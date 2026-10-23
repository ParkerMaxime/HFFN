"""Reusable forward-only Metal runtime for exact-offset WANDA FFNs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .compact import CompactWandaInferenceLayer
from .shaders import inference_source

_LIBRARY_CACHE: dict[tuple[int, int, int, int, int, int], Any] = {}


def select_inference_layout(spec: CompactWandaInferenceLayer, n: int) -> str:
    """Measured forward-only policy for exact-offset Pythia/WANDA layers.

    A cooperative owner-row reduction fills the SIMD lanes at N=1. Persistent
    feature ownership covers the remaining small-N regime. Owner tiling
    amortizes its fixed tile at N>=64 and wins across the measured
    70M/160M/410M and low/high-density boundaries.
    """
    n = int(n)
    if n < 1:
        raise ValueError("N must be positive")
    row_reduce_max_n = 4 if spec.d_model <= 512 else 2
    if n <= row_reduce_max_n:
        return "owner_row_reduce"
    if n < 64:
        return "feature_persistent"
    return "feature_owner_tile"


@dataclass
class _DeviceLayer:
    spec: CompactWandaInferenceLayer
    flat: torch.Tensor
    up_val: torch.Tensor
    down_val: torch.Tensor
    b_up: torch.Tensor
    b_down: torch.Tensor
    up_offsets: torch.Tensor
    up_idx: torch.Tensor
    down_owner_offsets: torch.Tensor
    down_route_hidden: torch.Tensor


@dataclass
class _Workspace:
    act: torch.Tensor
    y_nt: torch.Tensor
    x_feature: torch.Tensor | None
    y_feature: torch.Tensor | None

    @property
    def numel(self) -> int:
        values = (self.act, self.y_nt, self.x_feature, self.y_feature)
        return sum(value.numel() for value in values if value is not None)


def _to_device_layer(spec: CompactWandaInferenceLayer) -> _DeviceLayer:
    # Inference weights are immutable, so the down values can be permuted once
    # into output-major route order.  This is a bijection, not padding or a
    # duplicated execution view, and removes the row-offset indirection.
    down_output_order = spec.down_val.index_select(0, spec.down_by_output.slot_order)
    flat = torch.cat((spec.up_val, down_output_order, spec.b_up, spec.b_down)).to("mps")
    up_count = spec.up_val.numel()
    down_count = spec.down_val.numel()
    up_bias_count = spec.b_up.numel()
    return _DeviceLayer(
        spec=spec,
        flat=flat,
        up_val=flat[:up_count],
        down_val=flat[up_count:up_count + down_count],
        b_up=flat[up_count + down_count:up_count + down_count + up_bias_count],
        b_down=flat[up_count + down_count + up_bias_count:],
        up_offsets=spec.up_row_offsets.to("mps"),
        up_idx=spec.up_idx.to("mps"),
        down_owner_offsets=spec.down_by_output.owner_offsets.to("mps"),
        down_route_hidden=spec.down_by_output.hidden_in_output_order.to("mps"),
    )


class HFFNMetalInferenceRuntime:
    """Inference-only runtime with immutable layer state and cached workspaces."""

    def __init__(self, specs: list[CompactWandaInferenceLayer]) -> None:
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available")
        if not specs:
            raise ValueError("inference runtime requires at least one layer")
        for spec in specs:
            spec.validate()
        self.layers = tuple(_to_device_layer(spec) for spec in specs)
        self._workspaces: dict[tuple[int, int, str], _Workspace] = {}
        self._workspace_allocations = 0
        self._forward_calls = 0
        self._layout_calls = {
            "token_major": 0,
            "owner_row_reduce": 0,
            "feature_persistent": 0,
            "feature_owner_tile": 0,
        }
        self._pending_inputs: list[torch.Tensor] = []
        self._inflight: list[tuple[torch.mps.Event, tuple[torch.Tensor, ...]]] = []
        self._hook_handles: tuple[Any, ...] = ()
        self.forced_layout: str | None = None

    def set_forced_layout(self, layout: str | None) -> None:
        allowed = {"token_major", "owner_row_reduce", "feature_persistent", "feature_owner_tile"}
        if layout is not None and layout not in allowed:
            raise ValueError(f"layout must be one of {sorted(allowed)}")
        self.forced_layout = layout

    def _reap_inflight(self) -> None:
        self._inflight = [entry for entry in self._inflight if not entry[0].query()]

    def begin_model_call(self) -> None:
        self._reap_inflight()
        if self._pending_inputs:
            raise RuntimeError("previous HFFN model call was not sealed")

    def end_model_call(self) -> None:
        if not self._pending_inputs:
            return
        completion = torch.mps.Event()
        completion.record()
        self._inflight.append((completion, tuple(self._pending_inputs)))
        self._pending_inputs.clear()

    def attach_model(self, model: torch.nn.Module) -> None:
        if self._hook_handles:
            raise RuntimeError("runtime is already attached to a model")

        def pre_hook(_module, _args):
            self.begin_model_call()

        def post_hook(_module, _args, output):
            self.end_model_call()
            return output

        self._hook_handles = (
            model.register_forward_pre_hook(pre_hook),
            model.register_forward_hook(post_hook, always_call=True),
        )

    def close(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = ()
        self.synchronize()

    def synchronize(self) -> None:
        torch.mps.synchronize()
        self._pending_inputs.clear()
        self._inflight.clear()

    @staticmethod
    def _dispatch(kernel, *args: torch.Tensor, threads: int) -> None:
        kernel(*args, threads=int(threads), group_size=256)

    @staticmethod
    def _dispatch_persistent(kernel, *args: torch.Tensor, owners: int) -> None:
        groups = (int(owners) + 15) // 16
        kernel(*args, threads=groups * 512, group_size=512)

    @staticmethod
    def _dispatch_owner_tile(kernel, *args: torch.Tensor, owners: int, n: int) -> None:
        groups = (int(owners) + 15) // 16
        tiles = (int(n) + 127) // 128
        kernel(*args, threads=groups * tiles * 512, group_size=512)

    def _library(self, layer: _DeviceLayer, n: int):
        spec = layer.spec
        key = (
            spec.d_model,
            spec.d_ff,
            int(n),
            spec.down_by_output.edge_bits,
            spec.up_val.numel(),
            spec.down_val.numel(),
        )
        library = _LIBRARY_CACHE.get(key)
        if library is None:
            library = torch.mps.compile_shader(inference_source(*key))
            _LIBRARY_CACHE[key] = library
        return library

    def _workspace(self, layer_idx: int, n: int, layout: str) -> _Workspace:
        key = (int(layer_idx), int(n), layout)
        workspace = self._workspaces.get(key)
        if workspace is None:
            spec = self.layers[layer_idx].spec
            feature = layout not in {"token_major", "owner_row_reduce"}
            workspace = _Workspace(
                act=torch.empty(spec.d_ff, n, device="mps", dtype=torch.float32),
                y_nt=torch.empty(n, spec.d_model, device="mps", dtype=torch.float32),
                x_feature=torch.empty(spec.d_model, n, device="mps", dtype=torch.float32) if feature else None,
                y_feature=torch.empty(spec.d_model, n, device="mps", dtype=torch.float32) if feature else None,
            )
            self._workspaces[key] = workspace
            self._workspace_allocations += 1
        return workspace

    def forward(self, layer_idx: int, x_nt: torch.Tensor) -> torch.Tensor:
        if not 0 <= layer_idx < len(self.layers):
            raise IndexError("invalid HFFN layer index")
        layer = self.layers[layer_idx]
        spec = layer.spec
        if x_nt.device.type != "mps" or x_nt.dtype != torch.float32 or x_nt.dim() != 2 or x_nt.size(1) != spec.d_model:
            raise ValueError(f"expected MPS float32 [N,{spec.d_model}]")
        x_nt = x_nt.contiguous()
        n = int(x_nt.size(0))
        layout = self.forced_layout or select_inference_layout(spec, n)
        workspace = self._workspace(layer_idx, n, layout)
        library = self._library(layer, n)
        self._pending_inputs.append(x_nt)
        if layout == "token_major":
            self._dispatch(
                library.up_token_major_inference,
                x_nt, layer.up_offsets, layer.up_idx, layer.up_val, layer.b_up, workspace.act,
                threads=spec.d_ff * n,
            )
            self._dispatch(
                library.down_token_major_inference,
                workspace.act, layer.down_owner_offsets, layer.down_route_hidden,
                layer.down_val, layer.b_down, workspace.y_nt,
                threads=n * spec.d_model,
            )
        elif layout == "owner_row_reduce":
            self._dispatch_persistent(
                library.up_owner_row_reduce_inference,
                x_nt, layer.up_offsets, layer.up_idx, layer.up_val,
                layer.b_up, workspace.act,
                owners=spec.d_ff,
            )
            self._dispatch_persistent(
                library.down_owner_row_reduce_inference,
                workspace.act, layer.down_owner_offsets, layer.down_route_hidden,
                layer.down_val, layer.b_down, workspace.y_nt,
                owners=spec.d_model,
            )
        else:
            assert workspace.x_feature is not None and workspace.y_feature is not None
            workspace.x_feature.copy_(x_nt.transpose(0, 1))
            if layout == "feature_persistent":
                self._dispatch_persistent(
                    library.up_feature_persistent_inference,
                    workspace.x_feature, layer.up_offsets, layer.up_idx,
                    layer.up_val, layer.b_up, workspace.act,
                    owners=spec.d_ff,
                )
                self._dispatch_persistent(
                    library.down_feature_persistent_inference,
                    workspace.act, layer.down_owner_offsets, layer.down_route_hidden,
                    layer.down_val, layer.b_down, workspace.y_feature,
                    owners=spec.d_model,
                )
            else:
                self._dispatch_owner_tile(
                    library.up_feature_owner_tile_inference,
                    workspace.x_feature, layer.up_offsets, layer.up_idx,
                    layer.up_val, layer.b_up, workspace.act,
                    owners=spec.d_ff, n=n,
                )
                self._dispatch_owner_tile(
                    library.down_feature_owner_tile_inference,
                    workspace.act, layer.down_owner_offsets, layer.down_route_hidden,
                    layer.down_val, layer.b_down, workspace.y_feature,
                    owners=spec.d_model, n=n,
                )
            workspace.y_nt.copy_(workspace.y_feature.transpose(0, 1))
        self._forward_calls += 1
        self._layout_calls[layout] += 1
        return workspace.y_nt

    def workspace_stats(self) -> dict[str, Any]:
        self._reap_inflight()
        return {
            "allocated": self._workspace_allocations,
            "live": len(self._workspaces),
            "workspace_bytes": 4 * sum(workspace.numel for workspace in self._workspaces.values()),
            "forward_calls": self._forward_calls,
            "layout_calls": dict(self._layout_calls),
            "pending_inputs": len(self._pending_inputs),
            "inflight_calls": len(self._inflight),
            "forced_layout": self.forced_layout,
        }
