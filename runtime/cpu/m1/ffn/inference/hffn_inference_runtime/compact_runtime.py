"""Reusable forward-only workspaces for compact hidden-major HFFN."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from hffn_pythia.compact_format import HFFNCompactG4Layer, HFFNDualRoute, build_dual_route
from .compact_torch_extension import load_hffn_compact_inference_extension
from .forward_route import DECODE, DUAL, OUTPUTS_Y, route as owner_route


@dataclass
class _Workspace:
    x_t: torch.Tensor
    y_nt: torch.Tensor
    y_t: torch.Tensor
    z_t: torch.Tensor
    h_t: torch.Tensor
    y_local: torch.Tensor
    timing: torch.Tensor
    dual_timing: torch.Tensor


class HFFNCompactG4InferenceRuntime:
    """CPU float32 HFFN forward runtime with one workspace per ``(layer, N)``."""

    def __init__(self, layers: list[HFFNCompactG4Layer], *, build_dir: Path | None = None, execution: str = "baseline", dual_route_min_n: int = 64) -> None:
        if not layers:
            raise ValueError("compact inference runtime requires layers")
        if execution not in {"baseline", "dual_route", "adaptive"}:
            raise ValueError("execution must be baseline, dual_route or adaptive")
        if dual_route_min_n < 1:
            raise ValueError("dual_route_min_n must be positive")
        for layer in layers:
            layer.validate()
        self.layers = tuple(layers)
        self.execution = execution
        self.dual_route_min_n = int(dual_route_min_n)
        self._routes: tuple[HFFNDualRoute, ...] = tuple(build_dual_route(layer) for layer in self.layers) if execution != "baseline" else ()
        self.ops = load_hffn_compact_inference_extension(build_dir or (Path(__file__).resolve().parents[1] / "build_inference"))
        self._workspaces: dict[tuple[int, int], _Workspace] = {}
        # The five scratch tensors of a call (x_t, y_t, z_t, h_t, y_local) are written before being read and dead on return, so one set per shape serves every layer.
        self._scratch: dict[tuple[int, int, int], tuple[torch.Tensor, ...]] = {}
        self._workspace_allocations = 0
        self._forward_calls = 0
        self._baseline_calls = 0
        self._dual_calls = 0
        self._decode_calls = 0

    @staticmethod
    def _check_tensor(value: torch.Tensor, name: str) -> None:
        if value.device.type != "cpu" or value.dtype != torch.float32 or not value.is_contiguous():
            raise ValueError(f"{name} must be a contiguous CPU float32 tensor")

    def _workspace(self, layer_idx: int, n: int) -> _Workspace:
        key = (layer_idx, n)
        workspace = self._workspaces.get(key)
        if workspace is None:
            spec = self.layers[layer_idx]
            shape = (spec.d_model, spec.d_ff, n)
            scratch = self._scratch.get(shape)
            if scratch is None:
                scratch = (torch.empty(spec.d_model, n), torch.empty(spec.d_model, n), torch.empty(spec.d_ff, n), torch.empty(spec.d_ff, n), torch.empty(4, spec.d_model, n))
                self._scratch[shape] = scratch
            x_t, y_t, z_t, h_t, y_local = scratch
            workspace = _Workspace(x_t, torch.empty(n, spec.d_model), y_t, z_t, h_t, y_local, torch.empty(3, dtype=torch.int64), torch.empty(4, dtype=torch.int64))
            self._workspaces[key] = workspace
            self._workspace_allocations += 1
        return workspace

    def forward_into(self, layer_idx: int, x_nt: torch.Tensor, y_nt: torch.Tensor) -> torch.Tensor:
        if not 0 <= layer_idx < len(self.layers):
            raise IndexError("invalid HFFN layer index")
        self._check_tensor(x_nt, "x_nt")
        self._check_tensor(y_nt, "y_nt")
        spec = self.layers[layer_idx]
        if x_nt.dim() != 2 or tuple(x_nt.shape) != tuple(y_nt.shape) or x_nt.size(1) != spec.d_model:
            raise ValueError("forward_into expects matching [N, H] input/output")
        workspace = self._workspace(layer_idx, int(x_nt.size(0)))
        n = int(x_nt.size(0))
        chosen = owner_route(n, OUTPUTS_Y)  # the owner route for {y}: decode at N == 1, the fused no-store entry above
        # Labelled control / test switches (not the production route): `dual_route` forces the dual entry at N >= 2, `adaptive` forces it from `dual_route_min_n`.
        forced_dual = self.execution == "dual_route" or (self.execution == "adaptive" and n >= self.dual_route_min_n)  # counted as the previous runtime counted it, also at N == 1
        if n > 1 and forced_dual:
            chosen = DUAL
        dual = forced_dual or chosen == DUAL
        if chosen == DECODE:
            self.ops.forward_decode_nt_groups(x_nt, spec.group_hidden_offsets, spec.b_up, spec.up_row_offsets, spec.up_idx, spec.up_val, spec.down_row_offsets, spec.down_idx, spec.down_val, spec.b_down, y_nt, workspace.z_t, workspace.h_t, workspace.y_local, workspace.timing)
            self._decode_calls += 1
        elif dual:
            dual_tables = self._routes[layer_idx]
            self.ops.forward_dual_nt(x_nt, spec.group_hidden_offsets, spec.b_up, spec.up_row_offsets, spec.up_idx, spec.up_val, spec.down_val, spec.b_down, dual_tables.down_by_output_offsets, dual_tables.down_slot_indices, dual_tables.down_slot_hidden, dual_tables.output_group_offsets, workspace.x_t, y_nt, workspace.y_t, workspace.z_t, workspace.h_t, workspace.dual_timing)
        else:
            self.ops.forward_nt(x_nt, spec.group_hidden_offsets, spec.b_up, spec.up_row_offsets, spec.up_idx, spec.up_val, spec.down_row_offsets, spec.down_idx, spec.down_val, spec.b_down, workspace.x_t, y_nt, workspace.y_t, workspace.z_t, workspace.h_t, workspace.y_local, workspace.timing)
        self._forward_calls += 1
        self._baseline_calls += int(not dual and x_nt.size(0) != 1)
        self._dual_calls += int(dual)
        return y_nt

    def forward(self, layer_idx: int, x_nt: torch.Tensor) -> torch.Tensor:
        """Convenience API; benchmarks must supply a reusable output buffer."""
        self._check_tensor(x_nt, "x_nt")
        return self.forward_into(layer_idx, x_nt, torch.empty_like(x_nt))

    def workspace_for(self, layer_idx: int, n: int) -> torch.Tensor:
        """Return the reusable output workspace used by the HF module."""
        return self._workspace(layer_idx, n).y_nt

    def workspace_stats(self) -> dict[str, int | str]:
        return {"allocated": self._workspace_allocations, "live": len(self._workspaces), "free": 0, "forward_calls": self._forward_calls, "baseline_calls": self._baseline_calls, "dual_calls": self._dual_calls, "decode_calls": self._decode_calls, "execution": self.execution}
