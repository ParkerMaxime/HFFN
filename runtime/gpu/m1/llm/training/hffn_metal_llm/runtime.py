"""Adaptive exact-offset Metal FFN training runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch

from .compact import CompactWandaLayer
from .shaders import exact_offset_source

_INFERENCE = next(p for p in Path(__file__).resolve().parents if (p/'runtime'/'common'/'pythia'/'hffn_pythia').is_dir()) / 'runtime/gpu/m1/llm/inference'
if str(_INFERENCE) not in sys.path: sys.path.insert(0,str(_INFERENCE))
from native_transformer.hffn.training_entry import TrainingForward

_LIBRARY_CACHE: dict[tuple[int, int, int, int, int, int, int], object] = {}


@dataclass
class ForwardContext:
    x: torch.Tensor
    kernel_x: torch.Tensor
    z: torch.Tensor
    act: torch.Tensor
    layout: str


class ExactOffsetMetalFFN:
    """Exact WANDA support with token-major PyTorch boundaries.

    Token-major, persistent feature-major and owner-tile schedules all consume
    the same irregular compact support without padded or duplicated weights.
    """

    def __init__(
        self,
        layer: CompactWandaLayer,
        n: int,
        *,
        flat: torch.Tensor | None = None,
        layout: str = "token_major",
    ) -> None:
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available")
        if n < 1:
            raise ValueError("N must be positive")
        layer.validate()
        self.spec = layer
        self.n = int(n)
        if layout not in {"token_major", "feature_persistent", "feature_owner_tile"}:
            raise ValueError("layout must be token_major, feature_persistent or feature_owner_tile")
        self.layout = layout
        key = (
            layer.d_model, layer.d_ff, self.n,
            layer.routes.up_by_input.edge_bits,
            layer.routes.down_by_output.edge_bits,
            layer.up_val.numel(),
            layer.down_val.numel(),
        )
        library = _LIBRARY_CACHE.get(key)
        if library is None:
            library = torch.mps.compile_shader(exact_offset_source(*key))
            _LIBRARY_CACHE[key] = library
        self.lib = library
        expected = layer.up_val.numel() + layer.down_val.numel() + layer.b_up.numel() + layer.b_down.numel()
        if flat is None:
            flat = torch.cat((layer.up_val, layer.down_val, layer.b_up, layer.b_down)).to("mps")
        if flat.device.type != "mps" or flat.dtype != torch.float32 or flat.dim() != 1 or flat.numel() != expected:
            raise ValueError(f"expected flat MPS float32 parameter pool with {expected} entries")
        self.flat = flat
        u = layer.up_val.numel()
        d = layer.down_val.numel()
        bu = layer.b_up.numel()
        self.up_val = flat[:u]
        self.down_val = flat[u:u + d]
        self.b_up = flat[u + d:u + d + bu]
        self.b_down = flat[u + d + bu:]
        self.up_offsets = layer.up_row_offsets.to("mps")
        self.up_idx = layer.up_idx.to("mps")
        self.down_offsets = layer.down_row_offsets.to("mps")
        self.down_idx = layer.down_idx.to("mps")
        self.down_owner_offsets = layer.routes.down_by_output.owner_offsets.to("mps")
        self.down_route = layer.routes.down_by_output.packed_hidden_local.to("mps")
        self.down_route_slot = layer.routes.down_by_output.direct_slot.to("mps")
        self.down_route_hidden = layer.routes.down_by_output.direct_hidden.to("mps")
        self.up_owner_offsets = layer.routes.up_by_input.owner_offsets.to("mps")
        self.up_route = layer.routes.up_by_input.packed_hidden_local.to("mps")
        self.up_slot_hidden = layer.routes.up_slot_hidden.to("mps")
        self.down_slot_hidden = layer.routes.down_slot_hidden.to("mps")
        self.inference_forward = TrainingForward(
            d_in=layer.d_model,d_ff=layer.d_ff,d_out=layer.d_model,n=self.n,
            up_offsets=self.up_offsets,up_idx=self.up_idx,up_val=self.up_val,b_up=self.b_up,
            down_offsets=self.down_owner_offsets,down_hidden=self.down_route_hidden,
            down_slots=self.down_route_slot,down_val=self.down_val,b_down=self.b_down,
            layout='token_major' if layout=='token_major' else 'feature_major')

    @staticmethod
    def _dispatch(kernel, *args: torch.Tensor, threads: int) -> None:
        kernel(*args, threads=int(threads), group_size=256)

    @staticmethod
    def _dispatch_persistent(kernel, *args: torch.Tensor, owners: int) -> None:
        groups = (int(owners) + 15) // 16
        kernel(*args, threads=groups * 512, group_size=512)

    def _dispatch_owner_tile(self, kernel, *args: torch.Tensor, owners: int) -> None:
        groups = (int(owners) + 15) // 16
        tiles = (self.n + 127) // 128
        kernel(*args, threads=groups * tiles * 512, group_size=512)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ForwardContext]:
        if x.device.type != "mps" or x.dtype != torch.float32 or tuple(x.shape) != (self.n, self.spec.d_model):
            raise ValueError(f"expected contiguous MPS float32 [{self.n},{self.spec.d_model}]")
        x = x.contiguous()
        z = torch.empty(self.spec.d_ff, self.n, device="mps", dtype=torch.float32)
        act = torch.empty_like(z)
        kernel_x = x if self.layout == "token_major" else x.transpose(0, 1).contiguous()
        y = torch.empty(self.n, self.spec.d_model, device="mps", dtype=torch.float32)
        self.inference_forward.forward_into(kernel_x, z, act, y)
        return y, ForwardContext(x, kernel_x, z, act, self.layout)

    def backward(self, context: ForwardContext, gy: torch.Tensor) -> dict[str, torch.Tensor]:
        if gy.device.type != "mps" or gy.dtype != torch.float32 or tuple(gy.shape) != (self.n, self.spec.d_model):
            raise ValueError(f"expected contiguous MPS float32 grad [{self.n},{self.spec.d_model}]")
        gy = gy.contiguous()
        gz = torch.empty(self.spec.d_ff, self.n, device="mps", dtype=torch.float32)
        gflat = torch.empty_like(self.flat)
        u = self.up_val.numel()
        d = self.down_val.numel()
        bu = self.b_up.numel()
        gup = gflat[:u]
        gdown = gflat[u:u + d]
        gbup = gflat[u + d:u + d + bu]
        gbdown = gflat[u + d + bu:]
        # compile_shader mutates output tensors supplied as arguments.  Keep
        # every custom-kernel input/workspace alive until autograd consumes the
        # returned gradients; otherwise the MPS allocator can recycle storage
        # while an asynchronous dispatch still reads it.
        keepalive: list[torch.Tensor] = [
            context.x, context.kernel_x, context.z, context.act, gy, gz,
        ]
        if context.layout == "token_major":
            gx = torch.empty_like(context.x)
            self._dispatch(
                self.lib.hidden_backward,
                gy, context.z, self.down_offsets, self.down_idx, self.down_val, gz,
                threads=gz.numel(),
            )
            self._dispatch(
                self.lib.grad_down,
                gy, context.act, self.down_slot_hidden, self.down_idx, gdown,
                threads=gdown.numel(),
            )
            self._dispatch(
                self.lib.grad_up,
                gz, context.x, self.up_slot_hidden, self.up_idx, gup,
                threads=gup.numel(),
            )
            self._dispatch(self.lib.grad_b_up, gz, gbup, threads=gbup.numel())
            self._dispatch(self.lib.grad_b_down, gy, gbdown, threads=gbdown.numel())
            self._dispatch(
                self.lib.grad_x_token_major,
                gz, self.up_owner_offsets, self.up_route, self.up_offsets, self.up_val, gx,
                threads=gx.numel(),
            )
        elif context.layout == "feature_persistent":
            gy_feature = gy.transpose(0, 1).contiguous()
            gx_feature = torch.empty_like(context.kernel_x)
            keepalive.extend((gy_feature, gx_feature))
            self._dispatch_persistent(
                self.lib.hidden_backward_feature_persistent,
                gy_feature, context.z, self.down_offsets, self.down_idx,
                self.down_val, gz, gbup,
                owners=self.spec.d_ff,
            )
            self._dispatch_persistent(
                self.lib.down_grad_feature_persistent,
                context.act, gy_feature, self.down_owner_offsets, self.down_route,
                self.down_offsets, gdown, gbdown,
                owners=self.spec.d_model,
            )
            self._dispatch_persistent(
                self.lib.input_grad_feature_persistent,
                context.kernel_x, gz, self.up_owner_offsets, self.up_route,
                self.up_offsets, self.up_val, gup, gx_feature,
                owners=self.spec.d_model,
            )
            gx = gx_feature.transpose(0, 1).contiguous()
        else:
            gy_feature = gy.transpose(0, 1).contiguous()
            gx_feature = torch.empty_like(context.kernel_x)
            tiles = (self.n + 127) // 128
            partial = torch.empty(self.flat.numel() * tiles, device="mps", dtype=torch.float32)
            keepalive.extend((gy_feature, gx_feature, partial))
            self._dispatch_owner_tile(
                self.lib.hidden_backward_owner_tile,
                gy_feature, context.z, self.down_offsets, self.down_idx,
                self.down_val, gz, partial,
                owners=self.spec.d_ff,
            )
            self._dispatch_owner_tile(
                self.lib.down_grad_owner_tile,
                context.act, gy_feature, self.down_owner_offsets, self.down_route,
                self.down_offsets, partial,
                owners=self.spec.d_model,
            )
            self._dispatch_owner_tile(
                self.lib.input_grad_owner_tile,
                context.kernel_x, gz, self.up_owner_offsets, self.up_route,
                self.up_offsets, self.up_val, partial, gx_feature,
                owners=self.spec.d_model,
            )
            self._dispatch(
                self.lib.reduce_owner_tile_grad, partial, gflat,
                threads=gflat.numel(),
            )
            gx = gx_feature.transpose(0, 1).contiguous()
        return {
            "grad_x": gx,
            "grad_up": gup,
            "grad_down": gdown,
            "grad_b_up": gbup,
            "grad_b_down": gbdown,
            "grad_z": gz,
            "grad_flat": gflat,
            "_mps_keepalive": tuple(keepalive),
        }
