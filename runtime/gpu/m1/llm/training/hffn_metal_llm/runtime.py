"""Exact-offset Metal FFN forward of the training leaf (the backward is native: ``gpu/m1/ffn/training``)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch

from .compact import CompactWandaLayer

_INFERENCE = next(p for p in Path(__file__).resolve().parents if (p / "runtime" / "common" / "pythia" / "hffn_pythia").is_dir()) / 'runtime/gpu/m1/llm/inference'
if str(_INFERENCE) not in sys.path: sys.path.insert(0,str(_INFERENCE))
from native_transformer.hffn.training_entry import TrainingForward


@dataclass
class ForwardContext:
    x: torch.Tensor
    kernel_x: torch.Tensor
    z: torch.Tensor
    act: torch.Tensor
    layout: str


class ExactOffsetMetalFFN:
    """Exact WANDA support with token-major PyTorch boundaries (forward only).

    Token-major and feature-major schedules consume the same irregular compact support without padded or duplicated
    weights. The training backward of the block of N tokens is the owner's native staged context.
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
        if layout not in {
            "token_major", "feature_persistent", "feature_owner_tile",
            "feature_owner_tile_fused", "feature_owner_tile_fused_direct",
            "feature_owner_tile_fused_balanced",
            "feature_owner_tile_fused_gup",
            "feature_owner_tile_fused_gup_split4",
        }:
            raise ValueError(
                "layout must be token_major, feature_persistent, "
                "feature_owner_tile, feature_owner_tile_fused, or "
                "feature_owner_tile_fused_direct/balanced/gup/gup_split4"
            )
        self.layout = layout
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
        self.down_owner_offsets = layer.routes.down_by_output.owner_offsets.to("mps")
        self.down_route_slot = layer.routes.down_by_output.direct_slot.to("mps")
        self.down_route_hidden = layer.routes.down_by_output.direct_hidden.to("mps")

        self.inference_forward = TrainingForward(
            d_in=layer.d_model,d_ff=layer.d_ff,d_out=layer.d_model,n=self.n,
            up_offsets=self.up_offsets,up_idx=self.up_idx,up_val=self.up_val,b_up=self.b_up,
            down_offsets=self.down_owner_offsets,down_hidden=self.down_route_hidden,
            down_slots=self.down_route_slot,down_val=self.down_val,b_down=self.b_down,
            layout='token_major' if layout=='token_major' else 'feature_major')

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
