"""Fixed-support dense PyTorch reference for HFFN training comparisons.

The reference intentionally computes dense PyTorch FFNs, but its trainable
support must stay identical to the HFFN export.  A zero-valued pruned weight
otherwise receives a non-zero dense gradient and silently regrows on the first
AdamW step, which is not an iso-training comparison.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .padded_export import load_hffn_padded_training_layer
from .compact_export import load_hffn_compact_g4_layer


from hffn_pythia.dense_control import FixedSupportAdamW


def _layer_masks(layer) -> tuple[torch.Tensor, torch.Tensor]:
    up = torch.zeros((layer.d_ff, layer.d_model), dtype=torch.float32)
    down = torch.zeros((layer.d_model, layer.d_ff), dtype=torch.float32)
    max_group = int(layer.b_up_pad.shape[1])
    for hidden in range(layer.d_ff):
        group, row = divmod(hidden, max_group)
        up_degree, down_degree = int(layer.up_deg[group, row]), int(layer.down_deg[group, row])
        up[hidden, layer.up_idx_pad[group, row, :up_degree].long()] = 1.0
        down[layer.down_idx_pad[group, row, :down_degree].long(), hidden] = 1.0
    return up, down


def _compact_layer_masks(layer) -> tuple[torch.Tensor, torch.Tensor]:
    up = torch.zeros((layer.d_ff, layer.d_model), dtype=torch.float32)
    down = torch.zeros((layer.d_model, layer.d_ff), dtype=torch.float32)
    for hidden in range(layer.d_ff):
        us, ue = int(layer.up_row_offsets[hidden]), int(layer.up_row_offsets[hidden + 1])
        ds, de = int(layer.down_row_offsets[hidden]), int(layer.down_row_offsets[hidden + 1])
        up[hidden, layer.up_idx[us:ue].long()] = 1.0
        down[layer.down_idx[ds:de].long(), hidden] = 1.0
    return up, down


def build_fixed_support_pythia_adamw(model: nn.Module, export_dir: str | Path, *, lr: float, weight_decay: float) -> FixedSupportAdamW:
    """Create the paper-equivalent masked dense AdamW reference for Pythia."""
    blocks = list(getattr(getattr(model, "gpt_neox", None), "layers", ()))
    root = Path(export_dir)
    if not blocks:
        raise ValueError("Expected a GPT-NeoX/Pythia model.")
    masks: dict[nn.Parameter, torch.Tensor] = {}
    for index, block in enumerate(blocks):
        compact = root / f"layer{index}" / "hffn_training_compact_g4_v2.pt"
        if compact.is_file():
            up, down = _compact_layer_masks(load_hffn_compact_g4_layer(compact))
        else:
            up, down = _layer_masks(load_hffn_padded_training_layer(root / f"layer{index}" / "hffn_training_padded_v1.pt"))
        dense_up = block.mlp.dense_h_to_4h.weight
        dense_down = block.mlp.dense_4h_to_h.weight
        if dense_up.shape != up.shape or dense_down.shape != down.shape:
            raise ValueError(f"HFFN support shape mismatch in Pythia layer {index}.")
        masks[dense_up] = up; masks[dense_down] = down
    # Ensure a sparse checkpoint cannot contain stale off-support values.
    with torch.no_grad():
        for parameter, mask in masks.items():
            parameter.mul_(mask)
    return FixedSupportAdamW(model.parameters(), masks, lr=lr, weight_decay=weight_decay)


@torch.no_grad()
def install_compact_hffn_values_in_dense_pythia(
    model: nn.Module, export_dir: str | Path
) -> None:
    """Install compact HFFN values in the dense fixed-support comparator.

    The HFFN runtime never calls this function or materializes dense FFN
    weights. It exists only to construct the PyTorch control from the same
    released support and active values.
    """
    blocks = list(getattr(getattr(model, "gpt_neox", None), "layers", ()))
    root = Path(export_dir)
    if not blocks:
        raise ValueError("Expected a GPT-NeoX/Pythia model.")
    for index, block in enumerate(blocks):
        layer = load_hffn_compact_g4_layer(
            root / f"layer{index}" / "hffn_training_compact_g4_v2.pt"
        )
        up = block.mlp.dense_h_to_4h
        down = block.mlp.dense_4h_to_h
        if tuple(up.weight.shape) != (layer.d_ff, layer.d_model) or tuple(
            down.weight.shape
        ) != (layer.d_model, layer.d_ff):
            raise ValueError(f"HFFN export/model shape mismatch in Pythia layer {index}.")
        up.weight.zero_()
        down.weight.zero_()
        for hidden in range(layer.d_ff):
            up_start = int(layer.up_row_offsets[hidden])
            up_end = int(layer.up_row_offsets[hidden + 1])
            down_start = int(layer.down_row_offsets[hidden])
            down_end = int(layer.down_row_offsets[hidden + 1])
            up.weight[hidden, layer.up_idx[up_start:up_end].long()] = layer.up_val[
                up_start:up_end
            ]
            down.weight[layer.down_idx[down_start:down_end].long(), hidden] = (
                layer.down_val[down_start:down_end]
            )
        up.bias.copy_(layer.b_up)
        down.bias.copy_(layer.b_down)
