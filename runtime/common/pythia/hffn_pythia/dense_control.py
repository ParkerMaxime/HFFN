"""The dense fixed-support control arm of a Pythia export: stock Hugging Face FFNs holding the active values of the
export, and the AdamW that keeps them on that support.

The HFFN runtime never calls this module and never materialises a dense FFN weight. It builds the PyTorch control
from the same released support and active values, for the checks and benchmarks of the hardware leaves. The values
come through ``read_layer`` (the one reader of the format); a file that is not the canonical export is refused.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .export import FORMAT, read_layer


class FixedSupportAdamW:
    """AdamW whose selected dense FFN matrices remain on a fixed support.

    The off-support entries start at zero (``build_fixed_support_pythia_adamw`` masks them). Masking the gradient
    before each step is then enough: their moments stay zero, and AdamW moves a zero entry by a decayed zero plus
    ``-lr * 0 / eps``, which is zero again.
    """

    def __init__(self, params, masks: dict[nn.Parameter, torch.Tensor], **kwargs) -> None:
        self.base = torch.optim.AdamW(params, **kwargs)
        self._masks = {parameter: mask.detach().to(dtype=torch.float32, device=parameter.device).contiguous() for parameter, mask in masks.items()}

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self) -> None:
        # Prevent pruned values from affecting Adam moments or the update.
        for parameter, mask in self._masks.items():
            if parameter.grad is not None:
                parameter.grad.mul_(mask)
        self.base.step()

    @property
    def param_groups(self):
        return self.base.param_groups


def _blocks(model: nn.Module) -> list:
    blocks = list(getattr(getattr(model, "gpt_neox", None), "layers", ()))
    if not blocks:
        raise ValueError("Expected a GPT-NeoX/Pythia model.")
    return blocks


def _layer(root: Path, index: int, block) -> dict:
    layer = read_layer((root / f"layer{index}" / f"{FORMAT}.pt").resolve())
    up, down = block.mlp.dense_h_to_4h.weight, block.mlp.dense_4h_to_h.weight
    if tuple(up.shape) != (layer["d_ff"], layer["d_model"]) or tuple(down.shape) != (layer["d_model"], layer["d_ff"]):
        raise ValueError(f"HFFN export/model shape mismatch in Pythia layer {index}.")
    return layer


@torch.no_grad()
def install_compact_hffn_values_in_dense_pythia(model: nn.Module, export_dir: str | Path) -> None:
    """Zero the dense FFN weights, then write the active values and the biases of the export."""
    root = Path(export_dir)
    for index, block in enumerate(_blocks(model)):
        layer = _layer(root, index, block)
        up, down = block.mlp.dense_h_to_4h, block.mlp.dense_4h_to_h
        up.weight.zero_()
        down.weight.zero_()
        for hidden in range(layer["d_ff"]):
            us, ue = int(layer["up_row_offsets"][hidden]), int(layer["up_row_offsets"][hidden + 1])
            ds, de = int(layer["down_row_offsets"][hidden]), int(layer["down_row_offsets"][hidden + 1])
            up.weight[hidden, layer["up_idx"][us:ue].long()] = layer["up_val"][us:ue]
            down.weight[layer["down_idx"][ds:de].long(), hidden] = layer["down_val"][ds:de]
        up.bias.copy_(layer["b_up"])
        down.bias.copy_(layer["b_down"])


def build_fixed_support_pythia_adamw(model: nn.Module, export_dir: str | Path, *, lr: float, weight_decay: float) -> FixedSupportAdamW:
    """The AdamW of the dense control, its masks being the support of the export (float32, on the parameters' device)."""
    root = Path(export_dir)
    masks: dict[nn.Parameter, torch.Tensor] = {}
    for index, block in enumerate(_blocks(model)):
        layer = _layer(root, index, block)
        up_mask = torch.zeros((layer["d_ff"], layer["d_model"]), dtype=torch.float32)
        down_mask = torch.zeros((layer["d_model"], layer["d_ff"]), dtype=torch.float32)
        for hidden in range(layer["d_ff"]):
            us, ue = int(layer["up_row_offsets"][hidden]), int(layer["up_row_offsets"][hidden + 1])
            ds, de = int(layer["down_row_offsets"][hidden]), int(layer["down_row_offsets"][hidden + 1])
            up_mask[hidden, layer["up_idx"][us:ue].long()] = 1.0
            down_mask[layer["down_idx"][ds:de].long(), hidden] = 1.0
        masks[block.mlp.dense_h_to_4h.weight] = up_mask
        masks[block.mlp.dense_4h_to_h.weight] = down_mask
    # A sparse checkpoint cannot keep stale off-support values.
    with torch.no_grad():
        for parameter, mask in masks.items():
            parameter.mul_(mask)
    return FixedSupportAdamW(model.parameters(), masks, lr=lr, weight_decay=weight_decay)
