from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_random_mask(
    out_features: int,
    in_features: int,
    density: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if not (0.0 < density <= 1.0):
        raise ValueError(f"density must be in (0, 1], got {density}")

    mask = (torch.rand(out_features, in_features, generator=generator) < density).float()

    row_sums = mask.sum(dim=1)
    empty_rows = torch.nonzero(row_sums == 0, as_tuple=False).flatten()
    for r in empty_rows.tolist():
        c = torch.randint(0, in_features, (1,), generator=generator).item()
        mask[r, c] = 1.0

    col_sums = mask.sum(dim=0)
    empty_cols = torch.nonzero(col_sums == 0, as_tuple=False).flatten()
    for c in empty_cols.tolist():
        r = torch.randint(0, out_features, (1,), generator=generator).item()
        mask[r, c] = 1.0

    return mask


class MaskedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        density: float,
        generator: torch.Generator,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.register_buffer(
            "mask",
            make_random_mask(
                out_features=out_features,
                in_features=in_features,
                density=density,
                generator=generator,
            ),
        )
        self.reset_parameters()
        self.apply_mask_()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    @torch.no_grad()
    def apply_mask_(self) -> None:
        self.weight.mul_(self.mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.mask, self.bias)


class SparseMaskedReferenceFFN(nn.Module):
    """
    Baseline reference:
      - pure PyTorch
      - CPU only in this project phase
      - fixed sparse mask
      - PyTorch native threading policy left on auto
    """

    def __init__(self, d_model: int, d_ff: int, density: float, generator: torch.Generator):
        super().__init__()
        self.up = MaskedLinear(d_model, d_ff, density, generator, bias=True)
        self.down = MaskedLinear(d_ff, d_model, density, generator, bias=True)

    @torch.no_grad()
    def apply_masks_(self) -> None:
        self.up.apply_mask_()
        self.down.apply_mask_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x), approximate="none"))


def build_hidden_major(
    up_weight: torch.Tensor,
    up_mask: torch.Tensor,
    down_weight: torch.Tensor,
    down_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    d_ff, _ = up_weight.shape
    up_ptr: List[int] = [0]
    up_idx_parts: List[torch.Tensor] = []
    up_val_parts: List[torch.Tensor] = []
    down_ptr: List[int] = [0]
    down_idx_parts: List[torch.Tensor] = []
    down_val_parts: List[torch.Tensor] = []

    for hidden in range(d_ff):
        up_active = torch.nonzero(up_mask[hidden] > 0, as_tuple=False).flatten()
        up_idx_parts.append(up_active.to(torch.long))
        up_val_parts.append(up_weight[hidden, up_active].contiguous())
        up_ptr.append(up_ptr[-1] + int(up_active.numel()))

        down_active = torch.nonzero(down_mask[:, hidden] > 0, as_tuple=False).flatten()
        down_idx_parts.append(down_active.to(torch.long))
        down_val_parts.append(down_weight[down_active, hidden].contiguous())
        down_ptr.append(down_ptr[-1] + int(down_active.numel()))

    return {
        "up_ptr": torch.tensor(up_ptr, dtype=torch.long),
        "up_idx": torch.cat(up_idx_parts, dim=0) if up_idx_parts else torch.empty(0, dtype=torch.long),
        "up_val": torch.cat(up_val_parts, dim=0) if up_val_parts else torch.empty(0),
        "down_ptr": torch.tensor(down_ptr, dtype=torch.long),
        "down_idx": torch.cat(down_idx_parts, dim=0) if down_idx_parts else torch.empty(0, dtype=torch.long),
        "down_val": torch.cat(down_val_parts, dim=0) if down_val_parts else torch.empty(0),
    }


class HFFNCompactFFN(nn.Module):
    def __init__(self, reference: SparseMaskedReferenceFFN):
        super().__init__()
        packed = build_hidden_major(
            reference.up.weight.detach() * reference.up.mask.detach(),
            reference.up.mask.detach(),
            reference.down.weight.detach() * reference.down.mask.detach(),
            reference.down.mask.detach(),
        )
        self.d_model = int(reference.up.in_features)
        self.d_ff = int(reference.up.out_features)

        self.register_buffer("up_ptr", packed["up_ptr"])
        self.register_buffer("up_idx", packed["up_idx"])
        self.register_buffer("down_ptr", packed["down_ptr"])
        self.register_buffer("down_idx", packed["down_idx"])

        self.up_val = nn.Parameter(packed["up_val"].float())
        self.down_val = nn.Parameter(packed["down_val"].float())
        self.b_up = nn.Parameter(reference.up.bias.detach().clone().float())
        self.b_down = nn.Parameter(reference.down.bias.detach().clone().float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        y = self.b_down.unsqueeze(0).expand(batch, self.d_model).clone()

        for hidden in range(self.d_ff):
            up_start = int(self.up_ptr[hidden].item())
            up_end = int(self.up_ptr[hidden + 1].item())
            if up_end > up_start:
                idx = self.up_idx[up_start:up_end]
                val = self.up_val[up_start:up_end]
                z = (x[:, idx] * val.unsqueeze(0)).sum(dim=1) + self.b_up[hidden]
            else:
                z = self.b_up[hidden].expand(batch)

            h = F.gelu(z, approximate="none")

            down_start = int(self.down_ptr[hidden].item())
            down_end = int(self.down_ptr[hidden + 1].item())
            if down_end > down_start:
                out_idx = self.down_idx[down_start:down_end]
                out_val = self.down_val[down_start:down_end]
                contrib = h.unsqueeze(1) * out_val.unsqueeze(0)
                scatter_index = out_idx.unsqueeze(0).expand(batch, -1)
                y = y.scatter_add(1, scatter_index, contrib)

        return y
