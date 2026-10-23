"""One FFN layer of a compact training export (``hffn_training_compact_g4_v2``) and its reverse schedules.

The canonical storage is hidden-major (offsets, int16 indices, float32 values). The reverse schedules only index it:
``down_slots`` / ``up_slots``. The forward reads the canonical down values through ``down_slots`` with the slots entry of
the inference kernel: the layer holds no copy of them.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

FORMAT = "hffn_training_compact_g4_v2"
GROUPS = 4


def balanced_groups(cost: torch.Tensor, groups: int = GROUPS) -> torch.Tensor:
    """The partition of ``compact_export._balanced_group_offsets`` (and of the inference loader for T = groups)."""
    total, boundaries, cumulative, target = int(cost.sum()), [0], 0, 1
    for index, value in enumerate(cost.tolist()):
        cumulative += int(value)
        while target < groups and cumulative >= round(total * target / groups):
            boundaries.append(index + 1)
            target += 1
    while len(boundaries) < groups:
        boundaries.append(boundaries[-1])
    boundaries.append(cost.numel())
    return torch.tensor(boundaries, dtype=torch.int32)


def _reverse(offsets: torch.Tensor, index: torch.Tensor, d_ff: int, d_model: int):
    count = torch.bincount(index.long(), minlength=d_model).to(torch.int32)
    by_feature = torch.zeros(d_model + 1, dtype=torch.int32)
    by_feature[1:] = torch.cumsum(count, 0)
    slots = torch.argsort(index.long(), stable=True)
    owner = torch.repeat_interleave(torch.arange(d_ff), offsets.diff().long())
    return by_feature, slots.to(torch.int32).contiguous(), owner[slots].to(torch.int16).contiguous(), balanced_groups(count)


@dataclass
class Layer:
    name: str
    d_model: int
    d_ff: int
    groups: torch.Tensor
    up_offsets: torch.Tensor
    up_idx: torch.Tensor
    up_val: torch.Tensor
    b_up: torch.Tensor
    down_offsets: torch.Tensor
    down_idx: torch.Tensor
    down_val: torch.Tensor
    b_down: torch.Tensor
    down_by_output: torch.Tensor
    down_slots: torch.Tensor
    down_hidden: torch.Tensor
    output_groups: torch.Tensor
    up_by_input: torch.Tensor
    up_slots: torch.Tensor
    up_hidden: torch.Tensor
    input_groups: torch.Tensor

    @property
    def density(self) -> float:
        return self.up_val.numel() / (self.d_model * self.d_ff)

    def clone(self) -> "Layer":
        """Own copies of the trainable values (AdamW writes them in place)."""
        values = {k: (v.clone() if k in ("up_val", "b_up", "down_val", "b_down") else v) for k, v in self.__dict__.items()}
        return Layer(**values)


def load(path: Path) -> Layer:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if raw.get("format") != FORMAT:
        raise ValueError(f"{path}: not {FORMAT}")
    d_model, d_ff = int(raw["d_model"]), int(raw["d_ff"])
    i32 = lambda key: raw[key].contiguous().to(torch.int32)  # noqa: E731
    i16 = lambda key: raw[key].contiguous().to(torch.int16)  # noqa: E731
    f32 = lambda key: raw[key].contiguous().float().clone()  # noqa: E731
    up_offsets, down_offsets = i32("up_row_offsets"), i32("down_row_offsets")
    up_idx, down_idx = i16("up_idx"), i16("down_idx")
    down = _reverse(down_offsets, down_idx, d_ff, d_model)
    up = _reverse(up_offsets, up_idx, d_ff, d_model)
    groups = i32("group_hidden_offsets")
    if groups.numel() != GROUPS + 1 or int(groups[0]) != 0 or int(groups[-1]) != d_ff:
        raise ValueError(f"{path}: invalid G4 partition")
    return Layer(f"{path.parents[2].name}/{path.parent.name}", d_model, d_ff, groups, up_offsets, up_idx, f32("up_val"),
                 f32("b_up"), down_offsets, down_idx, f32("down_val"), f32("b_down"), *down, *up)


def from_compact(name: str, compact) -> Layer:
    """A training layer from a shared synthetic ``CompactLayer`` (numpy arrays of the leaf's test layers)."""
    t = lambda a, dtype: torch.from_numpy(a).contiguous().to(dtype).clone()  # noqa: E731
    up_offsets, down_offsets = t(compact.up_offsets, torch.int32), t(compact.down_offsets, torch.int32)
    up_idx, down_idx = t(compact.up_idx, torch.int16), t(compact.down_idx, torch.int16)
    groups = t(compact.groups, torch.int32)
    if groups.numel() != GROUPS + 1 or int(groups[0]) != 0 or int(groups[-1]) != compact.d_ff:
        raise ValueError(f"{name}: invalid G4 partition")
    down = _reverse(down_offsets, down_idx, compact.d_ff, compact.d_model)
    up = _reverse(up_offsets, up_idx, compact.d_ff, compact.d_model)
    return Layer(name, compact.d_model, compact.d_ff, groups, up_offsets, up_idx, t(compact.up_val, torch.float32),
                 t(compact.b_up, torch.float32), down_offsets, down_idx, t(compact.down_val, torch.float32),
                 t(compact.b_down, torch.float32), *down, *up)


def export_layers(export_dir: Path) -> list[Path]:
    paths = sorted(export_dir.glob(f"layer*/{FORMAT}.pt"), key=lambda p: int(p.parent.name[5:]))
    if not paths:
        raise FileNotFoundError(f"no {FORMAT} layer under {export_dir}")
    return paths
