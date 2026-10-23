"""Canonical ``hffn_training_compact_g4_v2`` reader used by inference only."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

# The file reader of the export exists once, at the shared Pythia level (first ancestor that has common/pythia/hffn_pythia).
_PYTHIA = next(p / "common/pythia" for p in Path(__file__).resolve().parents if (p / "common/pythia/hffn_pythia").is_dir())
if str(_PYTHIA) not in sys.path:
    sys.path.insert(0, str(_PYTHIA))

from hffn_pythia.export import read_layer  # noqa: E402

FORMAT = "hffn_training_compact_g4_v2"


@dataclass(frozen=True)
class HFFNCompactG4Layer:
    d_model: int
    d_ff: int
    group_hidden_offsets: torch.Tensor
    up_row_offsets: torch.Tensor
    up_idx: torch.Tensor
    up_val: torch.Tensor
    b_up: torch.Tensor
    down_row_offsets: torch.Tensor
    down_idx: torch.Tensor
    down_val: torch.Tensor
    b_down: torch.Tensor

    def validate(self) -> None:
        if self.group_hidden_offsets.dtype != torch.int32 or tuple(self.group_hidden_offsets.shape) != (5,):
            raise ValueError("group_hidden_offsets must be int32[5]")
        if int(self.group_hidden_offsets[0]) != 0 or int(self.group_hidden_offsets[-1]) != self.d_ff:
            raise ValueError("invalid G4 hidden partition")
        if bool((self.group_hidden_offsets[1:] < self.group_hidden_offsets[:-1]).any()):
            raise ValueError("non-monotonic G4 hidden partition")
        for offsets, indices, values, name in (
            (self.up_row_offsets, self.up_idx, self.up_val, "up"),
            (self.down_row_offsets, self.down_idx, self.down_val, "down"),
        ):
            if offsets.dtype != torch.int32 or offsets.numel() != self.d_ff + 1:
                raise ValueError(f"invalid {name} row offsets")
            if int(offsets[0]) != 0 or int(offsets[-1]) != values.numel() or indices.numel() != values.numel():
                raise ValueError(f"invalid {name} hidden-major storage")
            if indices.dtype != torch.int16 or values.dtype != torch.float32:
                raise ValueError(f"invalid {name} compact types")
            if bool((offsets[1:] < offsets[:-1]).any()):
                raise ValueError(f"non-monotonic {name} offsets")
            if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= self.d_model):
                raise ValueError(f"invalid {name} feature index")
        if self.b_up.dtype != torch.float32 or self.b_down.dtype != torch.float32:
            raise ValueError("HFFN biases must be float32")
        if self.b_up.numel() != self.d_ff or self.b_down.numel() != self.d_model:
            raise ValueError("invalid HFFN bias dimensions")


@dataclass(frozen=True)
class HFFNDualRoute:
    """Ownership schedules over canonical slots; this never duplicates weights."""
    down_by_output_offsets: torch.Tensor
    down_slot_indices: torch.Tensor
    down_slot_hidden: torch.Tensor
    output_group_offsets: torch.Tensor


def _balanced_group_offsets(cost: torch.Tensor) -> torch.Tensor:
    total, boundaries, cumulative, target = int(cost.sum()), [0], 0, 1
    for feature, value in enumerate(cost.tolist()):
        cumulative += int(value)
        while target < 4 and cumulative >= round(total * target / 4):
            boundaries.append(feature + 1)
            target += 1
    while len(boundaries) < 4:
        boundaries.append(boundaries[-1])
    boundaries.append(cost.numel())
    return torch.tensor(boundaries, dtype=torch.int32)


def build_dual_route(layer: HFFNCompactG4Layer) -> HFFNDualRoute:
    """Create only a reverse slot schedule for the dual output reduction."""
    index = layer.down_idx
    count = torch.bincount(index.to(torch.int64), minlength=layer.d_model).to(torch.int32)
    offsets = torch.empty(layer.d_model + 1, dtype=torch.int32)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(count, 0)
    slots = torch.argsort(index.to(torch.int64), stable=True).to(torch.int32)
    degrees = (layer.down_row_offsets[1:] - layer.down_row_offsets[:-1]).to(torch.int64)
    hidden_by_slot = torch.repeat_interleave(torch.arange(layer.d_ff, dtype=torch.int64), degrees)
    return HFFNDualRoute(offsets, slots, hidden_by_slot.index_select(0, slots.to(torch.int64)).to(torch.int16), _balanced_group_offsets(count))


def load_hffn_compact_g4_layer(path: str | Path) -> HFFNCompactG4Layer:
    """Load v2 through the shared reader; no padded, dense, or legacy conversion exists here."""
    raw = read_layer(Path(path).resolve(), groups=True)
    layer = HFFNCompactG4Layer(
        raw["d_model"], raw["d_ff"], raw["group_hidden_offsets"],
        raw["up_row_offsets"], raw["up_idx"], raw["up_val"], raw["b_up"],
        raw["down_row_offsets"], raw["down_idx"], raw["down_val"], raw["b_down"],
    )
    layer.validate()
    return layer
