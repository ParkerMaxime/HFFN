"""The compact hidden-major G4 layer of the CPU runtimes, once: the layer, its validation, the reverse routes of the
dual-route kernels and the partition of the hidden neurons into the four groups.

Offsets delimit hidden-neuron rows in flat active-connection arrays. This is the HFFN representation of the published
training kernel and of the inference kernel, not a generic sparse format. The file reader is ``export.read_layer``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .export import read_layer

# Number of hidden-neuron groups of the G4 format (bound to native/g4_contract.h by the leaf checks).
G4_GROUPS = 4


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
        if self.group_hidden_offsets.dtype != torch.int32 or tuple(self.group_hidden_offsets.shape) != (G4_GROUPS + 1,):
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
    """Reverse routing over the canonical slots, never a second weight storage.

    The down route (by output feature) is always built. The up route (by input feature) is built on request, for the
    training backward: the inference forward never reads it."""
    down_by_output_offsets: torch.Tensor
    down_slot_indices: torch.Tensor
    down_slot_hidden: torch.Tensor
    output_group_offsets: torch.Tensor
    up_by_input_offsets: torch.Tensor | None = None
    up_slot_indices: torch.Tensor | None = None
    up_slot_hidden: torch.Tensor | None = None
    input_group_offsets: torch.Tensor | None = None


def balanced_group_offsets(cost: torch.Tensor) -> torch.Tensor:
    """Partition of ``cost.numel()`` items into the G4 groups, balancing the summed cost (int32[5])."""
    total, boundaries, cumulative, target = int(cost.sum()), [0], 0, 1
    for item, value in enumerate(cost.tolist()):
        cumulative += int(value)
        while target < G4_GROUPS and cumulative >= round(total * target / G4_GROUPS):
            boundaries.append(item + 1)
            target += 1
    while len(boundaries) < G4_GROUPS:
        boundaries.append(boundaries[-1])
    boundaries.append(cost.numel())
    return torch.tensor(boundaries, dtype=torch.int32)


def _reverse(layer: HFFNCompactG4Layer, offsets: torch.Tensor, index: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Slots grouped by feature (stable, so hidden-major order is kept inside a feature), the hidden neuron owning each,
    and the G4 partition of the features balancing their slot counts."""
    count = torch.bincount(index.to(torch.int64), minlength=layer.d_model).to(torch.int32)
    by_feature = torch.empty(layer.d_model + 1, dtype=torch.int32)
    by_feature[0] = 0
    by_feature[1:] = torch.cumsum(count, 0)
    slots = torch.argsort(index.to(torch.int64), stable=True)
    degrees = (offsets[1:] - offsets[:-1]).to(torch.int64)
    owner = torch.repeat_interleave(torch.arange(layer.d_ff, dtype=torch.int64), degrees)
    return by_feature, slots.to(torch.int32), owner.index_select(0, slots).to(torch.int16), balanced_group_offsets(count)


def build_dual_route(layer: HFFNCompactG4Layer, *, up: bool = False) -> HFFNDualRoute:
    """The reverse slot schedules from the hidden-major slots: down by output feature, and with ``up`` the up by input
    feature too (the training backward)."""
    down_offsets, down_slots, down_hidden, output_groups = _reverse(layer, layer.down_row_offsets, layer.down_idx)
    if not up:
        return HFFNDualRoute(down_offsets, down_slots, down_hidden, output_groups)
    up_offsets, up_slots, up_hidden, input_groups = _reverse(layer, layer.up_row_offsets, layer.up_idx)
    return HFFNDualRoute(down_offsets, down_slots, down_hidden, output_groups, up_offsets, up_slots, up_hidden, input_groups)


def load_hffn_compact_g4_layer(path: str | Path) -> HFFNCompactG4Layer:
    """Load a ``hffn_training_compact_g4_v2`` export through the shared reader and validate it."""
    raw = read_layer(Path(path).resolve(), groups=True)
    layer = HFFNCompactG4Layer(
        raw["d_model"], raw["d_ff"], raw["group_hidden_offsets"],
        raw["up_row_offsets"], raw["up_idx"], raw["up_val"], raw["b_up"],
        raw["down_row_offsets"], raw["down_idx"], raw["down_val"], raw["b_down"],
    )
    layer.validate()
    return layer
