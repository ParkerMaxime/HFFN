"""Minimal forward-only reader for canonical exact-offset WANDA exports."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

# The file reader of the export exists once, at the shared Pythia level, found from the root of the `runtime/` tree
# (the first parent that holds `runtime/common/pythia`).
_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "runtime" / "common" / "pythia" / "hffn_pythia").is_dir())
_PYTHIA = _ROOT / "runtime/common/pythia"
if str(_PYTHIA) not in sys.path:
    sys.path.insert(0, str(_PYTHIA))

from hffn_pythia.export import read_layer  # noqa: E402


@dataclass(frozen=True)
class DownOutputRoute:
    owner_offsets: torch.Tensor
    packed_hidden_local: torch.Tensor
    hidden_in_output_order: torch.Tensor
    slot_order: torch.Tensor
    edge_bits: int

    def validate(self, d_model: int, d_ff: int, row_offsets: torch.Tensor) -> None:
        if self.owner_offsets.dtype != torch.int32 or self.owner_offsets.numel() != d_model + 1:
            raise ValueError("down owner offsets must be int32[d_model+1]")
        if self.packed_hidden_local.dtype != torch.int32:
            raise ValueError("packed down route must be int32")
        if self.hidden_in_output_order.dtype != torch.int16:
            raise ValueError("direct hidden route must be int16")
        if self.slot_order.dtype != torch.int64:
            raise ValueError("down slot permutation must be int64")
        if int(self.owner_offsets[0]) != 0 or int(self.owner_offsets[-1]) != self.packed_hidden_local.numel():
            raise ValueError("invalid down route extent")
        if self.hidden_in_output_order.numel() != self.packed_hidden_local.numel() or self.slot_order.numel() != self.packed_hidden_local.numel():
            raise ValueError("reordered down route extent mismatch")
        if not 1 <= self.edge_bits < 31:
            raise ValueError("invalid down route edge bit count")
        packed = self.packed_hidden_local.to(torch.int64)
        hidden = packed >> self.edge_bits
        local = packed & ((1 << self.edge_bits) - 1)
        if packed.numel() and (int(hidden.min()) < 0 or int(hidden.max()) >= d_ff):
            raise ValueError("invalid hidden owner in down route")
        degree = row_offsets[1:].to(torch.int64) - row_offsets[:-1].to(torch.int64)
        if packed.numel() and bool((local >= degree.index_select(0, hidden)).any()):
            raise ValueError("invalid local edge in down route")
        if not torch.equal(self.hidden_in_output_order.to(torch.int64), hidden):
            raise ValueError("direct hidden route disagrees with packed route")
        if self.slot_order.numel() and not torch.equal(torch.sort(self.slot_order).values, torch.arange(self.slot_order.numel(), dtype=torch.int64)):
            raise ValueError("down slot order is not a permutation")


@dataclass(frozen=True)
class CompactWandaInferenceLayer:
    d_model: int
    d_ff: int
    up_row_offsets: torch.Tensor
    up_idx: torch.Tensor
    up_val: torch.Tensor
    b_up: torch.Tensor
    down_row_offsets: torch.Tensor
    down_idx: torch.Tensor
    down_val: torch.Tensor
    b_down: torch.Tensor
    down_by_output: DownOutputRoute

    def validate(self) -> None:
        for offsets, index, value, name in (
            (self.up_row_offsets, self.up_idx, self.up_val, "up"),
            (self.down_row_offsets, self.down_idx, self.down_val, "down"),
        ):
            if offsets.dtype != torch.int32 or offsets.numel() != self.d_ff + 1:
                raise ValueError(f"invalid {name} row offsets")
            if index.dtype != torch.int16 or value.dtype != torch.float32:
                raise ValueError(f"invalid {name} compact dtype")
            if int(offsets[0]) != 0 or int(offsets[-1]) != value.numel() or index.numel() != value.numel():
                raise ValueError(f"invalid {name} compact extent")
            if bool((offsets[1:] < offsets[:-1]).any()):
                raise ValueError(f"non-monotonic {name} row offsets")
            if index.numel() and (int(index.min()) < 0 or int(index.max()) >= self.d_model):
                raise ValueError(f"invalid {name} feature index")
        if self.b_up.dtype != torch.float32 or self.b_up.numel() != self.d_ff:
            raise ValueError("invalid up bias")
        if self.b_down.dtype != torch.float32 or self.b_down.numel() != self.d_model:
            raise ValueError("invalid down bias")
        self.down_by_output.validate(self.d_model, self.d_ff, self.down_row_offsets)

    @property
    def compact_numel(self) -> int:
        return self.up_val.numel() + self.down_val.numel() + self.b_up.numel() + self.b_down.numel()


def _build_down_output_route(
    row_offsets: torch.Tensor,
    output_index: torch.Tensor,
    d_model: int,
) -> DownOutputRoute:
    degree = (row_offsets[1:] - row_offsets[:-1]).to(torch.int64)
    hidden = torch.repeat_interleave(torch.arange(degree.numel(), dtype=torch.int64), degree)
    slot = torch.arange(output_index.numel(), dtype=torch.int64)
    local = slot - row_offsets.to(torch.int64).index_select(0, hidden)
    max_degree = int(degree.max()) if degree.numel() else 0
    edge_bits = max(1, max(0, max_degree - 1).bit_length())
    if degree.numel().bit_length() + edge_bits > 31:
        raise ValueError("WANDA down route does not fit signed int32 packing")
    order = torch.argsort(output_index.to(torch.int64), stable=True)
    packed = ((hidden << edge_bits) | local).index_select(0, order).to(torch.int32)
    counts = torch.bincount(output_index.to(torch.int64), minlength=d_model).to(torch.int32)
    owner_offsets = torch.empty(d_model + 1, dtype=torch.int32)
    owner_offsets[0] = 0
    owner_offsets[1:] = torch.cumsum(counts, 0)
    return DownOutputRoute(
        owner_offsets,
        packed,
        hidden.index_select(0, order).to(torch.int16),
        order.to(torch.int64),
        edge_bits,
    )


def load_compact_wanda_inference_layer(path: str | Path) -> CompactWandaInferenceLayer:
    raw = read_layer(Path(path).resolve(), groups=True)
    d_model, d_ff = raw["d_model"], raw["d_ff"]
    down_offsets, down_idx = raw["down_row_offsets"], raw["down_idx"]
    layer = CompactWandaInferenceLayer(
        d_model=d_model,
        d_ff=d_ff,
        up_row_offsets=raw["up_row_offsets"],
        up_idx=raw["up_idx"],
        up_val=raw["up_val"],
        b_up=raw["b_up"],
        down_row_offsets=down_offsets,
        down_idx=down_idx,
        down_val=raw["down_val"],
        b_down=raw["b_down"],
        down_by_output=_build_down_output_route(down_offsets, down_idx, d_model),
    )
    layer.validate()
    return layer
