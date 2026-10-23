"""Canonical WANDA loader and exact Metal scheduling metadata.

The CPU export remains the source of truth.  CPU G4 partitions are validated
but deliberately not used for GPU scheduling.
"""

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
class MetalReverseRoute:
    owner_offsets: torch.Tensor
    packed_hidden_local: torch.Tensor
    direct_slot: torch.Tensor
    direct_hidden: torch.Tensor
    edge_bits: int

    def validate(self, owners: int, hidden: int, row_offsets: torch.Tensor) -> None:
        if self.owner_offsets.dtype != torch.int32 or self.owner_offsets.numel() != owners + 1:
            raise ValueError("reverse owner offsets must be int32[owners+1]")
        if self.packed_hidden_local.dtype != torch.int32:
            raise ValueError("packed reverse route must be int32")
        if self.direct_slot.dtype != torch.int32 or self.direct_hidden.dtype != torch.int16:
            raise ValueError("direct reverse route must be int32 slots + int16 hidden owners")
        if int(self.owner_offsets[0]) != 0 or int(self.owner_offsets[-1]) != self.packed_hidden_local.numel():
            raise ValueError("invalid reverse route extent")
        if self.direct_slot.numel() != self.packed_hidden_local.numel() or self.direct_hidden.numel() != self.packed_hidden_local.numel():
            raise ValueError("direct reverse route extent mismatch")
        if self.edge_bits < 1 or self.edge_bits >= 31:
            raise ValueError("invalid reverse route edge bit count")
        packed = self.packed_hidden_local.to(torch.int64)
        route_hidden = packed >> self.edge_bits
        route_local = packed & ((1 << self.edge_bits) - 1)
        if packed.numel() and (int(route_hidden.min()) < 0 or int(route_hidden.max()) >= hidden):
            raise ValueError("invalid hidden owner in reverse route")
        degree = row_offsets[1:] - row_offsets[:-1]
        if packed.numel() and bool((route_local >= degree.index_select(0, route_hidden)).any()):
            raise ValueError("invalid local edge in reverse route")
        direct_hidden = self.direct_hidden.to(torch.int64)
        direct_slot = self.direct_slot.to(torch.int64)
        if direct_slot.numel():
            if int(direct_hidden.min()) < 0 or int(direct_hidden.max()) >= hidden:
                raise ValueError("invalid hidden owner in direct reverse route")
            direct_begin = row_offsets.to(torch.int64).index_select(0, direct_hidden)
            direct_end = row_offsets.to(torch.int64).index_select(0, direct_hidden + 1)
            if bool(((direct_slot < direct_begin) | (direct_slot >= direct_end)).any()):
                raise ValueError("invalid compact slot in direct reverse route")
            if not torch.equal(direct_hidden, route_hidden):
                raise ValueError("packed and direct reverse hidden owners disagree")
            if not torch.equal(direct_slot, direct_begin + route_local):
                raise ValueError("packed and direct reverse slots disagree")


@dataclass(frozen=True)
class MetalRoutes:
    down_by_output: MetalReverseRoute
    up_by_input: MetalReverseRoute
    up_slot_hidden: torch.Tensor
    down_slot_hidden: torch.Tensor


@dataclass(frozen=True)
class CompactWandaLayer:
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
    routes: MetalRoutes

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
        if self.routes.up_slot_hidden.dtype != torch.int16 or self.routes.up_slot_hidden.numel() != self.up_val.numel():
            raise ValueError("invalid up slot owners")
        if self.routes.down_slot_hidden.dtype != torch.int16 or self.routes.down_slot_hidden.numel() != self.down_val.numel():
            raise ValueError("invalid down slot owners")
        self.routes.down_by_output.validate(self.d_model, self.d_ff, self.down_row_offsets)
        self.routes.up_by_input.validate(self.d_model, self.d_ff, self.up_row_offsets)


def _reverse_route(row_offsets: torch.Tensor, index: torch.Tensor, owners: int) -> tuple[MetalReverseRoute, torch.Tensor]:
    degree = (row_offsets[1:] - row_offsets[:-1]).to(torch.int64)
    hidden = torch.repeat_interleave(torch.arange(degree.numel(), dtype=torch.int64), degree)
    slot = torch.arange(index.numel(), dtype=torch.int64)
    local = slot - row_offsets.to(torch.int64).index_select(0, hidden)
    max_degree = int(degree.max()) if degree.numel() else 0
    edge_bits = max(1, max(0, max_degree - 1).bit_length())
    if degree.numel().bit_length() + edge_bits > 31:
        raise ValueError("WANDA route does not fit signed int32 packing")
    order = torch.argsort(index.to(torch.int64), stable=True)
    packed = ((hidden << edge_bits) | local).index_select(0, order).to(torch.int32)
    direct_slot = slot.index_select(0, order).to(torch.int32)
    direct_hidden = hidden.index_select(0, order).to(torch.int16)
    counts = torch.bincount(index.to(torch.int64), minlength=owners).to(torch.int32)
    owner_offsets = torch.empty(owners + 1, dtype=torch.int32)
    owner_offsets[0] = 0
    owner_offsets[1:] = torch.cumsum(counts, 0)
    return MetalReverseRoute(owner_offsets, packed, direct_slot, direct_hidden, edge_bits), hidden.to(torch.int16)


def load_compact_wanda_layer(path: str | Path) -> CompactWandaLayer:
    raw = read_layer(Path(path).resolve(), groups=True)
    d_model, d_ff = raw["d_model"], raw["d_ff"]
    up_offsets, up_idx = raw["up_row_offsets"], raw["up_idx"]
    down_offsets, down_idx = raw["down_row_offsets"], raw["down_idx"]
    down_route, down_hidden = _reverse_route(down_offsets, down_idx, d_model)
    up_route, up_hidden = _reverse_route(up_offsets, up_idx, d_model)
    layer = CompactWandaLayer(
        d_model=d_model,
        d_ff=d_ff,
        up_row_offsets=up_offsets,
        up_idx=up_idx,
        up_val=raw["up_val"],
        b_up=raw["b_up"],
        down_row_offsets=down_offsets,
        down_idx=down_idx,
        down_val=raw["down_val"],
        b_down=raw["b_down"],
        routes=MetalRoutes(down_route, up_route, up_hidden, down_hidden),
    )
    layer.validate()
    return layer
