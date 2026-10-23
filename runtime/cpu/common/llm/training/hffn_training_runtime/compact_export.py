"""Canonical compact hidden-major G4 training export.

Offsets delimit hidden-neuron rows in flat active-connection arrays.  This is
the HFFN representation used by the published training kernel, not a generic
sparse matrix format.
"""
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

from .padded_export import G4_GROUPS, load_hffn_padded_training_layer  # noqa: E402


@dataclass(frozen=True)
class HFFNCompactG4Layer:
    d_model: int; d_ff: int
    group_hidden_offsets: torch.Tensor
    up_row_offsets: torch.Tensor; up_idx: torch.Tensor; up_val: torch.Tensor; b_up: torch.Tensor
    down_row_offsets: torch.Tensor; down_idx: torch.Tensor; down_val: torch.Tensor; b_down: torch.Tensor

    def validate(self) -> None:
        if self.group_hidden_offsets.dtype != torch.int32 or tuple(self.group_hidden_offsets.shape) != (G4_GROUPS + 1,): raise ValueError("compact G4 offsets must be int32[5]")
        if int(self.group_hidden_offsets[0]) != 0 or int(self.group_hidden_offsets[-1]) != self.d_ff: raise ValueError("invalid compact G4 partition")
        if bool((self.group_hidden_offsets[1:] < self.group_hidden_offsets[:-1]).any()): raise ValueError("non-monotonic compact G4 partition")
        for offsets, indices, values, name in ((self.up_row_offsets,self.up_idx,self.up_val,"up"),(self.down_row_offsets,self.down_idx,self.down_val,"down")):
            if offsets.dtype != torch.int32 or offsets.numel() != self.d_ff + 1: raise ValueError(f"invalid {name} row offsets")
            if int(offsets[0]) != 0 or int(offsets[-1]) != values.numel() or indices.numel() != values.numel(): raise ValueError(f"invalid compact {name} storage")
            if indices.dtype != torch.int16: raise ValueError(f"compact {name} indices must be int16")
            if bool((offsets[1:] < offsets[:-1]).any()): raise ValueError(f"non-monotonic {name} offsets")
            if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= self.d_model): raise ValueError(f"invalid {name} index")
        if self.up_val.dtype != torch.float32 or self.down_val.dtype != torch.float32 or self.b_up.dtype != torch.float32 or self.b_down.dtype != torch.float32: raise ValueError("compact values must be float32")
        if self.b_up.numel() != self.d_ff or self.b_down.numel() != self.d_model: raise ValueError("invalid compact biases")


@dataclass(frozen=True)
class HFFNDualRoute:
    """Reverse routing over HFFN slots, never a second weight storage."""
    down_by_output_offsets: torch.Tensor; down_slot_indices: torch.Tensor; down_slot_hidden: torch.Tensor
    up_by_input_offsets: torch.Tensor; up_slot_indices: torch.Tensor; up_slot_hidden: torch.Tensor
    output_group_offsets: torch.Tensor; input_group_offsets: torch.Tensor


def build_dual_route(layer: HFFNCompactG4Layer) -> HFFNDualRoute:
    """Build compact output/input ownership schedules from hidden-major slots."""
    def reverse(offsets: torch.Tensor, index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        count = torch.bincount(index.to(torch.int64), minlength=layer.d_model).to(torch.int32)
        out_offsets = torch.empty(layer.d_model + 1, dtype=torch.int32); out_offsets[0] = 0; out_offsets[1:] = torch.cumsum(count, 0)
        # `argsort` groups canonical slot ids by feature in native tensor code.
        # No per-edge Python fill: it would cost millions of scalar tensor conversions on a six-layer LLM export.
        slots64 = torch.argsort(index.to(torch.int64), stable=True)
        degrees = (offsets[1:] - offsets[:-1]).to(torch.int64)
        owner = torch.repeat_interleave(torch.arange(layer.d_ff, dtype=torch.int64), degrees)
        return out_offsets, slots64.to(torch.int32), owner.index_select(0, slots64).to(torch.int16)
    down_offsets, down_slots, down_hidden = reverse(layer.down_row_offsets, layer.down_idx)
    up_offsets, up_slots, up_hidden = reverse(layer.up_row_offsets, layer.up_idx)
    output_groups = _balanced_group_offsets(down_offsets[1:] - down_offsets[:-1])
    input_groups = _balanced_group_offsets(up_offsets[1:] - up_offsets[:-1])
    return HFFNDualRoute(down_offsets, down_slots, down_hidden, up_offsets, up_slots, up_hidden, output_groups, input_groups)


def load_hffn_compact_g4_layer(path: Path) -> HFFNCompactG4Layer:
    raw = read_layer(path.resolve(), groups=True)
    layer = HFFNCompactG4Layer(
        raw["d_model"], raw["d_ff"], raw["group_hidden_offsets"],
        raw["up_row_offsets"], raw["up_idx"], raw["up_val"], raw["b_up"],
        raw["down_row_offsets"], raw["down_idx"], raw["down_val"], raw["b_down"],
    ); layer.validate(); return layer


def _balanced_group_offsets(cost: torch.Tensor) -> torch.Tensor:
    total = int(cost.sum()); boundaries = [0]; cumulative = 0; target_group = 1
    for hidden, value in enumerate(cost.tolist()):
        cumulative += int(value)
        while target_group < G4_GROUPS and cumulative >= round(total * target_group / G4_GROUPS):
            boundaries.append(hidden + 1); target_group += 1
    while len(boundaries) < G4_GROUPS: boundaries.append(boundaries[-1])
    boundaries.append(cost.numel())
    return torch.tensor(boundaries, dtype=torch.int32)


def convert_padded_layer_to_compact(source: Path, destination: Path) -> dict[str, int]:
    """Losslessly flatten active padded rows; group boundaries balance active work."""
    padded = load_hffn_padded_training_layer(source)
    max_group = int(padded.b_up_pad.shape[1]); up_idx=[]; up_val=[]; down_idx=[]; down_val=[]; up_offsets=[0]; down_offsets=[0]; biases=[]; cost=[]
    for hidden in range(padded.d_ff):
        group, row = divmod(hidden, max_group); ud, dd = int(padded.up_deg[group,row]), int(padded.down_deg[group,row])
        up_idx.append(padded.up_idx_pad[group,row,:ud]); up_val.append(padded.up_val_pad[group,row,:ud]); down_idx.append(padded.down_idx_pad[group,row,:dd]); down_val.append(padded.down_val_pad[group,row,:dd])
        up_offsets.append(up_offsets[-1] + ud); down_offsets.append(down_offsets[-1] + dd); biases.append(padded.b_up_pad[group,row]); cost.append(ud + dd)
    out = {"format":"hffn_training_compact_g4_v2","d_model":padded.d_model,"d_ff":padded.d_ff,"group_hidden_offsets":_balanced_group_offsets(torch.tensor(cost)),"up_row_offsets":torch.tensor(up_offsets,dtype=torch.int32),"up_idx":torch.cat(up_idx).contiguous().to(torch.int16),"up_val":torch.cat(up_val).contiguous().float(),"b_up":torch.stack(biases).contiguous().float(),"down_row_offsets":torch.tensor(down_offsets,dtype=torch.int32),"down_idx":torch.cat(down_idx).contiguous().to(torch.int16),"down_val":torch.cat(down_val).contiguous().float(),"b_down":padded.b_down.contiguous().float()}
    destination.parent.mkdir(parents=True, exist_ok=True); torch.save(out, destination)
    return {"up_nnz":int(out["up_val"].numel()),"down_nnz":int(out["down_val"].numel())}
