"""Load-time checks of the tables the forward of execution ``adaptive`` hands to the inference kernel of a leaf (x86: the
slots entry of the shared x86 kernel; M1: the dual entry of the compact G4 kernel). One copy for the CPU leaves; each
entry point (``forward_entry.py``) calls it once per layer where the runtime is built."""
from __future__ import annotations

import torch

from .compact_export import HFFNCompactG4Layer, HFFNDualRoute


@torch.no_grad()
def check_forward_route(index: int, layer: HFFNCompactG4Layer, route: HFFNDualRoute, threads: int) -> None:
    """Structural checks, at load, of the tables the dual entry of an inference kernel reads by pointer.

    The kernel trusts them: a wrong table gives a wrong sum, not an error. Checked once per layer, where the runtime is
    built, never in the forward: dtype, length and contiguity of every array passed by pointer; the down reverse schedule
    is a permutation of the hidden-major down slots (each slot read exactly once), grouped by output feature with offsets
    equal to the counts of each feature, hidden-major order kept inside a feature (stable sort), and each slot owned by
    the one hidden neuron whose row holds it; the G4 hidden partition and the output partition each have ``threads`` + 1
    monotone entries covering [0, d_ff] and [0, d_model].
    """
    def fail(what: str) -> None:
        raise RuntimeError(f"layer {index}: {what}")

    d_model, d_ff, n_down = layer.d_model, layer.d_ff, layer.down_val.numel()
    if d_ff > 32768:
        fail(f"d_ff {d_ff} does not fit the int16 owner of a down slot")
    arrays = {"group_hidden_offsets": (layer.group_hidden_offsets, torch.int32, threads + 1),
              "up_row_offsets": (layer.up_row_offsets, torch.int32, d_ff + 1), "up_idx": (layer.up_idx, torch.int16, layer.up_val.numel()),
              "up_val": (layer.up_val, torch.float32, layer.up_idx.numel()), "b_up": (layer.b_up, torch.float32, d_ff),
              "b_down": (layer.b_down, torch.float32, d_model), "down_val": (layer.down_val, torch.float32, n_down),
              "down_by_output_offsets": (route.down_by_output_offsets, torch.int32, d_model + 1),
              "down_slot_indices": (route.down_slot_indices, torch.int32, n_down), "down_slot_hidden": (route.down_slot_hidden, torch.int16, n_down),
              "output_group_offsets": (route.output_group_offsets, torch.int32, threads + 1)}
    for name, (tensor, dtype, length) in arrays.items():
        if tensor.device.type != "cpu" or tensor.dtype != dtype or tensor.dim() != 1 or tensor.numel() != length or not tensor.is_contiguous():
            fail(f"{name} is not a contiguous CPU {dtype} vector of {length} entries")
    counts = torch.bincount(layer.down_idx.long(), minlength=d_model)
    offsets = route.down_by_output_offsets.long()
    if int(offsets[0]) != 0 or int(offsets[-1]) != n_down or bool((offsets.diff() < 0).any()) or not torch.equal(offsets.diff(), counts):
        fail("down output offsets are not monotone or not the counts of the output features")
    slots = route.down_slot_indices.long()
    if n_down and (int(slots.min()) < 0 or int(slots.max()) >= n_down):
        fail("a down slot index is out of range")
    if not torch.equal(torch.bincount(slots, minlength=n_down), torch.ones(n_down, dtype=torch.int64)):
        fail("the down schedule is not a permutation of the down slots")
    feature = torch.repeat_interleave(torch.arange(d_model), counts)
    if not torch.equal(layer.down_idx.long().index_select(0, slots), feature):
        fail("a down slot is not in the segment of its output feature")
    same = feature[1:] == feature[:-1]
    if bool((slots[1:][same] <= slots[:-1][same]).any()):
        fail("hidden-major slot order not kept inside an output feature")
    owner = torch.repeat_interleave(torch.arange(d_ff), layer.down_row_offsets.diff().long())
    if not torch.equal(route.down_slot_hidden.long(), owner.index_select(0, slots)):
        fail("the owner of a down slot is not the hidden neuron whose row holds it")
    for name, table, end in (("G4 hidden partition", layer.group_hidden_offsets, d_ff), ("output partition", route.output_group_offsets, d_model)):
        if int(table[0]) != 0 or int(table[-1]) != end or bool((table.diff() < 0).any()):
            fail(f"{name} is not monotone over [0, {end}]")
