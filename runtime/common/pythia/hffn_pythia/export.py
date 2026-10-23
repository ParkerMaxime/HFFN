"""Reader of the ``hffn_training_compact_g4_v2`` exports (one file per Pythia FFN).

An export stores a fixed-support FFN per hidden neuron: for neuron ``r``, its active incoming
connections ``up_idx/up_val[up_row_offsets[r]:up_row_offsets[r + 1]]``, its bias ``b_up[r]``
and its active outgoing connections ``down_idx/down_val[down_row_offsets[r]:...]``.
"""
from __future__ import annotations

from pathlib import Path

import torch

FORMAT = "hffn_training_compact_g4_v2"
_TYPES = {
    "up_row_offsets": torch.int32, "up_idx": torch.int16, "up_val": torch.float32, "b_up": torch.float32,
    "down_row_offsets": torch.int32, "down_idx": torch.int16, "down_val": torch.float32, "b_down": torch.float32,
}


def read_layer(path: Path, *, groups: bool = False) -> dict:
    """The validated arrays of one export file, as contiguous CPU tensors.

    With ``groups``, also ``group_hidden_offsets``: the partition of the hidden neurons into the four groups of the CPU
    G4 runtimes (int32[5], from 0 to ``d_ff``, monotonic)."""
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if raw.get("format") != FORMAT:
        raise ValueError(f"{path}: expected a {FORMAT} export")
    layer: dict = {"d_model": int(raw["d_model"]), "d_ff": int(raw["d_ff"])}
    for name, dtype in _TYPES.items():
        if raw[name].dtype != dtype:
            raise ValueError(f"{path}: {name} must be {dtype}")
        layer[name] = raw[name].contiguous()
    h, f = layer["d_model"], layer["d_ff"]
    if layer["b_up"].numel() != f or layer["b_down"].numel() != h:
        raise ValueError(f"{path}: bias dimensions do not match d_model/d_ff")
    for side in ("up", "down"):
        offsets, index, values = layer[f"{side}_row_offsets"], layer[f"{side}_idx"], layer[f"{side}_val"]
        if offsets.numel() != f + 1 or int(offsets[0]) != 0 or int(offsets[-1]) != values.numel() or index.numel() != values.numel():
            raise ValueError(f"{path}: invalid {side} hidden-major storage")
        if bool((offsets.diff() < 0).any()) or (index.numel() and (int(index.min()) < 0 or int(index.max()) >= h)):
            raise ValueError(f"{path}: invalid {side} offsets or feature index")
    if groups:
        partition = raw["group_hidden_offsets"]
        if partition.dtype != torch.int32 or tuple(partition.shape) != (5,) or int(partition[0]) != 0 or int(partition[-1]) != f \
                or bool((partition.diff() < 0).any()):
            raise ValueError(f"{path}: invalid group_hidden_offsets")
        layer["group_hidden_offsets"] = partition.contiguous()
    return layer


def read_layers(export_dir: str | Path, count: int) -> list[dict]:
    return [read_layer(Path(export_dir) / f"layer{index}" / f"{FORMAT}.pt") for index in range(count)]


def hidden_of_slot(layer: dict, side: str) -> torch.Tensor:
    """Hidden neuron owning each compact slot of one side (int64[nnz])."""
    return torch.repeat_interleave(torch.arange(layer["d_ff"]), layer[f"{side}_row_offsets"].diff().long())
