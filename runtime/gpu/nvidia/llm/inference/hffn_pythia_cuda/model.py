"""Pythia with its FFNs executed dense or by the CUDA HFFN runtime: the shared loader, bound to this leaf."""
from __future__ import annotations

from pathlib import Path

from torch import nn

from hffn_pythia import model as shared


def layer_from_export(layer: dict):
    """The kernel arrays of one export layer (``hffn_pythia.export.read_layer``)."""
    from hffn_cuda.runtime import HFFNLayer  # the dense arm needs neither CuPy nor the kernels
    return HFFNLayer.from_hidden_major(layer["d_model"], layer["d_ff"], layer["b_up"], layer["up_row_offsets"], layer["up_idx"],
                                       layer["up_val"], layer["b_down"], layer["down_row_offsets"], layer["down_idx"], layer["down_val"])


def _runtime(layers: list[dict], device: str):
    from .prepared_runtime import PreparedRuntime  # the decode FFN call launched from a descriptor (HFFNRuntime for the other cases)
    return PreparedRuntime([layer_from_export(layer) for layer in layers], device)


def load_pythia(model_dir: str | Path, export_dir: str | Path, ffn: str, device: str = "cuda") -> nn.Module:
    """Pythia in eval mode (float32, on ``device``) with its FFNs from ``export_dir``, executed dense or by HFFN.

    The HFFN arm exposes its runtime as ``model.hffn_runtime`` (None in the dense arm).
    """
    return shared.load_pythia(model_dir, export_dir, ffn, lambda layers: _runtime(layers, device), device)
