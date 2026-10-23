"""Pythia with its FFNs executed dense or by the x86 HFFN runtime: the shared loader, bound to this leaf."""
from __future__ import annotations

from pathlib import Path

from torch import nn

from hffn_pythia import model as shared

from .runtime import HFFNLayer, HFFNRuntime


def load_pythia(model_dir: str | Path, export_dir: str | Path, ffn: str, threads: int | None = None) -> nn.Module:
    """Pythia in eval mode (float32, CPU) with its FFNs from ``export_dir``, executed dense or by HFFN.

    ``threads`` is the HFFN thread count (default: the CPUs of the process, at most 8); PyTorch
    keeps its own setting (``torch.set_num_threads``).
    """
    return shared.load_pythia(model_dir, export_dir, ffn,
                              lambda layers: HFFNRuntime([HFFNLayer.from_export(layer) for layer in layers], threads))
