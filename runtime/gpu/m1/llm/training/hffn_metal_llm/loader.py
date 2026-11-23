"""Direct Pythia loader: compact FFNs plus PyTorch/MPS for all other state."""

from __future__ import annotations

import gc
from pathlib import Path

import torch
from torch import nn

from .model import HFFNMetalMLP, install_pythia_hffn_metal_mlps
from hffn_pythia.model import build_pythia  # on sys.path once `.model` (its `.compact`) is imported


def load_pythia_hffn_metal_for_training(
    model_dir: str | Path,
    export_dir: str | Path,
    *,
    native_n: tuple[int, ...],
    library: str | Path,
) -> tuple[nn.Module, tuple[HFFNMetalMLP, ...]]:
    model = build_pythia(model_dir, dense_ffn=False)
    modules = install_pythia_hffn_metal_mlps(model, export_dir, native_n=native_n, library=library)
    gc.collect()
    model.to(device="mps", dtype=torch.float32)
    return model.train(), modules
