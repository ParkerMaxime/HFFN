"""Direct Pythia loader: compact FFNs and PyTorch/MPS for all other state."""

from __future__ import annotations

import gc
from pathlib import Path

import torch

from .model import install_pythia_hffn_metal_inference_mlps
from hffn_pythia.model import build_pythia  # on sys.path once `.model` (its `.compact`) is imported


def load_pythia_hffn_metal_for_inference(
    model_dir: str | Path,
    export_dir: str | Path,
):
    model = build_pythia(model_dir, dense_ffn=False)
    runtime, modules = install_pythia_hffn_metal_inference_mlps(model, export_dir)
    gc.collect()
    model.to(device="mps", dtype=torch.float32)
    model.requires_grad_(False)
    model.eval()
    runtime.attach_model(model)
    return model, runtime, modules
