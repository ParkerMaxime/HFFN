"""Direct Pythia loader: compact FFNs plus PyTorch/MPS for all other state."""

from __future__ import annotations

import gc
from pathlib import Path

import torch
from torch import nn
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM

from .model import HFFNMetalMLP, install_pythia_hffn_metal_mlps


def load_pythia_hffn_metal_for_training(
    model_dir: str | Path,
    export_dir: str | Path,
) -> tuple[nn.Module, tuple[HFFNMetalMLP, ...]]:
    root = Path(model_dir)
    checkpoint = root / "model.safetensors"
    if not checkpoint.is_file():
        raise ValueError("direct Metal loader requires model.safetensors")
    model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(root), dtype=torch.float32)
    modules = install_pythia_hffn_metal_mlps(model, export_dir)
    targets = model.state_dict()
    loaded: set[str] = set()
    with torch.no_grad(), safe_open(checkpoint, framework="pt", device="cpu") as source:
        for name in source.keys():
            if ".mlp." in name:
                continue
            if name not in targets:
                if name.endswith((".attention.bias", ".attention.masked_bias", ".attention.rotary_emb.inv_freq")):
                    continue
                raise ValueError(f"unexpected non-FFN checkpoint tensor {name}")
            value = source.get_tensor(name)
            target = targets[name]
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(f"tensor shape mismatch {name}: {tuple(value.shape)} != {tuple(target.shape)}")
            if value.dtype != target.dtype:
                if not value.is_floating_point() or not target.is_floating_point():
                    raise ValueError(f"tensor dtype mismatch {name}: {value.dtype} != {target.dtype}")
                value = value.to(target.dtype)
            target.copy_(value)
            loaded.add(name)
    missing = [name for name in targets if ".mlp." not in name and name not in loaded]
    if missing:
        raise ValueError(f"missing non-FFN tensors: {missing}")
    del targets
    gc.collect()
    model.to(device="mps", dtype=torch.float32)
    return model.train(), modules
