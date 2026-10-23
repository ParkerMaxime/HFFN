"""Direct Pythia loader that reads non-FFN weights and v2 HFFN payloads only."""
from __future__ import annotations

import gc
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM

from .compact_model import install_pythia_hffn_compact_g4_mlps


def load_pythia_hffn_for_inference(model_dir: str | Path, export_dir: str | Path, *, execution: str = "baseline", batch_split32: bool = True, batch_split16: bool = False, batch_split16_t96: bool = False, batch_split_t96_size: int = 96):
    root = Path(model_dir)
    checkpoint = root / "model.safetensors"
    if not checkpoint.is_file():
        raise ValueError("direct inference loader requires model.safetensors")
    model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(root), dtype=torch.float32)
    runtime = install_pythia_hffn_compact_g4_mlps(model, export_dir, execution=execution, batch_split32=batch_split32, batch_split16=batch_split16, batch_split16_t96=batch_split16_t96, batch_split_t96_size=batch_split_t96_size)
    targets, loaded = model.state_dict(), set()
    with torch.no_grad(), safe_open(checkpoint, framework="pt", device="cpu") as source:
        for name in source.keys():
            if ".mlp." in name:
                continue
            if name not in targets:
                if name.endswith((".attention.bias", ".attention.masked_bias", ".attention.rotary_emb.inv_freq")):
                    continue
                raise ValueError(f"unexpected non-FFN checkpoint tensor {name}")
            value = source.get_tensor(name)
            if tuple(value.shape) != tuple(targets[name].shape):
                raise ValueError(f"tensor shape mismatch for {name}")
            if value.dtype != targets[name].dtype:
                if not value.is_floating_point() or not targets[name].is_floating_point():
                    raise ValueError(f"tensor dtype mismatch for {name}")
                value = value.to(dtype=targets[name].dtype)
            targets[name].copy_(value)
            loaded.add(name)
    missing = [name for name in targets if ".mlp." not in name and name not in loaded]
    if missing:
        raise ValueError(f"missing non-FFN tensors: {missing}")
    del targets
    gc.collect()
    return model.eval(), runtime
