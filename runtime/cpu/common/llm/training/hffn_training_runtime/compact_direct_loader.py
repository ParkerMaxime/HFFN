"""Direct Pythia loader for compact-G4 HFFN exports."""
from __future__ import annotations
import gc
from pathlib import Path
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM
from .compact_model import install_pythia_hffn_compact_g4_mlps

def load_pythia_hffn_compact_g4_for_training(model_dir:str|Path,export_dir:str|Path):
 root=Path(model_dir); checkpoint=root/"model.safetensors"
 if not checkpoint.is_file(): raise ValueError("compact direct loader requires model.safetensors")
 model=AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(root),dtype=torch.float32); runtime=install_pythia_hffn_compact_g4_mlps(model,export_dir); targets=model.state_dict(); loaded=set()
 with torch.no_grad(),safe_open(checkpoint,framework="pt",device="cpu") as source:
  for name in source.keys():
   if ".mlp." in name: continue
   if name not in targets: raise ValueError(f"unexpected non-FFN checkpoint tensor {name}")
   value=source.get_tensor(name)
   if tuple(value.shape)!=tuple(targets[name].shape) or value.dtype!=targets[name].dtype: raise ValueError(f"tensor mismatch {name}")
   targets[name].copy_(value);loaded.add(name)
 missing=[name for name in targets if ".mlp." not in name and name not in loaded]
 if missing: raise ValueError(f"missing non-FFN tensors: {missing}")
 del targets;gc.collect();return model.train(),runtime
