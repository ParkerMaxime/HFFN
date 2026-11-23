"""Direct Pythia loader for compact-G4 HFFN exports."""
from __future__ import annotations
import gc
from pathlib import Path
from hffn_pythia.model import build_pythia
from .compact_model import install_pythia_hffn_compact_g4_mlps

def load_pythia_hffn_compact_g4_for_training(model_dir:str|Path,export_dir:str|Path):
 """The shared construction without a dense FFN (``build_pythia``: non-FFN tensors loaded from the base checkpoint),
 then the compact HFFN modules of ``export_dir``."""
 model=build_pythia(model_dir,dense_ffn=False); runtime=install_pythia_hffn_compact_g4_mlps(model,export_dir)
 gc.collect();return model.train(),runtime
