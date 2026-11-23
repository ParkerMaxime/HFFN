"""Forward-only Pythia runtime for compact hidden-major HFFN exports."""

import sys
from pathlib import Path

# The Pythia modules (export reader, compact layer) exist once, at the shared level: first ancestor that has common/pythia/hffn_pythia.
_PYTHIA = next(p / "common/pythia" for p in Path(__file__).resolve().parents if (p / "common/pythia/hffn_pythia").is_dir())
if str(_PYTHIA) not in sys.path:
    sys.path.insert(0, str(_PYTHIA))

from hffn_pythia.compact_format import HFFNCompactG4Layer, load_hffn_compact_g4_layer
from .compact_runtime import HFFNCompactG4InferenceRuntime
from .compact_model import HFFNCompactG4InferenceMLP, install_pythia_hffn_compact_g4_mlps
from .compact_inference_loader import load_pythia_hffn_for_inference

__all__ = [
    "HFFNCompactG4Layer", "HFFNCompactG4InferenceRuntime",
    "HFFNCompactG4InferenceMLP", "install_pythia_hffn_compact_g4_mlps",
    "load_hffn_compact_g4_layer", "load_pythia_hffn_for_inference",
]
