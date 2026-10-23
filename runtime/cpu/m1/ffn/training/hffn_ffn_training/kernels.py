"""Entry point of the M1 leaf for the shared ``step.py``: the dual entry of the compact G4 inference kernel, which
stores ``z`` and ``h`` at every N >= 1. The binding is the one of the inference leaf
(``hffn_inference_runtime.dual_entry``)."""
from __future__ import annotations

import ctypes
import sys
from pathlib import Path

# The compact inference leaf is `runtime/cpu/m1/llm/inference/compact`.
_INFERENCE_LEAF = next(p for p in Path(__file__).resolve().parents if (p / "runtime" / "common" / "pythia" / "hffn_pythia").is_dir()) / "runtime/cpu/m1/llm/inference/compact"
if str(_INFERENCE_LEAF) not in sys.path:
    sys.path.insert(0, str(_INFERENCE_LEAF))

from hffn_inference_runtime.dual_entry import INFERENCE_MIN_N, InferenceKernel  # noqa: E402,F401

_P, _I, _F, _L = ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_int64


def _bind(lib: ctypes.CDLL, name: str, argtypes: list) -> ctypes._CFuncPtr:
    function = getattr(lib, name)
    function.argtypes, function.restype = argtypes, None
    return function
