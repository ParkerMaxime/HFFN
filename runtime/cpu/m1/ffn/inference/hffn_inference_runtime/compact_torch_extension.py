"""Build/load the inference extension, which exposes forward symbols only.

The recipe is the shared ``runtime/cpu/common/recipe/compact_extension.py`` (loaded by path from the repository
root); this leaf gives its constants (``build_entry.py``) and its two sources.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from .build_entry import COMPILER, FLAGS, LINK_ARGUMENTS, NAME, TORCH_LOAD

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "runtime" / "common" / "pythia" / "hffn_pythia").is_dir())


def _recipe():
    key = "hffn_cpu_compact_extension"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, _ROOT / "runtime/cpu/common/recipe/compact_extension.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


def load_hffn_compact_inference_extension(build_dir: Path):
    leaf = Path(__file__).resolve().parents[1]
    sources = [leaf / "torch_extension" / "hffn_compact_inference_torch_binding.cpp", leaf / "native" / "hffn_compact_g4_inference.cpp"]
    return _recipe().load_extension(build_dir, NAME, COMPILER, FLAGS, LINK_ARGUMENTS, TORCH_LOAD, sources, [])
