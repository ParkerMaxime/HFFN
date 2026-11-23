"""Nvidia model training: persistent contexts, existing FFN kernels."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "runtime" / "common" / "pythia" / "hffn_pythia").is_dir())
for _relative in (
    "runtime/common/pythia",
    "runtime/gpu/nvidia/common/kernels",
    "runtime/gpu/nvidia/ffn/training",
):
    _path = str(_ROOT / _relative)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from .context_runtime import ContextTrainRuntime  # noqa: E402
from .model import HFFNTrainingMLP, HFFNTrainingOptimizer, create_optimizer, load_pythia  # noqa: E402

__all__ = ["ContextTrainRuntime", "HFFNTrainingMLP", "HFFNTrainingOptimizer", "create_optimizer", "load_pythia"]
