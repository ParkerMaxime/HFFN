"""HFFN compact-G4 training runtime for Pythia on x86_64: compact hidden-major parameters, gradients and AdamW
state; four fixed groups of hidden neurons define execution ownership. A port of the CPU M1 runtime."""
from pathlib import Path

# compact_model.py and compact_direct_loader.py exist once for x86 and M1, at the shared level: they join this package by its path.
__path__.append(str(next(p for p in Path(__file__).resolve().parents if (p / "runtime" / "common" / "pythia" / "hffn_pythia").is_dir())
                    / "runtime/cpu/common/llm/training/hffn_training_runtime"))

from .compact_export import HFFNCompactG4Layer, load_hffn_compact_g4_layer
from .compact_runtime import HFFNCompactG4TrainingRuntime
from .compact_model import HFFNCompactG4Optimizer, install_pythia_hffn_compact_g4_mlps
from .compact_direct_loader import load_pythia_hffn_compact_g4_for_training
from .masked_reference import FixedSupportAdamW, build_fixed_support_pythia_adamw, install_compact_hffn_values_in_dense_pythia

__all__ = [
    "HFFNCompactG4Layer", "HFFNCompactG4TrainingRuntime", "HFFNCompactG4Optimizer", "load_hffn_compact_g4_layer",
    "install_pythia_hffn_compact_g4_mlps", "load_pythia_hffn_compact_g4_for_training",
    "FixedSupportAdamW", "build_fixed_support_pythia_adamw", "install_compact_hffn_values_in_dense_pythia",
]

HFFNTrainingOptimizer = HFFNCompactG4Optimizer
install_pythia_hffn_mlps = install_pythia_hffn_compact_g4_mlps
load_pythia_hffn_for_training = load_pythia_hffn_compact_g4_for_training
