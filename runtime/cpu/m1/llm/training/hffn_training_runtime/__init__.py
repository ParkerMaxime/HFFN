"""HFFN compact-G4 training runtime of the M1 leaf (fail-closed).

This leaf has no padded runtime; a padded export or training checkpoint stays readable through
the shared ``padded_export`` and ``padded_import`` modules, and a padded export converts to the compact format with
``compact_export.convert_padded_layer_to_compact``.
"""

from pathlib import Path

# compact_model.py and compact_direct_loader.py exist once for x86 and M1, at the shared level: they join this package by its path.
__path__.append(str(next(p for p in Path(__file__).resolve().parents if (p / "runtime" / "common" / "pythia" / "hffn_pythia").is_dir())
                    / "runtime/cpu/common/llm/training/hffn_training_runtime"))

from .padded_export import HFFNPaddedTrainingLayer, load_hffn_padded_training_layer
from .compact_export import HFFNCompactG4Layer, load_hffn_compact_g4_layer
from .compact_runtime import HFFNCompactG4TrainingRuntime
from .compact_model import HFFNCompactG4Optimizer, install_pythia_hffn_compact_g4_mlps
from .compact_direct_loader import load_pythia_hffn_compact_g4_for_training
from .masked_reference import (
    FixedSupportAdamW,
    build_fixed_support_pythia_adamw,
    install_compact_hffn_values_in_dense_pythia,
)

__all__ = [
    "HFFNPaddedTrainingLayer",
    "HFFNTrainingOptimizer",
    "install_pythia_hffn_mlps",
    "load_pythia_hffn_for_training",
    "HFFNCompactG4Layer",
    "HFFNCompactG4TrainingRuntime",
    "HFFNCompactG4Optimizer",
    "load_hffn_compact_g4_layer",
    "install_pythia_hffn_compact_g4_mlps",
    "load_pythia_hffn_compact_g4_for_training",
    "FixedSupportAdamW",
    "build_fixed_support_pythia_adamw",
    "install_compact_hffn_values_in_dense_pythia",
    "load_hffn_padded_training_layer",
]

# Publication-facing defaults: compact hidden-major parameters, gradients and AdamW state; fixed G4 workers only define
# execution ownership.
HFFNTrainingOptimizer = HFFNCompactG4Optimizer
install_pythia_hffn_mlps = install_pythia_hffn_compact_g4_mlps
load_pythia_hffn_for_training = load_pythia_hffn_compact_g4_for_training
