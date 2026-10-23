"""The only file of the update gate that touches the runtime: the six seams named below.

After a change of the training runtime this file is the one to edit. A seam that moved fails loudly (AttributeError,
ImportError), which `check.sh` reports as FAILED. Nothing here is skipped silently.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, os.environ["HFFN_LEAF"])  # the leaf under check, given by its check.sh

import parity_model as base  # noqa: E402,F401  (flat_support, dense_gradients, dense_model, CONFIGURATIONS, ratio)
from hffn_training_runtime import (HFFNCompactG4Optimizer, build_fixed_support_pythia_adamw,  # noqa: E402,F401
                                   load_pythia_hffn_compact_g4_for_training)

CONFIGURATIONS = base.CONFIGURATIONS


def load(model_dir, export_dir, name=None):
    """(model, runtime) for the retained inference forward; `name` identifies the single HFFN arm."""
    options = CONFIGURATIONS[name] if name else {}
    return load_pythia_hffn_compact_g4_for_training(model_dir, export_dir, **options)


def optimizer(model, runtime, lr, weight_decay):
    return HFFNCompactG4Optimizer(model, runtime, lr=lr, weight_decay=weight_decay)


def active_weights(model):
    return [tuple(t.detach() for t in (b.mlp.up_val, b.mlp.down_val, b.mlp.b_up, b.mlp.b_down)) for b in model.gpt_neox.layers]


def active_moments(runtime):
    """(first, second) per layer: the only place that reads the private `_moments`."""
    return ([(m.um, m.dm, m.bum, m.bdm) for m in runtime._moments], [(m.uv, m.dv, m.buv, m.bdv) for m in runtime._moments])


def supports(model):
    return [base.flat_support(b.mlp) for b in model.gpt_neox.layers]


def support_checksum(flats) -> int:
    """Order-sensitive checksum of the flat supports: the compact storage order and the gradient order must agree."""
    total = 0
    for pair in flats:
        for flat in pair:
            if flat is not None:
                total = (total * 1000003 + int((flat.long() * torch.arange(1, flat.numel() + 1)).sum() % 2**61)) % 2**61
    return total
