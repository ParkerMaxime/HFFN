"""Builds of the two kernels of the leaf into ``build/`` of the leaf.

- inference kernel: ``runtime/cpu/x86/common/kernels/hffn_compact_g4_inference.cpp``, built and read by
  ``hffn_x86_common.inference`` (its library goes to ``common/build/``);
- training kernel: ``runtime/cpu/common/llm/training/native/hffn_compact_g4_training.cpp`` (one copy for x86 and
  M1), read where it is.

One flag set for both, the one of ``hffn_x86_common.inference``. It adds ``-mavx2 -mfma`` to the build of the training
kernel in its own leaf, which has no AVX2 flag; this leaf is judged with them.
"""
from __future__ import annotations

from pathlib import Path

from hffn_x86_common.inference import FLAGS, INFERENCE, build, inference_lib, sha256  # noqa: F401

HERE = Path(__file__).resolve()
LEAF = HERE.parents[1]
ROOT = next(p for p in HERE.parents if (p / "runtime" / "common" / "pythia" / "hffn_pythia").is_dir())
BUILD = LEAF / "build"

TRAINING = ROOT / "runtime/cpu/common/llm/training/native/hffn_compact_g4_training.cpp"
# Included by the training kernel, so part of its build tag.
HEADER = TRAINING.parent / "g4_contract.h"


def training_lib() -> Path:
    return build(TRAINING, BUILD, dependencies=(HEADER,))


def sources() -> dict[str, str]:
    return {str(p.relative_to(ROOT)): sha256(p) for p in (INFERENCE, TRAINING)}
