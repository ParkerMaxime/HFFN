"""The two kernels of the leaf.

- inference kernel: the library that the compact inference leaf builds (``runtime/cpu/m1/ffn/inference``,
  in its own ``build/``), loaded as it is: no second build of the forward;
- training kernel: ``runtime/cpu/common/llm/training/native/hffn_compact_g4_training.cpp`` (one copy for x86 and
  M1), read where it is and built into ``build/`` of this leaf with the flags of the M1 training leaf
  (``clang++ -O3 -ffast-math -std=c++17 -fPIC -dynamiclib -pthread``); it includes no PyTorch header.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
LEAF = HERE.parents[1]
ROOT = next(p for p in HERE.parents if (p / "runtime" / "common" / "pythia" / "hffn_pythia").is_dir())
BUILD = LEAF / "build"

INFERENCE_LEAF = ROOT / "runtime/cpu/m1/ffn/inference"
INFERENCE = INFERENCE_LEAF / "native/hffn_compact_g4_inference.cpp"
TRAINING = ROOT / "runtime/cpu/common/llm/training/native/hffn_compact_g4_training.cpp"
# Included by the training kernel, so part of its build tag.
HEADER = TRAINING.parent / "g4_contract.h"
# The macros select the token-contiguous ``grad_up`` and down gradient of the backward and the block of eight slots of the
# down gradient; only this M1 build defines them.
FLAGS = ["-O3", "-ffast-math", "-std=c++17", "-fPIC", "-dynamiclib", "-pthread", "-DHFFN_G4_UP_TOKEN_CONTIGUOUS", "-DHFFN_G4_DOWN_TOKEN_CONTIGUOUS", "-DHFFN_G4_DOWN_BLOCK8"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inference_lib() -> Path:
    if str(INFERENCE_LEAF) not in sys.path:
        sys.path.insert(0, str(INFERENCE_LEAF))
    from hffn_inference_runtime.dual_entry import inference_library
    return inference_library()


def _recipe():
    key = "hffn_cpu_plain_dylib"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, ROOT / "runtime/cpu/common/recipe/plain_dylib.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


def training_lib() -> Path:
    return _recipe().build(TRAINING, BUILD, "clang++", FLAGS, "dylib", dependencies=(HEADER,))


def sources() -> dict[str, str]:
    return {str(p.relative_to(ROOT)): sha256(p) for p in (INFERENCE, TRAINING)}
