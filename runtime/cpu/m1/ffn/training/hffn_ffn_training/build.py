"""The two kernels of the leaf.

- inference kernel: the library that the compact inference leaf builds (``runtime/cpu/m1/llm/inference/compact``,
  in its own ``build/``), loaded as it is: no second build of the forward;
- training kernel: ``runtime/cpu/common/llm/training/native/hffn_compact_g4_training.cpp`` (one copy for x86 and
  M1), read where it is and built into ``build/`` of this leaf with the flags of the M1 training leaf
  (``clang++ -O3 -ffast-math -std=c++17 -fPIC -dynamiclib -pthread``); it includes no PyTorch header.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
LEAF = HERE.parents[1]
ROOT = next(p for p in HERE.parents if (p / "runtime" / "common" / "pythia" / "hffn_pythia").is_dir())
BUILD = LEAF / "build"

INFERENCE_LEAF = ROOT / "runtime/cpu/m1/llm/inference/compact"
INFERENCE = INFERENCE_LEAF / "native/hffn_compact_g4_inference.cpp"
TRAINING = ROOT / "runtime/cpu/common/llm/training/native/hffn_compact_g4_training.cpp"
# Included by the training kernel, so part of its build tag.
HEADER = TRAINING.parent / "g4_contract.h"
FLAGS = ["-O3", "-ffast-math", "-std=c++17", "-fPIC", "-dynamiclib", "-pthread"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inference_lib() -> Path:
    if str(INFERENCE_LEAF) not in sys.path:
        sys.path.insert(0, str(INFERENCE_LEAF))
    from hffn_inference_runtime.dual_entry import inference_library
    return inference_library()


def training_lib() -> Path:
    command = ["clang++", *FLAGS, str(TRAINING)]
    tag = hashlib.sha256((" ".join(command) + TRAINING.read_text() + HEADER.read_text()).encode()).hexdigest()[:12]
    lib = BUILD / f"{TRAINING.stem}.{tag}.dylib"
    if not lib.exists():
        BUILD.mkdir(exist_ok=True)
        partial = lib.with_suffix(f".{os.getpid()}.tmp")
        subprocess.run([*command, "-o", str(partial)], check=True)
        partial.rename(lib)
    return lib


def sources() -> dict[str, str]:
    return {str(p.relative_to(ROOT)): sha256(p) for p in (INFERENCE, TRAINING)}
