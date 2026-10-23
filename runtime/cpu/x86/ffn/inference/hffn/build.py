"""Compile the compact kernel into ``build/`` and return the shared library path.

The kernel (AVX2/FMA with a scalar fallback) is one source file, ``common/kernels/hffn_compact_g4_inference.cpp``,
found as the first ancestor folder of this leaf that holds ``common/kernels/<file>``. It only needs g++.

Environment overrides: ``HFFN_CXX`` (compiler, default g++), ``HFFN_CXXFLAGS`` (extra flags;
``-mavx2 -mfma`` or ``-march=native`` enable the AVX2 paths).
"""
from __future__ import annotations

import hashlib
import os
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"

COMPACT = "hffn_compact_g4_inference.cpp"

BASE_FLAGS = ["-O3", "-ffast-math", "-std=c++17", "-fPIC", "-shared", "-pthread"]


def source(name: str = COMPACT) -> Path:
    """The kernel source: ``common/kernels/<name>`` of the nearest ancestor that has it."""
    for folder in ROOT.parents:
        candidate = folder / "common" / "kernels" / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"common/kernels/{name} not found above {ROOT}")


def compiler() -> str:
    return os.environ.get("HFFN_CXX") or "g++"


def extra_flags() -> list[str]:
    return shlex.split(os.environ.get("HFFN_CXXFLAGS", ""))


def build(name: str = COMPACT) -> Path:
    cpp = source(name)
    command = [compiler(), *BASE_FLAGS, *extra_flags(), str(cpp)]
    tag = hashlib.sha256((" ".join(command) + cpp.read_text()).encode()).hexdigest()[:12]
    lib = BUILD / f"{cpp.stem}.{tag}.so"
    if not lib.exists():
        BUILD.mkdir(exist_ok=True)
        subprocess.run([*command, "-o", str(lib)], check=True)
    return lib
