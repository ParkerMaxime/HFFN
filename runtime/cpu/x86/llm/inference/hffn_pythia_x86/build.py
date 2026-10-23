"""Compile the kernel into ``build/`` and return the shared library path.

The kernel is the source shared by the x86 leaves, ``common/kernels/hffn_compact_g4_inference.cpp``, found as the
first ancestor folder of this leaf that holds ``common/kernels/<file>``. This leaf builds it with
``-DHFFN_GROUPS_OPENMP -fopenmp``: the groups of hidden neurons run on the OpenMP team of the process, which is
PyTorch's when the library is loaded after it.

The build needs g++ (>= 9) and a CPU with AVX2 and FMA: ``-mavx2 -mfma`` is always passed, there is no scalar
fallback in this leaf, and a CPU without them stops the process with an illegal instruction.

The library name carries a hash of the command and of the source: changing either recompiles.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL = "hffn_compact_g4_inference.cpp"
BUILD = ROOT / "build"
COMMAND = ["g++", "-O3", "-ffast-math", "-std=c++17", "-fPIC", "-shared", "-fopenmp", "-mavx2", "-mfma",
           "-DHFFN_GROUPS_OPENMP"]      # the groups run on the OpenMP team of the process (PyTorch's)


def _source() -> Path:
    for folder in ROOT.parents:
        candidate = folder / "common" / "kernels" / KERNEL
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"common/kernels/{KERNEL} not found above {ROOT}")


SOURCE = _source()


def build() -> Path:
    tag = hashlib.sha256((" ".join(COMMAND) + SOURCE.read_text()).encode()).hexdigest()[:12]
    library = BUILD / f"{SOURCE.stem}.{tag}.so"
    if not library.exists():
        BUILD.mkdir(exist_ok=True)
        subprocess.run([*COMMAND, str(SOURCE), "-o", str(library)], check=True)
    return library
