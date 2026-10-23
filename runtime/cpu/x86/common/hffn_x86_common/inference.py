"""Build and ctypes entries of the shared x86 inference kernels.

- kernel: ``runtime/cpu/x86/common/kernels/hffn_compact_g4_inference.cpp``, read where it is;
- training N = 1 sidecar: ``hffn_compact_g4_inference_n1.cpp``; it includes the shared kernel source and exports only
  the materializing entry;
- libraries: ``runtime/cpu/x86/common/build/``, one file per source and flag set (tag of the content).

One flag set for every kernel built from this package. A leaf that builds another source passes its own build
directory to ``build``.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = next(p for p in HERE.parents if (p / "runtime" / "common" / "pythia" / "hffn_pythia").is_dir())
COMMON = ROOT / "runtime/cpu/x86/common"
BUILD = COMMON / "build"

INFERENCE = COMMON / "kernels/hffn_compact_g4_inference.cpp"
INFERENCE_N1 = COMMON / "kernels/hffn_compact_g4_inference_n1.cpp"

FLAGS = ["-O3", "-ffast-math", "-mno-recip", "-mavx2", "-mfma", "-std=c++17", "-fPIC", "-shared", "-pthread"]
N1_FLAGS = ["-fvisibility=hidden", "-ffunction-sections", "-fdata-sections", "-Wl,--gc-sections"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source: Path, build_dir: Path, extra_flags: tuple[str, ...] = (), dependencies: tuple[Path, ...] = ()) -> Path:
    command = ["g++", *FLAGS, *extra_flags, str(source)]
    content = source.read_text() + "".join(path.read_text() for path in dependencies)
    tag = hashlib.sha256((" ".join(command) + content).encode()).hexdigest()[:12]
    lib = build_dir / f"{source.stem}.{tag}.so"
    if not lib.exists():
        build_dir.mkdir(exist_ok=True)
        partial = lib.with_suffix(f".{os.getpid()}.tmp")
        subprocess.run([*command, "-o", str(partial)], check=True)
        partial.rename(lib)
    return lib


def inference_lib() -> Path:
    return build(INFERENCE, BUILD)


def inference_n1_lib() -> Path:
    return build(INFERENCE_N1, BUILD, tuple(N1_FLAGS), (INFERENCE,))


def sources() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha256(path) for path in (INFERENCE, INFERENCE_N1)}


_P, _I, _F, _L = ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_int64


def _bind(lib: ctypes.CDLL, name: str, argtypes: list) -> ctypes._CFuncPtr:
    function = getattr(lib, name)
    function.argtypes, function.restype = argtypes, None
    return function


class InferenceKernel:
    """The slots entry of the dual route of the shared x86 inference kernel."""

    def __init__(self, path: Path, n1_path: Path | None = None):
        self._lib = ctypes.CDLL(str(path))
        self.forward_dual_slots_nt = _bind(self._lib, "hffn_compact_forward_dual_slots_nt_f32", [_I] + [_P] * 12 + [_I] * 3 + [_P] * 6)
        self._n1_lib = None
        if n1_path is not None:
            self._n1_lib = ctypes.CDLL(str(n1_path))
            self.forward_dual_slots_n1_nt = _bind(self._n1_lib, "hffn_compact_forward_dual_slots_n1_nt_f32", [_I] + [_P] * 12 + [_I] * 3 + [_P] * 6)
