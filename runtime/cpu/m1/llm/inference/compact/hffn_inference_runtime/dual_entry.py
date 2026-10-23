"""The dual entry ``hffn_compact_g4_forward_dual_nt_f32`` of the compact G4 inference kernel, bound once for the callers
that need ``z`` and ``h`` stored (the training leaves of the M1 CPU). The library is the one this leaf builds for its
own runtime, in its ``build/``: no second build of the forward."""
from __future__ import annotations

import ctypes
from pathlib import Path

from .compact_torch_extension import load_hffn_compact_inference_extension

LEAF = Path(__file__).resolve().parents[1]
INFERENCE_MIN_N = 1  # the dual entry has no branch on N and stores z and h at N = 1

_I, _P = ctypes.c_int, ctypes.c_void_p


def inference_library() -> Path:
    """The library of the inference leaf, built if needed and loaded once."""
    return Path(load_hffn_compact_inference_extension(LEAF / "build").__file__)


class InferenceKernel:
    """The dual entry of the compact inference kernel, under the call the shared training code makes."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lib = ctypes.CDLL(str(path))
        self._dual = self._lib.hffn_compact_g4_forward_dual_nt_f32
        self._dual.argtypes, self._dual.restype = [_P] * 12 + [_I] * 3 + [_P] * 6, None

    def forward_dual_slots_nt(self, threads, x, hidden_groups, b_up, up_offsets, up_idx, up_val, b_down, down_output_offsets, down_hidden, down_val,
                              down_slots, output_groups, n, h, f, x_t, y_nt, y_t, z, activation, timing) -> None:
        # Same tables as the x86 slots entry, in the argument order of the M1 kernel; `threads` is fixed by the G4 pool.
        self._dual(x, hidden_groups, b_up, up_offsets, up_idx, up_val, down_val, b_down, down_output_offsets, down_slots, down_hidden, output_groups,
                   n, h, f, x_t, y_nt, y_t, z, activation, timing)
