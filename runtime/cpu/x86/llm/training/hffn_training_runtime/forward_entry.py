"""Entry point of the x86 leaf for the shared ``compact_runtime``: the forward of execution ``adaptive`` runs on the
shared x86 inference kernel (slots entry ``hffn_compact_forward_dual_slots_nt_f32``)."""
from __future__ import annotations

import sys
from pathlib import Path

# The shared x86 inference kernel (build and ctypes entry) lives in `runtime/cpu/x86/common/hffn_x86_common`.
_COMMON = next(p for p in Path(__file__).resolve().parents if (p / "runtime" / "common" / "pythia" / "hffn_pythia").is_dir()) / "runtime/cpu/x86/common"
if str(_COMMON) not in sys.path:
    sys.path.insert(0, str(_COMMON))

from hffn_x86_common.inference import InferenceKernel, inference_lib, inference_n1_lib  # noqa: E402

from .compact_export import G4_GROUPS, HFFNCompactG4Layer, HFFNDualRoute  # noqa: E402
from .forward_route_check import check_forward_route  # noqa: E402  (the shared copy, joined to this package by its path)

DEFAULT_BUILD_DIR = Path(__file__).resolve().parents[1] / "build"
# One thread per G4 group of hidden neurons (and per output group of the reverse schedule): T = 4.
INFERENCE_THREADS = G4_GROUPS
# Smallest block the training forward takes from the sibling inference kernel.
INFERENCE_MIN_N = 1


class _TrainingInferenceForward:
    """Route the N = 1 training call to its dedicated sibling inference entry."""

    def __init__(self, kernel: InferenceKernel):
        self._kernel = kernel

    def forward_dual_slots_nt(self, threads, x_nt, hidden_groups, b_up, up_offsets, up_idx, up_val, b_down,
                              down_output_offsets, down_hidden, down_val, down_slots, output_groups, n, h, f,
                              x_t, y_nt, y_t, z, activation, timing):
        forward = self._kernel.forward_dual_slots_n1_nt if n == 1 else self._kernel.forward_dual_slots_nt
        forward(threads, x_nt, hidden_groups, b_up, up_offsets, up_idx, up_val, b_down,
                down_output_offsets, down_hidden, down_val, down_slots, output_groups, n, h, f,
                x_t, y_nt, y_t, z, activation, timing)


def open_inference_forward(layers: tuple[HFFNCompactG4Layer, ...], routes: tuple[HFFNDualRoute, ...]) -> _TrainingInferenceForward:
    """The inference kernel of the forward, after the structural checks of every layer (once, at load)."""
    kernel = InferenceKernel(inference_lib(), inference_n1_lib())
    for index, (layer, route) in enumerate(zip(layers, routes)):
        check_forward_route(index, layer, route, INFERENCE_THREADS)
    return _TrainingInferenceForward(kernel)
