"""The entries ``hffn_compact_g4_forward_dual_nt_f32`` and ``hffn_compact_g4_forward_stores_nt_f32`` of the compact G4 inference kernel, bound once for the callers
that need ``z`` and ``h`` stored (the training leaves of the M1 CPU). The library is the one this leaf builds for its
own runtime, in its ``build/``: no second build of the forward."""
from __future__ import annotations

import ctypes
from pathlib import Path

import torch

from . import forward_route
from .compact_torch_extension import load_hffn_compact_inference_extension
from .forward_route import OUTPUTS_YZH, STORES, route

LEAF = Path(__file__).resolve().parents[1]
INFERENCE_MIN_N = 1  # the dual entry has no branch on N and stores z and h at N = 1

_I, _P = ctypes.c_int, ctypes.c_void_p
G4_GROUPS = 4  # the G4 groups of the kernel's own pool (hffn_pythia.compact_format.G4_GROUPS)


def inference_library() -> Path:
    """The library of the inference leaf, built if needed and loaded once."""
    return Path(load_hffn_compact_inference_extension(LEAF / "build").__file__)


class InferenceKernel:
    """The {y, z, h} forward of the compact inference kernel: the owner route (``forward_route.route``) picks the stores or the dual entry.

    ``layers`` hands the hidden-owned down tables the stores entry reads (keyed by the ``up_offsets`` pointer of the call) and sizes its one
    persistent ``y_local`` workspace (G4 x d_model x ``STORES_N`` floats, shared by the layers: the calls of a forward are sequential), so nothing is
    allocated or converted in the hot path. The FFN training step hands its layer with ``bind_layer``. A caller that hands neither cannot take the stores route: a call whose route is stores then
    raises (a consumer that was not moved, never a silent second route)."""

    def __init__(self, path: Path, layers=()):
        self.path = Path(path)
        self._lib = ctypes.CDLL(str(path))
        self._dual = self._lib.hffn_compact_g4_forward_dual_nt_f32
        self._dual.argtypes, self._dual.restype = [_P] * 12 + [_I] * 3 + [_P] * 6, None
        self.route_counts = {"stores": {}, "dual": {}}  # calls per route and N (descriptive, read by the checks)
        self._tables: dict[int, tuple[int, int]] = {}
        self._y_local = None
        self._min_n = self._max_n = forward_route.STORES_N  # the stores range (production: the one N); `_max_n` also sizes the workspace; route() reads these, not the module global
        self._keep: list = []  # anchors the registered tables so that their addresses stay valid
        for layer in layers:
            self._register(layer.up_row_offsets, layer.down_row_offsets, layer.down_idx, int(layer.d_model))

    def _register(self, up_offsets, down_offsets, down_idx, d_model: int) -> None:
        """Hand one layer's hidden-owned down tables to the stores route (at load, never in the hot path)."""
        if self._max_n <= 0:
            return
        if down_idx.dtype != torch.int16 or down_offsets.dtype != torch.int32 or not (down_idx.is_contiguous() and down_offsets.is_contiguous()):
            raise RuntimeError("stores route: hidden-owned down tables must be contiguous int32 offsets and int16 indices")
        key = up_offsets.data_ptr()
        entry = (down_offsets.data_ptr(), down_idx.data_ptr())
        if key in self._tables:
            if self._tables[key] != entry:
                raise RuntimeError("stores route: one up_offsets buffer with two different down tables")
            return  # the same layer tables handed again (several steps on one layer): idempotent
        self._keep += [up_offsets, down_offsets, down_idx]
        if self._y_local is None:
            self._stores = self._lib.hffn_compact_g4_forward_stores_nt_f32
            self._stores.argtypes, self._stores.restype = [_P] * 10 + [_I] * 3 + [_P] * 7, None
        self._tables[key] = (down_offsets.data_ptr(), down_idx.data_ptr())
        need = G4_GROUPS * d_model * self._max_n
        if self._y_local is None or self._y_local.numel() < need:
            self._y_local = torch.zeros(need)  # kGroups x h x n, the entry zeroes what it uses; one workspace shared by the layers (sequential calls)

    def bind_layer(self, layer) -> None:
        """Layer of the shared FFN training step (``up_offsets``, ``down_offsets``, ``down_idx``): the same hand-over as ``layers`` at load."""
        self._register(layer.up_offsets, layer.down_offsets, layer.down_idx, int(layer.d_model))

    @property
    def workspace_bytes(self) -> int:
        """Bytes of the stores workspace this kernel holds (0 when no layers were handed)."""
        return 0 if self._y_local is None else self._y_local.numel() * self._y_local.element_size()

    def forward_dual_slots_nt(self, threads, x, hidden_groups, b_up, up_offsets, up_idx, up_val, b_down, down_output_offsets, down_hidden, down_val,
                              down_slots, output_groups, n, h, f, x_t, y_nt, y_t, z, activation, timing) -> None:
        if route(n, OUTPUTS_YZH, (self._min_n, self._max_n)) != STORES:
            self.route_counts["dual"][n] = self.route_counts["dual"].get(n, 0) + 1
            # Same tables as the x86 slots entry, in the argument order of the M1 kernel; `threads` is fixed by the G4 pool.
            self._dual(x, hidden_groups, b_up, up_offsets, up_idx, up_val, down_val, b_down, down_output_offsets, down_slots, down_hidden, output_groups,
                       n, h, f, x_t, y_nt, y_t, z, activation, timing)
            return
        if self._y_local is None:
            raise RuntimeError("the owner route is stores at this N but the caller handed no layers: the consumer is not moved onto the owner route")
        assert self._y_local.numel() >= G4_GROUPS * h * n, "stores workspace smaller than G4 x h x n"
        down_offsets, down_idx = self._tables[up_offsets]  # KeyError = a layer that was not handed at load: a harness defect, never a silent route change
        self.route_counts["stores"][n] = self.route_counts["stores"].get(n, 0) + 1
        self._stores(x, hidden_groups, b_up, up_offsets, up_idx, up_val, down_offsets, down_idx, down_val, b_down, n, h, f, x_t, y_nt, y_t, z, activation,
                     self._y_local.data_ptr(), timing)
