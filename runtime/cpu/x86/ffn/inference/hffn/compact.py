"""ctypes loader for the compact kernel (``hffn_compact_g4_inference.cpp``).

Threads: one call runs on T threads (``threads``, default: the CPUs of the process affinity
mask, at most ``MAX_DEFAULT_THREADS``), the calling thread plus T - 1 pool workers, each
owning one group of hidden neurons (and one group of output features). The partitions for
T are computed here; T = 4 gives the G4 partition of the test layers.
``decode`` and ``dual`` give the same bits for every T (each hidden value and each output is
computed by one thread, in slot order); ``baseline`` reduces T partial outputs in group order
(bits depend on T, within tolerance).

Routes: ``decode`` (N = 1) and ``dual`` (N >= 2), the recommended ones: up projection per hidden
neuron, down projection per output feature on the reverse schedule (down weights in output
order). ``baseline``: hidden-owned down projection with private [T, d, N] buffers; it is the
only route that reads the hidden-major down arrays. ``forward(x)`` without a route picks the
recommended one.

Work buffers (dense, owned by the loader, reused across calls):
- one buffer per role, sized for the largest batch seen and viewed as [.., N] for smaller ones,
  starting on a 64-byte boundary (``WORK_ALIGNMENT``);
- only the buffers the route uses (decode: act; dual: x_t, act; baseline: x_t, act, y_local);
- y_t shares x_t (x_t is dead once the hidden phase is over);
- batches above 96 tokens run as consecutive chunks of at most 96 (balanced, multiples of 8),
  so that every chunk takes the kernel's register-blocked paths.
No PyTorch import: the kernel only needs the C++ runtime and glibc's libm/libmvec.
"""
from __future__ import annotations

import ctypes
import dataclasses
import os

import numpy as np

from . import build
from hffn_ffn_layers.synth import CompactLayer

ROUTES = ("decode", "baseline", "dual")
MAX_BATCH = 96
MAX_DEFAULT_THREADS = 8

_ROLES = {"decode": ("act",), "baseline": ("x_t", "act", "y_local"), "dual": ("x_t", "act")}
# Work buffers start on a cache-line boundary. numpy only guarantees 16 bytes, and the kernel
# time then depends on where malloc happened to put each buffer: on an 8-core Zen 4, two
# loaders of the same layer differed by -7 % to +22 % on the same batch, reproducibly.
WORK_ALIGNMENT = 64


def aligned_empty(size: int, alignment: int = WORK_ALIGNMENT) -> np.ndarray:
    """Uninitialised float32[size] whose first byte is a multiple of ``alignment``."""
    raw = np.empty(size * 4 + alignment, np.uint8)
    start = -raw.ctypes.data % alignment
    return raw[start:start + size * 4].view(np.float32)


def default_threads() -> int:
    return max(1, min(len(os.sched_getaffinity(0)), MAX_DEFAULT_THREADS))


def recommended_route(n: int) -> str:
    return "decode" if n == 1 else "dual"


def batch_chunks(n: int, max_batch: int | None) -> list[tuple[int, int]]:
    """(start, size) of the consecutive token chunks of a batch of n."""
    if not max_batch or n <= max_batch:
        return [(0, n)]
    count = -(-n // max_batch)
    size = -(-(-(-n // count)) // 8) * 8
    return [(start, min(size, n - start)) for start in range(0, n, size)]


def _balanced_group_offsets(cost: np.ndarray, threads: int) -> np.ndarray:
    total, boundaries, cumulative, target = int(cost.sum()), [0], 0, 1
    for feature, value in enumerate(cost.tolist()):
        cumulative += int(value)
        while target < threads and cumulative >= round(total * target / threads):
            boundaries.append(feature + 1)
            target += 1
    while len(boundaries) < threads:
        boundaries.append(boundaries[-1])
    boundaries.append(cost.size)
    return np.array(boundaries, dtype=np.int32)


def hidden_partition(layer: CompactLayer, threads: int) -> np.ndarray:
    """Hidden neurons in T contiguous groups of equal size (the G4 groups of the layer for T = 4)."""
    return np.array([(group * layer.d_ff) // threads for group in range(threads + 1)], dtype=np.int32)


def dual_route(layer: CompactLayer, threads: int) -> dict:
    """Reverse schedule of the down projection: per output feature, its hidden neurons and its
    down weights in output order (a 2-byte hidden index and a 4-byte weight per edge, read sequentially by the output phase)."""
    index = layer.down_idx.astype(np.int64)
    count = np.bincount(index, minlength=layer.d_model).astype(np.int32)
    offsets = np.zeros(layer.d_model + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(count)
    slots = np.argsort(index, kind="stable")
    hidden_by_slot = np.repeat(np.arange(layer.d_ff), np.diff(layer.down_offsets))
    return {
        "offsets": offsets,
        "hidden": np.ascontiguousarray(hidden_by_slot[slots].astype(np.int16)),
        "weights": np.ascontiguousarray(layer.down_val[slots]),
        "groups": _balanced_group_offsets(count, threads),
    }


class CompactKernel:
    def __init__(self, layer: CompactLayer, max_batch: int | None = MAX_BATCH, threads: int | None = None,
                 hidden_owned: bool = True):
        """``hidden_owned=False`` keeps only what the recommended routes read (the up arrays and
        the reverse schedule): the hidden-major down arrays, a third of the model, are not
        referenced and the ``baseline`` route is unavailable."""
        self.max_batch = max_batch
        self._lib = ctypes.CDLL(str(build.build()))
        self.threads = threads or default_threads()
        self._groups = hidden_partition(layer, self.threads)
        self._route = dual_route(layer, self.threads)
        self.hidden_owned = hidden_owned
        empty = np.empty(0, np.int32)
        self.layer = layer if hidden_owned else dataclasses.replace(layer, down_offsets=empty, down_idx=empty.astype(np.int16), down_val=empty.astype(np.float32))
        p, i = ctypes.c_void_p, ctypes.c_int
        self._decode = self._lib.hffn_compact_forward_decode_nt_f32
        self._decode.argtypes = [i] + [p] * 11 + [i, i] + [p] * 3
        self._baseline = self._lib.hffn_compact_forward_nt_f32
        self._baseline.argtypes = [i] + [p] * 10 + [i, i, i] + [p] * 7
        self._dual = self._lib.hffn_compact_forward_dual_nt_f32
        self._dual.argtypes = [i] + [p] * 11 + [i, i, i] + [p] * 6
        for function in (self._decode, self._baseline, self._dual):
            function.restype = None
        self._flat: dict[str, np.ndarray] = {}
        self._timing: dict[int, np.ndarray] = {}
        self._chunked_timing: dict[int, np.ndarray] = {}
        self._arguments_cache: dict[tuple[str, int], tuple] = {}

    # Memory -------------------------------------------------------------------------------

    def model_bytes(self, route: str | None = None) -> int:
        """Bytes of the model arrays read by a route (None: the recommended routes, decode and dual)."""
        s = self.layer
        arrays = [self._groups, s.b_up, s.up_offsets, s.up_idx, s.up_val, s.b_down]
        arrays += [s.down_offsets, s.down_idx, s.down_val] if route == "baseline" else list(self._route.values())
        return sum(array.nbytes for array in arrays)

    def workspace_bytes(self) -> int:
        """Bytes of work buffers currently held."""
        return sum(a.nbytes for a in self._flat.values()) + sum(t.nbytes for t in self._timing.values())

    def _buffer(self, role: str, n: int) -> np.ndarray:
        h, f = self.layer.d_model, self.layer.d_ff
        shape = {"x_t": (h, n), "act": (f, n), "y_local": (self.threads, h, n)}[role]
        size = int(np.prod(shape))
        if role not in self._flat or self._flat[role].size < size:
            self._flat[role] = aligned_empty(size)
            self._arguments_cache.clear()  # every view moves
        return self._flat[role][:size].reshape(shape)

    def _workspace(self, n: int, route: str) -> dict:
        w = {role: self._buffer(role, n) for role in _ROLES[route]}
        w["timing"] = self._timing.setdefault(n, np.zeros(4, np.int64))
        return w

    # Calls ---------------------------------------------------------------------------------

    def _arguments(self, route: str, n: int) -> tuple:
        """Call arguments with every pointer resolved once; x and y are set per call."""
        key = (route, n)
        if key not in self._arguments_cache:
            if route == "baseline" and not self.hidden_owned:
                raise ValueError("the baseline route needs the hidden-major down arrays: build the kernel with hidden_owned=True")
            s, r, w, g = self.layer, self._route, self._workspace(n, route), self._groups
            a = lambda array: array.ctypes.data  # noqa: E731
            up = [a(g), a(s.b_up), a(s.up_offsets), a(s.up_idx), a(s.up_val)]
            reverse = [a(s.b_down), a(r["offsets"]), a(r["hidden"]), a(r["weights"]), a(r["groups"])]
            if route == "decode":
                assert n == 1
                function, arguments = self._decode, [self.threads, 0, *up, *reverse, s.d_model, s.d_ff, 0, a(w["act"]), a(w["timing"])]
                y_slot = 14
            elif route == "baseline":
                function, arguments = self._baseline, [self.threads, 0, *up, a(s.down_offsets), a(s.down_idx), a(s.down_val), a(s.b_down),
                                                       n, s.d_model, s.d_ff, a(w["x_t"]), 0, a(w["x_t"]), a(w["act"]), a(w["act"]),
                                                       a(w["y_local"]), a(w["timing"])]
                y_slot = 15
            elif route == "dual":
                function, arguments = self._dual, [self.threads, 0, *up, *reverse, n, s.d_model, s.d_ff,
                                                   a(w["x_t"]), 0, a(w["x_t"]), a(w["act"]), a(w["act"]), a(w["timing"])]
                y_slot = 16
            else:
                raise ValueError(route)
            self._arguments_cache[key] = (function, arguments, 1, y_slot)
        return self._arguments_cache[key]

    def forward(self, x_nt: np.ndarray, route: str | None = None, repeat: int = 1) -> np.ndarray:
        """y[N, d] for x[N, d] (float32, C-contiguous); route=None picks the recommended one."""
        n = x_nt.shape[0]
        assert x_nt.dtype == np.float32 and x_nt.flags.c_contiguous and x_nt.shape[1] == self.layer.d_model
        route = route or recommended_route(n)
        y = np.empty_like(x_nt)
        chunks = batch_chunks(n, self.max_batch if route != "decode" else None)
        calls = []
        for start, size in chunks:
            function, arguments, x_slot, y_slot = self._arguments(route, size)
            arguments = list(arguments)
            arguments[x_slot] = x_nt.ctypes.data + start * x_nt.strides[0]
            arguments[y_slot] = y.ctypes.data + start * y.strides[0]
            calls.append((function, arguments))
        if len(calls) == 1:
            function, arguments = calls[0]
            for _ in range(repeat):
                function(*arguments)
            return y
        for _ in range(repeat):
            total = np.zeros(4, np.int64)
            for (function, arguments), (_, size) in zip(calls, chunks):
                function(*arguments)
                total += self._timing[size]  # chunks of equal size share one timing array
            self._chunked_timing[n] = total
        return y

    def kernel_ns(self, n: int) -> np.ndarray:
        """Phase timings (ns) of the last call at batch n, measured inside the kernel."""
        if self.max_batch and n > self.max_batch and n in self._chunked_timing:
            return self._chunked_timing[n].copy()
        return self._timing[n].copy()
