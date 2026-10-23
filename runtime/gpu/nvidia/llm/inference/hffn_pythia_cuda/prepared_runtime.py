"""`HFFNRuntime` with the `coop`/`coop` call (N <= 4, every decode step) launched from a descriptor built once per (layer, N).

A descriptor holds the two kernel handles, their grids and the two argument blocks packed once as ctypes values (device
pointers and 32-bit scalars). A call writes the pointer of `x` into the first up argument and launches both kernels with the
driver call on the default stream. Every other case (other modes, forced modes, N above the cooperative range) runs the
control's `forward`. The kernels, grids, modes, workspace buffers and counters are those of `HFFNRuntime`.

Baked device pointers: the eight arrays of the layer (never reassigned, kept alive by the runtime) and `act` and `y`
(`_reserve` drops every descriptor when it reallocates them). `x` is never baked. `force()` drops every descriptor.

Assumptions, as for `HFFNRuntime`: one thread per instance; CuPy and PyTorch both on the default stream 0 (the launch passes
stream 0, and a descriptor is built only when CuPy's current stream is 0); the kernel modes are changed only through `force()`
(a descriptor keeps the modes of the thresholds `up_mode` and `down_mode` it was built with).
"""
from __future__ import annotations

import ctypes

import cupy
import torch
from cupy.cuda import driver

from hffn_cuda.runtime import BLOCK, MAX_TOKENS, HFFNRuntime, down_mode, up_mode

_UP_ARRAYS = ("up_offsets", "up_idx", "up_val", "b_up")
_DOWN_ARRAYS = ("out_offsets", "out_hidden", "out_val", "b_down")


def _pack(values):
    """The array of the addresses of the argument storage (the `void**` of the driver launch)."""
    return (ctypes.c_void_p * len(values))(*(ctypes.addressof(v) for v in values))


class _Descriptor:
    __slots__ = ("up_func", "down_func", "up_grid", "down_grid", "x", "_storage", "_tables", "up_table", "down_table")

    def __init__(self, runtime: "PreparedRuntime", index: int, n: int) -> None:
        model = runtime._model[index]
        _, act, y = runtime._work
        d, h, count = ctypes.c_uint32(runtime.d_model), ctypes.c_uint32(runtime.d_ff), ctypes.c_uint32(n)
        self.x = ctypes.c_uint64(0)                                   # patched at every call
        up = [self.x] + [ctypes.c_uint64(model[name].data.ptr) for name in _UP_ARRAYS] + [ctypes.c_uint64(act.data.ptr), d, h, count]
        down = [ctypes.c_uint64(act.data.ptr)] + [ctypes.c_uint64(model[name].data.ptr) for name in _DOWN_ARRAYS] + [ctypes.c_uint64(y.data.ptr), d, count]
        self._storage = (up, down)                                    # the storage and the tables stay alive with the descriptor
        self._tables = (_pack(up), _pack(down))
        self.up_table, self.down_table = (ctypes.addressof(t) for t in self._tables)
        self.up_func = runtime._up["coop"].kernel.ptr
        self.down_func = runtime._down["coop"].kernel.ptr
        self.up_grid = (runtime.d_ff + 15) // 16
        self.down_grid = runtime.d_model // 16


class PreparedRuntime(HFFNRuntime):
    def _reserve(self, n: int) -> None:
        super()._reserve(n)
        self._descriptors: dict = {}                                  # act and y moved: every pointer baked in a descriptor is stale

    def force(self, up, down) -> None:
        super().force(up, down)
        self._descriptors = {}

    def forward(self, index: int, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.d_model or x.dtype != torch.float32 or x.device != self._y.device or not x.is_contiguous():
            raise ValueError("expected a contiguous float32 [N, d_model] tensor on the runtime's device")
        n = x.shape[0]
        if not 1 <= n <= MAX_TOKENS:
            raise ValueError("unsupported token count")
        if n > self._tokens:
            self._reserve(n)
        key = (index, n)
        desc = self._descriptors.get(key)
        if desc is None:
            up, down = self.forced_up or up_mode(n), self.forced_down or down_mode(n, self.d_ff)
            desc = _Descriptor(self, index, n) if (up, down) == ("coop", "coop") and cupy.cuda.get_current_stream().ptr == 0 else False
            self._descriptors[key] = desc
        if desc is False:
            return super().forward(index, x)
        desc.x.value = x.data_ptr()                                   # the default stream orders the kernel before any reuse of x's block
        driver.launchKernel(desc.up_func, desc.up_grid, 1, 1, BLOCK, 1, 1, 0, 0, desc.up_table, 0)
        driver.launchKernel(desc.down_func, desc.down_grid, 1, 1, BLOCK, 1, 1, 0, 0, desc.down_table, 0)
        self.calls["coop", "coop"] += 1
        return self._y[:n]
