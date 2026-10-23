"""HFFN forward runtime for the FFNs of one model, on the x86 kernel.

Execution (T threads of the process's OpenMP team, the one PyTorch's operators run on: the
kernel has no thread pool of its own, and must be loaded after PyTorch to share its libgomp):
- up projection and GELU per hidden neuron, on T contiguous groups of hidden neurons;
- down projection per output feature, on T contiguous groups of output features, reading the
  same edges in output order (reverse schedule built at load, no reduction).
N = 1 takes the kernel's decode route, N >= 2 its dual route. Each hidden value and each output
is computed by one thread in slot order: the result depends neither on T nor on the partitions.

Model arrays are compact (values, indices, offsets). Work buffers are dense, shared by all
layers (one call at a time per process), sized for the largest batch seen and reused.
"""
from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass

import torch

from . import build
from hffn_pythia.export import hidden_of_slot

MAX_CHUNK = 96  # the kernel keeps a batch of up to 96 tokens in registers
MAX_DEFAULT_THREADS = 8


def default_threads() -> int:
    return max(1, min(len(os.sched_getaffinity(0)), MAX_DEFAULT_THREADS))


def chunks(n: int) -> list[tuple[int, int]]:
    """(start, size) of the consecutive token chunks of a batch: balanced, multiples of 8, at most MAX_CHUNK."""
    if n <= MAX_CHUNK:
        return [(0, n)]
    count = -(-n // MAX_CHUNK)
    size = -(-(-(-n // count)) // 8) * 8
    return [(start, min(size, n - start)) for start in range(0, n, size)]


def balanced_partition(cost: torch.Tensor, parts: int) -> torch.Tensor:
    """int32[parts + 1] offsets of contiguous groups of nearly equal total cost."""
    cumulative = torch.cumsum(cost.long(), 0)
    targets = (int(cumulative[-1]) * torch.arange(1, parts)) // parts
    inner = torch.searchsorted(cumulative, targets, right=False) + 1
    return torch.cat([torch.zeros(1, dtype=torch.long), inner.clamp(max=cost.numel()), torch.tensor([cost.numel()])]).to(torch.int32)


@dataclass(frozen=True)
class HFFNLayer:
    """What the kernel reads for one FFN: the up arrays per hidden neuron and the down edges in output order."""
    d_model: int
    d_ff: int
    b_up: torch.Tensor         # float32[d_ff]
    up_offsets: torch.Tensor   # int32[d_ff + 1]
    up_idx: torch.Tensor       # int16[nnz_up], input feature of each slot
    up_val: torch.Tensor       # float32[nnz_up]
    b_down: torch.Tensor       # float32[d_model]
    out_offsets: torch.Tensor  # int32[d_model + 1]
    out_hidden: torch.Tensor   # int16[nnz_down], hidden neuron of each edge
    out_val: torch.Tensor      # float32[nnz_down]

    @classmethod
    def from_export(cls, layer: dict) -> "HFFNLayer":
        output = layer["down_idx"].long()
        order = torch.argsort(output, stable=True)
        out_offsets = torch.zeros(layer["d_model"] + 1, dtype=torch.int32)
        out_offsets[1:] = torch.cumsum(torch.bincount(output, minlength=layer["d_model"]), 0)
        return cls(layer["d_model"], layer["d_ff"], layer["b_up"], layer["up_row_offsets"], layer["up_idx"], layer["up_val"],
                   layer["b_down"], out_offsets, hidden_of_slot(layer, "down")[order].to(torch.int16), layer["down_val"][order].contiguous())

    def model_bytes(self) -> int:
        return sum(value.numel() * value.element_size() for value in vars(self).values() if isinstance(value, torch.Tensor))


class HFFNRuntime:
    def __init__(self, layers: list[HFFNLayer], threads: int | None = None) -> None:
        self.layers = layers
        self.threads = threads or default_threads()
        self.d_model, self.d_ff = layers[0].d_model, layers[0].d_ff
        if any((layer.d_model, layer.d_ff) != (self.d_model, self.d_ff) for layer in layers):
            raise ValueError("all layers of a runtime share d_model and d_ff")
        library = ctypes.CDLL(str(build.build()))
        pointer, integer = ctypes.c_void_p, ctypes.c_int
        self._decode = library.hffn_compact_forward_decode_nt_f32
        self._decode.argtypes = [integer] + [pointer] * 11 + [integer] * 2 + [pointer] * 3
        self._dual = library.hffn_compact_forward_dual_nt_f32
        self._dual.argtypes = [integer] + [pointer] * 11 + [integer] * 3 + [pointer] * 6
        self._decode.restype = self._dual.restype = None
        # Partitions by work: up slots per hidden neuron, down edges per output feature.
        self._partitions = [(balanced_partition(layer.up_offsets.diff(), self.threads), balanced_partition(layer.out_offsets.diff(), self.threads)) for layer in layers]
        self._model = [self._model_pointers(layer, *partition) for layer, partition in zip(layers, self._partitions)]
        self._timing = (ctypes.c_int64 * 4)()  # phase times of the last kernel call
        self.kernel_ns = 0                     # total time measured inside the kernel since load
        self._chunk = self._tokens = 0
        self._reserve(1)

    @staticmethod
    def _model_pointers(layer: HFFNLayer, hidden_groups: torch.Tensor, output_groups: torch.Tensor) -> tuple:
        arrays = (hidden_groups, layer.b_up, layer.up_offsets, layer.up_idx, layer.up_val, layer.b_down, layer.out_offsets, layer.out_hidden, layer.out_val, output_groups)
        return tuple(array.data_ptr() for array in arrays)

    def _reserve(self, n: int) -> None:
        """Grow the work buffers for a batch of n tokens (PyTorch aligns CPU tensors on 64 bytes)."""
        chunk = min(n, MAX_CHUNK)
        if chunk > self._chunk:
            self._chunk = chunk
            self._x_t = torch.empty(self.d_model * chunk)  # [d_model, chunk], also the transposed output
            self._act = torch.empty(self.d_ff * chunk)     # [d_ff, chunk]
        if n > self._tokens:
            self._tokens = n
            self._y = torch.empty(n, self.d_model)

    def workspace_bytes(self) -> int:
        return 4 * (self._x_t.numel() + self._act.numel() + self._y.numel())

    def forward(self, index: int, x: torch.Tensor) -> torch.Tensor:
        """y[N, d_model] of layer ``index`` for x[N, d_model] (contiguous CPU float32).

        The result is a view of the runtime's output buffer, overwritten by the next call.
        """
        if x.dim() != 2 or x.shape[1] != self.d_model or x.dtype != torch.float32 or not x.is_contiguous():
            raise ValueError("expected a contiguous float32 [N, d_model] tensor")
        n = x.shape[0]
        if n > self._tokens or min(n, MAX_CHUNK) > self._chunk:
            self._reserve(n)
        y = self._y[:n]
        model, timing = self._model[index], self._timing
        x_ptr, y_ptr, x_t, act = x.data_ptr(), y.data_ptr(), self._x_t.data_ptr(), self._act.data_ptr()
        if n == 1:
            self._decode(self.threads, x_ptr, *model, self.d_model, self.d_ff, y_ptr, act, timing)
            self.kernel_ns += timing[1]
            return y
        row = 4 * self.d_model
        for start, size in chunks(n):
            self._dual(self.threads, x_ptr + start * row, *model, size, self.d_model, self.d_ff, x_t, y_ptr + start * row, x_t, act, act, timing)
            self.kernel_ns += timing[0] + timing[1] + timing[2] + timing[3]
        return y
