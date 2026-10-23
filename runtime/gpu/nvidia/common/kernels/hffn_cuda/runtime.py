"""HFFN forward runtime for a stack of FFNs of one shape (the FFNs of one model, or one isolated FFN), on CUDA.

One call runs the whole block of N tokens of one FFN: an up kernel (hidden owners) then a down
kernel (output owners), chosen from N and d_ff by `up_mode` and `down_mode` (see kernels.py).
The thresholds are those of the Metal reference, measured on an Apple M1: provisional here.

Model arrays are compact (values, indices, offsets) and stay on the device. Work buffers are
dense, shared by all layers, sized for the largest N seen and reused. Kernels are queued on
PyTorch's default CUDA stream, so they run in call order with the PyTorch operations around them.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import cupy
import numpy
import torch

from .kernels import BLOCK, DOWN_TILE, SOURCE, UP_TILE

UP_MODES = ("coop", "lane", "tile")
DOWN_MODES = ("coop", "tiled", "blocked")
COOP_MAX_TOKENS = 4       # cooperative reduction up to this N
BLOCK_MAX_WIDTH = 4096    # blocked down for a FFN up to the widest measured (4096), or up to two tiles
MAX_TOKENS = 1 << 18      # keeps every 32-bit address of the kernels in range


def up_mode(n: int) -> str:
    return "coop" if n <= COOP_MAX_TOKENS else "lane" if n < 64 else "tile"


def down_mode(n: int, d_ff: int) -> str:
    if n <= COOP_MAX_TOKENS:
        return "coop"
    if n >= DOWN_TILE and (d_ff <= BLOCK_MAX_WIDTH or n <= 2 * DOWN_TILE):
        return "blocked"
    return "tiled"


@dataclass(frozen=True)
class HFFNLayer:
    """What the kernels read for one FFN: the up arrays per hidden neuron and the down edges in output order."""
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
    def from_hidden_major(cls, d_model: int, d_ff: int, b_up: torch.Tensor, up_offsets: torch.Tensor, up_idx: torch.Tensor,
                          up_val: torch.Tensor, b_down: torch.Tensor, down_offsets: torch.Tensor, down_idx: torch.Tensor,
                          down_val: torch.Tensor) -> "HFFNLayer":
        """From the canonical hidden-major storage: for neuron r, its up slots and its down slots (output feature of
        each slot in ``down_idx``) at ``[offsets[r], offsets[r + 1])``. The up arrays are kept as given."""
        output = down_idx.long()
        order = torch.argsort(output, stable=True)
        out_offsets = torch.zeros(d_model + 1, dtype=torch.int32)
        out_offsets[1:] = torch.cumsum(torch.bincount(output, minlength=d_model), 0)
        hidden = torch.repeat_interleave(torch.arange(d_ff), down_offsets.diff().long())
        return cls(d_model, d_ff, b_up, up_offsets, up_idx, up_val,
                   b_down, out_offsets, hidden[order].to(torch.int16), down_val[order].contiguous())

    def model_bytes(self) -> int:
        return sum(value.numel() * value.element_size() for value in vars(self).values() if isinstance(value, torch.Tensor))


_ARRAYS = ("b_up", "up_offsets", "up_idx", "up_val", "b_down", "out_offsets", "out_hidden", "out_val")


class HFFNRuntime:
    def __init__(self, layers: list[HFFNLayer], device: str | torch.device = "cuda") -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda" or self.device.index not in (None, 0):
            raise ValueError("the HFFN CUDA runtime runs on cuda:0")
        if torch.cuda.current_stream(self.device).cuda_stream != 0:
            raise RuntimeError("the HFFN CUDA runtime requires PyTorch's default CUDA stream")
        self.layers = layers
        self.d_model, self.d_ff = layers[0].d_model, layers[0].d_ff
        if any((layer.d_model, layer.d_ff) != (self.d_model, self.d_ff) for layer in layers):
            raise ValueError("all layers of a runtime share d_model and d_ff")
        if self.d_model % 16:
            raise ValueError("d_model must be a multiple of 16")
        module = cupy.RawModule(code=SOURCE)
        self._up = {mode: module.get_function(f"up_{mode}") for mode in UP_MODES}
        self._down = {mode: module.get_function(f"down_{mode}") for mode in DOWN_MODES}
        # The device tensors own the memory; the CuPy arrays are views of them for the kernel calls.
        self._tensors = [{name: getattr(layer, name).to(self.device) for name in _ARRAYS} for layer in layers]
        self._model = [{name: cupy.asarray(tensor) for name, tensor in tensors.items()} for tensors in self._tensors]
        self._d, self._h = numpy.uint32(self.d_model), numpy.uint32(self.d_ff)
        self.forced_up: str | None = None
        self.forced_down: str | None = None
        self.calls: Counter = Counter()
        self._tokens = 0
        self._reserve(1)

    def force(self, up: str | None, down: str | None) -> None:
        """Diagnostic: fix the up and/or the down kernel regardless of N (None: the policy)."""
        if up not in (None, *UP_MODES) or down not in (None, *DOWN_MODES):
            raise ValueError("unknown kernel mode")
        self.forced_up, self.forced_down = up, down

    def _reserve(self, n: int) -> None:
        self._tokens = n
        self._x_t = torch.empty(self.d_model * n, device=self.device)   # [d_model, n]
        self._act = torch.empty(self.d_ff * n, device=self.device)      # [d_ff, n]
        self._y = torch.empty(n, self.d_model, device=self.device)
        self._work = tuple(cupy.asarray(tensor) for tensor in (self._x_t, self._act, self._y))

    def workspace_bytes(self) -> int:
        return 4 * (self._x_t.numel() + self._act.numel() + self._y.numel())

    def forward(self, index: int, x: torch.Tensor) -> torch.Tensor:
        """y[N, d_model] of layer ``index`` for x[N, d_model] (contiguous float32 on the device).

        The result is a view of the runtime's output buffer, overwritten by the next call.
        """
        if x.dim() != 2 or x.shape[1] != self.d_model or x.dtype != torch.float32 or x.device != self._y.device or not x.is_contiguous():
            raise ValueError("expected a contiguous float32 [N, d_model] tensor on the runtime's device")
        n = x.shape[0]
        if not 1 <= n <= MAX_TOKENS:
            raise ValueError("unsupported token count")
        if n > self._tokens:
            self._reserve(n)
        up, down = self._launch(index, x, n, self._up, None)
        self.calls[up, down] += 1
        return self._y[:n]

    def _launch(self, index: int, x: torch.Tensor, n: int, up_kernels: dict, z_ptr: numpy.uint64 | None) -> tuple[str, str]:
        """Launch the up then the down kernel of layer ``index`` for the N = ``n`` tokens of ``x`` (validated, buffers reserved)
        and return the (up, down) modes taken. ``up_kernels`` maps an up mode to its kernel: ``self._up``, or the training
        runtime's up kernels that also store ``z`` (then ``z_ptr`` is the device pointer of ``z``, passed after ``act``, and
        ``x`` is also copied into ``_x_t`` for every up mode: the backward reads it). Buffers are read from the attributes
        at each call."""
        up, down = self.forced_up or up_mode(n), self.forced_down or down_mode(n, self.d_ff)
        model, (x_t, act, y) = self._model[index], self._work
        d, h, count = self._d, self._h, numpy.uint32(n)
        owners = ((self.d_ff + 15) // 16,)
        if up == "coop":
            source = numpy.uint64(x.data_ptr())   # the default stream orders the kernel before any reuse of x's block
        else:
            source = x_t
            if up == "tile":
                owners += ((n + UP_TILE - 1) // UP_TILE,)
        if up != "coop" or z_ptr is not None:
            self._x_t[:self.d_model * n].view(self.d_model, n).copy_(x.T)
        if z_ptr is None:
            up_kernels[up](owners, (BLOCK,), (source, model["up_offsets"], model["up_idx"], model["up_val"], model["b_up"], act, d, h, count))
        else:
            up_kernels[up](owners, (BLOCK,), (source, model["up_offsets"], model["up_idx"], model["up_val"], model["b_up"], act, z_ptr, d, h, count))
        if down == "coop":
            blocks = self.d_model // 16
        elif down == "tiled":
            blocks = (n * self.d_model + BLOCK - 1) // BLOCK
        else:
            blocks = ((n + DOWN_TILE - 1) // DOWN_TILE) * self.d_model * 32 // BLOCK
        self._down[down]((blocks,), (BLOCK,), (act, model["out_offsets"], model["out_hidden"], model["out_val"], model["b_down"], y, d, count))
        return up, down
