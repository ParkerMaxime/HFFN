"""HFFN training step on CUDA over the shared inference runtime: forward (inference kernels, `z` stored), backward and
compact AdamW, one call per block of N tokens. Kernels, layouts, dispatch and reduction orders: `kernels_train.py`.

The wrapped `HFFNRuntime` keeps its compact arrays, its dispatch policy and its dense work buffers (`x_t`, `a = h`,
`y`); the step reads and updates them in place. Added here, all allocated at construction or by `reserve(n)`:
- compact, per layer: the gradients of `up_val`, `b_up`, `out_val`, `b_down` and their two AdamW moments, each of the
  length and order of its array (`out_val` and its gradient are output-major, as the runtime holds them); two
  transposed supports as positions into the runtime's arrays (int32 positions, int16 owners, int32 offsets);
- dense, shared by the layers, sized for the largest N reserved: `z`, `gz` `[d_ff, N]`, `gy_t` `[d_model, N]`, `gx`
  `[N, d_model]`;
- the partial workspace of grad_down / grad_up (`kernels_train`: `[tiles, S]` float32, S = nnz_down + d_model +
  nnz_up + d_ff columns), shared by the layers (S of the widest layer), sized at construction for `max_tokens`
  (default 2048: 8 tiles of 256 tokens) and grown by `reserve(n)` only for a larger N; none is used for N <= 256.
No dense `W_up`, `W_down`, their gradients or moments exist. `step`, `forward`, `backward` and `adamw` allocate no
device memory once `reserve(n)` has run for the largest N.

`backward(index, gy)` uses the state of the last `forward(index, x, z)` with a `z` buffer (`x_t`, `h`, `z` of that
call): it must follow it with no other forward of the runtime in between (checked). The runtime's work buffers are
shared by its layers, so a stack of layers needs per-layer activations; that is outside this isolated-FFN step.
"""
from __future__ import annotations

import math
from collections import Counter

import kernels_train  # noqa: F401  (puts the shared kernels on sys.path)
from kernels_train import TrainKernels, grad_tiles

import cupy  # noqa: E402
import numpy  # noqa: E402
import torch  # noqa: E402
from hffn_cuda.runtime import MAX_TOKENS, HFFNLayer, HFFNRuntime  # noqa: E402

PARAMETERS = ("up_val", "b_up", "out_val", "b_down")
SUPPORTS = ("in_offsets", "in_pos", "in_hidden", "hid_offsets", "hid_pos", "hid_output")
PARTIAL_MAX_TOKENS = 2048   # default largest N: the per-tile partials are sized at construction, never in a step


def transposed_supports(layer: HFFNLayer) -> dict[str, torch.Tensor]:
    """Up slots input-major and down edges hidden-major, as positions into the runtime's arrays (built once, on the host).

    Stable sorts: the up slots of one input are in ascending hidden order, the down edges of one neuron in ascending
    output order. These orders are the summation orders of `gx` and `dh`.
    """
    up_offsets, up_idx = layer.up_offsets.cpu().long(), layer.up_idx.cpu().long()
    out_offsets, out_hidden = layer.out_offsets.cpu().long(), layer.out_hidden.cpu().long()
    up_hidden = torch.repeat_interleave(torch.arange(layer.d_ff), up_offsets.diff())
    by_input = torch.argsort(up_idx, stable=True)
    in_offsets = torch.zeros(layer.d_model + 1, dtype=torch.int32)
    in_offsets[1:] = torch.cumsum(torch.bincount(up_idx, minlength=layer.d_model), 0)
    out_output = torch.repeat_interleave(torch.arange(layer.d_model), out_offsets.diff())
    by_hidden = torch.argsort(out_hidden, stable=True)
    hid_offsets = torch.zeros(layer.d_ff + 1, dtype=torch.int32)
    hid_offsets[1:] = torch.cumsum(torch.bincount(out_hidden, minlength=layer.d_ff), 0)
    return {"in_offsets": in_offsets, "in_pos": by_input.to(torch.int32), "in_hidden": up_hidden[by_input].to(torch.int16),
            "hid_offsets": hid_offsets, "hid_pos": by_hidden.to(torch.int32), "hid_output": out_output[by_hidden].to(torch.int16)}


class HFFNTrainRuntime:
    def __init__(self, runtime: HFFNRuntime, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 1e-2, max_tokens: int = PARTIAL_MAX_TOKENS) -> None:
        self.runtime = runtime
        self.d_model, self.d_ff = runtime.d_model, runtime.d_ff
        self.lr, self.betas, self.eps, self.weight_decay = lr, betas, eps, weight_decay
        self.kernels = TrainKernels()
        device = runtime.device
        self._supports, self._grads, self._moments = [], [], []
        for layer, params in zip(runtime.layers, runtime._tensors):
            self._supports.append({name: tensor.to(device) for name, tensor in transposed_supports(layer).items()})
            self._grads.append({name: torch.zeros_like(params[name]) for name in PARAMETERS})
            self._moments.append({name: (torch.zeros_like(params[name]), torch.zeros_like(params[name])) for name in PARAMETERS})
        # CuPy views of the device tensors, made once (the tensors own the memory).
        self._support_views = [{name: cupy.asarray(t) for name, t in s.items()} for s in self._supports]
        self._grad_views = [{name: cupy.asarray(t) for name, t in g.items()} for g in self._grads]
        self._moment_views = [{name: tuple(cupy.asarray(t) for t in pair) for name, pair in m.items()} for m in self._moments]
        self._d, self._h = numpy.uint32(self.d_model), numpy.uint32(self.d_ff)
        self.steps = [0] * len(runtime.layers)   # AdamW step count per layer (part of the optimiser state)
        self.calls: Counter = Counter()          # (up, down) of the forwards with a `z` buffer
        # Columns of the partial workspace per layer: g_down [0, c1), gb_down [c1, c2), g_up [c2, c3), gb_up [c3, S).
        self._columns = []
        for params in runtime._tensors:
            # Every column must be written by exactly one warp per tile: the offsets cover the whole compact array.
            for offsets, values in (("out_offsets", "out_val"), ("up_offsets", "up_val")):
                if int(params[offsets][0]) != 0 or int(params[offsets][-1]) != params[values].numel():
                    raise ValueError(f"{offsets} must cover {values} exactly (partial workspace columns)")
            c1 = params["out_val"].numel()
            c2 = c1 + self.d_model
            c3 = c2 + params["up_val"].numel()
            self._columns.append(tuple(numpy.uint32(c) for c in (c1, c2, c3, c3 + self.d_ff)))
        self._partial_tiles, self._partials, self._partial_views = 1, None, None
        self._saved: tuple | None = None
        self._tokens = 0
        self.growths = 0
        self.reserve(1)
        self._reserve_partials(max_tokens)
        self.growths = 0     # buffer growths after construction: a timed window must leave it unchanged

    # -- buffers -------------------------------------------------------------------------------------------------
    def reserve(self, n: int) -> None:
        """Size the runtime's and the step's dense buffers for N tokens. Allocates only when N grows: call it with the
        largest N before any timed step."""
        runtime = self.runtime
        if n > runtime._tokens:
            runtime._reserve(n)
            self._saved = None
            self.growths += 1
        if n > self._tokens:
            self._tokens = n
            self.growths += 1
            device = runtime.device
            self._z = torch.empty(self.d_ff * n, device=device)
            self._gz = torch.empty(self.d_ff * n, device=device)
            self._gy_t = torch.empty(self.d_model * n, device=device)
            self._gx = torch.empty(n, self.d_model, device=device)
            self._work = tuple(cupy.asarray(t) for t in (self._gz, self._gy_t, self._gx))
            self._saved = None
        self._reserve_partials(n)

    def _reserve_partials(self, n: int) -> None:
        """Size the partial workspace for grad_tiles(n) tiles; allocates only when the tile count grows past 1."""
        tiles = grad_tiles(n)
        if tiles <= self._partial_tiles:
            return
        stride = max(int(columns[3]) for columns in self._columns)
        if tiles * stride >= 2 ** 32:      # the kernels address the workspace with unsigned 32-bit indices
            raise ValueError(f"partial workspace of {tiles} tiles x {stride} columns exceeds 32-bit indexing")
        self._partial_tiles = tiles
        self.growths += 1
        self._partials = torch.empty(tiles * stride, device=self.runtime.device)
        ws = cupy.asarray(self._partials)
        # Per layer: the whole workspace (reduce_tiles), then the views at the first column of g_down (0), gb_down (c1),
        # g_up (c2) and gb_up (c3), passed to grad_down and grad_up.
        self._partial_views = [(ws, ws, ws[int(c1):], ws[int(c2):], ws[int(c3):]) for c1, c2, c3, _ in self._columns]

    def _check_tokens(self, t: torch.Tensor, what: str) -> int:
        if (t.dim() != 2 or t.shape[1] != self.d_model or t.dtype != torch.float32 or t.device != self.runtime._y.device
                or not t.is_contiguous()):
            raise ValueError(f"expected a contiguous float32 [N, d_model] tensor {what} on the runtime's device")
        if not 1 <= t.shape[0] <= MAX_TOKENS:
            raise ValueError("unsupported token count")
        return t.shape[0]

    # -- the step and its pieces ---------------------------------------------------------------------------------
    def forward(self, index: int, x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        """y[N, d_model] of layer ``index`` (a view of the runtime's output buffer).

        Without ``z``: the inference runtime's own call, unchanged. With ``z`` (contiguous float32 on the device, at
        least d_ff * N entries): the same dispatch, the up kernel also writes ``z`` hidden-major ``[d_ff, N]``, and the
        state the backward needs is kept (``z`` must stay alive until then).
        """
        runtime = self.runtime
        if z is None:
            return runtime.forward(index, x)
        n = self._check_tokens(x, "x")
        if z.dtype != torch.float32 or z.device != runtime._y.device or not z.is_contiguous() or z.numel() < self.d_ff * n:
            raise ValueError("z must be a contiguous float32 device tensor of at least d_ff * N entries")
        self.reserve(n)
        up, down = runtime._launch(index, x, n, self.kernels.up_z, numpy.uint64(z.data_ptr()))
        self.calls[up, down] += 1
        self._saved = (index, n, z.data_ptr(), sum(runtime.calls.values()))
        return runtime._y[:n]

    def backward(self, index: int, gy: torch.Tensor) -> torch.Tensor:
        """Compact gradients of layer ``index`` (see `gradients`) and gx[N, d_model] (a view of the step's buffer)
        for the upstream gradient gy[N, d_model] of the last forward with a `z` buffer."""
        runtime = self.runtime
        if self._saved is None or self._saved[0] != index or self._saved[3] != sum(runtime.calls.values()):
            raise RuntimeError("backward must follow a forward of the same layer with a z buffer, with no forward in between")
        _, n, z_ptr, _ = self._saved
        if self._check_tokens(gy, "gy") != n:
            raise ValueError("gy must have the N of the forward")
        model, support, grad = runtime._model[index], self._support_views[index], self._grad_views[index]
        x_t, act, _ = runtime._work
        gz, gy_t, gx = self._work
        d, h, count = self._d, self._h, numpy.uint32(n)
        self._gy_t[:self.d_model * n].view(self.d_model, n).copy_(gy.T)
        tiles = grad_tiles(n)
        if tiles == 1:   # the tile partial is the gradient: written in place, no reduction launch
            stride, g_down, gb_down, g_up, gb_up = numpy.uint32(0), grad["out_val"], grad["b_down"], grad["up_val"], grad["b_up"]
        else:            # tile partials in the workspace, summed in tile order by reduce_tiles
            c1, c2, c3, stride = self._columns[index]
            ws, g_down, gb_down, g_up, gb_up = self._partial_views[index]
        self.kernels.grad_down(n, self.d_model, (gy_t, act, model["out_offsets"], model["out_hidden"],
                                                 g_down, gb_down, d, count, stride))
        self.kernels.grad_hidden(n, self.d_ff, (gy_t, numpy.uint64(z_ptr), support["hid_offsets"], support["hid_pos"],
                                                support["hid_output"], model["out_val"], gz, h, count))
        self.kernels.grad_up(n, self.d_ff, (x_t, gz, model["up_offsets"], model["up_idx"], g_up, gb_up, h, count, stride))
        if tiles > 1:
            self.kernels.reduce_tiles(int(stride), (ws, stride, numpy.uint32(tiles), c1, c2, c3, grad["out_val"],
                                                    grad["b_down"], grad["up_val"], grad["b_up"]))
        self.kernels.grad_x(n, self.d_model, (gz, support["in_offsets"], support["in_pos"], support["in_hidden"],
                                              model["up_val"], gx, d, count))
        return self._gx[:n]

    def adamw(self, index: int) -> None:
        """One AdamW update of the four compact arrays of layer ``index`` from the gradients of the last backward."""
        self.steps[index] += 1
        k = self.steps[index]
        beta1, beta2 = self.betas
        f = numpy.float32
        scalars = (f(self.lr * self.weight_decay), f(beta1), f(1.0 - beta1), f(beta2), f(1.0 - beta2),
                   f(self.lr / (1.0 - beta1 ** k)), f(math.sqrt(1.0 - beta2 ** k)), f(self.eps))
        model, grad, moments = self.runtime._model[index], self._grad_views[index], self._moment_views[index]
        for name in PARAMETERS:
            p = model[name]
            m, v = moments[name]
            self.kernels.adamw(p.size, (p, grad[name], m, v, numpy.uint32(p.size)) + scalars)

    def step(self, index: int, x: torch.Tensor, gy: torch.Tensor) -> torch.Tensor:
        """Forward (z stored in the step's buffer), backward, AdamW. Returns gx (a view of the step's buffer)."""
        self.forward(index, x, self._z)
        gx = self.backward(index, gy)
        self.adamw(index)
        return gx

    # -- read access for the gates (views, no copy; not for the timed path) --------------------------------------
    def parameters(self, index: int) -> dict[str, torch.Tensor]:
        return {name: self.runtime._tensors[index][name] for name in PARAMETERS}

    def gradients(self, index: int) -> dict[str, torch.Tensor]:
        """Keyed by parameter: `up_val` hidden-major slot order, `out_val` output-major edge order, as the runtime."""
        return self._grads[index]

    def moments(self, index: int) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        return self._moments[index]

    def supports(self, index: int) -> dict[str, torch.Tensor]:
        return self._supports[index]

    def activations(self, n: int) -> dict[str, torch.Tensor]:
        """z, h, gz hidden-major [d_ff, N] and gx [N, d_model] of the last step on N tokens."""
        return {"z": self._z[:self.d_ff * n].view(self.d_ff, n), "h": self.runtime._act[:self.d_ff * n].view(self.d_ff, n),
                "gz": self._gz[:self.d_ff * n].view(self.d_ff, n), "gx": self._gx[:n]}

    def nbytes(self, index: int) -> dict[str, int]:
        """Bytes of layer ``index``: values, gradients, moments, indices (runtime and transposed supports); dense workspace;
        the partial workspace of grad_down / grad_up (shared by the layers, allocated size)."""
        def size(tensors) -> int:
            return sum(t.numel() * t.element_size() for t in tensors)
        params = self.runtime._tensors[index]
        return {"values": size(params[name] for name in PARAMETERS),
                "gradients": size(self._grads[index].values()),
                "moments": size(t for pair in self._moments[index].values() for t in pair),
                "indices": size(params[name] for name in ("up_offsets", "up_idx", "out_offsets", "out_hidden"))
                + size(self._supports[index].values()),
                "workspace": self.runtime.workspace_bytes() + size((self._z, self._gz, self._gy_t, self._gx)),
                "partials": size((self._partials,)) if self._partials is not None else 0}
