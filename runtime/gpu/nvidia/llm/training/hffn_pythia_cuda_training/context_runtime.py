"""S1 contexts around the isolated-FFN CUDA training runtime.

The forward, backward and AdamW calls are imported unchanged. Each layer owns
persistent x_t/z/h/y/gx buffers; backward scratch and tile partials remain shared.
Outputs and returned input gradients remain valid until the next use of that
layer after the complete backward and update. A context is never reused early.

One use of each layer per step, FP32, one Python thread and the default CUDA
stream are supported. Microbatch accumulation, checkpointing, retained graphs,
AMP and higher-order gradients are outside this first mechanism. Explicit reserve
is the only place this wrapper creates device storage or CuPy views.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from operator import index as integer_index

import cupy
import torch

from hffn_cuda.runtime import MAX_TOKENS
from kernels_train import grad_tiles
from train_runtime import HFFNTrainRuntime


@dataclass
class _State:
    index: int
    x_t: torch.Tensor
    z: torch.Tensor
    h: torch.Tensor
    y: torch.Tensor
    gx: torch.Tensor
    forward_work: tuple
    backward_work: tuple
    generation: int
    context_id: int | None = None
    n: int = 0
    consumed: bool = False
    saved: tuple | None = None


class ContextTrainRuntime:
    def __init__(self, train: HFFNTrainRuntime, max_tokens: int | None = None) -> None:
        self.train = self.train_runtime = train
        self.runtime = train.runtime
        self.device = self.runtime.device
        self.d_model, self.d_ff = train.d_model, train.d_ff
        self.layers = self.runtime.layers
        self._states: list[_State] = []
        self._contexts: dict[int, _State] = {}
        self._has_gradient = [False] * len(self.layers)
        self._next_context = 1
        self._generation = 0
        self._tokens = 0
        self._cycle_calls: int | None = None
        self._binding = False
        self._context_allocations = 0
        self.growths = 0
        if not self.layers:
            raise ValueError("S1 requires at least one layer")
        if not 1 <= self.d_model < 32768 or not 1 <= self.d_ff < 32768:
            raise ValueError("the current CUDA kernels require signed 16-bit feature and hidden indices")
        self.reserve(train._tokens if max_tokens is None else max_tokens)
        self.growths = 0

    @property
    def calls(self):
        return self.train.calls

    @property
    def steps(self):
        return self.train.steps

    def _check_stream(self) -> None:
        if (torch.cuda.current_stream(self.device).cuda_stream != 0
                or cupy.cuda.get_current_stream().ptr != 0):
            raise RuntimeError("S1 requires PyTorch and CuPy's default CUDA stream")

    def _check_cycle(self) -> None:
        if self._cycle_calls is not None and sum(self.runtime.calls.values()) != self._cycle_calls:
            raise RuntimeError("the underlying inference runtime was used while S1 contexts were retained")

    def _index(self, value: int) -> int:
        value = integer_index(value)
        if not 0 <= value < len(self.layers):
            raise ValueError("layer index is outside the runtime")
        return value

    def reserve(self, n: int) -> None:
        """Prepare all contexts for the largest whole token block, before use.

        Even a same-capacity reserve is refused while contexts are retained:
        their y/gx may still be referenced by the autograd graph.
        """
        n = integer_index(n)
        self._check_stream()
        if self._contexts or self._binding:
            raise RuntimeError("reserve with retained S1 contexts is forbidden")
        if not 1 <= n <= MAX_TOKENS:
            raise ValueError("unsupported token capacity")
        if max(self.d_model, self.d_ff) * n >= 2 ** 32:
            raise ValueError("activation workspace exceeds the kernels' 32-bit indexing")
        stride = max(int(columns[3]) for columns in self.train._columns)
        if grad_tiles(n) * stride >= 2 ** 32:
            raise ValueError("partial workspace exceeds the kernels' 32-bit indexing")
        if n <= self._tokens:
            return
        self.train.reserve(n)
        # The isolated runtime's own buffers form context 0, avoiding one
        # additional copy of the largest activation workspace.
        base = self.runtime
        generation = self._generation + 1
        states = []
        for layer_index in range(len(self.layers)):
            if layer_index == 0:
                x_t, z, h, y, gx = base._x_t, self.train._z, base._act, base._y, self.train._gx
            else:
                x_t = torch.empty(self.d_model * n, device=self.device, dtype=torch.float32)
                z = torch.empty(self.d_ff * n, device=self.device, dtype=torch.float32)
                h = torch.empty_like(z)
                y = torch.empty(n, self.d_model, device=self.device, dtype=torch.float32)
                gx = torch.empty_like(y)
            forward_work = tuple(cupy.asarray(tensor) for tensor in (x_t, h, y))
            backward_work = (self.train._work[0], self.train._work[1], cupy.asarray(gx))
            states.append(_State(layer_index, x_t, z, h, y, gx, forward_work, backward_work, generation))
        self._x_input = torch.empty(n, self.d_model, device=self.device, dtype=torch.float32)
        self._gy_input = torch.empty_like(self._x_input)
        self._states = states
        self._generation, self._tokens = generation, n
        self._context_allocations += len(states)
        self.growths += 1

    def _shape(self, tensor: torch.Tensor, what: str) -> tuple[int, tuple[int, ...]]:
        if (tensor.dim() < 2 or tensor.shape[-1] != self.d_model
                or tensor.dtype != torch.float32 or tensor.device != self.runtime._y.device):
            raise ValueError(f"{what} must be FP32 on the runtime device, with final dimension d_model")
        n = tensor.numel() // self.d_model
        if not 1 <= n <= self._tokens:
            raise ValueError("token block exceeds reserved capacity; call reserve before the step")
        return n, tuple(tensor.shape)

    def _tokens_view(self, tensor: torch.Tensor, n: int, staging: torch.Tensor) -> torch.Tensor:
        if tensor.is_contiguous():
            return tensor.view(n, self.d_model)
        target = staging[:n]
        target.view(tensor.shape).copy_(tensor)
        return target

    @contextmanager
    def _bind(self, state: _State):
        if self._binding:
            raise RuntimeError("S1 does not support reentrant or concurrent kernel calls")
        base, train = self.runtime, self.train
        original_base = base._x_t, base._act, base._y, base._work
        original_train = train._z, train._gx, train._work, train._saved
        self._binding = True
        base._x_t, base._act, base._y, base._work = state.x_t, state.h, state.y, state.forward_work
        train._z, train._gx, train._work, train._saved = state.z, state.gx, state.backward_work, state.saved
        try:
            yield
        finally:
            base._x_t, base._act, base._y, base._work = original_base
            train._z, train._gx, train._work, train._saved = original_train
            self._binding = False

    @torch.no_grad()
    def forward_with_context(self, index: int, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """One whole-block forward; returned y is owned by this layer's context."""
        index = self._index(index)
        self._check_stream()
        self._check_cycle()
        n, _shape = self._shape(x, "x")
        state = self._states[index]
        if state.context_id is not None or self._has_gradient[index]:
            raise RuntimeError("one use of a layer per S1 step; update before reusing it")
        if self._cycle_calls is None:
            self._cycle_calls = sum(self.runtime.calls.values())
        tokens = self._tokens_view(x, n, self._x_input)
        with self._bind(state):
            y = self.train.forward(index, tokens, state.z)
            state.saved = self.train._saved
        cid = self._next_context
        self._next_context += 1
        state.context_id, state.n, state.consumed = cid, n, False
        self._contexts[cid] = state
        return y, cid

    @torch.no_grad()
    def backward_with_context(self, context_id: int, index: int, gy: torch.Tensor) -> torch.Tensor:
        """Compact gradients remain in the isolated runtime's existing arrays."""
        index = self._index(index)
        self._check_stream()
        self._check_cycle()
        context_id = integer_index(context_id)
        state = self._contexts.get(context_id)
        if (state is None or state.index != index or state.generation != self._generation
                or state.consumed or state.saved is None):
            raise RuntimeError("invalid, stale or consumed S1 context")
        n, _shape = self._shape(gy, "gy")
        if n != state.n:
            raise ValueError("gy must have the token count of its forward")
        tokens = self._tokens_view(gy, n, self._gy_input)
        with self._bind(state):
            gx = self.train.backward(index, tokens)
        state.consumed = True
        self._has_gradient[index] = True
        return gx

    def discard_context(self, context_id: int) -> None:
        """Abandon an unconsumed graph only after the caller has finished with y.

        Consumed contexts retain gx until adamw_all; they cannot be discarded
        early. This method does not undo kernel writes or an optimizer update.
        """
        self._check_stream()
        self._check_cycle()
        context_id = integer_index(context_id)
        state = self._contexts.get(context_id)
        if state is None or state.generation != self._generation or state.consumed:
            raise RuntimeError("invalid or consumed S1 context cannot be discarded")
        del self._contexts[context_id]
        state.context_id, state.saved, state.n = None, None, 0
        if not self._contexts:
            self._cycle_calls = None

    @torch.no_grad()
    def forward(self, index: int, x: torch.Tensor) -> torch.Tensor:
        """No-grad inference using the same reserved per-layer output buffers.

        The view survives other layers, but the next inference of this same
        layer overwrites it. Inference is refused during a retained training
        graph; clone outside this API if a longer-lived output is needed.
        """
        index = self._index(index)
        self._check_stream()
        if self._contexts or self._binding:
            raise RuntimeError("inference while S1 training contexts are retained is forbidden")
        n, _shape = self._shape(x, "x")
        tokens = self._tokens_view(x, n, self._x_input)
        with self._bind(self._states[index]):
            return self.train.forward(index, tokens)

    def force(self, up: str | None, down: str | None) -> None:
        if self._contexts or self._binding:
            raise RuntimeError("dispatch cannot change while S1 contexts are retained")
        self.runtime.force(up, down)

    def ensure_can_step(self) -> None:
        """Preflight both optimizers before either one mutates a parameter."""
        self._check_stream()
        self._check_cycle()
        if self._binding or any(not state.consumed for state in self._contexts.values()):
            raise RuntimeError("AdamW requires the complete S1 backward")

    @torch.no_grad()
    def adamw_all(self) -> None:
        self.ensure_can_step()
        for index, has_gradient in enumerate(self._has_gradient):
            if has_gradient:
                self.train.adamw(index)
        # Queued kernels and autograd consumers precede future reuse on the
        # required default stream. Physical CUDA buffers are retained.
        for state in self._contexts.values():
            state.context_id, state.saved, state.n = None, None, 0
            state.consumed = False
        self._contexts.clear()
        self._has_gradient = [False] * len(self.layers)
        self._cycle_calls = None

    @torch.no_grad()
    def zero_grad(self) -> None:
        self._check_stream()
        if self._contexts or self._binding:
            raise RuntimeError("zero_grad while S1 contexts are retained is forbidden")
        for index in range(len(self.layers)):
            for gradient in self.train.gradients(index).values():
                gradient.zero_()
        self._has_gradient = [False] * len(self.layers)

    def parameters(self, index: int) -> dict[str, torch.Tensor]:
        """Canonical device values; out_val is output-major, never a host refresh."""
        return self.train.parameters(self._index(index))

    def gradients(self, index: int) -> dict[str, torch.Tensor]:
        return self.train.gradients(self._index(index))

    def moments(self, index: int) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        return self.train.moments(self._index(index))

    def supports(self, index: int) -> dict[str, torch.Tensor]:
        return self.train.supports(self._index(index))

    def flat_support(self, index: int, device=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Gate-only dense indices in exactly the parameter/gradient orders.

        Up follows hidden-major up_val; down follows output-major out_val.
        This helper allocates index tensors and is outside the runtime path.
        It reads immutable supports, never the stale host weight values.
        """
        layer = self.layers[self._index(index)]
        up_offsets = layer.up_offsets.detach().cpu().long()
        out_offsets = layer.out_offsets.detach().cpu().long()
        up_owner = torch.repeat_interleave(torch.arange(self.d_ff), up_offsets.diff())
        out_owner = torch.repeat_interleave(torch.arange(self.d_model), out_offsets.diff())
        up = up_owner * self.d_model + layer.up_idx.detach().cpu().long()
        down = out_owner * self.d_ff + layer.out_hidden.detach().cpu().long()
        if device is not None:
            up, down = up.to(device), down.to(device)
        return up, down

    def activations(self, context_id: int) -> dict[str, torch.Tensor]:
        """Gate-only views, valid while the context is retained."""
        state = self._contexts.get(integer_index(context_id))
        if state is None or state.generation != self._generation:
            raise RuntimeError("invalid or stale S1 context")
        n = state.n
        return {"x_t": state.x_t[:self.d_model * n].view(self.d_model, n),
                "z": state.z[:self.d_ff * n].view(self.d_ff, n),
                "h": state.h[:self.d_ff * n].view(self.d_ff, n),
                "y": state.y[:n], "gx": state.gx[:n]}

    def workspace_stats(self) -> dict[str, int]:
        return {"allocated": self._context_allocations, "capacity": self._tokens,
                "live": len(self._contexts), "awaiting_backward": sum(not s.consumed for s in self._contexts.values()),
                "growths": self.growths, "train_growths": self.train.growths}

    def nbytes(self) -> dict[str, int]:
        """Allocated bytes, with activation contexts counted once."""
        size = lambda tensors: sum(t.numel() * t.element_size() for t in tensors)
        compact = [self.train.nbytes(index) for index in range(len(self.layers))]
        return {**{name: sum(row[name] for row in compact) for name in ("values", "gradients", "moments", "indices")},
                "contexts": size(t for s in self._states for t in (s.x_t, s.z, s.h, s.y, s.gx)),
                "shared_workspace": size((self.train._gz, self.train._gy_t, self._x_input, self._gy_input)),
                "partials": size((self.train._partials,)) if self.train._partials is not None else 0}
