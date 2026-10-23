"""Explicitly leased, preallocated saved activations on one shared FFN Step.

Use ``with module.acquire_context() as saved: y = module(x, context=saved)``
and finish all wanted backwards inside that scope. Release invalidates that
graph; retain_graph works until release. Abandoned graphs need explicit release.
No-grad calls need no lease. Compact parameters/gradients/moments are shared.
Native ``module.step.update()`` after release uses the last backward gradient;
ordinary PyTorch optimizers use accumulated Parameter.grad.
External optimizer/checkpoint writes must occur outside concurrent native calls.
"""
from __future__ import annotations

from contextlib import contextmanager
import threading
import weakref

import torch

from .module import HFFNTrainFFN, PARAMETERS
from .step import LR, WD


class ContextLease:
    """One acquisition, one forward, arbitrary backwards until explicit release."""

    def __init__(self, owner, index, generation):
        self._owner = weakref.ref(owner)
        self._index, self._generation = index, generation
        self._released = False

    def release(self):
        if not self._released:
            owner = self._owner()
            if owner is not None:
                owner._release(self)
            self._released = True

    def __enter__(self):
        if self._released:
            raise RuntimeError('released context lease')
        return self

    def __exit__(self, *_):
        self.release()


class _PublicStep:
    """Read-through Step access; executing methods excludes live contexts."""

    def __init__(self, owner):
        self._owner = weakref.ref(owner)

    @property
    def lr(self):
        return self._owner()._step.lr

    @property
    def weight_decay(self):
        return self._owner()._step.weight_decay

    def __getattr__(self, name):
        owner = self._owner()
        if name in ('resize', 'forward', 'backward', 'update', 'step', 'results'):
            def guarded(*args, **kwargs):
                with owner._exclusive():
                    if any(slot['leased'] for slot in owner._slots):
                        raise RuntimeError('release saved contexts before a direct Step call')
                    if name == 'backward' and not owner._direct_ready:
                        raise RuntimeError('a direct Step forward is required before backward')
                    if name == 'results' and not owner._direct_results:
                        raise RuntimeError('direct Step results require a direct forward or step')
                    if name in ('resize', 'forward', 'step'):
                        owner._direct_ready = False
                        owner._direct_results = False
                        owner._scratch(owner._step.n)
                    result = getattr(owner._step, name)(*args, **kwargs)
                    if name == 'forward':
                        owner._direct_ready = True
                    elif name in ('resize', 'update', 'step'):
                        owner._direct_ready = False
                    if name in ('forward', 'step'):
                        owner._direct_results = True
                    return result
            return guarded
        return getattr(owner._step, name)


class _ContextFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, module, lease, up_val, down_val, b_up, b_down):
        with module._exclusive():
            slot = module._validate(lease)
            if slot['used']:
                raise RuntimeError('one forward per context lease')
            module.check_storage()
            x_nt = module._input(x)
            module._direct_ready = False
            module._direct_results = False
            slot['used'] = True  # a failed native call also requires release
            module._select(slot, x_nt.shape[0])
            module._step.forward(x_nt)
            ctx.module, ctx.index, ctx.generation = module, lease._index, lease._generation
            ctx.n, ctx.shape, ctx.steps = x_nt.shape[0], x.shape, module._step.steps
            ctx.save_for_backward(up_val, down_val, b_up, b_down)
            return module._step.y_nt.clone().view(x.shape)

    @staticmethod
    def backward(ctx, grad_y):
        module = ctx.module
        with module._exclusive():
            slot = module._slots[ctx.index]
            if not slot['leased'] or slot['generation'] != ctx.generation or not slot['used']:
                raise RuntimeError('stale backward: context released or reused')
            ctx.saved_tensors  # raises on in-place PyTorch/checkpoint updates
            module.check_storage()
            if module._step.steps != ctx.steps:
                raise RuntimeError('native update since saved forward')
            if grad_y.dtype != torch.float32 or grad_y.device.type != 'cpu':
                raise TypeError('grad_y must be float32 on the CPU')
            module._select(slot, ctx.n)
            need = ctx.needs_input_grad
            module._step.backward(grad_y.reshape(ctx.n, module._step.layer.d_model).contiguous(), input_grad=need[0])
            step = module._step
            gx = step.gx_nt.clone().view(ctx.shape) if need[0] else None
            grads = [g.clone() if need[3 + i] else None for i, g in enumerate((step.g_up, step.g_down, step.g_b_up, step.g_b_down))]
            return (gx, None, None, *grads)


class HFFNContextFFN(HFFNTrainFFN):
    """Shared FFN with an explicit maximum count of saved-forward contexts.

    Acquire/release leases outside calls. Grad-producing calls require a lease;
    no-grad calls use separate Step scratch. Calls sharing scratch are exclusive.
    ``step`` exposes the existing Step state and guarded execution methods.
    """

    def __init__(self, layer, n, *, contexts, n_min=None, kernels=None, lr=LR, weight_decay=WD):
        if type(contexts) is not int or contexts <= 0:
            raise ValueError('contexts must be a positive integer')
        super().__init__(layer, n, n_min=n_min, kernels=kernels, lr=lr, weight_decay=weight_decay)
        self._step = self.step
        self._lock = threading.Lock()
        self._direct_ready = False
        self._direct_results = False
        self._slots = [dict(leased=False, generation=0, used=False,
                            buffers={name: torch.zeros(rows * n, dtype=torch.float32)
                                     for name, rows in (('x_t', layer.d_model), ('z', layer.d_ff), ('act', layer.d_ff))})
                       for _ in range(contexts)]
        self.step = _PublicStep(self)

    @contextmanager
    def _exclusive(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('concurrent context/Step entry is not supported')
        try:
            yield
        finally:
            self._lock.release()

    def acquire_context(self, index=None):
        """Acquire a free slot outside forward/backward; explicit release required."""
        with self._exclusive():
            if index is None:
                index = next((i for i, slot in enumerate(self._slots) if not slot['leased']), None)
                if index is None:
                    raise RuntimeError('saved-context reservation exhausted')
            if type(index) is not int or not 0 <= index < len(self._slots):
                raise ValueError('invalid context index')
            slot = self._slots[index]
            if slot['leased']:
                raise RuntimeError('context is already leased')
            slot.update(leased=True, used=False, generation=slot['generation'] + 1)
            return ContextLease(self, index, slot['generation'])

    def _validate(self, lease):
        if not isinstance(lease, ContextLease) or lease._owner() is not self or lease._released:
            raise RuntimeError('foreign or released context lease')
        slot = self._slots[lease._index]
        if not slot['leased'] or slot['generation'] != lease._generation:
            raise RuntimeError('stale context lease')
        return slot

    def _release(self, lease):
        with self._exclusive():
            slot = self._validate(lease)
            slot.update(leased=False, used=False, generation=slot['generation'] + 1)

    def _input(self, x):
        if x.dtype != torch.float32 or x.device.type != 'cpu':
            raise TypeError('x must be float32 on the CPU')
        if x.ndim < 1 or x.shape[-1] != self._step.layer.d_model:
            raise ValueError('x has the wrong feature width')
        x_nt = x.reshape(-1, self._step.layer.d_model).contiguous()
        if not self._step.n_min <= x_nt.shape[0] <= self._step.n_max:
            raise ValueError('token count outside the reservation')
        return x_nt

    def _scratch(self, n):
        self._step.resize(n)  # restores the original reserved activation views

    def _select(self, slot, n):
        self._scratch(n)
        for name, flat in slot['buffers'].items():
            rows = self._step.layer.d_model if name == 'x_t' else self._step.layer.d_ff
            setattr(self._step, name, flat[:rows * n].view(rows, n))

    def forward(self, x, *, context=None):
        needs_graph = torch.is_grad_enabled() and (x.requires_grad or any(getattr(self, p).requires_grad for p in PARAMETERS))
        if needs_graph:
            if context is None:
                raise RuntimeError('acquire a context before a grad-producing forward')
            return _ContextFunction.apply(x, self, context, self.up_val, self.down_val, self.b_up, self.b_down)
        with self._exclusive():
            if context is not None:
                self._validate(context)
            self.check_storage()
            x_nt = self._input(x)
            self._direct_ready = False
            self._direct_results = False
            self._scratch(x_nt.shape[0])
            self._step.forward(x_nt)
            return self._step.y_nt.clone().view(x.shape)
