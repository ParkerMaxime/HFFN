"""A PyTorch module on the shared training step: its parameters are the step's compact values, and its autograd function
runs the native forward and backward of ``Step``.

- Parameters: ``up_val``, ``down_val``, ``b_up``, ``b_down`` share their storage with the step's layer. Any PyTorch
  optimiser (SGD, AdamW, ...) updates in place the values the next native call reads; the fused native AdamW of the step
  (``module.step.update()``) stays available. A parameter whose storage was replaced is refused at the next call.
- Token count: buffers reserved once for ``n`` tokens; a call with ``n_min <= m <= n`` tokens selects them with
  ``Step.resize`` (views, no allocation). ``x`` is ``[..., d_model]`` float32 on the CPU; its leading dimensions are the
  tokens.
- One forward in flight: the step's buffers hold the activations of its last forward only. The backward of an earlier
  forward fails closed; an in-place change of a parameter between a forward and its backward fails as in autograd.
- Outputs: ``y``, the input gradient and the four parameter gradients are new tensors owned by autograd (the work
  buffers are reused by the next call). The allocation-free path is ``Step`` itself.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .layer import Layer
from .step import LR, WD, InferenceKernel, Step, TrainingKernel, _validate_adamw_config

PARAMETERS = ("up_val", "down_val", "b_up", "b_down")


class _StepFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, module, up_val, down_val, b_up, b_down):
        step = module.step
        module.check_storage()
        if x.dtype != torch.float32 or x.device.type != "cpu":
            raise TypeError(f"x must be float32 on the CPU, not {x.dtype} on {x.device}")
        x_nt = x.reshape(-1, step.layer.d_model).contiguous()
        if x_nt.shape[0] != step.n:
            step.resize(x_nt.shape[0])
        step.forward(x_nt)
        module.generation += 1
        ctx.module, ctx.generation, ctx.n, ctx.shape = module, module.generation, step.n, x.shape
        ctx.save_for_backward(up_val, down_val, b_up, b_down)  # their version counters: an in-place change fails the backward
        return step.y_nt.clone().view(x.shape)

    @staticmethod
    def backward(ctx, grad_y):
        module = ctx.module
        step = module.step
        if ctx.generation != module.generation or ctx.n != step.n:
            raise RuntimeError("stale backward: the step's buffers hold a later forward (one forward in flight)")
        ctx.saved_tensors  # noqa: B018  (raises if a parameter was modified in place since the forward)
        if grad_y.dtype != torch.float32:
            raise TypeError(f"grad_y must be float32, not {grad_y.dtype}")
        need = ctx.needs_input_grad
        step.backward(grad_y.reshape(step.n, step.layer.d_model).contiguous(), input_grad=need[0])
        grad_x = step.gx_nt.clone().view(ctx.shape) if need[0] else None
        grads = [g.clone() if need[2 + i] else None for i, g in enumerate((step.g_up, step.g_down, step.g_b_up, step.g_b_down))]
        return (grad_x, None, *grads)


class HFFNTrainFFN(nn.Module):
    """One compact FFN layer trainable by PyTorch, on the shared step of the leaf (x86 or M1)."""

    def __init__(self, layer: Layer, n: int, *, n_min: int | None = None,
                 kernels: tuple[TrainingKernel, InferenceKernel] | None = None,
                 lr: float = LR, weight_decay: float = WD):
        lr, weight_decay = _validate_adamw_config(lr, weight_decay)
        super().__init__()
        if kernels is None:
            from . import build
            kernels = (TrainingKernel(build.training_lib()), InferenceKernel(build.inference_lib()))
        self.step = Step(layer, n, *kernels, n_min=n_min, lr=lr, weight_decay=weight_decay)
        for name in PARAMETERS:
            setattr(self, name, nn.Parameter(getattr(self.step.layer, name)))
        self.generation = 0
        self.check_storage()

    def check_storage(self) -> None:
        for name in PARAMETERS:
            parameter, value = getattr(self, name), getattr(self.step.layer, name)
            if parameter.data_ptr() != value.data_ptr() or parameter.dtype != torch.float32 or parameter.shape != value.shape:
                raise RuntimeError(f"parameter {name} no longer shares the storage of the step")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _StepFunction.apply(x, self, self.up_val, self.down_val, self.b_up, self.b_down)
