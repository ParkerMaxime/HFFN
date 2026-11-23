"""Leaf-local autograd MLP and Pythia loader for the S1 CUDA contexts.

Non-FFN parameters, attention, loss and their optimizer remain PyTorch. Compact
FFN arrays are owned and updated by the CUDA runtime, not registered again as
PyTorch parameters. Their gradients are available through runtime.gradients.
The caller reserves capacity before a step and calls the optimizer only after
the entire model backward. Runtime state is not a model.state_dict checkpoint.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.autograd.function import once_differentiable

from hffn_cuda.runtime import HFFNLayer, HFFNRuntime
from hffn_pythia import model as shared
from train_runtime import HFFNTrainRuntime

from .context_runtime import ContextTrainRuntime


class _FFN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, module: "HFFNTrainingMLP"):
        y, cid = module.runtime.forward_with_context(module.index, x)
        ctx.module, ctx.shape, ctx.context_id = module, tuple(x.shape), cid
        return y.view(ctx.shape)

    @staticmethod
    @once_differentiable
    def backward(ctx, gy: torch.Tensor):
        gx = ctx.module.runtime.backward_with_context(ctx.context_id, ctx.module.index, gy)
        return gx.view(ctx.shape), None


class HFFNTrainingMLP(nn.Module):
    def __init__(self, runtime: ContextTrainRuntime, index: int) -> None:
        super().__init__()
        self.runtime, self.index = runtime, index

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            return self.runtime.forward(self.index, hidden_states).view(hidden_states.shape)
        if not hidden_states.requires_grad:
            raise RuntimeError("S1 training requires inputs connected to trainable non-FFN parameters")
        return _FFN.apply(hidden_states, self)


def _runtime(layers: list[dict], device: str, max_tokens: int) -> ContextTrainRuntime:
    kernel_layers = [HFFNLayer.from_hidden_major(
        layer["d_model"], layer["d_ff"], layer["b_up"], layer["up_row_offsets"], layer["up_idx"], layer["up_val"],
        layer["b_down"], layer["down_row_offsets"], layer["down_idx"], layer["down_val"],
    ) for layer in layers]
    native = HFFNRuntime(kernel_layers, device)
    train = HFFNTrainRuntime(native, max_tokens=max_tokens)
    return ContextTrainRuntime(train, max_tokens=max_tokens)


def load_pythia(model_dir: str | Path, export_dir: str | Path, device: str = "cuda", *,
                max_tokens: int = 2048) -> nn.Module:
    """Trainable Pythia using the shared streamed loader, with no dense FFNs.

    Cache is disabled for training. The caller must keep the CUDA device and
    FP32 dtype fixed after loading: runtime arrays are not nn.Module buffers.
    """
    model = shared.load_pythia(model_dir, export_dir, "hffn",
                               lambda layers: _runtime(layers, device, max_tokens), device)
    runtime = model.hffn_runtime
    for index, block in enumerate(model.gpt_neox.layers):
        block.mlp = HFFNTrainingMLP(runtime, index)
    model.config.use_cache = False
    return model.train().requires_grad_(True)


class HFFNTrainingOptimizer:
    """Dense PyTorch non-FFN AdamW followed by the existing compact AdamW.

    Dense controls must instantiate the same non-FFN AdamW options and parameter
    order; their FFN optimizer is separate. The native compact state is read
    through runtime.parameters/gradients/moments rather than PyTorch .grad.
    """
    def __init__(self, model: nn.Module, runtime: ContextTrainRuntime, *, lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 1e-2, fused: bool = True) -> None:
        if runtime is not getattr(model, "hffn_runtime", None):
            raise ValueError("optimizer runtime must be the one installed in the model")
        if any(".mlp." in name for name, _ in model.named_parameters()):
            raise ValueError("S1's PyTorch optimizer must contain no FFN parameters")
        self.runtime = runtime
        runtime.ensure_can_step()
        if runtime.workspace_stats()["live"]:
            raise RuntimeError("create the optimizer before retaining an S1 graph")
        self.base = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay, fused=fused)
        # PyTorch validates the shared hyperparameters before the native state
        # changes, so a rejected optimizer cannot leave a different FFN update.
        runtime.train.lr, runtime.train.betas = lr, betas
        runtime.train.eps, runtime.train.weight_decay = eps, weight_decay

    def zero_grad(self, set_to_none: bool = False) -> None:
        # Preflight before mutating either collection of gradients.
        if self.runtime.workspace_stats()["live"]:
            raise RuntimeError("zero_grad while S1 contexts are retained is forbidden")
        self.runtime.zero_grad()
        self.base.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self.runtime.ensure_can_step()
        if len(self.base.param_groups) != 1:
            raise RuntimeError("S1 supports one shared AdamW hyperparameter group")
        group = self.base.param_groups[0]
        # Keep a scheduler's changes identical for compact and non-FFN weights.
        self.runtime.train.lr, self.runtime.train.betas = group["lr"], group["betas"]
        self.runtime.train.eps, self.runtime.train.weight_decay = group["eps"], group["weight_decay"]
        self.base.step()
        self.runtime.adamw_all()

    @property
    def param_groups(self):
        return self.base.param_groups


def create_optimizer(model: nn.Module, **kwargs) -> HFFNTrainingOptimizer:
    runtime = getattr(model, "hffn_runtime", None)
    if not isinstance(runtime, ContextTrainRuntime):
        raise ValueError("expected a Pythia model loaded with the S1 runtime")
    return HFFNTrainingOptimizer(model, runtime, **kwargs)
