"""Hybrid PyTorch + native compact AdamW for Pythia training.

the compact arm is the owner's optimizer-only step of ``gpu/m1/ffn/training`` (one dedicated
native context per layer, ``NativeFFNAdapter.optimizer_step``); the non-FFN weights stay the dense PyTorch AdamW. The consumer
keeps its moment form (lerp), its non-finite policy and its step counters; it computes the float32 scalars of each update.
"""

from __future__ import annotations

import math

import numpy
import torch
from torch import nn

from .model import HFFNMetalMLP


def _f32(value: float) -> float:
    return float(numpy.float32(value))


class HFFNMetalAdamW:
    def __init__(
        self,
        model: nn.Module,
        modules: tuple[HFFNMetalMLP, ...],
        *,
        lr: float,
        weight_decay: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        for module in modules:
            if getattr(module, "adapter", None) is None:
                raise TypeError("HFFNMetalAdamW: every compact module needs the native adapter (NativeBackwardMLP)")
        self.modules = modules
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.beta1, self.beta2 = map(float, betas)
        self.eps = float(eps)
        compact_ids = {id(module.flat) for module in modules}
        base_params = [parameter for parameter in model.parameters() if id(parameter) not in compact_ids]
        self.base = torch.optim.AdamW(
            base_params,
            lr=self.lr,
            betas=(self.beta1, self.beta2),
            eps=self.eps,
            weight_decay=self.weight_decay,
        )
        self.steps = [0 for _ in modules]
        for module in modules:
            module.adapter.prepare_optimizer()

    @property
    def moments(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Read-back of the compact moments per layer (CPU float32, flat order), for the gates that read them; cold, a copy."""
        return [module.adapter.export_moments() for module in self.modules]

    def _reap_inflight(self) -> None:
        """Kept for callers: every native update waits for its own completion, nothing stays in flight."""

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base.zero_grad(set_to_none=set_to_none)
        for module in self.modules:
            if set_to_none:
                module.flat.grad = None
            elif module.flat.grad is not None:
                module.flat.grad.zero_()

    def step(self) -> None:
        self.base.step()
        for index, module in enumerate(self.modules):
            grad = module.flat.grad
            if grad is None:
                continue
            self.steps[index] += 1
            step = self.steps[index]
            step_size = self.lr / (1.0 - self.beta1 ** step)
            beta2_scale = math.sqrt(1.0 - self.beta2 ** step)
            scalars = tuple(_f32(v) for v in (self.lr, self.beta1, self.beta2, self.eps, self.weight_decay,
                                              step_size, beta2_scale))
            module.adapter.optimizer_step(scalars, grad)

    def set_lr(self, lr: float) -> None:
        """Update both the PyTorch and compact-native learning rates."""
        self.lr = float(lr)
        for group in self.base.param_groups:
            group["lr"] = self.lr

    def state_dict(self) -> dict:
        """Return a portable checkpoint containing both optimizer arms."""
        compact = []
        for step, module in zip(self.steps, self.modules):
            moment1, moment2 = module.adapter.export_moments()
            compact.append({"step": step, "moment1": moment1, "moment2": moment2})
        return {
            "format": "hffn_metal_adamw_state_v1",
            "hyperparameters": {
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "betas": (self.beta1, self.beta2),
                "eps": self.eps,
            },
            "base": self.base.state_dict(),
            "compact": compact,
        }

    def load_state_dict(self, state: dict) -> None:
        if not isinstance(state, dict) or state.get("format") != "hffn_metal_adamw_state_v1":
            raise ValueError("expected hffn_metal_adamw_state_v1")
        compact = state.get("compact")
        if not isinstance(compact, list) or len(compact) != len(self.modules):
            raise ValueError("compact optimizer layer count mismatch")
        self.base.load_state_dict(state["base"])
        hyper = state["hyperparameters"]
        saved_betas = tuple(map(float, hyper["betas"]))
        if saved_betas != (self.beta1, self.beta2) or float(hyper["eps"]) != self.eps:
            raise ValueError("AdamW beta/epsilon mismatch")
        self.weight_decay = float(hyper["weight_decay"])
        self.set_lr(float(hyper["lr"]))
        for index, (saved, module) in enumerate(zip(compact, self.modules)):
            expected = tuple(module.flat.shape)
            if tuple(saved["moment1"].shape) != expected or tuple(saved["moment2"].shape) != expected:
                raise ValueError(f"compact optimizer shape mismatch at layer {index}")
            module.adapter.restore_moments(saved["moment1"], saved["moment2"], int(saved["step"]))
            self.steps[index] = int(saved["step"])
