"""Hybrid PyTorch + compact Metal AdamW for Pythia training."""

from __future__ import annotations

import math

import torch
from torch import nn

from .model import HFFNMetalMLP
from .shaders import adamw_source


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
        self.lib = torch.mps.compile_shader(adamw_source())
        self.moments = [
            (torch.zeros_like(module.flat), torch.zeros_like(module.flat)) for module in modules
        ]
        self.steps = [0 for _ in modules]
        self._inflight: list[tuple[torch.mps.Event, tuple[torch.Tensor, ...]]] = []

    def _reap_inflight(self) -> None:
        self._inflight = [entry for entry in self._inflight if not entry[0].query()]

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._reap_inflight()
        self.base.zero_grad(set_to_none=set_to_none)
        for module in self.modules:
            if set_to_none:
                module.flat.grad = None
            elif module.flat.grad is not None:
                module.flat.grad.zero_()

    def step(self) -> None:
        self.base.step()
        submitted_grads = []
        for index, (module, (moment1, moment2)) in enumerate(zip(self.modules, self.moments)):
            grad = module.flat.grad
            if grad is None:
                continue
            submitted_grads.append(grad)
            self.steps[index] += 1
            step = self.steps[index]
            step_size = self.lr / (1.0 - self.beta1 ** step)
            beta2_scale = math.sqrt(1.0 - self.beta2 ** step)
            self.lib.adamw_compact(
                module.flat, grad, moment1, moment2,
                self.lr, self.beta1, self.beta2, self.eps,
                self.weight_decay, step_size, beta2_scale,
                threads=module.flat.numel(), group_size=256,
            )
        if submitted_grads:
            completion = torch.mps.Event()
            completion.record()
            self._inflight.append((completion, tuple(submitted_grads)))

    def set_lr(self, lr: float) -> None:
        """Update both the PyTorch and compact-Metal learning rates."""
        self.lr = float(lr)
        for group in self.base.param_groups:
            group["lr"] = self.lr

    def state_dict(self) -> dict:
        """Return a portable checkpoint containing both optimizer arms."""
        return {
            "format": "hffn_metal_adamw_state_v1",
            "hyperparameters": {
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "betas": (self.beta1, self.beta2),
                "eps": self.eps,
            },
            "base": self.base.state_dict(),
            "compact": [
                {
                    "step": step,
                    "moment1": moment1.detach().cpu(),
                    "moment2": moment2.detach().cpu(),
                }
                for step, (moment1, moment2) in zip(self.steps, self.moments)
            ],
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
        for index, (saved, (moment1, moment2)) in enumerate(zip(compact, self.moments)):
            saved_m1 = saved["moment1"]
            saved_m2 = saved["moment2"]
            if tuple(saved_m1.shape) != tuple(moment1.shape) or tuple(saved_m2.shape) != tuple(moment2.shape):
                raise ValueError(f"compact optimizer shape mismatch at layer {index}")
            moment1.copy_(saved_m1.to(device=moment1.device, dtype=moment1.dtype))
            moment2.copy_(saved_m2.to(device=moment2.device, dtype=moment2.dtype))
            self.steps[index] = int(saved["step"])
