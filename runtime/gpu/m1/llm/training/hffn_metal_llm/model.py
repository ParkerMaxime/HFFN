"""No-fusion GPT-NeoX/Pythia integration for exact-offset Metal FFNs."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .compact import CompactWandaLayer, load_compact_wanda_layer
from .runtime import ExactOffsetMetalFFN


def select_training_layout(spec: CompactWandaLayer, n: int) -> str:
    """Select the measured exact-offset schedule from shape/support statistics.

    Three stable regions were measured on every canonical layer-0 export: token ownership through N=6, feature ownership from N=9, and a
    density-dependent boundary at N=7/8.  The boundary uses the actual compact
    degree so it also applies to irregular supports and unseen model names.
    """
    n = int(n)
    if n < 1:
        raise ValueError("N must be positive")
    if n <= 6:
        return "token_major"
    if n >= 256:
        return "feature_owner_tile"
    if n >= 9:
        return "feature_persistent"
    mean_up_degree = spec.up_val.numel() / spec.d_ff
    mean_down_degree = spec.down_val.numel() / spec.d_ff
    return "token_major" if max(mean_up_degree, mean_down_degree) <= 64.0 else "feature_persistent"


class _MetalFFNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, flat: torch.Tensor, module: "HFFNMetalMLP") -> torch.Tensor:
        shape = tuple(x.shape)
        x2 = x.reshape(-1, shape[-1])
        runtime = module.runtime_for_n(int(x2.size(0)))
        y, forward_context = runtime.forward(x2)
        ctx.runtime = runtime
        ctx.module = module
        ctx.forward_context = forward_context
        ctx.shape = shape
        return y.reshape(shape)

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        grad = ctx.runtime.backward(ctx.forward_context, grad_y.reshape(-1, grad_y.shape[-1]))
        ctx.module._pending_mps_keepalive = grad["_mps_keepalive"]
        return grad["grad_x"].reshape(ctx.shape), grad["grad_flat"], None


class HFFNMetalMLP(nn.Module):
    """One Pythia MLP backed by one flat compact trainable pool."""

    def __init__(self, layer_idx: int, spec: CompactWandaLayer) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.spec = spec
        self.flat = nn.Parameter(torch.cat((spec.up_val, spec.down_val, spec.b_up, spec.b_down)).to("mps"))
        self._runtimes: dict[tuple[int, str], ExactOffsetMetalFFN] = {}
        self.forced_layout: str | None = None
        self._pending_mps_keepalive: tuple[torch.Tensor, ...] = ()
        self.flat.register_post_accumulate_grad_hook(self._retain_mps_workspaces)

    def _retain_mps_workspaces(self, parameter: torch.Tensor) -> None:
        """Transfer custom-shader workspaces to the accumulated flat gradient."""
        grad = parameter.grad
        if grad is None or not self._pending_mps_keepalive:
            return
        previous = getattr(grad, "_hffn_mps_keepalive", ())
        grad._hffn_mps_keepalive = previous + self._pending_mps_keepalive
        self._pending_mps_keepalive = ()

    def set_forced_layout(self, layout: str | None) -> None:
        if layout is not None and layout not in {"token_major", "feature_persistent"}:
            raise ValueError("forced layout must be token_major or feature_persistent")
        self.forced_layout = layout

    def runtime_for_n(self, n: int) -> ExactOffsetMetalFFN:
        layout = self.forced_layout or select_training_layout(self.spec, int(n))
        key = (int(n), layout)
        runtime = self._runtimes.get(key)
        if runtime is None:
            runtime = ExactOffsetMetalFFN(self.spec, int(n), flat=self.flat, layout=layout)
            self._runtimes[key] = runtime
        return runtime

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            shape = tuple(x.shape)
            y, _ = self.runtime_for_n(int(x.numel() // shape[-1])).forward(x.reshape(-1, shape[-1]))
            return y.reshape(shape)
        return _MetalFFNFunction.apply(x, self.flat, self)

    @property
    def compact_numel(self) -> int:
        return self.flat.numel()


def install_pythia_hffn_metal_mlps(model: nn.Module, export_dir: str | Path) -> tuple[HFFNMetalMLP, ...]:
    blocks = list(getattr(getattr(model, "gpt_neox", None), "layers", ()))
    if not blocks:
        raise ValueError("expected a Pythia GPT-NeoX model")
    root = Path(export_dir)
    modules = []
    for layer_idx, block in enumerate(blocks):
        spec = load_compact_wanda_layer(root / f"layer{layer_idx}" / "hffn_training_compact_g4_v2.pt")
        module = HFFNMetalMLP(layer_idx, spec)
        block.mlp = module
        modules.append(module)
    return tuple(modules)
