"""No-fusion GPT-NeoX/Pythia integration for exact-offset Metal FFNs."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .compact import CompactWandaLayer, load_compact_wanda_layer
from .runtime import ExactOffsetMetalFFN


def select_training_layout(spec: CompactWandaLayer, n: int) -> str:
    """Select the measured exact-offset schedule from shape/support statistics.

    ABBA campaigns on every canonical layer-0 export establish three stable
    regions: token ownership through N=6, feature ownership from N=9, and a
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


class HFFNMetalMLP(nn.Module):
    """One Pythia MLP backed by one flat compact trainable pool; its forward only.

    The training backward is the owner's native staged context (``NativeBackwardMLP``, ``native_ffn_adapter.py``):
    this class holds no backward, and its grad-enabled forward is refused.
    """

    def __init__(self, layer_idx: int, spec: CompactWandaLayer) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.spec = spec
        self.flat = nn.Parameter(torch.cat((spec.up_val, spec.down_val, spec.b_up, spec.b_down)).to("mps"))
        self._runtimes: dict[tuple[int, str], ExactOffsetMetalFFN] = {}
        self.forced_layout: str | None = None

    def set_forced_layout(self, layout: str | None) -> None:
        if layout is not None and layout not in {"token_major", "feature_persistent"}:
            raise ValueError("forced layout (test switch) must be token_major or feature_persistent")
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
        if torch.is_grad_enabled():
            raise RuntimeError("HFFNMetalMLP has no backward: the training module is NativeBackwardMLP")
        shape = tuple(x.shape)
        y, _ = self.runtime_for_n(int(x.numel() // shape[-1])).forward(x.reshape(-1, shape[-1]))
        return y.reshape(shape)

    @property
    def compact_numel(self) -> int:
        return self.flat.numel()


def install_pythia_hffn_metal_mlps(
    model: nn.Module, export_dir: str | Path, *, native_n: tuple[int, ...], library: str | Path
) -> tuple[HFFNMetalMLP, ...]:
    """Installs the exact-offset MLPs whose backward is the owner's native staged context.

    ``native_n`` lists every block size N the training will use; it is validated here, and no context is created: each
    layer prepares the native context of an N at its first call of that N and keeps one resident, discarding the
    context of the previous N (a declared bridging cost, counted by the adapter's
    ``lazy_prepares`` and ``contexts_discarded``). ``library`` is the owner's ``libhffn_train``."""
    from native_ffn_adapter import NativeBackwardMLP  # the adapter; imported here, the module is the route of the leaf

    ns = tuple(sorted({int(n) for n in native_n}))
    if not ns or ns[0] < 1:
        raise ValueError("native_n: at least one positive block size N is required")
    blocks = list(getattr(getattr(model, "gpt_neox", None), "layers", ()))
    if not blocks:
        raise ValueError("expected a Pythia GPT-NeoX model")
    root = Path(export_dir)
    modules = []
    for layer_idx, block in enumerate(blocks):
        spec = load_compact_wanda_layer(root / f"layer{layer_idx}" / "hffn_training_compact_g4_v2.pt")
        module = NativeBackwardMLP(layer_idx, spec, library)
        block.mlp = module
        modules.append(module)
    return tuple(modules)
