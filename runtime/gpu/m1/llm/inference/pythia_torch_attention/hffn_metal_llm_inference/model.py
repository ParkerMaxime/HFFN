"""No-fusion GPT-NeoX/Pythia integration: every MLP runs the native HFFN FFN (`native_transformer.hffn`).

Each layer is one bank and one route of that FFN, read from the canonical exact-offset export
(`hffn_training_compact_g4_v2`). Weights, indices and offsets stay compact; the down values are placed in
output-owner order once, at load. The FFN keeps every tensor its queued kernels read until their MPS event.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

# The FFN kernel is the package `native_transformer.hffn` of this leaf.
_LEAF = next(p for p in Path(__file__).resolve().parents if (p / "runtime" / "common" / "pythia" / "hffn_pythia").is_dir()) / "runtime/gpu/m1/llm/inference"
if str(_LEAF) not in sys.path:
    sys.path.insert(0, str(_LEAF))

from native_transformer.hffn.bank_store import device_bank, validate  # noqa: E402
from native_transformer.hffn.ffn import MoEFFN
from native_transformer.hffn.stats import Stats

from .compact import load_compact_wanda_inference_layer

EXPORT = "hffn_training_compact_g4_v2.pt"
FIELDS = ("d_model", "d_ff", "up_row_offsets", "up_idx", "up_val", "b_up", "down_row_offsets", "down_idx", "down_val")


class PythiaExportStore:
    """The bank store of `MoEFFN` over exported layers (one file each): one bank per layer, route 0 runs bank 0
    alone."""

    def __init__(self, paths: list[Path]) -> None:
        specs = [load_compact_wanda_inference_layer(path) for path in paths]
        shapes = {(spec.d_model, spec.d_ff) for spec in specs}
        if len(shapes) != 1:
            raise ValueError(f"inconsistent FFN shapes: {sorted(shapes)}")
        self.layers, self.pairs, self.banks_per_layer = len(specs), ((),), 1
        self.d_model = specs[0].d_model
        self.widths = [[spec.d_ff] for spec in specs]
        self.output_biases = tuple(spec.b_down for spec in specs)
        self._raw = [{field: getattr(spec, field) for field in FIELDS} for spec in specs]

    def load(self, layer: int, bank: int):
        if bank != 0:
            raise KeyError("a Pythia layer has one bank")
        raw = self._raw[layer]
        validate(raw, self.d_model)
        return device_bank(raw)


class NativeHFFNInference:
    """The native FFN of every layer of one Pythia model, with the counters the leaf's checks read."""

    def __init__(self, store: PythiaExportStore, workspace_limit: int | None = None) -> None:
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available")
        self.store, self.stats = store, Stats()
        limit = {} if workspace_limit is None else {"workspace_limit": int(workspace_limit)}   # None: the kernel's default
        self.ffn = MoEFFN(store, store.output_biases, self.stats, **limit)
        self.ffn.prepare([0])
        self._gates: dict[int, torch.Tensor] = {}
        self._plans: dict[int, tuple] = {}

    def forward(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        """x: MPS FP32 [batch, steps, d_model]; one route covering every row."""
        rows, steps = int(x.shape[0]), int(x.shape[1])
        gates = self._gates.get(rows)
        if gates is None:     # one per batch size; never read by a one-bank route
            gates = self._gates[rows] = torch.ones(rows, device="mps")
            self._plans[rows] = ((0, 0, rows),)
        return self.ffn.forward(layer, x.contiguous(), self._plans[rows], gates, "decode" if steps == 1 else "prefill")

    def attach_model(self, model: nn.Module) -> None:
        """Nothing to attach: the FFN retains the tensors of its queued calls itself."""

    def bank_numel(self, layer: int) -> int:
        bank = self.ffn.banks[layer, 0]
        return bank.up_weights.numel() + bank.down_values.numel()

    def workspace_stats(self) -> dict:
        inflight = self.ffn.inflight
        while inflight and inflight[0][0].query():
            inflight.popleft()
        return {
            "allocated": self.stats.workspace_allocations,
            "pending_inputs": 0,
            "inflight_calls": len(inflight),
            "counters": self.stats.snapshot(),
        }

    def synchronize(self) -> None:
        torch.mps.synchronize()
        self.ffn.inflight.clear()

    def close(self) -> None:
        self.ffn.close()


class HFFNMetalInferenceMLP(nn.Module):
    def __init__(self, layer_idx: int, runtime: NativeHFFNInference) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.runtime = runtime

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled():
            raise RuntimeError("HFFN Metal inference requires torch.inference_mode() or torch.no_grad()")
        if hidden_states.device.type != "mps" or hidden_states.dtype != torch.float32:
            raise ValueError("HFFN Metal inference expects MPS float32 hidden states")
        shape = tuple(hidden_states.shape)
        if len(shape) < 2 or shape[-1] != self.runtime.store.d_model:
            raise ValueError("invalid HFFN hidden-state shape")
        x = hidden_states.reshape(shape[0], -1, shape[-1]) if len(shape) > 2 else hidden_states.unsqueeze(0)
        return self.runtime.forward(self.layer_idx, x).view(shape)

    @property
    def compact_numel(self) -> int:
        return self.runtime.bank_numel(self.layer_idx) + self.runtime.store.d_model


def install_pythia_hffn_metal_inference_mlps(
    model: nn.Module,
    export_dir: str | Path,
) -> tuple[NativeHFFNInference, tuple[HFFNMetalInferenceMLP, ...]]:
    blocks = list(getattr(getattr(model, "gpt_neox", None), "layers", ()))
    if not blocks:
        raise ValueError("expected a Pythia GPT-NeoX model")
    root = Path(export_dir)
    runtime = NativeHFFNInference(PythiaExportStore([root / f"layer{layer}" / EXPORT for layer in range(len(blocks))]))
    modules = []
    for layer_idx, block in enumerate(blocks):
        module = HFFNMetalInferenceMLP(layer_idx, runtime)
        block.mlp = module
        modules.append(module)
    return runtime, tuple(modules)
