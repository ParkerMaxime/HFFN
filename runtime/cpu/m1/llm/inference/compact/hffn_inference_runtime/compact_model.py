"""Pythia MLP replacement that never creates autograd state."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .compact_format import HFFNCompactG4Layer, load_hffn_compact_g4_layer
from .compact_runtime import HFFNCompactG4InferenceRuntime


class HFFNCompactG4InferenceMLP(nn.Module):
    def __init__(self, index: int, layer: HFFNCompactG4Layer, runtime: HFFNCompactG4InferenceRuntime) -> None:
        super().__init__()
        self.layer_idx = index
        self.runtime = runtime
        for name in ("group_hidden_offsets", "up_row_offsets", "up_idx", "up_val", "b_up", "down_row_offsets", "down_idx", "down_val", "b_down"):
            self.register_buffer(name, getattr(layer, name), persistent=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled():
            raise RuntimeError("HFFN inference MLP requires torch.inference_mode() or torch.no_grad()")
        if hidden_states.dim() != 3 or hidden_states.device.type != "cpu" or hidden_states.dtype != torch.float32 or not hidden_states.is_contiguous():
            raise ValueError("HFFN inference MLP expects contiguous CPU float32 [B, T, H]")
        shape = tuple(hidden_states.shape)
        x_nt = hidden_states.view(-1, shape[-1])
        y_nt = self.runtime.workspace_for(self.layer_idx, int(x_nt.size(0)))
        self.runtime.forward_into(self.layer_idx, x_nt, y_nt, sequence_length=shape[1])
        return y_nt.view(shape)


def install_pythia_hffn_compact_g4_mlps(model: nn.Module, export_dir: str | Path, *, execution: str = "baseline", batch_split32: bool = True, batch_split16: bool = False, batch_split16_t96: bool = False, batch_split_t96_size: int = 96) -> HFFNCompactG4InferenceRuntime:
    blocks = list(getattr(getattr(model, "gpt_neox", None), "layers", ()))
    if not blocks:
        raise ValueError("expected a Pythia/GPT-NeoX model")
    root = Path(export_dir)
    layers = [load_hffn_compact_g4_layer(root / f"layer{index}" / "hffn_training_compact_g4_v2.pt") for index in range(len(blocks))]
    runtime = HFFNCompactG4InferenceRuntime(layers, execution=execution, batch_split32=batch_split32, batch_split16=batch_split16, batch_split16_t96=batch_split16_t96, batch_split_t96_size=batch_split_t96_size)
    for index, (block, layer) in enumerate(zip(blocks, layers)):
        block.mlp = HFFNCompactG4InferenceMLP(index, layer, runtime)
    return runtime
