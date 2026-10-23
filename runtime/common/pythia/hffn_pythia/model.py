"""Pythia loader shared by the dense and the HFFN arms, and by the hardware leaves.

Both arms build the same GPT-NeoX model, load the same non-FFN tensors from the base
checkpoint and read the FFN from the same compact export. They differ only in the FFN call:
- ``ffn="dense"``: the model's own ``GPTNeoXMLP`` (two ``nn.Linear`` and a GELU), its weights
  set from the export on the fixed support, zero elsewhere;
- ``ffn="hffn"``: ``HFFNMLP``, which hands the same activations to the HFFN runtime.
Embeddings, attention, LayerNorm, KV cache and head stay PyTorch in both arms.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.gpt_neox import modeling_gpt_neox

from .export import hidden_of_slot, read_layers

# Buffers of older GPT-NeoX checkpoints that the model classes rebuild themselves.
_LEGACY = (".attention.bias", ".attention.masked_bias", ".attention.rotary_emb.inv_freq")


class HFFNMLP(nn.Module):
    """Replacement of ``block.mlp``. Its output is a view of the runtime's output buffer, which
    the block consumes (residual sum) before the next FFN call overwrites it."""

    def __init__(self, runtime, index: int) -> None:
        super().__init__()
        self.runtime, self.index = runtime, index

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        return self.runtime.forward(self.index, hidden_states.view(-1, shape[-1])).view(shape)


@torch.no_grad()
def _set_dense_mlp(mlp: nn.Module, layer: dict) -> None:
    up, down = mlp.dense_h_to_4h, mlp.dense_4h_to_h
    if tuple(up.weight.shape) != (layer["d_ff"], layer["d_model"]):
        raise ValueError("export and model dimensions differ")
    up.weight.zero_()
    up.weight[hidden_of_slot(layer, "up"), layer["up_idx"].long()] = layer["up_val"]
    up.bias.copy_(layer["b_up"])
    down.weight.zero_()
    down.weight[layer["down_idx"].long(), hidden_of_slot(layer, "down")] = layer["down_val"]
    down.bias.copy_(layer["b_down"])


@torch.no_grad()
def _load_non_ffn(model: nn.Module, checkpoint: Path) -> None:
    targets = {name: tensor for name, tensor in model.state_dict().items() if ".mlp." not in name}
    with safe_open(checkpoint, framework="pt", device="cpu") as source:
        names = [name for name in source.keys() if ".mlp." not in name and (name in targets or not name.endswith(_LEGACY))]
        if set(names) != set(targets):
            raise ValueError(f"checkpoint and model tensors differ: {sorted(set(names) ^ set(targets))}")
        for name in names:
            targets[name].copy_(source.get_tensor(name))


class _NoFFN(nn.Module):
    """Parameterless stand-in for ``GPTNeoXMLP`` while the HFFN arm is constructed: no dense FFN is allocated
    (nor initialised); ``HFFNMLP`` takes its place before the model is used."""

    def __init__(self, config):
        super().__init__()


@contextmanager
def _without_dense_ffn():
    original = modeling_gpt_neox.GPTNeoXMLP
    modeling_gpt_neox.GPTNeoXMLP = _NoFFN
    try:
        yield
    finally:
        modeling_gpt_neox.GPTNeoXMLP = original


def load_pythia(model_dir: str | Path, export_dir: str | Path, ffn: str, make_runtime, device: str = "cpu") -> nn.Module:
    """Pythia in eval mode (float32, on ``device``) with its FFNs from ``export_dir``, executed dense or by HFFN.

    ``make_runtime(layers)`` builds the HFFN runtime of the leaf from the export layers (``read_layers``); it is
    called for the HFFN arm only, so the dense arm needs nothing of the leaf's kernels. The HFFN arm exposes its
    runtime as ``model.hffn_runtime`` (None in the dense arm).
    """
    if ffn not in ("dense", "hffn"):
        raise ValueError("ffn must be 'dense' or 'hffn'")
    config = AutoConfig.from_pretrained(model_dir)
    if ffn == "hffn":
        with _without_dense_ffn():
            model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
        assert not any(".mlp." in name for name, _ in model.named_parameters())
    else:
        model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    _load_non_ffn(model, Path(model_dir) / "model.safetensors")
    blocks = model.gpt_neox.layers
    layers = read_layers(export_dir, len(blocks))
    runtime = None
    if ffn == "dense":
        for block, layer in zip(blocks, layers):
            _set_dense_mlp(block.mlp, layer)
    else:
        runtime = make_runtime(layers)
        for index, block in enumerate(blocks):
            block.mlp = HFFNMLP(runtime, index)
    model = model.to(device).eval().requires_grad_(False)
    model.hffn_runtime = runtime
    return model


def share_non_ffn(model: nn.Module, source: nn.Module) -> None:
    """Make ``model`` read the non-FFN parameters of ``source`` (both loaded from the same base checkpoint)
    instead of its own copy: one copy per process, at one place in memory. Used by the timing harness, whose arms
    must differ by the FFN only; two copies of the same weights at different addresses do not run at the same speed."""
    owned = dict(source.named_parameters())
    for prefix, module in model.named_modules():
        for key in module._parameters:
            name = f"{prefix}.{key}" if prefix else key
            if ".mlp." not in name and module._parameters[key] is not None:
                module._parameters[key] = owned[name]
