"""Pythia loader shared by the dense and the HFFN arms, and by the hardware leaves.

Both arms build the same GPT-NeoX model, load the same non-FFN tensors from the base
checkpoint and read the FFN from the same compact export. They differ only in the FFN call:
- ``ffn="dense"``: the model's own ``GPTNeoXMLP`` (two ``nn.Linear`` and a GELU), its weights
  set from the export on the fixed support, zero elsewhere;
- ``ffn="hffn"``: ``HFFNMLP``, which hands the same activations to the HFFN runtime.
Embeddings, attention, LayerNorm, KV cache and head stay PyTorch in both arms.

``build_pythia`` is the only construction of a Pythia model on ``runtime/``: ``load_pythia`` and the GPU M1 loaders
call it, then install their own FFN modules.
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

# Buffers of older GPT-NeoX checkpoints that current models rebuild themselves.
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
    """Copy every non-FFN tensor of ``checkpoint`` into ``model``: the two name sets must be equal (legacy buffers
    aside), and each tensor must have the target's shape and, unless both are floating, its dtype."""
    targets = {name: tensor for name, tensor in model.state_dict().items() if ".mlp." not in name}
    with safe_open(checkpoint, framework="pt", device="cpu") as source:
        names = [name for name in source.keys() if ".mlp." not in name and (name in targets or not name.endswith(_LEGACY))]
        if set(names) != set(targets):
            raise ValueError(f"checkpoint and model tensors differ: {sorted(set(names) ^ set(targets))}")
        for name in names:
            value, target = source.get_tensor(name), targets[name]
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(f"tensor shape mismatch {name}: {tuple(value.shape)} != {tuple(target.shape)}")
            if value.dtype != target.dtype and not (value.is_floating_point() and target.is_floating_point()):
                raise ValueError(f"tensor dtype mismatch {name}: {value.dtype} != {target.dtype}")
            target.copy_(value)


class _NoFFN(nn.Module):
    """Parameterless stand-in for ``GPTNeoXMLP`` while the HFFN arm is constructed: no dense FFN is allocated
    (nor initialised); ``HFFNMLP`` takes its place before the model is used."""

    def __init__(self, config):
        super().__init__()


@contextmanager
def _without_dense_ffn():
    original = modeling_gpt_neox.GPTNeoXMLP
    if original is _NoFFN:
        raise RuntimeError("_without_dense_ffn is not reentrant")
    modeling_gpt_neox.GPTNeoXMLP = _NoFFN
    try:
        yield
    finally:
        modeling_gpt_neox.GPTNeoXMLP = original


def build_pythia(model_dir: str | Path, *, dense_ffn: bool) -> nn.Module:
    """The Pythia model of ``model_dir`` in float32 on the CPU, its non-FFN tensors loaded from ``model.safetensors``.

    With ``dense_ffn=False`` no dense FFN is constructed: every ``block.mlp`` is a parameterless ``_NoFFN`` that the
    caller replaces by its HFFN module before use. With ``dense_ffn=True`` the blocks keep the model's ``GPTNeoXMLP``,
    left as initialised (the dense arm sets its values from the export)."""
    root = Path(model_dir)
    checkpoint = root / "model.safetensors"
    if not checkpoint.is_file():
        raise ValueError(f"Pythia loader requires {checkpoint}")
    config = AutoConfig.from_pretrained(root, local_files_only=True)
    if config.model_type != "gpt_neox":
        raise ValueError(f"expected a GPT-NeoX (Pythia) model, got {config.model_type!r}")
    if dense_ffn:
        model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    else:
        with _without_dense_ffn():
            model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
        blocks = list(getattr(getattr(model, "gpt_neox", None), "layers", ()))
        named = [name for name, _ in (*model.named_parameters(), *model.named_buffers())]
        if not blocks or not all(type(block.mlp) is _NoFFN for block in blocks) or any(".mlp." in n for n in named):
            raise RuntimeError("a dense FFN was constructed in the HFFN arm")
    _load_non_ffn(model, checkpoint)
    return model


def load_pythia(model_dir: str | Path, export_dir: str | Path, ffn: str, make_runtime, device: str = "cpu", *, groups: bool = False) -> nn.Module:
    """Pythia in eval mode (float32, on ``device``) with its FFNs from ``export_dir``, executed dense or by HFFN.

    ``make_runtime(layers)`` builds the HFFN runtime of the leaf from the export layers (``read_layers``); it is
    called for the HFFN arm only, so the dense arm needs nothing of the leaf's kernels. The HFFN arm exposes its
    runtime as ``model.hffn_runtime`` (None in the dense arm). ``groups`` is passed to ``read_layers``: with it the layer
    dicts carry ``group_hidden_offsets`` (the CPU M1 leaf builds its layers from them); the default is the former call.
    """
    if ffn not in ("dense", "hffn"):
        raise ValueError("ffn must be 'dense' or 'hffn'")
    model = build_pythia(model_dir, dense_ffn=(ffn == "dense"))
    blocks = model.gpt_neox.layers
    layers = read_layers(export_dir, len(blocks), groups=groups)
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
