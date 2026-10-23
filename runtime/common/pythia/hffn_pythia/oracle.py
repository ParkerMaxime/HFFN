"""Dense oracle of one fixed-support FFN of an export, for the checks only (never a runtime path).

``dense_weights`` materialises ``W_up[d_ff, d_model]``, ``b_up``, ``W_down[d_model, d_ff]`` and ``b_down`` in float64
(zeros outside the support); ``dense_forward`` computes ``W_down gelu(W_up x + b_up) + b_down`` with the exact GELU in
the dtype of ``x``: float64 for the oracle, float32 for the dense control. The float32 weights are the float64 ones
cast back, which are the bits of the export.
"""
from __future__ import annotations

import math

import torch

from .export import hidden_of_slot


def dense_weights(layer: dict, device: torch.device | None = None) -> tuple[torch.Tensor, ...]:
    """Control only: the float64 dense matrices and biases of the fixed-support FFN."""
    up = torch.zeros(layer["d_ff"], layer["d_model"], dtype=torch.float64)
    up[hidden_of_slot(layer, "up"), layer["up_idx"].long()] = layer["up_val"].double()
    down = torch.zeros(layer["d_model"], layer["d_ff"], dtype=torch.float64)
    down[layer["down_idx"].long(), hidden_of_slot(layer, "down")] = layer["down_val"].double()
    return tuple(tensor.to(device) for tensor in (up, layer["b_up"].double(), down, layer["b_down"].double()))


def dense_forward(source: dict | tuple[torch.Tensor, ...], x: torch.Tensor) -> torch.Tensor:
    """``source``: an export layer, or the result of ``dense_weights`` when it is reused over several inputs."""
    weights = dense_weights(source, x.device) if isinstance(source, dict) else source
    up, b_up, down, b_down = (tensor.to(x.dtype) for tensor in weights)
    z = x @ up.T + b_up
    return (0.5 * z * (1.0 + torch.erf(z / math.sqrt(2.0)))) @ down.T + b_down
