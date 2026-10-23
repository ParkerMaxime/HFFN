"""Model-level training gate, for every hardware: distance of an evaluation to a float64 reference, and the verdict.

An evaluation is `(loss, ffn, other)`: the loss, per layer the four gradients of the active FFN entries and biases (up, down,
b_up, b_down, in the order of the compact storage), and the gradients of the non-FFN parameters by name. The distance of a
float32 evaluation to the float64 one is max|g - g64| / max|g64| per tensor, then the maximum over the tensors of a group.

Criterion: the distance to float64 of each gradient group of an HFFN
arm is at most GRADIENT_FACTOR times that of the dense float32 control, and the relative distance of its loss to float64 is at
most LOSS. Two float32 arms are not compared step by step. Pure torch: no runtime, no export, no device named; the leaf builds
its arms and gives the tensors.
"""
from __future__ import annotations

import torch

GRADIENT_FACTOR, LOSS = 10.0, 1e-4


def ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    if not bool(torch.isfinite(a).all()):          # Python's max drops NaN: a non-finite value must not pass unseen
        return float("inf")
    scale = float(b.abs().max())
    return float((a - b).abs().max()) / scale if scale > 0 else float((a - b).abs().max())


def dense_gradients(model, ids, supports):
    """Loss and gradients of a dense control: active FFN entries and biases per layer, and non-FFN tensors by name."""
    model.zero_grad(set_to_none=True)
    loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    ffn = []
    for block, (up_flat, down_flat) in zip(model.gpt_neox.layers, supports):
        up, down = block.mlp.dense_h_to_4h, block.mlp.dense_4h_to_h
        ffn.append((up.weight.grad.reshape(-1).index_select(0, up_flat), down.weight.grad.reshape(-1).index_select(0, down_flat),
                    up.bias.grad.clone(), down.bias.grad.clone()))
    other = {name: p.grad.clone() for name, p in model.named_parameters() if ".mlp." not in name and p.grad is not None}
    return float(loss), ffn, other


def gap(a, b):
    """The distance ratio between two gradients of any precision, computed in float64 one tensor at a time."""
    return ratio(a.double(), b.double())


def distance(evaluation, reference):
    loss, ffn, other = evaluation
    loss64, ffn64, other64 = reference
    return {"loss": abs(loss - loss64) / abs(loss64),
            "gradient_ffn": max(gap(a, b) for layer, layer64 in zip(ffn, ffn64) for a, b in zip(layer, layer64)),
            "gradient_non_ffn": max(gap(other[name], g) for name, g in other64.items() if name in other and float(g.abs().max()) > 0)}


def judge(rows, control, names):
    """The checks of the gate for each HFFN arm of `names` (rows `hffn_<name>`) against the row of the dense float32 control, and the verdict."""
    gate = {name: {"loss": rows[f"hffn_{name}"]["loss"] <= LOSS,
                   **{key: rows[f"hffn_{name}"][key] <= GRADIENT_FACTOR * control[key] for key in ("gradient_ffn", "gradient_non_ffn")}}
            for name in names}
    passed = all(all(checks.values()) for checks in gate.values())
    return gate, passed
