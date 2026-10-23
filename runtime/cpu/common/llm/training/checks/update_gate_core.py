"""Closed form of the first AdamW step and the criteria of the update gate. Pure torch: no runtime, no checkpoint.

After one AdamW step from zero moments: `w1 = w0 (1 - lr wd) - lr g / (|g| + eps)`, `m1 = (1 - beta1) g`, `v1 = (1 - beta2) g^2`.
Criteria, a local update form: on entries with |g| >= CONDITIONED max|g| of the tensor,
max |w - w1| / lr <= WEIGHT; moments within MOMENT_FACTOR times the distance of the dense float32 control; all finite.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

# One definition of the distance ratio, at the shared Pythia level (first ancestor that has common/pythia/hffn_pythia).
sys.path.insert(0, str(next(p / "common/pythia" for p in Path(__file__).resolve().parents if (p / "common/pythia/hffn_pythia").is_dir())))

from hffn_pythia.training_gate import ratio  # noqa: E402,F401

WEIGHT, MOMENT_FACTOR, CONDITIONED = 1e-2, 10.0, 1e-2
KINDS = ("up", "down", "b_up", "b_down")


def exact_step(w0, g, lr, wd, beta1, beta2, eps):
    """The first AdamW step in float64: initial weights, expected weights and moments, and the well-conditioned mask."""
    w0 = w0.double()
    return w0, w0 * (1.0 - lr * wd) - lr * g / (g.abs() + eps), (1.0 - beta1) * g, (1.0 - beta2) * g * g, g.abs() >= CONDITIONED * g.abs().max()


def measure(weights, first, second, expected, lr, hyper):
    """Distances of one arm to the exact step. `weights`, `first`, `second`: per layer, the four tensors of KINDS in the
    order of the compact storage; `expected`: per layer and kind, (initial float32 values, float64 gradient) in that same order."""
    out = {"weight_conditioned": 0.0, "weight_all": 0.0, "moment_m": 0.0, "moment_v": 0.0, "flipped": 0, "entries": 0, "finite": True}
    for layer, (w, m, v) in enumerate(zip(weights, first, second)):
        for kind in range(4):
            if w[kind].numel() != expected[layer][kind][1].numel():
                raise ValueError(f"layer {layer} {KINDS[kind]}: {w[kind].numel()} active entries against {expected[layer][kind][1].numel()} gradients")
            w0, w1, m1, v1, keep = exact_step(*expected[layer][kind], lr, *hyper)
            delta = torch.nan_to_num((w[kind].double() - w1).abs() / lr, nan=float("inf"))
            out["weight_all"] = max(out["weight_all"], float(delta.max()))
            if bool(keep.any()):
                out["weight_conditioned"] = max(out["weight_conditioned"], float(delta[keep].max()))
            out["moment_m"] = max(out["moment_m"], ratio(m[kind].double(), m1))
            out["moment_v"] = max(out["moment_v"], ratio(v[kind].double(), v1))
            out["flipped"] += int((((w[kind].double() - w0) * (w1 - w0)) < 0).sum())
            out["entries"] += w1.numel()
            out["finite"] = out["finite"] and all(bool(torch.isfinite(t).all()) for t in (w[kind], m[kind], v[kind]))
    out["flipped_share"] = out["flipped"] / max(out["entries"], 1)
    return out


def judge(control: dict, arms: dict) -> dict:
    """`control`: measure() of the dense float32 control; `arms`: name -> measure() of an HFFN configuration."""
    checks = {"control": control["weight_conditioned"] <= WEIGHT}
    for name, row in arms.items():
        checks[name] = {"weight": row["weight_conditioned"] <= WEIGHT, "moment_m": row["moment_m"] <= MOMENT_FACTOR * control["moment_m"],
                        "moment_v": row["moment_v"] <= MOMENT_FACTOR * control["moment_v"], "finite": row["finite"]}
    passed = checks["control"] and all(all(checks[n].values()) for n in arms)
    return {"weight": WEIGHT, "moment_factor": MOMENT_FACTOR, "conditioned": CONDITIONED, "checks": checks, "passed": passed}
