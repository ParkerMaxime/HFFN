#!/usr/bin/env python3
"""Numerical parity gate for one compact-G4 HFFN training layer against the direct HFFN equation."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from hffn_training_runtime import HFFNCompactG4TrainingRuntime, load_hffn_compact_g4_layer


def reference(layer, x, up, down, b_up, b_down):
    """Direct HFFN equation, deliberately using the compact slots only: hidden r owns [offsets[r], offsets[r + 1])."""
    up_rows = torch.repeat_interleave(torch.arange(layer.d_ff), layer.up_row_offsets.diff().long())
    down_rows = torch.repeat_interleave(torch.arange(layer.d_ff), layer.down_row_offsets.diff().long())
    w_up = torch.zeros(layer.d_ff * layer.d_model, dtype=x.dtype).scatter_add(
        0, up_rows * layer.d_model + layer.up_idx.long(), up
    ).reshape(layer.d_ff, layer.d_model)
    w_down = torch.zeros(layer.d_model * layer.d_ff, dtype=x.dtype).scatter_add(
        0, layer.down_idx.long() * layer.d_ff + down_rows, down
    ).reshape(layer.d_model, layer.d_ff)
    z = x @ w_up.T + b_up
    hidden = 0.5 * z * (1.0 + torch.erf(z * 0.7071067811865475))
    return hidden @ w_down.T + b_down


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", required=True)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--atol", type=float, default=3e-5)
    args = parser.parse_args()

    # Keep the reference gate deterministic and avoid oversubscribing the
    # native G4 worker pool with a large BLAS pool on small test matrices.
    torch.set_num_threads(1)
    torch.manual_seed(17)
    layer = load_hffn_compact_g4_layer(Path(args.export))
    runtime = HFFNCompactG4TrainingRuntime([layer])
    x = torch.randn(args.n, layer.d_model)
    grad_y = torch.randn_like(x)
    y_native = runtime.forward(0, x).clone()
    native = runtime.backward(0, grad_y)

    x_ref = x.clone().requires_grad_(True)
    up = layer.up_val.clone().requires_grad_(True)
    down = layer.down_val.clone().requires_grad_(True)
    b_up = layer.b_up.clone().requires_grad_(True)
    b_down = layer.b_down.clone().requires_grad_(True)
    y_ref = reference(layer, x_ref, up, down, b_up, b_down)
    (y_ref * grad_y).sum().backward()
    results = {
        "forward": max_abs(y_native, y_ref.detach()),
        "grad_x": max_abs(native["grad_x"], x_ref.grad),
        "grad_up": max_abs(native["grad_up"], up.grad),
        "grad_down": max_abs(native["grad_down"], down.grad),
        "grad_b_up": max_abs(native["grad_b_up"], b_up.grad),
        "grad_b_down": max_abs(native["grad_b_down"], b_down.grad),
    }
    print(results)
    if max(results.values()) > args.atol:
        raise SystemExit(f"Parity failed: max error {max(results.values())} > atol {args.atol}")


if __name__ == "__main__":
    main()
