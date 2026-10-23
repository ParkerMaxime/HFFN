#!/usr/bin/env python3
"""Forward/backward parity against dense fixed-support PyTorch/MPS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from hffn_metal_llm import ExactOffsetMetalFFN, load_compact_wanda_layer
from hffn_pythia.oracle import dense_forward, dense_weights  # on sys.path once the runtime package is imported

LAYER_KEYS = ("d_ff", "d_model", "up_row_offsets", "up_idx", "up_val", "b_up",
              "down_row_offsets", "down_idx", "down_val", "b_down")


def active_dense_grad(spec, grad: torch.Tensor, projection: str) -> torch.Tensor:
    values = []
    offsets = spec.up_row_offsets if projection == "up" else spec.down_row_offsets
    index = spec.up_idx if projection == "up" else spec.down_idx
    for r in range(spec.d_ff):
        begin, end = int(offsets[r]), int(offsets[r + 1])
        ids = index[begin:end].to(device=grad.device, dtype=torch.int64)
        values.append(grad[r, ids] if projection == "up" else grad[ids, r])
    return torch.cat(values)


def errors(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    delta = (actual.detach().cpu() - expected.detach().cpu()).abs()
    denom = expected.detach().cpu().abs().clamp_min(1e-7)
    return {"max_abs": float(delta.max()), "max_rel": float((delta / denom).max())}


def validate(spec, n: int, seed: int, layout: str) -> dict:
    torch.manual_seed(seed + n)
    device = "mps"
    x = torch.randn(n, spec.d_model, dtype=torch.float32, device=device, requires_grad=True)
    gy = torch.randn(n, spec.d_model, dtype=torch.float32, device=device)
    x64 = x.detach().cpu().double().requires_grad_()
    up, bup, down, bdown = (t.requires_grad_() for t in dense_weights({key: getattr(spec, key) for key in LAYER_KEYS}))
    reference = dense_forward((up, bup, down, bdown), x64)
    reference.backward(gy.cpu().double())

    runtime = ExactOffsetMetalFFN(spec, n, layout=layout)
    y, context = runtime.forward(x.detach())
    grad = runtime.backward(context, gy)
    expected = {
        "output": reference,
        "grad_x": x64.grad,
        "grad_up": active_dense_grad(spec, up.grad, "up"),
        "grad_down": active_dense_grad(spec, down.grad, "down"),
        "grad_b_up": bup.grad,
        "grad_b_down": bdown.grad,
    }
    actual = {"output": y, **{key: grad[key] for key in expected if key != "output"}}
    result = {name: errors(actual[name], expected[name]) for name in expected}
    max_abs = max(value["max_abs"] for value in result.values())
    # The FFN kernel uses an erf approximation of GELU.  This gate compares
    # against torch GELU exact, hence a small activation approximation
    # tolerance is expected.
    if max_abs > 5e-4:
        raise AssertionError(f"N={n} exact-offset parity failed: {result}")
    return {"layout": layout, "n": n, "max_abs": max_abs, "errors": result}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=Path, required=True)
    parser.add_argument("--n", type=int, nargs="+", default=[1, 8, 64, 128, 256])
    parser.add_argument(
        "--layout",
        choices=("token_major", "feature_persistent", "feature_owner_tile"),
        nargs="+",
        default=["token_major", "feature_persistent"],
    )
    parser.add_argument("--seed", type=int, default=712)
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available; run outside the GPU sandbox")
    spec = load_compact_wanda_layer(args.layer)
    rows = [validate(spec, n, args.seed, layout) for layout in args.layout for n in args.n]
    print(json.dumps({
        "format": "hffn_metal_wanda_exact_offset_layer_parity_v1",
        "layer": str(args.layer.resolve()),
        "shape": {"d_model": spec.d_model, "d_ff": spec.d_ff},
        "nnz": {"up": spec.up_val.numel(), "down": spec.down_val.numel()},
        "route_edge_bits": {
            "up": spec.routes.up_by_input.edge_bits,
            "down": spec.routes.down_by_output.edge_bits,
        },
        "rows": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
