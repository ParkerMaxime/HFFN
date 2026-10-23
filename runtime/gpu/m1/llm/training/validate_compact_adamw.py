#!/usr/bin/env python3
"""Ten-step parity gate for the compact Metal AdamW shader."""

from __future__ import annotations

import argparse
import json
import math

import torch

from hffn_metal_llm.shaders import adamw_source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100_003)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=712)
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available; run outside the GPU sandbox")
    lr, beta1, beta2, eps, wd = 1e-3, 0.9, 0.999, 1e-8, 1e-2
    generator = torch.Generator().manual_seed(args.seed)
    initial = torch.randn(args.count, generator=generator, dtype=torch.float32)
    gradients = [torch.randn(args.count, generator=generator, dtype=torch.float32) for _ in range(args.steps)]
    metal_param = initial.to("mps")
    metal_grad = torch.empty_like(metal_param)
    metal_m = torch.zeros_like(metal_param)
    metal_v = torch.zeros_like(metal_param)
    reference = torch.nn.Parameter(initial.to("mps"))
    optimizer = torch.optim.AdamW([reference], lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=wd)
    lib = torch.mps.compile_shader(adamw_source())
    rows = []
    for step, grad_cpu in enumerate(gradients, 1):
        grad = grad_cpu.to("mps")
        reference.grad = grad.clone()
        optimizer.step()
        metal_grad.copy_(grad)
        lib.adamw_compact(
            metal_param, metal_grad, metal_m, metal_v,
            lr, beta1, beta2, eps, wd,
            lr / (1.0 - beta1 ** step), math.sqrt(1.0 - beta2 ** step),
            threads=args.count, group_size=256,
        )
        torch.mps.synchronize()
        state = optimizer.state[reference]
        error = {
            "parameter": float((metal_param - reference.detach()).abs().max().cpu()),
            "moment1": float((metal_m - state["exp_avg"]).abs().max().cpu()),
            "moment2": float((metal_v - state["exp_avg_sq"]).abs().max().cpu()),
            "cleared_grad": float(metal_grad.abs().max().cpu()),
        }
        if max(error.values()) > 5e-6:
            raise AssertionError(f"AdamW parity failed at step {step}: {error}")
        rows.append({"step": step, **error})
    recovery_param = torch.ones(3, device="mps", dtype=torch.float32)
    recovery_grad = torch.tensor([float("nan"), float("inf"), float("-inf")], device="mps")
    recovery_m = torch.zeros_like(recovery_param)
    recovery_v = torch.zeros_like(recovery_param)
    lib.adamw_compact(
        recovery_param, recovery_grad, recovery_m, recovery_v,
        lr, beta1, beta2, eps, wd,
        lr / (1.0 - beta1), math.sqrt(1.0 - beta2),
        threads=3, group_size=3,
    )
    torch.mps.synchronize()
    recovery = {
        "parameter_finite": bool(torch.isfinite(recovery_param).all().cpu()),
        "moment1_finite": bool(torch.isfinite(recovery_m).all().cpu()),
        "moment2_finite": bool(torch.isfinite(recovery_v).all().cpu()),
        "gradient_cleared": bool((recovery_grad == 0).all().cpu()),
    }
    if not all(recovery.values()):
        raise AssertionError(f"non-finite gradient recovery failed: {recovery}")
    print(json.dumps({
        "format": "hffn_metal_compact_adamw_parity_v1",
        "count": args.count,
        "steps": args.steps,
        "rows": rows,
        "nonfinite_gradient_recovery": recovery,
        "max_abs": max(max(row[key] for key in ("parameter", "moment1", "moment2", "cleared_grad")) for row in rows),
    }, indent=2))


if __name__ == "__main__":
    main()
