#!/usr/bin/env python3
"""CPU/MPS parity of the iso-support dense AdamW control."""

import argparse
import json
import sys
from pathlib import Path

import torch

from iso_support import synthetic_support_masks
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "runtime" / "common" / "pythia" / "hffn_pythia").is_dir())
sys.path.insert(0, str(ROOT / "runtime/common/pythia"))
from hffn_pythia.dense_control import FixedSupportAdamW  # noqa: E402  the one projection of the dense control


def build(device: torch.device, i: int, h: int, o: int, d: int):
    up = torch.zeros((h, i), device=device)
    down = torch.zeros((o, h), device=device)
    b_up, b_down = torch.zeros(h, device=device), torch.zeros(o, device=device)
    up_mask, down_mask = synthetic_support_masks(up, down, d)
    up.copy_(up_mask * 0.05)
    down.copy_(down_mask * -0.04)
    params = [torch.nn.Parameter(x) for x in (up, down, b_up, b_down)]
    masks = [up_mask, down_mask]
    grads = [torch.full_like(up, .003) * up_mask, torch.full_like(down, -.002) * down_mask,
             torch.full_like(b_up, .001), torch.full_like(b_down, -.001)]
    for p, g in zip(params, grads):
        p.grad = g
    opt = FixedSupportAdamW(params, {params[0]: up_mask, params[1]: down_mask}, lr=1e-3, betas=(.9, .999), eps=1e-8,
                            weight_decay=1e-2, foreach=False)
    return params, masks, opt


def step(params, masks, opt):
    opt.step()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=int, default=128)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--output-size", type=int, default=128)
    p.add_argument("--degree", type=int, default=13)
    p.add_argument("--steps", type=int, default=32)
    a = p.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS est requis")
    cpu = build(torch.device("cpu"), a.input, a.hidden, a.output_size, a.degree)
    mps = build(torch.device("mps"), a.input, a.hidden, a.output_size, a.degree)
    for _ in range(a.steps):
        step(*cpu)
        step(*mps)
    torch.mps.synchronize()
    fields = ["parameter", "exp_avg", "exp_avg_sq"]
    errors = {name: 0.0 for name in fields}
    for cp, mp in zip(cpu[0], mps[0]):
        errors["parameter"] = max(errors["parameter"], (cp.detach() - mp.detach().cpu()).abs().max().item())
        for name in fields[1:]:
            errors[name] = max(errors[name], (cpu[2].base.state[cp][name] - mps[2].base.state[mp][name].cpu()).abs().max().item())
    inactive = max(
        (mps[0][0].detach() * (1 - mps[1][0])).abs().max().item(),
        (mps[0][1].detach() * (1 - mps[1][1])).abs().max().item(),
        (mps[2].base.state[mps[0][0]]["exp_avg"] * (1 - mps[1][0])).abs().max().item(),
        (mps[2].base.state[mps[0][1]]["exp_avg_sq"] * (1 - mps[1][1])).abs().max().item(),
    )
    print(json.dumps({"kind": "pytorch_mps_dense_iso_support_adamw_parity", "steps": a.steps,
                      "max_abs_error_cpu_vs_mps": errors, "max_abs_inactive_slot": inactive}, indent=2))


if __name__ == "__main__":
    main()
