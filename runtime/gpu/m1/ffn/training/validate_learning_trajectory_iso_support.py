#!/usr/bin/env python3
"""Compare Metal HFFN and PyTorch/MPS on the same fixed sparse support.

The dense reference must project gradients, parameters, and AdamW moments back
onto the fixed support at every step. Otherwise it silently becomes a dense
network after the first update and no longer tests HFFN learning parity.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from iso_support import synthetic_support_masks
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "runtime" / "common" / "pythia" / "hffn_pythia").is_dir())
sys.path.insert(0, str(ROOT / "runtime/common/pythia"))
from hffn_pythia.dense_control import FixedSupportAdamW  # noqa: E402  the one projection of the dense control


LEAF = next(p for p in Path(__file__).resolve().parents if (p / "runtime" / "common" / "pythia" / "hffn_pythia").is_dir()) / "runtime/gpu/m1/ffn/training"
BUILD = LEAF / "build"   # binaries of the build_*.sh scripts of this leaf
DEFAULT_RUNNER = BUILD / "metal_hffn_train_step_runner"


def tensor_data(i, h, o, n, d, device):
    x = torch.empty((n, i), dtype=torch.float32)
    gy = torch.empty((n, o), dtype=torch.float32)
    up = torch.zeros((h, i), dtype=torch.float32)
    down = torch.zeros((o, h), dtype=torch.float32)
    bup = torch.empty(h, dtype=torch.float32)
    bdown = torch.empty(o, dtype=torch.float32)
    active_up = []
    active_down = []
    for feature in range(i):
        for row in range(n):
            x[row, feature] = ((feature * 17 + row * 13) % 29 - 14) * 0.02
    for output in range(o):
        bdown[output] = (output % 11 - 5) * 0.008
        for row in range(n):
            gy[row, output] = ((output * 11 + row * 7) % 23 - 11) * 0.03 + 0.0007
    for hidden in range(h):
        bup[hidden] = (hidden % 9 - 4) * 0.01
        for edge in range(d):
            slot = hidden * d + edge
            index = (hidden * 19 + edge * 7) % i
            target = (hidden * 17 + edge * 5) % o
            up[hidden, index] = ((slot * 3) % 17 - 8) * 0.015
            down[target, hidden] = ((slot * 5) % 19 - 9) * 0.012
            active_up.append((hidden, index))
            active_down.append((target, hidden))
    values = tuple(
        value.to(device)
        for value in (x, gy, up, down, bup, bdown)
    )
    values += synthetic_support_masks(values[2], values[3], d)
    for value in values[2:6]:
        value.requires_grad_()
    return values, active_up, active_down


def read(prefix, name, count):
    return torch.from_file(str(prefix) + f"_{name}.f32", size=count,
                           dtype=torch.float32).clone()


def active_values(tensor, indices):
    rows = torch.tensor([row for row, _ in indices], device=tensor.device)
    cols = torch.tensor([col for _, col in indices], device=tensor.device)
    return tensor[rows, cols].detach().cpu()


def dense(values, indices, rows, cols):
    result = torch.zeros((rows, cols), dtype=torch.float32)
    rr = torch.tensor([row for row, _ in indices])
    cc = torch.tensor([col for _, col in indices])
    result[rr, cc] = values
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--input", type=int, default=128)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--output-size", type=int, default=128)
    p.add_argument("--n", type=int, default=2048)
    p.add_argument("--degree", type=int, default=13)
    p.add_argument("--state-atol", type=float, default=2e-4)
    p.add_argument("--output-atol", type=float, default=2e-5)
    p.add_argument("--objective-rtol", type=float, default=2e-5)
    p.add_argument("--output-relative-rmse", type=float, default=1e-4)
    a = p.parse_args()
    if a.epochs < 1:
        p.error("--epochs must be positive")
    if not a.runner.exists():
        p.error(f"runner not found: {a.runner}")
    if not torch.backends.mps.is_available():
        p.error("PyTorch MPS is required")
    a.results_dir.mkdir(parents=True, exist_ok=True)
    prefix = a.results_dir / "metal_state"
    subprocess.run([
        str(a.runner), "--input", str(a.input), "--hidden", str(a.hidden),
        "--output-size", str(a.output_size), "--n", str(a.n),
        "--degree", str(a.degree), "--verify-steps", str(a.epochs),
        "--dump-prefix", str(prefix), "--output", str(a.results_dir / "metal.json")
    ], check=True)
    (x, gy, up, down, bup, bdown, up_mask, down_mask), au, ad = tensor_data(
        a.input, a.hidden, a.output_size, a.n, a.degree, "mps")
    params = [up, down, bup, bdown]
    opt = FixedSupportAdamW(params, {up: up_mask, down: down_mask}, lr=1e-3, betas=(.9, .999), eps=1e-8,
                            weight_decay=.01, foreach=False)
    losses = []
    for _ in range(a.epochs):
        opt.zero_grad(set_to_none=True)
        y = F.linear(F.gelu(F.linear(x, up, bup), approximate="none"), down, bdown)
        loss = (y * gy).sum()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    torch.mps.synchronize()
    e = a.hidden * a.degree
    names = {"up": e, "down": e, "bup": a.hidden, "bdown": a.output_size,
             "um": e, "uv": e, "dm": e, "dv": e,
             "bum": a.hidden, "buv": a.hidden,
             "bdm": a.output_size, "bdv": a.output_size}
    metal = {name: read(prefix, name, count) for name, count in names.items()}
    pytorch = {
        "up": active_values(up, au), "down": active_values(down, ad),
        "bup": bup.detach().cpu(), "bdown": bdown.detach().cpu(),
        "um": active_values(opt.base.state[up]["exp_avg"], au),
        "uv": active_values(opt.base.state[up]["exp_avg_sq"], au),
        "dm": active_values(opt.base.state[down]["exp_avg"], ad),
        "dv": active_values(opt.base.state[down]["exp_avg_sq"], ad),
        "bum": opt.base.state[bup]["exp_avg"].cpu(),
        "buv": opt.base.state[bup]["exp_avg_sq"].cpu(),
        "bdm": opt.base.state[bdown]["exp_avg"].cpu(),
        "bdv": opt.base.state[bdown]["exp_avg_sq"].cpu(),
    }
    errors = {name: float((metal[name] - value).abs().max())
              for name, value in pytorch.items()}
    inactive_max = max(
        float((up.detach() * (1 - up_mask)).abs().max().cpu()),
        float((down.detach() * (1 - down_mask)).abs().max().cpu()),
        float((opt.base.state[up]["exp_avg"] * (1 - up_mask)).abs().max().cpu()),
        float((opt.base.state[up]["exp_avg_sq"] * (1 - up_mask)).abs().max().cpu()),
        float((opt.base.state[down]["exp_avg"] * (1 - down_mask)).abs().max().cpu()),
        float((opt.base.state[down]["exp_avg_sq"] * (1 - down_mask)).abs().max().cpu()),
    )
    with torch.no_grad():
        metal_up = dense(metal["up"], au, a.hidden, a.input).to("mps")
        metal_down = dense(
            metal["down"], ad, a.output_size, a.hidden
        ).to("mps")
        metal_h = F.gelu(
            F.linear(x, metal_up, metal["bup"].to("mps")), approximate="none"
        )
        metal_y = F.linear(
            metal_h, metal_down, metal["bdown"].to("mps")
        )
        pytorch_y = F.linear(
            F.gelu(F.linear(x, up, bup), approximate="none"), down, bdown
        )
        objectives = [
            float((value * gy).sum().cpu()) for value in (metal_y, pytorch_y)
        ]
        output_delta = metal_y - pytorch_y
        output_error = float(output_delta.abs().max().cpu())
        output_rmse = float(output_delta.square().mean().sqrt().cpu())
        output_reference_rms = float(pytorch_y.square().mean().sqrt().cpu())
        output_relative_rmse = output_rmse / max(output_reference_rms, 1e-12)
    objective_abs = abs(objectives[0] - objectives[1])
    objective_relative = objective_abs / max(abs(objectives[1]), 1e-12)
    strict_numerical_pass = (
        max(errors.values()) <= a.state_atol
        and output_error <= a.output_atol
        and objective_relative <= a.objective_rtol
        and inactive_max == 0.0
    )
    functional_pass = (
        output_relative_rmse <= a.output_relative_rmse
        and objective_relative <= a.objective_rtol
        and inactive_max == 0.0
    )
    result = {"kind": "hffn_metal_vs_pytorch_mps_iso_support_trajectory",
              "epochs": a.epochs, "runner": str(a.runner), "losses": losses,
              "post_objective": {"metal": objectives[0], "pytorch": objectives[1],
                                 "abs_difference": objective_abs,
                                 "relative_difference": objective_relative,
                                 "output_max_abs": output_error,
                                 "output_rmse": output_rmse,
                                 "output_reference_rms": output_reference_rms,
                                 "output_relative_rmse": output_relative_rmse},
              "max_abs_by_tensor": errors,
              "max_abs": max(errors.values()),
              "max_abs_inactive": inactive_max,
              "tolerances": {"state_atol": a.state_atol,
                             "output_atol": a.output_atol,
                             "objective_rtol": a.objective_rtol,
                             "output_relative_rmse": a.output_relative_rmse},
              "strict_numerical_pass": strict_numerical_pass,
              "functional_pass": functional_pass,
              "support_invariant": "inactive gradients, parameters, and AdamW moments remain zero"}
    (a.results_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
