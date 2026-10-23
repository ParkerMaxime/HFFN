"""Parity gates of a CPU FFN training leaf (the leaf is given by ``HFFN_LEAF``; ``check.sh`` of a leaf sets it). Each gate
writes one JSON and returns 0 when it passes, 1 otherwise.

- gate 2: one forward and backward step against the direct equation in float64, every layer of the selected exports and
  the shared synthetic test layers;
- gate 3: one AdamW update against its closed form in float64, with the float64 gradients of gate 2.

    python -B checks/gates.py <2|3> --assets <HFFN_ASSETS> --out <json> [--models ...] [--densities ...]

Without ``--models`` and ``--densities`` the targeted exports (``pythia70m`` 10 %, ``pythia410m`` 80 %) and the synthetic
layers are used; with them, their product (``check.sh`` covers all 18 exports and no synthetic layer).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch

LEAF = Path(os.environ["HFFN_LEAF"]).resolve()
sys.path.insert(0, str(LEAF))

from hffn_ffn_training import build  # noqa: E402
from hffn_ffn_training.layer import Layer, export_layers, from_compact, load  # noqa: E402
from hffn_ffn_training.step import BETA1, BETA2, EPS, INFERENCE_MIN_N, LR, WD, InferenceKernel, Step, TrainingKernel  # noqa: E402

# The shared synthetic test layers (`hffn_ffn_layers`) of the runtime level.
sys.path.insert(0, str(build.ROOT / "runtime/common/ffn"))

MODELS = ("pythia70m", "pythia160m", "pythia410m")
DENSITIES = (10, 20, 30, 50, 70, 80)
TARGETED = (("pythia70m", 10), ("pythia410m", 80))
GATE_N = (8, 32, 128, 13, 61, 97, 203, 512)  # N not multiple of 8, on both sides of 96, and the largest
if INFERENCE_MIN_N == 1:
    GATE_N += (1, 2, 64, 192)  # a leaf whose inference entry stores z at one token: that N, and fixed-N templates of its kernel
UPDATE_N = 128
TOLERANCE = 1e-4
UPDATE_TOLERANCE = 1e-2
CONDITIONED = 1e-2
QUANTITIES = ("y", "z", "h", "grad_x", "grad_up", "grad_down", "grad_b_up", "grad_b_down")
SEED = 17


def export_dir(assets: Path, model: str, density: int) -> Path:
    return assets / f"pythia/exports/{model}_sparse{density}_ffn_only/export"


def inputs(layer: Layer, n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(n, layer.d_model, generator=generator), torch.randn(n, layer.d_model, generator=generator))


def oracle(layer: Layer, x: torch.Tensor, grad_y: torch.Tensor) -> dict[str, torch.Tensor]:
    """The direct equation in float64 and its gradients. The dense float64 matrices belong to the check only."""
    owner = lambda offsets: torch.repeat_interleave(torch.arange(offsets.numel() - 1), offsets.diff().long())  # noqa: E731
    leaf = lambda tensor: tensor.double().clone().requires_grad_(True)  # noqa: E731
    xd, up, down, b_up, b_down = leaf(x), leaf(layer.up_val), leaf(layer.down_val), leaf(layer.b_up), leaf(layer.b_down)
    w_up = torch.zeros(layer.d_ff, layer.d_model, dtype=torch.float64).index_put((owner(layer.up_offsets), layer.up_idx.long()), up)
    w_down = torch.zeros(layer.d_model, layer.d_ff, dtype=torch.float64).index_put((layer.down_idx.long(), owner(layer.down_offsets)), down)
    z = xd @ w_up.T + b_up
    h = 0.5 * z * (1.0 + torch.erf(z * 0.7071067811865476))
    y = h @ w_down.T + b_down
    (y * grad_y.double()).sum().backward()
    return {"y": y.detach(), "z": z.detach(), "h": h.detach(), "grad_x": xd.grad, "grad_up": up.grad, "grad_down": down.grad,
            "grad_b_up": b_up.grad, "grad_b_down": b_down.grad}


def relative(native: torch.Tensor, reference: torch.Tensor) -> float:
    scale = float(reference.abs().max())
    return float((native.double() - reference.double()).abs().max()) / (scale if scale > 0 else 1.0)


def kernels() -> tuple[TrainingKernel, InferenceKernel]:
    return TrainingKernel(build.training_lib()), InferenceKernel(build.inference_lib())


def pairs(args) -> list[tuple[str, int]]:
    if args.models is None and args.densities is None:
        return list(TARGETED)
    return [(m, d) for m in args.models or MODELS for d in args.densities or DENSITIES]


def targets(args) -> list[tuple[str, object]]:
    """(name, layers) of every gate target: all layers of each selected export, then the shared synthetic test layers
    (only for the default selection)."""
    out = [(f"{m}/d{d}", (load(p) for p in export_layers(export_dir(args.assets, m, d)))) for m, d in pairs(args)]
    if args.models is None and args.densities is None:
        from hffn_ffn_layers import layers as shared
        for name in shared.names():
            out.append((f"synth/{name}", iter([from_compact(name, shared.load(name)[0])])))
    return out


def gate2(args) -> tuple[bool, dict]:
    training, inference = kernels()
    rows, passed, worst = [], True, {}
    api_enabled = bool(getattr(args, "api_check", False))
    api_rows = []
    grid = GATE_N
    if api_enabled:
        from capabilities import APIState
        if INFERENCE_MIN_N not in grid:
            grid += (INFERENCE_MIN_N,)  # x86 N2: optional API check only; every other N is kept.
    for name, group in targets(args):
        for layer in group:
            api = APIState(layer, training, inference, Step, relative, INFERENCE_MIN_N) if api_enabled else None
            try:
                for n in grid:
                    x, grad_y = inputs(layer, n, SEED)
                    reference = oracle(layer, x, grad_y)
                    step = Step(layer, n, training, inference)
                    step.forward(x)
                    step.backward(grad_y)
                    native = step.results()
                    row = {"export": name, "layer": layer.name, "n": n}
                    row.update({q: relative(native[q], reference[q]) for q in QUANTITIES})
                    row["finite"] = all(bool(torch.isfinite(native[q]).all()) for q in QUANTITIES)
                    row["passed"] = row["finite"] and all(row[q] <= TOLERANCE for q in QUANTITIES)
                    for q in QUANTITIES:
                        worst[q] = max(worst.get(q, 0.0), row[q])
                    passed &= row["passed"]
                    rows.append(row)
                    if api is not None:
                        api.observe(layer, x, grad_y, reference, n)
                        # Selected references are held once by APIState; do not
                        # keep a second Step alive during API finalization.
                        del step, native, x, grad_y, reference
                if api is not None:
                    observed = [dict(export=name, layer=layer.name, **row) for row in api.finalize()]
                    api_rows.extend(observed)
                    passed &= all(row["passed"] for row in observed)
            finally:
                if api is not None:
                    api.close()
        print(f"gate2 {name}: {'pass' if all(r['passed'] for r in rows if r['export'] == name) else 'FAIL'}", flush=True)
        if api_enabled:
            print(f"gate2 API {name}: {'pass' if all(r['passed'] for r in api_rows if r['export'] == name) else 'FAIL'}", flush=True)
    if api_enabled:
        api_passed = bool(api_rows) and all(row["passed"] for row in api_rows)
        return passed and api_passed, {"rows": rows, "worst": worst, "api_rows": api_rows,
                                      "api_passed": api_passed, "api_check": True}
    return passed, {"rows": rows, "worst": worst}


def closed_form(value: torch.Tensor, grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """First AdamW step from zero moments, in float64."""
    v, g = value.double(), grad.double()
    m1, v1 = (1 - BETA1) * g, (1 - BETA2) * g * g
    m_hat, v_hat = m1 / (1 - BETA1), v1 / (1 - BETA2)
    return v * (1 - LR * WD) - LR * m_hat / (v_hat.sqrt() + EPS), m1, v1


def gate3(args) -> tuple[bool, dict]:
    """One AdamW step at N = 128 against the closed form on the entries with |g| >= CONDITIONED max|g| of their tensor
    (a local update form). Moment distances are reported without a threshold."""
    training, inference = kernels()
    keys = ("up_val", "down_val", "b_up", "b_down")
    oracle_keys = dict(zip(keys, ("grad_up", "grad_down", "grad_b_up", "grad_b_down")))
    rows, passed, n = [], True, UPDATE_N
    for name, group in targets(args):
        for layer in group:
            x, grad_y = inputs(layer, n, SEED)
            reference = oracle(layer, x, grad_y)
            step = Step(layer, n, training, inference)
            step.forward(x)
            step.backward(grad_y)
            before = {k: getattr(step.layer, k).clone() for k in keys}
            step.update()
            moments = dict(zip(keys, zip(step.moments[0::2], step.moments[1::2])))
            row = {"export": name, "layer": layer.name, "n": n, "finite": True, "weight_lr": 0.0, "moment_rel": 0.0}
            for key, value in before.items():
                after, m, v = getattr(step.layer, key), *moments[key]
                expected, m1, v1 = closed_form(value, reference[oracle_keys[key]])
                g = reference[oracle_keys[key]].abs()
                row["finite"] &= all(bool(torch.isfinite(t).all()) for t in (after, m, v))
                if float(g.max()) > 0:
                    conditioned = g >= CONDITIONED * float(g.max())
                    gap = float((after[conditioned].double() - expected[conditioned]).abs().max()) / LR
                    row["weight_lr"] = max(row["weight_lr"], gap)
                row["moment_rel"] = max(row["moment_rel"], relative(m, m1), relative(v, v1))
            row["passed"] = row["finite"] and row["weight_lr"] <= UPDATE_TOLERANCE
            passed &= row["passed"]
            rows.append(row)
        print(f"gate3 {name}: {'pass' if all(r['passed'] for r in rows if r['export'] == name) else 'FAIL'}", flush=True)
    return passed, {"rows": rows}


GATES = {"2": gate2, "3": gate3}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gate", choices=sorted(GATES))
    parser.add_argument("--assets", type=Path, default=os.environ.get("HFFN_ASSETS"), required="HFFN_ASSETS" not in os.environ)
    parser.add_argument("--models", nargs="+", default=None, help="with --densities: their product replaces the targeted exports")
    parser.add_argument("--densities", nargs="+", type=int, default=None)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--api-check", action="store_true", help="Check shared training APIs in gate2, reusing its selected FP64 references.")
    args = parser.parse_args()
    if args.api_check and args.gate != "2":
        parser.error("--api-check is supported by gate2 only")
    torch.set_num_threads(args.torch_threads)
    passed, payload = GATES[args.gate](args)
    payload = {"gate": args.gate, "passed": passed, "tolerance": TOLERANCE, "update_tolerance": UPDATE_TOLERANCE,
               "models": args.models, "densities": args.densities, **payload}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, default=lambda o: None if isinstance(o, float) and math.isnan(o) else o))
    print(f"gate{args.gate}: {'passed' if passed else 'FAILED'}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
