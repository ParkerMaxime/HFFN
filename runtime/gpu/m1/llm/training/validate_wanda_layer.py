#!/usr/bin/env python3
"""Forward and native-backward parity of the leaf's FFN module against the float64 dense oracle.

The backward is the owner's native staged context (``native_ffn_adapter.NativeBackwardMLP``); the layout only selects the
Torch forward schedule (``auto`` is the leaf's deterministic choice by N)."""

from __future__ import annotations

import argparse
import gc
import json
import traceback
from pathlib import Path

import torch
from hffn_metal_llm import load_compact_wanda_layer
from native_ffn_adapter import NativeBackwardMLP
from hffn_pythia.oracle import dense_forward, dense_weights  # on sys.path once the runtime package is imported

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "runtime" / "common" / "pythia" / "hffn_pythia").is_dir())
LIBRARY = ROOT / "runtime/gpu/m1/ffn/training/build/libhffn_train.dylib"
GATE = 5e-4  # the gate of this leaf; only the labelled test switch --test-gate lowers it (forced-failure witness, never a default)
LAYER = ""  # name of the layer folder being validated, for the record filename
RECORD_DIR: Path | None = None  # set by --record-dir; the failure record is written there before the error is raised
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


def worst_entry(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    """Position and values of the largest absolute difference (diagnostic only; computed after the verdict is known)."""
    a, e = actual.detach().cpu().double(), expected.detach().cpu().double()
    flat = (a - e).abs().flatten()
    k = int(flat.argmax())
    index = [int(i) for i in torch.unravel_index(torch.tensor(k), a.shape)]
    return {"index": index, "actual": float(a.flatten()[k]), "expected": float(e.flatten()[k]), "abs": float(flat[k])}


def write_failure_record(spec, n, seed, layout, x, gy, actual, expected, result, max_abs) -> Path | None:
    """Written only on a failing row, before the AssertionError: layout, N, seed, errors, worst entries, x and gy."""
    if RECORD_DIR is None:
        return None
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    stem = RECORD_DIR / f"failure_{LAYER or 'layer'}_{layout}_n{n}_seed{seed}"
    torch.save({"x": x.detach().cpu(), "gy": gy.detach().cpu()}, f"{stem}.pt")
    record = {
        "layer": LAYER, "layout": layout, "n": n, "seed": seed, "gate": GATE, "max_abs": max_abs,
        "shape": {"d_model": spec.d_model, "d_ff": spec.d_ff}, "errors": result,
        "worst": {name: worst_entry(actual[name], expected[name]) for name in expected},
        "tensors": f"{stem.name}.pt",
    }
    Path(f"{stem}.json").write_text(json.dumps(record, indent=2) + "\n")
    return Path(f"{stem}.json")


def validate(spec, n: int, seed: int, layout: str, module=None) -> dict:
    """One row. ``module``: a NativeBackwardMLP of the layer shared by the rows of a layer (its native context of an N is
    prepared once); without it, a module is built and closed for the row."""
    torch.manual_seed(seed + n)
    device = "mps"
    x = torch.randn(n, spec.d_model, dtype=torch.float32, device=device, requires_grad=True)
    gy = torch.randn(n, spec.d_model, dtype=torch.float32, device=device)
    x64 = x.detach().cpu().double().requires_grad_()
    up, bup, down, bdown = (t.requires_grad_() for t in dense_weights({key: getattr(spec, key) for key in LAYER_KEYS}))
    reference = dense_forward((up, bup, down, bdown), x64)
    reference.backward(gy.cpu().double())

    own = module is None
    if own:
        module = NativeBackwardMLP(0, spec, LIBRARY)
    else:
        module.flat.grad = None  # a row never accumulates the gradient of the previous one
    try:
        module.set_forced_layout(None if layout == "auto" else layout)
        module.prepare(n)
        x_leaf = x.detach().clone().requires_grad_()
        y = module(x_leaf)
        y.backward(gy)
        torch.mps.synchronize()
        u, d, bu = spec.up_val.numel(), spec.down_val.numel(), spec.b_up.numel()
        flat_grad = module.flat.grad
        grad = {
            "grad_x": x_leaf.grad,
            "grad_up": flat_grad[:u],
            "grad_down": flat_grad[u:u + d],
            "grad_b_up": flat_grad[u + d:u + d + bu],
            "grad_b_down": flat_grad[u + d + bu:],
        }
        y = y.detach()
        result = _compare(spec, n, seed, layout, x, gy, y, grad, reference, x64, up, down, bup, bdown)
    finally:
        if own:
            module.close()
    return result


def _compare(spec, n, seed, layout, x, gy, y, grad, reference, x64, up, down, bup, bdown) -> dict:
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
    # The FFN kernel uses the same erf approximation.  This first gate
    # compares against torch GELU exact, hence a small activation approximation
    # tolerance is expected before trajectory validation.
    if max_abs > GATE:
        record = write_failure_record(spec, n, seed, layout, x, gy, actual, expected, result, max_abs)
        raise AssertionError(f"N={n} layout={layout} exact-offset parity failed (record: {record}): {result}")
    return {"layout": layout, "n": n, "max_abs": max_abs, "errors": result}


def validate_layer(path: Path, ns: list[int], layouts: list[str], seed: int) -> dict:
    global LAYER
    LAYER = path.parent.name
    spec = load_compact_wanda_layer(path)
    module = NativeBackwardMLP(0, spec, LIBRARY)
    try:
        rows = [validate(spec, n, seed, layout, module) for layout in layouts for n in ns]
    finally:
        module.close()
    return {
        "format": "hffn_metal_wanda_exact_offset_layer_parity_v1",
        "layer": str(path.resolve()),
        "shape": {"d_model": spec.d_model, "d_ff": spec.d_ff},
        "nnz": {"up": spec.up_val.numel(), "down": spec.down_val.numel()},
        "route_edge_bits": {
            "up": spec.routes.up_by_input.edge_bits,
            "down": spec.routes.down_by_output.edge_bits,
        },
        "rows": rows,
    }


def validate_export(export_dir: Path, output_dir: Path, ns: list[int], layouts: list[str], seed: int) -> int:
    """Every layer of an export in one process: one record and one line `ok|FAIL <export> <layer>` per layer."""
    output_dir.mkdir(parents=True, exist_ok=True)
    failed = 0
    for directory in sorted(export_dir.glob("layer*"), key=lambda p: int(p.name[len("layer"):])):
        try:
            result = validate_layer(directory / "hffn_training_compact_g4_v2.pt", ns, layouts, seed)
            (output_dir / f"{directory.name}.json").write_text(json.dumps(result, indent=2) + "\n")
            print(f"ok   {export_dir.parent.name} {directory.name}", flush=True)
        except Exception:
            traceback.print_exc()
            print(f"FAIL {export_dir.parent.name} {directory.name}", flush=True)
            failed = 1
        gc.collect()
        torch.mps.empty_cache()
    return failed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=Path)
    parser.add_argument("--export-dir", type=Path, help="every layer of this export; needs --output-dir")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--n", type=int, nargs="+", default=[1, 8, 64, 128, 256])
    parser.add_argument(
        "--layout",
        choices=("auto", "token_major", "feature_persistent"),
        nargs="+",
        default=["auto"],
    )
    parser.add_argument("--seed", type=int, default=712)
    parser.add_argument("--record-dir", type=Path, help="write a record of a failing row here (layout, N, errors, worst entries, x, gy)")
    parser.add_argument("--test-gate", type=float, help="TEST SWITCH (witness of the record only): replaces the gate 5e-4")
    args = parser.parse_args()
    global RECORD_DIR, GATE
    RECORD_DIR = args.record_dir
    if args.test_gate is not None:
        GATE = args.test_gate
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available; run outside the GPU sandbox")
    if args.export_dir is not None:
        raise SystemExit(validate_export(args.export_dir, args.output_dir, args.n, args.layout, args.seed))
    print(json.dumps(validate_layer(args.layer, args.n, args.layout, args.seed), indent=2))


if __name__ == "__main__":
    main()
