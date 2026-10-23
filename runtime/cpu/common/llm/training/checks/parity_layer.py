"""Layer gate of the sole compact training forward against the direct fixed-support float64 equation.

For every exported and shared synthetic edge layer, deterministic inputs and output gradients are passed through the
sibling inference kernel and retained training backward. Output, z, h, input gradient, active up/down gradients and
both bias gradients are compared with the float64 oracle. Dense matrices exist only in the oracle, never in HFFN.
``--suppress-z`` is a negative control: it aliases the forward's z and h destinations and must fail at N = 1.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.environ["HFFN_LEAF"])
from hffn_training_runtime import (HFFNCompactG4Layer, HFFNCompactG4TrainingRuntime,
                                   load_hffn_compact_g4_layer)  # noqa: E402

TOLERANCE = 1e-4
NS = (1, 2, 7, 8, 31, 32, 36, 64, 128, 130, 512)
QUANTITIES = ("forward", "z", "h", "grad_x", "grad_up", "grad_down", "grad_b_up", "grad_b_down")


def oracle(layer, x: torch.Tensor, grad_y: torch.Tensor) -> dict[str, torch.Tensor]:
    """Direct fixed-support equation in float64; dense matrices are oracle-only."""
    owner = lambda offsets: torch.repeat_interleave(torch.arange(offsets.numel() - 1), offsets.diff().long())
    up_hidden, up_feature = owner(layer.up_row_offsets), layer.up_idx.long()
    down_hidden, down_feature = owner(layer.down_row_offsets), layer.down_idx.long()
    leaf = lambda tensor: tensor.double().clone().requires_grad_(True)
    xd, up, down, b_up, b_down = leaf(x), leaf(layer.up_val), leaf(layer.down_val), leaf(layer.b_up), leaf(layer.b_down)
    w_up = torch.zeros(layer.d_ff, layer.d_model, dtype=torch.float64).index_put((up_hidden, up_feature), up)
    w_down = torch.zeros(layer.d_model, layer.d_ff, dtype=torch.float64).index_put((down_feature, down_hidden), down)
    z = xd @ w_up.T + b_up
    h = 0.5 * z * (1.0 + torch.erf(z * 0.7071067811865476))
    y = h @ w_down.T + b_down
    (y * grad_y.double()).sum().backward()
    return {"forward": y.detach(), "z": z.detach(), "h": h.detach(), "grad_x": xd.grad, "grad_up": up.grad, "grad_down": down.grad,
            "grad_b_up": b_up.grad, "grad_b_down": b_down.grad}


def adapt_synthetic_layer(layer) -> HFFNCompactG4Layer:
    """Map the shared FFN check layer into the compact LLM runtime's typed layer contract."""
    adapted = HFFNCompactG4Layer(
        d_model=int(layer.d_model),
        d_ff=int(layer.d_ff),
        group_hidden_offsets=layer.groups.contiguous(),
        up_row_offsets=layer.up_offsets.contiguous(),
        up_idx=layer.up_idx.contiguous(),
        up_val=layer.up_val.contiguous(),
        b_up=layer.b_up.contiguous(),
        down_row_offsets=layer.down_offsets.contiguous(),
        down_idx=layer.down_idx.contiguous(),
        down_val=layer.down_val.contiguous(),
        b_down=layer.b_down.contiguous(),
    )
    adapted.validate()
    return adapted


def targets(export_dir: Path):
    """Export layers plus the shared synthetic edge cases used by the CPU FFN training gates."""
    paths = sorted(export_dir.glob("layer*/hffn_training_compact_g4_v2.pt"), key=lambda p: int(p.parent.name[5:]))
    if not paths:
        raise SystemExit(f"no compact layer under {export_dir}")
    rows = [(f"export/{path.parent.name}", load_hffn_compact_g4_layer(path), None) for path in paths]
    root = next(p for p in Path(__file__).resolve().parents if (p / "runtime" / "common" / "pythia" / "hffn_pythia").is_dir())
    sys.path.insert(0, str(root / "runtime/common/ffn"))
    sys.path.insert(0, str(root / "runtime/cpu/common/ffn/training"))
    from hffn_ffn_training.layer import from_compact
    from hffn_ffn_layers import layers as synthetic
    rows.extend((f"synthetic/{name}", adapt_synthetic_layer(from_compact(name, synthetic.load(name)[0])),
                 synthetic.load(name)[1]) for name in synthetic.names())
    return rows


def json_safe(value):
    """Represent non-finite diagnostics as JSON null while the finite gate records the failure."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--ns", type=int, nargs="+", default=list(NS))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--suppress-z", action="store_true", help="negative control: alias N = 1 z and h destinations")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if any(n < 1 for n in args.ns):
        parser.error("every N must be positive")
    torch.set_num_threads(args.torch_threads)
    layer_rows = targets(args.export_dir)
    from hffn_ffn_layers import layers as synthetic  # noqa: E402
    layers = [entry[1] for entry in layer_rows]
    runtime = HFFNCompactG4TrainingRuntime(layers)
    if args.suppress_z:
        new_state = runtime._new_state
        def poisoned_state(layer, n):
            state = new_state(layer, n)
            if n == 1:
                state.z_t.fill_(float("nan"))
            return state
        runtime._new_state = poisoned_state
        inference_forward = runtime._inference.forward_dual_slots_nt
        def suppress_z(*call):
            call = list(call)
            if call[13] == 1:
                call[19] = call[20]
            return inference_forward(*call)
        runtime._inference.forward_dual_slots_nt = suppress_z
    rows, worst = [], {name: 0.0 for name in QUANTITIES}
    finite = True
    for n in args.ns:
        torch.manual_seed(args.seed)
        n_rows = []
        for index, (label, layer, case) in enumerate(layer_rows):
            if case is None:
                x = torch.randn(n, layer.d_model)
            else:
                x = torch.from_numpy(synthetic.make_input(case, n).copy())
            grad_y = torch.randn(n, layer.d_model)
            y, context = runtime.forward_with_context(index, x)
            state = runtime._contexts[context]
            native = {"forward": y.clone(), "z": state.z_t.t().clone(), "h": state.h_t.t().clone()}
            native.update({name: tensor.clone() for name, tensor in runtime.backward_with_context(context, index, grad_y).items()})
            runtime.discard_context(context)
            reference = oracle(layer, x, grad_y)
            row = {"n": n, "layer": label}
            for name in QUANTITIES:
                gap = float((native[name].double() - reference[name]).abs().max())
                row[name] = gap / float(reference[name].abs().max())
                row[f"{name}_abs"] = gap
                finite = finite and bool(torch.isfinite(native[name]).all())
                worst[name] = max(worst[name], row[name])
            rows.append(row)
            n_rows.append(row)
        print(f"{args.export_dir.parent.name} [sibling inference, N = {n}]: {len(layers)} layers, worst relative gap "
              + ", ".join(f"{name} {max(row[name] for row in n_rows):.1e}" for name in QUANTITIES), flush=True)
    failed = [name for name in QUANTITIES if not worst[name] <= TOLERANCE] + ([] if finite else ["finite"])
    result = {"export": args.export_dir.parent.name, "ns": args.ns, "seed": args.seed, "tolerance": TOLERANCE,
              "forward": "sibling_inference", "suppressed_z_negative_control": args.suppress_z, "worst": worst,
              "worst_abs": {name: max(row[f"{name}_abs"] for row in rows) for name in QUANTITIES},
              "rows": rows, "passed": not failed, "failed": failed}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(json_safe(result), indent=1, allow_nan=False) + "\n")
    print(f"{result['export']}: layer gate worst {max(worst.values()):.1e}/{TOLERANCE:.0e} -> "
          f"{'PASSED' if not failed else 'FAILED ' + str(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
