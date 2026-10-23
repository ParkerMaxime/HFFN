"""Update gate of the training runtime, written against the adapter (`arm_adapter.py`). Criteria and closed form: `update_gate_core.py`.

    python -B checks/parity_update.py --model-dir <base checkpoint> --export-dir <export> --tokens <wikitext2_test_tokens.json> [--gate] [--out <json>]

One AdamW step from zero moments of the dense float32 control and the sole HFFN inference arm, compared with
the float64 closed form on the active entries. Exit code 1 with `--gate` when a criterion fails or a seam of the adapter is missing.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import arm_adapter as adapter  # noqa: E402
import update_gate_core as core  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    ids = torch.tensor(json.loads(args.tokens.read_text())["tokens"][:args.batch * args.seq_len]).view(args.batch, args.seq_len)
    torch.set_num_threads(4)
    lr, wd = args.lr, args.weight_decay
    base = adapter.base

    model, runtime = adapter.load(args.model_dir, args.export_dir)
    supports = adapter.supports(model)
    checksum = adapter.support_checksum(supports)
    initial = [tuple(t.clone() for t in layer) for layer in adapter.active_weights(model)]
    del model, runtime

    dense = base.dense_model(args.model_dir, args.export_dir).double()
    loss64, gradients, _other = base.dense_gradients(dense, ids, supports)
    del dense, _other
    gc.collect()
    defaults = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=lr, weight_decay=wd).defaults
    beta1, beta2 = defaults["betas"]
    eps = defaults["eps"]
    hyper = (wd, beta1, beta2, eps)
    expected = [tuple(zip(w0s, gs)) for w0s, gs in zip(initial, gradients)]

    rows = {}
    dense = base.dense_model(args.model_dir, args.export_dir)
    optimizer = adapter.build_fixed_support_pythia_adamw(dense, args.export_dir, lr=lr, weight_decay=wd)
    optimizer.zero_grad()
    loss = dense(input_ids=ids, labels=ids).loss
    loss.backward()
    optimizer.step()
    weights, first, second = [], [], []
    with torch.no_grad():
        for block, (up_flat, down_flat) in zip(dense.gpt_neox.layers, supports):
            up, down = block.mlp.dense_h_to_4h, block.mlp.dense_4h_to_h
            pick = lambda tensor, flat: tensor.reshape(-1).index_select(0, flat) if flat is not None else tensor.clone()
            parameters = ((up.weight, up_flat), (down.weight, down_flat), (up.bias, None), (down.bias, None))
            weights.append(tuple(pick(p, flat) for p, flat in parameters))
            first.append(tuple(pick(optimizer.base.state[p]["exp_avg"], flat) for p, flat in parameters))
            second.append(tuple(pick(optimizer.base.state[p]["exp_avg_sq"], flat) for p, flat in parameters))
        rows["dense32"] = {**core.measure(weights, first, second, expected, lr, hyper), "loss_before": float(loss),
                           "loss_after": float(dense(input_ids=ids, labels=ids).loss)}
    del dense, optimizer, weights, first, second, loss
    gc.collect()

    for name in adapter.CONFIGURATIONS:
        model, runtime = adapter.load(args.model_dir, args.export_dir, name)
        if adapter.support_checksum(adapter.supports(model)) != checksum:
            raise RuntimeError(f"{name}: the flat supports differ from those of the gradients (order contract broken)")
        opt = adapter.optimizer(model, runtime, lr, wd)
        opt.zero_grad()
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        opt.step()
        with torch.no_grad():
            first, second = adapter.active_moments(runtime)
            rows[f"hffn_{name}"] = {**core.measure(adapter.active_weights(model), first, second, expected, lr, hyper), "loss_before": float(loss),
                                    "loss_after": float(model(input_ids=ids, labels=ids).loss)}
        del model, runtime, opt, first, second, loss
        gc.collect()

    control = rows["dense32"]
    gate = core.judge(control, {n: rows[f"hffn_{n}"] for n in adapter.CONFIGURATIONS})
    result = {"export": args.export_dir.parent.name, "batch": args.batch, "seq_len": args.seq_len, "lr": lr, "weight_decay": wd, "betas": [beta1, beta2],
              "eps": eps, "loss_float64": loss64, "support_checksum": checksum, "gate": gate, "rows": rows}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"{result['export']}: loss float64 {loss64:.6f}")
    for name, row in rows.items():
        print(f"  {name:16s} |dw|/lr conditioned {row['weight_conditioned']:.1e} all {row['weight_all']:.1e}  moments m {row['moment_m']:.1e} v {row['moment_v']:.1e}  "
              f"flipped {row['flipped_share']:.1e}  loss {row['loss_before']:.4f} -> {row['loss_after']:.4f}", flush=True)
    print(f"  update gate -> {'PASSED' if gate['passed'] else 'FAILED'}", flush=True)
    return 0 if gate["passed"] or not args.gate else 1


if __name__ == "__main__":
    sys.exit(main())
