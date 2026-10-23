"""Model gate of the training runtime: distance to a float64 reference at the first step (no update).

The same model and the same batch of real text (the first tokens of the shared WikiText-2 prompt file) are
evaluated by the dense fixed-support control in float64 (the reference), by the same control in float32, and by the
HFFN arm whose forward unconditionally calls the sibling inference kernel. For the loss and for the gradients
(active FFN weights and biases; every non-FFN parameter): the distance of each float32 evaluation to float64, as
max|g - g64| / max|g64| per tensor, then the maximum over the tensors of a group. The dense models are controls of the
check; the HFFN arm never builds a dense matrix.

Criterion, with `--gate`: the HFFN distance to float64 of each
gradient group is at most GRADIENT_FACTOR times that of the dense float32 control, and the relative distance of the
loss to float64 is at most LOSS; exit code 1 otherwise. Two float32 arms are not
compared step by step: the dense control itself is 2e-4 to 3e-2 from float64 on these pruned models.

    python -B checks/parity_model.py --model-dir <base checkpoint> --export-dir <export> --tokens <wikitext2_test_tokens.json> [--gate] [--out <json>]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.insert(0, os.environ["HFFN_LEAF"])  # the leaf under check, given by its check.sh
# The metric and the verdict of the gate are those of the shared Pythia level (first ancestor that has common/pythia/hffn_pythia).
sys.path.insert(0, str(next(p / "common/pythia" for p in Path(__file__).resolve().parents if (p / "common/pythia/hffn_pythia").is_dir())))

from hffn_pythia.training_gate import GRADIENT_FACTOR, LOSS, dense_gradients, distance, gap, judge, ratio  # noqa: E402,F401
from hffn_training_runtime import install_compact_hffn_values_in_dense_pythia, load_pythia_hffn_compact_g4_for_training  # noqa: E402

CONFIGURATIONS = {"inference": {}}


def dense_model(model_dir: Path, export_dir: Path):
    """The dense control: the base checkpoint outside the FFNs, the export values on the fixed support inside."""
    model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(model_dir), dtype=torch.float32)
    targets = model.state_dict()
    with torch.no_grad(), safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as source:
        for name in source.keys():
            if ".mlp." not in name and name in targets:
                targets[name].copy_(source.get_tensor(name))
    install_compact_hffn_values_in_dense_pythia(model, export_dir)
    return model.train()


def support(mlp):
    """Row (hidden) and column (feature) of every compact slot of an HFFN MLP, for both sides."""
    owner = lambda offsets: torch.repeat_interleave(torch.arange(offsets.numel() - 1), offsets.diff().long())
    return owner(mlp.up_row_offsets), mlp.up_idx.long(), mlp.down_idx.long(), owner(mlp.down_row_offsets)


def flat_support(mlp):
    """One flat int32 index per side into the dense weights of the control (up: [d_ff, d_model]; down: [d_model, d_ff])."""
    up_row, up_col, down_row, down_col = support(mlp)
    d_model, d_ff = mlp.b_down.numel(), mlp.b_up.numel()
    return (up_row * d_model + up_col).to(torch.int32), (down_row * d_ff + down_col).to(torch.int32)


def hffn_gradients(model, ids):
    loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    ffn = [tuple(t.grad.clone() for t in (b.mlp.up_val, b.mlp.down_val, b.mlp.b_up, b.mlp.b_down)) for b in model.gpt_neox.layers]
    other = {name: p.grad.clone() for name, p in model.named_parameters() if ".mlp." not in name and p.grad is not None}
    return float(loss), ffn, other


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--gate", action="store_true", help="judge the gate and return its exit code")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    ids = torch.tensor(json.loads(args.tokens.read_text())["tokens"][:args.batch * args.seq_len]).view(args.batch, args.seq_len)
    torch.set_num_threads(4)
    # The supports, then the float64 reference first; each float32 evaluation is then reduced to its distances and
    # freed, so that the process never holds more than the reference, one model and one set of gradients.
    model, _runtime = load_pythia_hffn_compact_g4_for_training(args.model_dir, args.export_dir)
    supports = [flat_support(block.mlp) for block in model.gpt_neox.layers]
    del model, _runtime
    dense = dense_model(args.model_dir, args.export_dir).double()
    reference = dense_gradients(dense, ids, supports)
    del dense
    gc.collect()
    dense = dense_model(args.model_dir, args.export_dir)
    dense32 = {}
    for threads in (4, 1):
        torch.set_num_threads(threads)
        dense32[threads] = dense_gradients(dense, ids, supports)
    torch.set_num_threads(4)
    dense.zero_grad(set_to_none=True)
    del dense
    gc.collect()
    rows = {"dense32_threads4": distance(dense32[4], reference), "dense32_threads1": distance(dense32[1], reference)}
    between = {"dense32_threads1_vs_threads4": distance(dense32[1], dense32[4])}
    del dense32[1]
    if args.gate:                # the gate needs the distances to float64 only: the float32 control is freed
        dense32.clear()
        gc.collect()
    for name, options in CONFIGURATIONS.items():
        model, _runtime = load_pythia_hffn_compact_g4_for_training(args.model_dir, args.export_dir, **options)
        evaluation = hffn_gradients(model, ids)
        del model, _runtime
        rows[f"hffn_{name}"] = distance(evaluation, reference)
        if dense32:
            between[f"hffn_{name}_vs_dense32_threads4"] = distance(evaluation, dense32[4])
        del evaluation
        gc.collect()
    control = rows["dense32_threads4"]
    gate, passed = judge(rows, control, CONFIGURATIONS)
    result = {"gate": {"gradient_factor": GRADIENT_FACTOR, "loss": LOSS, "control": "dense32_threads4", "checks": gate, "passed": passed}, "export": args.export_dir.parent.name, "batch": args.batch, "seq_len": args.seq_len, "batch_source": "wikitext2_test_tokens",
              "loss_float64": reference[0], "to_float64": rows, "between_float32": between}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"{result['export']}: loss float64 {reference[0]:.6f}")
    for name, row in {**rows, **between}.items():
        print(f"  {name:36s} loss {row['loss']:.1e}  gradient ffn {row['gradient_ffn']:.1e}  non-ffn {row['gradient_non_ffn']:.1e}", flush=True)
    print(f"  model gate: worst hffn/dense32 gradient ffn {max(rows[f'hffn_{n}']['gradient_ffn'] for n in CONFIGURATIONS) / control['gradient_ffn']:.2f}, "
          f"non-ffn {max(rows[f'hffn_{n}']['gradient_non_ffn'] for n in CONFIGURATIONS) / control['gradient_non_ffn']:.2f} (limit {GRADIENT_FACTOR:.0f}); "
          f"worst hffn loss {max(rows[f'hffn_{n}']['loss'] for n in CONFIGURATIONS):.1e} (limit {LOSS:.0e}) -> {'PASSED' if passed else 'FAILED'}", flush=True)
    return 0 if passed or not args.gate else 1


if __name__ == "__main__":
    sys.exit(main())
