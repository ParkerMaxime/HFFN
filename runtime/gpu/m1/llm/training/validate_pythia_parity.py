#!/usr/bin/env python3
"""One-step full-model gate of the M1 GPU training runtime on the shared float64 oracle.

The dense float32 control (MPS) and the HFFN Metal model run the same training step; each is judged
against a float64 reference of the same model and token ids (built on the CPU), by the shared gates of `hffn_pythia.training_gate`
(loss, gradient groups) and `update_gate_core` (the first AdamW step against its closed form). Two float32 arms are not compared
step by step; the five direct quantities are reported without a verdict.

Exit code 0: every gate holds and both fault injections are detected. 1: a gate failed or a fault was not detected. 3: only the
update gate is inconclusive because the dense control itself fails it. The JSON is written (stdout, and `--out`) in every case.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "runtime" / "common" / "pythia" / "hffn_pythia").is_dir())
sys.path.insert(0, str(ROOT / "runtime/gpu/m1/llm/training"))
sys.path.insert(0, str(ROOT / "runtime/common/pythia"))
sys.path.insert(0, str(ROOT / "runtime/cpu/common/llm/training/checks"))

from hffn_pythia.dense_control import (  # noqa: E402
    build_fixed_support_pythia_adamw,
    install_compact_hffn_values_in_dense_pythia,
)
from hffn_metal_llm import HFFNMetalAdamW, load_pythia_hffn_metal_for_training  # noqa: E402
from hffn_metal_llm.compact import load_compact_wanda_layer  # noqa: E402
from hffn_pythia import training_gate as tg  # noqa: E402
import update_gate_core as core  # noqa: E402

KINDS = core.KINDS


def flat_supports(spec) -> tuple[torch.Tensor, torch.Tensor]:
    """Flat index of every compact slot in the dense up ([d_ff, d_model]) and down ([d_model, d_ff]) weights, in the compact order."""
    owner = lambda offsets: torch.repeat_interleave(torch.arange(offsets.numel() - 1), offsets.diff().long())
    up = owner(spec.up_row_offsets) * spec.d_model + spec.up_idx.long()
    down = spec.down_idx.long() * spec.d_ff + owner(spec.down_row_offsets)
    return up, down


def split(flat: torch.Tensor, spec) -> tuple[torch.Tensor, ...]:
    sizes = (spec.up_val.numel(), spec.down_val.numel(), spec.d_ff, spec.d_model)
    return tuple(t.clone() for t in torch.split(flat, sizes))


def dense_ffn(model, supports, source):
    """Per layer (up, down, b_up, b_down) active entries of `source(parameter)` of the dense control, cloned to the CPU."""
    out = []
    for block, (up_flat, down_flat) in zip(model.gpt_neox.layers, supports):
        up, down = block.mlp.dense_h_to_4h, block.mlp.dense_4h_to_h
        out.append((source(up.weight).reshape(-1).index_select(0, up_flat.to(up.weight.device)).cpu(),
                    source(down.weight).reshape(-1).index_select(0, down_flat.to(down.weight.device)).cpu(),
                    source(up.bias).cpu(), source(down.bias).cpu()))
    return out


def grad_names(model) -> set[str]:
    return {n for n, p in model.named_parameters() if ".mlp." not in n and p.grad is not None}


def other_gradients(model) -> dict[str, torch.Tensor]:
    return {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters() if ".mlp." not in n and p.grad is not None}


def finite(tensors) -> bool:
    return all(bool(torch.isfinite(t).all()) for t in tensors)


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().cpu().double() - b.detach().cpu().double()).abs().max())


def run_dense_arm(args, specs, supports, tokens) -> dict:
    """The dense float32 control (MPS): its step, then only what the gates read, as CPU clones. Its model and optimiser die with this frame."""
    dense = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.float32).train()
    install_compact_hffn_values_in_dense_pythia(dense, args.export_dir)
    dense_optimizer = build_fixed_support_pythia_adamw(dense, args.export_dir, lr=args.lr, weight_decay=args.weight_decay)
    masks = {}
    for index, block in enumerate(dense.gpt_neox.layers):
        masks[index] = tuple(dense_optimizer._masks[p].detach().cpu().reshape(-1) != 0
                             for p in (block.mlp.dense_h_to_4h.weight, block.mlp.dense_4h_to_h.weight))
    dense.to("mps")
    dense_optimizer._masks = {p: m.to(device=p.device) for p, m in dense_optimizer._masks.items()}
    group = dense_optimizer.base.param_groups[0]
    kept = {"masks": masks, "lr": float(group["lr"]),
            "hyper": (float(group["weight_decay"]), *map(float, group["betas"]), float(group["eps"]))}

    dense_optimizer.zero_grad()
    kept["paired_rng"] = torch.mps.get_rng_state()
    output = dense(tokens, labels=tokens)
    output.loss.backward()
    kept["after_dense_rng"] = torch.mps.get_rng_state()
    # Everything is read and cloned to the CPU before the step.
    kept["names"] = grad_names(dense)
    kept["evaluation"] = (float(output.loss.detach().cpu()), dense_ffn(dense, supports, lambda p: p.grad), other_gradients(dense))
    kept["logits"] = output.logits.detach().cpu().clone()
    kept["initial"] = dense_ffn(dense, supports, lambda p: p.detach())
    del output

    dense_optimizer.step()
    torch.mps.synchronize()
    kept["weights"] = dense_ffn(dense, supports, lambda p: p.detach())
    state = lambda p, key: dense_optimizer.base.state[p][key]
    first, second = [], []
    for block, (up_flat, down_flat) in zip(dense.gpt_neox.layers, supports):
        up, down = block.mlp.dense_h_to_4h, block.mlp.dense_4h_to_h
        pick = lambda p, flat, key: (state(p, key).reshape(-1).index_select(0, flat.to(p.device)) if flat is not None else state(p, key)).cpu()
        parameters = ((up.weight, up_flat), (down.weight, down_flat), (up.bias, None), (down.bias, None))
        first.append(tuple(pick(p, f, "exp_avg") for p, f in parameters))
        second.append(tuple(pick(p, f, "exp_avg_sq") for p, f in parameters))
    kept["first"], kept["second"] = first, second
    with torch.no_grad():
        kept["next_loss"] = float(dense(tokens, labels=tokens).loss.cpu())
    return kept


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--seed", type=int, default=712)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available; run outside the GPU sandbox")
    torch.manual_seed(args.seed)
    config = AutoConfig.from_pretrained(args.model_dir)
    guards = {"dropout_zero": float(config.hidden_dropout) == 0.0 and float(config.attention_dropout) == 0.0}
    tokens = torch.randint(0, config.vocab_size, (1, args.seq_len), device="mps")
    ids = tokens.cpu()
    guards["token_ids_sha256"] = hashlib.sha256(ids.numpy().tobytes()).hexdigest()

    # Supports of every layer from the compact files (CPU), then the float64 reference first and freed before the MPS models.
    layer_files = sorted(args.export_dir.glob("layer*/hffn_training_compact_g4_v2.pt"), key=lambda p: int(p.parent.name[5:]))
    specs = [load_compact_wanda_layer(path) for path in layer_files]
    supports = [flat_supports(spec) for spec in specs]
    reference = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.float32)
    install_compact_hffn_values_in_dense_pythia(reference, args.export_dir)
    reference = reference.double().train()
    loss64, ffn64, other64 = tg.dense_gradients(reference, ids, supports)
    ref_names = {n for n, g in other64.items()}
    del reference
    gc.collect()
    reference_eval = (loss64, ffn64, other64)

    # Dense control arm first: it runs to the end of its step, everything the gates read is kept as
    # CPU clones, then it is freed (del, gc, MPS cache) before the HFFN arm is built.
    kept = run_dense_arm(args, specs, supports, tokens)
    dense_names, dense_eval, initial_dense = kept["names"], kept["evaluation"], kept["initial"]
    dense_weights, dense_first, dense_second, dense_next = kept["weights"], kept["first"], kept["second"], kept["next_loss"]
    masks = kept["masks"]
    gc.collect()
    torch.mps.empty_cache()

    hffn, modules = load_pythia_hffn_metal_for_training(args.model_dir, args.export_dir)
    hffn_optimizer = HFFNMetalAdamW(hffn, modules, lr=args.lr, weight_decay=args.weight_decay)
    hyper = (hffn_optimizer.weight_decay, hffn_optimizer.beta1, hffn_optimizer.beta2, hffn_optimizer.eps)
    guards["optimizer_hyperparameters_equal"] = kept["hyper"] == hyper
    guards["lr_equal"] = kept["lr"] == hffn_optimizer.lr == args.lr

    # E1: the mask of the dense fixed-support optimiser (another loader than the Metal one) is exactly the support of the spec.
    e1 = len(specs) == len(modules) == len(masks)
    for spec, (up_flat, down_flat), (up_mask, down_mask) in zip(specs, supports, (masks[i] for i in range(len(specs)))):
        for flat, mask in ((up_flat, up_mask), (down_flat, down_mask)):
            expected = torch.zeros_like(mask)
            expected[flat] = True
            e1 = e1 and bool(torch.equal(expected, mask)) and int(expected.sum()) == flat.numel()

    hffn_optimizer.zero_grad()
    torch.mps.set_rng_state(kept["paired_rng"])
    hffn_output = hffn(tokens, labels=tokens)
    hffn_output.loss.backward()
    torch.mps.set_rng_state(kept["after_dense_rng"])

    # Everything is read and cloned to the CPU before the step (the shader zeroes `grad` in place and maps a non-finite one to 0).
    hffn_names = grad_names(hffn)
    hffn_eval = (float(hffn_output.loss.detach().cpu()),
                 [split(m.flat.grad.detach().cpu(), s) for m, s in zip(modules, specs)], other_gradients(hffn))
    old_logits = max_abs(hffn_output.logits, kept["logits"])
    old_loss = abs(hffn_eval[0] - dense_eval[0])
    initial_hffn = [split(m.flat.detach().cpu(), s) for m, s in zip(modules, specs)]

    e2 = (dense_names == hffn_names == ref_names
          and all(m.flat.numel() == s.up_val.numel() + s.down_val.numel() + s.d_ff + s.d_model for m, s in zip(modules, specs)))
    e3_gradients = finite([t for layer in hffn_eval[1] + dense_eval[1] for t in layer] + list(hffn_eval[2].values()) + list(dense_eval[2].values()))
    e4 = all(torch.equal(a, b) for la, lb in zip(initial_hffn, initial_dense) for a, b in zip(la, lb))
    del hffn_output

    hffn_optimizer.step()
    torch.mps.synchronize()
    hffn_weights = [split(m.flat.detach().cpu(), s) for m, s in zip(modules, specs)]
    hffn_first = [split(m1.detach().cpu(), s) for (m1, _), s in zip(hffn_optimizer.moments, specs)]
    hffn_second = [split(m2.detach().cpu(), s) for (_, m2), s in zip(hffn_optimizer.moments, specs)]
    with torch.no_grad():
        hffn_next = float(hffn(tokens, labels=tokens).loss.cpu())

    # Gates. T1, T2: step-0 loss and gradients against float64; T3: the update against the closed form from the float64 gradient.
    middle = len(modules) // 2
    rows = {"dense32": tg.distance(dense_eval, reference_eval), "hffn_variant": tg.distance(hffn_eval, reference_eval)}
    control_c = rows["dense32"]["gradient_ffn"]
    scale = 1.0 + max(1e-3, 20.0 * control_c)
    faulty = [tuple(t * scale for t in layer) if i == middle else layer for i, layer in enumerate(hffn_eval[1])]
    rows["hffn_fault_grad"] = tg.distance((hffn_eval[0], faulty, hffn_eval[2]), reference_eval)
    gate, t12 = tg.judge(rows, rows["dense32"], ["variant", "fault_grad"])
    expected = [tuple(zip(w0, g)) for w0, g in zip(initial_hffn, ffn64)]
    update = {"dense32": core.measure(dense_weights, dense_first, dense_second, expected, args.lr, hyper),
              "hffn_variant": core.measure(hffn_weights, hffn_first, hffn_second, expected, args.lr, hyper)}
    displaced = [tuple(w0 + 1.05 * (w - w0) for w0, w in zip(initial_hffn[i], hffn_weights[i])) if i == middle else hffn_weights[i]
                 for i in range(len(modules))]
    update["hffn_fault_update"] = core.measure(displaced, hffn_first, hffn_second, expected, args.lr, hyper)
    t3 = core.judge(update["dense32"], {"variant": update["hffn_variant"], "fault_update": update["hffn_fault_update"]})
    control_t3 = t3["checks"]["control"]
    t3_variant = all(t3["checks"]["variant"].values())
    fault_grad_detected = not (gate["fault_grad"]["gradient_ffn"])
    fault_update_detected = not t3["checks"]["fault_update"]["weight"]

    e3 = e3_gradients and update["hffn_variant"]["finite"] and update["dense32"]["finite"]
    exact = {"E1_support_equals_dense_masks": e1, "E2_gradient_names_and_sizes": e2, "E3_finite": e3, "E4_initial_values_equal": e4}
    t1 = gate["variant"]["loss"]
    t2 = gate["variant"]["gradient_ffn"] and gate["variant"]["gradient_non_ffn"]
    context_ok = guards["dropout_zero"] and guards["optimizer_hyperparameters_equal"] and guards["lr_equal"]
    faults_ok = fault_grad_detected and fault_update_detected
    base_ok = all(exact.values()) and t1 and t2 and faults_ok and context_ok
    if base_ok and control_t3 and t3_variant:
        status, code = "passed", 0
    elif base_ok and not control_t3:
        status, code = "inconclusive_update", 3
    else:
        status, code = "failed", 1
    direct = {"logits_max_abs": old_logits, "loss_abs": old_loss,
              "compact_grad_max_abs_per_layer": [max(max_abs(a, b) for a, b in zip(la, lb)) for la, lb in zip(hffn_eval[1], dense_eval[1])],
              "compact_parameter_max_abs_per_layer": [max(max_abs(a, b) for a, b in zip(la, lb)) for la, lb in zip(hffn_weights, dense_weights)],
              "dense_next_loss": dense_next, "hffn_next_loss": hffn_next, "next_loss_abs": abs(hffn_next - dense_next)}
    result = {
        "format": "hffn_metal_pythia_model_oracle_gate_v1",
        "export": args.export_dir.parent.name, "status": status, "seq_len": args.seq_len, "seed": args.seed,
        "lr": args.lr, "weight_decay": args.weight_decay, "layers": len(modules), "middle_layer": middle,
        "loss_float64": loss64, "context": guards, "exact": exact,
        "T1_loss": t1, "T2_gradients": {"ffn": gate["variant"]["gradient_ffn"], "non_ffn": gate["variant"]["gradient_non_ffn"]},
        "T3_update": {"control_passes": control_t3, "hffn": t3["checks"]["variant"]},
        "distance_to_float64": {k: rows[k] for k in ("dense32", "hffn_variant")},
        "update_rows": {k: update[k] for k in ("dense32", "hffn_variant")},
        "fault_grad": {"scale": scale, "detected": fault_grad_detected, "distance": rows["hffn_fault_grad"]},
        "fault_update": {"displacement_scale": 1.05, "detected": fault_update_detected,
                        "weight_conditioned": update["hffn_fault_update"]["weight_conditioned"]},
        "reported_unjudged_direct_quantities": direct,
    }
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    return code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:  # the JSON is written even when a seam raises; the exit code stays non-zero
        import traceback

        trace = traceback.format_exc()
        print(trace, file=sys.stderr)
        argv = sys.argv
        if "--out" in argv:
            target = Path(argv[argv.index("--out") + 1])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps({"format": "hffn_metal_pythia_model_oracle_gate_v1", "status": "error",
                                          "error": repr(error), "traceback": trace}, indent=2) + "\n")
        sys.exit(2)
