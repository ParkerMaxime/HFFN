#!/usr/bin/env python3
"""One complete no-fusion Pythia training step on MPS + exact-offset HFFN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hffn_metal_llm import HFFNMetalAdamW, load_pythia_hffn_metal_for_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=712)
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available; run outside the GPU sandbox")
    torch.manual_seed(args.seed)
    model, modules = load_pythia_hffn_metal_for_training(args.model_dir, args.export_dir)
    optimizer = HFFNMetalAdamW(
        model, modules, lr=args.lr, weight_decay=args.weight_decay
    )
    if args.accum_steps < 1:
        raise ValueError("accum-steps must be positive")
    tokens = [
        torch.randint(0, model.config.vocab_size, (1, args.seq_len), device="mps")
        for _ in range(args.accum_steps)
    ]
    before = [module.flat.detach().clone() for module in modules]
    non_ffn = next(parameter for name, parameter in model.named_parameters() if ".mlp." not in name)
    non_ffn_before = non_ffn.detach().clone()
    optimizer.zero_grad()
    losses = []
    for microbatch in tokens:
        loss = model(microbatch, labels=microbatch).loss
        (loss / args.accum_steps).backward()
        losses.append(float(loss.detach().cpu()))
    optimizer.step()
    torch.mps.synchronize()
    changed = [bool((old - module.flat).abs().max().item() > 0.0) for old, module in zip(before, modules)]
    finite = [bool(torch.isfinite(module.flat).all().item()) for module in modules]
    print(json.dumps({
        "format": "hffn_metal_pythia_training_smoke_v1",
        "losses": losses,
        "accum_steps": args.accum_steps,
        "layers": len(modules),
        "seq_len": args.seq_len,
        "compact_numel_per_layer": [module.compact_numel for module in modules],
        "all_compact_layers_changed": all(changed),
        "all_compact_layers_finite": all(finite),
        "non_ffn_parameter_changed": bool((non_ffn_before - non_ffn).abs().max().item() > 0.0),
        "adamw_steps": optimizer.steps,
        "layouts": [module.runtime_for_n(args.seq_len).layout for module in modules],
    }, indent=2))


if __name__ == "__main__":
    main()
