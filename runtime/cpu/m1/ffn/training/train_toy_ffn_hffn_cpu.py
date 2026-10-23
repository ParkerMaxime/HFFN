#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch


PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p/'runtime'/'common'/'pythia'/'hffn_pythia').is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.cpu.m1.ffn.training.config import (
    ARCH_PRESETS,
    TrainConfig,
    resolve_batch_size,
    resolve_cpu_out_dir,
    resolve_dataset_path,
    resolve_model_dims,
)
from runtime.cpu.m1.ffn.training.data import load_dataset, set_seed
from runtime.cpu.m1.ffn.training.io import save_hffn_checkpoint, save_reference_checkpoint
from runtime.cpu.m1.ffn.training.models import SparseMaskedReferenceFFN
from runtime.cpu.m1.ffn.training.shared_toy import HFFNToyFFN
from runtime.cpu.m1.ffn.training.hffn_ffn_training.checkpoint import MAX_DIMENSION
from runtime.cpu.m1.ffn.training.parity import build_optimizer, optimizer_step_parity, parity_snapshot
from runtime.cpu.m1.ffn.training.paths import FFN_SHARED, RUNS_ROOT
from runtime.cpu.m1.ffn.training.torch_info import REFERENCE_BACKEND_NAME, get_torch_cpu_runtime_info

HFFN_BACKEND_NAME = "hffn_shared_g4_autograd_cpu"
from runtime.cpu.m1.ffn.training.trainer import evaluate, train_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CPU HFFN prototype and validate parity.")
    parser.add_argument("--arch", type=str, default=None, choices=sorted(ARCH_PRESETS))
    parser.add_argument("--dataset_path", type=str, default=str(FFN_SHARED / "dataset" / "toy_ffn_dataset.pt"))
    parser.add_argument("--out_dir", type=str, default=str(RUNS_ROOT))
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--density", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--parity_samples", type=int, default=16)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=0)
    parser.add_argument("--max_test_samples", type=int, default=0)
    parser.add_argument("--no_parity_every_epoch", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--untimed", action="store_true", help="Disable duration and throughput reporting (parity/check runs).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    d_model, d_ff = resolve_model_dims(args.arch, args.d_model, args.d_ff)
    if not (0 < d_model <= MAX_DIMENSION and 0 < d_ff <= MAX_DIMENSION):
        raise ValueError("v1 requires 1 <= d_model,d_ff <= 32767")
    if args.parity_samples <= 0 or args.epochs < 0:
        raise ValueError("parity_samples must be positive and epochs nonnegative")
    dataset_path = resolve_dataset_path(args.dataset_path, d_model, d_ff)
    out_dir = resolve_cpu_out_dir(args.out_dir, d_model, d_ff)
    batch_size = resolve_batch_size(args.batch_size, d_model, d_ff)

    config = TrainConfig(
        dataset_path=str(dataset_path),
        out_dir=str(out_dir),
        d_model=d_model,
        d_ff=d_ff,
        density=args.density,
        seed=args.seed,
        batch_size=batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        parity_samples=args.parity_samples,
        parity_every_epoch=not args.no_parity_every_epoch,
    )

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hffn_path = out_dir / "hffn_student.pt"
    ref_path = out_dir / "masked_reference_student.pt"
    log_path = out_dir / "hffn_train_log.json"
    existing = [p for p in [hffn_path, ref_path, log_path] if p.exists()]
    if existing and not args.overwrite:
        existing_str = "\n".join(str(p) for p in existing)
        raise FileExistsError(
            "Output files already exist. Use --overwrite to replace them:\n"
            f"{existing_str}"
        )

    set_seed(config.seed)
    dataset = load_dataset(Path(config.dataset_path))
    x_train = dataset["x_train"].float().cpu()
    y_train = dataset["y_train"].float().cpu()
    x_val = dataset["x_val"].float().cpu()
    y_val = dataset["y_val"].float().cpu()
    x_test = dataset["x_test"].float().cpu()
    y_test = dataset["y_test"].float().cpu()

    if args.max_train_samples > 0:
        x_train = x_train[: args.max_train_samples]
        y_train = y_train[: args.max_train_samples]
    if args.max_val_samples > 0:
        x_val = x_val[: args.max_val_samples]
        y_val = y_val[: args.max_val_samples]
    if args.max_test_samples > 0:
        x_test = x_test[: args.max_test_samples]
        y_test = y_test[: args.max_test_samples]

    if x_train.shape[1] != config.d_model:
        raise ValueError(f"d_model mismatch: dataset has {x_train.shape[1]}, config has {config.d_model}")

    mask_gen = torch.Generator(device="cpu")
    mask_gen.manual_seed(config.seed + 12345)
    shuffle_gen = torch.Generator(device="cpu")
    shuffle_gen.manual_seed(config.seed + 200)

    reference = SparseMaskedReferenceFFN(config.d_model, config.d_ff, config.density, mask_gen).cpu()
    hffn = HFFNToyFFN(reference, max(config.batch_size, min(config.parity_samples, x_train.shape[0]))).cpu()

    ref_opt = build_optimizer(config.optimizer, reference.parameters(), config.lr, config.weight_decay)
    hffn_opt = build_optimizer(config.optimizer, hffn.parameters(), config.lr, config.weight_decay)

    parity_x = x_train[: config.parity_samples].clone()
    parity_y = y_train[: config.parity_samples].clone()
    init_parity = parity_snapshot(reference, hffn, parity_x, parity_y)
    init_step_parity = optimizer_step_parity(
        reference,
        hffn,
        parity_x,
        parity_y,
        config.optimizer,
        config.lr,
        config.weight_decay,
    )

    log: dict[str, object] = {
        "config": asdict(config),
        "timing_enabled": not args.untimed,
        "backends": {
            "reference": REFERENCE_BACKEND_NAME,
            "hffn": HFFN_BACKEND_NAME,
        },
        "runtime": get_torch_cpu_runtime_info(),
        "initial_parity": init_parity,
        "initial_step_parity": init_step_parity,
        "epochs": [],
    }

    print(json.dumps({
        "d_model": config.d_model,
        "d_ff": config.d_ff,
        "density": config.density,
        "batch_size": config.batch_size,
        "optimizer": config.optimizer,
        "backends": log["backends"],
        "runtime": log["runtime"],
        "initial_parity": init_parity,
        "initial_step_parity": init_step_parity,
    }, indent=2))

    total_t0 = None if args.untimed else time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        epoch_t0 = None if args.untimed else time.perf_counter()
        train_stats = train_one_epoch(
            reference,
            hffn,
            ref_opt,
            hffn_opt,
            x_train,
            y_train,
            config.batch_size,
            shuffle_gen,
        )
        ref_val = evaluate(reference, x_val, y_val, config.batch_size)
        hffn_val = evaluate(hffn, x_val, y_val, config.batch_size)
        row: dict[str, object] = {
            "epoch": epoch,
            **train_stats,
            "reference_val_mse": ref_val["mse"],
            "reference_val_relative_l2": ref_val["relative_l2"],
            "hffn_val_mse": hffn_val["mse"],
            "hffn_val_relative_l2": hffn_val["relative_l2"],
            "epoch_seconds": None if args.untimed else time.perf_counter() - epoch_t0,
        }
        row["train_samples_per_second"] = None if args.untimed else float(train_stats["train_samples"]) / max(float(row["epoch_seconds"]), 1e-12)
        if config.parity_every_epoch:
            row["epoch_parity"] = parity_snapshot(reference, hffn, parity_x, parity_y)
        log["epochs"].append(row)
        progress = (
            f"epoch {epoch:03d}/{config.epochs} | "
            f"ref train {train_stats['reference_train_mse']:.6e} val {ref_val['mse']:.6e} | "
            f"hffn train {train_stats['hffn_train_mse']:.6e} val {hffn_val['mse']:.6e} | "
            f"gap {train_stats['mean_step_forward_gap']:.3e}"
        )
        if not args.untimed:
            progress += f" | {row['train_samples_per_second']:.1f} samples/s | {row['epoch_seconds']:.2f}s"
        print(progress)


    ref_test = evaluate(reference, x_test, y_test, config.batch_size)
    hffn_test = evaluate(hffn, x_test, y_test, config.batch_size)
    final_parity = parity_snapshot(reference, hffn, parity_x, parity_y)

    log["test"] = {"reference": ref_test, "hffn": hffn_test}
    log["final_parity"] = final_parity
    log["total_seconds"] = None if args.untimed else time.perf_counter() - total_t0
    log["train_samples_total"] = int(x_train.shape[0] * config.epochs)
    log["mean_train_samples_per_second"] = None if args.untimed else float(log["train_samples_total"]) / max(float(log["total_seconds"]), 1e-12)

    save_hffn_checkpoint(hffn_path, hffn.cpu(), config, {"test": hffn_test, "final_parity": final_parity})
    save_reference_checkpoint(ref_path, reference.cpu().state_dict(), config, {"test": ref_test, "final_parity": final_parity})
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

    print(json.dumps({"test": log["test"], "final_parity": final_parity}, indent=2))


if __name__ == "__main__":
    main()
