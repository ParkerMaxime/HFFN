#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p/'runtime'/'common'/'pythia'/'hffn_pythia').is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.cpu.m1.ffn.training.config import (
    ARCH_PRESETS,
    TrainConfig,
    resolve_batch_size,
    resolve_dataset_path,
    resolve_model_dims,
)
from runtime.cpu.m1.ffn.training.data import load_dataset, set_seed
from runtime.cpu.m1.ffn.training.io import save_hffn_checkpoint, save_reference_checkpoint
from runtime.cpu.m1.ffn.training.models import SparseMaskedReferenceFFN
from runtime.cpu.m1.ffn.training.shared_toy import HFFNToyFFN
from runtime.cpu.m1.ffn.training.parity import build_optimizer, parity_snapshot
from runtime.cpu.m1.ffn.training.paths import TRAINING_ROOT, FFN_SHARED
from runtime.cpu.m1.ffn.training.torch_info import get_torch_cpu_runtime_info
from runtime.cpu.m1.ffn.training.trainer import evaluate



REFERENCE_BACKEND = "pytorch_cpu_auto_thread_masked"
HFFN_BACKEND_FULL = "hffn_shared_g4_autograd_full_cpu"
HFFN_BACKEND_WEIGHTS_ONLY = "hffn_shared_g4_autograd_weights_only_cpu"


def tensor_nbytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel() * tensor.element_size())


def optimizer_state_nbytes(optimizer: torch.optim.Optimizer | None) -> int:
    if optimizer is None:
        return 0
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                total += tensor_nbytes(value)
    return total


def activation_bytes_estimate(batch_size: int, d_model: int, d_ff: int, *, include_grad_x: bool) -> dict[str, int]:
    bytes_per_float = 4
    x_bytes = batch_size * d_model * bytes_per_float
    y_bytes = batch_size * d_model * bytes_per_float
    z_bytes = batch_size * d_ff * bytes_per_float
    h_bytes = batch_size * d_ff * bytes_per_float
    grad_y_bytes = batch_size * d_model * bytes_per_float
    grad_z_bytes = batch_size * d_ff * bytes_per_float
    grad_x_bytes = batch_size * d_model * bytes_per_float if include_grad_x else 0
    grad_h_bytes = batch_size * d_ff * bytes_per_float
    return {
        "x_bytes": x_bytes,
        "y_bytes": y_bytes,
        "z_bytes": z_bytes,
        "h_bytes": h_bytes,
        "grad_y_bytes": grad_y_bytes,
        "grad_z_bytes": grad_z_bytes,
        "grad_x_bytes": grad_x_bytes,
        "grad_h_bytes": grad_h_bytes,
        "total_bytes": x_bytes + y_bytes + z_bytes + h_bytes + grad_y_bytes + grad_z_bytes + grad_x_bytes + grad_h_bytes,
    }


def reference_memory_profile(
    reference: SparseMaskedReferenceFFN,
    optimizer: torch.optim.Optimizer | None,
    *,
    batch_size: int,
    d_model: int,
    d_ff: int,
    include_grad_x: bool,
) -> dict[str, object]:
    param_tensors = [
        reference.up.weight,
        reference.up.bias,
        reference.down.weight,
        reference.down.bias,
    ]
    mask_tensors = [reference.up.mask, reference.down.mask]
    param_bytes = sum(tensor_nbytes(t) for t in param_tensors)
    mask_bytes = sum(tensor_nbytes(t) for t in mask_tensors)
    grad_bytes_estimate = param_bytes
    optimizer_bytes = optimizer_state_nbytes(optimizer)
    if optimizer_bytes == 0 and optimizer is not None and type(optimizer).__name__ == "AdamW":
        optimizer_bytes = 2 * param_bytes
    activations = activation_bytes_estimate(batch_size, d_model, d_ff, include_grad_x=include_grad_x)
    active_up = int(reference.up.mask.sum().item())
    active_down = int(reference.down.mask.sum().item())
    structural_total = param_bytes + mask_bytes + grad_bytes_estimate + optimizer_bytes
    total_with_activations = structural_total + activations["total_bytes"]
    return {
        "trainable_parameter_elements": int(sum(t.numel() for t in param_tensors)),
        "active_weight_elements": {
            "up": active_up,
            "down": active_down,
            "total": active_up + active_down,
        },
        "param_bytes": param_bytes,
        "mask_bytes": mask_bytes,
        "grad_bytes_estimate": grad_bytes_estimate,
        "optimizer_state_bytes": optimizer_bytes,
        "activation_bytes_estimate": activations,
        "structural_total_bytes": structural_total,
        "total_with_activations_bytes_estimate": total_with_activations,
    }


def hffn_runtime_buffer_profile(hffn: HFFNToyFFN) -> dict[str, int]:
    """Persistent Step allocations, excluding compact parameters, metadata and moments.

    Count reservations rather than their current minibatch views, once each.
    """
    step = hffn.step
    buffers = {name + "_bytes": tensor_nbytes(flat) for name, (flat, _, _) in step._reserved.items()}
    buffers.update({name + "_bytes": tensor_nbytes(getattr(step, name))
                    for name in ("_gx_local", "g_up", "g_down", "g_b_up", "g_b_down", "timing")})
    buffers["total_bytes"] = sum(buffers.values())
    return buffers


def hffn_memory_profile(
    hffn: HFFNToyFFN,
    optimizer: torch.optim.Optimizer | None,
    *,
    batch_size: int,
    d_model: int,
    d_ff: int,
    include_grad_x: bool,
) -> dict[str, object]:
    parameters = [hffn.up_val, hffn.down_val, hffn.b_up, hffn.b_down]
    parameter_names = {"up_val", "down_val", "b_up", "b_down"}
    metadata = [value for name, value in vars(hffn.step.layer).items()
                if name not in parameter_names and isinstance(value, torch.Tensor)]
    param_bytes = sum(tensor_nbytes(value) for value in parameters)
    index_bytes = sum(tensor_nbytes(value) for value in metadata)
    # Parameter.grad is the autograd-owned copy, separate from Step.g_*.
    compact_grad_bytes = param_bytes
    # The shared Step reserves its moments even when the caller selects SGD.
    optimizer_bytes = sum(tensor_nbytes(value) for value in hffn.step.moments) + optimizer_state_nbytes(optimizer)
    runtime_buffers = hffn_runtime_buffer_profile(hffn)
    # Step z/h and native work arrays are already in runtime_buffers. These
    # additional activation-space tensors are ordinary autograd/input copies.
    activation_elements = batch_size * d_model
    activations = {name: activation_elements * 4 for name in ("x_bytes", "y_bytes", "grad_y_bytes")}
    activations["grad_x_bytes"] = activation_elements * 4 if include_grad_x else 0
    activations["total_bytes"] = sum(activations.values())
    structural_total = param_bytes + index_bytes + compact_grad_bytes + optimizer_bytes + runtime_buffers["total_bytes"]
    return {
        "trainable_parameter_elements": sum(value.numel() for value in parameters),
        "active_weight_elements": {"up": hffn.up_val.numel(), "down": hffn.down_val.numel(),
                                   "total": hffn.up_val.numel() + hffn.down_val.numel()},
        "param_bytes": param_bytes,
        "index_bytes": index_bytes,
        "compact_grad_bytes": compact_grad_bytes,
        "optimizer_state_bytes": optimizer_bytes,
        "runtime_buffer_bytes": runtime_buffers,
        "activation_bytes_estimate": activations,
        "structural_total_bytes": structural_total,
        "total_with_activations_bytes_estimate": structural_total + activations["total_bytes"],
    }


def memory_snapshot(reference, hffn, ref_opt, hffn_opt, config, hffn_mode):
    snapshot = {
        "reference": reference_memory_profile(reference, ref_opt, batch_size=config.batch_size,
                                               d_model=config.d_model, d_ff=config.d_ff,
                                               include_grad_x=hffn_mode == "full"),
        "hffn": hffn_memory_profile(hffn, hffn_opt, batch_size=config.batch_size,
                                   d_model=config.d_model, d_ff=config.d_ff,
                                   include_grad_x=hffn_mode == "full"),
    }
    snapshot["ratios"] = {
        "reference_over_hffn_structural_total": snapshot["reference"]["structural_total_bytes"] /
            max(snapshot["hffn"]["structural_total_bytes"], 1),
        "reference_over_hffn_total_with_activations": snapshot["reference"]["total_with_activations_bytes_estimate"] /
            max(snapshot["hffn"]["total_with_activations_bytes_estimate"], 1),
    }
    return snapshot


def memory_display(snapshot):
    if snapshot is None:
        return {key: None for key in ("reference_structural_mb", "hffn_structural_mb",
                                      "reference_total_estimated_mb", "hffn_total_estimated_mb",
                                      "reference_over_hffn_total_ratio")}
    mib = 1024.0 * 1024.0
    return {
        "reference_structural_mb": snapshot["reference"]["structural_total_bytes"] / mib,
        "hffn_structural_mb": snapshot["hffn"]["structural_total_bytes"] / mib,
        "reference_total_estimated_mb": snapshot["reference"]["total_with_activations_bytes_estimate"] / mib,
        "hffn_total_estimated_mb": snapshot["hffn"]["total_with_activations_bytes_estimate"] / mib,
        "reference_over_hffn_total_ratio": snapshot["ratios"]["reference_over_hffn_total_with_activations"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare masked PyTorch and shared HFFN training with ordinary PyTorch MSE/autograd."
    )
    parser.add_argument("--arch", type=str, default="256_1024", choices=sorted(ARCH_PRESETS))
    parser.add_argument("--dataset_path", type=str, default=str(FFN_SHARED / "dataset" / "toy_ffn_dataset.pt"))
    parser.add_argument("--out_dir", type=str, default=str(TRAINING_ROOT / "runs"))
    parser.add_argument("--tag", type=str, default="compare_100epochs")
    parser.add_argument("--density", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--hffn_mode", type=str, default="full", choices=["full", "weights_only"])
    parser.add_argument("--parity_samples", type=int, default=16)
    parser.add_argument("--parity_every", type=int, default=0)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=0)
    parser.add_argument("--max_test_samples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--untimed", action="store_true",
                        help="Skip all duration, throughput and structural-memory accounting (parity/check runs).")
    return parser.parse_args()


def resolve_compare_out_dir(raw_out_dir: str, d_model: int, d_ff: int, tag: str) -> Path:
    out_root = Path(raw_out_dir)
    default_root = TRAINING_ROOT / "runs"
    if out_root == default_root:
        out_root = out_root / f"{d_model}_{d_ff}"
    return out_root / tag


def train_epoch(
    model,
    optimizer,
    x_train,
    y_train,
    batch_size,
    shuffle_generator,
    *,
    native_adamw=False,
    input_grad=True,
    untimed=False,
):
    """Ordinary PyTorch MSE/autograd on both arms; shared Step owns native FFN work.

    Phase boundaries surround Python operations. Forward and MSE are measured
    together; they are not the phases of a fused MSE.
    """
    model.train()
    n = int(x_train.shape[0])
    perm = torch.randperm(n, generator=shuffle_generator)
    total_loss, total_samples = 0.0, 0
    phases = {name: None if untimed else 0.0 for name in
              ("zero_grad_seconds", "forward_loss_seconds", "backward_seconds", "update_seconds")}
    epoch_t0 = None if untimed else time.perf_counter()
    for start in range(0, n, batch_size):
        idx = perm[start:start + batch_size]
        xb, yb = x_train[idx], y_train[idx]
        t0 = None if untimed else time.perf_counter()
        if optimizer is None:
            model.zero_grad(set_to_none=True)
        else:
            optimizer.zero_grad(set_to_none=True)
        if not untimed:
            phases["zero_grad_seconds"] += time.perf_counter() - t0

        x = xb.detach().clone().requires_grad_(input_grad)
        t0 = None if untimed else time.perf_counter()
        loss = F.mse_loss(model(x), yb)
        if not untimed:
            phases["forward_loss_seconds"] += time.perf_counter() - t0
        t0 = None if untimed else time.perf_counter()
        loss.backward()
        if not untimed:
            phases["backward_seconds"] += time.perf_counter() - t0
        t0 = None if untimed else time.perf_counter()
        if native_adamw:
            model.step.update()
        else:
            optimizer.step()
        if isinstance(model, SparseMaskedReferenceFFN):
            model.apply_masks_()
        if not untimed:
            phases["update_seconds"] += time.perf_counter() - t0
        bs = int(xb.shape[0])
        total_loss += float(loss.item()) * bs
        total_samples += bs
    epoch_seconds = None if untimed else time.perf_counter() - epoch_t0
    return {
        "train_mse": total_loss / max(total_samples, 1),
        "train_samples": float(total_samples),
        "epoch_seconds": epoch_seconds,
        "samples_per_second": None if untimed else total_samples / max(epoch_seconds, 1e-12),
        **phases,
        "timed_total_seconds": None if untimed else sum(phases.values()),
    }


def write_history_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_write_plot(path: Path, rows: list[dict[str, object]]) -> str | None:
    cache_root = Path(tempfile.gettempdir()) / "toy_sparse_ffn_matplotlib_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    epochs = [int(row["epoch"]) for row in rows]
    ref_train = [float(row["reference_train_mse"]) for row in rows]
    hffn_train = [float(row["hffn_train_mse"]) for row in rows]
    ref_val = [float(row["reference_val_mse"]) for row in rows]
    hffn_val = [float(row["hffn_val_mse"]) for row in rows]
    throughput_available = bool(rows) and all(row["reference_samples_per_second"] is not None and
                                               row["hffn_samples_per_second"] is not None for row in rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, ref_train, label="PyTorch train", color="#1f77b4", linewidth=2)
    axes[0].plot(epochs, hffn_train, label="HFFN train", color="#d62728", linewidth=2)
    axes[0].plot(epochs, ref_val, label="PyTorch val", color="#1f77b4", linestyle="--", linewidth=1.8)
    axes[0].plot(epochs, hffn_val, label="HFFN val", color="#d62728", linestyle="--", linewidth=1.8)
    axes[0].set_title("Loss curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    if throughput_available:
        axes[1].plot(epochs, [row["reference_samples_per_second"] for row in rows],
                     label="PyTorch samples/s", color="#1f77b4", linewidth=2)
        axes[1].plot(epochs, [row["hffn_samples_per_second"] for row in rows],
                     label="HFFN samples/s", color="#d62728", linewidth=2)
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "Throughput unavailable (untimed run)", ha="center", va="center",
                     transform=axes[1].transAxes)
    axes[1].set_title("Training throughput")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Samples/s")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def main() -> None:
    args = parse_args()
    d_model, d_ff = resolve_model_dims(args.arch, 256, 1024)
    if args.parity_samples <= 0 or args.epochs < 0:
        raise ValueError("parity_samples must be positive and epochs nonnegative")
    dataset_path = resolve_dataset_path(args.dataset_path, d_model, d_ff)
    batch_size = resolve_batch_size(args.batch_size, d_model, d_ff)
    out_dir = resolve_compare_out_dir(args.out_dir, d_model, d_ff, args.tag)

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
        parity_every_epoch=False,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training_compare_log.json"
    csv_path = out_dir / "training_compare_history.csv"
    plot_path = out_dir / "training_compare_curves.png"
    hffn_path = out_dir / "hffn_native_student.pt"
    ref_path = out_dir / "pytorch_reference_student.pt"
    existing = [p for p in [log_path, csv_path, plot_path, hffn_path, ref_path] if p.exists()]
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

    mask_gen = torch.Generator(device="cpu")
    mask_gen.manual_seed(config.seed + 12345)
    ref_shuffle_gen = torch.Generator(device="cpu")
    ref_shuffle_gen.manual_seed(config.seed + 200)
    hffn_shuffle_gen = torch.Generator(device="cpu")
    hffn_shuffle_gen.manual_seed(config.seed + 200)

    reference = SparseMaskedReferenceFFN(config.d_model, config.d_ff, config.density, mask_gen).cpu()
    reservation = max(config.batch_size, min(config.parity_samples, x_train.shape[0]), 1)
    hffn = HFFNToyFFN(reference, reservation, lr=config.lr, weight_decay=config.weight_decay).cpu()

    ref_opt = build_optimizer(config.optimizer, reference.parameters(), config.lr, config.weight_decay)
    hffn_opt = None if config.optimizer == "adamw" else build_optimizer(
        config.optimizer, hffn.parameters(), config.lr, config.weight_decay
    )

    parity_x = x_train[: config.parity_samples].clone()
    parity_y = y_train[: config.parity_samples].clone()
    init_parity = parity_snapshot(reference, hffn, parity_x, parity_y)

    hffn_backend = HFFN_BACKEND_FULL if args.hffn_mode == "full" else HFFN_BACKEND_WEIGHTS_ONLY
    log: dict[str, object] = {
        "config": asdict(config),
        "timing_enabled": not args.untimed,
        "memory_accounting_enabled": not args.untimed,
        "experiment": {
            "kind": "training_compare",
            "hffn_mode": args.hffn_mode,
            "tag": args.tag,
            "optimizer_path": "shared_step_native_adamw" if args.optimizer == "adamw" else "pytorch_sgd",
            "timing_perimeter": "ordinary_pytorch_mse_autograd_shared_ffn; Python phase walls; epoch includes minibatch indexing and input copy",
            "phase_fields": "forward and loss are timed together: hffn_forward_loss_seconds",
            "input_gradient_control": "full on both arms, or weights-only on both arms",
        },
        "backends": {
            "reference": REFERENCE_BACKEND,
            "hffn": hffn_backend,
        },
        "runtime": get_torch_cpu_runtime_info(),
        "initial_parity": init_parity,
        "epochs": [],
    }

    initial_memory = None if args.untimed else memory_snapshot(
        reference, hffn, ref_opt, hffn_opt, config, args.hffn_mode)
    log["memory"] = {
        "measurement_contract": {
            "unit": "bytes",
            "scope": "backend_structural_memory_plus_batch_runtime_buffers",
            "reference_activation_memory": "deterministic_estimate",
            "hffn_activation_memory": "autograd_copy_estimate_excluding_persistent_Step_arrays",
            "hffn_runtime_buffers": "exact_persistent_Step_reservations_including_native_gradients",
            "hffn_metadata": "all_compact_support_and_reverse_schedules",
            "hffn_optimizer_state": "all_reserved_Step_moments_plus_torch_optimizer_state",
            "reference_autograd_buffers": "not_measured_exactly",
            "note": "Structural tensor accounting and activation estimates; process RSS and allocator peaks are not attributed.",
        },
        "initial": initial_memory,
    }
    print(json.dumps({
        "arch": args.arch, "density": config.density, "batch_size": config.batch_size,
        "epochs": config.epochs, "optimizer": config.optimizer, "backends": log["backends"],
        "runtime": log["runtime"], "initial_parity": init_parity,
        "initial_memory": memory_display(initial_memory),
    }, indent=2))

    total_t0 = None if args.untimed else time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        ref_stats = train_epoch(
            reference,
            ref_opt,
            x_train,
            y_train,
            config.batch_size,
            ref_shuffle_gen,
            input_grad=args.hffn_mode == "full",
            untimed=args.untimed,
        )
        hffn_stats = train_epoch(
            hffn,
            hffn_opt,
            x_train,
            y_train,
            config.batch_size,
            hffn_shuffle_gen,
            native_adamw=config.optimizer == "adamw",
            input_grad=args.hffn_mode == "full",
            untimed=args.untimed,
        )
        ref_val = evaluate(reference, x_val, y_val, config.batch_size)
        hffn_val = evaluate(hffn, x_val, y_val, config.batch_size)

        row: dict[str, object] = {
            "epoch": epoch,
            "reference_train_mse": ref_stats["train_mse"],
            "hffn_train_mse": hffn_stats["train_mse"],
            "reference_val_mse": ref_val["mse"],
            "hffn_val_mse": hffn_val["mse"],
            "reference_val_relative_l2": ref_val["relative_l2"],
            "hffn_val_relative_l2": hffn_val["relative_l2"],
            "reference_epoch_seconds": ref_stats["epoch_seconds"],
            "hffn_epoch_seconds": hffn_stats["epoch_seconds"],
            "reference_samples_per_second": ref_stats["samples_per_second"],
            "hffn_samples_per_second": hffn_stats["samples_per_second"],
            "reference_zero_grad_seconds": ref_stats["zero_grad_seconds"],
            "reference_forward_loss_seconds": ref_stats["forward_loss_seconds"],
            "reference_backward_seconds": ref_stats["backward_seconds"],
            "reference_update_seconds": ref_stats["update_seconds"],
            "hffn_zero_grad_seconds": hffn_stats["zero_grad_seconds"],
            "hffn_forward_seconds": None,
            "hffn_forward_loss_seconds": hffn_stats["forward_loss_seconds"],
            "hffn_loss_grad_y_seconds": None,
            "hffn_backward_seconds": hffn_stats["backward_seconds"],
            "hffn_update_seconds": hffn_stats["update_seconds"],
            "epoch_speedup_hffn_vs_reference": None if args.untimed else ref_stats["epoch_seconds"] / max(hffn_stats["epoch_seconds"], 1e-12),
        }
        if args.parity_every > 0 and (epoch == 1 or epoch % args.parity_every == 0 or epoch == config.epochs):
            row["epoch_parity"] = parity_snapshot(reference, hffn, parity_x, parity_y)
        log["epochs"].append(row)
        progress = (
            f"epoch {epoch:03d}/{config.epochs} | "
            f"ref train {row['reference_train_mse']:.6e} val {row['reference_val_mse']:.6e} | "
            f"hffn train {row['hffn_train_mse']:.6e} val {row['hffn_val_mse']:.6e}"
        )
        if not args.untimed:
            progress += (f" | speedup {row['epoch_speedup_hffn_vs_reference']:.2f}x | "
                         f"ref {row['reference_samples_per_second']:.1f} s/s | "
                         f"hffn {row['hffn_samples_per_second']:.1f} s/s")
        print(progress)

    ref_test = evaluate(reference, x_test, y_test, config.batch_size)
    hffn_test = evaluate(hffn, x_test, y_test, config.batch_size)
    final_parity = parity_snapshot(reference, hffn, parity_x, parity_y)
    total_seconds = None if args.untimed else time.perf_counter() - total_t0
    final_memory = None if args.untimed else memory_snapshot(
        reference, hffn, ref_opt, hffn_opt, config, args.hffn_mode)
    log["memory"]["final"] = final_memory
    reference_total_epoch_seconds = None if args.untimed else sum(float(row["reference_epoch_seconds"]) for row in log["epochs"])
    hffn_total_epoch_seconds = None if args.untimed else sum(float(row["hffn_epoch_seconds"]) for row in log["epochs"])
    final_memory_display = memory_display(final_memory)

    summary = {
        "reference_test": ref_test,
        "hffn_test": hffn_test,
        "final_parity": final_parity,
        "reference_total_epoch_seconds": reference_total_epoch_seconds,
        "hffn_total_epoch_seconds": hffn_total_epoch_seconds,
        "end_to_end_speedup_hffn_vs_reference": None if args.untimed else reference_total_epoch_seconds / max(hffn_total_epoch_seconds, 1e-12),
        "total_wall_clock_seconds": total_seconds,
        "memory_reference_structural_mb": final_memory_display["reference_structural_mb"],
        "memory_hffn_structural_mb": final_memory_display["hffn_structural_mb"],
        "memory_reference_total_estimated_mb": final_memory_display["reference_total_estimated_mb"],
        "memory_hffn_total_estimated_mb": final_memory_display["hffn_total_estimated_mb"],
        "memory_reference_over_hffn_total_ratio": final_memory_display["reference_over_hffn_total_ratio"],
    }
    log["summary"] = summary

    write_history_csv(csv_path, list(log["epochs"]))
    plot_result = maybe_write_plot(plot_path, list(log["epochs"]))
    if plot_result is None and plot_path.exists():
        plot_path.unlink()
    if plot_result is not None:
        log["artifacts"] = {"history_csv": str(csv_path), "curve_png": plot_result}
    else:
        log["artifacts"] = {"history_csv": str(csv_path), "curve_png": None}

    save_hffn_checkpoint(hffn_path, hffn.cpu(), config, summary)
    save_reference_checkpoint(ref_path, reference.cpu().state_dict(), config, summary)
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
