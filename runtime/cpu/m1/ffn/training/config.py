from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from runtime.cpu.m1.ffn.training.paths import FFN_ROOT, FFN_SHARED, RUNS_ROOT


ARCH_PRESETS = {
    "128_512": (128, 512),
    "256_512": (256, 512),
    "256_1024": (256, 1024),
    "256_2048": (256, 2048),
    "512_1024": (512, 1024),
    "512_2048": (512, 2048),
    "512_4096": (512, 4096),
    "1024_4096": (1024, 4096),
}
BASELINE_DIMS = (256, 1024)


@dataclass
class TrainConfig:
    dataset_path: str = ""
    out_dir: str = ""
    d_model: int = 256
    d_ff: int = 1024
    density: float = 0.10
    seed: int = 43
    batch_size: int = 0
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adamw"
    dtype: str = "float32"
    device: str = "cpu"
    parity_samples: int = 16
    parity_every_epoch: bool = True


def resolve_model_dims(arch: str | None, d_model: int, d_ff: int) -> tuple[int, int]:
    if arch is None:
        return int(d_model), int(d_ff)
    return ARCH_PRESETS[arch]


def resolve_dataset_path(raw_dataset_path: str, d_model: int, d_ff: int) -> Path:
    baseline = FFN_SHARED / "dataset" / "toy_ffn_dataset.pt"
    path = Path(raw_dataset_path)
    if path != baseline or (d_model, d_ff) == BASELINE_DIMS:
        return path
    return FFN_ROOT / f"{d_model}_{d_ff}" / "dataset" / "toy_ffn_dataset.pt"


def resolve_cpu_out_dir(raw_out_dir: str, d_model: int, d_ff: int) -> Path:
    out_dir = Path(raw_out_dir)
    default_root = RUNS_ROOT
    if out_dir != default_root:
        return out_dir
    if (d_model, d_ff) == BASELINE_DIMS:
        return default_root / "256_1024"
    return default_root / f"{d_model}_{d_ff}"


def resolve_batch_size(raw_batch_size: int, d_model: int, d_ff: int) -> int:
    if raw_batch_size > 0:
        return int(raw_batch_size)
    if d_model >= 1024 or d_ff >= 4096:
        return 32
    if d_model >= 512 or d_ff >= 2048:
        return 64
    return 256
