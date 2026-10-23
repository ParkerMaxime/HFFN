"""Atomic training checkpoints for compact-Metal Pythia models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .optimizer import HFFNMetalAdamW


def save_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: HFFNMetalAdamW,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    payload = {
        "format": "hffn_metal_pythia_training_checkpoint_v1",
        "model": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "metadata": dict(metadata or {}),
    }
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: HFFNMetalAdamW,
) -> dict[str, Any]:
    try:
        payload = torch.load(Path(path).resolve(), map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(Path(path).resolve(), map_location="cpu")
    if not isinstance(payload, dict) or payload.get("format") != "hffn_metal_pythia_training_checkpoint_v1":
        raise ValueError("expected hffn_metal_pythia_training_checkpoint_v1")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata must be a dictionary")
    return metadata
