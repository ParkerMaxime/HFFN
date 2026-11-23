from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Dict

import torch

from runtime.cpu.m1.ffn.training.config import TrainConfig

if TYPE_CHECKING:
    from runtime.cpu.m1.ffn.training.shared_toy import HFFNToyFFN


def save_hffn_checkpoint(path: Path, model: HFFNToyFFN, config: TrainConfig, extra: Dict) -> None:
    from runtime.cpu.m1.ffn.training.hffn_ffn_training.checkpoint import export_checkpoint
    shared = export_checkpoint(model)
    payload = {
        "state_dict": shared["state_dict"],
        "config": asdict(config),
        "extra": extra,
        "architecture": {
            "type": model.__class__.__name__,
            "d_model": model.d_model,
            "d_ff": model.d_ff,
            "activation": "gelu_exact",
            "layout": "hidden_major_compact",
        },
    }
    payload["hffn_runtime"] = {key: value for key, value in shared.items() if key != "state_dict"}
    torch.save(payload, path)


def save_reference_checkpoint(path: Path, state_dict: Dict, config: TrainConfig, extra: Dict) -> None:
    torch.save(
        {
            "state_dict": state_dict,
            "config": asdict(config),
            "extra": extra,
        },
        path,
    )
