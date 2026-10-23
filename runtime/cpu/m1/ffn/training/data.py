from __future__ import annotations

import random
from pathlib import Path
from typing import Dict

import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def load_dataset(path: Path) -> Dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")
