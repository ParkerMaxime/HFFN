"""Dense iso-support control of the CPU M1 isolated-FFN inference leaf.

PyTorch references that the HFFN arms are compared against: the dense student FFN and the sparse-dense (masked dense) payload with
its forward. A control only, never an HFFN path: the dense weights live here to be timed against, not to be
executed by the runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseStudentFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.up = nn.Linear(d_model, d_ff, bias=True)
        self.act = nn.GELU()
        self.down = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.act(self.up(x)))


def safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_dense_student(path: Path) -> DenseStudentFFN:
    payload = safe_torch_load(path)
    cfg = payload.get("config", {})
    d_model = int(cfg.get("d_model", 256))
    d_ff = int(cfg.get("d_ff", 1024))
    model = DenseStudentFFN(d_model=d_model, d_ff=d_ff)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model.cpu()


def load_sparse_dense_payload(path: Path) -> Dict[str, torch.Tensor]:
    payload = safe_torch_load(path)
    return {
        "W_up_dense": payload["W_up_dense"].float().cpu().contiguous(),
        "W_down_dense": payload["W_down_dense"].float().cpu().contiguous(),
        "b_up": payload["b_up"].float().cpu().contiguous(),
        "b_down": payload["b_down"].float().cpu().contiguous(),
    }


@torch.no_grad()
def sparse_dense_torch_forward(x: torch.Tensor, payload: Dict[str, torch.Tensor]) -> torch.Tensor:
    return F.gelu(x @ payload["W_up_dense"] + payload["b_up"]) @ payload["W_down_dense"] + payload["b_down"]
