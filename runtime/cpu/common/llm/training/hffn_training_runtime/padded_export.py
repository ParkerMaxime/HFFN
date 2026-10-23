"""Strict loader for the canonical HFFN padded training artifact."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

# Number of hidden-neuron groups of the G4 format (bound to native/g4_contract.h by the leaf checks).
G4_GROUPS = 4


_REQUIRED = {
    "format", "d_model", "d_ff", "n_hidden_group", "b_up_pad",
    "up_idx_pad", "up_val_pad", "up_deg", "down_idx_pad",
    "down_val_pad", "down_deg", "b_down",
}
_FORBIDDEN_TOKENS = ("ptr", "csr", "csc", "mask", "dense_weight")


@dataclass(frozen=True)
class HFFNPaddedTrainingLayer:
    d_model: int
    d_ff: int
    n_hidden_group: torch.Tensor
    b_up_pad: torch.Tensor
    up_idx_pad: torch.Tensor
    up_val_pad: torch.Tensor
    up_deg: torch.Tensor
    down_idx_pad: torch.Tensor
    down_val_pad: torch.Tensor
    down_deg: torch.Tensor
    b_down: torch.Tensor

    def validate(self) -> None:
        if self.n_hidden_group.shape != (G4_GROUPS,):
            raise ValueError("HFFN padded training export must contain exactly four hidden groups.")
        groups, max_group = self.b_up_pad.shape
        if groups != G4_GROUPS or int(self.n_hidden_group.sum()) != self.d_ff:
            raise ValueError("Invalid hidden-group partition.")
        if int(self.n_hidden_group.min()) < 0 or int(self.n_hidden_group.max()) > max_group:
            raise ValueError("Invalid hidden-group size.")
        if self.up_deg.shape != (G4_GROUPS, max_group) or self.down_deg.shape != (G4_GROUPS, max_group):
            raise ValueError("Invalid degree tensor shape.")
        if self.up_idx_pad.shape[:2] != (G4_GROUPS, max_group) or self.up_val_pad.shape != self.up_idx_pad.shape:
            raise ValueError("Invalid padded up-projection shape.")
        if self.down_idx_pad.shape[:2] != (G4_GROUPS, max_group) or self.down_val_pad.shape != self.down_idx_pad.shape:
            raise ValueError("Invalid padded down-projection shape.")
        if self.b_down.shape != (self.d_model,):
            raise ValueError("Invalid down bias shape.")
        if int(self.up_deg.max()) > self.up_idx_pad.shape[2] or int(self.down_deg.max()) > self.down_idx_pad.shape[2]:
            raise ValueError("Degree exceeds padded width.")
        # Structural validation belongs to export loading, never to the
        # per-step HFFN forward hot path.  Only active padded slots matter.
        for indices, degrees, name in ((self.up_idx_pad, self.up_deg, "up"), (self.down_idx_pad, self.down_deg, "down")):
            slots = torch.arange(indices.shape[2], device=indices.device).view(1, 1, -1)
            active = slots < degrees.unsqueeze(-1)
            values = indices[active]
            if values.numel() and (int(values.min()) < 0 or int(values.max()) >= self.d_model):
                raise ValueError(f"Invalid active {name} HFFN index in export.")


def _load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_hffn_padded_training_layer(path: Path) -> HFFNPaddedTrainingLayer:
    payload = _load(path.resolve())
    if not isinstance(payload, dict):
        raise ValueError("HFFN padded export must be a tensor dictionary.")
    missing = _REQUIRED.difference(payload)
    if missing:
        raise ValueError(f"Not an hffn_training_padded_v1 artifact; missing {sorted(missing)}.")
    if payload["format"] != "hffn_training_padded_v1":
        raise ValueError("Unsupported HFFN training export format.")
    forbidden = [key for key in payload if any(token in key.lower() for token in _FORBIDDEN_TOKENS)]
    if forbidden:
        raise ValueError(f"CSR/dense compatibility tensors are forbidden: {forbidden}.")
    layer = HFFNPaddedTrainingLayer(
        d_model=int(payload["d_model"]), d_ff=int(payload["d_ff"]),
        n_hidden_group=payload["n_hidden_group"].detach().cpu().contiguous().to(torch.int32),
        b_up_pad=payload["b_up_pad"].detach().cpu().contiguous().float(),
        up_idx_pad=payload["up_idx_pad"].detach().cpu().contiguous(),
        up_val_pad=payload["up_val_pad"].detach().cpu().contiguous().float(),
        up_deg=payload["up_deg"].detach().cpu().contiguous().to(torch.int32),
        down_idx_pad=payload["down_idx_pad"].detach().cpu().contiguous(),
        down_val_pad=payload["down_val_pad"].detach().cpu().contiguous().float(),
        down_deg=payload["down_deg"].detach().cpu().contiguous().to(torch.int32),
        b_down=payload["b_down"].detach().cpu().contiguous().float(),
    )
    layer.validate()
    return layer
