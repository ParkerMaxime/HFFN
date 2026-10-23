#!/usr/bin/env python3
"""Checkpoint/resume gate for compact parameters and both AdamW arms."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch
from torch import nn

from hffn_metal_llm import (
    HFFNMetalAdamW,
    HFFNMetalMLP,
    load_compact_wanda_layer,
    load_training_checkpoint,
    save_training_checkpoint,
)


class TinyTrainingState(nn.Module):
    def __init__(self, spec) -> None:
        super().__init__()
        self.base = nn.Parameter(torch.randn(31, device="mps"))
        self.mlp = HFFNMetalMLP(0, spec)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=Path, required=True)
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available; run outside the GPU sandbox")
    torch.manual_seed(712)
    spec = load_compact_wanda_layer(args.layer)
    source = TinyTrainingState(spec)
    source_optimizer = HFFNMetalAdamW(source, (source.mlp,), lr=3e-5, weight_decay=1e-2)
    source.base.grad = torch.randn_like(source.base)
    source.mlp.flat.grad = torch.randn_like(source.mlp.flat)
    source_optimizer.step()
    torch.mps.synchronize()
    with tempfile.TemporaryDirectory(prefix="hffn-metal-resume-") as directory:
        checkpoint = Path(directory) / "state.pt"
        save_training_checkpoint(checkpoint, source, source_optimizer, metadata={"global_step": 17})
        resumed = TinyTrainingState(spec)
        resumed_optimizer = HFFNMetalAdamW(resumed, (resumed.mlp,), lr=9e-4, weight_decay=0.0)
        metadata = load_training_checkpoint(checkpoint, resumed, resumed_optimizer)
    errors = {
        "base_parameter": float((source.base - resumed.base).detach().abs().max().cpu()),
        "compact_parameter": float((source.mlp.flat - resumed.mlp.flat).detach().abs().max().cpu()),
        "compact_moment1": float((source_optimizer.moments[0][0] - resumed_optimizer.moments[0][0]).detach().abs().max().cpu()),
        "compact_moment2": float((source_optimizer.moments[0][1] - resumed_optimizer.moments[0][1]).detach().abs().max().cpu()),
    }
    passed = max(errors.values()) == 0.0 and resumed_optimizer.steps == [1] and metadata == {"global_step": 17}
    if not passed:
        raise AssertionError({"errors": errors, "steps": resumed_optimizer.steps, "metadata": metadata})
    print(json.dumps({
        "format": "hffn_metal_optimizer_resume_gate_v1",
        "passed": True,
        "errors": errors,
        "steps": resumed_optimizer.steps,
        "lr": resumed_optimizer.lr,
        "weight_decay": resumed_optimizer.weight_decay,
        "metadata": metadata,
    }, indent=2))


if __name__ == "__main__":
    main()
