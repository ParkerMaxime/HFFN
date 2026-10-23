#!/usr/bin/env python3
"""Gate PyTorch -> custom Metal -> PyTorch with no explicit synchronization."""

from __future__ import annotations

import json
import torch


def main() -> None:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available; run outside the GPU sandbox")
    lib = torch.mps.compile_shader(
        "kernel void add_one(device float* x, uint i [[thread_position_in_grid]]) { x[i] += 1.0f; }"
    )
    x = torch.arange(257, dtype=torch.float32, device="mps")
    y = x * 2.0
    lib.add_one(y, threads=y.numel(), group_size=32)
    z = y * 3.0
    expected = (torch.arange(257, dtype=torch.float32) * 2.0 + 1.0) * 3.0
    actual = z.cpu()
    if not torch.equal(actual, expected):
        raise AssertionError(f"MPS ordering failure: max error {(actual - expected).abs().max().item()}")
    print(json.dumps({
        "torch": torch.__version__,
        "mps_available": True,
        "compile_shader": True,
        "ordered_without_explicit_sync": True,
        "thread_execution_width": lib.add_one.thread_execution_width,
        "elements": x.numel(),
    }, indent=2))


if __name__ == "__main__":
    main()

