"""Parity of the CUDA runtime on one FFN of a runtime against a float64 oracle, shared by the checks of the leaves.

For one input, the runtime runs with its policy and with each of the nine forced (up, down) kernel pairs.
Criteria: two consecutive calls on the same input give the same bits; ``max |y - y_float64| / max(1, |y_float64|)``
is reported for the leaf to compare with its guard. Reported without threshold: whether the four sequential pairs
(up lane/tile, down tiled/blocked) give the same bits.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .runtime import DOWN_MODES, UP_MODES, HFFNRuntime

PAIRS = [(None, None)] + [(up, down) for up in UP_MODES for down in DOWN_MODES]


@dataclass
class Parity:
    error: float           # max |y - y_float64| over the pairs
    relative: float        # max |y - y_float64| / max(1, |y_float64|) over the pairs
    repeatable: bool       # every pair gives the same bits on a second call
    sequential_bits: bool  # the four sequential pairs give the same bits


def check(runtime: HFFNRuntime, index: int, x: torch.Tensor, expected: torch.Tensor) -> Parity:
    """``x`` float32 [N, d_model] on the runtime's device, ``expected`` its float64 oracle output."""
    scale = expected.abs().clamp(min=1.0)
    result, sequential = Parity(0.0, 0.0, True, True), None
    for up, down in PAIRS:
        runtime.force(up, down)
        y = runtime.forward(index, x).clone()
        result.repeatable &= bool(torch.equal(y, runtime.forward(index, x)))
        distance = (y.double() - expected).abs()
        result.error = max(result.error, float(distance.max()))
        result.relative = max(result.relative, float((distance / scale).max()))
        if up in ("lane", "tile") and down in ("tiled", "blocked"):
            if sequential is None:
                sequential = y
            result.sequential_bits &= bool(torch.equal(y, sequential))
    runtime.force(None, None)
    return result
