"""Parity checks shared by pytest and by ``python -m hffn.checks``.

The kernel must match, on every route and batch size, the float64 reference (platform-independent oracle)
within ATOL, on the shared test layers (``hffn_ffn_layers.layers``, pinned by their fingerprints).
"""
from __future__ import annotations

import sys
from functools import lru_cache

import numpy as np

from hffn_ffn_layers import layers, reference

from .compact import CompactKernel

# Outputs have magnitude ~6; -ffast-math and FMA contraction make bits compiler- and
# ISA-dependent, so parity is a tolerance.
ATOL = 2e-5

COMPACT_N = (1, 7, 8, 32, 64, 96)          # the batch sizes of the main parity check
EXTRA_N = (2, 3, 24, 40, 100, 128, 200)    # generic paths, lane tails, chunking above 96


def compact_routes(n: int) -> tuple[str, ...]:
    return ("decode",) if n == 1 else ("baseline", "dual")


@lru_cache(maxsize=None)
def _case(name: str):
    return layers.load(name)


@lru_cache(maxsize=None)
def oracle(name: str, n: int) -> np.ndarray:
    layer, case = _case(name)
    return reference.forward(layer, layers.make_input(case, n))


def outputs(name: str, batch_sizes=None) -> dict[str, np.ndarray]:
    """Kernel outputs of one test layer, keyed ``<route>/N<n>``."""
    layer, case = _case(name)
    kernel = CompactKernel(layer)
    return {f"{route}/N{n}": kernel.forward(layers.make_input(case, n), route)
            for n in batch_sizes or COMPACT_N for route in compact_routes(n)}


def check(name: str, batch_sizes=None) -> list[tuple[str, float]]:
    """Return ``(key, max |y - float64|)`` and raise on any violation."""
    rows = []
    for key, y in outputs(name, batch_sizes).items():
        if not np.isfinite(y).all():
            raise AssertionError(f"{name} {key}: non-finite output")
        error = float(np.abs(y - oracle(name, int(key.rsplit("N", 1)[1]))).max())
        rows.append((key, error))
        if error > ATOL:
            raise AssertionError(f"{name} {key}: |y-float64|={error:.3e} > {ATOL:.0e}")
    return rows


def main() -> int:
    failed = 0
    for name in layers.names():
        try:
            rows = check(name)
        except AssertionError as error:
            failed += 1
            print(f"FAIL {name}: {error}")
            continue
        print(f"ok   {name}: {len(rows)} outputs, max |y-float64|={max(r[1] for r in rows):.2e}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
