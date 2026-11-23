"""The one forward route of the M1 CPU compact G4 kernel: a deterministic function of the token count and of the outputs the caller asks of the entry.

Hardware is the M1 CPU (the leaf that holds this file). A consumer never picks a route: it asks for outputs, and the same (n, outputs) takes the
same route whatever the consumer. Thresholds live here. ``STORES_N`` is the one N at which the stores entry is
taken for {y, z, h}: the route the LLM training leaf took before the routes were unified."""
from __future__ import annotations

DECODE, FUSED, STORES, DUAL = "decode", "fused", "stores", "dual"
OUTPUTS_Y = frozenset({"y"})
OUTPUTS_YZH = frozenset({"y", "z", "h"})
STORES_N = 8  # stores for outputs {y, z, h} at exactly this N, dual at every other N; N = 2..7 is not a stores range (written limit, no timed cell)


def route(n: int, outputs: frozenset[str], stores_range: tuple[int, int] | None = None) -> str:
    """``stores_range`` = (lowest, highest) N of the stores entry; a kernel passes the one it sized its workspace with (production: (STORES_N, STORES_N); the labelled
    test switches of a harness may widen it, never a production entry)."""
    lo, hi = (STORES_N, STORES_N) if stores_range is None else stores_range
    if n < 1:
        raise ValueError("route expects n >= 1")
    if n == 1:
        return DECODE if outputs == OUTPUTS_Y else DUAL  # the dual entry has no branch on N and stores z and h at N = 1 (dual_entry.INFERENCE_MIN_N)
    if outputs == OUTPUTS_Y:
        return FUSED
    if outputs == OUTPUTS_YZH:
        return STORES if lo <= n <= hi else DUAL
    raise ValueError(f"unsupported outputs {sorted(outputs)}")
