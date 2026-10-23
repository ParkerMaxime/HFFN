"""HFFN against dense on this machine: performance (hot and cold cache) and memory.

Methods: see ``methods.py``. Fairness controls: same input/output layout and dtype; every
method checked against the float64 oracle before timing; the dense references use
``--dense-threads`` threads (default: the HFFN thread count, ``--threads``); methods alternate block by
block (rotating order) with a pause longer than any spin-wait (HFFN pool <= 100 us, OpenMP,
OpenBLAS ~0.1 s); each hot
block first runs untimed calls for at least ``BLOCK_WARMUP`` calls and ``BLOCK_WARMUP_S``
seconds (threads asleep and cores idle after the pause).

Cold cache: before each timed call a 640 MB buffer is rewritten (more than twice any current
L3), as in a multi-layer model whose other layers evict this one.

Memory phase (separate process, fixed mmap threshold, see procfs.py): model bytes, buffers
held between calls, per-call peak RSS, pages touched per call.

    python bench/vs_dense.py --phase perf   --cpus 0-3 --out results/<tag>/vs_dense_perf.json
    python bench/vs_dense.py --phase memory --cpus 0-3 --out results/<tag>/vs_dense_memory.json
"""
from __future__ import annotations

import sys

import common  # noqa: F401
import procfs

if "memory" in sys.argv:
    procfs.fix_mmap_threshold()
elif "MALLOC_MMAP_THRESHOLD_" in procfs.os.environ:
    raise SystemExit("performance phase: unset MALLOC_MMAP_THRESHOLD_")

import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from hffn import checks  # noqa: E402
from hffn_ffn_layers import layers  # noqa: E402
from methods import METHODS  # noqa: E402

N_ALL = (1, 2, 4, 7, 8, 16, 24, 32, 40, 64, 96, 128)
N_COLD = (1, 8, 32, 96, 128)
PAUSE_S = 0.005
BLOCK_WARMUP = 10
BLOCK_WARMUP_S = 0.005


def timed_block(method, x, calls, warmup, flush=None):
    # Hot blocks warm up for a duration, not only a call count: after an idle pause every method
    # runs about twice slower for 1-2 ms (cores leaving idle), which ten calls of a fast method
    # do not cover while ten calls of a slow one do.
    warm_start, count = time.perf_counter(), 0
    while count < warmup or (flush is None and time.perf_counter() - warm_start < BLOCK_WARMUP_S):
        method.forward(x)
        count += 1
    values = []
    for _ in range(calls):
        if flush is not None:
            np.add(flush, 1.0, out=flush)
        start = time.perf_counter_ns()
        method.forward(x)
        values.append((time.perf_counter_ns() - start) / 1e3)
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_arguments(parser)
    parser.add_argument("--phase", choices=("perf", "memory"), default="perf")
    parser.add_argument("--method", action="append", choices=sorted(METHODS), help="default: all")
    parser.add_argument("--case", action="append", choices=layers.names())
    parser.add_argument("--n", type=int, action="append")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--cold-calls", type=int, default=12)
    parser.add_argument("--no-cold", action="store_true")
    parser.add_argument("--ws-repeat", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    common.set_torch_threads()
    perf = args.phase == "perf"
    flush = None if (args.no_cold or not perf) else np.zeros(640 * 2**20 // 4, np.float32)

    rows = []
    for name in args.case or layers.names():
        layer, case = layers.load(name)
        methods = [cls(layer) for key, cls in METHODS.items() if not args.method or key in args.method]
        for n in args.n or N_ALL:
            x = layers.make_input(case, n)
            expected = checks.oracle(name, n)
            active = [m for m in methods if m.supports(n)]
            for m in active:  # parity first (also compiles dense_compile for this shape)
                error = float(np.abs(m.forward(x) - expected).max())
                assert error <= checks.ATOL, f"{name} {m.name} N{n}: |y-float64|={error:.2e}"
                rows.append({"case": name, "n": n, "method": m.name, "kind": "parity", "max_abs_vs_float64": error})
            # The parity calls above ran every method: wait out the longest spin-wait before the first block.
            settle = max(getattr(m, "settle_s", PAUSE_S) for m in active)
            if perf:
                hot = {m.name: [] for m in active}
                for sample in range(args.samples):
                    shift = sample % len(active)
                    for m in active[shift:] + active[:shift]:
                        time.sleep(max(PAUSE_S, settle))
                        hot[m.name] += timed_block(m, x, args.repeat, BLOCK_WARMUP)
                        settle = getattr(m, "settle_s", PAUSE_S)
                for m in active:
                    values = hot[m.name]
                    rows.append({"case": name, "n": n, "method": m.name, "kind": "hot", "median_us": statistics.median(values),
                                 "p10_us": common.percentile(values, 0.1), "p90_us": common.percentile(values, 0.9)})
                print(f"{name:15} N{n:<4} " + "  ".join(f"{m.name} {statistics.median(hot[m.name]):8.1f}" for m in active), flush=True)
            if flush is not None and n in N_COLD:
                cold = {m.name: [] for m in active}
                for sample in range(2):
                    for m in (active if sample == 0 else active[::-1]):
                        time.sleep(max(PAUSE_S, settle))
                        cold[m.name] += timed_block(m, x, args.cold_calls // 2, BLOCK_WARMUP, flush)
                        settle = getattr(m, "settle_s", PAUSE_S)
                for m in active:
                    rows.append({"case": name, "n": n, "method": m.name, "kind": "cold", "median_us": statistics.median(cold[m.name])})
            if not perf:
                for m in active:
                    m.forward(x)
                    start = procfs.reset_peak()
                    m.forward(x)
                    steady = procfs.peak_delta_kb(start)
                    touched = procfs.touched_kb(lambda: m.forward(x), args.ws_repeat)
                    rows.append({"case": name, "n": n, "method": m.name, "kind": "memory", "model_bytes": m.model_bytes,
                                 "held_bytes": m.held_bytes(), "call_peak_kb": steady, "touched_kb": touched})
                print(f"{name:15} N{n:<4} memory done", flush=True)
        del methods
        gc.collect()

    if args.out:
        record = {"bench": f"vs_dense_{args.phase}", **common.machine(),
                  "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, "rows": rows}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
