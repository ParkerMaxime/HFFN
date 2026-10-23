"""Scaling of the compact kernel with its thread count T (recommended route, kernel time).

For each test layer and N, one loader per T (all share the process-wide pool); the T values
alternate block by block in a rotating order, with a pause longer than the pool's spin
bound, and the first calls of each block are dropped. Reported per T: kernel and wall time
(median), speedup over T = 1, parallel efficiency; per (case, N): the fastest T and the
smallest T within ``--margin`` of it (the thread count worth paying for).

    python bench/threads.py --cpus 0-7 --out results/<tag>/threads.json
    python bench/threads.py --cpus 0-7 --thread-list 1,2,4,6,8
"""
from __future__ import annotations

import common  # noqa: F401  (first: pinning and thread environment)

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np

from hffn import checks
from hffn.compact import CompactKernel, recommended_route
from hffn_ffn_layers import layers

DEFAULT_N = (1, 2, 8, 24, 32, 64, 96, 128)
BLOCK_PAUSE_S = 0.002
BLOCK_WARMUP = 3
BLOCK_WARMUP_S = 0.005


def default_thread_list() -> list[int]:
    cpus = len(os.sched_getaffinity(0))
    values = {1, cpus} | {t for t in (2, 4, 6, 8, 12, 16) if t < cpus}
    return sorted(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_arguments(parser)
    parser.add_argument("--thread-list", default=None, help="comma-separated T values (default: 1, 2, 4, ... up to the affinity CPUs)")
    parser.add_argument("--case", action="append", choices=layers.names())
    parser.add_argument("--n", type=int, action="append")
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--margin", type=float, default=0.05, help="'worth' T: smallest T within this fraction of the fastest")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    thread_list = [int(t) for t in args.thread_list.split(",")] if args.thread_list else default_thread_list()

    rows = []
    for name in args.case or layers.names():
        layer, case = layers.load(name)
        kernels = {t: CompactKernel(layer, threads=t) for t in thread_list}
        for n in args.n or DEFAULT_N:
            x = layers.make_input(case, n)
            route = recommended_route(n)
            for t, kernel in kernels.items():  # parity before timing
                error = float(np.abs(kernel.forward(x) - checks.oracle(name, n)).max())
                assert error <= checks.ATOL, f"{name} T={t} N{n}: |y-float64|={error:.2e}"
            kernel_us = {t: [] for t in thread_list}
            wall_us = {t: [] for t in thread_list}
            for sample in range(args.samples):
                shift = sample % len(thread_list)
                for t in thread_list[shift:] + thread_list[:shift]:
                    time.sleep(BLOCK_PAUSE_S)
                    # Warm up for a duration: during a block on fewer threads the other cores went
                    # idle, and a core leaving idle runs about twice slower for 1-2 ms.
                    warm_start, count = time.perf_counter(), 0
                    while count < BLOCK_WARMUP or time.perf_counter() - warm_start < BLOCK_WARMUP_S:
                        kernels[t].forward(x)
                        count += 1
                    for _ in range(args.repeat):
                        start = time.perf_counter_ns()
                        kernels[t].forward(x)
                        wall_us[t].append((time.perf_counter_ns() - start) / 1e3)
                        kernel_us[t].append(kernels[t].kernel_ns(n).sum() / 1e3)
            median = {t: statistics.median(values) for t, values in kernel_us.items()}
            base = median.get(1)  # speedups need T = 1 in the list
            fastest = min(median, key=median.get)
            worth = min(t for t in thread_list if median[t] <= median[fastest] * (1 + args.margin))
            for t in thread_list:
                rows.append({"case": name, "n": n, "route": route, "threads": t,
                             "kernel_us": median[t], "kernel_p10_us": common.percentile(kernel_us[t], 0.1),
                             "kernel_p90_us": common.percentile(kernel_us[t], 0.9), "wall_us": statistics.median(wall_us[t]),
                             "speedup_vs_1": base / median[t] if base else None,
                             "efficiency": base / median[t] / t if base else None,
                             "fastest": t == fastest, "worth": t == worth})
            print(f"{name:15} N{n:<4} {route:7} " + "  ".join(f"T{t} {median[t]:7.1f}" for t in thread_list)
                  + f"   fastest T{fastest}  worth T{worth}", flush=True)

    if args.out:
        record = {"bench": "threads", **common.machine(), "thread_list": thread_list,
                  "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, "rows": rows}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
