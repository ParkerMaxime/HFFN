"""Memory of the HFFN runtime: model, work buffers held, per-call peak RSS, pages touched.

For each test layer, route and N, on a fresh loader: first-call peak (buffers allocated),
steady-call peak, pages touched by a call. Then a serving scenario (the batch sizes a
runtime sees one after another, on the recommended routes) on one loader: buffers held at
the end and peak over the scenario.
Re-executes itself with a fixed glibc mmap threshold (see procfs.py).

    python bench/memory.py --cpus 0-3 --out results/<tag>/memory.json
"""
from __future__ import annotations

import common  # noqa: F401
import procfs

procfs.fix_mmap_threshold()

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

from hffn import checks  # noqa: E402
from hffn_ffn_layers import layers  # noqa: E402
from methods import HffnCompact  # noqa: E402

COMPACT_N = (1, 7, 8, 24, 32, 40, 64, 96, 128)
SCENARIO = (1, 8, 32, 7, 96, 24, 64, 40, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_arguments(parser)
    parser.add_argument("--ws-repeat", type=int, default=7)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for name in layers.names():
        layer, case = layers.load(name)
        model = HffnCompact(layer).model_bytes
        for n in COMPACT_N:
            x = layers.make_input(case, n)
            for route in checks.compact_routes(n):
                kernel = common.compact_kernel(layer)
                start = procfs.reset_peak()
                kernel.forward(x, route)
                first = procfs.peak_delta_kb(start)
                start = procfs.reset_peak()
                kernel.forward(x, route)
                steady = procfs.peak_delta_kb(start)
                touched = procfs.touched_kb(lambda: kernel.forward(x, route), args.ws_repeat)
                rows.append({"case": name, "kernel": "compact", "route": route, "n": n, "threads": kernel.threads, "model_bytes": kernel.model_bytes(route),
                             "workspace_bytes": kernel.workspace_bytes(), "first_call_peak_kb": first,
                             "steady_call_peak_kb": steady, "touched_kb": touched})
                del kernel
        kernel = common.compact_kernel(layer, hidden_owned=False)
        inputs = [layers.make_input(case, n) for n in SCENARIO]
        start = procfs.reset_peak()
        for x in inputs:
            kernel.forward(x)
        rows.append({"case": name, "kernel": "compact", "route": "scenario", "n": max(SCENARIO), "model_bytes": model,
                     "workspace_bytes": kernel.workspace_bytes(), "first_call_peak_kb": procfs.peak_delta_kb(start)})
        del kernel
    for row in rows:
        held = f"held {row['workspace_bytes'] / 1024:7.0f} kB" if "workspace_bytes" in row else " " * 15
        touched = f"touched {row['touched_kb']:6.0f} kB" if "touched_kb" in row else ""
        print(f"{row['case']:15} {row['kernel']:7} {row['route']:8} N{row['n']:<4} model {row['model_bytes'] / 1024:6.0f} kB  {held}  "
              f"first-call peak {row['first_call_peak_kb']:6d} kB  {touched}")

    if args.out:
        record = {"bench": "memory", **common.machine(), "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, "rows": rows}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
