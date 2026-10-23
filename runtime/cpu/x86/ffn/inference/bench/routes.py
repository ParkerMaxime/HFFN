"""Route choice per batch size: ``baseline`` against ``dual`` on the x86 kernel.

Both routes of one kernel alternate call by call (rotating order) and compare on the time
measured inside the kernel. ``hffn.compact.recommended_route`` uses ``dual`` for every N >= 2
(it won or tied everywhere on the cloud VM); re-run this on a new machine before trusting it.

    python bench/routes.py --cpus 0-3 --out results/<tag>/routes.json
"""
from __future__ import annotations

import common  # noqa: F401

import argparse
import json
import statistics
from pathlib import Path

import hffn  # noqa: F401  (puts the shared test layers on sys.path)
from hffn_ffn_layers import layers

ROUTES = ("baseline", "dual")
DEFAULT_N = (2, 7, 8, 24, 32, 40, 64, 96)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_arguments(parser)
    parser.add_argument("--n", type=int, action="append")
    parser.add_argument("--calls", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--margin", type=float, default=0.05, help="below this relative gap, report 'equal'")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for name in layers.names():
        layer, case = layers.load(name)
        kernel = common.compact_kernel(layer)
        for n in args.n or DEFAULT_N:
            x = layers.make_input(case, n)
            times = {route: [] for route in ROUTES}
            for index in range(-args.warmup, args.calls):
                for route in (ROUTES if index % 2 == 0 else ROUTES[::-1]):
                    kernel.forward(x, route)
                    if index >= 0:
                        times[route].append(kernel.kernel_ns(n).sum() / 1e3)
            median = {route: statistics.median(values) for route, values in times.items()}
            ratio = median["dual"] / median["baseline"]
            best = "equal" if abs(ratio - 1) < args.margin else ("dual" if ratio < 1 else "baseline")
            rows.append({"case": name, "n": n, "baseline_us": median["baseline"], "dual_us": median["dual"], "dual_over_baseline": ratio, "best": best})
            print(f"{name:16} N{n:<4} baseline {median['baseline']:8.1f} us  dual {median['dual']:8.1f} us  dual/baseline {ratio:5.2f}  -> {best}", flush=True)

    if args.out:
        record = {"bench": "routes", **common.machine(), "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, "rows": rows}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
