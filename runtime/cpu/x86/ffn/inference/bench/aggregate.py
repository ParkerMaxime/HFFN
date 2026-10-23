"""Status manifest and cross-session tables of one campaign directory (written by campaign.sh).

    python bench/aggregate.py results/<tag> --mode run --exit-code 0 --started <UTC> --threads 8

Writes ``manifest.json`` (status, steps, source, CPU steal) and, when session results exist,
``AGGREGATE.md``: every ratio as the median over the sessions with [min - max]. A session is
one process; ratios are taken inside a process and never across campaigns. Beside the best of the dense
arms (an envelope chosen per cell), every table gives the fixed arm ``dense_eager`` / hffn_compact, so that
a figure is also read against one dense definition.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hffn import provenance  # noqa: E402

DENSE = ("dense_eager", "dense_prealloc", "dense_compile", "dense_numpy")
FIXED = "dense_eager"  # the single fixed dense arm reported beside the envelope
STABLE_SPREAD = 0.10  # (max - min) / median of a ratio over the sessions


def steal_fraction(directory: Path) -> float | None:
    try:
        first, last = ([int(v) for v in (directory / f"proc_stat_{k}.txt").read_text().split()[1:9]] for k in ("start", "end"))
    except (OSError, ValueError):
        return None
    delta = [b - a for a, b in zip(first, last)]
    return delta[7] / sum(delta) if sum(delta) else None


def speedups(path: Path, kind: str) -> dict:
    """(case, n) -> (best dense / hffn_compact, hffn µs, best dense method, dense_eager / hffn_compact or None)."""
    grid = defaultdict(dict)
    for row in json.loads(path.read_text())["rows"]:
        if row["kind"] == kind:
            grid[(row["case"], row["n"])][row["method"]] = row["median_us"]
    result = {}
    for key, methods in grid.items():
        dense = {m: methods[m] for m in DENSE if m in methods}
        if dense and "hffn_compact" in methods:
            best = min(dense, key=dense.get)
            fixed = methods[FIXED] / methods["hffn_compact"] if FIXED in methods else None
            result[key] = (dense[best] / methods["hffn_compact"], methods["hffn_compact"], best, fixed)
    return result


def spread(values: list[float]) -> str:
    return f"{statistics.median(values):.2f} [{min(values):.2f}–{max(values):.2f}]"


def table(header: list[str], rows: list[list]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(str(v) for v in row) + " |" for row in rows] + [""]


def sessions_table(sessions: list[dict], title: str) -> tuple[list[str], dict]:
    """Table of one cache regime over the sessions, and (case, n) -> list of speedups."""
    keys = [key for key in sessions[0] if all(key in s for s in sessions)]
    rows, ratios = [], {}
    for key in keys:
        values = [s[key][0] for s in sessions]
        ratios[key] = values
        stable = (max(values) - min(values)) / statistics.median(values) <= STABLE_SPREAD
        best = sorted({s[key][2] for s in sessions})
        fixed = [s[key][3] for s in sessions]
        rows.append([key[0], key[1], f"{statistics.median(s[key][1] for s in sessions):.1f}", spread(values), ", ".join(best),
                     "yes" if min(values) > 1 else "no", "yes" if stable else "NO", "—" if None in fixed else spread(fixed)])
    header = ["case", "N", "hffn_compact µs", "best dense / hffn", "best dense", "min > 1", "stable", f"{FIXED} / hffn"]
    return [f"### {title}", ""] + table(header, rows), ratios


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--mode", default="run")
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--started", default=None)
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()
    directory = args.directory

    steps = []
    for line in (directory / "steps.tsv").read_text().splitlines():
        name, code, seconds = line.split("\t")
        steps.append({"name": name, "exit_code": int(code), "seconds": int(seconds)})
    status = "passed" if args.exit_code == 0 and steps and all(s["exit_code"] == 0 for s in steps) else ("partial" if args.exit_code == 124 else "failed")
    session_dirs = sorted(p for p in directory.glob("s[0-9]*") if (p / "vs_dense_perf.json").exists())
    manifest = {"tag": directory.name, "mode": args.mode, "status": status, "exit_code": args.exit_code, "started_utc": args.started,
                "hffn_threads": args.threads, "sessions": len(session_dirs), "source": provenance.record(),
                "cpu_steal_fraction": steal_fraction(directory), "seconds": sum(s["seconds"] for s in steps), "steps": steps}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest: status={status} exit_code={args.exit_code} steps={len(steps)} sessions={len(session_dirs)}")
    if status != "passed" or not session_dirs:
        return  # a failed or partial campaign yields no performance table

    lines = [f"# Campaign `{directory.name}` ({args.mode})", "",
             f"Status **{status}**; {len(session_dirs)} session(s), one process each; HFFN compact on {args.threads} threads; "
             f"CPU steal {manifest['cpu_steal_fraction']:.4%}." if manifest["cpu_steal_fraction"] is not None else f"Status **{status}**.",
             "", "Ratios are best dense / hffn_compact (wall time, same Python entry point), above 1 when HFFN is faster: "
             "median over the sessions [min–max]. `stable` = (max − min) / median ≤ 10 %. "
             f"`{FIXED} / hffn` is the same ratio against the single fixed dense arm `{FIXED}`.", ""]
    verdict, hot = [], {}
    for kind, title in (("hot", f"Hot cache, dense on {args.threads} threads"), ("cold", f"Cold cache, dense on {args.threads} threads")):
        data = [speedups(p / "vs_dense_perf.json", kind) for p in session_dirs]
        if not data[0]:
            continue
        block, ratios = sessions_table(data, title)
        lines += block
        verdict.append(f"- {title}: {sum(min(v) > 1 for v in ratios.values())} of {len(ratios)} cells with every session above 1; "
                       f"{sum((max(v) - min(v)) / statistics.median(v) > STABLE_SPREAD for v in ratios.values())} unstable.")
        if kind == "hot":
            hot = ratios

    others = {name: speedups(directory / name / "vs_dense_perf.json", "hot") for name in ("dense4", "dense1") if (directory / name / "vs_dense_perf.json").exists()}
    if others:
        rows = []
        for key, values in hot.items():
            row = [statistics.median(values)] + [others[name][key][0] for name in others if key in others[name]]
            rows.append([key[0], key[1], *[f"{v:.2f}" for v in row], f"{min(row):.2f}"])
        lines += ["### Dense thread count (hot cache, HFFN unchanged)", "",
                  "One process per dense thread count: the last column is the lowest ratio, i.e. HFFN against the dense thread count that suits dense best.", ""]
        lines += table(["case", "N", f"dense on {args.threads} (median)", *[f"dense on {name[5:]}" for name in others], "lowest"], rows)
        verdict.append(f"- Lowest ratio over the dense thread counts: {sum(float(r[-1]) > 1 for r in rows)} of {len(rows)} cells above 1 (single process for the reduced thread counts).")

    t4 = directory / "t4" / "vs_dense_perf.json"
    if t4.exists():
        lines += ["### HFFN on 4 threads against dense on 4 threads (hot cache, one process)", ""]
        lines += table(["case", "N", "hffn_compact µs", "best dense / hffn", "best dense", f"{FIXED} / hffn"],
                       [[c, n, f"{v[1]:.1f}", f"{v[0]:.2f}", v[2], "—" if v[3] is None else f"{v[3]:.2f}"] for (c, n), v in speedups(t4, "hot").items()])

    scaling = defaultdict(list)
    for p in session_dirs:
        if (p / "threads.json").exists():
            record = json.loads((p / "threads.json").read_text())
            top = max(record["thread_list"])
            for row in record["rows"]:
                if row["threads"] == top and row["speedup_vs_1"]:
                    scaling[(row["case"], row["n"], top)].append(row["speedup_vs_1"])
    if scaling:
        lines += ["### Scaling of the compact kernel (kernel time, T = 1 / largest T)", ""]
        lines += table(["case", "N", "T", "speedup"], [[c, n, t, spread(v)] for (c, n, t), v in scaling.items()])

    lines += ["## Counts", "", *verdict, ""]
    (directory / "AGGREGATE.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {directory / 'AGGREGATE.md'}")


if __name__ == "__main__":
    main()
