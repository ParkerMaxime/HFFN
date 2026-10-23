"""Tables of one results directory (written by campaign.sh) into <dir>/summary.md.

    python bench/summarize.py results/<tag>
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

DENSE = ("dense_eager", "dense_prealloc", "dense_compile", "dense_numpy")


def load(directory: Path, name: str):
    path = directory / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def table(header: list[str], rows: list[list]) -> list[str]:
    def cell(value):
        return f"{value:.1f}" if isinstance(value, float) else ("–" if value is None else str(value))
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]


def machine(record: dict) -> list[str]:
    cpu = record.get("lscpu", {})
    torch = record.get("torch", {})
    lines = ["## Machine", "",
             f"- CPU: {cpu.get('Model name', '?')} — {cpu.get('CPU(s)', '?')} CPU(s), {cpu.get('Thread(s) per core', '?')} thread(s)/core, "
             f"{cpu.get('Core(s) per socket', '?')} core(s)/socket, {cpu.get('Socket(s)', '?')} socket(s), NUMA nodes {cpu.get('NUMA node(s)', '?')}",
             f"- Caches: L1d {cpu.get('L1d cache', '?')}, L2 {cpu.get('L2 cache', '?')}, L3 {cpu.get('L3 cache', '?')}; hypervisor: {cpu.get('Hypervisor vendor', 'none')}",
             f"- SIMD flags: {' '.join(record.get('cpu_flags', []))}; governor: {record.get('governor')}",
             f"- Affinity: {record.get('affinity')}; HFFN compact threads: {record.get('hffn_threads', 4)}; dense threads: {record.get('dense_threads')}",
             f"- Compiler: {record.get('compiler')}; HFFN flags: `{' '.join(record.get('hffn_flags', []))}`",
             f"- Python {record.get('python')}, numpy {record.get('numpy')}, torch {torch.get('version')} ({torch.get('cpu_capability')})",
             f"- Platform: {record.get('platform')}", ""]
    return lines


def threads(record: dict) -> list[str]:
    values = record["thread_list"]
    out = ["## Scaling with the thread count T (`threads.py`, recommended route, kernel µs)", "",
           "speedup = T=1 / largest T; fastest = T of the lowest median; worth = smallest T within "
           f"{record['args']['margin']:.0%} of the fastest.", ""]
    grid = defaultdict(dict)
    for r in record["rows"]:
        grid[(r["case"], r["n"], r["route"])][r["threads"]] = r
    rows = []
    for (case, n, route), v in grid.items():
        fastest = next(t for t, r in v.items() if r["fastest"])
        worth = next(t for t, r in v.items() if r["worth"])
        top = v[max(values)]
        rows.append([case, n, route, *[v[t]["kernel_us"] for t in values],
                     f"{top['speedup_vs_1']:.2f}" if top["speedup_vs_1"] else None, fastest, worth])
    return out + table(["case", "N", "route", *[f"T={t}" for t in values], f"speedup T={max(values)}", "fastest", "worth"], rows) + [""]


def routes(record: dict) -> list[str]:
    out = ["## Routes baseline vs dual (`routes.py`, kernel µs)", ""]
    out += table(["case", "N", "baseline", "dual", "dual/baseline", "best"],
                 [[r["case"], r["n"], r["baseline_us"], r["dual_us"], f"{r['dual_over_baseline']:.2f}", r["best"]] for r in record["rows"]])
    return out + [""]


def vs_dense(record: dict) -> list[str]:
    out = ["## HFFN against dense (`vs_dense.py --phase perf`, wall µs per call, same Python entry point)", "",
           f"Threads: hffn_compact {record.get('hffn_threads', 4)}, dense {record.get('dense_threads')}. "
           "best dense = fastest of the dense methods; speedup = best dense / hffn_compact.", ""]
    for kind, title in (("hot", "Hot cache"), ("cold", "Cold cache (640 MB rewritten before each call)")):
        grid = defaultdict(dict)
        for r in record["rows"]:
            if r["kind"] == kind:
                grid[(r["case"], r["n"])][r["method"]] = r["median_us"]
        if not grid:
            continue
        methods = ["hffn_compact", *DENSE]
        rows = []
        for (case, n), v in grid.items():
            dense = {m: v[m] for m in DENSE if m in v}
            best = min(dense, key=dense.get) if dense else None
            rows.append([case, n, *[v.get(m) for m in methods], best, f"{dense[best] / v['hffn_compact']:.2f}" if best else None])
        out += [f"### {title}", ""] + table(["case", "N", *methods, "best dense", "speedup"], rows) + [""]
    return out


def vs_dense_memory(record: dict) -> list[str]:
    out = ["## Memory against dense (`vs_dense.py --phase memory`)", "",
           "model = weights/layout bytes; held = work buffers kept between calls; peak = VmHWM increase during one steady call; "
           "touched = pages referenced by one call.", ""]
    rows = []
    for r in record["rows"]:
        if r["kind"] == "memory" and r["n"] in (1, 32, 128):
            held = r["held_bytes"]
            rows.append([r["case"], r["n"], r["method"], f"{r['model_bytes'] / 1024:.0f}", None if held is None else f"{held / 1024:.0f}",
                         r["call_peak_kb"], f"{r['touched_kb']:.0f}"])
    return out + table(["case", "N", "method", "model kB", "held kB", "call peak kB", "touched kB"], rows) + [""]


def memory(record: dict) -> list[str]:
    out = ["## HFFN runtime memory (`memory.py`)", "",
           "scenario = N 1, 8, 32, 7, 96, 24, 64, 40, 1 on one loader (recommended routes).", ""]
    rows = [[r["case"], r["kernel"], r["route"], r["n"], f"{r['model_bytes'] / 1024:.0f}",
             f"{r['workspace_bytes'] / 1024:.0f}" if "workspace_bytes" in r else None, r["first_call_peak_kb"],
             f"{r['touched_kb']:.0f}" if "touched_kb" in r else None]
            for r in record["rows"] if r["route"] in ("scenario", "decode") or r["n"] in (32, 96)]
    return out + table(["case", "kernel", "route", "N", "model kB", "held kB", "first-call peak kB", "touched kB"], rows) + [""]


def footprint(record: dict) -> list[str]:
    out = ["## Whole-process footprint, homog_512_2048, N = 1..128 (`process_footprint.py`)", ""]
    return out + table(["method", "VmHWM MB", "VmRSS MB", "PyTorch imported"],
                       [[r["method"], r["vm_hwm_kb"] / 1024, r["vm_rss_kb"] / 1024, r["torch_imported"]] for r in record["rows"]]) + [""]


def main() -> None:
    directory = Path(sys.argv[1])
    lines = [f"# Results `{directory.name}`", ""]
    records = {name: load(directory, name) for name in ("threads", "routes", "vs_dense_perf", "vs_dense_memory", "memory", "footprint")}
    first = next((r for r in records.values() if r), None)
    if first:
        lines += machine(first)
    for name, render in (("threads", threads), ("routes", routes), ("vs_dense_perf", vs_dense), ("vs_dense_memory", vs_dense_memory),
                         ("memory", memory), ("footprint", footprint)):
        if records[name]:
            lines += render(records[name])
    (directory / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {directory / 'summary.md'}")


if __name__ == "__main__":
    main()
