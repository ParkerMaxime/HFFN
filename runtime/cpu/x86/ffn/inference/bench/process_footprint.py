"""Whole-process memory of one method serving N = 1..128 (fresh process per method).

    python bench/process_footprint.py hffn_compact homog_512_2048
    python bench/process_footprint.py --all --out results/<tag>/footprint.json

Prints VmHWM / VmRSS and the large shared libraries mapped. hffn_* and dense_numpy must
run without PyTorch: the script checks that it was never imported.
"""
from __future__ import annotations

import common  # noqa: F401

import argparse
import json
import subprocess
import sys
from pathlib import Path

import hffn  # noqa: F401  (puts the shared test layers on sys.path)
from hffn_ffn_layers import layers
from methods import METHODS

N_ALL = (1, 2, 4, 7, 8, 16, 24, 32, 40, 64, 96, 128)


def status_kb(key: str) -> int:
    return next(int(line.split()[1]) for line in open("/proc/self/status") if line.startswith(key))


def one(method: str, name: str) -> dict:
    layer, case = layers.load(name)
    inputs = [layers.make_input(case, n) for n in N_ALL]
    before = status_kb("VmRSS:")
    m = METHODS[method](layer)
    del layer  # each method keeps what it needs: the layer arrays it does not use are freed
    for _ in range(3):
        for x in inputs:
            m.forward(x)
    libs = sorted({line.split()[-1].rsplit("/", 1)[-1] for line in open("/proc/self/maps")
                   if ".so" in line and any(k in line for k in ("torch", "c10", "hffn", "mkl", "gomp", "openblas", "scipy"))})
    if method.startswith("hffn") or method == "dense_numpy":
        assert "torch" not in sys.modules, "PyTorch was imported"
    return {"method": method, "case": name, "vm_hwm_kb": status_kb("VmHWM:"), "vm_rss_kb": status_kb("VmRSS:"),
            "rss_before_method_kb": before, "torch_imported": "torch" in sys.modules, "libs": libs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_arguments(parser)
    parser.add_argument("method", nargs="?", choices=sorted(METHODS))
    parser.add_argument("case", nargs="?", choices=layers.names())
    parser.add_argument("--all", action="store_true", help="every method on homog_512_2048, one process each")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if not args.all:
        print(json.dumps(one(args.method, args.case)))
        return
    rows = []
    passthrough = [arg for option in ("--cpus", "--threads", "--dense-threads") if getattr(args, option[2:].replace("-", "_"))
                   for arg in (option, str(getattr(args, option[2:].replace("-", "_"))))]
    for method in METHODS:
        output = subprocess.run([sys.executable, __file__, method, "homog_512_2048", *passthrough], capture_output=True, text=True, check=True).stdout
        row = json.loads(output.strip().splitlines()[-1])
        rows.append(row)
        print(f"{method:15} VmHWM {row['vm_hwm_kb'] / 1024:7.1f} MB  VmRSS {row['vm_rss_kb'] / 1024:7.1f} MB  "
              f"torch imported: {row['torch_imported']}  libs: {', '.join(row['libs'])[:100]}", flush=True)
    if args.out:
        record = {"bench": "process_footprint", **common.machine(), "rows": rows}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
