"""Runs the independent gates of a check, several at a time, without timing anything.

    run_gates.py [--limits <prefix>=<jobs>,...] < plan
    plan: one line per job, `<name><TAB><shell command>`; limits default to $HFFN_CHECK_LIMITS

A job takes the limit of the first prefix its name starts with; a name that matches none runs alone. Consecutive jobs of
the same prefix form a tier, and tiers run one after the other, so a plan orders its memory peaks. Each command runs
under `bash -o pipefail`, in the working directory of this process. A failed job does not stop the others. The output of
each job is printed whole and in the order of the plan. Exit code 1 if any job failed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import groupby
import os
import subprocess
import sys


def parse_limits(text):
    limits = {}
    for item in filter(None, text.split(",")):
        prefix, jobs = item.split("=")
        if int(jobs) < 1:
            raise ValueError(f"limit below 1: {item}")
        limits[prefix] = int(jobs)
    return limits


def execute(job):
    name, command = job
    print(f"start {name}", flush=True)
    done = subprocess.run(["bash", "-o", "pipefail", "-c", command], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, errors="replace")
    return name, done.returncode, done.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limits", default=os.environ.get("HFFN_CHECK_LIMITS", ""))
    args = parser.parse_args()
    limits = parse_limits(args.limits)
    jobs = [tuple(line.rstrip("\n").split("\t", 1)) for line in sys.stdin if line.strip()]
    failed = []
    key = lambda job: next((prefix for prefix in limits if job[0].startswith(prefix)), None)
    for prefix, tier in groupby(jobs, key):
        with ThreadPoolExecutor(limits.get(prefix, 1)) as pool:
            for name, code, output in pool.map(execute, list(tier)):
                print(f"== {name}\n{output}", end="" if output.endswith("\n") else "\n", flush=True)
                if code:
                    failed.append(name)
                    print(f"FAIL {name} (exit code {code})", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
