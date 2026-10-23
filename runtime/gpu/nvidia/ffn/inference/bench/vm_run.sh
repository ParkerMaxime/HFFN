#!/usr/bin/env bash
# Isolated-FFN timing on a CUDA machine (python of the PATH: the CUDA environment), from the folder of this script.
# Usage: BASE_COMMIT=<commit> vm_run.sh smoke|run
# Writes runs/<mode>_<UTC>_<pid>/: run.log, EXIT, config.json, provenance.json, manifest.json, raw.jsonl, summary.json; base.txt holds the base commit given in BASE_COMMIT.
set -uo pipefail
mode="${1:-}"
case "$mode" in smoke|run) ;; *) echo "usage: vm_run.sh smoke|run" >&2; exit 2 ;; esac
capsule=$(cd "$(dirname "$0")" && pwd)
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
out="$capsule/runs/${mode}_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "$out"
echo "${BASE_COMMIT:-unknown}" > "$out/base.txt"
python -B "$capsule/bench_ffn.py" --mode "$mode" --out "$out" 2>&1 | tee "$out/run.log"
code=${PIPESTATUS[0]}
echo "$code" > "$out/EXIT"
echo "EXIT=$code  $out"
exit "$code"
