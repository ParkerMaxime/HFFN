#!/usr/bin/env bash
# All 18 exports and the shared synthetic layers: numerical gates and shared APIs. Needs HFFN_ASSETS.
set -uo pipefail
leaf=$(cd "$(dirname "$0")" && pwd)
cpu=$(cd "$leaf/../../.." && pwd)    # runtime/cpu
hardware=m1
mode=${1:-check}; seconds=7200; batch_mode=run
case "$mode" in check) ;; smoke) seconds=600; batch_mode=smoke;; *) exit 2;; esac
[ $# -le 1 ] || exit 2
py=${HFFN_PYTHON:-python}
[ -n "${HFFN_ASSETS:-}" ] || { echo "HFFN_ASSETS must name the assets folder"; exit 2; }
[ -z "${PYTORCH_ENABLE_MPS_FALLBACK:-}" ] || { echo 'MPS fallback is forbidden'; exit 2; }
checks=$cpu/common/ffn/training/checks
out=$leaf/runs/${mode}_$(date -u +%Y%m%dT%H%M%SZ)_$$
[ ! -e "$out" ] || exit 2
mkdir -p "$out"
export HFFN_LEAF="$leaf" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
HFFN_LAUNCHER_PID=$$ HFFN_QUAL_STARTED_EPOCH=$(date +%s) "$py" -B "$checks/supervise.py" --seconds "$seconds" --out "$out" -- \
 "$py" -B "$checks/batch.py" --mode "$batch_mode" --hardware "$hardware" --assets "$HFFN_ASSETS" --out "$out" 2>&1 | tee "$out/check.log" &
logger=$!
trap 'touch "$out/STOP"; wait "$logger"; exit 130' INT TERM HUP
wait "$logger"; logger_code=$?
trap - INT TERM HUP
code=1; [ ! -f "$out/EXIT" ] || code=$(cat "$out/EXIT")
[ "$code" -ne 0 ] || [ "$logger_code" -eq 0 ] || code=$logger_code
printf 'EXIT=%s %s\n' "$code" "$out"
exit "$code"
