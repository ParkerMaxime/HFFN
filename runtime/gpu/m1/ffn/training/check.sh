#!/usr/bin/env zsh
# Untimed every-case check of the GPU M1 isolated-FFN training leaf (gates: check_leaf.py docstring).
# Usage: zsh check.sh check
set -uo pipefail
[[ ${1:-} == check ]] || { echo "usage: zsh check.sh check" >&2; exit 2; }
leaf=${0:A:h}
py=${HFFN_PYTHON:-python}
unset PYTORCH_ENABLE_MPS_FALLBACK
run=$leaf/runs/check_$(date -u +%Y%m%dT%H%M%SZ)_$$
mkdir -p "$leaf/runs"
KMP_DUPLICATE_LIB_OK=TRUE PYTHONUNBUFFERED=1 "$py" -B "$leaf/check_leaf.py" --out "$run" 2>&1 | tee "$run.log"
code=${pipestatus[1]}
mv "$run.log" "$run/run.log"
echo "$code" > "$run/EXIT"
echo "run: $run EXIT=$code"
exit "$code"
