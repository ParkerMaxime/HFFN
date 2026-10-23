#!/usr/bin/env bash
# Verification on the Apple GPU (MPS), without timing: the layer gate of every layer of every export, every layout.
# Usage: ./check.sh check <label> [export name ...]
#   <label>: short name of this check (written into the run folder name).
#   without export names: every `pythia*_sparse*_ffn_only` export of $HFFN_ASSETS/pythia/exports (HFFN_ASSETS is required).
#   HFFN_PYTHON=<interpreter> when `python` of the PATH is not the environment of the project.
# Settles: for each layer file, the four layouts of `validate_wanda_inference_layer.py` against the shared float64 oracle at
# N = 1, 8, 32, 128, 256, 512 (gate 5e-4, finite). Times nothing and bounds no memory. It does not run a model.
# Output: <leaf>/runs/check_<UTC>_<label>/ with check.log, START, END, EXIT, layer/<export>/layer<K>.json. EXIT is the exit code.
set -uo pipefail
cd "$(dirname "$0")"
[ "${1:-}" = "check" ] && [ -n "${2:-}" ] || { echo "usage: ./check.sh check <label> [export name ...]" >&2; exit 2; }
[ -z "${PYTORCH_ENABLE_MPS_FALLBACK:-}" ] || { echo "refused: PYTORCH_ENABLE_MPS_FALLBACK is set" >&2; exit 2; }
label="$2"; shift 2
case "$label" in *[!A-Za-z0-9_]*|"") echo "label: letters, digits and _ only" >&2; exit 2;; esac
python=${HFFN_PYTHON:-python}
here=$PWD
[ -n "${HFFN_ASSETS:-}" ] || { echo "HFFN_ASSETS is not set" >&2; exit 2; }
exports=$HFFN_ASSETS/pythia/exports
[ -d "$exports" ] || { echo "no exports at $exports" >&2; exit 2; }
if [ "$#" -eq 0 ]; then set -- $(cd "$exports" && ls -d pythia*_sparse*_ffn_only | sort); fi
[ "$#" -gt 0 ] || { echo "no export to check" >&2; exit 2; }
run=$(cd "$here/.." && pwd)/runs/check_$(date -u +%Y%m%dT%H%MZ)_$label
[ ! -e "$run" ] || { echo "run folder exists: $run" >&2; exit 2; }
mkdir -p "$run/layer" || exit 2
date -u +%Y-%m-%dT%H:%M:%SZ > "$run/START"
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

body() {
    echo "check: $# exports, run $run"
    failed=0; expected=0
    for name in "$@"; do
        layers=$(cd "$exports/$name/export" && ls -d layer* | sed 's/^layer//' | sort -n)
        [ -n "$layers" ] || { echo "FAIL $name: no layer"; failed=1; continue; }
        for k in $layers; do
            expected=$((expected + 1))
            file="$exports/$name/export/layer$k/hffn_training_compact_g4_v2.pt"
            if "$python" -B validate_wanda_inference_layer.py --layer "$file" --output "$run/layer/$name/layer$k.json" > /dev/null; then
                echo "ok   $name layer$k"
            else
                echo "FAIL $name layer$k"; failed=1
            fi
        done
    done
    "$python" -I - "$run/layer" "$expected" <<'PY' || failed=1
import json, sys
from pathlib import Path
root, expected = Path(sys.argv[1]), int(sys.argv[2])
files = sorted(root.glob("*/layer*.json"))
passed = worst = 0
for f in files:
    d = json.loads(f.read_text())
    rows = d["rows"]
    ok = d["gate"]["passed"] and rows and all(r["finite"] for r in rows)
    passed += bool(ok)
    worst = max(worst, max(r["max_abs_error"] for r in rows))
print(f"layers: {passed} of {expected} passed ({len(files)} records), max error {worst:.3e}")
sys.exit(0 if passed == expected == len(files) else 1)
PY
    if [ "$failed" -eq 0 ]; then echo "check: PASSED"; else echo "check: FAILED"; fi
    return "$failed"
}

body "$@" 2>&1 | tee "$run/check.log"
code=${PIPESTATUS[0]}
date -u +%Y-%m-%dT%H:%M:%SZ > "$run/END"
echo "$code" > "$run/EXIT"
exit "$code"
