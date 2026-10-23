#!/usr/bin/env bash
# Verification on the Apple GPU (MPS), without timing: layer and one-step model parity on every export.
# Usage: ./check.sh check <label> [export name ...]
#   <label>: short name of this check (written into the run folder name).
#   without export names: every `pythia*_sparse*_ffn_only` export of $HFFN_ASSETS/pythia/exports (HFFN_ASSETS is required).
#   HFFN_PYTHON=<interpreter> when `python` of the PATH is not the environment of the project.
# Settles, per export: (1) `validate_wanda_layer.py` on every layer file (forward and active gradients of the exact-offset
# runtime against the shared float64 oracle, layouts token_major, feature_persistent and feature_owner_tile at N = 1, 8, 64,
# 128, 256, gate 5e-4); (2) `validate_pythia_parity.py` (one training step of the whole model against the fixed-support dense
# model: loss, logits, compact gradients, compact parameters after AdamW, next loss). Times nothing and bounds no memory.
# Output: <leaf>/runs/check_<UTC>_<label>/ with check.log, START, END, EXIT, layer/<export>/layer<K>.json,
# model/<export>.json. EXIT is the exit code.
set -uo pipefail
cd "$(dirname "$0")"
[ "${1:-}" = "check" ] && [ -n "${2:-}" ] || { echo "usage: ./check.sh check <label> [export name ...]" >&2; exit 2; }
[ -z "${PYTORCH_ENABLE_MPS_FALLBACK:-}" ] || { echo "refused: PYTORCH_ENABLE_MPS_FALLBACK is set" >&2; exit 2; }
label="$2"; shift 2
case "$label" in *[!A-Za-z0-9_]*|"") echo "label: letters, digits and _ only" >&2; exit 2;; esac
python=${HFFN_PYTHON:-python}
here=$PWD
[ -n "${HFFN_ASSETS:-}" ] || { echo "HFFN_ASSETS is not set" >&2; exit 2; }
assets=$HFFN_ASSETS/pythia
exports=$assets/exports
models=$assets/models/base
[ -d "$exports" ] && [ -d "$models" ] || { echo "no exports or base models under $assets" >&2; exit 2; }
if [ "$#" -eq 0 ]; then set -- $(cd "$exports" && ls -d pythia*_sparse*_ffn_only | sort); fi
[ "$#" -gt 0 ] || { echo "no export to check" >&2; exit 2; }
run=$here/runs/check_$(date -u +%Y%m%dT%H%MZ)_$label
[ ! -e "$run" ] || { echo "run folder exists: $run" >&2; exit 2; }
mkdir -p "$run/layer" "$run/model" || exit 2
date -u +%Y-%m-%dT%H:%M:%SZ > "$run/START"
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

body() {
    echo "check: $# exports, run $run"
    failed=0; layers_expected=0; layers_passed=0; models_passed=0
    for name in "$@"; do
        mkdir -p "$run/layer/$name"
        for dir in $(cd "$exports/$name/export" && ls -d layer* | sort -V); do
            layers_expected=$((layers_expected + 1))
            if "$python" -B validate_wanda_layer.py --layer "$exports/$name/export/$dir/hffn_training_compact_g4_v2.pt" \
                --layout token_major feature_persistent feature_owner_tile > "$run/layer/$name/$dir.json"; then
                layers_passed=$((layers_passed + 1)); echo "ok   $name $dir"
            else
                failed=1; echo "FAIL $name $dir"
            fi
        done
        if "$python" -B validate_pythia_parity.py --model-dir "$models/${name%%_*}" --export-dir "$exports/$name/export" > "$run/model/$name.json"; then
            models_passed=$((models_passed + 1)); echo "ok   $name model step"
        else
            failed=1; echo "FAIL $name model step"
        fi
    done
    echo "layers: $layers_passed of $layers_expected passed; model steps: $models_passed of $# passed"
    [ "$layers_passed" -eq "$layers_expected" ] && [ "$models_passed" -eq "$#" ] || failed=1
    if [ "$failed" -eq 0 ]; then echo "check: PASSED"; else echo "check: FAILED"; fi
    return "$failed"
}

body "$@" 2>&1 | tee "$run/check.log"
code=${PIPESTATUS[0]}
date -u +%Y-%m-%dT%H:%M:%SZ > "$run/END"
echo "$code" > "$run/EXIT"
exit "$code"
