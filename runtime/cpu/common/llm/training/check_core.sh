#!/usr/bin/env bash
# The layer, model and update gates of a CPU LLM training leaf, without timing. Called by the check.sh of a leaf.
# Usage: check_core.sh <leaf> <base checkpoints dir> <exports dir> [export name ...]
#   <base checkpoints dir>/pythia70m, ...;  <exports dir>/<model>_sparse<density>_ffn_only/export
#   without export names: every export of <exports dir>. HFFN_CHECK_OUT=<dir> keeps the JSON of each check.
#   HFFN_CHECK_LIMITS=<prefix>=<jobs>,...: gates run at the same time (common/tools/run_gates.py); unset, one at a time.
# The prompt tokens are those of the shared Pythia level (first ancestor that has common/pythia/hffn_pythia).
# Smallest model first. The first export runs alone, since it builds the native libraries.
set -uo pipefail
python=${HFFN_PYTHON:-python}
HFFN_LEAF=$(cd "$1" && pwd); export HFFN_LEAF
models="$2"; exports="$3"; shift 3
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
out=${HFFN_CHECK_OUT:-}
tokens=$("$python" -B -c "
from pathlib import Path
print(next(p / 'common/pythia/hffn_pythia/wikitext2_test_tokens.json' for p in Path.cwd().parents if (p / 'common/pythia/hffn_pythia/wikitext2_test_tokens.json').is_file()))") || { echo "prompt tokens of the shared Pythia level not found" >&2; exit 1; }
runner=$("$python" -B -c "
from pathlib import Path
print(next(p / 'common/tools/run_gates.py' for p in Path.cwd().parents if (p / 'common/tools/run_gates.py').is_file()))") || { echo "common/tools/run_gates.py not found" >&2; exit 1; }
if [ "$#" -eq 0 ]; then set -- $(cd "$exports" && ls -d */export | cut -d/ -f1); fi
ordered=$(printf '%s\n' "$@" | sed -E 's/^pythia([0-9]+)m/\1 &/' | sort -n -s | cut -d' ' -f2)

layer_job() {
    printf '%s\t%s\n' "$1" "$(printf '%q ' "$python" -B checks/parity_layer.py --export-dir "$exports/$2/export" ${out:+--out "$out/layer/$2.json"})"
}
gates_job() {
    printf '%s\t%s 2>&1 | grep -v "Loading weights"\n' "$1" "$(printf '%q ' "$python" -B checks/parity_gates.py --model-dir "$models/${2%%_*}" --export-dir "$exports/$2/export" --tokens "$tokens" \
        ${out:+--out-model "$out/model/$2.json" --out-update "$out/update/$2.json"})"
}
plan() {
    first=${ordered%%$'\n'*}
    layer_job "first/layer/$first" "$first"
    gates_job "first/gates/$first" "$first"
    for model in $(printf '%s\n' "$ordered" | sed 's/_.*//' | uniq); do
        for name in $ordered; do [ "${name%%_*}" != "$model" ] || [ "$name" = "$first" ] || layer_job "layer/$name" "$name"; done
        for name in $ordered; do [ "${name%%_*}" != "$model" ] || [ "$name" = "$first" ] || gates_job "gates/$name" "$name"; done
    done
}
plan | "$python" -I "$runner"
failed=$?
if [ "$failed" -eq 0 ]; then echo "check: PASSED"; else echo "check: FAILED"; fi
exit "$failed"
