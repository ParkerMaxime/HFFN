#!/usr/bin/env bash
# The layer, model and update gates of a CPU LLM training leaf, without timing. Called by the check.sh of a leaf.
# Usage: check_core.sh <leaf> <base checkpoints dir> <exports dir> [export name ...]
#   <base checkpoints dir>/pythia70m, ...;  <exports dir>/<model>_sparse<density>_ffn_only/export
#   without export names: every export of <exports dir>. HFFN_CHECK_OUT=<dir> keeps the JSON of each check.
# The prompt tokens are those of the shared Pythia level (first ancestor that has common/pythia/hffn_pythia).
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
if [ "$#" -eq 0 ]; then set -- $(cd "$exports" && ls -d */export | cut -d/ -f1); fi
failed=0
for name in "$@"; do
    "$python" -B checks/parity_layer.py --export-dir "$exports/$name/export" ${out:+--out "$out/layer/$name.json"} || failed=1
    "$python" -B checks/parity_model.py --gate --model-dir "$models/${name%%_*}" --export-dir "$exports/$name/export" --tokens "$tokens" \
        ${out:+--out "$out/model/$name.json"} 2>&1 | grep -v "Loading weights"
    [ "${PIPESTATUS[0]}" -eq 0 ] || failed=1
    "$python" -B checks/parity_update.py --gate --model-dir "$models/${name%%_*}" --export-dir "$exports/$name/export" --tokens "$tokens" \
        ${out:+--out "$out/update/$name.json"} 2>&1 | grep -v "Loading weights"
    [ "${PIPESTATUS[0]}" -eq 0 ] || failed=1
done
if [ "$failed" -eq 0 ]; then echo "check: PASSED"; else echo "check: FAILED"; fi
exit "$failed"
