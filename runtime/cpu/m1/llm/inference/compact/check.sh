#!/usr/bin/env bash
# Verification of the CPU M1 compact leaf, without timing: compilation of the Python sources, then for every sparse
# Pythia export the parity per layer, the token gap against the dense arm and the negative control of the tie band.
# Needs the prompt tokens of the shared Pythia level (see ../../../../common/pythia/tools/make_prompt_tokens.py) and
# a Mac free of any other job (it builds the extension and loads the models).
# Usage: ./check.sh [<base checkpoints dir> <exports dir>]
#   <base checkpoints dir>/pythia70m, .../pythia160m, .../pythia410m
#   <exports dir>/<model>_sparse<density>_ffn_only/export
# Without arguments, the two folders are those of $HFFN_ASSETS (pythia/models/base and pythia/exports).
# HFFN_PYTHON=<interpreter> when `python` of the PATH is not the environment of the project.
set -uo pipefail
cd "$(dirname "$0")"
PY=${HFFN_PYTHON:-python}
unset PYTORCH_ENABLE_MPS_FALLBACK
if [ "$#" -ge 2 ]; then
    models="$1"
    exports="$2"
else
    models=$("$PY" -B -c "import hffn_pythia_m1; from hffn_pythia.assets import models_dir; print(models_dir())") || exit 2
    exports=$("$PY" -B -c "import hffn_pythia_m1; from hffn_pythia.assets import exports_dir; print(exports_dir())") || exit 2
fi
"$PY" -B -m py_compile hffn_pythia_m1/*.py hffn_inference_runtime/*.py checks/*.py || exit 1
failed=0
count=0
for export_dir in "$exports"/*_sparse*_ffn_only/export; do
    [ -d "$export_dir" ] || continue
    name=$(basename "$(dirname "$export_dir")")
    count=$((count + 1))
    echo "== $name"
    "$PY" -B -m checks.parity_layers --export-dir "$export_dir" | tail -n 1 || failed=1
    "$PY" -B -m checks.parity_model --model-dir "$models/${name%%_*}" --export-dir "$export_dir" --threads 4 --negative-control 2>&1 \
        | grep -v "Loading weights" || failed=1
done
[ "$count" -eq 18 ] || { echo "check: expected 18 sparse exports, found $count"; failed=1; }
if [ "$failed" -eq 0 ]; then echo "check: PASSED ($count exports)"; else echo "check: FAILED"; fi
exit "$failed"
