#!/usr/bin/env bash
# Verification on a CUDA machine, without timing: sources, then for every export the parity
# per layer and the token gap against the dense arm.
# Usage: ./check.sh [<base checkpoints dir> <exports dir>]
#   <base checkpoints dir>/pythia70m, .../pythia160m, .../pythia410m
#   <exports dir>/<model>_sparse<density>_ffn_only/export
# Without arguments, the two folders are those of $HFFN_ASSETS (pythia/models/base and pythia/exports).
set -uo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
if [ "$#" -ge 2 ]; then
    models="$1"
    exports="$2"
else
    models=$(python -B -W ignore -c "import hffn_pythia_cuda; from hffn_pythia.assets import models_dir; print(models_dir())") || exit 2
    exports=$(python -B -W ignore -c "import hffn_pythia_cuda; from hffn_pythia.assets import exports_dir; print(exports_dir())") || exit 2
fi
python -B -c "import sys, hffn_pythia_cuda; from hffn_pythia.verify_sources import main; sys.exit(main('.'))" || exit 1
failed=0
for export_dir in "$exports"/*/export; do
    name=$(basename "$(dirname "$export_dir")")
    echo "== $name"
    python -B -m checks.parity_layers --export-dir "$export_dir" || failed=1
    python -B -m checks.parity_model --model-dir "$models/${name%%_*}" --export-dir "$export_dir" 2>&1 || failed=1
done
if [ "$failed" -eq 0 ]; then echo "check: PASSED"; else echo "check: FAILED"; fi
exit "$failed"
