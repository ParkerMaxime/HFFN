#!/usr/bin/env bash
# Verification on Apple Silicon, without timing: compilation, the layer gate, the model gate and the update gate.
# Usage: ./check.sh <base checkpoints dir> <exports dir> [export name ...]
#   <base checkpoints dir>/pythia70m, ...;  <exports dir>/<model>_sparse<density>_ffn_only/export
#   without export names: every export of <exports dir>. HFFN_CHECK_OUT=<dir> keeps the JSON of each check.
#   HFFN_PYTHON=<interpreter> when `python` of the PATH is not the environment of the project.
# The gates are those of the shared level (../../../common/llm/training/check_core.sh), run on this leaf.
set -uo pipefail
cd "$(dirname "$0")"
[ -z "${PYTORCH_ENABLE_MPS_FALLBACK:-}" ] || { echo "refused: PYTORCH_ENABLE_MPS_FALLBACK is set" >&2; exit 2; }
# Gates at the same time, per model size (16 GB of memory; the model gates hold a float64 model).
export HFFN_CHECK_LIMITS=${HFFN_CHECK_LIMITS-layer/pythia70m=4,layer/pythia160m=3,gates/pythia70m=3,gates/pythia160m=2}
exec ../../../common/llm/training/check_core.sh "$PWD" "$@"
