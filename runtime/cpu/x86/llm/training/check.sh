#!/usr/bin/env bash
# Verification on Linux x86_64, without timing: sources, compilation, the layer gate, the model gate and the update gate.
# Usage: ./check.sh <base checkpoints dir> <exports dir> [export name ...]
#   <base checkpoints dir>/pythia70m, ...;  <exports dir>/<model>_sparse<density>_ffn_only/export
#   without export names: every export of <exports dir>. HFFN_CHECK_OUT=<dir> keeps the JSON of each check.
# The gates are those of the shared level (../../../common/llm/training/check_core.sh), run on this leaf.
set -uo pipefail
cd "$(dirname "$0")"
python=${HFFN_PYTHON:-python}
"$python" -B -c "
import hashlib, json, sys
failed = 0
for entry in json.load(open('sources/manifest.json'))['files']:
    ok = hashlib.sha256(open(entry['path'], 'rb').read()).hexdigest() == entry['sha256']
    failed += not ok
    print(('ok   ' if ok else 'FAIL ') + entry['path'])
sys.exit(1 if failed else 0)" || exit 1
exec ../../../common/llm/training/check_core.sh "$PWD" "$@"
