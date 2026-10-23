#!/usr/bin/env bash
# Runs check.sh and keeps its record: runs/port_check_<UTC stamp>.log and .EXIT (exit code).
# The log opens with the GPU, the library versions and the hashes of the sources (tools/provenance.py), and ends with
# the coverage of the check: one "-> PASSED" line per export for the layer check and for the token check.
# Usage: ./launch.sh [<base checkpoints dir> <exports dir>]     (python of the PATH: the CUDA environment;
#        without arguments, the folders of $HFFN_ASSETS)
set -uo pipefail
cd "$(dirname "$0")"
if [ "$#" -ge 2 ]; then
    models="$1"
    exports="$2"
else
    models=$(python -B -W ignore -c "import hffn_pythia_cuda; from hffn_pythia.assets import models_dir; print(models_dir())") || exit 2
    exports=$(python -B -W ignore -c "import hffn_pythia_cuda; from hffn_pythia.assets import exports_dir; print(exports_dir())") || exit 2
fi
stamp=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p runs
log="runs/port_check_$stamp.log"
{
    echo "start $stamp"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    python -B -W ignore -c "import sys, torch, cupy, transformers; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.version.cuda, 'cupy', cupy.__version__, 'transformers', transformers.__version__)"
    python -B tools/provenance.py
    python -B tools/provenance.py --files
    echo "models $models"
    echo "exports $exports"
} 2>&1 | tee "$log"
PYTHONWARNINGS=ignore ./check.sh "$models" "$exports" 2>&1 | tee -a "$log"
code=${PIPESTATUS[0]}
expected=$(ls -d "$exports"/*/export 2>/dev/null | wc -l)
layers=$(grep -c -- "max relative distance .* -> PASSED" "$log")
tokens=$(grep -c -- "unexplained HFFN tokens -> PASSED" "$log")
echo "coverage: $expected exports, $layers layer lines, $tokens token lines" | tee -a "$log"
if [ "$code" -eq 0 ] && { [ "$expected" -eq 0 ] || [ "$layers" -ne "$expected" ] || [ "$tokens" -ne "$expected" ]; }; then
    code=1
    echo "coverage: FAILED" | tee -a "$log"
fi
echo "end $(date -u +%Y%m%dT%H%M%SZ) exit $code" | tee -a "$log"
echo "$code" > "runs/port_check_$stamp.EXIT"
exit "$code"
