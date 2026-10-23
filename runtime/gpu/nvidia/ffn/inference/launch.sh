#!/usr/bin/env bash
# Runs check.sh and keeps its record: runs/ffn_check_<UTC stamp>.log and .EXIT (exit code).
# The log opens with the GPU, the library versions and the hashes of the sources (tools/provenance.py), and ends with
# the coverage of the check: one "-> PASSED" line per test layer.
# Usage: ./launch.sh      (python of the PATH: the CUDA environment)
set -uo pipefail
cd "$(dirname "$0")"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p runs
log="runs/ffn_check_$stamp.log"
{
    echo "start $stamp"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    python -B -W ignore -c "import sys, numpy, torch, cupy; print('python', sys.version.split()[0], 'numpy', numpy.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda, 'cupy', cupy.__version__)"
    python -B tools/provenance.py
    python -B tools/provenance.py --files
} 2>&1 | tee "$log"
PYTHONWARNINGS=ignore ./check.sh 2>&1 | tee -a "$log"
code=${PIPESTATUS[0]}
expected=$(python -B -c "import hffn_ffn_cuda; from hffn_ffn_layers.layers import names; print(len(names()))" 2>/dev/null || echo 0)
passed=$(grep -c -- "(tolerance .*) -> PASSED" "$log")
echo "coverage: $expected test layers, $passed passed lines" | tee -a "$log"
if [ "$code" -eq 0 ] && { [ "$expected" -eq 0 ] || [ "$passed" -ne "$expected" ]; }; then
    code=1
    echo "coverage: FAILED" | tee -a "$log"
fi
echo "end $(date -u +%Y%m%dT%H%M%SZ) exit $code" | tee -a "$log"
echo "$code" > "runs/ffn_check_$stamp.EXIT"
exit "$code"
