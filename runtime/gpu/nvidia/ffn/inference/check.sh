#!/usr/bin/env bash
# Verification on a CUDA machine, without timing: the test layers against their pinned fingerprints, then the parity
# of every kernel pair on each test layer against the float64 oracle.
# Keeps its own record in runs/check_<UTC stamp>/: check.log, manifest.json (status, planned, done) and EXIT (exit code
# of the check; the first failing step is returned, a failing fingerprint step as 1, a failing parity as its own code).
# manifest.json reads "running" from the start: a run killed before the end (hard stop, timeout) keeps "running", no EXIT.
# Usage: ./check.sh      (python of the PATH: the CUDA environment)
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run="runs/check_$stamp"
mkdir -p "$run"
log="$run/check.log"
printf '{"status": "running", "planned": 1, "done": 0, "exit": null}\n' > "$run/manifest.json"
echo "start $stamp" | tee "$log"
python -B -c "import hffn_ffn_cuda, sys; from hffn_ffn_layers.layers import main; sys.exit(main())" 2>&1 | tee -a "$log"
code=${PIPESTATUS[0]}
if [ "$code" -ne 0 ]; then
    code=1
else
    python -B -m checks.parity 2>&1 | tee -a "$log"
    code=${PIPESTATUS[0]}
    if [ "$code" -eq 0 ]; then echo "check: PASSED" | tee -a "$log"; else echo "check: FAILED" | tee -a "$log"; fi
fi
if [ "$code" -eq 0 ]; then status=passed; done_n=1; else status=failed; done_n=0; fi
printf '{"status": "%s", "planned": 1, "done": %s, "exit": %s}\n' "$status" "$done_n" "$code" > "$run/manifest.json"
echo "end $(date -u +%Y%m%dT%H%M%SZ) exit $code" | tee -a "$log"
echo "$code" > "$run/EXIT"
exit "$code"
