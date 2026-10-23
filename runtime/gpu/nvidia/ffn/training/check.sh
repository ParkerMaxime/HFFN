#!/usr/bin/env bash
# Verification of the CUDA FFN training step on a CUDA machine, without timing: the gates of check.py on the
# test layer 256 x 1024 at N = 1 to 2100, then the guards of the partial workspace.
# Keeps its own record in runs/check_<UTC stamp>/: check.log, manifest.json (status, planned, done) and EXIT
# (exit code of check.py; fail-closed: exit 0 without the final line "check: PASSED" is recorded as 1).
# manifest.json reads "running" from the start: a run killed before the end (hard stop, timeout) keeps "running", no EXIT.
# Usage on the VM: ./check.sh      (python of the PATH: the CUDA environment)
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run="runs/check_$stamp"
mkdir -p "$run"
log="$run/check.log"
printf '{"status": "running", "planned": 1, "done": 0, "exit": null}\n' > "$run/manifest.json"
echo "start $stamp" | tee "$log"
python -B check.py 2>&1 | tee -a "$log"
code=${PIPESTATUS[0]}
if [ "$code" -eq 0 ] && ! grep -q "^check: PASSED$" "$log"; then
    code=1
    echo "coverage: FAILED (no 'check: PASSED' line)" | tee -a "$log"
fi
if [ "$code" -eq 0 ]; then status=passed; done_n=1; else status=failed; done_n=0; fi
printf '{"status": "%s", "planned": 1, "done": %s, "exit": %s}\n' "$status" "$done_n" "$code" > "$run/manifest.json"
echo "end $(date -u +%Y%m%dT%H%M%SZ) exit $code" | tee -a "$log"
echo "$code" > "$run/EXIT"
exit "$code"
