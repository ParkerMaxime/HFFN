#!/usr/bin/env bash
# Verification of the leaf on a CUDA machine, without timing: the unit test of the update gate, then the S2 qualification
# (contexts, 70M d10, 410M d80, checks/qualification.py). The qualification is the check of the leaf: the two exports it judges
# are covered here, the other 16 by the every-export check.
# Keeps its own record in runs/check_<UTC stamp>_<pid>/: check.log, manifest.json (status, planned, done), qualification/ (the
# qualification records) and EXIT (exit code of the check; the first failing step is returned).
# manifest.json reads "running" from the start: a run killed before the end (hard stop, timeout) keeps "running", no EXIT.
# Needs HFFN_BASE_COMMIT (or BASE_COMMIT), the source base recorded in provenance.json.
# Usage: ./check.sh      (python of the PATH: the CUDA environment; Linux only)
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
if [ "$(uname -s)" != Linux ]; then
    echo "check.sh runs on a Linux machine with a CUDA GPU" >&2
    exit 2
fi
base_commit=${HFFN_BASE_COMMIT:-${BASE_COMMIT:-}}
if [ -z "$base_commit" ]; then
    echo "Set HFFN_BASE_COMMIT or BASE_COMMIT to the source base" >&2
    exit 2
fi
cuda_python=${HFFN_CUDA_PYTHON:-python}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run="runs/check_${stamp}_$$"
mkdir -p "$run"
log="$run/check.log"
printf '{"status": "running", "planned": 2, "done": 0, "exit": null}\n' > "$run/manifest.json"
echo "start $stamp" | tee "$log"
"$cuda_python" -B checks/test_update_gate.py 2>&1 | tee -a "$log"
code=${PIPESTATUS[0]}
done_n=0
if [ "$code" -eq 0 ]; then
    done_n=1
    "$cuda_python" -u -B checks/qualification.py --out "$run/qualification" --base "$base_commit" 2>&1 | tee -a "$log"
    code=${PIPESTATUS[0]}
    if [ "$code" -eq 0 ]; then done_n=2; fi
fi
if [ "$code" -eq 0 ]; then status=passed; echo "check: PASSED" | tee -a "$log"; else status=failed; echo "check: FAILED" | tee -a "$log"; fi
printf '{"status": "%s", "planned": 2, "done": %s, "exit": %s}\n' "$status" "$done_n" "$code" > "$run/manifest.json"
echo "end $(date -u +%Y%m%dT%H%M%SZ) exit $code" | tee -a "$log"
echo "$code" > "$run/EXIT"
exit "$code"
