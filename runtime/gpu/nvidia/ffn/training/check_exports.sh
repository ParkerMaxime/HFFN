#!/usr/bin/env bash
# Verification of the CUDA FFN training step on the real FFN layers of the 70M and 410M exports, without timing:
# `check_exports.py` (gate 5 local sets its exit; the other gates are listed apart), arguments passed through unchanged.
# Keeps its own record in runs/check_exports_<UTC stamp>/: check_exports.log, manifest.json (status, planned, done),
# provenance.json (hashes of the executed sources, GPU, names and sizes of the exports read) and EXIT (exit code of
# check_exports.py; fail-closed: exit 0 without the final line "check_exports: PASSED" is recorded as 1).
# manifest.json reads "running" from the start: a run killed before the end (hard stop, timeout) keeps "running", no EXIT.
# Usage: ./check_exports.sh [check_exports.py arguments]      (python of the PATH: the CUDA environment;
# HFFN_ASSETS names the assets folder, as for check_exports.py)
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run="runs/check_exports_$stamp"
mkdir -p "$run"
log="$run/check_exports.log"
printf '{"status": "running", "planned": 1, "done": 0, "exit": null}\n' > "$run/manifest.json"
python -B - "$run" <<'PY'
import hashlib, json, os, subprocess, sys
from pathlib import Path
run, leaf = Path(sys.argv[1]), Path.cwd()
repo = next((p for p in leaf.parents if (p / "runtime" / "common" / "pythia" / "hffn_pythia").is_dir()), None)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
out = {"head": None, "gpu": None, "leaf": {}, "shared": {}, "exports": {}}
if repo:
    out["head"] = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None
    out["missing"] = []
    for sub, pattern in (("runtime/gpu/nvidia/common/kernels/hffn_cuda", "*.py"),
                         ("runtime/common/ffn/hffn_ffn_layers", "*.py"),
                         ("runtime/gpu/nvidia/ffn/inference/hffn_ffn_cuda", "*.py")):
        files = sorted((repo / sub).glob(pattern))
        if not files:
            out["missing"].append(f"{sub}/{pattern}")   # an empty source folder is recorded, never silent
        for f in files:
            out["shared"][f"{sub}/{f.name}"] = sha(f)
    assets = Path(os.environ["HFFN_ASSETS"])
    exports = Path(os.environ["HFFN_EXPORTS"]) if "HFFN_EXPORTS" in os.environ else assets / "pythia/exports"
    for f in sorted(exports.rglob("*")) if exports.is_dir() else []:
        if f.is_file():
            st = f.stat()
            out["exports"][str(f.relative_to(exports))] = {"size": st.st_size, "mtime": int(st.st_mtime)}
for f in ("check_exports.py", "check.py", "kernels_train.py", "train_runtime.py", "train_step.py"):
    out["leaf"][f] = sha(leaf / f) if (leaf / f).is_file() else None
try:
    out["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip() or None
except OSError:
    pass
json.dump(out, open(run / "provenance.json", "w"), indent=1)
PY
echo "start $stamp" | tee "$log"
python -B check_exports.py "$@" 2>&1 | tee -a "$log"
code=${PIPESTATUS[0]}
if [ "$code" -eq 0 ] && ! grep -q "^check_exports: PASSED$" "$log"; then
    code=1
    echo "coverage: FAILED (no 'check_exports: PASSED' line)" | tee -a "$log"
fi
if [ "$code" -eq 0 ]; then status=passed; done_n=1; else status=failed; done_n=0; fi
printf '{"status": "%s", "planned": 1, "done": %s, "exit": %s}\n' "$status" "$done_n" "$code" > "$run/manifest.json"
echo "end $(date -u +%Y%m%dT%H%M%SZ) exit $code" | tee -a "$log"
echo "$code" > "$run/EXIT"
exit "$code"
