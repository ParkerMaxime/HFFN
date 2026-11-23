#!/usr/bin/env bash
# Leaf llm/training on Nvidia: the S2 qualification (contexts, 70M d10, 410M d80 at B1 x S4) and the model checker
# (README). Nothing here is timed.
# check: local, no GPU, no timing: syntax, imports that need no CUDA, the files bound in provenance.json, the planned ids.
# unit:  local, no GPU, no timing: the unit test of update_gate.py (float64 closed form, faults, decay chain, bound tree).
# smoke|run: CUDA, on a Linux machine with a CUDA GPU. HFFN_BASE_COMMIT (or
#   BASE_COMMIT) identifies the source base recorded in provenance.json.
#   smoke                      the contexts checker, smoke grid (S0)
#   smoke --export <export>    the model checker, mode smoke, on pythia70m_sparse10_ffn_only or pythia410m_sparse80_ffn_only
#   run [--attempt 1|2]        the S2 qualification: contexts, 70M d10, 410M d80, in one process tree (checks/qualification.py)
# Usage: bash launch.sh check|unit
#        bash launch.sh smoke [--export <export>]
#        bash launch.sh run [--attempt N]
set -uo pipefail

leaf_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd) || exit 2
repo_dir="$leaf_dir"
while [ ! -d "$repo_dir/runtime/common/pythia/hffn_pythia" ]; do
    [ "$repo_dir" != / ] || { printf 'Tree root with runtime/ not found\n' >&2; exit 2; }
    repo_dir=$(dirname -- "$repo_dir")
done
mode=${1:-check}
if [ "$#" -gt 0 ]; then shift; fi

if [ "$(uname -s)" = Darwin ]; then
    local_python=${HFFN_CHECK_PYTHON:-${HFFN_PYTHON:-python}}
else
    local_python=${HFFN_CHECK_PYTHON:-${HFFN_CUDA_PYTHON:-python}}
fi

case "$mode" in
check|unit)
    if [ "$#" -ne 0 ]; then
        printf '%s takes no extra arguments\n' "$mode" >&2
        exit 2
    fi
    ;;
esac

if [ "$mode" = check ]; then
    "$local_python" -B - "$leaf_dir" "$repo_dir" <<'PY'
import ast
import sys
from pathlib import Path

root, repo = Path(sys.argv[1]), Path(sys.argv[2])
paths = sorted((root / "checks").glob("*.py")) + sorted((root / "hffn_pythia_cuda_training").glob("*.py"))
if not paths:
    raise SystemExit("No Python source files found")
for path in paths:
    ast.parse(path.read_text(), filename=str(path))
print(f"AST syntax passed: {len(paths)} source files", flush=True)

# Import the checker without CUDA (its heavy imports are inside import_runtime) and verify what it binds and plans.
sys.path.insert(0, str(root / "checks"))
sys.dont_write_bytecode = True
import model_check as mc

extra = mc.extra_sources(mc.TOKENS)
missing = [str(path) for path in extra if not path.is_file()]
if missing:
    raise SystemExit(f"files bound in provenance.json are missing: {missing}")
source = mc.context_helpers.provenance(None)
bound = set(source["source_hashes"])
if not any("hffn_pythia_cuda_training/" in name for name in bound):
    raise SystemExit("the package hffn_pythia_cuda_training is not bound in provenance.json")
print(f"provenance sources: {len(bound)} bound by the leaf, {len(extra)} added by the model checker, all present", flush=True)

for stage, fault in (("70M", True), ("410M", False)):
    ids = mc.planned_ids(fault)
    if len(set(ids)) != len(ids) or ids[-1] != "model_gate":
        raise SystemExit(f"planned ids of the {stage} stage are not unique or do not end with model_gate: {ids}")
    print(f"planned ids {stage}: {len(ids)}", flush=True)
PY
    code=$?
    [ "$code" -eq 0 ] || exit "$code"
    bash -n "$leaf_dir/launch.sh" || exit $?
    if [ -f "$leaf_dir/check.sh" ]; then bash -n "$leaf_dir/check.sh" || exit $?; fi
    printf 'Bash syntax passed\n'
    exit 0
fi

if [ "$mode" = unit ]; then
    PYTHONDONTWRITEBYTECODE=1 "$local_python" -B "$leaf_dir/checks/test_update_gate.py"
    exit $?
fi

# Resolved after the argument screen below: smoke without --export is the contexts checker, with --export the model checker.
case "$mode" in
    smoke|run) ;;
    *) printf 'Unknown mode: %s\n' "$mode" >&2; exit 2 ;;
esac

# CUDA modes run only on a Linux machine with a CUDA GPU.
if [ "$(uname -s)" != Linux ]; then
    printf 'Run %s on a Linux machine with a CUDA GPU\n' "$mode" >&2
    exit 2
fi
base_commit=${HFFN_BASE_COMMIT:-${BASE_COMMIT:-}}
if [ -z "$base_commit" ]; then
    printf 'Set HFFN_BASE_COMMIT or BASE_COMMIT to the source base\n' >&2
    exit 2
fi
for arg in "$@"; do
    case "$arg" in
        --out|--out=*|--base|--base=*|--smoke|--mode|--mode=*|--attention-backend|--attention-backend=*)
            printf 'The launcher owns --out, --base, the checker mode and the attention backend\n' >&2
            exit 2
            ;;
    esac
done
if [ "$mode" = run ]; then
    checker=qualification.py; checker_mode=()
    for arg in "$@"; do
        case "$arg" in --export|--export=*) printf 'run takes only --attempt\n' >&2; exit 2 ;; esac
    done
elif [ "$#" -eq 0 ]; then
    checker=context_check.py; checker_mode=(--smoke)
elif [ "$1" = --export ] && [ "$#" -eq 2 ]; then
    checker=model_check.py; checker_mode=(--mode smoke --attention-backend math)
else
    printf 'usage: smoke [--export <export>] | run [--attempt N]\n' >&2
    exit 2
fi
if [ ! -f "$leaf_dir/checks/$checker" ]; then
    printf 'Checker is not yet present: %s\n' "$checker" >&2
    exit 2
fi
cuda_python=${HFFN_CUDA_PYTHON:-python}
command -v "$cuda_python" >/dev/null 2>&1 || { printf 'CUDA Python is unavailable\n' >&2; exit 2; }
mkdir -p "$leaf_dir/runs" || exit 2
run_dir=$(mktemp -d "$leaf_dir/runs/${mode}_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX") || exit 2
export PYTHONUNBUFFERED=1
printf 'mode=%s base=%s output=%s\n' "$mode" "$base_commit" "$run_dir"
"$cuda_python" -u -B "$leaf_dir/checks/$checker" --out "$run_dir" --base "$base_commit" \
    "${checker_mode[@]}" "$@" 2>&1 | tee "$run_dir/run.log"
pipeline=("${PIPESTATUS[@]}")
code=${pipeline[0]}
tee_code=${pipeline[1]}

# Preserve a completed check's records. A killed interpreter or argument/import failure without a terminal record
# receives a failed launcher manifest and EXIT.
"$cuda_python" -B - "$run_dir" "$code" "$tee_code" "$mode" <<'PY'
import json
import os
from pathlib import Path
import sys
root = Path(sys.argv[1])
code, tee_code, mode = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
exit_path = root / "EXIT"
missing_exit = not exit_path.exists()
launcher_exit = code or tee_code or (1 if missing_exit else 0)
if missing_exit:
    effective = code if code != 0 else 1
    manifest_path = root / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    previous.update(status="failed", exit_code=effective, full_coverage=False,
                    error=f"launcher ended without a terminal checker EXIT (process={code}, tee={tee_code})")
    temporary = root / ".launcher-manifest.tmp"
    with temporary.open("x") as stream:
        json.dump(previous, stream, indent=2)
        stream.write("\n")
    os.replace(temporary, manifest_path)
    with exit_path.open("x") as stream:
        stream.write(f"{effective}\n")
launcher = {"status": "passed" if launcher_exit == 0 else "failed",
            "exit_code": launcher_exit, "process_exit_code": code, "tee_exit_code": tee_code,
            "mode": mode, "missing_checker_exit": missing_exit}
with (root / "launcher_manifest.json").open("x") as stream:
    json.dump(launcher, stream, indent=2)
    stream.write("\n")
with (root / "LAUNCH_EXIT").open("x") as stream:
    stream.write(f"{launcher_exit}\n")
raise SystemExit(launcher_exit)
PY
record_code=$?
if [ "$code" -eq 0 ] && [ "$record_code" -ne 0 ]; then code=$record_code; fi
if [ "$code" -eq 0 ] && [ "$tee_code" -ne 0 ]; then code=$tee_code; fi
printf 'output=%s launcher_exit=%s process_exit=%s tee_exit=%s\n' "$run_dir" "$code" "${pipeline[0]}" "$tee_code"
exit "$code"
