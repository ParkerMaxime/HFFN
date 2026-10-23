#!/usr/bin/env bash
# Request-level harness of the x86 LLM leaf: check | smoke | run   (README.md of the leaf; run on the x86 machine)
# Usage: bench.sh <mode> [<base checkpoints dir> <exports dir>] [extra arguments, e.g. --threads 4]
# Without the two folders, those of $HFFN_ASSETS are used. The python of the PATH is the environment of the leaf.
set -uo pipefail
MODE=${1:-}
case "$MODE" in
  check|smoke|run) ;;
  *) echo "usage: $0 check|smoke|run [<base checkpoints dir> <exports dir>] [extra arguments]" >&2; exit 2 ;;
esac
shift
FOLDERS=()
if [ "$#" -ge 2 ] && [ "${1#--}" = "$1" ] && [ "${2#--}" = "$2" ]; then
    FOLDERS=(--models "$1" --exports "$2")
    shift 2
fi
HERE=$(cd "$(dirname "$0")" && pwd)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
cd "$HERE"
if [ "$MODE" = check ]; then
    exec python -B bench_request.py --mode check ${FOLDERS[@]+"${FOLDERS[@]}"}
fi
RUN=$HERE/../runs/bench_${MODE}_$(date -u +%Y%m%dT%H%M%SZ)_$$
mkdir -p "$RUN"
python -B bench_request.py --mode "$MODE" ${FOLDERS[@]+"${FOLDERS[@]}"} --out "$RUN" "$@" 2>&1 | tee "$RUN/run.log"
STATUS=${PIPESTATUS[0]}
echo "$STATUS" > "$RUN/EXIT"
echo "EXIT=$STATUS  $RUN"
exit "$STATUS"
