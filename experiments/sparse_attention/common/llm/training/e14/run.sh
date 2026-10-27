#!/usr/bin/env bash
# Launcher of this step. Usage: run.sh check | smoke | prefill | decode | memory
#   check  the checks of this step only (exit non-zero on failure)
#   smoke  checks, gate, every cell in both modes (one process, 2 windows) and the memory run on one arm (proves nothing)
#   prefill | decode | memory   checks, (gate,) then the timing run of the mode, or the memory run; records go to runs/<mode>_<UTC>/
# Progress is unbuffered on stdout and in runs/<...>/run.log; manifest.json is the status manifest; EXIT is reliable.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
TRAIN=$(cd "$HERE/.." && pwd)
MODE=${1:?usage: run.sh check|smoke|prefill|decode|memory}
python "$TRAIN/fetch_enwik8.py" || exit 4
cd "$TRAIN" && exec python -u -m lib.e14 --step-dir "$HERE" --mode "$MODE"
