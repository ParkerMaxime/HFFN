#!/usr/bin/env bash
# Launcher of this step (run = smoke). Usage: run.sh check | smoke | run
#   check  pre-training checks of PLAN section 7 only (exit non-zero on failure)
#   smoke  checks, then every arm of the step for 40 steps (proves nothing about quality)
#   run    checks, then the stages of step.json; records go to runs/<mode>_<UTC>/
# Progress is unbuffered on stdout and in runs/<...>/run.log; manifest.json is the status manifest; EXIT is reliable.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
TRAIN=$(cd "$HERE/.." && pwd)
MODE=${1:?usage: run.sh check|smoke|run}
python "$TRAIN/fetch_enwik8.py" || exit 4
cd "$TRAIN" && exec python -u -m lib.prep3 --step-dir "$HERE" --mode "$MODE"
