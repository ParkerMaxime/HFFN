#!/usr/bin/env bash
# Publication campaign of the x86 LLM leaf (protocol: README, "Measurement campaign").
#
#   bench/campaign.sh check [<models> <exports>]        provenance, tests, assets of the grid; no measurement
#   bench/campaign.sh smoke [<models> <exports>]        every step once on a reduced grid, in results/smoke_<time>/
#   bench/campaign.sh run <tag> [<models> <exports>]    the campaign, in results/<tag>/ (run it inside tmux)
#   bench/campaign.sh bench <tag> [<models> <exports>]  its timing steps only (provenance, tests, the two harness runs)
#
# Without <models> <exports>, the folders of $HFFN_ASSETS are used (pythia/models/base, pythia/exports).
# smoke never starts the campaign. Every step logs unbuffered to the terminal and to <step>.log; steps.tsv records
# each step's exit code and duration; at the end (or at the first failing step) manifest.json states passed / failed
# / partial and EXIT holds the exit code. A step that exceeds its cap, or the ceiling of the whole campaign, ends it
# with EXIT=124 (status partial). PyTorch and HFFN run on 8 threads (4 in the control run).
set -uo pipefail
cd "$(dirname "$0")/.."
MODE=${1:?usage: bench/campaign.sh check | smoke | run <tag>}
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONWARNINGS=ignore
CEILING=14400                      # seconds: no step starts after four hours, and none outlasts them

case "$MODE" in
    check|smoke) TAG="${MODE}_$(date -u +%Y%m%dT%H%M%SZ)"; ARGS=("${@:2}") ;;
    run|bench)   TAG=${2:?usage: bench/campaign.sh $MODE <tag>}; ARGS=("${@:3}") ;;
    *)           echo "unknown mode: $MODE" >&2; exit 2 ;;
esac
if [ "${#ARGS[@]}" -ge 2 ]; then
    MODELS=${ARGS[0]}; EXPORTS=${ARGS[1]}
else
    MODELS=$(python -B -c "import hffn_pythia_x86; from hffn_pythia.assets import models_dir; print(models_dir())") || exit 2
    EXPORTS=$(python -B -c "import hffn_pythia_x86; from hffn_pythia.assets import exports_dir; print(exports_dir())") || exit 2
fi
if [ "$(nproc)" -lt 8 ]; then echo "refused: the protocol runs on 8 CPUs, this machine has $(nproc)" >&2; exit 2; fi
OUT=results/$TAG
if [ -e "$OUT" ]; then echo "refused: $OUT already exists (a measurement is never overwritten)" >&2; exit 2; fi
if pgrep -f "bench/(bench_request|memory)\.py|hffn_pythia_x86|checks\.parity|pytest" >/dev/null; then
    echo "refused: a benchmark or a check is already running on this machine" >&2; exit 2
fi
mkdir -p "$OUT"
: > "$OUT/steps.tsv"
STARTED=$(date -u +%FT%TZ)
BEGAN=$(date +%s)

finish() {  # finish <exit code>
    head -1 /proc/stat > "$OUT/proc_stat_end.txt"
    python -B bench/campaign_manifest.py "$OUT" --mode "$MODE" --exit-code "$1" --started "$STARTED" 2>&1 | tee "$OUT/manifest.log"
    echo "$1" > "$OUT/EXIT"
    echo "campaign $MODE $TAG: EXIT=$1  ($OUT/manifest.json)"
    exit "$1"
}
trap 'finish 130' INT TERM

record() {  # record <name> <exit code> <seconds>
    printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$OUT/steps.tsv"
}

step() {  # step <cap seconds> <name> <command...>; the name is also the log path under $OUT
    local cap=$1 name=$2 begin code left; shift 2
    left=$(( CEILING - ($(date +%s) - BEGAN) ))
    if [ "$left" -le 0 ]; then echo "ceiling of $CEILING s reached before $name" >&2; finish 124; fi
    [ "$cap" -gt "$left" ] && cap=$left
    echo "=== $(date -u +%T) $name: $*"
    begin=$(date +%s)
    timeout "$cap" "$@" 2>&1 | tee "$OUT/$name.log"
    code=${PIPESTATUS[0]}
    record "$name" "$code" "$(( $(date +%s) - begin ))"
    if [ "$code" -ne 0 ]; then echo "step $name failed with code $code" >&2; finish "$code"; fi
}

expect_count() {  # expect_count <name> <log> <pattern> <number>: the log holds exactly that many lines matching the pattern
    local found
    found=$(grep -c -- "$3" "$OUT/$2.log")
    echo "=== $1: $found lines match '$3' (expected $4)"
    record "$1" "$([ "$found" -eq "$4" ] && echo 0 || echo 1)" 0
    if [ "$found" -ne "$4" ]; then finish 1; fi
}

case "$MODE" in check) PLANNED=3 ;; bench) PLANNED=4 ;; run) PLANNED=8 ;; *) PLANNED=7 ;; esac   # steps: provenance tests check check_coverage [check_negative_control, run only] bench_run bench_threads4 memory
echo "$PLANNED" > "$OUT/planned.txt"
{
    echo "# $STARTED  mode=$MODE  nproc=$(nproc)  models=$MODELS  exports=$EXPORTS"
    echo "source: $(python -B tools/provenance.py)"
    lscpu; echo; lscpu -e
    echo; command -v numactl >/dev/null && numactl -H || echo "numactl: not installed"
    echo; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "governor: not exposed"
    echo; uptime; echo; free -m
} > "$OUT/topology.txt"
head -1 /proc/stat > "$OUT/proc_stat_start.txt"

step 300 provenance bash -c 'python -B tools/provenance.py && python -B tools/provenance.py --files > "$0/provenance_files.txt"' "$OUT"
step 1800 tests python -B -m pytest -q

if [ "$MODE" = check ]; then
    step 300 assets python -B bench/bench_request.py --mode check --models "$MODELS" --exports "$EXPORTS"
    finish 0
fi

if [ "$MODE" = smoke ]; then
    mkdir -p "$OUT/smoke_exports"
    for name in pythia70m_sparse10_ffn_only pythia70m_sparse50_ffn_only pythia160m_sparse20_ffn_only; do ln -s "$EXPORTS/$name" "$OUT/smoke_exports/$name"; done
    step 1800 check ./check.sh "$MODELS" "$OUT/smoke_exports"
    expect_count check_coverage check "unexplained HFFN tokens -> PASSED" 3
    step 900 bench_run python -B bench/bench_request.py --mode smoke --models "$MODELS" --exports "$EXPORTS" --out "$OUT/bench_run" --threads 8
    step 900 bench_threads4 python -B bench/bench_request.py --mode smoke --models "$MODELS" --exports "$EXPORTS" --out "$OUT/bench_threads4" --threads 4
    step 900 memory python -B bench/memory.py --models "$MODELS" --exports "$EXPORTS" --out "$OUT/memory.json" --threads 8 --sizes 70 --densities 20
else
    if [ "$MODE" = run ]; then
        step 3600 check ./check.sh "$MODELS" "$EXPORTS"
        expect_count check_coverage check "unexplained HFFN tokens -> PASSED" 18
        expect_count check_negative_control check "unexplained tokens -> DETECTED" 18
    fi
    step 14400 bench_run python -B bench/bench_request.py --mode run --models "$MODELS" --exports "$EXPORTS" --out "$OUT/bench_run" --threads 8
    step 3600 bench_threads4 python -B bench/bench_request.py --mode run --models "$MODELS" --exports "$EXPORTS" --out "$OUT/bench_threads4" --threads 4 --sessions 1 --densities 10,50
    [ "$MODE" = run ] && step 3600 memory python -B bench/memory.py --models "$MODELS" --exports "$EXPORTS" --out "$OUT/memory.json" --threads 8
fi
finish 0
