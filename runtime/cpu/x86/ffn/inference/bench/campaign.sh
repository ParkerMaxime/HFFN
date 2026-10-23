#!/usr/bin/env bash
# Measurement campaign on a dedicated multi-core machine (protocol: README, "Measurement campaign on a dedicated machine").
#
#   bench/campaign.sh check          build, pytest, parity; no measurement
#   bench/campaign.sh smoke          every step once with reduced arguments, in results/smoke_<time>/
#   bench/campaign.sh run <tag>      the campaign, in results/<tag>/ (run it inside tmux)
#
# smoke never starts the campaign. Every step logs unbuffered to the terminal and to
# <step>.log; steps.tsv records each step's exit code and duration; at the end (or at the
# first failing step) manifest.json states passed / failed / partial and EXIT holds the exit
# code. A step that exceeds its time cap ends the campaign with EXIT=124 (status partial).
# HFFN runs on T = min(CPUs, 8) threads; HFFN_CXXFLAGS defaults to "-mavx2 -mfma".
set -uo pipefail
cd "$(dirname "$0")/.."
MODE=${1:?usage: bench/campaign.sh check | smoke | run <tag>}
export PYTHONUNBUFFERED=1
export HFFN_CXXFLAGS=${HFFN_CXXFLAGS--mavx2 -mfma}
CPUS=$(nproc)
T=$(( CPUS < 8 ? CPUS : 8 ))
LIST="0-$(( T - 1 ))"

if [ "$MODE" = check ]; then
    python -c 'import hffn, hffn_ffn_layers.layers as layers; raise SystemExit(layers.main())' && python -m pytest -q && python -m hffn.checks
    exit $?
fi

case "$MODE" in
    smoke) TAG="smoke_$(date -u +%Y%m%dT%H%M%SZ)"; SESSIONS=1; SHORT_CAP=900; LONG_CAP=900
           REDUCED=(--case homog_256_1024 --n 1 --n 8 --n 32 --samples 2)
           ROUTES=(--n 8 --n 32 --calls 40); COLD=(--cold-calls 2)
           MEMORY=(--case homog_256_1024 --n 1 --n 8) ;;
    run)   TAG=${2:?usage: bench/campaign.sh run <tag>}; SESSIONS=3; SHORT_CAP=1800; LONG_CAP=3600
           REDUCED=(); ROUTES=(); COLD=(); MEMORY=() ;;
    *)     echo "unknown mode: $MODE" >&2; exit 2 ;;
esac
OUT=results/$TAG
if [ -e "$OUT" ]; then echo "refused: $OUT already exists (a measurement is never overwritten)" >&2; exit 2; fi
if pgrep -f "bench/(vs_dense|threads|routes|memory|process_footprint)\.py" >/dev/null; then
    echo "refused: a benchmark is already running on this machine" >&2; exit 2
fi
mkdir -p "$OUT"
: > "$OUT/steps.tsv"
STARTED=$(date -u +%FT%TZ)

finish() {  # finish <exit code>
    head -1 /proc/stat > "$OUT/proc_stat_end.txt"
    python bench/aggregate.py "$OUT" --mode "$MODE" --exit-code "$1" --started "$STARTED" --threads "$T" 2>&1 | tee "$OUT/aggregate.log"
    echo "$1" > "$OUT/EXIT"
    echo "campaign $MODE $TAG: EXIT=$1  ($OUT/manifest.json)"
    exit "$1"
}
trap 'finish 130' INT TERM

step() {  # step <cap seconds> <name> <command...>; the name is also the log path under $OUT
    local cap=$1 name=$2 begin code; shift 2
    mkdir -p "$(dirname "$OUT/$name")"
    echo "=== $(date -u +%T) $name: $*"
    begin=$(date +%s)
    timeout "$cap" "$@" 2>&1 | tee "$OUT/$name.log"
    code=${PIPESTATUS[0]}
    printf '%s\t%s\t%s\n' "$name" "$code" "$(( $(date +%s) - begin ))" >> "$OUT/steps.tsv"
    if [ "$code" -ne 0 ]; then echo "step $name failed with code $code" >&2; finish "$code"; fi
}

{
    echo "# $STARTED  mode=$MODE  nproc=$CPUS  threads=$T  cpus=$LIST  HFFN_CXXFLAGS=$HFFN_CXXFLAGS"
    echo "source: $(python -m hffn.provenance)"
    lscpu; echo; lscpu -e
    echo; command -v numactl >/dev/null && numactl -H || echo "numactl: not installed"
    echo; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "governor: not exposed"
    echo; uptime; echo; free -m
} > "$OUT/topology.txt"
head -1 /proc/stat > "$OUT/proc_stat_start.txt"

step "$SHORT_CAP" layers python -c 'import hffn, hffn_ffn_layers.layers as layers; raise SystemExit(layers.main())'
step "$SHORT_CAP" pytest python -m pytest -q
step "$SHORT_CAP" parity python -m hffn.checks

# Sessions: the same measurements in separate processes (HFFN on T threads, dense on T threads).
for session in $(seq 1 "$SESSIONS"); do
    S=s$session
    step "$SHORT_CAP" "$S/threads" python bench/threads.py --cpus "$LIST" "${REDUCED[@]}" --out "$OUT/$S/threads.json"
    step "$SHORT_CAP" "$S/routes" python bench/routes.py --cpus "$LIST" --threads "$T" "${ROUTES[@]}" --out "$OUT/$S/routes.json"
    step "$LONG_CAP" "$S/vs_dense_perf" python bench/vs_dense.py --phase perf --cpus "$LIST" --threads "$T" "${REDUCED[@]}" "${COLD[@]}" --out "$OUT/$S/vs_dense_perf.json"
    step "$SHORT_CAP" "$S/summary" python bench/summarize.py "$OUT/$S"
done

# Dense on fewer threads (hot cache only): 8 threads are not necessarily the fastest dense setting.
for dense in 4 1; do
    [ "$dense" -lt "$T" ] || continue
    step "$LONG_CAP" "dense$dense/vs_dense_perf" python bench/vs_dense.py --phase perf --no-cold --cpus "$LIST" --threads "$T" --dense-threads "$dense" "${REDUCED[@]}" --out "$OUT/dense$dense/vs_dense_perf.json"
done
# HFFN on 4 threads against dense on 4 threads (comparable with a 4-core machine).
if [ "$T" -gt 4 ]; then
    step "$LONG_CAP" "t4/vs_dense_perf" python bench/vs_dense.py --phase perf --no-cold --cpus "$LIST" --threads 4 --dense-threads 4 "${REDUCED[@]}" --out "$OUT/t4/vs_dense_perf.json"
fi

# Once: memory.
step "$SHORT_CAP" once/vs_dense_memory python bench/vs_dense.py --phase memory --cpus "$LIST" --threads "$T" "${MEMORY[@]}" --out "$OUT/once/vs_dense_memory.json"
step "$SHORT_CAP" once/memory python bench/memory.py --cpus "$LIST" --threads "$T" --out "$OUT/once/memory.json"
step "$LONG_CAP" once/footprint python bench/process_footprint.py --all --cpus "$LIST" --threads "$T" --out "$OUT/once/footprint.json"
step "$SHORT_CAP" once/summary python bench/summarize.py "$OUT/once"

finish 0
