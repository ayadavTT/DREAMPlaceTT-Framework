#!/usr/bin/env bash
# run_sweep.sh — Run DREAMPlace end-to-end on TT for a set of benchmarks.
#
# Default: all 4 (adaptec1_{512,2048}, bigblue3_{512,2048}).
# Pass benchmark names (without .json) to override.
#
# Usage:
#   bash scripts/run_sweep.sh
#   bash scripts/run_sweep.sh sweep_adaptec1_2048
#   RESULTS_DIR=./results/sweep_latest bash scripts/run_sweep.sh
#
# Env vars:
#   CONTAINER   — Docker container name (required)
#   PYTHON      — Python interpreter (default: $FW/DREAMPlace/dp_env/bin/python3)
#   RESULTS_DIR — Output directory for metrics + per-iter CSVs

set -euo pipefail

FW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$FW_ROOT"

: "${CONTAINER:?CONTAINER env var required (Docker container name)}"
: "${PYTHON:=$FW_ROOT/DREAMPlace/dp_env/bin/python3}"
: "${RESULTS_DIR:=$FW_ROOT/results/sweep_latest}"

mkdir -p "$RESULTS_DIR"

if [[ "$#" -eq 0 ]]; then
    BENCHES=(sweep_adaptec1_512 sweep_adaptec1_2048 sweep_bigblue3_512 sweep_bigblue3_2048)
else
    BENCHES=("$@")
fi

CONFIGS_DIR="$FW_ROOT/benchmarks/configs"
TIMEOUT="${TIMEOUT:-1200}"

for bench in "${BENCHES[@]}"; do
    JSON="$CONFIGS_DIR/${bench}.json"
    if [[ ! -f "$JSON" ]]; then
        # fall back to DREAMPlace's own test/ispd2005 if present
        JSON="$FW_ROOT/DREAMPlace/test/ispd2005/${bench}.json"
    fi
    if [[ ! -f "$JSON" ]]; then
        echo "[run_sweep] SKIP $bench — no JSON found"
        continue
    fi
    echo "=========================================================="
    echo "[run_sweep] $bench"
    echo "[run_sweep] config: $JSON"
    echo "=========================================================="
    GATHER_MODE=v11 OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" \
    timeout "$TIMEOUT" "$PYTHON" "$FW_ROOT/integration/run_dreamplace.py" \
        --device scatter_ttnn \
        --container "$CONTAINER" \
        --benchmark "$JSON" \
        --results-dir "$RESULTS_DIR" \
        2>&1 | tail -3 || true

    METRICS="$RESULTS_DIR/${bench}_scatter_ttnn_metrics.json"
    if [[ -f "$METRICS" ]]; then
        echo "[result] $(grep -E '"n_ep_calls"|"hpwl"|"overflow"|"wall_time_s"' "$METRICS" | tr '\n' ' ')"
    else
        echo "[result] (no metrics file produced)"
    fi
    echo
done

echo "[run_sweep] DONE. Output in: $RESULTS_DIR"
