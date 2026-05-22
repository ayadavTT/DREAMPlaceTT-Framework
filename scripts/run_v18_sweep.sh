#!/bin/bash
# V18 hash-agg vs V11 sweep
# Runs both modes on the 18-config benchmark set, saves results + Tracy CSVs.
#
# Usage: scripts/run_v18_sweep.sh [config_glob]

set -uo pipefail

CONTAINER=bh-38-special-ayadav-for-reservation-75063
RESULTS_BASE=results/v18_sweep_$(date +%Y%m%d_%H%M)
mkdir -p "$RESULTS_BASE"

CONFIG_GLOB=${1:-sweep_*.json}

# Sort configs: adaptec1_512 first (smallest, smoke), then size-asc.
CONFIGS=$(ls benchmarks/configs/$CONFIG_GLOB | grep -E '(sweep_adaptec|sweep_bigblue)' | grep -v hybrid | sort)

echo "Sweep configs:"
echo "$CONFIGS" | sed 's/^/  /'
echo "Results dir: $RESULTS_BASE"
echo

run_one() {
    local mode=$1
    local cfg=$2
    local name=$(basename "$cfg" .json)
    local rdir="$RESULTS_BASE/${mode}_${name}"
    mkdir -p "$rdir"
    local logf="$rdir/run.log"
    echo "[$(date +%H:%M:%S)] $mode  $name ..."
    TT_METAL_HOME=$(pwd)/tt-metal CPU_DCT=1 GATHER_MODE=$mode \
      TT_METAL_DEVICE_PROFILER=1 \
      DREAMPlace/dp_env/bin/python3 integration/run_dreamplace.py \
        --device scatter_ttnn --container $CONTAINER \
        --benchmark "$cfg" \
        --results-dir "$rdir" \
        > "$logf" 2>&1
    local rc=$?
    # Copy Tracy CSV if exists
    if [ -f "tt-metal/generated/profiler/.logs/profile_log_device.csv" ]; then
        cp tt-metal/generated/profiler/.logs/profile_log_device.csv "$rdir/profile_log_device.csv"
    fi
    if [ -f "$rdir/${name}_scatter_ttnn_metrics.json" ]; then
        local hpwl=$(DREAMPlace/dp_env/bin/python3 -c "
import json; m=json.load(open('$rdir/${name}_scatter_ttnn_metrics.json'))
print(f'HPWL={m.get(\"hpwl\",0):.0f} ov={m.get(\"overflow\",0):.4f} sc={m.get(\"scatter_ttnn_scatter_ms_median\",0):.2f}ms ga={m.get(\"scatter_ttnn_gather_ms_median\",0):.2f}ms')
")
        echo "    $hpwl"
    else
        echo "    FAILED (rc=$rc)"
    fi
}

for cfg in $CONFIGS; do
    run_one v11      "$cfg"
    run_one v18 "$cfg"
done

echo
echo "=== Sweep complete ==="
echo "Results in $RESULTS_BASE"
