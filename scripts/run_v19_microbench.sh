#!/usr/bin/env bash
# Run V19 microbench across grid × cells × distribution.
# Writes a CSV (results/v19_microbench/v19_sweep.csv) + per-config logs.

set -u

FW_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONTAINER="${CONTAINER:-bh-38-special-ayadav-for-reservation-75661}"
OUT_DIR="${OUT_DIR:-$FW_ROOT/results/v19_microbench}"
TIMEOUT_SEC="${TIMEOUT_SEC:-180}"

# Grids and configs to sweep
GRIDS=(${GRIDS:-512 1024 2048})
CELLS=(${CELLS:-1000 10000 50000 200000 500000})
DISTS=(${DISTS:-uniform cluster hotcold})

# Enable Tracy device profiler when set
PROFILE="${TT_METAL_DEVICE_PROFILER:-0}"

mkdir -p "$OUT_DIR"
CSV="$OUT_DIR/v19_sweep.csv"
echo "grid,n_cells,dist,exit_code,scatter_ms,writeout_ms,dram_read_ms,mismatches,sum_actual,sum_expected,log" > "$CSV"

run_one() {
    local grid="$1"; local ncells="$2"; local dist="$3"
    local stem="v19mb_g${grid}_c${ncells}_${dist}"
    local log="$OUT_DIR/${stem}.log"
    local cmd=(
        docker exec -w /localdev/ayadav/tt-work/TTPort/DREAMPlaceTT-Framework
        -e "TT_METAL_HOME=/localdev/ayadav/tt-work/TTPort/DREAMPlaceTT-Framework/tt-metal"
        -e "TT_METAL_RUNTIME_ROOT=/localdev/ayadav/tt-work/TTPort/DREAMPlaceTT-Framework/tt-metal"
        -e "LD_LIBRARY_PATH=/localdev/ayadav/tt-work/TTPort/DREAMPlaceTT-Framework/tt-metal/build_Release/lib"
        -e "ARCH_NAME=blackhole"
        -e "TT_METAL_LOGGER_LEVEL=WARNING"
        -e "TT_METAL_DEVICE_PROFILER=${PROFILE}"
        "$CONTAINER"
        timeout "$TIMEOUT_SEC"
        host/build/v19_microbench_host "$grid" "$ncells" "$dist"
    )

    echo "[sweep] grid=$grid cells=$ncells dist=$dist ..." | tee -a "$log"
    local t0=$(date +%s.%N)
    set +e
    "${cmd[@]}" >>"$log" 2>&1
    local rc=$?
    set -e
    local t1=$(date +%s.%N)
    local wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')

    # Parse the RESULT,... line if present
    local result_line=""
    if grep -q "^RESULT," "$log"; then
        result_line=$(grep "^RESULT," "$log" | head -1)
    fi

    if [[ -n "$result_line" ]]; then
        # RESULT,grid,cells,dist,scatter,writeout,dram_read,mismatches,actual,expected
        local fields="${result_line#RESULT,}"
        local scatter_ms=$(echo "$fields" | cut -d, -f4)
        local writeout_ms=$(echo "$fields" | cut -d, -f5)
        local dram_ms=$(echo "$fields" | cut -d, -f6)
        local mismatches=$(echo "$fields" | cut -d, -f7)
        local actual=$(echo "$fields" | cut -d, -f8)
        local expected=$(echo "$fields" | cut -d, -f9)
        echo "${grid},${ncells},${dist},${rc},${scatter_ms},${writeout_ms},${dram_ms},${mismatches},${actual},${expected},${log}" >> "$CSV"
        echo "  → rc=$rc scatter=${scatter_ms}ms writeout=${writeout_ms}ms mismatches=${mismatches} (wall=${wall}s)"
    else
        echo "${grid},${ncells},${dist},${rc},,,,,,,${log}" >> "$CSV"
        if [[ "$rc" == "124" ]]; then
            echo "  → TIMEOUT (>${TIMEOUT_SEC}s)  log: $log"
        else
            echo "  → rc=$rc NO RESULT (wall=${wall}s)  log: $log"
        fi
    fi
}

echo "[sweep] writing → $CSV"
echo "[sweep] grids=${GRIDS[*]}  cells=${CELLS[*]}  dists=${DISTS[*]}  profile=${PROFILE}"
echo

for g in "${GRIDS[@]}"; do
    for n in "${CELLS[@]}"; do
        for d in "${DISTS[@]}"; do
            run_one "$g" "$n" "$d"
        done
    done
done

echo
echo "[sweep] DONE → $CSV"
column -s, -t "$CSV" | head -60
