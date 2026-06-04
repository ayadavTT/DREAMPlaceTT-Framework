#!/usr/bin/env bash
# Isolated test + profile for the v18-scatter kernel. Self-contained: uses only
# this folder's tt-metal, DREAMPlace dp_env, and the built v19_engine.so.
# ONE MeshDevice per process, so we invoke the test once per grid.
# Run inside the TT-Metal container.
#   bash run.sh --quick            # adaptec1 x {512,1024,2048} x 3 dists
#   bash run.sh                    # all 6 designs x 3 grids x 3 dists
#   GRIDS_OVERRIDE="2048" TT_METAL_DEVICE_PROFILER=1 bash run.sh --quick   # +Tracy
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"          # DREAMPlaceTT
cd "$ROOT"
export TT_METAL_HOME="$ROOT/tt-metal"
export TT_METAL_RUNTIME_ROOT="$ROOT/tt-metal"
export CPU_DCT=1                                  # isolate scatter (no DCT)
export GATHER_MODE=v18
PY="$ROOT/DREAMPlace/dp_env/bin/python3"

QUICK=""
for a in "$@"; do [ "$a" = "--quick" ] && QUICK="--quick"; done
GRIDS="${GRIDS_OVERRIDE:-512 1024 2048}"
if [ -n "$QUICK" ]; then DESIGNS="${DESIGNS_OVERRIDE:-adaptec1}"
else DESIGNS="${DESIGNS_OVERRIDE:-adaptec1 adaptec2 adaptec3 bigblue1 bigblue2 bigblue3}"; fi

# fresh sweep CSV
[ -n "$QUICK" ] && rm -f "$HERE/../profile/results_quick.csv" || rm -f "$HERE/../profile/results.csv"

# ONE process per (design,grid): engine NC_max == design cell count (accurate timing).
for DES in $DESIGNS; do
  for G in $GRIDS; do
    echo "================ $DES grid $G ================"
    "$PY" -u "$HERE/v18_scatter_test.py" --grid "$G" --design "$DES" $QUICK || echo "[run.sh] $DES grid $G FAILED"
  done
done

# Per-zone Tracy: copy + process the device profiler CSV if profiling was on.
CSV="$TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv"
if [ "${TT_METAL_DEVICE_PROFILER:-0}" = "1" ] && [ -f "$CSV" ]; then
    cp "$CSV" "$HERE/../profile/profile_log_device.csv"
    "$PY" "$ROOT/tools/profile_v11.py" "$CSV" > "$HERE/../profile/zones.txt" 2>/dev/null \
        && echo "[run.sh] wrote profile/zones.txt" || echo "[run.sh] zone processing skipped"
fi
