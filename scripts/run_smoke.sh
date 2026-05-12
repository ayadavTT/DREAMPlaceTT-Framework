#!/usr/bin/env bash
# run_smoke.sh — Run the V11 phase-2 single-iteration smoke test against
# the TT server. Pass criterion: rel_L2 < 1%.
#
# Usage:
#   bash scripts/run_smoke.sh
#   bash scripts/run_smoke.sh --grid 512 --cells 100000
#
# Env vars:
#   CONTAINER   — Docker container name running tt-metal
#   PYTHON      — Python interpreter (default: DREAMPlace dp_env python)

set -euo pipefail

FW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$FW_ROOT"

: "${CONTAINER:?CONTAINER env var required (Docker container name)}"
: "${PYTHON:=$FW_ROOT/DREAMPlace/dp_env/bin/python3}"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: $PYTHON not executable. Set PYTHON=<path/to/python>."
    exit 1
fi

GRID="${GRID:-2048}"
CELLS="${CELLS:-1500000}"

echo "[run_smoke] CONTAINER=$CONTAINER  grid=$GRID  cells=$CELLS"

# Extra args passed through to the smoke harness
exec "$PYTHON" "$FW_ROOT/tools/v11_phase2_smoke.py" \
    --container "$CONTAINER" \
    --grid "$GRID" \
    --cells "$CELLS" \
    "$@"
