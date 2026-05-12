#!/usr/bin/env bash
# build_server.sh — Build the V11 TT-Metal density scatter server.
#
# Must be run INSIDE the TT-Metal Docker container (it links against
# TT-Metal libraries in ${TT_METAL_HOME}/build_Release).
#
# Usage:
#   bash scripts/build_server.sh
#   bash scripts/build_server.sh -j 16        # parallelism override
#   TT_METAL_HOME=/path bash scripts/build_server.sh
#
# Output:
#   host/build/density_scatter_ttnn_server

set -euo pipefail

FW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$FW_ROOT"

: "${TT_METAL_HOME:=${FW_ROOT}/tt-metal}"

if [[ ! -d "$TT_METAL_HOME" ]]; then
    echo "ERROR: TT_METAL_HOME=$TT_METAL_HOME doesn't exist."
    echo "Either symlink ./tt-metal -> your tt-metal checkout, or export TT_METAL_HOME."
    exit 1
fi

if [[ ! -f "$TT_METAL_HOME/build_Release/lib/_ttnncpp.so" \
   && ! -f "$TT_METAL_HOME/build/lib/_ttnncpp.so" ]]; then
    echo "ERROR: TT-Metal hasn't been built (no _ttnncpp.so found)."
    echo "Build it first:"
    echo "  cd $TT_METAL_HOME && bash build_metal.sh --build-programming-examples"
    exit 1
fi

JOBS="${1:-$(nproc)}"
if [[ "$JOBS" == "-j" ]]; then JOBS="${2:-$(nproc)}"; fi

BUILD_DIR="$FW_ROOT/host/build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "[build_server] TT_METAL_HOME = $TT_METAL_HOME"
echo "[build_server] Build dir     = $BUILD_DIR"
echo "[build_server] Jobs          = $JOBS"

# clang-20 is required (TTNN headers use C++23 reflection).
: "${CXX:=clang++-20}"

cmake .. \
    -DCMAKE_CXX_COMPILER="$CXX" \
    -DTT_METAL_HOME="$TT_METAL_HOME"

make -j "$JOBS" density_scatter_ttnn_server

echo
echo "[build_server] OK → $BUILD_DIR/density_scatter_ttnn_server"
