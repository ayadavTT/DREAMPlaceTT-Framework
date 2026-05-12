#!/usr/bin/env bash
# reset_chip.sh — Reset the TT card. Required after a container restart
# (the IOMMU sysmem mapping gets a stale NOC address otherwise).
#
# Symptom this resolves:
#   UMD | error | Expected NOC address: 0x1000000000000000, but got 0x1000000040000000
#
# Usage:
#   CONTAINER=<container_name> bash scripts/reset_chip.sh

set -euo pipefail
: "${CONTAINER:?CONTAINER env var required}"

echo "[reset_chip] Resetting TT chip in container=$CONTAINER (chip ID 1)..."
docker exec "$CONTAINER" tt-smi -r 1
echo "[reset_chip] Done."
