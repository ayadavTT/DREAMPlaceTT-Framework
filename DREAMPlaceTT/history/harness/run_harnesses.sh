#!/usr/bin/env bash
# Build + run the isolated kernel harnesses, saving each kernel's output into
# its history/<stage>/<kernel>/profile/run.log. Self-contained (in-folder
# tt-metal + kernels). Run inside the TT-Metal container.
#   bash run_harnesses.sh            # build + run the safe set
#   bash run_harnesses.sh build      # build only
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DPTT="$(cd "$HERE/../.." && pwd)"
export TT_METAL_HOME="$DPTT/tt-metal"
export TT_METAL_RUNTIME_ROOT="$DPTT/tt-metal"
BUILD="$HERE/build"; mkdir -p "$BUILD"; cd "$BUILD"
CXX=clang++-20 cmake "$HERE" -DCMAKE_CXX_COMPILER=clang++-20 -DTT_METAL_HOME="$TT_METAL_HOME" >/tmp/hh_cmake.log 2>&1
make -j32 >/tmp/hh_build.log 2>&1 && echo "[harness] build OK" || { echo "[harness] build FAILED (see /tmp/hh_build.log)"; tail -5 /tmp/hh_build.log; }
[ "${1:-run}" = "build" ] && exit 0

H="$DPTT/history"
# kernel target -> history profile dir. v31_backward EXCLUDED (wedges device standalone).
declare -A OUT=(
  [v25_ef_l1gather]="$H/05-backward-ef-gather/v25-ef-l1gather"
  [v26]="$H/05-backward-ef-gather/v26-ef"
  [v27]="$H/04-backward-bucketing/v27-bucket"
  [v28]="$H/05-backward-ef-gather/v28-ef-multibin"
  [v29]="$H/05-backward-ef-gather/v29-gather"
  [v30]="$H/05-backward-ef-gather/v30-backward"
  [fccs]="$H/05-backward-ef-gather/fccs"
  [v32_regroup]="$H/04-backward-bucketing/v32-regroup"
  [atomic_bench_host]="$H/99-infra-probes/atomic-bench"
  [mcast_bw]="$H/99-infra-probes/mcast-bw"
  [v11op_bench_host]="$H/99-infra-probes/v11op-bench"
)
for t in "${!OUT[@]}"; do
  [ -x "$BUILD/$t" ] || continue
  timeout 120 "$BUILD/$t" > "${OUT[$t]}/profile/run.log" 2>&1
  printf '%-22s rc=%s  %s\n' "$t" "$?" "$(grep -E '^\[' "${OUT[$t]}/profile/run.log" | grep -vE ' \| ' | head -1 | cut -c1-70)"
done
echo "[harness] done. v31_backward skipped (standalone wedges device); v21/v22 are file-based (--inputs)."
