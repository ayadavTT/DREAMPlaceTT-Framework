#!/bin/bash
# NO-HOST DRIVE convergence sweep: V22_NOHOST=drive (gradient resident in b_dg,
# on-device combine/precond/nesterov/clamp + on-device line-search alpha; carry-
# optimized ~1 eval/step). Goal: confirm convergence (Overflow ≤ ~0.07) on the 16
# non-deferred configs (adaptec3_2048, bigblue2_2048 are the pre-existing stalls).
#
# Robustness (no-host is slower + the device soft-wedges after a timeout-kill):
#   - RESTART the container before each attempt (clean device state).
#   - FRESH JIT cache per config.
#   - RETRY (up to 3x) if no "Overflow=" or overflow>0.1 (transient blow-up).
#   - Generous 1800s timeout.
# Override configs by passing names as args.
C=bh-37-special-ayadav-for-reservation-82536
F=/localdev/ayadav/DREAMPlaceTT-Framework
PY=$F/DREAMPlace/dp_env/bin/python3
CONFIGS="${*:-adaptec1_512 adaptec1_1024 adaptec1_2048 adaptec2_512 adaptec2_1024 adaptec2_2048 adaptec3_512 adaptec3_1024 adaptec3_2048 bigblue1_512 bigblue1_1024 bigblue1_2048 bigblue2_512 bigblue2_1024 bigblue2_2048 bigblue3_512 bigblue3_1024 bigblue3_2048}"
SUM=$F/.verifymatrix_nohost_summary.txt; : > $SUM
echo "V22_NOHOST=drive sweep start $(date +%H:%M)  configs: $CONFIGS" | tee -a $SUM
for cfg in $CONFIGS; do
  ov=""; conv="NO"; ok=0
  for attempt in 1 2 3; do
    docker restart "$C" >/dev/null 2>&1; sleep 16
    docker exec "$C" bash -lc "rm -rf /dev/shm/ttcache_nh_* 2>/dev/null; sleep 2" 2>/dev/null
    docker exec "$C" bash -lc "
      export TT_METAL_HOME=$F/tt-metal TT_METAL_RUNTIME_ROOT=$F/tt-metal ARCH_NAME=blackhole
      export TT_METAL_CACHE=/dev/shm/ttcache_nh_${cfg}_${attempt}
      export DREAMPLACE_DIR=$F/DREAMPlace/install PYTHONPATH=$F/DREAMPlace/install:$F/DREAMPlaceTT/host/build:$F/integration
      export DPTT_DEVID=1 V22_NOHOST=drive
      timeout 1800 $PY $F/integration/run_dreamplace.py \
        --device scatter_ttnn_inprocess --benchmark $F/benchmarks/configs/sweep_${cfg}.json \
        --results-dir /tmp/nhsw_${cfg} > $F/.nhsw_${cfg}.log 2>&1
    " 2>/dev/null
    ov=$(grep -a "Overflow =" $F/.nhsw_${cfg}.log|tail -1|grep -oE "[0-9.]+$")
    if [ -n "$ov" ] && awk "BEGIN{exit !($ov<=0.1)}" 2>/dev/null; then ok=1; break; fi
    echo "  [retry] $cfg attempt=$attempt overflow=${ov:-NONE} (>0.1 or missing) -> retrying" | tee -a $SUM
  done
  hpwl=$(grep -a "HPWL     =" $F/.nhsw_${cfg}.log|tail -1|grep -oE "[0-9.e+]+$")
  if [ -n "$ov" ] && awk "BEGIN{exit !($ov<=0.0701)}" 2>/dev/null; then conv="YES"; fi
  case "$cfg" in adaptec3_2048|bigblue2_2048) note=" (DEFERRED baseline)";; *) note="";; esac
  echo "=== $cfg attempts=$attempt overflow=$ov hpwl=$hpwl converged=$conv$note" | tee -a $SUM
done
echo "VERIFYMATRIX-NOHOST DONE $(date +%H:%M)" | tee -a $SUM
