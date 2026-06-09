#!/bin/bash
# Regenerate the per-config per-stage timing table from the no-host run logs
# (.nhsw_<config>.log). Greps the median [FWD] + [BWD-V35] lines (FWD_TIMING /
# V35_TIMING are forced on by the production lock). Usage: bash tools/nohost_timing.sh
F=/localdev/ayadav/DREAMPlaceTT-Framework
printf "%-15s|%6s %7s %6s %7s|%6s %5s %6s %7s %7s %8s|%5s\n" \
  config h2d scatter DCT FWDtot count plan place gather unsort BWDtot iters
for cfg in adaptec1_512 adaptec1_1024 adaptec1_2048 adaptec2_512 adaptec2_1024 adaptec2_2048 \
           adaptec3_512 adaptec3_1024 adaptec3_2048 bigblue1_512 bigblue1_1024 bigblue1_2048 \
           bigblue2_512 bigblue2_1024 bigblue2_2048 bigblue3_512 bigblue3_1024 bigblue3_2048; do
  L=$F/.nhsw_${cfg}.log; [ -f "$L" ] || continue
  fwd=$(grep -a "^\[FWD\] " "$L"|tail -1); bwd=$(grep -a "\[BWD-V35\]" "$L"|tail -1)
  g(){ echo "$1"|grep -oE "$2=[0-9.]+"|cut -d= -f2; }
  printf "%-15s|%6s %7s %6s %7s|%6s %5s %6s %7s %7s %8s|%5s\n" "$cfg" \
    "$(g "$fwd" h2d)" "$(g "$fwd" scatter)" "$(echo "$fwd"|grep -oE 'DCT\(up/comp/dn\)=[0-9.]+/[0-9.]+/[0-9.]+'|awk -F'[=/]' '{print $5}')" \
    "$(g "$fwd" server)" "$(g "$bwd" count)" "$(g "$bwd" plan)" "$(g "$bwd" place)" \
    "$(g "$bwd" gather)" "$(g "$bwd" d2h)" "$(g "$bwd" bw_total)" \
    "$(echo "$fwd"|grep -oE 'n=[0-9]+'|cut -d= -f2)"
done
