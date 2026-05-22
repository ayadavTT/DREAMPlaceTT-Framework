#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""V11 vs V11-outer sweep comparison report."""
import json
import os
import sys
from glob import glob

DIR = sys.argv[1] if len(sys.argv) > 1 else "results/v11outer_sweep_20260520_2204"

# Group results by config.
results = {}
for d in sorted(glob(f"{DIR}/v11_sweep_*")) + sorted(glob(f"{DIR}/v11outer_sweep_*")):
    name = os.path.basename(d)
    if name.startswith("v11outer_"):
        mode, cfg = "v11outer", name[len("v11outer_"):]
    else:
        mode, cfg = "v11", name[len("v11_"):]
    f = os.path.join(d, f"{cfg}_scatter_ttnn_metrics.json")
    if not os.path.exists(f):
        continue
    m = json.load(open(f))
    results.setdefault(cfg, {})[mode] = m

# Header.
hdr = f"{'config':<22}  {'mode':<10} {'HPWL':>15}  {'overflow':>9}  {'sc_ms':>7}  {'ga_ms':>7}  {'sc+ga':>7}"
print(hdr)
print("-" * len(hdr))

total_v11_scga = 0.0
total_v11out_scga = 0.0
converged_match = 0
total = 0
sc_deltas = []
ga_deltas = []
scga_deltas = []

for cfg in sorted(results.keys()):
    for mode in ("v11", "v11outer"):
        m = results[cfg].get(mode)
        if m is None:
            print(f"{cfg:<22}  {mode:<10}  MISSING")
            continue
        sc = m.get("scatter_ttnn_scatter_ms_median", 0.0)
        ga = m.get("scatter_ttnn_gather_ms_median", 0.0)
        scga = sc + ga
        hpwl = m.get("hpwl", 0.0)
        ov = m.get("overflow", 0.0)
        print(f"{cfg:<22}  {mode:<10} {hpwl:>15,.0f}  {ov:>9.4f}  {sc:>7.2f}  {ga:>7.2f}  {scga:>7.2f}")
    # Delta line for this config.
    if "v11" in results[cfg] and "v11outer" in results[cfg]:
        v = results[cfg]["v11"]
        vo = results[cfg]["v11outer"]
        v_sc = v.get("scatter_ttnn_scatter_ms_median", 0)
        v_ga = v.get("scatter_ttnn_gather_ms_median", 0)
        vo_sc = vo.get("scatter_ttnn_scatter_ms_median", 0)
        vo_ga = vo.get("scatter_ttnn_gather_ms_median", 0)
        sc_pct = ((vo_sc - v_sc) / v_sc * 100) if v_sc else 0
        ga_pct = ((vo_ga - v_ga) / v_ga * 100) if v_ga else 0
        scga_pct = ((vo_sc + vo_ga - v_sc - v_ga) / (v_sc + v_ga) * 100) if (v_sc + v_ga) else 0
        match = "match" if v["hpwl"] == vo["hpwl"] else f"DIFF (v11={v['hpwl']:,.0f} v11out={vo['hpwl']:,.0f})"
        print(f"{'':<22}  {'Δ':<10} {match:>15}  {'':<9}  {sc_pct:>+6.1f}%  {ga_pct:>+6.1f}%  {scga_pct:>+6.1f}%")
        if v["hpwl"] == vo["hpwl"]:
            converged_match += 1
        sc_deltas.append(sc_pct)
        ga_deltas.append(ga_pct)
        scga_deltas.append(scga_pct)
        total += 1
        total_v11_scga += v_sc + v_ga
        total_v11out_scga += vo_sc + vo_ga
    print()

# Summary.
if total:
    print("=" * 80)
    print(f"Summary across {total} configs:")
    print(f"  HPWL bit-exact match: {converged_match}/{total}")
    print(f"  Mean scatter Δ:   {sum(sc_deltas)/len(sc_deltas):+.2f}%")
    print(f"  Mean gather Δ:    {sum(ga_deltas)/len(ga_deltas):+.2f}%")
    print(f"  Mean sc+ga Δ:     {sum(scga_deltas)/len(scga_deltas):+.2f}%")
    print(f"  Total V11 sc+ga: {total_v11_scga:.2f} ms")
    print(f"  Total V11out sc+ga: {total_v11out_scga:.2f} ms")
    print(f"  Total delta: {total_v11out_scga - total_v11_scga:+.2f} ms ({((total_v11out_scga - total_v11_scga)/total_v11_scga*100):+.2f}%)")

    # Sort scatter deltas by config for grid-size breakdown
    print()
    print("By grid size (scatter Δ):")
    for grid in ("512", "1024", "2048"):
        ds = []
        for cfg in sorted(results.keys()):
            if not cfg.endswith(f"_{grid}"): continue
            if "v11" in results[cfg] and "v11outer" in results[cfg]:
                v = results[cfg]["v11"]; vo = results[cfg]["v11outer"]
                v_sc = v.get("scatter_ttnn_scatter_ms_median", 0)
                vo_sc = vo.get("scatter_ttnn_scatter_ms_median", 0)
                if v_sc: ds.append((vo_sc - v_sc) / v_sc * 100)
        if ds:
            print(f"  {grid:>4}-grid: {len(ds)} configs, mean sc Δ = {sum(ds)/len(ds):+.2f}%  (min {min(ds):+.2f}%, max {max(ds):+.2f}%)")
