#!/usr/bin/env python3
"""Compute offset_x / ratio statistics for adaptec1 at grid 512 vs 2048 to
verify the magnitude of the bug where scatter_ttnn_client sends raw `pos`
and `node_size_x_clamped` to TT without applying CPU's `+ offset_x` shift
or `* ratio` density correction."""

import os
import sys
import math

# Make DREAMPlace importable
os.environ.setdefault("DREAMPLACE_DIR",
    "/localdev/ayadav/tt-work/TTPort/DREAMPlaceTT-Framework/DREAMPlace/install")
sys.path.insert(0, os.environ["DREAMPLACE_DIR"])
sys.path.insert(0, os.path.join(os.environ["DREAMPLACE_DIR"], "dreamplace"))

import numpy as np
import dreamplace.Params as Params
import dreamplace.PlaceDB as PlaceDB

cfg_path = "/localdev/ayadav/tt-work/TTPort/DREAMPlaceTT-Framework/benchmarks/configs/adaptec1_512.json"

# Change cwd to install dir BEFORE loading params (matches run_dreamplace.py behavior)
os.chdir(os.environ["DREAMPLACE_DIR"])

params = Params.Params()
params.load(cfg_path)
# aux_input is already relative to install dir; don't rewrite it

placedb = PlaceDB.PlaceDB()
placedb.read(params)

xl, yl, xh, yh = placedb.xl, placedb.yl, placedb.xh, placedb.yh
chip_w = xh - xl
chip_h = yh - yl
n_mov = placedb.num_movable_nodes
n_term = placedb.num_terminals
n_total = placedb.num_physical_nodes  # mov + term
n_fil = placedb.num_filler_nodes

# node_size_x is the original size; placedb has it for ALL nodes
nsx_all = np.asarray(placedb.node_size_x, dtype=np.float64)
nsy_all = np.asarray(placedb.node_size_y, dtype=np.float64)
print(f"Adaptec1 chip dimensions: ({chip_w:.1f}, {chip_h:.1f})")
print(f"  num_movable={n_mov}, num_terminals={n_term}, num_filler={n_fil}, total_phys={n_total}")
print(f"  node_size_x array length: {len(nsx_all)}")

# Filler sizes are not stored in placedb directly (filler is dynamic);
# they're set to fill the remaining whitespace area uniformly.
# DREAMPlace creates fillers with size = filler_size_x/y in PlaceObj.
# For our purposes, look at the SAVE arrays in placedb.

# Compute for both grid 512 and 2048
sqrt2 = math.sqrt(2)
print()
for ng in [512, 1024, 2048]:
    bsx = chip_w / ng
    bsy = chip_h / ng
    clamp_x = bsx * sqrt2
    clamp_y = bsy * sqrt2

    # Movable + terminals stats
    sx_clamped = np.maximum(nsx_all, clamp_x)
    sy_clamped = np.maximum(nsy_all, clamp_y)
    offset_x = (nsx_all - sx_clamped) * 0.5   # negative if stretched
    offset_y = (nsy_all - sy_clamped) * 0.5
    ratio_arr = (nsx_all * nsy_all) / (sx_clamped * sy_clamped)

    # Subset: only movable cells (first n_mov)
    mov_offx = offset_x[:n_mov]
    mov_offy = offset_y[:n_mov]
    mov_ratio = ratio_arr[:n_mov]
    mov_stretched = (nsx_all[:n_mov] < clamp_x) | (nsy_all[:n_mov] < clamp_y)

    # Subset: terminals
    term_offx = offset_x[n_mov:n_mov + n_term]
    term_ratio = ratio_arr[n_mov:n_mov + n_term]
    term_stretched = (nsx_all[n_mov:n_mov + n_term] < clamp_x) | (nsy_all[n_mov:n_mov + n_term] < clamp_y)

    print(f"=== grid {ng}: bsx={bsx:.3f}, clamp_thresh={clamp_x:.3f} (=bsx*sqrt2) ===")
    print(f"  MOVABLE ({n_mov} cells):")
    print(f"    {mov_stretched.sum()}/{n_mov} ({mov_stretched.mean()*100:.1f}%) have stretching")
    print(f"    offset_x:  min={mov_offx.min():.3f}  median={np.median(mov_offx):.3f}  max={mov_offx.max():.3f}")
    print(f"    |offset_x| in bin_units: max={abs(mov_offx).max()/bsx:.2f}")
    print(f"    ratio:     min={mov_ratio.min():.4f}  median={np.median(mov_ratio):.4f}  max={mov_ratio.max():.4f}")
    if n_term > 0:
        print(f"  TERMINALS ({n_term} fixed):")
        print(f"    {term_stretched.sum()}/{n_term} have stretching")
        print(f"    offset_x:  min={term_offx.min():.3f}  max={term_offx.max():.3f}")
        print(f"    ratio:     min={term_ratio.min():.4f}  max={term_ratio.max():.4f}")
    print()
