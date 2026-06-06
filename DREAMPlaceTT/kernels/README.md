# TT-Metal Kernels — live production set

This directory holds **only the kernels the production density pipeline JIT-compiles
at runtime** (V19 scatter + TT DCT + V35 backward), plus the V22 on-chip optimizer
kernels kept for the planned optimizer fusion. Every other kernel ever written lives —
documented + archived — under `../history/<stage>/<version>/`.

> **The authoritative pipeline map (flow, per-stage cost, optimization frontier) is
> [`../PIPELINE.md`](../PIPELINE.md).** This file is just the kernel inventory.

Blackhole p150b: 13×10 = 130 Tensix worker cores, each with BRISC + NCRISC + 3× TRISC
(compute) RISCs and a 1.5 MB L1 bank. Cells are batched 1024 per tile. Kernels are JIT-
compiled from this dir by path (`DENSITY_KERNEL_DIR`); to change one, edit it and clear
`$TT_METAL_CACHE` — no engine rebuild needed.

## Contents (28 `.cpp` + 1 `.h`)

**Forward — V19 scatter + density (9)**
```
 v4_reader              (BRISC)  read px,py,sx,sy SoA tiles → CBs
 v4_compute             (TRISC)  SFPU: bin (bxl,byl) + clamped triangle overlaps ox[0..7]/oy[0..7]
 v11_histogram          (—)      per-tile cell counts → shard table (one-time, iter 0)
 v31_scatter_stash_dm   (NCRISC) cells 0..511   : L1 atomic-add density + 128-B per-cell geom stash
 v31_scatter_b_stash_dm (BRISC)  cells 512..1023: same, BRISC half
 v11_accum_dm           (BRISC)  accumulate routed density (owner half b) + merge
 v11_accum_n_dm         (NCRISC) accumulate routed density (owner half n)
 v19_writeout_fp32_dm   (BRISC)  scale by 1/(bsx·bsy) + write fp32 density tile → DRAM
 v19_writeout_void_ncrisc (NCRISC void — pairs the BRISC writeout)
```
(The DCT is TTNN's built-in tilize/matmul/untilize kernels, not in this dir.)

**Backward — V35 halo-tile gather (5)**
```
 v35_count        per-tile counts for the grouping plan
 v35_place        scatter cells into one tile-grouped buffer (PASS 2)
 v35_gather_brisc (BRISC) load fx band page-by-page, build px/ratio/base_idx, drain oidx
 v28_ncrisc       (NCRISC) load fy band, drain gx/gy to DRAM
 v28_compute      (TRISC SFPU) gx/gy = ratio·Σ px[k]·py[h]·f[k,h]
```

**Backward — V21 exact fallback (3, prewarmed; not the gradient path)**
`v21_ef_brisc` / `v21_ef_ncrisc` / `v21_ef_compute` (+ `v21_ef_common.h`). V21 is the
slower **exact** on-chip backward (cos 1.0 vs CPU). The production lock prewarms it but
routes the actual gradient through the faster V35 gather. Use it as the oracle
(`DBG_V35_EF=0 DBG_V21_EF_USE_CPU_GRAD=1` → TT forward + CPU backward).

**On-chip optimizer — V22 (11, not yet wired)**
`v22_clamp_compute`, `v22_elt_reader/writer`, `v22_gradcombine_compute`,
`v22_nesterov_compute/reader/writer`, `v22_precond_compute`,
`v22_stepsize_compute/reader/writer`. Kept here for the planned **optimizer fusion** that
would keep the gradient on-device and eliminate the backward `d2h` grad-unsort — the
biggest residual cost (see `../PIPELINE.md` §4).

## Archive & isolation testing
- **Every past + present kernel** is documented in `../history/<stage>/<version>/KERNEL.md`
  with `src/` + (where captured) `profile/`. Stages: `01-forward-scatter` …
  `08-dct-solve`, `99-infra-probes`. The shipping kernels above also appear there as their
  documented version (forward scatter+stash → `03-forward-geom-stash/v31-stash` +
  `01-forward-scatter/{v4-base,v19-scatter}`; V35 backward → `05-backward-ef-gather/
  {v35-halotile,v28-ef-multibin,v21-ef}`; DCT → `08-dct-solve`).
- **Isolated per-kernel microbenches:** `../history/harness/` (`run_harnesses.sh`), with its
  own vendored `kernels/` so it stays runnable independent of this production dir.

Note: the engine source (`../host/v19_engine.cpp`) still contains gated `CreateKernel(...)`
calls for removed experimental kernels (v6/v13/v14/…); those branches are dead under the
production lock and were intentionally not pruned. Only the kernels actually loaded are
kept here.
