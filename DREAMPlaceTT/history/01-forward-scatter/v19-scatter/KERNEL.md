# V19 scatter — direct L1 atomic-add density-map construction

> **TL;DR:** Each cell's bin overlaps are atomic-added (`noc_semaphore_inc`) as
> fixed-point area directly into the *owning core's* L1 density slab — no
> `route_buf`, no staging, no separate gather kernel. Block-partition ownership
> lets the writeout dump each core's slab straight to row-major DRAM. This is the
> **shipped production forward** density path.

---

## 1. Identity
- **Stage:** forward — density-map construction (scatter)
- **Status:** ⭐ SHIPPED (production forward; `GATHER_MODE=v19`)
- **Lineage:** V4 base scatter → V11 tile-owner route+accum → V18 hash pre-agg → **V19 (L1 atomic-add)**. Successor experiments: V19-mb block-permute (V20), V31 geom-stash (adds a per-cell geometry stash on top of this kernel for the backward).
- **Source files:** `src/v19_scatter_dm.cpp` (216 ln, NCRISC), `src/v19_scatter_b_dm.cpp` (187 ln, BRISC). Pairs with **v19-writeout** (`02-forward-gather-accum/v19-writeout/`).
- **Activated by:** `GATHER_MODE=v19`. Host builds the program in `host/v19_engine.cpp` (`use_v19` branch, ~L1640–1680 for the scatter program `prog_v11_sc`; writeout program `prog_v19_wo` ~L1567–1601).

## 2. Problem & contract
**Computation.** DREAMPlace density forward: each movable/filler cell, placed at `pos` with size `(sx,sy)`, overlaps a small set of grid bins. The fractional x-overlap of cell *c* with bin column `bxl+j` is `ox[j]`, the y-overlap with bin row `byl+k` is `oy[k]`; the area deposited into bin `(bxl+j, byl+k)` is `ox[j]·oy[k]`. The density map `D[M,N]` is the sum of these areas over all cells. V19 builds `D` on-chip.

**Inputs:**
- DRAM tiles `px,py,sx,sy` (cell positions/sizes), one tile = 1024 cells, interleaved, `page_size = tile_pgsz`. Read by the BRISC reader (CBs `c_0..c_3`).
- Per-tile, per-cell geometry produced by the **v4_compute** SFPU stage into CBs: `bxl` (`c_4`), `byl` (`c_5`), `ox[0..7]` (`c_6..c_13`), `oy[0..7]` (`c_14..c_21`). `bxl/byl` = base bin indices; `ox[j]/oy[k]` = the (already `sqrt(inv_bin_area)`-prescaled) overlap fractions. Up to `V11_MAX_OVERLAP=8` bins per axis.
- Common runtime args: `noc_coords[0..nc_all-1]` — packed `(x|y<<16)` NoC coord of each owner core (broadcast via `SetCommonRuntimeArgs`).
- Key runtime args: `nc_all` (#owner cores=110), `nbx,nby` (grid bins), `n_tiles`, `density_l1_off`, `density_slab_bins`, `scale_bits`.

**Output:** each core's L1 density slab `density_l1[0..density_slab_bins)` as **uint32 fixed-point** (area × `2^scale_bits`). Consumed by v19-writeout.

**Invariants / preconditions:**
- `density_slab_bins` is padded to a multiple of 8 (host) so DRAM page-size + de-stride stay in sync (fix #4 below).
- Block partition: core `c` owns global bins `[c·slab_bins, (c+1)·slab_bins)`.
- `noc_coords` must be delivered via common runtime args, **not** a kernel-side DRAM read (fix #3 below).

## 3. Mechanism (how it works)
**Data residency.** The density slab + the `noc_coords` table both live in `CB_SCRATCH` (`c_24`): slab at `density_l1_off`, coords immediately after at `(density_l1_off + slab_bins*4 + 31) & ~31` (32-B aligned). A `ScatterShared` struct at `shared_state_off` carries the L1 pointers of the current tile's `bxl/byl/ox/oy` CB buffers from NCRISC → BRISC.

**RISC division of labor (both run the same atomic-add loop on different cell halves):**
- **NCRISC** (`v19_scatter_dm.cpp`): owns cells `[0..512)` of each tile. Also does **init** (`V19N-INIT`): zeroes its own slab and copies `noc_coords` from common args into L1; signals BRISC via `tables_ready_sem`. Per tile: waits on the v4_compute CBs (`V19N-CB-WAIT`), publishes the CB read-pointers into `ScatterShared`, bumps `data_ready_sem`, then runs the atomic loop (`V19N-ATOMIC`).
- **BRISC** (`v19_scatter_b_dm.cpp`): is the **reader** (Phase 1: `noc_async_read_page` of px/py/sx/sy → CBs, zone `V19B-DRAM-READ`) *and* the scatterer for cells `[512..1024)` (Phase 2: `V19B-ATOMIC`). It's software-pipelined: tile *t* is read while tile *t-1* is scattered. It consumes the shared pointers NCRISC published.
- **TRISC (v4_compute, separate kernel)** produces `bxl/byl/ox/oy` on the SFPU — not part of V19's two files but upstream in the same program.

**The atomic-add inner loop (identical on both RISCs).** For each cell `ci`:
1. Clamp base bins `bxl_u,byl_u` to `[0,nbx/nby)`.
2. Walk `j=0..7` x-overlaps and `k=0..7` y-overlaps, with an early-out: two consecutive `ox[j]<=0` (or `oy[k]<=0`) breaks the loop (cells rarely span all 8 bins).
3. `area_fixed = (uint32)(ox[j]*oy[k] * 2^scale_bits)`; skip if 0.
4. `bin_global = bx_val*nby + by_val`; `owner_idx = bin_global / slab_bins`; `local_idx = bin_global - owner_idx*slab_bins`.
5. Look up `owner_x,owner_y` from `noc_coords[owner_idx]`, form the NoC address of that core's `density_l1[local_idx]`, and **`noc_semaphore_inc(target, area_fixed)`** — a hardware atomic add over the NoC.
6. After the loop, **one** `noc_async_atomic_barrier()`.

**THE TRICK (two of them):**
- **Direct cross-core atomic-add gather.** Skip the entire V11 route_buf→accum→reduce pipeline; each contribution is a single hardware atomic into the owner's L1. Determinism + correctness come from `noc_semaphore_inc` (blind atomic add is *exact* at any contention; see [[atomic_fetchadd_contention_deadlock]]) plus a separate writeout kernel so all atomics settle before any read.
- **Block-partition ownership.** Owner = `bin_global / slab_bins` (contiguous block), not `bin_global % nc_all` (stride-1). Stride-1 minimizes atomic contention but forces a costly host de-stride at readout; block partition lets writeout dump each slab straight to row-major DRAM. Microbench @2048: contention costs ~+1 ms/scatter, far less than the ~15–20 ms host de-stride it removes.

## 4. Performance (MEASURED — isolated synthetic sweep, 2026-06-03; `profile/results.csv`)
- **54/54 configs PASS** (6 designs × {512,1024,2048} × {uniform,cluster,clouds}), rel_l2 ≤ 2.4e-5.
- **Scatter ms (median), cell-bound & grid-insensitive & distribution-insensitive:**
  adaptec1 (211K) 1.45–1.58 · adaptec2 (255K) 2.04–2.17 · bigblue1 (278K) 2.00–2.16 · adaptec3 (451K) 3.22–3.37 · bigblue2 (558K) 3.20–3.36 · bigblue3 (1.10M) 6.07–6.47.
- **Key measured finding:** cost scales ~linearly with #cells, barely moves with grid (512→2048: +5–10%), and is **insensitive to clustering** — unlike v11 (whose route_buf cap overflows on clustered bigblue3). The atomic-add design has no per-page cap, so it stays correct + stable across all distributions.
- **Critical path:** per-cell BRISC/NCRISC atomic loop; NoC/atomic traffic bound (not SFPU).
- **Per-zone Tracy:** zones instrumented (`V19N-INIT/CB-WAIT/ATOMIC`, `V19B-DRAM-READ/ATOMIC`); profiler IS built in (`ENABLE_TRACY=ON`). `zones.txt` pending a `dump_profiler()` + safe-grid capture (see `profile/PROFILE.md`).

## 5. Correctness
- **Method:** HPWL convergence vs CPU 16T + cross-run determinism.
- **Result:** adaptec1_512 HPWL **70.40M vs CPU 70.25M (+0.22%)**; all 6 designs @512 converge within ±0.51% ([[dreamplacett_validated_512]]). Drift <0.15% vs V11 across grids ([[v19_works_all_grids]]).

## 6. Gotchas / pitfalls (all are real bugs that were fixed here)
1. **Writeout must be a *separate* kernel.** In-kernel writeout raced cross-core atomics → 4–8 ULP cross-run diffs → broke convergence. Split into v19-writeout so all atomics land first.
2. **CB allocation order must match between scatter and writeout programs** so `c_24` lands at the same L1 base address.
3. **Deliver `noc_coords` via `SetCommonRuntimeArgs`, NOT a kernel-side `noc_async_read`.** At 2048, 110 cores reading the same DRAM page into high-L1-offset destinations silently returned shifted/garbage data → bogus NoC targets → `noc_async_atomic_barrier` hung. ([[v19_works_all_grids]])
4. **Pad `density_slab_bins` to a multiple of 8** so DRAM page-size + readback de-stride stay in sync.
5. Use **`noc_semaphore_inc`** (static VC), never raw `noc_atomic_*`; and **count, don't fetch-add a return slot** — fetch-add return values double-read under cross-core contention. ([[atomic_fetchadd_contention_deadlock]])

## 7. When to use / avoid (the lesson)
- **Wins when:** you need a deterministic, low-host-overhead density scatter at any grid; atomic contention from spatial clusters is cheap relative to staging/de-stride.
- **Avoid / watch:** extreme clustering creates hot owner cores (load imbalance) — but still correct. If you need the per-cell geometry later (backward), use the **V31 geom-stash** variant which stashes `ox·oy` during this scatter instead of discarding it.
- **What it replaced:** V11's route_buf→accum→reduce (more kernels, host de-stride). V19 collapses that to atomic-add + on-chip fp32 writeout.

## 8. Provenance
- **Memory:** [[v19_l1_atomic_converges]], [[v19_works_all_grids]], [[v19_block_fp32_outcome]], [[atomic_fetchadd_contention_deadlock]], [[blackhole_dram_read_align_64]], [[dreamplacett_validated_512]]
- **Handoff docs:** `docs/V19_L1_ATOMIC_HANDOFF.md`, `docs/V19_RUNG3_INPROCESS_HANDOFF.md`
- **Host entry:** `host/v19_engine.cpp` (`use_v19` program build); pybind `host/v19_engine_pybind.cpp::scatter_from_pos`.
