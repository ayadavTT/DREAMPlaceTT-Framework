# V11 scatter — tile-ownership route + accumulate density construction

> **TL;DR:** Each grid tile is *owned* by one Tensix core. The scatter kernel
> walks every cell's 8×8 bin neighborhood, looks up each bin's owner via a
> `tile_to_core[]` map, stages `(bx,by,area)` tuples in per-receiver L1 buffers,
> and bulk-NoC-writes them to per-(writer,reader) DRAM `route_buf` pages. A
> second kernel then accumulates each core's inbound route_buf into its local
> density slab and reduces. The validated pre-V19 production path.

---

## 1. Identity
- **Stage:** forward — density-map construction (scatter + route)
- **Status:** ⭐ SHIPPED (`GATHER_MODE=v11`, and `v11outer_auto` which routes the Stage-B′ outer-product variant by grid size)
- **Lineage:** V4 base → **V11 (tile-ownership route+accum)** → V18 (hash pre-agg) → V19 (L1 atomic-add, collapses the route+accum into one atomic). Stage B′ (`v11_outer_scatter`) is the SFPU-outer-product refinement of V11.
- **Source files:** `src/v11_scatter_dm.cpp` (452 ln, NCRISC), `src/v11_scatter_b_dm.cpp` (386 ln, BRISC), `src/v11_outer_scatter_dm.cpp` / `src/v11_outer_scatter_b_dm.cpp` (the Stage-B′ outer-product variant). Accumulate/reduce stage: see `02-forward-gather-accum/v11-accum/`.
- **Activated by:** `GATHER_MODE=v11` (or `v11outer_auto`). Built in-process by `host/v19_engine.cpp` (shared V11 setup) and the IPC server.

## 2. Problem & contract
**Computation.** Same density forward as V19: deposit `ox[j]·oy[k]` area into bin `(bxl+j, byl+k)` for every cell, summed → `D[M,N]`. V11 splits this into **route** (this kernel: send each contribution to its bin's owner) + **accumulate** (`v11_accum`: each owner sums its inbound contributions).

**Inputs (CBs from v4_compute, identical to V19):** `bxl`(c_4), `byl`(c_5), `ox[0..7]`(c_6..c_13), `oy[0..7]`(c_14..c_21). Plus the `tile_to_core[]` map + shard table (loaded from DRAM by NCRISC).

**Output:** `route_buf` DRAM pages — per (writer core, reader core) a page of `[uint32 count][pad][V11Contrib[count]]`, each `V11Contrib` = 16 B (8 B pad for 16-B NoC alignment). Consumed by `v11_accum`.

**Invariants:** L1 staging capped at `MAX_IN_FLIGHT` tuples/receiver; `route_buf` page = `16 + MAX_PER_PAGE_TUPLES*16` (≤4096 tuples with `SRC_CHUNK=8`, else L1 budget FATAL). Page header 16 B; tuples 16-B aligned even at odd counts.

## 3. Mechanism (how it works)
**Ownership.** The M×N tile grid is partitioned across the 110 cores via `tile_to_core[]`; each core owns a set of tiles (with optional hot-tile sharding). A contribution to bin `(bx,by)` is routed to that bin's tile's owner.

**RISC division of labor (one Program, BRISC+NCRISC same-core):**
- **NCRISC** (`v11_scatter_dm.cpp`): loads `tile_to_core` + shard table; per tile walks the 8×8 neighborhood for its cell half, stages `(bx,by,area)` into per-receiver L1 buffers, flushes a receiver buffer to its `route_buf` page when it fills (`MAX_IN_FLIGHT`), publishes per-tile CB read-ptrs to a shared struct, bumps `data_ready_sem`.
- **BRISC** (`v11_scatter_b_dm.cpp`): the reader (px/py/sx/sy → CBs) + scatters the other cell half; waits on `tables_ready_sem` then `data_ready_sem`.
- **Accumulate** (`v11_accum_dm.cpp`, separate program): each core loads `owned_lookup[]`, zeroes its `dense[]`, reads `route_buf[*][me]` in `SRC_CHUNK` batches, accumulates tuples into `dense[]`; shard owners write their shard tile to `shard_reduce_buf` and atomically signal the primary. Merged Phase 2+3 so the host pays **one** Finish() for the whole gather (the old 3-launch split cost ~16 ms in round-trips).

**THE TRICK(s):**
- **Tile-ownership routing** decouples scatter (any core writes) from accumulate (each owner reads only its inbound pages) — no cross-core atomics, deterministic.
- **Stage B′ (`v11_outer_scatter`)**: compute the 8×8 `ox⊗oy` outer product on the **SFPU** (12.4 ns/cell) instead of 64 scalar BRISC multiplies (~150 ns/cell). Wins by freeing NCRISC route cycles, not by the multiply itself — routing is ~85% of per-cell time. Net grid-dependent: −19.7% sc+ga @2048, wash @512 → shipped as `v11outer_auto` (B′ only for large grids).
- **G-PRESCALE**: pre-multiply `ox/oy` by `sqrt(inv_bin_area)` in v4_compute so `area=ox·oy` is already bin-area-normalized (removes a separate scale pass; gather −8.9%@512, −25%@2048).
- **G-PMERGE**: BRISC and NCRISC each merge half of `dense_b += dense_n` in parallel (−9.1% gather @2048).

## 4. Performance (MEASURED — isolated synthetic sweep, 2026-06-03; see `profile/results.csv`)
- **Scatter (route) ms, cell-bound:** adaptec1 1.16–1.27 / adaptec2 1.61–1.72 / bigblue1 1.57–1.70 / adaptec3 2.52–2.61 / bigblue2 2.56–2.69 / bigblue3 4.82–5.08. (Note: this is the **route** time only; V11 then needs a separate accum/gather stage — see `02-forward-gather-accum/v11-accum/` — so total sc+ga > V19's single atomic-scatter.)
- **Critical path:** NCRISC route (per-cell tuple staging + NoC flushes), confirmed by per-pass Tracy ([[v11_ox0_backpressure]]). Gather 3.0–17.8× max/median core imbalance ([[v11_gather_skew_diagnostic]]).
- **Scaling:** B′ routing helps only at 2048; hot-tile sharding default OFF.

## 5. Correctness (MEASURED — 48/54 PASS; 6 real failures)
- **PASS** for adaptec1/2/3, bigblue1/2 (all grids/dists) and bigblue3-uniform: rel_l2 ≤ 2.2e-5.
- **FAIL** at **bigblue3 (1.1M cells) + cluster/clouds** — rel_l2 **0.057–0.346** (lost density mass). Root cause = `route_buf` per-page tuple cap (`MAX_PER_PAGE_TUPLES`) **overflows on hot tiles** when 1.1M cells cluster → contributions dropped. Uniform spreads load → fits → passes. This is the measured proof of the [[v11_a2_perf_win]] cap warning, and **the reason V19's atomic-add (no cap) superseded V11.**
- HPWL convergence vs CPU at moderate scale ≈1.04× (HPWL 70.34M); Stage-B′ HPWL bit-exact 18/18.

## 6. Gotchas / pitfalls
- `MAX_PER_PAGE_TUPLES` too high → `FATAL: V11 accum scratch exceeds L1 budget`. Keep ≤4096 with `SRC_CHUNK=8`.
- Shrinking `max_per_page_tuples` 4096→256 gave scatter −54%; but `pg=128` DIVERGES (hot tiles overflow the cap → dropped mass).
- `CPU_DCT=1` required in the live loop (the V11 engine assumes host add-back), else HPWL diverges to 155M ([[cpu_dct_required_for_v11]]).
- Hot-tile sharding (K>1) can hurt wall time by sending DREAMPlace down longer convergence paths — keep default OFF.

## 7. When to use / avoid
- **Wins when:** you want a deterministic, contention-free scatter and can afford the route_buf DRAM staging; large grids benefit from B′.
- **Avoid:** at scale the route+accum + host de-stride is more expensive than V19's direct atomic-add → V19 superseded it for the production forward. Keep V11 as the validated reference / fallback.

## 8. Provenance
- **Memory:** [[v11_a2_perf_win]], [[v11_stage_bprime_outer_product]], [[v11_stage_bprime_sweep_outcome]], [[v11_g_prescale_win]], [[v11_g_pmerge_win]], [[v11_ox0_backpressure]], [[v11_gather_skew_diagnostic]], [[v11_gather_sharding_investigation]], [[v11_sweep_config]]
- **Handoff:** `docs/V11_PHASE3_HANDOFF.md`, `docs/V11_V13_GATHER_HANDOFF.md`
- **Host:** `host/v19_engine.cpp` (shared V11 setup) + `host/v11_tile_ownership.h`
