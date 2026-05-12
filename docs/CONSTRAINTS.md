# Constraints & Design Spec — DREAMPlace on Tenstorrent Blackhole

This document is the **design spec / constraints contract** that any new V11
variant (V12, …) must respect. It also catalogs the constraints and pitfalls
we've discovered the hard way while bringing V11 up.

Read this before proposing a new kernel design.

---

## 0. The big picture

DREAMPlace's per-iteration electric-potential step is:

```
   positions (px, py, sx, sy)
         │
         ▼  ─────────────────────────────  scatter (cells → bins)
   density map ρ[M][N]
         │
         ▼  ─────────────────────────────  DCT solve (Poisson)
   potential ψ[M][N]
         │
         ▼  ─────────────────────────────  gather (bins → cells)
   per-cell forces (fx, fy)
```

We accelerate **scatter** and **DCT** on TT. (Gather is currently CPU-side; future work.) The fundamental challenge is that scatter is **input-dependent**: which cells overlap which bins changes every iteration as cells migrate. Any static partition of work that was load-balanced at iter 0 becomes imbalanced by iter 200.

This document enumerates what we **must** preserve in any new design, what we **cannot afford** to do, and the device-level capabilities we have to work with.

---

## 1. Hard constraints

These are non-negotiable. A design that violates one of these is not a candidate.

### 1.1 Mathematical correctness

The density at bin `(bx, by)` is:
```
ρ[bx][by] = (1 / bin_area) · Σ_cells   overlap_x(cell, bx) · overlap_y(cell, by)
```
where `overlap_x / overlap_y` is the 1-D Manhattan-style cell-bin overlap. Any kernel re-formulation must produce the **same** ρ within `rel_L2 < 1%` against a CPU reference (the `tools/v11_phase2_smoke.py` harness enforces this).

### 1.2 Host-side preprocessing budget

- **Allowed**: any preprocessing that is done **once** at run start and stays valid for the full DREAMPlace run. Examples that are fine:
  - Sort cells by `sx` (or any static cell property) — cell width never changes.
  - Build the `tile_to_core` ownership map.
  - Build initial `shard_table` from the first iter's histogram.
  - Pre-split mega-cells whose `sx > 7·bsx` into sub-cells (already done in `integration/scatter_ttnn_client.py`).
- **Not allowed**: per-iteration host preprocessing that scales with `nc` (cell count) or `M*N` (grid). The host's only per-iter job is to shovel `px, py, sx, sy` into shared memory.

Why: DREAMPlace runs 600–1000 iters. Per-iter host preprocessing adds 5–50 ms × 1000 iters and erases the TT speedup.

### 1.3 Load balance across all 110 Tensix cores

The chip has **110 Tensix worker cores**. Every per-iter kernel **must** keep all 110 busy. The chip is paid for at "max core wall-time"; the iteration's wall time is bounded by the slowest core, not the median.

**Concrete target**: `max_core_time / median_core_time < 1.5×` on every kernel phase (`V11A-ACC`, `V11N-ACC`, `V11-ROUTE`, etc.). Today this ratio is ~10× on hot benchmarks — see `docs/KNOWN_ISSUES.md` #1.

**Crucially**, the balance must hold for **every iteration**, not just iter 0. A partition that's balanced at iter 0 becomes imbalanced by iter 500 as cells migrate. The design must either:
- Be **inherently invariant** to cell positions (e.g. partition by cell-id, where load only depends on cell *count* per core, not where the cells are), **or**
- Have a **cheap re-balance** mechanism that runs every K iters and adapts to current cell positions (the histogram refresh approach — currently broken; see KNOWN_ISSUES.md #2).

### 1.4 L1 memory budget: 1.5 MB per core, hard cap

Each Tensix core has **1.5 MB** of L1 SRAM that holds CBs, kernel scratch, and per-core state. The TT-Metal runtime enforces this — exceeding it crashes the program at launch.

For the V11 accum kernel today (`grid 2048`, `n_owned_max = 38`):
| L1 region | Bytes | %  |
|---|---:|---:|
| `owned_lookup`        |   8 KB | 0.5 % |
| `inbound_hdrs` × 2    |  14 KB | 0.9 % |
| `inbound_buf` × 2     |1024 KB | 67 %  |
| 128 B safety gap      | 128 B  | <0.1 %|
| `dense_b` + `dense_n` |   436 KB | 28 %  |
| `tmp_shard`           |   4 KB | 0.3 % |
| **Total**             |**1481 KB** | **96 %** |

That's the maximum we can fit. Any new design **must** stay under 1.5 MB even at `grid 2048` with `max_owned_tiles = 38`. Designs that need ≥ 2 MB are dead on arrival.

### 1.5 Grid + cell-count range

The framework must support:
- Grid sizes from **`512 × 512`** to **`2048 × 2048`** (DREAMPlace allows others, but these are the standard sweep points).
- Cell counts from **~100k** (adaptec1) to **~2M** (bigblue3); future benchmarks go to 5M+.

Performance can taper at the extremes, but **correctness must hold everywhere**.

### 1.6 Default parallelization axis: cells

Unless you have a strong reason otherwise, **parallelize across cells**, not across bins. Reasons:
- Cells are equi-cost (same SFPU work per cell, regardless of position).
- The cell count is roughly known at startup and stable.
- Bins are not equi-cost: dense regions have many cells overlapping; empty regions have ~none.

A cell-parallel scatter where every core processes the same number of cells is **automatically load-balanced for the scatter step**. The imbalance creeps in at gather: some cores happen to own bins that many cells touched. The design challenge is to keep the **gather** balanced too, decoupled from where cells happen to be.

---

## 2. The fundamental scatter–gather problem (today)

V11 is a **cell-centric, tile-routed scatter** with a **bin-centric gather**.
- Scatter is balanced (cells per writer ≈ constant).
- Gather is **not** balanced. Receivers own tiles, and "hot tiles" with many cell overlaps make their owning core slow.

We've tried two mitigations, both incomplete:

| Mitigation | What it does | Why it isn't enough |
|---|---|---|
| **K-way hot-tile sharding** (`v11_tile_ownership.h::build_shard_table`) | Hot tiles are replicated across K receivers; scatter round-robins; primary sums shards back. | Shard table is built at iter 0 and stays static. By iter 500, *different* tiles are hot. The histogram-refresh mechanism that would adapt this is disabled due to a semaphore-leak bug (KNOWN_ISSUES.md #2). |
| **Per-page tuple cap** (`v11_max_per_page_tuples = 4096`) + **scatter-side dedup** (insertion sort + combine in `flush_recv`) | Caps the worst-case writer→receiver tuple count; coalesces duplicates from cells overlapping the same bin. | Cap is still hit on the very hottest pairs at grid 512 (only ~3 tiles per receiver). Dedup is per-64-tuple-batch, doesn't catch cross-batch duplicates. |

**The structurally correct fix** is one of:

1. **Refresh + sharding** (the existing design, fixed). Re-run the histogram every ~200 iters and rebuild the shard table. Requires re-using a single Program across refreshes so the per-core semaphore pool isn't exhausted.

2. **Bin-parallel gather** (re-formulate). Instead of "each core owns N bins", have each core own **a slice of cells** and accumulate locally into a dense buffer covering only the bins those cells touch. Then a global reduction step merges per-core dense buffers. This naturally balances *every* phase by cell count; the merge cost scales with `unique_bins` rather than `cells`.

3. **Two-phase reduce** ("tree-reduce on dense"). Each writer emits its full dense map for the bins it touched (sparse-then-dense). Pairs of writers exchange and sum. Log-depth tree reduction. Eliminates the cap-hit problem; cost is `O(n_writers × unique_bins_per_writer × log n_writers)` DRAM traffic.

4. **SFPU-accelerated dense accumulate**. Tuples → bucket-sort on bin → per-tile SFPU MAC. Reduces V11A-ACC scalar overhead 5–10× but doesn't help imbalance.

Any new V11-variant proposal must explain which of these (or a new fifth option) it picks and why.

---

## 3. Soft constraints / design principles

These are strong preferences but can be violated with justification.

### 3.1 Reuse the existing front-end

`v4_compute.cpp` (SFPU bin geometry) and `v4_reader.cpp` (DRAM streaming of `px/py/sx/sy`) are stable and well-tested. **Don't rewrite them.** They produce the same outputs any scatter variant needs: `bxl`, `byl`, `overlap_x[0..7]`, `overlap_y[0..7]` per cell.

### 3.2 One program launch per phase, where possible

Every `EnqueueMeshWorkload(cq, …); Finish(cq);` pair costs ~3–5 ms of host↔device round trip. V11 collapsed scatter+accum+reduce into 2 Programs (scatter + accum) by using on-chip NOC semaphores for inter-phase sync — see `docs/V11_PHASE3_HANDOFF.md`. **Prefer fewer Program launches** with on-chip sync over more launches with host barriers.

### 3.4 Build artifacts stay out of the repo

Server binary is rebuilt from source via `scripts/build_server.sh`. Kernels are JIT-compiled on server startup from source files in `kernels/`. No precompiled `.o` or `.so` should end up tracked.

---

## 4. Cumulative known constraints (what's bitten us)

Each entry below = a real bug or design corner we hit during V11 development.

| # | Constraint | Evidence | Workaround |
|---|---|---|---|
| 1 | **Per-page tuple cap silently drops tuples** | At grid 512, hot (writer, receiver) pairs produce > 2048 tuples; the cap at `v11_max_per_page_tuples/2` truncates the write. Mass deficit at iter 100 was 5 % (now 0.5 % with dedup); HPWL diverges 2–3× CPU baseline. | Raise cap to 4096 + halve `SRC_CHUNK` to fit; insertion sort + run-length combine in scatter's `flush_recv` (Option B). Doesn't fully solve grid 512. |
| 2 | **Per-core semaphore limit = 16** | `CreateSemaphore` inside the V11 dbg-first refresh block burns 1 merge_sem + up to `H_max_hot` shard_sems per refresh. After ~4 refreshes, the per-core pool is exhausted; subsequent launches deadlock. | Disabled refresh (`V11_HIST_REFRESH_ITERS = 1_000_000`). Real fix: reuse one Program's semaphores via `SetRuntimeArgs` only. |
| 3 | **Blackhole NOC writes overshoot up to 16 B** | NOC writes that aren't 32-B aligned can write 1–16 bytes past the requested length, corrupting neighboring data. | All `noc_async_write` sizes are rounded up to multiples of 32 bytes. V11Contrib is 8 bytes → pad to multiples of 4 tuples (= 32 B). Use 64-byte page headers (writes to byte 32+ stay within a single cache line). |
| 4 | **SFPU `floor` is 1-ULP imprecise** | `(int)floor((cx - xl) / bsx)` can return one bin index too low at exact boundaries. Caused cells to spread over the wrong bins on rare aligned coordinates. | `correct_bin_idx()` in `v4_compute.cpp` checks against exact bin edges and adjusts ±1. |
| 5 | **Mega-cells with `sx > 7·bsx` overflow the 8-wide overlap grid** | V4 SFPU produces 8 overlap_x and 8 overlap_y values, so a cell can span at most an 8×8 bin footprint. Cells wider than `7·bsx` must be sub-divided. | `scatter_ttnn_client.py` pre-splits cells with `sx > 7·bsx` into sub-cells before sending to the server. Done once at startup (cell sizes are static). |
| 6 | **Same-RISC LSU → NIU ordering is implicit** | We worried CPU stores to L1 might not be visible to the NIU's NOC reads, so added `asm("fence")` at 6 sites. **Result: no measurable effect.** Blackhole's small L1 cache is write-through; same-core LSU → NIU ordering is correct without explicit fences. | Don't add fences without a measurement showing they help. (Saved as a "dead end" in `docs/KNOWN_ISSUES.md` #5 so the next dev doesn't repeat.) |
| 7 | **No float atomic-add in shared L1** | BRISC and NCRISC share L1 but cannot both `dense[i] += a` to the same location safely. There's no atomic float-add instruction on Tensix. | Each RISC writes to its own dense slice (`dense_b` and `dense_n`); BRISC merges at the end after NCRISC signals via semaphore. Halves theoretical accumulator throughput per core but is the only correct option. |
| 8 | **Per-core L1 budget is tight at grid 2048** | `max_owned_tiles = 38` → dense = 152 KB × 2 RISCs = 304 KB, plus 256 KB × 2 for inbound_buf, plus ~25 KB overhead = 1.42 MB. Any growth in `MAX_K`, `n_owned`, or staging can crash startup with "v11_ac_scratch exceeds L1 budget". | Conservative `V11_CB_SLOT_HEADROOM = 16` (= 2 × MAX_K) to absorb hot-tile shards. New designs must compute this budget upfront and assert. |
| 9 | **TT-Metal's per-RISC dispatch order isn't program-ordered** | NCRISC fires before BRISC on the same Program. If BRISC needs NCRISC's setup, BRISC must explicitly wait (e.g. `noc_semaphore_wait(tables_ready_sem, 1)`). | V11 uses `tables_ready_sem` so BRISC scatter doesn't read `tile_to_core` / `shard_table` before NCRISC has loaded them. |
| 10| **JIT compile is slow on cold cache** | First-time server startup takes ~60 s to JIT all kernels. Subsequent runs hit the cache and start in ~2 s. | Smoke test harness blocks on `ready.flag` with a 60 s default timeout. Don't shorten this. |
| 11| **Container restart breaks IOMMU sysmem mapping** | After `docker restart`, the first kernel launch fails with `Expected NOC address: 0x1000000000000000, but got 0x1000000040000000`. | `tt-smi -r 1` chip reset. `scripts/reset_chip.sh`. |
| 12| **Server JIT cache invalidates on kernel-source hash, not path** | Edit a kernel file and restart the server → new JIT. But TT-Metal also has an in-process cache; rare hash collisions can let stale binaries persist. | If you've edited a kernel and behavior didn't change, `rm -rf ~/.cache/tt-metal`. |

---

## 5. Tensix capabilities reference

Each Tensix worker core has **three RISC processors** and **a compute unit** sharing one L1 bank. This section is the canonical reference for what each piece can do — used when designing new kernels.

### 5.1 BRISC (RISC-V data movement, NOC_0)

- **Role**: primary data movement core; usually owns DRAM reads and the user-facing kernel entry.
- **Instruction set**: RV32IMC.
- **NOC channel**: NOC_0 (the "even" NOC; has its own VC pool).
- **What it can do**:
  - `noc_async_read(src_noc_addr, dst_l1, size)` — bring DRAM or another core's L1 into our L1.
  - `noc_async_write(src_l1, dst_noc_addr, size)` — push our L1 to DRAM or another core.
  - `noc_async_read_barrier()` / `noc_async_write_barrier()` — block until in-flight ops on NOC_0 complete.
  - `noc_semaphore_inc(sem_noc_addr, val)` — atomic add to a remote semaphore.
  - `noc_semaphore_wait(sem_l1_ptr, target)` — local poll-spin on a semaphore in our L1.
  - `cb_reserve_back(cb, n)` / `cb_push_back` / `cb_wait_front` / `cb_pop_front` — producer/consumer on circular buffers in L1 shared with TRISC.
  - Direct L1 reads/writes (volatile `tt_l1_ptr` pointers) — coherent within the same core; visible to other cores' NOC reads via L1 SRAM.
- **What it cannot do**:
  - Issue NOC ops on NOC_1 (that's NCRISC's channel).
  - Run TRISC compute instructions directly. Compute work must flow through CBs.
  - Atomic float-add to shared L1 (no instruction exists; cross-RISC float updates require fences or per-RISC partitioned buffers).
- **Pipeline**: in-order, scalar; no SIMD; ~few hundred MIPS at 1.35 GHz.

### 5.2 NCRISC (RISC-V data movement, NOC_1)

- **Role**: secondary data movement core; runs in parallel with BRISC for 2× NOC throughput per core.
- **Identical RISC-V ISA** to BRISC.
- **NOC channel**: NOC_1 (independent VC pool from NOC_0).
- **Why it matters**: a core has **two independent NOC channels**, so a well-designed kernel can issue 2 NOC writes (or 1 read + 1 write) per cycle. V11 uses both — NCRISC scatter to writer half-0, BRISC scatter to writer half-1.
- **Same shared L1** as BRISC. Reads of L1 locations written by BRISC are coherent through L1 SRAM (no explicit fence needed on BH — see constraint #6).

### 5.3 TRISC (3 cores: unpack / math / pack)

- **Role**: the compute unit. Three TRISCs cooperate per Tensix.
  - **TRISC_0 (unpack)**: pull tiles from CBs into Tensix DEST registers.
  - **TRISC_1 (math)**: do the actual compute (FPU matmul, SFPU vector ops).
  - **TRISC_2 (pack)**: write DEST tiles back to a CB.
- **Tile granularity**: a *tile* is **32 × 32 = 1024 elements** (so 4 KB for float32). All TRISC compute is tile-grain.
- **FPU**: tile-grain matmul (`matmul_tiles`), tile add/sub/mul (`add_tiles`, `eltwise_*`). One tile per cycle peak.
- **SFPU** (Special Function Processing Unit): 32-lane SIMD per face (a face = 16×16 = 256 elements; 4 faces per tile). Used for:
  - Element-wise non-matmul ops: `exp`, `log`, `reciprocal`, `floor`, `abs`.
  - Conditional moves: `v_if(cond) {...} v_endif`.
  - Vector arithmetic in `vFloat` lanes.
  - **Cannot** do native **scatter** (write to a varying destination per lane). Gather-style ops work.
- **What TRISCs cannot do**:
  - Issue NOC operations directly. Must route through a RISC-V (typically NCRISC).
  - Free-form addressing — operations are tile-grain (1024 elements at a time).
- **Pipeline**: pipelined matmul/SFPU at high throughput; the cost is mostly the CB plumbing around it.

### 5.4 L1 SRAM (per core, 1.5 MB)

- **Layout**: byte-addressable from any RISC. CBs are carved out by `CreateCircularBuffer`. Kernel scratch is conventionally `CB_SCRATCH = c_24` (a single CB used as a flat L1 region).
- **Cache**: on Blackhole, the per-RISC L1 cache is tiny (4 lines × 16 B = 64 B total) and **write-through**. Cross-RISC visibility is via L1 SRAM, no flush needed. (Quasar arch is different — has a 4 KB write-back L1 D$ that needs `flush_l1_dcache(0)`; we don't run on Quasar.)
- **Coherence between BRISC and NCRISC on the same core**: writes by one RISC are visible to the other within a few cycles via L1 SRAM. Sync via:
  - A semaphore (`noc_semaphore_inc` from the producer, `noc_semaphore_wait` on the consumer), **or**
  - A direct L1 store + compiler barrier (`*sem = val; asm volatile("" ::: "memory");`).
- **DON'T** use `asm("fence")` or `invalidate_l1_cache()` — we tested, they're no-ops on BH for our access patterns.

### 5.5 NOC (Network-on-Chip)

- **Two physical networks**: NOC_0 (BRISC) and NOC_1 (NCRISC). Independent route tables and VC pools, so a Tensix can have two NOC ops in flight simultaneously.
- **Bandwidth**: nominal ~100 GB/s aggregate to DRAM, ~200 GB/s Tensix-to-Tensix at peak. Real throughput is dominated by **alignment** and **message size**:
  - 32-byte aligned writes are ideal.
  - Writes < 256 bytes are header-dominated; batch your tuples.
  - Unaligned writes can **overshoot up to 16 bytes** (KNOWN_ISSUES #3) — pad accordingly.
- **NOC reads** are blocking-ish: the issuing RISC can do other work, but to *use* the data you must `noc_async_read_barrier()`.
- **NOC writes** are fire-and-forget: `noc_async_write_barrier()` waits for **all** in-flight writes on that NOC to complete. Beware: barriering on NOC_0 doesn't drain NOC_1.

### 5.6 Semaphores

- **Quantity**: 16 per Tensix core, total per Program. Allocated via `CreateSemaphore(prog, core_range, init_val)`.
- **Operations**:
  - `noc_semaphore_inc(sem_noc_addr, val)` — atomic add to a semaphore *anywhere on chip* via the NOC. Used for cross-core signaling.
  - `noc_semaphore_wait(sem_l1_ptr, target)` — local poll-spin (`while (*sem != target);`).
  - `noc_semaphore_wait_min(sem_l1_ptr, target)` — same, but stops at `*sem >= target`.
  - `noc_semaphore_set(sem_l1_ptr, val)` — direct L1 write to reset the semaphore for the next launch.
- **Trap (KNOWN_ISSUES #2)**: each `CreateSemaphore` call allocates from a fixed per-core pool. Recreating a `Program` and re-calling `CreateSemaphore` does NOT free the previous program's slots — once you've allocated 16 per core across a session, you're done.

### 5.7 DRAM

- **Banked, interleaved**: Buffers allocated with `InterleavedAddrGen` are spread across DRAM banks one page per bank.
- **Page-size determines stride** through banks. Choose page size to maximize sequential single-bank reads / writes.
- **Total**: 4 GB+ usable on Blackhole. Allocations beyond `2^32 = 4 GB` size_t can hit cap checks — `route_buf` has a runtime check for this.

### 5.8 Profiler (`DeviceZoneScopedN`)

- Drop `DeviceZoneScopedN("MY-NAME")` inside any kernel scope. Captured when `TT_METAL_DEVICE_PROFILER=1` at server startup. CSV written to `$TT_METAL_HOME/generated/profiler/.logs/`.
- Aggregated by `tools/profile_v11.py` (see `docs/PROFILING.md`).
- **Cost**: ~100 cycles per zone entry+exit. Free if profiler not enabled.

### 5.9 What you can't do (anti-features to remember)

- No float atomic add in shared L1 → can't have BRISC and NCRISC simultaneously `+=` to the same dense buffer.
- No native scatter in SFPU → can't do `dst[varying_idx_per_lane] += value` directly. Must bucket-sort or use scalar accumulation.
- No per-iter host preprocessing (constraint 1.2).
- No more than 16 semaphores per core per session.
- No more than ~1.4 MB of scratch per core (constraint 1.4).
- No NOC ops from TRISC compute (data movement must go through a RISC).
- No tile-grain addressing inside the dense buffer of a Tensix core when iterating with scalar RISC code — that's `dense[idx]++` on plain RISC-V, ~1 op per element.

---

## 6. Acceptance tests for any new V11 variant

A new design needs to clear these bars before replacing the current V11 path:

1. **Smoke** (`scripts/run_smoke.sh`): `rel_L2 < 1%` against CPU reference on synthetic 1.5M cells, grid 2048.
2. **Convergence**: at least matches V11-current on the 4 sweep benchmarks (no regression on `adaptec1_2048`).
3. **Profile** (`tools/profile_v11.py`): `max_core_time / median_core_time < 3.0×` on the hottest zone for the 4 benchmarks.
4. **L1 footprint**: prints `v11_ac_scratch = <KB>` at startup; must be `< 1500 KB` on `grid 2048, n_owned_max ≤ 40`.
5. **Compile**: builds inside the standard container without changing the global `CMakeLists.txt` or `clang` version.
6. **End-to-end perf**: scatter+gather wall time on `adaptec1_2048` is `≤ 1.2 × V11-current` (no major perf regression).

---

## 7. Open design questions

These are the things to think about when proposing the next variant:

1. **Where does the load come from?**
   - The accumulator inner loop on the slowest core is ~21 ms; the median is ~2 ms. The 10× ratio comes from one or two cores owning hot tiles. What partitioning makes this O(1) instead of O(hot-tile-mass)?
2. **Can we move from a bin-centric receiver model to a cell-centric receiver model?**
   - Today: bins are partitioned across receivers; cells route to receivers based on which bins they touch. Slow because cells cluster on bins.
   - Alternative: cells are partitioned across receivers; receivers accumulate locally into a dense map of *only the bins their cells touched*. Then a reduction phase merges.
   - This is option 2 in section 2. Open: does the reduction fit in L1 / DRAM? At grid 2048 and 38 tiles/receiver, ~ 38 × 1024 = 38912 unique bins per receiver × 4 B = 156 KB. That fits.
3. **Is sharding salvageable?**
   - The mechanism exists but needs the semaphore leak fixed. Worth ~1 day of refactor.
4. **Can SFPU do the accumulate?**
   - Per-tile dense MAC after a bucket-sort. We don't have a bucket-sort kernel for tuples yet.
5. **What about gathering forces, not just scattering density?**
   - The framework offloads scatter+DCT to TT but does gather on CPU. Moving gather to TT would unlock another ~20 % iteration speedup. See `docs/FAST_SCATTER_GATHER_PLAN.md` for prior thinking.

If you're proposing a new variant, please write up a short design doc (≤ 2 pages) that says which of these questions you're answering, why, and how the design satisfies sections 1, 2, and 4 above.

---

## Appendix A — Where the numbers in this doc come from

- Per-core L1 budget breakdown (§ 1.4): from `host/density_scatter_ttnn_server_host.cpp` `v11_dense_offset_bytes` calculation, instrumented with `printf("...v11_ac_scratch = %u KB ...")` and observed on `sweep_adaptec1_2048` startup.
- `V11A-ACC` 21 ms slowest core / 2 ms median (§ 2): from `TT_METAL_DEVICE_PROFILER=1` traces on `sweep_adaptec1_2048` at iter 100, aggregated with `tools/profile_v11.py`.
- Mass deficit numbers (§ 4 #1): from `tools/cpu_vs_tt_density.py` on `sweep_adaptec1_512` at iter 100 with the iter-100 position dump.
- Semaphore-leak observation (§ 4 #2): empirical — we set `V11_HIST_REFRESH_ITERS = 200` and the server died with `n_ep_calls=0` after ~3 refreshes.
- 16-byte NOC overshoot (§ 4 #3): TT-Metal team comments + symptomatic corruption when adjacent 32-byte L1 destinations shared a 64-byte cache line.

If a future change invalidates one of these numbers, please update the corresponding section here.
