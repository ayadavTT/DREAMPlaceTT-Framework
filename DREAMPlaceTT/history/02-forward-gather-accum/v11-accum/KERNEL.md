# V11 accum — route_buf accumulate + reduce (density gather stage)

> **TL;DR:** The gather half of V11: each core reads its inbound `route_buf`
> pages, accumulates the `(bx,by,area)` tuples into its local density slab, and
> (shard owners) reduces shard tiles into the primary. Merged accum+reduce into
> one kernel so the host pays a single Finish() for the whole gather. Plus the
> histogram/reduce variants. Shared by V11 and V19 forward.

---

## 1. Identity
- **Stage:** forward — density gather / accumulate / reduce
- **Status:** ⭐ SHIPPED (the V11 gather; also used in the V11/V19 forward setup).
- **Lineage:** original 3-launch accum+reduce_a+reduce_bc (~16 ms host round-trips) → **V11 merged accum+reduce (one launch)**; G-PRESCALE + G-PMERGE optimizations.
- **Source files:** `src/v11_accum_dm.cpp` (403, BRISC accum+reduce), `src/v11_accum_n_dm.cpp` (248, NCRISC half), `src/v11_histogram.cpp` (140), `src/v11_reduce_a_dm.cpp` (71), `src/v11_reduce_bc_dm.cpp` (128).
- **Activated by:** `GATHER_MODE=v11` (and the V11 setup shared by V19); built in `host/v19_engine.cpp`.

## 2. Problem & contract
**Computation.** Consume the `route_buf` produced by the V11 scatter and produce the density map: each core sums its inbound tuples into a local `dense[]`, then shard owners reduce their owned shard tiles into the primary. **Inputs:** `route_buf[*][me]` DRAM pages + `owned_lookup[]` + shard table. **Output:** per-core density slab (→ DRAM). **Invariants:** `n_total = n_primary + n_shard` tiles fit L1; per-shard barrier before the atomic increment to the primary.

## 3. Mechanism (how it works)
1. Load per-core `owned_lookup[]`; 2. zero `dense[]`; 3. read `route_buf[*][me]` in `SRC_CHUNK` batches, accumulate tuples into `dense[]` via `owned_lookup` (tile_id → local slot); 4. shard owners write each owned shard tile to `shard_reduce_buf`, barrier, then atomically signal the primary.
- **G-PMERGE:** BRISC and NCRISC each merge **half** of `dense_b += dense_n` in parallel (−9.1% gather @2048).
- **G-PRESCALE:** the `area` was pre-multiplied by `sqrt(inv_bin_area)` in v4_compute, so no separate scale pass here.
- **THE TRICK:** merge accum + reduce into a single program launch — the old 3-launch split cost ~16 ms in host-device round-trips; one launch + one Finish() removes that.

## 4. Performance (measured — TT)
- Gather ≈ 3.14 ms @ adaptec1_512 (memory); G-PRESCALE −8.9%@512/−25%@2048, G-PMERGE −9.1%@2048 → ~30% total gather reduction at 2048 ([[v11_g_prescale_win]], [[v11_g_pmerge_win]]).
- **Critical path:** NCRISC route upstream; gather has 3.0–17.8× core imbalance ([[v11_gather_skew_diagnostic]]).

## 5. Correctness
- HPWL convergence vs CPU (with the V11 scatter). Note: inherits the route_buf cap-overflow limit — see `01-forward-scatter/v11-scatter` (clustered bigblue3 fails).

## 6. Gotchas / pitfalls
- L1 budget: `n_total` tiles + scratch must fit (FATAL if `MAX_PER_PAGE_TUPLES` too high).
- Per-shard barrier MUST precede the primary's semaphore increment (else the increment races ahead of the DRAM shard data).
- Gather load imbalance (hot tiles) — hot-tile sharding default OFF after it hurt convergence paths.

## 7. When to use / avoid
- **Wins:** as the V11 gather. **Avoid:** at scale + clustering the route+accum overflows; V19's atomic-add folds gather into scatter (no separate accum) and has no cap → preferred.

## 8. Provenance
- **Memory:** [[v11_g_pmerge_win]], [[v11_g_prescale_win]], [[v11_gather_skew_diagnostic]], [[v11_gather_sharding_investigation]]
- **Handoff:** `docs/V11_V13_GATHER_HANDOFF.md`
- **Host:** `host/v19_engine.cpp`
