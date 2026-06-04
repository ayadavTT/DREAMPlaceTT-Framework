# V27 bucket — on-chip cell-bucketing scatter (DRAM-resident, no atomics)

> **TL;DR:** Group cells into field-band buckets at the **cell level** (route[src]
> [band], DRAM-resident, no atomics) so the EF-backward gather reads each band's
> cells contiguously — without touching the V19 forward. Validated 0 mismatch;
> 0.34–3.2 ms @ 211K–2.2M. The first clean cell-bucketing front-end for the
> backward gather.

---

## 1. Identity
- **Stage:** backward — cell bucketing
- **Status:** ✓ VALIDATED (0 mismatch all scales). Feeds the V25/V26 EF-backward gather.
- **Lineage:** **V27 (cell-level band bucketing)** → V28 (consumes it for multi-bin gather) → V29 (prep+bucket fused) → V32 (count-prefix). V27 is the standalone bucketing proof.
- **Source files:** `src/v27_bucket.cpp` (82).
- **Activated by:** `host/v27_host.cpp` (target `v27`).

## 2. Problem & contract
**Computation.** Assign each cell to the field band(s) it overlaps and write it into that band's DRAM-resident bucket, so the backward gather can stream a band's cells contiguously. **Inputs:** per-cell footprint + band layout. **Output:** `route[src][band]` DRAM buckets of cells. **Invariants:** cell-level (moves cells, not records — no transpose); 64-align reads (cpc → multiple of 8, [[blackhole_dram_read_align_64]]); no atomics.

## 3. Mechanism (how it works)
- For each cell, compute the bands it overlaps and append it to each band's bucket (per-source-core, per-band layout) in DRAM.
- **DRAM-resident, no atomics:** buckets live in DRAM; each source core writes its own region → no cross-core contention.
- **THE TRICK:** bucket at the **cell** granularity (not record/transpose) so it composes with the V19 forward untouched and feeds the L1-gather backward directly. This is the "move cells, not records" insight that V33 later confirmed beats the transpose-based regroup.

## 4. Performance (measured — TT; see `profile/run.log`)
- **MEDIAN 0.793 ms** (measured this run) for the backward grouping; 0.34–3.2 ms @ 211K–2.2M cells; vs CPU EF ~5–7 ms ([[v27_bucketing_scatter]]). Validated 0 mismatch.

## 5. Correctness
- 0 mismatch (exact bucketing) at all scales.

## 6. Gotchas / pitfalls
- **64-B aligned reads** on Blackhole — pad cells-per-core (cpc) to a multiple of 8 ([[blackhole_dram_read_align_64]]).
- Cell-level (not record/transpose) — keep it that way; the transpose-based regroup (V32 record form) loses at 2048 ([[v33_cell_gather_pivot]]).

## 7. When to use / avoid
- **Wins:** as a simple, atomic-free, clustering-tolerant bucketing front-end for the L1-gather backward. **Avoid:** if you can skip grouping (FCCS field-cast).

## 8. Provenance
- **Memory:** [[v27_bucketing_scatter]], [[v28_multibin_ef_gather]], [[v33_cell_gather_pivot]], [[blackhole_dram_read_align_64]]
- **Host:** `host/v27_host.cpp`
