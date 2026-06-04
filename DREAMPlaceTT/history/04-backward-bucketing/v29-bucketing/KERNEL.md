# V29 bucketing — prep+bucket (reads forward geometry stash)

> **TL;DR:** The prep+bucket front-end of the V29 full on-chip EF-backward
> pipeline. Buckets cells into L1-resident field bands; when the forward stashed
> per-cell geometry (`V31_GEOM`), `bucket_only` skips the soft-float prep and
> reuses it. Bit-exact; part of the live V29 backward (`V29_EF=1`).

---

## 1. Identity
- **Stage:** backward — cell bucketing (prep + bucket)
- **Status:** ✓ VALIDATED (bit-exact). Front-end of the V29 pipeline (pairs with `v29-gather`).
- **Lineage:** V27 (standalone bucket) → **V29 prep+bucket** (fused into the full pipeline) → uses V31 geom-stash for prep-free operation.
- **Source files:** `src/v29_prepbucket.cpp` (111), `src/v29_bucket_only.cpp` (66).
- **Activated by:** live via engine `V29EFEngine` + `V29_EF=1`; full pipeline `host/v29_host.cpp` (target `v29`).

## 2. Problem & contract
**Computation.** Compute each cell's band overlaps (prep) and place cells into per-band L1-resident buckets (bucket), feeding `v29-gather`. **Inputs:** cells + field-band layout; optionally the forward `V31_GEOM` stash (cell,area) at `eng_.v31_geom_addr()`. **Output:** per-band cell buckets. **Invariants:** `prepbucket` computes overlaps in soft-float; `bucket_only` reads the stash instead (prep-free).

## 3. Mechanism (how it works)
- **prepbucket:** soft-float computes per-cell band overlaps, then buckets — the self-contained path.
- **bucket_only:** if the forward ran with `V31_STASH=1 V31_GEOM=1`, the per-cell px·py geometry is already in DRAM; `bucket_only` reads it and skips the soft-float prep (the prep was the measured bottleneck — [[v30_prep_is_redundant]]).
- **THE TRICK:** **prep-reuse** — the forward (v4_compute/v19_scatter) already computed px·py on the SFPU and discarded it; stashing it lets the backward bucket without recomputing in soft-float.

## 4. Performance (measured — TT; see `profile/run.log`)
- Within the V29 pipeline: prep+bucket = **4.99 ms** (measured this run); gather+compute+scatter = 5.26 ms; TOTAL 10.25 ms ([[v29_full_pipeline]]). The prep (soft-float) dominates → `bucket_only` + stash cuts it (4.94→1.2 ms, [[v33_geom_stash_step1]]).

## 5. Correctness
- Bit-exact (rel_l2 8.7e-8) all grids in the V29 pipeline.

## 6. Gotchas / pitfalls
- Soft-float prep is the bottleneck — use `bucket_only` + the forward stash whenever available.
- The V29 band partition is **clustering-fragile** (`random_center_init` clusters cells in center x-bins → hottest band huge → cold bands grind empty) — this is why live V29 was slower despite bit-exact convergence ([[v29_live_swap]]). V30 adaptive bands / V32 count-prefix / FCCS field-cast fix it.

## 7. When to use / avoid
- **Wins:** with the forward geom-stash (bucket_only) for prep-free bucketing. **Avoid:** the band partition under clustering — prefer adaptive (V30) or no-bucketing (FCCS).

## 8. Provenance
- **Memory:** [[v29_full_pipeline]], [[v29_live_swap]], [[v30_prep_is_redundant]], [[v33_geom_stash_step1]], [[ef_backward_2x_target]]
- **Host:** `host/v29_host.cpp`; engine `host/v29_ef_engine.cpp`
