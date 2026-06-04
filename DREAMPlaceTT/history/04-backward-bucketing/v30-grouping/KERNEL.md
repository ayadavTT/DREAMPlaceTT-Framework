# V30 grouping — bin-owner adaptive slab grouping (clustering-robust)

> **TL;DR:** Group cells by a field-slab (bin-owner) partition with **adaptive
> slab balancing**, so each core's backward workload stays balanced even under
> heavy clustering. The decisive experiment showed imbalance 1.25–1.54× (vs the
> V29 band approach's 3.8× that killed it). The grouping front-end of the V30
> field-stationary backward.

---

## 1. Identity
- **Stage:** backward — cell bucketing / grouping (bin-owner, adaptive)
- **Status:** ✓ VALIDATED (clustering-robust; core beats CPU uniform 1.3×, ~ties clustered@2048).
- **Lineage:** V29 band-partition (clustering-fragile, 3.8× imbalance) → **V30 bin-owner adaptive slab** (1.25–1.54× imbalance). Pairs with `v30-backward` (the gather+atomic-grad).
- **Source files:** `src/v30_prepcount.cpp` (88), `src/v30_band.cpp` (99), `src/v30_zero.cpp` (16).
- **Activated by:** `host/v30_host.cpp` (target `v30`).

## 2. Problem & contract
**Computation.** Partition the field into slabs (bin-owner bands) and assign each cell to the slab(s) it overlaps, with **adaptive** slab widths so cell counts per slab are balanced. **Inputs:** per-cell footprint + field-slab layout. **Output:** per-slab cell lists for the field-stationary gather. **Invariants:** count-prefix placement (not atomic-slot); slab widths tuned to balance load.

## 3. Mechanism (how it works)
- **prepcount:** count cells per slab (local histogram).
- **band:** assign adaptive slab boundaries so each slab gets ~equal cells (the anti-clustering fix — fixed-width bands put all center-clustered cells in one band → 3.8× imbalance).
- **zero:** clear the per-slab accumulators.
- **THE TRICK:** make the partition **data-adaptive** (balance by cell count, not by fixed bin width). `random_center_init` clusters cells in center bins; fixed bands → one hot band; adaptive bands spread the work → imbalance drops from 3.8× to ~1.3×.

## 4. Performance (measured — TT; see `profile/run.log`)
- Imbalance 1.25–1.54× (vs V29 band 3.8×); core beats CPU uniform 1.3×, ~ties clustered@2048 ([[v30_clustering_verdict]]). The decisive clustered-vs-uniform experiment.

## 5. Correctness
- rel_l2 4.5e-4 @ span-6 (the V30 backward it feeds); grouping itself is exact (count-prefix).

## 6. Gotchas / pitfalls
- **Fixed-width bands are the trap** — they collapse under center-clustering (the V29 failure). Use adaptive (count-balanced) boundaries.
- Forward stash should use count-prefix, not atomic-slot, to feed this.
- Core-only result (no live h2d/d2h in the microbench) — the real win is the whole density backward at scale.

## 7. When to use / avoid
- **Wins:** when you need a bin-owner/field-stationary backward that survives clustering. **Avoid:** FCCS's field-cast avoids grouping entirely and is also clustering-immune — prefer it unless you specifically need field-stationary.

## 8. Provenance
- **Memory:** [[v30_clustering_verdict]], [[bigblue1_1024_metastability]], [[v29_live_swap]]
- **Host:** `host/v30_host.cpp`
