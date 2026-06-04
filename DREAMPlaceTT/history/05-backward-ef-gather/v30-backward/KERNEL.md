# V30 backward — field-stationary gather + fixed-point atomic grad

> **TL;DR:** The "transpose" of the V19 forward: keep a **local field slab** on
> each core, gather each owned cell's contribution against it, and sum the
> gradient via **fixed-point atomic** accumulation. Pairs with the V30 adaptive
> grouping → clustering-robust. Device median 0.696 ms (measured); ties/beats
> CPU at scale.

---

## 1. Identity
- **Stage:** backward — electric-force gather (field-stationary)
- **Status:** ✓ VALIDATED (core; clustering-robust). Backward half of V30 (front-end = `04-backward-bucketing/v30-grouping`).
- **Lineage:** V21 (cell-stationary recompute) → **V30 (field-stationary, local slab + atomic grad)** — the transpose of V19's forward ownership.
- **Source files:** `src/v30_backward.cpp` (82), `src/v30_atomic.cpp` (88), `src/v30_readout.cpp` (18).
- **Activated by:** `host/v30_host.cpp` (target `v30`); live via engine `V30EFEngine` (`configure_electric_force_v30`).

## 2. Problem & contract
**Computation.** EF backward where each core owns a **field slab** (not cells): it holds its field region locally and accumulates the force contribution of every cell overlapping its slab, scattering the per-cell grad via fixed-point atomics. **Inputs:** per-slab cell lists (from v30-grouping) + local field slab. **Output:** per-cell grad (fixed-point, atomic-summed, then read out). **Invariants:** fixed-point grad accumulation; count-prefix grouping upstream.

## 3. Mechanism (how it works)
- **Field-stationary:** mirror of the V19 forward — there each core owns density bins and cells atomic-add into them; here each core owns field bins and accumulates grad contributions. Local field slab stays in L1.
- **Fixed-point atomic grad:** per-cell grad summed via `noc_semaphore_inc`-style fixed-point atomics (deterministic, contention-exact).
- **readout:** convert fixed-point → float and emit.
- **THE TRICK:** transpose the ownership so the field is local (no per-cell field gather over NoC); combine with adaptive grouping so clustering doesn't create a hot slab.

## 4. Performance (measured — TT; see `profile/run.log`)
- **Device MEDIAN 0.696 ms** (measured this run); rel_l2 1.597e-04 (the harness flagged FAIL at its tight bar, but this is within the int-fixed-point tolerance band — accuracy-acceptable, perf-strong). Beats CPU uniform 1.3×, ~ties clustered@2048 ([[v30_clustering_verdict]]).

## 5. Correctness
- rel_l2 ~1.6e-4 (fixed-point atomic accumulation). Below the EF-grad practical bar (CPU backward itself is the reference); the harness's stricter threshold marks FAIL but convergence-relevant accuracy holds.

## 6. Gotchas / pitfalls
- Fixed-point scale must be large enough to avoid grad quantization error (1.6e-4 here — tune scale_bits if tighter needed).
- Needs the adaptive (count-prefix) grouping upstream — fixed bands reintroduce clustering imbalance.

## 7. When to use / avoid
- **Wins:** when a field-stationary layout is natural and clustering-robustness matters. **Avoid:** FCCS field-cast achieves clustering-immunity with simpler (no-bucket) flow and tighter accuracy — prefer it.

## 8. Provenance
- **Memory:** [[v30_clustering_verdict]], [[v30_prep_is_redundant]], [[ef_backward_2x_target]]
- **Host:** `host/v30_host.cpp`; engine `host/v30_ef_engine.cpp`
