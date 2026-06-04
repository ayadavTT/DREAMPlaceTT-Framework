# V26 EF — SFPU fp32 4-corner gather

> **TL;DR:** V25's L1-gather idea computed in **SFPU fp32** (BRISC gather + SFPU
> weighted sum) rather than BRISC integer fixed-point. A 4-corner gather, so —
> like V25 — accuracy-correct only where cells span ≤2×2 bins. The predecessor
> that V28 generalized to multi-bin.

---

## 1. Identity
- **Stage:** backward — electric-force gather (SFPU fp32, 4-corner)
- **Status:** ✓ VALIDATED (correct at small/medium grid). Stepping stone V25 → V28.
- **Lineage:** V25 (BRISC int fixed-point) → **V26 (SFPU fp32)** → V28 (multi-bin, the correct-at-all-grids generalization).
- **Source files:** `src/v26_brisc.cpp` (99), `src/v26_compute.cpp` (85, SFPU), `src/v26_writer.cpp` (33).
- **Activated by:** `host/v26_host.cpp` (target `v26`).

## 2. Problem & contract
**Computation.** Same EF backward; 4-corner bilinear field sample per cell, weighted by px·py, summed in SFPU fp32. **Inputs:** L1-resident field + per-cell corner indices/weights. **Output:** per-cell grad (fp32).

## 3. Mechanism (how it works)
- **BRISC gather:** loads the 4 corner field samples from the L1-resident field.
- **SFPU compute:** the weighted sum in fp32 on the SFPU (vs V25's integer MAC on BRISC) — cleaner fp accuracy, but uses the SFPU path.
- **THE TRICK:** keep V25's L1-resident gather but move the arithmetic to the SFPU (fp32) — useful when fp32 accuracy is wanted over int fixed-point. V28 then keeps SFPU compute but fixes the footprint (multi-bin).

## 4. Performance (measured — TT; see `profile/run.log`)
- This run reported a grouping split: field-shard(artifact)=13.16 ms / cell-bucket(real)=13.15 ms — i.e., dominated by the grouping it was measured with, not the gather itself. The gather core is comparable to V25/V28; V28 (1.68 ms this run) is the productionized successor ([[v28_multibin_ef_gather]]).

## 5. Correctness
- Correct for ≤2×2-bin cells (small/medium grid); **wrong at 2048** (4-corner footprint) — same limit as V25. Use V28 there.

## 6. Gotchas / pitfalls
- **4-corner is invalid at 2048** (cells span 6×3) — V26 inherits V25's footprint limit; V28 fixes it.
- SFPU fp32 vs BRISC int: SFPU is cleaner accuracy but the BRISC int path (V25) was faster on the FPU-less BRISC for the pure gather.

## 7. When to use / avoid
- **Wins:** small/medium grid where fp32 SFPU accuracy is preferred. **Avoid:** 2048 (footprint wrong) → V28.

## 8. Provenance
- **Memory:** [[v28_multibin_ef_gather]], [[v25_ef_l1gather_breakthrough]]
- **Host:** `host/v26_host.cpp`
