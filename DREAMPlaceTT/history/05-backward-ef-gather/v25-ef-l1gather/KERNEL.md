# V25 EF L1-gather — the electric-force backward breakthrough

> **TL;DR:** Keep the field map **L1-resident** and gather each cell's 4 corner
> field samples by **direct L1 load** (no DRAM round-trip), computing the
> weighted force in **integer fixed-point on BRISC** (which has no FPU, so
> soft-float was the bottleneck). Result: **0.38 ms @540K cells ≈ 19× CPU**.
> This is the EF backward kernel that finally beat CPU.

---

## 1. Identity
- **Stage:** backward — electric-force gather (sample field at cell, weight, → grad)
- **Status:** ⭐ SHIPPED-as-design / breakthrough (microbench; the design that all later EF backward builds adopt)
- **Lineage:** V21 (SFPU outer-product, 17 ms — prep-bound) → V24 (grid_sample pool) → **V25 (L1-resident direct-load gather + int fixed-point)** → V26/V28 (SFPU/multi-bin generalizations) → FCCS (field-cast, balanced).
- **Source files:** `src/v25_ef_l1gather.cpp` (111 ln).
- **Activated by:** standalone microbench `host/v25_ef_l1gather_host.cpp` (target `v25_ef_l1gather`).

## 2. Problem & contract
**Computation.** Backward of the density forward: given the solved field maps `field_x`, `field_y` (`[M,N]` each) and each cell's position/footprint, the electric force on a cell = Σ over its overlapping bins of `field · overlap_weight`. V25 computes `grad_x`, `grad_y` per cell.

**Inputs:** field maps (L1-resident), per-cell corner bin indices + bilinear weights (the px·py geometry). **Output:** per-cell `(grad_x, grad_y)`.

**Invariants:** field must fit L1 (it does at all grids when band-partitioned — see V28/FCCS); 64-B aligned DRAM reads if any spill ([[blackhole_dram_read_align_64]]).

## 3. Mechanism (how it works)
- **Field L1-resident:** the field slab lives in L1, so the per-cell gather is a **direct L1 load** at the corner addresses — not a DRAM `noc_async_read`. This is the core idea: the gather itself is 0.31 ms (24× CPU) once the field is local.
- **Integer fixed-point compute on BRISC:** BRISC has **no FPU** — the original soft-float weighted-sum was the measured bottleneck (Tracy). V25-v2 does the MAC in **integer fixed-point** on BRISC, eliminating soft-float → 0.383 ms total, rel_l2 8.6e-5.
- **4-corner gather:** bilinear sample at the cell's 4 surrounding bin corners with px·py weights.

**THE TRICK:** make the field local (L1) so the gather is a load, and do the arithmetic in integer fixed-point on the FPU-less BRISC. Two independent wins that together cross the CPU line.

## 4. Performance (measured — TT; see `profile/` once harness re-run)
- **0.383 ms @540K cells = 19× CPU**, rel_l2 8.6e-5 ([[v25_ef_l1gather_breakthrough]]). The gather core alone 0.31 ms (24× CPU).
- **Bottleneck found by Tracy:** BRISC soft-float (fixed by int fixed-point). Field-locality is the enabler (V21 was prep/gather-overhead bound at 17 ms).

## 5. Correctness
- rel_l2 8.6e-5 vs CPU EF reference (int fixed-point quantization). PASS.

## 6. Gotchas / pitfalls
- BRISC has no FPU → never do float MAC there; use integer fixed-point.
- 4-corner gather is only valid where cells span ≤ a 2×2 bin window — at large grids cells span more bins, so the **4-corner model is wrong at 2048**; V28 generalizes to variable k×h overlap.
- Field must be L1-resident; at scale band-partition it (V28/FCCS) so it fits.

## 7. When to use / avoid
- **Wins when:** the field can be made L1-resident and cells span a small bin window (small/medium grid). The 19× headline is real.
- **Avoid:** at 2048 the 4-corner approximation drops accuracy → use V28 (multi-bin) which keeps the L1-gather + int idea but handles variable overlap.

## 8. Provenance
- **Memory:** [[v25_ef_l1gather_breakthrough]], [[v21_ef_geom_ablation]], [[v24_ef_grid_sample_pool]], [[ef_backward_2x_target]], [[blackhole_dram_read_align_64]]
- **Host:** `host/v25_ef_l1gather_host.cpp`
