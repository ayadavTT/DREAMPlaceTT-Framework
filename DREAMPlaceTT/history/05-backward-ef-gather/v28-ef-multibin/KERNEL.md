# V28 EF multi-bin gather — V25's L1-gather generalized to any overlap

> **TL;DR:** V25's L1-resident direct-load gather, generalized from a fixed
> 4-corner sample to **variable k×h bin overlap** so it's correct at *every*
> grid (4-corner is 0% valid at 2048, where cells span ~6×3 bins). L1-resident
> column-band field + V21-style SFPU compute. 0.39/0.88/2.5 ms @ 512/1024/2048
> spans, rel_l2 5e-8 — beats CPU 2–18× and V21 ~5×.

---

## 1. Identity
- **Stage:** backward — electric-force gather (multi-bin)
- **Status:** ✓ VALIDATED (correct at all grids; beats CPU + V21). The accuracy-correct generalization of V25.
- **Lineage:** V25 (4-corner L1-gather) → V26 (SFPU fp32 4-corner) → **V28 (variable k×h multi-bin)** → folded into FCCS / V29 live path.
- **Source files:** `src/v28_brisc.cpp` (118), `src/v28_compute.cpp` (79), `src/v28_ncrisc.cpp` (71).
- **Activated by:** standalone microbench `host/v28_host.cpp` (target `v28`).

## 2. Problem & contract
**Computation.** Same EF backward as V25, but a cell overlaps a `k×h` window of bins (not just 2×2). V28 gathers all overlapped field samples and weights them by the per-bin px·py overlaps, producing `grad_x,grad_y`. **Why it matters:** at grid 2048 a standard cell spans ~6×3 bins → the V25/V26 4-corner model deposits/gathers the wrong set of bins (0% valid). V28 handles the real footprint.

**Inputs:** L1-resident **column-band** field slice + per-cell overlap geometry. **Output:** per-cell grad. **Invariants:** band must fit L1 (column-band partition keeps it bounded at all grids); resolve field row-vs-column-major before wiring.

## 3. Mechanism (how it works)
- **L1-resident column-band field:** partition the field into column bands so each core's band fits L1 (the scalability fix over V25's whole-field-L1 assumption).
- **Variable k×h gather:** for each cell, loop its actual `max_k × max_h` overlapped bins (bounded by the design's max cell width), gathering field samples by direct L1 load.
- **V21-style SFPU compute:** the weighted sum uses the SFPU (fp32), reusing V21's outer-product compute structure.

**THE TRICK:** keep V25's L1-resident-direct-load gather (the thing that beat CPU) but make it **footprint-correct** by iterating the true k×h overlap instead of a fixed 4 corners — and bound L1 with a column-band partition so it scales to 2048.

## 4. Performance (measured — TT; see `profile/`)
- 0.39 / 0.88 / 2.5 ms @ 512/1024/2048 spans, rel_l2 5e-8 — **2–18× CPU, ~5× V21** ([[v28_multibin_ef_gather]]).

## 5. Correctness
- rel_l2 5e-8 at all grids (essentially exact) — the key win over V25/V26 which are wrong at 2048.

## 6. Gotchas / pitfalls
- **4-corner is invalid at 2048** — this is the whole reason V28 exists; don't regress to it for large grids.
- Field **row- vs column-major** orientation must match between forward field layout and the band gather (a wiring hazard flagged before live integration).
- `max_k/max_h` must be bounded by the design's max cell width (don't loop a fixed 8×8 if cells are smaller — wasted cycles).

## 7. When to use / avoid
- **Wins when:** you need a correct EF gather at any grid, especially 2048. This is the accuracy-correct L1-gather.
- **Avoid:** if cells are guaranteed ≤2×2 bins (small grid) V25's simpler 4-corner is marginally cheaper — but V28 is the safe default.

## 8. Provenance
- **Memory:** [[v28_multibin_ef_gather]], [[v25_ef_l1gather_breakthrough]], [[v21_v23_ef_profile]]
- **Host:** `host/v28_host.cpp`
