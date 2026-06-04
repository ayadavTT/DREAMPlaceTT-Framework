# V32 regroup — atomic-free count-prefix bin-owner grouping

> **TL;DR:** Group cells by their bin-owner using **local histogram → on-chip
> prefix-scan → local-cursor place** — no atomics, no fetch-add slots. Validated
> at all scales and **immune to clustering** (clustered == uniform speed), which
> is the failure mode that killed the band/atomic approaches. The robust
> replacement for the fragile atomic-slot grouping.

---

## 1. Identity
- **Stage:** backward — cell bucketing / regroup
- **Status:** ✓ VALIDATED (all scales; clustered==uniform). The atomic-free grouping primitive.
- **Lineage:** atomic-slot fetch-add (fragile, double-reads under contention) → V30 bin-owner (atomic count) → **V32 count-prefix regroup** (atomic-free). Feeds the V30/V32 bin-owner backward.
- **Source files:** `src/v32_count.cpp` (43), `src/v32_prefix.cpp` (44), `src/v32_place.cpp` (46); tuned variants `v32b_*` (place-sorted), `v32c_*`, `v32d_*` (prefix/place split).
- **Activated by:** `host/v32_regroup_host.cpp` (target `v32_regroup`); end-to-end `host/v32_e2e_host.cpp` / `v32d_e2e_host.cpp`.

## 2. Problem & contract
**Computation.** Given cells with bin-owner ids, produce a contiguous per-owner grouping (so the backward gather reads each owner's cells in one pass). **Inputs:** per-cell owner id. **Output:** cells reordered into per-owner contiguous runs + per-owner offsets. **Invariants:** 3 passes (count, prefix, place); per-owner pages; no atomics.

## 3. Mechanism (how it works)
- **Pass 1 — count:** each core builds a **local histogram** of how many cells it has per owner.
- **Pass 2 — prefix:** an on-chip **prefix-scan** over the histograms yields each (core,owner)'s write base.
- **Pass 3 — place:** each core walks its cells and writes each to its owner's run using a **local cursor** (incremented locally, no cross-core atomic).
- **THE TRICK:** replace the cross-core atomic fetch-add (which **double-reads the return slot under pipelined contention** — correct only at MAXO≤2, see [[atomic_fetchadd_contention_deadlock]]) with blind local counting + a global scan + local placement. Counting is always exact; the scan resolves global order deterministically. Result: **clustered == uniform** (the band/atomic approaches' hot-owner imbalance vanishes).

## 4. Performance (measured — TT; see `profile/run.log`)
- 0.57 / 2.55 / 5.84 ms @ 220K / 2.2M / 5.5M cells; clustered==uniform speed ([[v32_count_prefix_regroup]]). The **place pass (scattered 16-B writes) is ~70%** → optimize via local counting-sort.

## 5. Correctness
- Validated all scales; grad validated standalone vs CPU rel_l2 < 1e-3 with the real V31 record format ([[v32_e2e_backward_chain]]).

## 6. Gotchas / pitfalls
- **Never use atomic fetch-add for the unique slot** — it double-reads under cross-core contention ([[atomic_fetchadd_contention_deadlock]]). Blind count + prefix + local cursor is the safe pattern.
- The place pass's scattered 16-B writes dominate (~70%); a local counting-sort before placing would cut it.
- Per-owner workers can be unbalanced at extreme scale → needs adaptive workers (the V32 e2e found 2048/6² = 44 ms with naive per-owner workers).

## 7. When to use / avoid
- **Wins:** as the clustering-robust grouping front-end for any bin-owner backward (V30). **Avoid:** if you can skip grouping entirely (FCCS's field-cast does — no bucketing at all), prefer that.

## 8. Provenance
- **Memory:** [[v32_count_prefix_regroup]], [[v32_e2e_backward_chain]], [[atomic_fetchadd_contention_deadlock]], [[v33_cell_gather_pivot]]
- **Host:** `host/v32_regroup_host.cpp`, `host/v32_e2e_host.cpp`
