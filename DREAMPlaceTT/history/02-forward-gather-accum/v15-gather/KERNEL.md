# V15 gather — DRAM-spill gather for large grids

> **TL;DR:** A gather that **spills accumulation state to DRAM** when the density
> working set exceeds L1 (large grids / high overlap). Works after a critical
> page-size fix: `noc_async_write` > ~96 KB silently corrupts state, so the
> spill page must stay ≤80 KB (cap the bucket size). With cap=2048 + 16 chunks,
> adaptec3_512 converges (was diverging).

---

## 1. Identity
- **Stage:** forward — density gather (DRAM-spill)
- **Status:** ✓ VALIDATED (after the page-size fix). `GATHER_MODE=v15`.
- **Lineage:** V11/V13 in-L1 accum → **V15 (DRAM-spill when L1 overflows)** for the largest working sets.
- **Source files:** `src/v15_gather_brisc.cpp` (521), `src/v15_gather_compute.cpp` (88), `src/v15_gather_ncrisc.cpp` (227), `src/v15_gather_brisc_stream.cpp` (349, streaming variant).
- **Activated by:** `GATHER_MODE=v15`.

## 2. Problem & contract
**Computation.** Density gather where the per-core accumulation buffer doesn't fit L1 → spill partial sums to DRAM and merge. **Inputs:** scatter output / route tuples. **Output:** density map. **Invariants:** **spill page ≤ 80 KB** (clamp `v15_bucket_cap`); cap=2048 + chunks=16 is a validated config.

## 3. Mechanism (how it works)
- Accumulate in L1 up to a bucket cap; when full, **spill** the bucket to a DRAM page (`noc_async_write`), reset L1, continue; finally merge spilled pages.
- **THE TRICK (and the bug):** the spill is what lets gather scale past L1 — but `noc_async_write` of > ~96 KB **silently corrupts state across iterations** on Blackhole. The fix is to clamp the bucket cap so `spill_pgsz ≤ 80 KB`. With that, adaptec3_512 converges (it diverged before the clamp).

## 4. Performance (measured — TT)
- Functional at large working sets; see `profile/`. Primary value is **correctness at scale**, not peak speed (V19 atomic-add superseded it for the production forward).

## 5. Correctness
- Converges with cap=2048 + chunks=16 (adaptec3_512); diverges if the spill page-size clamp is removed.

## 6. Gotchas / pitfalls
- ⚠️ **`noc_async_write` > ~96 KB silently corrupts** on Blackhole — keep spill page ≤ 80 KB ([[v15_spill_pgsz_bug]]). This is a general Blackhole NoC-write hazard worth remembering for any large DRAM write.
- Bucket cap too small → more spill round-trips (slower); too large → corruption. cap=2048 is the sweet spot found.

## 7. When to use / avoid
- **Wins:** when the gather working set genuinely exceeds L1 and you must spill. **Avoid:** V19 atomic-add avoids the whole accumulation-buffer problem (each contribution is an atomic into the owner's L1 slab) → preferred for production.

## 8. Provenance
- **Memory:** [[v15_spill_pgsz_bug]]
- **Handoff:** `docs/V15_HANDOFF.md`
- **Host:** `host/v19_engine.cpp` (v15 path, IPC server)
