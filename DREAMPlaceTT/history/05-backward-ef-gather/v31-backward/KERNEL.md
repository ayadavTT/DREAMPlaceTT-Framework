# V31 backward — no-host backward reading the forward stash

> **TL;DR:** A backward kernel that reads the forward's `V31_GEOM` per-cell
> geometry stash directly and runs a band-discovering gather — **no host
> geometry, prep-free**. Converges live. ⚠️ **The standalone microbench wedges
> the device** (it needs the forward stash that the standalone harness doesn't
> populate); run it only via the live engine path, not `host/v31_backward`.

---

## 1. Identity
- **Stage:** backward — electric-force gather (no-host, stash-fed)
- **Status:** ✓ VALIDATED live (converges); ⚠️ standalone harness crashes the chip — DO NOT run `host/v31_backward` standalone.
- **Lineage:** V29/V30 (host-fed bucketing) → **V31 (reads forward V31_GEOM stash, host-free)** → V33 geom-stash wired live.
- **Source files:** `src/v31_backward.cpp` (98). Forward side: `03-forward-geom-stash/v31-stash`.
- **Activated by:** live via the engine `compute_electric_force_v31` (`V31_STASH=1` forward); standalone `host/v31_backward_host.cpp` exists but **hangs** without the stash.

## 2. Problem & contract
**Computation.** EF backward that consumes the forward's stashed `(cell, bxl, px·py area)` records (no host-side geometry, no recompute). **Inputs:** the `V31_GEOM` DRAM stash (written by `v31-stash` forward) + field maps + `sel`/`ratio`. **Output:** raw +force into `grad_full[sel]`. **Invariants:** the forward MUST have run with `V31_STASH=1 V31_GEOM=1` to populate the stash; otherwise the kernel reads garbage → hang/crash.

## 3. Mechanism (how it works)
- Reads the stash records, discovers the bands each cell touches, gathers the field, and computes the force — entirely from stashed geometry, so **no soft-float prep** and **no host round-trip**.
- **THE TRICK:** the forward already computed px·py on the SFPU and discarded it; V31 stashes it and the backward reuses it (prep elimination — 4.94→1.2 ms, [[v33_geom_stash_step1]]).

## 4. Performance (measured — TT)
- Live (V33 graft): prep eliminated; converges (adaptec1_512 HPWL 60.87M). But gather (14.5 ms) + grad d2h (6.8 ms) made the live backward ~26 ms — the gather/transfer became the new bottleneck after prep was removed ([[v33_geom_stash_step1]]).
- **Standalone harness: device timeout/crash** (no stash) — this is the harness that wedged the chip in the 2026-06-03 sweep. Profile via the live engine, not standalone.

## 5. Correctness
- Converges live (HPWL = ref). Grad correctness validated in the V32/V33 e2e chains.

## 6. Gotchas / pitfalls
- ⚠️ **Do not run `host/v31_backward` standalone** — it needs the forward stash and hangs/crashes the device without it. Use the live engine path.
- Needs BOTH `V21_EF=1` AND the stash flags live, else silent CPU fallback ([[v33_geom_stash_step1]]).
- After prep-elimination, gather+d2h dominate → the V33/FCCS direction.

## 7. When to use / avoid
- **Wins:** prep-free backward when the forward stash is available (live). **Avoid:** standalone (crashes); and post-prep the gather/d2h is the new floor → FCCS/V28 are faster end-to-end.

## 8. Provenance
- **Memory:** [[v33_geom_stash_step1]], [[v30_prep_is_redundant]], [[v32_e2e_backward_chain]]
- **Host:** `host/v31_backward_host.cpp` (standalone — crashes); live via engine.
