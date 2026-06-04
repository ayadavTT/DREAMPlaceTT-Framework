# V21 EF — SFPU outer-product electric-force backward (the first chip EF)

> **TL;DR:** The first on-chip electric-force backward: a balanced 3-RISC kernel
> that gathers the field and computes the force via an **SFPU outer product** of
> the px/py geometry weights. Bit-exact, shipped (`V21_EF=1`), but **prep/gather-
> overhead bound** (~17 ms @2048, ~1.4× slower than CPU). Its profiling drove the
> whole later EF line (V23 redesign, V25 L1-gather). The `_v5` files are the V23
> adaptive-window redesign (~11 ms).

---

## 1. Identity
- **Stage:** backward — electric-force gather + compute
- **Status:** ⭐ SHIPPED (`V21_EF=1`, live backward), bit-exact; superseded on speed by V25/V28/FCCS.
- **Lineage:** (density forward) → **V21 (SFPU outer-product EF)** → V23 (`_v5`: adaptive 5×5 + full tiles, 11.2 ms) → V25 (L1-gather, 0.38 ms). V21 is the reference correctness baseline for EF.
- **Source files:** `src/v21_ef_brisc.cpp` (363), `src/v21_ef_ncrisc.cpp` (347), `src/v21_ef_compute.cpp` (225), `src/v21_ef_common.h` (352); V23 redesign: `src/v21_ef_*_v5.cpp`.
- **Activated by:** standalone microbench `host/v21_electric_force_microbench_host.cpp` (target `v21_ef_microbench`); live via the engine `V21EFEngine` + `V21_EF=1`.

## 2. Problem & contract
**Computation.** EF backward: per cell, force = Σ over overlapped bins of `field · (px·py overlap weight)`, negated to match `electric_potential_cpp.electric_force`. **Inputs:** pos (2·num_nodes), field_x/field_y (M·N each), per-cell constants (ox,oy,nsx,nsy,ratio). **Output:** `grad_x,grad_y` per node (−grad). **Invariants:** field uploaded h2d (or read from chip via `latest_field_addrs` zero-copy); needs 2-point sem sync for mutably-shared L1 (see gotchas).

## 3. Mechanism (how it works)
- **3-RISC split:** BRISC + NCRISC gather the field samples + compute per-cell px/py geometry; TRISC runs the SFPU **outer product** `px ⊗ py` to form the per-bin weights and the weighted field sum.
- **Geometry recompute:** V21 recomputes the px/py overlap weights each backward (the "prep") — later found redundant ([[v30_prep_is_redundant]]) since the forward already computes them.
- **`_v5` (V23):** adaptive 5×5 window + full-tile packing — 1.55× faster (17.4→11.2 ms) but still > CPU.

**THE TRICK (and its limit):** doing the force as an SFPU outer product is elegant and bit-exact, but fine-grained Tracy showed **TRISC is starved ~49% waiting for the field-gather + px/py producers** ([[v21_ef_finegrain_starvation]]) — the kernel is **producer/prep bound, not SFPU bound**. That insight (field-locality + prep-reuse) produced V25/V28.

## 4. Performance (measured — TT; see `profile/`)
- ~17.4 ms @2048 (1.64× faster than scalar, **1.43× slower than CPU 16T**); `_v5`/V23 11.2 ms ([[v21_sfpu_brisc_bottleneck]], [[v21_v23_ef_profile]]).
- **Bottleneck:** producers (field gather + px/py geometry) starve TRISC; SFPU offload of geometry justified but field-gather is the floor ([[v21_ef_geom_ablation]]). Field locality refuted as a lever ([[v21_ef_locality_refuted]]); bf16 fields refuted ([[v23_ef_bf16_refuted]]).

## 5. Correctness
- Bit-exact 0/540K, 0/1.1M vs CPU EF. The gold correctness reference for EF backward.

## 6. Gotchas / pitfalls
- **2-point semaphore sync** required for mutably-shared L1 (after prep AND after tile-pack) — else BRISC races and overwrites NCRISC's source data ([[v21_sfpu_brisc_bottleneck]]).
- The per-backward geometry recompute is **redundant** — the forward already has px·py; stash + reuse (V31) instead.
- Don't chase SFPU speedups: it's producer-bound, not SFPU-bound.

## 7. When to use / avoid
- **Wins:** as the bit-exact correctness reference, and when you need a self-contained EF backward without the forward stash. **Avoid for speed:** at 17 ms it loses to CPU; use V25/V28 (L1-gather) or FCCS for performance.

## 8. Provenance
- **Memory:** [[v21_sfpu_brisc_bottleneck]], [[v21_ef_finegrain_starvation]], [[v21_ef_geom_ablation]], [[v21_v23_ef_profile]], [[v21_ef_locality_refuted]], [[v23_ef_redesign_design]], [[v23_ef_bf16_refuted]], [[v30_prep_is_redundant]]
- **Handoff:** `docs/V21A_SFPU_HANDOFF.md`, `docs/V21_DREAMPLACE_ON_CHIP_RESEARCH.md`, `docs/V23_EF_REDESIGN.md`
- **Host:** `host/v21_electric_force_microbench_host.cpp`; engine `host/v21_ef_engine.cpp`
