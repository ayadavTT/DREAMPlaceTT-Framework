# V31 geom-stash — forward stashes per-cell geometry for a prep-free backward

> **TL;DR:** A V19-style forward scatter that *also* **stashes each cell's px·py
> geometry** (`bxl, area` per overlapped bin) to DRAM as it computes the density.
> The backward then reads this stash instead of recomputing overlaps in
> soft-float — eliminating the backward's prep stage (4.94→1.2 ms). Wired live
> (`V31_STASH=1 V31_GEOM=1`); adaptec1_512 converges.

---

## 1. Identity
- **Stage:** forward — geometry stash (prep-reuse enabler)
- **Status:** ✓ VALIDATED + LIVE (converges; prep eliminated).
- **Lineage:** V19 scatter (computes px·py, discards it) → **V31 stash (keeps it)** → consumed by V29/V31/FCCS backward (`bucket_only`, `compute_electric_force_v31`, FCCS `set_geom_source`).
- **Source files:** `src/v31_scatter_stash_dm.cpp` (330, NCRISC), `src/v31_scatter_b_stash_dm.cpp` (289, BRISC).
- **Activated by:** `V31_STASH=1` (+ `V31_GEOM=1`) on the V19 forward, via the engine.

## 2. Problem & contract
**Computation.** Same as V19 scatter (build density via atomic-add) **plus** write a per-cell geometry record `(cell_idx, bxl, byl, px·py area per bin)` into a DRAM stash buffer. **Inputs:** same as V19 (cells + v4_compute overlaps). **Output:** density (as V19) + the `V31_GEOM` stash at `eng_.v31_geom_addr()` (32 int words/record). **Invariants:** stash record format must match what the backward reads; 64-align.

## 3. Mechanism (how it works)
- Runs the V19 atomic-add scatter, and in the same per-cell loop **also emits the geometry** (which the SFPU/loop already has in hand) to the stash buffer — near-free since the data is already computed.
- **THE TRICK (prep-reuse):** the forward computes px·py overlaps on the SFPU then throws them away; the backward then recomputes them in **soft-float** (the measured backward bottleneck). Stashing the forward's result lets the backward skip that entirely — prep 4.94→1.2 ms ([[v30_prep_is_redundant]], [[v33_geom_stash_step1]]).

## 4. Performance (measured — TT)
- Forward cost ≈ V19 scatter + a small stash write (the geometry is already computed). Backward prep eliminated: 4.94→1.2 ms. Live adaptec1_512 converges (HPWL 60.87M) ([[v33_geom_stash_step1]]).

## 5. Correctness
- Converges live (HPWL = ref). Stash record format validated by the V32/V31 backward consumers (grad rel_l2 < 1e-3).

## 6. Gotchas / pitfalls
- Live needs **BOTH `V21_EF=1` AND the stash/V29 flags** — else silent CPU fallback that still converges (masks whether the chip path ran) ([[v33_geom_stash_step1]]).
- `TT_METAL_RUNTIME_ROOT` required; gather loops `max_k×max_h` (bound `max_k` by orig cell width, not a fixed 8).
- Stash record layout (32 int words) must match the backward reader exactly.

## 7. When to use / avoid
- **Wins:** whenever a chip backward follows the forward in the same session — stash once, reuse, skip soft-float prep. **Avoid:** if no chip backward (pure forward), the stash is wasted DRAM writes — use plain V19.

## 8. Provenance
- **Memory:** [[v33_geom_stash_step1]], [[v30_prep_is_redundant]], [[v32_e2e_backward_chain]]
- **Host:** engine `host/v19_engine.cpp` (V31_STASH path)
