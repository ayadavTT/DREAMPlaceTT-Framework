# V4 base — reader + SFPU bin/overlap compute (the shared foundation)

> **TL;DR:** The foundation every forward density path builds on: `v4_reader`
> streams px/py/sx/sy from DRAM into CBs, and `v4_compute` computes — **on the
> SFPU in fp32** — each cell's base bin (`bxl,byl`) and its up-to-8 x/y bin
> overlap fractions (`ox[j],oy[k]`). V11/V18/V19 all consume these CBs. The
> box-overlap model here is the ground truth the testbench reference matches.

---

## 1. Identity
- **Stage:** forward — reader + overlap compute (shared by all scatter modes)
- **Status:** ⭐ SHIPPED (shared foundation; live in the V19 production forward).
- **Lineage:** V4 is the base; V11/V13/V18/V19 reuse `v4_reader` + `v4_compute`. `v4_outer_compute` is the Stage-B′ outer-product variant for v11outer/v18outer.
- **Source files:** `src/v4_reader.cpp` (67), `src/v4_compute.cpp` (289, SFPU/TRISC), `src/v4_outer_compute.cpp` (347).
- **Activated by:** every `GATHER_MODE` (it's the front of the scatter program).

## 2. Problem & contract
**Computation.** From each cell's position `(cx,cy)` and size `(csx,csy)`, compute: `bxl = max(0, floor((cx−xl)/bsx))` and `overlap_x[J] = max(0, min(cx+csx, bin_right) − max(cx, bin_left))` where `bin_left = xl + (bxl+J)·bsx` (and the same for y). **Inputs:** DRAM px/py/sx/sy tiles. **Output CBs:** `bxl`(c_4), `byl`(c_5), `ox[0..7]`(c_6..c_13), `oy[0..7]`(c_14..c_21). **Invariants:** 2-DST-tile budget; reciprocal bin-index corrected against exact bin edges (±1 ULP fix).

## 3. Mechanism (how it works)
- **v4_reader (BRISC/NCRISC):** `noc_async_read_page` of px/py/sx/sy into CBs c_0..c_3.
- **v4_compute (TRISC SFPU):** `sfpu_floor((cx−xl)·inv_bsx)` then `correct_bin_idx()` against exact bin edges (the reciprocal multiply can be ±1 ULP off → floor lands wrong; correction uses exact `origin + idx·bin_size`). Then the box-overlap clamp for each of 8 bins.
- **G-PRESCALE:** optionally pre-multiplies `ox/oy` by `sqrt(inv_bin_area)` (host-injected) so downstream `area = ox·oy` is already bin-area-normalized.
- **THE TRICK / why it matters for the testbench:** this **box-overlap** model (`overlap = clamp(min(hi)−max(lo))`) is exactly what `synth.ref_density` computes — which is why the isolated forward tests get rel_l2 ~1e-6 (exact match).

## 4. Performance (measured — TT)
- Folded into each scatter mode's time (it's the front stage). The SFPU outer-product micro-cost is ~12.4 ns/cell ([[v11_stage_bprime_outer_product]]). Per-pass Tracy shows V4C-OX0 time is mostly NCRISC route back-pressure, not SFPU ([[v11_ox0_backpressure]]).

## 5. Correctness
- The box-overlap output is the ground-truth overlap model; matches the CPU reference exactly (rel_l2 ~1e-6 across the forward sweeps).

## 6. Gotchas / pitfalls
- **Reciprocal bin-index ±1 ULP** — must correct against exact bin edges (`correct_bin_idx`), else floor lands on the wrong bin.
- 2-DST-tile budget: `v4_compute` recomputes `bxl` inline inside `face_overlap_x` to stay within budget (LREG peak ~5).
- `sfpu_floor` precision; G-PRESCALE scale must be consistent with the downstream normalization.

## 7. When to use / avoid
- **Wins:** always — it's the shared front of every forward path. **Avoid:** N/A.

## 8. Provenance
- **Memory:** [[v11_stage_bprime_outer_product]], [[v11_g_prescale_win]], [[v11_ox0_backpressure]]
- **Host:** `host/v19_engine.cpp` (all GATHER_MODEs)
