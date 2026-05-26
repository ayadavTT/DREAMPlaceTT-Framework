# V21a SFPU port — handoff

**Status as of 2026-05-26 (this commit)**: scalar BRISC/NCRISC microbench shipped, **bit-exact** vs DREAMPlace's CPU `electric_force_cpp` at 32×32 / a1_2048 / bb3_2048 scales (0 mismatches, max_diff = 0), **but 0.30–0.39× CPU 16T**. The TT-Tracy device profile shows the bottleneck unambiguously: `V21MB-EF-MATH` zone (per-cell × per-bin scalar float on data-movement RISCs) is **89.5% of per-batch time**; DRAM barrier is 0.7%. The next step is to move the inner-loop math to the SFPU on TRISC, following the V4-compute / V11outer pattern.

This handoff is what a fresh session needs to land the SFPU port without re-deriving anything. It is concrete enough to start coding day 1.

---

## 1. The starting point in this commit

Files (all in this commit, except where noted):
- `kernels/v21_ef_common.h` — current scalar BRISC/NCRISC body. The `V21MB-EF-MATH` zone is where the 90% time is.
- `kernels/v21_ef_brisc.cpp`, `kernels/v21_ef_ncrisc.cpp` — thin wrappers around `v21_ef_main()`.
- `host/v21_electric_force_microbench_host.cpp` — host driver (loads inputs from binary, allocates DRAM, builds program with 2 kernels, JIT warmup + 5 timed runs).
- `integration/v21_ef_microbench.py` — Python orchestrator (synthesizes inputs, runs binary, calls DREAMPlace's `electric_potential_cpp.electric_force` on the same inputs at OMP_NUM_THREADS=16, diff + report).
- `host/CMakeLists.txt` — already has the `v21_ef_microbench` target (committed in `f8da93c`).
- `docs/V21_DREAMPLACE_ON_CHIP_RESEARCH.md` — §11 has the full measured table + profile breakdown.

Build + run (chip + Python 3.8 venv set up per REPRODUCE.md):
```bash
cd $FW && bash scripts/build_server.sh && \
  cd host/build && make v21_ef_microbench
export TT_METAL_HOME=$FW/tt-metal TT_METAL_RUNTIME_ROOT=$FW/tt-metal
DREAMPlace/dp_env/bin/python3 integration/v21_ef_microbench.py --M 2048 --N 2048 --num-nodes 540000
```

Reference timings on bh-38 (Blackhole p150b, 110 cores) — keep these visible to know whether the SFPU port is helping:

| Workload | TT scalar | CPU 16T | Target after SFPU |
|---|---:|---:|---:|
| a1_2048 (540 K cells) | 27.6 ms | 10.9 ms | ≤ 5 ms (≥ 2× CPU) |
| bb3_2048 (1.1 M cells) | 55.9 ms | 16.7 ms | ≤ 8 ms (≥ 2× CPU) |

Target = roughly the per-cell `V21MB-EF-MATH` time (10 µs scalar) dropping to ≤ 1 µs once on SFPU (proven achievable in `v11_outer_scatter_dm.cpp` and `v4_outer_compute.cpp`).

---

## 2. Math the SFPU kernel must implement (bit-exact reference)

For each cell `c` in `[0, num_movable) ∪ [num_nodes - num_filler, num_nodes)`:

```
node_x = pos[c] + offset_x[c];   node_y = pos[num_nodes + c] + offset_y[c]
nsx    = node_size_x_clamped[c]; nsy = node_size_y_clamped[c];  r = ratio[c]
bin_xl = max(0, floor((node_x - xl) * inv_bsx))
bin_xh = min(M, floor((node_x + nsx - xl) * inv_bsx) + 1)
bin_yl = max(0, floor((node_y - yl) * inv_bsy))
bin_yh = min(N, floor((node_y + nsy - yl) * inv_bsy) + 1)

gx = gy = 0
for k in [bin_xl, bin_xh):
    px = min(node_x + nsx, xl + (k+1)*bsx) - max(node_x, xl + k*bsx)
    for h in [bin_yl, bin_yh):
        py    = min(node_y + nsy, yl + (h+1)*bsy) - max(node_y, yl + h*bsy)
        area  = px * py
        gx   += area * field_x[k*N + h]
        gy   += area * field_y[k*N + h]
gx = -gx * r;   gy = -gy * r            // negation baked in (matches electric_potential.py:235)
```

Critical semantics (matches `DREAMPlace/dreamplace/ops/electric_potential/src/electric_force.cpp:133-147`):
- `+1` happens BEFORE the upper-bound clamp, not after.
- `node_size_x_clamped` (post-`sqrt(2)*bsx` clamp) is used in the loop; `ratio` corrects post-loop.
- `triangle_density_function` does NOT clamp px or py to ≥ 0. (They're nonneg in practice because the bin range is computed to overlap.)
- Field is row-major `idx = k * num_bins_y + h`, NOT the transpose.
- Per-cell output is negated: `output[c] = -gx_c, output[num_nodes + c] = -gy_c`.

The current scalar kernel in `kernels/v21_ef_common.h` already implements this and validates bit-exact. **Keep it as the reference.** Implement the SFPU version alongside, with a JIT-compile-time switch (e.g., `V21_USE_SFPU` define) so the scalar version remains buildable for accuracy regression.

---

## 3. Architecture: 3-RISC split with circular-buffer plumbing

Mirror V4-compute / V11-outer. Each Tensix core runs three kernels in parallel:

```
   BRISC (NoC 0)                NCRISC (NoC 1)               TRISC compute
   ─────────────────            ─────────────────            ──────────────────
   Read 32 cells'      ───┐     Read 32 cells'      ───┐     Wait on all 4 CBs
   ox, oy, nsx, nsy,      │     pos_x, pos_y,           │
   ratio constants        │     ratio                   │     SFPU computes
   from per-core page     │     from pos buffer         │       node_x, node_y,
                          │                             │       bin range,
   For each (k, h)    pack│     For each (k, h)     pack│       px, py per (k,h),
   in cell's bin range:   │     in cell's bin range:    │       area*field_x sum,
   read field_x[k,h]   ──►│     read field_y[k,h]    ──►│       area*field_y sum
                       cb_fx│                          cb_fy│
   Pack into 32-cell      │     Pack into 32-cell        │     Negate, * ratio
   tile (cb_fx)           │     tile (cb_fy)             │     Pack to cb_grad
                          │                              │
                          │                              │     ──► cb_grad
                          │     Constants tile ────► cb_const │
                                                              │     Read cb_grad,
   Output writer (could be either BRISC or NCRISC, post-SFPU)       write to DRAM
```

Circular buffers (in `host/v21_electric_force_microbench_host.cpp`):
- `cb_const` (c_0): per-tile per-cell constants — `[ox, oy, nsx, nsy, ratio, _]` packed. 32 cells × 32 B = 1 KB per tile.
- `cb_pos` (c_1): per-tile per-cell positions — `[pos_x, pos_y]`. 32 cells × 8 B = 256 B per tile.
- `cb_fx` (c_2): per-tile field_x values, one 32-lane vector per cell, padded to 64 (k,h) max. 32 cells × 64 floats = 8 KB per tile. **This is the biggest CB.**
- `cb_fy` (c_3): same as cb_fx for field_y.
- `cb_grad` (c_4): per-tile output gx, gy. 32 cells × 8 B = 256 B per tile.

Allocate each CB with 2 slots (double-buffering for pipeline). Total CB L1 ≈ 35 KB per core. Fits comfortably in the 1 MB L1 budget.

**Tile = 32 cells.** A "tile" in this kernel is the SFPU vector-tile unit (32 lanes × 32 rows = 1024 elements). For V21a we map 32 cells across lanes, with the per-cell (k,h) iteration done as a sequential outer loop within the SFPU compute (mask out lanes whose cells have shorter (k,h) ranges via `v_if`).

---

## 4. SFPU kernel skeleton (write this in `kernels/v21_ef_compute.cpp`)

Use `kernels/v4_outer_compute.cpp` as the structural template — it already has the per-cell-vector v_if masking + bin-range arithmetic on SFPU. The new piece is the per-(k,h) accumulate.

```cpp
// V21a SFPU compute kernel — per-cell electric_force on TRISC SFPU.
//
// Lane mapping: 32 lanes = 32 cells per tile.
// Outer loop within the SFPU "face function": iterate (k, h) sequentially across
// the maximum bin range any cell in the tile reaches, mask lanes whose cell
// doesn't reach this (k, h).

#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "llk_math_eltwise_ternary_sfpu_params.h"

#define V21_MAX_K 8     // matches scalar kernel's V21_MAX_K_RANGE
#define V21_MAX_H 8

#ifdef TRISC_MATH
template <int K_RANGE, int H_RANGE>
static void face_v21_ef(uint32_t d_const, uint32_t d_pos, uint32_t d_fx, uint32_t d_out) {
    // d_const lanes hold (ox, oy, nsx, nsy, ratio, _, _, _)
    // d_pos   lanes hold (pos_x, pos_y, _, _, _, _, _, _)
    // d_fx    lanes hold field_x[k*H_RANGE + h] for this cell, flat 64 elements
    //          across 2 rows (1 row = 32 lanes = 32 (k,h) pairs for this cell).
    // d_out   lane[0] = gx, lane[1] = gy   (or use two passes for gx and gy).

    vFloat pos_x = dst_reg[d_pos * N + 0];
    vFloat pos_y = dst_reg[d_pos * N + 1];
    vFloat ox    = dst_reg[d_const * N + 0];
    vFloat oy    = dst_reg[d_const * N + 1];
    vFloat nsx   = dst_reg[d_const * N + 2];
    vFloat nsy   = dst_reg[d_const * N + 3];
    vFloat ratio = dst_reg[d_const * N + 4];

    vFloat node_x = pos_x + ox;
    vFloat node_y = pos_y + oy;

    // Bin range — match scalar kernel's truncation/clamp semantics.
    vFloat bin_xl_f = sfpu_floor((node_x         - XL) * IBSX);
    vFloat bin_xh_f = sfpu_floor((node_x + nsx   - XL) * IBSX) + 1.0f;
    vFloat bin_yl_f = sfpu_floor((node_y         - YL) * IBSY);
    vFloat bin_yh_f = sfpu_floor((node_y + nsy   - YL) * IBSY) + 1.0f;
    v_if (bin_xl_f < 0.0f) { bin_xl_f = 0.0f; } v_endif;
    v_if (bin_xh_f > M_F) { bin_xh_f = M_F; } v_endif;
    v_if (bin_yl_f < 0.0f) { bin_yl_f = 0.0f; } v_endif;
    v_if (bin_yh_f > N_F) { bin_yh_f = N_F; } v_endif;

    vFloat gx_acc = vConst0;
    vFloat gy_acc = vConst0;

    // Outer (k, h) iteration — runs the same K_RANGE * H_RANGE times for every
    // lane; lanes whose cell has a smaller range are masked out per iter.
    for (int kk = 0; kk < K_RANGE; ++kk) {
        vFloat k_f = bin_xl_f + (float)kk;
        v_if (k_f < bin_xh_f) {
            // px = min(node_x + nsx, xl + (k+1)*bsx) - max(node_x, xl + k*bsx)
            vFloat bin_x_lo = XL + k_f * BSX;
            vFloat bin_x_hi = bin_x_lo + BSX;
            vFloat node_x_top = node_x + nsx;
            vFloat px_top = node_x_top;
            v_if (bin_x_hi < node_x_top) { px_top = bin_x_hi; } v_endif;
            vFloat px_bot = node_x;
            v_if (bin_x_lo > node_x) { px_bot = bin_x_lo; } v_endif;
            vFloat px = px_top - px_bot;

            for (int hh = 0; hh < H_RANGE; ++hh) {
                vFloat h_f = bin_yl_f + (float)hh;
                v_if (h_f < bin_yh_f) {
                    // py similar to px
                    vFloat bin_y_lo = YL + h_f * BSY;
                    vFloat bin_y_hi = bin_y_lo + BSY;
                    vFloat node_y_top = node_y + nsy;
                    vFloat py_top = node_y_top;
                    v_if (bin_y_hi < node_y_top) { py_top = bin_y_hi; } v_endif;
                    vFloat py_bot = node_y;
                    v_if (bin_y_lo > node_y) { py_bot = bin_y_lo; } v_endif;
                    vFloat py = py_top - py_bot;

                    vFloat area = px * py;
                    // d_fx[N*kk + hh] holds field_x[k_cell, h_cell] for THIS lane (cell)
                    vFloat fx = dst_reg[d_fx * N + kk * H_RANGE + hh];
                    gx_acc += area * fx;
                    // ... similarly field_y (separate compute pass or share buffer)
                } v_endif;
            }
        } v_endif;
    }

    dst_reg[d_out * N + 0] = -gx_acc * ratio;
    dst_reg[d_out * N + 1] = -gy_acc * ratio;  // gy in a parallel pass
}
#endif // TRISC_MATH
```

This is the **skeleton**, not the full kernel. Things still to nail down on top of v4_outer_compute.cpp's structure:
- Where the field_x / field_y values get packed into the SFPU dst_reg lanes (BRISC/NCRISC have to write them in the right tile-layout positions). Refer to `memory/v16_dst_register_layout.md` — Blackhole `dst_reg[i]` is **4 rows × 8 cols col-parity interleaved**, NOT 2×16.
- Whether `cb_fx` should hold one 64-element vector per cell (one row of 32 lanes × 2 = 64 floats per cell) or be split into 2-3 tiles per cell. Likely 2 tiles per cell.
- Per-cell tile re-init cost: if it's > 200 ns per cell, the 32-cells-per-tile batching may not be a win — measure early.

---

## 5. File-level work list (target order)

1. **`kernels/v21_ef_compute.cpp`** (new, ~300 LoC) — TRISC compute kernel. Template off `kernels/v4_outer_compute.cpp` for SFPU patterns + `kernels/v11op_bench_compute.cpp` for the simpler accumulate pattern.
2. **`kernels/v21_ef_brisc.cpp`** (rewrite, ~80 LoC) — pure data-movement: read constants, read field_x, pack into cb_const + cb_fx.
3. **`kernels/v21_ef_ncrisc.cpp`** (rewrite, ~80 LoC) — pure data-movement: read pos, read field_y, pack into cb_pos + cb_fy. (Also handle the output writer phase.)
4. **`host/v21_electric_force_microbench_host.cpp`** (modify, ~50 LoC delta) — allocate 5 CBs, register the 3rd kernel as `DataMovementProcessor::COMPUTE`, JIT defines for `V21_BSX, V21_INV_BSX, V21_XL, V21_M_F, V21_N_F` etc.
5. **`integration/v21_ef_microbench.py`** — unchanged. Reuses the same binary format.
6. **`docs/V21_DREAMPLACE_ON_CHIP_RESEARCH.md`** — append new sub-section under §11 with measured SFPU numbers.

---

## 6. Validation plan

After each step that compiles + runs, re-run:
```bash
python3 integration/v21_ef_microbench.py --M 32 --N 32 --num-nodes 80
```
Required: **0 mismatches**. If non-zero, suspect:
- Tile dst_reg layout (per V16 memory note)
- `v_if` masking missed a code path → uninitialized lane value
- FMA semantics — SFPU FMAs may round differently than CPU. If only the LSB drifts (rel_diff < 1e-5) and HPWL still converges, that's OK; tighten the tolerance in the orchestrator to `1e-4` (current default).

Once tiny is clean, escalate to a1_2048 (540 K cells) then bb3_2048 (1.1 M).

---

## 7. Risks (from prior memory notes)

- **Blackhole SFPU dst_reg layout** — `memory/v16_dst_register_layout.md`: 4×8 col-parity, NOT 2×16. Per-lane writes need `vConstTileId` predicates and cost the same per-instruction as the 2×16 the V16 handoff had assumed. **Expected SFPU-pack speedup vs BRISC scalar drops from 2× to ~1.2× for that specific pattern.** Our V21a pattern is different (we don't need per-lane writes — we want 32-cell tile-wise compute), but watch out for layout assumptions.
- **SFPU FMA rounding vs CPU** — single-precision is single-precision, but accumulation order can differ. Bit-exact may not hold; aim for `max_rel_diff < 1e-5` and verify HPWL doesn't drift when integrated.
- **CB sizing** — `cb_fx` is the largest at 8 KB × 2 slots = 16 KB. Plus the existing scalar-kernel L1 of ~15 KB. Total > 30 KB per core — should still fit (1 MB L1), but verify with cmake `cb_*_cfg` traces.
- **TRISC profile zones** — `DeviceZoneScopedN` works inside compute kernels too (see v4_compute.cpp). Add V21C-PREP, V21C-INNER, V21C-PACK zones early so we can confirm the SFPU is actually doing work and not stalling on CBs.
- **Per-iter performance regression on small cells** — at 80 cells (tiny test) the scalar version is 0.108 ms. The SFPU version has CB-init + JIT overhead per kernel launch (~500 µs?) that may make it SLOWER at small scale. Acceptable as long as it's faster at production scales (540K + 1.1M cells).

---

## 8. Definition of done

1. `v21_ef_microbench` builds with all 3 kernels via `bash scripts/build_server.sh` + `cd host/build && make v21_ef_microbench`.
2. Tiny smoke (32×32, 80 cells): 0 mismatches, max_rel_diff < 1e-5.
3. a1_2048 (540 K cells): 0 mismatches (or max_rel_diff < 1e-4), median kernel ms < 5 ms, **speedup ≥ 2× vs CPU 16T**.
4. bb3_2048 (1.1 M cells): same, median < 10 ms, speedup ≥ 2× vs CPU 16T.
5. Tracy profile shows V21C-INNER (TRISC compute) as the new dominant zone, with V21MB-EF-MATH (old scalar zone) gone.
6. `docs/V21_DREAMPLACE_ON_CHIP_RESEARCH.md` §11 appended with the new numbers + a "SFPU port outcome" subsection.
7. (Stretch) Engine integration — wire into `host/v19_engine.cpp` after `solve_device`, save 6–10 ms/iter on bb3 in the full DREAMPlace pipeline.

---

## 9. Related memory + handoff notes

- `memory/v16_dst_register_layout.md` — Blackhole SFPU `dst_reg` layout warning.
- `memory/v11_stage_bprime_outer_product.md` — past benchmark: SFPU 12 ns/cell vs BRISC 150–180 ns/cell — same order of magnitude as the gap we'd need to close on V21a.
- `memory/v19_block_fp32_outcome.md` — V19 architecture (which v21a will eventually plug into).
- `kernels/v4_outer_compute.cpp` — the closest template: per-cell × per-bin SFPU outer product with floor/clamp.
- `kernels/v11op_bench_compute.cpp` — simpler SFPU accumulate template.
- `docs/V21_DREAMPLACE_ON_CHIP_RESEARCH.md` §11 — the measured scalar-V21a perf + bottleneck analysis this handoff builds on.
