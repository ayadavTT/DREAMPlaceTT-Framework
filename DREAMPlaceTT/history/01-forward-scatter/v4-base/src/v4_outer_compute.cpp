// SPDX-License-Identifier: Apache-2.0
//
// V4 Outer-Product TRISC Compute Kernel — Stage B' variant of v4_compute.
//
// Identical to v4_compute for bxl/byl/overlap_x/overlap_y, PLUS 64 additional
// SFPU passes that pre-compute outer[j, k] = ox[j] * oy[k] per cell, so the
// scatter side doesn't have to do the BRISC scalar multiply.
//
// Output CB layout (per tile of 1024 cells):
//   c_4     bxl                     (1 tile)
//   c_5     byl                     (1 tile)
//   c_6..c_13  ox[0..7]             (1 tile each — INTERNAL, consumed by THIS kernel)
//   c_14..c_21 oy[0..7]             (1 tile each — INTERNAL, consumed by THIS kernel)
//   c_22..c_29 outer[J=0..7][K=0..7] — c_(22+J) gets 8 tiles per batch (K=0..7)
//                                       SCATTER reads these.
//
// V11-outer scatter (v11_outer_scatter_*_dm.cpp) reads bxl, byl, and the 64
// outer-product values from c_22..c_29 — skips the inline ox*oy multiply.
//
// Per memory/v11_stage_bprime_outer_product.md: predicted ~6-8% V11 scatter
// improvement (multiplies were ~15% of per-cell BRISC time).
//
// Geometry constants injected as JIT #defines by host (ComputeConfig::defines).
//
// Uses the proven 2-DST-tile in-place pattern:
//   DST[0] = primary input (pos_x or pos_y), overwritten in-place with output
//   DST[1] = secondary input (sx or sy)
//
// Float32 tile element ordering is linear (no face permutation), verified by
// passthrough tests in tt-sfpu/.
//
// Runtime args:
//   0: n_tiles    number of tiles this core processes

#include <cstdint>
#if __has_include("api/compute/common.h")
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"
#else
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api.h"
#endif
#ifdef TRISC_MATH
#include "llk_math_eltwise_ternary_sfpu_params.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"

// ── Geometry constants (JIT #defines from host, with fallback defaults) ──────
#ifndef V4_BSX_F
#define V4_BSX_F 3.90625f
#endif
#ifndef V4_BSY_F
#define V4_BSY_F 3.90625f
#endif
#ifndef V4_INV_BSX_F
#define V4_INV_BSX_F 0.256f
#endif
#ifndef V4_INV_BSY_F
#define V4_INV_BSY_F 0.256f
#endif
#ifndef V4_XL_F
#define V4_XL_F 0.0f
#endif
#ifndef V4_YL_F
#define V4_YL_F 0.0f
#endif
#ifndef V4_MAX_OVERLAP
#define V4_MAX_OVERLAP 8
#endif
// G-PRESCALE: optional per-output scale (host injects sqrt(inv_bin_area) in
// V11 mode so area = ox*oy is pre-multiplied by inv_bin_area at scatter
// time; V11A-SCALE in gather then becomes a no-op).
#ifndef V4_OX_OY_SCALE_F
#define V4_OX_OY_SCALE_F 1.0f
#endif

// ── SFPU face functions — inside #ifdef TRISC_MATH ──────────────────────────
#ifdef TRISC_MATH

static constexpr float BSX  = V4_BSX_F;
static constexpr float BSY  = V4_BSY_F;
static constexpr float IBSX = V4_INV_BSX_F;
static constexpr float IBSY = V4_INV_BSY_F;
static constexpr float XL   = V4_XL_F;
static constexpr float YL   = V4_YL_F;
static constexpr float OX_OY_SCALE = V4_OX_OY_SCALE_F;

static constexpr uint32_t N = 32;
static constexpr float BIG  = 8388608.0f;  // 2^23

inline vFloat sfpu_floor(vFloat x) {
    vFloat r = (x + BIG) - BIG;
    v_if (r > x) { r -= 1.0f; } v_endif;
    return r;
}

// Correct a reciprocal-estimated bin index so that
//   origin + idx * bin_size <= pos < origin + (idx+1) * bin_size
// The reciprocal multiply can be off by ±1 ULP on the SFPU, causing floor()
// to land on the wrong side of an integer. This correction uses exact bin
// boundaries (origin + idx * bin_size) to guarantee the correct index.
inline vFloat correct_bin_idx(vFloat idx, vFloat pos, float origin, float bin_sz) {
    vFloat bl = origin + idx * bin_sz;
    v_if (bl > pos) { idx = idx - 1.0f; } v_endif;
    bl = origin + (idx + 1.0f) * bin_sz;
    v_if (bl <= pos) { idx = idx + 1.0f; } v_endif;
    return idx;
}

// bxl = max(0, floor((cx - xl) * inv_bsx)), corrected against exact bin edges
static void face_bxl(uint32_t d_px, uint32_t d_sx, uint32_t, uint32_t d_out) {
    for (uint32_t i = 0; i < 8; ++i) {
        vFloat cx = dst_reg[d_px * N + i];
        vFloat val = sfpu_floor((cx - XL) * IBSX);
        v_if (val < sfpi::vConst0) { val = sfpi::vConst0; } v_endif;
        val = correct_bin_idx(val, cx, XL, BSX);
        v_if (val < sfpi::vConst0) { val = sfpi::vConst0; } v_endif;
        dst_reg[d_out * N + i] = val;
    }
}

// byl = max(0, floor((cy - yl) * inv_bsy)), corrected against exact bin edges
static void face_byl(uint32_t d_py, uint32_t d_sy, uint32_t, uint32_t d_out) {
    for (uint32_t i = 0; i < 8; ++i) {
        vFloat cy = dst_reg[d_py * N + i];
        vFloat val = sfpu_floor((cy - YL) * IBSY);
        v_if (val < sfpi::vConst0) { val = sfpi::vConst0; } v_endif;
        val = correct_bin_idx(val, cy, YL, BSY);
        v_if (val < sfpi::vConst0) { val = sfpi::vConst0; } v_endif;
        dst_reg[d_out * N + i] = val;
    }
}

// overlap_x[J] = max(0, min(cx+csx, bin_right) - max(cx, bin_left))
// where bin_left = xl + (bxl + J) * bsx, bin_right = bin_left + bsx
// bxl recomputed from cx inline to stay within 2-DST-tile budget.
// The reciprocal-based estimate is corrected against exact bin edges.
// LREG peak: ~5 (cx, csx, t, lo, hi — bl lives only during correction)
template <int J>
static void face_overlap_x(uint32_t d_px, uint32_t d_sx, uint32_t, uint32_t d_out) {
    constexpr float Jf = static_cast<float>(J);
    for (uint32_t i = 0; i < 8; ++i) {
        vFloat cx  = dst_reg[d_px * N + i];
        vFloat csx = dst_reg[d_sx * N + i];
        vFloat t = sfpu_floor((cx - XL) * IBSX);
        v_if (t < sfpi::vConst0) { t = sfpi::vConst0; } v_endif;
        t = correct_bin_idx(t, cx, XL, BSX);
        v_if (t < sfpi::vConst0) { t = sfpi::vConst0; } v_endif;
        t = XL + (t + Jf) * BSX;                    // bin_left
        vFloat lo = cx;
        v_if (t > cx) { lo = t; } v_endif;          // lo = max(cx, bin_left)
        vFloat hi = cx + csx;
        t = t + BSX;                                 // bin_right
        v_if (t < hi) { hi = t; } v_endif;          // hi = min(cx+csx, bin_right)
        hi = hi - lo;
        v_if (hi < sfpi::vConst0) { hi = sfpi::vConst0; } v_endif;
        dst_reg[d_out * N + i] = hi * OX_OY_SCALE;
    }
}

// overlap_y[K] — identical structure with Y constants
template <int K>
static void face_overlap_y(uint32_t d_py, uint32_t d_sy, uint32_t, uint32_t d_out) {
    constexpr float Kf = static_cast<float>(K);
    for (uint32_t i = 0; i < 8; ++i) {
        vFloat cy  = dst_reg[d_py * N + i];
        vFloat csy = dst_reg[d_sy * N + i];
        vFloat t = sfpu_floor((cy - YL) * IBSY);
        v_if (t < sfpi::vConst0) { t = sfpi::vConst0; } v_endif;
        t = correct_bin_idx(t, cy, YL, BSY);
        v_if (t < sfpi::vConst0) { t = sfpi::vConst0; } v_endif;
        t = YL + (t + Kf) * BSY;
        vFloat lo = cy;
        v_if (t > cy) { lo = t; } v_endif;
        vFloat hi = cy + csy;
        t = t + BSY;
        v_if (t < hi) { hi = t; } v_endif;
        hi = hi - lo;
        v_if (hi < sfpi::vConst0) { hi = sfpi::vConst0; } v_endif;
        dst_reg[d_out * N + i] = hi * OX_OY_SCALE;
    }
}

// Stage B' outer-product face function — multiplies two DST tiles elementwise.
// d_a holds ox[J] (from CB c_(6+J)), d_b holds oy[K] (from CB c_(14+K)).
// d_out receives outer[J, K] = ox[J] * oy[K] for all 1024 cells in the tile.
static void face_outer_product(uint32_t d_a, uint32_t d_b, uint32_t, uint32_t d_out) {
    for (uint32_t i = 0; i < 8; ++i) {
        vFloat a = dst_reg[d_a * N + i];
        vFloat b = dst_reg[d_b * N + i];
        dst_reg[d_out * N + i] = a * b;
    }
}

#endif  // TRISC_MATH

// ── SFPU_PASS macro — one complete acquire→copy→compute→pack→release cycle ──
// face_fn is only expanded inside MATH() which compiles to nothing on non-MATH
// TRISCs, so the face function name never reaches the compiler on TRISC_UNPACK
// or TRISC_PACK.
// Wrapped in a brace-scoped DeviceZoneScopedN so each of the 18 passes per
// tile shows up as its own Tracy zone; the outer V4C-SFPU-TILE zone still
// nests them for tile-level totals.
#define SFPU_PASS(cb_a, cb_b, cb_out, face_fn, zone_label)  \
    {                                                       \
        DeviceZoneScopedN(zone_label);                      \
        tile_regs_acquire();                                \
        copy_tile_init(cb_a);  copy_tile(cb_a, 0, 0);       \
        copy_tile_init(cb_b);  copy_tile(cb_b, 0, 1);       \
        MATH(_llk_math_eltwise_ternary_sfpu_params_(        \
            face_fn, 0, 1, 0, 0,                            \
            static_cast<int>(VectorMode::RC)));             \
        tile_regs_commit();                                 \
        cb_reserve_back(cb_out, 1);                         \
        tile_regs_wait();                                   \
        pack_reconfig_data_format(cb_out);                  \
        pack_tile(0, cb_out);                               \
        tile_regs_release();                                \
        cb_push_back(cb_out, 1);                            \
    }

void kernel_main() {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_px = tt::CBIndex::c_0;
    constexpr auto cb_py = tt::CBIndex::c_1;
    constexpr auto cb_sx = tt::CBIndex::c_2;
    constexpr auto cb_sy = tt::CBIndex::c_3;

    init_sfpu(cb_px, tt::CBIndex::c_4);

    for (uint32_t ti = 0; ti < n_tiles; ++ti) {
        {
            DeviceZoneScopedN("V4C-CB-WAIT");
            cb_wait_front(cb_px, 1);
            cb_wait_front(cb_py, 1);
            cb_wait_front(cb_sx, 1);
            cb_wait_front(cb_sy, 1);
        }

        // Pass 1-2: bxl, byl
        {
        DeviceZoneScopedN("V4C-SFPU-TILE");
        SFPU_PASS(cb_px, cb_sx, 4u,  face_bxl, "V4C-BXL");
        SFPU_PASS(cb_py, cb_sy, 5u,  face_byl, "V4C-BYL");

        // Passes 3-10: overlap_x[0..7]
#if V4_MAX_OVERLAP >= 1
        SFPU_PASS(cb_px, cb_sx, 6u,  face_overlap_x<0>, "V4C-OX0");
#endif
#if V4_MAX_OVERLAP >= 2
        SFPU_PASS(cb_px, cb_sx, 7u,  face_overlap_x<1>, "V4C-OX1");
#endif
#if V4_MAX_OVERLAP >= 3
        SFPU_PASS(cb_px, cb_sx, 8u,  face_overlap_x<2>, "V4C-OX2");
#endif
#if V4_MAX_OVERLAP >= 4
        SFPU_PASS(cb_px, cb_sx, 9u,  face_overlap_x<3>, "V4C-OX3");
#endif
#if V4_MAX_OVERLAP >= 5
        SFPU_PASS(cb_px, cb_sx, 10u, face_overlap_x<4>, "V4C-OX4");
#endif
#if V4_MAX_OVERLAP >= 6
        SFPU_PASS(cb_px, cb_sx, 11u, face_overlap_x<5>, "V4C-OX5");
#endif
#if V4_MAX_OVERLAP >= 7
        SFPU_PASS(cb_px, cb_sx, 12u, face_overlap_x<6>, "V4C-OX6");
#endif
#if V4_MAX_OVERLAP >= 8
        SFPU_PASS(cb_px, cb_sx, 13u, face_overlap_x<7>, "V4C-OX7");
#endif

        // Passes 11-18: overlap_y[0..7]
#if V4_MAX_OVERLAP >= 1
        SFPU_PASS(cb_py, cb_sy, 14u, face_overlap_y<0>, "V4C-OY0");
#endif
#if V4_MAX_OVERLAP >= 2
        SFPU_PASS(cb_py, cb_sy, 15u, face_overlap_y<1>, "V4C-OY1");
#endif
#if V4_MAX_OVERLAP >= 3
        SFPU_PASS(cb_py, cb_sy, 16u, face_overlap_y<2>, "V4C-OY2");
#endif
#if V4_MAX_OVERLAP >= 4
        SFPU_PASS(cb_py, cb_sy, 17u, face_overlap_y<3>, "V4C-OY3");
#endif
#if V4_MAX_OVERLAP >= 5
        SFPU_PASS(cb_py, cb_sy, 18u, face_overlap_y<4>, "V4C-OY4");
#endif
#if V4_MAX_OVERLAP >= 6
        SFPU_PASS(cb_py, cb_sy, 19u, face_overlap_y<5>, "V4C-OY5");
#endif
#if V4_MAX_OVERLAP >= 7
        SFPU_PASS(cb_py, cb_sy, 20u, face_overlap_y<6>, "V4C-OY6");
#endif
#if V4_MAX_OVERLAP >= 8
        SFPU_PASS(cb_py, cb_sy, 21u, face_overlap_y<7>, "V4C-OY7");
#endif

        } // end V4C-SFPU-TILE

        // ── Stage B': 64 outer-product passes ────────────────────────────
        // For each (J, K) ∈ [0, 8)^2, compute outer[J, K] = ox[J] * oy[K]
        // by reading c_(6+J) and c_(14+K) (just pushed above) and writing
        // to c_(22+J). Each c_(22+J) receives 8 tile pushes (one per K),
        // so by the end of the batch c_22..c_29 each hold 8 tiles.
        //
        // copy_tile(cb, 0, ...) reads tile-index-0 (the front tile) of cb.
        // Since this kernel just pushed one tile each to c_6..c_21 and hasn't
        // popped them, tile-0 is the correct tile.
        {
            DeviceZoneScopedN("V4OP-OUTER-TILE");
            for (uint32_t j = 0; j < 8; ++j) {
                for (uint32_t k = 0; k < 8; ++k) {
                    tile_regs_acquire();
                    copy_tile_init(6u + j);  copy_tile(6u + j, 0, 0);
                    copy_tile_init(14u + k); copy_tile(14u + k, 0, 1);
                    MATH((_llk_math_eltwise_ternary_sfpu_params_(
                        face_outer_product, 0, 1, 0, 0,
                        static_cast<int>(VectorMode::RC))));
                    tile_regs_commit();
                    cb_reserve_back(32u + j, 1);
                    tile_regs_wait();
                    pack_reconfig_data_format(32u + j);
                    pack_tile(0, 32u + j);
                    tile_regs_release();
                    cb_push_back(32u + j, 1);
                }
            }
        } // end V4OP-OUTER-TILE

        // Pop the ox/oy intermediate tiles — we've consumed them and no
        // downstream kernel reads c_6..c_21 in v11outer mode.
        for (uint32_t i = 0; i < 8; ++i) {
            cb_pop_front(6u + i, 1);
            cb_pop_front(14u + i, 1);
        }

        cb_pop_front(cb_px, 1);
        cb_pop_front(cb_py, 1);
        cb_pop_front(cb_sx, 1);
        cb_pop_front(cb_sy, 1);
    }
}
