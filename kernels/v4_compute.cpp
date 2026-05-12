// SPDX-License-Identifier: Apache-2.0
//
// V4 TRISC Compute Kernel — SFPU (hardware FP32) bin-index + overlap computation.
//
// For each tile of 1024 cells, produces 18 output tiles:
//   bxl, byl                 (2 tiles — bin index lower bounds as float)
//   overlap_x[0..7]          (8 tiles — 1D X overlap for bin bxl+j)
//   overlap_y[0..7]          (8 tiles — 1D Y overlap for bin byl+k)
//
// NCRISC reads these outputs and forms (bx, by, area) tuples via:
//   bx = (int)bxl + j,  by = (int)byl + k,  area = overlap_x[j] * overlap_y[k]
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

// ── SFPU face functions — inside #ifdef TRISC_MATH ──────────────────────────
#ifdef TRISC_MATH

static constexpr float BSX  = V4_BSX_F;
static constexpr float BSY  = V4_BSY_F;
static constexpr float IBSX = V4_INV_BSX_F;
static constexpr float IBSY = V4_INV_BSY_F;
static constexpr float XL   = V4_XL_F;
static constexpr float YL   = V4_YL_F;

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
        dst_reg[d_out * N + i] = hi;
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
        dst_reg[d_out * N + i] = hi;
    }
}

#endif  // TRISC_MATH

// ── SFPU_PASS macro — one complete acquire→copy→compute→pack→release cycle ──
// face_fn is only expanded inside MATH() which compiles to nothing on non-MATH
// TRISCs, so the face function name never reaches the compiler on TRISC_UNPACK
// or TRISC_PACK.
#define SFPU_PASS(cb_a, cb_b, cb_out, face_fn)       \
    tile_regs_acquire();                              \
    copy_tile_init(cb_a);  copy_tile(cb_a, 0, 0);    \
    copy_tile_init(cb_b);  copy_tile(cb_b, 0, 1);    \
    MATH(_llk_math_eltwise_ternary_sfpu_params_(      \
        face_fn, 0, 1, 0, 0,                         \
        static_cast<int>(VectorMode::RC)));           \
    tile_regs_commit();                               \
    cb_reserve_back(cb_out, 1);                       \
    tile_regs_wait();                                 \
    pack_reconfig_data_format(cb_out);                \
    pack_tile(0, cb_out);                             \
    tile_regs_release();                              \
    cb_push_back(cb_out, 1)

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
        SFPU_PASS(cb_px, cb_sx, 4u,  face_bxl);
        SFPU_PASS(cb_py, cb_sy, 5u,  face_byl);

        // Passes 3-10: overlap_x[0..7]
#if V4_MAX_OVERLAP >= 1
        SFPU_PASS(cb_px, cb_sx, 6u,  face_overlap_x<0>);
#endif
#if V4_MAX_OVERLAP >= 2
        SFPU_PASS(cb_px, cb_sx, 7u,  face_overlap_x<1>);
#endif
#if V4_MAX_OVERLAP >= 3
        SFPU_PASS(cb_px, cb_sx, 8u,  face_overlap_x<2>);
#endif
#if V4_MAX_OVERLAP >= 4
        SFPU_PASS(cb_px, cb_sx, 9u,  face_overlap_x<3>);
#endif
#if V4_MAX_OVERLAP >= 5
        SFPU_PASS(cb_px, cb_sx, 10u, face_overlap_x<4>);
#endif
#if V4_MAX_OVERLAP >= 6
        SFPU_PASS(cb_px, cb_sx, 11u, face_overlap_x<5>);
#endif
#if V4_MAX_OVERLAP >= 7
        SFPU_PASS(cb_px, cb_sx, 12u, face_overlap_x<6>);
#endif
#if V4_MAX_OVERLAP >= 8
        SFPU_PASS(cb_px, cb_sx, 13u, face_overlap_x<7>);
#endif

        // Passes 11-18: overlap_y[0..7]
#if V4_MAX_OVERLAP >= 1
        SFPU_PASS(cb_py, cb_sy, 14u, face_overlap_y<0>);
#endif
#if V4_MAX_OVERLAP >= 2
        SFPU_PASS(cb_py, cb_sy, 15u, face_overlap_y<1>);
#endif
#if V4_MAX_OVERLAP >= 3
        SFPU_PASS(cb_py, cb_sy, 16u, face_overlap_y<2>);
#endif
#if V4_MAX_OVERLAP >= 4
        SFPU_PASS(cb_py, cb_sy, 17u, face_overlap_y<3>);
#endif
#if V4_MAX_OVERLAP >= 5
        SFPU_PASS(cb_py, cb_sy, 18u, face_overlap_y<4>);
#endif
#if V4_MAX_OVERLAP >= 6
        SFPU_PASS(cb_py, cb_sy, 19u, face_overlap_y<5>);
#endif
#if V4_MAX_OVERLAP >= 7
        SFPU_PASS(cb_py, cb_sy, 20u, face_overlap_y<6>);
#endif
#if V4_MAX_OVERLAP >= 8
        SFPU_PASS(cb_py, cb_sy, 21u, face_overlap_y<7>);
#endif

        } // end V4C-SFPU-TILE

        cb_pop_front(cb_px, 1);
        cb_pop_front(cb_py, 1);
        cb_pop_front(cb_sx, 1);
        cb_pop_front(cb_sy, 1);
    }
}
