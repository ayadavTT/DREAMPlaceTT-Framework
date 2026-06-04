// SPDX-License-Identifier: Apache-2.0
//
// V14 Architecture-A — Two-phase TRISC compute kernel.
//
// Phase 1 (SFPU): identical to v4_compute.cpp. For each cell-tile of 1024
//   cells, computes bxl, byl, overlap_x[0..7], overlap_y[0..7] from
//   (px, py, sx, sy) inputs. Outputs 18 fp32 tiles to CBs c_4..c_21.
//
// Phase 2 (FPU matmul): identical to v13_accum_compute_mt.cpp. Loops until
//   BRISC pushes an ALL_DONE marker in CB_OX_FPU (c_22). Each tile group
//   ends with a per-tile SENTINEL (0xFFFFFFFF). After ALL tile groups, BRISC
//   pushes an ALL_DONE (0xFFFFFFFE) batch. TRISC detects it and exits.
//
// Runtime args:
//   0: n_scatter_tiles   — number of cell-tiles for Phase 1 (SFPU)

#include <cstdint>
#if __has_include("api/compute/common.h")
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/experimental/matmul_custom.h"
#else
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api.h"
#include "compute_kernel_api/experimental/matmul_custom.h"
#endif
#ifdef TRISC_MATH
#include "llk_math_eltwise_ternary_sfpu_params.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"
#include "v13_llk_math_kaccum.h"
#include "v13_llk_unpack_paired.h"

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
#ifndef V4_OX_OY_SCALE_F
#define V4_OX_OY_SCALE_F 1.0f
#endif

// ── Phase 1 CB indices ──────────────────────────────────────────────────────
constexpr auto cb_px = tt::CBIndex::c_0;
constexpr auto cb_py = tt::CBIndex::c_1;
constexpr auto cb_sx = tt::CBIndex::c_2;
constexpr auto cb_sy = tt::CBIndex::c_3;

// ── Phase 2 CB indices ──────────────────────────────────────────────────────
constexpr uint32_t CB_OX_FPU  = tt::CBIndex::c_22;
constexpr uint32_t CB_OY_FPU  = tt::CBIndex::c_23;
constexpr uint32_t CB_PARTIAL = tt::CBIndex::c_25;

// Per-tile-group end sentinel (bf16 NaN in all elements).
constexpr uint32_t SENTINEL   = 0xFFFFFFFFu;
// Global "all tile groups done" sentinel.
constexpr uint32_t ALL_DONE   = 0xFFFFFFFEu;
constexpr uint32_t N_INFLIGHT = 4u;

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

inline vFloat correct_bin_idx(vFloat idx, vFloat pos, float origin, float bin_sz) {
    vFloat bl = origin + idx * bin_sz;
    v_if (bl > pos) { idx = idx - 1.0f; } v_endif;
    bl = origin + (idx + 1.0f) * bin_sz;
    v_if (bl <= pos) { idx = idx + 1.0f; } v_endif;
    return idx;
}

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
        t = XL + (t + Jf) * BSX;
        vFloat lo = cx;
        v_if (t > cx) { lo = t; } v_endif;
        vFloat hi = cx + csx;
        t = t + BSX;
        v_if (t < hi) { hi = t; } v_endif;
        hi = hi - lo;
        v_if (hi < sfpi::vConst0) { hi = sfpi::vConst0; } v_endif;
        dst_reg[d_out * N + i] = hi * OX_OY_SCALE;
    }
}

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

#endif  // TRISC_MATH

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
    const uint32_t n_scatter_tiles = get_arg_val<uint32_t>(0);

    // ═══════════════════════════════════════════════════════════════════════
    // Phase 1: SFPU overlap computation (clone of v4_compute body)
    // ═══════════════════════════════════════════════════════════════════════
    init_sfpu(cb_px, tt::CBIndex::c_4);

    for (uint32_t ti = 0; ti < n_scatter_tiles; ++ti) {
        {
            DeviceZoneScopedN("V14C-CB-WAIT");
            cb_wait_front(cb_px, 1);
            cb_wait_front(cb_py, 1);
            cb_wait_front(cb_sx, 1);
            cb_wait_front(cb_sy, 1);
        }

        {
        DeviceZoneScopedN("V14C-SFPU-TILE");
        SFPU_PASS(cb_px, cb_sx, 4u,  face_bxl);
        SFPU_PASS(cb_py, cb_sy, 5u,  face_byl);

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

        } // end V14C-SFPU-TILE

        cb_pop_front(cb_px, 1);
        cb_pop_front(cb_py, 1);
        cb_pop_front(cb_sx, 1);
        cb_pop_front(cb_sy, 1);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Phase 2: FPU matmul accumulation (clone of v13_accum_compute_mt body)
    //
    // TRISC loops until BRISC pushes an ALL_DONE marker. Each tile group
    // ends with a per-tile SENTINEL. After all groups, BRISC pushes ALL_DONE.
    // ═══════════════════════════════════════════════════════════════════════
    mm_init(CB_OX_FPU, CB_OY_FPU, CB_PARTIAL);
    UNPACK(( v13::_v13_unpack_paired_init_() ));
    MATH(( v13::_llk_math_matmul_kaccum_init_<MathFidelity::HiFi4>() ));

    while (true) {
        DeviceZoneScopedN("V14C-FPU-TILE");
        tile_regs_acquire();

        bool done = false;
        {
            DeviceZoneScopedN("V14C-MATMUL-LOOP");
            while (true) {
                {
                    DeviceZoneScopedN("V14C-WAIT-OX");
                    cb_wait_front(CB_OX_FPU, N_INFLIGHT);
                }
                uint32_t v = read_tile_value(CB_OX_FPU, 0u, 0u);
                if (v == ALL_DONE) {
                    cb_pop_front(CB_OX_FPU, N_INFLIGHT);
                    done = true;
                    break;
                }
                if (v == SENTINEL) {
                    cb_pop_front(CB_OX_FPU, N_INFLIGHT);
                    break;
                }
                {
                    DeviceZoneScopedN("V14C-WAIT-OY");
                    cb_wait_front(CB_OY_FPU, N_INFLIGHT);
                }

                {
                    DeviceZoneScopedN("V14C-UNPACK");
                    UNPACK(( v13::_v13_unpack_paired_run_(CB_OX_FPU, CB_OY_FPU, N_INFLIGHT) ));
                }

                {
                    DeviceZoneScopedN("V14C-MATH");
                    MATH(( v13::_llk_math_matmul_kaccum_<MathFidelity::HiFi4>(0u, N_INFLIGHT) ));
                }

                {
                    DeviceZoneScopedN("V14C-POP");
                    cb_pop_front(CB_OX_FPU, N_INFLIGHT);
                    cb_pop_front(CB_OY_FPU, N_INFLIGHT);
                }
            }
        }

        if (done) {
            tile_regs_release();
            break;
        }

        tile_regs_commit();
        {
            DeviceZoneScopedN("V14C-PACK");
            tile_regs_wait();
            cb_reserve_back(CB_PARTIAL, 1);
            pack_reconfig_data_format(CB_PARTIAL);
            pack_tile(0, CB_PARTIAL);
            cb_push_back(CB_PARTIAL, 1);
            tile_regs_release();
        }
    }
}
