// SPDX-License-Identifier: Apache-2.0
//
// V21a electric_force — TRISC SFPU compute kernel
// (v6: N-kk-per-acquire — Plan A item #2, refined for 4-DST-slot constraint).
//
// fp32_dest_acc_en gives only 4 DST slots, so the original "merge gx+gy into
// one acquire by adding a 5th slot" approach overflows DST and clobbers px/py.
// Instead, this version reduces acquire count by holding the SAME accumulator
// in DST[3] across V21C_KK_PER_ACQ kk iterations within one acquire — the
// per-acquire LLK overhead is the dominant TRISC cost, not the FMA work itself
// (Tracy: V21C-BATCH 1674 µs / 20 acquires ≈ 80 µs/acquire of fixed overhead).
//
// V21C_KK_PER_ACQ = 1 → 20 acquires/batch (= v4 baseline)
// V21C_KK_PER_ACQ = 2 → 10 acquires/batch
// V21C_KK_PER_ACQ = 4 →  6 acquires/batch
// V21C_KK_PER_ACQ = 8 →  4 acquires/batch (whole-phase single acquire)
//
// cb_fx and cb_fy must have ≥ V21C_KK_PER_ACQ * 8 slots so all required tiles
// can be in-flight when TRISC enters the acquire. Host sets these.
//
// DST budget (4 slots, fp32):
//   DST[0]=px, DST[1]=py, DST[2]=fx_or_fy, DST[3]=acc (alive across full acquire)
//
// CB contract (must match BRISC + host):
//   c_0  cb_px        fp32 8 tiles/batch — tile k = px[c, k]
//   c_1  cb_py        fp32 8 tiles/batch — tile h = py[c, h]
//   c_2  cb_fx        fp32 64 tiles/batch (kk*8+hh order) — needs ≥ V21C_KK_PER_ACQ*8 slots
//   c_3  cb_fy        fp32 64 tiles/batch                — needs ≥ V21C_KK_PER_ACQ*8 slots
//   c_4  cb_neg_ratio fp32 1 tile/batch — -ratio[c]
//   c_5  cb_acc       fp32 cycled accumulator (4 slots)
//   c_16 cb_gx_out    fp32 1 tile/batch
//   c_17 cb_gy_out    fp32 1 tile/batch

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

#ifndef V21C_KK_PER_ACQ
#define V21C_KK_PER_ACQ 1
#endif

// Adaptive work bounds (host passes the global max bin span; default 8 = V21).
#ifndef V21S_MAX_K
#define V21S_MAX_K 8
#endif
#ifndef V21S_MAX_H
#define V21S_MAX_H 8
#endif

#ifdef TRISC_MATH

static constexpr uint32_t N = 32;

// DST[d_out] = DST[d_a] * DST[d_b] * DST[d_c]   (seed: overwrite)
static void face_mul3(uint32_t d_a, uint32_t d_b, uint32_t d_c, uint32_t d_out) {
    for (uint32_t i = 0; i < 8; ++i) {
        vFloat a = dst_reg[d_a * N + i];
        vFloat b = dst_reg[d_b * N + i];
        vFloat c = dst_reg[d_c * N + i];
        dst_reg[d_out * N + i] = a * b * c;
    }
}

// DST[d_out] += DST[d_a] * DST[d_b] * DST[d_c]  (in-place FMA into d_out)
static void face_fma3(uint32_t d_a, uint32_t d_b, uint32_t d_c, uint32_t d_out) {
    for (uint32_t i = 0; i < 8; ++i) {
        vFloat acc = dst_reg[d_out * N + i];
        vFloat a   = dst_reg[d_a   * N + i];
        vFloat b   = dst_reg[d_b   * N + i];
        vFloat c   = dst_reg[d_c   * N + i];
        dst_reg[d_out * N + i] = acc + a * b * c;
    }
}

// DST[d_out] = DST[d_a] * DST[d_b]
static void face_mul(uint32_t d_a, uint32_t d_b, uint32_t /*c*/, uint32_t d_out) {
    for (uint32_t i = 0; i < 8; ++i) {
        vFloat a = dst_reg[d_a * N + i];
        vFloat b = dst_reg[d_b * N + i];
        dst_reg[d_out * N + i] = a * b;
    }
}

#endif  // TRISC_MATH

// One acquire processing kk in [kk_base, kk_base + V21C_KK_PER_ACQ).
// is_first_acq: this is the GX (or GY) phase's first acquire — seed with mul3.
// is_last_acq:  this is the phase's last acquire — don't pack to cb_acc, the
//               caller will do the negscale directly (saves 1 pack+pop pair).
//               (Currently always pack; negscale stays separate.)
template <uint32_t CB_PX, uint32_t CB_PY, uint32_t CB_F>
static inline void run_kk_group(
    uint32_t kk_base, bool is_first_acq,
    uint32_t cb_acc, uint32_t /*cb_neg_ratio*/) {
    // Caller has already cb_wait_front'd CB_F for V21C_KK_PER_ACQ*8 tiles
    // and (if !is_first_acq) cb_wait_front'd cb_acc for 1 tile.

    tile_regs_acquire();
    if (!is_first_acq) {
        copy_tile_init(cb_acc); copy_tile(cb_acc, 0, 3);  // DST[3] = prev acc
    }

    for (uint32_t kk_off = 0; kk_off < V21C_KK_PER_ACQ; ++kk_off) {
        const uint32_t kk = kk_base + kk_off;
        copy_tile_init(CB_PX); copy_tile(CB_PX, kk, 0);   // DST[0] = px[kk]
        for (uint32_t hh = 0; hh < V21S_MAX_H; ++hh) {
            const uint32_t f_idx = kk_off * V21S_MAX_H + hh;  // tile index within CB_F front
            copy_tile_init(CB_PY); copy_tile(CB_PY, hh,    1);  // DST[1] = py[hh]
            copy_tile_init(CB_F);  copy_tile(CB_F,  f_idx, 2);  // DST[2] = f[kk_off, hh]
            if (is_first_acq && kk_off == 0 && hh == 0) {
                MATH(( _llk_math_eltwise_ternary_sfpu_params_(
                    face_mul3, 0, 1, 2, 3, static_cast<int>(VectorMode::RC)) ));
            } else {
                MATH(( _llk_math_eltwise_ternary_sfpu_params_(
                    face_fma3, 0, 1, 2, 3, static_cast<int>(VectorMode::RC)) ));
            }
        }
    }
    tile_regs_commit();

    cb_reserve_back(cb_acc, 1);
    tile_regs_wait();
    pack_reconfig_data_format(cb_acc);
    pack_tile(3, cb_acc);
    tile_regs_release();
    cb_push_back(cb_acc, 1);
}

void kernel_main() {
    const uint32_t n_batches = get_arg_val<uint32_t>(0);

    constexpr auto cb_px        = tt::CBIndex::c_0;
    constexpr auto cb_py        = tt::CBIndex::c_1;
    constexpr auto cb_fx        = tt::CBIndex::c_2;
    constexpr auto cb_fy        = tt::CBIndex::c_3;
    constexpr auto cb_neg_ratio = tt::CBIndex::c_4;
    constexpr auto cb_acc       = tt::CBIndex::c_5;
    constexpr auto cb_gx_out    = tt::CBIndex::c_16;
    constexpr auto cb_gy_out    = tt::CBIndex::c_17;

    constexpr uint32_t TILES_PER_GROUP = V21C_KK_PER_ACQ * V21S_MAX_H;
    constexpr uint32_t N_GROUPS        = V21S_MAX_K / V21C_KK_PER_ACQ;

    init_sfpu(cb_px, cb_acc);

    for (uint32_t batch = 0; batch < n_batches; ++batch) {
        DeviceZoneScopedN("V21C-BATCH");
        cb_wait_front(cb_px, V21S_MAX_K);
        cb_wait_front(cb_py, V21S_MAX_H);
        cb_wait_front(cb_neg_ratio, 1);

        // ════════════ GX phase ════════════
        for (uint32_t grp = 0; grp < N_GROUPS; ++grp) {
            cb_wait_front(cb_fx, TILES_PER_GROUP);
            if (grp > 0) cb_wait_front(cb_acc, 1);

            run_kk_group<cb_px, cb_py, cb_fx>(
                grp * V21C_KK_PER_ACQ, grp == 0, cb_acc, cb_neg_ratio);

            cb_pop_front(cb_fx, TILES_PER_GROUP);
            if (grp > 0) cb_pop_front(cb_acc, 1);
        }

        // GX negscale: gx_out = acc * neg_ratio
        cb_wait_front(cb_acc, 1);
        tile_regs_acquire();
        copy_tile_init(cb_acc);       copy_tile(cb_acc,       0, 0);
        copy_tile_init(cb_neg_ratio); copy_tile(cb_neg_ratio, 0, 1);
        MATH(( _llk_math_eltwise_ternary_sfpu_params_(
            face_mul, 0, 1, 0, 0, static_cast<int>(VectorMode::RC)) ));
        tile_regs_commit();
        cb_reserve_back(cb_gx_out, 1);
        tile_regs_wait();
        pack_reconfig_data_format(cb_gx_out);
        pack_tile(0, cb_gx_out);
        tile_regs_release();
        cb_push_back(cb_gx_out, 1);
        cb_pop_front(cb_acc, 1);

        // ════════════ GY phase ════════════
        for (uint32_t grp = 0; grp < N_GROUPS; ++grp) {
            cb_wait_front(cb_fy, TILES_PER_GROUP);
            if (grp > 0) cb_wait_front(cb_acc, 1);

            run_kk_group<cb_px, cb_py, cb_fy>(
                grp * V21C_KK_PER_ACQ, grp == 0, cb_acc, cb_neg_ratio);

            cb_pop_front(cb_fy, TILES_PER_GROUP);
            if (grp > 0) cb_pop_front(cb_acc, 1);
        }

        // GY negscale
        cb_wait_front(cb_acc, 1);
        tile_regs_acquire();
        copy_tile_init(cb_acc);       copy_tile(cb_acc,       0, 0);
        copy_tile_init(cb_neg_ratio); copy_tile(cb_neg_ratio, 0, 1);
        MATH(( _llk_math_eltwise_ternary_sfpu_params_(
            face_mul, 0, 1, 0, 0, static_cast<int>(VectorMode::RC)) ));
        tile_regs_commit();
        cb_reserve_back(cb_gy_out, 1);
        tile_regs_wait();
        pack_reconfig_data_format(cb_gy_out);
        pack_tile(0, cb_gy_out);
        tile_regs_release();
        cb_push_back(cb_gy_out, 1);
        cb_pop_front(cb_acc, 1);

        cb_pop_front(cb_px, V21S_MAX_K);
        cb_pop_front(cb_py, V21S_MAX_H);
        cb_pop_front(cb_neg_ratio, 1);
    }
}
