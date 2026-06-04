// SPDX-License-Identifier: Apache-2.0
//
// V21a electric_force — TRISC SFPU compute kernel v5 (FACE-MERGE).
//
// Design (combines GX+GY into one acquire-loop using lane interleaving):
//   Each tile in cb_px, cb_py, cb_neg_ratio has px/py/neg_ratio for cell c
//   DUPLICATED at lane c (0..511) AND lane c+512 (512..1023).
//
//   Each tile in cb_fxy has:
//     - lanes 0..511   = fx[c, kk, hh]   (written by BRISC)
//     - lanes 512..1023 = fy[c, kk, hh]  (written by NCRISC via shared L1)
//
//   The accumulator DST[3] therefore has:
//     - lanes 0..511   accumulating gx
//     - lanes 512..1023 accumulating gy
//
//   One face_fma3 MATH op processes ALL 1024 lanes, so per (kk, hh) iteration
//   we accumulate BOTH gx (lower) and gy (upper) in a single FMA. Halves the
//   per-batch MATH count from 128 → 64 vs v1.
//
// Per batch: 8 acquires + 1 negscale = 9 (vs v1's 20). cb_acc still cycled.
//
// CB contract (must match BRISC v5 + NCRISC v5 + host):
//   c_0  cb_px        fp32 8 tiles/batch — px[c, kk] duplicated to both halves
//   c_1  cb_py        fp32 8 tiles/batch — py[c, hh] duplicated
//   c_2  cb_fxy       fp32 64 tiles/batch — lower=fx, upper=fy (dual producer)
//   c_4  cb_neg_ratio fp32 1 tile/batch — duplicated
//   c_5  cb_acc       fp32 cycled accumulator
//   c_16 cb_gxy_out   fp32 1 tile/batch — lower=gx, upper=gy (TRISC writes)
//
// DST budget: 4 fp32 slots — px (DST[0]), py (DST[1]), fxy (DST[2]), acc (DST[3]).

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

#ifdef TRISC_MATH
static constexpr uint32_t N = 32;
static void face_mul3(uint32_t d_a, uint32_t d_b, uint32_t d_c, uint32_t d_out) {
    for (uint32_t i = 0; i < 8; ++i) {
        vFloat a = dst_reg[d_a * N + i];
        vFloat b = dst_reg[d_b * N + i];
        vFloat c = dst_reg[d_c * N + i];
        dst_reg[d_out * N + i] = a * b * c;
    }
}
static void face_fma3(uint32_t d_a, uint32_t d_b, uint32_t d_c, uint32_t d_out) {
    for (uint32_t i = 0; i < 8; ++i) {
        vFloat acc = dst_reg[d_out * N + i];
        vFloat a   = dst_reg[d_a   * N + i];
        vFloat b   = dst_reg[d_b   * N + i];
        vFloat c   = dst_reg[d_c   * N + i];
        dst_reg[d_out * N + i] = acc + a * b * c;
    }
}
static void face_mul(uint32_t d_a, uint32_t d_b, uint32_t /*c*/, uint32_t d_out) {
    for (uint32_t i = 0; i < 8; ++i) {
        vFloat a = dst_reg[d_a * N + i];
        vFloat b = dst_reg[d_b * N + i];
        dst_reg[d_out * N + i] = a * b;
    }
}
#endif

void kernel_main() {
    const uint32_t n_batches = get_arg_val<uint32_t>(0);

    constexpr auto cb_px        = tt::CBIndex::c_0;
    constexpr auto cb_py        = tt::CBIndex::c_1;
    constexpr auto cb_fxy       = tt::CBIndex::c_2;
    constexpr auto cb_neg_ratio = tt::CBIndex::c_4;
    constexpr auto cb_acc       = tt::CBIndex::c_5;
    constexpr auto cb_gxy_out   = tt::CBIndex::c_16;

    init_sfpu(cb_px, cb_acc);

    for (uint32_t batch = 0; batch < n_batches; ++batch) {
        DeviceZoneScopedN("V21C-BATCH");

        cb_wait_front(cb_px, 8);
        cb_wait_front(cb_py, 8);
        cb_wait_front(cb_neg_ratio, 1);

        // ═══════════ Combined GX+GY phase (face-merged) ═══════════
        // Per kk: load px[kk], iterate 8 hh × (load py[hh], load fxy[kk,hh], fma).
        // The single FMA accumulates BOTH gx (lower lanes) and gy (upper lanes).
        for (uint32_t kk = 0; kk < 8; ++kk) {
            // Fine-grained stall diagnostic: WAITFXY = TRISC blocked waiting for
            // the field tiles from BRISC(fx)+NCRISC(fy). If this dominates, TRISC
            // is STARVED → the field producers are the true bottleneck (not TRISC
            // compute). MATH = the actual copy_tile + SFPU fma work.
            { DeviceZoneScopedN("V21C-WAITFXY");
              cb_wait_front(cb_fxy, 8);
              if (kk > 0) cb_wait_front(cb_acc, 1);
            }

            { DeviceZoneScopedN("V21C-MATH");
            tile_regs_acquire();
            copy_tile_init(cb_px); copy_tile(cb_px, kk, 0);
            if (kk > 0) {
                copy_tile_init(cb_acc); copy_tile(cb_acc, 0, 3);
            }
            for (uint32_t hh = 0; hh < 8; ++hh) {
                copy_tile_init(cb_py);  copy_tile(cb_py,  hh, 1);
                copy_tile_init(cb_fxy); copy_tile(cb_fxy, hh, 2);
                if (kk == 0 && hh == 0) {
                    MATH(( _llk_math_eltwise_ternary_sfpu_params_(
                        face_mul3, 0, 1, 2, 3, static_cast<int>(VectorMode::RC)) ));
                } else {
                    MATH(( _llk_math_eltwise_ternary_sfpu_params_(
                        face_fma3, 0, 1, 2, 3, static_cast<int>(VectorMode::RC)) ));
                }
            }
            tile_regs_commit();

            cb_reserve_back(cb_acc, 1);
            tile_regs_wait();
            pack_reconfig_data_format(cb_acc);
            pack_tile(3, cb_acc);
            tile_regs_release();
            cb_push_back(cb_acc, 1);

            cb_pop_front(cb_fxy, 8);
            if (kk > 0) cb_pop_front(cb_acc, 1);
            }
        }

        // ─── Combined negscale: gxy_out = acc * neg_ratio (both halves) ───
        cb_wait_front(cb_acc, 1);
        tile_regs_acquire();
        copy_tile_init(cb_acc);       copy_tile(cb_acc,       0, 0);
        copy_tile_init(cb_neg_ratio); copy_tile(cb_neg_ratio, 0, 1);
        MATH(( _llk_math_eltwise_ternary_sfpu_params_(
            face_mul, 0, 1, 0, 0, static_cast<int>(VectorMode::RC)) ));
        tile_regs_commit();
        cb_reserve_back(cb_gxy_out, 1);
        tile_regs_wait();
        pack_reconfig_data_format(cb_gxy_out);
        pack_tile(0, cb_gxy_out);
        tile_regs_release();
        cb_push_back(cb_gxy_out, 1);
        cb_pop_front(cb_acc, 1);

        cb_pop_front(cb_px, 8);
        cb_pop_front(cb_py, 8);
        cb_pop_front(cb_neg_ratio, 1);
    }
}
