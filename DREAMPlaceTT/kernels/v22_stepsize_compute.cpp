// SPDX-License-Identifier: Apache-2.0
//
// V22 K5 — Barzilai-Borwein step-size reduction, TRISC SFPU compute kernel.
//
// Production formula (NesterovAcceleratedGradientOptimizer.py:212-217):
//   s_k = v_k - v_k_1 ;  y_k = g_k - g_k_1   (subtraction done on host/K3; this
//                                             kernel takes s_k, y_k as inputs)
//   ss = sum(s_k . s_k) ;  sy = sum(s_k . y_k) ;  yy = sum(y_k . y_k)
//   bb_short = sy / yy ;  lip = sqrt(ss)/sqrt(yy)
//   step = bb_short if bb_short > 0 else min(lip, alpha_k)
//
// The three sums are global reductions over 2*num_nodes fp32. This kernel does
// the heavy per-core reduction: it lane-wise accumulates ss/sy/yy over the
// core's tile shard (one fp32 accumulator held in DST across the tile loop, the
// proven V21/V22 pattern — no matrix-engine reduce / packer_l1_acc), then packs
// 3 partial tiles (one per product). The host sums the nc_all*3 partials (a
// trivial 330-value cross-core reduce) into the 3 scalars + step_size.
//
// 3 passes over the shard (pass 0=ss, 1=sy, 2=yy) so a single DST accumulator
// suffices (4-slot fp32 dest: 0=s, 1=y, 2=product, 3=accumulator).
//
// CB contract:
//   c_0  cb_s   fp32  s_k tiles (streamed 3x by the reader)
//   c_1  cb_y   fp32  y_k tiles (streamed 3x)
//   c_16 cb_out fp32  3 partial tiles/core (lane-wise ss, sy, yy)

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

#ifdef TRISC_MATH
static constexpr uint32_t N = 32;
// out = a * b
static void face_mul(uint32_t d_a, uint32_t d_b, uint32_t /*c*/, uint32_t d_out) {
    for (uint32_t i = 0; i < 8; ++i) {
        vFloat a = dst_reg[d_a * N + i];
        vFloat b = dst_reg[d_b * N + i];
        dst_reg[d_out * N + i] = a * b;
    }
}
// out = src   (seed accumulator on first tile)
static void face_copy(uint32_t d_src, uint32_t /*b*/, uint32_t /*c*/, uint32_t d_out) {
    for (uint32_t i = 0; i < 8; ++i) {
        vFloat v = dst_reg[d_src * N + i];
        dst_reg[d_out * N + i] = v;
    }
}
// acc += x   (d_out += d_x)
static void face_addinto(uint32_t d_x, uint32_t /*b*/, uint32_t /*c*/, uint32_t d_out) {
    for (uint32_t i = 0; i < 8; ++i) {
        vFloat acc = dst_reg[d_out * N + i];
        vFloat x   = dst_reg[d_x   * N + i];
        dst_reg[d_out * N + i] = acc + x;
    }
}
#endif

void kernel_main() {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);
    constexpr auto cb_s   = tt::CBIndex::c_0;
    constexpr auto cb_y   = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;

    init_sfpu(cb_s, cb_out);

    // pass 0: ss = s*s ;  pass 1: sy = s*y ;  pass 2: yy = y*y
    for (uint32_t pass = 0; pass < 3u; ++pass) {
        const uint32_t ia = (pass == 2u) ? 1u : 0u;  // s,s / s,y / y,y
        const uint32_t ib = (pass == 0u) ? 0u : 1u;

        tile_regs_acquire();
        for (uint32_t ti = 0; ti < n_tiles; ++ti) {
            cb_wait_front(cb_s, 1);
            cb_wait_front(cb_y, 1);
            copy_tile_init(cb_s); copy_tile(cb_s, 0, 0);  // DST[0] = s
            copy_tile_init(cb_y); copy_tile(cb_y, 0, 1);  // DST[1] = y
            MATH(( _llk_math_eltwise_ternary_sfpu_params_(
                face_mul, ia, ib, 0, 2, static_cast<int>(VectorMode::RC)) ));  // DST[2] = product
            if (ti == 0) {
                MATH(( _llk_math_eltwise_ternary_sfpu_params_(
                    face_copy, 2, 0, 0, 3, static_cast<int>(VectorMode::RC)) ));   // DST[3] = product
            } else {
                MATH(( _llk_math_eltwise_ternary_sfpu_params_(
                    face_addinto, 2, 0, 0, 3, static_cast<int>(VectorMode::RC)) ));// DST[3] += product
            }
            cb_pop_front(cb_s, 1);
            cb_pop_front(cb_y, 1);
        }
        tile_regs_commit();

        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_reconfig_data_format(cb_out);
        pack_tile(3, cb_out);   // lane-wise partial sum for this product
        tile_regs_release();
        cb_push_back(cb_out, 1);
    }
}
