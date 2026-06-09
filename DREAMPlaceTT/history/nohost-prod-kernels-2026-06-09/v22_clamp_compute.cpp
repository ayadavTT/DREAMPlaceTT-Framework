// SPDX-License-Identifier: Apache-2.0
//
// V22 K2 — move_boundary / clamp, TRISC SFPU compute kernel.
//
// Per element:  out = min(max(pos, lo), hi)
//
// DreamPlace move_boundary (ops/move_boundary/src/move_boundary.cpp):
//   x[i] = min(max(x[i], xl), xh - node_size_x[i])  for movable + filler
//   (fixed nodes left untouched)
// The per-half / per-cell bounds and the fixed-node skip are folded into the
// `lo` / `hi` full-length arrays on the host (fixed nodes get lo=-BIG, hi=+BIG
// so the clamp is a no-op), keeping this kernel pure element-wise.
//
// CB contract:
//   c_0 cb_pos  fp32 1 tile/iter
//   c_1 cb_lo   fp32 1 tile/iter
//   c_2 cb_hi   fp32 1 tile/iter
//   c_16 cb_out fp32 1 tile/iter

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
static void face_clamp(uint32_t d_pos, uint32_t d_lo, uint32_t d_hi, uint32_t d_out) {
    for (uint32_t i = 0; i < 8; ++i) {
        vFloat p  = dst_reg[d_pos * N + i];
        vFloat lo = dst_reg[d_lo  * N + i];
        vFloat hi = dst_reg[d_hi  * N + i];
        v_if (p < lo) { p = lo; } v_endif;
        v_if (p > hi) { p = hi; } v_endif;
        dst_reg[d_out * N + i] = p;
    }
}
#endif

void kernel_main() {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);
    constexpr auto cb_pos = tt::CBIndex::c_0;
    constexpr auto cb_lo  = tt::CBIndex::c_1;
    constexpr auto cb_hi  = tt::CBIndex::c_2;
    constexpr auto cb_out = tt::CBIndex::c_16;

    init_sfpu(cb_pos, cb_out);

    for (uint32_t ti = 0; ti < n_tiles; ++ti) {
        cb_wait_front(cb_pos, 1);
        cb_wait_front(cb_lo, 1);
        cb_wait_front(cb_hi, 1);

        tile_regs_acquire();
        copy_tile_init(cb_pos); copy_tile(cb_pos, 0, 0);
        copy_tile_init(cb_lo);  copy_tile(cb_lo,  0, 1);
        copy_tile_init(cb_hi);  copy_tile(cb_hi,  0, 2);
        MATH(( _llk_math_eltwise_ternary_sfpu_params_(
            face_clamp, 0, 1, 2, 0, static_cast<int>(VectorMode::RC)) ));
        tile_regs_commit();

        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_reconfig_data_format(cb_out);
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);

        cb_pop_front(cb_pos, 1);
        cb_pop_front(cb_lo, 1);
        cb_pop_front(cb_hi, 1);
    }
}
