// SPDX-License-Identifier: Apache-2.0
//
// V22 K4 — precondition, TRISC SFPU compute kernel.
//
// Per element (PlaceObj.py:76-103, density_weight.size==1 case):
//   precond = sum_pin_weights + alpha * density_weight * node_areas
//   precond = max(precond, 1.0)
//   grad   /= precond
//
// alpha * density_weight are folded into ONE per-iter scalar `adw` (broadcast
// into a full tile, resident in c_3). pin_weights and node_areas are constants
// (uploaded once). precond is length num_nodes but applies to BOTH halves of
// the 2*nn grad — the host duplicates pin_w / area into both halves so this
// kernel stays pure element-wise over 2*nn (precond recomputed redundantly per
// half; cheap).
//
// No SFPI division — reciprocal via inline Newton-Raphson. precond >= 1 always
// (max with 1.0), so there are no zero/inf/NaN edge cases.
//
// CB contract:
//   c_0 cb_grad  fp32 1 tile/iter
//   c_1 cb_pw    fp32 1 tile/iter   sum_pin_weights (duplicated both halves)
//   c_2 cb_area  fp32 1 tile/iter   node_areas      (duplicated both halves)
//   c_3 cb_adw   fp32 1 tile total  alpha*density_weight broadcast (resident)
//   c_16 cb_out  fp32 1 tile/iter   grad / precond

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

// Reciprocal of d via magic-seed + 3 Newton-Raphson iterations. Valid for
// d > 0 (here d = precond >= 1). y_{n+1} = y_n * (2 - d*y_n).
sfpi_inline vFloat recip_pos(vFloat d) {
    vInt i = sfpi::reinterpret<vInt>(d);
    i = vInt(0x7EF127EA) - i;                 // initial 1/x bit estimate
    vFloat y = sfpi::reinterpret<vFloat>(i);
    y = y * (vFloat(2.0f) - d * y);
    y = y * (vFloat(2.0f) - d * y);
    y = y * (vFloat(2.0f) - d * y);
    return y;
}

// precond = max(pw + adw*area, 1)
static void face_precond(uint32_t d_pw, uint32_t d_area, uint32_t d_adw, uint32_t d_out) {
    for (uint32_t i = 0; i < 8; ++i) {
        vFloat pw   = dst_reg[d_pw   * N + i];
        vFloat area = dst_reg[d_area * N + i];
        vFloat adw  = dst_reg[d_adw  * N + i];
        vFloat pc = pw + adw * area;
        v_if (pc < vFloat(1.0f)) { pc = vFloat(1.0f); } v_endif;
        dst_reg[d_out * N + i] = pc;
    }
}

// out = grad * recip(precond)
static void face_div(uint32_t d_grad, uint32_t d_pc, uint32_t /*unused*/, uint32_t d_out) {
    for (uint32_t i = 0; i < 8; ++i) {
        vFloat pc = dst_reg[d_pc   * N + i];
        vFloat r  = recip_pos(pc);
        vFloat g  = dst_reg[d_grad * N + i];
        dst_reg[d_out * N + i] = g * r;
    }
}
#endif

void kernel_main() {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);
    constexpr auto cb_grad = tt::CBIndex::c_0;
    constexpr auto cb_pw   = tt::CBIndex::c_1;
    constexpr auto cb_area = tt::CBIndex::c_2;
    constexpr auto cb_adw  = tt::CBIndex::c_3;
    constexpr auto cb_out  = tt::CBIndex::c_16;

    init_sfpu(cb_grad, cb_out);
    cb_wait_front(cb_adw, 1);   // resident scalar

    for (uint32_t ti = 0; ti < n_tiles; ++ti) {
        cb_wait_front(cb_grad, 1);
        cb_wait_front(cb_pw, 1);
        cb_wait_front(cb_area, 1);

        tile_regs_acquire();
        copy_tile_init(cb_pw);   copy_tile(cb_pw,   0, 0);  // DST[0]=pw
        copy_tile_init(cb_area); copy_tile(cb_area, 0, 1);  // DST[1]=area
        copy_tile_init(cb_adw);  copy_tile(cb_adw,  0, 2);  // DST[2]=adw
        MATH(( _llk_math_eltwise_ternary_sfpu_params_(
            face_precond, 0, 1, 2, 3, static_cast<int>(VectorMode::RC)) ));  // DST[3]=precond

        copy_tile_init(cb_grad); copy_tile(cb_grad, 0, 0);  // DST[0]=grad (overwrite pw)
        MATH(( _llk_math_eltwise_ternary_sfpu_params_(
            face_div, 0, 3, 0, 0, static_cast<int>(VectorMode::RC)) ));      // DST[0]=grad/precond
        tile_regs_commit();

        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_reconfig_data_format(cb_out);
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);

        cb_pop_front(cb_grad, 1);
        cb_pop_front(cb_pw, 1);
        cb_pop_front(cb_area, 1);
    }
}
