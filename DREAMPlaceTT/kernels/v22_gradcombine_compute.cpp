// SPDX-License-Identifier: Apache-2.0
//
// V22 K3 — gradient combine, TRISC SFPU compute kernel.
//
// Per element:  total_grad = wl_grad + density_weight * density_grad
//
// `density_weight` is a per-iter scalar (size==1 case), broadcast by the host
// into a full tile and kept resident in c_3.
//
// CB contract:
//   c_0 cb_wl    fp32 1 tile/iter   wl_grad
//   c_1 cb_dg    fp32 1 tile/iter   density_grad
//   c_3 cb_dw    fp32 1 tile total  density_weight broadcast (resident)
//   c_16 cb_out  fp32 1 tile/iter   total_grad

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
static void face_combine(uint32_t d_wl, uint32_t d_dg, uint32_t d_dw, uint32_t d_out) {
    for (uint32_t i = 0; i < 8; ++i) {
        vFloat wl = dst_reg[d_wl * N + i];
        vFloat dg = dst_reg[d_dg * N + i];
        vFloat dw = dst_reg[d_dw * N + i];
        dst_reg[d_out * N + i] = wl + dw * dg;
    }
}
#endif

void kernel_main() {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);
    constexpr auto cb_wl  = tt::CBIndex::c_0;
    constexpr auto cb_dg  = tt::CBIndex::c_1;
    constexpr auto cb_dw  = tt::CBIndex::c_3;
    constexpr auto cb_out = tt::CBIndex::c_16;

    init_sfpu(cb_wl, cb_out);
    cb_wait_front(cb_dw, 1);   // resident scalar

    for (uint32_t ti = 0; ti < n_tiles; ++ti) {
        cb_wait_front(cb_wl, 1);
        cb_wait_front(cb_dg, 1);

        tile_regs_acquire();
        copy_tile_init(cb_wl); copy_tile(cb_wl, 0, 0);
        copy_tile_init(cb_dg); copy_tile(cb_dg, 0, 1);
        copy_tile_init(cb_dw); copy_tile(cb_dw, 0, 2);
        MATH(( _llk_math_eltwise_ternary_sfpu_params_(
            face_combine, 0, 1, 2, 0, static_cast<int>(VectorMode::RC)) ));
        tile_regs_commit();

        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_reconfig_data_format(cb_out);
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);

        cb_pop_front(cb_wl, 1);
        cb_pop_front(cb_dg, 1);
    }
}
