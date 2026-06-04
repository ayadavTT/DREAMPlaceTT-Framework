// SPDX-License-Identifier: Apache-2.0
//
// V26 EF backward — SFPU fp32 weighted-corner sum. BRISC gathers the 4 corner
// fx/fy values per cell (direct L1 loads, no math) into tiles; this SFPU kernel
// does gx = Σ_corner w·fx, gy = Σ_corner w·fy in hardware fp32. No int convert,
// no soft-float.
//
// CBs (1 tile = 1024 cells/batch):
//   c_0..c_3  fx_nw,fx_ne,fx_sw,fx_se
//   c_4..c_7  fy_nw,fy_ne,fy_sw,fy_se
//   c_8..c_11 w_nw,w_ne,w_sw,w_se
//   c_16 gx_out   c_17 gy_out

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
static void face_mul(uint32_t d_w, uint32_t d_f, uint32_t, uint32_t d_out) {
    for (uint32_t i = 0; i < 8; ++i) dst_reg[d_out*N+i] = dst_reg[d_w*N+i] * dst_reg[d_f*N+i];
}
static void face_fma(uint32_t d_w, uint32_t d_f, uint32_t, uint32_t d_out) {
    for (uint32_t i = 0; i < 8; ++i) dst_reg[d_out*N+i] = dst_reg[d_out*N+i] + dst_reg[d_w*N+i]*dst_reg[d_f*N+i];
}
#endif

void kernel_main() {
    const uint32_t n_batches = get_arg_val<uint32_t>(0);
    constexpr auto FXNW=tt::CBIndex::c_0, FXNE=tt::CBIndex::c_1, FXSW=tt::CBIndex::c_2, FXSE=tt::CBIndex::c_3;
    constexpr auto FYNW=tt::CBIndex::c_4, FYNE=tt::CBIndex::c_5, FYSW=tt::CBIndex::c_6, FYSE=tt::CBIndex::c_7;
    constexpr auto WNW=tt::CBIndex::c_8, WNE=tt::CBIndex::c_9, WSW=tt::CBIndex::c_10, WSE=tt::CBIndex::c_11;
    constexpr auto GX=tt::CBIndex::c_16, GY=tt::CBIndex::c_17;

    init_sfpu(FXNW, GX);
    for (uint32_t b = 0; b < n_batches; ++b) {
        cb_wait_front(FXNW,1); cb_wait_front(FXNE,1); cb_wait_front(FXSW,1); cb_wait_front(FXSE,1);
        cb_wait_front(FYNW,1); cb_wait_front(FYNE,1); cb_wait_front(FYSW,1); cb_wait_front(FYSE,1);
        cb_wait_front(WNW,1); cb_wait_front(WNE,1); cb_wait_front(WSW,1); cb_wait_front(WSE,1);

        // gx = Σ w·fx   (DST: w=0, f=1, acc=2)
        tile_regs_acquire();
        copy_tile_init(WNW);  copy_tile(WNW,0,0);  copy_tile_init(FXNW); copy_tile(FXNW,0,1);
        MATH((_llk_math_eltwise_ternary_sfpu_params_(face_mul,0,1,0,2,static_cast<int>(VectorMode::RC))));
        copy_tile_init(WNE);  copy_tile(WNE,0,0);  copy_tile_init(FXNE); copy_tile(FXNE,0,1);
        MATH((_llk_math_eltwise_ternary_sfpu_params_(face_fma,0,1,0,2,static_cast<int>(VectorMode::RC))));
        copy_tile_init(WSW);  copy_tile(WSW,0,0);  copy_tile_init(FXSW); copy_tile(FXSW,0,1);
        MATH((_llk_math_eltwise_ternary_sfpu_params_(face_fma,0,1,0,2,static_cast<int>(VectorMode::RC))));
        copy_tile_init(WSE);  copy_tile(WSE,0,0);  copy_tile_init(FXSE); copy_tile(FXSE,0,1);
        MATH((_llk_math_eltwise_ternary_sfpu_params_(face_fma,0,1,0,2,static_cast<int>(VectorMode::RC))));
        tile_regs_commit();
        cb_reserve_back(GX,1); tile_regs_wait(); pack_reconfig_data_format(GX); pack_tile(2,GX);
        tile_regs_release(); cb_push_back(GX,1);

        // gy = Σ w·fy
        tile_regs_acquire();
        copy_tile_init(WNW);  copy_tile(WNW,0,0);  copy_tile_init(FYNW); copy_tile(FYNW,0,1);
        MATH((_llk_math_eltwise_ternary_sfpu_params_(face_mul,0,1,0,2,static_cast<int>(VectorMode::RC))));
        copy_tile_init(WNE);  copy_tile(WNE,0,0);  copy_tile_init(FYNE); copy_tile(FYNE,0,1);
        MATH((_llk_math_eltwise_ternary_sfpu_params_(face_fma,0,1,0,2,static_cast<int>(VectorMode::RC))));
        copy_tile_init(WSW);  copy_tile(WSW,0,0);  copy_tile_init(FYSW); copy_tile(FYSW,0,1);
        MATH((_llk_math_eltwise_ternary_sfpu_params_(face_fma,0,1,0,2,static_cast<int>(VectorMode::RC))));
        copy_tile_init(WSE);  copy_tile(WSE,0,0);  copy_tile_init(FYSE); copy_tile(FYSE,0,1);
        MATH((_llk_math_eltwise_ternary_sfpu_params_(face_fma,0,1,0,2,static_cast<int>(VectorMode::RC))));
        tile_regs_commit();
        cb_reserve_back(GY,1); tile_regs_wait(); pack_reconfig_data_format(GY); pack_tile(2,GY);
        tile_regs_release(); cb_push_back(GY,1);

        cb_pop_front(FXNW,1); cb_pop_front(FXNE,1); cb_pop_front(FXSW,1); cb_pop_front(FXSE,1);
        cb_pop_front(FYNW,1); cb_pop_front(FYNE,1); cb_pop_front(FYSW,1); cb_pop_front(FYSE,1);
        cb_pop_front(WNW,1); cb_pop_front(WNE,1); cb_pop_front(WSW,1); cb_pop_front(WSE,1);
    }
}
