// SPDX-License-Identifier: Apache-2.0
//
// V28 EF backward — multi-bin SFPU sum. For each (k,h) with k<max_k, h<max_h:
//   w = px[k]·py[h];  gx += w·fx[k,h];  gy += w·fy[k,h]
// then gx·=ratio, gy·=ratio. Cells with span < (k,h) carry px[k]=0/py[h]=0 so
// absent bins are nullified — the loop runs the batch-global max span only.
// DST: 0=w/scratch, 1=scratch(f or ratio), 2=gx acc, 3=gy acc.  (4 fp32 regs)
//
// CBs: c_0 px[max_k]  c_1 py[max_h]  c_2 fx(stream)  c_3 fy(stream)
//      c_4 ratio  c_16 gx_out  c_17 gy_out.

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
static void face_mul(uint32_t a, uint32_t b, uint32_t, uint32_t o) {
    for (uint32_t i=0;i<8;++i) dst_reg[o*N+i]=dst_reg[a*N+i]*dst_reg[b*N+i];
}
static void face_fma(uint32_t a, uint32_t b, uint32_t, uint32_t o) {
    for (uint32_t i=0;i<8;++i) dst_reg[o*N+i]=dst_reg[o*N+i]+dst_reg[a*N+i]*dst_reg[b*N+i];
}
#endif

void kernel_main() {
    const uint32_t n_batches = get_arg_val<uint32_t>(0);
    const uint32_t max_k     = get_arg_val<uint32_t>(1);
    const uint32_t max_h     = get_arg_val<uint32_t>(2);
    constexpr auto PX=tt::CBIndex::c_0, PY=tt::CBIndex::c_1, FX=tt::CBIndex::c_2,
                   FY=tt::CBIndex::c_3, RATIO=tt::CBIndex::c_4, GX=tt::CBIndex::c_16, GY=tt::CBIndex::c_17;
    constexpr int RC = static_cast<int>(VectorMode::RC);

    init_sfpu(FX, GX);
    for (uint32_t b=0;b<n_batches;++b) {
        cb_wait_front(PX,max_k); cb_wait_front(PY,max_h); cb_wait_front(RATIO,1);
        tile_regs_acquire();
        bool first=true;
        for (uint32_t k=0;k<max_k;++k) {
            for (uint32_t h=0;h<max_h;++h) {
                cb_wait_front(FX,1); cb_wait_front(FY,1);
                // w = px[k]·py[h]  → DST0
                copy_tile_init(PX); copy_tile(PX,k,0);
                copy_tile_init(PY); copy_tile(PY,h,1);
                MATH((_llk_math_eltwise_ternary_sfpu_params_(face_mul,0,1,0,0,RC)));
                // gx, gy accumulate
                copy_tile_init(FX); copy_tile(FX,0,1);
                MATH((_llk_math_eltwise_ternary_sfpu_params_(first?face_mul:face_fma,0,1,0,2,RC)));
                copy_tile_init(FY); copy_tile(FY,0,1);
                MATH((_llk_math_eltwise_ternary_sfpu_params_(first?face_mul:face_fma,0,1,0,3,RC)));
                first=false;
                cb_pop_front(FX,1); cb_pop_front(FY,1);
            }
        }
        // gx·=ratio, gy·=ratio
        copy_tile_init(RATIO); copy_tile(RATIO,0,1);
        MATH((_llk_math_eltwise_ternary_sfpu_params_(face_mul,2,1,0,2,RC)));
        copy_tile_init(RATIO); copy_tile(RATIO,0,1);
        MATH((_llk_math_eltwise_ternary_sfpu_params_(face_mul,3,1,0,3,RC)));
        tile_regs_commit();
        cb_reserve_back(GX,1); cb_reserve_back(GY,1); tile_regs_wait();
        pack_reconfig_data_format(GX); pack_tile(2,GX);
        pack_reconfig_data_format(GY); pack_tile(3,GY);
        tile_regs_release(); cb_push_back(GX,1); cb_push_back(GY,1);
        cb_pop_front(PX,max_k); cb_pop_front(PY,max_h); cb_pop_front(RATIO,1);
    }
}
