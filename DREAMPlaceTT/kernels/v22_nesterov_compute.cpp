// SPDX-License-Identifier: Apache-2.0
//
// V22 K1 — Nesterov optimizer update, TRISC SFPU compute kernel.
//
// Per element (one of 2*num_nodes fp32, processed as 32x32 tiles):
//   u_kp1 = v_k  - alpha * g_k
//   v_kp1 = u_kp1 + coef * (u_kp1 - u_k)
//
// `alpha` and `coef` are per-iter scalars. They are broadcast by the host into
// full tiles (all 1024 lanes hold the same value) and streamed in c_3 / c_4,
// then kept resident (waited once, never popped) — matches how the integration
// will feed per-iter scalars without JIT-recompiling the kernel.
//
// The whole update fits in ONE acquire using the 4-tile fp32 DST budget:
//   DST[0]=v_k, DST[1]=g_k, DST[2]=alpha  --MATH face_ukp1-->  DST[3]=u_kp1
//   (overwrite) DST[1]=u_k, DST[2]=coef   --MATH face_vkp1-->  DST[0]=v_kp1
// then pack DST[3]->cb_u_out (u_kp1, becomes next u_prev) and DST[0]->cb_v_out
// (v_kp1, next pos). Mixing copy_tile and MATH within one acquire mirrors
// v21_ef_compute_v5.cpp.
//
// CB contract (must match v22_nesterov_reader/writer + host):
//   c_0  cb_v      fp32 1 tile/iter   v_k
//   c_1  cb_g      fp32 1 tile/iter   g_k
//   c_2  cb_u      fp32 1 tile/iter   u_k (previous u)
//   c_3  cb_alpha  fp32 1 tile total  alpha broadcast (resident)
//   c_4  cb_coef   fp32 1 tile total  coef  broadcast (resident)
//   c_16 cb_u_out  fp32 1 tile/iter   u_kp1
//   c_17 cb_v_out  fp32 1 tile/iter   v_kp1
//
// DST budget: 4 fp32 slots (0..3). Each face fn uses <=4 vFloat reads + 1 write
// (within the ~5 LREG budget; exceeding it spills into dst_reg and corrupts
// faces 2/3 — see tt-sfpu/SFPU_GUIDE.md).

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

// u_kp1 = v - alpha*g
static void face_ukp1(uint32_t d_v, uint32_t d_g, uint32_t d_alpha, uint32_t d_out) {
    for (uint32_t i = 0; i < 8; ++i) {
        vFloat v = dst_reg[d_v     * N + i];
        vFloat g = dst_reg[d_g     * N + i];
        vFloat a = dst_reg[d_alpha * N + i];
        dst_reg[d_out * N + i] = v - a * g;
    }
}

// v_kp1 = u_kp1 + coef*(u_kp1 - u_k)
static void face_vkp1(uint32_t d_ukp1, uint32_t d_uk, uint32_t d_coef, uint32_t d_out) {
    for (uint32_t i = 0; i < 8; ++i) {
        vFloat u1 = dst_reg[d_ukp1 * N + i];
        vFloat uk = dst_reg[d_uk   * N + i];
        vFloat c  = dst_reg[d_coef * N + i];
        dst_reg[d_out * N + i] = u1 + c * (u1 - uk);
    }
}
#endif  // TRISC_MATH

void kernel_main() {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_v     = tt::CBIndex::c_0;
    constexpr auto cb_g     = tt::CBIndex::c_1;
    constexpr auto cb_u     = tt::CBIndex::c_2;
    constexpr auto cb_alpha = tt::CBIndex::c_3;
    constexpr auto cb_coef  = tt::CBIndex::c_4;
    constexpr auto cb_u_out = tt::CBIndex::c_16;
    constexpr auto cb_v_out = tt::CBIndex::c_17;

    init_sfpu(cb_v, cb_u_out);

    // alpha / coef are global scalars for this launch — wait once, keep resident.
    cb_wait_front(cb_alpha, 1);
    cb_wait_front(cb_coef, 1);

    for (uint32_t ti = 0; ti < n_tiles; ++ti) {
        cb_wait_front(cb_v, 1);
        cb_wait_front(cb_g, 1);
        cb_wait_front(cb_u, 1);

        tile_regs_acquire();
        copy_tile_init(cb_v);     copy_tile(cb_v,     0, 0);  // DST[0] = v_k
        copy_tile_init(cb_g);     copy_tile(cb_g,     0, 1);  // DST[1] = g_k
        copy_tile_init(cb_alpha); copy_tile(cb_alpha, 0, 2);  // DST[2] = alpha
        MATH(( _llk_math_eltwise_ternary_sfpu_params_(
            face_ukp1, 0, 1, 2, 3, static_cast<int>(VectorMode::RC)) ));  // DST[3] = u_kp1

        copy_tile_init(cb_u);     copy_tile(cb_u,    0, 1);   // DST[1] = u_k (overwrite g)
        copy_tile_init(cb_coef);  copy_tile(cb_coef, 0, 2);   // DST[2] = coef (overwrite alpha)
        MATH(( _llk_math_eltwise_ternary_sfpu_params_(
            face_vkp1, 3, 1, 2, 0, static_cast<int>(VectorMode::RC)) ));  // DST[0] = v_kp1
        tile_regs_commit();

        cb_reserve_back(cb_u_out, 1);
        cb_reserve_back(cb_v_out, 1);
        tile_regs_wait();
        pack_reconfig_data_format(cb_u_out);
        pack_tile(3, cb_u_out);   // u_kp1
        pack_reconfig_data_format(cb_v_out);
        pack_tile(0, cb_v_out);   // v_kp1
        tile_regs_release();
        cb_push_back(cb_u_out, 1);
        cb_push_back(cb_v_out, 1);

        cb_pop_front(cb_v, 1);
        cb_pop_front(cb_g, 1);
        cb_pop_front(cb_u, 1);
    }
}
