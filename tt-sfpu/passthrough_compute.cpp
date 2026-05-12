// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Passthrough / copy compute kernel — diagnostic only.
//
// Copies 4 input tiles (c_0..c_3) verbatim to 4 output tiles (c_16..c_19).
// No arithmetic — purely exercises the copy_tile / pack_tile path.
// Used to verify that SFPU element ordering and packing are correct.
//
// Control defines:
//   PASSTHROUGH_ONLY   (default) — copy inputs straight to outputs
//   PASSTHROUGH_FLOOR  — apply the floor trick to input[0] only (bxl = floor(pos_x))
//                        and forward the other 3 inputs unchanged.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_ternary_sfpu_params.h"
#endif

static constexpr uint32_t N = 32;

#ifdef PASSTHROUGH_FLOOR
// BIG constant for the round-then-adjust floor trick
static constexpr float BIG = 8388608.0f;  // 2^23
#endif

#ifdef TRISC_MATH
inline void passthrough_face(
    const uint32_t d_in0,
    const uint32_t d_in1,
    const uint32_t d_in2,
    const uint32_t d_out0)
{
    const uint32_t d_in3  = 3;          // sy  (hardcoded, matches bin_index_compute)
    const uint32_t d_out1 = d_out0 + 1; // bxh
    const uint32_t d_out2 = d_out0 + 2; // byl
    const uint32_t d_out3 = d_out0 + 3; // byh

    for (uint32_t i = 0; i < 8; ++i) {
        vFloat v0 = dst_reg[d_in0 * N + i];
        vFloat v1 = dst_reg[d_in1 * N + i];
        vFloat v2 = dst_reg[d_in2 * N + i];
        vFloat v3 = dst_reg[d_in3 * N + i];

#ifdef PASSTHROUGH_FLOOR
        // Apply the floor trick to v0 only; forward rest unchanged.
        vFloat r0 = (v0 + BIG) - BIG;
        v_if (r0 > v0) { r0 -= 1.0f; } v_endif;
        dst_reg[d_out0 * N + i] = r0;
#else
        dst_reg[d_out0 * N + i] = v0;
#endif
        dst_reg[d_out1 * N + i] = v1;
        dst_reg[d_out2 * N + i] = v2;
        dst_reg[d_out3 * N + i] = v3;
    }
}
#endif

inline void run_passthrough(uint32_t d_in0, uint32_t d_in1,
                             uint32_t d_in2, uint32_t d_out0) {
    MATH(_llk_math_eltwise_ternary_sfpu_params_<true>(
        passthrough_face, d_in0, d_in1, d_in2, d_out0,
        static_cast<int>(VectorMode::RC)));
}

void kernel_main() {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_px  = tt::CBIndex::c_0;
    constexpr auto cb_py  = tt::CBIndex::c_1;
    constexpr auto cb_sx  = tt::CBIndex::c_2;
    constexpr auto cb_sy  = tt::CBIndex::c_3;
    constexpr auto cb_bxl = tt::CBIndex::c_16;
    constexpr auto cb_bxh = tt::CBIndex::c_17;
    constexpr auto cb_byl = tt::CBIndex::c_18;
    constexpr auto cb_byh = tt::CBIndex::c_19;

    init_sfpu(cb_px, cb_bxl);
    pack_reconfig_data_format(cb_bxl);

    for (uint32_t ti = 0; ti < n_tiles; ++ti) {
        cb_wait_front(cb_px, 1);
        cb_wait_front(cb_py, 1);
        cb_wait_front(cb_sx, 1);
        cb_wait_front(cb_sy, 1);

        tile_regs_acquire();

        copy_tile_init(cb_px);  copy_tile(cb_px, 0, 0);
        copy_tile_init(cb_py);  copy_tile(cb_py, 0, 1);
        copy_tile_init(cb_sx);  copy_tile(cb_sx, 0, 2);
        copy_tile_init(cb_sy);  copy_tile(cb_sy, 0, 3);

        run_passthrough(0, 1, 2, 4);

        tile_regs_commit();

        cb_reserve_back(cb_bxl, 1);
        cb_reserve_back(cb_bxh, 1);
        cb_reserve_back(cb_byl, 1);
        cb_reserve_back(cb_byh, 1);

        tile_regs_wait();

        pack_reconfig_data_format(cb_bxl);  pack_tile(4, cb_bxl);
        pack_reconfig_data_format(cb_bxh);  pack_tile(5, cb_bxh);
        pack_reconfig_data_format(cb_byl);  pack_tile(6, cb_byl);
        pack_reconfig_data_format(cb_byh);  pack_tile(7, cb_byh);

        tile_regs_release();

        cb_push_back(cb_bxl, 1);
        cb_push_back(cb_bxh, 1);
        cb_push_back(cb_byl, 1);
        cb_push_back(cb_byh, 1);

        cb_pop_front(cb_px, 1);
        cb_pop_front(cb_py, 1);
        cb_pop_front(cb_sx, 1);
        cb_pop_front(cb_sy, 1);
    }
}
