// SPDX-License-Identifier: Apache-2.0
//
// V14 Architecture-A — TRISC compute kernel for FPU tile reduction.
//
// For each owned density tile:
//   1. Acquire DST (zeros it).
//   2. Loop nc_all/2 times: wait for tiles in CB_A and CB_B,
//      add_tiles with acc_to_dest to accumulate in DST[0].
//   3. Pack the accumulated result to CB_OUT.
//
// Runtime args:
//   0: n_owned    number of owned density tiles this core reduces
//   1: nc_all     number of writer cores (MUST be even)

#include <cstdint>
#if __has_include("api/compute/common.h")
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/compute_kernel_api.h"
#else
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api.h"
#endif

void kernel_main() {
    const uint32_t n_owned = get_arg_val<uint32_t>(0);
    const uint32_t nc_all  = get_arg_val<uint32_t>(1);
    const uint32_t n_pairs = nc_all / 2u;

    constexpr auto CB_A   = tt::CBIndex::c_0;
    constexpr auto CB_B   = tt::CBIndex::c_1;
    constexpr auto CB_OUT = tt::CBIndex::c_16;

    binary_op_init_common(CB_A, CB_B, CB_OUT);
    add_tiles_init(CB_A, CB_B, true);

    for (uint32_t t = 0; t < n_owned; ++t) {
        tile_regs_acquire();
        for (uint32_t p = 0; p < n_pairs; ++p) {
            cb_wait_front(CB_A, 1);
            cb_wait_front(CB_B, 1);
            add_tiles(CB_A, CB_B, 0, 0, 0);
            cb_pop_front(CB_A, 1);
            cb_pop_front(CB_B, 1);
        }
        tile_regs_commit();

        tile_regs_wait();
        cb_reserve_back(CB_OUT, 1);
        pack_tile(0, CB_OUT);
        cb_push_back(CB_OUT, 1);
        tile_regs_release();
    }
}
