// SPDX-License-Identifier: Apache-2.0
//
// V13_fpu Phase 1.1 — TRISC compute kernel: per-tile FPU matmul accumulate.
//
// For one owned density tile, loop n_batches times:
//   cb_wait_front(CB_OX, 1)  // bf16 32x32 from BRISC
//   cb_wait_front(CB_OY, 1)  // bf16 32x32 from BRISC
//   matmul_tiles(CB_OX, CB_OY, 0, 0, 0)  // DST[0] += OX @ OY (fp32 accumulator)
//   cb_pop_front(CB_OX, 1); cb_pop_front(CB_OY, 1);
//
// After last batch: pack fp32 DST → bf16 CB_DENSE, push to NCRISC writer.
//
// Compute config (set on host): fp32_dest_acc_en = true, MathFidelity::LoFi.
//
// CB layout (must match v13_accum_brisc + v13_accum_ncrisc):
//   c_0   CB_OX     bf16  double-buffered (2 tiles)
//   c_1   CB_OY     bf16  double-buffered (2 tiles)
//   c_16  CB_DENSE  bf16  1 tile
//
// Runtime args:
//   0: n_batches   number of (OX, OY) tile pairs to accumulate before packing

#if __has_include("api/compute/common.h")
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#else
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/tile_move_copy.h"
#endif

constexpr uint32_t CB_OX    = tt::CBIndex::c_0;
constexpr uint32_t CB_OY    = tt::CBIndex::c_1;
constexpr uint32_t CB_DENSE = tt::CBIndex::c_16;

void kernel_main() {
    const uint32_t n_batches = get_arg_val<uint32_t>(0);

    mm_init(CB_OX, CB_OY, CB_DENSE);

    tile_regs_acquire();  // zeros DST[0]
    for (uint32_t i = 0; i < n_batches; ++i) {
        cb_wait_front(CB_OX, 1);
        cb_wait_front(CB_OY, 1);
        matmul_tiles(CB_OX, CB_OY, 0, 0, 0);
        cb_pop_front(CB_OX, 1);
        cb_pop_front(CB_OY, 1);
    }
    tile_regs_commit();

    tile_regs_wait();
    cb_reserve_back(CB_DENSE, 1);
    pack_tile(0, CB_DENSE);
    cb_push_back(CB_DENSE, 1);
    tile_regs_release();
}
