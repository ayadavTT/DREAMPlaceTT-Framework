// SPDX-License-Identifier: Apache-2.0
//
// V13 Phase 0a — FPU matmul smoke (TRISC compute kernel).
//
// Reads one bf16 32x32 tile from CB_IN0 (matrix A) and one from CB_IN1 (B),
// computes A·B on the FPU (DST is fp32 via fp32_dest_acc_en in host config),
// and packs the result to bf16 in CB_OUT.
//
// CB layout (must match v13_matmul_smoke_host):
//   c_0  CB_IN0  bf16  1 tile
//   c_1  CB_IN1  bf16  1 tile
//   c_16 CB_OUT  bf16  1 tile

#if __has_include("api/compute/common.h")
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#else
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/tile_move_copy.h"
#endif

constexpr uint32_t CB_IN0 = tt::CBIndex::c_0;
constexpr uint32_t CB_IN1 = tt::CBIndex::c_1;
constexpr uint32_t CB_OUT = tt::CBIndex::c_16;

void kernel_main() {
    mm_init(CB_IN0, CB_IN1, CB_OUT);

    // Wait for input tiles A and B to be available.
    cb_wait_front(CB_IN0, 1);
    cb_wait_front(CB_IN1, 1);

    // Acquire DST (zeros it). Accumulate one matmul: DST[0] += A @ B.
    tile_regs_acquire();
    matmul_tiles(CB_IN0, CB_IN1, 0, 0, 0);
    tile_regs_commit();

    cb_pop_front(CB_IN0, 1);
    cb_pop_front(CB_IN1, 1);

    // Pack fp32 DST → bf16 output.
    tile_regs_wait();
    cb_reserve_back(CB_OUT, 1);
    pack_tile(0, CB_OUT);
    cb_push_back(CB_OUT, 1);
    tile_regs_release();
}
