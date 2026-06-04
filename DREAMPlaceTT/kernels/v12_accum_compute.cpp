// SPDX-License-Identifier: Apache-2.0
//
// V12 TRISC Compute Kernel — FPU BF16 matmul accumulation, Phase 2.
//
// Protocol (sentinel-based, deadlock-free):
//   BRISC pushes (CB_OX, CB_OY) pairs for real contributions, then one
//   CB_OX sentinel tile (element[0] = V12_OX_DONE_SENTINEL = 0xFFFF,
//   rest zero) to signal end-of-tile contributions. No matching CB_OY
//   is pushed for the sentinel.
//
// Per owned tile loop:
//   tile_regs_acquire();   // zeros DST, acquires exclusive lock on DST
//   loop:
//     cb_wait_front(CB_OX, 1)
//     read_tile_value(CB_OX, 0, 0) → broadcasts element[0] to all 3 TRISCs
//     if element[0] == V12_OX_DONE_SENTINEL: pop OX; break
//     cb_wait_front(CB_OY, 1)
//     matmul_tiles(CB_OX, CB_OY, 0, 0, 0)   → DST[0] += OX @ OY (BF16 → FP32)
//     cb_pop_front(CB_OX, 1); cb_pop_front(CB_OY, 1)
//   tile_regs_commit(); tile_regs_wait();  // synchronize MATH → PACK
//   cb_reserve_back(CB_ACCUM, 1); pack_tile(0, CB_ACCUM); cb_push_back(CB_ACCUM, 1)
//   tile_regs_release()
//
// NCRISC reads CB_ACCUM, applies FP32 inv_bin_area scale, writes density_buf.
//
// CB layout (must match v12_brisc_combined.cpp + v12_ncrisc_combined.cpp):
//   c_0  CB_OX     BF16  double-buffered (2 tiles)
//   c_1  CB_OY     BF16  double-buffered (2 tiles)
//   c_3  CB_ACCUM  Float32  single (1 tile)
//
// Runtime args:
//   0: n_tiles_owned   number of density tiles this core owns

#if __has_include("api/compute/common.h")
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/compute_kernel_api.h"
#else
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api.h"
#endif

constexpr uint32_t CB_OX    = tt::CBIndex::c_0;
constexpr uint32_t CB_OY    = tt::CBIndex::c_1;
constexpr uint32_t CB_ACCUM = tt::CBIndex::c_3;

// Sentinel: BF16 all-ones in element[0] (= 0xFFFF) marks end-of-tile.
// This is a BF16 NaN, never produced by a real overlap_x product.
constexpr uint16_t V12_OX_DONE_SENTINEL = 0xFFFFu;

void kernel_main() {
    // V12 direct-accum mode: BRISC accumulates FP32 directly and pushes to CB_ACCUM.
    // TRISC is not needed in this path; return immediately on all cores.
    // (n_tiles_owned is always 0 in this mode, but check for safety.)
    const uint32_t n_tiles_owned = get_arg_val<uint32_t>(0);
    if (n_tiles_owned == 0u) return;

    // Initialize FPU matmul engine: BF16 inputs → FP32 DST accumulation.
    mm_init(CB_OX, CB_OY, CB_ACCUM);

    for (uint32_t tile_local = 0u; tile_local < n_tiles_owned; ++tile_local) {
        // tile_regs_acquire() acquires DST and zeros it (per TT-Metal spec).
        tile_regs_acquire();

        while (true) {
            // Wait for next OX tile (or sentinel) from BRISC.
            cb_wait_front(CB_OX, 1u);

            // read_tile_value broadcasts element[0] to all 3 TRISC threads via
            // inter-thread mailboxes, giving all of them the same value.
            // element_offset=0: reads first uint32 of tile = BF16 elements [0..1].
            // In little-endian, lower 16 bits = element[0].
            uint32_t ox0 = read_tile_value(CB_OX, 0u, 0u);
            if ((ox0 & 0xFFFFu) == (uint32_t)V12_OX_DONE_SENTINEL) {
                cb_pop_front(CB_OX, 1u);
                break;
            }

            // Real contribution: wait for OY then accumulate via FPU matmul.
            cb_wait_front(CB_OY, 1u);
            // DST[0] += OX[0] @ OY[0]  (BF16 inputs, FP32 accumulate in DST)
            matmul_tiles(CB_OX, CB_OY, 0u, 0u, 0u);
            cb_pop_front(CB_OX, 1u);
            cb_pop_front(CB_OY, 1u);
        }

        // Commit accumulated FP32 result: MATH signals PACK, PACK packs to CB_ACCUM.
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(CB_ACCUM, 1u);
        pack_tile(0u, CB_ACCUM);
        cb_push_back(CB_ACCUM, 1u);
        tile_regs_release();
    }
}
