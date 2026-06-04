// SPDX-License-Identifier: Apache-2.0
//
// V14 Architecture-A — BRISC writer for FPU reduce.
//
// For each owned density tile:
//   1. Wait for TRISC to push the accumulated tile into CB_OUT.
//   2. Convert tile format (face layout) to row-major bf16 into CB_SCRATCH.
//   3. Write 32 row-strips from CB_SCRATCH to the density DRAM buffer.
//
// NOTE: NOC DMA reads from the CB-allocated L1 region, NOT from stack/BSS.
//       Using CB_SCRATCH (c_2) as the conversion buffer guarantees NOC access.
//
// Runtime args:
//   0: density_dram      DRAM base of density_buf
//   1: density_pgsz      page size for density row = N * sizeof(bf16)
//   2: N_tiles
//   3: nbx
//   4: nby
//   5: n_owned_tiles
//   6..: owned_tile_ids[0 .. n_owned_tiles)

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t TILE_BF16_BYTES = 32u * 32u * 2u;  // 2048
constexpr uint32_t CB_OUT          = 16u;              // tt::CBIndex::c_16
constexpr uint32_t CB_SCRATCH      = 2u;               // tt::CBIndex::c_2

static inline uint32_t face_idx(int row, int col) {
    uint32_t face = ((uint32_t)(row >> 4) * 2u) + (uint32_t)(col >> 4);
    return face * 256u + ((uint32_t)row & 15u) * 16u + ((uint32_t)col & 15u);
}

void kernel_main() {
    const uint32_t density_dram  = get_arg_val<uint32_t>(0);
    const uint32_t density_pgsz  = get_arg_val<uint32_t>(1);
    const uint32_t N_tiles       = get_arg_val<uint32_t>(2);
    const uint32_t nbx           = get_arg_val<uint32_t>(3);
    const uint32_t nby           = get_arg_val<uint32_t>(4);
    const uint32_t n_owned_tiles = get_arg_val<uint32_t>(5);

    if (n_owned_tiles == 0u) return;

    const InterleavedAddrGen<true> dgen = {
        .bank_base_address = density_dram,
        .page_size         = density_pgsz,
    };

    // Get the L1 address of CB_SCRATCH for row-major conversion.
    // This is in the CB-allocated region, so NOC DMA can read from it.
    cb_reserve_back(CB_SCRATCH, 1);
    uint16_t* dense_rm = reinterpret_cast<uint16_t*>(get_write_ptr(CB_SCRATCH));

    for (uint32_t t = 0; t < n_owned_tiles; ++t) {
        const uint32_t T = get_arg_val<uint32_t>(6u + t);

        cb_wait_front(CB_OUT, 1);

        const uint16_t* tile_data = reinterpret_cast<const uint16_t*>(get_read_ptr(CB_OUT));

        for (int row = 0; row < 32; ++row) {
            for (int col = 0; col < 32; ++col) {
                dense_rm[row * 32 + col] = tile_data[face_idx(row, col)];
            }
        }

        {
            DeviceZoneScopedN("V14R-DENSITY-WRITE");
            uint32_t tile_x = T / N_tiles;
            uint32_t tile_y = T % N_tiles;
            for (uint32_t bxw = 0; bxw < 32u; ++bxw) {
                uint32_t bx = tile_x * 32u + bxw;
                if (bx >= nbx) break;
                uint64_t row_base = dgen.get_noc_addr(bx);
                uint64_t dst = row_base + (uint64_t)tile_y * 32u * sizeof(uint16_t);
                uint32_t y_count = 32u;
                if (tile_y * 32u + y_count > nby)
                    y_count = nby - tile_y * 32u;
                uint32_t src_l1 = reinterpret_cast<uint32_t>(&dense_rm[bxw * 32]);
                noc_async_write(src_l1, dst, y_count * sizeof(uint16_t));
            }
            noc_async_write_barrier();
        }

        cb_pop_front(CB_OUT, 1);
    }
}
