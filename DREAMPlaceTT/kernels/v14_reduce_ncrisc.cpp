// SPDX-License-Identifier: Apache-2.0
//
// V14 Architecture-A — NCRISC reader for FPU reduce.
//
// For each owned density tile T, reads nc_all partial bf16 tiles from
// partial_buf_v14 in DRAM (tile-major layout: page = T * nc_all + w)
// and pushes them alternately into CB_A and CB_B for TRISC FPU add_tiles.
//
// Runtime args:
//   0: partial_dram     DRAM base of partial_buf_v14
//   1: partial_pgsz     page size = TILE_BF16_BYTES = 2048
//   2: nc_all           number of writer cores (MUST be even)
//   3: n_owned_tiles
//   4..: owned_tile_ids[0 .. n_owned_tiles)

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif

constexpr uint32_t TILE_BF16_BYTES = 32u * 32u * 2u;  // 2048
constexpr uint32_t CB_A            = 0u;               // tt::CBIndex::c_0
constexpr uint32_t CB_B            = 1u;               // tt::CBIndex::c_1

void kernel_main() {
    const uint32_t partial_dram  = get_arg_val<uint32_t>(0);
    const uint32_t partial_pgsz  = get_arg_val<uint32_t>(1);
    const uint32_t nc_all        = get_arg_val<uint32_t>(2);
    const uint32_t n_owned_tiles = get_arg_val<uint32_t>(3);

    if (n_owned_tiles == 0u) return;

    const InterleavedAddrGen<true> pgen = {
        .bank_base_address = partial_dram,
        .page_size         = partial_pgsz,
    };

    const uint32_t n_pairs = nc_all / 2u;

    for (uint32_t t = 0; t < n_owned_tiles; ++t) {
        const uint32_t T = get_arg_val<uint32_t>(4u + t);

        for (uint32_t p = 0; p < n_pairs; ++p) {
            uint32_t w = p * 2u;

            cb_reserve_back(CB_A, 1);
            uint64_t src_a = pgen.get_noc_addr(T * nc_all + w);
            noc_async_read(src_a, get_write_ptr(CB_A), TILE_BF16_BYTES);

            cb_reserve_back(CB_B, 1);
            uint64_t src_b = pgen.get_noc_addr(T * nc_all + w + 1u);
            noc_async_read(src_b, get_write_ptr(CB_B), TILE_BF16_BYTES);

            noc_async_read_barrier();

            cb_push_back(CB_A, 1);
            cb_push_back(CB_B, 1);
        }
    }
}
