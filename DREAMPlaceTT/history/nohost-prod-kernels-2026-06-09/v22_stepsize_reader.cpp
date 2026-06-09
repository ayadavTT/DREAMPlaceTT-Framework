// SPDX-License-Identifier: Apache-2.0
//
// V22 K5 reader (RISCV_0). Streams s_k (c_0) and y_k (c_1) for this core's tile
// shard, THREE times (one per product pass: ss, sy, yy) so the compute kernel's
// single DST accumulator can cover all three reductions. Interleaved DRAM, page
// = one 32x32 fp32 tile.

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif

void kernel_main() {
    const uint32_t n_tiles    = get_arg_val<uint32_t>(0);
    const uint32_t tile_start = get_arg_val<uint32_t>(1);
    const uint32_t addr_s     = get_arg_val<uint32_t>(2);
    const uint32_t addr_y     = get_arg_val<uint32_t>(3);

    constexpr uint32_t TILE_BYTES = 32u * 32u * 4u;
    constexpr auto cb_s = tt::CBIndex::c_0;
    constexpr auto cb_y = tt::CBIndex::c_1;

    const InterleavedAddrGenFast<true> s_s = {.bank_base_address = addr_s, .page_size = TILE_BYTES, .data_format = DataFormat::Float32};
    const InterleavedAddrGenFast<true> s_y = {.bank_base_address = addr_y, .page_size = TILE_BYTES, .data_format = DataFormat::Float32};

    for (uint32_t pass = 0; pass < 3u; ++pass) {
        for (uint32_t ti = 0; ti < n_tiles; ++ti) {
            const uint32_t tid = tile_start + ti;
            cb_reserve_back(cb_s, 1);
            noc_async_read_tile(tid, s_s, get_write_ptr(cb_s));
            cb_reserve_back(cb_y, 1);
            noc_async_read_tile(tid, s_y, get_write_ptr(cb_y));
            noc_async_read_barrier();
            cb_push_back(cb_s, 1);
            cb_push_back(cb_y, 1);
        }
    }
}
