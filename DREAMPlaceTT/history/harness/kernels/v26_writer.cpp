// SPDX-License-Identifier: Apache-2.0
//
// V26 EF backward — writer (NCRISC). Drains the SFPU's gx/gy tiles (c_16/c_17)
// to two DRAM output buffers (one page per core).

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif

void kernel_main() {
    const uint32_t my_core   = get_arg_val<uint32_t>(0);
    const uint32_t n_batches = get_arg_val<uint32_t>(1);
    const uint32_t gx_base   = get_arg_val<uint32_t>(2);
    const uint32_t gy_base   = get_arg_val<uint32_t>(3);
    const uint32_t out_pg    = get_arg_val<uint32_t>(4);   // per-core page bytes (max_cells*4)

    constexpr uint32_t TILE_BYTES = 1024u*4u;
    constexpr auto GX=tt::CBIndex::c_16, GY=tt::CBIndex::c_17;
    const InterleavedAddrGen<true> gxg = {.bank_base_address=gx_base, .page_size=out_pg};
    const InterleavedAddrGen<true> gyg = {.bank_base_address=gy_base, .page_size=out_pg};
    const uint64_t gx_page = gxg.get_noc_addr(my_core);
    const uint64_t gy_page = gyg.get_noc_addr(my_core);

    for (uint32_t b = 0; b < n_batches; ++b) {
        cb_wait_front(GX,1); cb_wait_front(GY,1);
        noc_async_write(get_read_ptr(GX), gx_page + (uint64_t)b*TILE_BYTES, TILE_BYTES);
        noc_async_write(get_read_ptr(GY), gy_page + (uint64_t)b*TILE_BYTES, TILE_BYTES);
        noc_async_write_barrier();
        cb_pop_front(GX,1); cb_pop_front(GY,1);
    }
}
