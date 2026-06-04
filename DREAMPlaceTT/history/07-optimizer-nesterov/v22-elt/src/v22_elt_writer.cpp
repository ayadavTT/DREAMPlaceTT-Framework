// SPDX-License-Identifier: Apache-2.0
//
// V22 generic element-wise writer (RISCV_1). Drains the single output CB (c_16)
// to an interleaved DRAM buffer. Shared by K2/K3/K4.

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif

void kernel_main() {
    const uint32_t n_tiles    = get_arg_val<uint32_t>(0);
    const uint32_t tile_start = get_arg_val<uint32_t>(1);
    const uint32_t addr_out   = get_arg_val<uint32_t>(2);

    constexpr uint32_t TILE_BYTES = 32u * 32u * 4u;
    constexpr auto cb_out = tt::CBIndex::c_16;

    const InterleavedAddrGenFast<true> s_out = {.bank_base_address = addr_out, .page_size = TILE_BYTES, .data_format = DataFormat::Float32};

    for (uint32_t ti = 0; ti < n_tiles; ++ti) {
        const uint32_t tid = tile_start + ti;
        cb_wait_front(cb_out, 1);
        noc_async_write_tile(tid, s_out, get_read_ptr(cb_out));
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
