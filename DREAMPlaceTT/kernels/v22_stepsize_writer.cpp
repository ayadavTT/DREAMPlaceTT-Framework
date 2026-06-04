// SPDX-License-Identifier: Apache-2.0
//
// V22 K5 writer (RISCV_1). Drains the 3 partial tiles this core produced
// (lane-wise ss, sy, yy) to out[core*3 + {0,1,2}] in interleaved DRAM.

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif

void kernel_main() {
    const uint32_t out_tile_base = get_arg_val<uint32_t>(0);  // = core_idx * 3
    const uint32_t addr_out      = get_arg_val<uint32_t>(1);

    constexpr uint32_t TILE_BYTES = 32u * 32u * 4u;
    constexpr auto cb_out = tt::CBIndex::c_16;

    const InterleavedAddrGenFast<true> s_out = {.bank_base_address = addr_out, .page_size = TILE_BYTES, .data_format = DataFormat::Float32};

    for (uint32_t k = 0; k < 3u; ++k) {
        cb_wait_front(cb_out, 1);
        noc_async_write_tile(out_tile_base + k, s_out, get_read_ptr(cb_out));
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
