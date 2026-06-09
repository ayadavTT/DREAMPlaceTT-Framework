// SPDX-License-Identifier: Apache-2.0
//
// V22 K1 — Nesterov update writer (RISCV_1 / NCRISC dataflow).
//
// Drains this core's tile range of u_kp1 (c_16) and v_kp1 (c_17) from the
// compute kernel out to the two interleaved DRAM output buffers.

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif

void kernel_main() {
    const uint32_t n_tiles    = get_arg_val<uint32_t>(0);
    const uint32_t tile_start = get_arg_val<uint32_t>(1);
    const uint32_t addr_u_out = get_arg_val<uint32_t>(2);
    const uint32_t addr_v_out = get_arg_val<uint32_t>(3);

    constexpr auto cb_u_out = tt::CBIndex::c_16;
    constexpr auto cb_v_out = tt::CBIndex::c_17;

    constexpr uint32_t TILE_BYTES = 32u * 32u * 4u;  // 4096 B fp32 tile

    const InterleavedAddrGenFast<true> su = {.bank_base_address = addr_u_out, .page_size = TILE_BYTES, .data_format = DataFormat::Float32};
    const InterleavedAddrGenFast<true> sv = {.bank_base_address = addr_v_out, .page_size = TILE_BYTES, .data_format = DataFormat::Float32};

    for (uint32_t ti = 0; ti < n_tiles; ++ti) {
        const uint32_t tid = tile_start + ti;
        cb_wait_front(cb_u_out, 1);
        cb_wait_front(cb_v_out, 1);
        noc_async_write_tile(tid, su, get_read_ptr(cb_u_out));
        noc_async_write_tile(tid, sv, get_read_ptr(cb_v_out));
        noc_async_write_barrier();
        cb_pop_front(cb_u_out, 1);
        cb_pop_front(cb_v_out, 1);
    }
}
