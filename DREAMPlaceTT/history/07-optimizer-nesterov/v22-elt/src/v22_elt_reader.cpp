// SPDX-License-Identifier: Apache-2.0
//
// V22 generic element-wise reader (RISCV_0). Shared by K2 (clamp), K3 (combine),
// K4 (precond). Streams n_streams input arrays (c_0/c_1/c_2) tile-by-tile and
// reads n_scalars broadcast scalar tiles (c_3/c_4) once (kept resident by the
// compute kernel). All buffers interleaved DRAM, page = one 32x32 fp32 tile.

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif

void kernel_main() {
    const uint32_t n_tiles    = get_arg_val<uint32_t>(0);
    const uint32_t tile_start = get_arg_val<uint32_t>(1);
    const uint32_t n_streams  = get_arg_val<uint32_t>(2);
    const uint32_t n_scalars  = get_arg_val<uint32_t>(3);
    const uint32_t addr_in0   = get_arg_val<uint32_t>(4);
    const uint32_t addr_in1   = get_arg_val<uint32_t>(5);
    const uint32_t addr_in2   = get_arg_val<uint32_t>(6);
    const uint32_t addr_s0    = get_arg_val<uint32_t>(7);
    const uint32_t addr_s1    = get_arg_val<uint32_t>(8);

    constexpr uint32_t TILE_BYTES = 32u * 32u * 4u;
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_in2 = tt::CBIndex::c_2;
    constexpr auto cb_s0  = tt::CBIndex::c_3;
    constexpr auto cb_s1  = tt::CBIndex::c_4;

    const InterleavedAddrGenFast<true> s_in0 = {.bank_base_address = addr_in0, .page_size = TILE_BYTES, .data_format = DataFormat::Float32};
    const InterleavedAddrGenFast<true> s_in1 = {.bank_base_address = addr_in1, .page_size = TILE_BYTES, .data_format = DataFormat::Float32};
    const InterleavedAddrGenFast<true> s_in2 = {.bank_base_address = addr_in2, .page_size = TILE_BYTES, .data_format = DataFormat::Float32};
    const InterleavedAddrGenFast<true> s_s0  = {.bank_base_address = addr_s0,  .page_size = TILE_BYTES, .data_format = DataFormat::Float32};
    const InterleavedAddrGenFast<true> s_s1  = {.bank_base_address = addr_s1,  .page_size = TILE_BYTES, .data_format = DataFormat::Float32};

    // Resident scalar tiles, read once.
    if (n_scalars >= 1) {
        cb_reserve_back(cb_s0, 1);
        noc_async_read_tile(0, s_s0, get_write_ptr(cb_s0));
        noc_async_read_barrier();
        cb_push_back(cb_s0, 1);
    }
    if (n_scalars >= 2) {
        cb_reserve_back(cb_s1, 1);
        noc_async_read_tile(0, s_s1, get_write_ptr(cb_s1));
        noc_async_read_barrier();
        cb_push_back(cb_s1, 1);
    }

    for (uint32_t ti = 0; ti < n_tiles; ++ti) {
        const uint32_t tid = tile_start + ti;
        cb_reserve_back(cb_in0, 1);
        noc_async_read_tile(tid, s_in0, get_write_ptr(cb_in0));
        if (n_streams >= 2) { cb_reserve_back(cb_in1, 1); noc_async_read_tile(tid, s_in1, get_write_ptr(cb_in1)); }
        if (n_streams >= 3) { cb_reserve_back(cb_in2, 1); noc_async_read_tile(tid, s_in2, get_write_ptr(cb_in2)); }
        noc_async_read_barrier();
        cb_push_back(cb_in0, 1);
        if (n_streams >= 2) cb_push_back(cb_in1, 1);
        if (n_streams >= 3) cb_push_back(cb_in2, 1);
    }
}
