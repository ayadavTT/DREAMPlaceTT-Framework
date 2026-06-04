// SPDX-License-Identifier: Apache-2.0
//
// V22 K1 — Nesterov update reader (RISCV_0 / BRISC dataflow).
//
// Streams this core's tile range of v_k / g_k / u_k from DRAM into c_0/c_1/c_2,
// and reads the two broadcast scalar tiles (alpha, coef) ONCE into c_3/c_4
// (the compute kernel keeps them resident). All buffers are interleaved DRAM
// with page_size = 4096 B (one 32x32 fp32 tile).

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif

void kernel_main() {
    const uint32_t n_tiles    = get_arg_val<uint32_t>(0);
    const uint32_t tile_start = get_arg_val<uint32_t>(1);
    const uint32_t addr_v     = get_arg_val<uint32_t>(2);
    const uint32_t addr_g     = get_arg_val<uint32_t>(3);
    const uint32_t addr_u     = get_arg_val<uint32_t>(4);
    const uint32_t addr_alpha = get_arg_val<uint32_t>(5);
    const uint32_t addr_coef  = get_arg_val<uint32_t>(6);

    constexpr auto cb_v     = tt::CBIndex::c_0;
    constexpr auto cb_g     = tt::CBIndex::c_1;
    constexpr auto cb_u     = tt::CBIndex::c_2;
    constexpr auto cb_alpha = tt::CBIndex::c_3;
    constexpr auto cb_coef  = tt::CBIndex::c_4;

    constexpr uint32_t TILE_BYTES = 32u * 32u * 4u;  // 4096 B fp32 tile

    const InterleavedAddrGenFast<true> sv     = {.bank_base_address = addr_v,     .page_size = TILE_BYTES, .data_format = DataFormat::Float32};
    const InterleavedAddrGenFast<true> sg     = {.bank_base_address = addr_g,     .page_size = TILE_BYTES, .data_format = DataFormat::Float32};
    const InterleavedAddrGenFast<true> su     = {.bank_base_address = addr_u,     .page_size = TILE_BYTES, .data_format = DataFormat::Float32};
    const InterleavedAddrGenFast<true> salpha = {.bank_base_address = addr_alpha, .page_size = TILE_BYTES, .data_format = DataFormat::Float32};
    const InterleavedAddrGenFast<true> scoef  = {.bank_base_address = addr_coef,  .page_size = TILE_BYTES, .data_format = DataFormat::Float32};

    // ── Broadcast scalar tiles: read once into resident CBs ──
    cb_reserve_back(cb_alpha, 1);
    cb_reserve_back(cb_coef, 1);
    noc_async_read_tile(0, salpha, get_write_ptr(cb_alpha));
    noc_async_read_tile(0, scoef,  get_write_ptr(cb_coef));
    noc_async_read_barrier();
    cb_push_back(cb_alpha, 1);
    cb_push_back(cb_coef, 1);

    // ── Stream v / g / u tile-by-tile ──
    for (uint32_t ti = 0; ti < n_tiles; ++ti) {
        const uint32_t tid = tile_start + ti;
        cb_reserve_back(cb_v, 1);
        cb_reserve_back(cb_g, 1);
        cb_reserve_back(cb_u, 1);
        noc_async_read_tile(tid, sv, get_write_ptr(cb_v));
        noc_async_read_tile(tid, sg, get_write_ptr(cb_g));
        noc_async_read_tile(tid, su, get_write_ptr(cb_u));
        noc_async_read_barrier();
        cb_push_back(cb_v, 1);
        cb_push_back(cb_g, 1);
        cb_push_back(cb_u, 1);
    }
}
