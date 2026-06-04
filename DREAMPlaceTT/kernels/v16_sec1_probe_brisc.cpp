// SPDX-License-Identifier: Apache-2.0
//
// V16 Step 1 — BRISC reader/writer paired with v16_sec1_probe_compute.cpp.
//
// Reads pre-staged OX and OY tiles (32x32 bf16, face layout) from DRAM
// into CB_OX/CB_OY. Then waits for CB_DENSE and writes the matmul result
// to DRAM.
//
// Runtime args:
//   0: ox_dram_addr     bank base of OX input buffer (2048 B = 1 page)
//   1: oy_dram_addr     bank base of OY input buffer
//   2: dout_dram_addr   bank base of D output buffer
//   3: pgsz             page size = TILE_BF16_BYTES = 2048

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif

constexpr uint32_t CB_OX    = tt::CBIndex::c_0;
constexpr uint32_t CB_OY    = tt::CBIndex::c_1;
constexpr uint32_t CB_DENSE = tt::CBIndex::c_16;

constexpr uint32_t TILE_BF16_BYTES = 32u * 32u * 2u;  // 2048

void kernel_main() {
    const uint32_t ox_dram   = get_arg_val<uint32_t>(0);
    const uint32_t oy_dram   = get_arg_val<uint32_t>(1);
    const uint32_t dout_dram = get_arg_val<uint32_t>(2);
    const uint32_t pgsz      = get_arg_val<uint32_t>(3);

    const InterleavedAddrGen<true> ox_gen   = {.bank_base_address = ox_dram,   .page_size = pgsz};
    const InterleavedAddrGen<true> oy_gen   = {.bank_base_address = oy_dram,   .page_size = pgsz};
    const InterleavedAddrGen<true> dout_gen = {.bank_base_address = dout_dram, .page_size = pgsz};

    // Stage OX → CB_OX, OY → CB_OY.
    cb_reserve_back(CB_OX, 1);
    cb_reserve_back(CB_OY, 1);
    uint32_t ox_l1 = (uint32_t)get_write_ptr(CB_OX);
    uint32_t oy_l1 = (uint32_t)get_write_ptr(CB_OY);
    noc_async_read(ox_gen.get_noc_addr(0u), ox_l1, TILE_BF16_BYTES);
    noc_async_read(oy_gen.get_noc_addr(0u), oy_l1, TILE_BF16_BYTES);
    noc_async_read_barrier();
    cb_push_back(CB_OX, 1);
    cb_push_back(CB_OY, 1);

    // Wait for compute output → DRAM.
    cb_wait_front(CB_DENSE, 1);
    uint32_t dout_l1 = (uint32_t)get_read_ptr(CB_DENSE);
    noc_async_write(dout_l1, dout_gen.get_noc_addr(0u), TILE_BF16_BYTES);
    noc_async_write_barrier();
    cb_pop_front(CB_DENSE, 1);
}
