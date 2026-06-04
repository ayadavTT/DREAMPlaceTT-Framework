// SPDX-License-Identifier: Apache-2.0
//
// V13 Phase 0a — FPU matmul smoke (NCRISC reader kernel).
//
// Reads one tile from each of two DRAM buffers (A, B) into circular buffers
// CB_IN0 and CB_IN1. Single tile, single core.
//
// Runtime args:
//   0: src_a_dram_addr   — base of A buffer (1 tile = 2048 B bf16)
//   1: src_b_dram_addr   — base of B buffer (1 tile = 2048 B bf16)

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif

constexpr uint32_t CB_IN0 = 0;
constexpr uint32_t CB_IN1 = 1;
constexpr uint32_t TILE_BYTES_BF16 = 32 * 32 * 2;  // 2048

void kernel_main() {
    const uint32_t src_a = get_arg_val<uint32_t>(0);
    const uint32_t src_b = get_arg_val<uint32_t>(1);

    const InterleavedAddrGen<true> gen_a = {
        .bank_base_address = src_a,
        .page_size         = TILE_BYTES_BF16,
    };
    const InterleavedAddrGen<true> gen_b = {
        .bank_base_address = src_b,
        .page_size         = TILE_BYTES_BF16,
    };

    cb_reserve_back(CB_IN0, 1);
    {
        uint32_t l1_a = get_write_ptr(CB_IN0);
        noc_async_read(gen_a.get_noc_addr(0), l1_a, TILE_BYTES_BF16);
        noc_async_read_barrier();
    }
    cb_push_back(CB_IN0, 1);

    cb_reserve_back(CB_IN1, 1);
    {
        uint32_t l1_b = get_write_ptr(CB_IN1);
        noc_async_read(gen_b.get_noc_addr(0), l1_b, TILE_BYTES_BF16);
        noc_async_read_barrier();
    }
    cb_push_back(CB_IN1, 1);
}
