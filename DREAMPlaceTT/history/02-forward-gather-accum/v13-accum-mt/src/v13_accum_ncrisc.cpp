// SPDX-License-Identifier: Apache-2.0
//
// V13_fpu Phase 1.1 — NCRISC writer: drain CB_DENSE (bf16 32x32 tile) to DRAM.
//
// Runtime args:
//   0: dst_dram_addr   DRAM destination for the produced bf16 density tile.

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif

constexpr uint32_t CB_DENSE = tt::CBIndex::c_16;
constexpr uint32_t TILE_BF16_BYTES = 32 * 32 * 2;  // 2048

void kernel_main() {
    const uint32_t dst = get_arg_val<uint32_t>(0);

    const InterleavedAddrGen<true> gen = {
        .bank_base_address = dst,
        .page_size         = TILE_BF16_BYTES,
    };

    cb_wait_front(CB_DENSE, 1);
    uint32_t l1_addr = get_read_ptr(CB_DENSE);
    noc_async_write(l1_addr, gen.get_noc_addr(0), TILE_BF16_BYTES);
    noc_async_write_barrier();
    cb_pop_front(CB_DENSE, 1);
}
