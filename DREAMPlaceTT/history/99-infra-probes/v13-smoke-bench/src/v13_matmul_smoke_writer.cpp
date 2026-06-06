// SPDX-License-Identifier: Apache-2.0
//
// V13 Phase 0a — FPU matmul smoke (BRISC writer kernel).
//
// Pops one tile from CB_OUT and writes it to a DRAM output buffer.
//
// Runtime args:
//   0: dst_dram_addr   — base of C buffer (1 tile = 2048 B bf16)

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif

constexpr uint32_t CB_OUT = 16;
constexpr uint32_t TILE_BYTES_BF16 = 32 * 32 * 2;

void kernel_main() {
    const uint32_t dst = get_arg_val<uint32_t>(0);

    const InterleavedAddrGen<true> gen = {
        .bank_base_address = dst,
        .page_size         = TILE_BYTES_BF16,
    };

    cb_wait_front(CB_OUT, 1);
    uint32_t l1_out = get_read_ptr(CB_OUT);
    noc_async_write(l1_out, gen.get_noc_addr(0), TILE_BYTES_BF16);
    noc_async_write_barrier();
    cb_pop_front(CB_OUT, 1);
}
