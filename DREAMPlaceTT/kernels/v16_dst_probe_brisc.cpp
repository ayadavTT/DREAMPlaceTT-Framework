// SPDX-License-Identifier: Apache-2.0
//
// V16 Phase 1.0 — BRISC dumper paired with v16_dst_probe_compute.cpp.
//
// Drains the single tile pushed by TRISC (in native L1 face layout) and
// writes the 2048 raw bytes to DRAM page 0 of an interleaved buffer the
// host pre-allocated. Host then reads back and decodes the dst_reg slot →
// (face, row_pair) mapping.
//
// Runtime args:
//   0: dst_dram_addr     bank base of output buffer (32x32 bf16 = 2048 B)
//   1: dst_pgsz          page size (= 2048 B)

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif

constexpr uint32_t CB_OUT = tt::CBIndex::c_16;
constexpr uint32_t TILE_BF16_BYTES = 32u * 32u * 2u;

void kernel_main() {
    const uint32_t dst_dram = get_arg_val<uint32_t>(0);
    const uint32_t dst_pgsz = get_arg_val<uint32_t>(1);

    cb_wait_front(CB_OUT, 1);
    uint32_t src_l1 = (uint32_t)get_read_ptr(CB_OUT);

    const InterleavedAddrGen<true> dgen = {
        .bank_base_address = dst_dram,
        .page_size         = dst_pgsz,
    };
    noc_async_write(src_l1, dgen.get_noc_addr(0u), TILE_BF16_BYTES);
    noc_async_write_barrier();

    cb_pop_front(CB_OUT, 1);
}
