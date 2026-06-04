// SPDX-License-Identifier: Apache-2.0
//
// FCCS SFPU field-quantize — WRITER (BRISC). Writes the int32 tiles produced by
// the compute kernel (CB c_16) back to the field DRAM IN PLACE (same offsets the
// reader read), float→int32 conversion done on the SFPU. See fccs_quant_reader.cpp.
//
// Args: 0 my_core 1 fx_base 2 fy_base 3 ntiles_per_buf 4 tpc 5 total_tiles 6 field_pg

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif

constexpr uint32_t CB_OUT = tt::CBIndex::c_16;   // Int32 output tiles
constexpr uint32_t TILE_BYTES = 1024u * 4u;

void kernel_main() {
    const uint32_t my_core = get_arg_val<uint32_t>(0);
    const uint32_t fx_base = get_arg_val<uint32_t>(1);
    const uint32_t fy_base = get_arg_val<uint32_t>(2);
    const uint32_t ntiles_per_buf = get_arg_val<uint32_t>(3);
    const uint32_t tpc     = get_arg_val<uint32_t>(4);
    const uint32_t total   = get_arg_val<uint32_t>(5);
    const uint32_t field_pg= get_arg_val<uint32_t>(6);

    const InterleavedAddrGen<true> fxg = {.bank_base_address=fx_base, .page_size=field_pg};
    const InterleavedAddrGen<true> fyg = {.bank_base_address=fy_base, .page_size=field_pg};

    uint32_t lo = my_core * tpc; if (lo >= total) return;
    uint32_t hi = lo + tpc; if (hi > total) hi = total;
    for (uint32_t g = lo; g < hi; ++g) {
        cb_wait_front(CB_OUT, 1);
        uint32_t l1 = get_read_ptr(CB_OUT);
        uint64_t dst = (g < ntiles_per_buf)
            ? (fxg.get_noc_addr(0) + (uint64_t)g * TILE_BYTES)
            : (fyg.get_noc_addr(0) + (uint64_t)(g - ntiles_per_buf) * TILE_BYTES);
        noc_async_write(l1, dst, TILE_BYTES);
        noc_async_write_barrier();
        cb_pop_front(CB_OUT, 1);
    }
}
