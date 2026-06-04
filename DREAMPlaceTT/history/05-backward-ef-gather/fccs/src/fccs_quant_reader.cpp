// SPDX-License-Identifier: Apache-2.0
//
// FCCS SFPU field-quantize — READER (NCRISC). Each core reads its contiguous range
// of 1024-float "tiles" from the float field DRAM (fx buffer then fy buffer) into
// CB c_0. For an elementwise op the 32×32 tile layout is irrelevant: a flat 1024
// contiguous floats read into the tile slot, SFPU-scaled per lane, then written back
// contiguously is an identity mapping (the face permutation cancels). So no tilize.
//
// Args: 0 my_core 1 fx_base 2 fy_base 3 ntiles_per_buf 4 tpc(tiles/core) 5 total_tiles
//       6 field_pg

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif

constexpr uint32_t CB_IN = tt::CBIndex::c_0;   // Float32 input tiles
constexpr uint32_t TILE_BYTES = 1024u * 4u;    // 1024 floats

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
        cb_reserve_back(CB_IN, 1);
        uint32_t l1 = get_write_ptr(CB_IN);
        uint64_t src = (g < ntiles_per_buf)
            ? (fxg.get_noc_addr(0) + (uint64_t)g * TILE_BYTES)
            : (fyg.get_noc_addr(0) + (uint64_t)(g - ntiles_per_buf) * TILE_BYTES);
        noc_async_read(src, l1, TILE_BYTES);
        noc_async_read_barrier();
        cb_push_back(CB_IN, 1);
    }
}
