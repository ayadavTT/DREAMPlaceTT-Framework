// SPDX-License-Identifier: Apache-2.0
//
// FCCS chip-field de-interleave — removes the host field upload (h2d). The chip-DCT
// field (V19 latest_field_x/y_mb) is ROW_MAJOR but DRAM-INTERLEAVED (page = 1 row =
// N floats, M pages — the V21 zero-copy layout). FCCS's quantize+producer want a
// CONTIGUOUS single-page float buffer. This pass: each core copies its row range from
// the interleaved chip field → contiguous fxb/fyb (DRAM→L1→DRAM), for both planes.
// Then the existing SFPU quantize (flat 1024-tiles) + producer run unchanged, with NO
// host round-trip (chip-DCT → chip backward, host-free).
//
// Args: 0 my_core 1 fx_chip 2 fy_chip 3 fxb_dst 4 fyb_dst 5 NBX(rows) 6 NBY(cols)
//       7 rpc(rows/core) 8 chip_pg(=NBY*4, interleaved row page) 9 dst_pg(contig buffer bytes)

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif

constexpr uint32_t CB_SCRATCH = tt::CBIndex::c_0;

void kernel_main() {
    const uint32_t my_core = get_arg_val<uint32_t>(0);
    const uint32_t fx_chip = get_arg_val<uint32_t>(1);
    const uint32_t fy_chip = get_arg_val<uint32_t>(2);
    const uint32_t fxb_dst = get_arg_val<uint32_t>(3);
    const uint32_t fyb_dst = get_arg_val<uint32_t>(4);
    const uint32_t NBX     = get_arg_val<uint32_t>(5);
    const uint32_t NBY     = get_arg_val<uint32_t>(6);
    const uint32_t rpc     = get_arg_val<uint32_t>(7);
    const uint32_t chip_pg = get_arg_val<uint32_t>(8);
    const uint32_t dst_pg  = get_arg_val<uint32_t>(9);

    const uint32_t row_bytes = NBY * 4u;
    // chip field: interleaved, page = 1 row (chip_pg, 64-aligned)
    const InterleavedAddrGen<true> fxc = {.bank_base_address=fx_chip, .page_size=chip_pg};
    const InterleavedAddrGen<true> fyc = {.bank_base_address=fy_chip, .page_size=chip_pg};
    // dst: single contiguous page (whole buffer)
    const InterleavedAddrGen<true> fxd = {.bank_base_address=fxb_dst, .page_size=dst_pg};
    const InterleavedAddrGen<true> fyd = {.bank_base_address=fyb_dst, .page_size=dst_pg};

    uint32_t r0 = my_core * rpc; if (r0 >= NBX) return;
    uint32_t r1 = r0 + rpc; if (r1 > NBX) r1 = NBX;

    uint32_t lx = get_write_ptr(CB_SCRATCH);          // NBY*4 scratch (fx)
    uint32_t ly = lx + ((row_bytes + 63u) & ~63u);    // NBY*4 scratch (fy), 64-aligned

    for (uint32_t r = r0; r < r1; ++r) {
        noc_async_read(fxc.get_noc_addr(r), lx, row_bytes);
        noc_async_read(fyc.get_noc_addr(r), ly, row_bytes);
        noc_async_read_barrier();
        noc_async_write(lx, fxd.get_noc_addr(0) + (uint64_t)r * row_bytes, row_bytes);
        noc_async_write(ly, fyd.get_noc_addr(0) + (uint64_t)r * row_bytes, row_bytes);
        noc_async_write_barrier();
    }
}
