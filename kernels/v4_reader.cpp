// SPDX-License-Identifier: Apache-2.0
//
// V4 BRISC Reader (RISCV_0) — streams 4 SoA tile buffers from DRAM into input CBs.
//
// Runtime args:
//   0: addr_px           DRAM base address of pos_x buffer
//   1: addr_py           DRAM base address of pos_y buffer
//   2: addr_sx           DRAM base address of sx buffer
//   3: addr_sy           DRAM base address of sy buffer
//   4: tile_page_size    bytes per tile page (4096 for Float32)
//   5: first_tile        first tile index for this core
//   6: n_tiles           number of tiles this core processes

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    const uint32_t addr_px    = get_arg_val<uint32_t>(0);
    const uint32_t addr_py    = get_arg_val<uint32_t>(1);
    const uint32_t addr_sx    = get_arg_val<uint32_t>(2);
    const uint32_t addr_sy    = get_arg_val<uint32_t>(3);
    const uint32_t pgsz       = get_arg_val<uint32_t>(4);
    const uint32_t first_tile = get_arg_val<uint32_t>(5);
    const uint32_t n_tiles    = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_px = tt::CBIndex::c_0;
    constexpr uint32_t cb_py = tt::CBIndex::c_1;
    constexpr uint32_t cb_sx = tt::CBIndex::c_2;
    constexpr uint32_t cb_sy = tt::CBIndex::c_3;

    const InterleavedAddrGen<true> gen_px = { .bank_base_address = addr_px, .page_size = pgsz };
    const InterleavedAddrGen<true> gen_py = { .bank_base_address = addr_py, .page_size = pgsz };
    const InterleavedAddrGen<true> gen_sx = { .bank_base_address = addr_sx, .page_size = pgsz };
    const InterleavedAddrGen<true> gen_sy = { .bank_base_address = addr_sy, .page_size = pgsz };

    for (uint32_t t = 0; t < n_tiles; ++t) {
        uint32_t page_id = first_tile + t;

        cb_reserve_back(cb_px, 1);
        cb_reserve_back(cb_py, 1);
        cb_reserve_back(cb_sx, 1);
        cb_reserve_back(cb_sy, 1);

        uint32_t l1_px = get_write_ptr(cb_px);
        uint32_t l1_py = get_write_ptr(cb_py);
        uint32_t l1_sx = get_write_ptr(cb_sx);
        uint32_t l1_sy = get_write_ptr(cb_sy);

        {
            DeviceZoneScopedN("V4R-DRAM-READ");
            noc_async_read_page(page_id, gen_px, l1_px);
            noc_async_read_page(page_id, gen_py, l1_py);
            noc_async_read_page(page_id, gen_sx, l1_sx);
            noc_async_read_page(page_id, gen_sy, l1_sy);
            noc_async_read_barrier();
        }

        cb_push_back(cb_px, 1);
        cb_push_back(cb_py, 1);
        cb_push_back(cb_sx, 1);
        cb_push_back(cb_sy, 1);
    }
}
