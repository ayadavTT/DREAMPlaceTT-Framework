// SPDX-License-Identifier: Apache-2.0
//
// V14 Architecture-A — NCRISC scatter kernel.
//
// Loads tile_to_core[] from DRAM into the start of CB_SCRATCH (shared L1)
// and signals tables_ready_sem so BRISC can start Phase 1. Then idle —
// BRISC handles all 1024 cells per cell-tile (no NCRISC split in V14).
//
// Runtime args:
//   0: tile_map_dram
//   1: tile_map_pgsz
//   2: tile_map_bytes

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif

constexpr uint32_t CB_SCRATCH = tt::CBIndex::c_24;

void kernel_main() {
    const uint32_t tile_map_dram       = get_arg_val<uint32_t>(0);
    const uint32_t tile_map_pgsz       = get_arg_val<uint32_t>(1);
    const uint32_t tile_map_bytes      = get_arg_val<uint32_t>(2);
    const uint32_t tables_ready_sem_id = get_arg_val<uint32_t>(3);

    uint8_t* base = reinterpret_cast<uint8_t*>(get_write_ptr(CB_SCRATCH));
    uint16_t* tile_to_core = reinterpret_cast<uint16_t*>(base);

    // Load tile_to_core[] from DRAM.
    {
        const InterleavedAddrGen<true> mgen = {
            .bank_base_address = tile_map_dram,
            .page_size         = tile_map_pgsz,
        };
        noc_async_read(mgen.get_noc_addr(0),
                       reinterpret_cast<uint32_t>(tile_to_core),
                       tile_map_bytes);
        noc_async_read_barrier();
    }

    // Signal BRISC that tile_to_core[] is loaded.
    {
        volatile tt_l1_ptr uint32_t* tr_sem =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(tables_ready_sem_id));
        *tr_sem = 1u;
        asm volatile("" ::: "memory");
    }
}
