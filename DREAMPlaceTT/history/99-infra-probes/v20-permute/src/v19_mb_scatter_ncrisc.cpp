// SPDX-License-Identifier: Apache-2.0
//
// V19 microbench NCRISC scatter — emits atomic increments to per-core L1
// density slabs given synthetic (bx, by, area_fixed) triplets from DRAM.
//
// Replicates v19_scatter_dm.cpp's atomic-emit path but without v4_compute
// dataflow — cells come pre-baked from the host, allowing controlled stress
// tests of the atomic-dispatch pipeline.
//
// Tracy zones (fine-grained):
//   V19MB-INIT          zero density slab + DRAM read of noc_coords table
//   V19MB-CELLS-READ    DRAM read of cell triplets into L1
//   V19MB-EMIT-BATCH    batch of 1024 atomic emits (one zone per batch)
//   V19MB-BARRIER       noc_async_atomic_barrier — should be ~0 if no contention

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t CB_SCRATCH = tt::CBIndex::c_24;
constexpr uint32_t EMIT_BATCH_SIZE = 1024u;

void kernel_main() {
    const uint32_t my_core_idx       = get_arg_val<uint32_t>(0);
    const uint32_t n_cells           = get_arg_val<uint32_t>(1);
    const uint32_t cell_dram_addr    = get_arg_val<uint32_t>(2);
    const uint32_t cell_dram_pgsz    = get_arg_val<uint32_t>(3);
    const uint32_t n_pages           = get_arg_val<uint32_t>(4);
    const uint32_t first_page_id     = get_arg_val<uint32_t>(5);
    const uint32_t density_l1_off    = get_arg_val<uint32_t>(6);
    const uint32_t density_slab_bins = get_arg_val<uint32_t>(7);
    const uint32_t noc_coord_dram    = get_arg_val<uint32_t>(8);
    const uint32_t noc_coord_pgsz    = get_arg_val<uint32_t>(9);
    const uint32_t cells_l1_off      = get_arg_val<uint32_t>(10);
    const uint32_t nbx               = get_arg_val<uint32_t>(11);
    const uint32_t nby               = get_arg_val<uint32_t>(12);
    const uint32_t nc_all            = get_arg_val<uint32_t>(13);
    const uint32_t coords_off        = get_arg_val<uint32_t>(14);  // explicit (host controls layout)

    uint8_t* base = reinterpret_cast<uint8_t*>(get_write_ptr(CB_SCRATCH));

    uint32_t* density_l1 = reinterpret_cast<uint32_t*>(base + density_l1_off);
    uint32_t* noc_coords = reinterpret_cast<uint32_t*>(base + coords_off);
    uint32_t* cells_l1   = reinterpret_cast<uint32_t*>(base + cells_l1_off);

    // Phase 1: INIT — copy noc_coords from common runtime args into L1.
    // Avoids noc_async_read because that read corrupts at high L1 offsets
    // on 2048 grid (V19 2048 hang root cause).
    {
        DeviceZoneScopedN("V19MB-INIT");
        for (uint32_t i = 0; i < nc_all; ++i) {
            noc_coords[i] = get_common_arg_val<uint32_t>((int)i);
        }
    }
    (void)noc_coord_dram;
    (void)noc_coord_pgsz;

    // Phase 2: CELLS-READ — DRAM read of this core's cell triplets.
    {
        DeviceZoneScopedN("V19MB-CELLS-READ");
        const InterleavedAddrGen<true> pgen = {
            .bank_base_address = cell_dram_addr,
            .page_size         = cell_dram_pgsz,
        };
        for (uint32_t p = 0; p < n_pages; ++p) {
            uint32_t page_id = first_page_id + p;
            uint32_t dst_l1 = reinterpret_cast<uint32_t>(cells_l1) + p * cell_dram_pgsz;
            noc_async_read_page(page_id, pgen, dst_l1);
        }
        noc_async_read_barrier();
    }

    // Phase 3: EMIT — one atomic per cell. Per-batch zones so Tracy shows
    // throughput vs back-pressure over the run.
    uint32_t i = 0;
    while (i < n_cells) {
        uint32_t batch_end = i + EMIT_BATCH_SIZE;
        if (batch_end > n_cells) batch_end = n_cells;
        {
            DeviceZoneScopedN("V19MB-EMIT-BATCH");
            for (; i < batch_end; ++i) {
                uint32_t bx         = cells_l1[i * 3u + 0u];
                uint32_t by         = cells_l1[i * 3u + 1u];
                uint32_t area_fixed = cells_l1[i * 3u + 2u];
                if (bx >= nbx) continue;
                if (by >= nby) continue;
                if (area_fixed == 0u) continue;

                uint32_t bin_global = bx * nby + by;
                uint32_t owner_idx  = bin_global % nc_all;
                uint32_t local_idx  = bin_global / nc_all;
                uint32_t packed_xy  = noc_coords[owner_idx];
                uint32_t owner_x    = packed_xy & 0xFFFFu;
                uint32_t owner_y    = packed_xy >> 16;
                uint64_t target = get_noc_addr(owner_x, owner_y,
                                               (uint32_t)density_l1 + local_idx * 4u);
                noc_semaphore_inc(target, area_fixed);
            }
        }
    }

    // Phase 4: BARRIER — wait for all outgoing atomics to ACK.
    {
        DeviceZoneScopedN("V19MB-BARRIER");
        noc_async_atomic_barrier();
    }
}
