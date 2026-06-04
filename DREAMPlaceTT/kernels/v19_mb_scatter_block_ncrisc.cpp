// SPDX-License-Identifier: Apache-2.0
//
// V19 microbench scatter — BLOCK partitioning variant.
//
// Ownership formula:
//   owner_idx = bin_global / density_slab_bins
//   local_idx = bin_global % density_slab_bins
//
// vs stride-1 (the original):
//   owner_idx = bin_global % nc_all
//   local_idx = bin_global / nc_all
//
// Why block: each core's L1 slab now holds a CONTIGUOUS chunk of bin_globals.
// The writeout kernel can dump it to a single contiguous row-major DRAM
// region per core — no de-stride needed.
//
// Trade-off: spatial cell clusters concentrate atomic-add load on fewer
// cores (~4× more concentrated for a 32×32 hot tile). This benchmark
// measures whether that's worth the gather savings.

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
    const uint32_t coords_off        = get_arg_val<uint32_t>(14);

    uint8_t* base = reinterpret_cast<uint8_t*>(get_write_ptr(CB_SCRATCH));

    uint32_t* density_l1 = reinterpret_cast<uint32_t*>(base + density_l1_off);
    uint32_t* noc_coords = reinterpret_cast<uint32_t*>(base + coords_off);
    uint32_t* cells_l1   = reinterpret_cast<uint32_t*>(base + cells_l1_off);

    {
        DeviceZoneScopedN("V19MB-INIT");
        for (uint32_t i = 0; i < nc_all; ++i) {
            noc_coords[i] = get_common_arg_val<uint32_t>((int)i);
        }
    }
    (void)noc_coord_dram;
    (void)noc_coord_pgsz;

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
                // BLOCK PARTITION: contiguous bins per core.
                uint32_t owner_idx  = bin_global / density_slab_bins;
                uint32_t local_idx  = bin_global % density_slab_bins;
                // Guard: with padded slab_bins, owner_idx can technically reach
                // nc_all (rare). Skip — those would be padding bins past
                // total_bins anyway, already filtered by nbx/nby checks.
                if (owner_idx >= nc_all) continue;

                uint32_t packed_xy = noc_coords[owner_idx];
                uint32_t owner_x   = packed_xy & 0xFFFFu;
                uint32_t owner_y   = packed_xy >> 16;
                uint64_t target = get_noc_addr(owner_x, owner_y,
                                               (uint32_t)density_l1 + local_idx * 4u);
                noc_semaphore_inc(target, area_fixed);
            }
        }
    }

    {
        DeviceZoneScopedN("V19MB-BARRIER");
        noc_async_atomic_barrier();
    }
}
