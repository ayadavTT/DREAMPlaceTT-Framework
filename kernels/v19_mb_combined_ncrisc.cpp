// SPDX-License-Identifier: Apache-2.0
//
// V19 microbench DIAGNOSTIC kernel — combined scatter + writeout in ONE
// program. Used to isolate whether the 2048-grid sum_actual=0 bug is caused
// by L1-base shift between scatter and writeout programs (kernel binaries
// of different sizes might push the unreserved-L1 base by a different amount).
//
// CAVEAT: cross-core atomic races are not avoided here — incoming atomics
// from other cores can still arrive after this core's noc_async_atomic_barrier
// (which only waits on outgoing). For the microbench, all cores start at the
// same time and we accept some imperfection vs. the production split.
//
// Tracy zones: V19MB-INIT-COORDS, V19MB-CELLS-READ, V19MB-EMIT-BATCH,
//              V19MB-BARRIER, V19MB-WRITEOUT.

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t CB_SCRATCH = tt::CBIndex::c_24;
constexpr uint32_t EMIT_BATCH_SIZE = 1024u;

void kernel_main() {
    const uint32_t my_core_idx        = get_arg_val<uint32_t>(0);
    const uint32_t n_cells            = get_arg_val<uint32_t>(1);
    const uint32_t cell_dram_addr     = get_arg_val<uint32_t>(2);
    const uint32_t cell_dram_pgsz     = get_arg_val<uint32_t>(3);
    const uint32_t n_pages            = get_arg_val<uint32_t>(4);
    const uint32_t first_page_id      = get_arg_val<uint32_t>(5);
    const uint32_t density_l1_off     = get_arg_val<uint32_t>(6);
    const uint32_t density_slab_bins  = get_arg_val<uint32_t>(7);
    const uint32_t noc_coord_dram     = get_arg_val<uint32_t>(8);
    const uint32_t noc_coord_pgsz     = get_arg_val<uint32_t>(9);
    const uint32_t cells_l1_off       = get_arg_val<uint32_t>(10);
    const uint32_t nbx                = get_arg_val<uint32_t>(11);
    const uint32_t nby                = get_arg_val<uint32_t>(12);
    const uint32_t nc_all             = get_arg_val<uint32_t>(13);
    const uint32_t density_dram_base  = get_arg_val<uint32_t>(14);
    const uint32_t density_dram_pgsz  = get_arg_val<uint32_t>(15);
    const uint32_t diag_dram_base     = get_arg_val<uint32_t>(16);  // separate diag DRAM
    const uint32_t diag_dram_pgsz     = get_arg_val<uint32_t>(17);  // bytes per core diag page
    const uint32_t coords_off         = get_arg_val<uint32_t>(18);  // L1 byte offset of coords in c_24
                                                                    // (host now passes explicit value so we
                                                                    //  can use a layout where coords is at low
                                                                    //  L1 offset, density slab at high offset)

    uint8_t* base = reinterpret_cast<uint8_t*>(get_write_ptr(CB_SCRATCH));

    // coords_off now provided by host (rather than derived from density_l1_off
    // + density_slab_bins) so the layout can be: [coords][cells_l1][density_l1].
    uint32_t* density_l1 = reinterpret_cast<uint32_t*>(base + density_l1_off);
    uint32_t* noc_coords = reinterpret_cast<uint32_t*>(base + coords_off);
    uint32_t* cells_l1   = reinterpret_cast<uint32_t*>(base + cells_l1_off);

    // INIT: zero density slab + COPY coords from common runtime args into L1.
    // We avoid noc_async_read of the coord DRAM buffer because that read has
    // been observed to silently return shifted data when the L1 destination
    // is at a high L1 offset (the 2048-grid V19 hang root cause).
    // Common args are broadcast to all cores by tt-metal at kernel launch
    // and read directly via get_common_arg_val<>, no NoC reads involved.
    {
        DeviceZoneScopedN("V19MB-INIT-COORDS");
        for (uint32_t i = 0; i < density_slab_bins; ++i) density_l1[i] = 0u;
        for (uint32_t i = 0; i < nc_all; ++i) {
            noc_coords[i] = get_common_arg_val<uint32_t>((int)i);
        }
    }
    (void)noc_coord_dram;
    (void)noc_coord_pgsz;

    // CELLS-READ
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

    // EMIT
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

    // BARRIER: wait for outgoing atomics.
    {
        DeviceZoneScopedN("V19MB-BARRIER");
        noc_async_atomic_barrier();
        // Crude cross-core stall — busy-wait ~10us so peers' outgoing atomics
        // have time to ACK at this core's L1 inbox.
        volatile uint32_t spin = 0;
        for (uint32_t k = 0; k < 20000u; ++k) spin += k;
        (void)spin;
    }

    // DIAGNOSTIC: dump per-core state into the LAST 8 padded slots so the
    // host can read it back. Slots:
    //   [N-1]: density_l1 byte address
    //   [N-2]: noc_coords[0] (packed (y<<16)|x for core 0)
    //   [N-3]: noc_coord_dram (DRAM page-0 NoC byte addr LO)
    //   [N-4]: cells_l1[0] = first cell bx as seen by kernel
    //   [N-5]: cells_l1[1] = first cell by
    //   [N-6]: cells_l1[2] = first cell area_fixed
    //   [N-7]: my_core_idx
    //   [N-8]: n_cells
    // DIAGNOSTIC: dump per-core state into a SEPARATE DRAM diag buffer so we
    // don't clobber valid density bin slots. Layout per core (16 u32 slots):
    //   [0]  density_l1 byte addr   [1]  noc_coords[0] (kernel-read)
    //   [2]  noc_coord_dram arg     [3]  cells_l1[0]  = first cell bx (kernel-read)
    //   [4]  cells_l1[1] (by)       [5]  cells_l1[2] (area_fixed)
    //   [6]  my_core_idx            [7]  n_cells
    //   [8]  cell_dram_addr arg     [9]  cell_dram_pgsz arg
    //   [10] first_page_id arg      [11] n_pages arg
    //   [12] coords[my_core_idx] (sanity)
    //   [13] cells_l1[3] (next cell bx)
    //   [14] (uint32_t)cells_l1 (L1 byte addr)
    //   [15] 0xCAFEBABE (sentinel)
    {
        const InterleavedAddrGen<true> diag_gen = {
            .bank_base_address = diag_dram_base,
            .page_size         = diag_dram_pgsz,
        };
        // Write directly via noc_async_write from a small L1 staging buffer.
        // Use the marker padding region of density slab (NOT the addressable
        // bin range — only the last 16 u32 = density_l1[density_slab_bins-16..-1]
        // which is reserved padding via the 8-align).
        if (density_slab_bins >= 16u) {
            uint32_t b = density_slab_bins;
            uint32_t* diag = &density_l1[b - 16u];
            diag[ 0] = (uint32_t)density_l1;
            diag[ 1] = noc_coords[0];
            diag[ 2] = noc_coord_dram;
            diag[ 3] = cells_l1[0];
            diag[ 4] = cells_l1[1];
            diag[ 5] = cells_l1[2];
            diag[ 6] = my_core_idx;
            diag[ 7] = n_cells;
            diag[ 8] = cell_dram_addr;
            diag[ 9] = cell_dram_pgsz;
            diag[10] = first_page_id;
            diag[11] = n_pages;
            diag[12] = (my_core_idx < 110u) ? noc_coords[my_core_idx] : 0xBADBADu;
            diag[13] = cells_l1[3];
            diag[14] = (uint32_t)cells_l1;
            diag[15] = 0xCAFEBABEu;
            // Send diag page to separate DRAM staging.
            uint64_t diag_page = diag_gen.get_noc_addr(my_core_idx);
            noc_async_write((uint32_t)diag, diag_page, 16u * 4u);
            noc_async_write_barrier();
            // Zero the markers in L1 so they don't pollute the writeout.
            for (uint32_t k = 0; k < 16u; ++k) diag[k] = 0u;
        }
    }

    // WRITEOUT: stream this core's L1 density slab to DRAM staging.
    {
        DeviceZoneScopedN("V19MB-WRITEOUT");
        const InterleavedAddrGen<true> dgen = {
            .bank_base_address = density_dram_base,
            .page_size         = density_dram_pgsz,
        };
        constexpr uint32_t MAX_CHUNK_BYTES = 64u * 1024u;
        const uint32_t total_bytes = density_slab_bins * 4u;
        uint64_t page_base = dgen.get_noc_addr(my_core_idx);
        uint32_t l1_addr   = reinterpret_cast<uint32_t>(density_l1);
        for (uint32_t off = 0; off < total_bytes; off += MAX_CHUNK_BYTES) {
            uint32_t chunk = total_bytes - off;
            if (chunk > MAX_CHUNK_BYTES) chunk = MAX_CHUNK_BYTES;
            noc_async_write(l1_addr + off, page_base + (uint64_t)off, chunk);
        }
        noc_async_write_barrier();
    }
}
