// SPDX-License-Identifier: Apache-2.0
//
// V20 permute microbench — shared body for BRISC + NCRISC siblings.
//
// Models the proposed V20 chip-side permute pass:
//   For each output slot c in this kernel's range:
//     read perm_table[c] = (ix, iy, ox, oy)      [DRAM → L1]
//     read pos[ix] (64 B aligned chunk)           [random DRAM → L1]
//     read pos[iy] (64 B aligned chunk)           [random DRAM → L1]
//     px = pos[ix] + ox    py = pos[iy] + oy
//     write px → px_buf[c]                        [L1 → DRAM]
//     write py → py_buf[c]                        [L1 → DRAM]
//
// Each Tensix core's cells are split into two contiguous ranges:
//   BRISC handles [0, cpc/2)        on NoC 0
//   NCRISC handles [cpc/2, cpc)     on NoC 1
// so the two NoCs issue reads/writes in parallel.
//
// `V20_CB_SCRATCH` must be defined by the including .cpp before this header.
// BRISC and NCRISC use *different* CBs so their L1 staging buffers don't collide.
//
// Pos reads use NOC_DRAM_READ_ALIGNMENT=64 on Blackhole (see memory note
// blackhole_dram_read_align_64.md).

#pragma once

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"

#ifndef V20_CB_SCRATCH
#error "V20_CB_SCRATCH must be defined before including v19_mb_permute_common.h"
#endif

#ifndef V20_BATCH_OVERRIDE
#define V20_BATCH_OVERRIDE 128
#endif
constexpr uint32_t V20_BATCH          = V20_BATCH_OVERRIDE;
constexpr uint32_t V20_POS_READ_ALIGN = 64;
constexpr uint32_t V20_L1_WRITE_ALIGN = 16;

// PERM_CHUNK_BATCHES = N → outer loop reads N BATCHes worth of perm at once,
// then iterates the inner pos-read/compute/write loop N times before another
// perm read. Sweep showed N=1..4 all perf-equivalent (perm read isn't on the
// critical path); N≥8 SILENTLY corrupts because the noc_async_read multi-packet
// (any_len) path doesn't honor 64-B intra-packet alignment when the total read
// > NOC_MAX_BURST_SIZE. Keep N=1 for safety; structure is preserved for future
// experimentation.
#ifndef V20_PERM_CHUNK_BATCHES_OVERRIDE
#define V20_PERM_CHUNK_BATCHES_OVERRIDE 1
#endif
constexpr uint32_t V20_PERM_CHUNK_BATCHES = V20_PERM_CHUNK_BATCHES_OVERRIDE;
constexpr uint32_t V20_PERM_CHUNK_CELLS   = V20_PERM_CHUNK_BATCHES * V20_BATCH;

inline void v20_permute_main() {
    const uint32_t my_core_idx       = get_arg_val<uint32_t>(0);
    const uint32_t cells_start       = get_arg_val<uint32_t>(1);   // first cell in this kernel's range
    const uint32_t cells_end         = get_arg_val<uint32_t>(2);   // one past last
    const uint32_t pos_dram_base     = get_arg_val<uint32_t>(3);
    const uint32_t pos_dram_pgsz     = get_arg_val<uint32_t>(4);
    const uint32_t floats_per_pos_pg = get_arg_val<uint32_t>(5);
    const uint32_t perm_dram_base    = get_arg_val<uint32_t>(6);
    const uint32_t perm_pg_bytes     = get_arg_val<uint32_t>(7);
    const uint32_t pxbuf_dram_base   = get_arg_val<uint32_t>(8);
    const uint32_t pybuf_dram_base   = get_arg_val<uint32_t>(9);
    const uint32_t xy_pg_bytes       = get_arg_val<uint32_t>(10);

    uint8_t* base = reinterpret_cast<uint8_t*>(get_write_ptr(V20_CB_SCRATCH));

    // L1 layout (per kernel, 64-B aligned throughout):
    //   perm_l1     : PERM_CHUNK_CELLS × 16 B   (default 8 KB for 512 cells)
    //   pos_stage_x : BATCH × 64 B              = 8192 B
    //   pos_stage_y : BATCH × 64 B              = 8192 B
    //   px_out_l1   : BATCH × 4 B               = 512 B
    //   py_out_l1   : BATCH × 4 B               = 512 B
    // Total ~25 KB per kernel, fits in 32 KB CB.
    constexpr uint32_t PERM_BYTES      = V20_PERM_CHUNK_CELLS * 16;
    constexpr uint32_t POS_STAGE_BYTES = V20_BATCH * V20_POS_READ_ALIGN;
    constexpr uint32_t OUT_BYTES       = V20_BATCH * 4;
    const uint32_t perm_l1_off     = 0;
    const uint32_t pos_stage_x_off = perm_l1_off     + PERM_BYTES;
    const uint32_t pos_stage_y_off = pos_stage_x_off + POS_STAGE_BYTES;
    const uint32_t px_out_l1_off   = pos_stage_y_off + POS_STAGE_BYTES;
    const uint32_t py_out_l1_off   = px_out_l1_off   + OUT_BYTES;

    uint32_t* perm_l1     = reinterpret_cast<uint32_t*>(base + perm_l1_off);
    uint8_t*  pos_stage_x = base + pos_stage_x_off;
    uint8_t*  pos_stage_y = base + pos_stage_y_off;
    float*    px_out_l1   = reinterpret_cast<float*>(base + px_out_l1_off);
    float*    py_out_l1   = reinterpret_cast<float*>(base + py_out_l1_off);

    const InterleavedAddrGen<true> perm_pgen = {
        .bank_base_address = perm_dram_base,
        .page_size         = perm_pg_bytes,
    };
    const InterleavedAddrGen<true> pos_pgen = {
        .bank_base_address = pos_dram_base,
        .page_size         = pos_dram_pgsz,
    };
    const InterleavedAddrGen<true> pxbuf_pgen = {
        .bank_base_address = pxbuf_dram_base,
        .page_size         = xy_pg_bytes,
    };
    const InterleavedAddrGen<true> pybuf_pgen = {
        .bank_base_address = pybuf_dram_base,
        .page_size         = xy_pg_bytes,
    };

    // BRISC and NCRISC on this core share the same perm/px/py pages
    // (they just touch disjoint cell ranges).
    const uint64_t my_perm_base  = perm_pgen.get_noc_addr(my_core_idx);
    const uint64_t my_pxbuf_base = pxbuf_pgen.get_noc_addr(my_core_idx);
    const uint64_t my_pybuf_base = pybuf_pgen.get_noc_addr(my_core_idx);

    {
        DeviceZoneScopedN("V20MB-PERMUTE-ALL");
        uint32_t cells_done = cells_start;
        while (cells_done < cells_end) {
            // ── Outer loop: read perm for up to PERM_CHUNK_CELLS cells. ──
            // cells_done is mult of 4 (host invariant); chunk is mult of 4 too
            // (cells_end - cells_done is mult of 4, and PERM_CHUNK_CELLS is mult of 4).
            // So chunk * 16 is mult of 64 → 64-aligned read.
            uint32_t chunk = cells_end - cells_done;
            if (chunk > V20_PERM_CHUNK_CELLS) chunk = V20_PERM_CHUNK_CELLS;
            {
                DeviceZoneScopedN("V20MB-READ-PERM");
                const uint32_t perm_off   = cells_done * 16;
                const uint32_t perm_bytes = chunk * 16u;
                noc_async_read(my_perm_base + perm_off,
                               (uint32_t)perm_l1, perm_bytes);
                noc_async_read_barrier();
            }

            // ── Inner loop: process the chunk in BATCH-sized sub-batches. ──
            uint32_t local_done = 0;
            while (local_done < chunk) {
                uint32_t this_batch = chunk - local_done;
                if (this_batch > V20_BATCH) this_batch = V20_BATCH;

                // ── 2. Per-cell 64-B aligned reads into pos. ──
                {
                    DeviceZoneScopedN("V20MB-READ-POS");
                    for (uint32_t c = 0; c < this_batch; ++c) {
                        const uint32_t lc = local_done + c;
                        const uint32_t ix = perm_l1[lc * 4 + 0];
                        const uint32_t iy = perm_l1[lc * 4 + 1];
                        const uint32_t ix_pg          = ix / floats_per_pos_pg;
                        const uint32_t ix_byte_in_pg  = (ix % floats_per_pos_pg) * 4;
                        const uint32_t ix_aligned_off = ix_byte_in_pg & ~(V20_POS_READ_ALIGN - 1u);
                        const uint32_t iy_pg          = iy / floats_per_pos_pg;
                        const uint32_t iy_byte_in_pg  = (iy % floats_per_pos_pg) * 4;
                        const uint32_t iy_aligned_off = iy_byte_in_pg & ~(V20_POS_READ_ALIGN - 1u);

                        const uint64_t ix_noc = pos_pgen.get_noc_addr(ix_pg) + ix_aligned_off;
                        const uint64_t iy_noc = pos_pgen.get_noc_addr(iy_pg) + iy_aligned_off;
                        noc_async_read(ix_noc,
                                       (uint32_t)(pos_stage_x + c * V20_POS_READ_ALIGN), V20_POS_READ_ALIGN);
                        noc_async_read(iy_noc,
                                       (uint32_t)(pos_stage_y + c * V20_POS_READ_ALIGN), V20_POS_READ_ALIGN);
                    }
                    noc_async_read_barrier();
                }

                // ── 3. Extract correct float from each 64-B chunk + compute. ──
                {
                    DeviceZoneScopedN("V20MB-COMPUTE");
                    union { uint32_t u; float f; } cvt;
                    for (uint32_t c = 0; c < this_batch; ++c) {
                        const uint32_t lc = local_done + c;
                        const uint32_t ix = perm_l1[lc * 4 + 0];
                        const uint32_t iy = perm_l1[lc * 4 + 1];
                        const uint32_t ix_intra = ((ix % floats_per_pos_pg) * 4) & (V20_POS_READ_ALIGN - 1u);
                        const uint32_t iy_intra = ((iy % floats_per_pos_pg) * 4) & (V20_POS_READ_ALIGN - 1u);
                        const float ix_val = *reinterpret_cast<float*>(pos_stage_x + c * V20_POS_READ_ALIGN + ix_intra);
                        const float iy_val = *reinterpret_cast<float*>(pos_stage_y + c * V20_POS_READ_ALIGN + iy_intra);
                        cvt.u = perm_l1[lc * 4 + 2]; const float ox = cvt.f;
                        cvt.u = perm_l1[lc * 4 + 3]; const float oy = cvt.f;
                        px_out_l1[c] = ix_val + ox;
                        py_out_l1[c] = iy_val + oy;
                    }
                }

                // ── 4. Write px_out / py_out to DRAM (write align = 16 B). ──
                {
                    DeviceZoneScopedN("V20MB-WRITE-OUT");
                    const uint32_t out_off   = (cells_done + local_done) * 4;
                    const uint32_t out_bytes = (this_batch * 4u + V20_L1_WRITE_ALIGN - 1u) & ~(V20_L1_WRITE_ALIGN - 1u);
                    noc_async_write((uint32_t)px_out_l1,
                                    my_pxbuf_base + out_off, out_bytes);
                    noc_async_write((uint32_t)py_out_l1,
                                    my_pybuf_base + out_off, out_bytes);
                    noc_async_write_barrier();
                }

                local_done += this_batch;
            }
            cells_done += chunk;
        }
    }
}
