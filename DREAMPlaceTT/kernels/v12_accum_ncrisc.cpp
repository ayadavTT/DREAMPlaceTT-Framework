// SPDX-License-Identifier: Apache-2.0
//
// V12 NCRISC Accumulate Kernel (RISCV_1, NOC_1) — Phase 2.
//
// Drains CB_ACCUM (FP32 32×32 packed tiles produced by TRISC), applies
// inv_bin_area scaling in-place, and writes 32 row-slices per tile to
// density_buf (row-major DRAM layout: page = bx row, 32 floats at offset
// ty*32*sizeof(float)).
//
// Runs concurrently with BRISC (which feeds TRISC). NCRISC uses NOC_1 so
// it doesn't contend with BRISC's NOC_0 route_buf reads.
//
// Runtime args (positions match the full v12 arg list):
//   11: nc_all
//   12: M_tiles
//   13: N_tiles
//   14: nbx    (used for row-boundary clamp)
//   15: nby    (used for column-boundary clamp)
//   23: inv_ba_u32
//   24: n_tiles_owned
//   25..25+n_tiles_owned-1: tile linear indices

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"
#include <cstring>

static constexpr uint32_t CB_ACCUM   = tt::CBIndex::c_3;
static constexpr uint32_t CB_SCRATCH = tt::CBIndex::c_24;
static constexpr uint32_t TILE_W     = 32u;
static constexpr uint32_t TILE_H     = 32u;
static constexpr uint32_t TILE_FLOATS= TILE_W * TILE_H;
static constexpr uint32_t TILE_BYTES = TILE_FLOATS * sizeof(float);

void kernel_main() {
    const uint32_t nc_all        = get_arg_val<uint32_t>(11);
    const uint32_t M_tiles       = get_arg_val<uint32_t>(12);
    const uint32_t N_tiles       = get_arg_val<uint32_t>(13);
    const uint32_t nbx           = get_arg_val<uint32_t>(14);
    const uint32_t nby           = get_arg_val<uint32_t>(15);
    const uint32_t density_dram  = get_arg_val<uint32_t>(16);  // reuse arg slot 16
    // Actually, density_dram needs its own slot. Check the full arg list:
    // arg 16 is route_dram in scatter kernels. We need a new slot for accum.
    // Host must place density_dram and density_pgsz at args 25+n_tiles_owned
    // OR we can place accum-specific args after the per-tile indices.
    // For now, place density args at fixed offsets known by host:
    //   arg 16 = route_dram (scatter), but accum NCRISC only runs in Phase 2
    //   after scatter, so route_dram is no longer needed. We REUSE arg 16 as
    //   density_dram and arg 17 as density_pgsz in this kernel.
    // The host uses the same runtime args for both scatter and accum kernels;
    // since these are different .cpp files (different kernel objects), they
    // have independent arg lists. See v12 host for exact assignment.
    const uint32_t density_pgsz  = get_arg_val<uint32_t>(17);
    const uint32_t inv_ba_u32    = get_arg_val<uint32_t>(23);
    const uint32_t n_tiles_owned = get_arg_val<uint32_t>(24);

    union { uint32_t u; float f; } cv; cv.u = inv_ba_u32;
    const float inv_bin_area = cv.f;

    if (n_tiles_owned == 0u) return;

    // L1 scratch: we need a TILE_BYTES FP32 buffer to receive CB_ACCUM,
    // apply scale, and then write row-by-row to density_buf.
    // We allocate it in CB_SCRATCH just past Phase-1 + BRISC accum regions.
    // For simplicity, use a fixed offset that is guaranteed to be free.
    // The CB_SCRATCH was sized by the host to include this buffer.
    // We use get_write_ptr(CB_ACCUM) after popping (but CB_ACCUM is a TRISC
    // output CB so we read it, not write it).
    //
    // NCRISC reads from CB_ACCUM via get_read_ptr, applies scale, then writes
    // directly from L1 to DRAM. We need a local writable copy for scaling.
    // Place it at a fixed L1 offset in CB_SCRATCH.
    uint8_t* scratch = reinterpret_cast<uint8_t*>(get_write_ptr(CB_SCRATCH));
    // Generous offset: skip all scatter + BRISC accum regions. Use a large
    // fixed offset (host guarantees CB_SCRATCH is large enough).
    // We compute the offset the same way as v12_accum_brisc.cpp then add
    // a density_tile_buf region.
    {
        uint32_t tile_map_bytes = M_tiles * N_tiles * (uint32_t)sizeof(uint16_t);
        uint32_t skip = 0u;
        skip += tile_map_bytes;                 skip = (skip + 63u) & ~63u;
        skip += 4u * TILE_BYTES;                skip = (skip + 63u) & ~63u;
        constexpr uint32_t B_CAP = 4u, N_CAP = 4u;
        skip += nc_all * B_CAP * 80u;           skip = (skip + 3u) & ~3u;
        skip += nc_all * 2u * sizeof(uint32_t); skip = (skip + 63u) & ~63u;
        skip += 32u;                            skip = (skip + 63u) & ~63u;
        skip += nc_all * N_CAP * 80u;           skip = (skip + 3u) & ~3u;
        skip += nc_all * 2u * sizeof(uint32_t); skip = (skip + 63u) & ~63u;
        skip += 32u;                            skip = (skip + 63u) & ~63u;
        skip += 4u * TILE_BYTES;                skip = (skip + 63u) & ~63u;
        // BRISC Phase 2 regions: hdrs + ov_stage + ox/oy scratch
        skip += nc_all * 32u;                   skip = (skip + 63u) & ~63u;
        skip += 8u * 80u;                       skip = (skip + 63u) & ~63u;
        skip += 2u * (TILE_W * TILE_H * sizeof(uint16_t)); skip = (skip + 63u) & ~63u;
        // NCRISC's density tile buffer lands here.
        scratch += skip;
    }
    float* density_tile = reinterpret_cast<float*>(scratch);

    const InterleavedAddrGen<true> dgen = {
        .bank_base_address = density_dram,
        .page_size         = density_pgsz,
    };

    for (uint32_t tile_local = 0u; tile_local < n_tiles_owned; ++tile_local) {
        DeviceZoneScopedN("V12N2-TILE");

        uint32_t tile_idx = get_arg_val<uint32_t>(25u + tile_local);
        uint32_t tx = tile_idx / N_tiles;
        uint32_t ty = tile_idx % N_tiles;

        // Wait for TRISC to push packed FP32 tile to CB_ACCUM.
        cb_wait_front(CB_ACCUM, 1u);

        // Copy from CB_ACCUM L1 buffer to our writable scratch, apply scale.
        {
            const float* src = reinterpret_cast<const float*>(get_read_ptr(CB_ACCUM));
            for (uint32_t i = 0u; i < TILE_FLOATS; ++i)
                density_tile[i] = src[i] * inv_bin_area;
        }
        cb_pop_front(CB_ACCUM, 1u);

        // Write 32 bx rows (each = 32 floats starting at column ty*32).
        {
            DeviceZoneScopedN("V12N2-DRAM-WRITE");
            for (uint32_t bxw = 0u; bxw < TILE_W; ++bxw) {
                uint32_t bx = tx * TILE_W + bxw;
                if (bx >= nbx) break;
                uint32_t y_count = TILE_H;
                if (ty * TILE_H + y_count > nby) y_count = nby - ty * TILE_H;
                if (y_count == 0u) break;

                uint64_t page_base = dgen.get_noc_addr(bx);
                uint64_t dst_addr  = page_base
                                   + (uint64_t)(ty * TILE_H) * sizeof(float);
                uint32_t src_l1    = reinterpret_cast<uint32_t>(
                                         &density_tile[bxw * TILE_W]);
                noc_async_write(src_l1, dst_addr, y_count * (uint32_t)sizeof(float));
            }
            noc_async_write_barrier();
        }
    }
}
