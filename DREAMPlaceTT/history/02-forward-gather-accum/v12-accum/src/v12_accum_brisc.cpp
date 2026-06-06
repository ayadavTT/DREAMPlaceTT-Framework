// SPDX-License-Identifier: Apache-2.0
//
// V12 BRISC Accumulate Kernel (RISCV_0, NOC_0) — Phase 2.
//
// Runs AFTER the Phase 1 scatter barrier has fired (v12_scatter_brisc.cpp
// incremented the coordinator and received the go signal).
//
// For each density tile owned by this core:
//   1. Scan all nc_all writers' V12Overlap vectors from route_buf.
//   2. For each vector that contributes to this tile:
//        a. Build BF16 32×32 OX tile:  OX[bxw][col] = overlap_x[j] ∀ col.
//        b. Build BF16 32×32 OY tile:  OY[row][byw] = overlap_y[k] ∀ row.
//        c. Push OX to CB_OX, push OY to CB_OY (TRISC accumulates).
//   3. Push a sentinel OX tile (element[0] = 0xFFFF) → TRISC sees end-of-tile.
//   4. Wait for CB_ACCUM (TRISC pushes packed FP32 result after sentineling).
//      NCRISC drains CB_ACCUM independently (scale + density_buf writeback).
//
// OX definition:  row = bxw = (bxl+j) & 31,  all 32 columns set to overlap_x[j].
// OY definition:  col = byw = (byl+k) & 31,  all 32 rows set to overlap_y[k].
// OX @ OY product: result[bxw][byw] = overlap_x[j] * overlap_y[k] — exactly
//   the bin-area contribution of this cell to density[bx][by].
//
// Sentinel tile: element[0] = 0xFFFFu (BF16 NaN). TRISC pops only this OX
//   tile (no matching OY), breaks out of inner loop, and packs accumulated DST.
//
// This kernel runs on BRISC in the SAME Program as the scatter kernels. It is
// a separate .cpp file (different kernel entry point). The host creates one
// Program with both scatter and accum kernels registered on BRISC.
//
// Runtime args (positions match the full v12 arg list; only accum-relevant
// args are used here):
//   10: my_core_id
//   11: nc_all
//   12: M_tiles
//   13: N_tiles
//   14: nbx
//   15: nby
//   16: route_dram
//   17: route_pgsz
//   18: max_overlaps_half
//   23: inv_ba_u32  (passed to NCRISC; also accessible by BRISC if needed)
//   24: n_tiles_owned
//   25..25+n_tiles_owned-1: tile linear indices (tx * N_tiles + ty)

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"
#include <cstring>

static constexpr uint32_t CB_OX      = tt::CBIndex::c_0;
static constexpr uint32_t CB_OY      = tt::CBIndex::c_1;
static constexpr uint32_t CB_ACCUM   = tt::CBIndex::c_3;
static constexpr uint32_t CB_SCRATCH = tt::CBIndex::c_24;

static constexpr uint32_t TILE_W       = 32u;
static constexpr uint32_t TILE_H       = 32u;
static constexpr uint32_t TILE_BF16_SZ = TILE_W * TILE_H * (uint32_t)sizeof(uint16_t);
static constexpr uint32_t TILE_FP32_SZ = TILE_W * TILE_H * (uint32_t)sizeof(float);
static constexpr uint32_t MAX_OVERLAP  = 8u;
static constexpr uint32_t V12_HDR_BYTES= 32u;

static constexpr uint16_t V12_OX_DONE_SENTINEL = 0xFFFFu;

struct V12Overlap {
    float    overlap_x[8];
    float    overlap_y[8];
    uint16_t bxl;
    uint16_t byl;
    uint8_t  wx;
    uint8_t  wy;
    uint8_t  pad[10];
};
static_assert(sizeof(V12Overlap) == 80u, "V12Overlap must be 80 bytes");

inline uint16_t f32_to_bf16(float x) {
    uint32_t bits;
    __builtin_memcpy(&bits, &x, 4);
    return (uint16_t)(bits >> 16);
}

void kernel_main() {
    const uint32_t my_core_id       = get_arg_val<uint32_t>(10);
    const uint32_t nc_all           = get_arg_val<uint32_t>(11);
    const uint32_t M_tiles          = get_arg_val<uint32_t>(12);
    const uint32_t N_tiles          = get_arg_val<uint32_t>(13);
    const uint32_t nbx              = get_arg_val<uint32_t>(14);
    const uint32_t nby              = get_arg_val<uint32_t>(15);
    const uint32_t route_dram       = get_arg_val<uint32_t>(16);
    const uint32_t route_pgsz       = get_arg_val<uint32_t>(17);
    const uint32_t max_overlaps_half= get_arg_val<uint32_t>(18);
    const uint32_t n_tiles_owned    = get_arg_val<uint32_t>(24);

    if (n_tiles_owned == 0u) {
        // No tiles owned; still need to push sentinels so TRISC doesn't hang
        // if n_tiles_owned was passed to TRISC as 0 (TRISC returns early).
        return;
    }

    // ── L1 layout: reuse CB_SCRATCH already set up by scatter phase ──────────
    // The scatter phase already loaded tile_to_core. We skip past all scatter
    // regions to reach Phase 2 private buffers.
    uint8_t* base = reinterpret_cast<uint8_t*>(get_write_ptr(CB_SCRATCH));
    uint32_t off = 0u;

    const uint32_t tile_map_bytes = M_tiles * N_tiles * (uint32_t)sizeof(uint16_t);

    // Skip scatter regions (must mirror v12_scatter_brisc.cpp exactly).
    off += tile_map_bytes;           off = (off + 63u) & ~63u;  // tile_to_core
    off += 4u * TILE_FP32_SZ;       off = (off + 63u) & ~63u;  // cell px/py/sx/sy
    // BRISC scatter staging
    constexpr uint32_t B_STAGE_CAP = 4u;
    off += nc_all * B_STAGE_CAP * (uint32_t)sizeof(V12Overlap);
    off  = (off + 3u) & ~3u;
    off += nc_all * (uint32_t)sizeof(uint32_t);  // staging_count_b
    off += nc_all * (uint32_t)sizeof(uint32_t);  // dram_off_vecs_b
    off  = (off + 63u) & ~63u;
    off += V12_HDR_BYTES;            off = (off + 63u) & ~63u;  // hdr scratch
    // NCRISC scatter staging
    constexpr uint32_t N_STAGE_CAP = 4u;
    off += nc_all * N_STAGE_CAP * (uint32_t)sizeof(V12Overlap);
    off  = (off + 3u) & ~3u;
    off += nc_all * (uint32_t)sizeof(uint32_t);  // staging_count_n
    off += nc_all * (uint32_t)sizeof(uint32_t);  // dram_off_vecs_n
    off  = (off + 63u) & ~63u;
    off += V12_HDR_BYTES;            off = (off + 63u) & ~63u;  // NCRISC hdr
    // NCRISC private cell buffers
    off += 4u * TILE_FP32_SZ;       off = (off + 63u) & ~63u;

    // ── Phase 2 private buffers ───────────────────────────────────────────────
    // Inbound page headers from all writers targeting this core.
    constexpr uint32_t HDR_STRIDE = V12_HDR_BYTES;  // 32B per writer
    uint32_t* inbound_hdrs = reinterpret_cast<uint32_t*>(base + off);
    off += nc_all * HDR_STRIDE;
    off  = (off + 63u) & ~63u;

    // V12Overlap staging: OV_STAGE_CAP vectors per read chunk.
    constexpr uint32_t OV_STAGE_CAP = 8u;
    V12Overlap* ov_stage = reinterpret_cast<V12Overlap*>(base + off);
    off += OV_STAGE_CAP * (uint32_t)sizeof(V12Overlap);
    off  = (off + 63u) & ~63u;

    // BF16 OX and OY scratch tiles (one pair per contribution, reused).
    uint16_t* ox_scratch = reinterpret_cast<uint16_t*>(base + off);
    off += TILE_BF16_SZ;
    uint16_t* oy_scratch = reinterpret_cast<uint16_t*>(base + off);
    off += TILE_BF16_SZ;
    off  = (off + 63u) & ~63u;

    // ── Route buffer address generator ───────────────────────────────────────
    const InterleavedAddrGen<true> rgen = {
        .bank_base_address = route_dram,
        .page_size         = route_pgsz,
    };

    // ── Load all page headers for pages targeting this core ──────────────────
    {
        DeviceZoneScopedN("V12B2-HDR-LOAD");
        for (uint32_t w = 0u; w < nc_all; ++w) {
            uint64_t page = rgen.get_noc_addr(w * nc_all + my_core_id);
            uint32_t dst  = reinterpret_cast<uint32_t>(inbound_hdrs)
                          + w * HDR_STRIDE;
            noc_async_read(page, dst, HDR_STRIDE);
        }
        noc_async_read_barrier();
    }

    const uint32_t ncrisc_data_off = V12_HDR_BYTES;
    const uint32_t brisc_data_off  = V12_HDR_BYTES + max_overlaps_half * (uint32_t)sizeof(V12Overlap);

    // ── Per-tile loop ─────────────────────────────────────────────────────────
    for (uint32_t tile_local = 0u; tile_local < n_tiles_owned; ++tile_local) {
        DeviceZoneScopedN("V12B2-TILE");

        uint32_t tile_idx = get_arg_val<uint32_t>(25u + tile_local);
        uint32_t tx = tile_idx / N_tiles;
        uint32_t ty = tile_idx % N_tiles;

        // Iterate all writers × halves × overlap vectors.
        for (uint32_t w = 0u; w < nc_all; ++w) {
            const uint32_t* hdr = inbound_hdrs + w * (HDR_STRIDE / sizeof(uint32_t));
            uint32_t cnts[2] = { hdr[0], hdr[1] };  // count_n, count_b
            uint32_t offs[2] = { ncrisc_data_off, brisc_data_off };

            for (uint32_t half = 0u; half < 2u; ++half) {
                uint32_t cnt = cnts[half];
                if (cnt > max_overlaps_half) cnt = max_overlaps_half;
                if (cnt == 0u) continue;

                uint64_t page = rgen.get_noc_addr(w * nc_all + my_core_id);
                uint32_t vec_off = 0u;

                while (vec_off < cnt) {
                    uint32_t chunk = cnt - vec_off;
                    if (chunk > OV_STAGE_CAP) chunk = OV_STAGE_CAP;
                    uint64_t src = page
                                 + (uint64_t)offs[half]
                                 + (uint64_t)vec_off * sizeof(V12Overlap);
                    noc_async_read(src,
                                   reinterpret_cast<uint32_t>(ov_stage),
                                   chunk * (uint32_t)sizeof(V12Overlap));
                    noc_async_read_barrier();

                    for (uint32_t vi = 0u; vi < chunk; ++vi) {
                        const V12Overlap& ov = ov_stage[vi];
                        int32_t  bxl = (int32_t)(uint32_t)ov.bxl;
                        int32_t  byl = (int32_t)(uint32_t)ov.byl;
                        uint32_t wx  = (uint32_t)ov.wx;
                        uint32_t wy  = (uint32_t)ov.wy;
                        if (wx == 0u || wy == 0u) continue;

                        // Check if this vector contributes to tile (tx,ty).
                        bool has_x = false, has_y = false;
                        for (uint32_t j = 0u; j < wx; ++j)
                            if (((uint32_t)(bxl + (int32_t)j) >> 5u) == tx)
                                { has_x = true; break; }
                        for (uint32_t k = 0u; k < wy; ++k)
                            if (((uint32_t)(byl + (int32_t)k) >> 5u) == ty)
                                { has_y = true; break; }
                        if (!has_x || !has_y) continue;

                        // Build OX: row bxw = (bxl+j)&31, all 32 cols = overlap_x[j].
                        for (uint32_t i = 0u; i < TILE_W * TILE_H; ++i) ox_scratch[i] = 0u;
                        for (uint32_t j = 0u; j < wx; ++j) {
                            uint32_t bx  = (uint32_t)(bxl + (int32_t)j);
                            if ((bx >> 5u) != tx) continue;
                            uint32_t bxw = bx & 31u;
                            uint16_t v   = f32_to_bf16(ov.overlap_x[j]);
                            for (uint32_t c = 0u; c < TILE_W; ++c)
                                ox_scratch[bxw * TILE_W + c] = v;
                        }

                        // Build OY: all 32 rows = overlap_y[k], col byw = (byl+k)&31.
                        for (uint32_t i = 0u; i < TILE_W * TILE_H; ++i) oy_scratch[i] = 0u;
                        for (uint32_t k = 0u; k < wy; ++k) {
                            uint32_t by  = (uint32_t)(byl + (int32_t)k);
                            if ((by >> 5u) != ty) continue;
                            uint32_t byw = by & 31u;
                            uint16_t v   = f32_to_bf16(ov.overlap_y[k]);
                            for (uint32_t r = 0u; r < TILE_H; ++r)
                                oy_scratch[r * TILE_W + byw] = v;
                        }

                        // Push OX tile.
                        cb_reserve_back(CB_OX, 1u);
                        {
                            uint16_t* dst = reinterpret_cast<uint16_t*>(
                                                get_write_ptr(CB_OX));
                            for (uint32_t i = 0u; i < TILE_W * TILE_H; ++i)
                                dst[i] = ox_scratch[i];
                        }
                        cb_push_back(CB_OX, 1u);

                        // Push OY tile.
                        cb_reserve_back(CB_OY, 1u);
                        {
                            uint16_t* dst = reinterpret_cast<uint16_t*>(
                                                get_write_ptr(CB_OY));
                            for (uint32_t i = 0u; i < TILE_W * TILE_H; ++i)
                                dst[i] = oy_scratch[i];
                        }
                        cb_push_back(CB_OY, 1u);
                    }  // for vi
                    vec_off += chunk;
                }  // while vec_off < cnt
            }  // for half
        }  // for w

        // Push sentinel OX tile: element[0] = 0xFFFF, rest zero.
        cb_reserve_back(CB_OX, 1u);
        {
            uint16_t* dst = reinterpret_cast<uint16_t*>(get_write_ptr(CB_OX));
            for (uint32_t i = 0u; i < TILE_W * TILE_H; ++i) dst[i] = 0u;
            dst[0] = V12_OX_DONE_SENTINEL;
        }
        cb_push_back(CB_OX, 1u);

        // NCRISC independently drains CB_ACCUM (BRISC doesn't need to wait).
    }  // for tile_local
}
