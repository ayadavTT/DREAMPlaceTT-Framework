// SPDX-License-Identifier: Apache-2.0
//
// V11 Phase 3 Reduce-BC Kernel (RISCV_0, NOC_0) — shard sum + scale + write.
//
// Runs after wl_v11_reduce_a has finished (host Finish() ensures all shard
// writes from reduce_a are in DRAM before this kernel reads them).
//
// Phase B: for each hot primary tile (K>1), read K-1 shard pages from
//          shard_reduce_buf and BRISC-add them into the primary's dense slot.
// Phase C: scale all primary dense tiles by inv_bin_area; write each
//          primary tile to density_buf (same layout as original accum).
//
// Dense is still live in CB_SCRATCH at offset `dense_offset_bytes`.
// tmp_shard is placed immediately after the dense region in the same CB.
//
// Runtime args:
//   0:  dense_offset_bytes      offset of dense[0] from CB_SCRATCH base
//   1:  shard_reduce_dram_addr
//   2:  shard_reduce_pgsz       (= TILE_BYTES = 4096)
//   3:  density_dram_addr
//   4:  density_pgsz            (= nby * 4)
//   5:  inv_bin_area_u32
//   6:  M_tiles
//   7:  N_tiles
//   8:  nbx
//   9:  nby
//  10:  max_K
//  11:  n_primary
//  12:  n_primary_hot           number of primary tiles with K > 1
//  13:  n_shard                 number of shard slots (to find tmp_shard offset)
//  14..14+n_primary-1:          primary tile_ids (tile_x*N_tiles+tile_y)
//  14+n_primary..+3*n_primary_hot-1:
//       hot-primary triples (local_idx, hot_tile_seq, K) × n_primary_hot
//       (local_idx is the index 0..n_primary-1 within the primary list)

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif

constexpr uint32_t CB_SCRATCH  = tt::CBIndex::c_24;
constexpr uint32_t TILE_W      = 32u;
constexpr uint32_t TILE_H      = 32u;
constexpr uint32_t TILE_FLOATS = TILE_W * TILE_H;
constexpr uint32_t TILE_BYTES  = TILE_FLOATS * sizeof(float);

void kernel_main() {
    const uint32_t dense_offset   = get_arg_val<uint32_t>(0);
    const uint32_t srb_dram       = get_arg_val<uint32_t>(1);
    const uint32_t srb_pgsz       = get_arg_val<uint32_t>(2);
    const uint32_t density_dram   = get_arg_val<uint32_t>(3);
    const uint32_t density_pgsz   = get_arg_val<uint32_t>(4);
    const uint32_t inv_ba_u32     = get_arg_val<uint32_t>(5);
    const uint32_t M_tiles        = get_arg_val<uint32_t>(6);
    const uint32_t N_tiles        = get_arg_val<uint32_t>(7);
    const uint32_t nbx            = get_arg_val<uint32_t>(8);
    const uint32_t nby            = get_arg_val<uint32_t>(9);
    const uint32_t max_K          = get_arg_val<uint32_t>(10);
    const uint32_t n_primary      = get_arg_val<uint32_t>(11);
    const uint32_t n_primary_hot  = get_arg_val<uint32_t>(12);
    const uint32_t n_shard        = get_arg_val<uint32_t>(13);

    union { uint32_t u; float f; } cv;
    cv.u = inv_ba_u32;
    const float inv_bin_area = cv.f;

    uint8_t* base  = reinterpret_cast<uint8_t*>(get_write_ptr(CB_SCRATCH));
    float*   dense = reinterpret_cast<float*>(base + dense_offset);
    // tmp_shard sits immediately after all primary+shard dense slots
    float*   tmp_shard = dense + (n_primary + n_shard) * TILE_FLOATS;

    // ── Phase B: sum K-1 shard pages into each hot primary's dense slot ───
    if (n_primary_hot > 0u) {
        const InterleavedAddrGen<true> rgen = {
            .bank_base_address = srb_dram,
            .page_size         = srb_pgsz,
        };
        uint32_t trip_base = 14u + n_primary;  // arg index of first hot triple
        for (uint32_t h = 0; h < n_primary_hot; ++h) {
            uint32_t local_idx = get_arg_val<uint32_t>(trip_base + 3u * h);
            uint32_t hot_seq   = get_arg_val<uint32_t>(trip_base + 3u * h + 1u);
            uint32_t K         = get_arg_val<uint32_t>(trip_base + 3u * h + 2u);
            float*   prim_slot = dense + local_idx * TILE_FLOATS;
            for (uint32_t s = 1u; s < K; ++s) {
                uint32_t page = hot_seq * max_K + s;
                noc_async_read(rgen.get_noc_addr(page),
                               reinterpret_cast<uint32_t>(tmp_shard),
                               TILE_BYTES);
                noc_async_read_barrier();
                for (uint32_t i = 0; i < TILE_FLOATS; ++i)
                    prim_slot[i] += tmp_shard[i];
            }
        }
    }

    // ── Phase C: scale by inv_bin_area ───────────────────────────────────
    {
        const uint32_t total_floats = n_primary * TILE_FLOATS;
        for (uint32_t i = 0; i < total_floats; ++i) dense[i] *= inv_bin_area;
    }

    // ── Phase C: write primary tiles to density_buf ──────────────────────
    {
        const InterleavedAddrGen<true> dgen = {
            .bank_base_address = density_dram,
            .page_size         = density_pgsz,
        };
        for (uint32_t local = 0; local < n_primary; ++local) {
            uint32_t tile_idx = get_arg_val<uint32_t>(14u + local);
            uint32_t tile_x   = tile_idx / N_tiles;
            uint32_t tile_y   = tile_idx % N_tiles;
            for (uint32_t bxw = 0; bxw < 32u; ++bxw) {
                uint32_t bx = tile_x * 32u + bxw;
                if (bx >= nbx) break;
                uint64_t page_base = dgen.get_noc_addr(bx);
                uint64_t dst = page_base + (uint64_t)tile_y * 32u * sizeof(float);
                uint32_t src_l1 = reinterpret_cast<uint32_t>(
                    &dense[local * TILE_FLOATS + bxw * TILE_W]);
                uint32_t y_count = 32u;
                if (tile_y * 32u + y_count > nby)
                    y_count = nby - tile_y * 32u;
                noc_async_write(src_l1, dst, y_count * sizeof(float));
            }
        }
        noc_async_write_barrier();
    }
}
