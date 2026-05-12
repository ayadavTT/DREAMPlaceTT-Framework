// SPDX-License-Identifier: Apache-2.0
//
// V11 BRISC Accumulator + Reduce Kernel (RISCV_0, NOC_0).
//
// Combined Phase 2 + Phase 3 reduce. Runs as a single program launch so the
// host pays only ONE Finish() barrier for the whole gather phase (instead of
// 3 with the old accum + reduce_a + reduce_bc split, which cost ~16ms in
// extra host-device round-trips).
//
// Steps performed by every Tensix core in this kernel:
//   1. Load per-core owned_lookup[] from DRAM.
//   2. Zero dense[] in L1 (n_total = n_primary + n_shard tiles).
//   3. Read route_buf[*][me] in SRC_CHUNK batches and accumulate tuples
//      into dense[] using owned_lookup to map tile_id → local slot.
//   4. (Shard owners only) Write each owned shard tile to shard_reduce_buf
//      DRAM page, barrier, then atomically increment the primary's
//      semaphore. The barrier is per-shard so the increment cannot race
//      ahead of the DRAM data.
//   5. (Hot primary only — at most 1 per core) Wait for K-1 semaphore incs
//      from the K-1 shard owners, then read each shard page back and add
//      it into the primary's dense slot.
//   6. (All primary) Scale all n_primary dense tiles by inv_bin_area.
//   7. (All primary) Write each primary tile out as 32 partial-page writes
//      into density_buf (column-major-per-row layout for TTNN).
//
// Important: this kernel assumes at most 1 hot primary tile per core.
// build_shard_table picks primaries by core load, so this holds in practice.
//
// L1 layout (in CB_SCRATCH = c_24):
//   owned_lookup[M_tiles*N_tiles]              uint16
//   inbound_hdrs[nc_all][V11_PAGE_HDR_BYTES]   uint32 (count at offset 0)
//   inbound_buf[SRC_CHUNK][max_per_writer]     V11Contrib
//   safety gap (128 bytes)
//   dense[n_primary + n_shard][32][32]         float
//   tmp_shard[32][32]                          float (Phase B scratch)
//
// Runtime args:
//   0:  owned_lookup_dram_addr
//   1:  owned_lookup_pgsz
//   2:  my_core_id
//   3:  nc_all
//   4:  M_tiles
//   5:  N_tiles
//   6:  nbx
//   7:  nby
//   8:  route_dram_addr
//   9:  route_pgsz
//  10:  max_per_writer
//  11:  density_dram_addr
//  12:  density_pgsz
//  13:  inv_bin_area_u32
//  14:  n_primary
//  15:  n_shard
//  16:  srb_dram             (shard_reduce_buf DRAM base)
//  17:  srb_pgsz             (= TILE_BYTES = 4096)
//  18:  max_K
//  19:  n_primary_hot        (number of primary tiles with K>1 on this core)
//  20:  nc_split             writer split point: BRISC reads [0..nc_split)
//  21:  merge_sem_id         semaphore BRISC waits on (NCRISC inc's once)
//  22:  prim_noc_x           THIS core's NOC X (unused on BRISC)
//  23:  prim_noc_y           THIS core's NOC Y (unused on BRISC)
//  [24..24+n_primary-1]:                        primary_tile_ids
//  [24+n_primary..+4*n_primary_hot-1]:          hot quads
//                                               (local_idx, hot_seq, K, sem_id)
//  [24+n_primary+4*n_primary_hot..+5*n_shard-1]: shard quints
//                                               (hot_seq, shard_idx, prim_noc_x, prim_noc_y, sem_id)
//
// Multiple hot primaries on the same core use different sem_id slots
// (allocated as separate semaphores by the host), so each tile's signal
// stream is isolated.

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t CB_SCRATCH = tt::CBIndex::c_24;
// SRC_CHUNK is how many writer pages we batch-read into L1 in one go before
// running the per-tuple accumulate. The total L1 staging is
//   SRC_CHUNK * max_per_writer * 8 bytes  (per RISC, doubled for BRISC+NCRISC).
// Halved from 16 to 8 to free L1 budget for a doubled max_per_writer (4096),
// which is needed to avoid silent tuple drops at hot (writer, receiver) pairs.
constexpr uint32_t SRC_CHUNK  = 8u;
constexpr uint32_t TILE_W     = 32u;
constexpr uint32_t TILE_H     = 32u;
constexpr uint32_t TILE_FLOATS= TILE_W * TILE_H;
constexpr uint32_t TILE_BYTES = TILE_FLOATS * sizeof(float);

// Page header is 64 bytes for NOC cache-line alignment (matches v11_scatter_dm.cpp).
constexpr uint32_t V11_PAGE_HDR_BYTES = 64u;

// 8-byte tuple (matches v11_scatter_dm.cpp). Scatter rounds cnt up to a
// multiple of 4 with zero-padding so DRAM writes stay 32-byte aligned, and
// the cnt observed here is also a multiple of 4. Reading cnt*8 bytes is
// always a multiple of 32 → safe.
struct V11Contrib { uint16_t bx; uint16_t by; float area; };
static_assert(sizeof(V11Contrib) == 8, "V11Contrib must be 8 bytes");

void kernel_main() {
    const uint32_t owned_lookup_dram = get_arg_val<uint32_t>(0);
    const uint32_t owned_lookup_pgsz = get_arg_val<uint32_t>(1);
    const uint32_t my_core_id        = get_arg_val<uint32_t>(2);
    const uint32_t nc_all            = get_arg_val<uint32_t>(3);
    const uint32_t M_tiles           = get_arg_val<uint32_t>(4);
    const uint32_t N_tiles           = get_arg_val<uint32_t>(5);
    const uint32_t nbx               = get_arg_val<uint32_t>(6);
    const uint32_t nby               = get_arg_val<uint32_t>(7);
    const uint32_t route_dram        = get_arg_val<uint32_t>(8);
    const uint32_t route_pgsz        = get_arg_val<uint32_t>(9);
    const uint32_t max_per_writer    = get_arg_val<uint32_t>(10);
    const uint32_t density_dram      = get_arg_val<uint32_t>(11);
    const uint32_t density_pgsz      = get_arg_val<uint32_t>(12);
    const uint32_t inv_ba_u32        = get_arg_val<uint32_t>(13);
    const uint32_t n_primary         = get_arg_val<uint32_t>(14);
    const uint32_t n_shard           = get_arg_val<uint32_t>(15);
    const uint32_t srb_dram          = get_arg_val<uint32_t>(16);
    const uint32_t srb_pgsz          = get_arg_val<uint32_t>(17);
    const uint32_t max_K             = get_arg_val<uint32_t>(18);
    const uint32_t n_primary_hot     = get_arg_val<uint32_t>(19);
    const uint32_t nc_split          = get_arg_val<uint32_t>(20);
    const uint32_t merge_sem_id      = get_arg_val<uint32_t>(21);
    // args 22/23 (my_noc_x/y) unused on BRISC; consumed by NCRISC.

    const uint32_t n_total = n_primary + n_shard;

    union { uint32_t u; float f; } cv;
    cv.u = inv_ba_u32;
    const float inv_bin_area = cv.f;

    // ── L1 layout ─────────────────────────────────────────────────────────
    uint8_t* base = reinterpret_cast<uint8_t*>(get_write_ptr(CB_SCRATCH));
    uint32_t off = 0;

    uint16_t* owned_lookup = reinterpret_cast<uint16_t*>(base + off);
    const uint32_t total_tiles = M_tiles * N_tiles;
    off += total_tiles * sizeof(uint16_t);
    off = (off + 7u) & ~7u;

    uint32_t* inbound_hdrs = reinterpret_cast<uint32_t*>(base + off);
    off += nc_all * V11_PAGE_HDR_BYTES;
    off = (off + 63u) & ~63u;
    // NCRISC's inbound_hdrs region (we don't read it, just skip past).
    off += nc_all * V11_PAGE_HDR_BYTES;
    off = (off + 63u) & ~63u;

    V11Contrib* inbound_buf = reinterpret_cast<V11Contrib*>(base + off);
    off += SRC_CHUNK * max_per_writer * sizeof(V11Contrib);
    off = (off + 63u) & ~63u;
    // NCRISC's inbound_buf region (skip).
    off += SRC_CHUNK * max_per_writer * sizeof(V11Contrib);
    off = (off + 63u) & ~63u;
    off += 128u;  // safety gap for NOC overshoot

    float* dense = reinterpret_cast<float*>(base + off);
    // Layout: [dense_b][dense_n][tmp_shard].
    float* dense_n = dense + n_total * TILE_FLOATS;
    float* tmp_shard = dense_n + n_total * TILE_FLOATS;

    // ── Step 0: load per-core owned_lookup[] from DRAM ────────────────────
    {
        DeviceZoneScopedN("V11A-LOOKUP-LOAD");
        const InterleavedAddrGen<true> lgen = {
            .bank_base_address = owned_lookup_dram,
            .page_size         = owned_lookup_pgsz,
        };
        noc_async_read(lgen.get_noc_addr(my_core_id),
                       reinterpret_cast<uint32_t>(owned_lookup),
                       total_tiles * sizeof(uint16_t));
        noc_async_read_barrier();
    }

    // ── Step 1: zero accumulator (n_total = primary + shard) ──────────────
    {
        DeviceZoneScopedN("V11A-ZERO");
        const uint32_t total_floats = n_total * TILE_FLOATS;
        for (uint32_t i = 0; i < total_floats; ++i) dense[i] = 0.0f;
    }

    // ── Setup route_buf address generator ─────────────────────────────────
    const InterleavedAddrGen<true> rgen = {
        .bank_base_address = route_dram,
        .page_size         = route_pgsz,
    };

    // ── Step 2a: bulk-read headers for BRISC's writer range [0..nc_split) ─
    {
        DeviceZoneScopedN("V11A-HDR-READ-ALL");
        for (uint32_t s = 0; s < nc_split; ++s) {
            uint64_t page_base = rgen.get_noc_addr(s * nc_all + my_core_id);
            uint32_t dst_l1 = reinterpret_cast<uint32_t>(inbound_hdrs)
                            + s * V11_PAGE_HDR_BYTES;
            noc_async_read(page_base, dst_l1, V11_PAGE_HDR_BYTES);
        }
        noc_async_read_barrier();
    }

    // ── Step 2b: per-chunk tuple read + accumulate over [0..nc_split) ─────
    // Each writer page is split: NCRISC's tuples in [0..max_per_writer/2),
    // BRISC's tuples in [max_per_writer/2..max_per_writer). We read the
    // whole tuple region in one NOC read then iterate cnt_n + cnt_b.
    const uint32_t HALF_CAP = max_per_writer / 2u;
    for (uint32_t cs = 0; cs < nc_split; cs += SRC_CHUNK) {
        uint32_t cs_end = cs + SRC_CHUNK;
        if (cs_end > nc_split) cs_end = nc_split;
        const uint32_t batch = cs_end - cs;

        {
            DeviceZoneScopedN("V11A-DATA-READ");
            for (uint32_t s = cs; s < cs_end; ++s) {
                uint32_t cnt_n = inbound_hdrs[s * (V11_PAGE_HDR_BYTES / 4) + 0];
                uint32_t cnt_b = inbound_hdrs[s * (V11_PAGE_HDR_BYTES / 4) + 8];
                if (cnt_n == 0u && cnt_b == 0u) continue;
                if (cnt_n > HALF_CAP) cnt_n = HALF_CAP;
                if (cnt_b > HALF_CAP) cnt_b = HALF_CAP;
                uint64_t page_base = rgen.get_noc_addr(s * nc_all + my_core_id);
                uint64_t src_addr  = page_base + (uint64_t)V11_PAGE_HDR_BYTES;
                uint32_t dst_l1    = reinterpret_cast<uint32_t>(
                                        &inbound_buf[(s - cs) * max_per_writer]);
                // Read entire tuple region in one NOC read (max_per_writer
                // bytes-wise = max_per_writer * 8). Padding tuples have
                // area=0 and are skipped on the inner loop's a==0 check.
                noc_async_read(src_addr, dst_l1, max_per_writer * sizeof(V11Contrib));
            }
            noc_async_read_barrier();
        }

        {
            DeviceZoneScopedN("V11A-ACC");
            for (uint32_t bs = 0; bs < batch; ++bs) {
                uint32_t s = cs + bs;
                uint32_t cnt_n = inbound_hdrs[s * (V11_PAGE_HDR_BYTES / 4) + 0];
                uint32_t cnt_b = inbound_hdrs[s * (V11_PAGE_HDR_BYTES / 4) + 8];
                if (cnt_n > HALF_CAP) cnt_n = HALF_CAP;
                if (cnt_b > HALF_CAP) cnt_b = HALF_CAP;
                if (cnt_n == 0u && cnt_b == 0u) continue;
                const V11Contrib* base = &inbound_buf[bs * max_per_writer];
                const V11Contrib* halves[2] = { base, base + HALF_CAP };
                uint32_t halves_cnt[2] = { cnt_n, cnt_b };
                for (uint32_t half = 0; half < 2u; ++half) {
                    uint32_t cnt = halves_cnt[half];
                    if (cnt == 0u) continue;
                    const V11Contrib* src = halves[half];
                    uint32_t i = 0;
                    uint32_t cnt4 = cnt & ~3u;
                    for (; i < cnt4; i += 4) {
                        #pragma GCC unroll 4
                        for (uint32_t k = 0; k < 4u; ++k) {
                            uint32_t bx = src[i + k].bx;
                            uint32_t by = src[i + k].by;
                            float    a  = src[i + k].area;
                            if (a == 0.0f) continue;
                            uint32_t map_idx = (bx >> 5) * N_tiles + (by >> 5);
                            uint32_t local = (uint32_t)owned_lookup[map_idx];
                            if (local >= n_total) continue;
                            uint32_t bxw = bx & 31u;
                            uint32_t byw = by & 31u;
                            dense[local * TILE_FLOATS + bxw * TILE_W + byw] += a;
                        }
                    }
                    for (; i < cnt; ++i) {
                        uint32_t bx = src[i].bx;
                        uint32_t by = src[i].by;
                        float    a  = src[i].area;
                        if (a == 0.0f) continue;
                        uint32_t map_idx = (bx >> 5) * N_tiles + (by >> 5);
                        uint32_t local = (uint32_t)owned_lookup[map_idx];
                        if (local >= n_total) continue;
                        uint32_t bxw = bx & 31u;
                        uint32_t byw = by & 31u;
                        dense[local * TILE_FLOATS + bxw * TILE_W + byw] += a;
                    }
                }
            }
        }
    }

    // ── Merge: wait for NCRISC to finish, then sum dense_n into dense_b ──
    {
        DeviceZoneScopedN("V11A-MERGE");
        volatile tt_l1_ptr uint32_t* mptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(merge_sem_id));
        noc_semaphore_wait(mptr, 1u);
        noc_semaphore_set(mptr, 0u);  // reset for next launch
        const uint32_t total_floats = n_total * TILE_FLOATS;
        for (uint32_t i = 0; i < total_floats; ++i) dense[i] += dense_n[i];
    }

    // ── Phase A: shard owners write their dense slot to srb + signal primary
    const uint32_t hot_quad_base   = 24u + n_primary;
    const uint32_t shard_quint_base = hot_quad_base + 4u * n_primary_hot;

    if (n_shard > 0u) {
        DeviceZoneScopedN("V11A-SHARD-WRITE");
        const InterleavedAddrGen<true> srb_gen = {
            .bank_base_address = srb_dram,
            .page_size         = srb_pgsz,
        };
        for (uint32_t i = 0; i < n_shard; ++i) {
            uint32_t hot_seq   = get_arg_val<uint32_t>(shard_quint_base + 5u * i + 0);
            uint32_t shard_idx = get_arg_val<uint32_t>(shard_quint_base + 5u * i + 1);
            uint32_t prim_x    = get_arg_val<uint32_t>(shard_quint_base + 5u * i + 2);
            uint32_t prim_y    = get_arg_val<uint32_t>(shard_quint_base + 5u * i + 3);
            uint32_t sem_id    = get_arg_val<uint32_t>(shard_quint_base + 5u * i + 4);
            uint32_t local_idx = n_primary + i;
            uint32_t page      = hot_seq * max_K + shard_idx;
            uint32_t src_l1    = reinterpret_cast<uint32_t>(dense + local_idx * TILE_FLOATS);
            noc_async_write(src_l1, srb_gen.get_noc_addr(page), TILE_BYTES);
            // Per-shard barrier: ensures the DRAM write committed before we
            // signal the primary, so primary's read after wait sees the data.
            noc_async_write_barrier();
            uint32_t sem_l1_off = (uint32_t)get_semaphore(sem_id);
            uint64_t sem_noc_addr = get_noc_addr(prim_x, prim_y, sem_l1_off);
            noc_semaphore_inc(sem_noc_addr, 1u);
        }
    }

    // ── Phase B: hot primary waits for shard signals + sums ───────────────
    if (n_primary_hot > 0u) {
        DeviceZoneScopedN("V11A-SHARD-SUM");
        const InterleavedAddrGen<true> srb_gen = {
            .bank_base_address = srb_dram,
            .page_size         = srb_pgsz,
        };
        for (uint32_t h = 0; h < n_primary_hot; ++h) {
            uint32_t local_idx = get_arg_val<uint32_t>(hot_quad_base + 4u * h + 0);
            uint32_t hot_seq   = get_arg_val<uint32_t>(hot_quad_base + 4u * h + 1);
            uint32_t K         = get_arg_val<uint32_t>(hot_quad_base + 4u * h + 2);
            uint32_t sem_id    = get_arg_val<uint32_t>(hot_quad_base + 4u * h + 3);
            volatile tt_l1_ptr uint32_t* sem_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));
            // Wait for K-1 shard owners on the dedicated slot for this tile.
            noc_semaphore_wait(sem_ptr, K - 1u);
            noc_semaphore_set(sem_ptr, 0u);  // reset for next launch

            float* prim_slot = dense + local_idx * TILE_FLOATS;
            for (uint32_t s = 1u; s < K; ++s) {
                uint32_t page = hot_seq * max_K + s;
                noc_async_read(srb_gen.get_noc_addr(page),
                               reinterpret_cast<uint32_t>(tmp_shard),
                               TILE_BYTES);
                noc_async_read_barrier();
                for (uint32_t i = 0; i < TILE_FLOATS; ++i)
                    prim_slot[i] += tmp_shard[i];
            }
        }
    }

    // ── Phase C: scale primary tiles by inv_bin_area ──────────────────────
    if (n_primary > 0u) {
        DeviceZoneScopedN("V11A-SCALE");
        const uint32_t total_floats = n_primary * TILE_FLOATS;
        for (uint32_t i = 0; i < total_floats; ++i) dense[i] *= inv_bin_area;
    }

    // ── Phase C: write primary tiles to density_buf ───────────────────────
    if (n_primary > 0u) {
        DeviceZoneScopedN("V11A-DENSITY-WRITE");
        const InterleavedAddrGen<true> dgen = {
            .bank_base_address = density_dram,
            .page_size         = density_pgsz,
        };
        for (uint32_t local = 0; local < n_primary; ++local) {
            uint32_t tile_idx = get_arg_val<uint32_t>(24u + local);
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
