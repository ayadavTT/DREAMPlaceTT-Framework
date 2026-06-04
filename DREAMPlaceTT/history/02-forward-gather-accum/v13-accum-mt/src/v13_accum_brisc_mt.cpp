// SPDX-License-Identifier: Apache-2.0
//
// V13_fpu Step 1.3.1 — multi-tile BRISC accumulate driver.
//
// One core's perspective:
//   1. Load owned_lookup[total_tiles] from DRAM (host built it; owned_lookup
//      [tile_id] = local_idx for owned tiles, 0xFFFF otherwise).
//   2. Read all writer page headers in one bulk NOC read.
//   3. For each owned tile T (sequentially):
//        For each writer w:
//          stream records from route_buf[w, my_core] in chunks of
//          MAX_READ_RECORDS. For each record:
//            if owned_lookup[record.tile_id] == T:
//              append to staging[T]; when full (K=32) pack OX/OY,
//              push to TRISC, reset.
//        End of writer loop: flush partial batch with zero padding,
//        push sentinel to CB_OX (signals TRISC to commit DST).
//        Wait for CB_DENSE; convert face→row-major; write 32-row tile
//        slab to density_buf[T's (tile_x, tile_y)] in DRAM.
//   4. Done — all density tiles written.
//
// Compute kernel pairs with this via sentinels; see v13_accum_compute_mt.cpp.

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t CB_OX      = tt::CBIndex::c_0;
constexpr uint32_t CB_OY      = tt::CBIndex::c_1;
constexpr uint32_t CB_DENSE   = tt::CBIndex::c_16;
constexpr uint32_t CB_SCRATCH = tt::CBIndex::c_24;
constexpr uint32_t TILE_BF16_BYTES = 32 * 32 * 2;  // 2048
constexpr uint32_t V13_RECORD_BYTES   = 40u;
constexpr uint32_t V13_PAGE_HDR_BYTES = 64u;
constexpr uint32_t K_BATCH = 32u;
// OPT-1: batch N_INFLIGHT matmul tiles per cb_push_back/cb_reserve_back to
// amortize the ~12 µs/matmul CB sync + driver overhead. BRISC packs
// PUSH_BATCH = N_INFLIGHT × K_BATCH records into N_INFLIGHT contiguous CB
// slots, pushes them all in one call. TRISC fires N_INFLIGHT matmul_tiles
// back-to-back. CB depth must be ≥ 2 × N_INFLIGHT for pipelining.
constexpr uint32_t N_INFLIGHT = 4u;
constexpr uint32_t PUSH_BATCH = N_INFLIGHT * K_BATCH;   // 128
constexpr uint32_t MAX_READ_RECORDS = 64u;        // chunk size for route_buf reads
constexpr uint32_t SENTINEL_U32 = 0xFFFFFFFFu;

struct V13Record {
    int16_t  bxl_local;
    int16_t  byl_local;
    uint16_t tile_id;
    uint16_t pad;
    uint16_t overlap_x[8];
    uint16_t overlap_y[8];
};
static_assert(sizeof(V13Record) == 40, "V13Record must be 40 bytes");

// Tile/face element offset (in uint16_t units) for 32x32 bf16 tile.
static inline uint32_t face_idx(int row, int col) {
    uint32_t face = ((uint32_t)(row >> 4) * 2u) + (uint32_t)(col >> 4);
    return face * 256u + ((uint32_t)row & 15u) * 16u + ((uint32_t)col & 15u);
}

// Pack K cells into OX (col=k contains cell k's overlap_x[..]) and OY (row=k
// contains cell k's overlap_y[..]). Tiles are zeroed first.
static inline void pack_ox_oy(
    const V13Record* batch, uint32_t k_used,
    uint16_t* ox, uint16_t* oy)
{
    uint32_t* ox32 = reinterpret_cast<uint32_t*>(ox);
    uint32_t* oy32 = reinterpret_cast<uint32_t*>(oy);
    for (uint32_t i = 0; i < TILE_BF16_BYTES / 4; ++i) {
        ox32[i] = 0;
        oy32[i] = 0;
    }
    for (uint32_t k = 0; k < k_used; ++k) {
        const V13Record& r = batch[k];
        int bxl = r.bxl_local;
        int byl = r.byl_local;
        for (uint32_t j = 0; j < 8; ++j) {
            int row = bxl + (int)j;
            if (row >= 0 && row < 32) {
                ox[face_idx(row, (int)k)] = r.overlap_x[j];
            }
        }
        for (uint32_t j = 0; j < 8; ++j) {
            int col = byl + (int)j;
            if (col >= 0 && col < 32) {
                oy[face_idx((int)k, col)] = r.overlap_y[j];
            }
        }
    }
}

void kernel_main() {
    const uint32_t owned_lookup_dram  = get_arg_val<uint32_t>(0);
    const uint32_t owned_lookup_pgsz  = get_arg_val<uint32_t>(1);
    const uint32_t my_core_id         = get_arg_val<uint32_t>(2);
    const uint32_t nc_all             = get_arg_val<uint32_t>(3);
    const uint32_t M_tiles            = get_arg_val<uint32_t>(4);
    const uint32_t N_tiles            = get_arg_val<uint32_t>(5);
    const uint32_t nbx                = get_arg_val<uint32_t>(6);
    const uint32_t nby                = get_arg_val<uint32_t>(7);
    const uint32_t route_dram         = get_arg_val<uint32_t>(8);
    const uint32_t route_pgsz         = get_arg_val<uint32_t>(9);
    const uint32_t records_cap        = get_arg_val<uint32_t>(10);
    const uint32_t density_dram       = get_arg_val<uint32_t>(11);
    const uint32_t density_pgsz       = get_arg_val<uint32_t>(12);
    const uint32_t n_owned_tiles      = get_arg_val<uint32_t>(13);
    // owned_tile_ids[0..n_owned_tiles): at args [14 .. 14+n_owned_tiles)

    (void)M_tiles;

    const uint32_t total_tiles = M_tiles * N_tiles;
    const uint32_t half_cap    = records_cap / 2u;

    // ── L1 layout ─────────────────────────────────────────────────────────
    uint8_t* base = reinterpret_cast<uint8_t*>(get_write_ptr(CB_SCRATCH));
    uint32_t off = 0;

    uint16_t* owned_lookup = reinterpret_cast<uint16_t*>(base + off);
    off += total_tiles * sizeof(uint16_t);
    off = (off + 63u) & ~63u;

    // Headers buffer: nc_all writer headers, 64 bytes each (full page header).
    // Blackhole NOC requires 32-byte-aligned multi-of-32 reads; read each
    // 64-byte header in a single transaction.
    uint8_t* inbound_hdrs_raw = reinterpret_cast<uint8_t*>(base + off);
    off += nc_all * V13_PAGE_HDR_BYTES;
    off = (off + 63u) & ~63u;

    // Per-writer record chunk buffer (read up to MAX_READ_RECORDS at a time).
    V13Record* inbound_buf = reinterpret_cast<V13Record*>(base + off);
    off += MAX_READ_RECORDS * V13_RECORD_BYTES;
    off = (off + 63u) & ~63u;

    // Per-tile staging buffers (n_owned_tiles × PUSH_BATCH × 40 B).
    // PUSH_BATCH = N_INFLIGHT × K_BATCH records to support OPT-1 batching.
    V13Record* per_tile_staging = reinterpret_cast<V13Record*>(base + off);
    off += n_owned_tiles * PUSH_BATCH * V13_RECORD_BYTES;
    off = (off + 63u) & ~63u;
    uint32_t* per_tile_count = reinterpret_cast<uint32_t*>(base + off);
    off += n_owned_tiles * sizeof(uint32_t);
    off = (off + 63u) & ~63u;

    // Row-major staging for density write (one 32-row × 32-col bf16 tile).
    uint16_t* dense_row_major = reinterpret_cast<uint16_t*>(base + off);
    off += TILE_BF16_BYTES;
    off = (off + 63u) & ~63u;

    // ── Load owned_lookup from DRAM ───────────────────────────────────────
    {
        DeviceZoneScopedN("V13A-LOOKUP-LOAD");
        const InterleavedAddrGen<true> lgen = {
            .bank_base_address = owned_lookup_dram,
            .page_size         = owned_lookup_pgsz,
        };
        noc_async_read(lgen.get_noc_addr(my_core_id),
                       reinterpret_cast<uint32_t>(owned_lookup),
                       total_tiles * sizeof(uint16_t));
        noc_async_read_barrier();
    }

    // ── Read all writer page headers in one shot ──────────────────────────
    const InterleavedAddrGen<true> rgen = {
        .bank_base_address = route_dram,
        .page_size         = route_pgsz,
    };
    {
        DeviceZoneScopedN("V13A-HDR-READ-ALL");
        for (uint32_t w = 0; w < nc_all; ++w) {
            uint64_t page = rgen.get_noc_addr(w * nc_all + my_core_id);
            uint32_t dst = reinterpret_cast<uint32_t>(
                inbound_hdrs_raw + (size_t)w * V13_PAGE_HDR_BYTES);
            noc_async_read(page, dst, V13_PAGE_HDR_BYTES);
        }
        noc_async_read_barrier();
    }

    // ── Density write helper: face → row-major, then NOC write 32 strides ─
    const InterleavedAddrGen<true> dgen = {
        .bank_base_address = density_dram,
        .page_size         = density_pgsz,
    };

    auto write_density_tile = [&](uint32_t tile_id) {
        uint32_t tile_x = tile_id / N_tiles;
        uint32_t tile_y = tile_id % N_tiles;

        // Convert CB_DENSE (face layout) → dense_row_major.
        const uint16_t* src = reinterpret_cast<const uint16_t*>(get_read_ptr(CB_DENSE));
        for (int row = 0; row < 32; ++row) {
            for (int col = 0; col < 32; ++col) {
                dense_row_major[row * 32 + col] = src[face_idx(row, col)];
            }
        }

        // Write 32 row strips of 32 bf16 = 64 bytes each.
        // Each strip targets density_buf[row, tile_y*32 .. tile_y*32+31].
        for (uint32_t bxw = 0; bxw < 32u; ++bxw) {
            uint32_t bx = tile_x * 32u + bxw;
            if (bx >= nbx) break;  // off-grid row (x = row, bounded by M = nbx)
            uint64_t row_base = dgen.get_noc_addr(bx);
            uint64_t dst = row_base + (uint64_t)tile_y * 32u * sizeof(uint16_t);
            uint32_t y_count = 32u;
            if (tile_y * 32u + y_count > nby)
                y_count = nby - tile_y * 32u;
            uint32_t src_l1 = reinterpret_cast<uint32_t>(
                &dense_row_major[bxw * 32]);
            noc_async_write(src_l1, dst, y_count * sizeof(uint16_t));
        }
        noc_async_write_barrier();
    };

    // ── Per owned tile loop ───────────────────────────────────────────────
    for (uint32_t t = 0; t < n_owned_tiles; ++t) {
        DeviceZoneScopedN("V13A-TILE");
        const uint32_t target_tile_id = get_arg_val<uint32_t>(14u + t);
        per_tile_count[t] = 0u;

        // Stream records from each writer, filter by tile_id == target.
        {
        DeviceZoneScopedN("V13A-READ-PACK");
        for (uint32_t w = 0; w < nc_all; ++w) {
            const uint32_t* hdr32 = reinterpret_cast<const uint32_t*>(
                inbound_hdrs_raw + (size_t)w * V13_PAGE_HDR_BYTES);
            uint32_t cnt_n = hdr32[0];        // cnt_n at byte 0
            uint32_t cnt_b = hdr32[8];        // cnt_b at byte 32 = u32 index 8
            if (cnt_n > half_cap) cnt_n = half_cap;
            if (cnt_b > half_cap) cnt_b = half_cap;
            uint32_t total_recs = cnt_n + cnt_b;
            if (total_recs == 0u) continue;

            // Two regions to read: NCRISC half [page + hdr .. page + hdr + cnt_n*40),
            // BRISC half [page + hdr + half_cap*40 .. page + hdr + half_cap*40 + cnt_b*40).
            uint64_t page_base = rgen.get_noc_addr(w * nc_all + my_core_id);
            uint64_t n_region_base = page_base + (uint64_t)V13_PAGE_HDR_BYTES;
            uint64_t b_region_base = n_region_base + (uint64_t)half_cap * V13_RECORD_BYTES;

            // Read region in chunks. Round read size up to multiple of
            // 32 bytes (Blackhole NOC requirement). Records past `to_read`
            // in the buffer are over-read but never iterated — safe.
            auto read_and_process = [&](uint64_t region_base, uint32_t count) {
                uint32_t done = 0;
                while (done < count) {
                    uint32_t to_read = count - done;
                    if (to_read > MAX_READ_RECORDS) to_read = MAX_READ_RECORDS;
                    uint64_t src = region_base + (uint64_t)done * V13_RECORD_BYTES;
                    uint32_t read_bytes = to_read * V13_RECORD_BYTES;
                    read_bytes = (read_bytes + 31u) & ~31u;  // round up to 32 B
                    noc_async_read(src,
                                   reinterpret_cast<uint32_t>(inbound_buf),
                                   read_bytes);
                    noc_async_read_barrier();
                    for (uint32_t i = 0; i < to_read; ++i) {
                        const V13Record& r = inbound_buf[i];
                        if ((uint32_t)r.tile_id != target_tile_id) continue;
                        // Append to per-tile staging[t]. Capacity PUSH_BATCH.
                        uint32_t c = per_tile_count[t];
                        per_tile_staging[t * PUSH_BATCH + c] = r;
                        c++;
                        per_tile_count[t] = c;
                        if (c >= PUSH_BATCH) {
                            // OPT-1: pack N_INFLIGHT=4 tiles SEQUENTIALLY
                            // (1 cb_reserve+push per tile to handle ring wrap).
                            // TRISC waits for full N_INFLIGHT and fires 4
                            // matmuls per cb_wait/pop pair — amortizes the
                            // CB sync overhead 4×.
                            DeviceZoneScopedN("V13A-PACK");
                            for (uint32_t q = 0; q < N_INFLIGHT; ++q) {
                                {
                                    DeviceZoneScopedN("V13A-PACK-RESERVE");
                                    cb_reserve_back(CB_OX, 1);
                                    cb_reserve_back(CB_OY, 1);
                                }
                                {
                                    DeviceZoneScopedN("V13A-PACK-FILL");
                                    pack_ox_oy(&per_tile_staging[t * PUSH_BATCH + q * K_BATCH],
                                               K_BATCH,
                                               reinterpret_cast<uint16_t*>(get_write_ptr(CB_OX)),
                                               reinterpret_cast<uint16_t*>(get_write_ptr(CB_OY)));
                                }
                                {
                                    DeviceZoneScopedN("V13A-PACK-PUSH");
                                    cb_push_back(CB_OX, 1);
                                    cb_push_back(CB_OY, 1);
                                }
                            }
                            per_tile_count[t] = 0;
                        }
                    }
                    done += to_read;
                }
            };
            if (cnt_n > 0u) read_and_process(n_region_base, cnt_n);
            if (cnt_b > 0u) read_and_process(b_region_base, cnt_b);
        }

        // Flush partial batch. With PUSH_BATCH = N_INFLIGHT × K_BATCH,
        // c_remain ∈ [0, PUSH_BATCH-1]. OPT-1 contract: TRISC always
        // consumes a FULL N_INFLIGHT-tile batch per cb_wait/pop pair. So
        // BRISC pads any partial flush to exactly N_INFLIGHT tiles using
        // zero-record packs (which produce zero OX/OY → zero matmul
        // contribution, safe to accumulate into DST).
        uint32_t c_remain = per_tile_count[t];
        if (c_remain > 0u) {
            uint32_t left = c_remain;
            uint32_t q = 0;
            while (q < N_INFLIGHT) {
                uint32_t k_used = (left > K_BATCH) ? K_BATCH : left;
                cb_reserve_back(CB_OX, 1);
                cb_reserve_back(CB_OY, 1);
                pack_ox_oy(&per_tile_staging[t * PUSH_BATCH + q * K_BATCH],
                           k_used,
                           reinterpret_cast<uint16_t*>(get_write_ptr(CB_OX)),
                           reinterpret_cast<uint16_t*>(get_write_ptr(CB_OY)));
                cb_push_back(CB_OX, 1);
                cb_push_back(CB_OY, 1);
                left = (left > k_used) ? left - k_used : 0;
                ++q;
            }
            per_tile_count[t] = 0;
        }

        // Push N_INFLIGHT sentinel OX tiles (a full sentinel batch so TRISC's
        // cb_wait_front(N_INFLIGHT) won't deadlock on tile-end). The OX
        // sentinel is detected by TRISC reading element[0] of the first
        // tile of the batch.
        for (uint32_t q = 0; q < N_INFLIGHT; ++q) {
            cb_reserve_back(CB_OX, 1);
            uint32_t* sent = reinterpret_cast<uint32_t*>(get_write_ptr(CB_OX));
            for (uint32_t i = 0; i < TILE_BF16_BYTES / 4; ++i) sent[i] = SENTINEL_U32;
            cb_push_back(CB_OX, 1);
        }
        }  // end V13A-READ-PACK

        // Wait for TRISC to produce the per-tile density and write it to DRAM.
        {
            DeviceZoneScopedN("V13A-WAIT-DENSE");
            cb_wait_front(CB_DENSE, 1);
        }
        {
            DeviceZoneScopedN("V13A-DENSITY-WRITE");
            write_density_tile(target_tile_id);
            cb_pop_front(CB_DENSE, 1);
        }
    }
}
