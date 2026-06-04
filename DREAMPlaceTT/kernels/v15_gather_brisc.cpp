// SPDX-License-Identifier: Apache-2.0
//
// V15 — Single-pass bucket-by-tile-id gather (BRISC: bucket+drain).
//
// NCRISC is a parallel BUCKETING reader only — it scans writers [55..110),
// buckets records into bucket_n[][] in L1, then signals BRISC via semaphore.
// BRISC scans writers [0..55), buckets into bucket_b[][], waits for NCRISC
// signal, then drains BOTH buckets per owned tile into ONE CB pair (CB_OX/
// CB_OY) — matching V13's compute-kernel contract exactly.
//
// Compared to V13: V15 eliminates V13's 37× per-owned-tile rescan of writer
// pages (V13A-READ-PACK ~231 µs/tile → V15A-DRAIN ~34 µs/tile at 2048-grid).
// Compared to V15-v1 (dual-CB compute): V15 uses a single CB pair, so the
// TRISC compute kernel is unchanged from V13 — no risk of unpacker state
// drift between CB pairs.
//
// Padding-record filter (critical):
//   V13 scatter pads each (writer, receiver) flush with all-zero records to
//   align to 8. These records have tile_id=0 and all-zero overlap. V13 gather
//   naturally skipped them via `tile_id != target` filter. V15's bucket lookup
//   would otherwise route padding into bucket[owned_lookup[0]] on the core
//   that owns tile 0, flooding it and dropping real records. We detect
//   padding via "any overlap nonzero?" early in pass 1.

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

constexpr uint32_t TILE_BF16_BYTES    = 32u * 32u * 2u;  // 2048
constexpr uint32_t V13_RECORD_BYTES   = 40u;
constexpr uint32_t V13_PAGE_HDR_BYTES = 64u;
constexpr uint32_t K_BATCH            = 32u;
constexpr uint32_t N_INFLIGHT         = 4u;
constexpr uint32_t PUSH_BATCH         = N_INFLIGHT * K_BATCH;  // 128
constexpr uint32_t MAX_READ_RECORDS   = 64u;
constexpr uint32_t SENTINEL_U32       = 0xFFFFFFFFu;

struct V13Record {
    int16_t  bxl_local;
    int16_t  byl_local;
    uint16_t tile_id;
    uint16_t pad;
    uint16_t overlap_x[8];
    uint16_t overlap_y[8];
};
static_assert(sizeof(V13Record) == 40, "V13Record must be 40 bytes");

static inline uint32_t face_idx(int row, int col) {
    uint32_t face = ((uint32_t)(row >> 4) * 2u) + (uint32_t)(col >> 4);
    return face * 256u + ((uint32_t)row & 15u) * 16u + ((uint32_t)col & 15u);
}

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
    const uint32_t bucket_cap         = get_arg_val<uint32_t>(14);
    const uint32_t ncrisc_done_sem_id = get_arg_val<uint32_t>(15);
    const uint32_t ncrisc_bucket_off  = get_arg_val<uint32_t>(16);
    const uint32_t brisc_only         = get_arg_val<uint32_t>(17);
    const uint32_t stats_dram         = get_arg_val<uint32_t>(18);
    const uint32_t spill_dram         = get_arg_val<uint32_t>(19);
    const uint32_t spill_pgsz         = get_arg_val<uint32_t>(20);
    const uint32_t max_spill_chunks   = get_arg_val<uint32_t>(21);
    const uint32_t n_owned_max        = get_arg_val<uint32_t>(22);
    // owned_tile_ids[0..n_owned_tiles): args [23 .. 23+n_owned_tiles)

    (void)M_tiles;

    const uint32_t total_tiles      = M_tiles * N_tiles;
    const uint32_t half_cap         = records_cap / 2u;
    // BRISC-only mode (used for low-n_owned dense configs where a hot tile
    // can hold > cap records per RISC — bucket overflow would otherwise drop
    // records and break convergence on adaptec3 / bigblue3). BRISC scans all
    // writers; NCRISC just immediately raises the done sem.
    const uint32_t brisc_writer_end =
        brisc_only ? nc_all : ((nc_all + 1u) / 2u);

    // ── L1 layout (BRISC region) ─────────────────────────────────────────
    uint8_t* base = reinterpret_cast<uint8_t*>(get_write_ptr(CB_SCRATCH));
    uint32_t off = 0;

    uint16_t* owned_lookup = reinterpret_cast<uint16_t*>(base + off);
    off += total_tiles * sizeof(uint16_t);
    off = (off + 63u) & ~63u;

    uint8_t* inbound_hdrs_raw = reinterpret_cast<uint8_t*>(base + off);
    off += nc_all * V13_PAGE_HDR_BYTES;
    off = (off + 63u) & ~63u;

    V13Record* inbound_buf = reinterpret_cast<V13Record*>(base + off);
    off += MAX_READ_RECORDS * V13_RECORD_BYTES;
    off = (off + 63u) & ~63u;

    V13Record* bucket_b = reinterpret_cast<V13Record*>(base + off);
    off += n_owned_tiles * bucket_cap * V13_RECORD_BYTES;
    off = (off + 63u) & ~63u;
    uint32_t* bucket_b_count = reinterpret_cast<uint32_t*>(base + off);
    off += n_owned_tiles * sizeof(uint32_t);
    off = (off + 63u) & ~63u;
    // DRAM spill tracking: how many spill chunks each owned tile has flushed.
    // L1 only — never goes to DRAM. Counts how many cap-full bucket dumps
    // we wrote out to v15_spill_buf for that owned tile.
    uint32_t* spill_chunks_b = reinterpret_cast<uint32_t*>(base + off);
    off += n_owned_tiles * sizeof(uint32_t);
    off = (off + 63u) & ~63u;

    V13Record* staging = reinterpret_cast<V13Record*>(base + off);
    off += PUSH_BATCH * V13_RECORD_BYTES;
    off = (off + 63u) & ~63u;

    uint16_t* dense_row_major = reinterpret_cast<uint16_t*>(base + off);
    off += TILE_BF16_BYTES;
    off = (off + 63u) & ~63u;

    // NCRISC's bucket arrays live at ncrisc_bucket_off (computed by host;
    // matches NCRISC kernel's L1 layout). BRISC reads them during pass 2.
    // bucket_n_count is at the 64-byte-aligned offset AFTER bucket_n — must
    // match NCRISC kernel's `off = (off + 63u) & ~63u;` alignment exactly.
    V13Record* bucket_n = reinterpret_cast<V13Record*>(base + ncrisc_bucket_off);
    uint32_t bucket_n_count_off = ncrisc_bucket_off
        + n_owned_tiles * bucket_cap * V13_RECORD_BYTES;
    bucket_n_count_off = (bucket_n_count_off + 63u) & ~63u;
    uint32_t* bucket_n_count = reinterpret_cast<uint32_t*>(base + bucket_n_count_off);

    // ── Load owned_lookup from DRAM ───────────────────────────────────────
    {
        DeviceZoneScopedN("V15A-LOOKUP-LOAD");
        const InterleavedAddrGen<true> lgen = {
            .bank_base_address = owned_lookup_dram,
            .page_size         = owned_lookup_pgsz,
        };
        noc_async_read(lgen.get_noc_addr(my_core_id),
                       reinterpret_cast<uint32_t>(owned_lookup),
                       total_tiles * sizeof(uint16_t));
        noc_async_read_barrier();
    }

    // ── Read BRISC-half writer page headers in one shot ──────────────────
    const InterleavedAddrGen<true> rgen = {
        .bank_base_address = route_dram,
        .page_size         = route_pgsz,
    };
    {
        DeviceZoneScopedN("V15A-HDR-READ-ALL");
        for (uint32_t w = 0; w < brisc_writer_end; ++w) {
            uint64_t page = rgen.get_noc_addr(w * nc_all + my_core_id);
            uint32_t dst = reinterpret_cast<uint32_t>(
                inbound_hdrs_raw + (size_t)w * V13_PAGE_HDR_BYTES);
            noc_async_read(page, dst, V13_PAGE_HDR_BYTES);
        }
        noc_async_read_barrier();
    }

    // ── Density write helper ─────────────────────────────────────────────
    const InterleavedAddrGen<true> dgen = {
        .bank_base_address = density_dram,
        .page_size         = density_pgsz,
    };

    auto write_density_tile = [&](uint32_t tile_id) {
        uint32_t tile_x = tile_id / N_tiles;
        uint32_t tile_y = tile_id % N_tiles;
        const uint16_t* src = reinterpret_cast<const uint16_t*>(get_read_ptr(CB_DENSE));
        for (int row = 0; row < 32; ++row) {
            for (int col = 0; col < 32; ++col) {
                dense_row_major[row * 32 + col] = src[face_idx(row, col)];
            }
        }
        for (uint32_t bxw = 0; bxw < 32u; ++bxw) {
            uint32_t bx = tile_x * 32u + bxw;
            if (bx >= nbx) break;
            uint64_t row_base = dgen.get_noc_addr(bx);
            uint64_t dst = row_base + (uint64_t)tile_y * 32u * sizeof(uint16_t);
            uint32_t y_count = 32u;
            if (tile_y * 32u + y_count > nby)
                y_count = nby - tile_y * 32u;
            uint32_t src_l1 = reinterpret_cast<uint32_t>(&dense_row_major[bxw * 32]);
            noc_async_write(src_l1, dst, y_count * sizeof(uint16_t));
        }
        noc_async_write_barrier();
    };

    // Helper to pack a contiguous run of records into N_INFLIGHT-aligned
    // CB tiles (K_BATCH=32 records per tile, partial trailing tile padded
    // with zero records, full N_INFLIGHT batch always pushed).
    auto pack_and_push = [&](const V13Record* recs, uint32_t count) {
        uint32_t done = 0;
        bool first = true;
        while (first || done < count) {
            first = false;
            uint32_t left = (count > done) ? (count - done) : 0u;
            uint32_t to_take = (left > PUSH_BATCH) ? PUSH_BATCH : left;
            for (uint32_t i = 0; i < to_take; ++i) staging[i] = recs[done + i];

            DeviceZoneScopedN("V15A-PACK");
            for (uint32_t q = 0; q < N_INFLIGHT; ++q) {
                uint32_t k_off = q * K_BATCH;
                uint32_t k_used = 0;
                if (k_off < to_take) {
                    k_used = ((to_take - k_off) > K_BATCH) ? K_BATCH : (to_take - k_off);
                }
                cb_reserve_back(CB_OX, 1);
                cb_reserve_back(CB_OY, 1);
                {
                    DeviceZoneScopedN("V15A-PACK-FILL");
                    pack_ox_oy(&staging[k_off], k_used,
                               reinterpret_cast<uint16_t*>(get_write_ptr(CB_OX)),
                               reinterpret_cast<uint16_t*>(get_write_ptr(CB_OY)));
                }
                cb_push_back(CB_OX, 1);
                cb_push_back(CB_OY, 1);
            }
            done += to_take;
            if (count == 0u) break;
        }
    };

    auto push_sentinel = [&]() {
        for (uint32_t q = 0; q < N_INFLIGHT; ++q) {
            cb_reserve_back(CB_OX, 1);
            uint32_t* sent = reinterpret_cast<uint32_t*>(get_write_ptr(CB_OX));
            for (uint32_t i = 0; i < TILE_BF16_BYTES / 4; ++i) sent[i] = SENTINEL_U32;
            cb_push_back(CB_OX, 1);
        }
    };

    // ── Pass 1: BRISC single-pass scan + bucket with DRAM spill ─────────
    for (uint32_t j = 0; j < n_owned_tiles; ++j) {
        bucket_b_count[j] = 0;
#ifdef V15_SPILL_ENABLED
        spill_chunks_b[j] = 0;
#endif
    }

#ifdef V15_SPILL_ENABLED
    const InterleavedAddrGen<true> spill_gen = {
        .bank_base_address = spill_dram,
        .page_size         = spill_pgsz,
    };
    const uint32_t brisc_spill_base = my_core_id * n_owned_max * max_spill_chunks;

    auto flush_bucket_to_dram = [&](uint32_t local_idx) {
        uint32_t chunk = spill_chunks_b[local_idx];
        if (chunk >= max_spill_chunks) return;  // ran out of DRAM chunks — drop
        uint32_t slot = brisc_spill_base + local_idx * max_spill_chunks + chunk;
        uint64_t dst_base = spill_gen.get_noc_addr(slot);
        uint32_t l1_src_base = reinterpret_cast<uint32_t>(
            &bucket_b[local_idx * bucket_cap]);
        uint32_t total_size = bucket_cap * V13_RECORD_BYTES;
        // CHUNKED WRITE: split into ≤4 KB sub-writes (NOC_MAX_BURST_SIZE).
        // A single >80 KB noc_async_write triggers a kernel-side hang via
        // ncrisc_noc_fast_write_any_len internal state corruption. Chunked
        // writes stay within burst-size limit per call.
        constexpr uint32_t SUB_WRITE_SIZE = 4096u;
        uint32_t written = 0;
        while (written < total_size) {
            uint32_t to_write = (total_size - written < SUB_WRITE_SIZE)
                                ? (total_size - written) : SUB_WRITE_SIZE;
            noc_async_write(l1_src_base + written,
                            dst_base + (uint64_t)written, to_write);
            written += to_write;
        }
        noc_async_write_barrier();
        spill_chunks_b[local_idx] = chunk + 1u;
        bucket_b_count[local_idx] = 0u;
    };
#endif  // V15_SPILL_ENABLED

    {
        DeviceZoneScopedN("V15A-BUCKET");
        for (uint32_t w = 0; w < brisc_writer_end; ++w) {
            const uint32_t* hdr32 = reinterpret_cast<const uint32_t*>(
                inbound_hdrs_raw + (size_t)w * V13_PAGE_HDR_BYTES);
            uint32_t cnt_n = hdr32[0];
            uint32_t cnt_b = hdr32[8];
            if (cnt_n > half_cap) cnt_n = half_cap;
            if (cnt_b > half_cap) cnt_b = half_cap;
            if ((cnt_n + cnt_b) == 0u) continue;

            uint64_t page_base = rgen.get_noc_addr(w * nc_all + my_core_id);
            uint64_t n_region_base = page_base + (uint64_t)V13_PAGE_HDR_BYTES;
            uint64_t b_region_base = n_region_base + (uint64_t)half_cap * V13_RECORD_BYTES;

            auto read_and_bucket = [&](uint64_t region_base, uint32_t count) {
                uint32_t done = 0;
                while (done < count) {
                    uint32_t to_read = count - done;
                    if (to_read > MAX_READ_RECORDS) to_read = MAX_READ_RECORDS;
                    uint64_t src = region_base + (uint64_t)done * V13_RECORD_BYTES;
                    uint32_t read_bytes = to_read * V13_RECORD_BYTES;
                    read_bytes = (read_bytes + 31u) & ~31u;
                    noc_async_read(src,
                                   reinterpret_cast<uint32_t>(inbound_buf),
                                   read_bytes);
                    noc_async_read_barrier();
                    for (uint32_t i = 0; i < to_read; ++i) {
                        const V13Record& r = inbound_buf[i];
                        if (r.tile_id == 0xFFFFu) continue;
                        uint32_t local_idx = owned_lookup[r.tile_id];
                        if (local_idx == 0xFFFFu) continue;
                        uint32_t c = bucket_b_count[local_idx];
                        if (c >= bucket_cap) {
#ifdef V15_SPILL_ENABLED
                            flush_bucket_to_dram(local_idx);
                            c = 0u;
#else
                            continue;
#endif
                        }
                        bucket_b[local_idx * bucket_cap + c] = r;
                        bucket_b_count[local_idx] = c + 1u;
                    }
                    done += to_read;
                }
            };
            if (cnt_n > 0u) read_and_bucket(n_region_base, cnt_n);
            if (cnt_b > 0u) read_and_bucket(b_region_base, cnt_b);
        }
    }

    // ── DEBUG: dump bucket stats AND first record of each bucket ─────────
    // Stats layout: 256 B per core in DRAM:
    //   [0]   = n_owned_tiles
    //   [1]   = bucket_cap
    //   [2]   = total records bucketed
    //   [3+t] = bucket_b_count[t] for t in [0..n_owned)
    //   [16+t] = tile_pushes[t] (written in pass 2)
    //   [24+t*4..27+t*4] = first record of bucket[t]: bxl_local|byl_local|tile_id|pad, ox[0:1], ox[2:3], oy[0:1]
    {
        const InterleavedAddrGen<true> sgen = {
            .bank_base_address = stats_dram,
            .page_size = 256u,
        };
        uint32_t total_bucketed = 0u;
        for (uint32_t t = 0; t < n_owned_tiles; ++t) total_bucketed += bucket_b_count[t];
        uint64_t base_addr = sgen.get_noc_addr(my_core_id);
        noc_inline_dw_write(base_addr + 0u,  n_owned_tiles);
        noc_inline_dw_write(base_addr + 4u,  bucket_cap);
        noc_inline_dw_write(base_addr + 8u,  total_bucketed);
        for (uint32_t t = 0; t < n_owned_tiles; ++t) {
            noc_inline_dw_write(base_addr + 12u + t * 4u, bucket_b_count[t]);
        }
        // First record of each bucket — verify record fields.
        // V13Record layout (uint32 view):
        //   w[0] = bxl_local | byl_local (each int16)
        //   w[1] = tile_id | pad
        //   w[2..5] = overlap_x[0..7]
        //   w[6..9] = overlap_y[0..7]
        for (uint32_t t = 0; t < n_owned_tiles && t < 4u; ++t) {
            if (bucket_b_count[t] > 0u) {
                const V13Record* r0 = &bucket_b[t * bucket_cap];
                const uint32_t* w = reinterpret_cast<const uint32_t*>(r0);
                noc_inline_dw_write(base_addr + 96u + t * 16u + 0u,  w[0]);  // bxl|byl
                noc_inline_dw_write(base_addr + 96u + t * 16u + 4u,  w[1]);  // tile_id|pad
                noc_inline_dw_write(base_addr + 96u + t * 16u + 8u,  w[2]);  // ox[0:1]
                noc_inline_dw_write(base_addr + 96u + t * 16u + 12u, w[6]);  // oy[0:1]
            }
        }
    }

    // ── Wait for NCRISC to finish its bucketing pass ─────────────────────
    {
        DeviceZoneScopedN("V15A-WAIT-NCRISC");
        volatile tt_l1_ptr uint32_t* ncrisc_done_sem =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ncrisc_done_sem_id));
        noc_semaphore_wait(ncrisc_done_sem, 1u);
    }

    // ── Pass 2: drain bucket_b[t] + bucket_n[t] into ONE CB pair ─────────
    uint32_t total_tiles_pushed = 0;
    for (uint32_t t = 0; t < n_owned_tiles; ++t) {
        DeviceZoneScopedN("V15A-TILE");
        const uint32_t target_tile_id = get_arg_val<uint32_t>(23u + t);
        uint32_t tile_pushes_before = total_tiles_pushed;

        {
            DeviceZoneScopedN("V15A-DRAIN");
            // 1. Drain L1 bucket_b residual (any records still in L1 after spill).
            uint32_t count_b = bucket_b_count[t];
            uint32_t batches_b = (count_b == 0u) ? 1u : ((count_b + PUSH_BATCH - 1u) / PUSH_BATCH);
            total_tiles_pushed += batches_b * N_INFLIGHT;
            pack_and_push(&bucket_b[t * bucket_cap], count_b);

            // 2. Drain any DRAM spill chunks for tile t — read each chunk back
#ifdef V15_SPILL_ENABLED
            // into bucket_b[t]'s L1 slot (residual already consumed) and pack.
            uint32_t n_chunks_b = spill_chunks_b[t];
            for (uint32_t chunk = 0; chunk < n_chunks_b; ++chunk) {
                uint32_t slot = brisc_spill_base + t * max_spill_chunks + chunk;
                uint64_t src_base = spill_gen.get_noc_addr(slot);
                uint32_t l1_dst_base = reinterpret_cast<uint32_t>(
                    &bucket_b[t * bucket_cap]);
                uint32_t total_size = bucket_cap * V13_RECORD_BYTES;
                constexpr uint32_t SUB_READ_SIZE = 4096u;
                uint32_t readn = 0;
                while (readn < total_size) {
                    uint32_t to_read = (total_size - readn < SUB_READ_SIZE)
                                       ? (total_size - readn) : SUB_READ_SIZE;
                    noc_async_read(src_base + (uint64_t)readn,
                                   l1_dst_base + readn, to_read);
                    readn += to_read;
                }
                noc_async_read_barrier();
                uint32_t batches_sp = (bucket_cap + PUSH_BATCH - 1u) / PUSH_BATCH;
                total_tiles_pushed += batches_sp * N_INFLIGHT;
                pack_and_push(&bucket_b[t * bucket_cap], bucket_cap);
            }
#endif  // V15_SPILL_ENABLED

            // 3. Same for NCRISC's bucket_n + its spill chunks (dual-RISC only).
            if (!brisc_only) {
                uint32_t count_n = bucket_n_count[t];
                uint32_t batches_n = (count_n == 0u) ? 1u : ((count_n + PUSH_BATCH - 1u) / PUSH_BATCH);
                total_tiles_pushed += batches_n * N_INFLIGHT;
                pack_and_push(&bucket_n[t * bucket_cap], count_n);
#ifdef V15_SPILL_ENABLED
                // NCRISC spill chunks
                const uint32_t ncrisc_spill_base =
                    (nc_all + my_core_id) * n_owned_max * max_spill_chunks;
                uint32_t* ncrisc_spill_chunks_ptr =
                    reinterpret_cast<uint32_t*>(
                        reinterpret_cast<uint8_t*>(bucket_n_count)
                        + ((n_owned_tiles * sizeof(uint32_t) + 63u) & ~63u));
                uint32_t n_chunks_n = ncrisc_spill_chunks_ptr[t];
                for (uint32_t chunk = 0; chunk < n_chunks_n; ++chunk) {
                    uint32_t slot = ncrisc_spill_base + t * max_spill_chunks + chunk;
                    uint64_t src_base = spill_gen.get_noc_addr(slot);
                    uint32_t l1_dst_base = reinterpret_cast<uint32_t>(
                        &bucket_n[t * bucket_cap]);
                    uint32_t total_size = bucket_cap * V13_RECORD_BYTES;
                    constexpr uint32_t SUB_READ_SIZE = 4096u;
                    uint32_t readn = 0;
                    while (readn < total_size) {
                        uint32_t to_read = (total_size - readn < SUB_READ_SIZE)
                                           ? (total_size - readn) : SUB_READ_SIZE;
                        noc_async_read(src_base + (uint64_t)readn,
                                       l1_dst_base + readn, to_read);
                        readn += to_read;
                    }
                    noc_async_read_barrier();
                    uint32_t batches_sp = (bucket_cap + PUSH_BATCH - 1u) / PUSH_BATCH;
                    total_tiles_pushed += batches_sp * N_INFLIGHT;
                    pack_and_push(&bucket_n[t * bucket_cap], bucket_cap);
                }
#endif  // V15_SPILL_ENABLED
            }
            total_tiles_pushed += N_INFLIGHT;  // sentinel batch
            push_sentinel();
        }

        // DEBUG: write per-tile push count to stats DRAM (slot 16+t).
        {
            const InterleavedAddrGen<true> sgen = {
                .bank_base_address = stats_dram,
                .page_size = 256u,
            };
            uint64_t base_addr = sgen.get_noc_addr(my_core_id);
            noc_inline_dw_write(base_addr + 64u + t * 4u,
                                 total_tiles_pushed - tile_pushes_before);
        }

        {
            DeviceZoneScopedN("V15A-WAIT-DENSE");
            cb_wait_front(CB_DENSE, 1);
        }
        {
            DeviceZoneScopedN("V15A-DENSITY-WRITE");
            write_density_tile(target_tile_id);
            cb_pop_front(CB_DENSE, 1);
        }
    }
}
