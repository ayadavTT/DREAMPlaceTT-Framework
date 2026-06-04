// SPDX-License-Identifier: Apache-2.0
//
// V15 — NCRISC parallel BUCKETING reader (no CB pushes).
//
// NCRISC scans writer pages [brisc_writer_end..nc_all) ONCE, filters padding
// records, and buckets real records into bucket_n[][] in L1. When done, it
// signals BRISC via a semaphore. BRISC handles all CB pushes in pass 2,
// draining bucket_b[] + bucket_n[] sequentially per owned tile.

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t CB_SCRATCH = tt::CBIndex::c_24;

constexpr uint32_t V13_RECORD_BYTES   = 40u;
constexpr uint32_t V13_PAGE_HDR_BYTES = 64u;
constexpr uint32_t MAX_READ_RECORDS   = 64u;

struct V13Record {
    int16_t  bxl_local;
    int16_t  byl_local;
    uint16_t tile_id;
    uint16_t pad;
    uint16_t overlap_x[8];
    uint16_t overlap_y[8];
};
static_assert(sizeof(V13Record) == 40, "V13Record must be 40 bytes");

void kernel_main() {
    const uint32_t my_core_id         = get_arg_val<uint32_t>(0);
    const uint32_t nc_all             = get_arg_val<uint32_t>(1);
    const uint32_t M_tiles            = get_arg_val<uint32_t>(2);
    const uint32_t N_tiles            = get_arg_val<uint32_t>(3);
    const uint32_t route_dram         = get_arg_val<uint32_t>(4);
    const uint32_t route_pgsz         = get_arg_val<uint32_t>(5);
    const uint32_t records_cap        = get_arg_val<uint32_t>(6);
    const uint32_t n_owned_tiles      = get_arg_val<uint32_t>(7);
    const uint32_t ncrisc_scratch_off = get_arg_val<uint32_t>(8);
    const uint32_t owned_lookup_dram  = get_arg_val<uint32_t>(9);
    const uint32_t owned_lookup_pgsz  = get_arg_val<uint32_t>(10);
    const uint32_t bucket_cap         = get_arg_val<uint32_t>(11);
    const uint32_t ncrisc_done_sem_id = get_arg_val<uint32_t>(12);
    const uint32_t brisc_only         = get_arg_val<uint32_t>(13);
    const uint32_t spill_dram         = get_arg_val<uint32_t>(14);
    const uint32_t spill_pgsz         = get_arg_val<uint32_t>(15);
    const uint32_t max_spill_chunks   = get_arg_val<uint32_t>(16);
    const uint32_t n_owned_max        = get_arg_val<uint32_t>(17);

    (void)M_tiles; (void)N_tiles;

    // BRISC-only mode: NCRISC has nothing to bucket; just signal done
    // immediately so BRISC's wait succeeds.
    if (brisc_only) {
        volatile tt_l1_ptr uint32_t* sem =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(ncrisc_done_sem_id));
        *sem = 1u;
        asm volatile("" ::: "memory");
        return;
    }

    const uint32_t total_tiles      = M_tiles * N_tiles;
    const uint32_t half_cap         = records_cap / 2u;
    const uint32_t brisc_writer_end = (nc_all + 1u) / 2u;

    uint8_t* base = reinterpret_cast<uint8_t*>(get_write_ptr(CB_SCRATCH));
    uint32_t off = ncrisc_scratch_off;

    uint16_t* owned_lookup_n = reinterpret_cast<uint16_t*>(base + off);
    off += total_tiles * sizeof(uint16_t);
    off = (off + 63u) & ~63u;

    uint8_t* inbound_hdrs_raw_n = reinterpret_cast<uint8_t*>(base + off);
    off += nc_all * V13_PAGE_HDR_BYTES;
    off = (off + 63u) & ~63u;

    V13Record* inbound_buf_n = reinterpret_cast<V13Record*>(base + off);
    off += MAX_READ_RECORDS * V13_RECORD_BYTES;
    off = (off + 63u) & ~63u;

    V13Record* bucket_n = reinterpret_cast<V13Record*>(base + off);
    off += n_owned_tiles * bucket_cap * V13_RECORD_BYTES;
    off = (off + 63u) & ~63u;
    uint32_t* bucket_n_count = reinterpret_cast<uint32_t*>(base + off);
    off += n_owned_tiles * sizeof(uint32_t);
    off = (off + 63u) & ~63u;
    // DRAM spill chunk counter — must be at the offset BRISC expects:
    // immediately after bucket_n_count, 64-aligned. BRISC re-derives this
    // address from `bucket_n_count + n_owned*4 + align`.
    uint32_t* spill_chunks_n = reinterpret_cast<uint32_t*>(base + off);
    off += n_owned_tiles * sizeof(uint32_t);
    off = (off + 63u) & ~63u;

    // Load owned_lookup (private copy).
    {
        const InterleavedAddrGen<true> lgen = {
            .bank_base_address = owned_lookup_dram,
            .page_size         = owned_lookup_pgsz,
        };
        noc_async_read(lgen.get_noc_addr(my_core_id),
                       reinterpret_cast<uint32_t>(owned_lookup_n),
                       total_tiles * sizeof(uint16_t));
        noc_async_read_barrier();
    }

    const InterleavedAddrGen<true> rgen = {
        .bank_base_address = route_dram,
        .page_size         = route_pgsz,
    };

    // NCRISC-half headers in one shot.
    for (uint32_t w = brisc_writer_end; w < nc_all; ++w) {
        uint64_t page = rgen.get_noc_addr(w * nc_all + my_core_id);
        uint32_t dst = reinterpret_cast<uint32_t>(
            inbound_hdrs_raw_n + (size_t)w * V13_PAGE_HDR_BYTES);
        noc_async_read(page, dst, V13_PAGE_HDR_BYTES);
    }
    noc_async_read_barrier();

    // ── Pass 1: single-pass scan + bucket NCRISC's writer half ───────────
    for (uint32_t j = 0; j < n_owned_tiles; ++j) {
        bucket_n_count[j] = 0;
#ifdef V15_SPILL_ENABLED
        spill_chunks_n[j] = 0;
#endif
    }

#ifdef V15_SPILL_ENABLED
    const InterleavedAddrGen<true> spill_gen = {
        .bank_base_address = spill_dram,
        .page_size         = spill_pgsz,
    };
    const uint32_t ncrisc_spill_base =
        (nc_all + my_core_id) * n_owned_max * max_spill_chunks;

    auto flush_bucket_n_to_dram = [&](uint32_t local_idx) {
        uint32_t chunk = spill_chunks_n[local_idx];
        if (chunk >= max_spill_chunks) return;
        uint32_t slot = ncrisc_spill_base + local_idx * max_spill_chunks + chunk;
        uint64_t dst_base = spill_gen.get_noc_addr(slot);
        uint32_t l1_src_base = reinterpret_cast<uint32_t>(
            &bucket_n[local_idx * bucket_cap]);
        uint32_t total_size = bucket_cap * V13_RECORD_BYTES;
        // CHUNKED WRITE: stay under NOC_MAX_BURST_SIZE per call.
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
        spill_chunks_n[local_idx] = chunk + 1u;
        bucket_n_count[local_idx] = 0u;
    };
#endif  // V15_SPILL_ENABLED

    for (uint32_t w = brisc_writer_end; w < nc_all; ++w) {
        const uint32_t* hdr32 = reinterpret_cast<const uint32_t*>(
            inbound_hdrs_raw_n + (size_t)w * V13_PAGE_HDR_BYTES);
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
                               reinterpret_cast<uint32_t>(inbound_buf_n),
                               read_bytes);
                noc_async_read_barrier();
                for (uint32_t i = 0; i < to_read; ++i) {
                    const V13Record& r = inbound_buf_n[i];
                    if (r.tile_id == 0xFFFFu) continue;
                    uint32_t local_idx = owned_lookup_n[r.tile_id];
                    if (local_idx == 0xFFFFu) continue;
                    uint32_t c = bucket_n_count[local_idx];
                    if (c >= bucket_cap) {
#ifdef V15_SPILL_ENABLED
                        flush_bucket_n_to_dram(local_idx);
                        c = 0u;
#else
                        continue;
#endif
                    }
                    bucket_n[local_idx * bucket_cap + c] = r;
                    bucket_n_count[local_idx] = c + 1u;
                }
                done += to_read;
            }
        };
        if (cnt_n > 0u) read_and_bucket(n_region_base, cnt_n);
        if (cnt_b > 0u) read_and_bucket(b_region_base, cnt_b);
    }

    // ── Signal BRISC: bucketing complete ─────────────────────────────────
    // Use a HARDWARE memory fence before setting the semaphore so that all
    // earlier L1 stores (bucket_n[] and bucket_n_count[]) are committed and
    // visible to BRISC before it sees the sem rise. A plain `asm volatile`
    // is only a COMPILER barrier — Tensix RISC-V can otherwise reorder the
    // semaphore write ahead of the bucket-array stores in its store buffer.
    {
        asm volatile("fence rw, rw" ::: "memory");
        volatile tt_l1_ptr uint32_t* sem =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(ncrisc_done_sem_id));
        *sem = 1u;
        asm volatile("fence rw, rw" ::: "memory");
    }
}
