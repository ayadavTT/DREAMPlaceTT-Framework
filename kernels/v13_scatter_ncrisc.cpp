// SPDX-License-Identifier: Apache-2.0
//
// V13_fpu Phase 1.2 — NCRISC scatter kernel.
//
// Twin of v13_scatter_brisc.cpp for cells [0..512). Also responsible for:
//   * Loading tile_to_core[] from DRAM at startup (signals tables_ready_sem
//     when done).
//   * Publishing per-tile CB read pointers via ScatterShared + data_ready_sem
//     so BRISC can read the same SFPU outputs.
//
// Routes records into route_buf[my_writer_id, recv] in the NCRISC half
// (first half of the tuple area, immediately after the 64-byte header).
//
// Header: NCRISC writes cnt_n at offset 0; BRISC writes cnt_b at offset 32.
//
// Runtime args (mirror v11_scatter_dm.cpp arg layout):
//   0: tile_map_dram
//   1: tile_map_pgsz
//   2: tile_map_bytes
//   3: my_core_id
//   4: nc_all
//   5: M_tiles
//   6: N_tiles
//   7: nbx
//   8: nby
//   9: n_tiles
//  10: route_dram
//  11: route_pgsz
//  12: records_cap
//  13: my_writer_id
//  14: data_ready_sem_id
//  15: brisc_done_sem_id
//  16: shared_state_off
//  17: brisc_state_off       (unused on NCRISC)
//  18: tables_ready_sem_id
//  19: overflow_dram

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif

constexpr uint32_t V13_MAX_OVERLAP = 8u;
constexpr uint32_t TILE_ELEMS      = 1024u;
constexpr uint32_t MAX_IN_FLIGHT   = 16u;
constexpr uint32_t V13_PAGE_HDR_BYTES = 64u;
constexpr uint32_t V13_RECORD_BYTES   = 40u;
constexpr uint32_t CB_SCRATCH = tt::CBIndex::c_24;
constexpr uint32_t CB_BXL     = 4u;
constexpr uint32_t CB_BYL     = 5u;
constexpr uint32_t CB_OX_BASE = 6u;
constexpr uint32_t CB_OY_BASE = 14u;

struct V13Record {
    int16_t  bxl_local;
    int16_t  byl_local;
    uint16_t tile_id;       // global tile index (= tx * N_tiles + ty)
    uint16_t pad;
    uint16_t overlap_x[8];
    uint16_t overlap_y[8];
};
static_assert(sizeof(V13Record) == 40, "V13Record must be 40 bytes");

struct ScatterShared {
    uint32_t bxl_ptr;
    uint32_t byl_ptr;
    uint32_t ox_ptr[V13_MAX_OVERLAP];
    uint32_t oy_ptr[V13_MAX_OVERLAP];
};

// fp32 → bf16 with round-to-nearest-even. See sibling note in v13_scatter_brisc.cpp.
static inline uint16_t fp32_to_bf16_trunc(float f) {
    uint32_t bits;
    __builtin_memcpy(&bits, &f, 4);
    uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7FFFu + lsb;
    return (uint16_t)(bits >> 16);
}

void kernel_main() {
    const uint32_t tile_map_dram      = get_arg_val<uint32_t>(0);
    const uint32_t tile_map_pgsz      = get_arg_val<uint32_t>(1);
    const uint32_t tile_map_bytes     = get_arg_val<uint32_t>(2);
    (void)get_arg_val<uint32_t>(3);   // my_core_id (unused, see arg 13 my_writer_id)
    const uint32_t nc_all             = get_arg_val<uint32_t>(4);
    const uint32_t M_tiles            = get_arg_val<uint32_t>(5);
    const uint32_t N_tiles            = get_arg_val<uint32_t>(6);
    const uint32_t nbx                = get_arg_val<uint32_t>(7);
    const uint32_t nby                = get_arg_val<uint32_t>(8);
    const uint32_t n_tiles            = get_arg_val<uint32_t>(9);
    const uint32_t route_dram         = get_arg_val<uint32_t>(10);
    const uint32_t route_pgsz         = get_arg_val<uint32_t>(11);
    const uint32_t records_cap        = get_arg_val<uint32_t>(12);
    const uint32_t my_writer_id       = get_arg_val<uint32_t>(13);
    const uint32_t data_ready_sem_id  = get_arg_val<uint32_t>(14);
    const uint32_t brisc_done_sem_id  = get_arg_val<uint32_t>(15);
    const uint32_t shared_state_off   = get_arg_val<uint32_t>(16);
    (void)get_arg_val<uint32_t>(17);  // brisc_state_off unused on NCRISC
    const uint32_t tables_ready_sem_id= get_arg_val<uint32_t>(18);
    const uint32_t overflow_dram      = get_arg_val<uint32_t>(19);

    (void)nbx; (void)nby; (void)M_tiles;

    const uint32_t ncrisc_cap_records = records_cap / 2u;

    // ── L1 layout (NCRISC region starts at offset 0) ─────────────────────
    uint8_t* base = reinterpret_cast<uint8_t*>(get_write_ptr(CB_SCRATCH));
    uint32_t off = 0u;

    uint16_t* tile_to_core = reinterpret_cast<uint16_t*>(base + off);
    off += tile_map_bytes;
    off = (off + 7u) & ~7u;

    V13Record* staging = reinterpret_cast<V13Record*>(base + off);
    off += nc_all * MAX_IN_FLIGHT * V13_RECORD_BYTES;
    off = (off + 3u) & ~3u;
    uint32_t* staging_count = reinterpret_cast<uint32_t*>(base + off);
    off += nc_all * sizeof(uint32_t);
    uint32_t* dram_offset_records = reinterpret_cast<uint32_t*>(base + off);
    off += nc_all * sizeof(uint32_t);
    off = (off + 63u) & ~63u;
    uint32_t* hdr_scratch = reinterpret_cast<uint32_t*>(base + off);
    off += V13_PAGE_HDR_BYTES;
    off = (off + 31u) & ~31u;
    // Local overflow flags: 1 byte per receiver, padded to 128 B (32-aligned).
    // NCRISC writes its 128 B half to DRAM at page_base + 128 (BRISC at + 0).
    uint8_t* ov_buf = reinterpret_cast<uint8_t*>(base + off);
    off += 128u;

    // Shared per-tile pointer struct (BRISC reads via shared_state_off).
    volatile ScatterShared* shared =
        reinterpret_cast<volatile ScatterShared*>(base + shared_state_off);

    // ── Step 0: load tile_to_core[] from DRAM ────────────────────────────
    {
        const InterleavedAddrGen<true> mgen = {
            .bank_base_address = tile_map_dram,
            .page_size         = tile_map_pgsz,
        };
        noc_async_read(mgen.get_noc_addr(0),
                       reinterpret_cast<uint32_t>(tile_to_core),
                       tile_map_bytes);
        noc_async_read_barrier();
    }

    // Signal BRISC that tile_to_core[] is loaded.
    {
        volatile tt_l1_ptr uint32_t* tr_sem =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(tables_ready_sem_id));
        *tr_sem = 1u;
        asm volatile("" ::: "memory");
    }

    // Init per-receiver state.
    for (uint32_t i = 0; i < nc_all; ++i) {
        staging_count[i]       = 0u;
        dram_offset_records[i] = 0u;
    }
    for (uint32_t i = 0; i < 128u; ++i) ov_buf[i] = 0u;

    // ── Address generators ───────────────────────────────────────────────
    const InterleavedAddrGen<true> rgen = {
        .bank_base_address = route_dram,
        .page_size         = route_pgsz,
    };
    // Overflow page is 256 B per writer: BRISC half [0..128), NCRISC half [128..256).
    const InterleavedAddrGen<true> ogen = {
        .bank_base_address = overflow_dram,
        .page_size         = 256u,
    };

    auto set_overflow = [&](uint32_t recv) {
        // Set locally; flush to DRAM once at end of scatter.
        if (recv < 128u) ov_buf[recv] = 1u;
    };

    auto flush_recv = [&](uint32_t recv, uint32_t cnt) {
        if (cnt == 0u) return;
        uint32_t already = dram_offset_records[recv];

        if (already >= ncrisc_cap_records) {
            set_overflow(recv);
            staging_count[recv] = 0u;
            return;
        }
        uint32_t to_write = cnt;
        if (already + to_write > ncrisc_cap_records) {
            set_overflow(recv);
            to_write = ncrisc_cap_records - already;
        }
        if (to_write == 0u) {
            staging_count[recv] = 0u;
            return;
        }

        uint32_t cnt_padded = (to_write + 7u) & ~7u;
        if (cnt_padded > MAX_IN_FLIGHT) cnt_padded = MAX_IN_FLIGHT;
        if (already + cnt_padded > ncrisc_cap_records) {
            cnt_padded = ncrisc_cap_records - already;
        }
        for (uint32_t i = to_write; i < cnt_padded; ++i) {
            V13Record& pad_rec = staging[recv * MAX_IN_FLIGHT + i];
            pad_rec.bxl_local = 0; pad_rec.byl_local = 0;
            pad_rec.tile_id = 0; pad_rec.pad = 0;
            for (uint32_t j = 0; j < 8; ++j) {
                pad_rec.overlap_x[j] = 0u;
                pad_rec.overlap_y[j] = 0u;
            }
        }

        uint64_t page_base = rgen.get_noc_addr(my_writer_id * nc_all + recv);
        uint64_t dst = page_base + (uint64_t)V13_PAGE_HDR_BYTES
                     + (uint64_t)already * V13_RECORD_BYTES;
        uint32_t src_l1 = reinterpret_cast<uint32_t>(
            &staging[recv * MAX_IN_FLIGHT]);
        noc_async_write(src_l1, dst, cnt_padded * V13_RECORD_BYTES);

        dram_offset_records[recv] = already + cnt_padded;
        staging_count[recv]       = 0u;
    };

    // ── Per-cell-tile loop ────────────────────────────────────────────────
    volatile tt_l1_ptr uint32_t* dr_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(data_ready_sem_id));
    volatile tt_l1_ptr uint32_t* bd_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(brisc_done_sem_id));

    for (uint32_t t = 0; t < n_tiles; ++t) {
        cb_wait_front(CB_BXL, 1);
        cb_wait_front(CB_BYL, 1);
        for (uint32_t j = 0; j < V13_MAX_OVERLAP; ++j) {
            cb_wait_front(CB_OX_BASE + j, 1);
            cb_wait_front(CB_OY_BASE + j, 1);
        }

        const float* bxl_data = reinterpret_cast<const float*>(get_read_ptr(CB_BXL));
        const float* byl_data = reinterpret_cast<const float*>(get_read_ptr(CB_BYL));
        const float* ox_data[V13_MAX_OVERLAP];
        const float* oy_data[V13_MAX_OVERLAP];
        for (uint32_t j = 0; j < V13_MAX_OVERLAP; ++j) {
            ox_data[j] = reinterpret_cast<const float*>(get_read_ptr(CB_OX_BASE + j));
            oy_data[j] = reinterpret_cast<const float*>(get_read_ptr(CB_OY_BASE + j));
        }

        // Publish read pointers for BRISC.
        shared->bxl_ptr = (uint32_t)bxl_data;
        shared->byl_ptr = (uint32_t)byl_data;
        for (uint32_t j = 0; j < V13_MAX_OVERLAP; ++j) {
            shared->ox_ptr[j] = (uint32_t)ox_data[j];
            shared->oy_ptr[j] = (uint32_t)oy_data[j];
        }
        asm volatile("" ::: "memory");
        *dr_sem = t + 1u;
        asm volatile("" ::: "memory");

        // NCRISC: cells [0..512).
        for (uint32_t ci = 0; ci < TILE_ELEMS / 2u; ++ci) {
            int bxl = (int)bxl_data[ci];
            int byl = (int)byl_data[ci];

            int tx_lo = bxl >> 5;
            int tx_hi = (bxl + 7) >> 5;
            int ty_lo = byl >> 5;
            int ty_hi = (byl + 7) >> 5;
            if (tx_hi < 0 || tx_lo >= (int)M_tiles) continue;
            if (ty_hi < 0 || ty_lo >= (int)N_tiles) continue;
            if (tx_lo < 0) tx_lo = 0;
            if (ty_lo < 0) ty_lo = 0;
            if (tx_hi >= (int)M_tiles) tx_hi = (int)M_tiles - 1;
            if (ty_hi >= (int)N_tiles) ty_hi = (int)N_tiles - 1;

            uint16_t ox_bf[8], oy_bf[8];
            for (uint32_t j = 0; j < 8; ++j) {
                ox_bf[j] = fp32_to_bf16_trunc(ox_data[j][ci]);
                oy_bf[j] = fp32_to_bf16_trunc(oy_data[j][ci]);
            }

            for (int tx = tx_lo; tx <= tx_hi; ++tx) {
                uint32_t tx_row = (uint32_t)tx * N_tiles;
                for (int ty = ty_lo; ty <= ty_hi; ++ty) {
                    uint32_t tile_id = tx_row + (uint32_t)ty;
                    uint32_t owner = (uint32_t)tile_to_core[tile_id];
                    if (owner >= nc_all) continue;

                    uint32_t cnt = staging_count[owner];
                    V13Record& rec = staging[owner * MAX_IN_FLIGHT + cnt];
                    rec.bxl_local = (int16_t)(bxl - tx * 32);
                    rec.byl_local = (int16_t)(byl - ty * 32);
                    rec.tile_id   = (uint16_t)tile_id;
                    rec.pad       = 0;
                    for (uint32_t j = 0; j < 8; ++j) rec.overlap_x[j] = ox_bf[j];
                    for (uint32_t j = 0; j < 8; ++j) rec.overlap_y[j] = oy_bf[j];

                    cnt++;
                    staging_count[owner] = cnt;
                    if (cnt >= MAX_IN_FLIGHT) {
                        flush_recv(owner, MAX_IN_FLIGHT);
                        noc_async_write_barrier();
                    }
                }
            }
        }

        // Wait for BRISC to finish processing tile t.
        noc_semaphore_wait_min(bd_sem, t + 1u);

        cb_pop_front(CB_BXL, 1);
        cb_pop_front(CB_BYL, 1);
        for (uint32_t j = 0; j < V13_MAX_OVERLAP; ++j) {
            cb_pop_front(CB_OX_BASE + j, 1);
            cb_pop_front(CB_OY_BASE + j, 1);
        }
    }

    // Final flush.
    for (uint32_t r = 0; r < nc_all; ++r) {
        uint32_t cnt = staging_count[r];
        if (cnt == 0u) continue;
        flush_recv(r, cnt);
    }
    noc_async_write_barrier();

    // Embed per-writer total record count into ov_buf bytes [120..123].
    // Read by host for "no records dropped during scatter" audit.
    {
        uint32_t total = 0u;
        for (uint32_t r = 0; r < nc_all; ++r) total += dram_offset_records[r];
        volatile tt_l1_ptr uint32_t* p =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(&ov_buf[120]);
        *p = total;
    }

    // ov_flush: write NCRISC's 128 B overflow half to DRAM at page_base + 128.
    {
        uint64_t ov_addr = ogen.get_noc_addr(my_writer_id) + 128u;
        noc_async_write(reinterpret_cast<uint32_t>(ov_buf), ov_addr, 128u);
        noc_async_write_barrier();
    }

    // Header write: NCRISC owns bytes [0..32), with cnt_n at byte 0.
    for (uint32_t i = 1; i < 8u; ++i) hdr_scratch[i] = 0u;
    for (uint32_t r = 0; r < nc_all; ++r) {
        hdr_scratch[0] = dram_offset_records[r];  // cnt_n
        uint64_t page_base = rgen.get_noc_addr(my_writer_id * nc_all + r);
        noc_async_write(reinterpret_cast<uint32_t>(hdr_scratch),
                        page_base, 32u);
        noc_async_write_barrier();
    }
}
