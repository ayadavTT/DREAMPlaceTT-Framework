// SPDX-License-Identifier: Apache-2.0
//
// V13_fpu Phase 1.2 — BRISC scatter kernel.
//
// Reads fp32 (bxl, byl, overlap_x[8], overlap_y[8]) from CBs c_4..c_21
// (produced by v4_compute SFPU front-end), for cells [512..1024) of each
// cell-tile. For each cell, determines which 32x32 bin tiles its 8x8
// overlap footprint touches (at most 2x2 = 4 tiles), and emits ONE 40-byte
// record per affected tile to route_buf[my_writer_id, owner].
//
// Precision contract:
//   * SFPU outputs (bxl, byl, ox[8], oy[8]) read as fp32 from CBs.
//   * fp32 → bf16 cast happens HERE when building each record (single
//     truncation per overlap value).
//   * Bin-index computation uses fp32 inputs (no rounding error in
//     cell→bin assignment).
//
// Overflow detection:
//   * Per-page cap (V13_RECORDS_CAP/2 records per RISC) is enforced.
//   * If a flush would exceed the cap, the kernel sets a sticky byte in
//     route_overflow_buf[my_writer_id, receiver] BEFORE clamping. Host
//     audits these bytes after Finish() to detect any drops.
//
// Record layout (40 bytes, matches v13_accum_brisc):
//   int16  bxl_local       signed offset of overlap_x[0] within receiver tile
//   int16  byl_local
//   uint16 pad[2]
//   bfloat16 overlap_x[8]
//   bfloat16 overlap_y[8]
//
// route_buf page layout (per writer, per receiver):
//   [64 B header] [BRISC half: cap/2 × 40 B] [NCRISC half: cap/2 × 40 B]
//   Header: byte 0 = cnt_n (uint32), byte 32 = cnt_b (uint32)
//
// Sync pattern (mirrors V11):
//   * NCRISC loads tile_to_core[] from DRAM at startup, signals
//     tables_ready_sem when done.
//   * BRISC waits, then enters the per-cell-tile loop.
//   * Per cell-tile: BRISC reads (px,py,sx,sy) from DRAM, pushes to input
//     CBs. NCRISC waits, runs SFPU, publishes output CB pointers via
//     ScatterShared + data_ready_sem. BRISC reads SFPU outputs, runs scatter
//     for its cell half. brisc_done_sem signals NCRISC to safely pop CBs.
//
// Runtime args (numbered to match v11_scatter_b_dm.cpp where possible):
//   0..6:  reader args (px, py, sx, sy DRAM addrs, tile_pgsz, first_tile, n_tiles)
//   7:     tile_map_bytes
//   8:     my_core_id (unused on BRISC, here for arg-slot compatibility)
//   9:     nc_all
//  10,11:  M_tiles, N_tiles
//  12,13:  nbx, nby
//  14,15:  route_dram, route_pgsz
//  16:     records_cap         per-page cap (BRISC uses half)
//  17:     my_writer_id        = my_core_id (BRISC and NCRISC share writer slot)
//  18:     data_ready_sem_id
//  19:     brisc_done_sem_id
//  20:     shared_state_off
//  21:     brisc_state_off
//  22:     tables_ready_sem_id
//  23:     overflow_dram       DRAM base of route_overflow_buf (nc_all² bytes)

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif

constexpr uint32_t V13_MAX_OVERLAP = 8u;
constexpr uint32_t TILE_ELEMS      = 1024u;
constexpr uint32_t MAX_IN_FLIGHT   = 16u;       // V13 records per receiver staging
constexpr uint32_t V13_PAGE_HDR_BYTES = 64u;
constexpr uint32_t V13_RECORD_BYTES   = 40u;
constexpr uint32_t CB_SCRATCH = tt::CBIndex::c_24;
constexpr uint32_t CB_PX = 0u, CB_PY = 1u, CB_SX = 2u, CB_SY = 3u;

struct V13Record {
    int16_t  bxl_local;
    int16_t  byl_local;
    uint16_t tile_id;       // global tile index (= tx * N_tiles + ty)
    uint16_t pad;
    uint16_t overlap_x[8];  // bf16 raw bits
    uint16_t overlap_y[8];
};
static_assert(sizeof(V13Record) == 40, "V13Record must be 40 bytes");

struct ScatterShared {
    uint32_t bxl_ptr;
    uint32_t byl_ptr;
    uint32_t ox_ptr[V13_MAX_OVERLAP];
    uint32_t oy_ptr[V13_MAX_OVERLAP];
};

// fp32 → bf16 with round-to-nearest-even. Truncate-toward-zero (bits>>16)
// introduces a systematic mass deficit (~0.85% on adaptec1_512 data) that
// breaks DREAMPlace convergence at grid 512. RNE is unbiased; ox/oy here are
// always non-negative so we don't need the negative-rounding branch.
static inline uint16_t fp32_to_bf16_trunc(float f) {
    uint32_t bits;
    __builtin_memcpy(&bits, &f, 4);
    uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7FFFu + lsb;
    return (uint16_t)(bits >> 16);
}

void kernel_main() {
    const uint32_t addr_px            = get_arg_val<uint32_t>(0);
    const uint32_t addr_py            = get_arg_val<uint32_t>(1);
    const uint32_t addr_sx            = get_arg_val<uint32_t>(2);
    const uint32_t addr_sy            = get_arg_val<uint32_t>(3);
    const uint32_t tile_pgsz          = get_arg_val<uint32_t>(4);
    const uint32_t first_tile         = get_arg_val<uint32_t>(5);
    const uint32_t n_tiles            = get_arg_val<uint32_t>(6);
    const uint32_t tile_map_bytes     = get_arg_val<uint32_t>(7);
    (void)get_arg_val<uint32_t>(8);   // my_core_id (unused on BRISC)
    const uint32_t nc_all             = get_arg_val<uint32_t>(9);
    const uint32_t M_tiles            = get_arg_val<uint32_t>(10);
    const uint32_t N_tiles            = get_arg_val<uint32_t>(11);
    const uint32_t nbx                = get_arg_val<uint32_t>(12);
    const uint32_t nby                = get_arg_val<uint32_t>(13);
    const uint32_t route_dram         = get_arg_val<uint32_t>(14);
    const uint32_t route_pgsz         = get_arg_val<uint32_t>(15);
    const uint32_t records_cap        = get_arg_val<uint32_t>(16);
    const uint32_t my_writer_id       = get_arg_val<uint32_t>(17);
    const uint32_t data_ready_sem_id  = get_arg_val<uint32_t>(18);
    const uint32_t brisc_done_sem_id  = get_arg_val<uint32_t>(19);
    const uint32_t shared_state_off   = get_arg_val<uint32_t>(20);
    const uint32_t brisc_state_off    = get_arg_val<uint32_t>(21);
    const uint32_t tables_ready_sem_id= get_arg_val<uint32_t>(22);
    const uint32_t overflow_dram      = get_arg_val<uint32_t>(23);

    (void)nbx; (void)nby; (void)M_tiles;

    const uint32_t brisc_cap_records = records_cap / 2u;

    // ── L1 layout (BRISC region) ──────────────────────────────────────────
    uint8_t* base = reinterpret_cast<uint8_t*>(get_write_ptr(CB_SCRATCH));
    volatile ScatterShared* shared =
        reinterpret_cast<volatile ScatterShared*>(base + shared_state_off);

    uint32_t off = brisc_state_off;
    V13Record* staging = reinterpret_cast<V13Record*>(base + off);
    off += nc_all * MAX_IN_FLIGHT * V13_RECORD_BYTES;
    off = (off + 3u) & ~3u;
    uint32_t* staging_count = reinterpret_cast<uint32_t*>(base + off);
    off += nc_all * sizeof(uint32_t);
    uint32_t* dram_offset_records = reinterpret_cast<uint32_t*>(base + off);
    off += nc_all * sizeof(uint32_t);
    off = (off + 63u) & ~63u;
    uint32_t* hdr_scratch = reinterpret_cast<uint32_t*>(base + off);
    off += 32u;  // hdr_scratch is 32B
    off = (off + 31u) & ~31u;
    // Local overflow flags: 1 byte per receiver, padded to 128 B (32-aligned).
    // BRISC OWNS bytes [0..nc_all). Written to DRAM page_base + 0 in one go at
    // end of scatter. The previous "1-byte unaligned NOC write per event" path
    // was silently dropped on Blackhole (sub-32B unaligned writes are NO-OPs),
    // which caused V11-style silent contribution drops at iter 0 / hot tiles.
    uint8_t* ov_buf = reinterpret_cast<uint8_t*>(base + off);
    off += 128u;  // 32-aligned, holds 110 bytes

    // tile_to_core[] is loaded by NCRISC at offset 0 in CB_SCRATCH.
    const uint16_t* tile_to_core = reinterpret_cast<const uint16_t*>(base);

    // Wait for NCRISC to finish loading the tile_to_core[] table.
    {
        volatile tt_l1_ptr uint32_t* tr_sem =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(tables_ready_sem_id));
        noc_semaphore_wait(tr_sem, 1u);
    }

    // Init per-receiver staging counts and DRAM offsets.
    for (uint32_t i = 0; i < nc_all; ++i) {
        staging_count[i]       = 0u;
        dram_offset_records[i] = 0u;
    }
    // Zero local overflow buffer.
    for (uint32_t i = 0; i < 128u; ++i) ov_buf[i] = 0u;

    // ── Address generators ───────────────────────────────────────────────
    const InterleavedAddrGen<true> rgen = {
        .bank_base_address = route_dram,
        .page_size         = route_pgsz,
    };
    // overflow_buf: one 256-byte page per writer. BRISC owns bytes [0..128),
    // NCRISC owns bytes [128..256). Host OR's the two halves to get final
    // (writer, receiver) overflow flags.
    const InterleavedAddrGen<true> ogen = {
        .bank_base_address = overflow_dram,
        .page_size         = 256u,
    };

    // BRISC's tuple region starts after NCRISC's half within the route_buf
    // tuple area (i.e., after header + cap/2 records).
    const uint32_t brisc_offset_bytes =
        V13_PAGE_HDR_BYTES + (records_cap / 2u) * V13_RECORD_BYTES;

    auto set_overflow = [&](uint32_t recv) {
        // Set bit locally; flushed to DRAM in one 128-byte 32-aligned write
        // at end of scatter (see "ov_flush" near return).
        if (recv < 128u) ov_buf[recv] = 1u;
    };

    auto flush_recv = [&](uint32_t recv, uint32_t cnt) {
        if (cnt == 0u) return;
        uint32_t already = dram_offset_records[recv];

        if (already >= brisc_cap_records) {
            set_overflow(recv);
            staging_count[recv] = 0u;
            return;
        }
        uint32_t to_write = cnt;
        if (already + to_write > brisc_cap_records) {
            set_overflow(recv);
            to_write = brisc_cap_records - already;
        }
        if (to_write == 0u) {
            staging_count[recv] = 0u;
            return;
        }

        // Round up to multiple of 8 records (320 B = 32-byte aligned writes).
        // Padding records have all-zero overlap → contribute nothing on the
        // accumulator side (each entry is multiplied by zero in the matmul).
        uint32_t cnt_padded = (to_write + 7u) & ~7u;
        if (cnt_padded > MAX_IN_FLIGHT) cnt_padded = MAX_IN_FLIGHT;
        if (already + cnt_padded > brisc_cap_records) {
            cnt_padded = brisc_cap_records - already;
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
        uint64_t dst = page_base + (uint64_t)brisc_offset_bytes
                     + (uint64_t)already * V13_RECORD_BYTES;
        uint32_t src_l1 = reinterpret_cast<uint32_t>(
            &staging[recv * MAX_IN_FLIGHT]);
        noc_async_write(src_l1, dst, cnt_padded * V13_RECORD_BYTES);

        dram_offset_records[recv] = already + cnt_padded;
        staging_count[recv]       = 0u;
    };

    // ── Reader+scatter interleaved per cell-tile ─────────────────────────
    volatile tt_l1_ptr uint32_t* dr_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(data_ready_sem_id));
    volatile tt_l1_ptr uint32_t* bd_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(brisc_done_sem_id));

    const InterleavedAddrGen<true> gen_px = {.bank_base_address = addr_px, .page_size = tile_pgsz};
    const InterleavedAddrGen<true> gen_py = {.bank_base_address = addr_py, .page_size = tile_pgsz};
    const InterleavedAddrGen<true> gen_sx = {.bank_base_address = addr_sx, .page_size = tile_pgsz};
    const InterleavedAddrGen<true> gen_sy = {.bank_base_address = addr_sy, .page_size = tile_pgsz};

    (void)tile_map_bytes;

    for (uint32_t t = 0; t <= n_tiles; ++t) {
        // Step A: push tile t into input CBs.
        if (t < n_tiles) {
            uint32_t page_id = first_tile + t;
            cb_reserve_back(CB_PX, 1);
            cb_reserve_back(CB_PY, 1);
            cb_reserve_back(CB_SX, 1);
            cb_reserve_back(CB_SY, 1);
            uint32_t l1_px = get_write_ptr(CB_PX);
            uint32_t l1_py = get_write_ptr(CB_PY);
            uint32_t l1_sx = get_write_ptr(CB_SX);
            uint32_t l1_sy = get_write_ptr(CB_SY);
            noc_async_read_page(page_id, gen_px, l1_px);
            noc_async_read_page(page_id, gen_py, l1_py);
            noc_async_read_page(page_id, gen_sx, l1_sx);
            noc_async_read_page(page_id, gen_sy, l1_sy);
            noc_async_read_barrier();
            cb_push_back(CB_PX, 1);
            cb_push_back(CB_PY, 1);
            cb_push_back(CB_SX, 1);
            cb_push_back(CB_SY, 1);
        }

        // Step B: scatter tile t-1.
        if (t == 0) continue;
        uint32_t st = t - 1u;
        noc_semaphore_wait_min(dr_sem, st + 1u);

        const float* bxl_data = reinterpret_cast<const float*>(shared->bxl_ptr);
        const float* byl_data = reinterpret_cast<const float*>(shared->byl_ptr);
        const float* ox_data[V13_MAX_OVERLAP];
        const float* oy_data[V13_MAX_OVERLAP];
        for (uint32_t j = 0; j < V13_MAX_OVERLAP; ++j) {
            ox_data[j] = reinterpret_cast<const float*>(shared->ox_ptr[j]);
            oy_data[j] = reinterpret_cast<const float*>(shared->oy_ptr[j]);
        }

        // BRISC: cells [512..1024).
        for (uint32_t ci = TILE_ELEMS / 2u; ci < TILE_ELEMS; ++ci) {
            int bxl = (int)bxl_data[ci];
            int byl = (int)byl_data[ci];

            // Tile range (a cell's 8-wide footprint can straddle 1-2 tiles
            // per axis, so 1-4 tiles total).
            int tx_lo = bxl >> 5;
            int tx_hi = (bxl + 7) >> 5;
            int ty_lo = byl >> 5;
            int ty_hi = (byl + 7) >> 5;
            // Skip cells whose footprint is entirely outside the grid.
            if (tx_hi < 0 || tx_lo >= (int)M_tiles) continue;
            if (ty_hi < 0 || ty_lo >= (int)N_tiles) continue;
            // Clamp tile indices to valid range.
            if (tx_lo < 0) tx_lo = 0;
            if (ty_lo < 0) ty_lo = 0;
            if (tx_hi >= (int)M_tiles) tx_hi = (int)M_tiles - 1;
            if (ty_hi >= (int)N_tiles) ty_hi = (int)N_tiles - 1;

            // bf16-cast overlap arrays once per cell.
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

        // Signal NCRISC: BRISC done reading tile `st`'s CB data.
        *bd_sem = st + 1u;
        asm volatile("" ::: "memory");
    }

    // Final flush of all partial buffers.
    for (uint32_t r = 0; r < nc_all; ++r) {
        uint32_t cnt = staging_count[r];
        if (cnt == 0u) continue;
        flush_recv(r, cnt);
    }
    noc_async_write_barrier();

    // Embed per-writer total record count into ov_buf bytes [120..123].
    // Used by host audit to verify "no records dropped during scatter".
    // sum(dram_offset_records[r]) = total records this RISC wrote to route_buf.
    {
        uint32_t total = 0u;
        for (uint32_t r = 0; r < nc_all; ++r) total += dram_offset_records[r];
        volatile tt_l1_ptr uint32_t* p =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(&ov_buf[120]);
        *p = total;
    }

    // ov_flush: write 128-byte local overflow buffer to DRAM (BRISC region).
    {
        uint64_t ov_addr = ogen.get_noc_addr(my_writer_id) + 0u;
        noc_async_write(reinterpret_cast<uint32_t>(ov_buf), ov_addr, 128u);
        noc_async_write_barrier();
    }

    // Header write: BRISC owns bytes [32..64), with cnt_b at offset 32 (u32
    // index 0 of the second-half scratch). NCRISC writes the first 32 B.
    for (uint32_t i = 1; i < 8u; ++i) hdr_scratch[i] = 0u;
    for (uint32_t r = 0; r < nc_all; ++r) {
        hdr_scratch[0] = dram_offset_records[r];  // cnt_b for this receiver
        uint64_t page_base = rgen.get_noc_addr(my_writer_id * nc_all + r);
        noc_async_write(reinterpret_cast<uint32_t>(hdr_scratch),
                        page_base + 32u, 32u);
        noc_async_write_barrier();
    }
}
