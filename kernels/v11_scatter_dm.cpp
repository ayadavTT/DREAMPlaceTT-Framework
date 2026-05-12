// SPDX-License-Identifier: Apache-2.0
//
// V11 NCRISC Scatter Kernel (RISCV_1, NOC_1) — Phase 2 real version.
//
// Reads SFPU output tiles from CBs c_4..c_21 (produced by v4_compute), walks
// the 8x8 neighbor grid for each cell, looks up the owning core for each
// (bx, by) bin via the tile_to_core[] map, stages the (bx, by, area) tuple
// in a per-receiver L1 buffer, and bulk NOC-writes per-receiver buffers to
// per-(writer, reader) DRAM pages in route_buf.
//
// L1 staging is capped at MAX_IN_FLIGHT tuples per receiver. When a
// receiver's buffer fills, the kernel flushes it via NOC and resets the
// L1 buffer. The route_buf DRAM page accumulates multiple flushes from
// the same writer at progressive byte offsets.
//
// route_buf page layout (per-writer, per-reader):
//   [uint32 count][12 bytes pad] [V11Contrib[count]]
//   Header is 16 bytes for NOC 16-byte alignment.
//   Each V11Contrib is 16 bytes (with 8 bytes of pad) so tuple writes/reads
//   stay 16-byte aligned even when cnt is odd.
//   page_pgsz = 16 + MAX_PER_PAGE_TUPLES * 16, padded to 32-byte alignment
//
// Input CBs (same as v4_ncrisc_scatter):
//   c_4  = bxl
//   c_5  = byl
//   c_6..c_13  = overlap_x[0..7]
//   c_14..c_21 = overlap_y[0..7]
//
// Runtime args:
//   0..15: same as before
//  16: my_writer_id            for NCRISC: my_core_id; for BRISC: my_core_id + nc_all
//  17: data_ready_sem_id       NCRISC publishes per-tile L1 ptrs then bumps this
//  18: brisc_done_sem_id       BRISC bumps after processing each tile
//  19: shared_state_off        L1 byte offset to ScatterShared (per-tile ptrs)
//  20: brisc_state_off         L1 byte offset to BRISC's private staging region
//  21: tables_ready_sem_id     NCRISC bumps once after loading tile_map+shard_table
//
// Sync pattern (same-core, BRISC↔NCRISC, both kernels in one Program):
//   Tables loaded by NCRISC; BRISC waits on tables_ready_sem before reading.
//   Per tile: NCRISC publishes get_read_ptr() values into shared struct, sets
//   data_ready = t+1 via direct L1 store; BRISC polls until value ≥ t+1 via
//   noc_semaphore_wait_min. After BRISC processes its half it sets
//   brisc_done = t+1 via direct L1 store; NCRISC polls until ≥ t+1 before
//   cb_pop_front (so the producer doesn't overwrite the in-flight tile).

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t V11_MAX_OVERLAP = 8u;
constexpr uint32_t TILE_ELEMS      = 1024u;

// In-flight buffer per receiver. flush_recv sorts+dedupes before DRAM write,
// so larger MAX_IN_FLIGHT = more duplicates combined per flush. But insertion
// sort is O(n²), so we keep MAX_IN_FLIGHT modest. 64 makes each flush cost
// ~2K compares ≈ 20 µs; with ~1k flushes per writer, ≈ 20 ms scatter overhead.
constexpr uint32_t MAX_IN_FLIGHT = 64u;

// Page header is 64 bytes (count + 15 padding uint32s) for NOC cache-line
// alignment. 32-byte headers exposed an even/odd writer race: adjacent L1
// 32-byte destinations share a 64-byte cache line, and parallel NOC writes
// to the same line produced garbage on the odd-indexed slot.
constexpr uint32_t V11_PAGE_HDR_BYTES = 64u;

constexpr uint32_t CB_SCRATCH = tt::CBIndex::c_24;
constexpr uint32_t CB_BXL     = 4u;
constexpr uint32_t CB_BYL     = 5u;
constexpr uint32_t CB_OX_BASE = 6u;
constexpr uint32_t CB_OY_BASE = 14u;

// 8-byte tuple. To keep NOC writes 32-byte aligned (Blackhole over-writes
// up to ~16 bytes if the size isn't a 32-byte multiple), we always emit
// tuples in groups of 4 (= 32 bytes). MAX_IN_FLIGHT below is a multiple of 4
// so mid-loop flushes need no padding; the final flush rounds up and the
// padding slots are zeroed (area=0 tuples are no-ops on the reader).
struct V11Contrib { uint16_t bx; uint16_t by; float area; };
static_assert(sizeof(V11Contrib) == 8, "V11Contrib must be 8 bytes");

// Per-tile pointers published by NCRISC, read by BRISC (must match the
// layout in v11_scatter_b_dm.cpp).
struct ScatterShared {
    uint32_t bxl_ptr;
    uint32_t byl_ptr;
    uint32_t ox_ptr[V11_MAX_OVERLAP];
    uint32_t oy_ptr[V11_MAX_OVERLAP];
};

void kernel_main() {
    const uint32_t tile_map_dram      = get_arg_val<uint32_t>(0);
    const uint32_t tile_map_pgsz      = get_arg_val<uint32_t>(1);
    const uint32_t tile_map_bytes     = get_arg_val<uint32_t>(2);
    const uint32_t my_core_id         = get_arg_val<uint32_t>(3);
    const uint32_t nc_all             = get_arg_val<uint32_t>(4);
    const uint32_t M_tiles            = get_arg_val<uint32_t>(5);
    const uint32_t N_tiles            = get_arg_val<uint32_t>(6);
    const uint32_t nbx                = get_arg_val<uint32_t>(7);
    const uint32_t nby                = get_arg_val<uint32_t>(8);
    const uint32_t n_tiles            = get_arg_val<uint32_t>(9);
    const uint32_t route_dram         = get_arg_val<uint32_t>(10);
    const uint32_t route_pgsz         = get_arg_val<uint32_t>(11);
    const uint32_t max_per_page_tuples= get_arg_val<uint32_t>(12);
    const uint32_t inv_ba_u32         = get_arg_val<uint32_t>(13);
    const uint32_t shard_dram         = get_arg_val<uint32_t>(14);
    const uint32_t shard_pgsz         = get_arg_val<uint32_t>(15);
    const uint32_t my_writer_id       = get_arg_val<uint32_t>(16);
    const uint32_t data_ready_sem_id  = get_arg_val<uint32_t>(17);
    const uint32_t brisc_done_sem_id  = get_arg_val<uint32_t>(18);
    const uint32_t shared_state_off   = get_arg_val<uint32_t>(19);
    (void)get_arg_val<uint32_t>(20);  // brisc_state_off unused on NCRISC
    const uint32_t tables_ready_sem_id= get_arg_val<uint32_t>(21);

    union { uint32_t u; float f; } cv;
    cv.u = inv_ba_u32;
    const float inv_bin_area = cv.f;
    (void)inv_bin_area;  // unused in scatter

    // ── L1 scratch layout ─────────────────────────────────────────────────
    uint8_t* base = reinterpret_cast<uint8_t*>(get_write_ptr(CB_SCRATCH));
    uint32_t off = 0;

    uint16_t* tile_to_core = reinterpret_cast<uint16_t*>(base + off);
    off += tile_map_bytes;
    off = (off + 7u) & ~7u;  // 8-byte align for tuples

    V11Contrib* staging = reinterpret_cast<V11Contrib*>(base + off);
    off += nc_all * MAX_IN_FLIGHT * sizeof(V11Contrib);
    off = (off + 3u) & ~3u;

    uint32_t* staging_count = reinterpret_cast<uint32_t*>(base + off);
    off += nc_all * sizeof(uint32_t);

    uint32_t* dram_offset_tuples = reinterpret_cast<uint32_t*>(base + off);
    off += nc_all * sizeof(uint32_t);
    off = (off + 63u) & ~63u;

    // Shard table copy in L1: SHARD_BYTES per tile.
    // Loaded once at startup from DRAM. K=1 means no sharding.
    uint8_t* shard_table = reinterpret_cast<uint8_t*>(base + off);
    off += shard_pgsz;
    off = (off + 63u) & ~63u;

    // 64-byte aligned scratch for header writes at the end (full cache line).
    uint32_t* hdr_scratch = reinterpret_cast<uint32_t*>(base + off);
    off += V11_PAGE_HDR_BYTES;

    // Shared per-tile pointer struct (BRISC reads this at shared_state_off).
    volatile ScatterShared* shared =
        reinterpret_cast<volatile ScatterShared*>(base + shared_state_off);

    // ── Step 0: load tile_to_core[] and shard_table[] from DRAM ───────────
    {
        DeviceZoneScopedN("V11-MAP-LOAD");
        const InterleavedAddrGen<true> mgen = {
            .bank_base_address = tile_map_dram,
            .page_size         = tile_map_pgsz,
        };
        noc_async_read(mgen.get_noc_addr(0),
                       reinterpret_cast<uint32_t>(tile_to_core),
                       tile_map_bytes);
        const InterleavedAddrGen<true> sgen = {
            .bank_base_address = shard_dram,
            .page_size         = shard_pgsz,
        };
        noc_async_read(sgen.get_noc_addr(0),
                       reinterpret_cast<uint32_t>(shard_table),
                       shard_pgsz);
        noc_async_read_barrier();
    }

    // Signal BRISC that the shared L1 tables (tile_to_core, shard_table)
    // are loaded and safe to read. Direct L1 store + compiler barrier:
    // same-core inter-RISC L1 is coherent within a few cycles.
    {
        volatile tt_l1_ptr uint32_t* tr_sem =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(tables_ready_sem_id));
        *tr_sem = 1u;
        asm volatile("" ::: "memory");
    }

    // ── Init per-receiver bookkeeping ─────────────────────────────────────
    for (uint32_t i = 0; i < nc_all; ++i) {
        staging_count[i]      = 0u;
        dram_offset_tuples[i] = 0u;
    }

    // Global tuple emit counter (for shard round-robin)
    uint32_t emit_counter = 0u;

    // ── Setup route_buf address generator ────────────────────────────────
    const InterleavedAddrGen<true> rgen = {
        .bank_base_address = route_dram,
        .page_size         = route_pgsz,
    };

    // Inline mid-loop flush helper. Sorts staging[recv] by (bx, by), combines
    // adjacent duplicates (so multiple cells overlapping the same bin emit ONE
    // tuple instead of N), pads to a multiple of 4 for 32-byte NOC alignment,
    // and writes the result to DRAM. Combining writer-side bounds the tuple
    // count per (writer, receiver) by the number of unique bins this writer
    // touched in this receiver's owned tiles (≤ ~3 tiles × 1024 bins) — well
    // below the per-page cap — and eliminates the silent tuple drops that
    // were causing TT density to be biased on dense placements.
    // Uses a single shared NOC channel (RISCV_1's NOC_1).
    // Step 5b: NCRISC owns the FIRST half of each writer page (tuples
    // 0..max/2). BRISC owns the second half. Per-RISC cap is half of the
    // total page tuple budget.
    const uint32_t ncrisc_cap_tuples = max_per_page_tuples / 2u;
    auto flush_recv = [&](uint32_t recv, uint32_t cnt) {
        if (cnt == 0u) return;
        V11Contrib* arr = &staging[recv * MAX_IN_FLIGHT];

        // Insertion sort by composite key (bx << 16 | by). MAX_IN_FLIGHT ≤ 64
        // makes O(n²) cheap (≤ ~2K cycles per flush). Stable order doesn't
        // matter — duplicates are combined next.
        for (uint32_t i = 1; i < cnt; ++i) {
            V11Contrib v = arr[i];
            uint32_t key_v = ((uint32_t)v.bx << 16) | (uint32_t)v.by;
            int32_t j = (int32_t)i - 1;
            while (j >= 0) {
                uint32_t key_j = ((uint32_t)arr[j].bx << 16) | (uint32_t)arr[j].by;
                if (key_j <= key_v) break;
                arr[j + 1] = arr[j];
                --j;
            }
            arr[j + 1] = v;
        }

        // Combine adjacent equal-key entries: arr[w-1].area += arr[r].area.
        uint32_t w = 0;
        for (uint32_t r = 0; r < cnt; ++r) {
            if (w > 0 && arr[w - 1].bx == arr[r].bx && arr[w - 1].by == arr[r].by) {
                arr[w - 1].area += arr[r].area;
            } else {
                arr[w] = arr[r];
                ++w;
            }
        }
        cnt = w;

        // Pad to multiple of 4 (32 bytes) for NOC alignment; padding tuples
        // have area=0 and are skipped by the accumulator's a==0 check.
        uint32_t cnt_padded = (cnt + 3u) & ~3u;
        for (uint32_t i = cnt; i < cnt_padded; ++i) {
            arr[i].bx = 0;
            arr[i].by = 0;
            arr[i].area = 0.0f;
        }
        cnt = cnt_padded;
        if (cnt == 0u) {
            staging_count[recv] = 0u;
            return;
        }

        uint32_t already = dram_offset_tuples[recv];
        if (already >= ncrisc_cap_tuples) {
            staging_count[recv] = 0u;
            return;
        }
        if (already + cnt > ncrisc_cap_tuples) {
            cnt = ncrisc_cap_tuples - already;
        }
        uint64_t page_base = rgen.get_noc_addr(my_writer_id * nc_all + recv);
        uint64_t dst = page_base
                     + (uint64_t)V11_PAGE_HDR_BYTES
                     + (uint64_t)already * sizeof(V11Contrib);
        uint32_t src_l1 = reinterpret_cast<uint32_t>(
            &staging[recv * MAX_IN_FLIGHT]);
        noc_async_write(src_l1, dst, cnt * sizeof(V11Contrib));
        dram_offset_tuples[recv] = already + cnt;
        staging_count[recv]      = 0u;
    };

    // ── Step 1: walk SFPU output, push tuples, flush mid-loop on overflow ─
    // BRISC processes the upper half of each tile in parallel via a shared
    // L1 ScatterShared struct + data_ready/brisc_done direct-L1 semaphores.
    volatile tt_l1_ptr uint32_t* dr_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(data_ready_sem_id));
    volatile tt_l1_ptr uint32_t* bd_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(brisc_done_sem_id));

    for (uint32_t t = 0; t < n_tiles; ++t) {
        {
            DeviceZoneScopedN("V11-CB-WAIT");
            cb_wait_front(CB_BXL, 1);
            cb_wait_front(CB_BYL, 1);
            for (uint32_t j = 0; j < V11_MAX_OVERLAP; ++j) {
                cb_wait_front(CB_OX_BASE + j, 1);
                cb_wait_front(CB_OY_BASE + j, 1);
            }
        }

        const float* bxl_data = reinterpret_cast<const float*>(get_read_ptr(CB_BXL));
        const float* byl_data = reinterpret_cast<const float*>(get_read_ptr(CB_BYL));
        const float* ox_data[V11_MAX_OVERLAP];
        const float* oy_data[V11_MAX_OVERLAP];
        for (uint32_t j = 0; j < V11_MAX_OVERLAP; ++j) {
            ox_data[j] = reinterpret_cast<const float*>(get_read_ptr(CB_OX_BASE + j));
            oy_data[j] = reinterpret_cast<const float*>(get_read_ptr(CB_OY_BASE + j));
        }

        // Publish read pointers + signal BRISC. Direct L1 stores; same-core
        // L1 SRAM is byte-coherent across RISCs.
        shared->bxl_ptr = (uint32_t)bxl_data;
        shared->byl_ptr = (uint32_t)byl_data;
        for (uint32_t j = 0; j < V11_MAX_OVERLAP; ++j) {
            shared->ox_ptr[j] = (uint32_t)ox_data[j];
            shared->oy_ptr[j] = (uint32_t)oy_data[j];
        }
        asm volatile("" ::: "memory");
        *dr_sem = t + 1u;
        asm volatile("" ::: "memory");

        {
            DeviceZoneScopedN("V11-ROUTE");
            // NCRISC processes the lower half of each tile (cells 0..511).
            for (uint32_t ci = 0; ci < TILE_ELEMS / 2u; ++ci) {
                int bxl = (int)bxl_data[ci];
                int byl = (int)byl_data[ci];

                // Track one-step precision blips: break only after TWO consecutive
                // zero overlaps. Single zeros may be V4 SFPU floor-precision
                // artifacts; two in a row means we've genuinely walked off the cell.
                uint32_t zero_streak_j = 0;
                for (uint32_t j = 0; j < V11_MAX_OVERLAP; ++j) {
                    float ox_j = ox_data[j][ci];
                    if (ox_j <= 0.0f) {
                        if (++zero_streak_j >= 2) break;
                        continue;
                    }
                    zero_streak_j = 0;
                    int bx_s = bxl + (int)j;
                    if (bx_s < 0) continue;
                    uint32_t bx_val = (uint32_t)bx_s;
                    if (bx_val >= nbx) continue;
                    uint32_t tile_x = bx_val >> 5;
                    if (tile_x >= M_tiles) continue;
                    // Hoist tile_x * N_tiles outside k-loop (one mul per j)
                    uint32_t tile_x_row = tile_x * N_tiles;

                    uint32_t zero_streak_k = 0;
                    for (uint32_t k = 0; k < V11_MAX_OVERLAP; ++k) {
                        float oy_k = oy_data[k][ci];
                        if (oy_k <= 0.0f) {
                            if (++zero_streak_k >= 2) break;
                            continue;
                        }
                        zero_streak_k = 0;
                        int by_s = byl + (int)k;
                        if (by_s < 0) continue;
                        uint32_t by_val = (uint32_t)by_s;
                        if (by_val >= nby) continue;
                        uint32_t tile_y = by_val >> 5;
                        if (tile_y >= N_tiles) continue;

                        uint32_t map_idx = tile_x_row + tile_y;
                        uint32_t primary = (uint32_t)tile_to_core[map_idx];
                        if (primary >= nc_all) continue;

                        // Shard round-robin: route to alt owners for hot tiles (K>1).
                        // For cold tiles (K=1), owner == primary always.
                        const uint8_t* sh = shard_table + map_idx * 16u;
                        uint32_t K = (uint32_t)sh[0];
                        uint32_t owner = (K <= 1u) ? primary
                                       : ((emit_counter % K) == 0u ? primary
                                                                    : (uint32_t)sh[emit_counter % K]);
                        emit_counter++;

                        float area = ox_j * oy_k;
                        uint32_t cnt = staging_count[owner];
                        V11Contrib& slot = staging[owner * MAX_IN_FLIGHT + cnt];
                        slot.bx   = (uint16_t)bx_val;
                        slot.by   = (uint16_t)by_val;
                        slot.area = area;
                        cnt++;
                        staging_count[owner] = cnt;

                        if (cnt >= MAX_IN_FLIGHT) {
                            flush_recv(owner, MAX_IN_FLIGHT);
                            noc_async_write_barrier();
                        }
                    }
                }
            }
        }

        // Wait for BRISC to finish reading SFPU output for this tile before
        // popping (so the producer doesn't overwrite the tile's L1 slot).
        noc_semaphore_wait_min(bd_sem, t + 1u);

        cb_pop_front(CB_BXL, 1);
        cb_pop_front(CB_BYL, 1);
        for (uint32_t j = 0; j < V11_MAX_OVERLAP; ++j) {
            cb_pop_front(CB_OX_BASE + j, 1);
            cb_pop_front(CB_OY_BASE + j, 1);
        }
    }

    // ── Step 2: final flush of partial buffers ────────────────────────────
    // flush_recv handles sort+dedup+pad internally, so pass the raw count.
    {
        DeviceZoneScopedN("V11-FINAL-FLUSH");
        for (uint32_t r = 0; r < nc_all; ++r) {
            uint32_t cnt = staging_count[r];
            if (cnt == 0u) continue;
            flush_recv(r, cnt);
        }
        noc_async_write_barrier();
    }

    // ── Step 3: write headers — NCRISC owns the FIRST 32 bytes (cnt_n at
    // offset 0). BRISC owns bytes [32..64). Page header is read in full
    // (64 bytes) by accum, which reads cnt_n at hdr[0] and cnt_b at hdr[8].
    {
        DeviceZoneScopedN("V11-HDR-WRITE");
        for (uint32_t i = 1; i < 8u; ++i) hdr_scratch[i] = 0u;
        for (uint32_t r = 0; r < nc_all; ++r) {
            hdr_scratch[0] = dram_offset_tuples[r];  // cnt_n
            uint64_t page_base = rgen.get_noc_addr(my_writer_id * nc_all + r);
            noc_async_write(reinterpret_cast<uint32_t>(hdr_scratch),
                            page_base, 32u);
            noc_async_write_barrier();
        }
    }
}
