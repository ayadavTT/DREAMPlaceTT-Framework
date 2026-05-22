// SPDX-License-Identifier: Apache-2.0
//
// V18-outer NCRISC Scatter (RISCV_1, NOC_1) — V11outer + V18 hash-agg.
//
// Combines V11outer's precomputed SFPU outer-product (saves inline ox*oy
// scalar multiplies; wins on grid≥2048) with V18's per-source hash-table
// dedup (saves gather work; wins on small grids). Goal: net win on 2048
// configs where V18 alone regresses ~+13% sc+ga.
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

// In-flight buffer per receiver. Sort+dedupe removed (A2). MIF must stay
// ≤ max_per_page/2 (= half_cap per RISC) to avoid silent tuple truncation at
// flush time. With max_per_page=128, MIF=64.
constexpr uint32_t MAX_IN_FLIGHT = 64u;

// Page header is 64 bytes (count + 15 padding uint32s) for NOC cache-line
// alignment. 32-byte headers exposed an even/odd writer race: adjacent L1
// 32-byte destinations share a 64-byte cache line, and parallel NOC writes
// to the same line produced garbage on the odd-indexed slot.
constexpr uint32_t V11_PAGE_HDR_BYTES = 64u;

constexpr uint32_t CB_SCRATCH = tt::CBIndex::c_24;
constexpr uint32_t CB_BXL     = 4u;
constexpr uint32_t CB_BYL     = 5u;
constexpr uint32_t CB_OX_BASE    = 6u;       // V11-outer: NCRISC pops these too
constexpr uint32_t CB_OY_BASE    = 14u;
constexpr uint32_t CB_OUTER_BASE = 32u;      // 8 CBs c_32..c_39, 8 tiles each per batch
                                              // (skip c_24 which is CB_SCRATCH in V11)
constexpr uint32_t OUTER_TILES_PER_BATCH = 8u;

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

// V18 hash helpers (same as v18_scatter_dm.cpp).
inline uint32_t v18_hash32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x85ebca6bu;
    x ^= x >> 13;
    return x;
}
constexpr uint32_t V18_KEY_EMPTY = 0xFFFFFFFFu;

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
    // Drop-counter instrumentation: one uint32 per writer in a small DRAM
    // buffer. We write the per-iter total at kernel end via inline-DW.
    const uint32_t drop_dram          = get_arg_val<uint32_t>(22);
    // V18 hash table + dirty list args (same layout as v18_scatter_dm.cpp).
    const uint32_t v18_hash_off       = get_arg_val<uint32_t>(23);
    const uint32_t v18_hash_bits      = get_arg_val<uint32_t>(24);
    const uint32_t v18_dirty_off      = get_arg_val<uint32_t>(25);
    const uint32_t v18_bf16_area      = get_arg_val<uint32_t>(26);
    const uint32_t v18_no_dirty       = get_arg_val<uint32_t>(27);
    const uint32_t v18_hash_slots     = 1u << v18_hash_bits;
    const uint32_t v18_hash_mask      = v18_hash_slots - 1u;
    uint32_t total_drops              = 0;

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

    // V18 hash table + dirty list.
    uint32_t* v18_keys       = reinterpret_cast<uint32_t*>(base + v18_hash_off);
    float*    v18_areas      = reinterpret_cast<float*>   (base + v18_hash_off + v18_hash_slots * 4u);
    uint16_t* v18_areas_bf16 = reinterpret_cast<uint16_t*>(base + v18_hash_off + v18_hash_slots * 4u);
    uint16_t* v18_dirty = v18_no_dirty ? nullptr
                        : reinterpret_cast<uint16_t*>(base + v18_dirty_off);
    uint32_t  v18_dirty_count = 0u;

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

    // ── V18: clear hash table keys to EMPTY sentinel ──────────────────────
    {
        DeviceZoneScopedN("V18N-HASH-CLEAR");
        for (uint32_t i = 0; i < v18_hash_slots; ++i) v18_keys[i] = V18_KEY_EMPTY;
    }

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

        // A2: Skip the insertion-sort + dedup pass. Gather accumulates via
        // atomic-add (commutative), so duplicate (bx, by) tuples are summed
        // correctly even when sent ungrouped. fp32 sum order changes are
        // <1 ULP and well within the 1% HPWL tolerance. Only padding to a
        // 4-tuple boundary remains, required for 32-byte NOC alignment.
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
            // Entire flush dropped — record original cnt as dropped.
            total_drops += cnt;
            staging_count[recv] = 0u;
            return;
        }
        if (already + cnt > ncrisc_cap_tuples) {
            uint32_t kept = ncrisc_cap_tuples - already;
            total_drops += cnt - kept;
            cnt = kept;
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

    const uint32_t v18_flush_threshold = (v18_hash_slots > 512u)
        ? (v18_hash_slots - 256u) : (v18_hash_slots >> 1);
    auto v18_emit_slot = [&](uint32_t h_d) {
        uint32_t key_d = v18_keys[h_d];
        if (key_d == V18_KEY_EMPTY) return;
        uint32_t bx_d = key_d >> 16;
        uint32_t by_d = key_d & 0xFFFFu;
        float a_d;
        if (v18_bf16_area) {
            union { uint32_t u; float f; } cv;
            cv.u = ((uint32_t)v18_areas_bf16[h_d]) << 16;
            a_d = cv.f;
        } else {
            a_d = v18_areas[h_d];
        }
        v18_keys[h_d] = V18_KEY_EMPTY;
        uint32_t tx_d = bx_d >> 5;
        uint32_t ty_d = by_d >> 5;
        if (tx_d >= M_tiles || ty_d >= N_tiles) return;
        uint32_t owner_d = (uint32_t)tile_to_core[tx_d * N_tiles + ty_d];
        if (owner_d >= nc_all) return;
        uint32_t cnt_d = staging_count[owner_d];
        V11Contrib& slot_d = staging[owner_d * MAX_IN_FLIGHT + cnt_d];
        slot_d.bx   = (uint16_t)bx_d;
        slot_d.by   = (uint16_t)by_d;
        slot_d.area = a_d;
        cnt_d++;
        staging_count[owner_d] = cnt_d;
        if (cnt_d >= MAX_IN_FLIGHT) {
            flush_recv(owner_d, MAX_IN_FLIGHT);
            noc_async_write_barrier();
        }
    };
    auto v18_drain_hash = [&]() {
        if (v18_no_dirty) {
            for (uint32_t h = 0; h < v18_hash_slots; ++h) v18_emit_slot(h);
        } else {
            for (uint32_t i = 0; i < v18_dirty_count; ++i) {
                v18_emit_slot((uint32_t)v18_dirty[i]);
            }
        }
        v18_dirty_count = 0u;
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
            // V11-outer: wait on the outer-product CBs (c_22..c_29), 8 tiles
            // each (one per K value). The intermediate ox/oy CBs (c_6..c_21)
            // are produced and consumed by v4_outer_compute internally — we
            // still need to cb_wait_front + cb_pop_front them so the CB's
            // ring buffer advances.
            for (uint32_t j = 0; j < V11_MAX_OVERLAP; ++j) {
                cb_wait_front(CB_OX_BASE + j, 1);
                cb_wait_front(CB_OY_BASE + j, 1);
                cb_wait_front(CB_OUTER_BASE + j, OUTER_TILES_PER_BATCH);
            }
        }

        const float* bxl_data = reinterpret_cast<const float*>(get_read_ptr(CB_BXL));
        const float* byl_data = reinterpret_cast<const float*>(get_read_ptr(CB_BYL));
        // Stage B': outer_base[J] = address of c_(22+J)'s front tile; K-th
        // outer-product tile is at outer_base[J] + K * 4096 bytes.
        constexpr uint32_t OUTER_TILE_BYTES = TILE_ELEMS * sizeof(float);
        const float* outer_data[V11_MAX_OVERLAP][V11_MAX_OVERLAP];
        for (uint32_t j = 0; j < V11_MAX_OVERLAP; ++j) {
            uint32_t base = (uint32_t)get_read_ptr(CB_OUTER_BASE + j);
            for (uint32_t k = 0; k < V11_MAX_OVERLAP; ++k) {
                outer_data[j][k] = reinterpret_cast<const float*>(base + k * OUTER_TILE_BYTES);
            }
        }

        // Publish read pointers + signal BRISC. Direct L1 stores; same-core
        // L1 SRAM is byte-coherent across RISCs.
        // V11-outer: BRISC uses ox_ptr[J] as the BASE address of c_(22+J)
        // (re-purposed slot). oy_ptr is unused in V11-outer mode.
        shared->bxl_ptr = (uint32_t)bxl_data;
        shared->byl_ptr = (uint32_t)byl_data;
        for (uint32_t j = 0; j < V11_MAX_OVERLAP; ++j) {
            shared->ox_ptr[j] = (uint32_t)get_read_ptr(CB_OUTER_BASE + j);
            shared->oy_ptr[j] = 0u;  // unused
        }
        asm volatile("" ::: "memory");
        *dr_sem = t + 1u;
        asm volatile("" ::: "memory");

        // V18: mid-batch drain if hash table is near full.
        if (v18_dirty_count >= v18_flush_threshold) {
            DeviceZoneScopedN("V18N-MIDFLUSH");
            v18_drain_hash();
        }

        {
            DeviceZoneScopedN("V18N-HASH-INSERT");
            // NCRISC processes the lower half of each tile (cells 0..511).
            for (uint32_t ci = 0; ci < TILE_ELEMS / 2u; ++ci) {
                int bxl_i = (int)bxl_data[ci];
                int byl_i = (int)byl_data[ci];
                if (bxl_i < 0) bxl_i = 0;
                if (byl_i < 0) byl_i = 0;
                uint32_t bxl_u = (uint32_t)bxl_i;
                uint32_t byl_u = (uint32_t)byl_i;
                if (bxl_u >= nbx) continue;
                if (byl_u >= nby) continue;

                // V18-outer: area = precomputed outer-product (no scalar ox*oy).
                // V18: hash-agg (bx, by, area). No shard_table / emit_counter.
                for (uint32_t j = 0; j < V11_MAX_OVERLAP; ++j) {
                    uint32_t bx_val = bxl_u + j;
                    if (bx_val >= nbx) continue;

                    uint32_t zero_streak_k = 0;
                    for (uint32_t k = 0; k < V11_MAX_OVERLAP; ++k) {
                        float area = outer_data[j][k][ci];
                        if (area <= 0.0f) {
                            if (++zero_streak_k >= 2) break;
                            continue;
                        }
                        zero_streak_k = 0;
                        uint32_t by_val = byl_u + k;
                        if (by_val >= nby) continue;

                        uint32_t key = (bx_val << 16) | by_val;
                        uint32_t h = v18_hash32(key) & v18_hash_mask;
                        bool inserted = false;
                        for (uint32_t pass = 0; pass < 2u && !inserted; ++pass) {
                            uint32_t probes = 0u;
                            for (; probes < 64u; ++probes) {
                                uint32_t k_at = v18_keys[h];
                                if (k_at == key) {
                                    if (v18_bf16_area) {
                                        union { uint32_t u; float f; } cv;
                                        cv.u = ((uint32_t)v18_areas_bf16[h]) << 16;
                                        cv.f += area;
                                        v18_areas_bf16[h] = (uint16_t)(cv.u >> 16);
                                    } else {
                                        v18_areas[h] += area;
                                    }
                                    inserted = true; break;
                                }
                                if (k_at == V18_KEY_EMPTY) {
                                    v18_keys[h] = key;
                                    if (v18_bf16_area) {
                                        union { uint32_t u; float f; } cv;
                                        cv.f = area;
                                        v18_areas_bf16[h] = (uint16_t)(cv.u >> 16);
                                    } else {
                                        v18_areas[h] = area;
                                    }
                                    if (!v18_no_dirty) {
                                        v18_dirty[v18_dirty_count++] = (uint16_t)h;
                                    } else {
                                        v18_dirty_count++;
                                    }
                                    inserted = true; break;
                                }
                                h = (h + 1u) & v18_hash_mask;
                            }
                            if (!inserted) {
                                v18_drain_hash();
                                h = v18_hash32(key) & v18_hash_mask;
                            }
                        }
                        if (!inserted) total_drops += 1;
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
            cb_pop_front(CB_OUTER_BASE + j, OUTER_TILES_PER_BATCH);
        }
    }

    // ── Step 2: final flush of partial buffers ────────────────────────────
    // flush_recv handles sort+dedup+pad internally, so pass the raw count.
    // V18: drain remaining hash entries first.
    {
        DeviceZoneScopedN("V18N-HASH-FLUSH");
        v18_drain_hash();
    }
    {
        DeviceZoneScopedN("V18N-FINAL-FLUSH");
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

    // ── Drop-counter: write total_drops to drop_dram[my_writer_id] ────────
    // Single inline-DW write (4 bytes) targeting a 32-byte-aligned slot.
    {
        const InterleavedAddrGen<true> dgen = {
            .bank_base_address = drop_dram,
            .page_size = 32u,
        };
        noc_inline_dw_write(dgen.get_noc_addr(my_writer_id), total_drops);
    }
}
