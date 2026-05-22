// SPDX-License-Identifier: Apache-2.0
//
// V18 BRISC combined Reader + Scatter (RISCV_0, NOC_0).
//
// V18 = V11 + per-source hash-table pre-aggregation. Each (bx, by, area)
// tuple from BRISC's half of each tile is summed into a separate L1 hash
// table (BRISC and NCRISC have independent tables — they handle disjoint
// cell ranges and can't share fp32 atomicity). At end-of-batch we walk the
// table and emit one tuple per unique (bx, by) into staging, then the
// existing flush_recv path delivers them to route_buf.
//
// Two extra runtime args vs V11:
//  24: v18_hash_off        L1 byte offset of BRISC's hash table region
//  25: v18_hash_bits       log2(num_slots); slots = 1 << bits
//
// BRISC has two jobs:
//   Phase 1 (reader): for each cell-tile this core processes, read
//   px/py/sx/sy from DRAM and push to CBs c_0..c_3 (formerly v4_reader.cpp).
//   The compute kernel (TRISC) consumes from c_0..c_3, runs the SFPU bin
//   geometry, and pushes outputs to c_4..c_21. NCRISC (v11_scatter_dm.cpp)
//   consumes those.
//
//   Phase 2 (scatter): in parallel with NCRISC, walk cells [512..1024) of
//   each tile (NCRISC does [0..512)) and route each (bx, by, area) tuple
//   to its owning core. BRISC writes to a SEPARATE writer slot in
//   route_buf at `my_writer_id = my_core_id + nc_all`. Sync with NCRISC
//   per-tile via two L1 semaphores: `data_ready_sem` (NCRISC publishes
//   per-tile read pointers in a shared L1 struct, then bumps to t+1) and
//   `brisc_done_sem` (BRISC bumps after processing each tile so NCRISC can
//   safely cb_pop_front).
//
// Phase 1 runs first (reader work blocks on cb_reserve when CBs are full,
// naturally pacing with compute). After all reads are pushed, BRISC enters
// Phase 2 and catches up to NCRISC's scatter progress.
//
// Runtime args:
//   0: addr_px              DRAM base of pos_x
//   1: addr_py              DRAM base of pos_y
//   2: addr_sx              DRAM base of sx
//   3: addr_sy              DRAM base of sy
//   4: tile_pgsz            bytes per tile page (4096)
//   5: first_tile           first cell-tile id this core processes
//   6: n_tiles              number of cell-tiles
//   7: tile_map_bytes       (used to compute shard_table offset)
//   8: my_core_id
//   9: nc_all
//  10: M_tiles
//  11: N_tiles
//  12: nbx
//  13: nby
//  14: route_dram
//  15: route_pgsz
//  16: max_per_page_tuples
//  17: my_writer_id         = my_core_id + nc_all
//  18: data_ready_sem_id
//  19: brisc_done_sem_id
//  20: shared_state_off     L1 byte offset to ScatterShared
//  21: brisc_state_off      L1 byte offset to BRISC's private staging block
//  22: tables_ready_sem_id  bumped by NCRISC after loading tile_map+shard_table

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t V11_MAX_OVERLAP = 8u;
constexpr uint32_t TILE_ELEMS      = 1024u;
constexpr uint32_t MAX_IN_FLIGHT   = 64u;  // matches v11_scatter_dm.cpp; ≤ max_per_page/2
constexpr uint32_t V11_PAGE_HDR_BYTES = 64u;
constexpr uint32_t CB_SCRATCH = tt::CBIndex::c_24;
constexpr uint32_t CB_PX = 0u;
constexpr uint32_t CB_PY = 1u;
constexpr uint32_t CB_SX = 2u;
constexpr uint32_t CB_SY = 3u;

struct V11Contrib { uint16_t bx; uint16_t by; float area; };
static_assert(sizeof(V11Contrib) == 8, "V11Contrib must be 8 bytes");

struct ScatterShared {
    uint32_t bxl_ptr;
    uint32_t byl_ptr;
    uint32_t ox_ptr[V11_MAX_OVERLAP];
    uint32_t oy_ptr[V11_MAX_OVERLAP];
};

inline uint32_t v18_hash32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x85ebca6bu;
    x ^= x >> 13;
    return x;
}

constexpr uint32_t V18_KEY_EMPTY = 0xFFFFFFFFu;

void kernel_main() {
    const uint32_t addr_px            = get_arg_val<uint32_t>(0);
    const uint32_t addr_py            = get_arg_val<uint32_t>(1);
    const uint32_t addr_sx            = get_arg_val<uint32_t>(2);
    const uint32_t addr_sy            = get_arg_val<uint32_t>(3);
    const uint32_t tile_pgsz          = get_arg_val<uint32_t>(4);
    const uint32_t first_tile         = get_arg_val<uint32_t>(5);
    const uint32_t n_tiles            = get_arg_val<uint32_t>(6);
    const uint32_t tile_map_bytes     = get_arg_val<uint32_t>(7);
    const uint32_t nc_all             = get_arg_val<uint32_t>(9);
    const uint32_t M_tiles            = get_arg_val<uint32_t>(10);
    const uint32_t N_tiles            = get_arg_val<uint32_t>(11);
    const uint32_t nbx                = get_arg_val<uint32_t>(12);
    const uint32_t nby                = get_arg_val<uint32_t>(13);
    const uint32_t route_dram         = get_arg_val<uint32_t>(14);
    const uint32_t route_pgsz         = get_arg_val<uint32_t>(15);
    const uint32_t max_per_page_tuples= get_arg_val<uint32_t>(16);
    const uint32_t my_writer_id       = get_arg_val<uint32_t>(17);
    const uint32_t data_ready_sem_id  = get_arg_val<uint32_t>(18);
    const uint32_t brisc_done_sem_id  = get_arg_val<uint32_t>(19);
    const uint32_t shared_state_off   = get_arg_val<uint32_t>(20);
    const uint32_t brisc_state_off    = get_arg_val<uint32_t>(21);
    const uint32_t tables_ready_sem_id= get_arg_val<uint32_t>(22);
    const uint32_t drop_dram          = get_arg_val<uint32_t>(23);
    const uint32_t v18_hash_off       = get_arg_val<uint32_t>(24);
    const uint32_t v18_hash_bits      = get_arg_val<uint32_t>(25);
    const uint32_t v18_dirty_off      = get_arg_val<uint32_t>(26);
    const uint32_t v18_bf16_area      = get_arg_val<uint32_t>(27);
    const uint32_t v18_no_dirty       = get_arg_val<uint32_t>(28);
    const uint32_t v18_hash_slots     = 1u << v18_hash_bits;
    const uint32_t v18_hash_mask      = v18_hash_slots - 1u;
    uint32_t total_drops              = 0;

    // ── Setup: locate L1 regions for scatter (interleaved with reader) ────
    uint8_t* base = reinterpret_cast<uint8_t*>(get_write_ptr(CB_SCRATCH));

    volatile ScatterShared* shared =
        reinterpret_cast<volatile ScatterShared*>(base + shared_state_off);

    uint32_t off = brisc_state_off;
    V11Contrib* staging = reinterpret_cast<V11Contrib*>(base + off);
    off += nc_all * MAX_IN_FLIGHT * sizeof(V11Contrib);
    off = (off + 3u) & ~3u;
    uint32_t* staging_count = reinterpret_cast<uint32_t*>(base + off);
    off += nc_all * sizeof(uint32_t);
    uint32_t* dram_offset_tuples = reinterpret_cast<uint32_t*>(base + off);
    off += nc_all * sizeof(uint32_t);
    off = (off + 63u) & ~63u;
    uint32_t* hdr_scratch = reinterpret_cast<uint32_t*>(base + off);

    // tile_to_core sits at offset 0 in CB_SCRATCH (loaded by NCRISC).
    const uint16_t* tile_to_core = reinterpret_cast<const uint16_t*>(base);
    // shard_table offset must mirror NCRISC's exact L1 layout calculation.
    uint32_t st_off = (tile_map_bytes + 7u) & ~7u;
    st_off += nc_all * MAX_IN_FLIGHT * sizeof(V11Contrib);
    st_off  = (st_off + 3u) & ~3u;
    st_off += nc_all * sizeof(uint32_t);
    st_off += nc_all * sizeof(uint32_t);
    st_off  = (st_off + 63u) & ~63u;
    const uint8_t* shard_table = reinterpret_cast<const uint8_t*>(base + st_off);

    // Wait for NCRISC to finish loading the L1 tables.
    {
        volatile tt_l1_ptr uint32_t* tr_sem =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(tables_ready_sem_id));
        noc_semaphore_wait(tr_sem, 1u);
    }

    for (uint32_t i = 0; i < nc_all; ++i) {
        staging_count[i]      = 0u;
        dram_offset_tuples[i] = 0u;
    }

    // V18 hash table (BRISC's private region): keys[] then areas[].
    uint32_t* v18_keys       = reinterpret_cast<uint32_t*>(base + v18_hash_off);
    float*    v18_areas      = reinterpret_cast<float*>   (base + v18_hash_off + v18_hash_slots * 4u);
    uint16_t* v18_areas_bf16 = reinterpret_cast<uint16_t*>(base + v18_hash_off + v18_hash_slots * 4u);
    uint16_t* v18_dirty = v18_no_dirty ? nullptr
                        : reinterpret_cast<uint16_t*>(base + v18_dirty_off);
    uint32_t  v18_dirty_count = 0u;
    {
        DeviceZoneScopedN("V18B-HASH-CLEAR");
        for (uint32_t i = 0; i < v18_hash_slots; ++i) v18_keys[i] = V18_KEY_EMPTY;
    }

    const InterleavedAddrGen<true> rgen = {
        .bank_base_address = route_dram,
        .page_size         = route_pgsz,
    };

    // BRISC's tuple region starts at offset (max_per_page_tuples/2) * 8 inside
    // the page's data area (right after NCRISC's first half). BRISC's per-RISC
    // cap is also half: max_per_page_tuples / 2.
    // Sort+combine duplicates before each NOC write (see v11_scatter_dm.cpp's
    // flush_recv for design notes — same logic, just BRISC's half of the page).
    const uint32_t brisc_offset_bytes = (max_per_page_tuples / 2u) * sizeof(V11Contrib);
    const uint32_t brisc_cap_tuples   = max_per_page_tuples / 2u;
    // Mid-batch hash-flush threshold (mirror of NCRISC). Prevents infinite
    // probe loop when the per-RISC unique-bin count would exceed hash_slots.
    const uint32_t v18_flush_threshold = (v18_hash_slots > 512u)
        ? (v18_hash_slots - 256u) : (v18_hash_slots >> 1);
    auto flush_recv = [&](uint32_t recv, uint32_t cnt) {
        if (cnt == 0u) return;
        V11Contrib* arr = &staging[recv * MAX_IN_FLIGHT];

        // A2: Skip writer-side sort + dedup. Gather atomic-add is commutative
        // (mirror of v11_scatter_dm.cpp). Only NOC alignment padding remains.
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
        if (already >= brisc_cap_tuples) {
            total_drops += cnt;
            staging_count[recv] = 0u;
            return;
        }
        if (already + cnt > brisc_cap_tuples) {
            uint32_t kept = brisc_cap_tuples - already;
            total_drops += cnt - kept;
            cnt = kept;
        }
        uint64_t page_base = rgen.get_noc_addr(my_writer_id * nc_all + recv);
        uint64_t dst = page_base + (uint64_t)V11_PAGE_HDR_BYTES
                     + (uint64_t)brisc_offset_bytes
                     + (uint64_t)already * sizeof(V11Contrib);
        uint32_t src_l1 = reinterpret_cast<uint32_t>(&staging[recv * MAX_IN_FLIGHT]);
        noc_async_write(src_l1, dst, cnt * sizeof(V11Contrib));
        dram_offset_tuples[recv] = already + cnt;
        staging_count[recv]      = 0u;
    };

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

    // ── Reader+scatter interleaved per tile ───────────────────────────────
    // Iter t pushes tile t's positions (if t<n_tiles) then scatters tile t-1
    // (if t>=1). This ordering avoids the deadlock where pushing fills the
    // 2-slot input CBs and stalls compute → NCRISC → brisc_done.
    volatile tt_l1_ptr uint32_t* dr_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(data_ready_sem_id));
    volatile tt_l1_ptr uint32_t* bd_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(brisc_done_sem_id));

    const InterleavedAddrGen<true> gen_px = {.bank_base_address = addr_px, .page_size = tile_pgsz};
    const InterleavedAddrGen<true> gen_py = {.bank_base_address = addr_py, .page_size = tile_pgsz};
    const InterleavedAddrGen<true> gen_sx = {.bank_base_address = addr_sx, .page_size = tile_pgsz};
    const InterleavedAddrGen<true> gen_sy = {.bank_base_address = addr_sy, .page_size = tile_pgsz};

    for (uint32_t t = 0; t <= n_tiles; ++t) {
        // Step A: push tile t to input CBs (skipped on the trailing iter).
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
            {
                DeviceZoneScopedN("V11B-DRAM-READ");
                noc_async_read_page(page_id, gen_px, l1_px);
                noc_async_read_page(page_id, gen_py, l1_py);
                noc_async_read_page(page_id, gen_sx, l1_sx);
                noc_async_read_page(page_id, gen_sy, l1_sy);
                noc_async_read_barrier();
            }
            cb_push_back(CB_PX, 1);
            cb_push_back(CB_PY, 1);
            cb_push_back(CB_SX, 1);
            cb_push_back(CB_SY, 1);
        }

        // Step B: scatter tile t-1 (skipped on the leading iter).
        if (t == 0) continue;
        uint32_t st = t - 1u;  // tile being scattered
        noc_semaphore_wait_min(dr_sem, st + 1u);

        const float* bxl_data = reinterpret_cast<const float*>(shared->bxl_ptr);
        const float* byl_data = reinterpret_cast<const float*>(shared->byl_ptr);
        const float* ox_data[V11_MAX_OVERLAP];
        const float* oy_data[V11_MAX_OVERLAP];
        for (uint32_t j = 0; j < V11_MAX_OVERLAP; ++j) {
            ox_data[j] = reinterpret_cast<const float*>(shared->ox_ptr[j]);
            oy_data[j] = reinterpret_cast<const float*>(shared->oy_ptr[j]);
        }

        // Mid-batch drain if the hash table is nearly full.
        if (v18_dirty_count >= v18_flush_threshold) {
            DeviceZoneScopedN("V18B-MIDFLUSH");
            v18_drain_hash();
        }

        {
            DeviceZoneScopedN("V18B-HASH-INSERT");
            // BRISC processes the upper half of each tile (cells 512..1023).
            // V18: no shard_table consultation, no emit_counter, no 2x2 owner
            // pre-dispatch. Owner determined at flush time from tile_to_core.
            for (uint32_t ci = TILE_ELEMS / 2u; ci < TILE_ELEMS; ++ci) {
                int bxl_i = (int)bxl_data[ci];
                int byl_i = (int)byl_data[ci];
                if (bxl_i < 0) bxl_i = 0;
                if (byl_i < 0) byl_i = 0;
                uint32_t bxl_u = (uint32_t)bxl_i;
                uint32_t byl_u = (uint32_t)byl_i;
                if (bxl_u >= nbx) continue;
                if (byl_u >= nby) continue;

                uint32_t zero_streak_j = 0;
                for (uint32_t j = 0; j < V11_MAX_OVERLAP; ++j) {
                    float ox_j = ox_data[j][ci];
                    if (ox_j <= 0.0f) {
                        if (++zero_streak_j >= 2) break;
                        continue;
                    }
                    zero_streak_j = 0;
                    uint32_t bx_val = bxl_u + j;
                    if (bx_val >= nbx) continue;

                    uint32_t zero_streak_k = 0;
                    for (uint32_t k = 0; k < V11_MAX_OVERLAP; ++k) {
                        float oy_k = oy_data[k][ci];
                        if (oy_k <= 0.0f) {
                            if (++zero_streak_k >= 2) break;
                            continue;
                        }
                        zero_streak_k = 0;
                        uint32_t by_val = byl_u + k;
                        if (by_val >= nby) continue;

                        float area = ox_j * oy_k;
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

        // Signal NCRISC: BRISC done with tile `st`'s CB data.
        *bd_sem = st + 1u;
        asm volatile("" ::: "memory");
    }

    // ── V18: drain remaining hash entries into staging ───────────────────
    {
        DeviceZoneScopedN("V18B-HASH-FLUSH");
        v18_drain_hash();
    }

    // ── Final flush + header write ────────────────────────────────────────
    {
        DeviceZoneScopedN("V18B-FINAL-FLUSH");
        for (uint32_t r = 0; r < nc_all; ++r) {
            uint32_t cnt = staging_count[r];
            if (cnt == 0u) continue;
            flush_recv(r, cnt);
        }
        noc_async_write_barrier();
    }
    {
        DeviceZoneScopedN("V11B-HDR-WRITE");
        // BRISC writes only the SECOND 32 bytes of the header, with cnt_b at
        // byte 32 (= u32 index 0 of the second-half buffer). NCRISC writes
        // the FIRST 32 bytes with cnt_n at byte 0.
        for (uint32_t i = 1; i < 8u; ++i) hdr_scratch[i] = 0u;
        for (uint32_t r = 0; r < nc_all; ++r) {
            hdr_scratch[0] = dram_offset_tuples[r];  // count_b
            uint64_t page_base = rgen.get_noc_addr(my_writer_id * nc_all + r);
            noc_async_write(reinterpret_cast<uint32_t>(hdr_scratch),
                            page_base + 32u, 32u);
            noc_async_write_barrier();
        }
    }

    // ── Drop-counter: BRISC uses slot (my_writer_id + nc_all) so it does
    // not collide with NCRISC (which uses slot my_writer_id).
    {
        const InterleavedAddrGen<true> dgen = {
            .bank_base_address = drop_dram,
            .page_size = 32u,
        };
        noc_inline_dw_write(dgen.get_noc_addr(my_writer_id + nc_all), total_drops);
    }
}
