// SPDX-License-Identifier: Apache-2.0
//
// V19 BRISC combined Reader + Scatter — Phase 1 atomic-add gather variant.
//
// Phase 1 (reader): same as V11 — load px/py/sx/sy from DRAM into CBs c_0..c_3.
// Phase 2 (scatter): for cells [512..1024), atomic-add area into owner's L1
//   density slab. No route_buf, no staging.

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t V11_MAX_OVERLAP = 8u;
constexpr uint32_t TILE_ELEMS      = 1024u;
constexpr uint32_t CB_SCRATCH = tt::CBIndex::c_24;
constexpr uint32_t CB_PX = 0u;
constexpr uint32_t CB_PY = 1u;
constexpr uint32_t CB_SX = 2u;
constexpr uint32_t CB_SY = 3u;

struct ScatterShared {
    uint32_t bxl_ptr;
    uint32_t byl_ptr;
    uint32_t ox_ptr[V11_MAX_OVERLAP];
    uint32_t oy_ptr[V11_MAX_OVERLAP];
};

void kernel_main() {
    const uint32_t addr_px            = get_arg_val<uint32_t>(0);
    const uint32_t addr_py            = get_arg_val<uint32_t>(1);
    const uint32_t addr_sx            = get_arg_val<uint32_t>(2);
    const uint32_t addr_sy            = get_arg_val<uint32_t>(3);
    const uint32_t tile_pgsz          = get_arg_val<uint32_t>(4);
    const uint32_t first_tile         = get_arg_val<uint32_t>(5);
    const uint32_t n_tiles            = get_arg_val<uint32_t>(6);
    (void)get_arg_val<uint32_t>(7);   // tile_map_bytes unused
    (void)get_arg_val<uint32_t>(8);
    const uint32_t nc_all             = get_arg_val<uint32_t>(9);
    (void)get_arg_val<uint32_t>(10);  // M_tiles
    (void)get_arg_val<uint32_t>(11);  // N_tiles
    const uint32_t nbx                = get_arg_val<uint32_t>(12);
    const uint32_t nby                = get_arg_val<uint32_t>(13);
    (void)get_arg_val<uint32_t>(14);
    (void)get_arg_val<uint32_t>(15);
    (void)get_arg_val<uint32_t>(16);
    (void)get_arg_val<uint32_t>(17);  // my_writer_id
    const uint32_t data_ready_sem_id  = get_arg_val<uint32_t>(18);
    const uint32_t brisc_done_sem_id  = get_arg_val<uint32_t>(19);
    const uint32_t shared_state_off   = get_arg_val<uint32_t>(20);
    (void)get_arg_val<uint32_t>(21);  // brisc_state_off
    const uint32_t tables_ready_sem_id= get_arg_val<uint32_t>(22);
    (void)get_arg_val<uint32_t>(23);  // drop_dram
    const uint32_t density_l1_off     = get_arg_val<uint32_t>(24);
    const uint32_t density_slab_bins  = get_arg_val<uint32_t>(25);
    const uint32_t scale_bits         = get_arg_val<uint32_t>(26);

    const float SCALE_F = (float)(1u << scale_bits);

    uint8_t* base = reinterpret_cast<uint8_t*>(get_write_ptr(CB_SCRATCH));
    volatile ScatterShared* shared =
        reinterpret_cast<volatile ScatterShared*>(base + shared_state_off);

    // density_l1 + noc_coords share c_24 with the same offsets used by NCRISC.
    const uint32_t coords_off =
        (density_l1_off + density_slab_bins * 4u + 31u) & ~31u;
    uint32_t* density_l1 = reinterpret_cast<uint32_t*>(base + density_l1_off);
    uint32_t* noc_coords = reinterpret_cast<uint32_t*>(base + coords_off);

    // Wait for NCRISC to load the noc_coords table.
    {
        volatile tt_l1_ptr uint32_t* tr_sem =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(tables_ready_sem_id));
        noc_semaphore_wait(tr_sem, 1u);
    }

    volatile tt_l1_ptr uint32_t* dr_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(data_ready_sem_id));
    volatile tt_l1_ptr uint32_t* bd_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(brisc_done_sem_id));

    const InterleavedAddrGen<true> gen_px = {.bank_base_address = addr_px, .page_size = tile_pgsz};
    const InterleavedAddrGen<true> gen_py = {.bank_base_address = addr_py, .page_size = tile_pgsz};
    const InterleavedAddrGen<true> gen_sx = {.bank_base_address = addr_sx, .page_size = tile_pgsz};
    const InterleavedAddrGen<true> gen_sy = {.bank_base_address = addr_sy, .page_size = tile_pgsz};

    for (uint32_t t = 0; t <= n_tiles; ++t) {
        // Phase 1: push tile t into reader CBs (skipped on trailing iter).
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
                DeviceZoneScopedN("V19B-DRAM-READ");
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

        // Phase 2: scatter tile t-1 (skipped on leading iter).
        if (t == 0) continue;
        uint32_t st = t - 1u;
        noc_semaphore_wait_min(dr_sem, st + 1u);

        const float* bxl_data = reinterpret_cast<const float*>(shared->bxl_ptr);
        const float* byl_data = reinterpret_cast<const float*>(shared->byl_ptr);
        const float* ox_data[V11_MAX_OVERLAP];
        const float* oy_data[V11_MAX_OVERLAP];
        for (uint32_t j = 0; j < V11_MAX_OVERLAP; ++j) {
            ox_data[j] = reinterpret_cast<const float*>(shared->ox_ptr[j]);
            oy_data[j] = reinterpret_cast<const float*>(shared->oy_ptr[j]);
        }

        {
            DeviceZoneScopedN("V19B-ATOMIC");
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

                        float    area_fp32  = ox_j * oy_k;
                        uint32_t area_fixed = (uint32_t)(area_fp32 * SCALE_F);
                        if (area_fixed == 0u) continue;

                        uint32_t bin_global = bx_val * nby + by_val;
                        // BLOCK PARTITION (see v19_scatter_dm.cpp for rationale).
                        uint32_t owner_idx  = bin_global / density_slab_bins;
                        uint32_t local_idx  = bin_global - owner_idx * density_slab_bins;
                        if (owner_idx >= nc_all) continue;
                        uint32_t packed_xy  = noc_coords[owner_idx];
                        uint32_t owner_x    = packed_xy & 0xFFFFu;
                        uint32_t owner_y    = packed_xy >> 16;
                        uint64_t target = get_noc_addr(owner_x, owner_y,
                                                       (uint32_t)density_l1 + local_idx * 4u);
                        noc_semaphore_inc(target, area_fixed);
                    }
                }
            }
            noc_async_atomic_barrier();
        }

        *bd_sem = st + 1u;
        asm volatile("" ::: "memory");
    }
}
