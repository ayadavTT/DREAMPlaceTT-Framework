// SPDX-License-Identifier: Apache-2.0
//
// V19 NCRISC scatter — direct L1 atomic-add gather.
//
// Per cell (j,k) nonzero overlap: atomic-add fixed-point area into the owning
// core's L1 density slab. No route_buf, no staging, no separate gather kernel.
//
// Density layout (across the 110-core mesh):
//   M*N density bins of uint32 fixed-point.
//   Strided ownership: bin_global = bx*N + by; owner_idx = bin_global % nc_all
//   Per core local slot: local_idx = bin_global / nc_all (0..⌈M*N/nc_all⌉)
//   Density slab lives in CB_SCRATCH (c_24) at offset density_l1_off.

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t V11_MAX_OVERLAP = 8u;
constexpr uint32_t TILE_ELEMS      = 1024u;

constexpr uint32_t CB_SCRATCH = tt::CBIndex::c_24;
constexpr uint32_t CB_BXL     = 4u;
constexpr uint32_t CB_BYL     = 5u;
constexpr uint32_t CB_OX_BASE = 6u;
constexpr uint32_t CB_OY_BASE = 14u;

struct ScatterShared {
    uint32_t bxl_ptr;
    uint32_t byl_ptr;
    uint32_t ox_ptr[V11_MAX_OVERLAP];
    uint32_t oy_ptr[V11_MAX_OVERLAP];
};

void kernel_main() {
    const uint32_t my_core_id          = get_arg_val<uint32_t>(3);
    const uint32_t nc_all              = get_arg_val<uint32_t>(4);
    (void)get_arg_val<uint32_t>(5);   // M_tiles unused
    (void)get_arg_val<uint32_t>(6);   // N_tiles unused
    const uint32_t nbx                 = get_arg_val<uint32_t>(7);
    const uint32_t nby                 = get_arg_val<uint32_t>(8);
    const uint32_t n_tiles             = get_arg_val<uint32_t>(9);
    const uint32_t data_ready_sem_id   = get_arg_val<uint32_t>(17);
    const uint32_t brisc_done_sem_id   = get_arg_val<uint32_t>(18);
    const uint32_t shared_state_off    = get_arg_val<uint32_t>(19);
    (void)get_arg_val<uint32_t>(20);  // brisc_state_off
    const uint32_t tables_ready_sem_id = get_arg_val<uint32_t>(21);
    (void)get_arg_val<uint32_t>(22);  // drop_dram unused
    const uint32_t density_l1_off      = get_arg_val<uint32_t>(23);
    const uint32_t density_slab_bins   = get_arg_val<uint32_t>(24);
    const uint32_t noc_coord_dram      = get_arg_val<uint32_t>(25);
    const uint32_t noc_coord_pgsz      = get_arg_val<uint32_t>(26);
    const uint32_t scale_bits          = get_arg_val<uint32_t>(27);
    const uint32_t density_dram_base   = get_arg_val<uint32_t>(28);
    const uint32_t density_dram_pgsz   = get_arg_val<uint32_t>(29);
    const uint32_t my_core_idx_arg     = get_arg_val<uint32_t>(30);
    (void)my_core_id;

    const float SCALE_F = (float)(1u << scale_bits);

    uint8_t* base = reinterpret_cast<uint8_t*>(get_write_ptr(CB_SCRATCH));

    // Density slab at host-controlled L1 offset. noc_coords table follows
    // immediately after density slab (32-byte aligned). Both live in c_24.
    const uint32_t coords_off =
        (density_l1_off + density_slab_bins * 4u + 31u) & ~31u;
    uint32_t* density_l1 = reinterpret_cast<uint32_t*>(base + density_l1_off);
    uint32_t* noc_coords = reinterpret_cast<uint32_t*>(base + coords_off);

    volatile ScatterShared* shared =
        reinterpret_cast<volatile ScatterShared*>(base + shared_state_off);

    // Step 0: zero own density slab + COPY worker_noc_coords[] from common
    // runtime args into L1.
    //
    // CRITICAL FIX (2026-05-22): replaced noc_async_read of coord_buf with
    // common-runtime-args copy. At 2048 grid, all 110 cores hitting DRAM
    // bank 0 simultaneously to read coord_buf (single page) AND the L1
    // destination being at a high offset caused the read to silently return
    // shifted/garbage data — the kernel computed bogus NoC targets and
    // noc_async_atomic_barrier hung waiting for ACKs that would never return.
    // SetCommonRuntimeArgs from the host broadcasts the 110-entry coords
    // table to all cores with no NoC reads needed.
    {
        DeviceZoneScopedN("V19N-INIT");
        for (uint32_t i = 0; i < density_slab_bins; ++i) density_l1[i] = 0u;
        for (uint32_t i = 0; i < nc_all; ++i) {
            noc_coords[i] = get_common_arg_val<uint32_t>((int)i);
        }
    }
    (void)noc_coord_dram;
    (void)noc_coord_pgsz;

    // Signal BRISC: shared state ready.
    {
        volatile tt_l1_ptr uint32_t* tr_sem =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(tables_ready_sem_id));
        *tr_sem = 1u;
        asm volatile("" ::: "memory");
    }

    volatile tt_l1_ptr uint32_t* dr_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(data_ready_sem_id));
    volatile tt_l1_ptr uint32_t* bd_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(brisc_done_sem_id));

    for (uint32_t t = 0; t < n_tiles; ++t) {
        {
            DeviceZoneScopedN("V19N-CB-WAIT");
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
            DeviceZoneScopedN("V19N-ATOMIC");
            for (uint32_t ci = 0; ci < TILE_ELEMS / 2u; ++ci) {
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
                        // BLOCK PARTITION ownership: each core owns a
                        // contiguous range of bin_globals so writeout can
                        // dump each core's slab to row-major DRAM directly
                        // (no host de-stride). Microbench at 2048 showed
                        // atomic contention from spatial clusters costs
                        // ~+1 ms/scatter vs stride-1 — far less than the
                        // ~15-20 ms host de-stride it eliminates.
                        uint32_t owner_idx  = bin_global / density_slab_bins;
                        uint32_t local_idx  = bin_global - owner_idx * density_slab_bins;
                        if (owner_idx >= nc_all) continue;  // padding bin past total_bins
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

        noc_semaphore_wait_min(bd_sem, t + 1u);
        cb_pop_front(CB_BXL, 1);
        cb_pop_front(CB_BYL, 1);
        for (uint32_t j = 0; j < V11_MAX_OVERLAP; ++j) {
            cb_pop_front(CB_OX_BASE + j, 1);
            cb_pop_front(CB_OY_BASE + j, 1);
        }
    }

    // DRAM writeout moved to v19_writeout_dm.cpp (separate kernel) to ensure
    // ALL cores' atomics have landed before any core reads its L1 slab.
    // The in-kernel writeout race was producing 4-8 ULP differences across
    // runs and breaking convergence.
    (void)density_dram_base;
    (void)density_dram_pgsz;
    (void)my_core_idx_arg;
}
