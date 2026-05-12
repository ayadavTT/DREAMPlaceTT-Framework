// SPDX-License-Identifier: Apache-2.0
//
// V11 NCRISC Histogram Kernel — Phase 3 hot-tile pre-pass.
//
// Counts how many tuples each writer would emit per global tile_id, without
// actually doing the NOC routing. Output: per-writer count array dumped to
// a DRAM page. Host reads all writer pages and reduces to a global histogram
// to identify hot tiles for K-way sharding.
//
// Reuses v4_reader (BRISC) + v4_compute (TRISC) to populate input CBs
// c_4..c_21 (bxl, byl, ovx[8], oy[8]) — same as v11_scatter_dm.
//
// L1 scratch (CB_SCRATCH = c_24):
//   local_count[total_tiles] uint32  (16 KB at 2048² = 4096 tiles)
//
// Runtime args:
//   0: my_core_id              writer linear id
//   1: M_tiles, 2: N_tiles
//   3: nbx, 4: nby
//   5: n_tiles                 cell tiles to process
//   6: hist_dram_addr          DRAM base of per-core histogram pages
//   7: hist_pgsz               bytes per page (= total_tiles * 4, padded)

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

void kernel_main() {
    const uint32_t my_core_id  = get_arg_val<uint32_t>(0);
    const uint32_t M_tiles     = get_arg_val<uint32_t>(1);
    const uint32_t N_tiles     = get_arg_val<uint32_t>(2);
    const uint32_t nbx         = get_arg_val<uint32_t>(3);
    const uint32_t nby         = get_arg_val<uint32_t>(4);
    const uint32_t n_tiles     = get_arg_val<uint32_t>(5);
    const uint32_t hist_dram   = get_arg_val<uint32_t>(6);
    const uint32_t hist_pgsz   = get_arg_val<uint32_t>(7);

    const uint32_t total_tiles = M_tiles * N_tiles;

    uint8_t* base = reinterpret_cast<uint8_t*>(get_write_ptr(CB_SCRATCH));
    uint32_t* local_count = reinterpret_cast<uint32_t*>(base);

    // Init counts to zero
    for (uint32_t i = 0; i < total_tiles; ++i) local_count[i] = 0u;

    // Walk SFPU outputs, count per-tile contribs
    for (uint32_t t = 0; t < n_tiles; ++t) {
        cb_wait_front(CB_BXL, 1);
        cb_wait_front(CB_BYL, 1);
        for (uint32_t j = 0; j < V11_MAX_OVERLAP; ++j) {
            cb_wait_front(CB_OX_BASE + j, 1);
            cb_wait_front(CB_OY_BASE + j, 1);
        }

        const float* bxl_data = reinterpret_cast<const float*>(get_read_ptr(CB_BXL));
        const float* byl_data = reinterpret_cast<const float*>(get_read_ptr(CB_BYL));
        const float* ox_data[V11_MAX_OVERLAP];
        const float* oy_data[V11_MAX_OVERLAP];
        for (uint32_t j = 0; j < V11_MAX_OVERLAP; ++j) {
            ox_data[j] = reinterpret_cast<const float*>(get_read_ptr(CB_OX_BASE + j));
            oy_data[j] = reinterpret_cast<const float*>(get_read_ptr(CB_OY_BASE + j));
        }

        {
            DeviceZoneScopedN("V11H-COUNT");
            for (uint32_t ci = 0; ci < TILE_ELEMS; ++ci) {
                int bxl = (int)bxl_data[ci];
                int byl = (int)byl_data[ci];

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
                        local_count[map_idx]++;
                    }
                }
            }
        }

        cb_pop_front(CB_BXL, 1);
        cb_pop_front(CB_BYL, 1);
        for (uint32_t j = 0; j < V11_MAX_OVERLAP; ++j) {
            cb_pop_front(CB_OX_BASE + j, 1);
            cb_pop_front(CB_OY_BASE + j, 1);
        }
    }

    // Write local_count to DRAM (single bulk write per core)
    {
        DeviceZoneScopedN("V11H-WRITE");
        const InterleavedAddrGen<true> hgen = {
            .bank_base_address = hist_dram,
            .page_size         = hist_pgsz,
        };
        noc_async_write(reinterpret_cast<uint32_t>(local_count),
                        hgen.get_noc_addr(my_core_id),
                        total_tiles * sizeof(uint32_t));
        noc_async_write_barrier();
    }
}
