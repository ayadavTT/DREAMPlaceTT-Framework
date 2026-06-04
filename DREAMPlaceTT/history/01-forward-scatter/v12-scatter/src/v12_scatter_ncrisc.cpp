// SPDX-License-Identifier: Apache-2.0
//
// V12 NCRISC Scatter Kernel (RISCV_1, NOC_1) — Phase 1.
//
// Mirror of v12_scatter_brisc.cpp but processes the SECOND HALF of each
// SoA tile (cells [512..1023]) and uses NOC_1. Both RISCs read the same
// four SoA pages concurrently (no race: reads only). NCRISC writes to the
// FIRST HALF of each route page (offsets [HDR..HDR + max_half*80)).
// BRISC writes to the SECOND HALF (offset [HDR + max_half*80..]).
//
// After completing its half, NCRISC participates in the same all-cores
// barrier: increments the coordinator, then waits for go.
//
// Geometry constants, V12Overlap struct, and route page layout are
// identical to v12_scatter_brisc.cpp (both files compiled together as one
// Program).
//
// Runtime args: identical layout to v12_scatter_brisc.cpp (args 0-23).
// Arg 22 (phase2_go_sem_id) is the NCRISC-side wait semaphore.

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"
#include <cstring>

#ifndef V12_BSX_F
#define V12_BSX_F 3.90625f
#endif
#ifndef V12_BSY_F
#define V12_BSY_F 3.90625f
#endif
#ifndef V12_INV_BSX_F
#define V12_INV_BSX_F 0.256f
#endif
#ifndef V12_INV_BSY_F
#define V12_INV_BSY_F 0.256f
#endif
#ifndef V12_XL_F
#define V12_XL_F 0.0f
#endif
#ifndef V12_YL_F
#define V12_YL_F 0.0f
#endif

static constexpr float BSX    = V12_BSX_F;
static constexpr float BSY    = V12_BSY_F;
static constexpr float INV_BSX= V12_INV_BSX_F;
static constexpr float INV_BSY= V12_INV_BSY_F;
static constexpr float XL     = V12_XL_F;
static constexpr float YL     = V12_YL_F;

static constexpr uint32_t TILE_ELEMS  = 1024u;
static constexpr uint32_t TILE_BYTES  = TILE_ELEMS * sizeof(float);
static constexpr uint32_t MAX_OVERLAP = 8u;
static constexpr uint32_t CB_SCRATCH  = tt::CBIndex::c_24;

struct V12Overlap {
    float    overlap_x[8];
    float    overlap_y[8];
    uint16_t bxl;
    uint16_t byl;
    uint8_t  wx;
    uint8_t  wy;
    uint8_t  pad[10];
};
static_assert(sizeof(V12Overlap) == 80u, "V12Overlap must be 80 bytes");

static constexpr uint32_t V12_HDR_BYTES = 32u;

inline int32_t correct_bin_idx_x(int32_t idx, float pos) {
    float bl = XL + (float)idx * BSX;
    if (bl > pos) idx--;
    bl = XL + (float)(idx + 1) * BSX;
    if (bl <= pos) idx++;
    return idx;
}
inline int32_t correct_bin_idx_y(int32_t idx, float pos) {
    float bl = YL + (float)idx * BSY;
    if (bl > pos) idx--;
    bl = YL + (float)(idx + 1) * BSY;
    if (bl <= pos) idx++;
    return idx;
}

void kernel_main() {
    const uint32_t addr_px          = get_arg_val<uint32_t>(0);
    const uint32_t addr_py          = get_arg_val<uint32_t>(1);
    const uint32_t addr_sx          = get_arg_val<uint32_t>(2);
    const uint32_t addr_sy          = get_arg_val<uint32_t>(3);
    const uint32_t tile_pgsz        = get_arg_val<uint32_t>(4);
    const uint32_t first_tile       = get_arg_val<uint32_t>(5);
    const uint32_t n_soa_tiles      = get_arg_val<uint32_t>(6);
    const uint32_t tile_map_dram    = get_arg_val<uint32_t>(7);
    const uint32_t tile_map_bytes   = get_arg_val<uint32_t>(8);
    const uint32_t tile_map_pgsz    = get_arg_val<uint32_t>(9);
    const uint32_t my_core_id       = get_arg_val<uint32_t>(10);
    const uint32_t nc_all           = get_arg_val<uint32_t>(11);
    const uint32_t M_tiles          = get_arg_val<uint32_t>(12);
    const uint32_t N_tiles          = get_arg_val<uint32_t>(13);
    const uint32_t nbx              = get_arg_val<uint32_t>(14);
    const uint32_t nby              = get_arg_val<uint32_t>(15);
    const uint32_t route_dram       = get_arg_val<uint32_t>(16);
    const uint32_t route_pgsz       = get_arg_val<uint32_t>(17);
    const uint32_t max_overlaps_half= get_arg_val<uint32_t>(18);
    const uint32_t barrier_sem_id   = get_arg_val<uint32_t>(19);
    const uint32_t coord_noc_x      = get_arg_val<uint32_t>(20);
    const uint32_t coord_noc_y      = get_arg_val<uint32_t>(21);
    const uint32_t phase2_go_sem_id = get_arg_val<uint32_t>(22);

    // ── L1 layout mirrors BRISC exactly so both share the same CB_SCRATCH.
    // NCRISC's private data is placed AFTER BRISC's regions.
    uint8_t* base = reinterpret_cast<uint8_t*>(get_write_ptr(CB_SCRATCH));
    uint32_t off = 0u;

    // Skip BRISC's tile_to_core region — NCRISC reads from the same region
    // (BRISC loads it, NCRISC reads it — safe via L1 coherence + barrier).
    uint16_t* tile_to_core = reinterpret_cast<uint16_t*>(base + off);
    off += tile_map_bytes;
    off  = (off + 63u) & ~63u;

    // Cell buffers (BRISC's — NCRISC reads the same arrays after BRISC loads).
    float* cell_px = reinterpret_cast<float*>(base + off); off += TILE_BYTES;
    float* cell_py = reinterpret_cast<float*>(base + off); off += TILE_BYTES;
    float* cell_sx = reinterpret_cast<float*>(base + off); off += TILE_BYTES;
    float* cell_sy = reinterpret_cast<float*>(base + off); off += TILE_BYTES;
    off  = (off + 63u) & ~63u;

    // Skip BRISC's staging region.
    constexpr uint32_t BRISC_STAGE_CAP = 4u;
    off += nc_all * BRISC_STAGE_CAP * (uint32_t)sizeof(V12Overlap);
    off  = (off + 3u) & ~3u;
    off += nc_all * sizeof(uint32_t);  // BRISC staging_count
    off += nc_all * sizeof(uint32_t);  // BRISC dram_off_vecs
    off  = (off + 63u) & ~63u;

    // NCRISC's private staging region (same structure, separate slots).
    constexpr uint32_t NCRISC_STAGE_CAP = 4u;
    V12Overlap* staging = reinterpret_cast<V12Overlap*>(base + off);
    off += nc_all * NCRISC_STAGE_CAP * (uint32_t)sizeof(V12Overlap);
    off  = (off + 3u) & ~3u;
    uint32_t* staging_count = reinterpret_cast<uint32_t*>(base + off);
    off += nc_all * sizeof(uint32_t);
    uint32_t* dram_off_vecs = reinterpret_cast<uint32_t*>(base + off);
    off += nc_all * sizeof(uint32_t);
    off  = (off + 63u) & ~63u;

    // NCRISC needs to wait until BRISC has loaded tile_to_core + cell data
    // before reading them. BRISC signals via a direct L1 store after its
    // DRAM reads complete. We use the same tables_ready semaphore slot.
    // For V12 we use a simpler approach: NCRISC issues its own DRAM reads
    // independently on NOC_1 (parallel with BRISC's NOC_0 reads).

    for (uint32_t i = 0; i < nc_all; ++i) {
        staging_count[i] = 0u;
        dram_off_vecs[i] = 0u;
    }

    // NCRISC data occupies the FIRST HALF of each route page (at header + 0).
    // BRISC data occupies the SECOND HALF.
    const uint32_t ncrisc_data_off = V12_HDR_BYTES;  // immediately after header

    const InterleavedAddrGen<true> rgen = {
        .bank_base_address = route_dram,
        .page_size         = route_pgsz,
    };

    auto flush_recv = [&](uint32_t recv) {
        uint32_t cnt = staging_count[recv];
        if (cnt == 0u) return;

        uint32_t already = dram_off_vecs[recv];
        if (already >= max_overlaps_half) { staging_count[recv] = 0u; return; }
        uint32_t to_write = cnt;
        if (already + to_write > max_overlaps_half)
            to_write = max_overlaps_half - already;

        uint64_t page_base = rgen.get_noc_addr(my_core_id * nc_all + recv);
        uint64_t dst = page_base
                     + (uint64_t)ncrisc_data_off
                     + (uint64_t)already * sizeof(V12Overlap);
        uint32_t src_l1 = reinterpret_cast<uint32_t>(&staging[recv * NCRISC_STAGE_CAP]);
        uint32_t write_bytes = (to_write * (uint32_t)sizeof(V12Overlap) + 31u) & ~31u;
        noc_async_write(src_l1, dst, write_bytes);
        dram_off_vecs[recv] = already + to_write;
        staging_count[recv] = 0u;
    };

    // NCRISC loads its own tile_to_core copy (same DRAM page, NOC_1).
    {
        DeviceZoneScopedN("V12N-MAP-LOAD");
        const InterleavedAddrGen<true> mgen = {
            .bank_base_address = tile_map_dram,
            .page_size         = tile_map_pgsz,
        };
        noc_async_read(mgen.get_noc_addr(0),
                       reinterpret_cast<uint32_t>(tile_to_core),
                       tile_map_bytes);
        noc_async_read_barrier();
    }

    // DRAM generators for cell data (NCRISC reads independently on NOC_1).
    const InterleavedAddrGen<true> gen_px = { .bank_base_address = addr_px, .page_size = tile_pgsz };
    const InterleavedAddrGen<true> gen_py = { .bank_base_address = addr_py, .page_size = tile_pgsz };
    const InterleavedAddrGen<true> gen_sx = { .bank_base_address = addr_sx, .page_size = tile_pgsz };
    const InterleavedAddrGen<true> gen_sy = { .bank_base_address = addr_sy, .page_size = tile_pgsz };

    // L1 for NCRISC's private cell buffers (immediately after staging).
    float* ncell_px = reinterpret_cast<float*>(base + off); off += TILE_BYTES;
    float* ncell_py = reinterpret_cast<float*>(base + off); off += TILE_BYTES;
    float* ncell_sx = reinterpret_cast<float*>(base + off); off += TILE_BYTES;
    float* ncell_sy = reinterpret_cast<float*>(base + off); off += TILE_BYTES;
    off  = (off + 63u) & ~63u;

    {
        DeviceZoneScopedN("V12N-SCATTER");

        for (uint32_t t = 0u; t < n_soa_tiles; ++t) {
            uint32_t page_id = first_tile + t;

            noc_async_read_page(page_id, gen_px, reinterpret_cast<uint32_t>(ncell_px));
            noc_async_read_page(page_id, gen_py, reinterpret_cast<uint32_t>(ncell_py));
            noc_async_read_page(page_id, gen_sx, reinterpret_cast<uint32_t>(ncell_sx));
            noc_async_read_page(page_id, gen_sy, reinterpret_cast<uint32_t>(ncell_sy));
            noc_async_read_barrier();

            // Process SECOND HALF [512..1023].
            for (uint32_t ci = TILE_ELEMS / 2u; ci < TILE_ELEMS; ++ci) {
                float cx  = ncell_px[ci];
                float cy  = ncell_py[ci];
                float csx = ncell_sx[ci];
                float csy = ncell_sy[ci];

                if (csx <= 0.0f || csy <= 0.0f) continue;

                float half_sx = csx * 0.5f;
                float half_sy = csy * 0.5f;
                float cx_lo = cx - half_sx;
                float cx_hi = cx + half_sx;
                float cy_lo = cy - half_sy;
                float cy_hi = cy + half_sy;

                int32_t bxl = (int32_t)((cx_lo - XL) * INV_BSX);
                if (bxl < 0) bxl = 0;
                bxl = correct_bin_idx_x(bxl, cx_lo);
                if (bxl < 0) bxl = 0;
                int32_t bxr = (int32_t)((cx_hi - XL) * INV_BSX);
                if (bxr >= (int32_t)nbx) bxr = (int32_t)nbx - 1;
                bxr = correct_bin_idx_x(bxr, cx_hi - 0.0001f * BSX);
                if (bxr < bxl) bxr = bxl;
                if (bxr >= (int32_t)nbx) bxr = (int32_t)nbx - 1;
                uint32_t wx = (uint32_t)(bxr - bxl + 1);
                if (wx > MAX_OVERLAP) wx = MAX_OVERLAP;

                float ovx[MAX_OVERLAP];
                for (uint32_t j = 0u; j < wx; ++j) {
                    float bl = XL + (float)(bxl + (int32_t)j) * BSX;
                    float br = bl + BSX;
                    float lo = (cx_lo > bl) ? cx_lo : bl;
                    float hi = (cx_hi < br) ? cx_hi : br;
                    float ov = hi - lo;
                    ovx[j] = (ov > 0.0f) ? ov : 0.0f;
                }

                int32_t byl = (int32_t)((cy_lo - YL) * INV_BSY);
                if (byl < 0) byl = 0;
                byl = correct_bin_idx_y(byl, cy_lo);
                if (byl < 0) byl = 0;
                int32_t byr = (int32_t)((cy_hi - YL) * INV_BSY);
                if (byr >= (int32_t)nby) byr = (int32_t)nby - 1;
                byr = correct_bin_idx_y(byr, cy_hi - 0.0001f * BSY);
                if (byr < byl) byr = byl;
                if (byr >= (int32_t)nby) byr = (int32_t)nby - 1;
                uint32_t wy = (uint32_t)(byr - byl + 1);
                if (wy > MAX_OVERLAP) wy = MAX_OVERLAP;

                float ovy[MAX_OVERLAP];
                for (uint32_t k = 0u; k < wy; ++k) {
                    float bl = YL + (float)(byl + (int32_t)k) * BSY;
                    float br = bl + BSY;
                    float lo = (cy_lo > bl) ? cy_lo : bl;
                    float hi = (cy_hi < br) ? cy_hi : br;
                    float ov = hi - lo;
                    ovy[k] = (ov > 0.0f) ? ov : 0.0f;
                }

                uint32_t tx_lo = (uint32_t)bxl >> 5u;
                uint32_t tx_hi = (uint32_t)(bxl + (int32_t)wx - 1) >> 5u;
                uint32_t ty_lo = (uint32_t)byl >> 5u;
                uint32_t ty_hi = (uint32_t)(byl + (int32_t)wy - 1) >> 5u;
                if (tx_hi >= M_tiles) tx_hi = M_tiles - 1u;
                if (ty_hi >= N_tiles) ty_hi = N_tiles - 1u;

                for (uint32_t tx = tx_lo; tx <= tx_hi; ++tx) {
                    for (uint32_t ty = ty_lo; ty <= ty_hi; ++ty) {
                        uint32_t owner = (uint32_t)tile_to_core[tx * N_tiles + ty];
                        if (owner >= nc_all) continue;

                        V12Overlap ov_vec;
                        for (uint32_t j = 0u; j < MAX_OVERLAP; ++j) {
                            ov_vec.overlap_x[j] = 0.0f;
                            ov_vec.overlap_y[j] = 0.0f;
                        }
                        for (uint32_t p = 0u; p < 10u; ++p) ov_vec.pad[p] = 0u;
                        ov_vec.bxl = (uint16_t)(uint32_t)bxl;
                        ov_vec.byl = (uint16_t)(uint32_t)byl;
                        ov_vec.wx  = (uint8_t)wx;
                        ov_vec.wy  = (uint8_t)wy;
                        for (uint32_t j = 0u; j < wx; ++j) ov_vec.overlap_x[j] = ovx[j];
                        for (uint32_t k = 0u; k < wy; ++k) ov_vec.overlap_y[k] = ovy[k];

                        uint32_t cnt = staging_count[owner];
                        staging[owner * NCRISC_STAGE_CAP + cnt] = ov_vec;
                        cnt++;
                        staging_count[owner] = cnt;
                        if (cnt >= NCRISC_STAGE_CAP) flush_recv(owner);
                    }
                }
            }  // for ci
        }  // for t
    }

    {
        DeviceZoneScopedN("V12N-FLUSH");
        for (uint32_t r = 0u; r < nc_all; ++r) flush_recv(r);
        noc_async_write_barrier();
    }

    // NCRISC writes count_n at byte 0 of the page header (BRISC writes count_b
    // at byte 4). We write a full 32B header but only set word 0 (NCRISC count).
    {
        DeviceZoneScopedN("V12N-HDR-WRITE");
        uint32_t* hdr_scratch = reinterpret_cast<uint32_t*>(base + off);
        for (uint32_t r = 0u; r < nc_all; ++r) {
            hdr_scratch[0] = dram_off_vecs[r];  // count_n
            hdr_scratch[1] = 0u;                // count_b placeholder
            for (uint32_t i = 2u; i < 8u; ++i) hdr_scratch[i] = 0u;
            uint64_t page_base = rgen.get_noc_addr(my_core_id * nc_all + r);
            noc_async_write(reinterpret_cast<uint32_t>(hdr_scratch),
                            page_base, V12_HDR_BYTES);
            noc_async_write_barrier();
        }
    }

    // All-cores barrier: NCRISC also increments and waits.
    {
        DeviceZoneScopedN("V12N-BARRIER");
        uint64_t coord_sem_addr = get_noc_addr(coord_noc_x, coord_noc_y,
                                               get_semaphore(barrier_sem_id));
        noc_semaphore_inc(coord_sem_addr, 1u);

        volatile tt_l1_ptr uint32_t* go_sem =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(phase2_go_sem_id));
        noc_semaphore_wait(go_sem, 1u);
        noc_semaphore_set(go_sem, 0u);
    }
}
