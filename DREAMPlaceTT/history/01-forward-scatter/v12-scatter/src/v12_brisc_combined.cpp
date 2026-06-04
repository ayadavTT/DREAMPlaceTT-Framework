// SPDX-License-Identifier: Apache-2.0
//
// V12 BRISC Combined Kernel (RISCV_0, NOC_0) — Phase 1 Scatter + Phase 2 Accum.
//
// Single kernel_main() running both phases sequentially:
//
//   Phase 1 (Scatter):
//     Reads cell data (first half of each SoA tile) from DRAM, computes FP32
//     overlaps, routes V12Overlap vectors to tile-owning cores via route_buf.
//     Core 0 is the barrier coordinator.
//
//   Barrier:
//     All BRISCs and NCRISCs (2 * nc_all total) atomically increment the
//     coordinator's (core 0 BRISC) barrier semaphore.  Coordinator waits for
//     count == 2*nc_all, then broadcasts phase2_go signals to every core via
//     per-core unicast writes using the NOC table stored in tile_map_buf.
//     Non-coordinator BRISCs increment coordinator and then wait for their
//     own go semaphore.
//
//   Phase 2 (Accum — BRISC side, direct FP32):
//     Reads all V12Overlap vectors from route_buf[writer][this_core] pages and
//     directly accumulates density contributions in FP32 using RISC-V arithmetic
//     (no TRISC/FPU involved).  After all contributions for a tile are accumulated,
//     pushes the FP32 density tile to CB_ACCUM for NCRISC to write to DRAM.
//
// Runtime args (args 0..24 + per-tile indices):
//   0:  addr_px              DRAM base of px SoA
//   1:  addr_py              DRAM base of py SoA
//   2:  addr_sx              DRAM base of sx SoA
//   3:  addr_sy              DRAM base of sy SoA
//   4:  tile_pgsz            bytes per SoA page (4096)
//   5:  first_tile           first SoA tile index for this core
//   6:  n_soa_tiles          number of SoA tiles this core processes
//   7:  tile_map_dram        DRAM base of tile_map_buf
//                            layout: [tile_to_core (uint16 × M×N)] then
//                                    [noc_table (2×uint32 × nc_all)]
//   8:  tile_map_bytes       bytes of just the tile_to_core portion (total_tiles*2)
//   9:  tile_map_pgsz        page size of tile_map_buf
//  10:  my_core_id           linear core id (0..nc_all-1)
//  11:  nc_all               total number of Tensix cores
//  12:  M_tiles              density grid width in 32-wide tiles
//  13:  N_tiles              density grid height in 32-wide tiles
//  14:  nbx                  density grid width in bins
//  15:  nby                  density grid height in bins
//  16:  route_dram           DRAM base of route_buf
//  17:  route_pgsz           bytes per (writer, receiver) route page
//  18:  max_overlaps_half    max V12Overlap vectors BRISC may write per page half
//  19:  barrier_sem_id       on-chip semaphore for all-cores Phase 1→2 barrier
//  20:  coord_noc_x          NOC X of coordinator core (core 0)
//  21:  coord_noc_y          NOC Y of coordinator core (core 0)
//  22:  phase2_go_sem_b_id   BRISC go semaphore (BRISC waits on this)
//  23:  phase2_go_sem_n_id   NCRISC go semaphore (coordinator also broadcasts this)
//  24:  n_tiles_owned        number of density tiles owned by this core
//  25 .. 25+n_tiles_owned-1: tile linear indices (tx * N_tiles + ty)

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"
#include <cstring>

// ── Geometry constants (JIT #defines) ───────────────────────────────────────
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

static constexpr uint32_t TILE_ELEMS   = 1024u;
static constexpr uint32_t TILE_BYTES   = TILE_ELEMS * sizeof(float);
static constexpr uint32_t MAX_OVERLAP  = 8u;
static constexpr uint32_t CB_OX        = tt::CBIndex::c_0;
static constexpr uint32_t CB_OY        = tt::CBIndex::c_1;
static constexpr uint32_t CB_ACCUM     = tt::CBIndex::c_3;
static constexpr uint32_t CB_SCRATCH   = tt::CBIndex::c_24;
static constexpr uint32_t TILE_FLOATS  = TILE_W * TILE_H;
static constexpr uint32_t TILE_W       = 32u;
static constexpr uint32_t TILE_H       = 32u;
static constexpr uint32_t TILE_BF16_SZ = TILE_W * TILE_H * (uint32_t)sizeof(uint16_t);
// 64 bytes: bytes 0..31 = NCRISC region (count_n), bytes 32..63 = BRISC region (count_b).
// Two 32-byte aligned non-overlapping regions prevent BRISC/NCRISC write races.
static constexpr uint32_t V12_HDR_BYTES= 64u;
static constexpr uint32_t BRISC_STAGE_CAP = 4u;
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
    // ── Runtime args ─────────────────────────────────────────────────────────
    const uint32_t addr_px           = get_arg_val<uint32_t>(0);
    const uint32_t addr_py           = get_arg_val<uint32_t>(1);
    const uint32_t addr_sx           = get_arg_val<uint32_t>(2);
    const uint32_t addr_sy           = get_arg_val<uint32_t>(3);
    const uint32_t tile_pgsz         = get_arg_val<uint32_t>(4);
    const uint32_t first_tile        = get_arg_val<uint32_t>(5);
    const uint32_t n_soa_tiles       = get_arg_val<uint32_t>(6);
    const uint32_t tile_map_dram     = get_arg_val<uint32_t>(7);
    const uint32_t tile_map_bytes    = get_arg_val<uint32_t>(8);
    const uint32_t tile_map_pgsz     = get_arg_val<uint32_t>(9);
    const uint32_t my_core_id        = get_arg_val<uint32_t>(10);
    const uint32_t nc_all            = get_arg_val<uint32_t>(11);
    const uint32_t M_tiles           = get_arg_val<uint32_t>(12);
    const uint32_t N_tiles           = get_arg_val<uint32_t>(13);
    const uint32_t nbx               = get_arg_val<uint32_t>(14);
    const uint32_t nby               = get_arg_val<uint32_t>(15);
    const uint32_t route_dram        = get_arg_val<uint32_t>(16);
    const uint32_t route_pgsz        = get_arg_val<uint32_t>(17);
    const uint32_t max_overlaps_half = get_arg_val<uint32_t>(18);
    const uint32_t barrier_sem_id    = get_arg_val<uint32_t>(19);
    const uint32_t coord_noc_x       = get_arg_val<uint32_t>(20);
    const uint32_t coord_noc_y       = get_arg_val<uint32_t>(21);
    const uint32_t phase2_go_sem_b   = get_arg_val<uint32_t>(22);
    const uint32_t phase2_go_sem_n   = get_arg_val<uint32_t>(23);
    const uint32_t n_tiles_owned     = get_arg_val<uint32_t>(24);

    // ── L1 layout ─────────────────────────────────────────────────────────────
    uint8_t* base = reinterpret_cast<uint8_t*>(get_write_ptr(CB_SCRATCH));
    uint32_t off  = 0u;

    // tile_to_core[M*N]: uint16, immediately at base.
    uint16_t* tile_to_core = reinterpret_cast<uint16_t*>(base + off);
    off += tile_map_bytes;
    off  = (off + 63u) & ~63u;

    // NOC coord table: 2 × uint32_t per core (noc_x, noc_y), stored right
    // after tile_to_core in the same DRAM page / L1 region.
    const uint32_t noc_table_bytes = nc_all * 2u * sizeof(uint32_t);
    uint32_t* noc_table = reinterpret_cast<uint32_t*>(base + off);
    off += noc_table_bytes;
    off  = (off + 63u) & ~63u;

    // Cell chunk buffers: 4 SoA arrays (px, py, sx, sy), 1024 floats each.
    float* cell_px = reinterpret_cast<float*>(base + off); off += TILE_BYTES;
    float* cell_py = reinterpret_cast<float*>(base + off); off += TILE_BYTES;
    float* cell_sx = reinterpret_cast<float*>(base + off); off += TILE_BYTES;
    float* cell_sy = reinterpret_cast<float*>(base + off); off += TILE_BYTES;
    off  = (off + 63u) & ~63u;

    // Per-receiver staging: BRISC_STAGE_CAP vectors per receiver + counts.
    V12Overlap* staging       = reinterpret_cast<V12Overlap*>(base + off);
    off += nc_all * BRISC_STAGE_CAP * sizeof(V12Overlap);
    off  = (off + 3u) & ~3u;
    uint32_t* staging_count   = reinterpret_cast<uint32_t*>(base + off);
    off += nc_all * sizeof(uint32_t);
    uint32_t* dram_off_vecs   = reinterpret_cast<uint32_t*>(base + off);
    off += nc_all * sizeof(uint32_t);
    off  = (off + 63u) & ~63u;

    // Header scratch for per-page header writes (V12_HDR_BYTES = 32 B).
    uint32_t* hdr_scratch = reinterpret_cast<uint32_t*>(base + off);
    off += V12_HDR_BYTES;
    off  = (off + 63u) & ~63u;

    // -- Phase 2 private buffers (past NCRISC scatter region which is sized
    //    identically to the BRISC scatter region).
    constexpr uint32_t N_STAGE_CAP = 4u;
    // Skip NCRISC scatter staging: same sized regions as BRISC.
    off += nc_all * N_STAGE_CAP * sizeof(V12Overlap); off = (off + 3u) & ~3u;
    off += nc_all * sizeof(uint32_t);  // ncrisc staging_count
    off += nc_all * sizeof(uint32_t);  // ncrisc dram_off_vecs
    off  = (off + 63u) & ~63u;
    off += V12_HDR_BYTES;              // ncrisc hdr_scratch
    off  = (off + 63u) & ~63u;
    // NCRISC private cell buffers (ncell_px/py/sx/sy).
    off += 4u * TILE_BYTES;
    off  = (off + 63u) & ~63u;

    // Phase 2 BRISC accum buffers.
    constexpr uint32_t HDR_STRIDE  = V12_HDR_BYTES;
    constexpr uint32_t OV_STAGE_CAP= 8u;
    uint32_t* inbound_hdrs  = reinterpret_cast<uint32_t*>(base + off);
    off += nc_all * HDR_STRIDE;
    off  = (off + 63u) & ~63u;

    V12Overlap* ov_stage = reinterpret_cast<V12Overlap*>(base + off);
    off += OV_STAGE_CAP * sizeof(V12Overlap);
    off  = (off + 63u) & ~63u;

    uint16_t* ox_scratch = reinterpret_cast<uint16_t*>(base + off);
    off += TILE_BF16_SZ;
    uint16_t* oy_scratch = reinterpret_cast<uint16_t*>(base + off);
    off += TILE_BF16_SZ;
    off  = (off + 63u) & ~63u;

    // ── Load tile_to_core + NOC table from DRAM ───────────────────────────────
    // DRAM layout: [tile_to_core: tile_map_bytes][noc_table: noc_table_bytes]
    // L1 layout has 64B alignment between them — use two separate NOC reads.
    {
        DeviceZoneScopedN("V12B-MAP-LOAD");
        const InterleavedAddrGen<true> mgen = {
            .bank_base_address = tile_map_dram,
            .page_size         = tile_map_pgsz,
        };
        uint64_t dram_base = mgen.get_noc_addr(0);
        // Read tile_to_core.
        noc_async_read(dram_base,
                       reinterpret_cast<uint32_t>(tile_to_core),
                       tile_map_bytes);
        // Read noc_table (immediately after tile_to_core in DRAM).
        noc_async_read(dram_base + (uint64_t)tile_map_bytes,
                       reinterpret_cast<uint32_t>(noc_table),
                       noc_table_bytes);
        noc_async_read_barrier();
    }

    // Init bookkeeping.
    for (uint32_t i = 0; i < nc_all; ++i) {
        staging_count[i] = 0u;
        dram_off_vecs[i] = 0u;
    }

    const InterleavedAddrGen<true> rgen = {
        .bank_base_address = route_dram,
        .page_size         = route_pgsz,
    };

    // BRISC data region within each route page starts after header + NCRISC half.
    const uint32_t brisc_data_off = V12_HDR_BYTES
                                  + max_overlaps_half * (uint32_t)sizeof(V12Overlap);

    // Flush helper.
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
                     + (uint64_t)brisc_data_off
                     + (uint64_t)already * sizeof(V12Overlap);
        uint32_t src_l1    = reinterpret_cast<uint32_t>(&staging[recv * BRISC_STAGE_CAP]);
        uint32_t write_bytes = (to_write * (uint32_t)sizeof(V12Overlap) + 31u) & ~31u;
        noc_async_write(src_l1, dst, write_bytes);
        dram_off_vecs[recv] = already + to_write;
        staging_count[recv] = 0u;
    };

    const InterleavedAddrGen<true> gen_px = { .bank_base_address = addr_px, .page_size = tile_pgsz };
    const InterleavedAddrGen<true> gen_py = { .bank_base_address = addr_py, .page_size = tile_pgsz };
    const InterleavedAddrGen<true> gen_sx = { .bank_base_address = addr_sx, .page_size = tile_pgsz };
    const InterleavedAddrGen<true> gen_sy = { .bank_base_address = addr_sy, .page_size = tile_pgsz };

    // ── PHASE 1: Scatter ───────────────────────────────────────────────────────
    {
        DeviceZoneScopedN("V12B-SCATTER");
        for (uint32_t t = 0; t < n_soa_tiles; ++t) {
            uint32_t page_id = first_tile + t;
            noc_async_read_page(page_id, gen_px, reinterpret_cast<uint32_t>(cell_px));
            noc_async_read_page(page_id, gen_py, reinterpret_cast<uint32_t>(cell_py));
            noc_async_read_page(page_id, gen_sx, reinterpret_cast<uint32_t>(cell_sx));
            noc_async_read_page(page_id, gen_sy, reinterpret_cast<uint32_t>(cell_sy));
            noc_async_read_barrier();

            // First half [0..511].
            for (uint32_t ci = 0u; ci < TILE_ELEMS / 2u; ++ci) {
                float cx  = cell_px[ci];
                float cy  = cell_py[ci];
                float csx = cell_sx[ci];
                float csy = cell_sy[ci];
                if (csx <= 0.0f || csy <= 0.0f) continue;

                float half_sx = csx * 0.5f;
                float half_sy = csy * 0.5f;
                float cx_lo = cx - half_sx, cx_hi = cx + half_sx;
                float cy_lo = cy - half_sy, cy_hi = cy + half_sy;

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
                        staging[owner * BRISC_STAGE_CAP + cnt] = ov_vec;
                        staging_count[owner] = cnt + 1u;
                        if (staging_count[owner] >= BRISC_STAGE_CAP) flush_recv(owner);
                    }
                }
            }
        }
    }

    // Final flush.
    {
        DeviceZoneScopedN("V12B-FLUSH");
        for (uint32_t r = 0u; r < nc_all; ++r) flush_recv(r);
        noc_async_write_barrier();
    }

    // Write BRISC count to the BRISC header region (bytes 32..63 of the page).
    // NCRISC concurrently writes bytes 0..31 (its own region), so both writes
    // are to non-overlapping 32-byte aligned regions — no race condition.
    {
        DeviceZoneScopedN("V12B-HDR-WRITE");
        hdr_scratch[0] = 0u;  // will be overwritten per-receiver below
        for (uint32_t i = 1u; i < 8u; ++i) hdr_scratch[i] = 0u;
        for (uint32_t r = 0u; r < nc_all; ++r) {
            hdr_scratch[0] = dram_off_vecs[r];  // count_b
            uint64_t page_base = rgen.get_noc_addr(my_core_id * nc_all + r);
            // Write 32 bytes to byte-offset 32 (BRISC region; 32-byte aligned).
            noc_async_write(reinterpret_cast<uint32_t>(hdr_scratch),
                            page_base + (uint64_t)V12_HDR_BYTES / 2u,
                            V12_HDR_BYTES / 2u);
            noc_async_write_barrier();
        }
    }

    // ── BARRIER: all 2*nc_all participants converge ───────────────────────────
    {
        DeviceZoneScopedN("V12B-BARRIER");

        volatile tt_l1_ptr uint32_t* my_go_sem =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(phase2_go_sem_b));

        if (my_core_id == 0u) {
            // Coordinator: also counts itself (increment own barrier_sem directly).
            volatile tt_l1_ptr uint32_t* bar_sem =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(barrier_sem_id));
            noc_semaphore_inc(
                get_noc_addr(coord_noc_x, coord_noc_y, get_semaphore(barrier_sem_id)),
                1u);
            // Wait for all 2*nc_all increments (including our own).
            noc_semaphore_wait(bar_sem, 2u * nc_all);
            noc_semaphore_set(bar_sem, 0u);

            // Broadcast phase2 go to every core.
            uint32_t go_val = 1u;
            uint32_t go_val_l1 = reinterpret_cast<uint32_t>(&go_val);
            // Build a local L1 word holding 1u — use hdr_scratch[7] as scratch.
            hdr_scratch[7] = 1u;
            uint32_t go_src_l1 = reinterpret_cast<uint32_t>(&hdr_scratch[7]);

            for (uint32_t c = 0u; c < nc_all; ++c) {
                uint32_t cx = noc_table[c * 2u + 0u];
                uint32_t cy = noc_table[c * 2u + 1u];
                // Signal BRISC go.
                noc_semaphore_set_remote(
                    go_src_l1,
                    get_noc_addr(cx, cy, get_semaphore(phase2_go_sem_b)));
                // Signal NCRISC go.
                noc_semaphore_set_remote(
                    go_src_l1,
                    get_noc_addr(cx, cy, get_semaphore(phase2_go_sem_n)));
            }
            noc_async_write_barrier();
        } else {
            // Non-coordinator: increment coordinator's barrier semaphore.
            noc_semaphore_inc(
                get_noc_addr(coord_noc_x, coord_noc_y, get_semaphore(barrier_sem_id)),
                1u);
        }

        // All cores (including coordinator) wait for their own go signal.
        noc_semaphore_wait(my_go_sem, 1u);
        noc_semaphore_set(my_go_sem, 0u);
    }

    // ── PHASE 2: Accumulate (BRISC side) ─────────────────────────────────────
    if (n_tiles_owned == 0u) return;

    // Load all inbound page headers (from all nc_all writers targeting this core).
    {
        DeviceZoneScopedN("V12B2-HDR-LOAD");
        for (uint32_t w = 0u; w < nc_all; ++w) {
            uint64_t page  = rgen.get_noc_addr(w * nc_all + my_core_id);
            uint32_t dst   = reinterpret_cast<uint32_t>(inbound_hdrs) + w * HDR_STRIDE;
            noc_async_read(page, dst, HDR_STRIDE);
        }
        noc_async_read_barrier();
    }

    const uint32_t ncrisc_data_off = V12_HDR_BYTES;
    const uint32_t brisc_data_off2 = V12_HDR_BYTES + max_overlaps_half * (uint32_t)sizeof(V12Overlap);

    // Reuse ox_scratch + oy_scratch memory (combined 4096 bytes) as FP32 density tile.
    // ox_scratch = 2048 bytes + oy_scratch = 2048 bytes = 4096 bytes = 1024 floats.
    // BRISC accumulates directly in FP32 (no BF16/FPU TRISC involved), then pushes
    // the result to CB_ACCUM for NCRISC to read and write to DRAM.
    float* density_tile = reinterpret_cast<float*>(ox_scratch);

    for (uint32_t tile_local = 0u; tile_local < n_tiles_owned; ++tile_local) {
        DeviceZoneScopedN("V12B2-TILE");

        uint32_t tile_idx = get_arg_val<uint32_t>(25u + tile_local);
        uint32_t tx = tile_idx / N_tiles;
        uint32_t ty = tile_idx % N_tiles;

        // Zero the density tile for this tile group.
        for (uint32_t i = 0u; i < TILE_FLOATS; ++i) density_tile[i] = 0.0f;

        for (uint32_t w = 0u; w < nc_all; ++w) {
            const uint32_t* hdr = inbound_hdrs + w * (HDR_STRIDE / sizeof(uint32_t));
            // hdr[0] = count_n (NCRISC region, offset 0)
            // hdr[8] = count_b (BRISC region, offset 32 = 8 uint32s)
            uint32_t cnts[2] = { hdr[0], hdr[8] };
            uint32_t offs[2] = { ncrisc_data_off, brisc_data_off2 };

            for (uint32_t half = 0u; half < 2u; ++half) {
                uint32_t cnt = cnts[half];
                if (cnt > max_overlaps_half) cnt = max_overlaps_half;
                if (cnt == 0u) continue;

                uint64_t page    = rgen.get_noc_addr(w * nc_all + my_core_id);
                uint32_t vec_off = 0u;

                while (vec_off < cnt) {
                    uint32_t chunk = cnt - vec_off;
                    if (chunk > OV_STAGE_CAP) chunk = OV_STAGE_CAP;
                    uint64_t src = page
                                 + (uint64_t)offs[half]
                                 + (uint64_t)vec_off * sizeof(V12Overlap);
                    noc_async_read(src,
                                   reinterpret_cast<uint32_t>(ov_stage),
                                   chunk * (uint32_t)sizeof(V12Overlap));
                    noc_async_read_barrier();

                    for (uint32_t vi = 0u; vi < chunk; ++vi) {
                        const V12Overlap& ov = ov_stage[vi];
                        int32_t  ibxl = (int32_t)(uint32_t)ov.bxl;
                        int32_t  ibyl = (int32_t)(uint32_t)ov.byl;
                        uint32_t wx   = (uint32_t)ov.wx;
                        uint32_t wy   = (uint32_t)ov.wy;
                        if (wx == 0u || wy == 0u) continue;

                        // Direct FP32 accumulation: density_tile[bxw][byw] += ovx * ovy.
                        // Only accumulate bins that fall in this tile (tx, ty).
                        for (uint32_t j = 0u; j < wx; ++j) {
                            uint32_t bx = (uint32_t)(ibxl + (int32_t)j);
                            if ((bx >> 5u) != tx) continue;
                            uint32_t bxw = bx & 31u;
                            float ovx = ov.overlap_x[j];
                            if (ovx == 0.0f) continue;
                            for (uint32_t k = 0u; k < wy; ++k) {
                                uint32_t by = (uint32_t)(ibyl + (int32_t)k);
                                if ((by >> 5u) != ty) continue;
                                uint32_t byw = by & 31u;
                                density_tile[bxw * TILE_W + byw] += ovx * ov.overlap_y[k];
                            }
                        }
                    }
                    vec_off += chunk;
                }
            }
        }

        // Push accumulated FP32 density tile to CB_ACCUM for NCRISC.
        cb_reserve_back(CB_ACCUM, 1u);
        {
            float* dst = reinterpret_cast<float*>(get_write_ptr(CB_ACCUM));
            for (uint32_t i = 0u; i < TILE_FLOATS; ++i) dst[i] = density_tile[i];
        }
        cb_push_back(CB_ACCUM, 1u);
    }
}
