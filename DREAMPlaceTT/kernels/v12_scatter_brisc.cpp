// SPDX-License-Identifier: Apache-2.0
//
// V12 BRISC Scatter Kernel (RISCV_0, NOC_0) — Phase 1.
//
// Reads the FIRST HALF of cell tiles directly from DRAM (no SFPU pipeline),
// computes FP32 overlap_x[0..7] / overlap_y[0..7] for each cell using a
// scalar RISC-V fast path identical in precision to v4_compute.cpp (same
// correct_bin_idx logic), and routes per-cell V12Overlap vectors to the
// route_buf DRAM page belonging to the tile-owning core.
//
// After all scatter writes are committed, signals the on-chip all-cores
// barrier. Once the barrier fires, transitions to Phase 2 by waiting for
// a signal from v12_accum_brisc logic (which is the same kernel continuing
// after the barrier — same RISC processes Phase 1 then Phase 2).
//
// Geometry constants injected as JIT #defines (same as v4_compute.cpp):
//   V12_BSX_F, V12_BSY_F, V12_INV_BSX_F, V12_INV_BSY_F, V12_XL_F, V12_YL_F
//
// V12Overlap struct (80 bytes, multiple of 32 for NOC alignment):
//   float overlap_x[8]  (32 B)
//   float overlap_y[8]  (32 B)
//   uint16_t bxl, byl   (4 B)
//   uint8_t  wx, wy     (2 B)
//   uint8_t  pad[10]    (10 B) — zero-filled, brings total to 48 B
// Wait — actual struct with 8+8 floats + 4 + 2 + 10 = 80 bytes.
//
// Route buffer page layout (per (writer, receiver) pair):
//   [uint32_t count_b][28B pad] = 32B header for NOC cache-line alignment
//   [V12Overlap[0..count_b-1]]   BRISC half (indices 0..MAX_OVERLAPS_HALF-1)
//   (NCRISC half follows at +MAX_OVERLAPS_HALF*80 within the same page)
//
// Runtime args:
//   0:  addr_px          DRAM base of px SoA
//   1:  addr_py          DRAM base of py SoA
//   2:  addr_sx          DRAM base of sx SoA
//   3:  addr_sy          DRAM base of sy SoA
//   4:  tile_pgsz        bytes per SoA page (4096)
//   5:  first_tile       first SoA tile index for this core
//   6:  n_soa_tiles      number of SoA tiles this core processes
//   7:  tile_map_dram    DRAM base of tile_to_core[] (uint16 per grid tile)
//   8:  tile_map_bytes   bytes to load (total_tiles * 2, padded)
//   9:  tile_map_pgsz    page size of tile_map_buf
//  10:  my_core_id       linear core id (0..nc_all-1)
//  11:  nc_all           total number of Tensix cores
//  12:  M_tiles          density grid width in 32-wide tiles
//  13:  N_tiles          density grid height in 32-wide tiles
//  14:  nbx              density grid width in bins (= M)
//  15:  nby              density grid height in bins (= N)
//  16:  route_dram       DRAM base of route_buf
//  17:  route_pgsz       bytes per (writer, receiver) route page
//  18:  max_overlaps_half  max V12Overlap vectors BRISC may write per page half
//  19:  barrier_sem_id   semaphore used for all-cores Phase 1→2 barrier
//                        (BRISC increments coordinator, then waits for go)
//  20:  coord_noc_x      NOC X of barrier coordinator core (core 0)
//  21:  coord_noc_y      NOC Y of barrier coordinator core
//  22:  phase2_go_sem_id semaphore BRISC waits on after incrementing coordinator
//  23:  inv_ba_u32       inv_bin_area as uint32 bitcast (used in Phase 2 scale)
//
// CB usage: NONE in Phase 1 (BRISC reads DRAM directly without TRISC pipeline)
// CB c_24 is used as scratch for L1 layout (same as V11 pattern).

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
static constexpr uint32_t CB_SCRATCH   = tt::CBIndex::c_24;

// V12Overlap: 80 bytes, 32B aligned.
struct V12Overlap {
    float    overlap_x[8];   // 32 B
    float    overlap_y[8];   // 32 B
    uint16_t bxl;            //  2 B
    uint16_t byl;            //  2 B
    uint8_t  wx;             //  1 B
    uint8_t  wy;             //  1 B
    uint8_t  pad[10];        // 10 B  (zero-filled, NOC-alignment pad)
};
static_assert(sizeof(V12Overlap) == 80u, "V12Overlap must be 80 bytes");

// Route page header: 32 bytes.
// count_b at offset 0; NCRISC's count_n is at offset 8 in the same header
// (written by NCRISC kernel — BRISC never touches that word).
// 28 bytes of padding follow for NOC cache-line alignment.
static constexpr uint32_t V12_HDR_BYTES = 32u;

// correct_bin_idx: same precision guarantee as v4_compute.cpp SFPU version.
// Returns bin index such that XL + idx*bs <= pos < XL + (idx+1)*bs.
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
    // arg 23 (inv_ba_u32) used by the accum phase (v12_accum_brisc) — not used here.

    // ── L1 layout (CB_SCRATCH = c_24) ────────────────────────────────────────
    uint8_t* base = reinterpret_cast<uint8_t*>(get_write_ptr(CB_SCRATCH));
    uint32_t off = 0u;

    // tile_to_core[]: uint16, one per density grid tile
    uint16_t* tile_to_core = reinterpret_cast<uint16_t*>(base + off);
    off += tile_map_bytes;
    off  = (off + 63u) & ~63u;

    // Cell chunk buffers: 4 SOA arrays × TILE_ELEMS floats (px, py, sx, sy)
    float* cell_px = reinterpret_cast<float*>(base + off); off += TILE_BYTES;
    float* cell_py = reinterpret_cast<float*>(base + off); off += TILE_BYTES;
    float* cell_sx = reinterpret_cast<float*>(base + off); off += TILE_BYTES;
    float* cell_sy = reinterpret_cast<float*>(base + off); off += TILE_BYTES;
    off  = (off + 63u) & ~63u;

    // Per-receiver staging: MAX_STAGE vectors per receiver + count + dram_offset
    // BRISC_STAGE_CAP = 4 vectors per receiver to keep L1 modest.
    constexpr uint32_t BRISC_STAGE_CAP = 4u;
    V12Overlap* staging = reinterpret_cast<V12Overlap*>(base + off);
    // staging[recv * BRISC_STAGE_CAP + slot]
    off += nc_all * BRISC_STAGE_CAP * sizeof(V12Overlap);
    off  = (off + 3u) & ~3u;

    uint32_t* staging_count  = reinterpret_cast<uint32_t*>(base + off);
    off += nc_all * sizeof(uint32_t);
    uint32_t* dram_off_vecs  = reinterpret_cast<uint32_t*>(base + off);
    off += nc_all * sizeof(uint32_t);
    off  = (off + 63u) & ~63u;

    // ── Step 0: load tile_to_core from DRAM ──────────────────────────────────
    {
        DeviceZoneScopedN("V12B-MAP-LOAD");
        const InterleavedAddrGen<true> mgen = {
            .bank_base_address = tile_map_dram,
            .page_size         = tile_map_pgsz,
        };
        noc_async_read(mgen.get_noc_addr(0),
                       reinterpret_cast<uint32_t>(tile_to_core),
                       tile_map_bytes);
        noc_async_read_barrier();
    }

    // ── Init per-receiver bookkeeping ─────────────────────────────────────────
    for (uint32_t i = 0; i < nc_all; ++i) {
        staging_count[i] = 0u;
        dram_off_vecs[i] = 0u;
    }

    // route_buf address generator
    const InterleavedAddrGen<true> rgen = {
        .bank_base_address = route_dram,
        .page_size         = route_pgsz,
    };

    // my_writer_id for BRISC = my_core_id (BRISC owns the second half of each
    // route page; NCRISC owns the first half — they use the same writer page
    // slot but different byte offsets within the page).
    // BRISC data starts at: V12_HDR_BYTES + max_overlaps_half * 80
    const uint32_t brisc_data_off = V12_HDR_BYTES + max_overlaps_half * (uint32_t)sizeof(V12Overlap);

    // Flush helper: write staging[recv] to DRAM if staging_count[recv] > 0.
    auto flush_recv = [&](uint32_t recv) {
        uint32_t cnt = staging_count[recv];
        if (cnt == 0u) return;

        uint32_t already = dram_off_vecs[recv];
        if (already >= max_overlaps_half) { staging_count[recv] = 0u; return; }
        uint32_t to_write = cnt;
        if (already + to_write > max_overlaps_half)
            to_write = max_overlaps_half - already;

        // page index: writer=my_core_id (BRISC reuses NCRISC's writer slot),
        // so page_idx = my_core_id * nc_all + recv
        uint64_t page_base = rgen.get_noc_addr(my_core_id * nc_all + recv);
        uint64_t dst = page_base
                     + (uint64_t)brisc_data_off
                     + (uint64_t)already * sizeof(V12Overlap);
        uint32_t src_l1 = reinterpret_cast<uint32_t>(&staging[recv * BRISC_STAGE_CAP]);
        // Size must be multiple of 32 for NOC alignment; V12Overlap = 80B,
        // so to_write * 80 is always a multiple of 80. For 32B alignment pad
        // to next multiple of 32 (80 = 2.5 × 32; groups of 2 vectors = 160B
        // which is 5 × 32). We allow partial writes by rounding up to 32B.
        uint32_t write_bytes = (to_write * (uint32_t)sizeof(V12Overlap) + 31u) & ~31u;
        noc_async_write(src_l1, dst, write_bytes);
        dram_off_vecs[recv] = already + to_write;
        staging_count[recv] = 0u;
    };

    // ── DRAM address generators for cell SoA ─────────────────────────────────
    const InterleavedAddrGen<true> gen_px = { .bank_base_address = addr_px, .page_size = tile_pgsz };
    const InterleavedAddrGen<true> gen_py = { .bank_base_address = addr_py, .page_size = tile_pgsz };
    const InterleavedAddrGen<true> gen_sx = { .bank_base_address = addr_sx, .page_size = tile_pgsz };
    const InterleavedAddrGen<true> gen_sy = { .bank_base_address = addr_sy, .page_size = tile_pgsz };

    // ── Phase 1: scatter (BRISC processes FIRST HALF of each SoA tile) ───────
    // Each SoA tile holds TILE_ELEMS=1024 cells. BRISC takes cells [0..511],
    // NCRISC takes cells [512..1023] (parallel, independent NOC channels).
    {
        DeviceZoneScopedN("V12B-SCATTER");

        for (uint32_t t = 0; t < n_soa_tiles; ++t) {
            uint32_t page_id = first_tile + t;

            // Load the full SoA tile into L1 (BRISC reads on NOC_0).
            noc_async_read_page(page_id, gen_px, reinterpret_cast<uint32_t>(cell_px));
            noc_async_read_page(page_id, gen_py, reinterpret_cast<uint32_t>(cell_py));
            noc_async_read_page(page_id, gen_sx, reinterpret_cast<uint32_t>(cell_sx));
            noc_async_read_page(page_id, gen_sy, reinterpret_cast<uint32_t>(cell_sy));
            noc_async_read_barrier();

            // Process first half [0..511].
            for (uint32_t ci = 0u; ci < TILE_ELEMS / 2u; ++ci) {
                float cx  = cell_px[ci];
                float cy  = cell_py[ci];
                float csx = cell_sx[ci];
                float csy = cell_sy[ci];

                // Zero-area cells (padding cells at end of last tile) — skip.
                if (csx <= 0.0f || csy <= 0.0f) continue;

                float half_sx = csx * 0.5f;
                float half_sy = csy * 0.5f;
                float cx_lo = cx - half_sx;
                float cx_hi = cx + half_sx;
                float cy_lo = cy - half_sy;
                float cy_hi = cy + half_sy;

                // ── X overlap ────────────────────────────────────────────
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

                // ── Y overlap ────────────────────────────────────────────
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

                // ── Build V12Overlap and route to tile owners ─────────────
                // A cell overlaps tiles [tx_lo..tx_hi] x [ty_lo..ty_hi].
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

                        // Build overlap vector for this tile.
                        // For bins that fall in this tile only — include only
                        // j values where (bxl+j)>>5 == tx, same for k and ty.
                        V12Overlap ov_vec;
                        // Zero-init pad and overlap arrays.
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

                        // Stage into per-receiver buffer.
                        uint32_t cnt = staging_count[owner];
                        staging[owner * BRISC_STAGE_CAP + cnt] = ov_vec;
                        cnt++;
                        staging_count[owner] = cnt;
                        if (cnt >= BRISC_STAGE_CAP) {
                            flush_recv(owner);
                        }
                    }
                }
            }  // for ci
        }  // for t (soa tiles)
    }  // V12B-SCATTER zone

    // ── Final flush of all receivers ──────────────────────────────────────────
    {
        DeviceZoneScopedN("V12B-FLUSH");
        for (uint32_t r = 0u; r < nc_all; ++r) flush_recv(r);
        noc_async_write_barrier();
    }

    // ── Write BRISC header word: count at byte 4 of the page header ──────────
    // Header layout: [count_n uint32 at 0][count_b uint32 at 4][24B pad]
    // BRISC writes dram_off_vecs[r] (its contribution count) to offset 4.
    {
        DeviceZoneScopedN("V12B-HDR-WRITE");
        // Use a small L1 scratch for the 32-byte header write.
        uint32_t* hdr_scratch = reinterpret_cast<uint32_t*>(base + off);
        for (uint32_t r = 0u; r < nc_all; ++r) {
            hdr_scratch[0] = 0u;                  // count_n placeholder (NCRISC fills this)
            hdr_scratch[1] = dram_off_vecs[r];    // count_b
            for (uint32_t i = 2u; i < 8u; ++i) hdr_scratch[i] = 0u;
            uint64_t page_base = rgen.get_noc_addr(my_core_id * nc_all + r);
            // Write only the second 4-byte word (count_b at offset 4).
            // To keep NOC writes 32B aligned, write the full 32B header.
            noc_async_write(reinterpret_cast<uint32_t>(hdr_scratch),
                            page_base, V12_HDR_BYTES);
            noc_async_write_barrier();
        }
    }

    // ── All-cores Phase 1→2 barrier ──────────────────────────────────────────
    // BRISC increments the coordinator's barrier semaphore, then waits for
    // the coordinator to broadcast the "phase 2 go" signal.
    {
        DeviceZoneScopedN("V12B-BARRIER");
        uint64_t coord_sem_addr = get_noc_addr(coord_noc_x, coord_noc_y,
                                               get_semaphore(barrier_sem_id));
        noc_semaphore_inc(coord_sem_addr, 1u);

        volatile tt_l1_ptr uint32_t* go_sem =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(phase2_go_sem_id));
        noc_semaphore_wait(go_sem, 1u);
        // Reset for next iteration.
        noc_semaphore_set(go_sem, 0u);
    }

    // Phase 2 (accumulate) is handled by v12_accum_brisc.cpp, which is a
    // separate kernel on BRISC in the same Program. Control passes to it
    // after this kernel exits.
}
