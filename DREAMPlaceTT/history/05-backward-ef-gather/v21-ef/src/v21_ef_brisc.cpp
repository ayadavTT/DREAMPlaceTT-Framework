// SPDX-License-Identifier: Apache-2.0
//
// V21a electric_force — BRISC kernel (minimal-but-correct for the 1024-cells-
// per-tile linear-layout design).
//
// MINIMAL DESIGN: each per-core launch pushes ONE batch of tiles for the SFPU:
//   cb_px:        8 tiles (one per k=0..7)
//   cb_py:        8 tiles (one per h=0..7)
//   cb_neg_ratio: 1 tile
//   cb_fx:        64 tiles (kk*8 + hh order) — ALL before any cb_fy
//   cb_fy:        64 tiles (kk*8 + hh order)
//
// Linear-layout: tile[c] = value (cell c at element c). UnpackToDestFp32 makes
// this end-to-end transparent (proven by tt-sfpu/bin_index_compute at 500K cells).
//
// Single 256-cell BRISC sub-batch per launch (fits 128 KB scratch). Each SFPU
// tile has 256 real elements + 768 zero-padding.

#define V21_CB_SCRATCH (tt::CBIndex::c_24)

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"
#include "api/debug/dprint.h"

#ifndef V21_USE_SFPU
#include "v21_ef_common.h"
void kernel_main() { v21_ef_main(); }
#else

#ifndef V21S_BATCH_OVERRIDE
#define V21S_BATCH_OVERRIDE 512
#endif
constexpr uint32_t V21S_BATCH            = V21S_BATCH_OVERRIDE;
constexpr uint32_t V21S_TILE_CELLS       = 1024;
constexpr uint32_t V21S_MAX_KH           = 8;  // array allocation bound (always 8)
// Adaptive work bounds: host computes the global max bin span and passes these so
// the kk/hh loops only cover bins cells actually touch (default 8 = V21 behaviour).
#ifndef V21S_MAX_K
#define V21S_MAX_K 8
#endif
#ifndef V21S_MAX_H
#define V21S_MAX_H 8
#endif
constexpr uint32_t MAX_K = V21S_MAX_K;
constexpr uint32_t MAX_H = V21S_MAX_H;
constexpr uint32_t V21S_DRAM_READ_ALIGN  = 64;
constexpr uint32_t V21S_POS_PG_BYTES     = 4096;
constexpr uint32_t V21S_POS_FLOATS_PER_PG = V21S_POS_PG_BYTES / 4;
constexpr uint32_t V21S_READ_CHUNK       = 128;  // up to 2 reads/cell → 256 in-flight max

constexpr uint32_t CB_PX        = tt::CBIndex::c_0;
constexpr uint32_t CB_PY        = tt::CBIndex::c_1;
constexpr uint32_t CB_FX        = tt::CBIndex::c_2;
constexpr uint32_t CB_FY        = tt::CBIndex::c_3;
constexpr uint32_t CB_NEG_RATIO = tt::CBIndex::c_4;

void kernel_main() {
    const uint32_t my_core_idx        = get_arg_val<uint32_t>(0);
    const uint32_t cells_start        = get_arg_val<uint32_t>(1);
    const uint32_t cells_end          = get_arg_val<uint32_t>(2);
    const uint32_t cells_start_global = get_arg_val<uint32_t>(3);
    const uint32_t num_nodes          = get_arg_val<uint32_t>(4);
    const uint32_t num_bins_x         = get_arg_val<uint32_t>(5);
    const uint32_t num_bins_y         = get_arg_val<uint32_t>(6);
    const uint32_t field_x_base       = get_arg_val<uint32_t>(7);
    const uint32_t field_y_base       = get_arg_val<uint32_t>(8);
    const uint32_t field_pg_bytes     = get_arg_val<uint32_t>(9);
    const uint32_t pos_base           = get_arg_val<uint32_t>(10);
    const uint32_t const_base         = get_arg_val<uint32_t>(11);
    const uint32_t const_pg_bytes     = get_arg_val<uint32_t>(12);
    union { uint32_t u; float f; } cvt;
    cvt.u = get_arg_val<uint32_t>(15); const float xl      = cvt.f;
    cvt.u = get_arg_val<uint32_t>(16); const float yl      = cvt.f;
    cvt.u = get_arg_val<uint32_t>(17); const float bsx     = cvt.f;
    cvt.u = get_arg_val<uint32_t>(18); const float bsy     = cvt.f;
    cvt.u = get_arg_val<uint32_t>(19); const float inv_bsx = cvt.f;
    cvt.u = get_arg_val<uint32_t>(20); const float inv_bsy = cvt.f;
    const uint32_t sem_brisc_done_id  = get_arg_val<uint32_t>(21);
    const uint32_t sem_ncrisc_done_id = get_arg_val<uint32_t>(22);

    volatile tt_l1_ptr uint32_t* sem_brisc_done =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_brisc_done_id));
    volatile tt_l1_ptr uint32_t* sem_ncrisc_done =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_ncrisc_done_id));

    uint8_t* base = reinterpret_cast<uint8_t*>(get_write_ptr(V21_CB_SCRATCH));
    uint32_t off = 0u;
    auto reserve = [&](uint32_t bytes) -> uint8_t* {
        uint8_t* p = base + off;
        off += (bytes + 63u) & ~63u;
        return p;
    };
    uint32_t* const_l1     = reinterpret_cast<uint32_t*>(reserve(V21S_BATCH * 32u));
    uint8_t*  pos_x_l1     = reserve(V21S_BATCH * V21S_DRAM_READ_ALIGN);
    uint8_t*  pos_y_l1     = reserve(V21S_BATCH * V21S_DRAM_READ_ALIGN);
    // Transposed: px_l1[k * V21S_BATCH + c] — gives contiguous read for cb_px[k] tile pack.
    float*    px_l1        = reinterpret_cast<float*>(reserve(V21S_BATCH * V21S_MAX_KH * 4u));
    float*    py_l1        = reinterpret_cast<float*>(reserve(V21S_BATCH * V21S_MAX_KH * 4u));
    float*    neg_ratio_l1 = reinterpret_cast<float*>(reserve(V21S_BATCH * 4u));
    uint32_t* bin_xl_l1    = reinterpret_cast<uint32_t*>(reserve(V21S_BATCH * 4u));
    uint32_t* bin_yl_l1    = reinterpret_cast<uint32_t*>(reserve(V21S_BATCH * 4u));
    uint32_t* k_count_l1   = reinterpret_cast<uint32_t*>(reserve(V21S_BATCH * 4u));
    uint32_t* h_count_l1   = reinterpret_cast<uint32_t*>(reserve(V21S_BATCH * 4u));
    // Field staging: 128 B per cell (2× align, handles bin_yl+h_count crossing 64-B boundary).
    constexpr uint32_t V21S_FIELD_STAGE_BYTES = 2u * V21S_DRAM_READ_ALIGN;
    uint8_t*  fx_stage     = reserve(V21S_BATCH * V21S_FIELD_STAGE_BYTES);
    uint8_t*  fy_stage     = reserve(V21S_BATCH * V21S_FIELD_STAGE_BYTES);

    const InterleavedAddrGen<true> pos_pgen     = {.bank_base_address = pos_base,     .page_size = V21S_POS_PG_BYTES};
    const InterleavedAddrGen<true> field_x_pgen = {.bank_base_address = field_x_base, .page_size = field_pg_bytes};
    const InterleavedAddrGen<true> field_y_pgen = {.bank_base_address = field_y_base, .page_size = field_pg_bytes};
    const InterleavedAddrGen<true> const_pgen   = {.bank_base_address = const_base,   .page_size = const_pg_bytes};
    const uint64_t my_const_base = const_pgen.get_noc_addr(my_core_idx);

    DeviceZoneScopedN("V21B-ALL");
    uint32_t cells_done = cells_start;
    uint32_t batch_idx = 0u;
    while (cells_done < cells_end) {
        uint32_t batch = cells_end - cells_done;
        if (batch > V21S_BATCH) batch = V21S_BATCH;

        // Parallel-prep split: BRISC owns cells [0..half), NCRISC owns [half..batch).
        const uint32_t my_start = 0u;
        const uint32_t my_end   = batch / 2u;

        // Read constants (only my half)
        {
            const uint32_t const_byte_off = (cells_done + my_start) * 32;
            const uint32_t my_cnt = my_end - my_start;
            const uint32_t const_bytes = (my_cnt * 32 + V21S_DRAM_READ_ALIGN - 1u) & ~(V21S_DRAM_READ_ALIGN - 1u);
            noc_async_read(my_const_base + const_byte_off, (uint32_t)(const_l1 + my_start * 8), const_bytes);
            noc_async_read_barrier();
        }

        // Read pos (chunked, only my half)
        for (uint32_t c0 = my_start; c0 < my_end; c0 += V21S_READ_CHUNK) {
            const uint32_t c1 = (c0 + V21S_READ_CHUNK < my_end) ? (c0 + V21S_READ_CHUNK) : my_end;
            for (uint32_t c = c0; c < c1; ++c) {
                const uint32_t g  = cells_start_global + cells_done + c;
                const uint32_t gx = g;
                const uint32_t gy = num_nodes + g;
                const uint32_t gx_pg          = gx / V21S_POS_FLOATS_PER_PG;
                const uint32_t gx_aligned_off = ((gx % V21S_POS_FLOATS_PER_PG) * 4) & ~(V21S_DRAM_READ_ALIGN - 1u);
                const uint32_t gy_pg          = gy / V21S_POS_FLOATS_PER_PG;
                const uint32_t gy_aligned_off = ((gy % V21S_POS_FLOATS_PER_PG) * 4) & ~(V21S_DRAM_READ_ALIGN - 1u);
                noc_async_read(pos_pgen.get_noc_addr(gx_pg) + gx_aligned_off,
                               (uint32_t)(pos_x_l1 + c * V21S_DRAM_READ_ALIGN), V21S_DRAM_READ_ALIGN);
                noc_async_read(pos_pgen.get_noc_addr(gy_pg) + gy_aligned_off,
                               (uint32_t)(pos_y_l1 + c * V21S_DRAM_READ_ALIGN), V21S_DRAM_READ_ALIGN);
            }
            noc_async_read_barrier();
        }

        // Scalar prep (only my half)
        { DeviceZoneScopedN("V21B-PREP");
        for (uint32_t c = my_start; c < my_end; ++c) {
            const uint32_t g       = cells_start_global + cells_done + c;
            const uint32_t gx_idx  = g;
            const uint32_t gy_idx  = num_nodes + g;
            const uint32_t gx_intra = ((gx_idx % V21S_POS_FLOATS_PER_PG) * 4) & (V21S_DRAM_READ_ALIGN - 1u);
            const uint32_t gy_intra = ((gy_idx % V21S_POS_FLOATS_PER_PG) * 4) & (V21S_DRAM_READ_ALIGN - 1u);
            const float pos_x = *reinterpret_cast<float*>(pos_x_l1 + c * V21S_DRAM_READ_ALIGN + gx_intra);
            const float pos_y = *reinterpret_cast<float*>(pos_y_l1 + c * V21S_DRAM_READ_ALIGN + gy_intra);

            cvt.u = const_l1[c * 8 + 0]; const float ox    = cvt.f;
            cvt.u = const_l1[c * 8 + 1]; const float oy    = cvt.f;
            cvt.u = const_l1[c * 8 + 2]; const float nsx   = cvt.f;
            cvt.u = const_l1[c * 8 + 3]; const float nsy   = cvt.f;
            cvt.u = const_l1[c * 8 + 4]; const float ratio = cvt.f;

            const float node_x = pos_x + ox;
            const float node_y = pos_y + oy;

            int bin_xl_i = (int)((node_x       - xl) * inv_bsx);
            int bin_xh_i = (int)((node_x + nsx - xl) * inv_bsx) + 1;
            int bin_yl_i = (int)((node_y       - yl) * inv_bsy);
            int bin_yh_i = (int)((node_y + nsy - yl) * inv_bsy) + 1;
            if (bin_xl_i < 0) bin_xl_i = 0;
            if (bin_xh_i > (int)num_bins_x) bin_xh_i = (int)num_bins_x;
            if (bin_yl_i < 0) bin_yl_i = 0;
            if (bin_yh_i > (int)num_bins_y) bin_yh_i = (int)num_bins_y;

            neg_ratio_l1[c] = -ratio;
            uint32_t k_count = (bin_xh_i > bin_xl_i) ? (uint32_t)(bin_xh_i - bin_xl_i) : 0u;
            uint32_t h_count = (bin_yh_i > bin_yl_i) ? (uint32_t)(bin_yh_i - bin_yl_i) : 0u;
            if (k_count > MAX_K) k_count = MAX_K;
            if (h_count > MAX_H) h_count = MAX_H;
            k_count_l1[c]  = k_count;
            h_count_l1[c]  = h_count;
            bin_xl_l1[c]   = (uint32_t)bin_xl_i;
            bin_yl_l1[c]   = (uint32_t)bin_yl_i;

            // Hoist invariants + use running sum: bin_x_lo += bsx per k.
            const float node_x_top = node_x + nsx;
            const float node_y_top = node_y + nsy;
            const float bin_xl_lo = xl + (float)(uint32_t)bin_xl_i * bsx;
            const float bin_yl_lo = yl + (float)(uint32_t)bin_yl_i * bsy;
#ifdef V21_PREP_FAST
            // Interior bins are fully covered → overlap = bsx (no min/max). Only the
            // first/last bin needs the clamp. Same result as the min/max form; exact
            // on integer bin grids (bsx=1). Cuts ~half the soft-float prep ops.
            for (uint32_t k = 0; k < MAX_K; ++k) {
                float px_val;
                if (k >= k_count)               px_val = 0.0f;
                else if (k + 1u == k_count && k == 0u) px_val = node_x_top - node_x;
                else if (k == 0u)               px_val = (bin_xl_lo + bsx) - node_x;
                else if (k + 1u == k_count)     px_val = node_x_top - (bin_xl_lo + (float)k * bsx);
                else                            px_val = bsx;
                px_l1[k * V21S_BATCH + c] = px_val;
            }
            for (uint32_t h = 0; h < MAX_H; ++h) {
                float py_val;
                if (h >= h_count)               py_val = 0.0f;
                else if (h + 1u == h_count && h == 0u) py_val = node_y_top - node_y;
                else if (h == 0u)               py_val = (bin_yl_lo + bsy) - node_y;
                else if (h + 1u == h_count)     py_val = node_y_top - (bin_yl_lo + (float)h * bsy);
                else                            py_val = bsy;
                py_l1[h * V21S_BATCH + c] = py_val;
            }
#else
            float bin_x_lo = bin_xl_lo;
            for (uint32_t k = 0; k < MAX_K; ++k) {
                float px_val = 0.0f;
                if (k < k_count) {
                    const float bin_x_hi = bin_x_lo + bsx;
                    const float top = (node_x_top < bin_x_hi) ? node_x_top : bin_x_hi;
                    const float bot = (node_x     > bin_x_lo) ? node_x     : bin_x_lo;
                    px_val = top - bot;
                }
                px_l1[k * V21S_BATCH + c] = px_val;  // transposed
                bin_x_lo += bsx;
            }
            float bin_y_lo = bin_yl_lo;
            for (uint32_t h = 0; h < MAX_H; ++h) {
                float py_val = 0.0f;
                if (h < h_count) {
                    const float bin_y_hi = bin_y_lo + bsy;
                    const float top = (node_y_top < bin_y_hi) ? node_y_top : bin_y_hi;
                    const float bot = (node_y     > bin_y_lo) ? node_y     : bin_y_lo;
                    py_val = top - bot;
                }
                py_l1[h * V21S_BATCH + c] = py_val;  // transposed
                bin_y_lo += bsy;
            }
#endif
        }
        }  // V21B-PREP

        // Signal BRISC prep done (counter = batch*2 + 1), wait for NCRISC prep done.
        asm volatile("fence rw, rw" ::: "memory");
        *sem_brisc_done = batch_idx * 2u + 1u;
        noc_semaphore_wait_min(sem_ncrisc_done, batch_idx * 2u + 1u);
        asm volatile("fence rw, rw" ::: "memory");


        // Pack cb_px (MAX_K tiles, one per k). tile[c] = px[c, k]; pad rest with 0.
        for (uint32_t k = 0; k < MAX_K; ++k) {
            cb_reserve_back(CB_PX, 1);
            float* tile = reinterpret_cast<float*>(get_write_ptr(CB_PX));
            // Contiguous copy thanks to transposed px_l1.
            const float* src = px_l1 + k * V21S_BATCH;
            for (uint32_t c = 0; c < batch; ++c) tile[c] = src[c];
            cb_push_back(CB_PX, 1);
        }

        // Pack cb_py (MAX_H tiles)
        for (uint32_t h = 0; h < MAX_H; ++h) {
            cb_reserve_back(CB_PY, 1);
            float* tile = reinterpret_cast<float*>(get_write_ptr(CB_PY));
            const float* src = py_l1 + h * V21S_BATCH;
            for (uint32_t c = 0; c < batch; ++c) tile[c] = src[c];
            cb_push_back(CB_PY, 1);
        }

        // Pack cb_neg_ratio (1 tile)
        cb_reserve_back(CB_NEG_RATIO, 1);
        {
            float* tile = reinterpret_cast<float*>(get_write_ptr(CB_NEG_RATIO));
            for (uint32_t c = 0; c < batch; ++c) tile[c] = neg_ratio_l1[c];
            (void)0;  // skip padding writes — L1 starts at zero per launch
        }
        cb_push_back(CB_NEG_RATIO, 1);

        // ALL 64 cb_fx tiles first. Per cell, issue ONE OR TWO 64-byte reads:
        // first chunk [aligned_off, aligned_off+64); second chunk only if the
        // cell's h-range crosses the 64-B alignment boundary. fx_stage is
        // sized 128 B per cell to hold both chunks side-by-side.
        for (uint32_t kk = 0; kk < MAX_K; ++kk) {
            for (uint32_t c0 = 0; c0 < batch; c0 += V21S_READ_CHUNK) {
                const uint32_t c1 = (c0 + V21S_READ_CHUNK < batch) ? (c0 + V21S_READ_CHUNK) : batch;
                for (uint32_t c = c0; c < c1; ++c) {
                    if (kk >= k_count_l1[c]) continue;
                    const uint32_t bin_k    = bin_xl_l1[c] + kk;
                    const uint32_t bin_yl   = bin_yl_l1[c];
                    const uint32_t h_count  = h_count_l1[c];
                    const uint32_t intra_byte_start = bin_yl * 4;
                    const uint32_t intra_byte_end   = (bin_yl + h_count) * 4;
                    const uint32_t aligned_off1     = intra_byte_start & ~(V21S_DRAM_READ_ALIGN - 1u);
                    noc_async_read(field_x_pgen.get_noc_addr(bin_k) + aligned_off1,
                                   (uint32_t)(fx_stage + c * V21S_FIELD_STAGE_BYTES),
                                   V21S_DRAM_READ_ALIGN);
                    // Second chunk if h-range crosses the boundary.
                    if (intra_byte_end > aligned_off1 + V21S_DRAM_READ_ALIGN
                        && aligned_off1 + V21S_DRAM_READ_ALIGN < field_pg_bytes) {
                        noc_async_read(field_x_pgen.get_noc_addr(bin_k) + aligned_off1 + V21S_DRAM_READ_ALIGN,
                                       (uint32_t)(fx_stage + c * V21S_FIELD_STAGE_BYTES + V21S_DRAM_READ_ALIGN),
                                       V21S_DRAM_READ_ALIGN);
                    }
                }
                noc_async_read_barrier();
            }
            for (uint32_t hh = 0; hh < MAX_H; ++hh) {
                cb_reserve_back(CB_FX, 1);
#ifdef V21_FIELD_BF16
                uint16_t* tile = reinterpret_cast<uint16_t*>(get_write_ptr(CB_FX));
#else
                float* tile = reinterpret_cast<float*>(get_write_ptr(CB_FX));
#endif
                for (uint32_t c = 0; c < batch; ++c) {
                    float fx_val = 0.0f;
                    if (kk < k_count_l1[c] && hh < h_count_l1[c]) {
                        const uint32_t bin_yl = bin_yl_l1[c];
                        const uint32_t aligned_off = (bin_yl * 4) & ~(V21S_DRAM_READ_ALIGN - 1u);
                        const uint32_t intra_off   = (bin_yl * 4) - aligned_off;
                        const uint8_t* row = fx_stage + c * V21S_FIELD_STAGE_BYTES + intra_off;
                        fx_val = *reinterpret_cast<const float*>(row + hh * 4);
                    }
#ifdef V21_FIELD_BF16
                    union { float f; uint32_t u; } cv; cv.f = fx_val;
                    // bf16 unpacker expects tilized layout (4 faces of 16×16), not
                    // linear. Place cell c at its tilized L1 position.
                    const uint32_t row = c >> 5, col = c & 31u;
                    const uint32_t q = (((row >> 4) << 1 | (col >> 4)) << 8)
                                       | ((row & 15u) << 4) | (col & 15u);
                    tile[q] = (uint16_t)((cv.u + 0x8000u) >> 16);  // round-to-nearest bf16
#else
                    tile[c] = fx_val;
#endif
                }
                (void)0;  // skip padding writes — L1 starts at zero per launch
                cb_push_back(CB_FX, 1);
            }
        }

        // cb_fy is now produced by NCRISC in parallel.
        // End-of-batch sync: ensure NCRISC has finished consuming bin_xl_l1/etc.
        // for this batch before we overwrite them in the next batch's prep.
        // Reuse the same sems with monotonic counter: each side increments twice
        // per batch (after prep, after tile-pack). Counter B*2+1 = prep-done; B*2+2 = pack-done.
        asm volatile("fence rw, rw" ::: "memory");
        *sem_brisc_done = (batch_idx + 1u) * 2u;
        noc_semaphore_wait_min(sem_ncrisc_done, (batch_idx + 1u) * 2u);

        cells_done += batch;
        ++batch_idx;
    }
}

#endif  // V21_USE_SFPU
