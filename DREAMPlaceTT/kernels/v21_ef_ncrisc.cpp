// SPDX-License-Identifier: Apache-2.0
//
// V21a electric_force — NCRISC kernel (v3: parallel prep + cb_fy producer).
//
// V21_USE_SFPU mode:
//   1. Reads const+pos for second half of cells [batch/2..batch).
//   2. Scalar prep for second half, writes to SHARED L1 in c_24 (same offsets
//      as BRISC).
//   3. Signals sem_ncrisc_done, waits for sem_brisc_done.
//   4. Reads field_y + packs cb_fy (64 tiles per batch, all 512 cells).
//   5. Drains cb_gx_out + cb_gy_out as TRISC produces them.
//
// Scalar mode: falls back to v21_ef_common.h.

#define V21_CB_SCRATCH (tt::CBIndex::c_25)

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
constexpr uint32_t V21S_READ_CHUNK       = 128;

constexpr uint32_t CB_FY        = tt::CBIndex::c_3;
constexpr uint32_t CB_GX_OUT    = tt::CBIndex::c_16;
constexpr uint32_t CB_GY_OUT    = tt::CBIndex::c_17;
constexpr uint32_t V21_BRISC_SCRATCH = (uint32_t)tt::CBIndex::c_24;

void kernel_main() {
    const uint32_t my_core_idx        = get_arg_val<uint32_t>(0);
    const uint32_t cells_start        = get_arg_val<uint32_t>(1);
    const uint32_t cells_end          = get_arg_val<uint32_t>(2);
    const uint32_t cells_start_global = get_arg_val<uint32_t>(3);
    const uint32_t num_nodes          = get_arg_val<uint32_t>(4);
    const uint32_t num_bins_x         = get_arg_val<uint32_t>(5);
    const uint32_t num_bins_y         = get_arg_val<uint32_t>(6);
    // field_x_base at arg 7 (unused here)
    const uint32_t field_y_base       = get_arg_val<uint32_t>(8);
    const uint32_t field_pg_bytes     = get_arg_val<uint32_t>(9);
    const uint32_t pos_base           = get_arg_val<uint32_t>(10);
    const uint32_t const_base         = get_arg_val<uint32_t>(11);
    const uint32_t const_pg_bytes     = get_arg_val<uint32_t>(12);
    const uint32_t grad_base          = get_arg_val<uint32_t>(13);
    const uint32_t grad_pg_bytes      = get_arg_val<uint32_t>(14);
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

    // ── Shared L1 layout in BRISC's c_24 (MUST match BRISC reserve() order). ──
    uint8_t* shared_base = reinterpret_cast<uint8_t*>(get_write_ptr(V21_BRISC_SCRATCH));
    uint32_t shared_off = 0u;
    auto shared_reserve = [&](uint32_t bytes) -> uint8_t* {
        uint8_t* p = shared_base + shared_off;
        shared_off += (bytes + 63u) & ~63u;
        return p;
    };
    uint32_t* const_l1     = reinterpret_cast<uint32_t*>(shared_reserve(V21S_BATCH * 32u));
    uint8_t*  pos_x_l1     = shared_reserve(V21S_BATCH * V21S_DRAM_READ_ALIGN);
    uint8_t*  pos_y_l1     = shared_reserve(V21S_BATCH * V21S_DRAM_READ_ALIGN);
    float*    px_l1        = reinterpret_cast<float*>(shared_reserve(V21S_BATCH * V21S_MAX_KH * 4u));
    float*    py_l1        = reinterpret_cast<float*>(shared_reserve(V21S_BATCH * V21S_MAX_KH * 4u));
    float*    neg_ratio_l1 = reinterpret_cast<float*>(shared_reserve(V21S_BATCH * 4u));
    uint32_t* bin_xl_l1    = reinterpret_cast<uint32_t*>(shared_reserve(V21S_BATCH * 4u));
    uint32_t* bin_yl_l1    = reinterpret_cast<uint32_t*>(shared_reserve(V21S_BATCH * 4u));
    uint32_t* k_count_l1   = reinterpret_cast<uint32_t*>(shared_reserve(V21S_BATCH * 4u));
    uint32_t* h_count_l1   = reinterpret_cast<uint32_t*>(shared_reserve(V21S_BATCH * 4u));
    constexpr uint32_t V21S_FIELD_STAGE_BYTES = 2u * V21S_DRAM_READ_ALIGN;
    (void)shared_reserve(V21S_BATCH * V21S_FIELD_STAGE_BYTES);  // fx_stage (BRISC-owned)
    uint8_t*  fy_stage     = shared_reserve(V21S_BATCH * V21S_FIELD_STAGE_BYTES);

    // ── NCRISC-private drain stage in c_25 ──
    uint8_t* drain_base = reinterpret_cast<uint8_t*>(get_write_ptr(V21_CB_SCRATCH));
    float* drain_stage = reinterpret_cast<float*>(drain_base);

    const InterleavedAddrGen<true> pos_pgen     = {.bank_base_address = pos_base,     .page_size = V21S_POS_PG_BYTES};
    const InterleavedAddrGen<true> field_y_pgen = {.bank_base_address = field_y_base, .page_size = field_pg_bytes};
    const InterleavedAddrGen<true> const_pgen   = {.bank_base_address = const_base,   .page_size = const_pg_bytes};
    const InterleavedAddrGen<true> grad_pgen    = {.bank_base_address = grad_base,    .page_size = grad_pg_bytes};
    const uint64_t my_const_base = const_pgen.get_noc_addr(my_core_idx);
    const uint64_t my_grad_base  = grad_pgen.get_noc_addr(my_core_idx);

    DeviceZoneScopedN("V21N-ALL");
    uint32_t cells_done = cells_start;
    uint32_t batch_idx = 0u;
    while (cells_done < cells_end) {
        uint32_t batch = cells_end - cells_done;
        if (batch > V21S_BATCH) batch = V21S_BATCH;

        // Parallel-prep split: NCRISC owns cells [half..batch).
        const uint32_t my_start = batch / 2u;
        const uint32_t my_end   = batch;

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

        // Scalar prep (only my half) — must match BRISC exactly.
        { DeviceZoneScopedN("V21N-PREP");
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

            const float node_x_top = node_x + nsx;
            const float node_y_top = node_y + nsy;
            const float bin_xl_lo = xl + (float)(uint32_t)bin_xl_i * bsx;
            const float bin_yl_lo = yl + (float)(uint32_t)bin_yl_i * bsy;
#ifdef V21_PREP_FAST
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
                px_l1[k * V21S_BATCH + c] = px_val;
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
                py_l1[h * V21S_BATCH + c] = py_val;
                bin_y_lo += bsy;
            }
#endif
        }
        }  // V21N-PREP

        // Signal NCRISC prep done (counter = batch*2 + 1), wait for BRISC prep done.
        asm volatile("fence rw, rw" ::: "memory");
        *sem_ncrisc_done = batch_idx * 2u + 1u;
        noc_semaphore_wait_min(sem_brisc_done, batch_idx * 2u + 1u);
        asm volatile("fence rw, rw" ::: "memory");


        // ── Pack cb_fy (64 tiles, in kk*8+hh order) — uses FULL 512 cells ──
        { DeviceZoneScopedN("V21N-FY");
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
                    noc_async_read(field_y_pgen.get_noc_addr(bin_k) + aligned_off1,
                                   (uint32_t)(fy_stage + c * V21S_FIELD_STAGE_BYTES),
                                   V21S_DRAM_READ_ALIGN);
                    if (intra_byte_end > aligned_off1 + V21S_DRAM_READ_ALIGN
                        && aligned_off1 + V21S_DRAM_READ_ALIGN < field_pg_bytes) {
                        noc_async_read(field_y_pgen.get_noc_addr(bin_k) + aligned_off1 + V21S_DRAM_READ_ALIGN,
                                       (uint32_t)(fy_stage + c * V21S_FIELD_STAGE_BYTES + V21S_DRAM_READ_ALIGN),
                                       V21S_DRAM_READ_ALIGN);
                    }
                }
                noc_async_read_barrier();
            }
            for (uint32_t hh = 0; hh < MAX_H; ++hh) {
                cb_reserve_back(CB_FY, 1);
#ifdef V21_FIELD_BF16
                uint16_t* tile = reinterpret_cast<uint16_t*>(get_write_ptr(CB_FY));
#else
                float* tile = reinterpret_cast<float*>(get_write_ptr(CB_FY));
#endif
                for (uint32_t c = 0; c < batch; ++c) {
                    float fy_val = 0.0f;
                    if (kk < k_count_l1[c] && hh < h_count_l1[c]) {
                        const uint32_t bin_yl = bin_yl_l1[c];
                        const uint32_t aligned_off = (bin_yl * 4) & ~(V21S_DRAM_READ_ALIGN - 1u);
                        const uint32_t intra_off   = (bin_yl * 4) - aligned_off;
                        const uint8_t* row = fy_stage + c * V21S_FIELD_STAGE_BYTES + intra_off;
                        fy_val = *reinterpret_cast<const float*>(row + hh * 4);
                    }
#ifdef V21_FIELD_BF16
                    union { float f; uint32_t u; } cv; cv.f = fy_val;
                    const uint32_t row = c >> 5, col = c & 31u;
                    const uint32_t q = (((row >> 4) << 1 | (col >> 4)) << 8)
                                       | ((row & 15u) << 4) | (col & 15u);
                    tile[q] = (uint16_t)((cv.u + 0x8000u) >> 16);  // round-to-nearest bf16
#else
                    tile[c] = fy_val;
#endif
                }
                cb_push_back(CB_FY, 1);
            }
        }
        }  // V21N-FY

        // End-of-batch sync (counter = batch*2 + 2 = pack-done).
        // Ensures BRISC has also finished consuming bin_xl_l1 (for cb_fx) before
        // either side overwrites the shared arrays in the next batch's prep.
        asm volatile("fence rw, rw" ::: "memory");
        *sem_ncrisc_done = (batch_idx + 1u) * 2u;
        noc_semaphore_wait_min(sem_brisc_done, (batch_idx + 1u) * 2u);

        // ── Drain cb_gx_out + cb_gy_out, interleave, write to DRAM grad page ──
        cb_wait_front(CB_GX_OUT, 1);
        cb_wait_front(CB_GY_OUT, 1);
        const float* gx_tile = reinterpret_cast<const float*>(get_read_ptr(CB_GX_OUT));
        const float* gy_tile = reinterpret_cast<const float*>(get_read_ptr(CB_GY_OUT));

        { DeviceZoneScopedN("V21N-INTERLEAVE");
        for (uint32_t c = 0; c < batch; ++c) {
            drain_stage[c * 2 + 0] = gx_tile[c];
            drain_stage[c * 2 + 1] = gy_tile[c];
        }
        }
        { DeviceZoneScopedN("V21N-WRITE");
        const uint32_t out_off   = cells_done * 8u;
        const uint32_t out_bytes = (batch * 8u + 15u) & ~15u;
        noc_async_write((uint32_t)drain_stage, my_grad_base + out_off, out_bytes);
        noc_async_write_barrier();
        }

        cb_pop_front(CB_GX_OUT, 1);
        cb_pop_front(CB_GY_OUT, 1);

        cells_done += batch;
        ++batch_idx;
    }

    (void)inv_bsx; (void)inv_bsy;  // unused on this path (only prep uses them indirectly)
}

#endif  // V21_USE_SFPU
