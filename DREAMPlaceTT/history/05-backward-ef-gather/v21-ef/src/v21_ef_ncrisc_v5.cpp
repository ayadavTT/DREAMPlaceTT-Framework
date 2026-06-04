// SPDX-License-Identifier: Apache-2.0
//
// V21a electric_force — NCRISC kernel v5 (FACE-MERGE).
//
// 1. Parallel-prep (cells [batch/2..batch), shared L1 in c_24) — same as v1.
// 2. Sync via sem_brisc_done/sem_ncrisc_done.
// 3. Read field_y DRAM into fy_stage (shared L1).
// 4. For each cb_fxy tile slot (in BRISC's reservation order):
//      a. Wait for sem_brisc_fxy ≥ tile_idx+1 (BRISC has reserved + filled fx).
//      b. NCRISC calls get_write_ptr(cb_fxy) to get the L1 slot address.
//      c. Write fy[c] to lanes 512..1023 of the slot.
//      d. Increment sem_ncrisc_fxy → BRISC pushes the slot to TRISC.
// 5. After all 64 fxy tiles, drain cb_gxy_out (1 tile holding gx in lower lanes,
//    gy in upper). Interleave (gx0, gy0, gx1, gy1, ...) into DRAM grad page.

#define V21_CB_SCRATCH (tt::CBIndex::c_25)

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"
#include "api/debug/dprint.h"

constexpr uint32_t V21S_BATCH            = 512;
constexpr uint32_t V21S_TILE_CELLS       = 1024;
constexpr uint32_t V21S_MAX_KH           = 8;
constexpr uint32_t V21S_DRAM_READ_ALIGN  = 64;
constexpr uint32_t V21S_POS_PG_BYTES     = 4096;
constexpr uint32_t V21S_POS_FLOATS_PER_PG = V21S_POS_PG_BYTES / 4;
constexpr uint32_t V21S_READ_CHUNK       = 128;

constexpr uint32_t CB_FXY       = tt::CBIndex::c_2;
constexpr uint32_t CB_GXY_OUT   = tt::CBIndex::c_16;
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
    const uint32_t sem_brisc_fxy_id   = get_arg_val<uint32_t>(23);
    const uint32_t sem_ncrisc_fxy_id  = get_arg_val<uint32_t>(24);

    volatile tt_l1_ptr uint32_t* sem_brisc_done =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_brisc_done_id));
    volatile tt_l1_ptr uint32_t* sem_ncrisc_done =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_ncrisc_done_id));
    volatile tt_l1_ptr uint32_t* sem_brisc_fxy =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_brisc_fxy_id));
    volatile tt_l1_ptr uint32_t* sem_ncrisc_fxy =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_ncrisc_fxy_id));

    // Shared L1 layout in BRISC's c_24 (MUST match BRISC reserve() order).
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
    uint8_t*  fy_stage_a   = shared_reserve(V21S_BATCH * V21S_FIELD_STAGE_BYTES);  // even kk
    uint8_t*  fy_stage_b   = shared_reserve(V21S_BATCH * V21S_FIELD_STAGE_BYTES);  // odd kk (v5d)
    volatile uint32_t* shared_fxy_ptrs = reinterpret_cast<volatile uint32_t*>(shared_reserve(64u));

    // NCRISC-private drain stage in c_25
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
    uint32_t fxy_tiles_done = 0u;  // monotonic count of cb_fxy upper halves NCRISC has written
    while (cells_done < cells_end) {
        uint32_t batch = cells_end - cells_done;
        if (batch > V21S_BATCH) batch = V21S_BATCH;

        const uint32_t my_start = batch / 2u;
        const uint32_t my_end   = batch;

        // ── Read constants (my half) ──
        {
            const uint32_t const_byte_off = (cells_done + my_start) * 32;
            const uint32_t my_cnt = my_end - my_start;
            const uint32_t const_bytes = (my_cnt * 32 + V21S_DRAM_READ_ALIGN - 1u) & ~(V21S_DRAM_READ_ALIGN - 1u);
            noc_async_read(my_const_base + const_byte_off, (uint32_t)(const_l1 + my_start * 8), const_bytes);
            noc_async_read_barrier();
        }

        // ── Read pos (my half) ──
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

        // ── Scalar prep (my half) — must match BRISC exactly ──
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
            if (k_count > V21S_MAX_KH) k_count = V21S_MAX_KH;
            if (h_count > V21S_MAX_KH) h_count = V21S_MAX_KH;
            k_count_l1[c]  = k_count;
            h_count_l1[c]  = h_count;
            bin_xl_l1[c]   = (uint32_t)bin_xl_i;
            bin_yl_l1[c]   = (uint32_t)bin_yl_i;

            const float node_x_top = node_x + nsx;
            const float node_y_top = node_y + nsy;
            float bin_x_lo = xl + (float)(uint32_t)bin_xl_i * bsx;
            for (uint32_t k = 0; k < V21S_MAX_KH; ++k) {
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
            float bin_y_lo = yl + (float)(uint32_t)bin_yl_i * bsy;
            for (uint32_t h = 0; h < V21S_MAX_KH; ++h) {
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
        }
        }  // V21N-PREP

        // Parallel-prep sync
        asm volatile("fence rw, rw" ::: "memory");
        *sem_ncrisc_done = batch_idx * 2u + 1u;
        noc_semaphore_wait_min(sem_brisc_done, batch_idx * 2u + 1u);
        asm volatile("fence rw, rw" ::: "memory");

        // ── v5c: NCRISC just reads field_y DRAM into shared fy_stage per kk,
        //    signals BRISC after each kk. BRISC reads fy_stage and writes the
        //    upper half of cb_fxy itself (single producer). Removes all
        //    per-tile sync; only kk-level sync remains.
        { DeviceZoneScopedN("V21N-FY");
        for (uint32_t kk = 0; kk < V21S_MAX_KH; ++kk) {
            const uint32_t fxy_token = batch_idx * V21S_MAX_KH + kk + 1u;
            uint8_t* fy_stage = (kk & 1u) ? fy_stage_b : fy_stage_a;
            // v5d double-buffered: wait only when about to reuse the same buffer
            // (kk-2 within batch, or batch boundary if same parity).
            if (fxy_token > 2u) {
                noc_semaphore_wait_min(sem_ncrisc_fxy, fxy_token - 2u);
                asm volatile("fence rw, rw" ::: "memory");
            }
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
            // Signal BRISC: fy_stage for this kk is fully populated.
            asm volatile("fence rw, rw" ::: "memory");
            *sem_brisc_fxy = fxy_token;
        }
        (void)shared_fxy_ptrs;
        (void)fxy_tiles_done;
        }  // V21N-FY

        // End-of-batch sync (counter = batch*2 + 2)
        asm volatile("fence rw, rw" ::: "memory");
        *sem_ncrisc_done = (batch_idx + 1u) * 2u;
        noc_semaphore_wait_min(sem_brisc_done, (batch_idx + 1u) * 2u);

        // ── Drain cb_gxy_out: 1 tile with lower=gx, upper=gy → interleaved DRAM ──
        cb_wait_front(CB_GXY_OUT, 1);
        const float* gxy_tile = reinterpret_cast<const float*>(get_read_ptr(CB_GXY_OUT));

        { DeviceZoneScopedN("V21N-INTERLEAVE");
        for (uint32_t c = 0; c < batch; ++c) {
            drain_stage[c * 2 + 0] = gxy_tile[c];          // gx for cell c (lower lane)
            drain_stage[c * 2 + 1] = gxy_tile[c + 512];    // gy for cell c (upper lane)
        }
        }
        { DeviceZoneScopedN("V21N-WRITE");
        const uint32_t out_off   = cells_done * 8u;
        const uint32_t out_bytes = (batch * 8u + 15u) & ~15u;
        noc_async_write((uint32_t)drain_stage, my_grad_base + out_off, out_bytes);
        noc_async_write_barrier();
        }

        cb_pop_front(CB_GXY_OUT, 1);

        cells_done += batch;
        ++batch_idx;
    }

    (void)inv_bsx; (void)inv_bsy;
}
