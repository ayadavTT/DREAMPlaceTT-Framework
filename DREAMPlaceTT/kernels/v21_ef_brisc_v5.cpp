// SPDX-License-Identifier: Apache-2.0
//
// V21a electric_force — BRISC kernel v5 (FACE-MERGE).
//
// Same parallel-prep as v1 BRISC, but tile packing changes:
//   1. cb_px tile: tile[c] = tile[c+512] = px[c, kk]   (duplicated to both halves)
//   2. cb_py tile: tile[c] = tile[c+512] = py[c, hh]   (duplicated)
//   3. cb_neg_ratio tile: tile[c] = tile[c+512] = -ratio[c]  (duplicated)
//   4. cb_fxy tile (64 per batch): tile[c] = fx[c, kk, hh]  in LOWER half only.
//      NCRISC writes tile[c+512] = fy[c, kk, hh] in UPPER half via shared L1.
//      Sync: BRISC signals "lower half done" via sem_brisc_fxy_done;
//            NCRISC signals "upper half done" via sem_ncrisc_fxy_done;
//            BRISC pushes only after NCRISC signal.

#define V21_CB_SCRATCH (tt::CBIndex::c_24)

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

constexpr uint32_t CB_PX        = tt::CBIndex::c_0;
constexpr uint32_t CB_PY        = tt::CBIndex::c_1;
constexpr uint32_t CB_FXY       = tt::CBIndex::c_2;
constexpr uint32_t CB_NEG_RATIO = tt::CBIndex::c_4;

// Emulate bf16 precision: round to nearest and zero the low 16 mantissa bits.
// Returns an fp32 value whose bit-pattern matches `bf16-as-fp32` (zero-extend).
// Used in V21_FXY_BF16 mode to measure precision impact w/o changing CB format.
static inline float f32_truncate_to_bf16(float f) {
    union { float f; uint32_t u; } x;
    x.f = f;
    uint32_t bias = 0x7FFFu + ((x.u >> 16) & 1u);
    x.u = (x.u + bias) & 0xFFFF0000u;
    return x.f;
}

void kernel_main() {
    const uint32_t my_core_idx        = get_arg_val<uint32_t>(0);
    const uint32_t cells_start        = get_arg_val<uint32_t>(1);
    const uint32_t cells_end          = get_arg_val<uint32_t>(2);
    const uint32_t cells_start_global = get_arg_val<uint32_t>(3);
    const uint32_t num_nodes          = get_arg_val<uint32_t>(4);
    const uint32_t num_bins_x         = get_arg_val<uint32_t>(5);
    const uint32_t num_bins_y         = get_arg_val<uint32_t>(6);
    const uint32_t field_x_base       = get_arg_val<uint32_t>(7);
    // field_y_base at arg 8 (used by NCRISC)
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
    const uint32_t sem_brisc_done_id  = get_arg_val<uint32_t>(21);  // parallel-prep sync
    const uint32_t sem_ncrisc_done_id = get_arg_val<uint32_t>(22);
    const uint32_t sem_brisc_fxy_id   = get_arg_val<uint32_t>(23);  // NEW v5: per-tile cb_fxy sync
    const uint32_t sem_ncrisc_fxy_id  = get_arg_val<uint32_t>(24);

    volatile tt_l1_ptr uint32_t* sem_brisc_done =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_brisc_done_id));
    volatile tt_l1_ptr uint32_t* sem_ncrisc_done =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_ncrisc_done_id));
    volatile tt_l1_ptr uint32_t* sem_brisc_fxy =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_brisc_fxy_id));
    volatile tt_l1_ptr uint32_t* sem_ncrisc_fxy =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_ncrisc_fxy_id));

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
    float*    px_l1        = reinterpret_cast<float*>(reserve(V21S_BATCH * V21S_MAX_KH * 4u));
    float*    py_l1        = reinterpret_cast<float*>(reserve(V21S_BATCH * V21S_MAX_KH * 4u));
    float*    neg_ratio_l1 = reinterpret_cast<float*>(reserve(V21S_BATCH * 4u));
    uint32_t* bin_xl_l1    = reinterpret_cast<uint32_t*>(reserve(V21S_BATCH * 4u));
    uint32_t* bin_yl_l1    = reinterpret_cast<uint32_t*>(reserve(V21S_BATCH * 4u));
    uint32_t* k_count_l1   = reinterpret_cast<uint32_t*>(reserve(V21S_BATCH * 4u));
    uint32_t* h_count_l1   = reinterpret_cast<uint32_t*>(reserve(V21S_BATCH * 4u));
    constexpr uint32_t V21S_FIELD_STAGE_BYTES = 2u * V21S_DRAM_READ_ALIGN;
    uint8_t*  fx_stage     = reserve(V21S_BATCH * V21S_FIELD_STAGE_BYTES);
    uint8_t*  fy_stage_a   = reserve(V21S_BATCH * V21S_FIELD_STAGE_BYTES);  // even kk
    uint8_t*  fy_stage_b   = reserve(V21S_BATCH * V21S_FIELD_STAGE_BYTES);  // odd kk (v5d double-buf)
    volatile uint32_t* shared_fxy_ptrs = reinterpret_cast<volatile uint32_t*>(reserve(64u));

    const InterleavedAddrGen<true> pos_pgen     = {.bank_base_address = pos_base,     .page_size = V21S_POS_PG_BYTES};
    const InterleavedAddrGen<true> field_x_pgen = {.bank_base_address = field_x_base, .page_size = field_pg_bytes};
    const InterleavedAddrGen<true> const_pgen   = {.bank_base_address = const_base,   .page_size = const_pg_bytes};
    const uint64_t my_const_base = const_pgen.get_noc_addr(my_core_idx);

    DeviceZoneScopedN("V21B-ALL");
    uint32_t cells_done = cells_start;
    uint32_t batch_idx = 0u;
    uint32_t fxy_tiles_pushed = 0u;  // monotonic count of cb_fxy tiles fully pushed (for sem)
    while (cells_done < cells_end) {
        uint32_t batch = cells_end - cells_done;
        if (batch > V21S_BATCH) batch = V21S_BATCH;

        const uint32_t my_start = 0u;
        const uint32_t my_end   = batch / 2u;

        // ── Read constants (my half) ──
        {
            const uint32_t const_byte_off = (cells_done + my_start) * 32;
            const uint32_t my_cnt = my_end - my_start;
            const uint32_t const_bytes = (my_cnt * 32 + V21S_DRAM_READ_ALIGN - 1u) & ~(V21S_DRAM_READ_ALIGN - 1u);
            noc_async_read(my_const_base + const_byte_off, (uint32_t)(const_l1 + my_start * 8), const_bytes);
            noc_async_read_barrier();
        }

        // ── Read pos (chunked, my half) ──
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

        // ── Scalar prep (my half) ── (same as v1 BRISC)
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
        }  // V21B-PREP

        // Parallel-prep sync (counter = batch*2 + 1)
        asm volatile("fence rw, rw" ::: "memory");
        *sem_brisc_done = batch_idx * 2u + 1u;
        noc_semaphore_wait_min(sem_ncrisc_done, batch_idx * 2u + 1u);
        asm volatile("fence rw, rw" ::: "memory");

        // ── Pack cb_px (8 tiles, DUPLICATED across both halves) ──
        // Use __builtin_memcpy: contiguous source allows the compiler to emit
        // an unrolled / word-wise copy faster than the scalar `tile[c]=src[c]` loop.
        const uint32_t bytes_lo = batch * 4u;
        for (uint32_t k = 0; k < V21S_MAX_KH; ++k) {
            cb_reserve_back(CB_PX, 1);
            uint8_t* tile = reinterpret_cast<uint8_t*>(get_write_ptr(CB_PX));
            const uint8_t* src = reinterpret_cast<const uint8_t*>(px_l1 + k * V21S_BATCH);
            __builtin_memcpy(tile, src, bytes_lo);                // lanes 0..batch
            __builtin_memcpy(tile + 512u * 4u, src, bytes_lo);    // duplicate upper half
            cb_push_back(CB_PX, 1);
        }

        // ── Pack cb_py (8 tiles, DUPLICATED) ──
        for (uint32_t h = 0; h < V21S_MAX_KH; ++h) {
            cb_reserve_back(CB_PY, 1);
            uint8_t* tile = reinterpret_cast<uint8_t*>(get_write_ptr(CB_PY));
            const uint8_t* src = reinterpret_cast<const uint8_t*>(py_l1 + h * V21S_BATCH);
            __builtin_memcpy(tile, src, bytes_lo);
            __builtin_memcpy(tile + 512u * 4u, src, bytes_lo);
            cb_push_back(CB_PY, 1);
        }

        // ── Pack cb_neg_ratio (1 tile, DUPLICATED) ──
        cb_reserve_back(CB_NEG_RATIO, 1);
        {
            uint8_t* tile = reinterpret_cast<uint8_t*>(get_write_ptr(CB_NEG_RATIO));
            const uint8_t* src = reinterpret_cast<const uint8_t*>(neg_ratio_l1);
            __builtin_memcpy(tile, src, bytes_lo);
            __builtin_memcpy(tile + 512u * 4u, src, bytes_lo);
        }
        cb_push_back(CB_NEG_RATIO, 1);

        // ── Pack cb_fxy LOWER halves (64 tiles per batch), dual-producer sync with NCRISC ──
        // BRISC fills lanes 0..511 with fx data. NCRISC fills 512..1023 with fy data.
        // Per tile: BRISC reserves, writes fx, signals NCRISC, waits for NCRISC, pushes.
        for (uint32_t kk = 0; kk < V21S_MAX_KH; ++kk) {
            // Read field_x DRAM for all cells of this kk (same logic as v1 BRISC).
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
                    if (intra_byte_end > aligned_off1 + V21S_DRAM_READ_ALIGN
                        && aligned_off1 + V21S_DRAM_READ_ALIGN < field_pg_bytes) {
                        noc_async_read(field_x_pgen.get_noc_addr(bin_k) + aligned_off1 + V21S_DRAM_READ_ALIGN,
                                       (uint32_t)(fx_stage + c * V21S_FIELD_STAGE_BYTES + V21S_DRAM_READ_ALIGN),
                                       V21S_DRAM_READ_ALIGN);
                    }
                }
                noc_async_read_barrier();
            }

            // v5c SINGLE-PRODUCER: wait for NCRISC's fy reads for this kk, then BRISC
            // writes BOTH halves of cb_fxy (fx in lower from fx_stage; fy in upper
            // from NCRISC's fy_stage in shared L1). One sync per kk; no per-tile sync.
            const uint32_t fxy_token = batch_idx * V21S_MAX_KH + kk + 1u;
            noc_semaphore_wait_min(sem_brisc_fxy, fxy_token);
            asm volatile("fence rw, rw" ::: "memory");
            for (uint32_t hh = 0; hh < V21S_MAX_KH; ++hh) {
                cb_reserve_back(CB_FXY, 1);
                float* tile = reinterpret_cast<float*>(get_write_ptr(CB_FXY));
                for (uint32_t c = 0; c < batch; ++c) {
                    float fx_val = 0.0f, fy_val = 0.0f;
                    if (kk < k_count_l1[c] && hh < h_count_l1[c]) {
                        const uint32_t bin_yl = bin_yl_l1[c];
                        const uint32_t aligned_off = (bin_yl * 4) & ~(V21S_DRAM_READ_ALIGN - 1u);
                        const uint32_t intra_off   = (bin_yl * 4) - aligned_off;
                        const uint8_t* fx_row = fx_stage + c * V21S_FIELD_STAGE_BYTES + intra_off;
                        const uint8_t* fy_buf = (kk & 1u) ? fy_stage_b : fy_stage_a;
                        const uint8_t* fy_row = fy_buf + c * V21S_FIELD_STAGE_BYTES + intra_off;
                        fx_val = *reinterpret_cast<const float*>(fx_row + hh * 4);
                        fy_val = *reinterpret_cast<const float*>(fy_row + hh * 4);
                    }
#ifdef V21_FXY_BF16
                    // Emulate bf16 precision (store as fp32; low 16 bits cleared).
                    tile[c]       = f32_truncate_to_bf16(fx_val);
                    tile[c + 512] = f32_truncate_to_bf16(fy_val);
#else
                    tile[c]       = fx_val;
                    tile[c + 512] = fy_val;
#endif
                }
                cb_push_back(CB_FXY, 1);
            }
            // Signal NCRISC: done consuming fy_stage for this kk; it can read kk+1 now.
            asm volatile("fence rw, rw" ::: "memory");
            *sem_ncrisc_fxy = fxy_token;
            (void)shared_fxy_ptrs;
            (void)fxy_tiles_pushed;
        }

        // End-of-batch sync (counter = batch*2 + 2 = pack-done).
        asm volatile("fence rw, rw" ::: "memory");
        *sem_brisc_done = (batch_idx + 1u) * 2u;
        noc_semaphore_wait_min(sem_ncrisc_done, (batch_idx + 1u) * 2u);

        cells_done += batch;
        ++batch_idx;
    }
}
