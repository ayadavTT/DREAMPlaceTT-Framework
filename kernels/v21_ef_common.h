// SPDX-License-Identifier: Apache-2.0
//
// V21a electric_force microbench — shared body for BRISC + NCRISC siblings.
//
// For each cell c in this kernel's range, compute the per-cell density-gradient
// pair (gx, gy) and write it to the per-core output DRAM page. Math mirrors
// DREAMPlace's `computeElectricForceLauncher` (CPU baseline) at
// dreamplace/ops/electric_potential/src/electric_force.cpp:106-172, with the
// `-` from electric_potential.py:235 baked into the kernel output:
//
//   For cell c with global index g = cells_start_global + c:
//     pos_x   = pos[g];   pos_y = pos[num_nodes + g]
//     node_x  = pos_x + ox[c];    node_y = pos_y + oy[c]
//     nsx, nsy, ratio from constants[c]
//
//   bin_index_xl = floor((node_x - xl) * inv_bsx)
//   bin_index_xh = floor((node_x + nsx - xl) * inv_bsx) + 1     (exclusive)
//   bin_index_xl = max(bin_index_xl, 0)
//   bin_index_xh = min(bin_index_xh, num_bins_x)
//   (same for y)
//
//   gx = sum_{(k,h)} triangle(node_x, nsx, xl, k, bsx)
//                  * triangle(node_y, nsy, yl, h, bsy)
//                  * field_x[k * num_bins_y + h]
//   (same accumulation for gy with field_y)
//   gx = -gx * ratio;    gy = -gy * ratio
//
// BRISC handles cells [cells_start, brisc_end) on NoC 0.
// NCRISC handles cells [brisc_end, cells_end) on NoC 1.
// They use different CB_SCRATCH ids (V21_CB_SCRATCH must be defined per caller).
//
// DRAM read alignment: 64 B on Blackhole (NOT 32 — see
// memory note blackhole_dram_read_align_64.md). Pos + field reads must be
// 64-B aligned in src + size, else silent corruption.
//
// L1 layout (per kernel, all 64-B aligned):
//   const_l1     : V21_BATCH × 32 B           = 2 KB
//   pos_x_stage  : V21_BATCH × 64 B           = 4 KB
//   pos_y_stage  : V21_BATCH × 64 B           = 4 KB
//   fx_row_stage : V21_MAX_K × FIELD_ROW_BYTES = 2 KB
//   fy_row_stage : V21_MAX_K × FIELD_ROW_BYTES = 2 KB
//   grad_out_l1  : V21_BATCH × 8 B            = 512 B
// Total ≈ 15 KB — fits in 32 KB CB.

#pragma once

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"

#ifndef V21_CB_SCRATCH
#error "V21_CB_SCRATCH must be defined before including v21_ef_common.h"
#endif

constexpr uint32_t V21_BATCH             = 64;
constexpr uint32_t V21_FIELD_GROUP       = 4;     // # cells whose field reads share one barrier
constexpr uint32_t V21_DRAM_READ_ALIGN   = 64;
constexpr uint32_t V21_L1_WRITE_ALIGN    = 16;
constexpr uint32_t V21_MAX_K_RANGE       = 8;     // max bin_xh - bin_xl per cell
constexpr uint32_t V21_FIELD_ROW_BYTES   = 256;   // up to 16-wide bin-slice + 64-B alignment slack
constexpr uint32_t V21_POS_PG_BYTES      = 4096;
constexpr uint32_t V21_POS_FLOATS_PER_PG = V21_POS_PG_BYTES / 4;

inline void v21_ef_main() {
    const uint32_t my_core_idx        = get_arg_val<uint32_t>(0);
    const uint32_t cells_start        = get_arg_val<uint32_t>(1);   // intra-core inclusive
    const uint32_t cells_end          = get_arg_val<uint32_t>(2);   // intra-core exclusive
    const uint32_t cells_start_global = get_arg_val<uint32_t>(3);   // global cell idx of intra c=0
    const uint32_t num_nodes          = get_arg_val<uint32_t>(4);
    const uint32_t num_bins_x         = get_arg_val<uint32_t>(5);
    const uint32_t num_bins_y         = get_arg_val<uint32_t>(6);
    const uint32_t field_x_base       = get_arg_val<uint32_t>(7);
    const uint32_t field_y_base       = get_arg_val<uint32_t>(8);
    const uint32_t field_pg_bytes     = get_arg_val<uint32_t>(9);   // = num_bins_y * 4
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

    uint8_t* base = reinterpret_cast<uint8_t*>(get_write_ptr(V21_CB_SCRATCH));

    constexpr uint32_t CONST_BYTES     = V21_BATCH * 32;
    constexpr uint32_t POS_STAGE_BYTES = V21_BATCH * V21_DRAM_READ_ALIGN;
    // Group field-staging: V21_FIELD_GROUP cells share a single barrier.
    // Per cell, max V21_MAX_K_RANGE rows × V21_FIELD_ROW_BYTES each.
    constexpr uint32_t FIELD_ROW_BLOCK = V21_FIELD_GROUP * V21_MAX_K_RANGE * V21_FIELD_ROW_BYTES;
    constexpr uint32_t GRAD_OUT_BYTES  = V21_BATCH * 8;

    const uint32_t const_off    = 0;
    const uint32_t pos_x_off    = const_off  + CONST_BYTES;
    const uint32_t pos_y_off    = pos_x_off  + POS_STAGE_BYTES;
    const uint32_t fx_row_off   = pos_y_off  + POS_STAGE_BYTES;
    const uint32_t fy_row_off   = fx_row_off + FIELD_ROW_BLOCK;
    const uint32_t grad_off     = fy_row_off + FIELD_ROW_BLOCK;

    uint32_t* const_l1    = reinterpret_cast<uint32_t*>(base + const_off);
    uint8_t*  pos_x_l1    = base + pos_x_off;
    uint8_t*  pos_y_l1    = base + pos_y_off;
    uint8_t*  fx_row_l1   = base + fx_row_off;
    uint8_t*  fy_row_l1   = base + fy_row_off;
    float*    grad_out_l1 = reinterpret_cast<float*>(base + grad_off);

    const InterleavedAddrGen<true> pos_pgen     = {.bank_base_address = pos_base,     .page_size = V21_POS_PG_BYTES};
    const InterleavedAddrGen<true> field_x_pgen = {.bank_base_address = field_x_base, .page_size = field_pg_bytes};
    const InterleavedAddrGen<true> field_y_pgen = {.bank_base_address = field_y_base, .page_size = field_pg_bytes};
    const InterleavedAddrGen<true> const_pgen   = {.bank_base_address = const_base,   .page_size = const_pg_bytes};
    const InterleavedAddrGen<true> grad_pgen    = {.bank_base_address = grad_base,    .page_size = grad_pg_bytes};

    const uint64_t my_const_base = const_pgen.get_noc_addr(my_core_idx);
    const uint64_t my_grad_base  = grad_pgen.get_noc_addr(my_core_idx);

    {
        DeviceZoneScopedN("V21MB-EF-ALL");
        uint32_t cells_done = cells_start;
        while (cells_done < cells_end) {
            uint32_t batch = cells_end - cells_done;
            if (batch > V21_BATCH) batch = V21_BATCH;

            // ── 1. Read per-cell constants (BATCH × 32 B). ──
            // cells_done is multiple of V21_BATCH except possibly on the last
            // partial batch. const_off_bytes = cells_done * 32 — always 64-B
            // aligned because cells_done is multiple of 2 (BATCH=64) within the
            // batched stream, and cells_start is multiple of 4 (host invariant).
            // We round the size up to 64 B to be safe on the last partial batch.
            {
                DeviceZoneScopedN("V21MB-EF-READ-CONST");
                const uint32_t const_byte_off = cells_done * 32;
                const uint32_t const_bytes    = (batch * 32 + V21_DRAM_READ_ALIGN - 1u) & ~(V21_DRAM_READ_ALIGN - 1u);
                noc_async_read(my_const_base + const_byte_off,
                               (uint32_t)const_l1, const_bytes);
                noc_async_read_barrier();
            }

            // ── 2. Per-cell pos reads (one 64-B aligned chunk for pos_x, one for pos_y). ──
            {
                DeviceZoneScopedN("V21MB-EF-READ-POS");
                for (uint32_t c = 0; c < batch; ++c) {
                    const uint32_t g   = cells_start_global + cells_done + c;
                    const uint32_t gx  = g;
                    const uint32_t gy  = num_nodes + g;

                    const uint32_t gx_pg          = gx / V21_POS_FLOATS_PER_PG;
                    const uint32_t gx_byte_in_pg  = (gx % V21_POS_FLOATS_PER_PG) * 4;
                    const uint32_t gx_aligned_off = gx_byte_in_pg & ~(V21_DRAM_READ_ALIGN - 1u);

                    const uint32_t gy_pg          = gy / V21_POS_FLOATS_PER_PG;
                    const uint32_t gy_byte_in_pg  = (gy % V21_POS_FLOATS_PER_PG) * 4;
                    const uint32_t gy_aligned_off = gy_byte_in_pg & ~(V21_DRAM_READ_ALIGN - 1u);

                    const uint64_t gx_noc = pos_pgen.get_noc_addr(gx_pg) + gx_aligned_off;
                    const uint64_t gy_noc = pos_pgen.get_noc_addr(gy_pg) + gy_aligned_off;
                    noc_async_read(gx_noc, (uint32_t)(pos_x_l1 + c * V21_DRAM_READ_ALIGN), V21_DRAM_READ_ALIGN);
                    noc_async_read(gy_noc, (uint32_t)(pos_y_l1 + c * V21_DRAM_READ_ALIGN), V21_DRAM_READ_ALIGN);
                }
                noc_async_read_barrier();
            }

            // ── 3. Per-cell: compute bin range, read field rows, accumulate (gx, gy). ──
            // Process cells in groups of V21_FIELD_GROUP. Per group: issue all field
            // reads, single barrier, then compute. Cuts barrier count by FIELD_GROUP×
            // vs the per-cell barrier baseline.
            {
                DeviceZoneScopedN("V21MB-EF-INNER");
                // Per-cell state buffered between read-issue phase and compute phase.
                // Stored on the stack; small enough (~88 B per cell × 4 = 352 B).
                struct CellPrep {
                    uint32_t k_count;
                    uint32_t h_count;
                    uint32_t bin_xl;
                    uint32_t bin_yl;
                    uint32_t intra_off;   // start byte within the 64-B aligned read
                    uint32_t fx_l1_base;  // L1 addr of cell's fx staging block
                    uint32_t fy_l1_base;  // L1 addr of cell's fy staging block
                    float    node_x;
                    float    node_y;
                    float    nsx;
                    float    nsy;
                    float    ratio;
                };
                CellPrep prep[V21_FIELD_GROUP];

                for (uint32_t g_start = 0; g_start < batch; g_start += V21_FIELD_GROUP) {
                    uint32_t g_end = g_start + V21_FIELD_GROUP;
                    if (g_end > batch) g_end = batch;
                    const uint32_t g_size = g_end - g_start;

                    // Phase 1: per-cell prep + issue all field reads (no barrier).
                    {
                    DeviceZoneScopedN("V21MB-EF-PREP");
                    for (uint32_t gc = 0; gc < g_size; ++gc) {
                        const uint32_t c       = g_start + gc;
                        const uint32_t g       = cells_start_global + cells_done + c;
                        const uint32_t gx_idx  = g;
                        const uint32_t gy_idx  = num_nodes + g;
                        const uint32_t gx_intra = ((gx_idx % V21_POS_FLOATS_PER_PG) * 4) & (V21_DRAM_READ_ALIGN - 1u);
                        const uint32_t gy_intra = ((gy_idx % V21_POS_FLOATS_PER_PG) * 4) & (V21_DRAM_READ_ALIGN - 1u);
                        const float pos_x = *reinterpret_cast<float*>(pos_x_l1 + c * V21_DRAM_READ_ALIGN + gx_intra);
                        const float pos_y = *reinterpret_cast<float*>(pos_y_l1 + c * V21_DRAM_READ_ALIGN + gy_intra);

                        cvt.u = const_l1[c * 8 + 0]; const float ox    = cvt.f;
                        cvt.u = const_l1[c * 8 + 1]; const float oy    = cvt.f;
                        cvt.u = const_l1[c * 8 + 2]; const float nsx   = cvt.f;
                        cvt.u = const_l1[c * 8 + 3]; const float nsy   = cvt.f;
                        cvt.u = const_l1[c * 8 + 4]; const float ratio = cvt.f;

                        const float node_x = pos_x + ox;
                        const float node_y = pos_y + oy;

                        // CPU semantics: int truncation toward zero on positive values,
                        // +1 BEFORE clamp on the upper bound.
                        int bin_xl_i = (int)((node_x - xl) * inv_bsx);
                        int bin_xh_i = (int)((node_x + nsx - xl) * inv_bsx) + 1;
                        int bin_yl_i = (int)((node_y - yl) * inv_bsy);
                        int bin_yh_i = (int)((node_y + nsy - yl) * inv_bsy) + 1;
                        if (bin_xl_i < 0) bin_xl_i = 0;
                        if (bin_xh_i > (int)num_bins_x) bin_xh_i = (int)num_bins_x;
                        if (bin_yl_i < 0) bin_yl_i = 0;
                        if (bin_yh_i > (int)num_bins_y) bin_yh_i = (int)num_bins_y;

                        prep[gc].node_x = node_x;
                        prep[gc].node_y = node_y;
                        prep[gc].nsx    = nsx;
                        prep[gc].nsy    = nsy;
                        prep[gc].ratio  = ratio;

                        if (bin_xh_i > bin_xl_i && bin_yh_i > bin_yl_i) {
                            const uint32_t bin_xl   = (uint32_t)bin_xl_i;
                            const uint32_t bin_xh   = (uint32_t)bin_xh_i;
                            const uint32_t bin_yl   = (uint32_t)bin_yl_i;
                            const uint32_t bin_yh   = (uint32_t)bin_yh_i;
                            const uint32_t k_count  = bin_xh - bin_xl;
                            const uint32_t h_count  = bin_yh - bin_yl;

                            const uint32_t intra_byte_start = bin_yl * 4;
                            const uint32_t intra_byte_end   = bin_yh * 4;
                            const uint32_t aligned_off      = intra_byte_start & ~(V21_DRAM_READ_ALIGN - 1u);
                            const uint32_t aligned_end      = (intra_byte_end + V21_DRAM_READ_ALIGN - 1u) & ~(V21_DRAM_READ_ALIGN - 1u);
                            const uint32_t aligned_bytes    = aligned_end - aligned_off;
                            const uint32_t intra_off        = intra_byte_start - aligned_off;

                            const uint32_t fx_base = (uint32_t)(fx_row_l1 + gc * V21_MAX_K_RANGE * V21_FIELD_ROW_BYTES);
                            const uint32_t fy_base = (uint32_t)(fy_row_l1 + gc * V21_MAX_K_RANGE * V21_FIELD_ROW_BYTES);

                            prep[gc].k_count    = k_count;
                            prep[gc].h_count    = h_count;
                            prep[gc].bin_xl     = bin_xl;
                            prep[gc].bin_yl     = bin_yl;
                            prep[gc].intra_off  = intra_off;
                            prep[gc].fx_l1_base = fx_base;
                            prep[gc].fy_l1_base = fy_base;

                            for (uint32_t kk = 0; kk < k_count; ++kk) {
                                const uint32_t k = bin_xl + kk;
                                const uint64_t fx_noc = field_x_pgen.get_noc_addr(k) + aligned_off;
                                const uint64_t fy_noc = field_y_pgen.get_noc_addr(k) + aligned_off;
                                noc_async_read(fx_noc, fx_base + kk * V21_FIELD_ROW_BYTES, aligned_bytes);
                                noc_async_read(fy_noc, fy_base + kk * V21_FIELD_ROW_BYTES, aligned_bytes);
                            }
                        } else {
                            prep[gc].k_count = 0;
                        }
                    }
                    }  // V21MB-EF-PREP

                    // Single barrier for all cells in this group.
                    {
                    DeviceZoneScopedN("V21MB-EF-WAIT");
                    noc_async_read_barrier();
                    }

                    // Phase 2: compute (gx, gy) from staged field data.
                    {
                    DeviceZoneScopedN("V21MB-EF-MATH");
                    for (uint32_t gc = 0; gc < g_size; ++gc) {
                        const uint32_t c = g_start + gc;
                        float gx_acc = 0.0f;
                        float gy_acc = 0.0f;

                        if (prep[gc].k_count > 0) {
                            const float    node_x    = prep[gc].node_x;
                            const float    node_y    = prep[gc].node_y;
                            const float    nsx       = prep[gc].nsx;
                            const float    nsy       = prep[gc].nsy;
                            const uint32_t k_count   = prep[gc].k_count;
                            const uint32_t h_count   = prep[gc].h_count;
                            const uint32_t bin_xl    = prep[gc].bin_xl;
                            const uint32_t bin_yl    = prep[gc].bin_yl;
                            const uint32_t intra_off = prep[gc].intra_off;
                            const uint8_t* fx_base   = reinterpret_cast<uint8_t*>(prep[gc].fx_l1_base);
                            const uint8_t* fy_base   = reinterpret_cast<uint8_t*>(prep[gc].fy_l1_base);

                            for (uint32_t kk = 0; kk < k_count; ++kk) {
                                const uint32_t k = bin_xl + kk;
                                const float bin_x_lo = xl + (float)k * bsx;
                                const float node_x_top = node_x + nsx;
                                const float bin_x_hi   = bin_x_lo + bsx;
                                const float px_top = (node_x_top < bin_x_hi) ? node_x_top : bin_x_hi;
                                const float px_bot = (node_x     > bin_x_lo) ? node_x     : bin_x_lo;
                                const float px = px_top - px_bot;

                                for (uint32_t hh = 0; hh < h_count; ++hh) {
                                    const uint32_t h = bin_yl + hh;
                                    const float bin_y_lo   = yl + (float)h * bsy;
                                    const float bin_y_hi   = bin_y_lo + bsy;
                                    const float node_y_top = node_y + nsy;
                                    const float py_top = (node_y_top < bin_y_hi) ? node_y_top : bin_y_hi;
                                    const float py_bot = (node_y     > bin_y_lo) ? node_y     : bin_y_lo;
                                    const float py = py_top - py_bot;

                                    const float fx = *reinterpret_cast<const float*>(fx_base + kk * V21_FIELD_ROW_BYTES + intra_off + hh * 4);
                                    const float fy = *reinterpret_cast<const float*>(fy_base + kk * V21_FIELD_ROW_BYTES + intra_off + hh * 4);
                                    const float area = px * py;
                                    gx_acc += area * fx;
                                    gy_acc += area * fy;
                                }
                            }
                        }

                        // Bake the negation from electric_potential.py:235.
                        grad_out_l1[c * 2 + 0] = -(gx_acc * prep[gc].ratio);
                        grad_out_l1[c * 2 + 1] = -(gy_acc * prep[gc].ratio);
                    }
                    }  // V21MB-EF-MATH
                }
            }

            // ── 4. Write (gx, gy) outputs to per-core DRAM page. ──
            // batch * 8 ≤ 64*8 = 512 B per write — well under the 96 KB silent-
            // corruption hazard on Blackhole (see memory note v15_spill_pgsz_bug).
            {
                DeviceZoneScopedN("V21MB-EF-WRITE-OUT");
                const uint32_t out_off   = cells_done * 8;
                const uint32_t out_bytes = (batch * 8 + V21_L1_WRITE_ALIGN - 1u) & ~(V21_L1_WRITE_ALIGN - 1u);
                noc_async_write((uint32_t)grad_out_l1, my_grad_base + out_off, out_bytes);
                noc_async_write_barrier();
            }

            cells_done += batch;
        }
    }
}
