// SPDX-License-Identifier: Apache-2.0
//
// V19 writeout (FP32) — converts uint32 fixed-point density slab in L1
// directly to fp32 and writes per-row to `density_buf` in DRAM, the same
// buffer TTNN-DCT consumes. Eliminates the host PCIe round-trip + uint32→fp32
// CPU loop that dominated the previous V19 gather (~9 ms of ~10 ms total
// at 2048 grid was host work; this kernel takes that to ~0 ms host).
//
// Layout assumptions (block partition):
//   * Core c owns bin_globals [c*slab_bins, (c+1)*slab_bins).
//   * Density slab in L1 is contiguous uint32, density_l1_off..+slab_bins*4.
//   * density_buf in DRAM is row-major fp32 [M, N], page_size = N*4 (1 row).
//   * slab_bins is multiple of 8 (host pads). M and N are powers of 2 ≥ 512,
//     so c*slab_bins is multiple of 8, and the per-row col_offsets are
//     multiples of 16 — no alignment hazards for noc_async_write.
//
// Algorithm:
//   1. In-place convert density_l1[i] uint32 → fp32 = (float)u * INV_SCALE.
//      (We no longer need the uint32 form once scatter is done.)
//   2. Walk the core's bin range row-by-row in density_buf:
//        first row partial (col_start .. N-1)
//        middle rows fully (col 0 .. N-1)
//        last row partial (col 0 .. col_end)
//      For each row, issue ONE noc_async_write of (cols_in_row * 4) bytes
//      from the L1 region holding that row's converted data.
//   3. Single barrier at the end — all writes target disjoint L1 regions and
//      disjoint DRAM regions, no cross-write hazards.

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t CB_SCRATCH = tt::CBIndex::c_24;

void kernel_main() {
    const uint32_t density_l1_off    = get_arg_val<uint32_t>(0);
    const uint32_t density_slab_bins = get_arg_val<uint32_t>(1);
    const uint32_t density_buf_addr  = get_arg_val<uint32_t>(2);
    const uint32_t density_buf_pgsz  = get_arg_val<uint32_t>(3);  // = N * sizeof(float)
    const uint32_t my_core_idx       = get_arg_val<uint32_t>(4);
    const uint32_t M                 = get_arg_val<uint32_t>(5);
    const uint32_t N                 = get_arg_val<uint32_t>(6);
    const uint32_t scale_bits        = get_arg_val<uint32_t>(7);

    const float INV_SCALE = 1.0f / (float)(1u << scale_bits);
    const uint32_t total_bins = M * N;

    uint8_t* base = reinterpret_cast<uint8_t*>(get_write_ptr(CB_SCRATCH));
    uint32_t* density_l1_u32 = reinterpret_cast<uint32_t*>(base + density_l1_off);
    float*    density_l1_fp  = reinterpret_cast<float*   >(base + density_l1_off);

    // Determine bin range this core owns within [0, total_bins).
    uint32_t bin_first = my_core_idx * density_slab_bins;
    if (bin_first >= total_bins) return;  // padding-only core
    uint32_t bin_last_exclusive = bin_first + density_slab_bins;
    if (bin_last_exclusive > total_bins) bin_last_exclusive = total_bins;
    uint32_t n_valid_bins = bin_last_exclusive - bin_first;

    // Phase 1: in-place uint32 -> fp32 conversion.
    {
        DeviceZoneScopedN("V19-FP-CONVERT");
        for (uint32_t i = 0; i < n_valid_bins; ++i) {
            density_l1_fp[i] = (float)density_l1_u32[i] * INV_SCALE;
        }
    }

    // Phase 2: per-row writes to density_buf.
    {
        DeviceZoneScopedN("V19-FP-WRITE");
        const InterleavedAddrGen<true> dgen = {
            .bank_base_address = density_buf_addr,
            .page_size         = density_buf_pgsz,
        };

        uint32_t row       = bin_first / N;
        uint32_t col       = bin_first - row * N;  // bin_first % N
        uint32_t l1_off    = 0;  // u32 index into density_l1_fp
        uint32_t bins_left = n_valid_bins;

        while (bins_left > 0u) {
            uint32_t bins_in_row = N - col;
            if (bins_in_row > bins_left) bins_in_row = bins_left;
            uint32_t bytes = bins_in_row * 4u;

            uint32_t l1_src_addr = reinterpret_cast<uint32_t>(&density_l1_fp[l1_off]);
            uint64_t dst_noc     = dgen.get_noc_addr(row, col * 4u);
            noc_async_write(l1_src_addr, dst_noc, bytes);

            l1_off    += bins_in_row;
            bins_left -= bins_in_row;
            col        = 0;
            ++row;
        }
        noc_async_write_barrier();
    }
}
