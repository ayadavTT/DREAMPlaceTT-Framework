// SPDX-License-Identifier: Apache-2.0
//
// V25 EF backward — L1-resident field, direct-L1-load corner gather (BRISC).
//
// Decisive finding (grid_sample C-sweep): the EF gather is READ-COUNT bound —
// 2.16M cell×corner field reads, each a NoC txn (~5 ns) → ~10 ms; CPU does them
// cache-resident (~free) → 7.3 ms. To beat CPU, each corner read must be a
// DIRECT L1 LOAD, not a NoC round-trip. This kernel:
//   1. Loads this core's field band (its rows + halo) into L1 ONCE.
//   2. For each cell assigned to this core, reads the 4 corner fx/fy via direct
//      L1 pointer loads (no NoC) + bilinear weighted sum.
//   3. Writes grad out.
//
// Cell record (24 B, 6 u32): {bxl, byl_local, w_nw, w_ne, w_sw, w_se}
//   weights fold in ratio (w = ratio·px·py). Field band: field[(r*W+col)*2+ch],
//   ch 0=fx 1=fy. Each DRAM buffer is interleaved with one PAGE per core.
//
// Tracy: V25-LOADBAND, V25-LOADCELLS, V25-GATHER (hot loop), V25-WRITE.

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    const uint32_t my_core         = get_arg_val<uint32_t>(0);
    const uint32_t ncells          = get_arg_val<uint32_t>(1);
    const uint32_t band_bytes      = get_arg_val<uint32_t>(2);
    const uint32_t W               = get_arg_val<uint32_t>(3);
    const uint32_t field_base      = get_arg_val<uint32_t>(4);
    const uint32_t cells_base      = get_arg_val<uint32_t>(5);
    const uint32_t grad_base       = get_arg_val<uint32_t>(6);
    const uint32_t cells_pg_bytes  = get_arg_val<uint32_t>(7);
    const uint32_t grad_pg_bytes   = get_arg_val<uint32_t>(8);

    constexpr uint32_t CB_FIELD = tt::CBIndex::c_0;
    constexpr uint32_t CB_CELLS = tt::CBIndex::c_1;
    constexpr uint32_t CB_GRAD  = tt::CBIndex::c_2;

    const InterleavedAddrGen<true> field_gen = {.bank_base_address = field_base, .page_size = band_bytes};
    const InterleavedAddrGen<true> cells_gen = {.bank_base_address = cells_base, .page_size = cells_pg_bytes};
    const InterleavedAddrGen<true> grad_gen  = {.bank_base_address = grad_base,  .page_size = grad_pg_bytes};

    // ── 1. Load this core's field band DRAM → L1 (once) ──
    float* field;
    { DeviceZoneScopedN("V25-LOADBAND");
      cb_reserve_back(CB_FIELD, 1);
      const uint32_t l1 = get_write_ptr(CB_FIELD);
      noc_async_read(field_gen.get_noc_addr(my_core), l1, band_bytes);
      noc_async_read_barrier();
      field = reinterpret_cast<float*>(l1);
    }

    // ── 2. Load this core's cell records DRAM → L1 ──
    uint32_t* cells;
    { DeviceZoneScopedN("V25-LOADCELLS");
      cb_reserve_back(CB_CELLS, 1);
      const uint32_t l1 = get_write_ptr(CB_CELLS);
      noc_async_read(cells_gen.get_noc_addr(my_core), l1, cells_pg_bytes);
      noc_async_read_barrier();
      cells = reinterpret_cast<uint32_t*>(l1);
    }

    cb_reserve_back(CB_GRAD, 1);
    const uint32_t grad_l1 = get_write_ptr(CB_GRAD);

    // V25-v2: FIXED-POINT INTEGER weighted sum. BRISC (RISC-V) has hardware
    // integer mul (M-ext) but NO FPU (float = ~90-cyc soft-float). The field is
    // uploaded as int32 (field·2^FRAC) and weights as int32 (w·2^FRAC); the
    // product is field·w·2^(2·FRAC). int64 accumulate avoids overflow; the
    // output is int32 (gx·2^(2·FRAC)) and the host descales by 2^(2·FRAC).
    const int32_t* fieldi = reinterpret_cast<const int32_t*>(field);
    const int32_t* cellsi = reinterpret_cast<const int32_t*>(cells);
    int32_t* gradi = reinterpret_cast<int32_t*>(grad_l1);

    // ── 3. Hot loop: direct-L1-load corner gather + integer weighted sum ──
    { DeviceZoneScopedN("V25-GATHER");
      for (uint32_t c = 0; c < ncells; ++c) {
          const uint32_t base = c * 6u;
          const uint32_t bxl  = (uint32_t)cellsi[base + 0];
          const uint32_t byl  = (uint32_t)cellsi[base + 1];
          const int32_t w_nw = cellsi[base + 2];
          const int32_t w_ne = cellsi[base + 3];
          const int32_t w_sw = cellsi[base + 4];
          const int32_t w_se = cellsi[base + 5];

          const uint32_t row0 = byl * W + bxl;
          const uint32_t i_nw = row0 * 2u;
          const uint32_t i_ne = (row0 + 1u) * 2u;
          const uint32_t i_sw = (row0 + W) * 2u;
          const uint32_t i_se = (row0 + W + 1u) * 2u;

          // int32 accumulate: products ≤ 2^26, sum of 4 ≤ 2^28 < int32 max.
          // (avoids RV32 int64 widening-multiply path)
          const int32_t gx = w_nw * fieldi[i_nw]     + w_ne * fieldi[i_ne]
                           + w_sw * fieldi[i_sw]     + w_se * fieldi[i_se];
          const int32_t gy = w_nw * fieldi[i_nw + 1] + w_ne * fieldi[i_ne + 1]
                           + w_sw * fieldi[i_sw + 1] + w_se * fieldi[i_se + 1];
          gradi[c * 2u]      = gx;   // scaled by 2^(2*FRAC); host descales
          gradi[c * 2u + 1u] = gy;
      }
    }

    // ── 4. Write grad L1 → DRAM ──
    { DeviceZoneScopedN("V25-WRITE");
      noc_async_write(grad_l1, grad_gen.get_noc_addr(my_core), grad_pg_bytes);
      noc_async_write_barrier();
    }
}
