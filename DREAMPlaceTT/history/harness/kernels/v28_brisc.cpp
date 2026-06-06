// SPDX-License-Identifier: Apache-2.0
//
// V28 EF backward — MULTI-BIN L1-resident gather (BRISC, no math).
// Generalizes V26 (4-corner) to variable k×h overlap sum, the real DREAMPlace
// electric_force. V21's field is COLUMN-major (page = x-column, M y-values), so
// a core owns a contiguous COLUMN-band [c·cpc, c·cpc+band_cols) (band_cols =
// cpc + MAX_KH halo) — one contiguous DRAM read, resident in L1. Per cell the
// gather reads field[(bin_xl_local+k)·M + (bin_yl+h)] by DIRECT L1 LOAD (no NoC)
// for k<max_k, h<max_h. SFPU (v28_compute) does Σ px[k]·py[h]·f.
//
// Cell record (96 B, 24 u32): [0]bin_xl_local [1]bin_yl [2]k_count [3]h_count
//   [4]ratio(f32) [5]pad [6..13]px[8] [14..21]py[8] [22,23]pad.

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    const uint32_t my_core    = get_arg_val<uint32_t>(0);
    const uint32_t ncells     = get_arg_val<uint32_t>(1);
    const uint32_t band_cols  = get_arg_val<uint32_t>(2);
    const uint32_t M          = get_arg_val<uint32_t>(3);   // num_bins_y (rows per column)
    const uint32_t max_k      = get_arg_val<uint32_t>(4);
    const uint32_t max_h      = get_arg_val<uint32_t>(5);
    const uint32_t fx_base    = get_arg_val<uint32_t>(6);
    const uint32_t fy_base    = get_arg_val<uint32_t>(7);
    const uint32_t field_pg   = get_arg_val<uint32_t>(8);   // full field page bytes (N*M*4)
    const uint32_t cells_base = get_arg_val<uint32_t>(9);
    const uint32_t cells_pg   = get_arg_val<uint32_t>(10);
    const uint32_t n_batches  = get_arg_val<uint32_t>(11);
    const uint32_t chunk_batches = get_arg_val<uint32_t>(12);
    const uint32_t col0       = get_arg_val<uint32_t>(13);  // this core's first global column (c*cpc)
    const uint32_t valid_cols = get_arg_val<uint32_t>(14);  // = min(band_cols, N-col0); band read clamped here

    constexpr uint32_t B = 1024;
    constexpr auto CB_PX=tt::CBIndex::c_0, CB_PY=tt::CBIndex::c_1, CB_FX=tt::CBIndex::c_2,
                   CB_FY=tt::CBIndex::c_3, CB_RATIO=tt::CBIndex::c_4, CB_BASE=tt::CBIndex::c_5;
    constexpr auto CB_FIELDX=tt::CBIndex::c_24, CB_CELLS=tt::CBIndex::c_25;
    uint32_t* base_idx;  // per-batch field base = bxl_loc*M+byl; pushed via CB_BASE to NCRISC (fy)

    const InterleavedAddrGen<true> fxg = {.bank_base_address=fx_base, .page_size=field_pg};
    const InterleavedAddrGen<true> fyg = {.bank_base_address=fy_base, .page_size=field_pg};
    const InterleavedAddrGen<true> cg  = {.bank_base_address=cells_base, .page_size=cells_pg};
    const uint32_t band_bytes = valid_cols * M * 4u;  // never read past the field buffer
    const uint32_t chunk_bytes = chunk_batches * B * 96u;
    const uint32_t REC = 24u;  // u32 per record

    // ── Field column-band: BRISC loads ONLY the fx band (NCRISC loads fy) ──
    float *fieldx;
    (void)fy_base;
    { DeviceZoneScopedN("V28-LOADBAND");
      cb_reserve_back(CB_FIELDX,1); uint32_t lx=get_write_ptr(CB_FIELDX);
      noc_async_read(fxg.get_noc_addr(0) + (uint64_t)col0*M*4u, lx, band_bytes);
      noc_async_read_barrier();
      fieldx=reinterpret_cast<float*>(lx);
      // zero the tail beyond the loaded band (edge cores / pad) so absent-bin
      // reads (unclamped) return 0, not uninitialized nan → px(0)*0 = 0.
      for (uint32_t i=valid_cols*M; i<band_cols*M+max_h+8u; ++i) fieldx[i]=0.f;
    }
    const uint32_t cells_l1 = get_write_ptr(CB_CELLS);
    const uint32_t* cu = reinterpret_cast<const uint32_t*>(cells_l1);
    const float*    cf = reinterpret_cast<const float*>(cells_l1);

    uint32_t chunk_base = 0;
    for (uint32_t b = 0; b < n_batches; ++b) {
        if (b % chunk_batches == 0) {
            DeviceZoneScopedN("V28-LOADCHUNK");
            uint32_t this_chunk = chunk_bytes;
            if (chunk_base + this_chunk > cells_pg) this_chunk = cells_pg - chunk_base;
            noc_async_read(cg.get_noc_addr(my_core) + chunk_base, cells_l1, this_chunk);
            noc_async_read_barrier();
            chunk_base += this_chunk;
        }
        const uint32_t g0 = b * B;
        const uint32_t lb = b % chunk_batches;

        // ── px[max_k], py[max_h], ratio tiles (lane = cell) ──
        { DeviceZoneScopedN("V28-PREP");
          cb_reserve_back(CB_PX,max_k); cb_reserve_back(CB_PY,max_h); cb_reserve_back(CB_RATIO,1);
          cb_reserve_back(CB_BASE,1);
          float* pxb=(float*)get_write_ptr(CB_PX);
          float* pyb=(float*)get_write_ptr(CB_PY);
          float* rab=(float*)get_write_ptr(CB_RATIO);
          base_idx=(uint32_t*)get_write_ptr(CB_BASE);
          for (uint32_t j=0;j<B;++j) {
              const uint32_t c=g0+j;
              if (c<ncells) {
                  const uint32_t base=(lb*B+j)*REC;
                  for (uint32_t k=0;k<max_k;++k) pxb[k*B+j]=cf[base+6+k];
                  for (uint32_t h=0;h<max_h;++h) pyb[h*B+j]=cf[base+14+h];
                  rab[j]=cf[base+4];
                  base_idx[j]=cu[base+0]*M + cu[base+1];   // hoist the per-(k,h) multiply
              } else {
                  for (uint32_t k=0;k<max_k;++k) pxb[k*B+j]=0.f;
                  for (uint32_t h=0;h<max_h;++h) pyb[h*B+j]=0.f;
                  rab[j]=0.f; base_idx[j]=0u;
              }
          }
          cb_push_back(CB_PX,max_k); cb_push_back(CB_PY,max_h); cb_push_back(CB_RATIO,1);
          cb_push_back(CB_BASE,1);   // hand base_idx to NCRISC for the fy gather
        }

        // ── per (k,h): BRISC gathers ONLY fx tiles (NCRISC does fy in parallel) ──
        for (uint32_t k=0;k<max_k;++k) {
            for (uint32_t h=0;h<max_h;++h) {
                DeviceZoneScopedN("V28-GATHER");
                cb_reserve_back(CB_FX,1);
                float* fxp=(float*)get_write_ptr(CB_FX);
                const uint32_t kh = k*M + h;            // constant per (k,h)
                for (uint32_t j=0;j<B;++j) fxp[j]=fieldx[base_idx[j]+kh];
                cb_push_back(CB_FX,1);
            }
        }
    }
}
