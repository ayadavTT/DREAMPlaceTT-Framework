// SPDX-License-Identifier: Apache-2.0
//
// V35 gather (BRISC) — V28 multi-bin L1-resident gather, but cells come from a
// flat slice of the single tile-grouped buffer (PASS-2 output) instead of a
// per-core page. Each gather worker owns [slice_first, slice_first+ncells) records
// of grouped_buf, all in ONE tile → field band [col0, col0+valid_cols) (col0 =
// tile*W) resident in L1, +max_k halo. Per cell: fx[((bxl_g-col0)+k)*M+(byl+h)] by
// direct L1 load; SFPU (v28_compute) does Σ px·py·f. 128 B record (32 u32, V31
// stash format, bxl GLOBAL → 64-B aligned offsets):
//   [0]gidx [1]bxl_g [2]byl [3]kc [4]hc [5]ratio [6..13]px [14..21]py.

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
    const uint32_t M          = get_arg_val<uint32_t>(3);
    const uint32_t max_k      = get_arg_val<uint32_t>(4);
    const uint32_t max_h      = get_arg_val<uint32_t>(5);
    const uint32_t fx_base    = get_arg_val<uint32_t>(6);
    const uint32_t fy_base    = get_arg_val<uint32_t>(7);
    const uint32_t field_pg   = get_arg_val<uint32_t>(8);
    const uint32_t grouped_base = get_arg_val<uint32_t>(9);
    const uint32_t grouped_pg = get_arg_val<uint32_t>(10);   // whole grouped buffer bytes
    const uint32_t slice_first= get_arg_val<uint32_t>(11);   // first record (in grouped_buf) of this worker
    const uint32_t n_batches  = get_arg_val<uint32_t>(12);
    const uint32_t chunk_batches = get_arg_val<uint32_t>(13);
    const uint32_t col0       = get_arg_val<uint32_t>(14);   // tile*W
    const uint32_t valid_cols = get_arg_val<uint32_t>(15);
    const uint32_t oidx_base  = get_arg_val<uint32_t>(16);   // per-core oidx page (gidx per output slot)
    const uint32_t oidx_pg    = get_arg_val<uint32_t>(17);

    constexpr uint32_t B = 1024;
    constexpr auto CB_PX=tt::CBIndex::c_0, CB_PY=tt::CBIndex::c_1, CB_FX=tt::CBIndex::c_2,
                   CB_RATIO=tt::CBIndex::c_4, CB_BASE=tt::CBIndex::c_5, CB_OIDX=tt::CBIndex::c_6;
    constexpr auto CB_FIELDX=tt::CBIndex::c_24, CB_CELLS=tt::CBIndex::c_25;
    uint32_t* base_idx;

    const InterleavedAddrGen<true> fxg = {.bank_base_address=fx_base, .page_size=field_pg};
    const InterleavedAddrGen<true> cg  = {.bank_base_address=grouped_base, .page_size=grouped_pg};
    const InterleavedAddrGen<true> oig = {.bank_base_address=oidx_base, .page_size=oidx_pg};
    (void)fy_base;
    const uint32_t REC = 32u;                                     // 128 B V31 record
    const uint32_t chunk_bytes = chunk_batches * B * REC * 4u;
    const uint64_t my_cell_byte0 = (uint64_t)slice_first * 128u;  // flat offset (64-B aligned)

    // ── fx band resident in L1 (BRISC) ──
    float *fieldx;
    { DeviceZoneScopedN("V35-LOADBAND");
      cb_reserve_back(CB_FIELDX,1); uint32_t lx=get_write_ptr(CB_FIELDX);
      // The field is page-interleaved (ONE page = M floats per x-bin) and striped
      // across DRAM banks, so a single contiguous read from page 0 + col0*M reads
      // garbage. Read page-by-page (x-bin col0+i → L1 offset i*M), like V21.
      for (uint32_t i=0;i<valid_cols;++i)
          noc_async_read(fxg.get_noc_addr(col0+i), lx + i*M*4u, M*4u);
      noc_async_read_barrier();
      fieldx=reinterpret_cast<float*>(lx);
      for (uint32_t i=valid_cols*M; i<band_cols*M+max_h+8u; ++i) fieldx[i]=0.f;
    }
    const uint32_t cells_l1 = get_write_ptr(CB_CELLS);
    const uint32_t* cu = reinterpret_cast<const uint32_t*>(cells_l1);
    const float*    cf = reinterpret_cast<const float*>(cells_l1);

    uint32_t chunk_base = 0;   // bytes within this worker's slice
    const uint32_t slice_bytes = ncells * 128u;
    for (uint32_t b = 0; b < n_batches; ++b) {
        if (b % chunk_batches == 0) {
            DeviceZoneScopedN("V35-LOADCHUNK");
            uint32_t this_chunk = chunk_bytes;
            if (chunk_base + this_chunk > slice_bytes) this_chunk = (chunk_base<slice_bytes)?(slice_bytes - chunk_base):0u;
            if (this_chunk) { noc_async_read(cg.get_noc_addr(0) + my_cell_byte0 + chunk_base, cells_l1, this_chunk);
                              noc_async_read_barrier(); }
            chunk_base += chunk_bytes;
        }
        const uint32_t g0 = b * B;
        const uint32_t lb = b % chunk_batches;

        { DeviceZoneScopedN("V35-PREP");
          cb_reserve_back(CB_PX,max_k); cb_reserve_back(CB_PY,max_h); cb_reserve_back(CB_RATIO,1);
          cb_reserve_back(CB_BASE,1);
          // CB_OIDX is NOT flow-controlled: nothing pops it (the oib slot is drained
          // straight to DRAM via noc_async_write below). Reserving/pushing it every
          // batch with only 2 CB slots dead-locks at the 3rd batch (>2048 cells/core).
          // Use it as a fixed L1 scratch instead — the write barrier serializes reuse.
          float* pxb=(float*)get_write_ptr(CB_PX);
          float* pyb=(float*)get_write_ptr(CB_PY);
          float* rab=(float*)get_write_ptr(CB_RATIO);
          base_idx=(uint32_t*)get_write_ptr(CB_BASE);
          uint32_t* oib=(uint32_t*)get_write_ptr(CB_OIDX);   // gidx per output slot (for host/on-chip grad map)
          for (uint32_t j=0;j<B;++j) {
              const uint32_t c=g0+j;
              if (c<ncells) {
                  const uint32_t base=(lb*B+j)*REC;   // V31: [0]gidx [1]bxl_g [2]byl [5]ratio [6..13]px [14..21]py
                  for (uint32_t k=0;k<max_k;++k) pxb[k*B+j]=cf[base+6+k];
                  for (uint32_t h=0;h<max_h;++h) pyb[h*B+j]=cf[base+14+h];
                  rab[j]=cf[base+5];
                  base_idx[j]=(cu[base+1]-col0)*M + cu[base+2];   // (bxl_g-col0)*M + byl
                  oib[j]=cu[base+0];
              } else {
                  for (uint32_t k=0;k<max_k;++k) pxb[k*B+j]=0.f;
                  for (uint32_t h=0;h<max_h;++h) pyb[h*B+j]=0.f;
                  rab[j]=0.f; base_idx[j]=0u; oib[j]=0xffffffffu;
              }
          }
          // drain this batch's gidx to the per-core oidx page (slot b*B+j)
          noc_async_write((uint32_t)oib, oig.get_noc_addr(my_core) + (uint64_t)b*B*4u, B*4u);
          noc_async_write_barrier();
          cb_push_back(CB_PX,max_k); cb_push_back(CB_PY,max_h); cb_push_back(CB_RATIO,1);
          cb_push_back(CB_BASE,1);   // CB_OIDX intentionally not pushed (see above)
        }

        for (uint32_t k=0;k<max_k;++k) {
            for (uint32_t h=0;h<max_h;++h) {
                DeviceZoneScopedN("V35-GATHER");
                cb_reserve_back(CB_FX,1);
                float* fxp=(float*)get_write_ptr(CB_FX);
                const uint32_t kh = k*M + h;
                for (uint32_t j=0;j<B;++j) fxp[j]=fieldx[base_idx[j]+kh];
                cb_push_back(CB_FX,1);
            }
        }
    }
}
