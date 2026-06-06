// SPDX-License-Identifier: Apache-2.0
//
// V28 EF backward — NCRISC half of the split gather. Loads ONLY the fy band into
// L1, consumes base_idx from BRISC (CB_BASE), and fills the fy tiles per (k,h) by
// direct L1 load — in PARALLEL with BRISC filling the fx tiles. Then drains the
// SFPU's gx/gy tiles to DRAM (writer role). Splitting fx/fy across the two
// dataflow cores halves the per-RISC L1-load throughput bottleneck AND the band
// load (each RISC loads one field map).

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    const uint32_t my_core    = get_arg_val<uint32_t>(0);
    const uint32_t n_batches  = get_arg_val<uint32_t>(1);
    const uint32_t M          = get_arg_val<uint32_t>(2);
    const uint32_t max_k      = get_arg_val<uint32_t>(3);
    const uint32_t max_h      = get_arg_val<uint32_t>(4);
    const uint32_t fy_base    = get_arg_val<uint32_t>(5);
    const uint32_t field_pg   = get_arg_val<uint32_t>(6);
    const uint32_t col0       = get_arg_val<uint32_t>(7);
    const uint32_t valid_cols = get_arg_val<uint32_t>(8);
    const uint32_t gx_base    = get_arg_val<uint32_t>(9);
    const uint32_t gy_base    = get_arg_val<uint32_t>(10);
    const uint32_t out_pg     = get_arg_val<uint32_t>(11);
    const uint32_t band_cols  = get_arg_val<uint32_t>(12);

    constexpr uint32_t B=1024, TILE_BYTES=1024u*4u;
    constexpr auto CB_FY=tt::CBIndex::c_3, CB_BASE=tt::CBIndex::c_5;
    constexpr auto GX=tt::CBIndex::c_16, GY=tt::CBIndex::c_17, CB_FIELDY=tt::CBIndex::c_26;

    const InterleavedAddrGen<true> fyg={.bank_base_address=fy_base, .page_size=field_pg};
    const InterleavedAddrGen<true> gxg={.bank_base_address=gx_base, .page_size=out_pg};
    const InterleavedAddrGen<true> gyg={.bank_base_address=gy_base, .page_size=out_pg};
    const uint64_t gx_page=gxg.get_noc_addr(my_core), gy_page=gyg.get_noc_addr(my_core);

    // ── fy band resident in L1 (NCRISC owns it) ──
    float* fieldy;
    { DeviceZoneScopedN("V28N-LOADBAND");
      cb_reserve_back(CB_FIELDY,1); uint32_t ly=get_write_ptr(CB_FIELDY);
      // page-interleaved field across DRAM banks → read page-by-page (x-bin
      // col0+i → L1 offset i*M), NOT one contiguous block (see v35_gather_brisc).
      for (uint32_t i=0;i<valid_cols;++i)
          noc_async_read(fyg.get_noc_addr(col0+i), ly + i*M*4u, M*4u);
      noc_async_read_barrier(); fieldy=reinterpret_cast<float*>(ly);
      for (uint32_t i=valid_cols*M; i<band_cols*M+max_h+8u; ++i) fieldy[i]=0.f;
    }

    for (uint32_t b=0;b<n_batches;++b) {
        cb_wait_front(CB_BASE,1);
        const uint32_t* base_idx=reinterpret_cast<const uint32_t*>(get_read_ptr(CB_BASE));
        for (uint32_t k=0;k<max_k;++k) for (uint32_t h=0;h<max_h;++h) {
            { DeviceZoneScopedN("NB-FYRESV"); cb_reserve_back(CB_FY,1); }   // NCRISC blocked by SFPU (CB full) if large
            { DeviceZoneScopedN("NB-FYFILL");                              // the actual fy-tile gather from L1 band
              float* fyp=(float*)get_write_ptr(CB_FY);
              const uint32_t kh=k*M+h;
              for (uint32_t j=0;j<B;++j) fyp[j]=fieldy[base_idx[j]+kh];
              cb_push_back(CB_FY,1); }
        }
        cb_pop_front(CB_BASE,1);
        // drain this batch's gx/gy (produced by the SFPU after consuming fx+fy)
        { DeviceZoneScopedN("NB-DRAINWAIT"); cb_wait_front(GX,1); cb_wait_front(GY,1); }  // NCRISC waiting for SFPU output → SFPU is the gate if large
        { DeviceZoneScopedN("NB-DRAINWR");
          noc_async_write(get_read_ptr(GX), gx_page+(uint64_t)b*TILE_BYTES, TILE_BYTES);
          noc_async_write(get_read_ptr(GY), gy_page+(uint64_t)b*TILE_BYTES, TILE_BYTES);
          noc_async_write_barrier();
          cb_pop_front(GX,1); cb_pop_front(GY,1);
        }
    }
}
