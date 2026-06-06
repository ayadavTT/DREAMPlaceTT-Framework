// SPDX-License-Identifier: Apache-2.0
//
// V31 backward — reads the forward STASH (route[c] = (cell,bin,weight) records,
// cell-owner pages, BALANCED: each core owns an equal cell slice) and computes
// grad with NO host, NO prep, NO atomic. Two cheap in-kernel passes:
//   PASS 1: chunk-stream route[c]'s bins → find this core's [min_bx,max_bx]
//           (V22_RELABEL spatial sort keeps it a narrow contiguous band).
//   load:   read that field band locally (int fixed-point).
//   PASS 2: chunk-stream route[c] again → accumulate gx/gy into an L1 grad slab
//           indexed by (cell - c*cpc); prod=(weight*f)>>15. Cell owned by c → no
//           cross-core atomic. One contiguous grad-slab write at the end.
// The band is discovered on-chip (no host computes it). route streamed in chunks
// so L1 holds only {chunk, band, grad-slab}, not the whole route.
// route record (16B): {cell, bin_global, weight_fixed(=ratio·px·py scaled), pad}.

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    const uint32_t my_core   = get_arg_val<uint32_t>(0);
    const uint32_t route_base= get_arg_val<uint32_t>(1);
    const uint32_t route_pg  = get_arg_val<uint32_t>(2);
    const uint32_t rcount    = get_arg_val<uint32_t>(3);
    const uint32_t slab_cells= get_arg_val<uint32_t>(4);   // cells this core owns (forward tile range)
    const uint32_t nbx_cap   = get_arg_val<uint32_t>(5);   // band CB capacity (columns)
    const uint32_t NBX       = get_arg_val<uint32_t>(6);
    const uint32_t fx_base   = get_arg_val<uint32_t>(8);
    const uint32_t fy_base   = get_arg_val<uint32_t>(9);
    const uint32_t col_pg    = get_arg_val<uint32_t>(10);  // NBY*4
    const uint32_t grad_base = get_arg_val<uint32_t>(11);
    const uint32_t grad_pg   = get_arg_val<uint32_t>(12);  // per-core grad page (max_slab_cells*8)
    const uint32_t cbase     = get_arg_val<uint32_t>(13);  // first global cell id this core owns
    const uint32_t chunk     = get_arg_val<uint32_t>(14);
    const uint32_t rcount1   = get_arg_val<uint32_t>(15);  // BRISC-half record count (page my_core+nc)
    const uint32_t nc        = get_arg_val<uint32_t>(16);  // core count (BRISC page = my_core+nc)

    constexpr auto CB_FX=tt::CBIndex::c_24, CB_FY=tt::CBIndex::c_25, CB_RT=tt::CBIndex::c_26, CB_G=tt::CBIndex::c_27;
    const InterleavedAddrGen<true> fxg={.bank_base_address=fx_base,.page_size=col_pg};
    const InterleavedAddrGen<true> fyg={.bank_base_address=fy_base,.page_size=col_pg};
    const InterleavedAddrGen<true> rg ={.bank_base_address=route_base,.page_size=route_pg};
    const InterleavedAddrGen<true> og ={.bank_base_address=grad_base,.page_size=grad_pg};
    const uint32_t NBYi=col_pg>>2;
    const uint32_t rt_l1=get_write_ptr(CB_RT);
    // The two half-pages: NCRISC cells [0,512)→page my_core, BRISC [512,1024)→page my_core+nc.
    const uint64_t pg_addr[2] = { rg.get_noc_addr(my_core), rg.get_noc_addr(my_core + nc) };
    const uint32_t pg_cnt[2]  = { rcount, rcount1 };

    // ── PASS 1: stream both half-pages' bins → [min_bx, max_bx] (band on-chip) ──
    uint32_t min_bx=0xFFFFFFFFu, max_bx=0;
    { DeviceZoneScopedN("V31B-SCAN");
      for(uint32_t p=0;p<2;++p){ const uint64_t base=pg_addr[p]; const uint32_t rc=pg_cnt[p];
        uint32_t done=0;
        while(done<rc){ uint32_t n=(rc-done<chunk)?(rc-done):chunk;
            noc_async_read(base+(uint64_t)done*16u, rt_l1, (n*16u+63u)&~63u); noc_async_read_barrier();
            const uint32_t* ri=(const uint32_t*)rt_l1;
            for(uint32_t i=0;i<n;++i){ uint32_t bx=ri[i*4u+1]/NBYi; if(bx<min_bx)min_bx=bx; if(bx>max_bx)max_bx=bx; }
            done+=n; } }
    }
    if(min_bx>max_bx){ min_bx=0; max_bx=0; }              // empty core
    uint32_t bx_lo=min_bx, n_bx=max_bx-min_bx+1u; if(n_bx>nbx_cap)n_bx=nbx_cap;

    // ── load field band [bx_lo, bx_lo+n_bx) locally (FLOAT — chip DCT field) ──
    float *fxs,*fys;
    { DeviceZoneScopedN("V31B-BAND");
      cb_reserve_back(CB_FX,1); uint32_t lx=get_write_ptr(CB_FX);
      cb_reserve_back(CB_FY,1); uint32_t ly=get_write_ptr(CB_FY);
      for(uint32_t c=0;c<n_bx;++c){ uint32_t col=bx_lo+c; if(col>=NBX)col=NBX-1u;
          noc_async_read(fxg.get_noc_addr(col), lx+c*col_pg, col_pg);
          noc_async_read(fyg.get_noc_addr(col), ly+c*col_pg, col_pg); }
      noc_async_read_barrier(); fxs=(float*)lx; fys=(float*)ly;
    }
    // grad slab (this core's cells), zeroed (FLOAT — area·field, no fixed-point scale)
    float* gs=(float*)get_write_ptr(CB_G);
    for(uint32_t i=0;i<slab_cells*2u;++i) gs[i]=0.0f;

    // ── PASS 2: stream both half-pages → accumulate grad (float; weight = area_fp32) ──
    { DeviceZoneScopedN("V31B-COMPUTE");
      for(uint32_t p=0;p<2;++p){ const uint64_t base=pg_addr[p]; const uint32_t rc=pg_cnt[p];
        uint32_t done=0;
        while(done<rc){ uint32_t n=(rc-done<chunk)?(rc-done):chunk;
            noc_async_read(base+(uint64_t)done*16u, rt_l1, (n*16u+63u)&~63u); noc_async_read_barrier();
            const uint32_t* ri=(const uint32_t*)rt_l1; const float* rf=(const float*)rt_l1;
            for(uint32_t i=0;i<n;++i){ const uint32_t b=i*4u;
                uint32_t cell=ri[b+0]; uint32_t bin=ri[b+1]; float w=rf[b+2];   // w = area (px·py)
                uint32_t bx=bin/NBYi, by=bin-bx*NBYi; uint32_t lc=bx-bx_lo;
                if(lc>=n_bx) continue;                       // guard: out-of-band (clamped) → skip
                uint32_t idx=lc*NBYi+by;
                uint32_t loc=(cell-cbase)*2u;
                gs[loc]+=w*fxs[idx]; gs[loc+1]+=w*fys[idx]; }
            done+=n; } }
    }
    noc_async_write((uint32_t)gs, og.get_noc_addr(my_core), slab_cells*8u);
    noc_async_write_barrier();
}
