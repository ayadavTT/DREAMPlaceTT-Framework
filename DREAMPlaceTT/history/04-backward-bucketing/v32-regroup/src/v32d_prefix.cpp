// SPDX-License-Identifier: Apache-2.0
// Bounded-page ADAPTIVE regroup — prefix/assignment (runs on ONE core). From the
// count matrix cnt[src][owner] it computes a worker assignment with a FIXED page
// capacity `cap` (≈ total/nc): a HOT owner (≥cap records) is split across several
// consecutive worker pages (each ≤cap, all reading that owner's 1 field slab); COLD
// owners are MERGED into shared workers (reading their few adjacent slabs). Total
// workers ≤ ~2·NC, route memory = workers·cap (no per-owner padding balloon). Mirrors
// the host-side V30 adaptive assignment, now on-chip (no host in the loop).
// Outputs: base[src][o] (within-owner src offset), w0[o]/slot0[o] (owner→first worker
// + start slot), and per-worker {lo_slab,n_slabs,nrec}+total_workers (wmeta).
#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    const uint32_t NC        = get_arg_val<uint32_t>(0);
    const uint32_t NOWN      = get_arg_val<uint32_t>(1);
    const uint32_t cntm_base = get_arg_val<uint32_t>(2);
    const uint32_t cntm_pg   = get_arg_val<uint32_t>(3);
    const uint32_t base_base = get_arg_val<uint32_t>(4);
    const uint32_t base_pg   = get_arg_val<uint32_t>(5);
    const uint32_t cap       = get_arg_val<uint32_t>(6);
    const uint32_t w0s_base  = get_arg_val<uint32_t>(7);   // page: w0[NOWN] then slot0[NOWN]
    const uint32_t w0s_pg    = get_arg_val<uint32_t>(8);
    const uint32_t wm_base   = get_arg_val<uint32_t>(9);    // page: [total_workers, lo_slab[MAXW], n_slabs[MAXW], nrec[MAXW]]
    const uint32_t wm_pg     = get_arg_val<uint32_t>(10);
    const uint32_t span_cap  = get_arg_val<uint32_t>(11);  // max slabs a merged cold worker may span (L1 field budget)
    const uint32_t MAXW      = get_arg_val<uint32_t>(12);  // worker-array stride in wmeta

    constexpr auto CB_CNT=tt::CBIndex::c_0, CB_BASE=tt::CBIndex::c_1, CB_TOT=tt::CBIndex::c_2,
                   CB_W0=tt::CBIndex::c_3, CB_WM=tt::CBIndex::c_4;
    const InterleavedAddrGen<true> cg={.bank_base_address=cntm_base,.page_size=cntm_pg};
    const InterleavedAddrGen<true> bg={.bank_base_address=base_base,.page_size=base_pg};
    const InterleavedAddrGen<true> w0g={.bank_base_address=w0s_base,.page_size=w0s_pg};
    const InterleavedAddrGen<true> wmg={.bank_base_address=wm_base,.page_size=wm_pg};
    const uint32_t stride=cntm_pg>>2;
    uint32_t* cnt =(uint32_t*)get_write_ptr(CB_CNT);   // NC*stride
    uint32_t* base=(uint32_t*)get_write_ptr(CB_BASE);  // NC*stride
    uint32_t* tot =(uint32_t*)get_write_ptr(CB_TOT);   // NOWN
    uint32_t* w0sl=(uint32_t*)get_write_ptr(CB_W0);    // 2*NOWN : w0[0..NOWN) , slot0[NOWN..2NOWN)
    uint32_t* wm  =(uint32_t*)get_write_ptr(CB_WM);    // 1+3*2NC

    { DeviceZoneScopedN("V32D-PREFIX");
      for(uint32_t c=0;c<NC;++c) noc_async_read(cg.get_noc_addr(c),(uint32_t)(cnt+c*stride),cntm_pg);
      noc_async_read_barrier();
      for(uint32_t o=0;o<NOWN;++o){ uint32_t t=0; for(uint32_t c=0;c<NC;++c) t+=cnt[c*stride+o]; tot[o]=t; }
      // base[src][o] = prefix over src of cnt (within-owner source offset)
      for(uint32_t o=0;o<NOWN;++o){ uint32_t run=0; for(uint32_t c=0;c<NC;++c){ base[c*stride+o]=run; run+=cnt[c*stride+o]; } }
      // assignment
      uint32_t* w0   = w0sl;            // [0..NOWN)
      uint32_t* slot0= w0sl+NOWN;       // [NOWN..2NOWN)
      uint32_t* lo_slab=wm+1, *n_slabs=wm+1+MAXW, *nrec=wm+1+2u*MAXW;
      uint32_t w=0, fill=0, merge_lo=0, merge_last=0; bool inmerge=false;
      for(uint32_t o=0;o<NOWN;++o){ uint32_t t=tot[o];
          if(t==0){ continue; }                                  // empty owner: no records (slab covered by spanning merge)
          if(t>=cap){                                            // HOT → dedicated split workers
              if(inmerge){ lo_slab[w]=merge_lo; n_slabs[w]=merge_last-merge_lo+1u; nrec[w]=fill; ++w; fill=0; inmerge=false; }
              w0[o]=w; slot0[o]=0u;
              uint32_t nparts=(t+cap-1u)/cap;
              for(uint32_t j=0;j<nparts;++j){ lo_slab[w]=o; n_slabs[w]=1u; uint32_t rem=t-j*cap; nrec[w]=(rem<cap)?rem:cap; ++w; }
          } else {                                               // COLD → merge into shared worker
              // close current merge if adding o would overflow records OR span > span_cap slabs
              if(inmerge && (fill+t>cap || (o-merge_lo+1u)>span_cap)){ lo_slab[w]=merge_lo; n_slabs[w]=merge_last-merge_lo+1u; nrec[w]=fill; ++w; fill=0; inmerge=false; }
              if(!inmerge){ merge_lo=o; fill=0; inmerge=true; }
              w0[o]=w; slot0[o]=fill; fill+=t; merge_last=o;
          }
      }
      if(inmerge){ lo_slab[w]=merge_lo; n_slabs[w]=merge_last-merge_lo+1u; nrec[w]=fill; ++w; }
      wm[0]=w;                                                   // total_workers
      for(uint32_t c=0;c<NC;++c) noc_async_write((uint32_t)(base+c*stride),bg.get_noc_addr(c),base_pg);
      noc_async_write((uint32_t)w0sl, w0g.get_noc_addr(0), w0s_pg);
      noc_async_write((uint32_t)wm,   wmg.get_noc_addr(0), wm_pg);
      noc_async_write_barrier();
    }
}
