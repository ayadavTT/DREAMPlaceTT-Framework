// SPDX-License-Identifier: Apache-2.0
//
// V35 place — 64 B SOURCE + FOOTPRINT-BUCKET variant (BRISC, CBs c_0..c_4). Like
// v35_place_s64.cpp but the counting-sort key is group = 2*tile + bucket (ngroups =
// 2*ntiles), bucket = large iff any px[k]!=0 for k in [K0,max_k) OR any py[h]!=0 for
// h in [H0,max_h). Because group 2t (small) is written before 2t+1 (large) and they are
// contiguous, each tile's run is small-cells-FIRST then the large tail — exactly what the
// per-batch-footprint gather needs (per-worker n_small). The emitted 64 B compact grouped
// record is unchanged (px[0..max_k), py @ 4+max_k). Args 13..16 = max_k,max_h,K0,H0.

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    const uint32_t my_core    = get_arg_val<uint32_t>(0);
    const uint32_t my_first   = get_arg_val<uint32_t>(1);
    const uint32_t my_n       = get_arg_val<uint32_t>(2);
    const uint32_t stash_base = get_arg_val<uint32_t>(3);
    const uint32_t stash_pg   = get_arg_val<uint32_t>(4);
    const uint32_t ntiles     = get_arg_val<uint32_t>(5);
    const uint32_t W          = get_arg_val<uint32_t>(6);
    const uint32_t grouped_base = get_arg_val<uint32_t>(7);
    const uint32_t grouped_pg = get_arg_val<uint32_t>(8);
    const uint32_t tilebase_base = get_arg_val<uint32_t>(9);
    const uint32_t srcpref_base  = get_arg_val<uint32_t>(10);
    const uint32_t plan_pg    = get_arg_val<uint32_t>(11);
    const uint32_t chunk      = get_arg_val<uint32_t>(12);
    const uint32_t max_k      = get_arg_val<uint32_t>(13);
    const uint32_t max_h      = get_arg_val<uint32_t>(14);
    const uint32_t K0         = get_arg_val<uint32_t>(15);
    const uint32_t H0         = get_arg_val<uint32_t>(16);

    constexpr uint32_t REC_S = 16u;   // 64 B source record
    constexpr uint32_t REC_C = 16u;   // 64 B compact grouped record
    const uint32_t ngroups = 2u*ntiles;

    constexpr auto CB_IN=tt::CBIndex::c_0, CB_SORT=tt::CBIndex::c_1, CB_TB=tt::CBIndex::c_2,
                   CB_SP=tt::CBIndex::c_3, CB_SCR=tt::CBIndex::c_4;
    const InterleavedAddrGen<true> ing  = {.bank_base_address=stash_base, .page_size=stash_pg};
    const InterleavedAddrGen<true> gg   = {.bank_base_address=grouped_base, .page_size=grouped_pg};
    const InterleavedAddrGen<true> tbg  = {.bank_base_address=tilebase_base, .page_size=ngroups*4u};
    const InterleavedAddrGen<true> spg  = {.bank_base_address=srcpref_base, .page_size=plan_pg};

    cb_reserve_back(CB_TB,1); uint32_t* tb=(uint32_t*)get_write_ptr(CB_TB);
    noc_async_read(tbg.get_noc_addr(0), (uint32_t)tb, ngroups*4u); noc_async_read_barrier();
    cb_reserve_back(CB_SP,1); uint32_t* sp=(uint32_t*)get_write_ptr(CB_SP);
    noc_async_read(spg.get_noc_addr(my_core), (uint32_t)sp, ngroups*4u); noc_async_read_barrier();
    cb_reserve_back(CB_SCR,1); uint32_t* scr=(uint32_t*)get_write_ptr(CB_SCR);
    uint32_t* wr=scr; uint32_t* hist=scr+ngroups; uint32_t* off=scr+2u*ngroups;
    for (uint32_t g=0;g<ngroups;++g) wr[g]=tb[g]+sp[g];

    const uint32_t in_l1   = get_write_ptr(CB_IN);
    const uint32_t sort_l1 = get_write_ptr(CB_SORT);

    { DeviceZoneScopedN("V35-PLACE");
      uint32_t done=0;
      while (done < my_n) {
          uint32_t this_n = (my_n-done < chunk) ? (my_n-done) : chunk;
          noc_async_read(ing.get_noc_addr(0) + (uint64_t)(my_first+done)*64u, in_l1, this_n*64u);
          noc_async_read_barrier();
          const uint32_t* rec = reinterpret_cast<const uint32_t*>(in_l1);
          for (uint32_t g=0;g<ngroups;++g) hist[g]=0;
          for (uint32_t i=0;i<this_n;++i){ const uint32_t* s=rec+i*REC_S;
              uint32_t t=s[1]/W; if(t>=ntiles)t=ntiles-1u; uint32_t big=0u;
              for (uint32_t k=K0;k<max_k;++k) big|=s[4u+k];
              for (uint32_t h=H0;h<max_h;++h) big|=s[12u+h];
              hist[2u*t+(big?1u:0u)]++; }
          uint32_t acc=0; for (uint32_t g=0;g<ngroups;++g){ off[g]=acc; acc+=hist[g]; }
          uint32_t* sorted = reinterpret_cast<uint32_t*>(sort_l1);
          uint32_t* cur = scr + 3u*ngroups; for (uint32_t g=0;g<ngroups;++g) cur[g]=off[g];
          for (uint32_t i=0;i<this_n;++i){ const uint32_t* s=rec+i*REC_S;
              uint32_t t=s[1]/W; if(t>=ntiles)t=ntiles-1u; uint32_t big=0u;
              for (uint32_t k=K0;k<max_k;++k) big|=s[4u+k];
              for (uint32_t h=H0;h<max_h;++h) big|=s[12u+h];
              uint32_t g=2u*t+(big?1u:0u);
              uint32_t d=cur[g]*REC_C;
              sorted[d+0]=s[0]; sorted[d+1]=s[1]; sorted[d+2]=s[2]; sorted[d+3]=s[3];
              for (uint32_t k=0;k<max_k;++k) sorted[d+4u+k]=s[4u+k];
              for (uint32_t h=0;h<max_h;++h) sorted[d+4u+max_k+h]=s[12u+h];
              cur[g]++; }
          for (uint32_t g=0;g<ngroups;++g){ if(hist[g]){
              noc_async_write(sort_l1 + (uint64_t)off[g]*64u, gg.get_noc_addr(0) + (uint64_t)wr[g]*64u, hist[g]*64u);
              wr[g]+=hist[g]; } }
          noc_async_write_barrier();
          done += this_n;
      }
    }
}
