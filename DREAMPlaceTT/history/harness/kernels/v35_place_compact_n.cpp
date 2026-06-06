// SPDX-License-Identifier: Apache-2.0
//
// V35 place (compact) — NCRISC half of the DUAL-RISC place. Identical to
// v35_place_compact.cpp but uses CB indices c_5..c_9 (instead of c_0..c_4) so the
// NCRISC half does not clobber the BRISC half's L1 when both run on the same core.
// arg0 (my_core) is the SOURCE id: it reads srcpref page `my_core` and writes its
// cells into the grouped buffer at tile_base[t]+srcpref[my_core][t] — disjoint from
// the BRISC half (which uses source id `c+nc`). Emits the 64 B compact record.

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

    constexpr uint32_t REC_S = 32u;   // source stash record (128 B)
    constexpr uint32_t REC_C = 16u;   // compact grouped record (64 B)

    constexpr auto CB_IN=tt::CBIndex::c_5, CB_SORT=tt::CBIndex::c_6, CB_TB=tt::CBIndex::c_7,
                   CB_SP=tt::CBIndex::c_8, CB_SCR=tt::CBIndex::c_9;   // NCRISC CBs (shifted)
    const InterleavedAddrGen<true> ing  = {.bank_base_address=stash_base, .page_size=stash_pg};
    const InterleavedAddrGen<true> gg   = {.bank_base_address=grouped_base, .page_size=grouped_pg};
    const InterleavedAddrGen<true> tbg  = {.bank_base_address=tilebase_base, .page_size=ntiles*4u};
    const InterleavedAddrGen<true> spg  = {.bank_base_address=srcpref_base, .page_size=plan_pg};

    cb_reserve_back(CB_TB,1); uint32_t* tb=(uint32_t*)get_write_ptr(CB_TB);
    noc_async_read(tbg.get_noc_addr(0), (uint32_t)tb, ntiles*4u); noc_async_read_barrier();
    cb_reserve_back(CB_SP,1); uint32_t* sp=(uint32_t*)get_write_ptr(CB_SP);
    noc_async_read(spg.get_noc_addr(my_core), (uint32_t)sp, ntiles*4u); noc_async_read_barrier();
    cb_reserve_back(CB_SCR,1); uint32_t* scr=(uint32_t*)get_write_ptr(CB_SCR);
    uint32_t* wr=scr; uint32_t* hist=scr+ntiles; uint32_t* off=scr+2u*ntiles;
    for (uint32_t t=0;t<ntiles;++t) wr[t]=tb[t]+sp[t];

    const uint32_t in_l1   = get_write_ptr(CB_IN);
    const uint32_t sort_l1 = get_write_ptr(CB_SORT);

    { DeviceZoneScopedN("V35-PLACE-N");
      uint32_t done=0;
      while (done < my_n) {
          uint32_t this_n = (my_n-done < chunk) ? (my_n-done) : chunk;
          noc_async_read(ing.get_noc_addr(0) + (uint64_t)(my_first+done)*128u, in_l1, this_n*128u);
          noc_async_read_barrier();
          const uint32_t* rec = reinterpret_cast<const uint32_t*>(in_l1);
          for (uint32_t t=0;t<ntiles;++t) hist[t]=0;
          for (uint32_t i=0;i<this_n;++i){ uint32_t t=rec[i*REC_S+1u]/W; if(t>=ntiles)t=ntiles-1u; hist[t]++; }
          uint32_t acc=0; for (uint32_t t=0;t<ntiles;++t){ off[t]=acc; acc+=hist[t]; }
          uint32_t* sorted = reinterpret_cast<uint32_t*>(sort_l1);
          uint32_t* cur = scr + 3u*ntiles; for (uint32_t t=0;t<ntiles;++t) cur[t]=off[t];
          for (uint32_t i=0;i<this_n;++i){ uint32_t t=rec[i*REC_S+1u]/W; if(t>=ntiles)t=ntiles-1u;
              uint32_t d=cur[t]*REC_C; const uint32_t* s=rec+i*REC_S;
              sorted[d+0]=s[0]; sorted[d+1]=s[1]; sorted[d+2]=s[2]; sorted[d+3]=s[5];
              for (uint32_t k=0;k<max_k;++k) sorted[d+4u+k]=s[6u+k];
              for (uint32_t h=0;h<max_h;++h) sorted[d+4u+max_k+h]=s[14u+h];
              cur[t]++; }
          for (uint32_t t=0;t<ntiles;++t){ if(hist[t]){
              noc_async_write(sort_l1 + (uint64_t)off[t]*64u, gg.get_noc_addr(0) + (uint64_t)wr[t]*64u, hist[t]*64u);
              wr[t]+=hist[t]; } }
          noc_async_write_barrier();
          done += this_n;
      }
    }
}
