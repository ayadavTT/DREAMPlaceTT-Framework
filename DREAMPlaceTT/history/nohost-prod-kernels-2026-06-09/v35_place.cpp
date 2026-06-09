// SPDX-License-Identifier: Apache-2.0
//
// V35 on-chip grouping — PASS 2 (place), LOCAL-COUNTING-SORT + BULK writes. The
// naive per-cell scatter (one 128 B NoC write/cell) was ~70% of the backward (the
// V32 lesson). Instead each core streams its stash slice in chunks, locally
// counting-sorts each chunk by TILE in L1 (cheap, no NoC), then emits ONE
// contiguous bulk write per (chunk,tile) into the grouped buffer at the running
// global offset wr[t] = tile_base[t] + srcprefix[t] + (cells of tile t this core
// already wrote). 128 B records kept verbatim (64-B aligned). tile_base[t] +
// srcprefix[t] from the host plan (PASS-1 counts) → disjoint per-core sub-ranges.
// Record (32 u32): [0]gidx [1]bxl_g [2]byl [3]kc [4]hc [5]ratio [6..13]px [14..21]py.

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

    constexpr auto CB_IN=tt::CBIndex::c_0, CB_SORT=tt::CBIndex::c_1, CB_TB=tt::CBIndex::c_2,
                   CB_SP=tt::CBIndex::c_3, CB_SCR=tt::CBIndex::c_4;
    const InterleavedAddrGen<true> ing  = {.bank_base_address=stash_base, .page_size=stash_pg};
    const InterleavedAddrGen<true> gg   = {.bank_base_address=grouped_base, .page_size=grouped_pg};
    const InterleavedAddrGen<true> tbg  = {.bank_base_address=tilebase_base, .page_size=ntiles*4u};
    const InterleavedAddrGen<true> spg  = {.bank_base_address=srcpref_base, .page_size=plan_pg};

    cb_reserve_back(CB_TB,1); uint32_t* tb=(uint32_t*)get_write_ptr(CB_TB);
    noc_async_read(tbg.get_noc_addr(0), (uint32_t)tb, ntiles*4u); noc_async_read_barrier();
    cb_reserve_back(CB_SP,1); uint32_t* sp=(uint32_t*)get_write_ptr(CB_SP);
    noc_async_read(spg.get_noc_addr(my_core), (uint32_t)sp, ntiles*4u); noc_async_read_barrier();
    cb_reserve_back(CB_SCR,1); uint32_t* scr=(uint32_t*)get_write_ptr(CB_SCR);
    uint32_t* wr=scr; uint32_t* hist=scr+ntiles; uint32_t* off=scr+2u*ntiles;   // running, per-chunk hist, per-chunk cursor
    for (uint32_t t=0;t<ntiles;++t) wr[t]=tb[t]+sp[t];

    const uint32_t in_l1   = get_write_ptr(CB_IN);
    const uint32_t sort_l1 = get_write_ptr(CB_SORT);

    { DeviceZoneScopedN("V35-PLACE");
      uint32_t done=0;
      while (done < my_n) {
          uint32_t this_n = (my_n-done < chunk) ? (my_n-done) : chunk;
          noc_async_read(ing.get_noc_addr(0) + (uint64_t)(my_first+done)*128u, in_l1, this_n*128u);
          noc_async_read_barrier();
          const uint32_t* rec = reinterpret_cast<const uint32_t*>(in_l1);
          // local counting sort of this chunk by tile
          for (uint32_t t=0;t<ntiles;++t) hist[t]=0;
          for (uint32_t i=0;i<this_n;++i){ uint32_t t=rec[i*32u+1u]/W; if(t>=ntiles)t=ntiles-1u; hist[t]++; }
          uint32_t acc=0; for (uint32_t t=0;t<ntiles;++t){ off[t]=acc; acc+=hist[t]; }   // off[t]=run start (records)
          uint32_t* sorted = reinterpret_cast<uint32_t*>(sort_l1);
          // place each record into its tile's run (off[] used as advancing cursor)
          // keep a second cursor so off[] stays = run start for the bulk write
          uint32_t* cur = scr + 3u*ntiles; for (uint32_t t=0;t<ntiles;++t) cur[t]=off[t];
          for (uint32_t i=0;i<this_n;++i){ uint32_t t=rec[i*32u+1u]/W; if(t>=ntiles)t=ntiles-1u;
              uint32_t d=cur[t]*32u; const uint32_t* s=rec+i*32u; for(uint32_t w=0;w<32u;++w) sorted[d+w]=s[w]; cur[t]++; }
          // bulk write each tile's contiguous run
          for (uint32_t t=0;t<ntiles;++t){ if(hist[t]){
              noc_async_write(sort_l1 + (uint64_t)off[t]*128u, gg.get_noc_addr(0) + (uint64_t)wr[t]*128u, hist[t]*128u);
              wr[t]+=hist[t]; } }
          noc_async_write_barrier();
          done += this_n;
      }
    }
}
