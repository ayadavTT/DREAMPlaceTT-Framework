// SPDX-License-Identifier: Apache-2.0
// Bin-owner regroup — pass 2 (place), OPTIMIZED format + block writes. Input record
// = {cell, owner, w_fixed, bin} (owner + fixed-point weight pre-computed by the
// forward). Per chunk: local counting-sort by owner → contiguous block writes to
// route[owner] page at the running global offset cur[o] (init = base[c][o]). Output
// record = {cell, w_fixed, bin, pad} (V30 backward format). PURE data movement — no
// division, no soft-float; the place loop is just int copies + block NoC writes.
#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    const uint32_t my_core   = get_arg_val<uint32_t>(0);
    const uint32_t recs_base = get_arg_val<uint32_t>(1);
    const uint32_t recs_pg   = get_arg_val<uint32_t>(2);
    const uint32_t M         = get_arg_val<uint32_t>(3);
    const uint32_t NOWN      = get_arg_val<uint32_t>(4);
    const uint32_t base_base = get_arg_val<uint32_t>(5);
    const uint32_t base_pg   = get_arg_val<uint32_t>(6);
    const uint32_t route_base= get_arg_val<uint32_t>(7);
    const uint32_t route_pg  = get_arg_val<uint32_t>(8);
    const uint32_t chunk     = get_arg_val<uint32_t>(9);

    constexpr auto CB_REC=tt::CBIndex::c_0, CB_CUR=tt::CBIndex::c_1, CB_STAGE=tt::CBIndex::c_2,
                   CB_HCNT=tt::CBIndex::c_3, CB_HOFF=tt::CBIndex::c_4;
    const InterleavedAddrGen<true> rg ={.bank_base_address=recs_base,.page_size=recs_pg};
    const InterleavedAddrGen<true> bg ={.bank_base_address=base_base,.page_size=base_pg};
    const InterleavedAddrGen<true> og ={.bank_base_address=route_base,.page_size=route_pg};
    const uint32_t rl   = get_write_ptr(CB_REC);
    uint32_t* cur  =(uint32_t*)get_write_ptr(CB_CUR);
    uint32_t* stage=(uint32_t*)get_write_ptr(CB_STAGE);
    uint32_t* hcnt =(uint32_t*)get_write_ptr(CB_HCNT);
    uint32_t* hoff =(uint32_t*)get_write_ptr(CB_HOFF);
    noc_async_read(bg.get_noc_addr(my_core),(uint32_t)cur,base_pg); noc_async_read_barrier();

    { DeviceZoneScopedN("V32C-PLACE-SORTED");
      const uint64_t base=rg.get_noc_addr(my_core);
      uint32_t done=0;
      while(done<M){ uint32_t n=(M-done<chunk)?(M-done):chunk;
          noc_async_read(base+(uint64_t)done*16u, rl, (n*16u+63u)&~63u); noc_async_read_barrier();
          const uint32_t* ri=(const uint32_t*)rl;
          for(uint32_t o=0;o<NOWN;++o) hcnt[o]=0u;
          for(uint32_t i=0;i<n;++i){ uint32_t o=ri[i*4u+1]; if(o<NOWN) hcnt[o]++; }
          uint32_t acc=0; for(uint32_t o=0;o<NOWN;++o){ hoff[o]=acc; acc+=hcnt[o]; }
          for(uint32_t i=0;i<n;++i){ uint32_t o=ri[i*4u+1]; if(o>=NOWN)continue;
              uint32_t* w=stage+(hoff[o]++)*4u;
              w[0]=ri[i*4u]; w[1]=ri[i*4u+2]; w[2]=ri[i*4u+3]; w[3]=0u; }  // {cell,w_fixed,bin,pad}
          uint32_t start=0;
          for(uint32_t o=0;o<NOWN;++o){ uint32_t cnt=hcnt[o];
              if(cnt){ noc_async_write((uint32_t)(stage+start*4u), og.get_noc_addr(o)+(uint64_t)cur[o]*16u, cnt*16u);
                       cur[o]+=cnt; }
              start+=cnt; }
          noc_async_write_barrier();
          done+=n; }
    }
}
