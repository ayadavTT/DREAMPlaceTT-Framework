// SPDX-License-Identifier: Apache-2.0
// Bin-owner regroup — pass 2 (place) + format transform, live-forward variant.
// Reads V31 stash records {cell, bin_global, area_fp32, pad}, owner = bin/slab_bins,
// and writes them grouped by owner into route[base[o]++] in the V30 backward format
// {cell, w_fixed, bin_global, pad} where w_fixed = round(area * WSCALE). Local
// cursor per owner — no atomics. (Float area → fixed-point happens here so the
// validated v30_atomic backward can run unchanged.)
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
    const uint32_t slab_bins = get_arg_val<uint32_t>(10);
    const uint32_t wscale_b  = get_arg_val<uint32_t>(11);  // WSCALE as float bits

    constexpr auto CB_REC=tt::CBIndex::c_0, CB_CUR=tt::CBIndex::c_1, CB_OUT=tt::CBIndex::c_2;
    const InterleavedAddrGen<true> rg ={.bank_base_address=recs_base,.page_size=recs_pg};
    const InterleavedAddrGen<true> bg ={.bank_base_address=base_base,.page_size=base_pg};
    const InterleavedAddrGen<true> og ={.bank_base_address=route_base,.page_size=route_pg};
    const uint32_t rl = get_write_ptr(CB_REC);
    uint32_t* cur=(uint32_t*)get_write_ptr(CB_CUR);
    uint32_t* out=(uint32_t*)get_write_ptr(CB_OUT);          // staged out-records (16B each)
    float WSCALE; { union{float f;uint32_t u;}cv; cv.u=wscale_b; WSCALE=cv.f; }
    noc_async_read(bg.get_noc_addr(my_core),(uint32_t)cur,base_pg); noc_async_read_barrier();

    { DeviceZoneScopedN("V32B-PLACE");
      const uint64_t base=rg.get_noc_addr(my_core);
      uint32_t done=0;
      while(done<M){ uint32_t n=(M-done<chunk)?(M-done):chunk;
          noc_async_read(base+(uint64_t)done*16u, rl, (n*16u+63u)&~63u); noc_async_read_barrier();
          const uint32_t* ri=(const uint32_t*)rl;
          for(uint32_t i=0;i<n;++i){ uint32_t cell=ri[i*4u]; uint32_t bin=ri[i*4u+1];
              float area; { union{float f;uint32_t u;}cv; cv.u=ri[i*4u+2]; area=cv.f; }
              uint32_t o=bin/slab_bins; if(o>=NOWN)continue;
              int32_t wf=(int32_t)(area*WSCALE + (area>=0.f?0.5f:-0.5f));
              uint32_t pos=cur[o]++; uint32_t* w=out+i*4u;                 // distinct slot per record (in-flight safe)
              w[0]=cell; w[1]=(uint32_t)wf; w[2]=bin; w[3]=0u;             // {cell,w_fixed,bin,pad}
              noc_async_write((uint32_t)w, og.get_noc_addr(o)+(uint64_t)pos*16u, 16u); }  // route PAGE o, slot pos
          noc_async_write_barrier();
          done+=n; }
    }
}
