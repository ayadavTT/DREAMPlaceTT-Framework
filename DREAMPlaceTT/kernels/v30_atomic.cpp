// SPDX-License-Identifier: Apache-2.0
//
// V30 backward P1: field-stationary gather + ON-CHIP grad sum via fixed-point
// NoC atomic-add (the V19 forward's write pattern, inverted). Each worker reads
// its contiguous field slab range (LOCAL, int32 fixed-point), iterates its route
// records, computes gx=(w*fx)>>15 / gy=(w*fy)>>15, and atomic-adds them into the
// CELL-OWNER's grad slab (CB c_28) at cell_local — across cores, like the forward.
// Cell-owner = cell/cpc_cell, cell_local = cell%cpc_cell. Signed two's-complement
// add accumulates correctly mod 2^32 (host descales). worker_noc_coords[] (each
// core's physical x,y, packed (x<<16|y)) come via common runtime args (no NoC read).
//
// route record (16 B): {cell_id, w_fixed, bin_global, pad}.  c_28 declared FIRST
// (same as P0/P2) so grad slab base matches across programs.

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    const uint32_t my_core   = get_arg_val<uint32_t>(0);
    const uint32_t slab_bins = get_arg_val<uint32_t>(1);
    const uint32_t fx_base   = get_arg_val<uint32_t>(2);
    const uint32_t fy_base   = get_arg_val<uint32_t>(3);
    const uint32_t field_pg  = get_arg_val<uint32_t>(4);
    const uint32_t route_base= get_arg_val<uint32_t>(5);
    const uint32_t route_pg  = get_arg_val<uint32_t>(6);
    const uint32_t lo_slab   = get_arg_val<uint32_t>(7);
    const uint32_t n_slabs   = get_arg_val<uint32_t>(8);
    const uint32_t sub_count = get_arg_val<uint32_t>(9);
    const uint32_t cpc_cell  = get_arg_val<uint32_t>(10);
    const uint32_t nc        = get_arg_val<uint32_t>(11);
    const uint32_t chunk     = get_arg_val<uint32_t>(12);
    const uint32_t base_bin  = lo_slab*slab_bins;

    constexpr auto CB_G=tt::CBIndex::c_28, CB_FX=tt::CBIndex::c_24, CB_FY=tt::CBIndex::c_25,
                   CB_RT=tt::CBIndex::c_26, CB_NC=tt::CBIndex::c_27;
    // grad slab base (same L1 addr on every core → atomic target offset)
    const uint32_t grad_off = get_write_ptr(CB_G);
    // copy worker_noc_coords[] into L1 from common args
    cb_reserve_back(CB_NC,1);
    uint32_t* nocc = reinterpret_cast<uint32_t*>(get_write_ptr(CB_NC));
    for (uint32_t i=0;i<nc;++i) nocc[i]=get_common_arg_val<uint32_t>((int)i);

    const InterleavedAddrGen<true> fxg={.bank_base_address=fx_base,.page_size=field_pg};
    const InterleavedAddrGen<true> fyg={.bank_base_address=fy_base,.page_size=field_pg};
    const InterleavedAddrGen<true> rg ={.bank_base_address=route_base,.page_size=route_pg};

    int32_t *fxs,*fys;
    { DeviceZoneScopedN("V30A-LOADSLAB");
      cb_reserve_back(CB_FX,1); uint32_t lx=get_write_ptr(CB_FX);
      cb_reserve_back(CB_FY,1); uint32_t ly=get_write_ptr(CB_FY);
      for (uint32_t j=0;j<n_slabs;++j){ const uint32_t off=j*slab_bins*4u;
          noc_async_read(fxg.get_noc_addr(lo_slab+j), lx+off, slab_bins*4u);
          noc_async_read(fyg.get_noc_addr(lo_slab+j), ly+off, slab_bins*4u); }
      noc_async_read_barrier(); fxs=(int32_t*)lx; fys=(int32_t*)ly;
    }
    cb_reserve_back(CB_RT,1);
    const uint32_t rt_l1 = get_write_ptr(CB_RT);
    const uint64_t my_rt = rg.get_noc_addr(my_core);

    // route[slab] is cell-sorted (host builds by iterating cells), so a cell's span
    // records are CONSECUTIVE → accumulate gx/gy in-register and emit ONE atomic
    // pair per cell instead of per record (~4× fewer atomics for 2×2 spans).
    uint32_t done=0; uint32_t cur=0xFFFFFFFFu; int32_t agx=0, agy=0;
    auto flush=[&](uint32_t cell){ if(cell==0xFFFFFFFFu)return;
        uint32_t owner=cell/cpc_cell; if(owner>=nc)owner=nc-1u; uint32_t loc=cell-owner*cpc_cell;
        uint32_t xy=nocc[owner]; uint64_t tg=get_noc_addr(xy>>16, xy&0xFFFFu, grad_off+loc*8u);
        noc_semaphore_inc(tg,(uint32_t)agx); noc_semaphore_inc(tg+4u,(uint32_t)agy); };
    while (done < sub_count) {
        uint32_t n=(sub_count-done<chunk)?(sub_count-done):chunk;
        noc_async_read(my_rt+(uint64_t)done*16u, rt_l1, (n*16u+63u)&~63u);
        noc_async_read_barrier();
        { DeviceZoneScopedN("V30A-COMPUTE");
          const int32_t* ri=(const int32_t*)rt_l1;
          for (uint32_t i=0;i<n;++i){ const uint32_t b=i*4u;
              uint32_t cell=(uint32_t)ri[b+0]; int32_t w=ri[b+1]; uint32_t lb=(uint32_t)ri[b+2]-base_bin;
              int32_t gx=(w*fxs[lb])>>15, gy=(w*fys[lb])>>15;
              if (cell!=cur){ flush(cur); cur=cell; agx=gx; agy=gy; }
              else { agx+=gx; agy+=gy; } }
        }
        done+=n;
    }
    flush(cur);
    noc_async_atomic_barrier();
}
