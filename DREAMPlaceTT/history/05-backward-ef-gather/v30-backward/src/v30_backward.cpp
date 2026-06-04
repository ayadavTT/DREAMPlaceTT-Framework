// SPDX-License-Identifier: Apache-2.0
//
// V30 field-stationary EF backward (validation variant). Each core OWNS a
// contiguous bin block (block partition, same as the V19 forward). It reads its
// field slab from DRAM with ONE LOCAL contiguous read (NO NoC per-bin gather —
// this is the whole point: it inverts V21's latency-bound scattered reads), then
// iterates the route records (cells touching its bins) and emits each cell's
// (gx,gy) contribution. The host sums contributions per cell (the on-chip
// atomic-add sum is the next step). Proves: local field reads + correct transpose.
//
// INTEGER FIXED-POINT compute (V25-v2 lever): BRISC has no FPU, so float mul is
// ~90-cyc soft-float. weight & field are pre-scaled to int (w·2^14, f·2^14) on the
// host; the per-record contribution is a single int32 mul + arithmetic shift
// (prod=(w*f)>>15). Host descales by 1/2^14. Kills the soft-float bottleneck.
//
// route record (16 B): {cell_id, w_fixed (int, =round(px·py·2^14)), local_bin, pad}
// field slab: int32 (=round(f·2^14))
// out record   (16 B): {cell_id, gx_fixed (int), gy_fixed (int), pad}  // host /2^14

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    const uint32_t my_core    = get_arg_val<uint32_t>(0);   // worker id → out page
    const uint32_t slab_bins  = get_arg_val<uint32_t>(1);   // bins per slab (padded mult of 16)
    const uint32_t fx_base    = get_arg_val<uint32_t>(2);
    const uint32_t fy_base    = get_arg_val<uint32_t>(3);
    const uint32_t field_pg   = get_arg_val<uint32_t>(4);   // per-slab field page bytes (slab_bins*4)
    const uint32_t route_base = get_arg_val<uint32_t>(5);
    const uint32_t route_pg   = get_arg_val<uint32_t>(6);   // per-WORKER route page bytes (balanced)
    const uint32_t lo_slab    = get_arg_val<uint32_t>(7);   // first field slab this worker covers
    const uint32_t n_slabs    = get_arg_val<uint32_t>(8);   // contiguous slabs this worker covers
    const uint32_t sub_count  = get_arg_val<uint32_t>(9);   // records this worker processes (its page, from 0)
    const uint32_t out_base   = get_arg_val<uint32_t>(10);
    const uint32_t out_pg     = get_arg_val<uint32_t>(11);
    const uint32_t chunk      = get_arg_val<uint32_t>(12);  // route records per L1 chunk
    const uint32_t base_bin   = lo_slab*slab_bins;          // global bin of L1 field[0]

    constexpr auto CB_FX=tt::CBIndex::c_24, CB_FY=tt::CBIndex::c_25, CB_RT=tt::CBIndex::c_26, CB_OUT=tt::CBIndex::c_27;
    const InterleavedAddrGen<true> fxg={.bank_base_address=fx_base, .page_size=field_pg};
    const InterleavedAddrGen<true> fyg={.bank_base_address=fy_base, .page_size=field_pg};
    const InterleavedAddrGen<true> rg ={.bank_base_address=route_base, .page_size=route_pg};
    const InterleavedAddrGen<true> og ={.bank_base_address=out_base, .page_size=out_pg};

    // ── LOCAL field: contiguous slab RANGE [lo_slab, lo_slab+n_slabs), assembled
    //    into L1 by per-slab interleaved reads (banks spread → no contention) ──
    int32_t *fxs, *fys;
    { DeviceZoneScopedN("V30-LOADSLAB");
      cb_reserve_back(CB_FX,1); uint32_t lx=get_write_ptr(CB_FX);
      cb_reserve_back(CB_FY,1); uint32_t ly=get_write_ptr(CB_FY);
      for (uint32_t j=0;j<n_slabs;++j){ const uint32_t off=j*slab_bins*4u;
          noc_async_read(fxg.get_noc_addr(lo_slab+j), lx+off, slab_bins*4u);
          noc_async_read(fyg.get_noc_addr(lo_slab+j), ly+off, slab_bins*4u); }
      noc_async_read_barrier(); fxs=(int32_t*)lx; fys=(int32_t*)ly;
    }
    const uint32_t rt_l1 = get_write_ptr(CB_RT);
    const uint32_t out_l1 = get_write_ptr(CB_OUT);
    const uint64_t my_out = og.get_noc_addr(my_core);
    const uint64_t my_rt  = rg.get_noc_addr(my_core);   // worker's own balanced route page

    uint32_t done=0;
    while (done < sub_count) {
        uint32_t n = (sub_count-done < chunk) ? (sub_count-done) : chunk;
        const uint32_t rd=(n*16u+63u)&~63u;
        noc_async_read(my_rt + (uint64_t)done*16u, rt_l1, rd);
        noc_async_read_barrier();
        { DeviceZoneScopedN("V30-COMPUTE");
          const int32_t* ri=(const int32_t*)rt_l1;
          int32_t* oi=(int32_t*)out_l1;
          for (uint32_t i=0;i<n;++i){ const uint32_t b=i*4u;
              int32_t cell=ri[b+0]; int32_t w=ri[b+1]; uint32_t lb=(uint32_t)ri[b+2]-base_bin;
              oi[b+0]=cell; oi[b+1]=(w*fxs[lb])>>15; oi[b+2]=(w*fys[lb])>>15; oi[b+3]=0; }
        }
        noc_async_write(out_l1, my_out + (uint64_t)done*16u, ((n*16u+15u)&~15u));
        noc_async_write_barrier();
        done += n;
    }
}
