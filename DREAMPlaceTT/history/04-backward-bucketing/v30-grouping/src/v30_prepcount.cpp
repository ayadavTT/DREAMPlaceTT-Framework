// SPDX-License-Identifier: Apache-2.0
//
// V30 backward P1 — on-chip PREP + COUNT + zero-grad (BRISC). Each core processes
// a contiguous slice of cells; for each it computes the V21 prep (bin_xl/yl, k/h
// span, px/py overlaps, matching v29_prepbucket so accuracy = CPU electric_force),
// emits 16B records {cell, w_fixed=round(ratio·px·py·WS), bin_global, pad} to its
// own SCRATCH page in CELL ORDER, and counts cnt[my_core][slab] (slab=bin/slab_bins).
// Host prefix-sums cnt → contiguous offsets; P2 places scratch→route_flat. Also
// zeros this core's grad slab (CB c_28, first CB → same L1 addr across P1..P4).
//
// in record (64B,16f): [0]x [1]y [2]ox [3]oy [4]nsx [5]nsy [6]ratio [7..]pad

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"
constexpr uint32_t MAXKH = 8;

void kernel_main() {
    const uint32_t my_core  = get_arg_val<uint32_t>(0);
    const uint32_t my_first = get_arg_val<uint32_t>(1);
    const uint32_t my_n     = get_arg_val<uint32_t>(2);
    const uint32_t in_base  = get_arg_val<uint32_t>(3);
    const uint32_t in_pg    = get_arg_val<uint32_t>(4);
    const uint32_t scr_base = get_arg_val<uint32_t>(5);
    const uint32_t scr_pg   = get_arg_val<uint32_t>(6);
    const uint32_t cnt_base = get_arg_val<uint32_t>(7);
    const uint32_t cnt_pg   = get_arg_val<uint32_t>(8);
    const uint32_t slab_bins= get_arg_val<uint32_t>(9);
    const uint32_t nc       = get_arg_val<uint32_t>(10);
    const uint32_t NBX      = get_arg_val<uint32_t>(11);
    const uint32_t NBY      = get_arg_val<uint32_t>(12);
    union{uint32_t u;float f;}cv;
    cv.u=get_arg_val<uint32_t>(13); const float xl=cv.f;
    cv.u=get_arg_val<uint32_t>(14); const float yl=cv.f;
    cv.u=get_arg_val<uint32_t>(15); const float bsx=cv.f;
    cv.u=get_arg_val<uint32_t>(16); const float bsy=cv.f;
    cv.u=get_arg_val<uint32_t>(17); const float inv_bsx=cv.f;
    cv.u=get_arg_val<uint32_t>(18); const float inv_bsy=cv.f;
    cv.u=get_arg_val<uint32_t>(19); const float WS=cv.f;
    const uint32_t grad_ints= get_arg_val<uint32_t>(20);
    const uint32_t chunk    = get_arg_val<uint32_t>(21);

    constexpr auto CB_G=tt::CBIndex::c_28, CB_IN=tt::CBIndex::c_0, CB_CNT=tt::CBIndex::c_1, CB_OUT=tt::CBIndex::c_2;
    // zero grad slab
    cb_reserve_back(CB_G,1); { uint32_t* g=(uint32_t*)get_write_ptr(CB_G); for(uint32_t i=0;i<grad_ints;++i)g[i]=0u; }
    cb_reserve_back(CB_CNT,1); uint32_t* cnt=(uint32_t*)get_write_ptr(CB_CNT); for(uint32_t b=0;b<nc;++b)cnt[b]=0u;
    const uint32_t in_l1=get_write_ptr(CB_IN);
    const uint32_t out_l1=get_write_ptr(CB_OUT);      // record staging (chunk*span*16)
    const InterleavedAddrGen<true> ing={.bank_base_address=in_base,.page_size=in_pg};
    const InterleavedAddrGen<true> scrg={.bank_base_address=scr_base,.page_size=scr_pg};
    const InterleavedAddrGen<true> cntg={.bank_base_address=cnt_base,.page_size=cnt_pg};
    const uint64_t my_scr=scrg.get_noc_addr(my_core);

    { DeviceZoneScopedN("V30-PREPCOUNT");
      uint32_t done=0, wrote=0;          // wrote = records emitted (scratch offset)
      while (done<my_n){
          uint32_t tn=(my_n-done<chunk)?(my_n-done):chunk;
          noc_async_read(ing.get_noc_addr(0)+(uint64_t)(my_first+done)*64u, in_l1, (tn*64u+63u)&~63u);
          noc_async_read_barrier();
          const float* rec=(const float*)in_l1;
          uint32_t* o=(uint32_t*)out_l1; float* of=(float*)out_l1; uint32_t oc=0;  // records this chunk
          for(uint32_t i=0;i<tn;++i){ const float* r=rec+i*16u;
              const float px0=r[0]+r[2], py0=r[1]+r[3], nsx=r[4], nsy=r[5], ratio=r[6];
              int bxl=(int)((px0-xl)*inv_bsx), bxh=(int)((px0+nsx-xl)*inv_bsx)+1;
              int byl=(int)((py0-yl)*inv_bsy), byh=(int)((py0+nsy-yl)*inv_bsy)+1;
              if(bxl<0)bxl=0; if(bxh>(int)NBX)bxh=NBX; if(byl<0)byl=0; if(byh>(int)NBY)byh=NBY;
              uint32_t kc=(bxh>bxl)?(uint32_t)(bxh-bxl):0u; if(kc>MAXKH)kc=MAXKH;
              uint32_t hc=(byh>byl)?(uint32_t)(byh-byl):0u; if(hc>MAXKH)hc=MAXKH;
              const float xt=px0+nsx, yt=py0+nsy; const uint32_t cell=my_first+done+i;
              float bxlo=xl+(float)(uint32_t)bxl*bsx;
              for(uint32_t k=0;k<kc;++k){ const float bhi=bxlo+bsx, pxk=((xt<bhi)?xt:bhi)-((px0>bxlo)?px0:bxlo); bxlo+=bsx;
                  float bylo=yl+(float)(uint32_t)byl*bsy;
                  for(uint32_t h=0;h<hc;++h){ const float bhy=bylo+bsy, pyh=((yt<bhy)?yt:bhy)-((py0>bylo)?py0:bylo); bylo+=bsy;
                      float w=ratio*pxk*pyh; int32_t wf=(int32_t)(w*WS + (w>=0.f?0.5f:-0.5f));
                      uint32_t bg=(uint32_t)(bxl+(int)k)*NBY+(uint32_t)(byl+(int)h);
                      uint32_t slab=bg/slab_bins; if(slab>=nc)slab=nc-1u; cnt[slab]++;
                      o[oc*4+0]=cell; ((int32_t*)o)[oc*4+1]=wf; o[oc*4+2]=bg; o[oc*4+3]=0u; ++oc; } }
          }
          if(oc){ noc_async_write(out_l1, my_scr+(uint64_t)wrote*16u, oc*16u); noc_async_write_barrier(); wrote+=oc; }
          done+=tn;
      }
    }
    noc_async_write((uint32_t)cnt, cntg.get_noc_addr(my_core), nc*4u);
    noc_async_write_barrier();
}
