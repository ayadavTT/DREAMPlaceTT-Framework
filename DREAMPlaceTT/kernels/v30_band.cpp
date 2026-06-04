// SPDX-License-Identifier: Apache-2.0
//
// V30 LEAN backward (BRISC, one kernel). Relies on V22_RELABEL spatial sort:
// each core owns a balanced contiguous cell slice [first,first+n) whose cells
// (sorted by x) touch a contiguous x-column band [bx_lo, bx_lo+n_bx). The core:
//   (1) reads its int32 field band LOCALLY (n_bx contiguous columns, no NoC gather),
//   (2) for each cell: prep (bins, px/py overlaps), gx=Σ(w*fx)>>15, gy=Σ(w*fy)>>15
//       in integer fixed-point (no soft-float mul on the hot path),
//   (3) writes grad_out[cell] DIRECTLY — the cell is owned by this core, so NO
//       cross-core atomic and NO grad sum is needed.
// No route, no bucketing, no grouping. Field local + grad local.
//
// in record (64B,16f): [0]x [1]y [2]ox [3]oy [4]nsx [5]nsy [6]ratio
// field: int32 per-x-column pages (page = NBY*4); column bx → page bx.
// grad_out: int32 [gx,gy] per cell (page=8B), host descales by wmax*fmax/2^15.

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"
constexpr uint32_t MAXKH = 8;

void kernel_main() {
    const uint32_t my_core = get_arg_val<uint32_t>(0);
    const uint32_t first   = get_arg_val<uint32_t>(1);
    const uint32_t n       = get_arg_val<uint32_t>(2);
    const uint32_t in_base = get_arg_val<uint32_t>(3);
    const uint32_t in_pg   = get_arg_val<uint32_t>(4);
    const uint32_t bx_lo   = get_arg_val<uint32_t>(5);
    const uint32_t n_bx    = get_arg_val<uint32_t>(6);
    const uint32_t NBX     = get_arg_val<uint32_t>(7);
    const uint32_t NBY     = get_arg_val<uint32_t>(8);
    const uint32_t fx_base = get_arg_val<uint32_t>(9);
    const uint32_t fy_base = get_arg_val<uint32_t>(10);
    const uint32_t col_pg  = get_arg_val<uint32_t>(11);   // NBY*4
    const uint32_t grad_base=get_arg_val<uint32_t>(12);
    const uint32_t grad_pg = get_arg_val<uint32_t>(13);   // whole grad buffer (na*8), single page
    union{uint32_t u;float f;}cv;
    cv.u=get_arg_val<uint32_t>(14); const float xl=cv.f;
    cv.u=get_arg_val<uint32_t>(15); const float yl=cv.f;
    cv.u=get_arg_val<uint32_t>(16); const float bsx=cv.f;
    cv.u=get_arg_val<uint32_t>(17); const float bsy=cv.f;
    cv.u=get_arg_val<uint32_t>(18); const float inv_bsx=cv.f;
    cv.u=get_arg_val<uint32_t>(19); const float inv_bsy=cv.f;
    cv.u=get_arg_val<uint32_t>(20); const float WS=cv.f;
    const uint32_t chunk   = get_arg_val<uint32_t>(21);

    constexpr auto CB_FX=tt::CBIndex::c_24, CB_FY=tt::CBIndex::c_25, CB_IN=tt::CBIndex::c_0, CB_O=tt::CBIndex::c_2;
    const InterleavedAddrGen<true> fxg={.bank_base_address=fx_base,.page_size=col_pg};
    const InterleavedAddrGen<true> fyg={.bank_base_address=fy_base,.page_size=col_pg};
    const InterleavedAddrGen<true> ing={.bank_base_address=in_base,.page_size=in_pg};
    const InterleavedAddrGen<true> og ={.bank_base_address=grad_base,.page_size=grad_pg};
    const uint64_t grad0=og.get_noc_addr(0);   // single-page buffer base

    // ── LOCAL field band: n_bx contiguous x-columns (int32) ──
    int32_t *fxs,*fys;
    { DeviceZoneScopedN("V30B-BAND");
      cb_reserve_back(CB_FX,1); uint32_t lx=get_write_ptr(CB_FX);
      cb_reserve_back(CB_FY,1); uint32_t ly=get_write_ptr(CB_FY);
      for(uint32_t c=0;c<n_bx;++c){ uint32_t col=bx_lo+c; if(col>=NBX)col=NBX-1u;
          noc_async_read(fxg.get_noc_addr(col), lx+c*col_pg, col_pg);
          noc_async_read(fyg.get_noc_addr(col), ly+c*col_pg, col_pg); }
      noc_async_read_barrier(); fxs=(int32_t*)lx; fys=(int32_t*)ly;
    }
    const uint32_t in_l1=get_write_ptr(CB_IN), o_l1=get_write_ptr(CB_O);

    uint32_t done=0;
    while(done<n){
        uint32_t tn=(n-done<chunk)?(n-done):chunk;
        noc_async_read(ing.get_noc_addr(0)+(uint64_t)(first+done)*64u, in_l1, (tn*64u+63u)&~63u);
        noc_async_read_barrier();
        { DeviceZoneScopedN("V30B-COMPUTE");
          const float* rec=(const float*)in_l1; int32_t* o=(int32_t*)o_l1;
          for(uint32_t i=0;i<tn;++i){ const float* r=rec+i*16u;
              const float px0=r[0]+r[2], py0=r[1]+r[3], nsx=r[4], nsy=r[5], ratio=r[6];
              int bxl=(int)((px0-xl)*inv_bsx), bxh=(int)((px0+nsx-xl)*inv_bsx)+1;
              int byl=(int)((py0-yl)*inv_bsy), byh=(int)((py0+nsy-yl)*inv_bsy)+1;
              if(bxl<0)bxl=0; if(bxh>(int)NBX)bxh=NBX; if(byl<0)byl=0; if(byh>(int)NBY)byh=NBY;
              uint32_t kc=(bxh>bxl)?(uint32_t)(bxh-bxl):0u; if(kc>MAXKH)kc=MAXKH;
              uint32_t hc=(byh>byl)?(uint32_t)(byh-byl):0u; if(hc>MAXKH)hc=MAXKH;
              const float xt=px0+nsx, yt=py0+nsy;
              int32_t gx=0, gy=0; float bxlo=xl+(float)(uint32_t)bxl*bsx;
              for(uint32_t k=0;k<kc;++k){ const float bhi=bxlo+bsx, pxk=((xt<bhi)?xt:bhi)-((px0>bxlo)?px0:bxlo); bxlo+=bsx;
                  uint32_t lcol=(uint32_t)(bxl+(int)k)-bx_lo;          // band-local column
                  const int32_t* fxc=fxs+lcol*(col_pg>>2); const int32_t* fyc=fys+lcol*(col_pg>>2);
                  float bylo=yl+(float)(uint32_t)byl*bsy;
                  for(uint32_t h=0;h<hc;++h){ const float bhy=bylo+bsy, pyh=((yt<bhy)?yt:bhy)-((py0>bylo)?py0:bylo); bylo+=bsy;
                      float w=ratio*pxk*pyh; int32_t wf=(int32_t)(w*WS+(w>=0.f?0.5f:-0.5f));
                      uint32_t by=(uint32_t)(byl+(int)h);
                      gx+=(wf*fxc[by])>>15; gy+=(wf*fyc[by])>>15; } }
              o[i*2+0]=gx; o[i*2+1]=gy; }
        }
        noc_async_write(o_l1, grad0+(uint64_t)(first+done)*8u, tn*8u);
        noc_async_write_barrier();
        done+=tn;
    }
}
