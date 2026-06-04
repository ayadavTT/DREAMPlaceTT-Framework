// SPDX-License-Identifier: Apache-2.0
//
// V29 EF backward stage 1 — fused PREP + BUCKET (BRISC, on chip, no host).
// Each core processes a contiguous slice of cells (cell order). For each cell it
// computes the V21 prep (bin_xl/yl, k/h_count, px[k], py[h], ratio) from pos +
// per-cell constants, then BUCKETS the resulting record into its OWN page
// route[my_core][band] (band = bin_xl/cpc_x, local slot — no cross-core atomic).
// The V28 gather later reads route[*][my_band]. Prep math is bit-identical to
// v21_ef_ncrisc_v5.cpp:160-213 so accuracy matches CPU electric_force.
//
// Input record  (64 B,16 f32): [0]x [1]y [2]ox [3]oy [4]nsx [5]nsy [6]ratio [7..]pad
// Output record (128 B,32 u32): [0]orig_idx [1]bin_xl_local [2]bin_yl [3]kc [4]hc
//   [5]ratio(f32) [6..13]px[8] [14..21]py[8] [22..31]pad. 128 B → any record
//   offset is 64-aligned, so the gather can read route sub-ranges aligned.

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t MAX_KH = 8;

void kernel_main() {
    const uint32_t my_core   = get_arg_val<uint32_t>(0);
    const uint32_t my_first  = get_arg_val<uint32_t>(1);
    const uint32_t my_n      = get_arg_val<uint32_t>(2);
    const uint32_t in_base   = get_arg_val<uint32_t>(3);
    const uint32_t in_pg     = get_arg_val<uint32_t>(4);   // whole input page bytes
    const uint32_t route_base= get_arg_val<uint32_t>(5);
    const uint32_t cap       = get_arg_val<uint32_t>(6);
    const uint32_t nc        = get_arg_val<uint32_t>(7);
    const uint32_t cpc_x     = get_arg_val<uint32_t>(8);   // x-bins per core (band width, no halo)
    const uint32_t counts_base = get_arg_val<uint32_t>(9);
    const uint32_t chunk     = get_arg_val<uint32_t>(10);
    const uint32_t num_bins_x= get_arg_val<uint32_t>(11);
    const uint32_t num_bins_y= get_arg_val<uint32_t>(12);
    const uint32_t cnt_pg    = get_arg_val<uint32_t>(19);  // 64-aligned counts page bytes
    union { uint32_t u; float f; } cvt;
    cvt.u=get_arg_val<uint32_t>(13); const float xl=cvt.f;
    cvt.u=get_arg_val<uint32_t>(14); const float yl=cvt.f;
    cvt.u=get_arg_val<uint32_t>(15); const float bsx=cvt.f;
    cvt.u=get_arg_val<uint32_t>(16); const float bsy=cvt.f;
    cvt.u=get_arg_val<uint32_t>(17); const float inv_bsx=cvt.f;
    cvt.u=get_arg_val<uint32_t>(18); const float inv_bsy=cvt.f;

    constexpr auto CB_IN=tt::CBIndex::c_0, CB_CNT=tt::CBIndex::c_1, CB_STAGE=tt::CBIndex::c_2;
    const uint32_t route_pg = cap*128u;
    const InterleavedAddrGen<true> ing = {.bank_base_address=in_base, .page_size=in_pg};
    const InterleavedAddrGen<true> rg  = {.bank_base_address=route_base, .page_size=route_pg};
    const InterleavedAddrGen<true> cntg= {.bank_base_address=counts_base, .page_size=cnt_pg};

    cb_reserve_back(CB_CNT,1);
    uint32_t* cnt = reinterpret_cast<uint32_t*>(get_write_ptr(CB_CNT));
    for (uint32_t b=0;b<nc;++b) cnt[b]=0;
    const uint32_t in_l1 = get_write_ptr(CB_IN);
    uint32_t* outrec = reinterpret_cast<uint32_t*>(get_write_ptr(CB_STAGE));  // 24-u32 staging
    float*    outf   = reinterpret_cast<float*>(outrec);

    { DeviceZoneScopedN("V29-PREPBUCKET");
      uint32_t done=0;
      while (done < my_n) {
          uint32_t this_n = (my_n-done < chunk) ? (my_n-done) : chunk;
          const uint32_t rd = (this_n*64u + 63u) & ~63u;     // 64-aligned read
          noc_async_read(ing.get_noc_addr(0) + (uint64_t)(my_first+done)*64u, in_l1, rd);
          noc_async_read_barrier();
          const float* rec = reinterpret_cast<const float*>(in_l1);
          for (uint32_t i=0;i<this_n;++i) {
              const float* r = rec + i*16u;
              const float pos_x=r[0], pos_y=r[1], ox=r[2], oy=r[3], nsx=r[4], nsy=r[5], ratio=r[6];
              const float node_x=pos_x+ox, node_y=pos_y+oy;
              int bxl=(int)((node_x      -xl)*inv_bsx);
              int bxh=(int)((node_x+nsx  -xl)*inv_bsx)+1;
              int byl=(int)((node_y      -yl)*inv_bsy);
              int byh=(int)((node_y+nsy  -yl)*inv_bsy)+1;
              if (bxl<0) bxl=0;
              if (bxh>(int)num_bins_x) bxh=(int)num_bins_x;
              if (byl<0) byl=0;
              if (byh>(int)num_bins_y) byh=(int)num_bins_y;
              uint32_t kc=(bxh>bxl)?(uint32_t)(bxh-bxl):0u; if (kc>MAX_KH) kc=MAX_KH;
              uint32_t hc=(byh>byl)?(uint32_t)(byh-byl):0u; if (hc>MAX_KH) hc=MAX_KH;
              const float node_x_top=node_x+nsx, node_y_top=node_y+nsy;
              // zero all MAX_KH slots, then compute overlap ONLY for the real span
              // (kc/hc) — most cells span ≤2 bins, so this is ~3× less float work
              // than looping the full 8 (the rest stay 0 → nullified downstream).
              for (uint32_t k=0;k<MAX_KH;++k) outf[6+k]=0.f;
              for (uint32_t h=0;h<MAX_KH;++h) outf[14+h]=0.f;
              float bxlo = xl + (float)(uint32_t)bxl*bsx;
              for (uint32_t k=0;k<kc;++k){ const float bhi=bxlo+bsx; const float top=(node_x_top<bhi)?node_x_top:bhi;
                  const float bot=(node_x>bxlo)?node_x:bxlo; outf[6+k]=top-bot; bxlo+=bsx; }
              float bylo = yl + (float)(uint32_t)byl*bsy;
              for (uint32_t h=0;h<hc;++h){ const float bhi=bylo+bsy; const float top=(node_y_top<bhi)?node_y_top:bhi;
                  const float bot=(node_y>bylo)?node_y:bylo; outf[14+h]=top-bot; bylo+=bsy; }
              uint32_t band=(uint32_t)bxl/cpc_x; if (band>=nc) band=nc-1;
              outrec[0]=my_first+done+i;                 // orig_idx
              outrec[1]=(uint32_t)bxl - band*cpc_x;       // bin_xl_local
              outrec[2]=(uint32_t)byl; outrec[3]=kc; outrec[4]=hc; cvt.f=ratio; outrec[5]=cvt.u;
              uint32_t slot=cnt[band];
              if (slot<cap){ cnt[band]=slot+1;
                  uint32_t page=my_core*nc+band;
                  noc_async_write((uint32_t)outrec, rg.get_noc_addr(page)+(uint64_t)slot*128u, 128u);
              }
          }
          noc_async_write_barrier();
          done += this_n;
      }
    }
    noc_async_write((uint32_t)cnt, cntg.get_noc_addr(my_core), nc*4u);
    noc_async_write_barrier();
}
