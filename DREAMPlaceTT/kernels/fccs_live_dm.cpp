// SPDX-License-Identifier: Apache-2.0
//
// FCCS live density-backward (int32-streaming) — production graft of the validated
// microbench kernels/fccs_dm.cpp into the live DREAMPlace loop. Same field-cast /
// cell-stationary structure and the same proven single-plane streaming (comp=t/Tx:
// fx plane → gx, fy plane → gy); only the I/O is adapted to the LIVE on-chip data:
//
//   * Field from chip-DCT: two FLOAT column-major planes fx[NBX*NBY], fy[NBX*NBY]
//     (col=bxl, row=byl). The PRODUCER (core 0) reads a column band of the current
//     plane, QUANTIZES it ×2^FS → int32 IN PLACE, and multicasts (NOC0 / receivers-
//     ready handshake). n_tiles = 2*Tx (Tx fx tiles then Tx fy tiles).
//   * Geometry from the FORWARD's V31_GEOM stash (128 B float record, indexed by orig):
//     g[0]=orig g[1]=bxl_g g[2]=byl g[3]=kc g[4]=hc g[5]=ratio g[6..13]=px g[14..21]=py.
//     The WORKER reads its BALANCED CONTIGUOUS slice, QUANTIZES px/py ×2^WS → int in
//     place, in-L1 counting-sorts by bxl, runs the int MAC, and writes FLOAT grad =
//     ratio·acc·2^-(WS+FS) per cell in ORIGINAL order (grad[orig] — no host unsort).
//
// Args: 0 my_core 1 nc_all 2 is_producer 3 cpc 4 ncells 5 NBX 6 NBY 7 max_k 8 max_h
//   9 W 10 Tx 11 fx_dram 12 fy_dram 13 geom_dram 14 field_l1_off 15 sortbuf_l1_off
//   16 grad_l1_off 17 buf_stride 18 core0_noc_x 19 core0_noc_y 20 ready_sid 21 tile_sid
//   22 done_sid 23 WS 24 FS 25 grad_dram 26 n_rects 27.. per-rect quint (xs,ys,xe,ye,nd)

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t CB_SCRATCH = tt::CBIndex::c_24;
constexpr uint32_t GREC = 32u;   // 128 B/cell forward V31_GEOM record

void kernel_main() {
    const uint32_t my_core    = get_arg_val<uint32_t>(0);
    const uint32_t nc_all     = get_arg_val<uint32_t>(1);
    const uint32_t is_producer= get_arg_val<uint32_t>(2);
    const uint32_t cpc        = get_arg_val<uint32_t>(3);
    const uint32_t ncells     = get_arg_val<uint32_t>(4);
    const uint32_t NBX        = get_arg_val<uint32_t>(5);
    const uint32_t NBY        = get_arg_val<uint32_t>(6);
    const uint32_t max_k      = get_arg_val<uint32_t>(7);
    const uint32_t dbg        = get_arg_val<uint32_t>(8);   // DEBUG: bit0=skip px/py quant, bit1=skip compute
    const uint32_t W          = get_arg_val<uint32_t>(9);
    const uint32_t Tx         = get_arg_val<uint32_t>(10);
    const uint32_t fx_dram    = get_arg_val<uint32_t>(11);
    const uint32_t fy_dram    = get_arg_val<uint32_t>(12);
    const uint32_t geom_dram  = get_arg_val<uint32_t>(13);
    const uint32_t field_l1_off = get_arg_val<uint32_t>(14);
    const uint32_t sortbuf_l1_off= get_arg_val<uint32_t>(15);
    const uint32_t grad_l1_off  = get_arg_val<uint32_t>(16);
    const uint32_t buf_stride = get_arg_val<uint32_t>(17);
    const uint32_t core0_noc_x = get_arg_val<uint32_t>(18);
    const uint32_t core0_noc_y = get_arg_val<uint32_t>(19);
    const uint32_t ready_sid  = get_arg_val<uint32_t>(20);
    const uint32_t tile_sid   = get_arg_val<uint32_t>(21);
    const uint32_t done_sid   = get_arg_val<uint32_t>(22);
    const uint32_t WS         = get_arg_val<uint32_t>(23);
    const uint32_t FS         = get_arg_val<uint32_t>(24);
    const uint32_t grad_dram  = get_arg_val<uint32_t>(25);
    const uint32_t n_rects    = get_arg_val<uint32_t>(26);
    const uint32_t nworkers   = nc_all - 1u;

    uint8_t* base = reinterpret_cast<uint8_t*>(get_write_ptr(CB_SCRATCH));
    int32_t* field0 = reinterpret_cast<int32_t*>(base + field_l1_off);
    const uint32_t n_tiles = 2u * Tx;

    volatile tt_l1_ptr uint32_t* tile_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(tile_sid));
    const InterleavedAddrGen<true> fxg = {.bank_base_address=fx_dram, .page_size=(uint32_t)((uint64_t)NBX*NBY*4u)};
    const InterleavedAddrGen<true> fyg = {.bank_base_address=fy_dram, .page_size=(uint32_t)((uint64_t)NBX*NBY*4u)};

    // ───────────────────────── PRODUCER (core 0) ─────────────────────────
    if (is_producer) {
        volatile tt_l1_ptr uint32_t* ready_sem =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ready_sid));
        noc_semaphore_wait_min(ready_sem, nworkers);
        volatile tt_l1_ptr uint32_t* done_sem =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(done_sid));
        uint32_t tile_l1_sem=(uint32_t)get_semaphore(tile_sid);
        { DeviceZoneScopedN("FCCS-FIELDCAST");
          for (uint32_t t=0;t<n_tiles;++t){
              uint32_t comp=t/Tx, wt=t%Tx, col0=wt*W;
              uint32_t vcols=(col0<NBX)?((NBX-col0<W)?(NBX-col0):W):0u;
              uint32_t bufp=(uint32_t)field0 + (t&1u)*buf_stride;
              if (t>=2u) { noc_semaphore_wait_min(done_sem, (t-1u)*nworkers); }   // credit-2
              const uint32_t plane = vcols*NBY;
              uint64_t coff=(uint64_t)col0*NBY*4u;
              // field is ALREADY int32 (quantized in parallel by the fccs_quantize pre-pass)
              // → just read the band and multicast, no per-producer soft-float.
              noc_async_read((comp==0u?fxg:fyg).get_noc_addr(0)+coff, bufp, plane*4u);
              noc_async_read_barrier();
              if ((dbg&8u) && t==2u){   // PROBE: is the producer's OWN bufp fully read? dump offsets 255/2814/60000
                  int64_t* sc = reinterpret_cast<int64_t*>(base + grad_l1_off);
                  sc[0]=(int64_t)((volatile int32_t*)bufp)[255];
                  sc[1]=(int64_t)((volatile int32_t*)bufp)[2814];
                  sc[2]=(int64_t)((volatile int32_t*)bufp)[60000];
                  sc[3]=(int64_t)plane;
                  const InterleavedAddrGen<true> gdb = {.bank_base_address=grad_dram, .page_size=(uint32_t)((uint64_t)ncells*16u)};
                  noc_async_write((uint32_t)sc, gdb.get_noc_addr(0), 32u);
                  noc_async_write_barrier();
              }
              // PULL model (not push): the band is now resident in the producer's bufp. We do NOT
              // multicast the data — the large multicast push only commits its early bursts to the
              // workers' SRAM before they read on buffer-reuse (acks≠commit), corrupting ~½ the cells.
              // Instead just publish tile_sem; each worker PULLS bufp via noc_async_read (acked →
              // full, reliable delivery). Credit-2 (done_sem) keeps bufp stable until all workers pull.
              noc_semaphore_set(tile_sem, t+1u);
              for (uint32_t rr=0;rr<n_rects;++rr){ const uint32_t b=27u+rr*5u;
                  uint32_t xs=get_arg_val<uint32_t>(b),ys=get_arg_val<uint32_t>(b+1),
                           xe=get_arg_val<uint32_t>(b+2),ye=get_arg_val<uint32_t>(b+3),nd=get_arg_val<uint32_t>(b+4);
                  if(!nd){ continue; }
                  uint64_t ma=get_noc_multicast_addr(xs,ys,xe,ye,tile_l1_sem);
                  noc_semaphore_set_multicast(tile_l1_sem, ma, nd);
              }
          }
          noc_async_atomic_barrier();
          noc_semaphore_wait_min(done_sem, n_tiles*nworkers);
        }
        return;
    }

    // ───────────────────────── WORKER (cores 1..nc-1) ─────────────────────────
    const uint32_t worker_idx = my_core - 1u;
    const uint32_t my_first = worker_idx * cpc;
    uint32_t my_n = 0;
    if (my_first < ncells) { my_n = ncells - my_first; if (my_n > cpc) my_n = cpc; }
    // INT64 grad accumulate (like the microbench) — NO per-cell soft-float on BRISC.
    // Host applies DESCALE·ratio during d2h placement (it has an FPU). [gx,gy]/cell, original order.
    int64_t* grad = reinterpret_cast<int64_t*>(base + grad_l1_off);
    for (uint32_t i=0;i<my_n*2u;++i) grad[i]=0;

    // sortbuf: [ irecs (my_n*GREC i32) | order (my_n u32) | sstart (NBX+1 u32) ]
    int32_t* irecs = reinterpret_cast<int32_t*>(base + sortbuf_l1_off);
    uint32_t* order = reinterpret_cast<uint32_t*>(irecs) + (uint32_t)my_n*GREC;
    uint32_t* sstart= order + (uint32_t)my_n;
    const InterleavedAddrGen<true> sg = {.bank_base_address=geom_dram,
        .page_size=(uint32_t)((uint64_t)ncells*GREC*4u)};
    const float WSCALE = (float)(1u << WS);

    if (my_n) {
        // load my slice (raw forward float record), quantize px/py → int in place
        noc_async_read(sg.get_noc_addr(0) + (uint64_t)my_first*GREC*4u, (uint32_t)irecs, my_n*GREC*4u);
        noc_async_read_barrier();
        // quantize ONLY the valid px/py slots (k<kc, h<hc) — the compute never reads
        // beyond kc/hc, so converting all 8+8 was ~2.7× wasted soft-float.
        if (!(dbg&1u)) for (uint32_t i=0;i<my_n;++i){ int32_t* r=irecs+i*GREC;
            uint32_t kc=(uint32_t)r[3], hc=(uint32_t)r[4];
            for (uint32_t k=0;k<kc;++k){ float v=*reinterpret_cast<float*>(&r[6+k]);  r[6+k]=(int32_t)(v*WSCALE); }
            for (uint32_t h=0;h<hc;++h){ float v=*reinterpret_cast<float*>(&r[14+h]); r[14+h]=(int32_t)(v*WSCALE); }
        }
        // counting-sort by bxl (record word [1])
        for (uint32_t b=0;b<=NBX;++b) sstart[b]=0u;
        for (uint32_t i=0;i<my_n;++i){ uint32_t bxl=(uint32_t)irecs[i*GREC+1]; if(bxl>=NBX)bxl=NBX-1u; sstart[bxl+1u]++; }
        for (uint32_t b=0;b<NBX;++b) sstart[b+1u]+=sstart[b];
        for (uint32_t i=0;i<my_n;++i){ uint32_t bxl=(uint32_t)irecs[i*GREC+1]; if(bxl>=NBX)bxl=NBX-1u; order[sstart[bxl]++]=i; }
        for (uint32_t b=NBX;b>0u;--b) sstart[b]=sstart[b-1u];
        sstart[0]=0u;
    }

    uint64_t ready_addr = get_noc_addr(core0_noc_x, core0_noc_y, (uint32_t)get_semaphore(ready_sid));
    noc_semaphore_inc(ready_addr, 1u);
    uint64_t done_addr = get_noc_addr(core0_noc_x, core0_noc_y, (uint32_t)get_semaphore(done_sid));

    (void)FS;   // descale moved to host (int64 grad)
    { DeviceZoneScopedN("FCCS-COMPUTE");
      for (uint32_t t=0;t<n_tiles;++t){
          uint32_t comp=t/Tx, wt=t%Tx, col0=wt*W;
          uint32_t vcols=(col0<NBX)?((NBX-col0<W)?(NBX-col0):W):0u;
          noc_semaphore_wait_min(tile_sem, t+1u);
          if (my_n && vcols && !(dbg&2u)){
              uint32_t wa=(col0>=(max_k-1u))?(col0-(max_k-1u)):0u; uint32_t wb=col0+vcols; if(wb>NBX)wb=NBX;
              uint32_t s_lo=sstart[wa], s_hi=sstart[wb];
              const uint32_t myb = (uint32_t)field0 + (t&1u)*buf_stride;
              const int32_t* fld = reinterpret_cast<const int32_t*>(myb);
              if (s_lo < s_hi) {
                  // read this tile's band DIRECTLY from field DRAM (distributed across banks).
                  // NOT an all-to-one pull from core0: at NBY=1024 the 512KB band × 110 workers
                  // pulling from one core's L1 delivers PARTIAL data once multiple tiles reuse the
                  // double-buffer (correct at the single center tile of iter 1, wrong as cells spread).
                  uint64_t coff = (uint64_t)col0*NBY*4u;
                  noc_async_read((comp==0u?fxg:fyg).get_noc_addr(0)+coff, myb, vcols*NBY*4u);
                  noc_async_read_barrier();
              }
              for (uint32_t s=s_lo; s<s_hi; ++s){
                  const int32_t* r = irecs + order[s]*GREC;
                  uint32_t orig=(uint32_t)r[0]; int32_t bxl=r[1]; int32_t byl=r[2];
                  uint32_t kc=(uint32_t)r[3], hc=(uint32_t)r[4];
                  int64_t acc=0; int64_t dbg_f=0; bool dbg_got=false;
                  for (uint32_t k=0;k<kc;++k){
                      int32_t col=bxl+(int32_t)k;
                      if (col<(int32_t)col0) continue;
                      if (col>=(int32_t)(col0+vcols)) continue;
                      int32_t pxk=r[6+k]; const int32_t* fcol=fld+(uint32_t)(col-(int32_t)col0)*NBY;
                      for (uint32_t h=0;h<hc;++h){
                          int64_t pxy = (int64_t)((int64_t)pxk * (int64_t)r[14+h]);
                          int64_t fv  = (int64_t)fcol[byl+(int32_t)h];
                          if (k==0u && h==0u && comp==0u){ dbg_f=fv; dbg_got=true; }  // capture k0h0 FX field only
                          acc += pxy * fv;
                      }
                  }
                  uint32_t slot = orig - my_first;
                  if (dbg&4u){ if(dbg_got){ grad[slot*2u+0u]=dbg_f; grad[slot*2u+1u]=(int64_t)bxl; } }  // DEBUG: k0h0 field + bxl
                  else        grad[slot*2u+comp] += acc;   // pure int64; host descales ·DESCALE·ratio
              }
          }
          noc_semaphore_inc(done_addr, 1u);
      }
    }
    if (my_n && !(dbg&2u)) {   // dbg&2 (probe mode) skips worker writeout so it can't clobber the producer's probe
        const InterleavedAddrGen<true> grg = {.bank_base_address=grad_dram, .page_size=(uint32_t)((uint64_t)ncells*16u)};
        noc_async_write((uint32_t)grad, grg.get_noc_addr(0) + (uint64_t)my_first*16u, my_n*16u);
        noc_async_write_barrier();
    }
}
