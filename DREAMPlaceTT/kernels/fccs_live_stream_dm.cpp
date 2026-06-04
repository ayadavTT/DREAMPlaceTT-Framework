// SPDX-License-Identifier: Apache-2.0
//
// FCCS live backward — STREAMING geom variant for cell counts whose per-worker slice
// exceeds L1 (adaptec3, 1.39M). Identical to fccs_live_dm.cpp (chip-int field multicast
// producer + int MAC + int64 grad in original order) EXCEPT the worker does NOT load+sort
// its slice in L1. Instead it reads its sort_start[NBX+1] prefix (from the fccs_sort
// pass) and STREAMS the bxl-sorted slice from sorted_geom_dram per tile-window (sequential
// chunks). L1 = field bufs + sort_start + grad + chunk — independent of cell count.
// px/py are already int32 in the stash (V31_GEOM_INT) → no quantize.
//
// Record (32 u32): [0]orig [1]bxl [2]byl [3]kc [4]hc [5]ratio [6..13]px [14..21]py.
//
// Args: 0 my_core 1 nc_all 2 is_producer 3 cpc 4 ncells 5 NBX 6 NBY 7 max_k 8 dbg 9 W
//   10 Tx 11 fx_dram 12 fy_dram 13 sorted_geom_dram 14 field_l1_off 15 sortstart_l1_off
//   16 grad_l1_off 17 buf_stride 18 c0x 19 c0y 20 ready_sid 21 tile_sid 22 done_sid
//   23 WS 24 FS 25 grad_dram 26 sort_start_dram 27 chunk_l1_off 28 n_rects 29.. rects

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t CB_SCRATCH = tt::CBIndex::c_24;
constexpr uint32_t GREC = 32u;
constexpr uint32_t CHUNK = 512u;

void kernel_main() {
    const uint32_t my_core    = get_arg_val<uint32_t>(0);
    const uint32_t nc_all     = get_arg_val<uint32_t>(1);
    const uint32_t is_producer= get_arg_val<uint32_t>(2);
    const uint32_t cpc        = get_arg_val<uint32_t>(3);
    const uint32_t ncells     = get_arg_val<uint32_t>(4);
    const uint32_t NBX        = get_arg_val<uint32_t>(5);
    const uint32_t NBY        = get_arg_val<uint32_t>(6);
    const uint32_t max_k      = get_arg_val<uint32_t>(7);
    const uint32_t dbg        = get_arg_val<uint32_t>(8);
    const uint32_t W          = get_arg_val<uint32_t>(9);
    const uint32_t Tx         = get_arg_val<uint32_t>(10);
    const uint32_t fx_dram    = get_arg_val<uint32_t>(11);
    const uint32_t fy_dram    = get_arg_val<uint32_t>(12);
    const uint32_t sorted_dram= get_arg_val<uint32_t>(13);
    const uint32_t field_l1_off = get_arg_val<uint32_t>(14);
    const uint32_t sortstart_l1_off = get_arg_val<uint32_t>(15);
    const uint32_t grad_l1_off  = get_arg_val<uint32_t>(16);
    const uint32_t buf_stride = get_arg_val<uint32_t>(17);
    const uint32_t core0_noc_x = get_arg_val<uint32_t>(18);
    const uint32_t core0_noc_y = get_arg_val<uint32_t>(19);
    const uint32_t ready_sid  = get_arg_val<uint32_t>(20);
    const uint32_t tile_sid   = get_arg_val<uint32_t>(21);
    const uint32_t done_sid   = get_arg_val<uint32_t>(22);
    const uint32_t WS         = get_arg_val<uint32_t>(23);
    const uint32_t grad_dram  = get_arg_val<uint32_t>(25);
    const uint32_t ss_dram    = get_arg_val<uint32_t>(26);
    const uint32_t chunk_l1_off = get_arg_val<uint32_t>(27);
    const uint32_t n_rects    = get_arg_val<uint32_t>(28);
    const uint32_t CG         = get_arg_val<uint32_t>(29);   // compact sorted record size (u32)
    const uint32_t PY_OFF     = get_arg_val<uint32_t>(30);   // py base in the compact record (=6+max_k)
    const uint32_t nworkers   = nc_all - 1u;

    uint8_t* base = reinterpret_cast<uint8_t*>(get_write_ptr(CB_SCRATCH));
    int32_t* field0 = reinterpret_cast<int32_t*>(base + field_l1_off);
    const uint32_t n_tiles = 2u * Tx;

    volatile tt_l1_ptr uint32_t* tile_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(tile_sid));
    const InterleavedAddrGen<true> fxg = {.bank_base_address=fx_dram, .page_size=(uint32_t)((uint64_t)NBX*NBY*4u)};
    const InterleavedAddrGen<true> fyg = {.bank_base_address=fy_dram, .page_size=(uint32_t)((uint64_t)NBX*NBY*4u)};

    // ───────────────────────── PRODUCER (core 0) — identical to fccs_live_dm ──────────
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
              if (t>=2u) { noc_semaphore_wait_min(done_sem, (t-1u)*nworkers); }
              const uint32_t plane = vcols*NBY;
              uint64_t coff=(uint64_t)col0*NBY*4u;
              noc_async_read((comp==0u?fxg:fyg).get_noc_addr(0)+coff, bufp, plane*4u);
              noc_async_read_barrier();
              // PULL model (see fccs_live_dm.cpp): the band is resident in bufp; do NOT multicast
              // (push dropped the tail on buffer reuse). Just publish tile_sem; workers PULL bufp
              // via reliable noc_async_read. Credit-2 keeps bufp stable until all workers pull.
              noc_semaphore_set(tile_sem, t+1u);
              for (uint32_t rr=0;rr<n_rects;++rr){ const uint32_t b=31u+rr*5u;
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

    // ───────────────────────── WORKER (cores 1..nc-1) — STREAMING ─────────────────────
    const uint32_t worker_idx = my_core - 1u;
    const uint32_t my_first = worker_idx * cpc;
    uint32_t my_n = 0;
    if (my_first < ncells) { my_n = ncells - my_first; if (my_n > cpc) my_n = cpc; }
    int64_t* grad = reinterpret_cast<int64_t*>(base + grad_l1_off);
    for (uint32_t i=0;i<my_n*2u;++i) grad[i]=0;

    uint32_t* sort_start = reinterpret_cast<uint32_t*>(base + sortstart_l1_off);   // [NBX+1] resident
    int32_t*  chunk      = reinterpret_cast<int32_t*>(base + chunk_l1_off);
    const InterleavedAddrGen<true> sg = {.bank_base_address=sorted_dram,
        .page_size=(uint32_t)((uint64_t)ncells*CG*4u)};   // sorted geom is COMPACT (CG u32/record)
    const uint32_t ssrow_bytes = (((NBX+1u)*4u)+63u)&~63u;
    const InterleavedAddrGen<true> ssg = {.bank_base_address=ss_dram, .page_size=ssrow_bytes};

    if (my_n) {   // read my sort_start prefix row (per worker page)
        noc_async_read(ssg.get_noc_addr(my_core), (uint32_t)sort_start, (NBX+1u)*4u);
        noc_async_read_barrier();
    }

    uint64_t ready_addr = get_noc_addr(core0_noc_x, core0_noc_y, (uint32_t)get_semaphore(ready_sid));
    noc_semaphore_inc(ready_addr, 1u);
    uint64_t done_addr = get_noc_addr(core0_noc_x, core0_noc_y, (uint32_t)get_semaphore(done_sid));

    { DeviceZoneScopedN("FCCS-COMPUTE");
      for (uint32_t t=0;t<n_tiles;++t){
          uint32_t comp=t/Tx, wt=t%Tx, col0=wt*W;
          uint32_t vcols=(col0<NBX)?((NBX-col0<W)?(NBX-col0):W):0u;
          noc_semaphore_wait_min(tile_sem, t+1u);
          if (my_n && vcols && !(dbg&2u)){
              uint32_t wa=(col0>=(max_k-1u))?(col0-(max_k-1u)):0u; uint32_t wb=col0+vcols; if(wb>NBX)wb=NBX;
              uint32_t s_lo=sort_start[wa], s_hi=sort_start[wb];
              const uint32_t myb = (uint32_t)field0 + (t&1u)*buf_stride;
              const int32_t* fld = reinterpret_cast<const int32_t*>(myb);
              if (s_lo < s_hi) {   // read this tile's band DIRECTLY from field DRAM (distributed across
                  // banks). NOT an all-to-one pull from core0 — at streaming scale 110 workers pulling
                  // 256 KB from one core's L1 WHILE also streaming geom chunks deadlocks the NoC.
                  uint64_t coff = (uint64_t)col0*NBY*4u;
                  noc_async_read((comp==0u?fxg:fyg).get_noc_addr(0)+coff, myb, vcols*NBY*4u);
                  noc_async_read_barrier();
              }
              for (uint32_t s=s_lo; s<s_hi; ){
                  uint32_t cn = (s_hi-s < CHUNK) ? (s_hi-s) : CHUNK;
                  noc_async_read(sg.get_noc_addr(0) + (uint64_t)(my_first+s)*CG*4u, (uint32_t)chunk, cn*CG*4u);
                  noc_async_read_barrier();
                  for (uint32_t j=0;j<cn;++j){
                      const int32_t* r = chunk + j*CG;
                      uint32_t orig=(uint32_t)r[0]; int32_t bxl=r[1]; int32_t byl=r[2];
                      uint32_t kc=(uint32_t)r[3], hc=(uint32_t)r[4];
                      int64_t acc=0;
                      for (uint32_t k=0;k<kc;++k){
                          int32_t col=bxl+(int32_t)k;
                          if (col<(int32_t)col0) continue;
                          if (col>=(int32_t)(col0+vcols)) continue;
                          int32_t pxk=r[6+k]; const int32_t* fcol=fld+(uint32_t)(col-(int32_t)col0)*NBY;
                          for (uint32_t h=0;h<hc;++h){
                              // FULL int64 product (no >>WS). Force true 64-bit: the chained
                              // (i64)a*(i64)b*(i64)c mis-compiled to 32-bit on RV32 (px·py fits 32b,
                              // ×f ~1e13 overflowed). Widen px·py to int64 before ×f.
                              int64_t pxy = (int64_t)((int64_t)pxk * (int64_t)r[PY_OFF+h]);
                              acc += pxy * (int64_t)fcol[byl+(int32_t)h];
                          }
                      }
                      uint32_t slot = orig - my_first;
                      grad[slot*2u+comp] += acc;
                  }
                  s += cn;
              }
          }
          noc_semaphore_inc(done_addr, 1u);
      }
    }
    if (my_n) {
        const InterleavedAddrGen<true> grg = {.bank_base_address=grad_dram, .page_size=(uint32_t)((uint64_t)ncells*16u)};
        noc_async_write((uint32_t)grad, grg.get_noc_addr(0) + (uint64_t)my_first*16u, my_n*16u);
        noc_async_write_barrier();
    }
}
