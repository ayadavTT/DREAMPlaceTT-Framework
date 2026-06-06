// SPDX-License-Identifier: Apache-2.0
//
// FCCS — Field-Cast, Cell-Stationary density backward (kernel), DOUBLE-BUFFERED.
// Mirrors the CPU's zero-reshuffle structure on TT:
//   * Cells are BALANCED contiguous slices (worker w owns [w*cpc,(w+1)*cpc)), resident
//     in L1 with geometry — no bucketing, no routing, clustering-immune.
//   * The FIELD is broadcast to ALL cores via NoC multicast ("shared-cache fill"),
//     streamed in column-band tiles into a 2-deep ping-pong buffer so the producer
//     streams ~2 tiles ahead of compute (no per-tile global barrier on the crit path).
//   * Core 0 = PURE PRODUCER (no cells). Cores 1..nc-1 = workers: gather their cells
//     from the resident tile (int64 fixed-point Σ px·py·f, no soft-float), grad IN PLACE.
// Multicast on NOC0/RISCV_0. Field DRAM: [fx|fy] column-major. Tile t: comp=t/Tx,
// col0=(t%Tx)*W. A cell column (bxl+k) contributes to whichever tile owns it; the
// persistent grad slab accumulates across tiles (correct without halo).
// Fixed-point: f×2^FS, px/py×2^WS; per term (px·py>>WS)·f in int64; host descales 2^(WS+FS).
//
// Args: 0 my_core 1 nc_all 2 is_producer 3 cpc 4 ncells 5 NBX 6 NBY 7 max_k 8 max_h
//  9 W 10 Tx 11 field_dram 12 geom_dram 13 grad_dram 14 field_l1_off 15 geom_l1_off
//  16 grad_l1_off 17 buf_stride 18 core0_noc_x 19 core0_noc_y 20 ready_sid 21 tile_sid
//  22 done_sid 23 WS 24 FS 25 n_rects 26.. per-rect quint (xs,ys,xe,ye,num_dests)

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t CB_SCRATCH = tt::CBIndex::c_24;
constexpr uint32_t GREC = 32u;   // 128 B/cell → my_first*128 is 64-B aligned (DRAM read align)

void kernel_main() {
    const uint32_t my_core    = get_arg_val<uint32_t>(0);
    const uint32_t nc_all     = get_arg_val<uint32_t>(1);
    const uint32_t is_producer= get_arg_val<uint32_t>(2);
    const uint32_t cpc        = get_arg_val<uint32_t>(3);
    const uint32_t ncells     = get_arg_val<uint32_t>(4);
    const uint32_t NBX        = get_arg_val<uint32_t>(5);
    const uint32_t NBY        = get_arg_val<uint32_t>(6);
    const uint32_t max_k      = get_arg_val<uint32_t>(7);
    const uint32_t W          = get_arg_val<uint32_t>(9);
    const uint32_t Tx         = get_arg_val<uint32_t>(10);
    const uint32_t field_dram = get_arg_val<uint32_t>(11);
    const uint32_t geom_dram  = get_arg_val<uint32_t>(12);
    const uint32_t grad_dram  = get_arg_val<uint32_t>(13);
    const uint32_t field_l1_off = get_arg_val<uint32_t>(14);
    const uint32_t chunk_l1_off  = get_arg_val<uint32_t>(15);   // streamed geom chunk buffer
    const uint32_t grad_l1_off  = get_arg_val<uint32_t>(16);
    const uint32_t buf_stride = get_arg_val<uint32_t>(17);
    const uint32_t core0_noc_x = get_arg_val<uint32_t>(18);
    const uint32_t core0_noc_y = get_arg_val<uint32_t>(19);
    const uint32_t ready_sid  = get_arg_val<uint32_t>(20);
    const uint32_t tile_sid   = get_arg_val<uint32_t>(21);
    const uint32_t done_sid   = get_arg_val<uint32_t>(22);
    const uint32_t WS         = get_arg_val<uint32_t>(23);
    const uint32_t skip_compute = get_arg_val<uint32_t>(24);  // DEBUG: workers wait+ack, no compute
    const uint32_t sorted_geom_dram = get_arg_val<uint32_t>(25);   // on-chip sort writes here
    const uint32_t sort_start_l1_off = get_arg_val<uint32_t>(26);  // L1 offsets computed in-kernel
    const uint32_t n_rects    = get_arg_val<uint32_t>(27);
    const uint32_t nworkers   = nc_all - 1u;
    constexpr uint32_t CHUNK = 512u;   // streamed geom records per read (bounds L1)

    uint8_t* base = reinterpret_cast<uint8_t*>(get_write_ptr(CB_SCRATCH));
    int32_t* field0 = reinterpret_cast<int32_t*>(base + field_l1_off);
    int64_t* grad   = reinterpret_cast<int64_t*>(base + grad_l1_off);
    const uint32_t n_tiles = 2u * Tx;

    volatile tt_l1_ptr uint32_t* tile_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(tile_sid));
    const InterleavedAddrGen<true> fg = {.bank_base_address=field_dram,
        .page_size=(uint32_t)(((uint64_t)2u*NBX*NBY*4u))};

    if (is_producer) {
        // PRODUCER (core 0): wait all workers ready, then double-buffered stream.
        volatile tt_l1_ptr uint32_t* ready_sem =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ready_sid));
        noc_semaphore_wait_min(ready_sem, nworkers);
        volatile tt_l1_ptr uint32_t* done_sem =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(done_sid));
        uint32_t tile_l1_sem=(uint32_t)get_semaphore(tile_sid);
        { DeviceZoneScopedN("FCCS-FIELDCAST");
          for (uint32_t t=0;t<n_tiles;++t){
              uint32_t comp=t/Tx, wt=t%Tx, col0=wt*W; uint32_t vcols=(col0<NBX)?((NBX-col0<W)?(NBX-col0):W):0u;
              uint32_t bufp=(uint32_t)field0 + (t&1u)*buf_stride;
              // credit-2: don't overwrite buf (last used by tile t-2) until workers consumed it
              if (t>=2u) { noc_semaphore_wait_min(done_sem, (t-1u)*nworkers); }
              uint64_t foff=((uint64_t)comp*NBX+col0)*NBY*4u;
              noc_async_read(fg.get_noc_addr(0)+foff, bufp, vcols*NBY*4u);
              noc_async_read_barrier();
              for (uint32_t rr=0;rr<n_rects;++rr){ const uint32_t b=28u+rr*5u;
                  uint32_t xs=get_arg_val<uint32_t>(b),ys=get_arg_val<uint32_t>(b+1),
                           xe=get_arg_val<uint32_t>(b+2),ye=get_arg_val<uint32_t>(b+3),nd=get_arg_val<uint32_t>(b+4);
                  if(!nd){ continue; }
                  uint64_t md=get_noc_multicast_addr(xs,ys,xe,ye,bufp);
                  noc_async_write_multicast(bufp, md, vcols*NBY*4u, nd);
              }
              noc_async_write_barrier();
              // release workers for tile t (ready := t+1, mcast) — fire-and-forget
              // (no atomic_barrier per tile: the sem arrives while we prep the next
              // tile; the data write_barrier above already guarantees data delivery).
              noc_semaphore_set(tile_sem, t+1u);
              for (uint32_t rr=0;rr<n_rects;++rr){ const uint32_t b=28u+rr*5u;
                  uint32_t xs=get_arg_val<uint32_t>(b),ys=get_arg_val<uint32_t>(b+1),
                           xe=get_arg_val<uint32_t>(b+2),ye=get_arg_val<uint32_t>(b+3),nd=get_arg_val<uint32_t>(b+4);
                  if(!nd){ continue; }
                  uint64_t ma=get_noc_multicast_addr(xs,ys,xe,ye,tile_l1_sem);
                  noc_semaphore_set_multicast(tile_l1_sem, ma, nd);
              }
          }
          noc_async_atomic_barrier();
          // stay alive until all workers consumed all tiles
          noc_semaphore_wait_min(done_sem, n_tiles*nworkers);
        }
        return;
    }

    // WORKER (cores 1..nc-1): balanced cell slice [worker_idx*cpc, +cpc).
    // ON-CHIP counting sort (count→prefix→place) reorders the UNSORTED geom (geom_dram)
    // into sorted_geom_dram by bxl, streaming in chunks (geometry NEVER fully resident).
    // grad is resident in ORIGINAL order (sorted record carries orig_idx) → host reads it
    // directly, no permutation. L1 = field bufs + grad + hist/cursor(reuse field pre-stream)
    // + sort_start + chunk — independent of cpc, so it scales to ANY cell count.
    const uint32_t worker_idx = my_core - 1u;
    const uint32_t my_first = worker_idx * cpc;
    uint32_t my_n = 0;
    if (my_first < ncells) { my_n = ncells - my_first; if (my_n > cpc) my_n = cpc; }
    for (uint32_t i=0;i<my_n*2u;++i) grad[i]=0;

    // FORWARD-ORDERED geometry: geom_dram is PRE-SORTED by bxl (per worker) with orig_idx
    // in record[20] — produced by the forward (here: host, representing forward output).
    // The backward does NO sort: just load the sort_start offsets + stream the sorted geom.
    uint32_t* sort_start = reinterpret_cast<uint32_t*>(base + sort_start_l1_off);   // [NBX+1] resident
    int32_t*  chunk      = reinterpret_cast<int32_t*>(base + chunk_l1_off);         // streamed records
    const InterleavedAddrGen<true> sg = {.bank_base_address=geom_dram, .page_size=(uint32_t)(((uint64_t)ncells*GREC*4u))};
    const uint32_t ssrow_bytes = (((NBX+1u)*4u)+63u)&~63u;
    // sort_start buffer is a SINGLE flat page (arg25); read my row by byte offset (not page index)
    const InterleavedAddrGen<true> ssg = {.bank_base_address=sorted_geom_dram, .page_size=(uint32_t)((uint64_t)nworkers*ssrow_bytes)};
    noc_async_read(ssg.get_noc_addr(0) + (uint64_t)worker_idx*ssrow_bytes, (uint32_t)sort_start, ssrow_bytes);
    noc_async_read_barrier();

    uint64_t ready_addr = get_noc_addr(core0_noc_x, core0_noc_y, (uint32_t)get_semaphore(ready_sid));
    noc_semaphore_inc(ready_addr, 1u);
    uint64_t done_addr = get_noc_addr(core0_noc_x, core0_noc_y, (uint32_t)get_semaphore(done_sid));

    { DeviceZoneScopedN("FCCS-COMPUTE");
      for (uint32_t t=0;t<n_tiles;++t){
          uint32_t comp=t/Tx, wt=t%Tx, col0=wt*W; uint32_t vcols=(col0<NBX)?((NBX-col0<W)?(NBX-col0):W):0u;
          noc_semaphore_wait_min(tile_sem, t+1u);
          if (skip_compute) { noc_semaphore_inc(done_addr, 1u); continue; }   // DEBUG: isolate sync
          const int32_t* fld = reinterpret_cast<const int32_t*>((uint32_t)field0 + (t&1u)*buf_stride);
          const uint32_t gcomp=comp;
          uint32_t wa=(col0>=max_k-1u)?(col0-(max_k-1u)):0u; uint32_t wb=col0+vcols; if(wb>NBX)wb=NBX;
          uint32_t s_lo=sort_start[wa], s_hi=sort_start[wb];
          for (uint32_t s=s_lo; s<s_hi; ) {
            uint32_t cn = (s_hi-s < CHUNK) ? (s_hi-s) : CHUNK;
            noc_async_read(sg.get_noc_addr(0) + (uint64_t)(my_first+s)*GREC*4u, (uint32_t)chunk, cn*GREC*4u);
            noc_async_read_barrier();
            for (uint32_t j=0;j<cn;++j){
              const int32_t* r = chunk + j*GREC;
              int32_t bxl=r[0], byl=r[1]; uint32_t kc=(uint32_t)r[2], hc=(uint32_t)r[3]; uint32_t orig=(uint32_t)r[20];
              int64_t acc=0;
              for (uint32_t k=0;k<kc;++k){
                  int32_t col=bxl+(int32_t)k;
                  if (col<(int32_t)col0) { continue; }
                  if (col>=(int32_t)(col0+vcols)) { continue; }
                  int32_t pxk=r[4+k];
                  const int32_t* fcol = fld + (uint32_t)(col-(int32_t)col0)*NBY;
                  for (uint32_t h=0;h<hc;++h){
                      // int32 products (px·py≤2^26, ·f≤2^26 fit int32), int64 accumulate.
                      // px·py>>WS keeps ~2^WS; ·f → 2^(WS+FS). Host descales 2^(WS+FS).
                      int32_t w = (int32_t)(((int32_t)pxk * (int32_t)r[12+h]) >> WS);
                      acc += (int64_t)(w * fcol[byl+(int32_t)h]);
                  }
              }
              grad[orig*2u+gcomp] += acc;   // ORIGINAL order — no host unsort
            }
            s += cn;
          }
          noc_semaphore_inc(done_addr, 1u);
      }
    }

    if (my_n) {
        const InterleavedAddrGen<true> grg = {.bank_base_address=grad_dram, .page_size=(uint32_t)(((uint64_t)ncells*16u))};
        noc_async_write((uint32_t)grad, grg.get_noc_addr(0) + (uint64_t)my_first*16u, my_n*16u);
        noc_async_write_barrier();
    }
}
