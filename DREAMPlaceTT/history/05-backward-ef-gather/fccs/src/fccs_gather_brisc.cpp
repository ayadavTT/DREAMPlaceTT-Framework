// SPDX-License-Identifier: Apache-2.0
//
// FCCS gather (BRISC) — live-graft variant of kernels/fccs_dm.cpp that FEEDS
// v28_compute (SFPU float reduce) instead of doing its own integer MAC. Grafts
// the field-cast architecture into the V29 P2 harness (v28_compute + v29_writer).
//
//   * Core 0 = PURE PRODUCER: multicasts HALO'd field column-tiles (W + max_k-1
//     columns) to all worker cores' L1 ping-pong buffer (NOC0 / receivers-ready
//     handshake, the validated multicast pattern). Producer's compute/writer run
//     n_batches=0 (idle).
//   * Cores 1..nc-1 = WORKERS: own a BALANCED CONTIGUOUS cell slice of the
//     forward's geometry stash (V31_GEOM 128 B float record: [0]orig [1]bxl_g
//     [2]byl [3]kc [4]hc [5]ratio [6..13]px [14..21]py), in-L1 counting-sort by
//     bxl, then per multicast tile emit px/py/fx/fy/ratio CB batches for the
//     cells OWNED by that tile (bxl in [col0,col0+W) → full window inside the
//     halo'd tile → v28's one-shot reduce is correct, no cross-tile accumulate).
//
// Field is FLOAT (chip-DCT), column-major [comp][NBX][NBY]. fx then fy back-to-back
// in DRAM (two NBX*NBY*4 planes). grad emitted as [gx,gy] interleaved per cell in
// ORIGINAL order via v29_writer (orig carried per batch through oidx scratch).
//
// Args: 0 my_core 1 nc_all 2 is_producer 3 cpc 4 ncells 5 NBX 6 NBY 7 max_k
//   8 max_h 9 W 10 Tx 11 field_dram 12 geom_dram 13 NA(writer owns grad) 14 field_l1_off
//   15 sortbuf_l1_off 16 NA 17 buf_stride 18 core0_noc_x 19 core0_noc_y 20 ready_sid
//   21 tile_sid 22 done_sid 23 oidx_dram 24 my_slice_first 25 my_slice_n
//   26 n_batches 27 n_rects 28.. per-rect quint (xs,ys,xe,ye,num_dests)

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t CB_SCRATCH = tt::CBIndex::c_24;
constexpr uint32_t GREC = 32u;   // 128 B/cell forward record (matches V31_GEOM stash)
constexpr uint32_t B = 1024u;    // v28 batch size

void kernel_main() {
    const uint32_t my_core    = get_arg_val<uint32_t>(0);
    const uint32_t nc_all     = get_arg_val<uint32_t>(1);
    const uint32_t is_producer= get_arg_val<uint32_t>(2);
    const uint32_t ncells     = get_arg_val<uint32_t>(4);
    const uint32_t NBX        = get_arg_val<uint32_t>(5);
    const uint32_t NBY        = get_arg_val<uint32_t>(6);
    const uint32_t max_k      = get_arg_val<uint32_t>(7);
    const uint32_t max_h      = get_arg_val<uint32_t>(8);
    const uint32_t W          = get_arg_val<uint32_t>(9);
    const uint32_t Tx         = get_arg_val<uint32_t>(10);
    const uint32_t field_dram = get_arg_val<uint32_t>(11);
    const uint32_t geom_dram  = get_arg_val<uint32_t>(12);
    const uint32_t field_l1_off = get_arg_val<uint32_t>(14);
    const uint32_t sortbuf_l1_off = get_arg_val<uint32_t>(15);
    const uint32_t buf_stride = get_arg_val<uint32_t>(17);
    const uint32_t core0_noc_x = get_arg_val<uint32_t>(18);
    const uint32_t core0_noc_y = get_arg_val<uint32_t>(19);
    const uint32_t ready_sid  = get_arg_val<uint32_t>(20);
    const uint32_t tile_sid   = get_arg_val<uint32_t>(21);
    const uint32_t done_sid   = get_arg_val<uint32_t>(22);
    const uint32_t oidx_dram  = get_arg_val<uint32_t>(23);
    const uint32_t my_first   = get_arg_val<uint32_t>(24);
    const uint32_t my_n       = get_arg_val<uint32_t>(25);
    const uint32_t n_batches  = get_arg_val<uint32_t>(26);
    const uint32_t n_rects    = get_arg_val<uint32_t>(27);
    const uint32_t nworkers   = nc_all - 1u;
    const uint32_t halo       = (max_k > 0u) ? (max_k - 1u) : 0u;   // extra cols per tile
    const uint32_t WH         = W + halo;                            // halo'd tile width

    uint8_t* base = reinterpret_cast<uint8_t*>(get_write_ptr(CB_SCRATCH));
    float* field0 = reinterpret_cast<float*>(base + field_l1_off);
    const uint32_t n_tiles = Tx;   // one comp plane (fx,fy read as two DRAM planes per tile)

    volatile tt_l1_ptr uint32_t* tile_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(tile_sid));
    // field DRAM holds fx plane then fy plane, each NBX*NBY*4 bytes, column-major.
    const uint64_t fy_plane_off = (uint64_t)NBX * NBY * 4u;
    const InterleavedAddrGen<true> fg = {.bank_base_address=field_dram,
        .page_size=(uint32_t)(2u*fy_plane_off)};

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
              uint32_t col0=t*W;
              uint32_t vcols=(col0<NBX)?((NBX-col0<WH)?(NBX-col0):WH):0u;   // halo'd, clamp to grid
              uint32_t bufp=(uint32_t)field0 + (t&1u)*buf_stride;
              if (t>=2u) { noc_semaphore_wait_min(done_sem, (t-1u)*nworkers); }   // credit-2
              // fx columns then fy columns, packed back-to-back into the L1 buf:
              // [ fx(vcols*NBY) | fy(vcols*NBY) ]
              uint64_t coff=(uint64_t)col0*NBY*4u;
              noc_async_read(fg.get_noc_addr(0)+coff, bufp, vcols*NBY*4u);
              noc_async_read(fg.get_noc_addr(0)+fy_plane_off+coff, bufp+vcols*NBY*4u, vcols*NBY*4u);
              noc_async_read_barrier();
              const uint32_t tile_bytes = 2u*vcols*NBY*4u;
              for (uint32_t rr=0;rr<n_rects;++rr){ const uint32_t b=28u+rr*5u;
                  uint32_t xs=get_arg_val<uint32_t>(b),ys=get_arg_val<uint32_t>(b+1),
                           xe=get_arg_val<uint32_t>(b+2),ye=get_arg_val<uint32_t>(b+3),nd=get_arg_val<uint32_t>(b+4);
                  if(!nd){ continue; }
                  uint64_t md=get_noc_multicast_addr(xs,ys,xe,ye,bufp);
                  noc_async_write_multicast(bufp, md, tile_bytes, nd);
              }
              noc_async_write_barrier();
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
          noc_semaphore_wait_min(done_sem, n_tiles*nworkers);
        }
        return;
    }

    // ───────────────────────── WORKER (cores 1..nc-1) ─────────────────────────
    constexpr auto CB_PX=tt::CBIndex::c_0, CB_PY=tt::CBIndex::c_1, CB_FX=tt::CBIndex::c_2,
                   CB_FY=tt::CBIndex::c_3, CB_RATIO=tt::CBIndex::c_4, CB_OISCRATCH=tt::CBIndex::c_5;

    // Forward geom records for my slice: [my_first, my_first+my_n). Resident in the
    // sort buffer; in-L1 counting-sort by bxl so the per-tile window is contiguous.
    // Layout of sortbuf: [ recs (my_n*GREC u32) | order (my_n u32) | sort_start (NBX+1 u32) ]
    uint32_t* recs   = reinterpret_cast<uint32_t*>(base + sortbuf_l1_off);
    uint32_t* order  = recs + (uint32_t)my_n * GREC;            // sorted cell order (indices into recs)
    uint32_t* sstart = order + (uint32_t)my_n;                  // [NBX+1] prefix offsets into order[]
    const InterleavedAddrGen<true> sg = {.bank_base_address=geom_dram,
        .page_size=(uint32_t)((uint64_t)ncells*GREC*4u)};

    if (my_n) {
        // load my slice (one contiguous DRAM read; my_first*128 is 64-aligned for cpc%4==0)
        noc_async_read(sg.get_noc_addr(0) + (uint64_t)my_first*GREC*4u, (uint32_t)recs, my_n*GREC*4u);
        noc_async_read_barrier();
        // counting sort by bxl (record word [1]). sstart[b] = #cells with bxl<b in slice.
        for (uint32_t b=0;b<=NBX;++b) sstart[b]=0u;
        for (uint32_t i=0;i<my_n;++i){ uint32_t bxl=recs[i*GREC+1]; if(bxl>=NBX)bxl=NBX-1u; sstart[bxl+1u]++; }
        for (uint32_t b=0;b<NBX;++b) sstart[b+1u]+=sstart[b];
        // scatter indices into order[] (cursor reuses... use a second pass via sstart copy in field buf area)
        // place: walk cells, append to their bin's running cursor. Use order[] built from sstart.
        for (uint32_t i=0;i<my_n;++i){ uint32_t bxl=recs[i*GREC+1]; if(bxl>=NBX)bxl=NBX-1u;
            order[sstart[bxl]++]=i; }
        // undo the cursor advance: sstart[b] was bumped to sstart[b+1]; shift back.
        for (uint32_t b=NBX;b>0u;--b) sstart[b]=sstart[b-1u];
        sstart[0]=0u;
    }

    volatile tt_l1_ptr uint32_t* tile_sem_w =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(tile_sid));
    uint64_t ready_addr = get_noc_addr(core0_noc_x, core0_noc_y, (uint32_t)get_semaphore(ready_sid));
    noc_semaphore_inc(ready_addr, 1u);
    uint64_t done_addr = get_noc_addr(core0_noc_x, core0_noc_y, (uint32_t)get_semaphore(done_sid));

    const InterleavedAddrGen<true> oig={.bank_base_address=oidx_dram, .page_size=n_batches*B*4u};
    uint32_t* oib=(uint32_t*)get_write_ptr(CB_OISCRATCH);

    // Emit one v28 batch of `count` cells drawn from order[off..off+count) (already
    // bxl-sorted). Field tile `tcol0` (halo'd, vcols wide) resident at `fld`.
    auto emit=[&](const uint32_t* idxs, uint32_t count, uint32_t bidx,
                  const float* fldx, const float* fldy, uint32_t tcol0, uint32_t vcols){
        cb_reserve_back(CB_PX,max_k); cb_reserve_back(CB_PY,max_h); cb_reserve_back(CB_RATIO,1);
        float* pxb=(float*)get_write_ptr(CB_PX); float* pyb=(float*)get_write_ptr(CB_PY);
        float* rab=(float*)get_write_ptr(CB_RATIO);
        for (uint32_t j=0;j<B;++j){
            if(j<count){ const uint32_t r=idxs[j]*GREC;
                for(uint32_t k=0;k<max_k;++k) pxb[k*B+j]=*reinterpret_cast<float*>(&recs[r+6+k]);
                for(uint32_t h=0;h<max_h;++h) pyb[h*B+j]=*reinterpret_cast<float*>(&recs[r+14+h]);
                rab[j]=*reinterpret_cast<float*>(&recs[r+5]); oib[j]=recs[r+0];
            } else { for(uint32_t k=0;k<max_k;++k)pxb[k*B+j]=0.f; for(uint32_t h=0;h<max_h;++h)pyb[h*B+j]=0.f;
                     rab[j]=0.f; oib[j]=0xFFFFFFFFu; }
        }
        noc_async_write((uint32_t)oib, oig.get_noc_addr(0)+(uint64_t)bidx*B*4u, B*4u);
        noc_async_write_barrier();
        cb_push_back(CB_PX,max_k); cb_push_back(CB_PY,max_h); cb_push_back(CB_RATIO,1);
        for (uint32_t k=0;k<max_k;++k) for (uint32_t h=0;h<max_h;++h){
            cb_reserve_back(CB_FX,1); cb_reserve_back(CB_FY,1);
            float* fxp=(float*)get_write_ptr(CB_FX); float* fyp=(float*)get_write_ptr(CB_FY);
            for (uint32_t j=0;j<B;++j){ if(j<count){ const uint32_t r=idxs[j]*GREC;
                    uint32_t bxl=recs[r+1]; uint32_t byl=recs[r+2];
                    uint32_t col=bxl-tcol0+k; if(col>=vcols)col=vcols-1u;
                    uint32_t row=byl+h; if(row>=NBY)row=NBY-1u; const uint32_t idx=col*NBY+row;
                    fxp[j]=fldx[idx]; fyp[j]=fldy[idx];
                } else { fxp[j]=0.f; fyp[j]=0.f; } }
            cb_push_back(CB_FX,1); cb_push_back(CB_FY,1);
        }
    };

    uint32_t emitted=0;
    { DeviceZoneScopedN("FCCS-COMPUTE");
      for (uint32_t t=0;t<n_tiles;++t){
          uint32_t col0=t*W; uint32_t vcols=(col0<NBX)?((NBX-col0<WH)?(NBX-col0):WH):0u;
          noc_semaphore_wait_min(tile_sem_w, t+1u);
          const float* fld = reinterpret_cast<const float*>((uint32_t)field0 + (t&1u)*buf_stride);
          const float* fldx = fld; const float* fldy = fld + vcols*NBY;
          if (my_n && vcols){
              // cells OWNED by this tile: bxl in [col0, col0+W). full window ⊆ halo'd vcols.
              uint32_t hi = col0+W; if(hi>NBX)hi=NBX;
              uint32_t s_lo=sstart[col0], s_hi=sstart[hi];
              for (uint32_t s=s_lo; s<s_hi && emitted<n_batches; ){
                  uint32_t cn=(s_hi-s<B)?(s_hi-s):B;
                  emit(order+s, cn, emitted, fldx, fldy, col0, vcols);
                  ++emitted; s+=cn;
              }
          }
          noc_semaphore_inc(done_addr, 1u);
      }
    }
    while (emitted<n_batches){ emit(order, 0, emitted, field0, field0, 0u, 1u); ++emitted; }  // pad
}
