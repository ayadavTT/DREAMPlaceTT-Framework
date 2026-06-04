// SPDX-License-Identifier: Apache-2.0
//
// FCCS on-chip counting-sort (streaming) — for cell counts whose per-worker geom slice
// exceeds L1 (e.g. adaptec3, 1.39M cells). Each worker counting-sorts its contiguous
// slice [my_first, my_first+my_n) of the forward's cell-order geom (V31_GEOM, 128 B/cell,
// int px/py) BY bxl, place-scattering the records into sorted_geom_dram[my_first + ...]
// and writing its sort_start[NBX+1] prefix row. The slice is STREAMED in chunks (never
// fully resident), so L1 use is independent of cell count. The streaming backward then
// reads sort_start + streams the sorted slice per tile-window (sequential).
//
// Record (32 u32): [0]orig [1]bxl [2]byl [3]kc [4]hc [5]ratio [6..13]px [14..21]py.
//
// Args: 0 my_core 1 geom_dram(cell-order src) 2 sorted_geom_dram(dst) 3 sort_start_dram
//   4 ncells 5 my_first 6 my_n 7 NBX 8 hist_l1_off 9 cursor_l1_off 10 chunk_l1_off

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif

constexpr uint32_t CB_SCRATCH = tt::CBIndex::c_24;
constexpr uint32_t GREC = 32u;          // u32 per record (128 B)
constexpr uint32_t CHUNK = 512u;        // records streamed per read

void kernel_main() {
    const uint32_t my_core   = get_arg_val<uint32_t>(0);
    const uint32_t geom_dram = get_arg_val<uint32_t>(1);
    const uint32_t sorted_dram = get_arg_val<uint32_t>(2);
    const uint32_t ss_dram   = get_arg_val<uint32_t>(3);
    const uint32_t ncells    = get_arg_val<uint32_t>(4);
    const uint32_t my_first  = get_arg_val<uint32_t>(5);
    const uint32_t my_n      = get_arg_val<uint32_t>(6);
    const uint32_t NBX       = get_arg_val<uint32_t>(7);
    const uint32_t hist_off  = get_arg_val<uint32_t>(8);
    const uint32_t cursor_off= get_arg_val<uint32_t>(9);
    const uint32_t chunk_off = get_arg_val<uint32_t>(10);
    const uint32_t CG        = get_arg_val<uint32_t>(11);   // compact record size (u32)
    const uint32_t PY_OFF    = get_arg_val<uint32_t>(12);   // py base in compact (=6+max_k)
    const uint32_t max_k     = get_arg_val<uint32_t>(13);
    const uint32_t max_h     = get_arg_val<uint32_t>(14);
    const uint32_t slice_off = get_arg_val<uint32_t>(15);
    if (my_n == 0u) return;

    uint8_t* base = reinterpret_cast<uint8_t*>(get_write_ptr(CB_SCRATCH));
    uint32_t* hist   = reinterpret_cast<uint32_t*>(base + hist_off);     // [NBX+1]
    uint32_t* cursor = reinterpret_cast<uint32_t*>(base + cursor_off);   // [NBX+1]
    int32_t*  chunk  = reinterpret_cast<int32_t*>(base + chunk_off);     // CHUNK*GREC (128 B read buf)
    int32_t*  slice  = reinterpret_cast<int32_t*>(base + slice_off);     // my_n*CG (L1 compact slice)

    const InterleavedAddrGen<true> sg = {.bank_base_address=geom_dram,   .page_size=(uint32_t)((uint64_t)ncells*GREC*4u)};
    const InterleavedAddrGen<true> dg = {.bank_base_address=sorted_dram, .page_size=(uint32_t)((uint64_t)ncells*CG*4u)};
    const uint32_t ssrow_bytes = (((NBX+1u)*4u)+63u)&~63u;               // 64-aligned sort_start row
    const InterleavedAddrGen<true> ssg = {.bank_base_address=ss_dram, .page_size=ssrow_bytes};

    for (uint32_t b=0;b<=NBX;++b) hist[b]=0u;

    // ── PASS 1: read the 128 B geom ONCE (chunked), extract the COMPACT record into the L1 slice,
    //    and histogram bxl. The slice (my_n*CG) stays resident so PASS 2 needs no DRAM re-read. ──
    for (uint32_t s=0; s<my_n; ) {
        uint32_t cn = (my_n-s < CHUNK) ? (my_n-s) : CHUNK;
        noc_async_read(sg.get_noc_addr(0) + (uint64_t)(my_first+s)*GREC*4u, (uint32_t)chunk, cn*GREC*4u);
        noc_async_read_barrier();
        for (uint32_t j=0;j<cn;++j){
            const int32_t* r = chunk + j*GREC; int32_t* c = slice + (s+j)*CG;
            for (uint32_t i=0;i<6u;++i)    c[i]        = r[i];        // header [0..5]
            for (uint32_t k=0;k<max_k;++k) c[6u+k]     = r[6u+k];     // px
            for (uint32_t h=0;h<max_h;++h) c[PY_OFF+h] = r[14u+h];    // py
            uint32_t bxl=(uint32_t)r[1]; if(bxl>=NBX)bxl=NBX-1u; hist[bxl+1u]++;
        }
        s += cn;
    }
    // prefix → hist[b] = #cells with bxl<b ; write sort_start row
    for (uint32_t b=0;b<NBX;++b) hist[b+1u]+=hist[b];
    noc_async_write((uint32_t)hist, ssg.get_noc_addr(my_core), (NBX+1u)*4u);
    noc_async_write_barrier();
    for (uint32_t b=0;b<=NBX;++b) cursor[b]=hist[b];   // running place cursor

    // ── PASS 2: place each compact record from the L1 slice → sorted_geom[my_first+cursor[bxl]++].
    //    No DRAM read; issue all scatter writes then a SINGLE barrier (the L1 source is stable). ──
    for (uint32_t i=0; i<my_n; ++i){
        const int32_t* c = slice + i*CG;
        uint32_t bxl=(uint32_t)c[1]; if(bxl>=NBX)bxl=NBX-1u;
        uint32_t dst = my_first + cursor[bxl]; cursor[bxl]++;
        noc_async_write((uint32_t)c, dg.get_noc_addr(0) + (uint64_t)dst*CG*4u, CG*4u);
    }
    noc_async_write_barrier();
}
