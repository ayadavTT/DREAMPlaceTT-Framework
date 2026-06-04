// SPDX-License-Identifier: Apache-2.0
//
// FCCS on-chip EF-geometry kernel — host-free replacement for fccs_compute_geom_host.
// Each core computes the ELECTRIC-FORCE overlap geometry for a contiguous slice of active
// cells and writes the V31_GEOM int records the FCCS backward reads. This is DREAMPlace's
// electric_force footprint, NOT the density footprint: triangle_density_function(
//   node_x = pos[g] + offset_x[g],  node_size = node_size_x_clamped[g] ).
// px/py are stashed as int32 = triangle · sqrt(inv_bin_area) · 2^WS (matches the kernel's
// descale: host applies · bin_area · ratio / 2^(2WS+FS)).
//
// Record (32 u32): [0]orig=i [1]bxl [2]byl [3]kc [4]hc [5]0 [6..13]px_int [14..21]py_int.
//
// Args: 0 my_core 1 nc 2 cpc 3 na 4 nn 5 NBX 6 NBY 7 WS 8 pos_dram 9 sel_dram
//   10 ox_dram 11 oy_dram 12 nsx_dram 13 nsy_dram 14 geom_dram
//   15 xl_bits 16 yl_bits 17 invbsx_bits 18 invbsy_bits 19 bsx_bits 20 bsy_bits 21 sqiba_bits

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif

constexpr uint32_t GREC = 32u;
constexpr uint32_t MAXKH = 8u;
constexpr uint32_t CB = tt::CBIndex::c_24;

static inline float fbits(uint32_t b){ float f; __builtin_memcpy(&f,&b,4); return f; }

void kernel_main() {
    const uint32_t my_core = get_arg_val<uint32_t>(0);
    const uint32_t nc      = get_arg_val<uint32_t>(1);
    const uint32_t cpc     = get_arg_val<uint32_t>(2);
    const uint32_t na      = get_arg_val<uint32_t>(3);
    const uint32_t nn      = get_arg_val<uint32_t>(4);
    const int32_t  NBX     = (int32_t)get_arg_val<uint32_t>(5);
    const int32_t  NBY     = (int32_t)get_arg_val<uint32_t>(6);
    const uint32_t WS      = get_arg_val<uint32_t>(7);
    const uint32_t pos_d   = get_arg_val<uint32_t>(8);
    const uint32_t sel_d   = get_arg_val<uint32_t>(9);
    const uint32_t ox_d    = get_arg_val<uint32_t>(10);
    const uint32_t oy_d    = get_arg_val<uint32_t>(11);
    const uint32_t nsx_d   = get_arg_val<uint32_t>(12);
    const uint32_t nsy_d   = get_arg_val<uint32_t>(13);
    const uint32_t geom_d  = get_arg_val<uint32_t>(14);
    const float xl     = fbits(get_arg_val<uint32_t>(15));
    const float yl     = fbits(get_arg_val<uint32_t>(16));
    const float invbsx = fbits(get_arg_val<uint32_t>(17));
    const float invbsy = fbits(get_arg_val<uint32_t>(18));
    const float bsx    = fbits(get_arg_val<uint32_t>(19));
    const float bsy    = fbits(get_arg_val<uint32_t>(20));
    const float sqiba  = fbits(get_arg_val<uint32_t>(21));   // sqrt(inv_bin_area)
    const float wscale = (float)(1u << WS);

    uint32_t my_first = my_core * cpc;
    uint32_t my_n = 0;
    if (my_first < na) { my_n = na - my_first; if (my_n > cpc) my_n = cpc; }
    if (!my_n) return;

    // single-page interleaved buffers (whole-array per page)
    const InterleavedAddrGen<true> posg = {.bank_base_address=pos_d, .page_size=(uint32_t)((uint64_t)2u*nn*4u)};
    const InterleavedAddrGen<true> selg = {.bank_base_address=sel_d, .page_size=(uint32_t)((uint64_t)na*4u)};
    const InterleavedAddrGen<true> oxg  = {.bank_base_address=ox_d,  .page_size=(uint32_t)((uint64_t)na*4u)};
    const InterleavedAddrGen<true> oyg  = {.bank_base_address=oy_d,  .page_size=(uint32_t)((uint64_t)na*4u)};
    const InterleavedAddrGen<true> nsxg = {.bank_base_address=nsx_d, .page_size=(uint32_t)((uint64_t)na*4u)};
    const InterleavedAddrGen<true> nsyg = {.bank_base_address=nsy_d, .page_size=(uint32_t)((uint64_t)na*4u)};
    const InterleavedAddrGen<true> geomg= {.bank_base_address=geom_d,.page_size=(uint32_t)((uint64_t)na*GREC*4u)};

    uint8_t* base = reinterpret_cast<uint8_t*>(get_write_ptr(CB));
    // scratch: a 64B-aligned staging area for one int32 read + one 128B geom record
    uint32_t* rd  = reinterpret_cast<uint32_t*>(base);                 // 16 words scratch
    uint32_t* rec = reinterpret_cast<uint32_t*>(base + 64);            // 32-word geom record

    for (uint32_t s=0; s<my_n; ++s) {
        const uint32_t i = my_first + s;
        // gather per-cell inputs (scattered single-element reads — cheap, host-free)
        noc_async_read(selg.get_noc_addr(0) + (uint64_t)i*4u, (uint32_t)rd, 4u); noc_async_read_barrier();
        const uint32_t g = rd[0];
        noc_async_read(oxg.get_noc_addr(0)  + (uint64_t)i*4u, (uint32_t)(rd+1), 4u);
        noc_async_read(oyg.get_noc_addr(0)  + (uint64_t)i*4u, (uint32_t)(rd+2), 4u);
        noc_async_read(nsxg.get_noc_addr(0) + (uint64_t)i*4u, (uint32_t)(rd+3), 4u);
        noc_async_read(nsyg.get_noc_addr(0) + (uint64_t)i*4u, (uint32_t)(rd+4), 4u);
        noc_async_read(posg.get_noc_addr(0) + (uint64_t)g*4u,        (uint32_t)(rd+5), 4u);
        noc_async_read(posg.get_noc_addr(0) + (uint64_t)(nn+g)*4u,   (uint32_t)(rd+6), 4u);
        noc_async_read_barrier();
        const float ox=fbits(rd[1]), oy=fbits(rd[2]), sx=fbits(rd[3]), sy=fbits(rd[4]);
        const float nx=fbits(rd[5])+ox, ny=fbits(rd[6])+oy;

        int32_t bxl=(int32_t)((nx-xl)*invbsx); int32_t bxh=(int32_t)((nx+sx-xl)*invbsx)+1;
        if(bxl<0)bxl=0; if(bxh>NBX)bxh=NBX; int32_t kc=bxh-bxl; if(kc<0)kc=0; if(kc>(int32_t)MAXKH)kc=MAXKH;
        int32_t byl=(int32_t)((ny-yl)*invbsy); int32_t byh=(int32_t)((ny+sy-yl)*invbsy)+1;
        if(byl<0)byl=0; if(byh>NBY)byh=NBY; int32_t hc=byh-byl; if(hc<0)hc=0; if(hc>(int32_t)MAXKH)hc=MAXKH;

        for(uint32_t z=0;z<GREC;++z) rec[z]=0;
        rec[0]=i; rec[1]=(uint32_t)bxl; rec[2]=(uint32_t)byl; rec[3]=(uint32_t)kc; rec[4]=(uint32_t)hc;
        const float nxr=nx+sx, nyr=ny+sy;
        for(int32_t k=0;k<kc;++k){ float bx=xl+(float)(bxl+k)*bsx; float r=bx+bsx; float lo=(nx>bx)?nx:bx;
            float hi=(nxr<r)?nxr:r; float tx=hi-lo; if(tx<0.f)tx=0.f;
            rec[6+k]=(uint32_t)(int32_t)(tx*sqiba*wscale); }
        for(int32_t h=0;h<hc;++h){ float by=yl+(float)(byl+h)*bsy; float r=by+bsy; float lo=(ny>by)?ny:by;
            float hi=(nyr<r)?nyr:r; float ty=hi-lo; if(ty<0.f)ty=0.f;
            rec[14+h]=(uint32_t)(int32_t)(ty*sqiba*wscale); }

        noc_async_write((uint32_t)rec, geomg.get_noc_addr(0) + (uint64_t)i*GREC*4u, GREC*4u);
        noc_async_write_barrier();
    }
}
