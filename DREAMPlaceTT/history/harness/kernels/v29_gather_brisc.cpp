// SPDX-License-Identifier: Apache-2.0
//
// V29 EF backward stage 2 — GATHER (BRISC). Reads this band's bucketed records
// route[*][my_band] (assembling 1024-cell batches across the nc source pages),
// loads the x-bin field band resident in L1, and per (k,h) gathers
// field[(bin_xl_local+k)·M+(bin_yl+h)] by direct L1 load. Emits px/py/fx/fy/
// ratio + orig_idx tiles; v28_compute does Σ px·py·f·ratio; v29_writer scatters
// grad[orig_idx]. Record 128 B (32 u32): [0]orig [1]bxl_loc [2]byl [3]kc [4]hc
// [5]ratio [6..13]px [14..21]py.

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    const uint32_t my_band    = get_arg_val<uint32_t>(0);
    const uint32_t route_base = get_arg_val<uint32_t>(1);
    const uint32_t cap        = get_arg_val<uint32_t>(2);
    const uint32_t counts_base= get_arg_val<uint32_t>(3);
    const uint32_t nc         = get_arg_val<uint32_t>(4);
    const uint32_t max_k      = get_arg_val<uint32_t>(5);
    const uint32_t max_h      = get_arg_val<uint32_t>(6);
    const uint32_t fx_base    = get_arg_val<uint32_t>(7);
    const uint32_t fy_base    = get_arg_val<uint32_t>(8);
    const uint32_t field_pg   = get_arg_val<uint32_t>(9);
    const uint32_t M          = get_arg_val<uint32_t>(10);
    const uint32_t cpc_x      = get_arg_val<uint32_t>(11);
    const uint32_t valid_cols = get_arg_val<uint32_t>(12);
    const uint32_t n_batches  = get_arg_val<uint32_t>(13);
    const uint32_t cnt_pg     = get_arg_val<uint32_t>(14);  // 64-aligned counts page bytes
    const uint32_t col0       = get_arg_val<uint32_t>(15);  // clamped (<N) band start col — safe field read
    const uint32_t oidx_base  = get_arg_val<uint32_t>(16);  // DRAM orig_idx buffer (page per CORE)
    const uint32_t max_batches= get_arg_val<uint32_t>(17);  // batches per core page (oidx layout)
    // ── Adaptive assignment: multiple cores share a hot band (field replicated),
    //    each handling a contiguous sub-range of the band's concatenated cells. ──
    const uint32_t my_core    = get_arg_val<uint32_t>(18);  // output page (oidx/grad), balanced
    const uint32_t skip       = get_arg_val<uint32_t>(19);  // cells to skip before my range (band-major)
    const uint32_t my_cells   = get_arg_val<uint32_t>(20);  // cells this core processes (≤ my range)
    const uint32_t n_bands    = get_arg_val<uint32_t>(21);  // # consecutive bands covered (cold-merge; ≥1)
    const uint32_t lo_band    = my_band;                    // arg0 = first band of the covered range

    constexpr uint32_t B=1024, REC=32u;  // u32 per record
    // c_5 is now a plain L1 scratch for building one batch's orig_idx (written to
    // DRAM, NOT a CB handshake) — keeps the compute↔writer handshake to GX/GY only
    // (the v28-proven 3-CB pattern; OIDX-as-CB was the deadlock source).
    constexpr auto CB_PX=tt::CBIndex::c_0, CB_PY=tt::CBIndex::c_1, CB_FX=tt::CBIndex::c_2,
                   CB_FY=tt::CBIndex::c_3, CB_RATIO=tt::CBIndex::c_4, CB_OISCRATCH=tt::CBIndex::c_5;
    constexpr auto CB_FIELDX=tt::CBIndex::c_24, CB_STAGE=tt::CBIndex::c_25, CB_FIELDY=tt::CBIndex::c_26, CB_CNT=tt::CBIndex::c_27;

    const uint32_t route_pg=cap*128u, oidx_pg=max_batches*B*4u;
    const InterleavedAddrGen<true> rg ={.bank_base_address=route_base, .page_size=route_pg};
    const InterleavedAddrGen<true> oig={.bank_base_address=oidx_base, .page_size=oidx_pg};
    const InterleavedAddrGen<true> fxg={.bank_base_address=fx_base, .page_size=field_pg};
    const InterleavedAddrGen<true> fyg={.bank_base_address=fy_base, .page_size=field_pg};
    const InterleavedAddrGen<true> cntg={.bank_base_address=counts_base, .page_size=cnt_pg};

    // counts column: cnt[src] = count[src][my_band]. The 64-B NoC read needs a
    // 64-B-aligned L1 destination — a bare stack array isn't aligned, so read
    // into the (aligned) tail of the CB_CNT L1 region instead.
    cb_reserve_back(CB_CNT,1); uint32_t* cnt=reinterpret_cast<uint32_t*>(get_write_ptr(CB_CNT));
    uint8_t* cnt_tmp=reinterpret_cast<uint8_t*>(((uint32_t)(cnt+nc)+63u)&~63u);  // 64-aligned scratch after cnt[]
    // cnt[src] = count[src][band] (clamped to cap). Reloaded per band in the assembly
    // loop so a cold-merge worker can stream several bands' route pages.
    auto load_cnt=[&](uint32_t band){
        const uint32_t aoff=(band*4u)&~63u; const uint32_t intra=(band*4u)-aoff;
        for (uint32_t s=0;s<nc;++s){ noc_async_read(cntg.get_noc_addr(s)+aoff,(uint32_t)cnt_tmp,64u); noc_async_read_barrier();
                                     uint32_t v=*reinterpret_cast<uint32_t*>(cnt_tmp+intra); cnt[s]=(v>cap)?cap:v; } };

    // field x-bin band resident: columns [col0, col0+valid_cols). col0 is clamped
    // <N by the host so empty edge bands (band*cpc_x≥N) don't read past the field.
    (void)cpc_x;
    float *fieldx,*fieldy;
    { DeviceZoneScopedN("V29-LOADBAND");
      cb_reserve_back(CB_FIELDX,1); uint32_t lx=get_write_ptr(CB_FIELDX);
      cb_reserve_back(CB_FIELDY,1); uint32_t ly=get_write_ptr(CB_FIELDY);
      noc_async_read(fxg.get_noc_addr(0)+(uint64_t)col0*M*4u, lx, valid_cols*M*4u);
      noc_async_read(fyg.get_noc_addr(0)+(uint64_t)col0*M*4u, ly, valid_cols*M*4u);
      noc_async_read_barrier(); fieldx=reinterpret_cast<float*>(lx); fieldy=reinterpret_cast<float*>(ly);
    }
    const uint32_t stage_l1=get_write_ptr(CB_STAGE);
    const uint32_t* su=reinterpret_cast<const uint32_t*>(stage_l1);
    const float*    sf=reinterpret_cast<const float*>(stage_l1);

    uint32_t* oib=(uint32_t*)get_write_ptr(CB_OISCRATCH);  // L1 scratch for one batch's orig_idx
    auto emit=[&](uint32_t count, uint32_t bidx){
        DeviceZoneScopedN("V29-EMIT");
        cb_reserve_back(CB_PX,max_k); cb_reserve_back(CB_PY,max_h); cb_reserve_back(CB_RATIO,1);
        float* pxb=(float*)get_write_ptr(CB_PX); float* pyb=(float*)get_write_ptr(CB_PY);
        float* rab=(float*)get_write_ptr(CB_RATIO);
        for (uint32_t j=0;j<B;++j){ if(j<count){ const uint32_t base=j*REC;
                for(uint32_t k=0;k<max_k;++k)pxb[k*B+j]=sf[base+6+k];
                for(uint32_t h=0;h<max_h;++h)pyb[h*B+j]=sf[base+14+h];
                rab[j]=sf[base+5]; oib[j]=su[base+0];
            } else { for(uint32_t k=0;k<max_k;++k)pxb[k*B+j]=0.f; for(uint32_t h=0;h<max_h;++h)pyb[h*B+j]=0.f;
                     rab[j]=0.f; oib[j]=0xFFFFFFFFu; } }
        // orig_idx → DRAM (page=band, offset=batch). Written before fx/fy push, so by
        // the time the writer gets GX(bidx) it's already landed. No CB handshake.
        noc_async_write((uint32_t)oib, oig.get_noc_addr(my_core)+(uint64_t)bidx*B*4u, B*4u);
        noc_async_write_barrier();
        cb_push_back(CB_PX,max_k); cb_push_back(CB_PY,max_h); cb_push_back(CB_RATIO,1);
        for (uint32_t k=0;k<max_k;++k) for (uint32_t h=0;h<max_h;++h){
            cb_reserve_back(CB_FX,1); cb_reserve_back(CB_FY,1);
            float* fxp=(float*)get_write_ptr(CB_FX); float* fyp=(float*)get_write_ptr(CB_FY);
            for (uint32_t j=0;j<B;++j){ if(j<count){ const uint32_t base=j*REC;
                    // su[base+1] = bin_xl GLOBAL; field band starts at global col0.
                    uint32_t col=su[base+1]-col0+k; if(col>=valid_cols)col=valid_cols-1;
                    uint32_t row=su[base+2]+h; if(row>=M)row=M-1; const uint32_t idx=col*M+row;
                    fxp[j]=fieldx[idx]; fyp[j]=fieldy[idx];
                } else { fxp[j]=0.f; fyp[j]=0.f; } }
            cb_push_back(CB_FX,1); cb_push_back(CB_FY,1);
        }
    };

    uint32_t fill=0, emitted=0;
#ifdef V29_EMPTY_DIAG
    while (emitted<n_batches){ emit(0,emitted); ++emitted; }
    return;
#endif
    // Assemble this core's sub-range [skip, skip+my_cells) of the concatenated cells
    // across the covered band range [lo_band, lo_band+n_bands) — band-major, src-minor
    // (route[0..nc-1][band]); emit n_batches=ceil(my_cells/1024). n_bands>1 = cold-merge
    // (one worker drains several low-count bands sharing one wide field band); a hot band
    // is n_bands=1 split across workers via skip/my_cells.
    uint32_t skipped=0, done_cells=0;
    for (uint32_t bi=0; bi<n_bands && emitted<n_batches; ++bi){
        const uint32_t band = lo_band + bi;
        load_cnt(band);
        for (uint32_t src=0; src<nc && emitted<n_batches; ++src){
            uint32_t rem=cnt[src], soff=0;
            if (skipped<skip){ uint32_t s=(rem<(skip-skipped))?rem:(skip-skipped); soff+=s; rem-=s; skipped+=s; }
            while (rem>0 && done_cells<my_cells && emitted<n_batches){
                uint32_t space=B-fill; uint32_t want=my_cells-done_cells;
                uint32_t take=(rem<space)?rem:space; if(take>want)take=want;
                noc_async_read(rg.get_noc_addr(src*nc+band)+(uint64_t)soff*128u,
                               stage_l1+fill*128u, take*128u);
                noc_async_read_barrier();
                fill+=take; soff+=take; rem-=take; done_cells+=take;
                if (fill==B){ emit(B,emitted); ++emitted; fill=0; }
            }
        }
    }
    if (fill>0 && emitted<n_batches){ emit(fill,emitted); ++emitted; fill=0; }
    while (emitted<n_batches){ emit(0,emitted); ++emitted; }   // pad to match compute/writer loop
}
