// SPDX-License-Identifier: Apache-2.0
//
// V35 EF backward engine — see v35_ef_engine.h. Fully on-chip count→plan→place→gather
// over the forward V31_GEOM stash + chip-DCT field. Persistent buffers/programs built
// on first compute; per iter re-runs count, recomputes the host plan (counts→tile_base
// +srcprefix+greedy alloc), re-sets gather args, runs place + gather, unsorts grad.

#include "v35_ef_engine.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <set>
#include <string>
#include <vector>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tt_metal.hpp>

using namespace tt; using namespace tt::tt_metal; using namespace tt::tt_metal::distributed;
using hrclock = std::chrono::high_resolution_clock;
template<class T> static double ms_since(T t){return std::chrono::duration<double,std::milli>(hrclock::now()-t).count();}
#ifndef DENSITY_KERNEL_DIR
#define DENSITY_KERNEL_DIR "kernels/"
#endif
namespace v35ef {
static constexpr uint32_t MAXKH = 8;

struct V35EFEngine::Impl {
    MeshDevice* dev; MeshCommandQueue* cq;
    int NBX, NBY, nmax; float xl, yl, bsx, bsy, inv_bsx, inv_bsy;
    int nc; CoreCoord grid; std::vector<CoreCoord> ccs; CoreRangeSet crs{std::set<CoreRange>{}};
    // configured
    int n_active=0, n_total=0; std::vector<int32_t> sel;
    std::vector<float> ox,oy,nsx,nsy,ratio; uint32_t max_k=1, max_h=1;
    uint32_t geom_addr=0, geom_pg=0;
    // sizing (first compute)
    bool built=false; bool compact=false; bool dual=false; int nsrc=0; uint32_t W=1, ntiles=1, band_cols=1, cpc_cell=1, mc=1024;
    uint32_t cnt_pg=0, tb_pg=0, plan_pg=0, grouped_pg=0, out_pg=0, field_pg=0, rpw=0;
    static constexpr uint32_t PCHUNK=1024, CHUNK_BATCHES=4;
    std::shared_ptr<MeshBuffer> cntb, tbb, spb, grb, gxb, gyb, oibb;
    MeshWorkload wl_count, wl_place, wl_gather; MeshCoordinateRange dr{MeshCoordinate{0,0},MeshCoordinate{0,0}};
    KernelHandle kc=0, kp=0, bk=0, ck=0, wk=0, kc_n=0, kp_n=0;
    uint32_t diag_n=0; uint32_t diag_n2=0;
    // host staging
    std::vector<uint32_t> cnts, tile_base, srcpref, oidx_host;
    std::vector<float> gradx, grady;
    std::vector<uint32_t> g_first, g_n, g_col0, g_nb;

    Impl(void* md,int M_,int N_,int nmax_,float xl_,float yl_,float bsx_,float bsy_)
      : dev(static_cast<MeshDevice*>(md)), NBX(M_), NBY(N_), nmax(nmax_),
        xl(xl_),yl(yl_),bsx(bsx_),bsy(bsy_),inv_bsx(1.f/bsx_),inv_bsy(1.f/bsy_) {
        cq=&dev->mesh_command_queue(); grid=dev->compute_with_storage_grid_size(); nc=grid.x*grid.y;
        std::set<CoreRange> s; for(uint32_t y=0;y<grid.y;++y)for(uint32_t x=0;x<grid.x;++x){CoreCoord c{x,y};ccs.push_back(c);s.insert(CoreRange{c,c});}
        crs=CoreRangeSet(s);
    }
};

V35EFEngine::V35EFEngine(void* md,int M,int N,int nmax,float xl,float yl,float bsx,float bsy)
  : impl_(std::make_unique<Impl>(md,M,N,nmax,xl,yl,bsx,bsy)) {}
V35EFEngine::~V35EFEngine() = default;

void V35EFEngine::configure_with_sel(const float* ox_f,const float* oy_f,const float* nsx_f,
        const float* nsy_f,const float* ratio_f,const int32_t* sel,int na,int nt){
    auto&I=*impl_; I.n_active=na; I.n_total=nt; I.sel.assign(sel,sel+na);
    I.ox.resize(na);I.oy.resize(na);I.nsx.resize(na);I.nsy.resize(na);I.ratio.resize(na);
    for(int i=0;i<na;++i){int g=sel[i];I.ox[i]=ox_f[g];I.oy[i]=oy_f[g];I.nsx[i]=nsx_f[g];I.nsy[i]=nsy_f[g];
        I.ratio[i]=ratio_f?ratio_f[g]:1.f;}
    // max_k/max_h from ORIG span (orig = clamped + 2|offset|, the forward's footprint), capped at 8.
    uint32_t mk=1,mh=1;
    for(int i=0;i<na;++i){ float ow=I.nsx[i]+2.f*std::fabs(I.ox[i]); float oh=I.nsy[i]+2.f*std::fabs(I.oy[i]);
        uint32_t k=(uint32_t)std::floor(ow*I.inv_bsx)+2u; uint32_t h=(uint32_t)std::floor(oh*I.inv_bsy)+2u; mk=std::max(mk,k); mh=std::max(mh,h);}
    I.max_k=std::min(mk,MAXKH); I.max_h=std::min(mh,MAXKH);
    I.cpc_cell=((uint32_t)na+I.nc-1)/I.nc;
    // Diagnostic: per-cell footprint (k,h) distribution. The gather field-production cost is
    // ∝ Σ_cells (kc·hc); today every cell pays the GLOBAL max_k·max_h. mean_product vs that
    // uniform product is the bucketing/filler-fast-path ceiling. (V35_FOOTPRINT=1, one-shot.)
    if(getenv("V35_FOOTPRINT")){
        static bool done=false;
        if(!done){ done=true;
            uint32_t hk[24]={0}, hh[24]={0}, hp[65]={0}; uint64_t prodsum=0; uint32_t maxprod=0;
            for(int i=0;i<na;++i){ float ow=I.nsx[i]+2.f*std::fabs(I.ox[i]); float oh=I.nsy[i]+2.f*std::fabs(I.oy[i]);
                uint32_t k=(uint32_t)std::floor(ow*I.inv_bsx)+2u, h=(uint32_t)std::floor(oh*I.inv_bsy)+2u;
                uint32_t kc=std::min(k,MAXKH), hc=std::min(h,MAXKH); uint32_t p=kc*hc;   // per-cell gather loop footprint
                prodsum+=p; if(p>maxprod)maxprod=p;
                if(k>23)k=23; if(h>23)h=23; hk[k]++; hh[h]++; uint32_t pp=p; if(pp>64)pp=64; hp[pp]++; }
            fprintf(stderr,"[v35_footprint] na=%d max_k=%u max_h=%u uniform=%u max_cell_product=%u mean_product=%.2f (headroom %.0f%%)\n",
                na,I.max_k,I.max_h,I.max_k*I.max_h,maxprod,(double)prodsum/(na>0?na:1),
                100.0*(1.0-(double)prodsum/((double)na*I.max_k*I.max_h)));
            fprintf(stderr,"  product CDF (%% of cells with footprint<=T):");
            uint32_t cum=0; const uint32_t Ts[]={4,8,12,16,20,24,32,48,64}; int ti=0;
            for(uint32_t p=1;p<=64;++p){ cum+=hp[p]; while(ti<9 && Ts[ti]==p){ fprintf(stderr," <=%u:%.1f",Ts[ti],100.0*cum/(na>0?na:1)); ++ti; } }
            fprintf(stderr,"\n  k-hist:"); for(uint32_t k=0;k<16;++k) if(hk[k]) fprintf(stderr," k%u=%u",k,hk[k]);
            fprintf(stderr,"\n  h-hist:"); for(uint32_t h=0;h<16;++h) if(hh[h]) fprintf(stderr," h%u=%u",h,hh[h]); fprintf(stderr,"\n");
        }
    }
}

void V35EFEngine::set_geom_source(uint32_t geom_addr,uint32_t geom_pg){ impl_->geom_addr=geom_addr; impl_->geom_pg=geom_pg; }

void V35EFEngine::compute_from_chip(uint32_t fx_addr,uint32_t fy_addr,float* grad,V35Timing* t){
    auto&I=*impl_; V35Timing tm{}; auto t0=hrclock::now();
    const int na=I.n_active, nn=I.n_total, NBX=I.NBX, NBY=I.NBY, nc=I.nc;
    using tt::CBIndex;
    auto cbf=[&](Program&p,uint32_t idx,uint32_t bytes,uint32_t slots){CircularBufferConfig c(slots*bytes,{{(CBIndex)idx,DataFormat::Float32}});c.set_page_size((CBIndex)idx,bytes);CreateCircularBuffer(p,I.crs,c);};

    if(!I.built){
        // tile width: band(+halo)/plane ≤ ~350 KB AND ntiles ≤ nc/2 (every tile ≥2 cores)
        uint32_t W=(350u*1024u)/((uint32_t)NBY*4u); if(W>I.max_k)W-=I.max_k; else W=1u;
        uint32_t halfnc=(uint32_t)std::max(1,nc/2); if(W<(((uint32_t)NBX+halfnc-1u)/halfnc))W=((uint32_t)NBX+halfnc-1u)/halfnc;
        if(W>(uint32_t)NBX)W=NBX; if(W<1u)W=1u; I.W=W;
        I.ntiles=((uint32_t)NBX+W-1u)/W; I.band_cols=std::min(W+I.max_k,(uint32_t)NBX);
        I.mc=(((3u*I.cpc_cell)+1023u)/1024u)*1024u;   // generous per-worker output page (imbalance margin)
        I.cnt_pg=((I.ntiles*4u+63u)&~63u); I.tb_pg=I.cnt_pg; I.plan_pg=I.cnt_pg; I.rpw=I.cnt_pg/4u;
        // COMPACT GROUPED RECORD: the gather consumes only gidx/bxl_g/byl/ratio/px[max_k]/py[max_h]
        // (never kc/hc, never px/py past max_k/max_h), so a 64 B (16 u32) record holds all of it when
        // 4+max_k+max_h ≤ 16 → halves place's write + the gather's record read (validated bit-exact in
        // the v35compact harness: place -20..29%, gather -6..12% at bigblue3). Falls back to the 128 B
        // verbatim record if it doesn't fit or V35_NO_COMPACT is set.
        I.compact = (getenv("V35_NO_COMPACT")==nullptr) && ((4u+I.max_k+I.max_h)<=16u);
        const uint32_t rec_b = I.compact ? 64u : 128u;
        // V35_STASH64: the FORWARD wrote a 64 B source stash (px[8]+py[4]) → count/place read
        // 64 B/cell instead of 128 B (forward+backward co-design; halves the DRAM read that
        // bounds count+place). Requires max_h<=4 (forward drops py[4..7]); compact implies it
        // when max_k=8 (4+8+4=16) but assert for the general case.
        const bool stash64 = (getenv("V35_STASH64")!=nullptr) && I.compact;
        if(stash64 && I.max_h>4u) fprintf(stderr,"[v35] ERROR: V35_STASH64 with max_h=%u>4 — forward dropped py[4..7], grad WILL be wrong\n", I.max_h);
        const uint32_t src_b = stash64 ? 64u : 128u;
        // V35_DUAL: split each core into TWO sources (NCRISC half = src c, BRISC half = src c+nc) so
        // count+place run on BOTH data-movement engines (reclaims the idle NCRISC). Requires stash64
        // (uses the 64 B count64/place_s64 + their _n NCRISC variants; disjoint count pages/srcpref).
        I.dual = (getenv("V35_DUAL")!=nullptr) && stash64;
        I.nsrc = I.dual ? 2*nc : nc;
        if(getenv("V35_DUAL")!=nullptr && !stash64) fprintf(stderr,"[v35] V35_DUAL ignored (requires V35_STASH64)\n");
        fprintf(stderr,"[v35] grouped=%uB(%s) source=%uB(%s) dual_risc=%s max_k=%u max_h=%u na=%d nsrc=%d\n",
                rec_b, I.compact?"COMPACT":"verbatim", src_b, stash64?"64B-codesign":"128B", I.dual?"YES":"no", I.max_k, I.max_h, na, I.nsrc);
        // field is page-interleaved: ONE page per x-bin = NBY floats (matches the
        // TTNN DCT output / V21's chip-field read). NOT a single NBX*NBY page.
        I.grouped_pg=(uint32_t)((uint64_t)na*rec_b+rec_b); I.out_pg=I.mc*4u; I.field_pg=(uint32_t)((uint64_t)NBY*4u);
        auto mb=[&](uint64_t b,uint32_t pg){DeviceLocalBufferConfig c{.page_size=pg,.buffer_type=BufferType::DRAM};ReplicatedBufferConfig r{.size=b};return MeshBuffer::create(r,c,I.dev);};
        I.cntb=mb((uint64_t)I.nsrc*I.cnt_pg,I.cnt_pg); I.tbb=mb(I.tb_pg,I.tb_pg); I.spb=mb((uint64_t)I.nsrc*I.plan_pg,I.plan_pg);
        I.grb=mb(I.grouped_pg,I.grouped_pg); I.gxb=mb((uint64_t)nc*I.out_pg,I.out_pg); I.gyb=mb((uint64_t)nc*I.out_pg,I.out_pg); I.oibb=mb((uint64_t)nc*I.out_pg,I.out_pg);
        I.cnts.assign((size_t)I.nsrc*I.rpw,0u); I.tile_base.assign(I.tb_pg/4u,0u); I.srcpref.assign((size_t)I.nsrc*I.rpw,0u);
        I.gradx.assign((size_t)nc*I.mc,0.f); I.grady.assign((size_t)nc*I.mc,0.f); I.oidx_host.assign((size_t)nc*I.mc,0u);
        I.g_first.assign(nc,0); I.g_n.assign(nc,0); I.g_col0.assign(nc,0); I.g_nb.assign(nc,0);
        const uint32_t sta=I.geom_addr, gra=I.grb->address(), cnta=I.cntb->address(), tba=I.tbb->address(), spa=I.spb->address();
        I.dr=MeshCoordinateRange{I.dev->shape()};
        // count program — dual-RISC (BRISC src c+nc, NCRISC src c) when I.dual; else BRISC-only.
        Program pc=CreateProgram(); cbf(pc,0,Impl::PCHUNK*src_b,1); cbf(pc,1,I.cnt_pg,1);
        I.kc=CreateKernel(pc,std::string(DENSITY_KERNEL_DIR)+(stash64?"v35_count64.cpp":"v35_count.cpp"),I.crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        if(I.dual){ cbf(pc,5,Impl::PCHUNK*src_b,1); cbf(pc,6,I.cnt_pg,1);
            I.kc_n=CreateKernel(pc,std::string(DENSITY_KERNEL_DIR)+"v35_count64_n.cpp",I.crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default}); }
        for(int c=0;c<nc;++c){uint32_t f=(uint32_t)c*I.cpc_cell,n=std::min(I.cpc_cell,(f<(uint32_t)na)?((uint32_t)na-f):0u);
            if(I.dual){ uint32_t nlo=n/2u;
                SetRuntimeArgs(pc,I.kc_n,I.ccs[c],{(uint32_t)c,      f,     nlo,   sta,I.geom_pg,I.ntiles,I.W,cnta,I.cnt_pg,Impl::PCHUNK});
                SetRuntimeArgs(pc,I.kc,  I.ccs[c],{(uint32_t)(c+nc), f+nlo, n-nlo, sta,I.geom_pg,I.ntiles,I.W,cnta,I.cnt_pg,Impl::PCHUNK}); }
            else SetRuntimeArgs(pc,I.kc,I.ccs[c],{(uint32_t)c,f,n,sta,I.geom_pg,I.ntiles,I.W,cnta,I.cnt_pg,Impl::PCHUNK}); }
        I.wl_count.add_program(I.dr,std::move(pc));
        // place program — dual-RISC (BRISC src c+nc, NCRISC src c) when I.dual; CB_SORT holds compact 64 B.
        Program pp=CreateProgram(); cbf(pp,0,Impl::PCHUNK*src_b,1); cbf(pp,1,Impl::PCHUNK*rec_b,1); cbf(pp,2,I.tb_pg,1); cbf(pp,3,I.plan_pg,1); cbf(pp,4,((4u*I.ntiles*4u+63u)&~63u),1);
        I.kp=CreateKernel(pp,std::string(DENSITY_KERNEL_DIR)+(stash64?"v35_place_s64.cpp":(I.compact?"v35_place_compact.cpp":"v35_place.cpp")),I.crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        if(I.dual){ cbf(pp,5,Impl::PCHUNK*src_b,1); cbf(pp,6,Impl::PCHUNK*rec_b,1); cbf(pp,7,I.tb_pg,1); cbf(pp,8,I.plan_pg,1); cbf(pp,9,((4u*I.ntiles*4u+63u)&~63u),1);
            I.kp_n=CreateKernel(pp,std::string(DENSITY_KERNEL_DIR)+"v35_place_s64_n.cpp",I.crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default}); }
        auto pargs=[&](uint32_t src,uint32_t f,uint32_t n){ std::vector<uint32_t> pa{src,f,n,sta,I.geom_pg,I.ntiles,I.W,gra,I.grouped_pg,tba,spa,I.plan_pg,Impl::PCHUNK}; if(I.compact){pa.push_back(I.max_k);pa.push_back(I.max_h);} return pa; };
        for(int c=0;c<nc;++c){uint32_t f=(uint32_t)c*I.cpc_cell,n=std::min(I.cpc_cell,(f<(uint32_t)na)?((uint32_t)na-f):0u);
            if(I.dual){ uint32_t nlo=n/2u;
                SetRuntimeArgs(pp,I.kp_n,I.ccs[c],pargs((uint32_t)c,      f,     nlo));
                SetRuntimeArgs(pp,I.kp,  I.ccs[c],pargs((uint32_t)(c+nc), f+nlo, n-nlo)); }
            else SetRuntimeArgs(pp,I.kp,I.ccs[c],pargs((uint32_t)c,f,n)); }
        I.wl_place.add_program(I.dr,std::move(pp));
        // gather program (args set per-iter)
        Program pg=CreateProgram();
        cbf(pg,0,4096,I.max_k);cbf(pg,1,4096,I.max_h);cbf(pg,2,4096,2);cbf(pg,3,4096,2);cbf(pg,4,4096,2);cbf(pg,5,4096,2);cbf(pg,6,4096,2);cbf(pg,16,4096,2);cbf(pg,17,4096,2);
        const uint32_t fld_bytes=I.band_cols*NBY*4u+(I.max_h+8u)*4u, chunk_bytes=Impl::CHUNK_BATCHES*1024u*rec_b;
        cbf(pg,24,fld_bytes,1); cbf(pg,25,chunk_bytes,1); cbf(pg,26,fld_bytes,1);
        std::vector<UnpackToDestMode> um(64,UnpackToDestMode::Default); for(uint32_t i=0;i<5;++i)um[i]=UnpackToDestMode::UnpackToDestFp32; um[16]=um[17]=UnpackToDestMode::UnpackToDestFp32;
        I.bk=CreateKernel(pg,std::string(DENSITY_KERNEL_DIR)+(I.compact?"v35_gather_brisc_compact.cpp":"v35_gather_brisc.cpp"),I.crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        I.wk=CreateKernel(pg,std::string(DENSITY_KERNEL_DIR)+"v28_ncrisc.cpp",I.crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
        I.ck=CreateKernel(pg,std::string(DENSITY_KERNEL_DIR)+"v28_compute.cpp",I.crs,ComputeConfig{.fp32_dest_acc_en=true,.unpack_to_dest_mode=um,.math_approx_mode=false});
        for(int c=0;c<nc;++c){ SetRuntimeArgs(pg,I.bk,I.ccs[c],std::vector<uint32_t>(18,0u)); SetRuntimeArgs(pg,I.ck,I.ccs[c],{0u,I.max_k,I.max_h}); SetRuntimeArgs(pg,I.wk,I.ccs[c],std::vector<uint32_t>(13,0u)); }
        I.wl_gather.add_program(I.dr,std::move(pg));
        I.built=true;
    }

    // Diagnostic toggles (default = original baseline behavior; flags are opt-in, off in prod).
    //   V35_NOFINISH : Lever-1 A/B — drop the redundant post-write + post-place Finishes
    //                  (FIFO ordering on the single CQ makes them unnecessary for correctness).
    //   V35_FINETIME : split every stage into host-dispatch (Enqueue) vs device-wait (Finish)
    //                  and print it. Forces the per-stage Finishes so attribution is clean.
    const bool NOFIN = getenv("V35_NOFINISH")!=nullptr;
    const bool FINE  = getenv("V35_FINETIME")!=nullptr;
    const bool wrFin = !NOFIN || FINE;   // keep the post-plan-write Finish
    const bool plFin = !NOFIN || FINE;   // keep the post-place Finish

    // ── PASS 1: count ── (enq = host dispatch of the 130-core program; fin = device wait)
    auto tc=hrclock::now(); EnqueueMeshWorkload(*I.cq,I.wl_count,false); double cnt_enq=ms_since(tc);
    auto tcf=hrclock::now(); Finish(*I.cq); double cnt_fin=ms_since(tcf);
    tm.count_ms=cnt_enq+cnt_fin;

    // ── host plan ──
    auto tp=hrclock::now();
    auto trd=hrclock::now(); EnqueueReadMeshBuffer(*I.cq,I.cnts,I.cntb,true); double rd_ms=ms_since(trd);
    const uint32_t rpw=I.rpw, W=I.W, ntiles=I.ntiles;
    auto tpc=hrclock::now();
    std::vector<uint32_t> total(ntiles,0);
    for(uint32_t tl=0;tl<ntiles;++tl){uint32_t acc=0; for(int s=0;s<I.nsrc;++s){I.srcpref[(size_t)s*rpw+tl]=acc; acc+=I.cnts[(size_t)s*rpw+tl];} total[tl]=acc;}
    {uint32_t b=0; for(uint32_t tl=0;tl<ntiles;++tl){I.tile_base[tl]=b; b+=total[tl];}}
    std::vector<uint32_t> alloc(ntiles,0); uint32_t nonempty=0; for(uint32_t tl=0;tl<ntiles;++tl)if(total[tl])++nonempty;
    if(nonempty){uint32_t used=0; for(uint32_t tl=0;tl<ntiles;++tl)if(total[tl]){alloc[tl]=1;++used;}
        while(used<(uint32_t)nc){uint32_t best=ntiles;double bl=-1;for(uint32_t tl=0;tl<ntiles;++tl)if(total[tl]){double l=(double)total[tl]/alloc[tl];if(l>bl){bl=l;best=tl;}}if(best==ntiles)break;alloc[best]++;++used;}}
    uint32_t cur=0,maxn=0;
    for(int c=0;c<nc;++c){I.g_first[c]=0;I.g_n[c]=0;I.g_col0[c]=0;I.g_nb[c]=0;}
    for(uint32_t tl=0;tl<ntiles;++tl){uint32_t a=alloc[tl]; if(!a)continue; uint32_t n=total[tl],col0=tl*W;
        for(uint32_t s=0;s<a && cur<(uint32_t)nc;++s){uint32_t lo=(uint64_t)n*s/a,hi=(uint64_t)n*(s+1)/a; uint32_t sn=hi-lo;
            if(sn>I.mc)sn=I.mc;   // guard worker page (shouldn't trigger with balanced alloc)
            I.g_first[cur]=I.tile_base[tl]+lo; I.g_n[cur]=sn; I.g_col0[cur]=col0; I.g_nb[cur]=(sn+1023)/1024; maxn=std::max(maxn,sn); ++cur;}}
    double planc_ms=ms_since(tpc);
    // write plan to device (FIFO-ordered before place; the Finish is redundant for correctness
    // and only present to bound timing / for the baseline A/B arm).
    auto twr=hrclock::now(); EnqueueWriteMeshBuffer(*I.cq,I.tbb,I.tile_base,false); EnqueueWriteMeshBuffer(*I.cq,I.spb,I.srcpref,false); double wr_enq=ms_since(twr);
    double wr_fin=0; if(wrFin){ auto twf=hrclock::now(); Finish(*I.cq); wr_fin=ms_since(twf); }
    // set gather args for this iter
    auto tsa=hrclock::now();
    Program& pg=I.wl_gather.get_programs().at(I.dr);
    const uint32_t gra=I.grb->address(),gxa=I.gxb->address(),gya=I.gyb->address(),oiba=I.oibb->address();
    for(int c=0;c<nc;++c){uint32_t col0=I.g_col0[c],vcols=(col0<(uint32_t)NBX)?std::min(I.band_cols,(uint32_t)NBX-col0):1u; uint32_t nb=I.g_nb[c];
        SetRuntimeArgs(pg,I.bk,I.ccs[c],{(uint32_t)c,I.g_n[c],I.band_cols,(uint32_t)NBY,I.max_k,I.max_h,fx_addr,fy_addr,I.field_pg,gra,I.grouped_pg,I.g_first[c],nb,Impl::CHUNK_BATCHES,col0,vcols,oiba,I.out_pg});
        SetRuntimeArgs(pg,I.ck,I.ccs[c],{nb,I.max_k,I.max_h});
        SetRuntimeArgs(pg,I.wk,I.ccs[c],{(uint32_t)c,nb,(uint32_t)NBY,I.max_k,I.max_h,fy_addr,I.field_pg,col0,vcols,gxa,gya,I.out_pg,I.band_cols});}
    double sa_ms=ms_since(tsa);
    tm.plan_ms=ms_since(tp);

    // ── PASS 2: place ── (enq = host dispatch; fin = device wait — the place kernel running)
    auto tpl=hrclock::now(); EnqueueMeshWorkload(*I.cq,I.wl_place,false); double pl_enq=ms_since(tpl);
    double pl_fin=0; if(plFin){ auto tpf=hrclock::now(); Finish(*I.cq); pl_fin=ms_since(tpf); }
    tm.place_ms=pl_enq+pl_fin;
    // ── PASS 3: gather ── (sole mandatory barrier: drains place+gather before the blocking d2h reads)
    auto tg=hrclock::now(); EnqueueMeshWorkload(*I.cq,I.wl_gather,false); double g_enq=ms_since(tg);
    auto tgf=hrclock::now(); Finish(*I.cq); double g_fin=ms_since(tgf);
    tm.gather_ms=g_enq+g_fin;

    // ── d2h grad + oidx → unsort grad_full[sel[oidx]] ──
    auto td=hrclock::now();
    // current tt-metal forbids non-blocking enqueue_read_mesh_buffer → all 3 blocking
    auto tr1=hrclock::now(); EnqueueReadMeshBuffer(*I.cq,I.gradx,I.gxb,true); double rdx=ms_since(tr1);
    auto tr2=hrclock::now(); EnqueueReadMeshBuffer(*I.cq,I.grady,I.gyb,true); double rdy=ms_since(tr2);
    auto tr3=hrclock::now(); EnqueueReadMeshBuffer(*I.cq,I.oidx_host,I.oibb,true); double rdo=ms_since(tr3);
    // sub-tiled big cells → multiple sub-cells share a node → ACCUMULATE. zero the
    // active node grads first (caller pre-zeros fixed nodes), then sum sub-cell grads.
    auto tus=hrclock::now();
    for(int i=0;i<na;++i){ int g=I.sel[i]; grad[g]=0.f; grad[nn+g]=0.f; }
    // grad is exact on-device now: stash word5 = ratio_c·bin_area (V31_EF_GEOM), so the
    // gather already produced ratio_c·Σ(lx·ly)·f = DREAMPlace's force. No host ratio scale.
    for(int c=0;c<nc;++c){ uint32_t n=I.g_n[c]; const size_t cb=(size_t)c*I.mc;
        for(uint32_t s=0;s<n;++s){ uint32_t oi=I.oidx_host[cb+s]; if(oi==0xffffffffu||(int)oi>=na)continue; int g=I.sel[oi]; grad[g]+=I.gradx[cb+s]; grad[nn+g]+=I.grady[cb+s]; } }
    double unsort_ms=ms_since(tus);
    tm.d2h_ms=ms_since(td);
    tm.total_ms=ms_since(t0); if(t)*t=tm;
    if(FINE){ static uint32_t fn=0; if(fn++<400) fprintf(stderr,
        "[v35_fine] cnt[enq=%.3f fin=%.3f] rd_cnts=%.3f planc=%.3f wr[enq=%.3f fin=%.3f] setargs=%.3f place[enq=%.3f fin=%.3f] gather[enq=%.3f fin=%.3f] d2h[rdx=%.3f rdy=%.3f rdo=%.3f unsort=%.3f] | count=%.3f plan=%.3f place=%.3f gather=%.3f d2h=%.3f TOTAL=%.3f\n",
        cnt_enq,cnt_fin,rd_ms,planc_ms,wr_enq,wr_fin,sa_ms,pl_enq,pl_fin,g_enq,g_fin,rdx,rdy,rdo,unsort_ms,
        tm.count_ms,tm.plan_ms,tm.place_ms,tm.gather_ms,tm.d2h_ms,tm.total_ms); }
    if(getenv("V35_DROP")){ uint64_t asg=0; uint32_t mxn=0; for(int c=0;c<nc;++c){asg+=I.g_n[c]; if(I.g_n[c]>mxn)mxn=I.g_n[c];}
        if(I.diag_n2++ < 2 || asg<(uint64_t)na) fprintf(stderr,"[v35_drop] na=%d assigned=%llu DROPPED=%lld maxn=%u mc=%u ntiles=%u nc=%d\n",
            na,(unsigned long long)asg,(long long)((int64_t)na-(int64_t)asg),mxn,I.mc,I.ntiles,nc); }
    if(getenv("V35_DIAG") && I.diag_n++ < 3) fprintf(stderr,"[v35_diag] na=%d nc=%d W=%u ntiles=%u max_k=%u max_h=%u mc=%u maxn=%u | count=%.3f plan=%.3f place=%.3f gather=%.3f d2h=%.3f total=%.3f\n",
        na,nc,I.W,I.ntiles,I.max_k,I.max_h,I.mc,maxn,tm.count_ms,tm.plan_ms,tm.place_ms,tm.gather_ms,tm.d2h_ms,tm.total_ms);
}
}  // namespace v35ef
