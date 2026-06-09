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
#include <cstdlib>
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
    bool built=false; bool compact=false; bool dual=false; bool bucket=false; int nsrc=0; uint32_t W=1, ntiles=1, band_cols=1, cpc_cell=1, mc=1024;
    uint32_t K0=4, H0=4, ngroups=1;
    uint32_t cnt_pg=0, tb_pg=0, plan_pg=0, grouped_pg=0, out_pg=0, field_pg=0, rpw=0;
    static constexpr uint32_t PCHUNK=1024, CHUNK_BATCHES=4;
    std::shared_ptr<MeshBuffer> cntb, tbb, spb, grb, gxb, gyb, oibb;
    MeshWorkload wl_count, wl_place, wl_gather; MeshCoordinateRange dr{MeshCoordinate{0,0},MeshCoordinate{0,0}};
    KernelHandle kc=0, kp=0, bk=0, ck=0, wk=0, kc_n=0, kp_n=0;
    uint32_t diag_n=0; uint32_t diag_n2=0;
    // host staging
    std::vector<uint32_t> cnts, tile_base, srcpref, oidx_host;
    std::vector<float> gradx, grady;
    std::vector<uint32_t> g_first, g_n, g_col0, g_nb, g_nsmall;
    // ── on-chip unsort -> V22 interleaved b_dg (gated V35_ONCHIP_UNSORT; validated vs CPU unsort) ──
    // node=sel[oidx] resolved ON DEVICE: sel uploaded once (page=64B, aligned gather), oidx read live.
    std::shared_ptr<MeshBuffer> bdgb, selb, coordbuf;
    MeshWorkload wl_unsort; KernelHandle ku=0; bool built_unsort=false;
    uint32_t epc=0, tpc_bdg=0, ntiles_bdg=0, sel_pad=0, maxsub=1;
    std::vector<uint32_t> coord_v;
    // No-host export: flush the on-chip unsort into an external b_dg (V22's) instead
    // of the internal validation buffer; optionally skip the host CPU unsort entirely.
    uint32_t ext_bdg_addr=0, ext_bdg_ntiles=0; bool skip_cpu_unsort=false;

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

void V35EFEngine::set_bdg_export(uint32_t addr,uint32_t ntiles,bool skip_cpu_unsort){
    impl_->ext_bdg_addr=addr; impl_->ext_bdg_ntiles=ntiles; impl_->skip_cpu_unsort=skip_cpu_unsort;
}

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
        // COMPACT GROUPED RECORD: the gather consumes only gidx/bxl_g/byl/ratio/px[max_k]/py[max_h]
        // (never kc/hc, never px/py past max_k/max_h), so a 64 B (16 u32) record holds all of it when
        // 4+max_k+max_h ≤ 16 → halves place's write + the gather's record read (validated bit-exact in
        // the v35compact harness: place -20..29%, gather -6..12% at bigblue3). Falls back to the 128 B
        // verbatim record if it doesn't fit or V35_NO_COMPACT is set.
        I.compact = (getenv("V35_NO_COMPACT")==nullptr) && ((4u+I.max_k+I.max_h)<=16u);
        const uint32_t rec_b = I.compact ? 64u : 128u;
        // ── Best-path optimizations are ON BY DEFAULT (all validated bit-exact, fastest). Opt out
        //    per-stage: V35_STASH64=0 / V35_DUAL=0 / V35_BUCKET=0, or V35_NO_COMPACT for the lot. ──
        auto env_on=[](const char* k){ const char* v=getenv(k); return v==nullptr || atoi(v)!=0; };  // default ON unless =0
        // V35_STASH64: the FORWARD wrote a 64 B source stash (px[8]+py[4]) → count/place read 64 B/cell
        // instead of 128 B (forward+backward co-design; halves the DRAM read that bounds count+place).
        // Valid when max_h<=4 (forward drops py[4..7]); the forward defaults to the matching 64 B stash.
        const bool stash64 = env_on("V35_STASH64") && I.compact;
        if(stash64 && I.max_h>4u) fprintf(stderr,"[v35] WARNING: 64 B stash with max_h=%u>4 — forward dropped py[4..7], grad WILL be wrong (set V35_STASH64=0)\n", I.max_h);
        // V35_DUAL: split each core into TWO sources (NCRISC half = src c, BRISC half = src c+nc) so
        // count+place run on BOTH data-movement engines (reclaims the idle NCRISC). Requires stash64.
        const uint32_t src_b = stash64 ? 64u : 128u;
        I.dual = env_on("V35_DUAL") && stash64;
        I.nsrc = I.dual ? 2*nc : nc;
        // V35_BUCKET (lever C): count/place group cells by (tile,bucket) where bucket = large iff
        // any px[k]!=0 (k>=K0) or py[h]!=0 (h>=H0); small cells are laid out FIRST within each tile.
        // The gather (pbf kernels) then runs a PER-BATCH footprint — pure-small 1024-batches loop
        // K0*H0 field reads instead of max_k*max_h. Bit-exact (small cell px[k>=K0]=py[h>=H0]=0 → the
        // dropped (k,h) terms are +0). Worker assignment stays by TILE so schedule balance is unchanged.
        // Requires the stash64 dual path (folds into v35_*_s64_bucket(_n)). ngroups = 2*ntiles.
        I.bucket = env_on("V35_BUCKET") && stash64 && I.dual;
        I.K0=getenv("V35_K0")?(uint32_t)atoi(getenv("V35_K0")):4u; if(I.K0>I.max_k)I.K0=I.max_k;
        I.H0=getenv("V35_H0")?(uint32_t)atoi(getenv("V35_H0")):4u; if(I.H0>I.max_h)I.H0=I.max_h;
        I.ngroups = I.bucket ? 2u*I.ntiles : I.ntiles;
        I.cnt_pg=((I.ngroups*4u+63u)&~63u); I.tb_pg=I.cnt_pg; I.plan_pg=I.cnt_pg; I.rpw=I.cnt_pg/4u;
        fprintf(stderr,"[v35] grouped=%uB(%s) source=%uB(%s) dual_risc=%s bucket=%s(K0=%u H0=%u) max_k=%u max_h=%u na=%d nsrc=%d\n",
                rec_b, I.compact?"COMPACT":"verbatim", src_b, stash64?"64B-codesign":"128B", I.dual?"YES":"no", I.bucket?"YES":"no", I.K0,I.H0, I.max_k, I.max_h, na, I.nsrc);
        // field is page-interleaved: ONE page per x-bin = NBY floats (matches the
        // TTNN DCT output / V21's chip-field read). NOT a single NBX*NBY page.
        I.grouped_pg=(uint32_t)((uint64_t)na*rec_b+rec_b); I.out_pg=I.mc*4u; I.field_pg=(uint32_t)((uint64_t)NBY*4u);
        auto mb=[&](uint64_t b,uint32_t pg){DeviceLocalBufferConfig c{.page_size=pg,.buffer_type=BufferType::DRAM};ReplicatedBufferConfig r{.size=b};return MeshBuffer::create(r,c,I.dev);};
        I.cntb=mb((uint64_t)I.nsrc*I.cnt_pg,I.cnt_pg); I.tbb=mb(I.tb_pg,I.tb_pg); I.spb=mb((uint64_t)I.nsrc*I.plan_pg,I.plan_pg);
        I.grb=mb(I.grouped_pg,I.grouped_pg); I.gxb=mb((uint64_t)nc*I.out_pg,I.out_pg); I.gyb=mb((uint64_t)nc*I.out_pg,I.out_pg); I.oibb=mb((uint64_t)nc*I.out_pg,I.out_pg);
        I.cnts.assign((size_t)I.nsrc*I.rpw,0u); I.tile_base.assign(I.tb_pg/4u,0u); I.srcpref.assign((size_t)I.nsrc*I.rpw,0u);
        I.gradx.assign((size_t)nc*I.mc,0.f); I.grady.assign((size_t)nc*I.mc,0.f); I.oidx_host.assign((size_t)nc*I.mc,0u);
        I.g_first.assign(nc,0); I.g_n.assign(nc,0); I.g_col0.assign(nc,0); I.g_nb.assign(nc,0); I.g_nsmall.assign(nc,0);
        const uint32_t sta=I.geom_addr, gra=I.grb->address(), cnta=I.cntb->address(), tba=I.tbb->address(), spa=I.spb->address();
        I.dr=MeshCoordinateRange{I.dev->shape()};
        // count program — dual-RISC (BRISC src c+nc, NCRISC src c) when I.dual; else BRISC-only.
        Program pc=CreateProgram(); cbf(pc,0,Impl::PCHUNK*src_b,1); cbf(pc,1,I.cnt_pg,1);
        const std::string countk = stash64 ? (I.bucket?"v35_count64_bucket.cpp":"v35_count64.cpp") : "v35_count.cpp";
        const std::string countk_n = I.bucket?"v35_count64_bucket_n.cpp":"v35_count64_n.cpp";
        I.kc=CreateKernel(pc,std::string(DENSITY_KERNEL_DIR)+countk,I.crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        if(I.dual){ cbf(pc,5,Impl::PCHUNK*src_b,1); cbf(pc,6,I.cnt_pg,1);
            I.kc_n=CreateKernel(pc,std::string(DENSITY_KERNEL_DIR)+countk_n,I.crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default}); }
        auto cargs=[&](uint32_t src,uint32_t f,uint32_t n){ std::vector<uint32_t> ca{src,f,n,sta,I.geom_pg,I.ntiles,I.W,cnta,I.cnt_pg,Impl::PCHUNK};
            if(I.bucket){ca.push_back(I.K0);ca.push_back(I.H0);ca.push_back(I.max_k);ca.push_back(I.max_h);} return ca; };
        for(int c=0;c<nc;++c){uint32_t f=(uint32_t)c*I.cpc_cell,n=std::min(I.cpc_cell,(f<(uint32_t)na)?((uint32_t)na-f):0u);
            if(I.dual){ uint32_t nlo=n/2u;
                SetRuntimeArgs(pc,I.kc_n,I.ccs[c],cargs((uint32_t)c,      f,     nlo));
                SetRuntimeArgs(pc,I.kc,  I.ccs[c],cargs((uint32_t)(c+nc), f+nlo, n-nlo)); }
            else SetRuntimeArgs(pc,I.kc,I.ccs[c],cargs((uint32_t)c,f,n)); }
        I.wl_count.add_program(I.dr,std::move(pc));
        // place program — dual-RISC (BRISC src c+nc, NCRISC src c) when I.dual; CB_SORT holds compact 64 B.
        const uint32_t scr_pg = ((4u*I.ngroups*4u+63u)&~63u);
        Program pp=CreateProgram(); cbf(pp,0,Impl::PCHUNK*src_b,1); cbf(pp,1,Impl::PCHUNK*rec_b,1); cbf(pp,2,I.tb_pg,1); cbf(pp,3,I.plan_pg,1); cbf(pp,4,scr_pg,1);
        const std::string placek = stash64 ? (I.bucket?"v35_place_s64_bucket.cpp":"v35_place_s64.cpp") : (I.compact?"v35_place_compact.cpp":"v35_place.cpp");
        I.kp=CreateKernel(pp,std::string(DENSITY_KERNEL_DIR)+placek,I.crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        if(I.dual){ cbf(pp,5,Impl::PCHUNK*src_b,1); cbf(pp,6,Impl::PCHUNK*rec_b,1); cbf(pp,7,I.tb_pg,1); cbf(pp,8,I.plan_pg,1); cbf(pp,9,scr_pg,1);
            I.kp_n=CreateKernel(pp,std::string(DENSITY_KERNEL_DIR)+(I.bucket?"v35_place_s64_bucket_n.cpp":"v35_place_s64_n.cpp"),I.crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default}); }
        auto pargs=[&](uint32_t src,uint32_t f,uint32_t n){ std::vector<uint32_t> pa{src,f,n,sta,I.geom_pg,I.ntiles,I.W,gra,I.grouped_pg,tba,spa,I.plan_pg,Impl::PCHUNK}; if(I.compact){pa.push_back(I.max_k);pa.push_back(I.max_h);} if(I.bucket){pa.push_back(I.K0);pa.push_back(I.H0);} return pa; };
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
        I.bk=CreateKernel(pg,std::string(DENSITY_KERNEL_DIR)+(I.bucket?"v35_gather_brisc_pbf.cpp":(I.compact?"v35_gather_brisc_compact.cpp":"v35_gather_brisc.cpp")),I.crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        I.wk=CreateKernel(pg,std::string(DENSITY_KERNEL_DIR)+(I.bucket?"v28_ncrisc_pbf.cpp":"v28_ncrisc.cpp"),I.crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
        I.ck=CreateKernel(pg,std::string(DENSITY_KERNEL_DIR)+(I.bucket?"v28_compute_pbf.cpp":"v28_compute.cpp"),I.crs,ComputeConfig{.fp32_dest_acc_en=true,.unpack_to_dest_mode=um,.math_approx_mode=false});
        const uint32_t bk_args=I.bucket?21u:18u, ck_args=I.bucket?6u:3u, wk_args=I.bucket?16u:13u;
        for(int c=0;c<nc;++c){ SetRuntimeArgs(pg,I.bk,I.ccs[c],std::vector<uint32_t>(bk_args,0u)); SetRuntimeArgs(pg,I.ck,I.ccs[c],std::vector<uint32_t>(ck_args,0u)); SetRuntimeArgs(pg,I.wk,I.ccs[c],std::vector<uint32_t>(wk_args,0u)); }
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
    uint32_t cur=0,maxn=0;
    for(int c=0;c<nc;++c){I.g_first[c]=0;I.g_n[c]=0;I.g_col0[c]=0;I.g_nb[c]=0;I.g_nsmall[c]=0;}
    if(I.bucket){
        // group totals (g=2*tile+bucket) + srcprefix; tile_base[] holds GROUP_BASE (ngroups entries).
        const uint32_t ng=I.ngroups;
        std::vector<uint32_t> gtot(ng,0);
        for(uint32_t g=0;g<ng;++g){uint32_t acc=0; for(int s=0;s<I.nsrc;++s){I.srcpref[(size_t)s*rpw+g]=acc; acc+=I.cnts[(size_t)s*rpw+g];} gtot[g]=acc;}
        {uint32_t b=0; for(uint32_t g=0;g<ng;++g){I.tile_base[g]=b; b+=gtot[g];}}
        // per-tile combined (small group 2t + large group 2t+1 are contiguous → tile run is small-first).
        std::vector<uint32_t> ttot(ntiles,0),tsmall(ntiles,0),tstart(ntiles,0);
        for(uint32_t tl=0;tl<ntiles;++tl){tsmall[tl]=gtot[2u*tl]; ttot[tl]=gtot[2u*tl]+gtot[2u*tl+1u]; tstart[tl]=I.tile_base[2u*tl];}
        // worker allocation by TILE (combined) — identical balance to the non-bucket path.
        std::vector<uint32_t> alloc(ntiles,0); uint32_t nonempty=0; for(uint32_t tl=0;tl<ntiles;++tl)if(ttot[tl])++nonempty;
        if(nonempty){uint32_t used=0; for(uint32_t tl=0;tl<ntiles;++tl)if(ttot[tl]){alloc[tl]=1;++used;}
            while(used<(uint32_t)nc){uint32_t best=ntiles;double bl=-1;for(uint32_t tl=0;tl<ntiles;++tl)if(ttot[tl]){double l=(double)ttot[tl]/alloc[tl];if(l>bl){bl=l;best=tl;}}if(best==ntiles)break;alloc[best]++;++used;}}
        for(uint32_t tl=0;tl<ntiles;++tl){uint32_t a=alloc[tl]; if(!a)continue; uint32_t n=ttot[tl],col0=tl*W;
            for(uint32_t s=0;s<a && cur<(uint32_t)nc;++s){uint32_t lo=(uint64_t)n*s/a,hi=(uint64_t)n*(s+1)/a; uint32_t sn=hi-lo;
                if(sn>I.mc)sn=I.mc;
                uint32_t ns=(tsmall[tl]>lo)?(tsmall[tl]-lo):0u; if(ns>sn)ns=sn;   // # small cells at front of this slice
                I.g_first[cur]=tstart[tl]+lo; I.g_n[cur]=sn; I.g_col0[cur]=col0; I.g_nb[cur]=(sn+1023)/1024; I.g_nsmall[cur]=ns; maxn=std::max(maxn,sn); ++cur;}}
    } else {
        std::vector<uint32_t> total(ntiles,0);
        for(uint32_t tl=0;tl<ntiles;++tl){uint32_t acc=0; for(int s=0;s<I.nsrc;++s){I.srcpref[(size_t)s*rpw+tl]=acc; acc+=I.cnts[(size_t)s*rpw+tl];} total[tl]=acc;}
        {uint32_t b=0; for(uint32_t tl=0;tl<ntiles;++tl){I.tile_base[tl]=b; b+=total[tl];}}
        std::vector<uint32_t> alloc(ntiles,0); uint32_t nonempty=0; for(uint32_t tl=0;tl<ntiles;++tl)if(total[tl])++nonempty;
        if(nonempty){uint32_t used=0; for(uint32_t tl=0;tl<ntiles;++tl)if(total[tl]){alloc[tl]=1;++used;}
            while(used<(uint32_t)nc){uint32_t best=ntiles;double bl=-1;for(uint32_t tl=0;tl<ntiles;++tl)if(total[tl]){double l=(double)total[tl]/alloc[tl];if(l>bl){bl=l;best=tl;}}if(best==ntiles)break;alloc[best]++;++used;}}
        for(uint32_t tl=0;tl<ntiles;++tl){uint32_t a=alloc[tl]; if(!a)continue; uint32_t n=total[tl],col0=tl*W;
            for(uint32_t s=0;s<a && cur<(uint32_t)nc;++s){uint32_t lo=(uint64_t)n*s/a,hi=(uint64_t)n*(s+1)/a; uint32_t sn=hi-lo;
                if(sn>I.mc)sn=I.mc;   // guard worker page (shouldn't trigger with balanced alloc)
                I.g_first[cur]=I.tile_base[tl]+lo; I.g_n[cur]=sn; I.g_col0[cur]=col0; I.g_nb[cur]=(sn+1023)/1024; maxn=std::max(maxn,sn); ++cur;}}
    }
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
        if(I.bucket){ uint32_t ns=I.g_nsmall[c];
            SetRuntimeArgs(pg,I.bk,I.ccs[c],{(uint32_t)c,I.g_n[c],I.band_cols,(uint32_t)NBY,I.max_k,I.max_h,fx_addr,fy_addr,I.field_pg,gra,I.grouped_pg,I.g_first[c],nb,Impl::CHUNK_BATCHES,col0,vcols,oiba,I.out_pg,ns,I.K0,I.H0});
            SetRuntimeArgs(pg,I.ck,I.ccs[c],{nb,I.max_k,I.max_h,ns,I.K0,I.H0});
            SetRuntimeArgs(pg,I.wk,I.ccs[c],{(uint32_t)c,nb,(uint32_t)NBY,I.max_k,I.max_h,fy_addr,I.field_pg,col0,vcols,gxa,gya,I.out_pg,I.band_cols,ns,I.K0,I.H0}); }
        else {
            SetRuntimeArgs(pg,I.bk,I.ccs[c],{(uint32_t)c,I.g_n[c],I.band_cols,(uint32_t)NBY,I.max_k,I.max_h,fx_addr,fy_addr,I.field_pg,gra,I.grouped_pg,I.g_first[c],nb,Impl::CHUNK_BATCHES,col0,vcols,oiba,I.out_pg});
            SetRuntimeArgs(pg,I.ck,I.ccs[c],{nb,I.max_k,I.max_h});
            SetRuntimeArgs(pg,I.wk,I.ccs[c],{(uint32_t)c,nb,(uint32_t)NBY,I.max_k,I.max_h,fy_addr,I.field_pg,col0,vcols,gxa,gya,I.out_pg,I.band_cols}); } }
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

    // ── d2h grad + oidx → unsort grad_full[sel[oidx]] (the host CPU unsort) ──
    // No-host mode (skip_cpu_unsort) elides this entirely — the gradient stays on
    // device in the external b_dg (written by the on-chip unsort block below).
    auto td=hrclock::now();
    double rdx=0,rdy=0,rdo=0,unsort_ms=0;
    if(!I.skip_cpu_unsort){
        // current tt-metal forbids non-blocking enqueue_read_mesh_buffer → all 3 blocking
        auto tr1=hrclock::now(); EnqueueReadMeshBuffer(*I.cq,I.gradx,I.gxb,true); rdx=ms_since(tr1);
        auto tr2=hrclock::now(); EnqueueReadMeshBuffer(*I.cq,I.grady,I.gyb,true); rdy=ms_since(tr2);
        auto tr3=hrclock::now(); EnqueueReadMeshBuffer(*I.cq,I.oidx_host,I.oibb,true); rdo=ms_since(tr3);
        // sub-tiled big cells → multiple sub-cells share a node → ACCUMULATE. zero the
        // active node grads first (caller pre-zeros fixed nodes), then sum sub-cell grads.
        auto tus=hrclock::now();
        for(int i=0;i<na;++i){ int g=I.sel[i]; grad[g]=0.f; grad[nn+g]=0.f; }
        // grad is exact on-device now: stash word5 = ratio_c·bin_area (V31_EF_GEOM), so the
        // gather already produced ratio_c·Σ(lx·ly)·f = DREAMPlace's force. No host ratio scale.
        for(int c=0;c<nc;++c){ uint32_t n=I.g_n[c]; const size_t cb=(size_t)c*I.mc;
            for(uint32_t s=0;s<n;++s){ uint32_t oi=I.oidx_host[cb+s]; if(oi==0xffffffffu||(int)oi>=na)continue; int g=I.sel[oi]; grad[g]+=I.gradx[cb+s]; grad[nn+g]+=I.grady[cb+s]; } }
        unsort_ms=ms_since(tus);
    }
    // COLLISION DIAGNOSTIC (V35_COLLIDE=1, one-shot): how many active nodes receive >1 slot
    // (= band-spanning sub-tiled cells that the += accumulate must sum). Sizes the on-chip
    // unsort's accumulate need. Cross-references multi-slot nodes with their (k,h) footprint.
    if(getenv("V35_COLLIDE")){ static bool cdone=false; if(!cdone){ cdone=true;
        std::vector<uint32_t> spo((size_t)na,0u); uint64_t totv=0;
        // collision-structure: per oi track band set (min band, parity bitmask, all-adjacent?)
        std::vector<uint32_t> bmin((size_t)na,0xffffffffu),bmax((size_t)na,0u); std::vector<uint8_t> parm((size_t)na,0u);
        for(int c=0;c<nc;++c){ uint32_t n=I.g_n[c]; const size_t cb=(size_t)c*I.mc; uint32_t band=I.W?(I.g_col0[c]/I.W):0u;
            for(uint32_t s=0;s<n;++s){ uint32_t oi=I.oidx_host[cb+s]; if(oi==0xffffffffu||(int)oi>=na)continue; spo[oi]++; totv++;
                if(band<bmin[oi])bmin[oi]=band; if(band>bmax[oi])bmax[oi]=band; parm[oi]|=(uint8_t)(1u<<(band&1u)); } }
        uint32_t multi=0,maxs=0,hist[17]={0}; uint64_t mslots=0;
        uint32_t par_distinct=0,par_same=0,span_gt1=0,same_band=0;   // among colliding nodes
        for(int i=0;i<na;++i){ if(spo[i]>1){ if(parm[i]==3u)par_distinct++; else par_same++;
            uint32_t span=bmax[i]-bmin[i]; if(span==0)same_band++; if(span>1)span_gt1++; } }
        for(int i=0;i<na;++i){ uint32_t s=spo[i]; if(s>maxs)maxs=s; hist[s<16?s:16]++; if(s>1){multi++;mslots+=s;} }
        fprintf(stderr,"[v35_collide2] colliding-node band structure: parity_distinct=%u parity_SAME=%u (same_band=%u band_span>1=%u) — parity sub-slot only works if parity_distinct==multi & span==1\n",
            par_distinct,par_same,same_band,span_gt1);
        // of the multi-slot nodes, how many also exceed the (k,h) window (k>max_k or h>max_h)?
        uint32_t multi_overk=0,multi_overh=0,zero_slot=0;
        for(int i=0;i<na;++i){ float ow=I.nsx[i]+2.f*std::fabs(I.ox[i]),oh=I.nsy[i]+2.f*std::fabs(I.oy[i]);
            uint32_t k=(uint32_t)std::floor(ow*I.inv_bsx)+2u,h=(uint32_t)std::floor(oh*I.inv_bsy)+2u;
            if(spo[i]==0)zero_slot++; if(spo[i]>1){ if(k>I.max_k)multi_overk++; if(h>I.max_h)multi_overh++; } }
        fprintf(stderr,"[v35_collide] na=%d total_valid_slots=%llu (%.4f slots/node) multi_slot_nodes=%u (%.4f%%) max_slots/node=%u zero_slot_nodes=%u\n",
            na,(unsigned long long)totv,(double)totv/(na>0?na:1),multi,100.0*multi/(na>0?na:1),maxs,zero_slot);
        fprintf(stderr,"  of multi-slot nodes: k>max_k(%u)=%u  h>max_h(%u)=%u   slots/node hist:",I.max_k,multi_overk,I.max_h,multi_overh);
        for(uint32_t s=0;s<=16;++s) if(hist[s]) fprintf(stderr," %u%s:%u",s,(s==16?"+":""),hist[s]); fprintf(stderr,"\n"); } }
    // ── ON-CHIP UNSORT (V35_ONCHIP_UNSORT env, or ext_bdg_addr set for no-host): scatter
    //    the gather output into an interleaved b_dg on device. When ext_bdg_addr is set the
    //    flush targets the EXTERNAL (V22) b_dg and the host validation is skipped. ──
    if(getenv("V35_ONCHIP_UNSORT") || I.ext_bdg_addr){
        using tt::CBIndex;
        const uint32_t two_nn=2u*(uint32_t)nn;
        if(!I.built_unsort){
            I.epc=((((uint32_t)nn + (uint32_t)nc - 1u)/(uint32_t)nc) + 511u)&~511u;   // nodes/core, mult of 512
            I.tpc_bdg=I.epc/512u; I.ntiles_bdg=(two_nn+1023u)/1024u; I.sel_pad=(((uint32_t)na+15u)&~15u);  // sel rounded to 16/page(64B)
            // sel is NON-injective (big cells split into consecutive active entries sharing a node).
            // Per-oi sub-index = rank among same-node entries (duplicates are CONSECUTIVE). Pack into
            // the resident sel value: sel_u32[oi] = (sub<<24)|node  -> the kernel routes each partial to
            // shard[node*maxsub + sub] (distinct sub-slots, no race) and the flush SUMS them. Constant -> once.
            std::vector<uint32_t> sel_u32((size_t)I.sel_pad,0u); { uint32_t mx=1,sub=0;
                for(int i=0;i<na;++i){ if(i>0 && I.sel[i]==I.sel[i-1]) ++sub; else sub=0; if(sub+1>mx)mx=sub+1;
                    sel_u32[i]=((sub&0xffu)<<24)|((uint32_t)I.sel[i]&0xffffffu); } I.maxsub=mx; }
            auto mbu=[&](uint64_t b,uint32_t pg){DeviceLocalBufferConfig c{.page_size=pg,.buffer_type=BufferType::DRAM};ReplicatedBufferConfig r{.size=b};return MeshBuffer::create(r,c,I.dev);};
            if(!I.ext_bdg_addr) I.bdgb=mbu((uint64_t)I.ntiles_bdg*4096u,4096u);   // internal buffer only when not exporting
            I.selb=mbu((uint64_t)I.sel_pad*4u,64u);   // page=64B (16 entries) -> sel[oi] is a 64B-aligned read
            I.coordbuf=mbu((uint64_t)nc*4u,(uint32_t)(nc*4u));
            I.coord_v.resize(nc); for(int c=0;c<nc;++c){auto p=I.dev->get_devices()[0]->worker_core_from_logical_core(I.ccs[c]); I.coord_v[c]=((uint32_t)p.x<<16)|(uint32_t)p.y;}
            EnqueueWriteMeshBuffer(*I.cq,I.coordbuf,I.coord_v,false);
            EnqueueWriteMeshBuffer(*I.cq,I.selb,sel_u32,false);   // packed (sub<<24|node), constant across iters
            auto cbu=[&](Program&p,uint32_t idx,uint32_t bytes){CircularBufferConfig c(bytes,{{(CBIndex)idx,DataFormat::Float32}});c.set_page_size((CBIndex)idx,bytes);CreateCircularBuffer(p,I.crs,c);};
            Program pu=CreateProgram();
            cbu(pu,0,I.epc*I.maxsub*16u+64u); cbu(pu,1,(uint32_t)(nc*4u)+64u); cbu(pu,2,I.mc*4u+64u);  // c_0: maxsub sub-slots/node
            cbu(pu,3,I.mc*4u+64u); cbu(pu,4,I.mc*4u+64u); cbu(pu,5,64u); cbu(pu,6,I.tpc_bdg*4096u+64u);
            cbu(pu,7,256u*64u+64u);   // CB_SEL: CHUNK(256)*64B sel-read scratch
            I.ku=CreateKernel(pu,std::string(DENSITY_KERNEL_DIR)+"v35_onchip_unsort.cpp",I.crs,
                DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
            I.wl_unsort.add_program(I.dr,std::move(pu));
            I.built_unsort=true;
            fprintf(stderr,"[v35_onchip_unsort] built: epc=%u tpc_bdg=%u ntiles_bdg=%u sel_pad=%u maxsub=%u shard=%.0fKB (b_dg=%.1fMB, on-device sel+accumulate)\n",
                I.epc,I.tpc_bdg,I.ntiles_bdg,I.sel_pad,I.maxsub,(double)I.epc*I.maxsub*16.0/1024.0,(double)I.ntiles_bdg*4096.0/1048576.0);
        }
        Program& pu=I.wl_unsort.get_programs().at(I.dr);
        const uint32_t bdga = I.ext_bdg_addr ? I.ext_bdg_addr : (uint32_t)I.bdgb->address();   // flush target: external (V22) or internal
        const uint32_t gxa2=I.gxb->address(),gya2=I.gyb->address(),oia2=I.oibb->address(),coorda=I.coordbuf->address(),sela=I.selb->address();
        auto sau=[&](uint32_t ph){ for(int c=0;c<nc;++c){ uint32_t t0=(uint32_t)c*I.tpc_bdg, nt=(t0<I.ntiles_bdg)?std::min(I.tpc_bdg,I.ntiles_bdg-t0):0u;
            SetRuntimeArgs(pu,I.ku,I.ccs[c],{ph,(uint32_t)c,I.g_n[c],gxa2,gya2,oia2,I.out_pg,coorda,(uint32_t)(nc*4u),(uint32_t)nc,I.epc,two_nn,t0,nt,bdga,4096u,(uint32_t)na,sela,I.maxsub}); } };
        sau(2u); EnqueueMeshWorkload(*I.cq,I.wl_unsort,false); Finish(*I.cq);   // zero shards
        sau(0u); EnqueueMeshWorkload(*I.cq,I.wl_unsort,false); Finish(*I.cq);   // scatter
        sau(1u); EnqueueMeshWorkload(*I.cq,I.wl_unsort,false); Finish(*I.cq);   // flush -> b_dg (external V22 buffer if exporting)
        if(!I.ext_bdg_addr){   // validation only against the internal buffer + the host CPU unsort
            std::vector<float> bdg((size_t)I.ntiles_bdg*1024u,0.f); EnqueueReadMeshBuffer(*I.cq,bdg,I.bdgb,true);
            size_t bad=0; double maxabs=0; int shown=0;
            for(int i=0;i<na;++i){ int g=I.sel[i];
                double dx=std::fabs((double)bdg[2u*g]-(double)grad[g]), dy=std::fabs((double)bdg[2u*g+1u]-(double)grad[nn+g]);
                if(dx>maxabs)maxabs=dx; if(dy>maxabs)maxabs=dy;
                if(dx>1e-6||dy>1e-6){ bad++;
                    if(getenv("V35_ONCHIP_DBG")&&shown<10){ float ow=I.nsx[i]+2.f*std::fabs(I.ox[i]),oh=I.nsy[i]+2.f*std::fabs(I.oy[i]);
                        uint32_t k=(uint32_t)std::floor(ow*I.inv_bsx)+2u,h=(uint32_t)std::floor(oh*I.inv_bsy)+2u;
                        fprintf(stderr,"  [ocu_dbg] i=%d g=%d k=%u h=%u exp=(%.4e,%.4e) got=(%.4e,%.4e)\n",
                            i,g,k,h,grad[g],grad[nn+g],bdg[2u*g],bdg[2u*g+1u]); ++shown; } } }
            fprintf(stderr,"[v35_onchip_unsort] vs CPU unsort: mismatches=%zu/%d max_abs=%.3e %s\n",bad,na,maxabs,(bad==0)?"OK":"<<< WRONG");
        }
    }
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
