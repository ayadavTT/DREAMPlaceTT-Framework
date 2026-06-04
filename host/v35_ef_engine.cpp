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
    std::vector<float> ox,oy,nsx,nsy; uint32_t max_k=1, max_h=1;
    uint32_t geom_addr=0, geom_pg=0;
    // sizing (first compute)
    bool built=false; uint32_t W=1, ntiles=1, band_cols=1, cpc_cell=1, mc=1024;
    uint32_t cnt_pg=0, tb_pg=0, plan_pg=0, grouped_pg=0, out_pg=0, field_pg=0, rpw=0;
    static constexpr uint32_t PCHUNK=1024, CHUNK_BATCHES=4;
    std::shared_ptr<MeshBuffer> cntb, tbb, spb, grb, gxb, gyb, oibb;
    MeshWorkload wl_count, wl_place, wl_gather; MeshCoordinateRange dr{MeshCoordinate{0,0},MeshCoordinate{0,0}};
    KernelHandle kc=0, kp=0, bk=0, ck=0, wk=0;
    uint32_t diag_n=0;
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
        const float* nsy_f,const float* /*ratio_f*/,const int32_t* sel,int na,int nt){
    auto&I=*impl_; I.n_active=na; I.n_total=nt; I.sel.assign(sel,sel+na);
    I.ox.resize(na);I.oy.resize(na);I.nsx.resize(na);I.nsy.resize(na);
    for(int i=0;i<na;++i){int g=sel[i];I.ox[i]=ox_f[g];I.oy[i]=oy_f[g];I.nsx[i]=nsx_f[g];I.nsy[i]=nsy_f[g];}
    // max_k/max_h from ORIG span (orig = clamped + 2|offset|, the forward's footprint), capped at 8.
    uint32_t mk=1,mh=1;
    for(int i=0;i<na;++i){ float ow=I.nsx[i]+2.f*std::fabs(I.ox[i]); float oh=I.nsy[i]+2.f*std::fabs(I.oy[i]);
        uint32_t k=(uint32_t)std::floor(ow*I.inv_bsx)+2u; uint32_t h=(uint32_t)std::floor(oh*I.inv_bsy)+2u; mk=std::max(mk,k); mh=std::max(mh,h);}
    I.max_k=std::min(mk,MAXKH); I.max_h=std::min(mh,MAXKH);
    I.cpc_cell=((uint32_t)na+I.nc-1)/I.nc;
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
        I.grouped_pg=(uint32_t)((uint64_t)na*128u+128u); I.out_pg=I.mc*4u; I.field_pg=(uint32_t)((uint64_t)NBX*NBY*4u);
        auto mb=[&](uint64_t b,uint32_t pg){DeviceLocalBufferConfig c{.page_size=pg,.buffer_type=BufferType::DRAM};ReplicatedBufferConfig r{.size=b};return MeshBuffer::create(r,c,I.dev);};
        I.cntb=mb((uint64_t)nc*I.cnt_pg,I.cnt_pg); I.tbb=mb(I.tb_pg,I.tb_pg); I.spb=mb((uint64_t)nc*I.plan_pg,I.plan_pg);
        I.grb=mb(I.grouped_pg,I.grouped_pg); I.gxb=mb((uint64_t)nc*I.out_pg,I.out_pg); I.gyb=mb((uint64_t)nc*I.out_pg,I.out_pg); I.oibb=mb((uint64_t)nc*I.out_pg,I.out_pg);
        I.cnts.assign((size_t)nc*I.rpw,0u); I.tile_base.assign(I.tb_pg/4u,0u); I.srcpref.assign((size_t)nc*I.rpw,0u);
        I.gradx.assign((size_t)nc*I.mc,0.f); I.grady.assign((size_t)nc*I.mc,0.f); I.oidx_host.assign((size_t)nc*I.mc,0u);
        I.g_first.assign(nc,0); I.g_n.assign(nc,0); I.g_col0.assign(nc,0); I.g_nb.assign(nc,0);
        const uint32_t sta=I.geom_addr, gra=I.grb->address(), cnta=I.cntb->address(), tba=I.tbb->address(), spa=I.spb->address();
        I.dr=MeshCoordinateRange{I.dev->shape()};
        // count program
        Program pc=CreateProgram(); cbf(pc,0,Impl::PCHUNK*128u,1); cbf(pc,1,I.cnt_pg,1);
        I.kc=CreateKernel(pc,std::string(DENSITY_KERNEL_DIR)+"v35_count.cpp",I.crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        for(int c=0;c<nc;++c){uint32_t f=(uint32_t)c*I.cpc_cell,n=std::min(I.cpc_cell,(f<(uint32_t)na)?((uint32_t)na-f):0u);
            SetRuntimeArgs(pc,I.kc,I.ccs[c],{(uint32_t)c,f,n,sta,I.geom_pg,I.ntiles,I.W,cnta,I.cnt_pg,Impl::PCHUNK});}
        I.wl_count.add_program(I.dr,std::move(pc));
        // place program
        Program pp=CreateProgram(); cbf(pp,0,Impl::PCHUNK*128u,1); cbf(pp,1,Impl::PCHUNK*128u,1); cbf(pp,2,I.tb_pg,1); cbf(pp,3,I.plan_pg,1); cbf(pp,4,((4u*I.ntiles*4u+63u)&~63u),1);
        I.kp=CreateKernel(pp,std::string(DENSITY_KERNEL_DIR)+"v35_place.cpp",I.crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        for(int c=0;c<nc;++c){uint32_t f=(uint32_t)c*I.cpc_cell,n=std::min(I.cpc_cell,(f<(uint32_t)na)?((uint32_t)na-f):0u);
            SetRuntimeArgs(pp,I.kp,I.ccs[c],{(uint32_t)c,f,n,sta,I.geom_pg,I.ntiles,I.W,gra,I.grouped_pg,tba,spa,I.plan_pg,Impl::PCHUNK});}
        I.wl_place.add_program(I.dr,std::move(pp));
        // gather program (args set per-iter)
        Program pg=CreateProgram();
        cbf(pg,0,4096,I.max_k);cbf(pg,1,4096,I.max_h);cbf(pg,2,4096,2);cbf(pg,3,4096,2);cbf(pg,4,4096,2);cbf(pg,5,4096,2);cbf(pg,6,4096,2);cbf(pg,16,4096,2);cbf(pg,17,4096,2);
        const uint32_t fld_bytes=I.band_cols*NBY*4u+(I.max_h+8u)*4u, chunk_bytes=Impl::CHUNK_BATCHES*1024u*128u;
        cbf(pg,24,fld_bytes,1); cbf(pg,25,chunk_bytes,1); cbf(pg,26,fld_bytes,1);
        std::vector<UnpackToDestMode> um(64,UnpackToDestMode::Default); for(uint32_t i=0;i<5;++i)um[i]=UnpackToDestMode::UnpackToDestFp32; um[16]=um[17]=UnpackToDestMode::UnpackToDestFp32;
        I.bk=CreateKernel(pg,std::string(DENSITY_KERNEL_DIR)+"v35_gather_brisc.cpp",I.crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        I.wk=CreateKernel(pg,std::string(DENSITY_KERNEL_DIR)+"v28_ncrisc.cpp",I.crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
        I.ck=CreateKernel(pg,std::string(DENSITY_KERNEL_DIR)+"v28_compute.cpp",I.crs,ComputeConfig{.fp32_dest_acc_en=true,.unpack_to_dest_mode=um,.math_approx_mode=false});
        for(int c=0;c<nc;++c){ SetRuntimeArgs(pg,I.bk,I.ccs[c],std::vector<uint32_t>(18,0u)); SetRuntimeArgs(pg,I.ck,I.ccs[c],{0u,I.max_k,I.max_h}); SetRuntimeArgs(pg,I.wk,I.ccs[c],std::vector<uint32_t>(13,0u)); }
        I.wl_gather.add_program(I.dr,std::move(pg));
        I.built=true;
    }

    // ── PASS 1: count ──
    auto tc=hrclock::now(); EnqueueMeshWorkload(*I.cq,I.wl_count,false); Finish(*I.cq); tm.count_ms=ms_since(tc);

    // ── host plan ──
    auto tp=hrclock::now();
    EnqueueReadMeshBuffer(*I.cq,I.cnts,I.cntb,true);
    const uint32_t rpw=I.rpw, W=I.W, ntiles=I.ntiles;
    std::vector<uint32_t> total(ntiles,0);
    for(uint32_t tl=0;tl<ntiles;++tl){uint32_t acc=0; for(int s=0;s<nc;++s){I.srcpref[(size_t)s*rpw+tl]=acc; acc+=I.cnts[(size_t)s*rpw+tl];} total[tl]=acc;}
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
    EnqueueWriteMeshBuffer(*I.cq,I.tbb,I.tile_base,false); EnqueueWriteMeshBuffer(*I.cq,I.spb,I.srcpref,false); Finish(*I.cq);
    // set gather args for this iter
    Program& pg=I.wl_gather.get_programs().at(I.dr);
    const uint32_t gra=I.grb->address(),gxa=I.gxb->address(),gya=I.gyb->address(),oiba=I.oibb->address();
    for(int c=0;c<nc;++c){uint32_t col0=I.g_col0[c],vcols=(col0<(uint32_t)NBX)?std::min(I.band_cols,(uint32_t)NBX-col0):1u; uint32_t nb=I.g_nb[c];
        SetRuntimeArgs(pg,I.bk,I.ccs[c],{(uint32_t)c,I.g_n[c],I.band_cols,(uint32_t)NBY,I.max_k,I.max_h,fx_addr,fy_addr,I.field_pg,gra,I.grouped_pg,I.g_first[c],nb,Impl::CHUNK_BATCHES,col0,vcols,oiba,I.out_pg});
        SetRuntimeArgs(pg,I.ck,I.ccs[c],{nb,I.max_k,I.max_h});
        SetRuntimeArgs(pg,I.wk,I.ccs[c],{(uint32_t)c,nb,(uint32_t)NBY,I.max_k,I.max_h,fy_addr,I.field_pg,col0,vcols,gxa,gya,I.out_pg,I.band_cols});}
    tm.plan_ms=ms_since(tp);

    // ── PASS 2: place ──
    auto tpl=hrclock::now(); EnqueueMeshWorkload(*I.cq,I.wl_place,false); Finish(*I.cq); tm.place_ms=ms_since(tpl);
    // ── PASS 3: gather ──
    auto tg=hrclock::now(); EnqueueMeshWorkload(*I.cq,I.wl_gather,false); Finish(*I.cq); tm.gather_ms=ms_since(tg);

    // ── d2h grad + oidx → unsort grad_full[sel[oidx]] ──
    auto td=hrclock::now();
    EnqueueReadMeshBuffer(*I.cq,I.gradx,I.gxb,false); EnqueueReadMeshBuffer(*I.cq,I.grady,I.gyb,false); EnqueueReadMeshBuffer(*I.cq,I.oidx_host,I.oibb,true);
    // sub-tiled big cells → multiple sub-cells share a node → ACCUMULATE. zero the
    // active node grads first (caller pre-zeros fixed nodes), then sum sub-cell grads.
    for(int i=0;i<na;++i){ int g=I.sel[i]; grad[g]=0.f; grad[nn+g]=0.f; }
    for(int c=0;c<nc;++c){ uint32_t n=I.g_n[c]; const size_t cb=(size_t)c*I.mc;
        for(uint32_t s=0;s<n;++s){ uint32_t oi=I.oidx_host[cb+s]; if(oi==0xffffffffu||(int)oi>=na)continue; int g=I.sel[oi]; grad[g]+=I.gradx[cb+s]; grad[nn+g]+=I.grady[cb+s]; } }
    tm.d2h_ms=ms_since(td);
    tm.total_ms=ms_since(t0); if(t)*t=tm;
    if(getenv("V35_DIAG") && I.diag_n++ < 3) fprintf(stderr,"[v35_diag] na=%d nc=%d W=%u ntiles=%u max_k=%u max_h=%u mc=%u maxn=%u | count=%.3f plan=%.3f place=%.3f gather=%.3f d2h=%.3f total=%.3f\n",
        na,nc,I.W,I.ntiles,I.max_k,I.max_h,I.mc,maxn,tm.count_ms,tm.plan_ms,tm.place_ms,tm.gather_ms,tm.d2h_ms,tm.total_ms);
}
}  // namespace v35ef
