// SPDX-License-Identifier: Apache-2.0
// V30 field-stationary EF backward engine (hybrid). See v30_ef_engine.h.
#include "v30_ef_engine.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <set>
#include <vector>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tt_metal.hpp>
using namespace tt; using namespace tt::tt_metal; using namespace tt::tt_metal::distributed;
using hrclock=std::chrono::high_resolution_clock;
template<class T> static double ms_since(T t){return std::chrono::duration<double,std::milli>(hrclock::now()-t).count();}
#ifndef DENSITY_KERNEL_DIR
#define DENSITY_KERNEL_DIR "kernels/"
#endif
namespace v30ef {
static constexpr uint32_t MAXKH=8;
static const float QS=32768.f;          // fixed-point quantum (2^15) per axis

struct V30EFEngine::Impl {
    MeshDevice* dev; MeshCommandQueue* cq;
    int NBX,NBY,nmax; float xl,yl,bsx,bsy,inv_bsx,inv_bsy;
    int nc; CoreCoord grid; std::vector<CoreCoord> ccs; CoreRangeSet crs{std::set<CoreRange>{}};
    uint32_t slab_bins=0, field_pg=0, span_cap=1, cpc_cell=0, slab_ints=0, grad_pg=0;
    int n_active=0, n_total=0; std::vector<int32_t> sel;
    std::vector<float> ox,oy,nsx,nsy,ratio;
    uint32_t maxk=1,maxh=1;
    bool built=false; uint32_t cap=0;
    std::shared_ptr<MeshBuffer> fxb,fyb,rb,gb;
    std::vector<uint32_t> nocc, fx_i, fy_i, rflat, gflat;
    Impl(void* md,int M,int N,int nm,float xl_,float yl_,float bx,float by)
      :dev((MeshDevice*)md),NBX(M),NBY(N),nmax(nm),xl(xl_),yl(yl_),bsx(bx),bsy(by),
       inv_bsx(1.f/bx),inv_bsy(1.f/by){
        cq=&dev->mesh_command_queue(); grid=dev->compute_with_storage_grid_size(); nc=grid.x*grid.y;
        std::set<CoreRange> s; for(uint32_t y=0;y<grid.y;++y)for(uint32_t x=0;x<grid.x;++x){CoreCoord c{x,y};ccs.push_back(c);s.insert(CoreRange{c,c});}
        crs=CoreRangeSet(s);
        slab_bins=((( (uint32_t)NBX*NBY + nc-1)/nc)+15u)&~15u; field_pg=slab_bins*4u;
        span_cap=std::max<uint32_t>(1u,150000u/slab_bins);
    }
};
V30EFEngine::V30EFEngine(void* md,int M,int N,int nm,float xl,float yl,float bx,float by)
  :impl_(std::make_unique<Impl>(md,M,N,nm,xl,yl,bx,by)){}
V30EFEngine::~V30EFEngine()=default;

void V30EFEngine::configure_with_sel(const float* ox,const float* oy,const float* nsx,
    const float* nsy,const float* ratio,const int32_t* sel,int na,int nt){
    auto&I=*impl_; I.n_active=na; I.n_total=nt; I.sel.assign(sel,sel+na);
    I.ox.resize(na);I.oy.resize(na);I.nsx.resize(na);I.nsy.resize(na);I.ratio.resize(na);
    for(int i=0;i<na;++i){int g=sel[i];I.ox[i]=ox[g];I.oy[i]=oy[g];I.nsx[i]=nsx[g];I.nsy[i]=nsy[g];I.ratio[i]=ratio[g];}
    uint32_t mk=1,mh=1; for(int i=0;i<na;++i){uint32_t k=(uint32_t)std::floor(I.nsx[i]*I.inv_bsx)+2,h=(uint32_t)std::floor(I.nsy[i]*I.inv_bsy)+2;mk=std::max(mk,k);mh=std::max(mh,h);}
    I.maxk=std::min(mk,MAXKH); I.maxh=std::min(mh,MAXKH);
    I.cpc_cell=((uint32_t)na+I.nc-1)/I.nc; I.slab_ints=I.cpc_cell*2u; I.grad_pg=I.slab_ints*4u;
}

void V30EFEngine::compute_from_full(const float* pos,const float* fx,const float* fy,float* grad,V30Timing* t){
    auto&I=*impl_; V30Timing tm{}; auto t0=hrclock::now();
    const int na=I.n_active,nc=I.nc,NBX=I.NBX,NBY=I.NBY; const uint32_t sb=I.slab_bins;

    // ── host PREP: per-cell records {cell,w=ratio·px·py,bin_global} bucketed by slab ──
    auto tp=hrclock::now();
    std::vector<std::vector<float>> rw(nc); std::vector<std::vector<uint32_t>> rc(nc), rb_(nc); // w, cell, bin
    float wmax=1e-30f;
    for(int i=0;i<na;++i){ int g=I.sel[i];
        float nx=pos[g]+I.ox[i], ny=pos[I.n_total+g]+I.oy[i];
        int bxl=(int)((nx-I.xl)*I.inv_bsx), bxh=(int)((nx+I.nsx[i]-I.xl)*I.inv_bsx)+1;
        int byl=(int)((ny-I.yl)*I.inv_bsy), byh=(int)((ny+I.nsy[i]-I.yl)*I.inv_bsy)+1;
        if(bxl<0)bxl=0; if(bxh>NBX)bxh=NBX; if(byl<0)byl=0; if(byh>NBY)byh=NBY;
        int kc=std::min(std::max(bxh-bxl,0),(int)MAXKH), hc=std::min(std::max(byh-byl,0),(int)MAXKH);
        float nxt=nx+I.nsx[i], nyt=ny+I.nsy[i], r=I.ratio[i];
        float bxlo=I.xl+(float)bxl*I.bsx;
        for(int k=0;k<kc;++k){ float bhi=bxlo+I.bsx, px=std::min(nxt,bhi)-std::max(nx,bxlo); bxlo+=I.bsx;
            float bylo=I.yl+(float)byl*I.bsy;
            for(int h=0;h<hc;++h){ float bhy=bylo+I.bsy, py=std::min(nyt,bhy)-std::max(ny,bylo); bylo+=I.bsy;
                float w=r*px*py; uint32_t bg=(uint32_t)(bxl+k)*NBY+(uint32_t)(byl+h);
                uint32_t s=bg/sb; if(s>=(uint32_t)nc)s=nc-1;
                rw[s].push_back(w); rc[s].push_back((uint32_t)i); rb_[s].push_back(bg);
                float a=std::fabs(w); if(a>wmax)wmax=a; }}
    }
    // field max for dynamic scale
    float fmax=1e-30f; const size_t NB=(size_t)NBX*NBY;
    for(size_t b=0;b<NB;++b){ float a=std::fabs(fx[b]); if(a>fmax)fmax=a; a=std::fabs(fy[b]); if(a>fmax)fmax=a; }
    const float WS=QS/wmax, FS=QS/fmax;            // scale so |fixed| ≤ 2^15
    const double DESC=(double)wmax*fmax/(double)(1u<<15); // gx = gx_fixed * DESC  (>>15 in kernel)
    tm.prep_ms=ms_since(tp);

    // ── field → int slab pages (per-slab, interleaved) ──
    std::vector<uint32_t> rcnt(nc); uint64_t R=0; for(int s=0;s<nc;++s){rcnt[s]=(uint32_t)rc[s].size(); R+=rcnt[s];}
    if(I.fx_i.empty()) I.fx_i.assign((size_t)nc*sb,0); else std::fill(I.fx_i.begin(),I.fx_i.end(),0);
    if(I.fy_i.empty()) I.fy_i.assign((size_t)nc*sb,0); else std::fill(I.fy_i.begin(),I.fy_i.end(),0);
    for(size_t b=0;b<NB;++b){ I.fx_i[b]=(uint32_t)(int32_t)std::lround(fx[b]*FS); I.fy_i[b]=(uint32_t)(int32_t)std::lround(fy[b]*FS); }

    // ── GROUPING: hot-split + cold-merge (≤nc workers, contiguous slab range) ──
    struct W{uint32_t lo,ns,cnt; std::vector<uint32_t> rec;}; std::vector<W> wk;
    uint32_t target=(uint32_t)((R+nc-1)/nc); if(target<8)target=8;
    for(int att=0;att<40;++att){ wk.clear(); int s=0;
        while(s<nc){
            if(rcnt[s]>=target){ uint32_t r=rcnt[s],np=(r+target-1)/target,base=r/np,rem=r%np,off=0;
                for(uint32_t j=0;j<np;++j){ uint32_t c=base+(j<rem?1u:0u); W w; w.lo=s;w.ns=1;w.cnt=c;
                    w.rec.reserve(c*4); for(uint32_t e=off;e<off+c;++e){ w.rec.push_back(rc[s][e]); w.rec.push_back((uint32_t)(int32_t)std::lround(rw[s][e]*WS)); w.rec.push_back(rb_[s][e]); w.rec.push_back(0u);} off+=c; wk.push_back(std::move(w)); }
                ++s;
            } else { W w; w.lo=s;w.ns=0;w.cnt=0; uint32_t recs=0;
                while(s<nc && w.ns<I.span_cap && rcnt[s]<target && recs+rcnt[s]<=target){
                    for(uint32_t e=0;e<rcnt[s];++e){ w.rec.push_back(rc[s][e]); w.rec.push_back((uint32_t)(int32_t)std::lround(rw[s][e]*WS)); w.rec.push_back(rb_[s][e]); w.rec.push_back(0u);} recs+=rcnt[s]; w.ns++; ++s; }
                if(w.ns==0){w.ns=1;++s;} w.cnt=recs; wk.push_back(std::move(w)); }
        }
        if((int)wk.size()<=nc) break; target+=target/4+8;
    }
    while((int)wk.size()<nc){ W w; w.lo=0;w.ns=1;w.cnt=0; wk.push_back(std::move(w)); }
    uint32_t max_cnt=0,max_ns=0; for(auto&w:wk){max_cnt=std::max(max_cnt,w.cnt);max_ns=std::max(max_ns,w.ns);}

    // ── (re)alloc persistent buffers if cap grew ──
    const uint32_t need_cap=((max_cnt+7u)&~7u)+8u;
    using tt::CBIndex;
    auto mb=[&](uint64_t b,uint32_t pg){DeviceLocalBufferConfig c{.page_size=pg,.buffer_type=BufferType::DRAM};ReplicatedBufferConfig r{.size=b};return MeshBuffer::create(r,c,I.dev);};
    if(!I.built || need_cap>I.cap){
        I.cap=need_cap;
        I.fxb=mb((uint64_t)nc*I.field_pg,I.field_pg); I.fyb=mb((uint64_t)nc*I.field_pg,I.field_pg);
        I.rb=mb((uint64_t)nc*I.cap*16u,I.cap*16u);
        if(!I.gb) I.gb=mb((uint64_t)nc*I.grad_pg,I.grad_pg);
        I.nocc.resize(nc); for(int c=0;c<nc;++c){auto pc=I.dev->worker_core_from_logical_core(I.ccs[c]); I.nocc[c]=((uint32_t)pc.x<<16)|(uint32_t)pc.y;}
        I.gflat.assign((size_t)nc*I.slab_ints,0);
        I.built=true;
    }
    if(I.rflat.size()!=(size_t)nc*I.cap*4u) I.rflat.assign((size_t)nc*I.cap*4u,0u); else std::fill(I.rflat.begin(),I.rflat.end(),0u);
    for(int w=0;w<nc;++w) std::copy(wk[w].rec.begin(),wk[w].rec.end(),I.rflat.begin()+(size_t)w*I.cap*4u);

    // ── h2d field + route ──
    auto th=hrclock::now();
    EnqueueWriteMeshBuffer(*I.cq,I.fxb,I.fx_i,false); EnqueueWriteMeshBuffer(*I.cq,I.fyb,I.fy_i,false);
    EnqueueWriteMeshBuffer(*I.cq,I.rb,I.rflat,false); Finish(*I.cq); tm.h2d_ms=ms_since(th);

    // ── build 3 programs (zero→atomic→readout); CB c_28 first in each ──
    auto tg=hrclock::now();
    const uint32_t chunk=2048; const uint32_t fxa=I.fxb->address(),fya=I.fyb->address(),ra=I.rb->address(),ga=I.gb->address();
    auto mkc=[&](Program&p,CBIndex id,uint32_t bytes){CircularBufferConfig c(bytes,{{id,DataFormat::Float32}});c.set_page_size(id,bytes);CreateCircularBuffer(p,I.crs,c);};
    auto dr=MeshCoordinateRange{I.dev->shape()};
    Program p0=CreateProgram(); mkc(p0,CBIndex::c_28,I.grad_pg);
    auto k0=CreateKernel(p0,std::string(DENSITY_KERNEL_DIR)+"v30_zero.cpp",I.crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
    for(int c=0;c<nc;++c)SetRuntimeArgs(p0,k0,I.ccs[c],{I.slab_ints});
    Program p1=CreateProgram(); mkc(p1,CBIndex::c_28,I.grad_pg); mkc(p1,CBIndex::c_24,max_ns*I.field_pg); mkc(p1,CBIndex::c_25,max_ns*I.field_pg); mkc(p1,CBIndex::c_26,chunk*16u+64u); mkc(p1,CBIndex::c_27,(nc+16u)*4u);
    auto k1=CreateKernel(p1,std::string(DENSITY_KERNEL_DIR)+"v30_atomic.cpp",I.crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
    SetCommonRuntimeArgs(p1,k1,I.nocc);
    for(int c=0;c<nc;++c){auto&w=wk[c]; SetRuntimeArgs(p1,k1,I.ccs[c],{(uint32_t)c,sb,fxa,fya,I.field_pg,ra,I.cap*16u,w.lo,w.ns,w.cnt,I.cpc_cell,(uint32_t)nc,chunk});}
    Program p2=CreateProgram(); mkc(p2,CBIndex::c_28,I.grad_pg);
    auto k2=CreateKernel(p2,std::string(DENSITY_KERNEL_DIR)+"v30_readout.cpp",I.crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
    for(int c=0;c<nc;++c)SetRuntimeArgs(p2,k2,I.ccs[c],{(uint32_t)c,I.slab_ints,ga,I.grad_pg});
    MeshWorkload w0,w1,w2; w0.add_program(dr,std::move(p0)); w1.add_program(dr,std::move(p1)); w2.add_program(dr,std::move(p2));
    EnqueueMeshWorkload(*I.cq,w0,false); EnqueueMeshWorkload(*I.cq,w1,false); EnqueueMeshWorkload(*I.cq,w2,false); Finish(*I.cq);
    tm.gather_ms=ms_since(tg);

    // ── d2h grad slabs → descale → grad[sel] (+force; hook negates) ──
    auto td=hrclock::now();
    EnqueueReadMeshBuffer(*I.cq,I.gflat,I.gb,true);
    for(int i=0;i<na;++i){ uint32_t owner=(uint32_t)i/I.cpc_cell; if(owner>=(uint32_t)nc)owner=nc-1; uint32_t loc=(uint32_t)i-owner*I.cpc_cell;
        size_t b=(size_t)owner*I.slab_ints+loc*2; int g=I.sel[i];
        grad[g]=(float)((int32_t)I.gflat[b]*DESC); grad[I.n_total+g]=(float)((int32_t)I.gflat[b+1]*DESC); }
    tm.d2h_ms=ms_since(td);
    tm.total_ms=ms_since(t0); if(t)*t=tm;
}
}  // namespace v30ef
