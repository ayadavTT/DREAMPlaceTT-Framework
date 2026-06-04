// SPDX-License-Identifier: Apache-2.0
//
// FCCS (Field-Cast, Cell-Stationary) density-backward microbench harness.
// Synthetic balanced cells + int32 fixed-point field; runs kernels/fccs_dm.cpp
// (field multicast to all cores + per-core in-place gather); checks rel_l2 vs a
// CPU float reference and reports device time vs the CPU 16T backward target.
// CLI: fccs [grid ncells span clustered]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>
#include <string>
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
static constexpr uint32_t GREC=32u, MAXKH=8u, WS=13u, FS=13u;

int main(int argc,char**argv){
    uint32_t grid=512, ncells=200000, span=2; int clustered=0, no_mcast=0;
    if(argc>=2)grid=atoi(argv[1]); if(argc>=3)ncells=atoi(argv[2]);
    if(argc>=4)span=atoi(argv[3]); if(argc>=5)clustered=atoi(argv[4]); if(argc>=6)no_mcast=atoi(argv[5]);
    const uint32_t NBX=grid,NBY=grid,kc=span,hc=span;
    auto mesh=MeshDevice::create_unit_mesh(0); auto&cq=mesh->mesh_command_queue();
    MeshCoordinateRange dr(mesh->shape());
    CoreCoord g=mesh->compute_with_storage_grid_size(); const uint32_t nc=g.x*g.y;
    std::vector<CoreCoord> lc,nocc;
    for(uint32_t y=0;y<g.y;++y)for(uint32_t x=0;x<g.x;++x){lc.push_back({x,y});nocc.push_back(mesh->worker_core_from_logical_core({x,y}));}
    // gap-aware rects
    std::set<uint32_t> ax; for(auto&c:nocc)ax.insert(c.x);
    std::vector<std::pair<uint32_t,uint32_t>> runs;{uint32_t lo=0,hi=0;bool in=false;
        for(uint32_t x:ax){if(!in){lo=hi=x;in=true;}else if(x==hi+1)hi=x;else{runs.emplace_back(lo,hi);lo=hi=x;}}if(in)runs.emplace_back(lo,hi);}
    struct Rect{uint32_t xs,ys,xe,ye,nd;};std::vector<Rect> rects;
    for(auto&xr:runs){uint32_t ymin=UINT32_MAX,ymax=0,cnt=0;for(auto&c:nocc)if(c.x>=xr.first&&c.x<=xr.second){ymin=std::min(ymin,(uint32_t)c.y);ymax=std::max(ymax,(uint32_t)c.y);++cnt;}rects.push_back({xr.first,ymin,xr.second,ymax,cnt});}

    const uint32_t nworkers=nc-1u;                 // core 0 = pure producer
    const uint32_t cpc=(ncells+nworkers-1)/nworkers;
    // tile width: bound each of the 2 ping-pong field bufs to ~256KB
    uint32_t W=std::min(NBX, std::max(1u, (256u*1024u)/(NBY*4u))); const uint32_t Tx=(NBX+W-1)/W;
    printf("[fccs] grid=%u ncells=%u span=%ux%u clustered=%d nc=%u cpc=%u W=%u Tx=%u tiles=%u tile=%uKB\n",
           grid,ncells,kc,hc,clustered,nc,cpc,W,Tx,2*Tx,(W*NBY*4u)>>10);

    // ── synthetic cells (balanced slices already, just index order) ──
    std::mt19937 rng(4242); std::uniform_real_distribution<float> wf(0.05f,1.0f);
    auto pick=[&](uint32_t hi)->uint32_t{ if(!clustered) return std::uniform_int_distribution<uint32_t>(0,hi)(rng);
        double m=hi*0.5,s=hi*0.10; int v=(int)std::round(std::normal_distribution<double>(m,s)(rng)); return (uint32_t)std::min<int>(std::max(v,0),(int)hi); };
    std::vector<int32_t> cbx(ncells),cby(ncells); std::vector<float> px((size_t)ncells*kc),py((size_t)ncells*hc),ratio(ncells);
    for(uint32_t i=0;i<ncells;++i){ cbx[i]=pick(NBX-kc); cby[i]=pick(NBY-hc); ratio[i]=0.5f+wf(rng);
        for(uint32_t k=0;k<kc;++k)px[i*kc+k]=wf(rng); for(uint32_t h=0;h<hc;++h)py[i*hc+h]=wf(rng); }
    // ── field fx,fy (float ref + int32 fixed-point, column-major [fx|fy]) ──
    const uint32_t total=NBX*NBY; const double WSCALE=(double)(1u<<WS),FSCALE=(double)(1u<<FS);
    std::vector<float> fxf(total),fyf(total); std::uniform_real_distribution<float> ff(-1.f,1.f);
    std::vector<int32_t> fld((size_t)2*total);
    for(uint32_t c=0;c<NBX;++c)for(uint32_t r=0;r<NBY;++r){ float a=ff(rng),b=ff(rng); uint32_t idx=c*NBY+r;
        fxf[idx]=a; fyf[idx]=b; fld[idx]=(int32_t)std::lround(a*FSCALE); fld[total+idx]=(int32_t)std::lround(b*FSCALE); }
    // ── geometry: FORWARD-ORDERED (PRE-SORTED by bxl per worker, orig_idx in [20]) ──
    // Represents the forward emitting ordered geometry; the backward does no sort.
    const uint32_t ssrow_i=(((NBX+1u)*4u+63u)&~63u)/4u;     // sort_start row, 64-B aligned
    std::vector<int32_t> geom((size_t)nworkers*cpc*GREC,0);
    std::vector<uint32_t> ss((size_t)nworkers*ssrow_i,0u);
    for(uint32_t w=0;w<nworkers;++w){
        uint32_t wf=w*cpc; uint32_t wn=(wf<ncells)?std::min(cpc,ncells-wf):0u;
        std::vector<uint32_t> ord(wn); for(uint32_t i=0;i<wn;++i)ord[i]=i;
        std::sort(ord.begin(),ord.end(),[&](uint32_t a,uint32_t b){return cbx[wf+a]<cbx[wf+b];});
        uint32_t* row=&ss[(size_t)w*ssrow_i];
        for(uint32_t i=0;i<wn;++i){int32_t bx=cbx[wf+i]; if(bx<0)bx=0; if(bx>=(int32_t)NBX)bx=NBX-1; row[bx+1]++;}
        for(uint32_t b=0;b<NBX;++b) row[b+1]+=row[b];     // prefix → row[b]=#cells bxl<b
        for(uint32_t s=0;s<wn;++s){ uint32_t i=ord[s]; uint32_t g=wf+i; size_t slot=(size_t)wf+s;
            int32_t* gr=&geom[slot*GREC]; gr[0]=cbx[g];gr[1]=cby[g];gr[2]=(int32_t)kc;gr[3]=(int32_t)hc;
            for(uint32_t k=0;k<kc;++k)gr[4+k]=(int32_t)std::lround(px[g*kc+k]*WSCALE);
            for(uint32_t h=0;h<hc;++h)gr[12+h]=(int32_t)std::lround(py[g*hc+h]*WSCALE);
            gr[20]=(int32_t)i;   // orig_idx (local) → grad lands in original order
        }
    }

    // ── DRAM buffers (single flat pages) ──
    auto mb=[&](uint64_t bytes){DeviceLocalBufferConfig c{.page_size=(uint32_t)bytes,.buffer_type=BufferType::DRAM};ReplicatedBufferConfig r{.size=bytes};return MeshBuffer::create(r,c,mesh.get());};
    const uint32_t CHUNK=512u;
    uint64_t field_bytes=(uint64_t)2*total*4, geom_bytes=(uint64_t)nworkers*cpc*GREC*4, grad_bytes=(uint64_t)nworkers*cpc*16;
    auto fb=mb(field_bytes),gb=mb(geom_bytes),ssb=mb((uint64_t)nworkers*ssrow_i*4),grb=mb(grad_bytes);  // gb=pre-sorted geom, ssb=sort_start
    EnqueueWriteMeshBuffer(cq,fb,fld,false); EnqueueWriteMeshBuffer(cq,gb,geom,false); EnqueueWriteMeshBuffer(cq,ssb,ss,false); Finish(cq);

    // ── L1 layout: 2 ping-pong field bufs + grad(resident) + sort_start(resident,NBX+1) + streamed geom chunk ──
    // field region is reused for the on-chip-sort histogram+cursor (2*NBX ints) BEFORE field streaming.
    const uint32_t ALIGN=64u;
    uint32_t buf_stride=(W*NBY*4u+ALIGN-1u)&~(ALIGN-1u);
    if(buf_stride < 2u*NBX*4u) buf_stride = (2u*NBX*4u+ALIGN-1u)&~(ALIGN-1u);   // ensure field region holds hist+cursor
    uint32_t field_l1_off=0u;
    uint32_t grad_l1_off=(2u*buf_stride+ALIGN-1u)&~(ALIGN-1u);
    uint32_t sort_start_l1_off=(grad_l1_off + cpc*16u + ALIGN-1u)&~(ALIGN-1u);
    uint32_t chunk_l1_off=(sort_start_l1_off + (NBX+1u)*4u + ALIGN-1u)&~(ALIGN-1u);
    uint32_t cb_size=chunk_l1_off + CHUNK*GREC*4u + ALIGN;
    if(cb_size>=1500u*1024u){ printf("[fccs] FATAL cb_size %uKB > L1 (lower W)\n",cb_size>>10); return 1; }

    std::set<CoreRange> crs_s; for(auto&c:lc)crs_s.insert(CoreRange{c,c}); CoreRangeSet crs(crs_s);
    Program prog=CreateProgram();
    CreateCircularBuffer(prog,crs,CircularBufferConfig(cb_size,{{24u,DataFormat::UInt32}}).set_page_size(24u,cb_size));
    auto k=CreateKernel(prog,std::string(DENSITY_KERNEL_DIR)+"fccs_dm.cpp",crs,
        DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
    CreateKernel(prog,std::string(DENSITY_KERNEL_DIR)+"v13_void_brisc.cpp",crs,
        DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
    uint32_t ready_sid=CreateSemaphore(prog,crs,0u),tile_sid=CreateSemaphore(prog,crs,0u),done_sid=CreateSemaphore(prog,crs,0u);
    auto c0=nocc[0]; uint32_t fa=(uint32_t)fb->address(),ga=(uint32_t)gb->address(),gra=(uint32_t)grb->address(),ssa=(uint32_t)ssb->address();
    for(uint32_t c=0;c<nc;++c){
        std::vector<uint32_t> a={c,nc,(c==0u)?1u:0u,cpc,ncells,NBX,NBY,MAXKH,MAXKH,W,Tx,fa,ga,gra,
            field_l1_off,chunk_l1_off,grad_l1_off,buf_stride,(uint32_t)c0.x,(uint32_t)c0.y,ready_sid,tile_sid,done_sid,WS,(uint32_t)no_mcast,ssa,sort_start_l1_off,(uint32_t)rects.size()};
        for(auto&r:rects){bool self=(nocc[c].x>=r.xs&&nocc[c].x<=r.xe&&nocc[c].y>=r.ys&&nocc[c].y<=r.ye);
            a.push_back(r.xs);a.push_back(r.ys);a.push_back(r.xe);a.push_back(r.ye);a.push_back(self?r.nd-1u:r.nd);}
        SetRuntimeArgs(prog,k,lc[c],a);
    }
    MeshWorkload wl; wl.add_program(dr,std::move(prog));
    EnqueueMeshWorkload(cq,wl,false); Finish(cq);   // warmup
    const int RN=7; std::vector<double> tv;
    for(int i=0;i<RN;++i){auto t=hrclock::now();EnqueueMeshWorkload(cq,wl,false);Finish(cq);tv.push_back(ms_since(t));}
    std::sort(tv.begin(),tv.end()); double med=tv[RN/2];

    // ── readback (grad in ORIGINAL order: slot g = cell g) + accuracy (no unsort) ──
    std::vector<int64_t> gd((size_t)nworkers*cpc*2,0); EnqueueReadMeshBuffer(cq,gd,grb,true);
    const double DESCALE=1.0/(double)((uint64_t)1<<(WS+FS));   // kernel pre-shifts px·py>>WS → acc = real×2^(WS+FS)
    double refn=0,difn=0; size_t nbad=0;
    for(uint32_t i=0;i<ncells;++i){ size_t slot=i;
        double ax=0,ay=0;
        for(uint32_t k=0;k<kc;++k)for(uint32_t h=0;h<hc;++h){ size_t idx=(size_t)(cbx[i]+k)*NBY+(cby[i]+h); double w=(double)px[i*kc+k]*py[i*hc+h]; ax+=w*fxf[idx]; ay+=w*fyf[idx]; }
        double gxr=ratio[i]*ax, gyr=ratio[i]*ay;
        double gxd=ratio[i]*(double)gd[slot*2]*DESCALE, gyd=ratio[i]*(double)gd[slot*2+1]*DESCALE;
        refn+=gxr*gxr+gyr*gyr; difn+=(gxd-gxr)*(gxd-gxr)+(gyd-gyr)*(gyd-gyr);
        if(std::fabs(gxd-gxr)>1e-2*std::max(1.0,std::fabs(gxr)))nbad++; }
    double rel=refn>0?std::sqrt(difn/refn):0;
    printf("[fccs] device MEDIAN=%.3f ms  rel_l2=%.3e nbad=%zu  %s\n",med,rel,nbad,(rel<1e-3)?"OK":"CHECK");
    printf("FCCS_RESULT,%u,%u,%u,%d,%.3f,%.3e\n",grid,ncells,span,clustered,med,rel);
    return 0;
}
