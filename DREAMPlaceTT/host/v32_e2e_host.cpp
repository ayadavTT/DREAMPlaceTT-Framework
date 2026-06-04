// SPDX-License-Identifier: Apache-2.0
// V32 END-TO-END backward chain (isolated): regroup (on-chip, count→prefix→place)
// → V30 bin-owner atomic backward → grad, vs CPU reference. Records use the V31
// forward STASH format {cell, bin_global, area_fp32, pad}, produced per source core
// (cell-owner). Regroup groups by bin-owner = bin/slab_bins (forward's partition),
// transforms area→fixed-point, writes per-owner pages; v30_atomic gathers each
// owner's field slab + scatters grad to cell-owners via noc_semaphore_inc.
// Validates the full new path before wiring into the live engine.  CLI: [grid ncells span]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
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
int main(int argc,char**argv){
    uint32_t grid=512, ncells=200000, span=2; int clustered=0;
    if(argc>=2)grid=atoi(argv[1]); if(argc>=3)ncells=atoi(argv[2]); if(argc>=4)span=atoi(argv[3]); if(argc>=5)clustered=atoi(argv[4]);
    const uint32_t nbx=grid,nby=grid,kc=span,hc=span;
    auto mesh=MeshDevice::create_unit_mesh(0); auto&cq=mesh->mesh_command_queue();
    auto g=mesh->compute_with_storage_grid_size(); const uint32_t nc=g.x*g.y; const uint32_t NOWN=nc;
    std::vector<CoreCoord> ccs; for(uint32_t y=0;y<g.y;++y)for(uint32_t x=0;x<g.x;++x)ccs.push_back({x,y});
    std::set<CoreRange> crs; for(auto c:ccs)crs.insert(CoreRange{c,c}); CoreRangeSet all(crs);
    auto dr=MeshCoordinateRange{mesh->shape()};
    const uint32_t total_bins=nbx*nby;
    const uint32_t slab_bins=(((total_bins+nc-1)/nc)+15u)&~15u;       // forward's partition, 64-B aligned
    const uint32_t field_pg=slab_bins*4u;
    const uint32_t cpc_cell=(ncells+nc-1)/nc;                          // cell-owner = cell/cpc_cell
    const float WSCALE=32768.f, FSCALE=16384.f, DESCALE=1.f/16384.f;

    // ── generate cells; build per-SOURCE-core record pages {cell,bin,area_fp32,pad} ──
    std::mt19937 rng(777); std::uniform_real_distribution<float> wf(0.05f,1.f), ff(-1.f,1.f);
    std::vector<uint32_t> cbx(ncells),cby(ncells); std::vector<float> px((size_t)ncells*kc),py((size_t)ncells*hc);
    auto pick=[&](uint32_t hi)->uint32_t{ if(!clustered)return std::uniform_int_distribution<uint32_t>(0,hi)(rng);
        double m=hi*0.5,s=hi*0.08; int v=(int)std::lround(std::normal_distribution<double>(m,s)(rng)); return (uint32_t)std::min<int>(std::max(v,0),(int)hi); };
    for(uint32_t i=0;i<ncells;++i){ cbx[i]=pick(nbx-kc); cby[i]=pick(nby-hc);
        for(uint32_t k=0;k<kc;++k)px[i*kc+k]=wf(rng); for(uint32_t h=0;h<hc;++h)py[i*hc+h]=wf(rng); }
    std::vector<float> fxf(total_bins),fyf(total_bins); for(uint32_t b=0;b<total_bins;++b){fxf[b]=ff(rng);fyf[b]=ff(rng);}

    const uint32_t M_src=cpc_cell*kc*hc;                               // max records per source core
    std::vector<uint32_t> srcM(nc,0); std::vector<uint32_t> recs((size_t)nc*M_src*4u,0u);
    std::vector<uint32_t> exp_tot(NOWN,0);
    // OPTIMIZED stash format {cell, owner, w_fixed, bin} — owner + fixed-point weight
    // pre-computed (mimics the forward, which already has both). Regroup = pure move.
    for(uint32_t i=0;i<ncells;++i){ uint32_t src=i/cpc_cell; if(src>=nc)src=nc-1;
        for(uint32_t k=0;k<kc;++k)for(uint32_t h=0;h<hc;++h){
            uint32_t bg=(cbx[i]+k)*nby+(cby[i]+h); uint32_t o=bg/slab_bins; if(o>=NOWN)o=NOWN-1;
            float area=px[i*kc+k]*py[i*hc+h]; int32_t wf=(int32_t)lroundf(area*WSCALE);
            size_t b=((size_t)src*M_src+srcM[src])*4u; recs[b]=i; recs[b+1]=o; recs[b+2]=(uint32_t)wf; recs[b+3]=bg;
            srcM[src]++; exp_tot[o]++; }}
    uint32_t max_tot=0; for(uint32_t o=0;o<NOWN;++o)max_tot=std::max(max_tot,exp_tot[o]);
    const uint32_t cap=((max_tot+7u)&~7u); const uint32_t route_pg=cap*16u;
    const uint32_t cm_pg=((NOWN*4u+63u)/64u)*64u; const uint32_t STR=cm_pg/4u;
    double balloon=(double)nc*cap*16.0/(1024*1024), ideal=(double)((uint64_t)ncells*kc*hc)*16.0/(1024*1024);
    printf("[v32e2e] grid=%u ncells=%u span=%ux%u clustered=%d nc=%u slab_bins=%u M_src=%u avg_owner=%u max_owner=%u cap=%u\n",
           grid,ncells,kc,hc,clustered,nc,slab_bins,M_src,(uint32_t)((uint64_t)ncells*kc*hc/nc),max_tot,cap);
    printf("[v32e2e] route mem: per-owner-padded=%.0f MB  vs ideal(flat)=%.0f MB  balloon=%.1fx\n",balloon,ideal,balloon/ideal);

    // ── buffers ──
    auto mk=[&](uint64_t b,uint32_t pg){DeviceLocalBufferConfig cf{.page_size=pg,.buffer_type=BufferType::DRAM};ReplicatedBufferConfig r{.size=b};return MeshBuffer::create(r,cf,mesh.get());};
    auto recs_b=mk((uint64_t)nc*M_src*16u,M_src*16u), cntm_b=mk((uint64_t)nc*cm_pg,cm_pg), base_b=mk((uint64_t)nc*cm_pg,cm_pg);
    auto route_b=mk((uint64_t)nc*route_pg,route_pg);
    auto fxb=mk((uint64_t)nc*field_pg,field_pg), fyb=mk((uint64_t)nc*field_pg,field_pg);
    // int32 fixed-point field per owner slab
    std::vector<int32_t> fx((size_t)nc*slab_bins,0),fy((size_t)nc*slab_bins,0);
    for(uint32_t o=0;o<nc;++o)for(uint32_t l=0;l<slab_bins;++l){uint32_t bg=o*slab_bins+l; if(bg<total_bins){fx[o*slab_bins+l]=(int32_t)lroundf(fxf[bg]*FSCALE); fy[o*slab_bins+l]=(int32_t)lroundf(fyf[bg]*FSCALE);}}
    EnqueueWriteMeshBuffer(cq,recs_b,recs,false); EnqueueWriteMeshBuffer(cq,fxb,fx,false); EnqueueWriteMeshBuffer(cq,fyb,fy,false); Finish(cq);
    uint32_t ra=recs_b->address(),ca=cntm_b->address(),ba=base_b->address(),oa=route_b->address(),fxa=fxb->address(),fya=fyb->address();
    using tt::CBIndex; const uint32_t chunk=2048u; uint32_t wsb; {union{float f;uint32_t u;}cv;cv.f=WSCALE;wsb=cv.u;}
    auto mkc=[&](Program&p,CoreRangeSet&cr,CBIndex id,uint32_t bytes){CircularBufferConfig c(bytes,{{id,DataFormat::Float32}});c.set_page_size(id,bytes);CreateCircularBuffer(p,cr,c);};
    CoreRangeSet one(CoreRange{ccs[0],ccs[0]});

    // ── regroup: count → prefix(page_mode) → place ──
    Program pc=CreateProgram();
    mkc(pc,all,CBIndex::c_0,chunk*16u); mkc(pc,all,CBIndex::c_1,cm_pg);
    auto kc_=CreateKernel(pc,std::string(DENSITY_KERNEL_DIR)+"v32c_count.cpp",all,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
    for(uint32_t c=0;c<nc;++c)SetRuntimeArgs(pc,kc_,ccs[c],{c,ra,M_src*16u,srcM[c],NOWN,ca,cm_pg,chunk});
    Program pp=CreateProgram();
    mkc(pp,one,CBIndex::c_0,nc*cm_pg); mkc(pp,one,CBIndex::c_1,nc*cm_pg); mkc(pp,one,CBIndex::c_2,cm_pg);
    auto kp=CreateKernel(pp,std::string(DENSITY_KERNEL_DIR)+"v32_prefix.cpp",one,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
    SetRuntimeArgs(pp,kp,ccs[0],{nc,NOWN,ca,cm_pg,ba,cm_pg,1u});
    const uint32_t pchunk=16384u;                                  // sort chunk (256KB staging)
    Program ppl=CreateProgram();
    mkc(ppl,all,CBIndex::c_0,pchunk*16u); mkc(ppl,all,CBIndex::c_1,cm_pg); mkc(ppl,all,CBIndex::c_2,pchunk*16u);
    mkc(ppl,all,CBIndex::c_3,cm_pg); mkc(ppl,all,CBIndex::c_4,cm_pg);
    auto kpl=CreateKernel(ppl,std::string(DENSITY_KERNEL_DIR)+"v32c_place_sorted.cpp",all,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
    for(uint32_t c=0;c<nc;++c)SetRuntimeArgs(ppl,kpl,ccs[c],{c,ra,M_src*16u,srcM[c],NOWN,ba,cm_pg,oa,route_pg,pchunk});

    // ── backward: zero → atomic → readout (V30 kernels, per-owner worker) ──
    const uint32_t slab_ints=cpc_cell*2u, grad_pg=slab_ints*4u; auto gb=mk((uint64_t)nc*grad_pg,grad_pg); uint32_t ga=gb->address();
    std::vector<uint32_t> nocc(nc); for(uint32_t c=0;c<nc;++c){auto p=mesh->worker_core_from_logical_core(ccs[c]); nocc[c]=((uint32_t)p.x<<16)|(uint32_t)p.y;}
    Program p0=CreateProgram(); mkc(p0,all,CBIndex::c_28,grad_pg);
    auto k0=CreateKernel(p0,std::string(DENSITY_KERNEL_DIR)+"v30_zero.cpp",all,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
    for(uint32_t c=0;c<nc;++c)SetRuntimeArgs(p0,k0,ccs[c],{slab_ints});
    Program p1=CreateProgram(); mkc(p1,all,CBIndex::c_28,grad_pg); mkc(p1,all,CBIndex::c_24,field_pg); mkc(p1,all,CBIndex::c_25,field_pg); mkc(p1,all,CBIndex::c_26,chunk*16u+64u); mkc(p1,all,CBIndex::c_27,(nc+16u)*4u);
    auto k1=CreateKernel(p1,std::string(DENSITY_KERNEL_DIR)+"v30_atomic.cpp",all,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
    SetCommonRuntimeArgs(p1,k1,nocc);
    for(uint32_t c=0;c<nc;++c)SetRuntimeArgs(p1,k1,ccs[c],{c,slab_bins,fxa,fya,field_pg,oa,route_pg,c,1u,exp_tot[c],cpc_cell,nc,chunk});
    Program p2=CreateProgram(); mkc(p2,all,CBIndex::c_28,grad_pg);
    auto k2=CreateKernel(p2,std::string(DENSITY_KERNEL_DIR)+"v30_readout.cpp",all,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
    for(uint32_t c=0;c<nc;++c)SetRuntimeArgs(p2,k2,ccs[c],{c,slab_ints,ga,grad_pg});

    MeshWorkload wc,wp,wpl,w0,w1,w2;
    wc.add_program(dr,std::move(pc)); wp.add_program(dr,std::move(pp)); wpl.add_program(dr,std::move(ppl));
    w0.add_program(dr,std::move(p0)); w1.add_program(dr,std::move(p1)); w2.add_program(dr,std::move(p2));
    auto ph=[&](MeshWorkload&w){auto t=hrclock::now(); EnqueueMeshWorkload(cq,w,false); Finish(cq); return ms_since(t);};
    { double a=ph(wc),b=ph(wp),c=ph(wpl); printf("[v32e2e] regroup phases: count=%.3f prefix=%.3f place=%.3f ms\n",a,b,c); }
    auto regroup=[&]{ EnqueueMeshWorkload(cq,wc,false); EnqueueMeshWorkload(cq,wp,false); EnqueueMeshWorkload(cq,wpl,false); Finish(cq); };
    auto backward=[&]{ EnqueueMeshWorkload(cq,w0,false); EnqueueMeshWorkload(cq,w1,false); EnqueueMeshWorkload(cq,w2,false); Finish(cq); };
    auto run=[&]{ regroup(); backward(); };
    run(); const int R=7; std::vector<double> ms,mr,mb;
    for(int i=0;i<R;++i){ auto t=hrclock::now(); regroup(); mr.push_back(ms_since(t));
                          auto t2=hrclock::now(); backward(); mb.push_back(ms_since(t2)); ms.push_back(ms_since(t)); }
    std::sort(ms.begin(),ms.end()); std::sort(mr.begin(),mr.end()); std::sort(mb.begin(),mb.end()); double med=ms[R/2];
    printf("[v32e2e] split: regroup=%.3f  backward=%.3f ms\n",mr[R/2],mb[R/2]);

    // ── readout grad, compare CPU ref (Σ px·py·field) ──
    std::vector<uint32_t> gflat((size_t)nc*slab_ints,0u); EnqueueReadMeshBuffer(cq,gflat,gb,true);
    std::vector<double> gx(ncells,0),gy(ncells,0),rx(ncells,0),ry(ncells,0);
    for(uint32_t i=0;i<ncells;++i){uint32_t o=i/cpc_cell; if(o>=nc)o=nc-1; uint32_t loc=i-o*cpc_cell; size_t b=(size_t)o*slab_ints+loc*2;
        gx[i]=(double)((int32_t)gflat[b])*DESCALE; gy[i]=(double)((int32_t)gflat[b+1])*DESCALE;}
    for(uint32_t i=0;i<ncells;++i)for(uint32_t k=0;k<kc;++k)for(uint32_t h=0;h<hc;++h){uint32_t bg=(cbx[i]+k)*nby+(cby[i]+h); double w=px[i*kc+k]*py[i*hc+h]; rx[i]+=w*fxf[bg]; ry[i]+=w*fyf[bg];}
    double nume=0,den=0; for(uint32_t i=0;i<ncells;++i){nume+=(gx[i]-rx[i])*(gx[i]-rx[i])+(gy[i]-ry[i])*(gy[i]-ry[i]); den+=rx[i]*rx[i]+ry[i]*ry[i];}
    double rel=std::sqrt(nume/(den+1e-30));
    printf("[v32e2e] chain MEDIAN=%.3f ms  rel_l2=%.3e  %s\n",med,rel,rel<1e-3?"OK":"FAIL");
    return 0;
}
