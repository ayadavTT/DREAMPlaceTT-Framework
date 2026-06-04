// SPDX-License-Identifier: Apache-2.0
//
// V31 backward microbench — tests the STASH-reuse backward on chip. Generates a
// realistic forward stash: cells spatially sorted by x (mimics V22_RELABEL), split
// into contiguous cell-owner slices; route[c] = that core's cells' (cell,bin,weight=
// ratio·px·py) records (cell-sorted). Each core reads route[c] + its local field
// band + accumulates grad (NO prep). Validates grad vs CPU + measures device time.
// CLI: v31_backward [grid ncells span]
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
    uint32_t grid=512, ncells=1390000, span=2;
    if(argc>=2)grid=atoi(argv[1]); if(argc>=3)ncells=atoi(argv[2]); if(argc>=4)span=atoi(argv[3]);
    const uint32_t NBX=grid, NBY=grid, kc=span, hc=span;
    auto mesh=MeshDevice::create_unit_mesh(0); auto&cq=mesh->mesh_command_queue();
    auto g=mesh->compute_with_storage_grid_size(); const int nc=(int)(g.x*g.y);
    std::vector<CoreCoord> ccs; for(uint32_t y=0;y<g.y;++y)for(uint32_t x=0;x<g.x;++x)ccs.push_back({x,y});
    std::set<CoreRange> crs; for(auto c:ccs)crs.insert(CoreRange{c,c}); CoreRangeSet all(crs);

    // ── cells: bxl spatially sorted (mimic V22_RELABEL), byl random, px/py/ratio ──
    std::mt19937 rng(99); std::uniform_real_distribution<float> wf(0.05f,1.f), rf(0.3f,1.2f);
    std::vector<uint32_t> cbx(ncells), cby(ncells); std::vector<float> px((size_t)ncells*kc), py((size_t)ncells*hc), ratio(ncells);
    for(uint32_t i=0;i<ncells;++i){ cbx[i]=std::uniform_int_distribution<uint32_t>(0,NBX-kc)(rng); cby[i]=std::uniform_int_distribution<uint32_t>(0,NBY-hc)(rng);
        for(uint32_t k=0;k<kc;++k)px[i*kc+k]=wf(rng); for(uint32_t h=0;h<hc;++h)py[i*hc+h]=wf(rng); ratio[i]=rf(rng); }
    std::vector<uint32_t> order(ncells); for(uint32_t i=0;i<ncells;++i)order[i]=i;
    std::sort(order.begin(),order.end(),[&](uint32_t a,uint32_t b){return cbx[a]<cbx[b];});  // spatial sort by x

    const uint32_t cpc=(ncells+nc-1)/nc;          // cells per owner (contiguous in sorted order)
    // ── field int (dynamic scale) + ref ──
    std::vector<float> fxf((size_t)NBX*NBY), fyf((size_t)NBX*NBY);
    std::uniform_real_distribution<float> ff(-1.f,1.f); float fmax=1e-30f;
    for(size_t b=0;b<(size_t)NBX*NBY;++b){ fxf[b]=ff(rng); fyf[b]=ff(rng); fmax=std::max(fmax,std::max(std::fabs(fxf[b]),std::fabs(fyf[b]))); }
    // weight max
    float wmax=1e-30f;
    for(uint32_t i=0;i<ncells;++i)for(uint32_t k=0;k<kc;++k)for(uint32_t h=0;h<hc;++h)wmax=std::max(wmax,ratio[i]*px[i*kc+k]*py[i*hc+h]);
    const float WS=16384.f/wmax, FS=16384.f/fmax; const double DESC=(double)wmax*fmax/16384.0/16384.0*32768.0;
    auto rfix=[](float v,float s){return (int32_t)std::lround(v*s);};
    std::vector<int32_t> fx((size_t)NBX*NBY), fy((size_t)NBX*NBY);
    for(size_t b=0;b<(size_t)NBX*NBY;++b){ fx[b]=rfix(fxf[b],FS); fy[b]=rfix(fyf[b],FS); }

    // ── build stash route[owner] (cell-sorted: cells in sorted order) + per-core band ──
    std::vector<std::vector<uint32_t>> route(nc); std::vector<uint32_t> bx_lo(nc,NBX), bx_hi(nc,0);
    for(uint32_t s=0;s<ncells;++s){ uint32_t i=order[s]; uint32_t owner=s/cpc; if(owner>=(uint32_t)nc)owner=nc-1;
        bx_lo[owner]=std::min(bx_lo[owner],cbx[i]); bx_hi[owner]=std::max(bx_hi[owner],cbx[i]+kc-1);
        for(uint32_t k=0;k<kc;++k)for(uint32_t h=0;h<hc;++h){ uint32_t bg=(cbx[i]+k)*NBY+(cby[i]+h);
            int32_t wfx=rfix(ratio[i]*px[i*kc+k]*py[i*hc+h],WS);
            route[owner].push_back(s); route[owner].push_back(bg); route[owner].push_back((uint32_t)wfx); route[owner].push_back(0u); } }
    uint32_t max_rt=0,max_nbx=0; for(int c=0;c<nc;++c){ max_rt=std::max(max_rt,(uint32_t)route[c].size()/4u);
        if(bx_hi[c]>=bx_lo[c]) max_nbx=std::max(max_nbx,bx_hi[c]-bx_lo[c]+1u); }
    const uint32_t cap=(max_rt+7u)&~7u, route_pg=cap*16u, col_pg=NBY*4u;
    const uint32_t grad_pg=cpc*8u;  // per-core grad page (cpc cells × [gx,gy])
    printf("[v31bw] grid=%u ncells=%u span=%ux%u nc=%d cpc=%u max_rt=%u max_nbx=%u (band=%u KB) field=%u MB\n",
           grid,ncells,kc,hc,nc,cpc,max_rt,max_nbx,(max_nbx*col_pg*2u)>>10,((uint32_t)NBX*NBY*8u)>>20);

    std::vector<uint32_t> rflat((size_t)nc*cap*4u,0u);
    for(int c=0;c<nc;++c)std::copy(route[c].begin(),route[c].end(),rflat.begin()+(size_t)c*cap*4u);

    auto mb=[&](uint64_t b,uint32_t pg){DeviceLocalBufferConfig cf{.page_size=pg,.buffer_type=BufferType::DRAM};ReplicatedBufferConfig r{.size=b};return MeshBuffer::create(r,cf,mesh.get());};
    auto fxb=mb((uint64_t)NBX*col_pg,col_pg), fyb=mb((uint64_t)NBX*col_pg,col_pg);   // per-x-column pages
    auto rb=mb((uint64_t)nc*route_pg,route_pg), gb=mb((uint64_t)nc*grad_pg,grad_pg);
    EnqueueWriteMeshBuffer(cq,fxb,fx,false); EnqueueWriteMeshBuffer(cq,fyb,fy,false); EnqueueWriteMeshBuffer(cq,rb,rflat,false); Finish(cq);

    using tt::CBIndex; Program prog=CreateProgram(); const uint32_t chunk=2048;
    auto mkc=[&](CBIndex id,uint32_t bytes){CircularBufferConfig c(bytes,{{id,DataFormat::Float32}});c.set_page_size(id,bytes);CreateCircularBuffer(prog,all,c);};
    mkc(CBIndex::c_24,(max_nbx+1u)*col_pg); mkc(CBIndex::c_25,(max_nbx+1u)*col_pg);
    mkc(CBIndex::c_26,chunk*16u+64u); mkc(CBIndex::c_27,cpc*8u+64u);
    auto k=CreateKernel(prog,std::string(DENSITY_KERNEL_DIR)+"v31_backward.cpp",all,
        DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
    uint32_t fxa=fxb->address(),fya=fyb->address(),ra=rb->address(),ga=gb->address();
    const uint32_t nbx_cap=max_nbx+1u;   // band CB capacity (kernel computes its own bx_lo/n_bx)
    for(int c=0;c<nc;++c){ uint32_t rcnt=(uint32_t)route[c].size()/4u;
        uint32_t cbase=(uint32_t)c*cpc; uint32_t slab=(cbase<ncells)?std::min(cpc,ncells-cbase):0u;
        SetRuntimeArgs(prog,k,ccs[c],{(uint32_t)c,ra,route_pg,rcnt,slab,nbx_cap,NBX,NBY,fxa,fya,col_pg,ga,grad_pg,cbase,chunk});}
    MeshWorkload wl; auto dr=MeshCoordinateRange{mesh->shape()}; wl.add_program(dr,std::move(prog));
    auto run=[&]{EnqueueMeshWorkload(cq,wl,false);Finish(cq);};
    run(); const int N=9; std::vector<double> ms; for(int i=0;i<N;++i){auto t=hrclock::now();run();ms.push_back(ms_since(t));}
    std::sort(ms.begin(),ms.end()); double med=ms[N/2];

    std::vector<uint32_t> gflat((size_t)nc*cpc*2u,0u); EnqueueReadMeshBuffer(cq,gflat,gb,true);
    std::vector<double> rx(ncells,0.0),ry(ncells,0.0);
    for(uint32_t i=0;i<ncells;++i)for(uint32_t k=0;k<kc;++k)for(uint32_t h=0;h<hc;++h){ uint32_t bg=(cbx[i]+k)*NBY+(cby[i]+h);
        double w=ratio[i]*px[i*kc+k]*py[i*hc+h]; rx[i]+=w*fxf[bg]; ry[i]+=w*fyf[bg]; }
    double nume=0,den=0; for(uint32_t s=0;s<ncells;++s){ uint32_t i=order[s]; double gx=(double)((int32_t)gflat[s*2])*DESC, gy=(double)((int32_t)gflat[s*2+1])*DESC;
        nume+=(gx-rx[i])*(gx-rx[i])+(gy-ry[i])*(gy-ry[i]); den+=rx[i]*rx[i]+ry[i]*ry[i]; }
    double rel=std::sqrt(nume/(den+1e-30));
    printf("[v31bw] device MEDIAN=%.3f ms   rel_l2=%.3e  %s\n",med,rel,rel<1e-3?"OK":"FAIL");
    return 0;
}
