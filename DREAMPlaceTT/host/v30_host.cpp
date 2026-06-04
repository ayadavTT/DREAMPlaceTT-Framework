// SPDX-License-Identifier: Apache-2.0
//
// V30 field-stationary EF backward microbench. Validates the TRANSPOSE of the
// V19 forward: each core owns a contiguous BIN block, reads its field slab with
// ONE local DRAM read (NO NoC per-bin gather — kills V21's latency floor), and
// emits each owned cell's (gx,gy) contribution. Host sums contributions per cell
// and checks rel_l2 vs a CPU reference. Measures the core's device time vs the
// CPU 16T backward target. CLI: v30 [grid ncells span clustered]
//   grid=nbx=nby, span=kc=hc per cell, clustered=1 → center-cluster cells.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <set>
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
    uint32_t grid=512, ncells=451000, span=2; int clustered=0, mode=0;  // mode 0=host-sum 1=on-chip atomic grad
    if(argc>=2)grid=atoi(argv[1]); if(argc>=3)ncells=atoi(argv[2]);
    if(argc>=4)span=atoi(argv[3]); if(argc>=5)clustered=atoi(argv[4]); if(argc>=6)mode=atoi(argv[5]);
    const uint32_t nbx=grid, nby=grid, kc=span, hc=span;
    auto mesh=MeshDevice::create_unit_mesh(0); auto&cq=mesh->mesh_command_queue();
    auto g=mesh->compute_with_storage_grid_size(); const int nc=(int)(g.x*g.y);
    std::vector<CoreCoord> ccs; for(uint32_t y=0;y<g.y;++y)for(uint32_t x=0;x<g.x;++x)ccs.push_back({x,y});
    std::set<CoreRange> crs; for(auto c:ccs)crs.insert(CoreRange{c,c}); CoreRangeSet all_crs(crs);

    const uint32_t total_bins=nbx*nby;
    const uint32_t slab_bins=(((total_bins+nc-1)/nc)+15u)&~15u;   // mult of 16 → 64-B aligned slab
    const uint32_t field_pg=slab_bins*4u;

    // ── generate cells: bxl,byl bin position + px[kc],py[hc] overlap weights ──
    std::mt19937 rng(777);
    std::uniform_real_distribution<float> wf(0.05f,1.0f);
    std::vector<uint32_t> cbx(ncells), cby(ncells);
    std::vector<float> px((size_t)ncells*kc), py((size_t)ncells*hc);
    auto pick=[&](uint32_t hi)->uint32_t{
        if(!clustered) return std::uniform_int_distribution<uint32_t>(0,hi)(rng);
        // center cluster: gaussian around middle, ~10% std
        double m=hi*0.5, s=hi*0.10; int v=(int)std::round(std::normal_distribution<double>(m,s)(rng));
        return (uint32_t)std::min<int>(std::max(v,0),(int)hi);
    };
    for(uint32_t i=0;i<ncells;++i){ cbx[i]=pick(nbx-kc); cby[i]=pick(nby-hc);
        for(uint32_t k=0;k<kc;++k)px[i*kc+k]=wf(rng);
        for(uint32_t h=0;h<hc;++h)py[i*hc+h]=wf(rng); }

    // ── field fx,fy: float ref + int32 fixed-point (×2^14) for the kernel ──
    const float WSCALE=32768.f, FSCALE=32768.f, DESCALE=1.f/32768.f;
    std::vector<float> fxf((size_t)total_bins,0.f), fyf((size_t)total_bins,0.f);
    std::vector<int32_t> fx((size_t)nc*slab_bins,0), fy((size_t)nc*slab_bins,0);
    std::uniform_real_distribution<float> ff(-1.f,1.f);
    auto rfix=[](float v,float s){return (int32_t)std::lround(v*s);};
    for(uint32_t b=0;b<total_bins;++b){ fxf[b]=ff(rng); fyf[b]=ff(rng);
        fx[b]=rfix(fxf[b],FSCALE); fy[b]=rfix(fyf[b],FSCALE); }

    // ── build route[slab] = {cell, w_fixed, bin_global, pad} bucketed by field slab ──
    std::vector<std::vector<uint32_t>> route(nc);
    for(uint32_t i=0;i<ncells;++i){
        for(uint32_t k=0;k<kc;++k)for(uint32_t h=0;h<hc;++h){
            uint32_t bg=(cbx[i]+k)*nby+(cby[i]+h);
            uint32_t owner=bg/slab_bins; if(owner>=(uint32_t)nc)owner=nc-1;
            int32_t wfx=rfix(px[i*kc+k]*py[i*hc+h],WSCALE);
            route[owner].push_back(i); route[owner].push_back((uint32_t)wfx);
            route[owner].push_back(bg); route[owner].push_back(0u); }}
    // ── ADAPTIVE worker assignment: decouple field-owner (slab) from worker ──
    // Each worker serves ONE slab (1 cheap contiguous field read) + a balanced
    // chunk (≤ target records) of that slab's route. Hot slabs get many workers
    // (field replicated, cheap); cold slabs share. Binary-search target so the
    // total worker count ≤ nc. Balances ANY distribution; field always 1 slab.
    std::vector<uint32_t> rcnt(nc); for(int s=0;s<nc;++s)rcnt[s]=(uint32_t)route[s].size()/4u;
    uint64_t R=0; for(int s=0;s<nc;++s)R+=rcnt[s];
    // ── WORKER GROUPING: balance records across ≤nc workers while keeping each
    //    worker's field read ONE contiguous slab range (≤ span_cap, fits L1).
    //    HOT slab (≥target recs) → split across several workers (field=that 1 slab).
    //    COLD/normal slabs → merge consecutive (incl. empties) into a group until
    //    target recs or span_cap. Grow target until #workers ≤ nc.
    const uint32_t span_cap = std::max<uint32_t>(1u, 150000u/slab_bins);  // ~1.2MB field/L1
    struct W{uint32_t lo_slab,n_slabs; std::vector<uint32_t> rec;};
    std::vector<W> wk;
    uint32_t target=(uint32_t)((R+nc-1)/nc); if(target<8)target=8;
    for(int attempt=0;attempt<40;++attempt){ wk.clear();
        int s=0;
        while(s<nc){
            if(rcnt[s]>=target){                          // hot slab → split
                uint32_t r=rcnt[s], nparts=(r+target-1)/target, base=r/nparts, rem=r%nparts, off=0;
                for(uint32_t j=0;j<nparts;++j){ uint32_t c=base+(j<rem?1u:0u); W w; w.lo_slab=s; w.n_slabs=1;
                    w.rec.assign(route[s].begin()+(size_t)off*4u, route[s].begin()+(size_t)(off+c)*4u); off+=c; wk.push_back(std::move(w)); }
                ++s;
            } else {                                       // merge contiguous slabs
                W w; w.lo_slab=s; w.n_slabs=0; uint32_t recs=0;
                while(s<nc && w.n_slabs<span_cap && rcnt[s]<target && recs+rcnt[s]<=target){
                    recs+=rcnt[s]; w.rec.insert(w.rec.end(),route[s].begin(),route[s].end()); w.n_slabs++; ++s; }
                if(w.n_slabs==0){ w.n_slabs=1; ++s; }      // safety
                wk.push_back(std::move(w));
            }
        }
        if((int)wk.size()<=nc) break;
        target += target/4 + 8;                            // too many workers → coarsen
    }
    while((int)wk.size()<nc){ W w; w.lo_slab=0; w.n_slabs=1; wk.push_back(std::move(w)); } // idle pad
    uint32_t max_cnt=0,max_ns=0; for(auto&w:wk){ max_cnt=std::max(max_cnt,(uint32_t)w.rec.size()/4u); max_ns=std::max(max_ns,w.n_slabs);}
    const uint32_t cap=((max_cnt+7u)&~7u);            // per-worker page, balanced
    const uint32_t route_pg=cap*16u, out_pg=cap*16u;
    double imbal=max_cnt/((double)ncells*kc*hc/nc);
    printf("[v30] grid=%u ncells=%u span=%ux%u nc=%d slab_bins=%u  workers=%zu target=%u max_worker=%u max_nslabs=%u imbal=%.2fx clustered=%d\n",
           grid,ncells,kc,hc,nc,slab_bins,wk.size(),target,max_cnt,max_ns,imbal,clustered);

    // flatten each worker's records → its own page
    std::vector<uint32_t> rflat((size_t)nc*cap*4u,0u);
    for(int w=0;w<nc;++w) std::copy(wk[w].rec.begin(),wk[w].rec.end(),rflat.begin()+(size_t)w*cap*4u);

    auto mb=[&](uint64_t b,uint32_t pg){DeviceLocalBufferConfig cf{.page_size=pg,.buffer_type=BufferType::DRAM};ReplicatedBufferConfig r{.size=b};return MeshBuffer::create(r,cf,mesh.get());};
    auto fxb=mb((uint64_t)nc*field_pg,field_pg), fyb=mb((uint64_t)nc*field_pg,field_pg);
    auto rb=mb((uint64_t)nc*route_pg,route_pg);
    EnqueueWriteMeshBuffer(cq,fxb,fx,false); EnqueueWriteMeshBuffer(cq,fyb,fy,false);
    EnqueueWriteMeshBuffer(cq,rb,rflat,false); Finish(cq);
    using tt::CBIndex;
    const uint32_t chunk=2048;
    uint32_t fxa=(uint32_t)fxb->address(),fya=(uint32_t)fyb->address(),ra=(uint32_t)rb->address();
    auto mkc=[&](Program&p,CBIndex id,uint32_t bytes){CircularBufferConfig c(bytes,{{id,DataFormat::Float32}});c.set_page_size(id,bytes);CreateCircularBuffer(p,all_crs,c);};
    auto dr=MeshCoordinateRange{mesh->shape()};
    std::vector<double> gx(ncells,0.0), gy(ncells,0.0);
    double med=0;

    if(mode==0){
        auto ob=mb((uint64_t)nc*out_pg,out_pg); uint32_t oa=(uint32_t)ob->address();
        Program prog=CreateProgram();
        mkc(prog,CBIndex::c_24,max_ns*field_pg); mkc(prog,CBIndex::c_25,max_ns*field_pg);
        mkc(prog,CBIndex::c_26,chunk*16u+64u); mkc(prog,CBIndex::c_27,chunk*16u+64u);
        auto k=CreateKernel(prog,std::string(DENSITY_KERNEL_DIR)+"v30_backward.cpp",all_crs,
            DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        for(int c=0;c<nc;++c){ auto&a=wk[c]; uint32_t cnt=(uint32_t)a.rec.size()/4u;
            SetRuntimeArgs(prog,k,ccs[c],{(uint32_t)c,slab_bins,fxa,fya,field_pg,ra,route_pg,a.lo_slab,a.n_slabs,cnt,oa,out_pg,chunk});}
        MeshWorkload wl; wl.add_program(dr,std::move(prog));
        auto run=[&]{EnqueueMeshWorkload(cq,wl,false);Finish(cq);};
        run(); const int N=9; std::vector<double> ms; for(int i=0;i<N;++i){auto t=hrclock::now();run();ms.push_back(ms_since(t));} std::sort(ms.begin(),ms.end()); med=ms[N/2];
        std::vector<uint32_t> oflat((size_t)nc*cap*4u,0u); EnqueueReadMeshBuffer(cq,oflat,ob,true);
        for(int c=0;c<nc;++c){ uint32_t rc2=(uint32_t)wk[c].rec.size()/4u;
            for(uint32_t i=0;i<rc2;++i){ size_t b=((size_t)c*cap+i)*4u;
                uint32_t cell=oflat[b+0]; gx[cell]+=(double)((int32_t)oflat[b+1])*DESCALE; gy[cell]+=(double)((int32_t)oflat[b+2])*DESCALE; }}
    } else {
        // ── ON-CHIP atomic grad: P0 zero → P1 gather+atomic → P2 readout ──
        const uint32_t cpc_cell=(ncells+nc-1)/nc, slab_ints=cpc_cell*2u, grad_pg=slab_ints*4u;
        auto gb=mb((uint64_t)nc*grad_pg,grad_pg); uint32_t ga=(uint32_t)gb->address();
        std::vector<uint32_t> nocc(nc); for(int c=0;c<nc;++c){auto pc=mesh->worker_core_from_logical_core(ccs[c]); nocc[c]=((uint32_t)pc.x<<16)|(uint32_t)pc.y;}
        Program p0=CreateProgram(); mkc(p0,CBIndex::c_28,grad_pg);
        auto k0=CreateKernel(p0,std::string(DENSITY_KERNEL_DIR)+"v30_zero.cpp",all_crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        for(int c=0;c<nc;++c)SetRuntimeArgs(p0,k0,ccs[c],{slab_ints});
        Program p1=CreateProgram(); mkc(p1,CBIndex::c_28,grad_pg); mkc(p1,CBIndex::c_24,max_ns*field_pg); mkc(p1,CBIndex::c_25,max_ns*field_pg); mkc(p1,CBIndex::c_26,chunk*16u+64u); mkc(p1,CBIndex::c_27,(nc+16u)*4u);
        auto k1=CreateKernel(p1,std::string(DENSITY_KERNEL_DIR)+"v30_atomic.cpp",all_crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        SetCommonRuntimeArgs(p1,k1,nocc);
        for(int c=0;c<nc;++c){ auto&a=wk[c]; uint32_t cnt=(uint32_t)a.rec.size()/4u;
            SetRuntimeArgs(p1,k1,ccs[c],{(uint32_t)c,slab_bins,fxa,fya,field_pg,ra,route_pg,a.lo_slab,a.n_slabs,cnt,cpc_cell,(uint32_t)nc,chunk});}
        Program p2=CreateProgram(); mkc(p2,CBIndex::c_28,grad_pg);
        auto k2=CreateKernel(p2,std::string(DENSITY_KERNEL_DIR)+"v30_readout.cpp",all_crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        for(int c=0;c<nc;++c)SetRuntimeArgs(p2,k2,ccs[c],{(uint32_t)c,slab_ints,ga,grad_pg});
        MeshWorkload w0,w1,w2; w0.add_program(dr,std::move(p0)); w1.add_program(dr,std::move(p1)); w2.add_program(dr,std::move(p2));
        auto run=[&]{EnqueueMeshWorkload(cq,w0,false);EnqueueMeshWorkload(cq,w1,false);EnqueueMeshWorkload(cq,w2,false);Finish(cq);};
        run(); const int N=9; std::vector<double> ms; for(int i=0;i<N;++i){auto t=hrclock::now();run();ms.push_back(ms_since(t));} std::sort(ms.begin(),ms.end()); med=ms[N/2];
        std::vector<uint32_t> gflat((size_t)nc*slab_ints,0u); EnqueueReadMeshBuffer(cq,gflat,gb,true);
        for(uint32_t i=0;i<ncells;++i){ uint32_t owner=i/cpc_cell; if(owner>=(uint32_t)nc)owner=nc-1; uint32_t loc=i-owner*cpc_cell;
            size_t b=(size_t)owner*slab_ints+loc*2; gx[i]=(double)((int32_t)gflat[b])*DESCALE; gy[i]=(double)((int32_t)gflat[b+1])*DESCALE; }
    }
    // CPU reference (float field)
    std::vector<double> rx(ncells,0.0), ry(ncells,0.0);
    for(uint32_t i=0;i<ncells;++i)for(uint32_t k=0;k<kc;++k)for(uint32_t h=0;h<hc;++h){
        uint32_t bg=(cbx[i]+k)*nby+(cby[i]+h); double w=px[i*kc+k]*py[i*hc+h];
        rx[i]+=w*fxf[bg]; ry[i]+=w*fyf[bg]; }
    double nume=0,den=0; for(uint32_t i=0;i<ncells;++i){
        nume+=(gx[i]-rx[i])*(gx[i]-rx[i])+(gy[i]-ry[i])*(gy[i]-ry[i]); den+=rx[i]*rx[i]+ry[i]*ry[i]; }
    double rel=std::sqrt(nume/(den+1e-30));
    printf("[v30] device MEDIAN=%.3f ms   rel_l2=%.3e  %s\n",med,rel,rel<1e-4?"OK":"FAIL");
    printf("V30_RESULT,%u,%u,%u,%d,%.3f,%.3e\n",grid,ncells,span,clustered,med,rel);
    return 0;
}
