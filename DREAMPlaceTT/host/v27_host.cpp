// SPDX-License-Identifier: Apache-2.0
//
// V27 on-chip cell-bucketing scatter microbench. Groups cells by field row-band
// (route[src][band]) for the V26 backward, without touching the V19 forward.
// Measures the scatter time + validates bucket counts. CLI: v27 [H ncells]

#include <algorithm>
#include <chrono>
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
    uint32_t H=2048, ncells=540000;
    if(argc>=3){H=atoi(argv[1]);ncells=atoi(argv[2]);}
    auto mesh=MeshDevice::create_unit_mesh(0); auto&cq=mesh->mesh_command_queue();
    auto grid=mesh->compute_with_storage_grid_size(); const int nc=(int)(grid.x*grid.y);
    std::vector<CoreCoord> ccs; for(uint32_t y=0;y<grid.y;++y)for(uint32_t x=0;x<grid.x;++x)ccs.push_back({x,y});
    std::set<CoreRange> crs; for(auto c:ccs)crs.insert(CoreRange{c,c}); CoreRangeSet all_crs(crs);
    const uint32_t rpc=(H+nc-1)/nc;

    // cells: byl(global), bxl, 4 weights
    std::mt19937 rng(12345); std::uniform_int_distribution<uint32_t> byd(0,H-2), bxd(0,2046);
    std::uniform_real_distribution<float> uf(-1,1);
    std::vector<uint32_t> in((size_t)ncells*6u+16u,0u); union{uint32_t u;float f;}cv;  // +64B pad
    for(uint32_t i=0;i<ncells;++i){in[i*6+0]=byd(rng); in[i*6+1]=bxd(rng);
        for(int k=2;k<6;++k){cv.f=uf(rng);in[i*6+k]=cv.u;}}

    // host pre-count max per (src,band) to size cap (each core scatters a contiguous slice).
    // cpc rounded to a multiple of 8 so each core's slice starts at a 64-byte-aligned
    // DRAM offset (8 records × 24 B = 192 = 3·64) — Blackhole DRAM reads need 64-B align.
    const uint32_t cpc=(((ncells+nc-1)/nc)+7u)&~7u;
    std::vector<uint32_t> hcnt((size_t)nc*nc,0); uint32_t maxc=0;
    for(int c=0;c<nc;++c){uint32_t f=(uint32_t)c*cpc, n=std::min(cpc, (f<ncells)?(ncells-f):0u);
        for(uint32_t i=0;i<n;++i){uint32_t byl=in[(f+i)*6]; uint32_t band=byl/rpc; if(band>=(uint32_t)nc)band=nc-1;
            uint32_t v=++hcnt[(size_t)c*nc+band]; maxc=std::max(maxc,v);}}
    const uint32_t cap = maxc + maxc/2 + 16;   // +50% margin
    const uint32_t route_pg=cap*24u, in_pg=ncells*24u, cnt_pg=nc*4u;
    printf("[v27] H=%u ncells=%u nc=%d rpc=%u cpc=%u  max/(src,band)=%u cap=%u  route_buf=%.0f MB\n",
           H,ncells,nc,rpc,cpc,maxc,cap,(double)nc*nc*route_pg/1048576.0);

    auto mb=[&](uint64_t b,uint32_t pg){DeviceLocalBufferConfig c{.page_size=pg,.buffer_type=BufferType::DRAM};ReplicatedBufferConfig r{.size=b};return MeshBuffer::create(r,c,mesh.get());};
    // pad input buffer by 64 B so the rounded-up (64-aligned) tail reads stay in-bounds
    auto inb=mb((uint64_t)in_pg+64u,in_pg+64u), rb=mb((uint64_t)nc*nc*route_pg,route_pg), cb=mb((uint64_t)nc*cnt_pg,cnt_pg);
    EnqueueWriteMeshBuffer(cq,inb,in,false); Finish(cq);

    using tt::CBIndex; Program prog=CreateProgram();
    const uint32_t chunk=2048;
    {CircularBufferConfig c(chunk*24u+24u,{{CBIndex::c_0,DataFormat::Float32}});c.set_page_size(CBIndex::c_0,chunk*24u+24u);CreateCircularBuffer(prog,all_crs,c);}
    {CircularBufferConfig c(nc*4u,{{CBIndex::c_1,DataFormat::Float32}});c.set_page_size(CBIndex::c_1,nc*4u);CreateCircularBuffer(prog,all_crs,c);}
    auto k=CreateKernel(prog,std::string(DENSITY_KERNEL_DIR)+"v27_bucket.cpp",all_crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
    uint32_t ia=(uint32_t)inb->address(),ra=(uint32_t)rb->address(),ca=(uint32_t)cb->address();
    for(int c=0;c<nc;++c){uint32_t f=(uint32_t)c*cpc, n=std::min(cpc,(f<ncells)?(ncells-f):0u);
        SetRuntimeArgs(prog,k,ccs[c],{(uint32_t)c,f,n,ia,in_pg,ra,cap,(uint32_t)nc,rpc,ca,chunk});}
    MeshWorkload wl; auto dr=MeshCoordinateRange{mesh->shape()}; wl.add_program(dr,std::move(prog));
    auto run=[&]{EnqueueMeshWorkload(cq,wl,false);Finish(cq);};
    run(); const int N=7; std::vector<double> ms; for(int i=0;i<N;++i){auto t=hrclock::now();run();ms.push_back(ms_since(t));}
    std::sort(ms.begin(),ms.end()); double med=ms[N/2];

    // validate counts: per band, Σ_src device-count == host expected
    std::vector<uint32_t> dcnt((size_t)nc*nc,0); EnqueueReadMeshBuffer(cq,dcnt,cb,true);
    std::vector<uint32_t> exp(nc,0); for(uint32_t i=0;i<ncells;++i){uint32_t band=in[i*6]/rpc; if(band>=(uint32_t)nc)band=nc-1; exp[band]++;}
    uint32_t bad=0,ovf=0; for(int b=0;b<nc;++b){uint32_t tot=0; for(int s=0;s<nc;++s){tot+=dcnt[(size_t)s*nc+b]; if(dcnt[(size_t)s*nc+b]>=cap)ovf++;} if(tot!=exp[b])bad++;}
    printf("[v27] bucketing scatter MEDIAN=%.3f ms  (vs CPU EF ~5-7 ms; this is the backward grouping)\n",med);
    printf("[v27] validate: bands_mismatch=%u cap_overflow=%u  %s\n",bad,ovf,(bad==0&&ovf==0)?"OK":"FAIL");
    printf("V27_RESULT,%u,%u,%.3f,%u,%u\n",H,ncells,med,bad,ovf);
    return 0;
}
