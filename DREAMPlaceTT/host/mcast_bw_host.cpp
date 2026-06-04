// SPDX-License-Identifier: Apache-2.0
//
// FCCS field-cast bandwidth microbench harness.
// One producer streams a DRAM field (n_tiles × tile_bytes) and multicasts each
// tile to all worker cores. Reports effective broadcast bandwidth (the make-or-
// break number for the FCCS density-backward design — docs/DENSITY_BACKWARD_NEXTGEN_DESIGN.md).
// CLI: mcast_bw [tile_bytes] [n_tiles] [repeats] [n_iters]
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

int main(int argc,char**argv){
    uint32_t tile_bytes=262144u, n_tiles=64u, repeats=8u; int n_iters=7;
    if(argc>=2)tile_bytes=(uint32_t)atoi(argv[1]);
    if(argc>=3)n_tiles=(uint32_t)atoi(argv[2]);
    if(argc>=4)repeats=(uint32_t)atoi(argv[3]);
    if(argc>=5)n_iters=atoi(argv[4]);
    tile_bytes=(tile_bytes+63u)&~63u;
    const uint64_t field_bytes=(uint64_t)tile_bytes*n_tiles;

    auto mesh=MeshDevice::create_unit_mesh(0); auto&cq=mesh->mesh_command_queue();
    MeshCoordinateRange dr(mesh->shape());
    CoreCoord grid=mesh->compute_with_storage_grid_size(); const uint32_t nc_all=grid.x*grid.y;
    std::vector<CoreCoord> lc, nocc;
    for(uint32_t y=0;y<grid.y;++y)for(uint32_t x=0;x<grid.x;++x){lc.push_back({x,y});nocc.push_back(mesh->worker_core_from_logical_core({x,y}));}
    // gap-aware multicast rects (contiguous-x runs) — from v13_mcast_smoke_host
    std::set<uint32_t> ax; for(auto&c:nocc)ax.insert(c.x);
    std::vector<std::pair<uint32_t,uint32_t>> runs; { uint32_t lo=0,hi=0; bool in=false;
        for(uint32_t x:ax){ if(!in){lo=hi=x;in=true;} else if(x==hi+1)hi=x; else {runs.emplace_back(lo,hi);lo=hi=x;} } if(in)runs.emplace_back(lo,hi);}
    struct Rect{uint32_t xs,ys,xe,ye,nd;}; std::vector<Rect> rects;
    for(auto&xr:runs){ uint32_t ymin=UINT32_MAX,ymax=0,cnt=0;
        for(auto&c:nocc) if(c.x>=xr.first&&c.x<=xr.second){ymin=std::min(ymin,(uint32_t)c.y);ymax=std::max(ymax,(uint32_t)c.y);++cnt;}
        rects.push_back({xr.first,ymin,xr.second,ymax,cnt}); }
    uint32_t totd=0; for(auto&r:rects)totd+=r.nd;
    printf("[mcast_bw] grid=%ux%u nc_all=%u tile=%uKB n_tiles=%u field=%.1fMB repeats=%u rects=%zu(dests=%u)\n",
           grid.x,grid.y,nc_all,tile_bytes>>10,n_tiles,(double)field_bytes/1048576.0,repeats,rects.size(),totd);

    // L1: dst region (mcast lands) at 0, producer srcbuf after
    const uint32_t ALIGN=64u;
    const uint32_t dst_off=0u, srcbuf_off=(tile_bytes+ALIGN-1u)&~(ALIGN-1u);
    const uint32_t cb_size=srcbuf_off+tile_bytes;
    if(cb_size>=1500u*1024u){ printf("[mcast_bw] FATAL cb_size %uKB > L1; lower tile_bytes\n",cb_size>>10); return 1; }

    std::set<CoreRange> crs_s; for(auto&c:lc)crs_s.insert(CoreRange{c,c}); CoreRangeSet crs(crs_s);
    // DRAM field: n_tiles pages of tile_bytes (interleaved across banks → parallel read)
    DeviceLocalBufferConfig dc{.page_size=tile_bytes,.buffer_type=BufferType::DRAM};
    ReplicatedBufferConfig rc{.size=field_bytes};
    auto fb=MeshBuffer::create(rc,dc,mesh.get());
    { std::vector<uint32_t> z(field_bytes/4u,0x3f800000u); EnqueueWriteMeshBuffer(cq,fb,z,false); Finish(cq); }

    Program prog=CreateProgram();
    CreateCircularBuffer(prog,crs,CircularBufferConfig(cb_size,{{24u,DataFormat::UInt32}}).set_page_size(24u,cb_size));
    // Multicast on NOC0 (RISCV_0) — matches the worker_core_from_logical_core (NOC0) coords.
    auto k=CreateKernel(prog,std::string(DENSITY_KERNEL_DIR)+"mcast_bw_dm.cpp",crs,
        DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
    CreateKernel(prog,std::string(DENSITY_KERNEL_DIR)+"v13_void_brisc.cpp",crs,
        DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
    uint32_t ready_sid=CreateSemaphore(prog,crs,0u), valid_sid=CreateSemaphore(prog,crs,0u);
    auto c0=nocc[0]; uint32_t fa=(uint32_t)fb->address();
    for(uint32_t c=0;c<nc_all;++c){
        std::vector<uint32_t> a={c,nc_all,(c==0u)?1u:0u,tile_bytes,n_tiles,repeats,fa,dst_off,srcbuf_off,
            (uint32_t)c0.x,(uint32_t)c0.y,ready_sid,valid_sid,(uint32_t)rects.size()};
        for(auto&r:rects){ bool self=(nocc[c].x>=r.xs&&nocc[c].x<=r.xe&&nocc[c].y>=r.ys&&nocc[c].y<=r.ye);
            a.push_back(r.xs);a.push_back(r.ys);a.push_back(r.xe);a.push_back(r.ye);a.push_back(self?r.nd-1u:r.nd); }
        SetRuntimeArgs(prog,k,lc[c],a);
    }
    MeshWorkload wl; wl.add_program(dr,std::move(prog));
    fprintf(stderr,"[mcast_bw] launching warmup...\n");
    EnqueueMeshWorkload(cq,wl,false); Finish(cq);   // JIT warmup
    fprintf(stderr,"[mcast_bw] warmup done; timed loop x%d...\n",n_iters);
    std::vector<double> tv;
    for(int i=0;i<n_iters;++i){ auto t=hrclock::now(); EnqueueMeshWorkload(cq,wl,false); Finish(cq); tv.push_back(ms_since(t)); fprintf(stderr,"[mcast_bw]  iter %d done %.3f ms\n",i,tv.back()); }
    std::sort(tv.begin(),tv.end()); double med=tv[tv.size()/2];
    // per-launch wall covers `repeats` full field streams. per-field-cast = med/repeats.
    double per_cast_ms=med/repeats;
    double bw=(double)field_bytes/(per_cast_ms*1e-3)/1e9;   // GB/s delivered to all cores
    printf("[mcast_bw] launch_median=%.3f ms  per_field_cast=%.4f ms  effective_BW=%.1f GB/s  (field %.1fMB to %u cores)\n",
           med,per_cast_ms,bw,(double)field_bytes/1048576.0,nc_all);
    printf("MCAST_BW,%u,%u,%.2f,%.4f,%.1f\n",tile_bytes,n_tiles,(double)field_bytes/1048576.0,per_cast_ms,bw);
    return 0;
}
