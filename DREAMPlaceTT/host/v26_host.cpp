// SPDX-License-Identifier: Apache-2.0
//
// V26 EF backward — SFPU fp32 gather variant, end-to-end per-iter.
// Field stays fp32 (NO int conversion — fixes Path A's 63ms lroundf). BRISC
// direct-L1 gather → SFPU fp32 weighted sum → writer. Reports grouping(host) +
// chip(3-kernel pipeline) vs CPU EF (~5-7 ms).
//
// CLI: v26 [H W ncells]

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
using hrclock = std::chrono::high_resolution_clock;
template<class T> static double ms_since(T t0){return std::chrono::duration<double,std::milli>(hrclock::now()-t0).count();}
#ifndef DENSITY_KERNEL_DIR
#define DENSITY_KERNEL_DIR "kernels/"
#endif

int main(int argc,char**argv){
    uint32_t H=2048,W=2048,ncells=540000;
    if(argc>=4){H=atoi(argv[1]);W=atoi(argv[2]);ncells=atoi(argv[3]);}
    auto mesh=MeshDevice::create_unit_mesh(0); auto&cq=mesh->mesh_command_queue();
    auto grid=mesh->compute_with_storage_grid_size(); const int nc=(int)(grid.x*grid.y);
    std::vector<CoreCoord> ccs; for(uint32_t y=0;y<grid.y;++y)for(uint32_t x=0;x<grid.x;++x)ccs.push_back({x,y});
    std::set<CoreRange> crs; for(auto c:ccs)crs.insert(CoreRange{c,c}); CoreRangeSet all_crs(crs);

    const uint32_t rpc=(H+nc-1)/nc, band_rows=rpc+1, band_floats=band_rows*W*2u, band_bytes=band_floats*4u;
    std::mt19937 rng(12345); std::uniform_real_distribution<float> uf(-1.f,1.f),u01(0.f,1.f);

    std::vector<float> gfield((size_t)H*W*2u); for(auto&v:gfield)v=uf(rng);
    struct Cell{uint32_t byl,bxl;float wnw,wne,wsw,wse;};
    std::vector<Cell> cellv(ncells);
    for(uint32_t i=0;i<ncells;++i){uint32_t byl=(uint32_t)(rng()%(H-2)),bxl=(uint32_t)(rng()%(W-2));
        float fy=u01(rng),fx=u01(rng),r=0.5f+u01(rng);
        cellv[i]={byl,bxl,r*(1-fy)*(1-fx),r*(1-fy)*fx,r*fy*(1-fx),r*fy*fx};}

    // ── PER-ITER GROUPING (host, timed): fp32 field-shard + bucket + records ──
    std::vector<float> field_b; std::vector<uint32_t> cells; std::vector<uint32_t> take(nc,0);
    std::vector<std::vector<uint32_t>> bucket(nc); uint32_t max_cells=0,max_batches=0;
    double group_ms=0, shard_ms=0, bucket_ms=0;
    {
      // (a) field shard — microbench artifact (free on-chip via LOADBAND)
      auto ts=hrclock::now();
      field_b.assign((size_t)nc*band_floats,0.f);
      for(int c=0;c<nc;++c){float*band=field_b.data()+(size_t)c*band_floats;
        for(uint32_t r=0;r<band_rows;++r){uint32_t gr=(uint32_t)c*rpc+r; if(gr>=H)break;
          std::memcpy(band+(size_t)r*W*2u, gfield.data()+(size_t)gr*W*2u, (size_t)W*2u*4u);}}
      shard_ms=ms_since(ts);
      // (b) cell bucketing — the real grouping (the candidate on-chip pass)
      auto tb=hrclock::now();
      for(uint32_t i=0;i<ncells;++i){uint32_t core=cellv[i].byl/rpc; if(core>=(uint32_t)nc)core=nc-1; bucket[core].push_back(i);}
      for(int c=0;c<nc;++c){take[c]=(uint32_t)bucket[c].size(); max_cells=std::max(max_cells,take[c]);}
      max_batches=(max_cells+1023)/1024;
      const uint32_t mc=max_batches*1024; max_cells=mc;            // pad to whole tiles
      cells.assign((size_t)nc*mc*6u,0u);
      union{uint32_t u;float f;}cv;
      for(int c=0;c<nc;++c)for(uint32_t j=0;j<take[c];++j){const Cell&cl=cellv[bucket[c][j]];
        uint32_t byl_local=cl.byl-(uint32_t)c*rpc; size_t b=((size_t)c*mc+j)*6u;
        cells[b+0]=cl.bxl; cells[b+1]=byl_local;
        cv.f=cl.wnw;cells[b+2]=cv.u; cv.f=cl.wne;cells[b+3]=cv.u; cv.f=cl.wsw;cells[b+4]=cv.u; cv.f=cl.wse;cells[b+5]=cv.u;}
      bucket_ms=ms_since(tb);
      group_ms=shard_ms+bucket_ms;
    }
    printf("[v26] grouping split: field-shard(artifact)=%.3f ms  cell-bucket(real)=%.3f ms\n", shard_ms, bucket_ms);
    const uint32_t mc=max_cells, cells_pg=mc*6u*4u, out_pg=max_batches*1024u*4u;
    printf("[v26] H=%u W=%u ncells=%u nc=%d band=%.0fKB mc=%u nbat_max=%u  GROUPING(host,fp32)=%.3f ms\n",
           H,W,ncells,nc,band_bytes/1024.0,mc,max_batches,group_ms);

    auto mb=[&](uint64_t b,uint32_t pg){DeviceLocalBufferConfig c{.page_size=pg,.buffer_type=BufferType::DRAM};ReplicatedBufferConfig r{.size=b};return MeshBuffer::create(r,c,mesh.get());};
    auto fbuf=mb((uint64_t)nc*band_bytes,band_bytes),cbuf=mb((uint64_t)nc*cells_pg,cells_pg);
    auto gxbuf=mb((uint64_t)nc*out_pg,out_pg),gybuf=mb((uint64_t)nc*out_pg,out_pg);
    EnqueueWriteMeshBuffer(cq,fbuf,field_b,false); EnqueueWriteMeshBuffer(cq,cbuf,cells,false); Finish(cq);

    using tt::CBIndex; Program prog=CreateProgram();
    auto cbf=[&](uint32_t idx,uint32_t bytes,uint32_t slots){CircularBufferConfig c(slots*bytes,{{(CBIndex)idx,DataFormat::Float32}});c.set_page_size((CBIndex)idx,bytes);CreateCircularBuffer(prog,all_crs,c);};
    const uint32_t chunk_batches = 4;               // 4096 cells/chunk → 96KB L1 (bounds cell L1 use)
    const uint32_t chunk_bytes = chunk_batches*1024u*6u*4u;
    for(uint32_t i=0;i<12;++i) cbf(i,4096,2);      // c_0..c_11 field+weight tiles, double-buffered
    cbf(16,4096,2); cbf(17,4096,2);                 // gx,gy out
    cbf(24,band_bytes,1); cbf(25,chunk_bytes,1);    // resident field band, ONE cell chunk (L1-bounded)
    std::vector<UnpackToDestMode> um(64,UnpackToDestMode::Default); for(uint32_t i=0;i<12;++i)um[i]=UnpackToDestMode::UnpackToDestFp32;
    auto bk=CreateKernel(prog,std::string(DENSITY_KERNEL_DIR)+"v26_brisc.cpp",all_crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
    auto wk=CreateKernel(prog,std::string(DENSITY_KERNEL_DIR)+"v26_writer.cpp",all_crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
    auto ck=CreateKernel(prog,std::string(DENSITY_KERNEL_DIR)+"v26_compute.cpp",all_crs,ComputeConfig{.fp32_dest_acc_en=true,.unpack_to_dest_mode=um,.math_approx_mode=false});
    uint32_t fa=(uint32_t)fbuf->address(),ca=(uint32_t)cbuf->address(),gxa=(uint32_t)gxbuf->address(),gya=(uint32_t)gybuf->address();
    for(int c=0;c<nc;++c){uint32_t nb=(take[c]+1023)/1024;
        SetRuntimeArgs(prog,bk,ccs[c],{(uint32_t)c,take[c],band_bytes,W,fa,ca,cells_pg,nb,chunk_batches});
        SetRuntimeArgs(prog,ck,ccs[c],{nb});
        SetRuntimeArgs(prog,wk,ccs[c],{(uint32_t)c,nb,gxa,gya,out_pg});}
    MeshWorkload wl; auto dr=MeshCoordinateRange{mesh->shape()}; wl.add_program(dr,std::move(prog));
    auto run=[&]{EnqueueMeshWorkload(cq,wl,false);Finish(cq);};
    run(); const int N=7; std::vector<double> ms; for(int i=0;i<N;++i){auto t=hrclock::now();run();ms.push_back(ms_since(t));}
    std::sort(ms.begin(),ms.end()); double chip_ms=ms[N/2];

    std::vector<float> gxo((size_t)nc*mc,0.f),gyo((size_t)nc*mc,0.f);
    EnqueueReadMeshBuffer(cq,gxo,gxbuf,true); EnqueueReadMeshBuffer(cq,gyo,gybuf,true);
    double max_abs=0,refn=0,difn=0; size_t nbad=0; const float*gf=gfield.data();
    for(int c=0;c<nc;++c)for(uint32_t j=0;j<take[c];++j){const Cell&cl=cellv[bucket[c][j]];
        uint32_t r0=cl.byl*W+cl.bxl,inw=r0*2,ine=(r0+1)*2,isw=(r0+W)*2,ise=(r0+W+1)*2;
        double gxr=(double)cl.wnw*gf[inw]+cl.wne*gf[ine]+cl.wsw*gf[isw]+cl.wse*gf[ise];
        double gyr=(double)cl.wnw*gf[inw+1]+cl.wne*gf[ine+1]+cl.wsw*gf[isw+1]+cl.wse*gf[ise+1];
        size_t o=(size_t)c*mc+j; double dx=std::fabs((double)gxo[o]-gxr),dy=std::fabs((double)gyo[o]-gyr);
        max_abs=std::max({max_abs,dx,dy}); refn+=gxr*gxr+gyr*gyr; difn+=dx*dx+dy*dy;
        if(dx>1e-3*std::max(1.0,std::fabs(gxr)))nbad++; if(dy>1e-3*std::max(1.0,std::fabs(gyr)))nbad++;}
    double rel_l2=refn>0?std::sqrt(difn/refn):0;
    printf("[v26] chip(3-kernel)=%.3f ms  grouping(host)=%.3f ms  TOTAL=%.3f ms  (CPU EF ~5-7 ms)\n",chip_ms,group_ms,chip_ms+group_ms);
    printf("[v26] accuracy: max_abs=%.6g rel_l2=%.6g nbad=%zu\n",max_abs,rel_l2,nbad);
    printf("V26_RESULT,%u,%u,%u,%.3f,%.3f,%.3f,%.6g\n",H,W,ncells,chip_ms,group_ms,chip_ms+group_ms,rel_l2);
    return 0;
}
