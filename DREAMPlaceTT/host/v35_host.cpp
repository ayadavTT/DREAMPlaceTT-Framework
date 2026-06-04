// SPDX-License-Identifier: Apache-2.0
//
// V35 EF backward — Forward-Grouped Halo-Tile gather (microbench).
//
// Goal: a SINGLE density-backward that is fast + robust at ALL grids, cell
// counts, and spatial distributions (uniform OR clustered), with no sort, no
// record-explosion, no host in the per-iter loop, and the MAC on the SFPU.
//
// Structure (reuses the V28 multi-bin SFPU gather kernels verbatim):
//   * The field is partitioned into FIXED-WIDTH column TILES sized so one tile
//     (+max_k halo) always fits L1 at every grid → no streaming (the FCCS death)
//     → a cell's whole k×h window is resident → NO SORT needed.
//   * Cells are assigned to the tile that contains their bxl. Because the tile
//     carries a +max_k halo, each cell stays WHOLE (one owner has its full
//     window) → NO record explosion (the V30 death).
//   * CLUSTERING IMMUNITY: cores are allocated to tiles in proportion to each
//     tile's cell count (V30's balance idea). A hot tile gets many helper cores,
//     each replicating that tile's SMALL band and processing an equal cell-chunk;
//     cold tiles get one core. So every core does ~equal work regardless of
//     distribution, and we only ever replicate a hot tile's small band (never
//     broadcast the whole field — the FCCS cost).
//
// This microbench computes the partition (histogram + proportional allocation)
// on the HOST — that is METADATA only (≈ntiles+nc ints), and the on-chip
// equivalent is V30 prepcount+band (separately measured). What is MEASURED here
// is the device gather+balance, validated vs the CPU multi-bin reference, on
// uniform AND clustered distributions at all grids.
//
//   CLI: v35 [N M ncells max_k max_h cluster]   cluster: 0=uniform 1=clustered
//        default adaptec1_2048-like: 2048 2048 210904 6 3 0

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

struct Cell{uint32_t bxl_g,byl,kc,hc;float ratio,px[8],py[8];};

int main(int argc,char**argv){
    uint32_t N=2048,M=2048,ncells=210904,max_k=6,max_h=3,cluster=0;
    if(argc>=6){N=atoi(argv[1]);M=atoi(argv[2]);ncells=atoi(argv[3]);max_k=atoi(argv[4]);max_h=atoi(argv[5]);}
    if(argc>=7)cluster=atoi(argv[6]);
    auto mesh=MeshDevice::create_unit_mesh(0); auto&cq=mesh->mesh_command_queue();
    auto grid=mesh->compute_with_storage_grid_size(); const int nc=(int)(grid.x*grid.y);
    std::vector<CoreCoord> ccs; for(uint32_t y=0;y<grid.y;++y)for(uint32_t x=0;x<grid.x;++x)ccs.push_back({x,y});
    std::set<CoreRange> crs; for(auto c:ccs)crs.insert(CoreRange{c,c}); CoreRangeSet all_crs(crs);

    // ── tile width: keep one band (W+halo) per plane ≤ ~350 KB so it always
    //    fits L1 (two planes resident + 384 KB cells chunk + small CBs < 1.5 MB).
    //    Also ensure ntiles ≤ nc so every nonempty tile can get a core.
    const uint32_t target_band_bytes = 350u*1024u;            // per plane
    uint32_t W=target_band_bytes/(M*4u); if(W>max_k)W-=max_k; else W=1u;
    // floor W so ntiles ≤ nc/2 → every tile gets ≥2 cores → the greedy can split
    // each tile evenly (avoids the ntiles≈nc case where some tile gets 1 core and
    // carries a full tile's load → ~2× imbalance even on a uniform distribution).
    uint32_t halfnc=(uint32_t)std::max(1,nc/2);
    if(W<((N+halfnc-1u)/halfnc)) W=(N+halfnc-1u)/halfnc;
    if(W>N)W=N; if(W<1u)W=1u;
    const uint32_t ntiles=(N+W-1u)/W;
    const uint32_t band_cols=std::min(W+max_k,N);             // +halo

    std::mt19937 rng(777); std::uniform_real_distribution<float> uf(-1,1),u01(0,1);

    // column-major field fx,fy : [col*M + row]
    std::vector<float> fx((size_t)N*M), fy((size_t)N*M);
    for(size_t i=0;i<fx.size();++i){fx[i]=uf(rng);fy[i]=uf(rng);}

    // ── generate cells with global bxl, grouped by tile ──
    std::vector<std::vector<Cell>> tcells(ntiles);
    // cluster: 0=uniform, 1=moderate (60% in ~15% cols, like random_center_init),
    //          2=extreme (90% in ~4% cols — stress the balancer's worst case).
    const float cfrac = (cluster>=2)?0.90f:0.60f;
    const float cwid  = (cluster>=2)?0.020f:0.075f;   // half-width as frac of N
    auto gen_bxl=[&]()->uint32_t{
        if(cluster){
            if(u01(rng)<cfrac){
                double c=0.5*N, s=cwid*N; double v=c+s*(uf(rng)+uf(rng)+uf(rng));
                if(v<0)v=0; if(v>(double)(N-1))v=N-1; return (uint32_t)v;
            }
        }
        return (uint32_t)(u01(rng)*(N-1));
    };
    for(uint32_t i=0;i<ncells;++i){
        Cell cl; cl.bxl_g=gen_bxl();
        cl.byl=(uint32_t)(u01(rng)*(M-max_h-1));
        cl.kc=1+(uint32_t)(u01(rng)*max_k); if(cl.kc>max_k)cl.kc=max_k;
        cl.hc=1+(uint32_t)(u01(rng)*max_h); if(cl.hc>max_h)cl.hc=max_h;
        if(cl.bxl_g+cl.kc>N)cl.kc=N-cl.bxl_g;
        if(cl.byl+cl.hc>M)cl.hc=M-cl.byl;
        cl.ratio=0.5f+u01(rng);
        for(uint32_t k=0;k<8;++k)cl.px[k]=(k<cl.kc)?(0.2f+u01(rng)):0.f;
        for(uint32_t h=0;h<8;++h)cl.py[h]=(h<cl.hc)?(0.2f+u01(rng)):0.f;
        tcells[cl.bxl_g/W].push_back(cl);
    }

    // ── proportional core allocation (V30 balance): cores per tile ∝ cell count,
    //    every nonempty tile ≥1 core, sum = nc (largest-remainder). ──
    std::vector<uint32_t> cnt(ntiles); uint32_t nonempty=0;
    for(uint32_t t=0;t<ntiles;++t){cnt[t]=(uint32_t)tcells[t].size(); if(cnt[t])++nonempty;}
    // GREEDY MAKESPAN: each nonempty tile starts with 1 core (its cells must be
    // processed somewhere); every remaining core goes to the tile with the worst
    // current per-core load cnt/alloc → directly minimizes max-cells-per-core
    // (the critical path). O(nc·ntiles), trivial; clustering-robust by construction.
    std::vector<uint32_t> alloc(ntiles,0);
    if(nonempty){
        uint32_t used=0; for(uint32_t t=0;t<ntiles;++t)if(cnt[t]){alloc[t]=1;++used;}
        while(used<(uint32_t)nc){
            uint32_t best=ntiles; double bestload=-1.0;
            for(uint32_t t=0;t<ntiles;++t)if(cnt[t]){double load=(double)cnt[t]/(double)alloc[t]; if(load>bestload){bestload=load;best=t;}}
            if(best==ntiles)break; alloc[best]++; ++used;
        }
    }

    // ── lay cells out per CORE: tile t's cells split into alloc[t] equal chunks ──
    std::vector<uint32_t> take(nc,0), core_col0(nc,0), core_vcols(nc,1);
    std::vector<std::vector<Cell>> ccell(nc);
    uint32_t cur=0;
    for(uint32_t t=0;t<ntiles;++t){
        uint32_t a=alloc[t]; if(!a)continue; uint32_t n=cnt[t];
        uint32_t col0=t*W, vcols=(col0<N)?std::min(band_cols,N-col0):1u;
        for(uint32_t s=0;s<a;++s){
            uint32_t lo=(uint64_t)n*s/a, hi=(uint64_t)n*(s+1)/a; int c=cur++;
            core_col0[c]=col0; core_vcols[c]=vcols;
            for(uint32_t j=lo;j<hi;++j)ccell[c].push_back(tcells[t][j]);
            take[c]=(uint32_t)ccell[c].size();
        }
    }
    uint32_t max_cells=0,sum_cells=0; for(int c=0;c<nc;++c){max_cells=std::max(max_cells,take[c]);sum_cells+=take[c];}
    double imbal=(sum_cells>0)?((double)max_cells/((double)sum_cells/nc)):1.0;
    const uint32_t max_batches=std::max(1u,(max_cells+1023)/1024), mc=max_batches*1024;
    const uint32_t REC=24u, cells_pg=mc*REC*4u, out_pg=max_batches*1024u*4u, field_pg=(uint32_t)((uint64_t)N*M*4u);

    // pack records: [0]bxl_loc [1]byl [2]kc [3]hc [4]ratio [5]pad [6..13]px [14..21]py
    std::vector<uint32_t> cells((size_t)nc*mc*REC,0u); union{uint32_t u;float f;}cv;
    for(int c=0;c<nc;++c)for(uint32_t j=0;j<take[c];++j){const Cell&cl=ccell[c][j];size_t b=((size_t)c*mc+j)*REC;
        uint32_t bxl_loc=cl.bxl_g-core_col0[c]; uint32_t kc=cl.kc;
        if(bxl_loc+kc>band_cols)kc=band_cols-bxl_loc;            // clamp window into the band
        cells[b+0]=bxl_loc;cells[b+1]=cl.byl;cells[b+2]=kc;cells[b+3]=cl.hc;cv.f=cl.ratio;cells[b+4]=cv.u;
        for(uint32_t k=0;k<8;++k){cv.f=cl.px[k];cells[b+6+k]=cv.u;}
        for(uint32_t h=0;h<8;++h){cv.f=cl.py[h];cells[b+14+h]=cv.u;}}
    printf("[v35] N=%u M=%u ncells=%u nc=%d cluster=%u W=%u ntiles=%u band_cols=%u band=%.0fKB mc=%u max_cells=%u imbal=%.2fx\n",
           N,M,ncells,nc,cluster,W,ntiles,band_cols,band_cols*M*4u/1024.0,mc,max_cells,imbal);

    auto mb=[&](uint64_t b,uint32_t pg){DeviceLocalBufferConfig c{.page_size=pg,.buffer_type=BufferType::DRAM};ReplicatedBufferConfig r{.size=b};return MeshBuffer::create(r,c,mesh.get());};
    auto fxb=mb((uint64_t)field_pg,field_pg),fyb=mb((uint64_t)field_pg,field_pg),cb=mb((uint64_t)nc*cells_pg,cells_pg);
    auto gxb=mb((uint64_t)nc*out_pg,out_pg),gyb=mb((uint64_t)nc*out_pg,out_pg);
    EnqueueWriteMeshBuffer(cq,fxb,fx,false);EnqueueWriteMeshBuffer(cq,fyb,fy,false);EnqueueWriteMeshBuffer(cq,cb,cells,false);Finish(cq);

    using tt::CBIndex; Program prog=CreateProgram();
    auto cbf=[&](uint32_t idx,uint32_t bytes,uint32_t slots){CircularBufferConfig c(slots*bytes,{{(CBIndex)idx,DataFormat::Float32}});c.set_page_size((CBIndex)idx,bytes);CreateCircularBuffer(prog,all_crs,c);};
    const uint32_t chunk_batches=4, chunk_bytes=chunk_batches*1024u*REC*4u;
    cbf(0,4096,max_k); cbf(1,4096,max_h); cbf(2,4096,2); cbf(3,4096,2); cbf(4,4096,2);  // px,py,fx,fy,ratio
    cbf(5,4096,2);                                                   // base_idx (1024 u32) BRISC→NCRISC
    cbf(16,4096,2); cbf(17,4096,2);                                  // gx,gy
    const uint32_t fld_bytes=band_cols*M*4u + (max_h+8u)*4u;         // +pad: unclamped idx stays in-bounds
    cbf(24,fld_bytes,1); cbf(25,chunk_bytes,1); cbf(26,fld_bytes,1); // fieldx,cells,fieldy
    std::vector<UnpackToDestMode> um(64,UnpackToDestMode::Default);
    for(uint32_t i=0;i<5;++i)um[i]=UnpackToDestMode::UnpackToDestFp32; um[16]=um[17]=UnpackToDestMode::UnpackToDestFp32;
    auto bk=CreateKernel(prog,std::string(DENSITY_KERNEL_DIR)+"v28_brisc.cpp",all_crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
    auto wk=CreateKernel(prog,std::string(DENSITY_KERNEL_DIR)+"v28_ncrisc.cpp",all_crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
    auto ck=CreateKernel(prog,std::string(DENSITY_KERNEL_DIR)+"v28_compute.cpp",all_crs,ComputeConfig{.fp32_dest_acc_en=true,.unpack_to_dest_mode=um,.math_approx_mode=false});
    uint32_t fxa=(uint32_t)fxb->address(),fya=(uint32_t)fyb->address(),ca=(uint32_t)cb->address(),gxa=(uint32_t)gxb->address(),gya=(uint32_t)gyb->address();
    for(int c=0;c<nc;++c){uint32_t nb=std::max(0u,(take[c]+1023)/1024); uint32_t col0=core_col0[c],vcols=core_vcols[c];
        SetRuntimeArgs(prog,bk,ccs[c],{(uint32_t)c,take[c],band_cols,M,max_k,max_h,fxa,fya,field_pg,ca,cells_pg,nb,chunk_batches,col0,vcols});
        SetRuntimeArgs(prog,ck,ccs[c],{nb,max_k,max_h});
        SetRuntimeArgs(prog,wk,ccs[c],{(uint32_t)c,nb,M,max_k,max_h,fya,field_pg,col0,vcols,gxa,gya,out_pg,band_cols});}
    MeshWorkload wl; auto dr=MeshCoordinateRange{mesh->shape()}; wl.add_program(dr,std::move(prog));
    auto run=[&]{EnqueueMeshWorkload(cq,wl,false);Finish(cq);};
    run(); const int RN=7; std::vector<double> ms; for(int i=0;i<RN;++i){auto t=hrclock::now();run();ms.push_back(ms_since(t));}
    std::sort(ms.begin(),ms.end()); double chip_ms=ms[RN/2];

    std::vector<float> gxo((size_t)nc*mc,0.f),gyo((size_t)nc*mc,0.f);
    EnqueueReadMeshBuffer(cq,gxo,gxb,true);EnqueueReadMeshBuffer(cq,gyo,gyb,true);
    double refn=0,difn=0,max_abs=0; size_t nbad=0;
    for(int c=0;c<nc;++c)for(uint32_t j=0;j<take[c];++j){const Cell&cl=ccell[c][j];
        double ax=0,ay=0;
        for(uint32_t k=0;k<cl.kc;++k)for(uint32_t h=0;h<cl.hc;++h){
            size_t idx=(size_t)(cl.bxl_g+k)*M+(cl.byl+h); double w=(double)cl.px[k]*cl.py[h];
            ax+=w*fx[idx]; ay+=w*fy[idx];}
        double gxr=ax*cl.ratio, gyr=ay*cl.ratio;
        size_t o=(size_t)c*mc+j; double dx=std::fabs(gxo[o]-gxr),dy=std::fabs(gyo[o]-gyr);
        max_abs=std::max({max_abs,dx,dy}); refn+=gxr*gxr+gyr*gyr; difn+=dx*dx+dy*dy;
        if(dx>1e-3*std::max(1.0,std::fabs(gxr)))nbad++; if(dy>1e-3*std::max(1.0,std::fabs(gyr)))nbad++;}
    double rel_l2=refn>0?std::sqrt(difn/refn):0;
    printf("[v35] chip MEDIAN=%.3f ms  imbal=%.2fx  (multi-bin, CPU EF ~5-7 ms)\n",chip_ms,imbal);
    printf("[v35] accuracy: max_abs=%.6g rel_l2=%.6g nbad=%zu  %s\n",max_abs,rel_l2,nbad,(rel_l2<1e-4&&nbad==0)?"OK":"CHECK");
    printf("V35_RESULT,%u,%u,%u,%u,%u,%u,%.3f,%.6g,%.2f\n",N,M,ncells,max_k,max_h,cluster,chip_ms,rel_l2,imbal);
    return 0;
}
