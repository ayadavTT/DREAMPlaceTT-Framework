// SPDX-License-Identifier: Apache-2.0
//
// V28 EF backward — MULTI-BIN L1-resident gather microbench (correct at all
// grids; generalizes V26's 4-corner). Field column-major (page=x-column), each
// core owns a contiguous column-band (cpc+max_k halo) resident in L1; per cell
// the gather reads field[(bin_xl+k)·M+(bin_yl+h)] by direct L1 load; SFPU does
// Σ px[k]·py[h]·f ·ratio. Validates vs CPU multi-bin + times.
//   CLI: v28 [N M ncells max_k max_h]   (default adaptec1_2048-like: 2048 2048 210904 6 3)

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
    uint32_t N=2048,M=2048,ncells=210904,max_k=6,max_h=3;
    if(argc>=6){N=atoi(argv[1]);M=atoi(argv[2]);ncells=atoi(argv[3]);max_k=atoi(argv[4]);max_h=atoi(argv[5]);}
    auto mesh=MeshDevice::create_unit_mesh(0); auto&cq=mesh->mesh_command_queue();
    auto grid=mesh->compute_with_storage_grid_size(); const int nc=(int)(grid.x*grid.y);
    std::vector<CoreCoord> ccs; for(uint32_t y=0;y<grid.y;++y)for(uint32_t x=0;x<grid.x;++x)ccs.push_back({x,y});
    std::set<CoreRange> crs; for(auto c:ccs)crs.insert(CoreRange{c,c}); CoreRangeSet all_crs(crs);

    const uint32_t cpc_col=(N+nc-1)/nc;                  // columns per core
    const uint32_t band_cols=std::min(cpc_col+max_k, N); // +halo
    std::mt19937 rng(777); std::uniform_real_distribution<float> uf(-1,1),u01(0,1);

    // column-major field fx,fy : [col*M + row]
    std::vector<float> fx((size_t)N*M), fy((size_t)N*M);
    for(size_t i=0;i<fx.size();++i){fx[i]=uf(rng);fy[i]=uf(rng);}

    // per-core column-bucketed cells with variable spans
    struct Cell{uint32_t bxl_loc,byl,kc,hc;float ratio,px[8],py[8];uint32_t bxl_g;};
    std::vector<std::vector<Cell>> bucket(nc); std::uniform_int_distribution<int> coredist(0,nc-1);
    std::vector<uint32_t> take(nc,0); uint32_t max_cells=0;
    for(uint32_t i=0;i<ncells;++i){
        int c=coredist(rng); uint32_t cstart=(uint32_t)c*cpc_col; if(cstart>=N){c=nc-1;cstart=(uint32_t)c*cpc_col;}
        uint32_t cend=std::min(cstart+cpc_col,N); if(cend<=cstart)continue;
        Cell cl; uint32_t span=cend-cstart;
        cl.bxl_g=cstart + (uint32_t)(u01(rng)*(span-1)); cl.bxl_loc=cl.bxl_g-cstart;
        cl.byl=(uint32_t)(u01(rng)*(M-max_h-1));
        cl.kc=1+(uint32_t)(u01(rng)*max_k); if(cl.kc>max_k)cl.kc=max_k;
        cl.hc=1+(uint32_t)(u01(rng)*max_h); if(cl.hc>max_h)cl.hc=max_h;
        // clamp span to stay within field bounds (mirrors DREAMPlace bin_xh/bin_yh clamp)
        if(cl.bxl_g+cl.kc>N)cl.kc=N-cl.bxl_g;
        if(cl.byl+cl.hc>M)cl.hc=M-cl.byl;
        if(cl.bxl_loc+cl.kc>band_cols)cl.kc=band_cols-cl.bxl_loc;
        cl.ratio=0.5f+u01(rng);
        for(uint32_t k=0;k<8;++k)cl.px[k]=(k<cl.kc)?(0.2f+u01(rng)):0.f;
        for(uint32_t h=0;h<8;++h)cl.py[h]=(h<cl.hc)?(0.2f+u01(rng)):0.f;
        bucket[c].push_back(cl);
    }
    for(int c=0;c<nc;++c){take[c]=(uint32_t)bucket[c].size();max_cells=std::max(max_cells,take[c]);}
    const uint32_t max_batches=(max_cells+1023)/1024, mc=max_batches*1024;
    const uint32_t REC=24u, cells_pg=mc*REC*4u, out_pg=max_batches*1024u*4u, field_pg=(uint32_t)((uint64_t)N*M*4u);

    // pack records: [0]bxl_loc [1]byl [2]kc [3]hc [4]ratio [5]pad [6..13]px [14..21]py
    std::vector<uint32_t> cells((size_t)nc*mc*REC,0u); union{uint32_t u;float f;}cv;
    for(int c=0;c<nc;++c)for(uint32_t j=0;j<take[c];++j){const Cell&cl=bucket[c][j];size_t b=((size_t)c*mc+j)*REC;
        cells[b+0]=cl.bxl_loc;cells[b+1]=cl.byl;cells[b+2]=cl.kc;cells[b+3]=cl.hc;cv.f=cl.ratio;cells[b+4]=cv.u;
        for(uint32_t k=0;k<8;++k){cv.f=cl.px[k];cells[b+6+k]=cv.u;}
        for(uint32_t h=0;h<8;++h){cv.f=cl.py[h];cells[b+14+h]=cv.u;}}
    printf("[v28] N=%u M=%u ncells=%u nc=%d cpc_col=%u band_cols=%u max_k=%u max_h=%u band=%.0fKB mc=%u\n",
           N,M,ncells,nc,cpc_col,band_cols,max_k,max_h,band_cols*M*4u/1024.0,mc);

    auto mb=[&](uint64_t b,uint32_t pg){DeviceLocalBufferConfig c{.page_size=pg,.buffer_type=BufferType::DRAM};ReplicatedBufferConfig r{.size=b};return MeshBuffer::create(r,c,mesh.get());};
    auto fxb=mb((uint64_t)field_pg,field_pg),fyb=mb((uint64_t)field_pg,field_pg),cb=mb((uint64_t)nc*cells_pg,cells_pg);
    auto gxb=mb((uint64_t)nc*out_pg,out_pg),gyb=mb((uint64_t)nc*out_pg,out_pg);
    EnqueueWriteMeshBuffer(cq,fxb,fx,false);EnqueueWriteMeshBuffer(cq,fyb,fy,false);EnqueueWriteMeshBuffer(cq,cb,cells,false);Finish(cq);

    using tt::CBIndex; Program prog=CreateProgram();
    auto cbf=[&](uint32_t idx,uint32_t bytes,uint32_t slots){CircularBufferConfig c(slots*bytes,{{(CBIndex)idx,DataFormat::Float32}});c.set_page_size((CBIndex)idx,bytes);CreateCircularBuffer(prog,all_crs,c);};
    const uint32_t chunk_batches=4, chunk_bytes=chunk_batches*1024u*REC*4u;
    cbf(0,4096,max_k); cbf(1,4096,max_h); cbf(2,4096,2); cbf(3,4096,2); cbf(4,4096,2);  // px,py,fx,fy,ratio
    cbf(5,4096,2);                                                   // base_idx (1024 u32) BRISC→NCRISC, double-buffered
    cbf(16,4096,2); cbf(17,4096,2);                                  // gx,gy
    const uint32_t fld_bytes=band_cols*M*4u + (max_h+8u)*4u;         // +pad: unclamped idx stays in-bounds
    cbf(24,fld_bytes,1); cbf(25,chunk_bytes,1); cbf(26,fld_bytes,1); // fieldx,cells,fieldy
    std::vector<UnpackToDestMode> um(64,UnpackToDestMode::Default);
    for(uint32_t i=0;i<5;++i)um[i]=UnpackToDestMode::UnpackToDestFp32; um[16]=um[17]=UnpackToDestMode::UnpackToDestFp32;
    auto bk=CreateKernel(prog,std::string(DENSITY_KERNEL_DIR)+"v28_brisc.cpp",all_crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
    auto wk=CreateKernel(prog,std::string(DENSITY_KERNEL_DIR)+"v28_ncrisc.cpp",all_crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
    auto ck=CreateKernel(prog,std::string(DENSITY_KERNEL_DIR)+"v28_compute.cpp",all_crs,ComputeConfig{.fp32_dest_acc_en=true,.unpack_to_dest_mode=um,.math_approx_mode=false});
    uint32_t fxa=(uint32_t)fxb->address(),fya=(uint32_t)fyb->address(),ca=(uint32_t)cb->address(),gxa=(uint32_t)gxb->address(),gya=(uint32_t)gyb->address();
    for(int c=0;c<nc;++c){uint32_t nb=(take[c]+1023)/1024;
        uint32_t col0=(uint32_t)c*cpc_col; uint32_t vcols=(col0<N)?std::min(band_cols,N-col0):1u;
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
    for(int c=0;c<nc;++c)for(uint32_t j=0;j<take[c];++j){const Cell&cl=bucket[c][j];
        double ax=0,ay=0;
        for(uint32_t k=0;k<cl.kc;++k)for(uint32_t h=0;h<cl.hc;++h){
            size_t idx=(size_t)(cl.bxl_g+k)*M+(cl.byl+h); double w=(double)cl.px[k]*cl.py[h];
            ax+=w*fx[idx]; ay+=w*fy[idx];}
        double gxr=ax*cl.ratio, gyr=ay*cl.ratio;
        size_t o=(size_t)c*mc+j; double dx=std::fabs(gxo[o]-gxr),dy=std::fabs(gyo[o]-gyr);
        max_abs=std::max({max_abs,dx,dy}); refn+=gxr*gxr+gyr*gyr; difn+=dx*dx+dy*dy;
        if(dx>1e-3*std::max(1.0,std::fabs(gxr)))nbad++; if(dy>1e-3*std::max(1.0,std::fabs(gyr)))nbad++;}
    double rel_l2=refn>0?std::sqrt(difn/refn):0;
    printf("[v28] chip MEDIAN=%.3f ms  (multi-bin, CPU EF ~5-7 ms)\n",chip_ms);
    printf("[v28] accuracy: max_abs=%.6g rel_l2=%.6g nbad=%zu  %s\n",max_abs,rel_l2,nbad,(rel_l2<1e-4&&nbad==0)?"OK":"CHECK");
    printf("V28_RESULT,%u,%u,%u,%u,%u,%.3f,%.6g\n",N,M,ncells,max_k,max_h,chip_ms,rel_l2);
    return 0;
}
