// SPDX-License-Identifier: Apache-2.0
//
// V29 EF backward — FULL on-chip pipeline microbench: prep+bucket → gather+
// compute → scatter-grad. Validates the whole backward vs a CPU electric_force
// reference (same prep math) + times each pass. No per-iter host compute (host
// only reads the tiny counts matrix to set the gather loop bound).
//   CLI: v29 [N M ncells]   (default adaptec1_2048-like: 2048 2048 210904)

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
static constexpr uint32_t MAXKH=8;

int main(int argc,char**argv){
    uint32_t N=2048,M=2048,ncells=210904;
    if(argc>=4){N=atoi(argv[1]);M=atoi(argv[2]);ncells=atoi(argv[3]);}
    auto mesh=MeshDevice::create_unit_mesh(0); auto&cq=mesh->mesh_command_queue();
    auto grid=mesh->compute_with_storage_grid_size(); const int nc=(int)(grid.x*grid.y);
    std::vector<CoreCoord> ccs; for(uint32_t y=0;y<grid.y;++y)for(uint32_t x=0;x<grid.x;++x)ccs.push_back({x,y});
    std::set<CoreRange> crs; for(auto c:ccs)crs.insert(CoreRange{c,c}); CoreRangeSet all_crs(crs);

    const float xl=0.f,yl=0.f,xh=1000.f,yh=1000.f;
    const float bsx=(xh-xl)/N, bsy=(yh-yl)/M, inv_bsx=1.f/bsx, inv_bsy=1.f/bsy;
    const uint32_t cpc_x=(N+nc-1)/nc, band_cols=std::min(cpc_x+MAXKH,N);
    const uint32_t cpc_cell=(ncells+nc-1)/nc;

    std::mt19937 rng(2029); std::uniform_real_distribution<float> u01(0,1);
    // field column-major [x-bin][y-bin] = [N][M] (matches on-chip ROW_MAJOR [num_bins_x,num_bins_y])
    std::vector<float> fx((size_t)N*M),fy((size_t)N*M);
    for(size_t i=0;i<fx.size();++i){fx[i]=2.f*u01(rng)-1.f;fy[i]=2.f*u01(rng)-1.f;}

    // cells: pos + constants. nsx ~ U(0.5,6)·bsx, nsy ~ 3·bsy (adaptec-like spans)
    std::vector<float> in((size_t)ncells*16u,0.f); union{uint32_t u;float f;}cv;
    struct Ref{int bxl,byl;uint32_t kc,hc;float ratio,px[8],py[8];};
    std::vector<Ref> ref(ncells);
    // host prep (CPU) — produces reference + cap/n_batches sizing (bit-matches kernel)
    std::vector<uint32_t> hcnt((size_t)nc*nc,0); uint32_t maxc=0,max_k=1,max_h=1;
    for(uint32_t i=0;i<ncells;++i){
        float nsx=(0.5f+5.5f*u01(rng))*bsx, nsy=3.f*bsy;
        float ox=0.f,oy=0.f,ratio=0.5f+u01(rng);
        float px_pos=xl+u01(rng)*(xh-xl-nsx), py_pos=yl+u01(rng)*(yh-yl-nsy);
        in[i*16+0]=px_pos;in[i*16+1]=py_pos;in[i*16+2]=ox;in[i*16+3]=oy;in[i*16+4]=nsx;in[i*16+5]=nsy;in[i*16+6]=ratio;
        float node_x=px_pos+ox,node_y=py_pos+oy;
        int bxl=(int)((node_x-xl)*inv_bsx),bxh=(int)((node_x+nsx-xl)*inv_bsx)+1;
        int byl=(int)((node_y-yl)*inv_bsy),byh=(int)((node_y+nsy-yl)*inv_bsy)+1;
        if(bxl<0)bxl=0; if(bxh>(int)N)bxh=(int)N; if(byl<0)byl=0; if(byh>(int)M)byh=(int)M;
        uint32_t kc=(bxh>bxl)?(uint32_t)(bxh-bxl):0u; if(kc>MAXKH)kc=MAXKH;
        uint32_t hc=(byh>byl)?(uint32_t)(byh-byl):0u; if(hc>MAXKH)hc=MAXKH;
        Ref&R=ref[i]; R.bxl=bxl;R.byl=byl;R.kc=kc;R.hc=hc;R.ratio=ratio;
        float bxlo=xl+(float)bxl*bsx;
        for(uint32_t k=0;k<8;++k){float pv=0.f; if(k<kc){float bhi=bxlo+bsx,top=std::min(node_x+nsx,bhi),bot=std::max(node_x,bxlo);pv=top-bot;} R.px[k]=pv;bxlo+=bsx;}
        float bylo=yl+(float)byl*bsy;
        for(uint32_t h=0;h<8;++h){float pv=0.f; if(h<hc){float bhi=bylo+bsy,top=std::min(node_y+nsy,bhi),bot=std::max(node_y,bylo);pv=top-bot;} R.py[h]=pv;bylo+=bsy;}
        max_k=std::max(max_k,kc); max_h=std::max(max_h,hc);
        uint32_t src=i/cpc_cell; if(src>=(uint32_t)nc)src=nc-1;
        uint32_t band=(uint32_t)bxl/cpc_x; if(band>=(uint32_t)nc)band=nc-1;
        uint32_t v=++hcnt[(size_t)src*nc+band]; maxc=std::max(maxc,v);
    }
    const uint32_t cap=maxc+maxc/2+16;
    std::vector<uint32_t> nbat(nc,0); uint32_t mbatch=1; for(int b=0;b<nc;++b){uint32_t tot=0;for(int s=0;s<nc;++s)tot+=hcnt[(size_t)s*nc+b];nbat[b]=(tot+1023)/1024;mbatch=std::max(mbatch,nbat[b]);}
    const uint32_t oidx_pg=mbatch*1024u*4u;   // per-band orig_idx page (one batch = 1024 u32)
    const uint32_t route_pg=cap*128u, in_pg=(uint32_t)((uint64_t)ncells*64u+64u), cnt_pg=((nc*4u+63u)&~63u), field_pg=(uint32_t)((uint64_t)N*M*4u), grad_pg=ncells*16u;
    printf("[v29] N=%u M=%u ncells=%u nc=%d cpc_x=%u band_cols=%u max_k=%u max_h=%u cap=%u route=%.0fMB\n",
           N,M,ncells,nc,cpc_x,band_cols,max_k,max_h,cap,(double)nc*nc*route_pg/1048576.0);

    auto mb=[&](uint64_t b,uint32_t pg){DeviceLocalBufferConfig c{.page_size=pg,.buffer_type=BufferType::DRAM};ReplicatedBufferConfig r{.size=b};return MeshBuffer::create(r,c,mesh.get());};
    auto inb=mb((uint64_t)in_pg,in_pg),rb=mb((uint64_t)nc*nc*route_pg,route_pg),cb=mb((uint64_t)nc*cnt_pg,cnt_pg);
    auto fxb=mb((uint64_t)field_pg,field_pg),fyb=mb((uint64_t)field_pg,field_pg),gb=mb((uint64_t)grad_pg,grad_pg);
    auto oib=mb((uint64_t)nc*oidx_pg,oidx_pg);   // per-band orig_idx (gather writes, writer reads)
    std::vector<uint32_t> inu((size_t)ncells*16u+16u,0u); std::memcpy(inu.data(),in.data(),(size_t)ncells*16u*4u);
    EnqueueWriteMeshBuffer(cq,inb,inu,false);EnqueueWriteMeshBuffer(cq,fxb,fx,false);EnqueueWriteMeshBuffer(cq,fyb,fy,false);Finish(cq);

    using tt::CBIndex;
    auto cbf=[&](Program&p,uint32_t idx,uint32_t bytes,uint32_t slots){CircularBufferConfig c(slots*bytes,{{(CBIndex)idx,DataFormat::Float32}});c.set_page_size((CBIndex)idx,bytes);CreateCircularBuffer(p,all_crs,c);};
    uint32_t ia=(uint32_t)inb->address(),ra=(uint32_t)rb->address(),ca=(uint32_t)cb->address();
    uint32_t fxa=(uint32_t)fxb->address(),fya=(uint32_t)fyb->address(),ga=(uint32_t)gb->address(),oa=(uint32_t)oib->address();

    // ── Program 1: prep + bucket ──
    Program p1=CreateProgram();
    const uint32_t pchunk=2048;
    cbf(p1,0,pchunk*64u,1); cbf(p1,1,nc*4u,1); cbf(p1,2,128u,1);
    auto k1=CreateKernel(p1,std::string(DENSITY_KERNEL_DIR)+"v29_prepbucket.cpp",all_crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
    auto pf=[&](float v){union{uint32_t u;float f;}c;c.f=v;return c.u;};
    for(int c=0;c<nc;++c){uint32_t f=(uint32_t)c*cpc_cell,n=std::min(cpc_cell,(f<ncells)?(ncells-f):0u);
        SetRuntimeArgs(p1,k1,ccs[c],{(uint32_t)c,f,n,ia,in_pg,ra,cap,(uint32_t)nc,cpc_x,ca,pchunk,N,M,pf(xl),pf(yl),pf(bsx),pf(bsy),pf(inv_bsx),pf(inv_bsy),cnt_pg});}
    MeshWorkload wl1; auto dr=MeshCoordinateRange{mesh->shape()}; wl1.add_program(dr,std::move(p1));

    // ── Program 2: gather + compute + writer ──
    Program p2=CreateProgram();
    cbf(p2,0,4096,max_k);cbf(p2,1,4096,max_h);cbf(p2,2,4096,2);cbf(p2,3,4096,2);cbf(p2,4,4096,2);
    cbf(p2,5,4096,1);cbf(p2,6,4096,1);   // c_5 gather oidx-build scratch, c_6 writer oidx-read scratch (separate L1)
    cbf(p2,16,4096,2);cbf(p2,17,4096,2);
    cbf(p2,24,band_cols*M*4u,1);cbf(p2,25,1024u*128u,1);cbf(p2,26,band_cols*M*4u,1);cbf(p2,27,nc*4u+128u,1);cbf(p2,28,1024u*16u,1);
    std::vector<UnpackToDestMode> um(64,UnpackToDestMode::Default);
    for(uint32_t i=0;i<5;++i)um[i]=UnpackToDestMode::UnpackToDestFp32; um[16]=um[17]=UnpackToDestMode::UnpackToDestFp32;
    std::map<std::string,std::string> gdef; if(getenv("V29_EMPTY_DIAG")) gdef["V29_EMPTY_DIAG"]="1";
    auto gk=CreateKernel(p2,std::string(DENSITY_KERNEL_DIR)+"v29_gather_brisc.cpp",all_crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default,.defines=gdef});
    auto wk=CreateKernel(p2,std::string(DENSITY_KERNEL_DIR)+"v29_writer.cpp",all_crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
    auto ck=CreateKernel(p2,std::string(DENSITY_KERNEL_DIR)+"v28_compute.cpp",all_crs,ComputeConfig{.fp32_dest_acc_en=true,.unpack_to_dest_mode=um,.math_approx_mode=false});
    for(int c=0;c<nc;++c){uint32_t col0=(uint32_t)c*cpc_x,vcols=(col0<N)?std::min(band_cols,N-col0):1u;
        uint32_t col0c=(col0<N)?col0:(N-1);   // clamp empty-band start so field read stays in-bounds
        SetRuntimeArgs(p2,gk,ccs[c],{(uint32_t)c,ra,cap,ca,(uint32_t)nc,max_k,max_h,fxa,fya,field_pg,M,cpc_x,vcols,nbat[c],cnt_pg,col0c,oa,mbatch});
        SetRuntimeArgs(p2,ck,ccs[c],{nbat[c],max_k,max_h});
        SetRuntimeArgs(p2,wk,ccs[c],{nbat[c],ga,grad_pg,(uint32_t)c,oa,mbatch});}
    MeshWorkload wl2; wl2.add_program(dr,std::move(p2));

    auto run_all=[&](){EnqueueMeshWorkload(cq,wl1,false);Finish(cq);EnqueueMeshWorkload(cq,wl2,false);Finish(cq);};
    fprintf(stderr,"[v29] launching P1 (prep+bucket)...\n");fflush(stderr);
    EnqueueMeshWorkload(cq,wl1,false);Finish(cq);
    fprintf(stderr,"[v29] P1 done. launching P2 (gather+compute+writer)...\n");fflush(stderr);
    EnqueueMeshWorkload(cq,wl2,false);Finish(cq);
    fprintf(stderr,"[v29] P2 done.\n");fflush(stderr);
    const int RN=7; std::vector<double> t1v,t2v;
    for(int i=0;i<RN;++i){auto a=hrclock::now();EnqueueMeshWorkload(cq,wl1,false);Finish(cq);double d1=ms_since(a);
                          auto b=hrclock::now();EnqueueMeshWorkload(cq,wl2,false);Finish(cq);double d2=ms_since(b);
                          t1v.push_back(d1);t2v.push_back(d2);}
    std::sort(t1v.begin(),t1v.end());std::sort(t2v.begin(),t2v.end());
    double bucket_ms=t1v[RN/2],gather_ms=t2v[RN/2];

    // validate vs CPU EF reference
    std::vector<float> gp((size_t)ncells*4u,0.f); EnqueueReadMeshBuffer(cq,gp,gb,true);
    double refn=0,difn=0,max_abs=0; size_t nbad=0;
    for(uint32_t i=0;i<ncells;++i){const Ref&R=ref[i]; double ax=0,ay=0;
        for(uint32_t k=0;k<R.kc;++k)for(uint32_t h=0;h<R.hc;++h){size_t idx=(size_t)(R.bxl+k)*M+(R.byl+h);double w=(double)R.px[k]*R.py[h];ax+=w*fx[idx];ay+=w*fy[idx];}
        double gxr=ax*R.ratio,gyr=ay*R.ratio; double dx=std::fabs(gp[i*4+0]-gxr),dy=std::fabs(gp[i*4+1]-gyr);
        max_abs=std::max({max_abs,dx,dy});refn+=gxr*gxr+gyr*gyr;difn+=dx*dx+dy*dy;
        if(dx>1e-3*std::max(1.0,std::fabs(gxr)))nbad++; if(dy>1e-3*std::max(1.0,std::fabs(gyr)))nbad++;}
    double rel_l2=refn>0?std::sqrt(difn/refn):0;
    printf("[v29] prep+bucket=%.3f ms  gather+compute+scatter=%.3f ms  TOTAL=%.3f ms  (CPU EF ~5-7 ms, V21 12.9 ms)\n",bucket_ms,gather_ms,bucket_ms+gather_ms);
    printf("[v29] accuracy: max_abs=%.6g rel_l2=%.6g nbad=%zu  %s\n",max_abs,rel_l2,nbad,(rel_l2<1e-4&&nbad==0)?"OK":"CHECK");
    printf("V29_RESULT,%u,%u,%u,%.3f,%.3f,%.6g\n",N,M,ncells,bucket_ms,gather_ms,rel_l2);
    return 0;
}
