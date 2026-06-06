// SPDX-License-Identifier: Apache-2.0
//
// V35 DUAL-RISC count+place microbench (on top of the compact 64 B record). count and
// place are normally BRISC-only (the NCRISC + SFPU sit idle through ~16 ms of the
// backward). Here each core is split into TWO sources — a NCRISC half (source id `c`)
// and a BRISC half (source id `c+nc`) — running v35_count{,_n} and v35_place_compact{,_n}
// in parallel on the two data-movement engines (NoC0/NoC1), writing to disjoint count
// pages / grouped-buffer regions (the forward-scatter convention). gather is unchanged.
// Goal: measure the count+place speedup from reclaiming the idle NCRISC, bit-exact vs CPU.
//   CLI: v35dual [N M ncells max_k max_h cluster]   (requires 4+max_k+max_h <= 16)

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
static uint32_t pf(float v){union{uint32_t u;float f;}c;c.f=v;return c.u;}

int main(int argc,char**argv){
    uint32_t N=2048,M=2048,ncells=210904,max_k=6,max_h=3,cluster=0;
    if(argc>=6){N=atoi(argv[1]);M=atoi(argv[2]);ncells=atoi(argv[3]);max_k=atoi(argv[4]);max_h=atoi(argv[5]);}
    if(argc>=7)cluster=atoi(argv[6]);
    if(4u+max_k+max_h>16u){ printf("[v35dual] ERROR: 4+max_k+max_h=%u > 16\n",4u+max_k+max_h); return 1; }
    auto mesh=MeshDevice::create_unit_mesh(0); auto&cq=mesh->mesh_command_queue();
    auto grid=mesh->compute_with_storage_grid_size(); const int nc=(int)(grid.x*grid.y);
    const int nsrc=2*nc;                                       // NCRISC halves [0,nc) + BRISC halves [nc,2nc)
    std::vector<CoreCoord> ccs; for(uint32_t y=0;y<grid.y;++y)for(uint32_t x=0;x<grid.x;++x)ccs.push_back({x,y});
    std::set<CoreRange> crs; for(auto c:ccs)crs.insert(CoreRange{c,c}); CoreRangeSet all_crs(crs);

    const uint32_t target_band=350u*1024u;
    uint32_t W=target_band/(M*4u); if(W>max_k)W-=max_k; else W=1u;
    uint32_t halfnc=(uint32_t)std::max(1,nc/2); if(W<((N+halfnc-1u)/halfnc))W=(N+halfnc-1u)/halfnc;
    if(W>N)W=N; if(W<1u)W=1u;
    const uint32_t ntiles=(N+W-1u)/W;
    const uint32_t band_cols=std::min(W+max_k,N);
    const uint32_t cpc_cell=((uint32_t)ncells+nc-1)/nc;

    // per-core cell counts + the NCRISC/BRISC split point
    std::vector<uint32_t> core_first(nc),core_n(nc),core_nlo(nc);
    for(int c=0;c<nc;++c){ uint32_t f=(uint32_t)c*cpc_cell; uint32_t n=(f<ncells)?std::min(cpc_cell,ncells-f):0u;
        core_first[c]=f; core_n[c]=n; core_nlo[c]=n/2u; }

    std::mt19937 rng(777); std::uniform_real_distribution<float> uf(-1,1),u01(0,1);
    std::vector<float> fx((size_t)N*M),fy((size_t)N*M); for(size_t i=0;i<fx.size();++i){fx[i]=uf(rng);fy[i]=uf(rng);}

    const float cfrac=(cluster>=2)?0.90f:0.60f, cwid=(cluster>=2)?0.020f:0.075f;
    auto gen_bxl=[&]()->uint32_t{ if(cluster && u01(rng)<cfrac){ double c=0.5*N,s=cwid*N,v=c+s*(uf(rng)+uf(rng)+uf(rng)); if(v<0)v=0; if(v>(double)(N-1))v=N-1; return(uint32_t)v;} return(uint32_t)(u01(rng)*(N-1)); };

    struct Cell{uint32_t bxl,byl,kc,hc;float ratio,px[8],py[8];};
    std::vector<Cell> cells(ncells);
    std::vector<uint32_t> stash((size_t)ncells*32u,0u);
    for(uint32_t i=0;i<ncells;++i){ Cell cl; cl.bxl=gen_bxl(); cl.byl=(uint32_t)(u01(rng)*(M-max_h-1));
        cl.kc=1+(uint32_t)(u01(rng)*max_k); if(cl.kc>max_k)cl.kc=max_k; cl.hc=1+(uint32_t)(u01(rng)*max_h); if(cl.hc>max_h)cl.hc=max_h;
        if(cl.bxl+cl.kc>N)cl.kc=N-cl.bxl; if(cl.byl+cl.hc>M)cl.hc=M-cl.byl; cl.ratio=0.5f+u01(rng);
        for(uint32_t k=0;k<8;++k)cl.px[k]=(k<cl.kc)?(0.2f+u01(rng)):0.f;
        for(uint32_t h=0;h<8;++h)cl.py[h]=(h<cl.hc)?(0.2f+u01(rng)):0.f;
        cells[i]=cl; uint32_t* r=&stash[(size_t)i*32u];
        r[0]=i; r[1]=cl.bxl; r[2]=cl.byl; r[3]=cl.kc; r[4]=cl.hc; r[5]=pf(cl.ratio);
        for(uint32_t k=0;k<8;++k){r[6+k]=pf(cl.px[k]); r[14+k]=pf(cl.py[k]);} }

    auto mb=[&](uint64_t b,uint32_t pg){DeviceLocalBufferConfig c{.page_size=pg,.buffer_type=BufferType::DRAM};ReplicatedBufferConfig r{.size=b};return MeshBuffer::create(r,c,mesh.get());};
    const uint32_t stash_pg=(uint32_t)((uint64_t)ncells*128u);
    const uint32_t cnt_pg=((ntiles*4u+63u)&~63u);
    const uint32_t grouped_pg=(uint32_t)((uint64_t)ncells*64u + 64u);
    const uint32_t plan_pg=cnt_pg;
    const uint32_t tb_pg=((ntiles*4u+63u)&~63u);
    const uint32_t field_pg=(uint32_t)((uint64_t)M*4u);
    auto stb=mb(stash_pg,stash_pg), cntb=mb((uint64_t)nsrc*cnt_pg,cnt_pg), grb=mb(grouped_pg,grouped_pg);
    auto tbb=mb(tb_pg,tb_pg), spb=mb((uint64_t)nsrc*plan_pg,plan_pg);
    auto fxb=mb((uint64_t)N*M*4u,field_pg), fyb=mb((uint64_t)N*M*4u,field_pg);
    EnqueueWriteMeshBuffer(cq,stb,stash,false); EnqueueWriteMeshBuffer(cq,fxb,fx,false); EnqueueWriteMeshBuffer(cq,fyb,fy,false); Finish(cq);
    uint32_t sta=stb->address(),cnta=cntb->address(),gra=grb->address(),tba=tbb->address(),spa=spb->address(),fxa=fxb->address(),fya=fyb->address();

    using tt::CBIndex;
    auto cbf=[&](Program&p,uint32_t idx,uint32_t bytes,uint32_t slots){CircularBufferConfig c(slots*bytes,{{(CBIndex)idx,DataFormat::Float32}});c.set_page_size((CBIndex)idx,bytes);CreateCircularBuffer(p,all_crs,c);};
    const uint32_t pchunk=1024u;
    auto dr=MeshCoordinateRange{mesh->shape()};

    // ── PASS 1: DUAL-RISC count (BRISC src c+nc, NCRISC src c) ──
    Program pc=CreateProgram();
    cbf(pc,0,pchunk*128u,1); cbf(pc,1,cnt_pg,1);          // BRISC CBs
    cbf(pc,5,pchunk*128u,1); cbf(pc,6,cnt_pg,1);          // NCRISC CBs
    auto kc_b=CreateKernel(pc,std::string(DENSITY_KERNEL_DIR)+"v35_count.cpp",  all_crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
    auto kc_n=CreateKernel(pc,std::string(DENSITY_KERNEL_DIR)+"v35_count_n.cpp",all_crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
    for(int c=0;c<nc;++c){ uint32_t f=core_first[c],nlo=core_nlo[c],nhi=core_n[c]-nlo;
        SetRuntimeArgs(pc,kc_n,ccs[c],{(uint32_t)c,        f,      nlo, sta,stash_pg,ntiles,W,cnta,cnt_pg,pchunk});   // NCRISC half → page c
        SetRuntimeArgs(pc,kc_b,ccs[c],{(uint32_t)(c+nc),   f+nlo,  nhi, sta,stash_pg,ntiles,W,cnta,cnt_pg,pchunk}); } // BRISC half → page c+nc
    MeshWorkload wlc; wlc.add_program(dr,std::move(pc));
    EnqueueMeshWorkload(cq,wlc,false); Finish(cq);   // warmup (JIT)
    auto tcount=hrclock::now(); EnqueueMeshWorkload(cq,wlc,false); Finish(cq); double count_ms=ms_since(tcount);

    // ── host plan over nsrc sources ──
    auto tplan=hrclock::now();
    std::vector<uint32_t> cnts((size_t)nsrc*(cnt_pg/4u),0u); EnqueueReadMeshBuffer(cq,cnts,cntb,true);
    const uint32_t rpw=cnt_pg/4u;
    std::vector<uint32_t> total(ntiles,0),tile_base(tb_pg/4u,0u); std::vector<uint32_t> srcpref((size_t)nsrc*(plan_pg/4u),0u);
    for(uint32_t t=0;t<ntiles;++t){uint32_t acc=0; for(int s=0;s<nsrc;++s){srcpref[(size_t)s*rpw+t]=acc; acc+=cnts[(size_t)s*rpw+t];} total[t]=acc;}
    {uint32_t b=0; for(uint32_t t=0;t<ntiles;++t){tile_base[t]=b; b+=total[t];}}
    std::vector<uint32_t> alloc(ntiles,0); uint32_t nonempty=0; for(uint32_t t=0;t<ntiles;++t)if(total[t])++nonempty;
    if(nonempty){uint32_t used=0; for(uint32_t t=0;t<ntiles;++t)if(total[t]){alloc[t]=1;++used;}
        while(used<(uint32_t)nc){uint32_t best=ntiles;double bl=-1;for(uint32_t t=0;t<ntiles;++t)if(total[t]){double l=(double)total[t]/alloc[t];if(l>bl){bl=l;best=t;}}if(best==ntiles)break;alloc[best]++;++used;}}
    std::vector<uint32_t> g_first(nc,0),g_n(nc,0),g_col0(nc,0); uint32_t cur=0,maxn=0;
    for(uint32_t t=0;t<ntiles;++t){uint32_t a=alloc[t]; if(!a)continue; uint32_t n=total[t],col0=t*W;
        for(uint32_t s=0;s<a && cur<(uint32_t)nc;++s){uint32_t lo=(uint64_t)n*s/a,hi=(uint64_t)n*(s+1)/a;
            g_first[cur]=tile_base[t]+lo; g_n[cur]=hi-lo; g_col0[cur]=col0; maxn=std::max(maxn,hi-lo); ++cur;}}
    const uint32_t max_batches=std::max(1u,(maxn+1023)/1024), mc=max_batches*1024, out_pg=mc*4u;
    EnqueueWriteMeshBuffer(cq,tbb,tile_base,false); EnqueueWriteMeshBuffer(cq,spb,srcpref,false); Finish(cq);
    double plan_ms=ms_since(tplan);
    // replay grouping order (per source, in src id order) for validation
    std::vector<uint32_t> g2o(ncells,0xffffffffu);
    {std::vector<uint32_t> cur2((size_t)nsrc*ntiles,0u);
     for(int s=0;s<nsrc;++s){ int c=(s<nc)?s:(s-nc); uint32_t f=(s<nc)?core_first[c]:(core_first[c]+core_nlo[c]);
        uint32_t n=(s<nc)?core_nlo[c]:(core_n[c]-core_nlo[c]);
        for(uint32_t j=0;j<n;++j){ uint32_t i=f+j; uint32_t t=cells[i].bxl/W; if(t>=ntiles)t=ntiles-1;
            uint32_t pos=tile_base[t]+srcpref[(size_t)s*rpw+t]+cur2[(size_t)s*ntiles+t]++; g2o[pos]=i; } } }

    // ── PASS 2: DUAL-RISC place (compact) ──
    Program pp=CreateProgram();
    cbf(pp,0,pchunk*128u,1); cbf(pp,1,pchunk*64u,1); cbf(pp,2,tb_pg,1); cbf(pp,3,plan_pg,1); cbf(pp,4,((4u*ntiles*4u+63u)&~63u),1);   // BRISC CBs
    cbf(pp,5,pchunk*128u,1); cbf(pp,6,pchunk*64u,1); cbf(pp,7,tb_pg,1); cbf(pp,8,plan_pg,1); cbf(pp,9,((4u*ntiles*4u+63u)&~63u),1);   // NCRISC CBs
    auto kp_b=CreateKernel(pp,std::string(DENSITY_KERNEL_DIR)+"v35_place_compact.cpp",  all_crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
    auto kp_n=CreateKernel(pp,std::string(DENSITY_KERNEL_DIR)+"v35_place_compact_n.cpp",all_crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
    for(int c=0;c<nc;++c){ uint32_t f=core_first[c],nlo=core_nlo[c],nhi=core_n[c]-nlo;
        SetRuntimeArgs(pp,kp_n,ccs[c],{(uint32_t)c,      f,     nlo, sta,stash_pg,ntiles,W,gra,grouped_pg,tba,spa,plan_pg,pchunk,max_k,max_h});
        SetRuntimeArgs(pp,kp_b,ccs[c],{(uint32_t)(c+nc), f+nlo, nhi, sta,stash_pg,ntiles,W,gra,grouped_pg,tba,spa,plan_pg,pchunk,max_k,max_h}); }
    MeshWorkload wlp; wlp.add_program(dr,std::move(pp));
    EnqueueMeshWorkload(cq,wlp,false); Finish(cq);   // warmup (JIT)
    auto tplace=hrclock::now(); EnqueueMeshWorkload(cq,wlp,false); Finish(cq); double place_ms=ms_since(tplace);

    if(getenv("V35_VERIFY_PLACE")){
        std::vector<uint32_t> gb((size_t)grouped_pg/4u,0u); EnqueueReadMeshBuffer(cq,gb,grb,true);
        size_t pbad=0; auto fbu=[&](uint32_t u){union{uint32_t u;float f;}c;c.u=u;return c.f;};
        for(uint32_t p=0;p<ncells;++p){ uint32_t oi=g2o[p]; if(oi>=ncells)continue;
            const uint32_t* o=&gb[(size_t)p*16u]; const Cell&cl=cells[oi];
            bool bad=(o[0]!=oi)||(o[1]!=cl.bxl)||(o[2]!=cl.byl)||(fbu(o[3])!=cl.ratio);
            for(uint32_t k=0;k<max_k;++k) if(fbu(o[4u+k])!=cl.px[k])bad=true;
            for(uint32_t h=0;h<max_h;++h) if(fbu(o[4u+max_k+h])!=cl.py[h])bad=true;
            if(bad)++pbad; }
        printf("[place_verify] dual compact full scan: %zu / %u mismatch %s\n",pbad,ncells,pbad?"<<< BUG":"(place OK)");
    }

    // ── PASS 3: gather (compact, unchanged) ──
    Program pg=CreateProgram();
    cbf(pg,0,4096,max_k);cbf(pg,1,4096,max_h);cbf(pg,2,4096,2);cbf(pg,3,4096,2);cbf(pg,4,4096,2);cbf(pg,5,4096,2);cbf(pg,6,4096,2);cbf(pg,16,4096,2);cbf(pg,17,4096,2);
    const uint32_t fld_bytes=band_cols*M*4u+(max_h+8u)*4u, chunk_batches=4, chunk_bytes=chunk_batches*1024u*64u;
    cbf(pg,24,fld_bytes,1); cbf(pg,25,chunk_bytes,1); cbf(pg,26,fld_bytes,1);
    std::vector<UnpackToDestMode> um(64,UnpackToDestMode::Default); for(uint32_t i=0;i<5;++i)um[i]=UnpackToDestMode::UnpackToDestFp32; um[16]=um[17]=UnpackToDestMode::UnpackToDestFp32;
    auto gxb=mb((uint64_t)nc*out_pg,out_pg), gyb=mb((uint64_t)nc*out_pg,out_pg); uint32_t gxa=gxb->address(),gya=gyb->address();
    auto oibb=mb((uint64_t)nc*out_pg,out_pg); uint32_t oiba=oibb->address();
    auto bk=CreateKernel(pg,std::string(DENSITY_KERNEL_DIR)+"v35_gather_brisc_compact.cpp",all_crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
    auto wk=CreateKernel(pg,std::string(DENSITY_KERNEL_DIR)+"v28_ncrisc.cpp",all_crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
    auto ck=CreateKernel(pg,std::string(DENSITY_KERNEL_DIR)+"v28_compute.cpp",all_crs,ComputeConfig{.fp32_dest_acc_en=true,.unpack_to_dest_mode=um,.math_approx_mode=false});
    for(int c=0;c<nc;++c){uint32_t nb=(g_n[c]+1023)/1024; uint32_t col0=g_col0[c],vcols=(col0<N)?std::min(band_cols,N-col0):1u;
        SetRuntimeArgs(pg,bk,ccs[c],{(uint32_t)c,g_n[c],band_cols,M,max_k,max_h,fxa,fya,field_pg,gra,grouped_pg,g_first[c],nb,chunk_batches,col0,vcols,oiba,out_pg});
        SetRuntimeArgs(pg,ck,ccs[c],{nb,max_k,max_h});
        SetRuntimeArgs(pg,wk,ccs[c],{(uint32_t)c,nb,M,max_k,max_h,fya,field_pg,col0,vcols,gxa,gya,out_pg,band_cols});}
    MeshWorkload wlg; wlg.add_program(dr,std::move(pg));
    auto run=[&]{EnqueueMeshWorkload(cq,wlg,false);Finish(cq);};
    run(); const int RN=7; std::vector<double> gms; for(int i=0;i<RN;++i){auto t=hrclock::now();run();gms.push_back(ms_since(t));}
    std::sort(gms.begin(),gms.end()); double gather_ms=gms[RN/2];

    std::vector<float> gxo((size_t)nc*mc,0.f),gyo((size_t)nc*mc,0.f);
    EnqueueReadMeshBuffer(cq,gxo,gxb,true); EnqueueReadMeshBuffer(cq,gyo,gyb,true);
    double refn=0,difn=0,maxabs=0; size_t nbad=0,checked=0;
    for(int c=0;c<nc;++c){uint32_t slice_first=g_first[c],n=g_n[c];
        for(uint32_t s=0;s<n;++s){uint32_t gpos=slice_first+s; uint32_t oi=g2o[gpos]; if(oi>=ncells)continue; const Cell&cl=cells[oi];
            double ax=0,ay=0; for(uint32_t k=0;k<cl.kc;++k)for(uint32_t h=0;h<cl.hc;++h){size_t idx=(size_t)(cl.bxl+k)*M+(cl.byl+h);double w=(double)cl.px[k]*cl.py[h];ax+=w*fx[idx];ay+=w*fy[idx];}
            double gxr=ax*cl.ratio,gyr=ay*cl.ratio; size_t o=(size_t)c*mc+s; double dx=std::fabs(gxo[o]-gxr),dy=std::fabs(gyo[o]-gyr);
            maxabs=std::max({maxabs,dx,dy}); refn+=gxr*gxr+gyr*gyr; difn+=dx*dx+dy*dy; ++checked;
            if(dx>1e-3*std::max(1.0,std::fabs(gxr)))nbad++; if(dy>1e-3*std::max(1.0,std::fabs(gyr)))nbad++;}}
    double rel_l2=refn>0?std::sqrt(difn/refn):0; double imbal=(ncells>0)?((double)maxn/((double)ncells/nc)):1.0;
    double prep_ms=count_ms+plan_ms+place_ms, total_ms=prep_ms+gather_ms;
    printf("[v35dual] N=%u M=%u ncells=%u nc=%d nsrc=%d cluster=%u W=%u ntiles=%u maxn=%u imbal=%.2fx checked=%zu REC=64B DUAL\n",N,M,ncells,nc,nsrc,cluster,W,ntiles,maxn,imbal,checked);
    printf("[v35dual] count=%.3f plan=%.3f place=%.3f gather=%.3f | PREP=%.3f TOTAL=%.3f ms\n",count_ms,plan_ms,place_ms,gather_ms,prep_ms,total_ms);
    printf("[v35dual] accuracy: maxabs=%.6g rel_l2=%.6g nbad=%zu %s\n",maxabs,rel_l2,nbad,(rel_l2<1e-4&&nbad==0)?"OK":"CHECK");
    printf("V35DUAL_RESULT,%u,%u,%u,%u,%u,%u,%.3f,%.3f,%.3f,%.3f,%.6g\n",N,M,ncells,max_k,max_h,cluster,count_ms,place_ms,gather_ms,total_ms,rel_l2);
    return 0;
}
