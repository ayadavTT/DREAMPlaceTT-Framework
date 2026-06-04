// SPDX-License-Identifier: Apache-2.0
//
// V35 ON-CHIP GROUPING microbench — validates the fully host-free grouping that the
// live backward needs: PASS1 count (per-core tile histogram) → host plan (tile bases
// + per-source prefix + greedy proportional core allocation; O(nc·ntiles) metadata)
// → PASS2 place (count-prefix scatter into ONE O(ncells) tile-grouped buffer) →
// PASS3 gather (V28 SFPU multi-bin, contiguous tile slices). Cells originate in the
// forward stash format (128 B, global order); no host touches per-cell data.
//
// Measures count+place+gather (the on-chip "prep+compute") and checks the grad is
// bit-exact vs the CPU multi-bin reference, uniform AND clustered, all grids.
//   CLI: v35live [N M ncells max_k max_h cluster]

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
    auto mesh=MeshDevice::create_unit_mesh(0); auto&cq=mesh->mesh_command_queue();
    auto grid=mesh->compute_with_storage_grid_size(); const int nc=(int)(grid.x*grid.y);
    std::vector<CoreCoord> ccs; for(uint32_t y=0;y<grid.y;++y)for(uint32_t x=0;x<grid.x;++x)ccs.push_back({x,y});
    std::set<CoreRange> crs; for(auto c:ccs)crs.insert(CoreRange{c,c}); CoreRangeSet all_crs(crs);

    // tile width: band(+halo) per plane ≤ ~350 KB, and ntiles ≤ nc/2 (every tile ≥2 cores)
    const uint32_t target_band=350u*1024u;
    uint32_t W=target_band/(M*4u); if(W>max_k)W-=max_k; else W=1u;
    uint32_t halfnc=(uint32_t)std::max(1,nc/2); if(W<((N+halfnc-1u)/halfnc))W=(N+halfnc-1u)/halfnc;
    if(W>N)W=N; if(W<1u)W=1u;
    const uint32_t ntiles=(N+W-1u)/W;
    const uint32_t band_cols=std::min(W+max_k,N);
    const uint32_t cpc_cell=((uint32_t)ncells+nc-1)/nc;   // stash cells per source core

    std::mt19937 rng(777); std::uniform_real_distribution<float> uf(-1,1),u01(0,1);
    std::vector<float> fx((size_t)N*M),fy((size_t)N*M); for(size_t i=0;i<fx.size();++i){fx[i]=uf(rng);fy[i]=uf(rng);}

    const float cfrac=(cluster>=2)?0.90f:0.60f, cwid=(cluster>=2)?0.020f:0.075f;
    auto gen_bxl=[&]()->uint32_t{ if(cluster && u01(rng)<cfrac){ double c=0.5*N,s=cwid*N,v=c+s*(uf(rng)+uf(rng)+uf(rng)); if(v<0)v=0; if(v>(double)(N-1))v=N-1; return(uint32_t)v;} return(uint32_t)(u01(rng)*(N-1)); };

    // ── stash: 128 B/cell (32 u32) [0]gidx [1]bxl_g [2]byl [3]kc [4]hc [5]ratio [6..13]px [14..21]py ──
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
    const uint32_t grouped_pg=(uint32_t)((uint64_t)ncells*128u + 128u);   // 128 B records (64-B aligned)
    const uint32_t plan_pg=cnt_pg;                               // per-core srcprefix page
    const uint32_t tb_pg=((ntiles*4u+63u)&~63u);
    const uint32_t field_pg=(uint32_t)((uint64_t)N*M*4u);
    auto stb=mb(stash_pg,stash_pg), cntb=mb((uint64_t)nc*cnt_pg,cnt_pg), grb=mb(grouped_pg,grouped_pg);
    auto tbb=mb(tb_pg,tb_pg), spb=mb((uint64_t)nc*plan_pg,plan_pg);
    auto fxb=mb(field_pg,field_pg), fyb=mb(field_pg,field_pg);
    EnqueueWriteMeshBuffer(cq,stb,stash,false); EnqueueWriteMeshBuffer(cq,fxb,fx,false); EnqueueWriteMeshBuffer(cq,fyb,fy,false); Finish(cq);
    uint32_t sta=stb->address(),cnta=cntb->address(),gra=grb->address(),tba=tbb->address(),spa=spb->address(),fxa=fxb->address(),fya=fyb->address();

    using tt::CBIndex;
    auto cbf=[&](Program&p,uint32_t idx,uint32_t bytes,uint32_t slots){CircularBufferConfig c(slots*bytes,{{(CBIndex)idx,DataFormat::Float32}});c.set_page_size((CBIndex)idx,bytes);CreateCircularBuffer(p,all_crs,c);};
    const uint32_t pchunk=1024u;

    // ── PASS 1: count ──
    Program pc=CreateProgram();
    cbf(pc,0,pchunk*128u,1); cbf(pc,1,cnt_pg,1);
    auto kc=CreateKernel(pc,std::string(DENSITY_KERNEL_DIR)+"v35_count.cpp",all_crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
    for(int c=0;c<nc;++c){uint32_t f=(uint32_t)c*cpc_cell,n=std::min(cpc_cell,(f<ncells)?(ncells-f):0u);
        SetRuntimeArgs(pc,kc,ccs[c],{(uint32_t)c,f,n,sta,stash_pg,ntiles,W,cnta,cnt_pg,pchunk});}
    MeshWorkload wlc; auto dr=MeshCoordinateRange{mesh->shape()}; wlc.add_program(dr,std::move(pc));
    EnqueueMeshWorkload(cq,wlc,false); Finish(cq);   // warmup (JIT)
    auto tcount=hrclock::now(); EnqueueMeshWorkload(cq,wlc,false); Finish(cq); double count_ms=ms_since(tcount);

    // ── host plan: counts → tile_base, srcprefix[src][t], proportional core alloc ──
    auto tplan=hrclock::now();
    std::vector<uint32_t> cnts((size_t)nc*(cnt_pg/4u),0u); EnqueueReadMeshBuffer(cq,cnts,cntb,true);
    const uint32_t rpw=cnt_pg/4u;
    std::vector<uint32_t> total(ntiles,0),tile_base(tb_pg/4u,0u); std::vector<uint32_t> srcpref((size_t)nc*(plan_pg/4u),0u);
    for(uint32_t t=0;t<ntiles;++t){uint32_t acc=0; for(int s=0;s<nc;++s){srcpref[(size_t)s*rpw+t]=acc; acc+=cnts[(size_t)s*rpw+t];} total[t]=acc;}
    {uint32_t b=0; for(uint32_t t=0;t<ntiles;++t){tile_base[t]=b; b+=total[t];}}
    // greedy makespan core allocation
    std::vector<uint32_t> alloc(ntiles,0); uint32_t nonempty=0; for(uint32_t t=0;t<ntiles;++t)if(total[t])++nonempty;
    if(nonempty){uint32_t used=0; for(uint32_t t=0;t<ntiles;++t)if(total[t]){alloc[t]=1;++used;}
        while(used<(uint32_t)nc){uint32_t best=ntiles;double bl=-1;for(uint32_t t=0;t<ntiles;++t)if(total[t]){double l=(double)total[t]/alloc[t];if(l>bl){bl=l;best=t;}}if(best==ntiles)break;alloc[best]++;++used;}}
    // per gather-core: (slice_first, slice_n, col0)
    std::vector<uint32_t> g_first(nc,0),g_n(nc,0),g_col0(nc,0); uint32_t cur=0,maxn=0;
    for(uint32_t t=0;t<ntiles;++t){uint32_t a=alloc[t]; if(!a)continue; uint32_t n=total[t],col0=t*W;
        for(uint32_t s=0;s<a && cur<(uint32_t)nc;++s){uint32_t lo=(uint64_t)n*s/a,hi=(uint64_t)n*(s+1)/a;
            g_first[cur]=tile_base[t]+lo; g_n[cur]=hi-lo; g_col0[cur]=col0; maxn=std::max(maxn,hi-lo); ++cur;}}
    const uint32_t max_batches=std::max(1u,(maxn+1023)/1024), mc=max_batches*1024, out_pg=mc*4u;
    EnqueueWriteMeshBuffer(cq,tbb,tile_base,false); EnqueueWriteMeshBuffer(cq,spb,srcpref,false); Finish(cq);
    double plan_ms=ms_since(tplan);
    // replay grouping order for validation: grouped_pos → orig cell
    std::vector<uint32_t> g2o(ncells,0xffffffffu);
    {std::vector<uint32_t> cur2((size_t)nc*ntiles,0u);
     for(uint32_t i=0;i<ncells;++i){uint32_t src=i/cpc_cell; if(src>=(uint32_t)nc)src=nc-1; uint32_t t=cells[i].bxl/W; if(t>=ntiles)t=ntiles-1;
        uint32_t pos=tile_base[t]+srcpref[(size_t)src*rpw+t]+cur2[(size_t)src*ntiles+t]++; g2o[pos]=i;}}

    // ── PASS 2: place ──
    Program pp=CreateProgram();
    cbf(pp,0,pchunk*128u,1); cbf(pp,1,pchunk*128u,1); cbf(pp,2,tb_pg,1); cbf(pp,3,plan_pg,1); cbf(pp,4,((4u*ntiles*4u+63u)&~63u),1);
    auto kp=CreateKernel(pp,std::string(DENSITY_KERNEL_DIR)+"v35_place.cpp",all_crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
    for(int c=0;c<nc;++c){uint32_t f=(uint32_t)c*cpc_cell,n=std::min(cpc_cell,(f<ncells)?(ncells-f):0u);
        SetRuntimeArgs(pp,kp,ccs[c],{(uint32_t)c,f,n,sta,stash_pg,ntiles,W,gra,grouped_pg,tba,spa,plan_pg,pchunk});}
    MeshWorkload wlp; wlp.add_program(dr,std::move(pp));
    EnqueueMeshWorkload(cq,wlp,false); Finish(cq);   // warmup (JIT)
    auto tplace=hrclock::now(); EnqueueMeshWorkload(cq,wlp,false); Finish(cq); double place_ms=ms_since(tplace);

    // ── DEBUG: verify grouped_buf records match g2o (place correctness) ──
    if(getenv("V35_VERIFY_PLACE")){
        std::vector<uint32_t> gb((size_t)grouped_pg/4u,0u); EnqueueReadMeshBuffer(cq,gb,grb,true);
        size_t pbad=0,nshow=0; auto fb=[&](uint32_t u){union{uint32_t u;float f;}c;c.u=u;return c.f;};
        for(uint32_t p=0;p<ncells;++p){ uint32_t oi=g2o[p]; if(oi>=ncells)continue;
            const uint32_t* o=&gb[(size_t)p*32u]; const Cell&cl=cells[oi];   // 128 B verbatim, bxl global
            bool bad = (o[0]!=oi)||(o[1]!=cl.bxl)||(o[2]!=cl.byl)||(o[3]!=cl.kc)||(o[4]!=cl.hc)||(fb(o[5])!=cl.ratio);
            for(uint32_t k=0;k<8;++k){ if(fb(o[6+k])!=cl.px[k])bad=true; if(fb(o[14+k])!=cl.py[k])bad=true; }
            if(bad){ ++pbad; if(nshow<6){++nshow; printf("[place_bad] p=%u oi=%u bxl(%u/%u) kc(%u/%u) ratio(%.3f/%.3f) px0(%.3f/%.3f)\n",
                p,oi,o[1],cl.bxl,o[3],cl.kc,fb(o[5]),cl.ratio,fb(o[6]),cl.px[0]); } } }
        printf("[place_verify] full-field scan: %zu / %u records mismatch %s\n", pbad,ncells,pbad?"<<< PLACE BUG":"(place OK)");
    }

    // ── PASS 3: gather (V35 brisc + v28 ncrisc + v28 compute) ──
    Program pg=CreateProgram();
    cbf(pg,0,4096,max_k);cbf(pg,1,4096,max_h);cbf(pg,2,4096,2);cbf(pg,3,4096,2);cbf(pg,4,4096,2);cbf(pg,5,4096,2);cbf(pg,6,4096,2);cbf(pg,16,4096,2);cbf(pg,17,4096,2);
    const uint32_t fld_bytes=band_cols*M*4u+(max_h+8u)*4u, chunk_batches=4, chunk_bytes=chunk_batches*1024u*128u;
    cbf(pg,24,fld_bytes,1); cbf(pg,25,chunk_bytes,1); cbf(pg,26,fld_bytes,1);
    std::vector<UnpackToDestMode> um(64,UnpackToDestMode::Default); for(uint32_t i=0;i<5;++i)um[i]=UnpackToDestMode::UnpackToDestFp32; um[16]=um[17]=UnpackToDestMode::UnpackToDestFp32;
    auto gxb=mb((uint64_t)nc*out_pg,out_pg), gyb=mb((uint64_t)nc*out_pg,out_pg); uint32_t gxa=gxb->address(),gya=gyb->address();
    auto oibb=mb((uint64_t)nc*out_pg,out_pg); uint32_t oiba=oibb->address();   // per-core oidx (gidx/slot)
    auto bk=CreateKernel(pg,std::string(DENSITY_KERNEL_DIR)+"v35_gather_brisc.cpp",all_crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
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

    // ── validate grad vs CPU multi-bin (grouped order via g2o) ──
    std::vector<float> gxo((size_t)nc*mc,0.f),gyo((size_t)nc*mc,0.f);
    EnqueueReadMeshBuffer(cq,gxo,gxb,true); EnqueueReadMeshBuffer(cq,gyo,gyb,true);
    double refn=0,difn=0,maxabs=0; size_t nbad=0,checked=0;
    for(int c=0;c<nc;++c){uint32_t slice_first=g_first[c],n=g_n[c];
        for(uint32_t s=0;s<n;++s){uint32_t gpos=slice_first+s; uint32_t oi=g2o[gpos]; if(oi>=ncells)continue; const Cell&cl=cells[oi];
            double ax=0,ay=0; for(uint32_t k=0;k<cl.kc;++k)for(uint32_t h=0;h<cl.hc;++h){size_t idx=(size_t)(cl.bxl+k)*M+(cl.byl+h);double w=(double)cl.px[k]*cl.py[h];ax+=w*fx[idx];ay+=w*fy[idx];}
            double gxr=ax*cl.ratio,gyr=ay*cl.ratio; size_t o=(size_t)c*mc+s; double dx=std::fabs(gxo[o]-gxr),dy=std::fabs(gyo[o]-gyr);
            maxabs=std::max({maxabs,dx,dy}); refn+=gxr*gxr+gyr*gyr; difn+=dx*dx+dy*dy; ++checked;
            if(dx>1e-3*std::max(1.0,std::fabs(gxr)))nbad++; if(dy>1e-3*std::max(1.0,std::fabs(gyr)))nbad++;}}
    if(getenv("V35_DBG")){ int c=0; for(uint32_t s=0;s<6 && s<g_n[c];++s){ uint32_t gp=g_first[c]+s; uint32_t oi=g2o[gp]; const Cell&cl=cells[oi];
        double ax=0,ay=0; for(uint32_t k=0;k<cl.kc;++k)for(uint32_t h=0;h<cl.hc;++h){size_t idx=(size_t)(cl.bxl+k)*M+(cl.byl+h);double w=(double)cl.px[k]*cl.py[h];ax+=w*fx[idx];ay+=w*fy[idx];}
        printf("[dbg] core=%d s=%u gp=%u oi=%u bxl=%u byl=%u kc=%u hc=%u | gxo=%.5f ref=%.5f %s\n",
               c,s,gp,oi,cl.bxl,cl.byl,cl.kc,cl.hc,gxo[(size_t)c*mc+s],ax*cl.ratio,(std::fabs(gxo[(size_t)c*mc+s]-ax*cl.ratio)<1e-4)?"ok":"BAD"); } }
    double rel_l2=refn>0?std::sqrt(difn/refn):0; double imbal=(ncells>0)?((double)maxn/((double)ncells/nc)):1.0;
    double prep_ms=count_ms+plan_ms+place_ms, total_ms=prep_ms+gather_ms;
    printf("[v35live] N=%u M=%u ncells=%u nc=%d cluster=%u W=%u ntiles=%u maxn=%u imbal=%.2fx checked=%zu\n",N,M,ncells,nc,cluster,W,ntiles,maxn,imbal,checked);
    printf("[v35live] count=%.3f plan=%.3f place=%.3f gather=%.3f | PREP=%.3f TOTAL=%.3f ms\n",count_ms,plan_ms,place_ms,gather_ms,prep_ms,total_ms);
    printf("[v35live] accuracy: maxabs=%.6g rel_l2=%.6g nbad=%zu %s\n",maxabs,rel_l2,nbad,(rel_l2<1e-4&&nbad==0)?"OK":"CHECK");
    printf("V35LIVE_RESULT,%u,%u,%u,%u,%u,%u,%.3f,%.3f,%.3f,%.3f,%.6g\n",N,M,ncells,max_k,max_h,cluster,count_ms,place_ms,gather_ms,total_ms,rel_l2);
    return 0;
}
