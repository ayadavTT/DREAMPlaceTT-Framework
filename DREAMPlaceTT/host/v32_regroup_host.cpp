// SPDX-License-Identifier: Apache-2.0
// COUNT-PREFIX REGROUP prototype (isolated). CLI: v32_regroup [M dist]
//   M    = records/core (default 20000)
//   dist = 0 uniform owners, 1 clustered (80% into a hot 10% of owners)
// Pipeline (NO atomics, NO host in the loop): count (local histogram, per core)
// → prefix (one core: owner_start + per-core base) → place (each core writes its
// records to route[base[o]++]). Verifies route is a permutation of all inputs,
// grouped contiguously by owner; times each device pass.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
    uint32_t M=20000; int dist=0;
    if(argc>=2)M=atoi(argv[1]); if(argc>=3)dist=atoi(argv[2]);
    auto mesh=MeshDevice::create_unit_mesh(0); auto&cq=mesh->mesh_command_queue();
    auto g=mesh->compute_with_storage_grid_size(); const uint32_t NC=g.x*g.y; const uint32_t NOWN=NC;
    std::vector<CoreCoord> ccs; for(uint32_t y=0;y<g.y;++y)for(uint32_t x=0;x<g.x;++x)ccs.push_back({x,y});
    std::set<CoreRange> crs; for(auto c:ccs)crs.insert(CoreRange{c,c}); CoreRangeSet all(crs);
    CoreRangeSet one(CoreRange{ccs[0],ccs[0]});
    auto dr=MeshCoordinateRange{mesh->shape()};

    // ── generate records: 16B = [owner, cell(=c*M+i, globally unique), bin, area_bits] ──
    const uint64_t TOT=(uint64_t)NC*M;
    std::vector<uint32_t> recs((size_t)TOT*4u);
    uint32_t hot=NOWN/10u; if(!hot)hot=1u; srand(1234);
    std::vector<uint64_t> exp_cnt(NOWN,0);
    for(uint32_t c=0;c<NC;++c)for(uint32_t i=0;i<M;++i){
        uint32_t o = (dist==1)? ((rand()%100<80)? (uint32_t)(rand()%hot) : (uint32_t)(rand()%NOWN))
                              : (uint32_t)(rand()%NOWN);
        uint64_t r=((uint64_t)c*M+i); size_t b=(size_t)r*4u;
        recs[b]=o; recs[b+1]=(uint32_t)r; recs[b+2]=o*7u+(i%7u); recs[b+3]=0x3f800000u; exp_cnt[o]++; }

    using tt::CBIndex;
    const uint32_t recs_pg=M*16u, cm_pg=((NOWN*4u+63u)/64u)*64u, route_pg=16u, chunk=512u; // cm_pg 64-B aligned (Blackhole DRAM read align)
    const uint32_t STR=cm_pg/4u;                                                            // padded row pitch in elements
    auto mk=[&](uint32_t pg,uint64_t npg){ DeviceLocalBufferConfig cfg{.page_size=pg,.buffer_type=BufferType::DRAM};
        ReplicatedBufferConfig rc{.size=npg*pg}; return MeshBuffer::create(rc,cfg,mesh.get()); };
    auto recs_b=mk(recs_pg,NC), cntm_b=mk(cm_pg,NC), base_b=mk(cm_pg,NC), route_b=mk(route_pg,TOT);
    EnqueueWriteMeshBuffer(cq,recs_b,recs,false);
    uint32_t ra=recs_b->address(), ca=cntm_b->address(), ba=base_b->address(), oa=route_b->address();

    // ── build the three programs ──
    auto p_count=std::make_shared<Program>(CreateProgram());
    { CircularBufferConfig r(chunk*16u,{{CBIndex::c_0,DataFormat::Float32}}); r.set_page_size(CBIndex::c_0,chunk*16u); CreateCircularBuffer(*p_count,all,r);
      CircularBufferConfig h(cm_pg,{{CBIndex::c_1,DataFormat::Float32}}); h.set_page_size(CBIndex::c_1,cm_pg); CreateCircularBuffer(*p_count,all,h);
      auto k=CreateKernel(*p_count,std::string(DENSITY_KERNEL_DIR)+"v32_count.cpp",all,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
      for(uint32_t c=0;c<NC;++c) SetRuntimeArgs(*p_count,k,ccs[c],{c,ra,recs_pg,M,NOWN,ca,cm_pg,chunk}); }

    auto p_pre=std::make_shared<Program>(CreateProgram());
    { CircularBufferConfig a(NC*cm_pg,{{CBIndex::c_0,DataFormat::Float32}}); a.set_page_size(CBIndex::c_0,NC*cm_pg); CreateCircularBuffer(*p_pre,one,a);
      CircularBufferConfig b(NC*cm_pg,{{CBIndex::c_1,DataFormat::Float32}}); b.set_page_size(CBIndex::c_1,NC*cm_pg); CreateCircularBuffer(*p_pre,one,b);
      CircularBufferConfig o(cm_pg,{{CBIndex::c_2,DataFormat::Float32}}); o.set_page_size(CBIndex::c_2,cm_pg); CreateCircularBuffer(*p_pre,one,o);
      auto k=CreateKernel(*p_pre,std::string(DENSITY_KERNEL_DIR)+"v32_prefix.cpp",one,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
      SetRuntimeArgs(*p_pre,k,ccs[0],{NC,NOWN,ca,cm_pg,ba,cm_pg,0u/*page_mode: flat array*/}); }

    auto p_place=std::make_shared<Program>(CreateProgram());
    { CircularBufferConfig r(chunk*16u,{{CBIndex::c_0,DataFormat::Float32}}); r.set_page_size(CBIndex::c_0,chunk*16u); CreateCircularBuffer(*p_place,all,r);
      CircularBufferConfig c(cm_pg,{{CBIndex::c_1,DataFormat::Float32}}); c.set_page_size(CBIndex::c_1,cm_pg); CreateCircularBuffer(*p_place,all,c);
      auto k=CreateKernel(*p_place,std::string(DENSITY_KERNEL_DIR)+"v32_place.cpp",all,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
      for(uint32_t c=0;c<NC;++c) SetRuntimeArgs(*p_place,k,ccs[c],{c,ra,recs_pg,M,NOWN,ba,cm_pg,oa,route_pg,chunk}); }

    auto runp=[&](std::shared_ptr<Program> p){ MeshWorkload wl; wl.add_program(dr,std::move(*p)); EnqueueMeshWorkload(cq,wl,false); Finish(cq); return wl; };
    auto time1=[&](const char* nm,std::shared_ptr<Program>& p){ MeshWorkload wl; wl.add_program(dr,std::move(*p));
        EnqueueMeshWorkload(cq,wl,false); Finish(cq);
        const int R=5; std::vector<double> ms; for(int i=0;i<R;++i){auto t=hrclock::now();EnqueueMeshWorkload(cq,wl,false);Finish(cq);ms.push_back(ms_since(t));}
        std::sort(ms.begin(),ms.end()); printf("  %-7s %.3f ms\n",nm,ms[R/2]); return ms[R/2]; };

    double tc=time1("count",p_count), tp=time1("prefix",p_pre), tpl=time1("place",p_place);

    // DEBUG: check the count matrix and base matrix against host-computed expectations
    { std::vector<uint32_t> cm((size_t)NC*STR,0u); EnqueueReadMeshBuffer(cq,cm,cntm_b,true);
      std::vector<uint32_t> bm((size_t)NC*STR,0u); EnqueueReadMeshBuffer(cq,bm,base_b,true);
      // host count matrix (padded stride STR)
      std::vector<uint32_t> hc((size_t)NC*STR,0u);
      for(uint32_t c=0;c<NC;++c)for(uint32_t i=0;i<M;++i){uint32_t o=recs[((uint64_t)c*M+i)*4u]; hc[c*STR+o]++;}
      uint64_t cnt_err=0; for(uint32_t c=0;c<NC;++c)for(uint32_t o=0;o<NOWN;++o) if(cm[c*STR+o]!=hc[c*STR+o])cnt_err++;
      // host base matrix from device count matrix
      std::vector<uint32_t> hb((size_t)NC*STR,0u); uint64_t acc=0;
      for(uint32_t o=0;o<NOWN;++o){ uint32_t run=(uint32_t)acc; uint64_t tot=0;
          for(uint32_t c=0;c<NC;++c){ hb[c*STR+o]=run; run+=cm[c*STR+o]; tot+=cm[c*STR+o]; } acc+=tot; }
      uint64_t base_err=0,first_be=0; for(uint32_t c=0;c<NC;++c)for(uint32_t o=0;o<NOWN;++o){size_t j=c*STR+o; if(bm[j]!=hb[j]){ if(!base_err)first_be=j; base_err++; }}
      printf("  [dbg] count_matrix_err=%llu  base_matrix_err=%llu (first idx %llu: dev=%u host=%u)\n",
             (unsigned long long)cnt_err,(unsigned long long)base_err,(unsigned long long)first_be,
             base_err?bm[first_be]:0u, base_err?hb[first_be]:0u); }

    // ── verify route: permutation of all cells + grouped contiguously by owner ──
    std::vector<uint32_t> rv((size_t)TOT*4u,0u); EnqueueReadMeshBuffer(cq,rv,route_b,true);
    std::vector<uint8_t> seen((size_t)TOT,0); int bad=0; uint32_t prev_o=0; uint64_t dups=0,oob=0,disorder=0;
    std::vector<uint64_t> got_cnt(NOWN,0);
    for(uint64_t p=0;p<TOT;++p){ uint32_t o=rv[p*4u], cell=rv[p*4u+1];
        if(o>=NOWN){oob++;continue;} got_cnt[o]++;
        if(o<prev_o)disorder++; prev_o=o;                       // owners must be non-decreasing
        if(cell>=TOT){oob++;continue;}
        if(seen[cell])dups++; else seen[cell]=1; }
    uint64_t missing=0; for(uint64_t r=0;r<TOT;++r) if(!seen[r])missing++;
    for(uint32_t o=0;o<NOWN;++o) if(got_cnt[o]!=exp_cnt[o]) bad++;
    bool ok = (dups==0 && missing==0 && oob==0 && disorder==0 && bad==0);
    printf("[regroup] M=%u/core NC=%u TOT=%llu dist=%s | count=%.3f prefix=%.3f place=%.3f TOTAL=%.3f ms\n",
           M,NC,(unsigned long long)TOT, dist?"clustered":"uniform", tc,tp,tpl, tc+tp+tpl);
    printf("          VERIFY %s  (dups=%llu missing=%llu oob=%llu disorder=%llu cnt_mismatch=%d)\n",
           ok?"OK (permutation + owner-grouped)":"FAIL",
           (unsigned long long)dups,(unsigned long long)missing,(unsigned long long)oob,(unsigned long long)disorder,bad);
    return 0;
}
