// SPDX-License-Identifier: Apache-2.0
//
// V22 on-chip optimizer END-TO-END validation. Chains the on-chip L1-target unsort
// (gather output -> interleaved density_grad in V22's b_dg, see onchip-unsort-64b-reliable)
// into the V22 optimizer (combine->precond->nesterov->clamp, pos/u resident) WITHOUT any
// host grad upload, then checks the resulting positions against a host reference of the
// same math. Proves the unsort->optimizer bridge end-to-end on hardware.
//   CLI: v22e2e [num_nodes]   env V35_DEVID
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <set>
#include <vector>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "v22_engine.h"
using namespace tt; using namespace tt::tt_metal; using namespace tt::tt_metal::distributed;
#ifndef DENSITY_KERNEL_DIR
#define DENSITY_KERNEL_DIR "kernels/"
#endif

int main(int argc,char**argv){
    const uint32_t nn = (argc>=2)?(uint32_t)atoi(argv[1]):131072u;
    const int devid=getenv("V35_DEVID")?atoi(getenv("V35_DEVID")):0;
    auto mesh=MeshDevice::create_unit_mesh(devid); auto&cq=mesh->mesh_command_queue();
    auto grid=mesh->compute_with_storage_grid_size(); const uint32_t nc=(uint32_t)(grid.x*grid.y);
    std::vector<CoreCoord> ccs; for(uint32_t y=0;y<grid.y;++y)for(uint32_t x=0;x<grid.x;++x)ccs.push_back({x,y});
    std::set<CoreRange> crs; for(auto c:ccs)crs.insert(CoreRange{c,c}); CoreRangeSet all_crs(crs);

    // ── host inputs in DreamPlace [x(nn)|y(nn)] layout ──
    std::mt19937 rng(7); std::uniform_real_distribution<float> U(-1.f,1.f), P(0.5f,3.f);
    const uint32_t ne=2u*nn;
    std::vector<float> lo(ne,-1e30f), hi(ne,1e30f);          // no-op clamp (validate the chain, not the bound)
    std::vector<float> pinw(ne), area(ne), vinit(ne), wlg(ne);
    for(uint32_t i=0;i<nn;++i){ float pw=P(rng),ar=P(rng);
        pinw[i]=pinw[nn+i]=pw; area[i]=area[nn+i]=ar;
        vinit[i]=U(rng)*100.f; vinit[nn+i]=U(rng)*100.f; wlg[i]=U(rng); wlg[nn+i]=U(rng); }
    const float alpha=0.03f, coef=0.7f, dw=1.5f;

    // ── gather-output source for the unsort: scrambled node perm + per-node gx,gy ──
    const uint32_t cpc=(((nn+nc-1)/nc)+15u)&~15u, padn=nc*cpc;
    std::vector<uint32_t> node((size_t)padn,0u); std::iota(node.begin(),node.begin()+nn,0u);
    std::shuffle(node.begin(),node.begin()+nn,rng);
    std::vector<float> gx(nn), gy(nn);
    std::vector<uint32_t> src((size_t)padn*4u,0u);   // 16B/slot: word0=gx, word1=gy
    for(uint32_t i=0;i<nn;++i){ gx[i]=U(rng); gy[i]=U(rng);
        std::memcpy(&src[4u*i], &gx[i], 4); std::memcpy(&src[4u*i+1u], &gy[i], 4); }

    // ── V22 engine (interleaved internal layout) ──
    v22::V22OptEngine eng(mesh.get(), (int)nn);
    eng.configure(lo.data(),hi.data(),pinw.data(),area.data());
    eng.set_pos(vinit.data());
    const uint32_t bdga=eng.density_grad_address();
    const uint32_t ntiles=eng.density_grad_num_tiles();   // == ceil(2nn/1024)
    const uint32_t epc=(((nn+nc-1u)/nc)+511u)&~511u, tpc=epc/512u;

    // ── unsort program: source(slot) -> interleaved b_dg (engine's), single-RISC ──
    auto mb=[&](uint64_t b,uint32_t pg){DeviceLocalBufferConfig c{.page_size=pg,.buffer_type=BufferType::DRAM};ReplicatedBufferConfig r{.size=b};return MeshBuffer::create(r,c,mesh.get());};
    std::vector<uint32_t> coord(nc); for(uint32_t c=0;c<nc;++c){auto p=mesh->worker_core_from_logical_core(ccs[c]); coord[c]=((uint32_t)p.x<<16)|(uint32_t)p.y;}
    const uint32_t src_pg=(uint32_t)((uint64_t)padn*16u), nidx_pg=(uint32_t)((uint64_t)padn*4u), coord_pg=nc*4u;
    auto srcb=mb(src_pg,src_pg), nidxb=mb(nidx_pg,nidx_pg), coordb=mb(coord_pg,coord_pg);
    EnqueueWriteMeshBuffer(cq,srcb,src,false); EnqueueWriteMeshBuffer(cq,nidxb,node,false); EnqueueWriteMeshBuffer(cq,coordb,coord,false); Finish(cq);
    uint32_t srca=srcb->address(),nidxa=nidxb->address(),coorda=coordb->address();

    using tt::CBIndex;
    auto cbf=[&](Program&p,uint32_t idx,uint32_t bytes){CircularBufferConfig c(bytes,{{(CBIndex)idx,DataFormat::Float32}});c.set_page_size((CBIndex)idx,bytes);CreateCircularBuffer(p,all_crs,c);};
    Program pp=CreateProgram();
    cbf(pp,0,epc*16u+64u); cbf(pp,1,nc*4u+64u); cbf(pp,2,cpc*16u+64u); cbf(pp,3,cpc*4u+64u); cbf(pp,4,tpc*4096u+64u);
    auto kp=CreateKernel(pp,std::string(DENSITY_KERNEL_DIR)+"v35_unsort_l1_full.cpp",all_crs,
        DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
    MeshWorkload wl; auto dr=MeshCoordinateRange{mesh->shape()}; wl.add_program(dr,std::move(pp));
    Program& prog=wl.get_programs().at(dr);
    auto setargs=[&](uint32_t phase){ for(uint32_t c=0;c<nc;++c){
        uint32_t f=c*cpc, rn=(f<nn)?cpc:0u, sn=std::min(cpc,(f<nn)?(nn-f):0u);
        uint32_t t0=c*tpc, nt=(t0<ntiles)?std::min(tpc,ntiles-t0):0u;
        SetRuntimeArgs(prog,kp,ccs[c],{phase,c,f,rn,0u,srca,src_pg,nidxa,nidx_pg,coorda,coord_pg,nc,epc,nn,0u,t0,nt,bdga,4096u,sn,1u}); } };
    setargs(0u); EnqueueMeshWorkload(cq,wl,false); Finish(cq);   // scatter -> L1 shards
    setargs(1u); EnqueueMeshWorkload(cq,wl,false); Finish(cq);   // flush -> engine b_dg (interleaved)

    // ── optimizer step (density_grad already in b_dg; only wl_grad+scalars uploaded) ──
    v22::OptTiming t; eng.step_resident_ondevice_grad(wlg.data(), alpha, coef, dw, alpha, &t);
    std::vector<float> out(ne); eng.get_pos(out.data());

    // ── host reference: combine -> precond -> nesterov(u_prev=vinit) -> clamp ──
    // density_grad at node n = grad of the SLOT that maps to n: dg_node[node[i]] = g[i].
    std::vector<float> dgx(nn), dgy(nn);
    for(uint32_t i=0;i<nn;++i){ dgx[node[i]]=gx[i]; dgy[node[i]]=gy[i]; }
    std::vector<float> ref(ne);
    double maxrel=0.0, maxabs=0.0; size_t nbad=0;
    auto step_ref=[&](float v,float wlv,float g,float pw,float ar)->float{
        float comb=wlv+dw*g;
        float pdiv=std::max(pw+alpha*dw*ar, 1.0f);
        float gp=comb/pdiv;
        float u=v-alpha*gp;            // u' (u_prev=v since set_pos seeds u_k=v_k)
        float vn=u+coef*(u - v);       // v' = u' + coef*(u' - u_prev)
        return vn;                      // clamp is no-op (lo/hi huge)
    };
    for(uint32_t i=0;i<nn;++i){
        ref[i]   =step_ref(vinit[i],   wlg[i],   dgx[i], pinw[i], area[i]);
        ref[nn+i]=step_ref(vinit[nn+i],wlg[nn+i],dgy[i], pinw[nn+i],area[nn+i]);
    }
    for(uint32_t k=0;k<ne;++k){ double a=std::fabs((double)out[k]-(double)ref[k]);
        double r=a/(std::fabs((double)ref[k])+1e-6); maxabs=std::max(maxabs,a); maxrel=std::max(maxrel,r);
        if(r>2e-3 && a>1e-3){ nbad++; if(getenv("E2E_DBG")&&nbad<=12) printf("  k=%u %s node=%u exp=%.6f got=%.6f rel=%.2e\n",k,(k<nn)?"x":"y",(k<nn)?k:k-nn,ref[k],out[k],r);} }

    printf("[v22e2e] nn=%u nc=%u ntiles=%u  unsort(scatter %.3f+flush, h2d %.3f, opt %.3f ms)\n",nn,nc,ntiles,0.0,t.h2d_ms,t.compute_ms);
    printf("[v22e2e] vs host ref: max_abs=%.3e max_rel=%.3e  out-of-tol=%zu/%u  %s\n",maxabs,maxrel,nbad,ne,(nbad==0)?"OK":"<<< WRONG");
    printf("V22E2E_RESULT,%u,%.3e,%.3e,%zu\n",nn,maxabs,maxrel,nbad);
    return 0;
}
