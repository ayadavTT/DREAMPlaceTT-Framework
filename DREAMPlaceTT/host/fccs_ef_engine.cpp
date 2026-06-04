// SPDX-License-Identifier: Apache-2.0
//
// FCCS EF backward engine — see fccs_ef_engine.h. Single persistent program:
// core 0 = field-cast producer, cores 1..nc-1 = workers (balanced cell slices).
// Field uploaded to two column-major DRAM planes; geometry read from the forward
// stash; grad read back in original order and scattered into grad[sel].

#include "fccs_ef_engine.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
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
namespace fccsef {
static constexpr uint32_t GREC=32u, MAXKH=8u, WS=13u, FS=8u, ALIGN=64u;   // FS=8: f·2^8 stays <2^23 (SFPU typecast wraps above 2^23)

struct Rect{uint32_t xs,ys,xe,ye,nd;};

struct FCCSEFEngine::Impl {
    MeshDevice* dev; MeshCommandQueue* cq;
    int NBX,NBY,nmax; float xl,yl,bsx,bsy,inv_bsx,inv_bsy;
    int nc; CoreCoord grid; std::vector<CoreCoord> lc,nocc; CoreRangeSet crs{std::set<CoreRange>{}};
    std::vector<Rect> rects;
    // configured
    int n_active=0,n_total=0; std::vector<int32_t> sel;
    std::vector<float> ox,oy,nsx,nsy,ratio;
    uint32_t max_k=1, max_h=1, cg=32u, py_off=14u;   // cg=compact sorted record size, py_off=py base
    uint32_t geom_addr=0, geom_pg=0;
    // sizing / buffers
    bool built=false; uint32_t nworkers=0,cpc=0,W=0,Tx=0,buf_stride=0,cb_size=0;
    uint32_t field_l1_off=0,sortbuf_l1_off=0,grad_l1_off=0;
    uint32_t fx_pg=0,fy_pg=0,grad_pg=0;
    uint32_t cpc_e=0,padded_elems=0;   // parallel-quantize chunk (mult of 16) + padded field length
    std::shared_ptr<MeshBuffer> fxb,fyb,grb;
    MeshWorkload wl, wlq, wld, wls; MeshCoordinateRange dr{MeshCoordinate{0,0},MeshCoordinate{0,0}};
    KernelHandle kde=0; uint32_t rpc=0, chip_pg=0;   // de-interleave (chip-resident field): rows/core, chip row-page bytes
    // STREAMING mode (geom slice exceeds L1, e.g. adaptec3 1.39M): on-chip sort → sorted DRAM
    // + sort_start, then streaming backward. set when in-L1 cb_size doesn't fit.
    bool streaming=false; KernelHandle ksort=0; std::shared_ptr<MeshBuffer> sortedb, ssb;
    uint32_t ss_pg=0, sorted_pg=0, sortstart_l1_off=0, chunk_l1_off=0;
    std::vector<int64_t> gradp;   // d2h staging int64 [gx,gy] per active cell (original order); host descales
    // HOST-GEOM validation (FCCS_HOST_GEOM): recompute the EF overlap geometry on the host
    // (triangle(pos+offset, node_size_clamped)) — the CORRECT electric-force footprint — instead
    // of the forward's density stash (which omits offset + uses orig size → wrong for clamped cells).
    bool use_host_geom=false; std::shared_ptr<MeshBuffer> host_geom; std::vector<int32_t> hg_data;

    Impl(void* md,int M_,int N_,int nmax_,float xl_,float yl_,float bsx_,float bsy_)
      : dev(static_cast<MeshDevice*>(md)),NBX(M_),NBY(N_),nmax(nmax_),
        xl(xl_),yl(yl_),bsx(bsx_),bsy(bsy_),inv_bsx(1.f/bsx_),inv_bsy(1.f/bsy_) {
        cq=&dev->mesh_command_queue(); grid=dev->compute_with_storage_grid_size(); nc=grid.x*grid.y;
        for(uint32_t y=0;y<grid.y;++y)for(uint32_t x=0;x<grid.x;++x){
            lc.push_back({x,y}); nocc.push_back(dev->worker_core_from_logical_core({x,y})); }
        std::set<CoreRange> s; for(auto&c:lc)s.insert(CoreRange{c,c}); crs=CoreRangeSet(s);
        // gap-aware multicast rects over the NOC0 worker coords (covers ALL cores;
        // per-core num_dests is decremented for the core's own rect at arg-build time)
        std::set<uint32_t> ax; for(auto&c:nocc)ax.insert(c.x);
        std::vector<std::pair<uint32_t,uint32_t>> runs; {uint32_t lo=0,hi=0;bool in=false;
            for(uint32_t x:ax){ if(!in){lo=hi=x;in=true;} else if(x==hi+1)hi=x; else{runs.emplace_back(lo,hi);lo=hi=x;} } if(in)runs.emplace_back(lo,hi);}
        for(auto&xr:runs){ uint32_t ymin=UINT32_MAX,ymax=0,cnt=0;
            for(auto&c:nocc) if(c.x>=xr.first&&c.x<=xr.second){ ymin=std::min(ymin,(uint32_t)c.y); ymax=std::max(ymax,(uint32_t)c.y); ++cnt; }
            rects.push_back({xr.first,ymin,xr.second,ymax,cnt}); }
    }
};

FCCSEFEngine::FCCSEFEngine(void* md,int M,int N,int nmax,float xl,float yl,float bsx,float bsy)
  : impl_(std::make_unique<Impl>(md,M,N,nmax,xl,yl,bsx,bsy)) {}
FCCSEFEngine::~FCCSEFEngine() = default;

void FCCSEFEngine::configure_with_sel(const float* ox_f,const float* oy_f,const float* nsx_f,
        const float* nsy_f,const float* ratio_f,const int32_t* sel,int na,int nt){
    auto&I=*impl_; I.n_active=na; I.n_total=nt; I.sel.assign(sel,sel+na);
    I.ox.resize(na);I.oy.resize(na);I.nsx.resize(na);I.nsy.resize(na);I.ratio.resize(na);
    for(int i=0;i<na;++i){int g=sel[i];I.ox[i]=ox_f[g];I.oy[i]=oy_f[g];I.nsx[i]=nsx_f[g];I.nsy[i]=nsy_f[g];I.ratio[i]=ratio_f[g];}
    // max_k from ORIG-size span (orig = clamped + 2·|offset|) — matches the forward stash
    // overlap count; capped at MAXKH (record holds 8 px/py slots).
    uint32_t mk=1; for(int i=0;i<na;++i){ float ow=I.nsx[i]+2.f*std::fabs(I.ox[i]);
        uint32_t k=(uint32_t)std::floor(ow*I.inv_bsx)+2u; mk=std::max(mk,k); }
    I.max_k=std::min(mk,MAXKH);
    // max_h from y-span (mirror of max_k) — needed to size the COMPACT sorted record.
    uint32_t mh=1; for(int i=0;i<na;++i){ float oh=I.nsy[i]+2.f*std::fabs(I.oy[i]);
        uint32_t h=(uint32_t)std::floor(oh*I.inv_bsy)+2u; mh=std::max(mh,h); }
    I.max_h=std::min(mh,MAXKH);
}

void FCCSEFEngine::set_geom_source(uint32_t geom_addr,uint32_t geom_pg){
    impl_->geom_addr=geom_addr; impl_->geom_pg=geom_pg;
}

void FCCSEFEngine::build_if_needed(){
    auto&I=*impl_;
    if(I.built) return;
    const int na=I.n_active,NBX=I.NBX,NBY=I.NBY,nc=I.nc;
    {
        I.nworkers=(uint32_t)nc-1u;
        I.cpc=((uint32_t)na+I.nworkers-1u)/I.nworkers;
        I.W=std::min((uint32_t)NBX,std::max(1u,(256u*1024u)/((uint32_t)NBY*4u)));
        I.Tx=((uint32_t)NBX+I.W-1u)/I.W;
        I.buf_stride=((I.W*(uint32_t)NBY+1u)*4u+ALIGN-1u)&~(ALIGN-1u);   // +1 word for the sentinel-in-data handshake flag
        I.field_l1_off=0u;
        I.sortbuf_l1_off=(2u*I.buf_stride+ALIGN-1u)&~(ALIGN-1u);
        const uint32_t sortbuf_bytes=I.cpc*GREC*4u + I.cpc*4u + ((uint32_t)NBX+1u)*4u;
        I.grad_l1_off=(I.sortbuf_l1_off+sortbuf_bytes+ALIGN-1u)&~(ALIGN-1u);
        I.cb_size=(I.grad_l1_off+I.cpc*16u+ALIGN);   // int64 [gx,gy]/cell
        // If the in-L1 slice doesn't fit, switch to STREAMING: on-chip counting-sort →
        // sorted DRAM + sort_start, then stream the sorted slice per tile-window (L1 use
        // independent of cell count). Enables adaptec3-class (1.39M cells) to run.
        I.streaming = (I.cb_size >= 1400u*1024u);
        if (I.streaming) {
            const uint32_t CHUNK=512u;
            I.sortstart_l1_off=(2u*I.buf_stride+ALIGN-1u)&~(ALIGN-1u);
            I.grad_l1_off=(I.sortstart_l1_off + ((uint32_t)NBX+1u)*4u + ALIGN-1u)&~(ALIGN-1u);
            I.chunk_l1_off=(I.grad_l1_off + I.cpc*16u + ALIGN-1u)&~(ALIGN-1u);
            I.cb_size=(I.chunk_l1_off + CHUNK*GREC*4u + ALIGN);
            fprintf(stderr,"[fccs_ef] STREAMING mode (na=%d cpc=%u W=%u): in-L1 too big → streaming, cb_size=%uKB\n",
                    na,I.cpc,I.W,I.cb_size>>10);
        }
        // SFPU field-quantize sizing: process the field as flat 1024-float "tiles"
        // (elementwise → tile layout irrelevant). ntiles_per_buf rounds up; buffers
        // padded to a whole number of tiles so the last tile is in-bounds.
        { const uint32_t total=(uint32_t)((uint64_t)NBX*NBY);
          I.cpc_e=(total+1023u)/1024u;              // ntiles per plane (reuse cpc_e field)
          I.padded_elems=I.cpc_e*1024u; }
        // DRAM: two field planes (tile-padded) + grad (interleaved [gx,gy]/cell)
        I.fx_pg=I.padded_elems*4u; I.fy_pg=I.fx_pg;
        I.grad_pg=(uint32_t)((uint64_t)na*16u); if(I.grad_pg<ALIGN)I.grad_pg=ALIGN;   // int64 [gx,gy]/cell
        auto mb=[&](uint64_t b,uint32_t pg){DeviceLocalBufferConfig c{.page_size=pg,.buffer_type=BufferType::DRAM};ReplicatedBufferConfig r{.size=b};return MeshBuffer::create(r,c,I.dev);};
        I.fxb=mb(I.fx_pg,I.fx_pg); I.fyb=mb(I.fy_pg,I.fy_pg); I.grb=mb(I.grad_pg,I.grad_pg);
        I.gradp.assign((size_t)na*2u,0);
        if (getenv("FCCS_HOST_GEOM") && atoi(getenv("FCCS_HOST_GEOM"))) {   // override geom_addr with our host-computed EF geometry
            uint32_t gpg=(uint32_t)((uint64_t)na*GREC*4u);
            I.host_geom=mb(gpg,gpg); I.geom_addr=(uint32_t)I.host_geom->address(); I.geom_pg=gpg; I.use_host_geom=true;
            I.hg_data.assign((size_t)na*GREC,0);
            fprintf(stderr,"[fccs_ef] HOST-GEOM on: recomputing EF overlap geometry on host (addr=%u)\n",I.geom_addr);
        }
        I.dr=MeshCoordinateRange{I.dev->shape()};
        // STREAMING: sorted-geom DRAM (na*128) + per-worker sort_start prefix rows
        if (I.streaming) {
            // COMPACT sorted record: the sort reads the full 128 B cell-order record but writes
            // only [header(6) + px(max_k) + py(max_h)] to sorted_geom (CG u32). This shrinks the
            // scatter-write (and the streaming gather's read) ~GREC/CG×, the sort's bottleneck.
            // CG must keep records 64-B aligned: Blackhole DRAM noc_async_read needs 64-B aligned
            // addr+size (else SILENTLY wrong data). round (6+max_k+max_h) up to a multiple of 16 u32.
            const uint32_t CG=((6u+I.max_k+I.max_h+15u)/16u)*16u, PY_OFF=6u+I.max_k;
            I.sorted_pg=(uint32_t)((uint64_t)na*CG*4u);
            I.ss_pg=(((uint32_t)NBX+1u)*4u+63u)&~63u;
            I.sortedb=mb(I.sorted_pg,I.sorted_pg);
            I.ssb=mb((uint64_t)nc*I.ss_pg,I.ss_pg);
            I.cg=CG; I.py_off=PY_OFF;
            // ── on-chip counting-sort program (per worker; core0 idle) ──
            Program ps=CreateProgram();
            const uint32_t hist_off=0u, cursor_off=(((uint32_t)NBX+1u)*4u+ALIGN-1u)&~(ALIGN-1u);
            const uint32_t chunk_off=(cursor_off+((uint32_t)NBX+1u)*4u+ALIGN-1u)&~(ALIGN-1u);
            // L1-resident COMPACT slice: read the 128 B geom ONCE (chunked) → extract compact here →
            // place from L1 (PASS 2 has no DRAM re-read + a single write barrier). cpc*CG*4 ≈ 0.8 MB fits.
            const uint32_t slice_off=(chunk_off+512u*GREC*4u+ALIGN-1u)&~(ALIGN-1u);
            const uint32_t sort_cb=(slice_off+I.cpc*CG*4u+ALIGN);
            CreateCircularBuffer(ps,I.crs,CircularBufferConfig(sort_cb,{{24u,DataFormat::UInt32}}).set_page_size(24u,sort_cb));
            I.ksort=CreateKernel(ps,std::string(DENSITY_KERNEL_DIR)+"fccs_sort_dm.cpp",I.crs,
                DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
            for(uint32_t c=0;c<(uint32_t)nc;++c){
                uint32_t mf=(c>=1u)?((c-1u)*I.cpc):0u;
                uint32_t mn=(c>=1u && mf<(uint32_t)na)?std::min(I.cpc,(uint32_t)na-mf):0u;
                SetRuntimeArgs(ps,I.ksort,I.lc[c],{c,I.geom_addr,(uint32_t)I.sortedb->address(),(uint32_t)I.ssb->address(),
                    (uint32_t)na,mf,mn,(uint32_t)NBX,hist_off,cursor_off,chunk_off,CG,PY_OFF,I.max_k,I.max_h,slice_off});
            }
            I.wls.add_program(I.dr,std::move(ps));
        }
        // ── backward program: in-L1 (fccs_live_dm) or streaming (fccs_live_stream_dm) ──
        Program prog=CreateProgram();
        CreateCircularBuffer(prog,I.crs,CircularBufferConfig(I.cb_size,{{24u,DataFormat::UInt32}}).set_page_size(24u,I.cb_size));
        const char* bwd_kernel = I.streaming ? "fccs_live_stream_dm.cpp" : "fccs_live_dm.cpp";
        auto k=CreateKernel(prog,std::string(DENSITY_KERNEL_DIR)+bwd_kernel,I.crs,
            DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        CreateKernel(prog,std::string(DENSITY_KERNEL_DIR)+"v13_void_brisc.cpp",I.crs,
            DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
        uint32_t ready_sid=CreateSemaphore(prog,I.crs,0u),tile_sid=CreateSemaphore(prog,I.crs,0u),done_sid=CreateSemaphore(prog,I.crs,0u);
        auto c0=I.nocc[0]; uint32_t fxa=(uint32_t)I.fxb->address(),fya=(uint32_t)I.fyb->address(),gra=(uint32_t)I.grb->address();
        uint32_t dbg = getenv("FCCS_DBG") ? (uint32_t)atoi(getenv("FCCS_DBG")) : 0u;  // arg8: bit0 skip pxpy-quant, bit1 skip compute
        // V31_GEOM_INT: forward stashed px/py as int32 already → backward must SKIP its
        // quantize (the wrapper overhead we moved to the forward). bit0=skip is then CORRECT.
        if (getenv("V31_GEOM_INT") && atoi(getenv("V31_GEOM_INT"))) dbg |= 1u;
        if (getenv("FCCS_DBG_INPUTS") && atoi(getenv("FCCS_DBG_INPUTS"))) dbg |= 4u;  // dump worker field+px·py
        if (getenv("FCCS_DBG_PROD") && atoi(getenv("FCCS_DBG_PROD"))) dbg |= (8u|2u);  // producer bufp probe + skip worker compute
        for(uint32_t c=0;c<(uint32_t)nc;++c){
            std::vector<uint32_t> a;
            if (I.streaming)
                a={c,(uint32_t)nc,(c==0u)?1u:0u,I.cpc,(uint32_t)na,(uint32_t)NBX,(uint32_t)NBY,
                   I.max_k,dbg,I.W,I.Tx,fxa,fya,(uint32_t)I.sortedb->address(),I.field_l1_off,I.sortstart_l1_off,I.grad_l1_off,
                   I.buf_stride,(uint32_t)c0.x,(uint32_t)c0.y,ready_sid,tile_sid,done_sid,WS,FS,gra,(uint32_t)I.ssb->address(),
                   I.chunk_l1_off,(uint32_t)I.rects.size(),I.cg,I.py_off};   // args 29=CG 30=PY_OFF; rects at 31+
            else
                a={c,(uint32_t)nc,(c==0u)?1u:0u,I.cpc,(uint32_t)na,(uint32_t)NBX,(uint32_t)NBY,
                   I.max_k,dbg,I.W,I.Tx,fxa,fya,I.geom_addr,I.field_l1_off,I.sortbuf_l1_off,I.grad_l1_off,
                   I.buf_stride,(uint32_t)c0.x,(uint32_t)c0.y,ready_sid,tile_sid,done_sid,WS,FS,gra,(uint32_t)I.rects.size()};
            for(auto&r:I.rects){ bool self=(I.nocc[c].x>=r.xs&&I.nocc[c].x<=r.xe&&I.nocc[c].y>=r.ys&&I.nocc[c].y<=r.ye);
                a.push_back(r.xs);a.push_back(r.ys);a.push_back(r.xe);a.push_back(r.ye);a.push_back(self?r.nd-1u:r.nd); }
            SetRuntimeArgs(prog,k,I.lc[c],a);
        }
        I.dr=MeshCoordinateRange{I.dev->shape()}; I.wl.add_program(I.dr,std::move(prog));

        // ── SFPU field-quantize pre-pass (reader NCRISC → compute TRISC SFPU → writer
        //    BRISC): float→int32 in place, ×2^FS, on the hardware float unit across all
        //    cores. ntiles_per_buf tiles/plane, total=2·ntiles, tpc tiles/core. ──
        const uint32_t ntiles_per_buf=I.cpc_e, total_tiles=2u*ntiles_per_buf;
        const uint32_t tpc=(total_tiles+(uint32_t)nc-1u)/(uint32_t)nc;
        union { float f; uint32_t u; } fsc; fsc.f=(float)(1u<<FS);
        Program pq=CreateProgram();
        const uint32_t TB=1024u*4u;   // one tile bytes
        { CircularBufferConfig ci(2u*TB,{{tt::CBIndex::c_0,DataFormat::Float32}}); ci.set_page_size(tt::CBIndex::c_0,TB);
          CreateCircularBuffer(pq,I.crs,ci);
          CircularBufferConfig co(2u*TB,{{tt::CBIndex::c_16,DataFormat::Int32}}); co.set_page_size(tt::CBIndex::c_16,TB);
          CreateCircularBuffer(pq,I.crs,co); }
        auto kqr=CreateKernel(pq,std::string(DENSITY_KERNEL_DIR)+"fccs_quant_reader.cpp",I.crs,
            DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
        auto kqw=CreateKernel(pq,std::string(DENSITY_KERNEL_DIR)+"fccs_quant_writer.cpp",I.crs,
            DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        auto kqc=CreateKernel(pq,std::string(DENSITY_KERNEL_DIR)+"fccs_quant_compute.cpp",I.crs,
            ComputeConfig{.fp32_dest_acc_en=true});
        for(uint32_t c=0;c<(uint32_t)nc;++c){
            SetRuntimeArgs(pq,kqr,I.lc[c],{c,fxa,fya,ntiles_per_buf,tpc,total_tiles,I.fx_pg});
            SetRuntimeArgs(pq,kqw,I.lc[c],{c,fxa,fya,ntiles_per_buf,tpc,total_tiles,I.fx_pg});
            SetRuntimeArgs(pq,kqc,I.lc[c],{c,tpc,total_tiles,fsc.u}); }
        I.wlq.add_program(I.dr,std::move(pq));

        // ── chip-resident field de-interleave (host-free): each core copies its row
        //    range from the interleaved chip-DCT field (latest_field_*_mb, page=1 row)
        //    → contiguous fxb/fyb (float). Then SFPU quantize + producer run as usual,
        //    NO host h2d. Chip src addrs (args 1,2) are set per-call in compute_from_chip. ──
        I.rpc=((uint32_t)NBX+(uint32_t)nc-1u)/(uint32_t)nc;
        I.chip_pg=(((uint32_t)NBY*4u)+63u)&~63u;
        Program pd=CreateProgram();
        const uint32_t de_cb=((2u*I.chip_pg)+ALIGN-1u)&~(ALIGN-1u);
        CreateCircularBuffer(pd,I.crs,CircularBufferConfig(de_cb,{{tt::CBIndex::c_0,DataFormat::Float32}}).set_page_size(tt::CBIndex::c_0,de_cb));
        I.kde=CreateKernel(pd,std::string(DENSITY_KERNEL_DIR)+"fccs_deinterleave_dm.cpp",I.crs,
            DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        for(uint32_t c=0;c<(uint32_t)nc;++c)
            SetRuntimeArgs(pd,I.kde,I.lc[c],{c,0u,0u,(uint32_t)I.fxb->address(),(uint32_t)I.fyb->address(),
                (uint32_t)NBX,(uint32_t)NBY,I.rpc,I.chip_pg,I.fx_pg});
        I.wld.add_program(I.dr,std::move(pd));
        I.built=true;
    }
}

// HOST-GEOM: recompute the electric-force overlap geometry per active cell on the host —
// triangle_density_function(node_x=pos+offset, node_size_clamped) — matching DREAMPlace's
// electric_force.cpp exactly. Writes V31_GEOM int records (px=triangle·sqrt(inv_ba)·2^WS).
static void fccs_compute_geom_host(FCCSEFEngine::Impl& I, const float* pos){
    const int na=I.n_active, nn=I.n_total;
    const double siba=std::sqrt((double)I.inv_bsx*(double)I.inv_bsy), ws=(double)(1u<<WS);
    for(int i=0;i<na;++i){ int g=I.sel[i]; int32_t* r=&I.hg_data[(size_t)i*GREC];
        const double nx=(double)pos[g]+(double)I.ox[i], ny=(double)pos[nn+g]+(double)I.oy[i];
        const double sx=(double)I.nsx[i], sy=(double)I.nsy[i];
        int bxl=(int)std::floor((nx-(double)I.xl)*(double)I.inv_bsx); int bxh=(int)std::floor((nx+sx-(double)I.xl)*(double)I.inv_bsx)+1;
        if(bxl<0)bxl=0; if(bxh>I.NBX)bxh=I.NBX; int kc=bxh-bxl; if(kc<0)kc=0; if(kc>(int)MAXKH)kc=MAXKH;
        int byl=(int)std::floor((ny-(double)I.yl)*(double)I.inv_bsy); int byh=(int)std::floor((ny+sy-(double)I.yl)*(double)I.inv_bsy)+1;
        if(byl<0)byl=0; if(byh>I.NBY)byh=I.NBY; int hc=byh-byl; if(hc<0)hc=0; if(hc>(int)MAXKH)hc=MAXKH;
        for(int z=0;z<(int)GREC;++z) r[z]=0;
        r[0]=i; r[1]=bxl; r[2]=byl; r[3]=kc; r[4]=hc;
        for(int k=0;k<kc;++k){ double bx=(double)I.xl+(double)(bxl+k)*(double)I.bsx;
            double tx=std::min(nx+sx,bx+(double)I.bsx)-std::max(nx,bx); if(tx<0)tx=0; r[6+k]=(int32_t)(tx*siba*ws); }
        for(int h=0;h<hc;++h){ double by=(double)I.yl+(double)(byl+h)*(double)I.bsy;
            double ty=std::min(ny+sy,by+(double)I.bsy)-std::max(ny,by); if(ty<0)ty=0; r[14+h]=(int32_t)(ty*siba*ws); }
    }
    EnqueueWriteMeshBuffer(*I.cq,I.host_geom,I.hg_data,false); Finish(*I.cq);
}

void FCCSEFEngine::compute_from_full(const float* pos,const float* fx,const float* fy,float* grad,FCCSTiming* t){
    auto&I=*impl_; FCCSTiming tm{}; auto t0=hrclock::now();
    const int na=I.n_active,nn=I.n_total,NBX=I.NBX,NBY=I.NBY,nc=I.nc; (void)nc;
    build_if_needed();
    if(I.use_host_geom){ auto tgh=hrclock::now(); fccs_compute_geom_host(I,pos);
        if(getenv("FCCS_SPLIT")) fprintf(stderr,"[fccs_split] HOST_GEOM compute=%.3f ms\n", ms_since(tgh)); }

    // ── upload field (two column-major planes; padded buffer, valid NBX*NBY at front) ──
    auto th=hrclock::now();
    { std::vector<float> fxv(fx,fx+(size_t)NBX*NBY),fyv(fy,fy+(size_t)NBX*NBY);
      fxv.resize(I.padded_elems,0.f); fyv.resize(I.padded_elems,0.f);
      EnqueueWriteMeshBuffer(*I.cq,I.fxb,fxv,false); EnqueueWriteMeshBuffer(*I.cq,I.fyb,fyv,false); }
    Finish(*I.cq); tm.h2d_ms=ms_since(th);

    // ── parallel field quantize (float→int32 in place, ~1/nc work per core) ──
    auto tq=hrclock::now(); EnqueueMeshWorkload(*I.cq,I.wlq,false); Finish(*I.cq); tm.quant_ms=ms_since(tq);

    // ── STREAMING: on-chip counting-sort of the geom (geom→sorted_geom + sort_start) MUST run
    //    before the streaming backward reads sort_start. compute_from_chip did this; the host-field
    //    path forgot it → uninitialized sort_start → the chunk read overflowed to an invalid NoC core. ──
    double sort_ms=0.0; if(I.streaming){ auto tso=hrclock::now(); EnqueueMeshWorkload(*I.cq,I.wls,false); Finish(*I.cq); sort_ms=ms_since(tso); }

    // ── run field-cast backward ──
    auto tr=hrclock::now(); EnqueueMeshWorkload(*I.cq,I.wl,false); auto tenq=hrclock::now(); Finish(*I.cq); tm.run_ms=ms_since(tr);
    if(getenv("FCCS_SPLIT") && I.streaming) fprintf(stderr,"[fccs_split] sort=%.3f ms\n", sort_ms);
    if(getenv("FCCS_SPLIT")){ double enq=std::chrono::duration<double,std::milli>(tenq-tr).count();
        fprintf(stderr,"[fccs_split] quant=%.3f bwd_enqueue=%.3f bwd_finish=%.3f h2d=%.3f\n",
                tm.quant_ms,enq,tm.run_ms-enq,tm.h2d_ms); }

    // ── d2h grad (int64, original order: slot i = active cell i) → grad[sel[i]].
    //    Host applies DESCALE·ratio[i] (FPU here, not soft-float on BRISC). ──
    auto td=hrclock::now();
    EnqueueReadMeshBuffer(*I.cq,I.gradp,I.grb,true);
    // scale = 1/2^(WS+FS) · bin_area. bin_area (= bsx·bsy) is the CONSTANT the forward
    // stashed at g[5] (undoes the v4_compute sqrt(inv_bin_area) prescale) — the working
    // float-grad version used exactly this per cell; it's the same for all cells.
    // grad[c] = bin_area·Σ(ox·oy·f)/2^(2WS+FS), ·ratio[c] when using HOST-GEOM (correct triangle
    // overlap → DREAMPlace's `gx*=ratio`). The old density-stash path omitted ratio AND used the
    // wrong footprint; the two errors cancelled in aggregate magnitude but not per-cell (rel_mov 0.25).
    const double s=(1.0/(double)((uint64_t)1u<<(2u*WS+FS)))*(double)I.bsx*(double)I.bsy;
    // ·ratio when the geometry is EF-correct (clamped+offset): host-geom OR the forward's
    // V31_EF_GEOM path. (The old orig-stash path baked ratio in via its wider footprint.)
    const bool efr=I.use_host_geom||(getenv("V31_EF_GEOM")&&atoi(getenv("V31_EF_GEOM")));
    // Big cells are sub-tiled (new_nc>nc): multiple sub-cells share a parent node (sel[i]).
    // Zero each touched slot once, then ACCUMULATE — so a wide cell's force = Σ of its sub-cells.
    for(int i=0;i<na;++i){ int g=I.sel[i]; grad[g]=0.f; grad[nn+g]=0.f; }
    for(int i=0;i<na;++i){ int g=I.sel[i]; double sr=efr?s*(double)I.ratio[i]:s;
        grad[g]+=(float)((double)I.gradp[(size_t)i*2]*sr); grad[nn+g]+=(float)((double)I.gradp[(size_t)i*2+1]*sr); }
    tm.d2h_ms=ms_since(td);
    tm.total_ms=ms_since(t0); if(t)*t=tm;
}

// Chip-resident field: read the field directly from the chip-DCT output (fx_addr/fy_addr
// = V19 latest_field_*_addr, ROW_MAJOR interleaved page=1 row) — NO host h2d. On-chip
// de-interleave → fxb/fyb (contiguous float) → SFPU quantize → producer multicast → backward.
void FCCSEFEngine::compute_from_chip(uint32_t fx_addr,uint32_t fy_addr,float* grad,FCCSTiming* t){
    auto&I=*impl_; FCCSTiming tm{}; auto t0=hrclock::now();
    const int na=I.n_active,nn=I.n_total;
    build_if_needed();

    // point the de-interleave reader at the current chip field addresses (args 1,2)
    { Program& pd=I.wld.get_programs().at(I.dr);
      for(uint32_t c=0;c<(uint32_t)I.nc;++c)
        SetRuntimeArgs(pd,I.kde,I.lc[c],{c,fx_addr,fy_addr,(uint32_t)I.fxb->address(),(uint32_t)I.fyb->address(),
            (uint32_t)I.NBX,(uint32_t)I.NBY,I.rpc,I.chip_pg,I.fx_pg}); }

    // ── on-chip de-interleave (chip field → contiguous fxb/fyb float) — replaces h2d ──
    auto th=hrclock::now(); EnqueueMeshWorkload(*I.cq,I.wld,false); Finish(*I.cq); tm.h2d_ms=ms_since(th);
    // ── SFPU field quantize (float→int32 in place) ──
    auto tq=hrclock::now(); EnqueueMeshWorkload(*I.cq,I.wlq,false); Finish(*I.cq); tm.quant_ms=ms_since(tq);
    // ── STREAMING: on-chip counting-sort cell-order geom → sorted DRAM + sort_start ──
    double sort_ms=0.0;
    if (I.streaming) { auto tso=hrclock::now(); EnqueueMeshWorkload(*I.cq,I.wls,false); Finish(*I.cq); sort_ms=ms_since(tso); }
    // ── field-cast backward (in-L1 or streaming) ──
    auto tr=hrclock::now(); EnqueueMeshWorkload(*I.cq,I.wl,false); auto tenq=hrclock::now(); Finish(*I.cq); tm.run_ms=ms_since(tr);
    if(getenv("FCCS_SPLIT")){ double enq=std::chrono::duration<double,std::milli>(tenq-tr).count();
        fprintf(stderr,"[fccs_split] deint=%.3f quant=%.3f sort=%.3f bwd_enqueue=%.3f bwd_finish=%.3f\n",
                tm.h2d_ms,tm.quant_ms,sort_ms,enq,tm.run_ms-enq); }

    auto td=hrclock::now();
    EnqueueReadMeshBuffer(*I.cq,I.gradp,I.grb,true);
    // grad[c] = bin_area·Σ(ox·oy·f)/2^(2WS+FS), ·ratio[c] when using HOST-GEOM (correct triangle
    // overlap → DREAMPlace's `gx*=ratio`). The old density-stash path omitted ratio AND used the
    // wrong footprint; the two errors cancelled in aggregate magnitude but not per-cell (rel_mov 0.25).
    const double s=(1.0/(double)((uint64_t)1u<<(2u*WS+FS)))*(double)I.bsx*(double)I.bsy;
    // ·ratio when the geometry is EF-correct (clamped+offset): host-geom OR the forward's
    // V31_EF_GEOM path. (The old orig-stash path baked ratio in via its wider footprint.)
    const bool efr=I.use_host_geom||(getenv("V31_EF_GEOM")&&atoi(getenv("V31_EF_GEOM")));
    // Big cells are sub-tiled (new_nc>nc): multiple sub-cells share a parent node (sel[i]).
    // Zero each touched slot once, then ACCUMULATE — so a wide cell's force = Σ of its sub-cells.
    for(int i=0;i<na;++i){ int g=I.sel[i]; grad[g]=0.f; grad[nn+g]=0.f; }
    for(int i=0;i<na;++i){ int g=I.sel[i]; double sr=efr?s*(double)I.ratio[i]:s;
        grad[g]+=(float)((double)I.gradp[(size_t)i*2]*sr); grad[nn+g]+=(float)((double)I.gradp[(size_t)i*2+1]*sr); }
    tm.d2h_ms=ms_since(td);
    tm.total_ms=ms_since(t0); if(t)*t=tm;
}
std::vector<int32_t> FCCSEFEngine::read_fxb(){ auto&I=*impl_; if(!I.fxb)return{};
    std::vector<int32_t> v(I.fx_pg/4u,0); EnqueueReadMeshBuffer(*I.cq,v,I.fxb,true); return v; }
std::vector<int64_t> FCCSEFEngine::read_grad_raw(){ auto&I=*impl_; if(!I.grb)return{};
    std::vector<int64_t> v(I.grad_pg/8u,0); EnqueueReadMeshBuffer(*I.cq,v,I.grb,true); return v; }
}  // namespace fccsef
