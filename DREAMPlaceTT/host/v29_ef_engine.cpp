// SPDX-License-Identifier: Apache-2.0
//
// V29 EF backward engine — persistent-buffer port of host/v29_host.cpp.
// See v29_ef_engine.h. Field buffer is ONE page (whole field) so x-bin bands are
// contiguous (matches the v29 gather's flat-offset band read).

#include "v29_ef_engine.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
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
namespace v29ef {
static constexpr uint32_t MAXKH = 8;

struct V29EFEngine::Impl {
    MeshDevice* dev; MeshCommandQueue* cq;
    int NBX, NBY, nmax; float xl, yl, bsx, bsy, inv_bsx, inv_bsy;
    int nc; CoreCoord grid; std::vector<CoreCoord> ccs; CoreRangeSet crs{std::set<CoreRange>{}};
    uint32_t cpc_x=0, band_cols=0, span_cap=1;   // span_cap = max consecutive bands one worker merges
    // configured
    int n_active=0, n_total=0; std::vector<int32_t> sel;
    std::vector<float> ox,oy,nsx,nsy,ratio;
    uint32_t max_k=1, max_h=1;
    // prep-reuse: forward per-cell geometry buffer (V31_GEOM). geom_addr!=0 → P1 is
    // v29_bucket_only reading pre-computed {idx,bxl,byl,kc,hc,ratio=bin_area,px[8],py[8]}.
    uint32_t geom_addr=0, geom_pg=0;
    uint32_t diag_n=0;   // V29_DIAG one-time print counter
    // sizing (set on first compute)
    bool built=false; uint32_t cap=0, mbatch=1, cpc_cell=0;
    uint32_t in_pg=0, route_pg=0, cnt_pg=0, field_pg=0, grad_pg=0, oidx_pg=0;
    std::shared_ptr<MeshBuffer> inb, rb, cb, fxb, fyb, gb, oib;
    MeshWorkload wl1, wl2; MeshCoordinateRange dr{MeshCoordinate{0,0},MeshCoordinate{0,0}};
    KernelHandle k1=0, gk=0, ck=0, wk=0;
    // staging
    std::vector<uint32_t> in_stage; std::vector<uint32_t> dcnt; std::vector<float> gradp; std::vector<uint32_t> nbat; std::vector<uint32_t> oidx_host; std::vector<uint32_t> core_nb;

    Impl(void* md,int M_,int N_,int nmax_,float xl_,float yl_,float bsx_,float bsy_)
      : dev(static_cast<MeshDevice*>(md)), NBX(M_), NBY(N_), nmax(nmax_),
        xl(xl_),yl(yl_),bsx(bsx_),bsy(bsy_),inv_bsx(1.f/bsx_),inv_bsy(1.f/bsy_) {
        cq=&dev->mesh_command_queue(); grid=dev->compute_with_storage_grid_size(); nc=grid.x*grid.y;
        std::set<CoreRange> s; for(uint32_t y=0;y<grid.y;++y)for(uint32_t x=0;x<grid.x;++x){CoreCoord c{x,y};ccs.push_back(c);s.insert(CoreRange{c,c});}
        crs=CoreRangeSet(s);
        cpc_x=(NBX+nc-1)/nc; band_cols=std::min(cpc_x+MAXKH,(uint32_t)NBX);
    }
};

static uint32_t pf(float v){union{uint32_t u;float f;}c;c.f=v;return c.u;}

V29EFEngine::V29EFEngine(void* md,int M,int N,int nmax,float xl,float yl,float bsx,float bsy)
  : impl_(std::make_unique<Impl>(md,M,N,nmax,xl,yl,bsx,bsy)) {}
V29EFEngine::~V29EFEngine() = default;

void V29EFEngine::configure_with_sel(const float* ox_f,const float* oy_f,const float* nsx_f,
        const float* nsy_f,const float* ratio_f,const int32_t* sel,int na,int nt){
    auto&I=*impl_; I.n_active=na; I.n_total=nt; I.sel.assign(sel,sel+na);
    I.ox.resize(na);I.oy.resize(na);I.nsx.resize(na);I.nsy.resize(na);I.ratio.resize(na);
    for(int i=0;i<na;++i){int g=sel[i];I.ox[i]=ox_f[g];I.oy[i]=oy_f[g];I.nsx[i]=nsx_f[g];I.nsy[i]=nsy_f[g];I.ratio[i]=ratio_f[g];}
    // span bounds from clamped sizes (stable across iters)
    uint32_t mk=1,mh=1; for(int i=0;i<na;++i){uint32_t k=(uint32_t)std::floor(I.nsx[i]*I.inv_bsx)+2;uint32_t h=(uint32_t)std::floor(I.nsy[i]*I.inv_bsy)+2;mk=std::max(mk,k);mh=std::max(mh,h);}
    I.max_k=std::min(mk,MAXKH); I.max_h=std::min(mh,MAXKH);
    I.cpc_cell=((uint32_t)na+I.nc-1)/I.nc;
}

void V29EFEngine::set_geom_source(uint32_t geom_addr,uint32_t geom_pg){
    impl_->geom_addr=geom_addr; impl_->geom_pg=geom_pg;
}

void V29EFEngine::compute_from_full(const float* pos,const float* fx,const float* fy,float* grad,V29Timing* t){
    auto&I=*impl_; V29Timing tm{}; auto t0=hrclock::now();
    const int na=I.n_active, nn=I.n_total, NBX=I.NBX, NBY=I.NBY, nc=I.nc;
    // ── build 64B input [x,y,ox,oy,nsx,nsy,ratio] per active cell ──
    if(I.in_stage.empty()) I.in_stage.assign((size_t)na*16u+16u,0u);
    {union{uint32_t u;float f;}c; float* s=reinterpret_cast<float*>(I.in_stage.data());
     for(int i=0;i<na;++i){int g=I.sel[i];float*r=s+(size_t)i*16;
        r[0]=pos[g];r[1]=pos[nn+g];r[2]=I.ox[i];r[3]=I.oy[i];r[4]=I.nsx[i];r[5]=I.nsy[i];r[6]=I.ratio[i];}}

    const bool use_geom = (I.geom_addr != 0u);
    // ── first call: size cap (pre-count bin_xl on this pos) + alloc + build ──
    if(!I.built){
        // Geom path reuses the forward's ORIG-size overlaps (orig = clamped + 2·offset),
        // which span MORE bins than the clamped sizes configure_with_sel measured. The
        // gather loops max_k×max_h (NOT per-cell kc) and drops any bin beyond it, so
        // max_k must cover the orig span — but no larger (each extra k/h is a full
        // 1024-cell field-gather pass; max_k=8 ballooned the gather ~16×). Bound it by
        // the orig width, capped at 8 (the record holds 8 px/py slots).
        if(use_geom){
            uint32_t mk=1,mh=1;
            for(int i=0;i<na;++i){
                float ow=I.nsx[i]+2.f*std::fabs(I.ox[i]); float oh=I.nsy[i]+2.f*std::fabs(I.oy[i]);
                uint32_t k=(uint32_t)std::floor(ow*I.inv_bsx)+2u; uint32_t h=(uint32_t)std::floor(oh*I.inv_bsy)+2u;
                mk=std::max(mk,k); mh=std::max(mh,h);
            }
            I.max_k=std::min(mk,MAXKH); I.max_h=std::min(mh,MAXKH);
        }
        std::vector<uint32_t> hc((size_t)nc*nc,0); uint32_t maxc=0;
        std::vector<uint32_t> bandtot(nc,0);
        const float* s=reinterpret_cast<const float*>(I.in_stage.data());
        for(int i=0;i<na;++i){const float*r=s+(size_t)i*16; float nx=r[0]+r[2];
            int bxl=(int)((nx-I.xl)*I.inv_bsx); if(bxl<0)bxl=0; if(bxl>=NBX)bxl=NBX-1;
            uint32_t src=(uint32_t)i/I.cpc_cell; if(src>=(uint32_t)nc)src=nc-1;
            uint32_t band=(uint32_t)bxl/I.cpc_x; if(band>=(uint32_t)nc)band=nc-1;
            uint32_t v=++hc[(size_t)src*nc+band]; maxc=std::max(maxc,v); bandtot[band]++;}
        uint32_t maxband=1; for(int b=0;b<nc;++b) maxband=std::max(maxband,bandtot[b]); (void)maxband;
        I.cap=maxc*3u+64u;                          // per-(src,band) page (initial = most clustered)
        // COLD-MERGE sizing: a worker covers up to span_cap consecutive bands sharing
        // ONE wide field band resident in L1 (bounded ~256 KB). Cold bands merge into
        // one worker (vs the old "≥1 core per non-empty band" that starved hot bands).
        { const uint32_t FIELD_L1=262144u;
          uint32_t budget_cols=std::max(I.max_k+1u, FIELD_L1/((uint32_t)NBY*4u));
          I.span_cap=std::max(1u,(budget_cols>I.max_k)?(budget_cols-I.max_k)/I.cpc_x:1u);
          I.band_cols=std::min(I.span_cap*I.cpc_x + I.max_k, (uint32_t)NBX); }
        // mbatch (per-worker page = grad/oidx d2h size) sized from the ACTUAL iter-0
        // grouping max worker load — NOT a blind ×3. iter-0 (random_center_init) is the
        // most clustered iter, so this is the worst case; +2 batches margin covers drift.
        // The old ×3 (=14 here) bloated the grad+oidx readback (~189 MB) → 6.5 ms d2h;
        // tight mbatch (~6) cuts the readback ~2.5× → smaller d2h.
        { uint32_t total=0; for(int b=0;b<nc;++b) total+=bandtot[b];
          uint32_t tgt=(total+(uint32_t)nc-1)/(uint32_t)nc; if(tgt<8)tgt=8;
          auto nw=[&](uint32_t t)->uint32_t{ uint32_t w=0; int b=0;
            while(b<nc){ if(bandtot[b]>=t){ w+=(bandtot[b]+t-1)/t; ++b; }
              else { uint32_t r=0,ns=0; while(b<nc && ns<I.span_cap && bandtot[b]<t && r+bandtot[b]<=t){ r+=bandtot[b]; ++ns; ++b; } if(!ns)++b; ++w; } } return w; };
          for(int it=0; it<40 && nw(tgt)>(uint32_t)nc; ++it) tgt+=tgt/4u+8u;
          I.mbatch=std::min((tgt+1023u)/1024u + 2u, 32u); }
        I.in_pg=(uint32_t)na*64u+64u; I.route_pg=I.cap*128u; I.cnt_pg=((nc*4u+63u)&~63u);
        I.field_pg=(uint32_t)((uint64_t)NBX*NBY*4u);
        I.grad_pg=I.mbatch*1024u*8u;        // per-band page: [gx,gy] per cell, interleaved → fast d2h
        I.oidx_pg=I.mbatch*1024u*4u;
        auto mb=[&](uint64_t b,uint32_t pg){DeviceLocalBufferConfig c{.page_size=pg,.buffer_type=BufferType::DRAM};ReplicatedBufferConfig r{.size=b};return MeshBuffer::create(r,c,I.dev);};
        I.inb=mb(I.in_pg,I.in_pg); I.rb=mb((uint64_t)nc*nc*I.route_pg,I.route_pg); I.cb=mb((uint64_t)nc*I.cnt_pg,I.cnt_pg);
        I.fxb=mb(I.field_pg,I.field_pg); I.fyb=mb(I.field_pg,I.field_pg);
        I.gb=mb((uint64_t)nc*I.grad_pg,I.grad_pg); I.oib=mb((uint64_t)nc*I.oidx_pg,I.oidx_pg);
        const size_t slots=(size_t)I.mbatch*1024u;
        I.dcnt.assign((size_t)nc*(I.cnt_pg/4u),0); I.gradp.assign((size_t)nc*slots*2u,0.f); I.oidx_host.assign((size_t)nc*slots,0u); I.nbat.assign(nc,0); I.core_nb.assign(nc,0);
        // ── programs ──
        using tt::CBIndex;
        auto cbf=[&](Program&p,uint32_t idx,uint32_t bytes,uint32_t slots){CircularBufferConfig c(slots*bytes,{{(CBIndex)idx,DataFormat::Float32}});c.set_page_size((CBIndex)idx,bytes);CreateCircularBuffer(p,I.crs,c);};
        uint32_t ia=I.inb->address(),ra=I.rb->address(),ca=I.cb->address(),fxa=I.fxb->address(),fya=I.fyb->address(),ga=I.gb->address(),oa=I.oib->address();
        // P1 bucket (prep-reuse: read forward geom records, no soft-float) OR legacy prep+bucket
        Program p1=CreateProgram();
        if(use_geom){
            const uint32_t pchunk=1024;
            cbf(p1,0,pchunk*128u,1); cbf(p1,1,nc*4u,1); cbf(p1,2,128u,1);   // c_0 holds 128B records
            I.k1=CreateKernel(p1,std::string(DENSITY_KERNEL_DIR)+"v29_bucket_only.cpp",I.crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
            for(int c=0;c<nc;++c){uint32_t f=(uint32_t)c*I.cpc_cell,n=std::min(I.cpc_cell,(f<(uint32_t)na)?((uint32_t)na-f):0u);
                SetRuntimeArgs(p1,I.k1,I.ccs[c],{(uint32_t)c,f,n,I.geom_addr,I.geom_pg,ra,I.cap,(uint32_t)nc,I.cpc_x,ca,pchunk,I.cnt_pg});}
        } else {
            const uint32_t pchunk=2048;
            cbf(p1,0,pchunk*64u,1); cbf(p1,1,nc*4u,1); cbf(p1,2,128u,1);
            I.k1=CreateKernel(p1,std::string(DENSITY_KERNEL_DIR)+"v29_prepbucket.cpp",I.crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
            for(int c=0;c<nc;++c){uint32_t f=(uint32_t)c*I.cpc_cell,n=std::min(I.cpc_cell,(f<(uint32_t)na)?((uint32_t)na-f):0u);
                SetRuntimeArgs(p1,I.k1,I.ccs[c],{(uint32_t)c,f,n,ia,I.in_pg,ra,I.cap,(uint32_t)nc,I.cpc_x,ca,pchunk,(uint32_t)NBX,(uint32_t)NBY,pf(I.xl),pf(I.yl),pf(I.bsx),pf(I.bsy),pf(I.inv_bsx),pf(I.inv_bsy),I.cnt_pg});}
        }
        I.dr=MeshCoordinateRange{I.dev->shape()}; I.wl1.add_program(I.dr,std::move(p1));
        // P2 gather+compute+writer
        Program p2=CreateProgram();
        cbf(p2,0,4096,I.max_k);cbf(p2,1,4096,I.max_h);cbf(p2,2,4096,2);cbf(p2,3,4096,2);cbf(p2,4,4096,2);
        cbf(p2,5,4096,1);cbf(p2,6,4096,1); cbf(p2,16,4096,2);cbf(p2,17,4096,2);
        cbf(p2,24,I.band_cols*NBY*4u,1);cbf(p2,25,1024u*128u,1);cbf(p2,26,I.band_cols*NBY*4u,1);cbf(p2,27,nc*4u+128u,1);cbf(p2,28,1024u*16u,1);
        std::vector<UnpackToDestMode> um(64,UnpackToDestMode::Default);
        for(uint32_t i=0;i<5;++i)um[i]=UnpackToDestMode::UnpackToDestFp32; um[16]=um[17]=UnpackToDestMode::UnpackToDestFp32;
        I.gk=CreateKernel(p2,std::string(DENSITY_KERNEL_DIR)+"v29_gather_brisc.cpp",I.crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        I.wk=CreateKernel(p2,std::string(DENSITY_KERNEL_DIR)+"v29_writer.cpp",I.crs,DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
        I.ck=CreateKernel(p2,std::string(DENSITY_KERNEL_DIR)+"v28_compute.cpp",I.crs,ComputeConfig{.fp32_dest_acc_en=true,.unpack_to_dest_mode=um,.math_approx_mode=false});
        for(int c=0;c<nc;++c){uint32_t col0=(uint32_t)c*I.cpc_x,vcols=(col0<(uint32_t)NBX)?std::min(I.band_cols,(uint32_t)NBX-col0):1u;uint32_t col0c=(col0<(uint32_t)NBX)?col0:(NBX-1);
            // fixed n_batches=mbatch for all cores: the gather reads per-src counts
            // on-chip, emits real batches then pads to mbatch; compute/writer loop
            // mbatch. No per-iter host counts readout or arg updates.
            SetRuntimeArgs(p2,I.gk,I.ccs[c],{(uint32_t)c,ra,I.cap,ca,(uint32_t)nc,I.max_k,I.max_h,fxa,fya,I.field_pg,(uint32_t)NBY,I.cpc_x,vcols,I.mbatch,I.cnt_pg,col0c,oa,I.mbatch,(uint32_t)c,0u,0u,1u});
            SetRuntimeArgs(p2,I.ck,I.ccs[c],{I.mbatch,I.max_k,I.max_h});
            SetRuntimeArgs(p2,I.wk,I.ccs[c],{I.mbatch,ga,I.grad_pg,(uint32_t)c});}
        I.wl2.add_program(I.dr,std::move(p2));
        I.built=true;
    }

    // ── upload field (+ input, unless the geom path reads the forward buffer) ──
    auto th=hrclock::now();
    if(!use_geom) EnqueueWriteMeshBuffer(*I.cq,I.inb,I.in_stage,false);
    {std::vector<float> fxv(fx,fx+(size_t)NBX*NBY),fyv(fy,fy+(size_t)NBX*NBY);
     EnqueueWriteMeshBuffer(*I.cq,I.fxb,fxv,false); EnqueueWriteMeshBuffer(*I.cq,I.fyb,fyv,false);}
    Finish(*I.cq); tm.h2d_ms=ms_since(th);

    // ── P1 prep+bucket ──
    auto tp=hrclock::now(); EnqueueMeshWorkload(*I.cq,I.wl1,false); Finish(*I.cq); tm.prep_ms=ms_since(tp);

    // ── read counts → ADAPTIVE band→core assignment. Each core gets ~na/nc cells
    //    (balanced); hot bands are split across multiple cores (field replicated),
    //    cold bands share one. Per-core output page → small uniform mbatch. ──
    EnqueueReadMeshBuffer(*I.cq,I.dcnt,I.cb,true);
    { Program& p2=I.wl2.get_programs().at(I.dr);
      const uint32_t rpw=I.cnt_pg/4u, ra=I.rb->address(),ca=I.cb->address(),fxa=I.fxb->address(),
                     fya=I.fyb->address(),ga=I.gb->address(),oa=I.oib->address();
      std::vector<uint32_t> bandtot(nc,0); uint32_t total=0;
      for(int b=0;b<nc;++b){ uint32_t t=0; for(int s=0;s<nc;++s)t+=I.dcnt[(size_t)s*rpw+b]; bandtot[b]=t; total+=t; }
      const uint32_t cap_cells=I.mbatch*1024u;          // max cells one worker can take
      // ── COLD-MERGE grouping (ported from v30_host): split a HOT band (≥target cells)
      //    across several workers (n_bands=1, skip/my_cells sub-range, field replicated);
      //    MERGE consecutive COLD bands (<target) into ONE worker (n_bands>1, one wide
      //    field band) up to span_cap bands / target cells. Grow target until #workers
      //    ≤ nc. Frees the cores the old "≥1 core per non-empty band" wasted on cold
      //    bands → the hot band gets enough workers (imbalance 3.8×→~1.5×). ──
      uint32_t target=(total+(uint32_t)nc-1)/(uint32_t)nc; if(target<8)target=8;
      auto nworkers=[&](uint32_t tgt)->uint32_t{ uint32_t w=0; int b=0;
        while(b<nc){ if(bandtot[b]>=tgt){ w+=(bandtot[b]+tgt-1)/tgt; ++b; }
          else { uint32_t r=0,ns=0; while(b<nc && ns<I.span_cap && bandtot[b]<tgt && r+bandtot[b]<=tgt){ r+=bandtot[b]; ++ns; ++b; } if(!ns)++b; ++w; } }
        return w; };
      for(int it=0; it<40 && nworkers(target)>(uint32_t)nc; ++it) target+=target/4u+8u;
      if(target>cap_cells) target=cap_cells;
      auto setcore=[&](int core,uint32_t lo_band,uint32_t n_bands,uint32_t skip,uint32_t mycells){
        uint32_t nb=(mycells+1023u)/1024u; if(nb>I.mbatch)nb=I.mbatch; if(nb<1)nb=1; I.core_nb[core]=(mycells?nb:0u);
        uint32_t col0=lo_band*I.cpc_x; uint32_t col0c=(col0<(uint32_t)NBX)?col0:(NBX-1);
        uint32_t wide=n_bands*I.cpc_x+I.max_k;
        uint32_t vcols=(col0<(uint32_t)NBX)?std::min(std::min(wide,I.band_cols),(uint32_t)NBX-col0):1u;
        SetRuntimeArgs(p2,I.gk,I.ccs[core],{lo_band,ra,I.cap,ca,(uint32_t)nc,I.max_k,I.max_h,fxa,fya,I.field_pg,(uint32_t)NBY,I.cpc_x,vcols,nb,I.cnt_pg,col0c,oa,I.mbatch,(uint32_t)core,skip,mycells,n_bands});
        SetRuntimeArgs(p2,I.ck,I.ccs[core],{nb,I.max_k,I.max_h});
        SetRuntimeArgs(p2,I.wk,I.ccs[core],{nb,ga,I.grad_pg,(uint32_t)core}); };
      int core=0,b=0;
      while(b<nc && core<nc){
        if(bandtot[b]>=target){                                   // hot band → split across workers
          uint32_t r=bandtot[b],nparts=(r+target-1)/target,base=r/nparts,rem=r%nparts,off=0;
          for(uint32_t j=0;j<nparts && core<nc;++j){ uint32_t c=base+(j<rem?1u:0u); setcore(core++,(uint32_t)b,1u,off,c); off+=c; } ++b;
        } else {                                                  // merge consecutive cold bands
          uint32_t r=0,ns=0,lo=(uint32_t)b;
          while(b<nc && ns<I.span_cap && bandtot[b]<target && r+bandtot[b]<=target){ r+=bandtot[b]; ++ns; ++b; }
          if(!ns){ ns=1; ++b; } setcore(core++,lo,ns,0u,r);
        }
      }
      for(;core<nc;++core) setcore(core,0,1,0,0);   // idle workers: 0 cells
      if(getenv("V29_DIAG") && I.diag_n++ < 3){
        uint32_t nbmax=0,nbsum=0,active=0,bmax=0; for(int c=0;c<nc;++c){ if(I.core_nb[c]){++active; nbsum+=I.core_nb[c]; nbmax=std::max(nbmax,I.core_nb[c]); } }
        for(int b=0;b<nc;++b) bmax=std::max(bmax,bandtot[b]);
        fprintf(stderr,"[v29_diag] na=%d nc=%d max_k=%u max_h=%u mbatch=%u cap=%u | total=%u target=%u bandtot_max=%u | nb: max=%u mean=%.2f active_cores=%u/%d | est_gather_passes=max_k*max_h*nbmax=%u\n",
                na,nc,I.max_k,I.max_h,I.mbatch,I.cap,total,target,bmax,nbmax,active?(double)nbsum/active:0.0,active,nc,I.max_k*I.max_h*nbmax);
      }
    }

    // ── P2 gather+compute+writer ──
    auto tg=hrclock::now(); EnqueueMeshWorkload(*I.cq,I.wl2,false); Finish(*I.cq); tm.gather_ms=ms_since(tg);

    // ── d2h grad_buf (interleaved [gx,gy]/cell, bucketed) + oidx_buf; host unsort ──
    auto td=hrclock::now();
    EnqueueReadMeshBuffer(*I.cq,I.gradp,I.gb,true);
    EnqueueReadMeshBuffer(*I.cq,I.oidx_host,I.oib,true);
    const size_t slots=(size_t)I.mbatch*1024u;
    for(int c=0;c<nc;++c){ const size_t cb=(size_t)c*slots; const size_t real=(size_t)I.core_nb[c]*1024u;
        for(size_t s=0;s<real;++s){ uint32_t oi=I.oidx_host[cb+s]; if(oi==0xFFFFFFFFu||(int)oi>=na)continue;
            int g=I.sel[oi]; size_t k=(cb+s)*2; grad[g]=I.gradp[k]; grad[nn+g]=I.gradp[k+1]; } }
    tm.d2h_ms=ms_since(td);
    tm.total_ms=ms_since(t0); if(t)*t=tm;
}
}  // namespace v29ef
