// SPDX-License-Identifier: Apache-2.0
//
// V25 EF backward — L1-resident field, direct-L1-load corner gather microbench.
// Tests the hypothesis that making each corner read a DIRECT L1 LOAD (field
// band resident in L1, cells spatially co-located) beats CPU's cache-resident
// gather. Self-contained: synth field+cells, run, in-C++ accuracy check vs a
// host reference, report kernel median + Tracy zones (TT_METAL_DEVICE_PROFILER=1).
//
// CLI: v25_ef_l1gather [H W ncells]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tt_metal.hpp>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;
using hrclock = std::chrono::high_resolution_clock;
template <class T> static double ms_since(T t0){return std::chrono::duration<double,std::milli>(hrclock::now()-t0).count();}

#ifndef DENSITY_KERNEL_DIR
#define DENSITY_KERNEL_DIR "kernels/"
#endif

int main(int argc, char** argv) {
    uint32_t H = 2048, W = 2048, ncells_total = 540000;
    if (argc >= 4) { H = atoi(argv[1]); W = atoi(argv[2]); ncells_total = atoi(argv[3]); }

    auto mesh = MeshDevice::create_unit_mesh(0);
    auto& cq = mesh->mesh_command_queue();
    auto grid = mesh->compute_with_storage_grid_size();
    const int nc = (int)(grid.x * grid.y);
    std::vector<CoreCoord> ccs;
    for (uint32_t y=0;y<grid.y;++y) for (uint32_t x=0;x<grid.x;++x) ccs.push_back({x,y});
    std::set<CoreRange> crs; for (auto c:ccs) crs.insert(CoreRange{c,c});
    CoreRangeSet all_crs(crs);

    // Band: rows per core + 1 halo (so byl and byl+1 are both in-band).
    const uint32_t band_rows = (H + nc - 1) / nc + 1;
    const uint32_t band_floats = band_rows * W * 2u;
    const uint32_t band_bytes  = band_floats * 4u;
    const uint32_t max_cells   = (ncells_total + nc - 1) / nc;
    const uint32_t cells_pg    = max_cells * 6u * 4u;  // 6 u32/cell
    const uint32_t grad_pg     = max_cells * 2u * 4u;  // gx,gy
    printf("[v25] H=%u W=%u ncells=%u  nc=%d band_rows=%u band=%.0f KB/core max_cells=%u\n",
           H, W, ncells_total, nc, band_rows, band_bytes/1024.0, max_cells);
    if (band_bytes > 1300u*1024u) { printf("[v25] WARN band %0.f KB may exceed L1\n", band_bytes/1024.0); }

    // ── Synthesize per-core field bands + cells, and host reference grad ──
    // Fixed-point: field & weights uploaded as int32 (×2^FRAC). Kernel does
    // integer madds → grad scaled by 2^(2*FRAC). Reference stays float.
    constexpr int FRAC = 13; const float SCALE = (float)(1 << FRAC);
    const double DESCALE = 1.0 / (double)((int64_t)1 << (2 * FRAC));
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> uf(-1.f, 1.f);
    std::vector<float>    field((size_t)nc * band_floats, 0.f);  // float (reference)
    std::vector<int32_t>  field_i((size_t)nc * band_floats, 0);  // int32 (upload)
    std::vector<uint32_t> cells((size_t)nc * max_cells * 6u, 0u);
    std::vector<float>    grad_ref((size_t)nc * max_cells * 2u, 0.f);
    for (size_t i = 0; i < field.size(); ++i) { field[i] = uf(rng); field_i[i] = (int32_t)lroundf(field[i] * SCALE); }

    uint32_t assigned = 0;
    for (int core = 0; core < nc; ++core) {
        const uint32_t take = std::min(max_cells, ncells_total - std::min(ncells_total, assigned));
        const float* fb = field.data() + (size_t)core * band_floats;
        for (uint32_t i = 0; i < max_cells; ++i) {
            const size_t cbase = ((size_t)core * max_cells + i) * 6u;
            if (i < take) {
                uint32_t bxl = (uint32_t)(rng() % (W - 2));
                uint32_t byl = (uint32_t)(rng() % (band_rows - 2));
                float wnw=uf(rng), wne=uf(rng), wsw=uf(rng), wse=uf(rng);
                cells[cbase+0]=bxl; cells[cbase+1]=byl;
                cells[cbase+2]=(uint32_t)(int32_t)lroundf(wnw*SCALE);
                cells[cbase+3]=(uint32_t)(int32_t)lroundf(wne*SCALE);
                cells[cbase+4]=(uint32_t)(int32_t)lroundf(wsw*SCALE);
                cells[cbase+5]=(uint32_t)(int32_t)lroundf(wse*SCALE);
                uint32_t r0 = byl*W + bxl;
                uint32_t inw=r0*2, ine=(r0+1)*2, isw=(r0+W)*2, ise=(r0+W+1)*2;
                float gx = wnw*fb[inw]+wne*fb[ine]+wsw*fb[isw]+wse*fb[ise];
                float gy = wnw*fb[inw+1]+wne*fb[ine+1]+wsw*fb[isw+1]+wse*fb[ise+1];
                grad_ref[((size_t)core*max_cells+i)*2+0]=gx;
                grad_ref[((size_t)core*max_cells+i)*2+1]=gy;
            }
        }
        assigned += take;
    }

    auto make_buf=[&](uint64_t bytes, uint32_t pg){
        DeviceLocalBufferConfig cfg{.page_size=pg,.buffer_type=BufferType::DRAM};
        ReplicatedBufferConfig r{.size=bytes};
        return MeshBuffer::create(r,cfg,mesh.get());
    };
    auto field_buf=make_buf((uint64_t)nc*band_bytes, band_bytes);
    auto cells_buf=make_buf((uint64_t)nc*cells_pg, cells_pg);
    auto grad_buf =make_buf((uint64_t)nc*grad_pg, grad_pg);

    EnqueueWriteMeshBuffer(cq, field_buf, field_i, false);   // int32 fixed-point field
    EnqueueWriteMeshBuffer(cq, cells_buf, cells, false);
    Finish(cq);

    using tt::CBIndex;
    Program prog = CreateProgram();
    auto mkcb=[&](uint32_t idx,uint32_t bytes){
        CircularBufferConfig c(bytes,{{(CBIndex)idx,DataFormat::Float32}});
        c.set_page_size((CBIndex)idx,bytes); CreateCircularBuffer(prog,all_crs,c);
    };
    mkcb((uint32_t)CBIndex::c_0, band_bytes);
    mkcb((uint32_t)CBIndex::c_1, cells_pg);
    mkcb((uint32_t)CBIndex::c_2, grad_pg);
    std::map<std::string,std::string> kdefs;
    if (std::getenv("V25_ABLATE_FLOAT")) kdefs["V25_ABLATE_FLOAT"] = "1";
    auto k = CreateKernel(prog, std::string(DENSITY_KERNEL_DIR)+"v25_ef_l1gather.cpp", all_crs,
        DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default,.defines=kdefs});
    const uint32_t fb=(uint32_t)field_buf->address(), cb=(uint32_t)cells_buf->address(), gb=(uint32_t)grad_buf->address();
    uint32_t assigned2=0;
    for (int core=0; core<nc; ++core){
        const uint32_t take=std::min(max_cells, ncells_total - std::min(ncells_total,assigned2));
        SetRuntimeArgs(prog,k,ccs[core],{(uint32_t)core,take,band_bytes,W,fb,cb,gb,cells_pg,grad_pg});
        assigned2+=take;
    }
    MeshWorkload wl; auto dr=MeshCoordinateRange{mesh->shape()}; wl.add_program(dr,std::move(prog));
    auto run=[&]{EnqueueMeshWorkload(cq,wl,false);Finish(cq);};

    printf("[v25] warmup...\n"); fflush(stdout);
    auto tj=hrclock::now(); run(); printf("[v25] warmup=%.2f ms\n", ms_since(tj));
    const int N=7; std::vector<double> ms;
    for (int i=0;i<N;++i){auto t=hrclock::now();run();ms.push_back(ms_since(t));}
    std::sort(ms.begin(),ms.end());
    double med=ms[N/2];
    printf("[v25] kernel runs: "); for(double m:ms)printf("%.3f ",m);
    printf("\n[v25] kernel MEDIAN = %.3f ms   (CPU EF baseline ~7.3 ms)\n", med);

    // ── Accuracy (descale int32 grad by 2^(2*FRAC)) ──
    std::vector<int32_t> grad_out_i((size_t)nc*max_cells*2u, 0);
    EnqueueReadMeshBuffer(cq, grad_out_i, grad_buf, true);
    double max_abs=0, ref_n=0, dif_n=0; size_t nbad=0;
    uint32_t assigned3=0;
    for (int core=0; core<nc; ++core){
        const uint32_t take=std::min(max_cells, ncells_total - std::min(ncells_total,assigned3));
        assigned3+=take;
        for (uint32_t i=0;i<take;++i){           // real cells only (skip uninitialized padding)
            const size_t idx=((size_t)core*max_cells+i)*2;
            for (int k=0;k<2;++k){
                double gk=(double)grad_out_i[idx+k]*DESCALE;
                double d=std::fabs(gk-grad_ref[idx+k]);
                max_abs=std::max(max_abs,d);
                ref_n+=(double)grad_ref[idx+k]*grad_ref[idx+k]; dif_n+=d*d;
                if (d>1e-3*std::max(1.0f,std::fabs(grad_ref[idx+k]))) nbad++;
            }
        }
    }
    double rel_l2 = ref_n>0 ? std::sqrt(dif_n/ref_n) : 0;
    printf("[v25] accuracy: max_abs=%.6g rel_l2=%.6g nbad=%zu\n", max_abs, rel_l2, nbad);
    printf("V25_RESULT,%u,%u,%u,%.3f,%.6g\n", H, W, ncells_total, med, rel_l2);
    return 0;
}
