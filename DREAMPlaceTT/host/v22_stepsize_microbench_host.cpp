// SPDX-License-Identifier: Apache-2.0
//
// V22 K5 — Barzilai-Borwein step-size reduction microbench (TT-side host driver).
//
// Computes the three global reductions ss=Σs·s, sy=Σs·y, yy=Σy·y over
// 2*num_nodes fp32, then the production step size
//   bb_short = sy/yy ; lip = sqrt(ss/yy) ; step = bb_short>0 ? bb_short : min(lip, alpha_k)
// (NesterovAcceleratedGradientOptimizer.py:212-217).
//
// The chip does the per-core lane-wise reduction (compute kernel → nc_all*3
// partial tiles); this host sums those partials (the trivial cross-core reduce)
// into the 3 scalars + step. The Python orchestrator runs the torch reference.
//
// Input binary (LE):
//   header : { u32 magic=0x56323250, u32 num_nodes, f32 alpha_k, u32 pad }    (16 B)
//   s_k    : 2*num_nodes fp32
//   y_k    : 2*num_nodes fp32
// Output binary:
//   header : { u32 magic=0x56323251, u32 num_nodes, u32 kernel_ms_x1000 }     (12 B)
//   vals   : f32 ss, f32 sy, f32 yy, f32 step

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tt_metal.hpp>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;
using hrclock = std::chrono::high_resolution_clock;
template <class T> static double ms_since(T t0) { return std::chrono::duration<double, std::milli>(hrclock::now() - t0).count(); }

#ifndef DENSITY_KERNEL_DIR
#define DENSITY_KERNEL_DIR "kernels/"
#endif

struct InputHeader  { uint32_t magic, num_nodes; float alpha_k; uint32_t pad; };
struct OutputHeader { uint32_t magic, num_nodes, kernel_ms_x1000; };

int main(int argc, char** argv) {
    std::string inputs_path, output_path;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--inputs" && i + 1 < argc)      inputs_path = argv[++i];
        else if (a == "--output" && i + 1 < argc) output_path = argv[++i];
    }
    if (inputs_path.empty() || output_path.empty()) {
        fprintf(stderr, "Usage: %s --inputs <p> --output <p>\n", argv[0]); return 1;
    }

    std::ifstream fin(inputs_path, std::ios::binary);
    if (!fin) { fprintf(stderr, "[v22ss] FATAL: cannot open %s\n", inputs_path.c_str()); return 1; }
    InputHeader hdr{};
    fin.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (hdr.magic != 0x56323250u) { fprintf(stderr, "[v22ss] FATAL: bad magic 0x%08x\n", hdr.magic); return 1; }
    const uint32_t num_nodes = hdr.num_nodes;
    const uint32_t n_elems   = num_nodes * 2u;
    const float alpha_k = hdr.alpha_k;
    printf("[v22ss] num_nodes=%u n_elems=%u alpha_k=%g\n", num_nodes, n_elems, alpha_k);

    auto read_floats = [&](size_t n) { std::vector<float> v(n); fin.read(reinterpret_cast<char*>(v.data()), (std::streamsize)(n*sizeof(float))); return v; };
    std::vector<float> s_k = read_floats(n_elems);
    std::vector<float> y_k = read_floats(n_elems);
    if (!fin) { fprintf(stderr, "[v22ss] FATAL: short read\n"); return 1; }
    fin.close();

    auto mesh_device = MeshDevice::create_unit_mesh(0);
    auto& cq = mesh_device->mesh_command_queue();
    auto grid = mesh_device->compute_with_storage_grid_size();
    const int nc_all = (int)(grid.x * grid.y);
    printf("[v22ss] grid=%ux%u nc_all=%d\n", grid.x, grid.y, nc_all);
    std::vector<CoreCoord> all_ccs;
    for (uint32_t y = 0; y < grid.y; ++y) for (uint32_t x = 0; x < grid.x; ++x) all_ccs.push_back(CoreCoord{x, y});
    std::set<CoreRange> crs_set;
    for (auto cc : all_ccs) crs_set.insert(CoreRange{cc, cc});
    CoreRangeSet all_crs(crs_set);

    using tt::CBIndex;
    constexpr uint32_t TILE_ELEMS = 32u * 32u;
    constexpr uint32_t TILE_BYTES = TILE_ELEMS * 4u;
    const uint32_t n_tiles_total = (n_elems + TILE_ELEMS - 1u) / TILE_ELEMS;
    const uint32_t padded_elems  = n_tiles_total * TILE_ELEMS;
    s_k.resize(padded_elems, 0.0f);
    y_k.resize(padded_elems, 0.0f);

    const uint32_t tiles_per_core = (n_tiles_total + (uint32_t)nc_all - 1u) / (uint32_t)nc_all;
    printf("[v22ss] n_tiles_total=%u tiles_per_core(max)=%u\n", n_tiles_total, tiles_per_core);

    auto make_buf = [&](uint64_t total_bytes, uint32_t pgsz) {
        DeviceLocalBufferConfig cfg{.page_size = pgsz, .buffer_type = BufferType::DRAM};
        ReplicatedBufferConfig rcfg{.size = total_bytes};
        return MeshBuffer::create(rcfg, cfg, mesh_device.get());
    };
    auto s_buf   = make_buf((uint64_t)n_tiles_total * TILE_BYTES, TILE_BYTES);
    auto y_buf   = make_buf((uint64_t)n_tiles_total * TILE_BYTES, TILE_BYTES);
    auto out_buf = make_buf((uint64_t)nc_all * 3u * TILE_BYTES, TILE_BYTES);

    auto t_h2d = hrclock::now();
    EnqueueWriteMeshBuffer(cq, s_buf, s_k, false);
    EnqueueWriteMeshBuffer(cq, y_buf, y_k, false);
    Finish(cq);
    printf("[v22ss] h2d = %.2f ms\n", ms_since(t_h2d));

    const uint32_t addr_s = (uint32_t)s_buf->address();
    const uint32_t addr_y = (uint32_t)y_buf->address();
    const uint32_t addr_o = (uint32_t)out_buf->address();

    Program prog = CreateProgram();
    auto make_cb = [&](uint32_t idx, uint32_t n_slots) {
        CircularBufferConfig cfg(n_slots * TILE_BYTES, {{(CBIndex)idx, DataFormat::Float32}});
        cfg.set_page_size((CBIndex)idx, TILE_BYTES);
        CreateCircularBuffer(prog, all_crs, cfg);
    };
    make_cb((uint32_t)CBIndex::c_0, 4);
    make_cb((uint32_t)CBIndex::c_1, 4);
    make_cb((uint32_t)CBIndex::c_16, 4);

    auto reader_k = CreateKernel(prog, std::string(DENSITY_KERNEL_DIR) + "v22_stepsize_reader.cpp", all_crs,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    auto writer_k = CreateKernel(prog, std::string(DENSITY_KERNEL_DIR) + "v22_stepsize_writer.cpp", all_crs,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    std::vector<UnpackToDestMode> unpack_modes(64, UnpackToDestMode::Default);
    unpack_modes[0] = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[1] = UnpackToDestMode::UnpackToDestFp32;
    auto compute_k = CreateKernel(prog, std::string(DENSITY_KERNEL_DIR) + "v22_stepsize_compute.cpp", all_crs,
        ComputeConfig{.fp32_dest_acc_en = true, .unpack_to_dest_mode = unpack_modes, .math_approx_mode = false});

    std::vector<int> core_nt(nc_all, 0);
    for (int c = 0; c < nc_all; ++c) {
        const uint32_t tile_start = std::min((uint32_t)c * tiles_per_core, n_tiles_total);
        const uint32_t tile_end   = std::min(tile_start + tiles_per_core, n_tiles_total);
        const uint32_t nt = tile_end - tile_start;
        core_nt[c] = (int)nt;
        SetRuntimeArgs(prog, reader_k, all_ccs[c], {nt, tile_start, addr_s, addr_y});
        SetRuntimeArgs(prog, writer_k, all_ccs[c], {(uint32_t)c * 3u, addr_o});
        SetRuntimeArgs(prog, compute_k, all_ccs[c], {nt});
    }

    MeshWorkload wl;
    auto device_range = MeshCoordinateRange{mesh_device->shape()};
    wl.add_program(device_range, std::move(prog));
    auto run_once = [&]() { EnqueueMeshWorkload(cq, wl, false); Finish(cq); };

    printf("[v22ss] JIT warmup...\n"); fflush(stdout);
    auto t_jit = hrclock::now(); run_once();
    printf("[v22ss] warmup = %.2f ms\n", ms_since(t_jit));

    const int N_RUNS = 5;
    std::vector<double> run_ms;
    for (int r = 0; r < N_RUNS; ++r) { auto t = hrclock::now(); run_once(); run_ms.push_back(ms_since(t)); }
    std::sort(run_ms.begin(), run_ms.end());
    double median_ms = run_ms[N_RUNS / 2];
    printf("[v22ss] runs (N=%d): ", N_RUNS);
    for (double m : run_ms) printf("%.3f ", m);
    printf("→ median %.3f ms\n", median_ms);

    // d2h partial tiles, sum on host (only cores that had tiles).
    std::vector<float> partials((size_t)nc_all * 3u * TILE_ELEMS, 0.0f);
    EnqueueReadMeshBuffer(cq, partials, out_buf, true);
    double ss = 0.0, sy = 0.0, yy = 0.0;
    for (int c = 0; c < nc_all; ++c) {
        if (core_nt[c] == 0) continue;  // skip cores with no shard (garbage partials)
        const size_t base = (size_t)c * 3u * TILE_ELEMS;
        double acc0 = 0, acc1 = 0, acc2 = 0;
        for (uint32_t i = 0; i < TILE_ELEMS; ++i) {
            acc0 += partials[base + 0 * TILE_ELEMS + i];
            acc1 += partials[base + 1 * TILE_ELEMS + i];
            acc2 += partials[base + 2 * TILE_ELEMS + i];
        }
        ss += acc0; sy += acc1; yy += acc2;
    }
    const float bb_short = (yy != 0.0) ? (float)(sy / yy) : 0.0f;
    const float lip      = (yy > 0.0)  ? (float)std::sqrt(ss / yy) : 0.0f;
    const float step     = (bb_short > 0.0f) ? bb_short : std::min(lip, alpha_k);
    printf("[v22ss] ss=%.6g sy=%.6g yy=%.6g  bb_short=%.6g lip=%.6g step=%.6g\n",
           ss, sy, yy, bb_short, lip, step);

    std::ofstream fout(output_path, std::ios::binary);
    OutputHeader ohdr{0x56323251u, num_nodes, (uint32_t)(median_ms * 1000.0)};
    fout.write(reinterpret_cast<const char*>(&ohdr), sizeof(ohdr));
    float vals[4] = {(float)ss, (float)sy, (float)yy, step};
    fout.write(reinterpret_cast<const char*>(vals), sizeof(vals));
    fout.close();
    printf("RESULT,stepsize,%u,%.3f\n", num_nodes, median_ms);
    return 0;
}
