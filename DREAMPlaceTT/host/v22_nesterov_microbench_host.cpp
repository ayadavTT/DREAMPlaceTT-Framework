// SPDX-License-Identifier: Apache-2.0
//
// V22 K1 — Nesterov optimizer update microbench (TT-side host driver).
//
// Stage 1 of the V22 chip-resident-pos arc (see docs/V22_CHIP_RESIDENT_POS_HANDOFF.md).
// Computes, on the SFPU, the element-wise Nesterov update over 2*num_nodes fp32:
//   u_kp1 = v_k  - alpha * g_k
//   v_kp1 = u_kp1 + coef * (u_kp1 - u_k)
//
// Two-process microbench: the Python orchestrator (integration/v22_nesterov_microbench.py)
// synthesizes inputs, dumps them to a flat binary, invokes this binary, then runs
// the torch Nesterov reference on the same inputs and diffs.
//
// CLI:  v22_nesterov_microbench --inputs <path.bin> --output <path.bin>
//
// Input binary (little-endian, packed):
//   header : { u32 magic=0x56323201, u32 num_nodes, f32 alpha, f32 coef }   (16 B)
//   v      : 2*num_nodes fp32   (v_k)
//   g      : 2*num_nodes fp32   (g_k)
//   u      : 2*num_nodes fp32   (u_k = previous u)
//
// Output binary:
//   header : { u32 magic=0x56323202, u32 num_nodes, u32 kernel_ms_x1000 }    (12 B)
//   u_out  : 2*num_nodes fp32   (u_kp1)
//   v_out  : 2*num_nodes fp32   (v_kp1)

#include <algorithm>
#include <chrono>
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

template <class T>
static double ms_since(T t0) {
    return std::chrono::duration<double, std::milli>(hrclock::now() - t0).count();
}

#ifndef DENSITY_KERNEL_DIR
#define DENSITY_KERNEL_DIR "kernels/"
#endif

struct InputHeader {
    uint32_t magic;
    uint32_t num_nodes;
    float    alpha;
    float    coef;
};
struct OutputHeader {
    uint32_t magic;
    uint32_t num_nodes;
    uint32_t kernel_ms_x1000;
};

int main(int argc, char** argv) {
    std::string inputs_path, output_path;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--inputs" && i + 1 < argc)      inputs_path = argv[++i];
        else if (a == "--output" && i + 1 < argc) output_path = argv[++i];
    }
    if (inputs_path.empty() || output_path.empty()) {
        fprintf(stderr, "Usage: %s --inputs <path.bin> --output <path.bin>\n", argv[0]);
        return 1;
    }

    // ── Load inputs ──
    std::ifstream fin(inputs_path, std::ios::binary);
    if (!fin) { fprintf(stderr, "[v22nag] FATAL: cannot open %s\n", inputs_path.c_str()); return 1; }
    InputHeader hdr{};
    fin.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (hdr.magic != 0x56323201u) {
        fprintf(stderr, "[v22nag] FATAL: bad input magic 0x%08x\n", hdr.magic);
        return 1;
    }
    const uint32_t num_nodes = hdr.num_nodes;
    const float alpha = hdr.alpha;
    const float coef  = hdr.coef;
    const uint32_t n_elems = num_nodes * 2u;   // pos layout is [x..|y..], 2*nn fp32
    printf("[v22nag] num_nodes=%u n_elems=%u alpha=%g coef=%g\n", num_nodes, n_elems, alpha, coef);

    auto read_floats = [&](size_t n) {
        std::vector<float> v(n);
        fin.read(reinterpret_cast<char*>(v.data()), (std::streamsize)(n * sizeof(float)));
        return v;
    };
    std::vector<float> v_in = read_floats(n_elems);
    std::vector<float> g_in = read_floats(n_elems);
    std::vector<float> u_in = read_floats(n_elems);
    if (!fin) { fprintf(stderr, "[v22nag] FATAL: short read on inputs\n"); return 1; }
    fin.close();

    // ── Open mesh device ──
    auto mesh_device = MeshDevice::create_unit_mesh(0);
    auto& cq = mesh_device->mesh_command_queue();
    auto grid = mesh_device->compute_with_storage_grid_size();
    const int nc_all = (int)(grid.x * grid.y);
    printf("[v22nag] grid=%ux%u nc_all=%d\n", grid.x, grid.y, nc_all);

    std::vector<CoreCoord> all_ccs;
    for (uint32_t y = 0; y < grid.y; ++y)
        for (uint32_t x = 0; x < grid.x; ++x)
            all_ccs.push_back(CoreCoord{x, y});
    std::set<CoreRange> crs_set;
    for (auto cc : all_ccs) crs_set.insert(CoreRange{cc, cc});
    CoreRangeSet all_crs(crs_set);

    using tt::CBIndex;
    constexpr uint32_t TILE_ELEMS = 32u * 32u;       // 1024 fp32/tile
    constexpr uint32_t TILE_BYTES = TILE_ELEMS * 4u; // 4096 B

    // ── Pad to whole tiles, distribute tiles across cores ──
    const uint32_t n_tiles_total = (n_elems + TILE_ELEMS - 1u) / TILE_ELEMS;
    const uint32_t padded_elems  = n_tiles_total * TILE_ELEMS;
    v_in.resize(padded_elems, 0.0f);
    g_in.resize(padded_elems, 0.0f);
    u_in.resize(padded_elems, 0.0f);

    // Broadcast scalar tiles: all 1024 lanes hold the scalar.
    std::vector<float> alpha_tile(TILE_ELEMS, alpha);
    std::vector<float> coef_tile(TILE_ELEMS, coef);

    // Per-core tile range (contiguous split, ceil to spread the remainder).
    const uint32_t tiles_per_core = (n_tiles_total + (uint32_t)nc_all - 1u) / (uint32_t)nc_all;
    printf("[v22nag] n_tiles_total=%u tiles_per_core(max)=%u padded_elems=%u\n",
           n_tiles_total, tiles_per_core, padded_elems);

    // ── DRAM buffers (interleaved, page = one tile) ──
    auto make_buf = [&](uint64_t total_bytes, uint32_t pgsz) {
        DeviceLocalBufferConfig cfg{.page_size = pgsz, .buffer_type = BufferType::DRAM};
        ReplicatedBufferConfig rcfg{.size = total_bytes};
        return MeshBuffer::create(rcfg, cfg, mesh_device.get());
    };
    const uint64_t io_bytes = (uint64_t)n_tiles_total * TILE_BYTES;
    auto v_buf     = make_buf(io_bytes,  TILE_BYTES);
    auto g_buf     = make_buf(io_bytes,  TILE_BYTES);
    auto u_buf     = make_buf(io_bytes,  TILE_BYTES);
    auto alpha_buf = make_buf(TILE_BYTES, TILE_BYTES);
    auto coef_buf  = make_buf(TILE_BYTES, TILE_BYTES);
    auto u_out_buf = make_buf(io_bytes,  TILE_BYTES);
    auto v_out_buf = make_buf(io_bytes,  TILE_BYTES);

    // ── H2D ──
    auto t_h2d = hrclock::now();
    EnqueueWriteMeshBuffer(cq, v_buf,     v_in,       false);
    EnqueueWriteMeshBuffer(cq, g_buf,     g_in,       false);
    EnqueueWriteMeshBuffer(cq, u_buf,     u_in,       false);
    EnqueueWriteMeshBuffer(cq, alpha_buf, alpha_tile, false);
    EnqueueWriteMeshBuffer(cq, coef_buf,  coef_tile,  false);
    Finish(cq);
    double h2d_ms = ms_since(t_h2d);
    printf("[v22nag] h2d = %.2f ms\n", h2d_ms);

    const uint32_t v_base     = (uint32_t)v_buf->address();
    const uint32_t g_base     = (uint32_t)g_buf->address();
    const uint32_t u_base     = (uint32_t)u_buf->address();
    const uint32_t alpha_base = (uint32_t)alpha_buf->address();
    const uint32_t coef_base  = (uint32_t)coef_buf->address();
    const uint32_t u_out_base = (uint32_t)u_out_buf->address();
    const uint32_t v_out_base = (uint32_t)v_out_buf->address();

    // ── Build program ──
    Program prog = CreateProgram();

    auto make_cb = [&](uint32_t idx, uint32_t n_slots) {
        CircularBufferConfig cfg(n_slots * TILE_BYTES, {{(CBIndex)idx, DataFormat::Float32}});
        cfg.set_page_size((CBIndex)idx, TILE_BYTES);
        CreateCircularBuffer(prog, all_crs, cfg);
    };
    make_cb((uint32_t)CBIndex::c_0,  2);   // cb_v
    make_cb((uint32_t)CBIndex::c_1,  2);   // cb_g
    make_cb((uint32_t)CBIndex::c_2,  2);   // cb_u
    make_cb((uint32_t)CBIndex::c_3,  1);   // cb_alpha (resident)
    make_cb((uint32_t)CBIndex::c_4,  1);   // cb_coef  (resident)
    make_cb((uint32_t)CBIndex::c_16, 2);   // cb_u_out
    make_cb((uint32_t)CBIndex::c_17, 2);   // cb_v_out

    auto reader_k = CreateKernel(
        prog, std::string(DENSITY_KERNEL_DIR) + "v22_nesterov_reader.cpp", all_crs,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                           .noc       = NOC::RISCV_0_default});
    auto writer_k = CreateKernel(
        prog, std::string(DENSITY_KERNEL_DIR) + "v22_nesterov_writer.cpp", all_crs,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                           .noc       = NOC::RISCV_1_default});

    // UnpackToDestFp32 on all fp32 input CBs (else FP32 → TF32 silent downcast;
    // see tt-sfpu/SFPU_GUIDE.md). Inputs to compute: c_0..c_4.
    std::vector<UnpackToDestMode> unpack_modes(64, UnpackToDestMode::Default);
    for (uint32_t i = 0; i <= 4; ++i)
        unpack_modes[i] = UnpackToDestMode::UnpackToDestFp32;

    auto compute_k = CreateKernel(
        prog, std::string(DENSITY_KERNEL_DIR) + "v22_nesterov_compute.cpp", all_crs,
        ComputeConfig{
            .fp32_dest_acc_en    = true,
            .unpack_to_dest_mode = unpack_modes,
            .math_approx_mode    = false,
        });

    // ── Runtime args per core ──
    for (int c = 0; c < nc_all; ++c) {
        const uint32_t tile_start = std::min((uint32_t)c * tiles_per_core, n_tiles_total);
        const uint32_t tile_end   = std::min(tile_start + tiles_per_core, n_tiles_total);
        const uint32_t n_tiles    = tile_end - tile_start;

        SetRuntimeArgs(prog, reader_k, all_ccs[c],
            {n_tiles, tile_start, v_base, g_base, u_base, alpha_base, coef_base});
        SetRuntimeArgs(prog, writer_k, all_ccs[c],
            {n_tiles, tile_start, u_out_base, v_out_base});
        SetRuntimeArgs(prog, compute_k, all_ccs[c], {n_tiles});
    }

    MeshWorkload wl;
    auto device_range = MeshCoordinateRange{mesh_device->shape()};
    wl.add_program(device_range, std::move(prog));

    auto run_once = [&]() {
        EnqueueMeshWorkload(cq, wl, false);
        Finish(cq);
    };

    // Warmup (JIT)
    printf("[v22nag] JIT warmup...\n"); fflush(stdout);
    auto t_jit = hrclock::now();
    run_once();
    printf("[v22nag] warmup = %.2f ms\n", ms_since(t_jit));

    const int N_RUNS = 5;
    std::vector<double> run_ms;
    for (int r = 0; r < N_RUNS; ++r) {
        auto t_run = hrclock::now();
        run_once();
        run_ms.push_back(ms_since(t_run));
    }
    std::sort(run_ms.begin(), run_ms.end());
    double median_ms = run_ms[N_RUNS / 2];
    printf("[v22nag] runs (N=%d): ", N_RUNS);
    for (double m : run_ms) printf("%.3f ", m);
    printf("→ median %.3f ms\n", median_ms);

    // ── D2H ──
    std::vector<float> u_out(padded_elems, 0.0f), v_out(padded_elems, 0.0f);
    auto t_d2h = hrclock::now();
    EnqueueReadMeshBuffer(cq, u_out, u_out_buf, true);
    EnqueueReadMeshBuffer(cq, v_out, v_out_buf, true);
    double d2h_ms = ms_since(t_d2h);
    printf("[v22nag] d2h = %.2f ms\n", d2h_ms);

    // ── Save output (trim padding back to n_elems) ──
    std::ofstream fout(output_path, std::ios::binary);
    if (!fout) { fprintf(stderr, "[v22nag] FATAL: cannot open output %s\n", output_path.c_str()); return 1; }
    OutputHeader ohdr{0x56323202u, num_nodes, (uint32_t)(median_ms * 1000.0)};
    fout.write(reinterpret_cast<const char*>(&ohdr), sizeof(ohdr));
    fout.write(reinterpret_cast<const char*>(u_out.data()), (std::streamsize)(n_elems * sizeof(float)));
    fout.write(reinterpret_cast<const char*>(v_out.data()), (std::streamsize)(n_elems * sizeof(float)));
    fout.close();

    printf("[v22nag] wrote %s (num_nodes=%u kernel_ms=%.3f)\n", output_path.c_str(), num_nodes, median_ms);
    printf("RESULT,%u,%.3f,%.3f,%.3f\n", num_nodes, median_ms, h2d_ms, d2h_ms);
    return 0;
}
