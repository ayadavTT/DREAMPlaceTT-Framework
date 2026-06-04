// SPDX-License-Identifier: Apache-2.0
//
// V22 K2/K3/K4 — generic element-wise op microbench (TT-side host driver).
//
// Shared driver for the V22 optimizer-side element-wise kernels (handoff §2):
//   --op clamp    : out = min(max(pos, lo), hi)                 (K2 move_boundary)
//   --op combine  : out = wl + density_weight*dg               (K3 grad combine)
//   --op precond  : out = grad / max(pin_w + adw*area, 1)      (K4 precondition)
//
// All are pure element-wise over 2*num_nodes fp32 with n_arrays streamed inputs
// (mapped to c_0/c_1/c_2 in order) + n_scalars broadcast scalars (c_3/c_4).
// The Python orchestrator (integration/v22_elt_microbench.py) synthesizes inputs
// + runs the reference and diffs.
//
// Input binary (LE, packed):
//   header : { u32 magic=0x56323210, u32 num_nodes, u32 n_arrays, u32 n_scalars,
//              f32 scalar0, f32 scalar1 }                                 (24 B)
//   arrays : n_arrays * (2*num_nodes fp32), in CB order c_0, c_1, c_2
// Output binary:
//   header : { u32 magic=0x56323211, u32 num_nodes, u32 kernel_ms_x1000 } (12 B)
//   out    : 2*num_nodes fp32

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
static double ms_since(T t0) { return std::chrono::duration<double, std::milli>(hrclock::now() - t0).count(); }

#ifndef DENSITY_KERNEL_DIR
#define DENSITY_KERNEL_DIR "kernels/"
#endif

struct InputHeader { uint32_t magic, num_nodes, n_arrays, n_scalars; float scalar0, scalar1; };
struct OutputHeader { uint32_t magic, num_nodes, kernel_ms_x1000; };

int main(int argc, char** argv) {
    std::string inputs_path, output_path, op;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--inputs" && i + 1 < argc)      inputs_path = argv[++i];
        else if (a == "--output" && i + 1 < argc) output_path = argv[++i];
        else if (a == "--op" && i + 1 < argc)     op = argv[++i];
    }
    if (inputs_path.empty() || output_path.empty() || op.empty()) {
        fprintf(stderr, "Usage: %s --op clamp|combine|precond --inputs <p> --output <p>\n", argv[0]);
        return 1;
    }
    std::string compute_file;
    if      (op == "clamp")   compute_file = "v22_clamp_compute.cpp";
    else if (op == "combine") compute_file = "v22_gradcombine_compute.cpp";
    else if (op == "precond") compute_file = "v22_precond_compute.cpp";
    else { fprintf(stderr, "[v22elt] FATAL: unknown --op %s\n", op.c_str()); return 1; }

    // ── Load inputs ──
    std::ifstream fin(inputs_path, std::ios::binary);
    if (!fin) { fprintf(stderr, "[v22elt] FATAL: cannot open %s\n", inputs_path.c_str()); return 1; }
    InputHeader hdr{};
    fin.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (hdr.magic != 0x56323210u) { fprintf(stderr, "[v22elt] FATAL: bad magic 0x%08x\n", hdr.magic); return 1; }
    const uint32_t num_nodes = hdr.num_nodes;
    const uint32_t n_arrays  = hdr.n_arrays;
    const uint32_t n_scalars = hdr.n_scalars;
    const uint32_t n_elems   = num_nodes * 2u;
    printf("[v22elt] op=%s num_nodes=%u n_elems=%u n_arrays=%u n_scalars=%u s0=%g s1=%g\n",
           op.c_str(), num_nodes, n_elems, n_arrays, n_scalars, hdr.scalar0, hdr.scalar1);
    if (n_arrays < 1 || n_arrays > 3) { fprintf(stderr, "[v22elt] FATAL: n_arrays=%u out of [1,3]\n", n_arrays); return 1; }

    auto read_floats = [&](size_t n) {
        std::vector<float> v(n);
        fin.read(reinterpret_cast<char*>(v.data()), (std::streamsize)(n * sizeof(float)));
        return v;
    };
    std::vector<std::vector<float>> arrays;
    for (uint32_t a = 0; a < n_arrays; ++a) arrays.push_back(read_floats(n_elems));
    if (!fin) { fprintf(stderr, "[v22elt] FATAL: short read\n"); return 1; }
    fin.close();

    // ── Device ──
    auto mesh_device = MeshDevice::create_unit_mesh(0);
    auto& cq = mesh_device->mesh_command_queue();
    auto grid = mesh_device->compute_with_storage_grid_size();
    const int nc_all = (int)(grid.x * grid.y);
    printf("[v22elt] grid=%ux%u nc_all=%d\n", grid.x, grid.y, nc_all);
    std::vector<CoreCoord> all_ccs;
    for (uint32_t y = 0; y < grid.y; ++y)
        for (uint32_t x = 0; x < grid.x; ++x) all_ccs.push_back(CoreCoord{x, y});
    std::set<CoreRange> crs_set;
    for (auto cc : all_ccs) crs_set.insert(CoreRange{cc, cc});
    CoreRangeSet all_crs(crs_set);

    using tt::CBIndex;
    constexpr uint32_t TILE_ELEMS = 32u * 32u;
    constexpr uint32_t TILE_BYTES = TILE_ELEMS * 4u;

    const uint32_t n_tiles_total = (n_elems + TILE_ELEMS - 1u) / TILE_ELEMS;
    const uint32_t padded_elems  = n_tiles_total * TILE_ELEMS;
    for (auto& v : arrays) v.resize(padded_elems, 0.0f);

    std::vector<float> s0_tile(TILE_ELEMS, hdr.scalar0);
    std::vector<float> s1_tile(TILE_ELEMS, hdr.scalar1);

    const uint32_t tiles_per_core = (n_tiles_total + (uint32_t)nc_all - 1u) / (uint32_t)nc_all;
    printf("[v22elt] n_tiles_total=%u tiles_per_core(max)=%u\n", n_tiles_total, tiles_per_core);

    auto make_buf = [&](uint64_t total_bytes, uint32_t pgsz) {
        DeviceLocalBufferConfig cfg{.page_size = pgsz, .buffer_type = BufferType::DRAM};
        ReplicatedBufferConfig rcfg{.size = total_bytes};
        return MeshBuffer::create(rcfg, cfg, mesh_device.get());
    };
    const uint64_t io_bytes = (uint64_t)n_tiles_total * TILE_BYTES;
    std::vector<std::shared_ptr<MeshBuffer>> in_bufs;
    for (uint32_t a = 0; a < n_arrays; ++a) in_bufs.push_back(make_buf(io_bytes, TILE_BYTES));
    auto s0_buf  = make_buf(TILE_BYTES, TILE_BYTES);
    auto s1_buf  = make_buf(TILE_BYTES, TILE_BYTES);
    auto out_buf = make_buf(io_bytes, TILE_BYTES);

    // ── H2D ──
    auto t_h2d = hrclock::now();
    for (uint32_t a = 0; a < n_arrays; ++a) EnqueueWriteMeshBuffer(cq, in_bufs[a], arrays[a], false);
    if (n_scalars >= 1) EnqueueWriteMeshBuffer(cq, s0_buf, s0_tile, false);
    if (n_scalars >= 2) EnqueueWriteMeshBuffer(cq, s1_buf, s1_tile, false);
    Finish(cq);
    printf("[v22elt] h2d = %.2f ms\n", ms_since(t_h2d));

    const uint32_t a0 = (uint32_t)in_bufs[0]->address();
    const uint32_t a1 = n_arrays >= 2 ? (uint32_t)in_bufs[1]->address() : a0;
    const uint32_t a2 = n_arrays >= 3 ? (uint32_t)in_bufs[2]->address() : a0;
    const uint32_t s0a = (uint32_t)s0_buf->address();
    const uint32_t s1a = (uint32_t)s1_buf->address();
    const uint32_t outa = (uint32_t)out_buf->address();

    // ── Program ──
    Program prog = CreateProgram();
    auto make_cb = [&](uint32_t idx, uint32_t n_slots) {
        CircularBufferConfig cfg(n_slots * TILE_BYTES, {{(CBIndex)idx, DataFormat::Float32}});
        cfg.set_page_size((CBIndex)idx, TILE_BYTES);
        CreateCircularBuffer(prog, all_crs, cfg);
    };
    make_cb((uint32_t)CBIndex::c_0, 2);
    make_cb((uint32_t)CBIndex::c_1, 2);
    make_cb((uint32_t)CBIndex::c_2, 2);
    make_cb((uint32_t)CBIndex::c_3, 1);
    make_cb((uint32_t)CBIndex::c_4, 1);
    make_cb((uint32_t)CBIndex::c_16, 2);

    auto reader_k = CreateKernel(prog, std::string(DENSITY_KERNEL_DIR) + "v22_elt_reader.cpp", all_crs,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    auto writer_k = CreateKernel(prog, std::string(DENSITY_KERNEL_DIR) + "v22_elt_writer.cpp", all_crs,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    std::vector<UnpackToDestMode> unpack_modes(64, UnpackToDestMode::Default);
    for (uint32_t i = 0; i <= 4; ++i) unpack_modes[i] = UnpackToDestMode::UnpackToDestFp32;
    auto compute_k = CreateKernel(prog, std::string(DENSITY_KERNEL_DIR) + compute_file, all_crs,
        ComputeConfig{.fp32_dest_acc_en = true, .unpack_to_dest_mode = unpack_modes, .math_approx_mode = false});

    for (int c = 0; c < nc_all; ++c) {
        const uint32_t tile_start = std::min((uint32_t)c * tiles_per_core, n_tiles_total);
        const uint32_t tile_end   = std::min(tile_start + tiles_per_core, n_tiles_total);
        const uint32_t nt = tile_end - tile_start;
        SetRuntimeArgs(prog, reader_k, all_ccs[c],
            {nt, tile_start, n_arrays, n_scalars, a0, a1, a2, s0a, s1a});
        SetRuntimeArgs(prog, writer_k, all_ccs[c], {nt, tile_start, outa});
        SetRuntimeArgs(prog, compute_k, all_ccs[c], {nt});
    }

    MeshWorkload wl;
    auto device_range = MeshCoordinateRange{mesh_device->shape()};
    wl.add_program(device_range, std::move(prog));
    auto run_once = [&]() { EnqueueMeshWorkload(cq, wl, false); Finish(cq); };

    printf("[v22elt] JIT warmup...\n"); fflush(stdout);
    auto t_jit = hrclock::now();
    run_once();
    printf("[v22elt] warmup = %.2f ms\n", ms_since(t_jit));

    const int N_RUNS = 5;
    std::vector<double> run_ms;
    for (int r = 0; r < N_RUNS; ++r) { auto t = hrclock::now(); run_once(); run_ms.push_back(ms_since(t)); }
    std::sort(run_ms.begin(), run_ms.end());
    double median_ms = run_ms[N_RUNS / 2];
    printf("[v22elt] runs (N=%d): ", N_RUNS);
    for (double m : run_ms) printf("%.3f ", m);
    printf("→ median %.3f ms\n", median_ms);

    std::vector<float> out(padded_elems, 0.0f);
    auto t_d2h = hrclock::now();
    EnqueueReadMeshBuffer(cq, out, out_buf, true);
    printf("[v22elt] d2h = %.2f ms\n", ms_since(t_d2h));

    std::ofstream fout(output_path, std::ios::binary);
    if (!fout) { fprintf(stderr, "[v22elt] FATAL: cannot open output %s\n", output_path.c_str()); return 1; }
    OutputHeader ohdr{0x56323211u, num_nodes, (uint32_t)(median_ms * 1000.0)};
    fout.write(reinterpret_cast<const char*>(&ohdr), sizeof(ohdr));
    fout.write(reinterpret_cast<const char*>(out.data()), (std::streamsize)(n_elems * sizeof(float)));
    fout.close();
    printf("[v22elt] wrote %s (op=%s num_nodes=%u kernel_ms=%.3f)\n", output_path.c_str(), op.c_str(), num_nodes, median_ms);
    printf("RESULT,%s,%u,%.3f\n", op.c_str(), num_nodes, median_ms);
    return 0;
}
