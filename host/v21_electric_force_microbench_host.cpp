// SPDX-License-Identifier: Apache-2.0
//
// V21a electric_force microbench — host driver.
//
// Stage 1 of the broader V21 arc to port DREAMPlace's density-backward
// (`electric_force`) onto the TT chip. See docs/V21_DREAMPLACE_ON_CHIP_RESEARCH.md
// + plan at /home/ayadav/.claude/plans/draft-the-plan-for-transient-abelson.md.
//
// This binary is the TT-side half of a two-process microbench. The Python
// orchestrator (`integration/v21_ef_microbench.py`) synthesizes inputs, dumps
// them to a flat binary file, invokes this binary, then runs DREAMPlace's
// production `electric_potential_cpp.electric_force(...)` on the same inputs
// and compares the two outputs.
//
// CLI:
//   v21_ef_microbench --inputs <path.bin> --output <path.bin>
//
// Input binary format (little-endian, packed):
//   header   : { u32 magic=0x56323121, u32 M, u32 N, u32 num_nodes,
//                f32 xl, f32 yl, f32 bsx, f32 bsy }                    (32 B)
//   field_x  : M * N  fp32       (row-major, [M][N])
//   field_y  : M * N  fp32
//   pos      : 2*num_nodes fp32  (x slice [0..nn), y slice [nn..2*nn))
//   ox       : num_nodes fp32
//   oy       : num_nodes fp32
//   nsx      : num_nodes fp32  (node_size_x_clamped)
//   nsy      : num_nodes fp32  (node_size_y_clamped)
//   ratio    : num_nodes fp32
//
// Output binary format:
//   header   : { u32 magic=0x56323122, u32 num_nodes,
//                u32 kernel_ms_x1000_int }                              (12 B)
//   grad_x   : num_nodes fp32   (= -gx_per_cell, negation baked in)
//   grad_y   : num_nodes fp32   (= -gy_per_cell)

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
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
    uint32_t M;          // num_bins_x
    uint32_t N;          // num_bins_y
    uint32_t num_nodes;
    float xl;
    float yl;
    float bsx;
    float bsy;
};

struct OutputHeader {
    uint32_t magic;
    uint32_t num_nodes;
    uint32_t kernel_ms_x1000;  // median * 1000, int
};

int main(int argc, char** argv) {
    std::string inputs_path;
    std::string output_path;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--inputs" && i + 1 < argc)      inputs_path = argv[++i];
        else if (a == "--output" && i + 1 < argc) output_path = argv[++i];
    }
    if (inputs_path.empty() || output_path.empty()) {
        fprintf(stderr, "Usage: %s --inputs <path.bin> --output <path.bin>\n", argv[0]);
        return 1;
    }

    // ── Load inputs from binary ──
    std::ifstream fin(inputs_path, std::ios::binary);
    if (!fin) { fprintf(stderr, "[v21ef] FATAL: cannot open inputs file %s\n", inputs_path.c_str()); return 1; }
    InputHeader hdr{};
    fin.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (hdr.magic != 0x56323121u) {
        fprintf(stderr, "[v21ef] FATAL: bad input magic 0x%08x\n", hdr.magic);
        return 1;
    }
    const uint32_t M = hdr.M;
    const uint32_t N = hdr.N;
    const uint32_t num_nodes = hdr.num_nodes;
    const float xl  = hdr.xl;
    const float yl  = hdr.yl;
    const float bsx = hdr.bsx;
    const float bsy = hdr.bsy;
    printf("[v21ef] M=%u N=%u num_nodes=%u  xl=%g yl=%g bsx=%g bsy=%g\n",
           M, N, num_nodes, xl, yl, bsx, bsy);

    auto read_floats = [&](size_t n) {
        std::vector<float> v(n);
        fin.read(reinterpret_cast<char*>(v.data()), (std::streamsize)(n * sizeof(float)));
        return v;
    };
    std::vector<float> field_x = read_floats((size_t)M * N);
    std::vector<float> field_y = read_floats((size_t)M * N);
    std::vector<float> pos     = read_floats((size_t)num_nodes * 2);
    std::vector<float> ox      = read_floats(num_nodes);
    std::vector<float> oy      = read_floats(num_nodes);
    std::vector<float> nsx     = read_floats(num_nodes);
    std::vector<float> nsy     = read_floats(num_nodes);
    std::vector<float> ratio   = read_floats(num_nodes);
    if (!fin) {
        fprintf(stderr, "[v21ef] FATAL: short read on inputs file\n");
        return 1;
    }
    fin.close();

    // ── Open mesh device ──
    auto mesh_device = MeshDevice::create_unit_mesh(0);
    auto& cq = mesh_device->mesh_command_queue();
    auto grid = mesh_device->compute_with_storage_grid_size();
    const int nc_all = (int)(grid.x * grid.y);
    printf("[v21ef] grid=%ux%u nc_all=%d\n", grid.x, grid.y, nc_all);

    std::vector<CoreCoord> all_ccs;
    for (uint32_t y = 0; y < grid.y; ++y)
        for (uint32_t x = 0; x < grid.x; ++x)
            all_ccs.push_back(CoreCoord{x, y});
    std::set<CoreRange> crs_set;
    for (auto cc : all_ccs) crs_set.insert(CoreRange{cc, cc});
    CoreRangeSet all_crs(crs_set);

    // ── Page sizing ──
    // Field maps: page = one row = N * 4 bytes. Multiple of 64 for N >= 16
    // (tested grids 32, 512, 1024, 2048 all satisfy). Read aligned slices
    // within a single row.
    if ((N * 4u) % 64u != 0u) {
        fprintf(stderr, "[v21ef] FATAL: row stride %u B not multiple of 64\n", N * 4u);
        return 1;
    }
    const uint32_t field_pg_bytes = N * 4u;
    const uint32_t field_total_pages = M;  // one page per row
    std::vector<float> field_x_padded = field_x;  // already row-major M*N
    std::vector<float> field_y_padded = field_y;
    field_x_padded.resize((size_t)field_total_pages * (field_pg_bytes / 4u), 0.0f);
    field_y_padded.resize((size_t)field_total_pages * (field_pg_bytes / 4u), 0.0f);

    // Pos: 4 KB pages = 1024 fp32/page. Pad to multiple-of-page.
    constexpr uint32_t POS_PG_BYTES = 4096u;
    const uint32_t pos_floats_per_pg = POS_PG_BYTES / 4u;
    const uint32_t pos_total = (uint32_t)(num_nodes * 2u);
    const uint32_t pos_pages = (pos_total + pos_floats_per_pg - 1u) / pos_floats_per_pg;
    pos.resize((size_t)pos_pages * pos_floats_per_pg, 0.0f);

    // Per-core partition: each core handles cpc cells. Pad num_nodes so
    // cpc = ceil(num_nodes/nc_all) rounded up to multiple of 8 (so BRISC's
    // cpc/2 is a multiple of 4 — host invariant from v20 pattern).
    const uint32_t PAD_GRAN = (uint32_t)nc_all * 8u;
    const uint32_t num_nodes_padded = ((num_nodes + PAD_GRAN - 1u) / PAD_GRAN) * PAD_GRAN;
    const uint32_t cpc = num_nodes_padded / (uint32_t)nc_all;
    if ((cpc & 7u) != 0u) {
        fprintf(stderr, "[v21ef] FATAL: cpc=%u not multiple of 8\n", cpc);
        return 1;
    }
    const uint32_t brisc_end  = cpc / 2u;
    const uint32_t ncrisc_end = cpc;
    if ((brisc_end & 3u) != 0u) {
        fprintf(stderr, "[v21ef] FATAL: brisc_end=%u not multiple of 4\n", brisc_end);
        return 1;
    }

    // Constants: per-core page of cpc cells × 32 B. Pack (ox, oy, nsx, nsy, ratio, _, _, _).
    const uint32_t const_pg_bytes = cpc * 32u;
    std::vector<uint32_t> const_pages((size_t)nc_all * (const_pg_bytes / 4u), 0u);
    for (uint32_t g = 0; g < num_nodes; ++g) {
        const uint32_t core = g / cpc;
        const uint32_t intra = g % cpc;
        const size_t base = (size_t)core * (const_pg_bytes / 4u) + intra * 8u;
        std::memcpy(&const_pages[base + 0], &ox[g],    4);
        std::memcpy(&const_pages[base + 1], &oy[g],    4);
        std::memcpy(&const_pages[base + 2], &nsx[g],   4);
        std::memcpy(&const_pages[base + 3], &nsy[g],   4);
        std::memcpy(&const_pages[base + 4], &ratio[g], 4);
        // remaining 3 dwords are zero padding (already from vector init)
    }
    // For padded extras [num_nodes, num_nodes_padded), constants are all-zero.
    // ratio=0 → kernel's bin loop may execute but final multiply gives 0.
    // The host comparison only looks at indices [0, num_nodes).

    // Output: per-core page of cpc × 8 B (gx, gy interleaved).
    const uint32_t grad_pg_bytes = cpc * 8u;
    // grad_pg_bytes may exceed 96 KB for bb3-sized runs (cpc≈18500 → 148 KB), but
    // the kernel does the actual writes in ≤ 512 B chunks, so no hazard.
    std::vector<float> grad_pages((size_t)nc_all * (grad_pg_bytes / 4u), 0.0f);

    printf("[v21ef] cpc=%u (brisc=[0,%u) ncrisc=[%u,%u))  pos_pages=%u  field_pg=%u B\n",
           cpc, brisc_end, brisc_end, ncrisc_end, pos_pages, field_pg_bytes);

    auto make_buf = [&](uint64_t total_bytes, uint32_t pgsz) {
        DeviceLocalBufferConfig cfg{.page_size = pgsz, .buffer_type = BufferType::DRAM};
        ReplicatedBufferConfig rcfg{.size = total_bytes};
        return MeshBuffer::create(rcfg, cfg, mesh_device.get());
    };
    auto field_x_buf = make_buf((uint64_t)field_total_pages * field_pg_bytes, field_pg_bytes);
    auto field_y_buf = make_buf((uint64_t)field_total_pages * field_pg_bytes, field_pg_bytes);
    auto pos_buf     = make_buf((uint64_t)pos_pages         * POS_PG_BYTES,    POS_PG_BYTES);
    auto const_buf   = make_buf((uint64_t)nc_all            * const_pg_bytes,  const_pg_bytes);
    auto grad_buf    = make_buf((uint64_t)nc_all            * grad_pg_bytes,   grad_pg_bytes);

    printf("[v21ef] DRAM: field_x=field_y=%u pages × %u B = %.1f MB each\n",
           field_total_pages, field_pg_bytes, (double)field_total_pages * field_pg_bytes / 1048576.0);
    printf("[v21ef] DRAM: pos=%u × %u B = %.1f MB\n",
           pos_pages, POS_PG_BYTES, (double)pos_pages * POS_PG_BYTES / 1048576.0);
    printf("[v21ef] DRAM: const=%d × %u B (1 page/core)\n", nc_all, const_pg_bytes);
    printf("[v21ef] DRAM: grad =%d × %u B (1 page/core)\n", nc_all, grad_pg_bytes);
    fflush(stdout);

    // ── H2D upload ──
    auto t_h2d = hrclock::now();
    EnqueueWriteMeshBuffer(cq, field_x_buf, field_x_padded, false);
    EnqueueWriteMeshBuffer(cq, field_y_buf, field_y_padded, false);
    EnqueueWriteMeshBuffer(cq, pos_buf,     pos,            false);
    EnqueueWriteMeshBuffer(cq, const_buf,   const_pages,    false);
    Finish(cq);
    double h2d_ms = ms_since(t_h2d);
    printf("[v21ef] h2d = %.2f ms\n", h2d_ms);

    // ── Build program ──
    constexpr uint32_t CB_SIZE = 32u * 1024u;
    using tt::CBIndex;
    Program prog;
    CircularBufferConfig cb24_cfg(CB_SIZE, {{CBIndex::c_24, DataFormat::Float32}});
    cb24_cfg.set_page_size(CBIndex::c_24, CB_SIZE);
    CreateCircularBuffer(prog, all_crs, cb24_cfg);
    CircularBufferConfig cb25_cfg(CB_SIZE, {{CBIndex::c_25, DataFormat::Float32}});
    cb25_cfg.set_page_size(CBIndex::c_25, CB_SIZE);
    CreateCircularBuffer(prog, all_crs, cb25_cfg);

    auto br_k = CreateKernel(
        prog,
        std::string(DENSITY_KERNEL_DIR) + "v21_ef_brisc.cpp",
        all_crs,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                            .noc       = NOC::RISCV_0_default});
    auto nc_k = CreateKernel(
        prog,
        std::string(DENSITY_KERNEL_DIR) + "v21_ef_ncrisc.cpp",
        all_crs,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                            .noc       = NOC::RISCV_1_default});

    const uint32_t field_x_base = (uint32_t)field_x_buf->address();
    const uint32_t field_y_base = (uint32_t)field_y_buf->address();
    const uint32_t pos_base     = (uint32_t)pos_buf->address();
    const uint32_t const_base   = (uint32_t)const_buf->address();
    const uint32_t grad_base    = (uint32_t)grad_buf->address();

    union { uint32_t u; float f; } cvt;
    auto f2u = [&](float f) { cvt.f = f; return cvt.u; };
    const float inv_bsx = 1.0f / bsx;
    const float inv_bsy = 1.0f / bsy;

    for (int c = 0; c < nc_all; ++c) {
        const uint32_t cells_start_global = (uint32_t)c * cpc;
        std::vector<uint32_t> common_tail = {
            cells_start_global,
            num_nodes,
            M, N,
            field_x_base, field_y_base, field_pg_bytes,
            pos_base,
            const_base, const_pg_bytes,
            grad_base,  grad_pg_bytes,
            f2u(xl), f2u(yl), f2u(bsx), f2u(bsy), f2u(inv_bsx), f2u(inv_bsy),
        };
        std::vector<uint32_t> br_args = {(uint32_t)c, 0u,        brisc_end};
        std::vector<uint32_t> nc_args = {(uint32_t)c, brisc_end, ncrisc_end};
        br_args.insert(br_args.end(), common_tail.begin(), common_tail.end());
        nc_args.insert(nc_args.end(), common_tail.begin(), common_tail.end());
        SetRuntimeArgs(prog, br_k, all_ccs[c], br_args);
        SetRuntimeArgs(prog, nc_k, all_ccs[c], nc_args);
    }
    MeshWorkload wl;
    auto device_range = MeshCoordinateRange{mesh_device->shape()};
    wl.add_program(device_range, std::move(prog));

    // ── JIT warmup, then 5 timed runs ──
    printf("[v21ef] JIT warmup...\n"); fflush(stdout);
    auto t_jit = hrclock::now();
    EnqueueMeshWorkload(cq, wl, false);
    Finish(cq);
    double warmup_ms = ms_since(t_jit);
    printf("[v21ef] warmup = %.2f ms\n", warmup_ms);

    const int N_RUNS = 5;
    std::vector<double> run_ms;
    for (int r = 0; r < N_RUNS; ++r) {
        auto t_run = hrclock::now();
        EnqueueMeshWorkload(cq, wl, false);
        Finish(cq);
        run_ms.push_back(ms_since(t_run));
    }
    std::sort(run_ms.begin(), run_ms.end());
    double median_ms = run_ms[N_RUNS / 2];
    printf("[v21ef] runs (N=%d): ", N_RUNS);
    for (double m : run_ms) printf("%.2f ", m);
    printf("→ median %.2f ms\n", median_ms);

    // ── D2H readback ──
    auto t_d2h = hrclock::now();
    EnqueueReadMeshBuffer(cq, grad_pages, grad_buf, true);
    double d2h_ms = ms_since(t_d2h);
    printf("[v21ef] d2h = %.2f ms\n", d2h_ms);

    // ── De-interleave per-core (gx, gy) into flat grad_x[], grad_y[] arrays. ──
    std::vector<float> grad_x_flat(num_nodes, 0.0f);
    std::vector<float> grad_y_flat(num_nodes, 0.0f);
    const uint32_t floats_per_grad_pg = grad_pg_bytes / 4u;
    for (uint32_t g = 0; g < num_nodes; ++g) {
        const uint32_t core = g / cpc;
        const uint32_t intra = g % cpc;
        const size_t base = (size_t)core * floats_per_grad_pg + intra * 2u;
        grad_x_flat[g] = grad_pages[base + 0];
        grad_y_flat[g] = grad_pages[base + 1];
    }

    // ── Save output binary ──
    std::ofstream fout(output_path, std::ios::binary);
    if (!fout) { fprintf(stderr, "[v21ef] FATAL: cannot open output %s\n", output_path.c_str()); return 1; }
    OutputHeader ohdr{};
    ohdr.magic = 0x56323122u;
    ohdr.num_nodes = num_nodes;
    ohdr.kernel_ms_x1000 = (uint32_t)(median_ms * 1000.0);
    fout.write(reinterpret_cast<const char*>(&ohdr), sizeof(ohdr));
    fout.write(reinterpret_cast<const char*>(grad_x_flat.data()), (std::streamsize)(num_nodes * sizeof(float)));
    fout.write(reinterpret_cast<const char*>(grad_y_flat.data()), (std::streamsize)(num_nodes * sizeof(float)));
    fout.close();

    printf("[v21ef] wrote %s (num_nodes=%u, kernel_ms=%.3f)\n",
           output_path.c_str(), num_nodes, median_ms);
    printf("RESULT,%u,%u,%u,%.3f,%.3f,%.3f\n",
           M, N, num_nodes, median_ms, h2d_ms, d2h_ms);
    return 0;
}
