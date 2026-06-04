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
    bool use_sfpu = false;
    bool use_v5 = false;
    bool use_v6 = false;  // v6 = v5 + bf16 cb_fxy
    bool use_v23 = false; // v23 = full-tile 1024-cell batches (vs V21's half-empty 512)
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--inputs" && i + 1 < argc)      inputs_path = argv[++i];
        else if (a == "--output" && i + 1 < argc) output_path = argv[++i];
        else if (a == "--mode"   && i + 1 < argc) use_sfpu = (std::string(argv[++i]) == "sfpu");
        else if (a == "--sfpu")                   use_sfpu = true;
        else if (a == "--v5")                     { use_sfpu = true; use_v5 = true; }
        else if (a == "--v6")                     { use_sfpu = true; use_v5 = true; use_v6 = true; }
        else if (a == "--v23")                    { use_sfpu = true; use_v23 = true; }
    }
    // V23: cells per SFPU batch (fills the whole 1024-lane tile; V21 uses 512 → half-empty).
    const uint32_t vbatch = use_v23 ? 1024u : 512u;
    // Experimental: bf16 field tiles (half the copy_tile bytes on TRISC). Opt-in.
    const bool field_bf16 = use_v23 && (getenv("V21_FIELD_BF16") != nullptr);
    if (inputs_path.empty() || output_path.empty()) {
        fprintf(stderr, "Usage: %s --inputs <path.bin> --output <path.bin> [--mode sfpu|scalar|--v5]\n", argv[0]);
        return 1;
    }
    printf("[v21ef] mode = %s%s%s\n", use_sfpu ? "SFPU (V21_USE_SFPU=1)" : "scalar",
           use_v5 ? " [v5 — face-merge cb_fxy]" : "",
           use_v6 ? " [v6 — bf16 cb_fxy]" : "");

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

    // Per-core partition: each core handles cpc cells.
    const uint32_t pad_unit = use_sfpu ? vbatch : 8u;
    const uint32_t PAD_GRAN = (uint32_t)nc_all * pad_unit;
    const uint32_t num_nodes_padded = ((num_nodes + PAD_GRAN - 1u) / PAD_GRAN) * PAD_GRAN;
    const uint32_t cpc = num_nodes_padded / (uint32_t)nc_all;
    if ((cpc % pad_unit) != 0u) {
        fprintf(stderr, "[v21ef] FATAL: cpc=%u not multiple of %u\n", cpc, pad_unit);
        return 1;
    }
    const uint32_t brisc_end  = use_sfpu ? cpc      : (cpc / 2u);
    const uint32_t ncrisc_end = cpc;
    if (!use_sfpu && (brisc_end & 3u) != 0u) {
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

    using tt::CBIndex;

    const uint32_t field_x_base = (uint32_t)field_x_buf->address();
    const uint32_t field_y_base = (uint32_t)field_y_buf->address();
    const uint32_t pos_base     = (uint32_t)pos_buf->address();
    const uint32_t const_base   = (uint32_t)const_buf->address();
    const uint32_t grad_base    = (uint32_t)grad_buf->address();

    union { uint32_t u; float f; } cvt;
    auto f2u = [&](float f) { cvt.f = f; return cvt.u; };
    const float inv_bsx = 1.0f / bsx;
    const float inv_bsy = 1.0f / bsy;

    // v23 adaptive bin-span: replicate the kernel's bin-range math on host to find
    // the global max k_count/h_count, so the kk/hh loops only cover bins actually
    // touched (V21 always does 8×8). Cuts total work on all three units.
    uint32_t max_k = 8u, max_h = 8u;
    if (use_v23) {
        uint32_t mk = 1u, mh = 1u;
        for (uint32_t g = 0; g < num_nodes; ++g) {
            const float node_x = pos[g] + ox[g];
            const float node_y = pos[(size_t)num_nodes + g] + oy[g];
            int bxl = (int)((node_x - xl) * inv_bsx);
            int bxh = (int)((node_x + nsx[g] - xl) * inv_bsx) + 1;
            int byl = (int)((node_y - yl) * inv_bsy);
            int byh = (int)((node_y + nsy[g] - yl) * inv_bsy) + 1;
            if (bxl < 0) bxl = 0; if (bxh > (int)M) bxh = (int)M;
            if (byl < 0) byl = 0; if (byh > (int)N) byh = (int)N;
            int kc = bxh - bxl; if (kc < 0) kc = 0; if (kc > 8) kc = 8;
            int hc = byh - byl; if (hc < 0) hc = 0; if (hc > 8) hc = 8;
            if ((uint32_t)kc > mk) mk = (uint32_t)kc;
            if ((uint32_t)hc > mh) mh = (uint32_t)hc;
        }
        max_k = mk; max_h = mh;
        printf("[v21ef] v23 adaptive bounds: max_k=%u max_h=%u  (work %u/64 = %.0f%% of V21)\n",
               max_k, max_h, max_k * max_h, 100.0 * (max_k * max_h) / 64.0);
    }

    // Parallel-prep sem IDs (captured by build_program on first call, reused).
    uint32_t sem_brisc_done_id  = 0u;
    uint32_t sem_ncrisc_done_id = 0u;
    // v5 per-tile cb_fxy dual-producer sync sem IDs (function-scope so update_args can see them).
    uint32_t sem_brisc_fxy_id   = 0u;
    uint32_t sem_ncrisc_fxy_id  = 0u;

    // Per-chunk Program builder: creates a fresh Program with all CBs, kernels,
    // and runtime args for processing cells [chunk_start, chunk_end) per core.
    auto build_program = [&](uint32_t chunk_start, uint32_t chunk_end) {
    Program prog = CreateProgram();
    {
        // v23: batch=1024 doubles per-cell staging to ~500 KB → 512 KB scratch.
        const uint32_t cb24_size = use_v23 ? (512u * 1024u)
                                  : use_sfpu ? ((use_v5 ? 384u : 256u) * 1024u) : (32u * 1024u);  // v5d needs +64KB for fy_stage_b
        const uint32_t cb25_size = use_sfpu ? (16u  * 1024u) : (32u * 1024u);
        CircularBufferConfig cb24_cfg(cb24_size, {{CBIndex::c_24, DataFormat::Float32}});
        cb24_cfg.set_page_size(CBIndex::c_24, cb24_size);
        CreateCircularBuffer(prog, all_crs, cb24_cfg);
        CircularBufferConfig cb25_cfg(cb25_size, {{CBIndex::c_25, DataFormat::Float32}});
        cb25_cfg.set_page_size(CBIndex::c_25, cb25_size);
        CreateCircularBuffer(prog, all_crs, cb25_cfg);
    }

    // SFPU-mode CBs. Each tile = 1024 fp32 cells = 4 KB.
    KernelHandle compute_k = 0;
    if (use_sfpu) {
        constexpr uint32_t TILE_BYTES = 32u * 32u * 4u;  // 4 KB fp32 tile
        auto make_cb = [&](uint32_t idx, uint32_t n_slots) {
            CircularBufferConfig cfg(n_slots * TILE_BYTES, {{idx, DataFormat::Float32}});
            cfg.set_page_size(idx, TILE_BYTES);
            CreateCircularBuffer(prog, all_crs, cfg);
        };
        auto make_cb_bf16 = [&](uint32_t idx, uint32_t n_slots) {
            constexpr uint32_t TB = 32u * 32u * 2u;  // 2 KB bf16 tile
            CircularBufferConfig cfg(n_slots * TB, {{idx, DataFormat::Float16_b}});
            cfg.set_page_size(idx, TB);
            CreateCircularBuffer(prog, all_crs, cfg);
        };
        // v23 adaptive: only max_k px / max_h py tiles per batch (V21 = 8/8).
        make_cb((uint32_t)CBIndex::c_0,  use_v23 ? max_k : 8u);   // cb_px (one per k)
        make_cb((uint32_t)CBIndex::c_1,  use_v23 ? max_h : 8u);   // cb_py (one per h)
        if (use_v5) {
            // v5 face-merge: single cb_fxy CB (replaces cb_fx + cb_fy), single cb_gxy_out.
            // v6 reuses fp32 cb_fxy but BRISC truncates values to bf16 precision —
            // this is a precision-impact experiment (no bandwidth saving).
            make_cb((uint32_t)CBIndex::c_2,  16);  // cb_fxy fp32 (precision-truncated in v6)
            // c_3 unused in v5
            make_cb((uint32_t)CBIndex::c_4,  4);   // cb_neg_ratio
            make_cb((uint32_t)CBIndex::c_5,  4);   // cb_acc
            make_cb((uint32_t)CBIndex::c_16, 4);   // cb_gxy_out   — face-merged: lower=gx, upper=gy
        } else {
            // CB tile size is always 4 KB regardless of vbatch — only c_24 scratch grows.
            // cb_fy holds the full per-batch fy set so NCRISC pushes ahead during GX
            // (avoids ~700 µs stall): max_k*max_h for v23 (= 64 when 8×8), else 64.
            if (field_bf16) {
                make_cb_bf16((uint32_t)CBIndex::c_2, max_h);          // cb_fx bf16
                make_cb_bf16((uint32_t)CBIndex::c_3, max_k * max_h);  // cb_fy bf16
            } else {
                make_cb((uint32_t)CBIndex::c_2,  use_v23 ? max_h : 8u);          // cb_fx — streamed (1 group)
                make_cb((uint32_t)CBIndex::c_3,  use_v23 ? (max_k * max_h) : 64u); // cb_fy — full run-ahead
            }
            make_cb((uint32_t)CBIndex::c_4,  4);   // cb_neg_ratio — pipelined
            make_cb((uint32_t)CBIndex::c_5,  4);   // cb_acc       — cycled accumulator
            make_cb((uint32_t)CBIndex::c_16, 4);   // cb_gx_out    — pipelined
            make_cb((uint32_t)CBIndex::c_17, 4);   // cb_gy_out
        }
    }

    std::map<std::string, std::string> dm_defines;
    std::map<std::string, std::string> compute_defines;
    if (use_sfpu) {
        dm_defines["V21_USE_SFPU"] = "1";
        compute_defines["V21_USE_SFPU"] = "1";
    }
    if (use_v23) {
        dm_defines["V21S_BATCH_OVERRIDE"] = "1024";  // full-tile batches (BRISC/NCRISC)
        dm_defines["V21S_MAX_K"]      = std::to_string(max_k);
        dm_defines["V21S_MAX_H"]      = std::to_string(max_h);
        dm_defines["V21_PREP_FAST"]   = "1";  // boundary-only overlap (interior = bsx)
        if (field_bf16) {
            dm_defines["V21_FIELD_BF16"]      = "1";
            compute_defines["V21_FIELD_BF16"] = "1";
        }
        compute_defines["V21S_MAX_K"] = std::to_string(max_k);
        compute_defines["V21S_MAX_H"] = std::to_string(max_h);
    }
    if (use_v6) {
        dm_defines["V21_FXY_BF16"] = "1";
        // compute kernel doesn't need to know — UnpackToDestMode::Default handles bf16→fp32.
    }
    // TEMP: enable layout diagnostic
    if (use_sfpu && getenv("V21_DIAG_LAYOUT")) {
        dm_defines["V21_DIAG_LAYOUT"] = "1";
        compute_defines["V21_DIAG_LAYOUT"] = "1";
        printf("[v21ef] V21_DIAG_LAYOUT enabled\n");
    }

    auto br_k = CreateKernel(
        prog,
        std::string(DENSITY_KERNEL_DIR) + (use_v5 ? "v21_ef_brisc_v5.cpp" : "v21_ef_brisc.cpp"),
        all_crs,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                           .noc       = NOC::RISCV_0_default,
                           .defines   = dm_defines});
    auto nc_k = CreateKernel(
        prog,
        std::string(DENSITY_KERNEL_DIR) + (use_v5 ? "v21_ef_ncrisc_v5.cpp" : "v21_ef_ncrisc.cpp"),
        all_crs,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                           .noc       = NOC::RISCV_1_default,
                           .defines   = dm_defines});

    // Parallel-prep semaphores (SFPU mode only): BRISC and NCRISC each signal
    // their prep-done; the other side waits. Monotonic counter across batches.
    // v5 adds 2 more semaphores (allocated at function scope) for per-tile cb_fxy dual-producer sync.
    if (use_sfpu) {
        sem_brisc_done_id  = CreateSemaphore(prog, all_crs, 0u);
        sem_ncrisc_done_id = CreateSemaphore(prog, all_crs, 0u);
        if (use_v5) {
            sem_brisc_fxy_id  = CreateSemaphore(prog, all_crs, 0u);
            sem_ncrisc_fxy_id = CreateSemaphore(prog, all_crs, 0u);
        }

        // UnpackToDestFp32 for all SFPU input CBs.
        std::vector<UnpackToDestMode> unpack_modes(64, UnpackToDestMode::Default);
        unpack_modes[(uint32_t)CBIndex::c_0] = UnpackToDestMode::UnpackToDestFp32;
        unpack_modes[(uint32_t)CBIndex::c_1] = UnpackToDestMode::UnpackToDestFp32;
        if (!field_bf16) unpack_modes[(uint32_t)CBIndex::c_2] = UnpackToDestMode::UnpackToDestFp32;
        if (!use_v5 && !field_bf16) unpack_modes[(uint32_t)CBIndex::c_3] = UnpackToDestMode::UnpackToDestFp32;
        unpack_modes[(uint32_t)CBIndex::c_4] = UnpackToDestMode::UnpackToDestFp32;
        unpack_modes[(uint32_t)CBIndex::c_5] = UnpackToDestMode::UnpackToDestFp32;

        compute_k = CreateKernel(
            prog,
            std::string(DENSITY_KERNEL_DIR) + (use_v5 ? "v21_ef_compute_v5.cpp" : "v21_ef_compute.cpp"),
            all_crs,
            ComputeConfig{
                .math_fidelity       = MathFidelity::HiFi4,
                .fp32_dest_acc_en    = true,
                .unpack_to_dest_mode = unpack_modes,
                .math_approx_mode    = false,
                .defines             = compute_defines,
            });
    }

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
            const uint32_t br_start = use_sfpu ? chunk_start : 0u;
            const uint32_t br_end   = use_sfpu ? chunk_end   : brisc_end;
            const uint32_t nc_start = use_sfpu ? chunk_start : brisc_end;
            const uint32_t nc_end   = use_sfpu ? chunk_end   : ncrisc_end;
            std::vector<uint32_t> br_args = {(uint32_t)c, br_start, br_end};
            std::vector<uint32_t> nc_args = {(uint32_t)c, nc_start, nc_end};
            br_args.insert(br_args.end(), common_tail.begin(), common_tail.end());
            nc_args.insert(nc_args.end(), common_tail.begin(), common_tail.end());
            if (use_sfpu) {
                br_args.push_back(sem_brisc_done_id);   // arg 21
                br_args.push_back(sem_ncrisc_done_id);  // arg 22
                nc_args.push_back(sem_brisc_done_id);
                nc_args.push_back(sem_ncrisc_done_id);
                if (use_v5) {
                    br_args.push_back(sem_brisc_fxy_id);   // arg 23 (v5 only)
                    br_args.push_back(sem_ncrisc_fxy_id);  // arg 24
                    nc_args.push_back(sem_brisc_fxy_id);
                    nc_args.push_back(sem_ncrisc_fxy_id);
                }
            }
            SetRuntimeArgs(prog, br_k, all_ccs[c], br_args);
            SetRuntimeArgs(prog, nc_k, all_ccs[c], nc_args);
            if (use_sfpu) {
                const uint32_t n_batches = (chunk_end - chunk_start) / vbatch;
                SetRuntimeArgs(prog, compute_k, all_ccs[c], {n_batches});
            }
        }
        return prog;
    };  // end build_program lambda

    // SFPU mode: split into chunks of ≤ MAX_BATCHES_PER_LAUNCH batches per core
    // (empirically > 9 batches per launch triggers a chip-side hang).
    // v23 doubles cells/batch, so cap batches/launch at 8 to keep cells/launch (and
    // chip-side state) within the empirical >9-batch hang limit.
    const uint32_t MAX_BATCHES_PER_LAUNCH = use_v23 ? 8u : 16u;
    const uint32_t chunk_cells = use_sfpu ? (MAX_BATCHES_PER_LAUNCH * vbatch) : cpc;
    const uint32_t n_chunks    = (cpc + chunk_cells - 1u) / chunk_cells;
    printf("[v21ef] launches per run: %u (chunk_cells=%u, cpc=%u)\n", n_chunks, chunk_cells, cpc);

    // Build ONE MeshWorkload up-front, then modify its program's runtime args
    // between EnqueueMeshWorkload calls. This avoids per-chunk Program leak.
    MeshWorkload wl;
    auto device_range = MeshCoordinateRange{mesh_device->shape()};
    wl.add_program(device_range, build_program(0u, std::min(chunk_cells, cpc)));

    // Helper to retrieve the held program and update its runtime args for a chunk.
    auto update_args = [&](uint32_t chunk_start, uint32_t chunk_end) {
        Program& prog_ref = wl.get_programs().at(device_range);
        // Need to know the kernel handles — they're returned by CreateKernel
        // inside build_program. Walk the program's kernels by index (BRISC, NCRISC,
        // COMPUTE were created in that order, so handles 0,1,2).
        constexpr KernelHandle br_handle      = 0;
        constexpr KernelHandle nc_handle      = 1;
        constexpr KernelHandle compute_handle = 2;
        for (int c = 0; c < nc_all; ++c) {
            const uint32_t cells_start_global = (uint32_t)c * cpc;
            std::vector<uint32_t> common_tail = {
                cells_start_global, num_nodes, M, N,
                field_x_base, field_y_base, field_pg_bytes,
                pos_base, const_base, const_pg_bytes,
                grad_base, grad_pg_bytes,
                f2u(xl), f2u(yl), f2u(bsx), f2u(bsy), f2u(inv_bsx), f2u(inv_bsy),
            };
            const uint32_t br_start = use_sfpu ? chunk_start : 0u;
            const uint32_t br_end   = use_sfpu ? chunk_end   : brisc_end;
            const uint32_t nc_start = use_sfpu ? chunk_start : brisc_end;
            const uint32_t nc_end   = use_sfpu ? chunk_end   : ncrisc_end;
            std::vector<uint32_t> br_args = {(uint32_t)c, br_start, br_end};
            std::vector<uint32_t> nc_args = {(uint32_t)c, nc_start, nc_end};
            br_args.insert(br_args.end(), common_tail.begin(), common_tail.end());
            nc_args.insert(nc_args.end(), common_tail.begin(), common_tail.end());
            if (use_sfpu) {
                br_args.push_back(sem_brisc_done_id);
                br_args.push_back(sem_ncrisc_done_id);
                nc_args.push_back(sem_brisc_done_id);
                nc_args.push_back(sem_ncrisc_done_id);
                if (use_v5) {
                    br_args.push_back(sem_brisc_fxy_id);
                    br_args.push_back(sem_ncrisc_fxy_id);
                    nc_args.push_back(sem_brisc_fxy_id);
                    nc_args.push_back(sem_ncrisc_fxy_id);
                }
            }
            SetRuntimeArgs(prog_ref, br_handle, all_ccs[c], br_args);
            SetRuntimeArgs(prog_ref, nc_handle, all_ccs[c], nc_args);
            if (use_sfpu) {
                const uint32_t n_batches = (chunk_end - chunk_start) / vbatch;
                SetRuntimeArgs(prog_ref, compute_handle, all_ccs[c], {n_batches});
            }
        }
    };

    auto run_all_chunks = [&]() {
        for (uint32_t chunk = 0; chunk < n_chunks; ++chunk) {
            const uint32_t cs = chunk * chunk_cells;
            const uint32_t ce = std::min(cs + chunk_cells, cpc);
            update_args(cs, ce);
            EnqueueMeshWorkload(cq, wl, false);
            Finish(cq);  // ensure each launch fully completes before next
        }
    };

    // Warmup
    printf("[v21ef] JIT warmup...\n"); fflush(stdout);
    auto t_jit = hrclock::now();
    run_all_chunks();
    double warmup_ms = ms_since(t_jit);
    printf("[v21ef] warmup = %.2f ms\n", warmup_ms);

    const int N_RUNS = 5;
    std::vector<double> run_ms;
    for (int r = 0; r < N_RUNS; ++r) {
        auto t_run = hrclock::now();
        run_all_chunks();
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
