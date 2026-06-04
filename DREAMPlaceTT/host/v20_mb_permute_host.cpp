// SPDX-License-Identifier: Apache-2.0
//
// V20 permute microbench — host driver.
//
// Models the proposed V20 chip-side permute (see docs/V20_CHIP_PERMUTE_HANDOFF.md).
// Measures the per-iter cost of the permute pass: how long does the chip take
// to do, for each output slot c:
//     px_buf[c] = pos[ix[c]] + ox[c]
//     py_buf[c] = pos[iy[c]] + oy[c]
//
// Inputs are synthetic; output is verified bit-identical to a host reference
// computation, then per-stage Tracy timing is captured.
//
// CLI:
//   v20_mb_permute_host <num_nodes> <new_nc>
//     num_nodes : size of synthetic pos[]  (host uploads 2*num_nodes floats)
//     new_nc    : number of output slots (post-subcell + post-sort cell count)
//
// Reference sizings:
//   adaptec1_2048 : num_nodes=~1100000  new_nc=~371000
//   bigblue3_2048 : num_nodes=~1100000  new_nc=~2040000

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
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

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <num_nodes> <new_nc>\n"
            "  num_nodes : pos array length (host uploads 2*num_nodes floats)\n"
            "  new_nc    : output slot count\n",
            argv[0]);
        return 1;
    }
    uint32_t num_nodes = (uint32_t)std::atoi(argv[1]);
    uint32_t new_nc    = (uint32_t)std::atoi(argv[2]);
    printf("[v20mb] num_nodes=%u  new_nc=%u\n", num_nodes, new_nc);
    fflush(stdout);

    // ── Open mesh device ──
    auto mesh_device = MeshDevice::create_unit_mesh(0);
    auto& cq = mesh_device->mesh_command_queue();
    auto grid = mesh_device->compute_with_storage_grid_size();
    const int nc_all = (int)(grid.x * grid.y);
    printf("[v20mb] grid=%ux%u nc_all=%d\n", grid.x, grid.y, nc_all);

    // Core list (110 cores on Blackhole).
    std::vector<CoreCoord> all_ccs;
    for (uint32_t y = 0; y < grid.y; ++y)
        for (uint32_t x = 0; x < grid.x; ++x)
            all_ccs.push_back(CoreCoord{x, y});
    std::set<CoreRange> crs_set;
    for (auto cc : all_ccs) crs_set.insert(CoreRange{cc, cc});
    CoreRangeSet all_crs(crs_set);

    // ── Synthetic input data ──
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> uxy(0.0f, 10000.0f);
    // pos layout: [x_0, x_1, ..., x_(nn-1), y_0, y_1, ..., y_(nn-1)]
    std::vector<float> pos(2u * num_nodes);
    for (uint32_t i = 0; i < pos.size(); ++i) pos[i] = uxy(rng);

    // perm_table: new_nc × {ix, iy, ox_bits, oy_bits} = new_nc × 16 bytes.
    // ix[c] picks a random x-coord (0..num_nodes-1).
    // iy[c] picks a corresponding y-coord (num_nodes..2*num_nodes-1).
    // ox/oy are random small floats (sub-cell offsets).
    std::vector<uint32_t> perm_table(new_nc * 4u);
    std::uniform_int_distribution<uint32_t> uidx(0, num_nodes - 1);
    std::uniform_real_distribution<float> uoff(0.0f, 5.0f);
    for (uint32_t c = 0; c < new_nc; ++c) {
        uint32_t ix = uidx(rng);
        uint32_t iy = num_nodes + uidx(rng);
        float ox = uoff(rng);
        float oy = uoff(rng);
        perm_table[c * 4 + 0] = ix;
        perm_table[c * 4 + 1] = iy;
        std::memcpy(&perm_table[c * 4 + 2], &ox, 4);
        std::memcpy(&perm_table[c * 4 + 3], &oy, 4);
    }

    // ── Page sizing ──
    // pos: 4 KB pages, random reads from arbitrary 32-B aligned offsets.
    // perm_table / px_buf / py_buf: per-core pages so each core writes only
    //   to its own page (no intra-page write-alignment problem).
    constexpr uint32_t POS_PG_BYTES  = 4096u;
    const uint32_t floats_per_pos_pg = POS_PG_BYTES / 4u;
    uint32_t pos_pages  = ((uint32_t)pos.size() + floats_per_pos_pg - 1u) / floats_per_pos_pg;
    pos.resize((size_t)pos_pages * floats_per_pos_pg, 0.0f);

    // Pad new_nc so each core gets a multiple-of-8 number of cells (so all
    // 32-B alignment requirements are trivially satisfied inside per-core pages).
    // PAD_GRAN = nc_all × 8 ensures cells_per_core_padded is a multiple of 8.
    uint32_t PAD_GRAN = (uint32_t)nc_all * 8u;
    uint32_t new_nc_padded = ((new_nc + PAD_GRAN - 1u) / PAD_GRAN) * PAD_GRAN;
    uint32_t cells_per_core_padded = new_nc_padded / (uint32_t)nc_all;
    if ((cells_per_core_padded & 7u) != 0u) {
        fprintf(stderr, "[v20mb] FATAL: cells_per_core_padded=%u not multiple of 8\n",
                cells_per_core_padded);
        return 1;
    }
    // Extend perm_table to full padded size (extra entries point to pos[0] with offset 0).
    perm_table.resize((size_t)new_nc_padded * 4u, 0u);
    uint32_t perm_pg_bytes = cells_per_core_padded * 16u;
    uint32_t xy_pg_bytes   = cells_per_core_padded * 4u;
    perm_pg_bytes = (perm_pg_bytes + 31u) & ~31u;
    xy_pg_bytes   = (xy_pg_bytes   + 31u) & ~31u;
    printf("[v20mb] padded new_nc=%u → %u (cells_per_core_padded=%u, perm_pg=%u B, xy_pg=%u B)\n",
           new_nc, new_nc_padded, cells_per_core_padded, perm_pg_bytes, xy_pg_bytes);

    std::vector<float> px_buf((size_t)nc_all * (xy_pg_bytes / 4u), 0.0f);
    std::vector<float> py_buf((size_t)nc_all * (xy_pg_bytes / 4u), 0.0f);

    auto make_buf = [&](uint64_t total_bytes, uint32_t pgsz) {
        DeviceLocalBufferConfig cfg{.page_size = pgsz, .buffer_type = BufferType::DRAM};
        ReplicatedBufferConfig rcfg{.size = total_bytes};
        return MeshBuffer::create(rcfg, cfg, mesh_device.get());
    };
    auto pos_buf_dram   = make_buf((uint64_t)pos_pages  * POS_PG_BYTES,  POS_PG_BYTES);
    auto perm_buf_dram  = make_buf((uint64_t)nc_all     * perm_pg_bytes, perm_pg_bytes);
    auto pxbuf_dram     = make_buf((uint64_t)nc_all     * xy_pg_bytes,   xy_pg_bytes);
    auto pybuf_dram     = make_buf((uint64_t)nc_all     * xy_pg_bytes,   xy_pg_bytes);

    printf("[v20mb] DRAM: pos=%u pages × %u B = %.1f MB\n",
           pos_pages, POS_PG_BYTES, (double)pos_pages * POS_PG_BYTES / 1048576.0);
    printf("[v20mb] DRAM: perm=%d pages × %u B = %.1f MB (1 page per core)\n",
           nc_all, perm_pg_bytes, (double)nc_all * perm_pg_bytes / 1048576.0);
    printf("[v20mb] DRAM: px_buf=py_buf=%d pages × %u B each (1 page per core)\n",
           nc_all, xy_pg_bytes);
    fflush(stdout);

    // ── Upload inputs ──
    // perm_table is flat [new_nc_padded × 4 × u32]; EnqueueWriteMeshBuffer
    // interleaves pages across DRAM banks. Page core_idx contains entries
    // for cells [core_idx * cpc, (core_idx+1) * cpc).
    auto t_h2d = hrclock::now();
    EnqueueWriteMeshBuffer(cq, pos_buf_dram,  pos,        false);
    EnqueueWriteMeshBuffer(cq, perm_buf_dram, perm_table, false);
    Finish(cq);
    double h2d_ms = ms_since(t_h2d);
    printf("[v20mb] h2d (pos + perm) = %.2f ms\n", h2d_ms);

    // ── Contiguous per-core partition (core C handles cells [C*cpc, (C+1)*cpc)). ──
    // No first_cell array needed: each core's first cell is implied by its
    // page assignment. The kernel just reads page my_core_idx of perm/px/py.

    // ── L1 layout ──
    // BRISC and NCRISC each need ~25 KB of staging (perm chunk = 4 × BATCH = 8 KB).
    // Allocate 32 KB per CB. c_24 for NCRISC, c_25 for BRISC — separate CBs so
    // the two kernels' L1 regions don't collide on the same core.
    constexpr uint32_t CB_SIZE = 32u * 1024u;
    using tt::CBIndex;

    // ── Split cells between BRISC and NCRISC. ──
    // BRISC handles [0, brisc_end); NCRISC handles [brisc_end, cells_per_core_padded).
    // Need brisc_end aligned to 4 so perm-read offset (×16) is 64-aligned and
    // out-write offset (×4) is 16-aligned. cells_per_core_padded is mult of 8
    // (host invariant) → cpc/2 is mult of 4 ✓.
    const uint32_t brisc_end   = cells_per_core_padded / 2u;
    const uint32_t ncrisc_end  = cells_per_core_padded;
    if ((brisc_end & 3u) != 0u) {
        fprintf(stderr,
                "[v20mb] FATAL: brisc_end=%u not a multiple of 4 (cpc=%u)\n",
                brisc_end, cells_per_core_padded);
        return 1;
    }
    printf("[v20mb] split: BRISC=[0,%u)  NCRISC=[%u,%u)\n",
           brisc_end, brisc_end, ncrisc_end);

    // ── Build the program ──
    auto* dev0 = mesh_device->get_devices()[0];
    Program prog;
    CircularBufferConfig cb24_cfg(CB_SIZE, {{CBIndex::c_24, DataFormat::Float32}});
    cb24_cfg.set_page_size(CBIndex::c_24, CB_SIZE);
    CreateCircularBuffer(prog, all_crs, cb24_cfg);
    CircularBufferConfig cb25_cfg(CB_SIZE, {{CBIndex::c_25, DataFormat::Float32}});
    cb25_cfg.set_page_size(CBIndex::c_25, CB_SIZE);
    CreateCircularBuffer(prog, all_crs, cb25_cfg);

    auto br_perm = CreateKernel(
        prog,
        std::string(DENSITY_KERNEL_DIR) + "v19_mb_permute_brisc.cpp",
        all_crs,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                            .noc       = NOC::RISCV_0_default});
    auto nc_perm = CreateKernel(
        prog,
        std::string(DENSITY_KERNEL_DIR) + "v19_mb_permute_ncrisc.cpp",
        all_crs,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                            .noc       = NOC::RISCV_1_default});

    uint32_t perm_base   = (uint32_t)perm_buf_dram->address();
    uint32_t pos_base    = (uint32_t)pos_buf_dram->address();
    uint32_t pxbuf_base  = (uint32_t)pxbuf_dram->address();
    uint32_t pybuf_base  = (uint32_t)pybuf_dram->address();

    for (int c = 0; c < nc_all; ++c) {
        // Kernel ABI (BRISC + NCRISC, same args except cells_start/end):
        //   arg  0: my_core_idx
        //   arg  1: cells_start         (intra-core, inclusive)
        //   arg  2: cells_end           (intra-core, exclusive)
        //   arg  3: pos_dram_base
        //   arg  4: pos_dram_pgsz
        //   arg  5: floats_per_pos_pg
        //   arg  6: perm_dram_base
        //   arg  7: perm_pg_bytes
        //   arg  8: pxbuf_dram_base
        //   arg  9: pybuf_dram_base
        //   arg 10: xy_pg_bytes
        std::vector<uint32_t> common_tail = {
            pos_base,  POS_PG_BYTES,  floats_per_pos_pg,
            perm_base, perm_pg_bytes,
            pxbuf_base, pybuf_base,
            xy_pg_bytes,
        };
        std::vector<uint32_t> br_args = {(uint32_t)c, 0u,         brisc_end};
        std::vector<uint32_t> nc_args = {(uint32_t)c, brisc_end,  ncrisc_end};
        br_args.insert(br_args.end(), common_tail.begin(), common_tail.end());
        nc_args.insert(nc_args.end(), common_tail.begin(), common_tail.end());
        SetRuntimeArgs(prog, br_perm, all_ccs[c], br_args);
        SetRuntimeArgs(prog, nc_perm, all_ccs[c], nc_args);
    }
    MeshWorkload wl;
    auto device_range = MeshCoordinateRange{mesh_device->shape()};
    wl.add_program(device_range, std::move(prog));

    // ── Warmup: launch once (triggers JIT) then time the second launch. ──
    printf("[v20mb] JIT warmup...\n"); fflush(stdout);
    auto t_jit = hrclock::now();
    EnqueueMeshWorkload(cq, wl, false);
    Finish(cq);
    double warmup_ms = ms_since(t_jit);
    printf("[v20mb] warmup (JIT) = %.2f ms\n", warmup_ms);

    // ── Timed run (already JIT'd) ──
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
    printf("[v20mb] permute kernel runs (N=%d): ", N_RUNS);
    for (double m : run_ms) printf("%.2f ", m);
    printf("→ median %.2f ms\n", median_ms);

    // ── Read back outputs ──
    auto t_d2h = hrclock::now();
    EnqueueReadMeshBuffer(cq, px_buf, pxbuf_dram, true);
    EnqueueReadMeshBuffer(cq, py_buf, pybuf_dram, true);
    double d2h_ms = ms_since(t_d2h);
    printf("[v20mb] d2h (px + py) = %.2f ms\n", d2h_ms);

    // ── Accuracy verify vs host reference ──
    // Output layout in DRAM: nc_all pages, each holding cells_per_core_padded
    // floats. Core C's slot is page C, intra-position c%cpc.
    // Read-back vector is the same flat layout.
    // floats per page = xy_pg_bytes / 4.
    const uint32_t floats_per_xy_pg = xy_pg_bytes / 4u;
    uint32_t n_mismatch = 0;
    float max_diff = 0.0f;
    int first_bad = -1;
    for (uint32_t c = 0; c < new_nc; ++c) {
        uint32_t core_idx = c / cells_per_core_padded;
        uint32_t intra    = c % cells_per_core_padded;
        size_t flat_idx   = (size_t)core_idx * floats_per_xy_pg + intra;
        uint32_t ix = perm_table[c * 4 + 0];
        uint32_t iy = perm_table[c * 4 + 1];
        float ox, oy;
        std::memcpy(&ox, &perm_table[c * 4 + 2], 4);
        std::memcpy(&oy, &perm_table[c * 4 + 3], 4);
        float ref_px = pos[ix] + ox;
        float ref_py = pos[iy] + oy;
        float got_px = px_buf[flat_idx];
        float got_py = py_buf[flat_idx];
        float dx = std::abs(ref_px - got_px);
        float dy = std::abs(ref_py - got_py);
        if (dx > max_diff) max_diff = dx;
        if (dy > max_diff) max_diff = dy;
        if (dx != 0.0f || dy != 0.0f) {
            if (first_bad < 0) first_bad = (int)c;
            n_mismatch++;
        }
    }

    printf("[v20mb] === RESULT num_nodes=%u  new_nc=%u ===\n", num_nodes, new_nc);
    printf("[v20mb]   permute median = %.3f ms  (h2d=%.2f, d2h=%.2f)\n",
           median_ms, h2d_ms, d2h_ms);
    printf("[v20mb]   mismatches=%u / %u   max_diff=%g\n",
           n_mismatch, new_nc, (double)max_diff);
    if (n_mismatch != 0u && first_bad >= 0) {
        uint32_t c = (uint32_t)first_bad;
        uint32_t core_idx = c / cells_per_core_padded;
        uint32_t intra    = c % cells_per_core_padded;
        size_t flat_idx   = (size_t)core_idx * floats_per_xy_pg + intra;
        uint32_t ix = perm_table[c * 4 + 0];
        uint32_t iy = perm_table[c * 4 + 1];
        float ox, oy;
        std::memcpy(&ox, &perm_table[c * 4 + 2], 4);
        std::memcpy(&oy, &perm_table[c * 4 + 3], 4);
        printf("[v20mb]   first mismatch c=%u (core=%u intra=%u)  ix=%u iy=%u ox=%g oy=%g\n"
               "[v20mb]      ref=(%g, %g)  got=(%g, %g)\n",
               c, core_idx, intra, ix, iy, (double)ox, (double)oy,
               (double)(pos[ix] + ox), (double)(pos[iy] + oy),
               (double)px_buf[flat_idx], (double)py_buf[flat_idx]);
        printf("[v20mb]   ✗ FAIL — host vs chip diverged\n");
        return 2;
    }
    printf("[v20mb]   ✓ accuracy OK\n");
    printf("RESULT,%u,%u,%.3f,%.3f,%.3f,%u,%g\n",
           num_nodes, new_nc, median_ms, h2d_ms, d2h_ms, n_mismatch, (double)max_diff);
    return 0;
}
