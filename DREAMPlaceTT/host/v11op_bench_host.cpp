// SPDX-License-Identifier: Apache-2.0
//
// V11 Stage B' — SFPU outer-product microbench host driver.
//
// Dispatches kernels/v11op_bench_compute.cpp on one core to measure the
// wall-clock cost of computing 64 (j, k) outer-product values per cell on
// the SFPU. This is what V11 Stage B' would do if the per-cell scalar
// ox * oy multiplies were moved from BRISC/NCRISC scalar to TRISC SFPU.
//
// Each "batch" processes 1024 cells through the SFPU. Per batch, the kernel
// runs 64 SFPU passes (one per (j, k) outer-product position).
//
// Reference:
//   V11 scatter at adaptec1_512: ~1.95 ms across 110 cores → ~17 µs/core
//     for ~1800 cells/core. Per-cell BRISC scalar work: ~10 ns/cell + 64
//     scalar multiplies/cell (the bottleneck per memory/v11_scatter_is_mul_bound).
//
// Usage:
//   v11op_bench_host [n_batches]   (default 1000)

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/allocator.hpp>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

using hrclock = std::chrono::high_resolution_clock;
template <class T>
static double ms_since(T t0) {
    return std::chrono::duration<double, std::milli>(hrclock::now() - t0).count();
}

int main(int argc, char* argv[]) {
    uint32_t n_batches = (argc >= 2) ? (uint32_t)atoi(argv[1]) : 1000u;
    fprintf(stderr, "[v11op-bench] n_batches=%u (each batch = 1024 cells × 64 outer products on SFPU)\n",
            n_batches);

    auto mesh_device = MeshDevice::create_unit_mesh(0);
    MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    MeshCoordinateRange device_range(mesh_device->shape());

    CoreCoord core{0, 0};
    std::set<CoreRange> crs_set{CoreRange{core, core}};
    CoreRangeSet core_crs(crs_set);

    constexpr uint32_t TILE_FP32_BYTES = 32u * 32u * sizeof(float);  // 4096

    Program prog = CreateProgram();
    // Input CBs: c_0 (surrogate ox), c_1 (surrogate oy). One tile each.
    CreateCircularBuffer(prog, core_crs,
        CircularBufferConfig(2u * TILE_FP32_BYTES, {{0u, tt::DataFormat::Float32}})
            .set_page_size(0u, TILE_FP32_BYTES));
    CreateCircularBuffer(prog, core_crs,
        CircularBufferConfig(2u * TILE_FP32_BYTES, {{1u, tt::DataFormat::Float32}})
            .set_page_size(1u, TILE_FP32_BYTES));
    // Output CB: 64 tiles per batch. Sized to hold a few batches for pipelining.
    CreateCircularBuffer(prog, core_crs,
        CircularBufferConfig(4u * TILE_FP32_BYTES, {{16u, tt::DataFormat::Float32}})
            .set_page_size(16u, TILE_FP32_BYTES));

    std::string kdir;
#ifdef DENSITY_KERNEL_DIR
    kdir = DENSITY_KERNEL_DIR;
#else
    kdir = "./kernels/";
#endif

    auto brisc_kernel = CreateKernel(prog,
        kdir + "v11op_bench_brisc.cpp",
        core_crs,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc       = NOC::RISCV_0_default});

    CreateKernel(prog,
        kdir + "v11op_bench_ncrisc.cpp",
        core_crs,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc       = NOC::RISCV_1_default});

    auto compute_kernel = CreateKernel(prog,
        kdir + "v11op_bench_compute.cpp",
        core_crs,
        ComputeConfig{
            .math_fidelity    = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .math_approx_mode = false,
        });

    SetRuntimeArgs(prog, brisc_kernel,   core, {n_batches});
    SetRuntimeArgs(prog, compute_kernel, core, {n_batches});

    MeshWorkload wl;
    wl.add_program(device_range, std::move(prog));

    // Warmup.
    fprintf(stderr, "[v11op-bench] warmup launch...\n");
    auto tw = hrclock::now();
    EnqueueMeshWorkload(cq, wl, false);
    Finish(cq);
    fprintf(stderr, "[v11op-bench] warmup: %.3f ms\n", ms_since(tw));

    // Timed launch.
    fprintf(stderr, "[v11op-bench] timed launch (n_batches=%u)...\n", n_batches);
    auto t0 = hrclock::now();
    EnqueueMeshWorkload(cq, wl, false);
    Finish(cq);
    double wall_ms = ms_since(t0);
    fprintf(stderr, "[v11op-bench] wall: %.3f ms\n", wall_ms);

    double per_batch_us = wall_ms * 1000.0 / (double)n_batches;
    double per_cell_ns  = wall_ms * 1e6 / (double)n_batches / 1024.0;
    double per_op_ns    = wall_ms * 1e6 / (double)n_batches / 1024.0 / 64.0;

    fprintf(stderr, "[v11op-bench] per-batch (1024 cells × 64 ops): %.3f µs\n", per_batch_us);
    fprintf(stderr, "[v11op-bench] per-cell (64 outer-product ops):  %.3f ns\n", per_cell_ns);
    fprintf(stderr, "[v11op-bench] per-(j,k) outer-product op:        %.3f ns\n", per_op_ns);
    fprintf(stderr, "\n");
    fprintf(stderr, "[v11op-bench] Reference:\n");
    fprintf(stderr, "[v11op-bench]   V11 BRISC scalar `area = ox*oy` (estimated): ~10-40 ns/multiply\n");
    fprintf(stderr, "[v11op-bench]   For full V11 Stage B' comparison, use Tracy profile vs V11 V4C/V11A zones.\n");

    printf("{\"n_batches\": %u, \"wall_ms\": %.4f, "
           "\"per_batch_us\": %.4f, "
           "\"per_cell_ns\": %.4f, "
           "\"per_op_ns\": %.4f}\n",
           n_batches, wall_ms, per_batch_us, per_cell_ns, per_op_ns);
    return 0;
}
