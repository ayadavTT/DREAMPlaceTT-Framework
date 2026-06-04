// SPDX-License-Identifier: Apache-2.0
//
// Atomic-increment microbench host.
//
// Modes:
//   loopback         1 core, target = self
//   cross            1 core, target = neighbour
//   contended N      N cores all incrementing same target (N=10/30/60/110)
//   valuesum         All 110 cores, each with a UNIQUE incr_value, verifies
//                    that the final counter == sum(per-core values)*n_iters
//                    (proves atomic_inc can sum arbitrary uint32s across cores)
//
// Usage:
//   atomic_bench_host all [n_iters=10000]
//   atomic_bench_host sweep [n_iters=10000]
//   atomic_bench_host valuesum [n_iters=10000]
//
// Reads back the actual counter from L1 to validate every atomic landed.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/allocator.hpp>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

static constexpr double GHZ = 1.35;
static constexpr double NS_PER_CYCLE = 1.0 / GHZ;

// CB_SCRATCH layout (must match kernel):
//   [0 .. 4)    counter (uint32)
//   [4 .. 8)    cyc_lo (uint32)
//   [8 .. 12)   cyc_hi (uint32)
constexpr uint32_t CB_SIZE = 64u;
constexpr uint32_t COUNTER_OFF = 0u;
constexpr uint32_t CYC_LO_OFF  = 4u;
constexpr uint32_t CYC_HI_OFF  = 8u;

struct Result {
    std::string label;
    uint32_t n_iters;
    uint32_t n_cores;
    double mean_ns;
    double median_ns;
    double max_ns;
    uint64_t expected_counter;
    uint64_t actual_counter;
    bool ok;
};

static Result run_one(MeshDevice* mesh, MeshCommandQueue& cq,
                      const std::string& kdir,
                      const std::string& label,
                      uint32_t n_iters,
                      const std::vector<CoreCoord>& source_cores,
                      CoreCoord target_core,
                      bool posted,
                      const std::vector<uint32_t>& per_src_value,
                      bool dram_mode = false,
                      std::shared_ptr<MeshBuffer> dram_buf = nullptr) {
    fprintf(stderr, "\n=== %s (n_iters=%u, n_sources=%zu, target=(%u,%u), posted=%d) ===\n",
            label.c_str(), n_iters, source_cores.size(),
            target_core.x, target_core.y, (int)posted);

    Program prog = CreateProgram();
    std::set<CoreRange> crs_set;
    for (const auto& c : source_cores) crs_set.insert(CoreRange{c, c});
    crs_set.insert(CoreRange{target_core, target_core});
    CoreRangeSet crs(crs_set);

    CreateCircularBuffer(prog, crs,
        CircularBufferConfig(CB_SIZE, {{24u, tt::DataFormat::UInt32}})
            .set_page_size(24u, CB_SIZE));

    auto brisc_kernel = CreateKernel(prog,
        kdir + "atomic_bench_brisc.cpp", crs,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                           .noc = NOC::RISCV_0_default});
    CreateKernel(prog,
        kdir + "atomic_bench_ncrisc.cpp", crs,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                           .noc = NOC::RISCV_1_default});

    CoreCoord target_noc = mesh->get_devices()[0]->worker_core_from_logical_core(target_core);

    uint32_t dram_addr = dram_buf ? (uint32_t)dram_buf->address() : 0u;
    uint64_t expected = 0;
    for (size_t i = 0; i < source_cores.size(); ++i) {
        SetRuntimeArgs(prog, brisc_kernel, source_cores[i], {
            n_iters,
            COUNTER_OFF,
            (uint32_t)target_noc.x, (uint32_t)target_noc.y,
            CYC_LO_OFF, CYC_HI_OFF,
            (uint32_t)(posted ? 1 : 0),
            per_src_value[i],
            (uint32_t)(dram_mode ? 1 : 0),
            dram_addr,
        });
        expected += (uint64_t)per_src_value[i] * (uint64_t)n_iters;
    }
    bool target_is_source = false;
    for (const auto& sc : source_cores)
        if (sc.x == target_core.x && sc.y == target_core.y) target_is_source = true;
    if (!target_is_source) {
        SetRuntimeArgs(prog, brisc_kernel, target_core, {
            0u, COUNTER_OFF,
            (uint32_t)target_noc.x, (uint32_t)target_noc.y,
            CYC_LO_OFF, CYC_HI_OFF, 0u, 0u,
            (uint32_t)(dram_mode ? 1 : 0), dram_addr,
        });
    }

    MeshWorkload wl;
    MeshCoordinateRange device_range(mesh->shape());
    wl.add_program(device_range, std::move(prog));

    EnqueueMeshWorkload(cq, wl, false); Finish(cq);  // warmup

    auto* device = mesh->get_devices()[0];
    uint32_t l1_base_pre = (uint32_t)device->allocator()->get_base_allocator_addr(HalMemType::L1);
    // Pre-zero the counter at the target (L1 or DRAM).
    if (dram_mode) {
        std::vector<uint32_t> zeros(8, 0u);  // 32-byte DRAM page
        EnqueueWriteMeshBuffer(cq, dram_buf, zeros, false);
        Finish(cq);
    } else {
        std::vector<uint8_t> zero_buf(4, 0);
        tt::tt_metal::detail::WriteToDeviceL1(
            device, target_core, l1_base_pre + COUNTER_OFF, zero_buf);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    EnqueueMeshWorkload(cq, wl, false); Finish(cq);
    double host_wall_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();

    // Read final counter (L1 or DRAM)
    uint32_t actual_counter = 0;
    if (dram_mode) {
        std::vector<uint32_t> readback(8, 0u);
        EnqueueReadMeshBuffer(cq, readback, dram_buf, true);
        actual_counter = readback[0];
    } else {
        std::vector<uint8_t> ctr_buf(4);
        tt::tt_metal::detail::ReadFromDeviceL1(
            device, target_core, l1_base_pre + COUNTER_OFF,
            std::span<uint8_t>(ctr_buf.data(), ctr_buf.size()));
        std::memcpy(&actual_counter, ctr_buf.data(), 4);
    }
    uint32_t l1_base = l1_base_pre;

    // Per-core cycle counts → ns-per-inc
    std::vector<double> ns_per_inc;
    ns_per_inc.reserve(source_cores.size());
    for (const auto& sc : source_cores) {
        std::vector<uint8_t> cyc_buf(8);
        tt::tt_metal::detail::ReadFromDeviceL1(
            device, sc, l1_base + CYC_LO_OFF,
            std::span<uint8_t>(cyc_buf.data(), cyc_buf.size()));
        uint32_t lo, hi;
        std::memcpy(&lo, cyc_buf.data(), 4);
        std::memcpy(&hi, cyc_buf.data() + 4, 4);
        uint32_t cycles = hi - lo;
        ns_per_inc.push_back((cycles * NS_PER_CYCLE) / std::max(1u, n_iters));
    }
    double mean = 0.0, maxv = 0.0;
    for (double v : ns_per_inc) { mean += v; maxv = std::max(maxv, v); }
    mean /= ns_per_inc.size();
    std::sort(ns_per_inc.begin(), ns_per_inc.end());
    double median = ns_per_inc[ns_per_inc.size() / 2];

    // expected is uint64, counter is uint32 — for very large sums it wraps,
    // which is fine for the integer-overflow case (still bit-correct).
    uint64_t expected_wrapped = expected & 0xFFFFFFFFu;
    bool ok = (uint64_t)actual_counter == expected_wrapped;

    Result r{label, n_iters, (uint32_t)source_cores.size(),
             mean, median, maxv, expected, actual_counter, ok};

    fprintf(stderr, "  host wall: %.2f ms\n", host_wall_ms);
    fprintf(stderr, "  per-inc latency: mean=%.1f ns  median=%.1f ns  max=%.1f ns\n",
            mean, median, maxv);
    fprintf(stderr, "  expected counter: %lu (low32=%lu)\n",
            (unsigned long)expected, (unsigned long)expected_wrapped);
    fprintf(stderr, "  actual counter:   %u  %s\n",
            actual_counter, ok ? "✓ MATCH" : "✗ MISMATCH");
    return r;
}

int main(int argc, char* argv[]) {
    std::string mode_arg = (argc >= 2) ? argv[1] : "all";
    uint32_t n_iters = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 10000u;

    auto mesh = MeshDevice::create_unit_mesh(0);
    MeshCommandQueue& cq = mesh->mesh_command_queue();
    auto grid = mesh->compute_with_storage_grid_size();
    fprintf(stderr, "[atomic-bench] grid = %ux%u, n_iters=%u\n",
            (unsigned)grid.x, (unsigned)grid.y, n_iters);

    std::string kdir;
#ifdef DENSITY_KERNEL_DIR
    kdir = DENSITY_KERNEL_DIR;
#else
    kdir = "./kernels/";
#endif

    auto all_worker_cores = [&]() {
        std::vector<CoreCoord> v;
        for (uint32_t y = 0; y < grid.y; ++y)
            for (uint32_t x = 0; x < grid.x; ++x)
                v.push_back(CoreCoord{x, y});
        return v;
    };

    std::vector<Result> results;
    CoreCoord tgt{0, 0};

    if (mode_arg == "loopback" || mode_arg == "all") {
        results.push_back(run_one(mesh.get(), cq, kdir,
            "loopback (1→self)", n_iters, {tgt}, tgt, false, {1u}));
    }
    if (mode_arg == "cross" || mode_arg == "all") {
        CoreCoord src{0, 0}; CoreCoord nbr{1, 0};
        results.push_back(run_one(mesh.get(), cq, kdir,
            "cross-core (1→neighbour)", n_iters, {src}, nbr, false, {1u}));
    }
    if (mode_arg == "sweep" || mode_arg == "all") {
        auto allc = all_worker_cores();
        for (uint32_t N : {10u, 30u, 60u, (uint32_t)allc.size()}) {
            std::vector<CoreCoord> srcs(allc.begin(), allc.begin() + N);
            std::vector<uint32_t> vals(N, 1u);
            results.push_back(run_one(mesh.get(), cq, kdir,
                "contended N=" + std::to_string(N), n_iters,
                srcs, tgt, false, vals));
        }
    }
    if (mode_arg == "dram_atomic" || mode_arg == "dram" || mode_arg == "all") {
        // DRAM-target atomic-add: same patterns as L1 but counter lives in DRAM.
        // Allocate a single 32-byte DRAM page as the counter.
        DeviceLocalBufferConfig dram_cfg{.page_size = 32u, .buffer_type = BufferType::DRAM};
        ReplicatedBufferConfig dram_rcfg{.size = 32u};
        auto dram_buf = MeshBuffer::create(dram_rcfg, dram_cfg, mesh.get());

        // 1) Single-source DRAM atomic
        results.push_back(run_one(mesh.get(), cq, kdir,
            "DRAM single-source", n_iters, {tgt}, tgt, false, {1u},
            /*dram_mode=*/true, dram_buf));

        // 2) Contention sweep to DRAM
        auto allc = all_worker_cores();
        for (uint32_t N : {10u, 30u, 60u, (uint32_t)allc.size()}) {
            std::vector<CoreCoord> srcs(allc.begin(), allc.begin() + N);
            std::vector<uint32_t> vals(N, 1u);
            results.push_back(run_one(mesh.get(), cq, kdir,
                "DRAM contended N=" + std::to_string(N), n_iters,
                srcs, tgt, false, vals,
                /*dram_mode=*/true, dram_buf));
        }
    }
    if (mode_arg == "fixedpt" || mode_arg == "all") {
        // Test fixed-point fp32 accumulation:
        //   per-core area_fp32 in [0.5, 0.6) — like a DREAMPlace density contrib
        //   fixed = (uint32_t)(area_fp32 * SCALE)
        //   atomic-inc n_iters times → counter
        //   density_fp32 = (float)counter / SCALE
        // Compare against fp64 reference sum, report relative error.
        // Sweep SCALE = 2^12 .. 2^20 to find precision/range balance.
        auto allc = all_worker_cores();
        const uint32_t n_test_iters = std::min<uint32_t>(n_iters, 10u);
        // Generate per-core area values (deterministic): area[k] = 0.5 + 0.001*k
        std::vector<float> area_fp32(allc.size());
        for (size_t k = 0; k < allc.size(); ++k)
            area_fp32[k] = 0.5f + 0.001f * (float)k;

        // Reference sum in fp64 (gold)
        double ref_sum_per_iter = 0.0;
        for (float v : area_fp32) ref_sum_per_iter += (double)v;
        double ref_total = ref_sum_per_iter * (double)n_test_iters;
        fprintf(stderr, "\n[fixedpt] per-core area_fp32 ∈ [%.3f, %.3f), "
                        "ref sum/iter = %.6f, iters = %u, ref total = %.6f\n",
                area_fp32.front(), area_fp32.back() + 0.001f,
                ref_sum_per_iter, n_test_iters, ref_total);

        for (uint32_t SCALE_BITS : {12u, 14u, 16u, 18u, 20u}) {
            uint32_t SCALE = 1u << SCALE_BITS;
            // Build per-core fixed-point increment values.
            std::vector<uint32_t> vals(allc.size());
            uint64_t expected_fixed_per_iter = 0;
            for (size_t k = 0; k < allc.size(); ++k) {
                vals[k] = (uint32_t)(area_fp32[k] * (float)SCALE);
                expected_fixed_per_iter += vals[k];
            }
            // Check uint32 wouldn't overflow: total = expected_fixed_per_iter × n_iters
            uint64_t expected_total = expected_fixed_per_iter * (uint64_t)n_test_iters;
            if (expected_total > 0xFFFFFFFFull) {
                fprintf(stderr, "\n[fixedpt SCALE=2^%u] skipped — would overflow uint32 (expected=%lu > 2^32)\n",
                        SCALE_BITS, (unsigned long)expected_total);
                continue;
            }
            std::string lbl = "fixedpt SCALE=2^" + std::to_string(SCALE_BITS);
            auto r = run_one(mesh.get(), cq, kdir, lbl,
                             n_test_iters, allc, tgt, false, vals);
            // Recover density_fp32 from counter
            double measured_fp32 = (double)r.actual_counter / (double)SCALE;
            double abs_err = measured_fp32 - ref_total;
            double rel_err = abs_err / ref_total;
            fprintf(stderr, "  measured fp32 sum (counter/SCALE) = %.6f\n", measured_fp32);
            fprintf(stderr, "  ref (fp64) sum                    = %.6f\n", ref_total);
            fprintf(stderr, "  abs error = %.6e  rel error = %.3e  (≈ %.4f ppm)\n",
                    abs_err, rel_err, rel_err * 1e6);
            results.push_back(r);
        }
    }
    if (mode_arg == "valuesum" || mode_arg == "all") {
        // Each core adds a UNIQUE uint32 value: core_k → value = k*7 + 1000
        // Final counter must == sum(k*7+1000) * n_iters mod 2^32
        auto allc = all_worker_cores();
        std::vector<uint32_t> vals;
        for (size_t i = 0; i < allc.size(); ++i) vals.push_back((uint32_t)i * 7u + 1000u);
        results.push_back(run_one(mesh.get(), cq, kdir,
            "valuesum (110 cores, distinct uint32 each)", n_iters,
            allc, tgt, false, vals));
    }

    printf("{\"results\":[");
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        printf("%s\n  {\"label\":\"%s\",\"n_iters\":%u,\"n_cores\":%u,"
               "\"mean_ns\":%.2f,\"median_ns\":%.2f,\"max_ns\":%.2f,"
               "\"expected\":%lu,\"actual\":%lu,\"ok\":%s}",
               i==0?"":",", r.label.c_str(), r.n_iters, r.n_cores,
               r.mean_ns, r.median_ns, r.max_ns,
               (unsigned long)r.expected_counter,
               (unsigned long)r.actual_counter,
               r.ok ? "true" : "false");
    }
    printf("\n]}\n");
    return 0;
}
