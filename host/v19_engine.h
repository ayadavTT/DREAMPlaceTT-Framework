// SPDX-License-Identifier: Apache-2.0
//
// V19 Rung 3 — In-process engine class.
//
// Replaces the IPC/docker-exec path used by `density_scatter_ttnn_server`.
// DREAMPlace's Python imports v19_engine (built as a pybind11 module) and
// calls `V19Engine::scatter()` once per iteration directly in-process.
//
// Supported gather modes: {v11, v11outer, v18, v18outer, v19}.
// V13/V14 stay on the IPC server; this engine targets only the V11-family
// modes that share the same setup (server_host.cpp lines 900-1722).
//
// Lifecycle:
//   - Ctor: open mesh device, allocate ~20 MeshBuffers, JIT 3-4 workloads.
//     All env-var reads (V11_MAX_K, V11_HOT_THRESHOLD, V18_HASH_BITS,
//     V19_SCALE_BITS, GATHER_MODE) happen here.
//   - set_initial_density(): one-time copy of initial_density_map_normalized
//     into the engine's host buffer. Folded into density_flat before TTNN DCT
//     when CPU_DCT=0. (When CPU_DCT=1 the host adds it back after readback.)
//   - scatter(): one iteration. H2D positions, run scatter+gather+writeout,
//     D2H density. Returns density + field_x + field_y + per-stage timings.
//   - Dtor: releases mesh device.

#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace v19 {

// Per-iteration timing breakdown (filled by scatter()).
struct ScatterTiming {
    double h2d_ms          = 0.0;  // EnqueueWriteMeshBuffer for px/py/sx/sy
    double scatter_ms      = 0.0;  // wl_v11_scatter + Finish(cq)
    double gather_ms       = 0.0;  // wl_v11_accum + (V19 writeout) + Finish(cq)
    double d2h_density_ms  = 0.0;  // EnqueueReadMeshBuffer for density
    double ttnn_upload_ms  = 0.0;  // density → ttnn::Tensor (CPU_DCT=0 only)
    double ttnn_compute_ms = 0.0;  // TTNN DCT solve (CPU_DCT=0 only)
    double ttnn_download_ms= 0.0;  // field tensors → host (CPU_DCT=0 only)
    double fw_ms           = 0.0;  // misc post-processing
    double total_server_ms = 0.0;  // wall time for this scatter() call
    uint32_t gather_mode   = 0;    // 0=v6 1=v7 2=v8 3=v9 4=v10 5=v11
};

class V19Engine {
public:
    // Construct engine. Opens the TT mesh device, allocates all MeshBuffers
    // (sized by M, N, NC_max), and JITs the V11/V19 workloads. Reads env
    // vars (GATHER_MODE, V11_MAX_K, V11_HOT_THRESHOLD, V18_HASH_BITS,
    // V19_SCALE_BITS, etc.) once at construction.
    //
    // gather_mode: one of "v11", "v11outer", "v11outer_auto", "v18",
    // "v18outer", "v19". Unsupported values throw std::runtime_error.
    V19Engine(int M, int N, int NC_max,
              float xl, float yl, float xh, float yh,
              const std::string& gather_mode);

    ~V19Engine();

    V19Engine(const V19Engine&)            = delete;
    V19Engine& operator=(const V19Engine&) = delete;

    // One-time copy of initial_density_map_normalized into the engine.
    // Must be called before the first scatter() if CPU_DCT=0 (TTNN DCT
    // consumes it pre-DCT). Safe to skip if CPU_DCT=1 (the host adds it
    // back after density readout).
    //   id_normalized: pointer to M*N floats (density / bin_area).
    void set_initial_density(const float* id_normalized);

    // Run one scatter iteration.
    //   px, py, sx, sy: pointers to nc_actual floats (cell positions/sizes).
    //                   Caller is responsible for any pre-sort.
    //   nc_actual:      number of valid cells (<= NC_max).
    //   density_out:    pointer to M*N floats (caller-allocated, filled).
    //                   When CPU_DCT=1 this is the only meaningful output.
    //   field_x_out, field_y_out: pointers to M*N floats (caller-allocated).
    //                   When CPU_DCT=0 these are filled with the TTNN DCT
    //                   field result; when CPU_DCT=1 they are written zero
    //                   (the Python side runs the CPU DCT).
    //   timing_out: filled with per-stage timings.
    void scatter(const float* px, const float* py,
                 const float* sx, const float* sy,
                 int32_t nc_actual,
                 float* density_out,
                 float* field_x_out, float* field_y_out,
                 ScatterTiming* timing_out);

    // Accessors for grid metadata (handy for the pybind11 wrapper).
    int M() const noexcept;
    int N() const noexcept;
    int NC_max() const noexcept;
    const std::string& gather_mode_str() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace v19
