// SPDX-License-Identifier: Apache-2.0
//
// V21 electric_force on-chip engine, sharing a MeshDevice with V19Engine.
// Used by V19EngineWrap to replace the CPU electric_force backward.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace v21ef {

struct EFTiming {
    double h2d_pos_ms       = 0.0;   // EnqueueWrite pos
    double h2d_field_ms     = 0.0;   // EnqueueWrite field_x + field_y
    double compute_ms       = 0.0;   // EnqueueMeshWorkload + Finish
    double d2h_grad_ms      = 0.0;   // EnqueueRead grad
    double total_ms         = 0.0;
    uint32_t launches       = 0;     // chunked launches (kernel limit ≤ 16 batches)
};

class V21EFEngine {
public:
    // Construct: takes a tt::tt_metal::distributed::MeshDevice* (opaque void*
    // for header decoupling). The caller owns the device — engine just uses it.
    // Allocates DRAM buffers sized for (num_nodes_max, M, N).
    V21EFEngine(void* mesh_device,
                int M, int N, int num_nodes_max,
                float xl, float yl, float bsx, float bsy);

    ~V21EFEngine();

    V21EFEngine(const V21EFEngine&)            = delete;
    V21EFEngine& operator=(const V21EFEngine&) = delete;

    // One-time upload of per-cell constants. Pack into the engine's const_buf.
    //   ox, oy, nsx, nsy, ratio: pointers to num_nodes floats each (packed,
    //                            not the full DREAMPlace layout).
    //   num_nodes: actual count (≤ num_nodes_max).
    void configure_constants(const float* ox, const float* oy,
                             const float* nsx, const float* nsy,
                             const float* ratio,
                             int num_nodes);

    // Like configure_constants() but also stores a selection mask so
    // compute_from_full() can read full-DREAMPlace-layout buffers and do the
    // sel gather/scatter in C++ (no Python masking per iter).
    //   ox_full..ratio_full: pointers to ratio_full_len floats (DREAMPlace layout).
    //   sel: indices into the full array, length num_active.
    //   num_total_nodes: full pos length is 2 * num_total_nodes.
    void configure_with_sel(const float* ox_full, const float* oy_full,
                            const float* nsx_full, const float* nsy_full,
                            const float* ratio_full,
                            const int32_t* sel, int num_active,
                            int num_total_nodes);

    // One iteration with packed pos / fx / fy arrays. Legacy API used by the
    // standalone microbench and the first integration. Prefer compute_from_full
    // when called from DREAMPlace to skip Python sel gather.
    void compute(const float* pos,
                 const float* field_x, const float* field_y,
                 float* grad_x, float* grad_y,
                 EFTiming* timing_out);

    // Full-DREAMPlace-layout compute: takes pos (length 2 * num_total_nodes),
    // fx/fy (M*N each), and writes grad_full (length 2 * num_total_nodes; only
    // entries at sel indices are written, others left untouched — caller
    // should pre-zero). The negation matches CPU electric_force semantics.
    // Requires configure_with_sel() to have been called.
    void compute_from_full(const float* pos_full,
                           const float* fx_full, const float* fy_full,
                           float* grad_full,
                           EFTiming* timing_out);

    // Zero-copy field variant: skip the field H2D and use TT chip DRAM
    // addresses (e.g. from V19Engine::latest_field_{x,y}_addr()) as the
    // V21 kernel's `field_x_base` / `field_y_base` runtime args. Saves the
    // ttnn_download + V21 EF h2d_field on every iter — important at large
    // grids (16 MB × 2 fields = ~12 ms/iter at a1_2048).
    //
    // The TT-side buffer layout must match V21's expectation: page_size
    // = N * 4 bytes, with the interleaved DRAM bank pattern (which is what
    // TTNN ROW_MAJOR layout uses by default). When in doubt, fall back to
    // compute_from_full() with host arrays.
    void compute_from_full_chip_fields(const float* pos_full,
                                        uint32_t fx_chip_addr,
                                        uint32_t fy_chip_addr,
                                        float* grad_full,
                                        EFTiming* timing_out);

    // Eagerly JIT-compile the V21 EF program and run a dummy launch. Called by
    // the constructor so the first real compute() call doesn't pay 9 s of
    // JIT cost mid-DREAMPlace-iteration.
    void prewarm();

    int num_nodes_configured() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace v21ef
