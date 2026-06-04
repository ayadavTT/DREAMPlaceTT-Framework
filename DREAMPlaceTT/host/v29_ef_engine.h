// SPDX-License-Identifier: Apache-2.0
//
// V29 electric_force on-chip backward engine — persistent-buffer port of the
// validated host/v29_host.cpp pipeline (prep+bucket → gather+compute → scatter).
// Shares a MeshDevice with V19Engine (opaque void*). Replaces the V21 EF
// backward in the live DREAMPlace loop: L1-resident field x-bin band + multi-bin
// SFPU gather, beating V21's NoC-per-read gather.

#pragma once
#include <cstdint>
#include <memory>
#include <vector>

namespace v29ef {

struct V29Timing {
    double h2d_ms      = 0.0;   // input + field upload
    double prep_ms     = 0.0;   // prep+bucket program
    double gather_ms   = 0.0;   // gather+compute+writer program
    double d2h_ms      = 0.0;   // grad readback + scatter
    double total_ms    = 0.0;
};

class V29EFEngine {
public:
    V29EFEngine(void* mesh_device, int M, int N, int num_nodes_max,
                float xl, float yl, float bsx, float bsy);
    ~V29EFEngine();
    V29EFEngine(const V29EFEngine&) = delete;
    V29EFEngine& operator=(const V29EFEngine&) = delete;

    // Store per-cell constants (sel-gathered from full DREAMPlace layout) + the
    // selection map. num_total_nodes: full pos length is 2*num_total_nodes.
    void configure_with_sel(const float* ox_full, const float* oy_full,
                            const float* nsx_full, const float* nsy_full,
                            const float* ratio_full,
                            const int32_t* sel, int num_active, int num_total_nodes);

    // PREP-REUSE: point P1 at the forward's per-cell geometry buffer (V31_GEOM)
    // so the backward buckets pre-computed overlaps instead of recomputing them in
    // soft-float. Must be set (with addr != 0) BEFORE the first compute_from_full so
    // the program is built with v29_bucket_only. addr=0 → legacy prepbucket path.
    void set_geom_source(uint32_t geom_addr, uint32_t geom_pg);

    // One backward: pos_full (2*num_total), fx/fy (M*N each, [num_bins_x][num_bins_y]
    // row-major), → grad_full (2*num_total; only sel entries written, caller pre-zeros).
    void compute_from_full(const float* pos_full, const float* fx_full,
                           const float* fy_full, float* grad_full, V29Timing* t);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace v29ef
