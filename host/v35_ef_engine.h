// SPDX-License-Identifier: Apache-2.0
//
// V35 EF backward engine — fully on-chip Forward-Grouped Halo-Tile density backward.
// Per iteration (no host per-cell work): COUNT (per-core tile histogram from the
// forward V31_GEOM stash) → host PLAN (tile_base + srcprefix + greedy proportional
// core allocation; O(nc·ntiles) metadata) → PLACE (local-counting-sort + bulk
// writes into one O(ncells) tile-grouped buffer) → GATHER (V28 SFPU multi-bin,
// contiguous tile slices, halo'd L1-resident field band). Grad emitted per worker +
// oidx; host maps grad_full[sel[oidx]] (the only residual host step, like V29).
// Validated bit-exact (rel_l2 7e-8) standalone; clustering-immune; no route_buf/OOM.

#pragma once
#include <cstdint>
#include <memory>
#include <vector>

namespace v35ef {

struct V35Timing {
    double count_ms  = 0.0;
    double plan_ms   = 0.0;   // host: read counts + prefix + alloc + h2d plan + set args
    double place_ms  = 0.0;
    double gather_ms = 0.0;
    double d2h_ms    = 0.0;   // grad+oidx readback + host sel-unsort (residual)
    double total_ms  = 0.0;
};

class V35EFEngine {
public:
    V35EFEngine(void* mesh_device, int M, int N, int num_nodes_max,
                float xl, float yl, float bsx, float bsy);
    ~V35EFEngine();
    V35EFEngine(const V35EFEngine&) = delete;
    V35EFEngine& operator=(const V35EFEngine&) = delete;

    void configure_with_sel(const float* ox_full, const float* oy_full,
                            const float* nsx_full, const float* nsy_full,
                            const float* ratio_full,
                            const int32_t* sel, int num_active, int num_total_nodes);

    // Point at the forward's per-cell geometry stash (V31_GEOM, 128 B/cell, active order).
    void set_geom_source(uint32_t geom_addr, uint32_t geom_pg);

    // One backward from the chip-DCT field in DRAM (host-free field): fx/fy column-major
    // [bxl*NBY + byl]. Writes -grad into grad_full[sel[i]] (caller pre-zeros). pos unused.
    void compute_from_chip(uint32_t fx_addr, uint32_t fy_addr, float* grad_full, V35Timing* t);

    struct Impl;
private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace v35ef
