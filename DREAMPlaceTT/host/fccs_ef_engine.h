// SPDX-License-Identifier: Apache-2.0
//
// FCCS electric_force backward engine — persistent-buffer port of the validated
// FCCS field-cast / cell-stationary density backward (kernels/fccs_live_dm.cpp).
// Grafts into the V29 live harness: reads the forward's V31_GEOM per-cell geometry
// stash (set_geom_source) + the chip-DCT field, runs the field-cast backward
// (core 0 multicasts the field, workers own balanced contiguous cell slices, in-L1
// counting-sort by bxl, int-MAC compute, FLOAT grad in original order — no host
// unsort, no bucketing). Beats CPU backward 17/18 configs (≤540K cells); 1.1M needs
// streaming (geom slice exceeds L1) — not covered by this in-L1 variant.

#pragma once
#include <cstdint>
#include <memory>
#include <vector>

namespace fccsef {

struct FCCSTiming {
    double h2d_ms   = 0.0;
    double quant_ms = 0.0;   // parallel SFPU-style field quantize pre-pass
    double run_ms   = 0.0;   // field-cast multicast backward
    double d2h_ms   = 0.0;
    double total_ms = 0.0;
};

class FCCSEFEngine {
public:
    FCCSEFEngine(void* mesh_device, int M, int N, int num_nodes_max,
                 float xl, float yl, float bsx, float bsy);
    ~FCCSEFEngine();
    FCCSEFEngine(const FCCSEFEngine&) = delete;
    FCCSEFEngine& operator=(const FCCSEFEngine&) = delete;

    // sel-gathered per-cell constants + selection map (full pos length = 2*num_total).
    void configure_with_sel(const float* ox_full, const float* oy_full,
                            const float* nsx_full, const float* nsy_full,
                            const float* ratio_full,
                            const int32_t* sel, int num_active, int num_total_nodes);

    // Point at the forward's per-cell geometry buffer (V31_GEOM). MUST be set (addr!=0)
    // before the first compute_from_full. Records are 128 B, indexed by active cell.
    void set_geom_source(uint32_t geom_addr, uint32_t geom_pg);

    // One backward: fx/fy (M*N each, column-major [bxl][byl]) → grad_full (2*num_total;
    // only sel entries written, caller pre-zeros). pos unused (geometry from stash).
    void compute_from_full(const float* pos_full, const float* fx_full,
                           const float* fy_full, float* grad_full, FCCSTiming* t);

    // Chip-resident field (host-free): read the field directly from chip-DCT DRAM at
    // fx_addr/fy_addr (V19 latest_field_*_addr) — no host h2d. → grad_full[sel].
    void compute_from_chip(uint32_t fx_addr, uint32_t fy_addr, float* grad_full, FCCSTiming* t);

    // DEBUG: read the int32 quantized field buffers + raw int64 grad buffer.
    std::vector<int32_t> read_fxb();
    std::vector<int64_t> read_grad_raw();

    struct Impl;   // public so build_if_needed (member) can use it across the .cpp
private:
    void build_if_needed();
    std::unique_ptr<Impl> impl_;
};

}  // namespace fccsef
