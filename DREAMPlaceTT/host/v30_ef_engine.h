// SPDX-License-Identifier: Apache-2.0
//
// V30 field-stationary EF backward engine (hybrid: host prep+route+grouping,
// chip field-stationary gather + fixed-point atomic grad sum + readout).
// Transpose of the V19 forward — see memory v30_field_stationary_backward.
// First cut prioritizes a real LIVE per-iter number + convergence; the host
// prep/route/h2d is the measured overhead to optimize on-chip next.

#pragma once
#include <cstdint>
#include <memory>

namespace v30ef {
struct V30Timing { double h2d_ms, prep_ms, gather_ms, d2h_ms, total_ms; };

class V30EFEngine {
public:
    V30EFEngine(void* mesh_device, int M, int N, int nmax,
                float xl, float yl, float bsx, float bsy);
    ~V30EFEngine();
    void configure_with_sel(const float* ox, const float* oy, const float* nsx,
        const float* nsy, const float* ratio, const int32_t* sel, int na, int nt);
    // pos (len 2*nt), fx/fy (M*N field), grad (len 2*nt, in-place +force into grad[sel]).
    void compute_from_full(const float* pos, const float* fx, const float* fy,
        float* grad, V30Timing* t);
private:
    struct Impl; std::unique_ptr<Impl> impl_;
};
}  // namespace v30ef
