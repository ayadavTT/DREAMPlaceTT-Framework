// SPDX-License-Identifier: Apache-2.0
//
// V22 optimizer-on-chip engine. Shares a MeshDevice with V19Engine and runs the
// ePlace Nesterov optimizer step entirely on chip by chaining the four validated
// element-wise SFPU kernels through persistent DRAM buffers:
//
//   combine   (K3): g  = wl_grad + density_weight * density_grad
//   precond   (K4): g  = g / max(pin_w + alpha*density_weight*area, 1)
//   nesterov  (K1): u' = v - alpha*g ;  v' = u' + coef*(u' - u_prev)
//   clamp     (K2): v' = min(max(v', lo), hi)
//
// Layout contract: everything is in DreamPlace optimizer order [movable|fixed|
// filler], pos vector length 2*num_nodes (x slice then y slice). Constants
// (lo/hi/pin_w/area) are uploaded once; per-iter scalars (alpha/coef/dw/adw)
// are broadcast into 1-tile buffers each call.
//
// Phase A (this engine): inputs uploaded + outputs downloaded each call so the
// chip step can be validated bit-against the Python Nesterov. Phase B will keep
// pos / u_prev resident and stop the per-iter round trip.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace v22 {

struct OptTiming {
    double h2d_ms     = 0.0;
    double compute_ms = 0.0;   // 4 chained launches + Finish
    double d2h_ms     = 0.0;
    double total_ms   = 0.0;
};

class V22OptEngine {
public:
    // mesh_device: tt::tt_metal::distributed::MeshDevice* (opaque). Caller owns it.
    // num_nodes: DreamPlace num_nodes (pos length is 2*num_nodes).
    V22OptEngine(void* mesh_device, int num_nodes);
    ~V22OptEngine();

    V22OptEngine(const V22OptEngine&)            = delete;
    V22OptEngine& operator=(const V22OptEngine&) = delete;

    // One-time upload of per-element constants, each length 2*num_nodes (fp32),
    // in optimizer order with x slice then y slice:
    //   lo / hi    : clamp bounds (xl/yl and xh-nsx / yh-nsy; fixed nodes use
    //                lo=-BIG, hi=+BIG so the clamp is a no-op).
    //   pin_w      : sum_pin_weights per node, duplicated into both halves.
    //   area       : node_areas per node, duplicated into both halves.
    void configure(const float* lo, const float* hi,
                   const float* pin_w, const float* area);

    // Build + JIT-warm the four programs so the first real step isn't slow.
    void prewarm();

    // One full optimizer step. All array args length 2*num_nodes (fp32):
    //   wl_grad, density_grad : the two gradient halves to combine.
    //   v_k, u_prev           : current reference solution + previous u.
    // Outputs (length 2*num_nodes): u_new (next u_prev), v_new (next pos).
    // Scalars are per-iter.
    void step(const float* wl_grad, const float* density_grad,
              const float* v_k, const float* u_prev,
              float alpha, float coef, float density_weight,
              float* u_new, float* v_new,
              OptTiming* timing_out);

    // ── Phase B: chip-resident pos/u ──
    // set_pos uploads the initial reference solution into the resident pos
    // buffer (b_v) and seeds u_k = v_k (matches torch u_k = p.data.clone()).
    void set_pos(const float* v_init);
    // One resident step: combine→precond→nesterov→clamp updates the resident
    // pos (b_v) and u (b_uprev) IN PLACE — no v_k/u_prev upload, no output
    // download. Only the two gradient halves + scalars are uploaded.
    void step_resident(const float* wl_grad, const float* density_grad,
                       float alpha, float coef, float density_weight,
                       OptTiming* timing_out);
    // Download the resident pos (for the host wirelength / HPWL eval).
    void get_pos(float* out);

    int num_nodes() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace v22
