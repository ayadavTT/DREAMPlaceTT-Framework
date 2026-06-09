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
    // Scalars are per-iter:
    //   step_size     : the Nesterov step (DreamPlace step_bb's BB stepsize).
    //   coef          : Nesterov momentum coef (a_k-1)/a_kp1.
    //   density_weight: the combine weight  g = wl + density_weight*density.
    //   precond_alpha : the PRECONDITIONER alpha (PlaceObj.alpha; =1.0 unless a
    //                   fence-region schedule doubles it). The precond divisor is
    //                   max(pin_w + precond_alpha*density_weight*area, 1). This is
    //                   DISTINCT from step_size (step_bb conflates neither).
    void step(const float* wl_grad, const float* density_grad,
              const float* v_k, const float* u_prev,
              float step_size, float coef, float density_weight, float precond_alpha,
              float* u_new, float* v_new,
              OptTiming* timing_out);

    // ── Phase B: chip-resident pos/u ──
    // set_pos uploads the initial reference solution into the resident pos
    // buffer (b_v) and seeds u_k = v_k (matches torch u_k = p.data.clone()).
    void set_pos(const float* v_init);
    // One resident step: combine→precond→nesterov→clamp updates the resident
    // pos (b_v) and u (b_uprev) IN PLACE — no v_k/u_prev upload, no output
    // download. Only the two gradient halves + scalars are uploaded.
    // (see step() for the scalar contract; step_size vs precond_alpha are distinct.)
    void step_resident(const float* wl_grad, const float* density_grad,
                       float step_size, float coef, float density_weight, float precond_alpha,
                       OptTiming* timing_out);
    // Download the resident pos (for the host wirelength / HPWL eval).
    void get_pos(float* out);

    // ── On-device density_grad (optimizer fusion with the on-chip unsort) ──
    // DRAM address + layout of the density_grad buffer (b_dg). The on-chip unsort
    // writes density_grad here directly (flat 2*num_nodes, x|y, tile-interleaved,
    // page = one 32x32 fp32 tile), so the host grad upload is eliminated.
    uint32_t density_grad_address() const;     // b_dg DRAM base address
    uint32_t density_grad_num_tiles() const;    // n_tiles_total (page count, page = TILE_BYTES)
    // Resident step that reads density_grad already-present in b_dg (NO host upload of it).
    // Only wl_grad + the per-iter scalars are uploaded. pos/u stay resident & update in place.
    // (see step() for the scalar contract; step_size vs precond_alpha are distinct.)
    void step_resident_ondevice_grad(const float* wl_grad,
                                     float step_size, float coef, float density_weight, float precond_alpha,
                                     OptTiming* timing_out);

    int num_nodes() const noexcept;

    // ── Step B: on-device step-size reduction ──
    // Compute the three global reduction sums over s, y (each length 2*num_nodes,
    // host layout [x|y]): ss = Σ s·s, sy = Σ s·y, yy = Σ y·y. The per-core
    // lane-wise reduction runs on device (v22_stepsize kernel → nc*3 partial
    // tiles); the host sums the (small) partials. The Nesterov line-search step
    // is alpha = sqrt(ss/yy); the BB short step is sy/yy. s,y are uploaded here;
    // when pos/grad are resident this becomes a no-d2h reduction over them.
    void stepsize_sums(const float* s, const float* y,
                       double* ss, double* sy, double* yy);

    // ── No-host: on-device combine + precond reading the resident b_dg ──
    // Runs ONLY the combine + precond passes (not nesterov/clamp), consuming the
    // density gradient already present in b_dg (written by the on-chip unsort), and
    // leaving the preconditioned combined gradient g = precond(wl_grad + (-dw)*b_dg)
    // in the internal b_g2 buffer. The sign is -density_weight because b_dg holds
    // +force while DreamPlace's density gradient is -force*density_weight. Only
    // wl_grad (the allowed h2d) + the scalars are uploaded.
    void combine_precond_ondevice(const float* wl_grad, float density_weight, float precond_alpha);
    // Download the preconditioned combined gradient (b_g2) — for validating that the
    // on-device combine+precond reproduces DreamPlace's host g_k.
    void get_precond_grad(float* out);

    // ── No-host line-search loop (resident pos/u + on-device step size) ──
    // Pos resident in b_v, u in b_uprev (seed once via set_pos). The preconditioned
    // gradient lives in b_g2 (filled by combine_precond_ondevice). Per optimizer step:
    //   combine_precond_ondevice(raw_wl@v_k) -> b_g2=g_k ; snapshot_precond_grad() -> b_g_k
    //   line search: nesterov_clamp_trial(step,coef) -> b_vclamp(v_kp1)/b_unew(u_kp1),
    //   leaving v_k(b_v)/u_k(b_uprev) intact for backtracking ; [host forward+backward
    //   @ v_kp1 writes b_dg] ; combine_precond_ondevice(raw_wl@v_kp1) -> b_g2=g_kp1 ;
    //   linesearch_sums() -> (Σ(v_kp1-v_k)², Σ(g_kp1-g_k)²) for α=sqrt(ss/yy) ;
    //   accept/retry ; commit_trial() -> b_v<-b_vclamp, b_uprev<-b_unew.
    // All reuse the combine/nesterov/clamp/stepsize kernels (combine doubles as a
    // scalar*-add for the diffs and a copy for snapshot/commit).
    void snapshot_precond_grad();                          // b_g_k <- b_g2 (g_k)
    void nesterov_clamp_trial(float step_size, float coef);// -> b_vclamp/b_unew (v_k,u_k kept)
    void linesearch_sums(double* ss, double* yy);          // Σ(b_vclamp-b_v)², Σ(b_g2-b_g_k)²
    void commit_trial();                                   // b_v<-b_vclamp, b_uprev<-b_unew
    void get_trial_pos(float* out);                        // download b_vclamp (v_kp1)

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace v22
