// SPDX-License-Identifier: Apache-2.0
//
// V19 Rung 3 — pybind11 bindings for V19Engine.
//
// The exposed Python class is `V19Engine` (implemented by `V19EngineWrap`
// here). The wrap class **owns the output numpy buffers and reuses them
// across scatter() calls**, eliminating ~16 MB of fresh allocation and
// 2× M·N memcpy overhead per iter on 2048-grid configs.
//
// Safety: DREAMPlace's pipeline consumes each iter's outputs (via
// `density_map = field_x.to(...)` → DCT → reassignment to a fresh tensor)
// *before* the next forward calls scatter() again. So the returned numpy
// arrays can safely be views of reused buffers; the next iter's overwrite
// happens after the previous iter's downstream torch ops are done.
//
// CPU_DCT=1 optimizations:
//   - density_out and field_x_out point to the SAME buffer → engine skips
//     the duplicate density→fx memcpy.
//   - field_y_out is pre-zeroed once at ctor; engine receives nullptr and
//     skips the per-iter memset.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "v19_engine.h"
#include "v21_ef_engine.h"
#include "v22_engine.h"
#include "v29_ef_engine.h"
#include "fccs_ef_engine.h"
#include "v35_ef_engine.h"
#include "v30_ef_engine.h"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>

namespace py = pybind11;
using namespace pybind11::literals;
using v19::V19Engine;
using v19::ScatterTiming;
using v21ef::V21EFEngine;
using v21ef::EFTiming;

// Build the per-iter timing dict in the same shape as the IPC client
// (scatter_ttnn_client.py:511-526). IPC-bucket keys are 0.0 — that's
// exactly the point of Rung 3.
static py::dict make_timing_dict(const ScatterTiming& t) {
    static const char* kModeNames[] = {"v6", "v7", "v8", "v9", "v10", "v11"};
    const char* gm = (t.gather_mode <= 5u) ? kModeNames[t.gather_mode] : "v6";
    return py::dict(
        "pos_write_ms"_a    = 0.0,
        "kernel_wait_ms"_a  = 0.0,
        "field_read_ms"_a   = 0.0,
        "h2d_ms"_a          = t.h2d_ms,
        "scatter_ms"_a      = t.scatter_ms,
        "gather_ms"_a       = t.gather_ms,
        "d2h_density_ms"_a  = t.d2h_density_ms,
        "ttnn_upload_ms"_a  = t.ttnn_upload_ms,
        "ttnn_compute_ms"_a = t.ttnn_compute_ms,
        "ttnn_download_ms"_a= t.ttnn_download_ms,
        "fw_ms"_a           = t.fw_ms,
        "total_server_ms"_a = t.total_server_ms,
        "total_client_ms"_a = 0.0,
        "gather_mode"_a     = std::string(gm)
    );
}

// Wrap class owning the engine + a pre-allocated zero buffer for field_y.
//
// Density and (in CPU_DCT=0 mode) field_x are allocated FRESH each call.
// Empirically, reusing the density buffer hurts CPU_DCT performance because
// the resident 16 MB buffer competes with the DCT working set in L2/L3.
// Fresh allocs avoid that cache pressure but cost a malloc + first-touch
// page faults — typically much cheaper than the resulting DCT slowdown.
//
// We still get the headline savings via two engine-side optimizations:
//   1. When CPU_DCT=1, we pass the SAME pointer for density_out and
//      field_x_out → engine sees aliasing and skips the duplicate memcpy.
//   2. fy is a single pre-zeroed buffer reused across all iters → engine
//      receives nullptr and skips the per-iter memset.
class V19EngineWrap {
public:
    V19EngineWrap(int M, int N, int NC_max,
                  float xl, float yl, float xh, float yh,
                  const std::string& gather_mode)
        : eng_(M, N, NC_max, xl, yl, xh, yh, gather_mode) {

        const char* cpu_dct_env = std::getenv("CPU_DCT");
        cpu_dct_ = (cpu_dct_env && std::string(cpu_dct_env) == "1");

        const py::ssize_t MM = (py::ssize_t)eng_.M();
        const py::ssize_t NN = (py::ssize_t)eng_.N();
        const size_t mn = (size_t)MM * (size_t)NN;

        // fy is constant zero in CPU_DCT=1 mode; allocate + zero once and
        // return the same view every iter. (DCT consumes density not fy,
        // so fy resident cost doesn't compete with DCT working set.)
        fy_zero_arr_ = py::array_t<float>({MM, NN});
        std::memset(fy_zero_arr_.mutable_data(), 0, mn * sizeof(float));

        std::printf("[v19_engine] V19EngineWrap init: CPU_DCT=%d  "
                    "(fresh density+fx per iter, fy pre-zeroed once, "
                    "duplicate memcpy via alias-detect; %.1f MB/buf)\n",
                    cpu_dct_ ? 1 : 0,
                    (double)(mn * sizeof(float)) / (1024.0 * 1024.0));
        std::fflush(stdout);
    }

    void set_initial_density(
        py::array_t<float, py::array::c_style | py::array::forcecast> id_arr) {
        const size_t mn = (size_t)eng_.M() * (size_t)eng_.N();
        if ((size_t)id_arr.size() < mn) {
            throw std::invalid_argument(
                "initial_density: expected at least " + std::to_string(mn) +
                " floats, got " + std::to_string(id_arr.size()));
        }
        eng_.set_initial_density(id_arr.data());
    }

    py::tuple scatter(
        py::array_t<float, py::array::c_style | py::array::forcecast> px,
        py::array_t<float, py::array::c_style | py::array::forcecast> py_arr,
        py::array_t<float, py::array::c_style | py::array::forcecast> sx,
        py::array_t<float, py::array::c_style | py::array::forcecast> sy,
        int32_t nc_actual) {

        if (nc_actual < 0 || nc_actual > eng_.NC_max()) {
            throw std::invalid_argument(
                "scatter: nc_actual=" + std::to_string(nc_actual) +
                " out of [0, " + std::to_string(eng_.NC_max()) + "]");
        }
        const size_t need = (size_t)nc_actual;
        if ((size_t)px.size()     < need || (size_t)py_arr.size() < need ||
            (size_t)sx.size()     < need || (size_t)sy.size()     < need) {
            throw std::invalid_argument("scatter: position array smaller than nc_actual");
        }

        // Fresh density (and fx for CPU_DCT=0) per iter to avoid the
        // resident-buffer cache pressure that hurt DCT in the V2 design.
        const py::ssize_t MM = (py::ssize_t)eng_.M();
        const py::ssize_t NN = (py::ssize_t)eng_.N();
        auto density = py::array_t<float>({MM, NN});

        // Output pointers. CPU_DCT=1: alias fx to density so the engine
        // skips the duplicate memcpy. fy is the pre-zeroed reused buffer
        // (engine receives nullptr → no memset). CPU_DCT=0: fresh fx
        // (TTNN writes); fy is fresh (TTNN writes).
        float* dp  = density.mutable_data();
        py::array_t<float> fx_local;
        float* fxp;
        float* fyp;
        if (cpu_dct_) {
            fxp = dp;                                 // alias-detect → skip dup memcpy
            fyp = nullptr;                            // skip per-iter memset
        } else {
            fx_local = py::array_t<float>({MM, NN});
            // For CPU_DCT=0 we need a fresh fy too (TTNN writes the field).
            // Re-zero our reused fy_zero_arr_'s buffer is not enough — engine
            // will overwrite. Allocate fresh for CPU_DCT=0 path.
            fxp = fx_local.mutable_data();
            // Fall through: we still need a separate fy buffer.
        }
        py::array_t<float> fy_local;
        if (!cpu_dct_) {
            fy_local = py::array_t<float>({MM, NN});
            fyp = fy_local.mutable_data();
        }

        ScatterTiming t;
        {
            py::gil_scoped_release rel;
            eng_.scatter(px.data(), py_arr.data(), sx.data(), sy.data(),
                         nc_actual, dp, fxp, fyp, &t);
        }

        // Return: density is fresh, fx is density-alias (CPU_DCT=1) or
        // fresh fx_local (CPU_DCT=0), fy is the reused zero buffer
        // (CPU_DCT=1) or fresh fy_local (CPU_DCT=0).
        return py::make_tuple(
            density,
            cpu_dct_ ? density   : fx_local,
            cpu_dct_ ? fy_zero_arr_ : fy_local,
            make_timing_dict(t)
        );
    }

    int M()       const { return eng_.M();      }
    int N()       const { return eng_.N();      }
    int NC_max()  const { return eng_.NC_max(); }
    std::string gather_mode_str() const { return eng_.gather_mode_str(); }

    // ── Fast path: do ALL per-iter prep work in C++ ─────────────────────
    //
    // Replaces the Python patch hook's per-iter chain:
    //   prep_cells (4× torch→numpy + 4× np.concatenate)
    //   + subcell apply (4× np.concatenate + 4× astype)
    //   + cell sort (2× np.ascontiguousarray(px[idx]))
    // with one fused C++ loop:
    //   for each output slot i:
    //     send_px[i] = pos[src_idx_x[i]] + offset_x[i]
    //
    // configure_cells() is called ONCE per design. It precomputes:
    //   - The big-cell sub-tiling layout
    //   - The area-based cell-sort permutation
    //   - A FUSED final-slot → (pos index, offset, send size) mapping
    //   so per-iter we touch each output slot exactly once.
    //
    // scatter_from_pos() takes the per-iter `pos` numpy view and runs the
    // fused loop, then dispatches to the engine.

    void configure_cells(
        py::array_t<float, py::array::c_style | py::array::forcecast> orig_sx_all,
        py::array_t<float, py::array::c_style | py::array::forcecast> orig_sy_all,
        int n_movable, int n_filler, int num_nodes,
        float bin_size_x, float bin_size_y,
        py::array_t<float, py::array::c_style | py::array::forcecast> offx_all = py::array_t<float>(),
        py::array_t<float, py::array::c_style | py::array::forcecast> offy_all = py::array_t<float>(),
        py::array_t<float, py::array::c_style | py::array::forcecast> ratio_all = py::array_t<float>()) {

        if ((py::ssize_t)orig_sx_all.size() < num_nodes ||
            (py::ssize_t)orig_sy_all.size() < num_nodes) {
            throw std::invalid_argument(
                "configure_cells: orig_sx_all/sy_all must have num_nodes entries");
        }
        const float* osx = orig_sx_all.data();
        const float* osy = orig_sy_all.data();

        // V31_EF_GEOM: retire the [[project_v13]] orig-size workaround. The forward
        // scattered orig-size-at-pos (orig=clamped+2offset) so the DENSITY (center+area)
        // matched CPU, but the electric FORCE (field-weighted, footprint-sensitive) did NOT.
        // When set, use the REAL clamped size at (pos+offset) — matches CPU's
        // triangle_density_function exactly → correct EF stash (backward applies ·ratio).
        const bool ef_geom = (std::getenv("V31_EF_GEOM") && std::atoi(std::getenv("V31_EF_GEOM")));
        const bool have_off = ef_geom && (py::ssize_t)offx_all.size() >= num_nodes
                                       && (py::ssize_t)offy_all.size() >= num_nodes;
        const float* aox = have_off ? offx_all.data() : nullptr;
        const float* aoy = have_off ? offy_all.data() : nullptr;
        const bool have_ratio = ef_geom && (py::ssize_t)ratio_all.size() >= num_nodes;
        const float* arat = have_ratio ? ratio_all.data() : nullptr;

        // ── Extract sx/sy (+ per-node offset) for movable+filler ──
        const int nc = n_movable + n_filler;
        std::vector<float> sx_full(nc), sy_full(nc);
        offx_nc_.assign(nc, 0.0f); offy_nc_.assign(nc, 0.0f);   // dreamplace offset, nc-space (0 unless EF-geom)
        auto gather=[&](float* dst, const float* src){
            std::memcpy(dst, src, n_movable*sizeof(float));
            std::memcpy(dst+n_movable, src+(num_nodes-n_filler), n_filler*sizeof(float)); };
        gather(sx_full.data(), osx); gather(sy_full.data(), osy);
        std::vector<float> ratio_nc(nc, 1.0f);
        if (have_off) {
            gather(offx_nc_.data(), aox); gather(offy_nc_.data(), aoy);
            // clamped = orig - 2*offset (offset = (orig-clamped)/2)
            for (int i=0;i<nc;++i){ sx_full[i]-=2.0f*offx_nc_[i]; sy_full[i]-=2.0f*offy_nc_[i]; }
        }
        if (have_ratio) gather(ratio_nc.data(), arat);

        // ── Big-cell sub-tiling layout ──
        // V11/V19 scatter kernels walk at most 8 bins per cell. A cell that
        // would span >8 bins is split into a grid of sub-cells, each ≤7×bin
        // wide (so even mid-bin starts cover ≤8 bins).
        constexpr int V11_SAFE = 7;
        std::vector<int32_t> nx(nc), ny(nc);
        std::vector<int32_t> small_pre, big_pre;  // indices into [0, nc)
        for (int i = 0; i < nc; ++i) {
            nx[i] = std::max(1, (int)std::ceil((double)sx_full[i] / (V11_SAFE * bin_size_x)));
            ny[i] = std::max(1, (int)std::ceil((double)sy_full[i] / (V11_SAFE * bin_size_y)));
            if (nx[i] > 1 || ny[i] > 1) big_pre.push_back((int32_t)i);
            else                        small_pre.push_back((int32_t)i);
        }
        const int n_small = (int)small_pre.size();
        int n_sub_total = 0;
        for (int p : big_pre) n_sub_total += nx[p] * ny[p];
        const int new_nc = n_small + n_sub_total;

        // ── Build pre-sort layout (small cells first, then big sub-cells). ──
        // For each pre-sort slot store: parent_cell (0..nc-1), i and j sub-cell
        // indices, send sx/sy. We'll use these to derive the FUSED layout
        // after we know the sort permutation.
        std::vector<int32_t> ps_parent(new_nc);
        std::vector<int32_t> ps_iflat(new_nc), ps_jflat(new_nc);
        std::vector<float>   ps_sx(new_nc), ps_sy(new_nc);
        for (int i = 0; i < n_small; ++i) {
            int p = small_pre[i];
            ps_parent[i] = p;
            ps_iflat[i] = 0; ps_jflat[i] = 0;
            ps_sx[i] = sx_full[p];
            ps_sy[i] = sy_full[p];
        }
        int cur = n_small;
        for (int p : big_pre) {
            int kx = nx[p], ky = ny[p];
            float dx = sx_full[p] / (float)kx;
            float dy = sy_full[p] / (float)ky;
            for (int i = 0; i < kx; ++i) {
                for (int j = 0; j < ky; ++j) {
                    ps_parent[cur] = p;
                    ps_iflat[cur] = i;
                    ps_jflat[cur] = j;
                    ps_sx[cur] = dx;
                    ps_sy[cur] = dy;
                    ++cur;
                }
            }
        }

        // ── Cell sort by descending area + V11 tile-balancing permutation. ──
        // Largest cells → lowest tile indices. Mirrors scatter_ttnn_client.py:436-467.
        //
        // V11 hot-tile sharding has been off-by-default since memory note
        // v11_gather_sharding_investigation.md (HOT_THRESHOLD=100M). The sort
        // is mostly inertia at this point — but it scrambles the per-iter
        // fancy-index reads from `pos`, making the fused loop random-access
        // (25 ms/iter on bigblue3 with 2 M cells). Setting `V19_SKIP_CELL_SORT=1`
        // uses an identity permutation: reads become near-sequential and the
        // fused loop drops to memory-write speed (~1-2 ms). Validate HPWL parity
        // before relying on this.
        const bool skip_sort = std::getenv("V19_SKIP_CELL_SORT")
                            && std::string(std::getenv("V19_SKIP_CELL_SORT")) == "1";
        std::vector<int32_t> sort_idx(new_nc);
        if (skip_sort) {
            std::iota(sort_idx.begin(), sort_idx.end(), 0);
            std::printf("[v19_engine] configure_cells: SKIP cell-sort (identity perm)\n");
            std::fflush(stdout);
        } else {
            std::vector<int32_t> area_rank(new_nc);
            std::iota(area_rank.begin(), area_rank.end(), 0);
            std::sort(area_rank.begin(), area_rank.end(),
                      [&](int32_t a, int32_t b) {
                          return ps_sx[a] * ps_sy[a] > ps_sx[b] * ps_sy[b];
                      });
            const int n_tiles = (new_nc + 1023) / 1024;
            std::fill(sort_idx.begin(), sort_idx.end(), area_rank[0]);
            std::vector<int32_t> invalid_ranks;
            std::vector<uint8_t> dest_filled(new_nc, 0);
            for (int r = 0; r < new_nc; ++r) {
                int d = (r % n_tiles) * 1024 + (r / n_tiles);
                if (d < new_nc) {
                    sort_idx[d] = area_rank[r];
                    dest_filled[d] = 1;
                } else {
                    invalid_ranks.push_back(r);
                }
            }
            std::vector<int32_t> unfilled;
            for (int i = 0; i < new_nc; ++i) if (!dest_filled[i]) unfilled.push_back(i);
            for (size_t i = 0; i < invalid_ranks.size() && i < unfilled.size(); ++i) {
                sort_idx[unfilled[i]] = area_rank[invalid_ranks[i]];
            }
        }

        // ── FUSED per-iter layout. ──
        // For each FINAL output slot i (post-sort, post-subcell):
        //   src_idx_x_[i] = index into pos[] for x of the parent cell
        //   src_idx_y_[i] = index into pos[] for y of the parent cell
        //   offset_x_[i]  = sub-cell offset to add (0 for small cells)
        //   offset_y_[i]  = ditto for y
        //   send_sx_[i]   = cell size to send (constant)
        //   send_sy_[i]   = ditto
        src_idx_x_.assign(new_nc, 0);
        src_idx_y_.assign(new_nc, 0);
        offset_x_.assign(new_nc, 0.0f);
        offset_y_.assign(new_nc, 0.0f);
        send_sx_arr_ = py::array_t<float>(new_nc);
        send_sy_arr_ = py::array_t<float>(new_nc);
        float* ssx = send_sx_arr_.mutable_data();
        float* ssy = send_sy_arr_.mutable_data();
        std::vector<float> send_ratio(new_nc, 1.0f);   // per-slot ratio (scatter order) for density ·ratio
        subcell_parent_.assign(new_nc, 0);   // per-(sub)cell parent active-cell index (0..nc-1)
        for (int i = 0; i < new_nc; ++i) {
            int pre = sort_idx[i];
            int parent = ps_parent[pre];  // 0..nc-1 (movable+filler space)
            subcell_parent_[i] = parent;  // FCCS: map each sub-cell's grad back to its parent
            send_ratio[i] = ratio_nc[parent];
            int pos_xi, pos_yi;
            if (parent < n_movable) {
                pos_xi = parent;
                pos_yi = num_nodes + parent;
            } else {
                int fidx = parent - n_movable;
                pos_xi = num_nodes - n_filler + fidx;
                pos_yi = 2 * num_nodes - n_filler + fidx;
            }
            src_idx_x_[i] = pos_xi;
            src_idx_y_[i] = pos_yi;
            // node_x = pos + dreamplace_offset(parent) + sub-cell offset. The dreamplace
            // offset is 0 unless V31_EF_GEOM (then sizes are clamped, so this puts the cell
            // at the CPU electric-force footprint clamped-at-(pos+offset)).
            offset_x_[i] = offx_nc_[parent] + (float)ps_iflat[pre] * ps_sx[pre];
            offset_y_[i] = offy_nc_[parent] + (float)ps_jflat[pre] * ps_sy[pre];
            ssx[i] = ps_sx[pre];
            ssy[i] = ps_sy[pre];
        }

        new_nc_ = new_nc;
        if (have_ratio) eng_.set_ef_ratio(send_ratio.data(), new_nc);   // V31_EF_GEOM: density ·ratio
        // Pre-allocate per-iter send buffers.
        send_px_arr_ = py::array_t<float>(new_nc);
        send_py_arr_ = py::array_t<float>(new_nc);
        n_movable_ = n_movable;
        n_filler_  = n_filler;
        num_nodes_ = num_nodes;
        configured_ = true;

        std::printf("[v19_engine] configure_cells: nc=%d  big=%zu  new_nc=%d (Δ=%+d, +%.1f%%)\n",
                    nc, big_pre.size(), new_nc, new_nc - nc,
                    (double)(new_nc - nc) * 100.0 / std::max(1, nc));
        std::fflush(stdout);
    }

    int new_nc() const { return new_nc_; }

    py::tuple scatter_from_pos(
        py::array_t<float, py::array::c_style | py::array::forcecast> pos_np) {

        if (!configured_) {
            throw std::runtime_error("scatter_from_pos: call configure_cells() first");
        }
        if ((py::ssize_t)pos_np.size() < 2 * num_nodes_) {
            throw std::invalid_argument(
                "scatter_from_pos: pos must have 2*num_nodes floats");
        }
        const float* pos = pos_np.data();

        // Allocate output buffers under the GIL (numpy alloc isn't safe without it).
        const py::ssize_t MM = (py::ssize_t)eng_.M();
        const py::ssize_t NN = (py::ssize_t)eng_.N();
        auto density = py::array_t<float>({MM, NN});
        py::array_t<float> fx_local, fy_local;
        if (!cpu_dct_) {
            fx_local = py::array_t<float>({MM, NN});
            fy_local = py::array_t<float>({MM, NN});
        }

        float* dp  = density.mutable_data();
        float* fxp = cpu_dct_ ? dp : fx_local.mutable_data();
        float* fyp = cpu_dct_ ? nullptr : fy_local.mutable_data();
        float* px_send = send_px_arr_.mutable_data();
        float* py_send = send_py_arr_.mutable_data();
        const int new_nc_local = new_nc_;
        const int32_t* ix = src_idx_x_.data();
        const int32_t* iy = src_idx_y_.data();
        const float*   ox = offset_x_.data();
        const float*   oy = offset_y_.data();

        ScatterTiming t;
        double fused_loop_ms = 0.0;
        {
            py::gil_scoped_release rel;
            // Fused per-iter loop: extract + apply sub-cell + permute by sort
            // in ONE pass over the output slots. Each slot is one fancy-index
            // read from pos and one add. No intermediate numpy temporaries.
            //
            // Memory-bound: at bigblue3 scale (2 M output slots, 8 MB pos)
            // single-threaded this is ~26 ms. OpenMP-parallelized across
            // 16 threads it should drop to a few ms.
            auto _t0 = std::chrono::high_resolution_clock::now();
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < new_nc_local; ++i) {
                px_send[i] = pos[ix[i]] + ox[i];
                py_send[i] = pos[iy[i]] + oy[i];
            }
            auto _t1 = std::chrono::high_resolution_clock::now();
            fused_loop_ms = std::chrono::duration<double, std::milli>(_t1 - _t0).count();

            eng_.scatter(px_send, py_send,
                         send_sx_arr_.data(), send_sy_arr_.data(),
                         new_nc_local,
                         dp, fxp, fyp, &t);
        }
        // Stash fused-loop time into the timing dict for visibility.
        t.fw_ms = t.fw_ms;  // (unchanged); we add a new key in make_timing_dict below.
        // Use fw_ms as a placeholder; we add `fused_loop_ms` via a side channel:
        last_fused_loop_ms_ = fused_loop_ms;

        py::dict td = make_timing_dict(t);
        td["fused_loop_ms"] = last_fused_loop_ms_;
        return py::make_tuple(
            density,
            cpu_dct_ ? density   : fx_local,
            cpu_dct_ ? fy_zero_arr_ : fy_local,
            td
        );
    }

    // ── V21 electric_force on-chip integration. ──
    //
    // configure_electric_force(ox, oy, nsx_clamped, nsy_clamped, ratio):
    //   one-time upload of per-cell constants. Lazily creates the V21EFEngine
    //   using V19's mesh device (shared chip session).
    //
    // compute_electric_force(pos, field_x, field_y):
    //   per-iter call. Uploads pos + field_x + field_y, runs V21 EF program,
    //   returns (grad_x, grad_y, timing_dict) with grad arrays each of length
    //   num_nodes. The grad is negated (-gx, -gy) — matches CPU
    //   electric_potential_cpp.electric_force output semantics.
    void configure_electric_force(
        py::array_t<float, py::array::c_style | py::array::forcecast> ox_np,
        py::array_t<float, py::array::c_style | py::array::forcecast> oy_np,
        py::array_t<float, py::array::c_style | py::array::forcecast> nsx_np,
        py::array_t<float, py::array::c_style | py::array::forcecast> nsy_np,
        py::array_t<float, py::array::c_style | py::array::forcecast> ratio_np,
        int num_nodes,
        float xl, float yl, float bsx, float bsy) {
        if (num_nodes <= 0) {
            throw std::invalid_argument(
                "configure_electric_force: num_nodes must be > 0");
        }
        if (!ef_engine_) {
            void* mesh_device = eng_.mesh_device_ptr();
            if (!mesh_device) {
                throw std::runtime_error(
                    "configure_electric_force: V19 mesh_device_ptr() is null");
            }
            ef_engine_ = std::make_unique<V21EFEngine>(
                mesh_device,
                eng_.M(), eng_.N(),
                num_nodes,
                xl, yl, bsx, bsy);
            // Eagerly JIT-compile the V21 EF program so the first compute()
            // call doesn't pay 9 s of warmup mid-iteration.
            ef_engine_->prewarm();
        }
        ef_engine_->configure_constants(
            ox_np.data(), oy_np.data(),
            nsx_np.data(), nsy_np.data(),
            ratio_np.data(),
            num_nodes);
    }

    // configure_electric_force_full: takes FULL-DREAMPlace-layout arrays plus
    // a `sel` index array; engine stores the sel and does gather/scatter in
    // C++ on each compute(), eliminating per-iter Python masking.
    void configure_electric_force_full(
        py::array_t<float, py::array::c_style | py::array::forcecast> ox_full,
        py::array_t<float, py::array::c_style | py::array::forcecast> oy_full,
        py::array_t<float, py::array::c_style | py::array::forcecast> nsx_full,
        py::array_t<float, py::array::c_style | py::array::forcecast> nsy_full,
        py::array_t<float, py::array::c_style | py::array::forcecast> ratio_full,
        py::array_t<int32_t, py::array::c_style | py::array::forcecast> sel_np,
        int num_total_nodes,
        float xl, float yl, float bsx, float bsy) {
        const int num_active = static_cast<int>(sel_np.size());
        if (num_active <= 0) {
            throw std::invalid_argument(
                "configure_electric_force_full: sel must be non-empty");
        }
        if (!ef_engine_) {
            void* mesh_device = eng_.mesh_device_ptr();
            if (!mesh_device) {
                throw std::runtime_error("V19 mesh_device_ptr() is null");
            }
            ef_engine_ = std::make_unique<V21EFEngine>(
                mesh_device,
                eng_.M(), eng_.N(),
                num_active,
                xl, yl, bsx, bsy);
            // V23: configure BEFORE prewarm so adaptive max_k/max_h are known
            // when the program (with its JIT defines + CB sizes) is built.
            ef_engine_->configure_with_sel(
                ox_full.data(), oy_full.data(),
                nsx_full.data(), nsy_full.data(),
                ratio_full.data(),
                sel_np.data(), num_active,
                num_total_nodes);
            ef_engine_->prewarm();
        } else {
            ef_engine_->configure_with_sel(
                ox_full.data(), oy_full.data(),
                nsx_full.data(), nsy_full.data(),
                ratio_full.data(),
                sel_np.data(), num_active,
                num_total_nodes);
        }
    }

    // compute_electric_force_full: in-place writes -grad into `grad_full`.
    // Returns timing dict only. The output array (grad_full) is assumed to
    // be pre-zeroed for indices outside `sel` (fixed terminals).
    py::dict compute_electric_force_full(
        py::array_t<float, py::array::c_style | py::array::forcecast> pos_full,
        py::array_t<float, py::array::c_style | py::array::forcecast> fx_full,
        py::array_t<float, py::array::c_style | py::array::forcecast> fy_full,
        py::array_t<float, py::array::c_style> grad_full) {
        if (!ef_engine_) {
            throw std::runtime_error(
                "compute_electric_force_full: configure_electric_force_full() first");
        }
        EFTiming t{};
        {
            py::gil_scoped_release rel;
            ef_engine_->compute_from_full(
                pos_full.data(), fx_full.data(), fy_full.data(),
                grad_full.mutable_data(),
                &t);
        }
        py::dict td;
        td["h2d_pos_ms"]   = t.h2d_pos_ms;
        td["h2d_field_ms"] = t.h2d_field_ms;
        td["compute_ms"]   = t.compute_ms;
        td["d2h_grad_ms"]  = t.d2h_grad_ms;
        td["total_ms"]     = t.total_ms;
        td["launches"]     = t.launches;
        return td;
    }

    // compute_electric_force_full_chip: zero-copy-fields variant. Reads
    // field_x and field_y directly from chip DRAM at the given addresses
    // (which V19's DCT solver kept alive — see latest_field_addrs()).
    py::dict compute_electric_force_full_chip(
        py::array_t<float, py::array::c_style | py::array::forcecast> pos_full,
        uint32_t fx_chip_addr, uint32_t fy_chip_addr,
        py::array_t<float, py::array::c_style> grad_full) {
        if (!ef_engine_) {
            throw std::runtime_error(
                "compute_electric_force_full_chip: configure_electric_force_full() first");
        }
        EFTiming t{};
        {
            py::gil_scoped_release rel;
            ef_engine_->compute_from_full_chip_fields(
                pos_full.data(), fx_chip_addr, fy_chip_addr,
                grad_full.mutable_data(),
                &t);
        }
        py::dict td;
        td["h2d_pos_ms"]   = t.h2d_pos_ms;
        td["h2d_field_ms"] = t.h2d_field_ms;  // 0 for zero-copy path
        td["compute_ms"]   = t.compute_ms;
        td["d2h_grad_ms"]  = t.d2h_grad_ms;
        td["total_ms"]     = t.total_ms;
        td["launches"]     = t.launches;
        return td;
    }

    // ── V29 EF backward (L1-resident multi-bin gather) ──
    void configure_electric_force_v29(
        py::array_t<float, py::array::c_style | py::array::forcecast> ox_full,
        py::array_t<float, py::array::c_style | py::array::forcecast> oy_full,
        py::array_t<float, py::array::c_style | py::array::forcecast> nsx_full,
        py::array_t<float, py::array::c_style | py::array::forcecast> nsy_full,
        py::array_t<float, py::array::c_style | py::array::forcecast> ratio_full,
        py::array_t<int32_t, py::array::c_style | py::array::forcecast> sel_np,
        int num_total_nodes, float xl, float yl, float bsx, float bsy) {
        const int num_active = static_cast<int>(sel_np.size());
        if (num_active <= 0) throw std::invalid_argument("configure_electric_force_v29: empty sel");
        if (!v29_engine_) {
            void* md = eng_.mesh_device_ptr();
            if (!md) throw std::runtime_error("V19 mesh_device_ptr() is null");
            v29_engine_ = std::make_unique<v29ef::V29EFEngine>(md, eng_.M(), eng_.N(), num_active, xl, yl, bsx, bsy);
        }
        v29_engine_->configure_with_sel(ox_full.data(), oy_full.data(), nsx_full.data(),
            nsy_full.data(), ratio_full.data(), sel_np.data(), num_active, num_total_nodes);
    }

    py::dict compute_electric_force_v29(
        py::array_t<float, py::array::c_style | py::array::forcecast> pos_full,
        py::array_t<float, py::array::c_style | py::array::forcecast> fx_full,
        py::array_t<float, py::array::c_style | py::array::forcecast> fy_full,
        py::array_t<float, py::array::c_style> grad_full) {
        if (!v29_engine_) throw std::runtime_error("compute_electric_force_v29: configure first");
        v29ef::V29Timing t{};
        // PREP-REUSE: if the forward stashed per-cell geometry (V31_STASH=1 V31_GEOM=1),
        // point P1 at it so the backward buckets pre-computed overlaps instead of
        // recomputing them in soft-float. Valid only after the forward has run once
        // (the buffer is allocated lazily in the forward build); 0 → legacy prepbucket.
        v29_engine_->set_geom_source(eng_.v31_geom_addr(), eng_.v31_geom_pg());
        { py::gil_scoped_release rel;
          v29_engine_->compute_from_full(pos_full.data(), fx_full.data(), fy_full.data(),
                                         grad_full.mutable_data(), &t); }
        py::dict td; td["h2d_ms"]=t.h2d_ms; td["prep_ms"]=t.prep_ms; td["gather_ms"]=t.gather_ms;
        td["d2h_ms"]=t.d2h_ms; td["total_ms"]=t.total_ms; return td;
    }

    // ── FCCS EF backward (field-cast / cell-stationary, balanced slices) ──
    void configure_electric_force_fccs(
        py::array_t<float, py::array::c_style | py::array::forcecast> ox_full,
        py::array_t<float, py::array::c_style | py::array::forcecast> oy_full,
        py::array_t<float, py::array::c_style | py::array::forcecast> nsx_full,
        py::array_t<float, py::array::c_style | py::array::forcecast> nsy_full,
        py::array_t<float, py::array::c_style | py::array::forcecast> ratio_full,
        py::array_t<int32_t, py::array::c_style | py::array::forcecast> sel_np,
        int num_total_nodes, float xl, float yl, float bsx, float bsy) {
        const int num_active = static_cast<int>(sel_np.size());
        if (num_active <= 0) throw std::invalid_argument("configure_electric_force_fccs: empty sel");
        if (!fccs_engine_) {
            void* md = eng_.mesh_device_ptr();
            if (!md) throw std::runtime_error("V19 mesh_device_ptr() is null");
            fccs_engine_ = std::make_unique<fccsef::FCCSEFEngine>(md, eng_.M(), eng_.N(), num_active, xl, yl, bsx, bsy);
        }
        fccs_engine_->configure_with_sel(ox_full.data(), oy_full.data(), nsx_full.data(),
            nsy_full.data(), ratio_full.data(), sel_np.data(), num_active, num_total_nodes);
    }
    py::dict compute_electric_force_fccs(
        py::array_t<float, py::array::c_style | py::array::forcecast> pos_full,
        py::array_t<float, py::array::c_style | py::array::forcecast> fx_full,
        py::array_t<float, py::array::c_style | py::array::forcecast> fy_full,
        py::array_t<float, py::array::c_style> grad_full) {
        if (!fccs_engine_) throw std::runtime_error("compute_electric_force_fccs: configure first");
        fccsef::FCCSTiming t{};
        // reads the forward's V31_GEOM stash (forward must run with V31_STASH=1 V31_GEOM=1)
        fccs_engine_->set_geom_source(eng_.v31_geom_addr(), eng_.v31_geom_pg());
        { py::gil_scoped_release rel;
          fccs_engine_->compute_from_full(pos_full.data(), fx_full.data(), fy_full.data(),
                                          grad_full.mutable_data(), &t); }
        py::dict td; td["h2d_ms"]=t.h2d_ms; td["quant_ms"]=t.quant_ms; td["run_ms"]=t.run_ms;
        td["gather_ms"]=t.run_ms; td["d2h_ms"]=t.d2h_ms; td["total_ms"]=t.total_ms; return td;
    }
    py::array_t<int32_t> read_fccs_fxb() {
        std::vector<int32_t> v = fccs_engine_ ? fccs_engine_->read_fxb() : std::vector<int32_t>{};
        py::array_t<int32_t> a((py::ssize_t)v.size());
        if (!v.empty()) std::memcpy(a.mutable_data(), v.data(), v.size()*sizeof(int32_t));
        return a;
    }
    py::array_t<int64_t> read_fccs_grad_raw() {
        std::vector<int64_t> v = fccs_engine_ ? fccs_engine_->read_grad_raw() : std::vector<int64_t>{};
        py::array_t<int64_t> a((py::ssize_t)v.size());
        if (!v.empty()) std::memcpy(a.mutable_data(), v.data(), v.size()*sizeof(int64_t));
        return a;
    }
    // chip-resident field variant: reads field from chip-DCT DRAM (latest_field_addrs) — no h2d.
    py::dict compute_electric_force_fccs_chip(
        py::array_t<float, py::array::c_style | py::array::forcecast> pos_full,
        uint32_t fx_chip_addr, uint32_t fy_chip_addr,
        py::array_t<float, py::array::c_style> grad_full) {
        if (!fccs_engine_) throw std::runtime_error("compute_electric_force_fccs_chip: configure first");
        (void)pos_full;
        fccsef::FCCSTiming t{};
        fccs_engine_->set_geom_source(eng_.v31_geom_addr(), eng_.v31_geom_pg());
        { py::gil_scoped_release rel;
          fccs_engine_->compute_from_chip(fx_chip_addr, fy_chip_addr, grad_full.mutable_data(), &t); }
        py::dict td; td["h2d_ms"]=t.h2d_ms; td["quant_ms"]=t.quant_ms; td["run_ms"]=t.run_ms;
        td["gather_ms"]=t.run_ms; td["d2h_ms"]=t.d2h_ms; td["total_ms"]=t.total_ms; return td;
    }

    // ── V35 fully-on-chip Forward-Grouped Halo-Tile EF backward ──
    void configure_electric_force_v35(
        py::array_t<float, py::array::c_style | py::array::forcecast> ox_full,
        py::array_t<float, py::array::c_style | py::array::forcecast> oy_full,
        py::array_t<float, py::array::c_style | py::array::forcecast> nsx_full,
        py::array_t<float, py::array::c_style | py::array::forcecast> nsy_full,
        py::array_t<float, py::array::c_style | py::array::forcecast> ratio_full,
        py::array_t<int32_t, py::array::c_style | py::array::forcecast> sel_np,
        int num_total_nodes, float xl, float yl, float bsx, float bsy) {
        const int num_active = static_cast<int>(sel_np.size());
        if (num_active <= 0) throw std::invalid_argument("configure_electric_force_v35: empty sel");
        if (!v35_engine_) {
            void* md = eng_.mesh_device_ptr();
            if (!md) throw std::runtime_error("V19 mesh_device_ptr() is null");
            v35_engine_ = std::make_unique<v35ef::V35EFEngine>(md, eng_.M(), eng_.N(), num_active, xl, yl, bsx, bsy);
        }
        v35_engine_->configure_with_sel(ox_full.data(), oy_full.data(), nsx_full.data(),
            nsy_full.data(), ratio_full.data(), sel_np.data(), num_active, num_total_nodes);
    }
    py::dict compute_electric_force_v35_chip(
        py::array_t<float, py::array::c_style | py::array::forcecast> pos_full,
        uint32_t fx_chip_addr, uint32_t fy_chip_addr,
        py::array_t<float, py::array::c_style> grad_full) {
        if (!v35_engine_) throw std::runtime_error("compute_electric_force_v35_chip: configure first");
        (void)pos_full;
        v35ef::V35Timing t{};
        v35_engine_->set_geom_source(eng_.v31_geom_addr(), eng_.v31_geom_pg());
        { py::gil_scoped_release rel;
          v35_engine_->compute_from_chip(fx_chip_addr, fy_chip_addr, grad_full.mutable_data(), &t); }
        py::dict td; td["count_ms"]=t.count_ms; td["plan_ms"]=t.plan_ms; td["place_ms"]=t.place_ms;
        td["gather_ms"]=t.gather_ms; td["d2h_ms"]=t.d2h_ms; td["run_ms"]=t.gather_ms; td["total_ms"]=t.total_ms; return td;
    }

    // ── V30 field-stationary EF backward (hybrid: host prep+route, chip gather+atomic) ──
    void configure_electric_force_v30(
        py::array_t<float, py::array::c_style | py::array::forcecast> ox_full,
        py::array_t<float, py::array::c_style | py::array::forcecast> oy_full,
        py::array_t<float, py::array::c_style | py::array::forcecast> nsx_full,
        py::array_t<float, py::array::c_style | py::array::forcecast> nsy_full,
        py::array_t<float, py::array::c_style | py::array::forcecast> ratio_full,
        py::array_t<int32_t, py::array::c_style | py::array::forcecast> sel_np,
        int num_total_nodes, float xl, float yl, float bsx, float bsy) {
        const int num_active = static_cast<int>(sel_np.size());
        if (num_active <= 0) throw std::invalid_argument("configure_electric_force_v30: empty sel");
        if (!v30_engine_) {
            void* md = eng_.mesh_device_ptr();
            if (!md) throw std::runtime_error("V19 mesh_device_ptr() is null");
            v30_engine_ = std::make_unique<v30ef::V30EFEngine>(md, eng_.M(), eng_.N(), num_active, xl, yl, bsx, bsy);
        }
        v30_engine_->configure_with_sel(ox_full.data(), oy_full.data(), nsx_full.data(),
            nsy_full.data(), ratio_full.data(), sel_np.data(), num_active, num_total_nodes);
    }
    py::dict compute_electric_force_v30(
        py::array_t<float, py::array::c_style | py::array::forcecast> pos_full,
        py::array_t<float, py::array::c_style | py::array::forcecast> fx_full,
        py::array_t<float, py::array::c_style | py::array::forcecast> fy_full,
        py::array_t<float, py::array::c_style> grad_full) {
        if (!v30_engine_) throw std::runtime_error("compute_electric_force_v30: configure first");
        v30ef::V30Timing t{};
        { py::gil_scoped_release rel;
          v30_engine_->compute_from_full(pos_full.data(), fx_full.data(), fy_full.data(),
                                         grad_full.mutable_data(), &t); }
        py::dict td; td["h2d_ms"]=t.h2d_ms; td["prep_ms"]=t.prep_ms; td["gather_ms"]=t.gather_ms;
        td["d2h_ms"]=t.d2h_ms; td["total_ms"]=t.total_ms; return td;
    }

    // ── V31: no-host backward reading the forward stash (V31_STASH=1 forward) ──
    py::dict compute_electric_force_v31(
        py::array_t<int32_t, py::array::c_style | py::array::forcecast> sel_np,
        py::array_t<float, py::array::c_style | py::array::forcecast> ratio_full,
        py::array_t<float, py::array::c_style | py::array::forcecast> fx_full,
        py::array_t<float, py::array::c_style | py::array::forcecast> fy_full,
        py::array_t<float, py::array::c_style> grad_full,
        int num_total_nodes) {
        const int na = static_cast<int>(sel_np.size());
        if (na <= 0) throw std::invalid_argument("compute_electric_force_v31: empty sel");
        double tm[4] = {0,0,0,0};
        { py::gil_scoped_release rel;
          eng_.compute_electric_force_v31(sel_np.data(), ratio_full.data(), na, num_total_nodes,
              fx_full.data(), fy_full.data(), grad_full.mutable_data(), tm); }
        py::dict td; td["h2d_ms"]=tm[0]; td["gather_ms"]=tm[1]; td["d2h_ms"]=tm[2]; td["total_ms"]=tm[3];
        return td;
    }

    // Latest DCT field DRAM addresses (from V19's last solve_device() call).
    // Returns (0, 0) if not yet computed. Used by the zero-copy backward.
    // DEBUG: read the V31_GEOM buffer (int32 words, 32/record). For diagnosing the stash.
    py::array_t<int32_t> read_geom() {
        auto v = eng_.read_geom();
        py::array_t<int32_t> a((py::ssize_t)v.size());
        if (!v.empty()) std::memcpy(a.mutable_data(), v.data(), v.size()*sizeof(int32_t));
        return a;
    }
    py::tuple latest_field_addrs() {
        return py::make_tuple(
            eng_.latest_field_x_addr(),
            eng_.latest_field_y_addr());
    }

    // Read+clear the device profiler buffer (call per-iter so it never overflows;
    // emits to generated/profiler/.logs/profile_log_device.csv). PROFILER builds only.
    void dump_profiler() {
        void* md = eng_.mesh_device_ptr();
        if (md) tt::tt_metal::ReadMeshDeviceProfilerResults(
            *static_cast<tt::tt_metal::distributed::MeshDevice*>(md));
    }

    // Skip the field d2h in V19's DCT solver. Use when V21 EF will read
    // the field maps from chip directly (zero-copy backward).
    void set_skip_field_d2h(bool skip) {
        eng_.set_skip_field_d2h(skip);
        skip_field_d2h_ = skip;
    }

    // Per-(sub)cell parent active-cell index (0..nc-1), length new_nc. After big-cell
    // sub-tiling the forward stashes new_nc records; the FCCS backward must configure
    // with new_nc cells and accumulate each sub-cell's gradient back into its parent.
    py::array_t<int32_t> subcell_parent() {
        return py::array_t<int32_t>((py::ssize_t)subcell_parent_.size(),
                                    subcell_parent_.data());
    }

    py::tuple compute_electric_force(
        py::array_t<float, py::array::c_style | py::array::forcecast> pos_np,
        py::array_t<float, py::array::c_style | py::array::forcecast> fx_np,
        py::array_t<float, py::array::c_style | py::array::forcecast> fy_np) {
        if (!ef_engine_) {
            throw std::runtime_error(
                "compute_electric_force: call configure_electric_force() first");
        }
        const int num_nodes = ef_engine_->num_nodes_configured();
        if ((py::ssize_t)pos_np.size() < 2 * num_nodes) {
            throw std::invalid_argument(
                "compute_electric_force: pos must have ≥ 2*num_nodes entries");
        }
        const py::ssize_t MN = (py::ssize_t)eng_.M() * (py::ssize_t)eng_.N();
        if ((py::ssize_t)fx_np.size() < MN ||
            (py::ssize_t)fy_np.size() < MN) {
            throw std::invalid_argument(
                "compute_electric_force: field arrays must have ≥ M*N entries");
        }

        auto grad_x = py::array_t<float>({(py::ssize_t)num_nodes});
        auto grad_y = py::array_t<float>({(py::ssize_t)num_nodes});

        EFTiming t{};
        {
            py::gil_scoped_release rel;
            ef_engine_->compute(pos_np.data(),
                                fx_np.data(), fy_np.data(),
                                grad_x.mutable_data(), grad_y.mutable_data(),
                                &t);
        }
        py::dict td;
        td["h2d_pos_ms"]   = t.h2d_pos_ms;
        td["h2d_field_ms"] = t.h2d_field_ms;
        td["compute_ms"]   = t.compute_ms;
        td["d2h_grad_ms"]  = t.d2h_grad_ms;
        td["total_ms"]     = t.total_ms;
        td["launches"]     = t.launches;
        return py::make_tuple(grad_x, grad_y, td);
    }

private:
    V19Engine eng_;
    bool cpu_dct_ = false;
    py::array_t<float> fy_zero_arr_;   // pre-zeroed, reused across iters (CPU_DCT=1 only)
    double last_fused_loop_ms_ = 0.0;
    std::vector<int32_t> subcell_parent_;   // [new_nc] parent active-cell index per (sub)cell

    // configure_cells() state — set once, reused per iter.
    std::vector<float> offx_nc_, offy_nc_;   // dreamplace offset (nc-space), 0 unless V31_EF_GEOM
    bool configured_ = false;
    int  new_nc_ = 0;
    int  n_movable_ = 0;
    int  n_filler_  = 0;
    int  num_nodes_ = 0;
    std::vector<int32_t> src_idx_x_;
    std::vector<int32_t> src_idx_y_;
    std::vector<float>   offset_x_;
    std::vector<float>   offset_y_;
    py::array_t<float>   send_sx_arr_;
    py::array_t<float>   send_sy_arr_;
    py::array_t<float>   send_px_arr_;
    py::array_t<float>   send_py_arr_;

    // V21 EF engine (created on first configure_electric_force()).
    std::unique_ptr<V21EFEngine> ef_engine_;
    bool skip_field_d2h_ = false;

    // V29 EF engine (L1-resident multi-bin gather backward; created on first
    // configure_electric_force_v29()).
    std::unique_ptr<v29ef::V29EFEngine> v29_engine_;
    // FCCS EF engine (field-cast / cell-stationary backward; created on first
    // configure_electric_force_fccs()).
    std::unique_ptr<fccsef::FCCSEFEngine> fccs_engine_;
    std::unique_ptr<v35ef::V35EFEngine> v35_engine_;
    std::unique_ptr<v30ef::V30EFEngine> v30_engine_;

    // ── V22 chip-resident optimizer (created on first v22_configure) ──
    std::unique_ptr<v22::V22OptEngine> opt_engine_;
    std::vector<float> v22_pos_buf_;  // reused d2h buffer for get_pos

public:
    void v22_configure(py::array_t<float, py::array::c_style | py::array::forcecast> lo,
                       py::array_t<float, py::array::c_style | py::array::forcecast> hi,
                       py::array_t<float, py::array::c_style | py::array::forcecast> pin_w,
                       py::array_t<float, py::array::c_style | py::array::forcecast> area,
                       int num_nodes) {
        if (!opt_engine_) {
            void* md = eng_.mesh_device_ptr();
            if (!md) throw std::runtime_error("V19 mesh_device_ptr() is null");
            opt_engine_ = std::make_unique<v22::V22OptEngine>(md, num_nodes);
            v22_pos_buf_.assign((size_t)num_nodes * 2u, 0.0f);
        }
        opt_engine_->configure(lo.data(), hi.data(), pin_w.data(), area.data());
        opt_engine_->prewarm();
    }
    void v22_set_pos(py::array_t<float, py::array::c_style | py::array::forcecast> v_init) {
        if (!opt_engine_) throw std::runtime_error("v22_set_pos: v22_configure() first");
        opt_engine_->set_pos(v_init.data());
    }
    py::dict v22_step_resident(py::array_t<float, py::array::c_style | py::array::forcecast> wl_grad,
                               py::array_t<float, py::array::c_style | py::array::forcecast> density_grad,
                               float alpha, float coef, float density_weight) {
        if (!opt_engine_) throw std::runtime_error("v22_step_resident: v22_configure() first");
        v22::OptTiming t{};
        opt_engine_->step_resident(wl_grad.data(), density_grad.data(),
                                   alpha, coef, density_weight, &t);
        py::dict d;
        d["h2d_ms"] = t.h2d_ms; d["compute_ms"] = t.compute_ms; d["total_ms"] = t.total_ms;
        return d;
    }
    void v22_get_pos(py::array_t<float, py::array::c_style> out) {
        if (!opt_engine_) throw std::runtime_error("v22_get_pos: v22_configure() first");
        opt_engine_->get_pos(out.mutable_data());
    }
};

PYBIND11_MODULE(v19_engine, m) {
    m.doc() = "V19 Rung 3 in-process engine — pybind11 replacement for the "
              "IPC/docker-exec path of density_scatter_ttnn_server, with "
              "reused output buffers to amortize numpy alloc + memcpy cost.";

    py::class_<V19EngineWrap>(m, "V19Engine")
        .def(py::init<int, int, int, float, float, float, float, std::string>(),
             "M"_a, "N"_a, "NC_max"_a,
             "xl"_a, "yl"_a, "xh"_a, "yh"_a,
             "gather_mode"_a = "v19",
             "Construct the engine and pre-allocate output buffers. Reads "
             "CPU_DCT env once: when set, fx is aliased to density (zero-copy) "
             "and fy is a one-time-zeroed constant buffer.")

        .def("set_initial_density", &V19EngineWrap::set_initial_density,
             "id_normalized"_a,
             "Forward the initial-density-map (= initial_density / bin_area) "
             "to the engine. Consumed by the TTNN DCT path when CPU_DCT=0.")

        .def("scatter", &V19EngineWrap::scatter,
             "px"_a, "py"_a, "sx"_a, "sy"_a, "nc_actual"_a,
             "Run one scatter iteration. Returns "
             "(density, field_x, field_y, timing_dict). The returned numpy "
             "arrays are VIEWS of internal reused buffers — copy them "
             "(.copy() / .clone()) if you need persistence past the next "
             "scatter() call. DREAMPlace's pipeline doesn't need to copy "
             "because it consumes each iter's outputs (DCT, HPWL eval, etc.) "
             "before the next forward.")

        .def_property_readonly("M",      &V19EngineWrap::M)
        .def_property_readonly("N",      &V19EngineWrap::N)
        .def_property_readonly("NC_max", &V19EngineWrap::NC_max)
        .def_property_readonly("gather_mode", &V19EngineWrap::gather_mode_str)

        .def("configure_cells", &V19EngineWrap::configure_cells,
             "orig_sx_all"_a, "orig_sy_all"_a,
             "n_movable"_a, "n_filler"_a, "num_nodes"_a,
             "bin_size_x"_a, "bin_size_y"_a,
             "offx_all"_a = py::array_t<float>(), "offy_all"_a = py::array_t<float>(),
             "ratio_all"_a = py::array_t<float>(),
             "One-time setup of the big-cell sub-tile + cell-sort layout. "
             "Pass orig_sx_all = node_size_x_clamped + 2*offset_x. When V31_EF_GEOM=1 "
             "AND offx_all/offy_all (dreamplace offset, len num_nodes) are given, the "
             "forward uses the REAL clamped size at pos+offset (matches CPU electric_force) "
             "instead of the orig-size workaround.")

        .def_property_readonly("new_nc", &V19EngineWrap::new_nc,
             "After configure_cells(), the cell count fed to the chip "
             "(movable + filler + sub-cell expansion).")

        .def("scatter_from_pos", &V19EngineWrap::scatter_from_pos,
             "pos"_a,
             "Run one scatter iteration from a `pos` numpy view (length "
             "2*num_nodes float32, layout [x_movable | x_fixed | x_filler | "
             "y_movable | y_fixed | y_filler]). Does cell extraction, "
             "big-cell sub-tile, sort permutation, and the chip dispatch "
             "in one C++ pass — no per-iter numpy temporaries. Returns "
             "(density, field_x, field_y, timing_dict).")

        .def("configure_electric_force",
             &V19EngineWrap::configure_electric_force,
             "ox"_a, "oy"_a, "nsx_clamped"_a, "nsy_clamped"_a, "ratio"_a,
             "num_nodes"_a,
             "xl"_a, "yl"_a, "bsx"_a, "bsy"_a,
             "One-time upload of per-cell constants for V21 electric_force on "
             "chip. Lazily creates the V21EFEngine using V19's mesh device.")

        .def("compute_electric_force",
             &V19EngineWrap::compute_electric_force,
             "pos"_a, "field_x"_a, "field_y"_a,
             "Run V21 electric_force on chip. Inputs: pos (length 2*num_nodes), "
             "field_x and field_y (each length M*N, row-major). Returns "
             "(grad_x, grad_y, timing_dict).")

        .def("configure_electric_force_full",
             &V19EngineWrap::configure_electric_force_full,
             "ox_full"_a, "oy_full"_a, "nsx_clamped_full"_a, "nsy_clamped_full"_a,
             "ratio_full"_a, "sel"_a, "num_total_nodes"_a,
             "xl"_a, "yl"_a, "bsx"_a, "bsy"_a,
             "Like configure_electric_force, but accepts full-DREAMPlace-layout "
             "arrays + a `sel` index array. Engine stores sel internally so "
             "compute_electric_force_full() can do gather/scatter in C++.")

        .def("compute_electric_force_full",
             &V19EngineWrap::compute_electric_force_full,
             "pos_full"_a, "fx_full"_a, "fy_full"_a, "grad_full"_a,
             "Per-iter V21 EF call with full-layout inputs. Reads pos_full "
             "(2*num_total_nodes), fx_full/fy_full (M*N each), writes -grad into "
             "grad_full at indices given by `sel` (caller pre-zeros grad_full). "
             "Returns the per-iter timing dict.")

        .def("dump_profiler", &V19EngineWrap::dump_profiler,
             "Read+clear device profiler buffer (per-iter; PROFILER builds).")
        .def("latest_field_addrs", &V19EngineWrap::latest_field_addrs,
             "Return (fx_chip_addr, fy_chip_addr) — DRAM addresses of V19's most "
             "recent DCT field maps (still resident on chip). Use with "
             "compute_electric_force_full_chip() to skip d2h+h2d.")
        .def("read_geom", &V19EngineWrap::read_geom,
             "DEBUG: read the V31_GEOM buffer as int32 (32 words/record).")
        .def("read_fccs_fxb", &V19EngineWrap::read_fccs_fxb, "DEBUG: FCCS quantized fx field (int32).")
        .def("read_fccs_grad_raw", &V19EngineWrap::read_fccs_grad_raw, "DEBUG: FCCS raw int64 grad buffer.")

        .def("subcell_parent", &V19EngineWrap::subcell_parent,
             "Per-(sub)cell parent active-cell index (len new_nc) for FCCS sub-cell mapping.")
        .def("set_skip_field_d2h", &V19EngineWrap::set_skip_field_d2h,
             "skip"_a,
             "Skip the field_x/field_y EnqueueReadMeshBuffer in V19's DCT solver. "
             "Set to True when V21 EF will read fields from chip directly via "
             "latest_field_addrs(). Saves M*N*4*2 bytes of d2h per iter. The "
             "Python-side numpy arrays still get allocated but are not filled — "
             "callers must NOT consume them when this flag is on.")

        .def("compute_electric_force_full_chip",
             &V19EngineWrap::compute_electric_force_full_chip,
             "pos_full"_a, "fx_chip_addr"_a, "fy_chip_addr"_a, "grad_full"_a,
             "Zero-copy-field variant: V21 EF reads field_x, field_y directly "
             "from chip DRAM at the given addresses (skips field_map d2h+h2d). "
             "Use V19's latest_field_addrs() to get them.")

        // ── V29 EF backward (L1-resident multi-bin gather) ──
        .def("configure_electric_force_v29", &V19EngineWrap::configure_electric_force_v29,
             "ox_full"_a, "oy_full"_a, "nsx_full"_a, "nsy_full"_a, "ratio_full"_a,
             "sel"_a, "num_total_nodes"_a, "xl"_a, "yl"_a, "bsx"_a, "bsy"_a,
             "Configure the V29 backward engine (same args as configure_electric_force_full).")
        .def("compute_electric_force_v29", &V19EngineWrap::compute_electric_force_v29,
             "pos_full"_a, "fx_full"_a, "fy_full"_a, "grad_full"_a,
             "V29 on-chip backward: prep+bucket → L1-resident multi-bin gather → "
             "scatter grad. Writes raw force (no negation) into grad_full[sel].")
        .def("configure_electric_force_fccs", &V19EngineWrap::configure_electric_force_fccs,
             "ox_full"_a, "oy_full"_a, "nsx_full"_a, "nsy_full"_a, "ratio_full"_a,
             "sel"_a, "num_total_nodes"_a, "xl"_a, "yl"_a, "bsx"_a, "bsy"_a,
             "Configure the FCCS field-cast / cell-stationary backward engine.")
        .def("compute_electric_force_fccs", &V19EngineWrap::compute_electric_force_fccs,
             "pos_full"_a, "fx_full"_a, "fy_full"_a, "grad_full"_a,
             "FCCS on-chip backward: field multicast to all cores + balanced cell slices + "
             "in-L1 sort + int MAC. Reads the forward V31_GEOM stash. Writes raw force "
             "(no negation) into grad_full[sel].")
        .def("compute_electric_force_fccs_chip", &V19EngineWrap::compute_electric_force_fccs_chip,
             "pos_full"_a, "fx_chip_addr"_a, "fy_chip_addr"_a, "grad_full"_a,
             "FCCS backward reading the field from chip-DCT DRAM (latest_field_addrs) — "
             "host-free, no field h2d. Writes raw force into grad_full[sel].")
        .def("configure_electric_force_v35", &V19EngineWrap::configure_electric_force_v35,
             "ox_full"_a, "oy_full"_a, "nsx_full"_a, "nsy_full"_a, "ratio_full"_a,
             "sel"_a, "num_total_nodes"_a, "xl"_a, "yl"_a, "bsx"_a, "bsy"_a,
             "Configure the V35 fully-on-chip Forward-Grouped Halo-Tile backward engine.")
        .def("compute_electric_force_v35_chip", &V19EngineWrap::compute_electric_force_v35_chip,
             "pos_full"_a, "fx_chip_addr"_a, "fy_chip_addr"_a, "grad_full"_a,
             "V35 fully-on-chip backward (count->plan->place->gather) reading the chip-DCT "
             "field from DRAM (latest_field_addrs). Reads the forward V31_GEOM stash. "
             "Writes raw force into grad_full[sel].")
        .def("configure_electric_force_v30", &V19EngineWrap::configure_electric_force_v30,
             "ox_full"_a, "oy_full"_a, "nsx_full"_a, "nsy_full"_a, "ratio_full"_a,
             "sel"_a, "num_total_nodes"_a, "xl"_a, "yl"_a, "bsx"_a, "bsy"_a,
             "Configure the V30 field-stationary backward engine.")
        .def("compute_electric_force_v30", &V19EngineWrap::compute_electric_force_v30,
             "pos_full"_a, "fx_full"_a, "fy_full"_a, "grad_full"_a,
             "V30 field-stationary backward: host prep+route → chip local-field gather "
             "+ fixed-point atomic grad sum. Writes raw force (no negation) into grad_full[sel].")
        .def("compute_electric_force_v31", &V19EngineWrap::compute_electric_force_v31,
             "sel"_a, "ratio_full"_a, "fx_full"_a, "fy_full"_a, "grad_full"_a, "num_total_nodes"_a,
             "V31 no-host backward: reads the forward stash (V31_STASH=1) + field, runs the "
             "band-discovering kernel, writes raw +force into grad_full[sel].")

        // ── V22 chip-resident optimizer (Phase B) ──
        .def("v22_configure", &V19EngineWrap::v22_configure,
             "lo"_a, "hi"_a, "pin_w"_a, "area"_a, "num_nodes"_a,
             "One-time setup of the chip-resident optimizer. lo/hi are the "
             "clamp bounds, pin_w/area the precondition constants (all length "
             "2*num_nodes, optimizer order). Lazily creates V22OptEngine on "
             "V19's mesh device + prewarms.")
        .def("v22_set_pos", &V19EngineWrap::v22_set_pos, "v_init"_a,
             "Upload the initial pos into the resident buffer (and seed u=v).")
        .def("v22_step_resident", &V19EngineWrap::v22_step_resident,
             "wl_grad"_a, "density_grad"_a, "alpha"_a, "coef"_a, "density_weight"_a,
             "One resident optimizer step (combine→precond→nesterov→clamp) "
             "updating chip-resident pos/u IN PLACE. Returns timing dict.")
        .def("v22_get_pos", &V19EngineWrap::v22_get_pos, "out"_a,
             "Download the resident pos into `out` (2*num_nodes, pre-allocated).");
}
