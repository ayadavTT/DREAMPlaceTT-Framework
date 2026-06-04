// SPDX-License-Identifier: Apache-2.0
//
// synth.hpp — synthetic DREAMPlace-like workload generator + CPU references.
// Shared by every per-kernel testbench (a library, not a runner).
//
// A "workload" is a set of cells, each at position (px,py) with size (sx,sy),
// all in BIN UNITS on a grid×grid bin map (so a cell of size 1.0 spans ~1 bin).
// This is the natural input to the forward density pipeline; backward EF tests
// additionally use gen_field().
//
// Distributions (tb::Dist): Uniform | Cluster (one dense blob) | Clouds (several
// blobs). Cell sizes follow a realistic bin-space model: ~95% sub-bin standard
// cells, ~5% multi-bin "macros" (this is what drives multi-bin overlap and
// cross-core atomic contention).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
#include "design_profiles.hpp"

namespace tb {

struct Workload {
    std::vector<float> px, py, sx, sy;   // length n_cells, bin units
    uint32_t n_cells = 0;
    uint32_t grid    = 0;                 // bins per axis (M=N=grid)
};

// Generate cells for (n_cells, grid, distribution). Deterministic in `seed`.
inline Workload gen_workload(uint32_t n_cells, uint32_t grid, Dist dist, uint64_t seed) {
    Workload w;
    w.n_cells = n_cells; w.grid = grid;
    w.px.resize(n_cells); w.py.resize(n_cells);
    w.sx.resize(n_cells); w.sy.resize(n_cells);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    std::normal_distribution<float>       gauss(0.0f, 1.0f);

    // ── cell sizes (bin units): 95% sub-bin std cells, 5% multi-bin macros ──
    std::uniform_real_distribution<float> small_sz(0.10f, 0.80f);
    std::uniform_real_distribution<float> macro_sz(1.50f, 8.00f);
    for (uint32_t i = 0; i < n_cells; ++i) {
        bool macro = (u01(rng) < 0.05f);
        w.sx[i] = macro ? macro_sz(rng) : small_sz(rng);
        w.sy[i] = macro ? macro_sz(rng) : small_sz(rng);
    }

    const float G = (float)grid;
    auto clamppos = [&](float v, float s) {
        float hi = G - s; if (hi < 0.f) hi = 0.f;
        return std::min(std::max(v, 0.0f), hi);
    };

    if (dist == Dist::Uniform) {
        for (uint32_t i = 0; i < n_cells; ++i) {
            w.px[i] = clamppos(u01(rng) * G, w.sx[i]);
            w.py[i] = clamppos(u01(rng) * G, w.sy[i]);
        }
    } else if (dist == Dist::Cluster) {
        // one large dense blob near center
        float cx = G * 0.5f, cy = G * 0.5f, sd = G * 0.08f;
        for (uint32_t i = 0; i < n_cells; ++i) {
            w.px[i] = clamppos(cx + gauss(rng) * sd, w.sx[i]);
            w.py[i] = clamppos(cy + gauss(rng) * sd, w.sy[i]);
        }
    } else {  // Clouds: several blobs
        const int K = 8;
        std::vector<float> ccx(K), ccy(K);
        for (int k = 0; k < K; ++k) { ccx[k] = u01(rng) * G; ccy[k] = u01(rng) * G; }
        float sd = G * 0.04f;
        std::uniform_int_distribution<int> pick(0, K - 1);
        for (uint32_t i = 0; i < n_cells; ++i) {
            int k = pick(rng);
            w.px[i] = clamppos(ccx[k] + gauss(rng) * sd, w.sx[i]);
            w.py[i] = clamppos(ccy[k] + gauss(rng) * sd, w.sy[i]);
        }
    }
    return w;
}

// Per-cell base bin + up-to-8 x/y overlap fractions (the v4_compute contract).
// ox[j] = overlap length of [px,px+sx] with bin column (bxl+j); same for oy.
struct CellOverlap {
    int32_t bxl, byl;
    float ox[8], oy[8];
};
inline CellOverlap overlap_of(float px, float py, float sx, float sy, uint32_t grid) {
    CellOverlap c{};
    c.bxl = (int32_t)std::floor(px);
    c.byl = (int32_t)std::floor(py);
    for (int j = 0; j < 8; ++j) {
        float lo = std::max((float)(c.bxl + j), px);
        float hi = std::min((float)(c.bxl + j + 1), px + sx);
        c.ox[j] = std::max(0.0f, hi - lo);
    }
    for (int k = 0; k < 8; ++k) {
        float lo = std::max((float)(c.byl + k), py);
        float hi = std::min((float)(c.byl + k + 1), py + sy);
        c.oy[k] = std::max(0.0f, hi - lo);
    }
    (void)grid;
    return c;
}

// CPU reference density map D[grid*grid] (row-major bin_global = bx*grid+by),
// the exact expected output of any forward scatter/gather kernel. fp64 accum.
inline std::vector<double> ref_density(const Workload& w) {
    std::vector<double> D((size_t)w.grid * w.grid, 0.0);
    for (uint32_t i = 0; i < w.n_cells; ++i) {
        CellOverlap c = overlap_of(w.px[i], w.py[i], w.sx[i], w.sy[i], w.grid);
        for (int j = 0; j < 8; ++j) {
            if (c.ox[j] <= 0.f) continue;
            uint32_t bx = (uint32_t)(c.bxl + j);
            if (bx >= w.grid) continue;
            for (int k = 0; k < 8; ++k) {
                if (c.oy[k] <= 0.f) continue;
                uint32_t by = (uint32_t)(c.byl + k);
                if (by >= w.grid) continue;
                D[(size_t)bx * w.grid + by] += (double)c.ox[j] * (double)c.oy[k];
            }
        }
    }
    return D;
}

// Synthetic field map [grid*grid] for backward-EF tests (smooth-ish random).
inline std::vector<float> gen_field(uint32_t grid, uint64_t seed) {
    std::vector<float> f((size_t)grid * grid);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (auto& v : f) v = u(rng);
    return f;
}

// relative L2 error between a device result and the fp64 reference.
template <class T>
inline double rel_l2(const std::vector<T>& got, const std::vector<double>& ref) {
    double num = 0.0, den = 0.0;
    size_t n = std::min(got.size(), ref.size());
    for (size_t i = 0; i < n; ++i) {
        double d = (double)got[i] - ref[i];
        num += d * d; den += ref[i] * ref[i];
    }
    return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

}  // namespace tb
