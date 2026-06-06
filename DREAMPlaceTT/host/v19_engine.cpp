// SPDX-License-Identifier: Apache-2.0
//
// V19 Rung 3 — in-process engine implementation.
//
// This file is a port of `density_scatter_ttnn_server_host.cpp`. The IPC
// layer (POSIX shared memory + state-flag polling + docker exec) is
// stripped; the per-iteration body now runs as a method that DREAMPlace's
// Python calls directly via the pybind11 binding in v19_engine_pybind.cpp.
//
// Layout:
//   1. Standard / TT-Metal / TTNN includes (same as the server).
//   2. Helper functions and utilities (same).
//   3. namespace v19 { ... } containing:
//      - V19Engine::Impl struct: owns the mesh device + all per-iter state.
//      - Impl::Impl(...): one-time setup (ported from server's main() lines
//        ~374-2545, with arg parsing replaced by ctor params).
//      - Impl::scatter(...): per-iteration body (ported from the server's
//        while(true) loop, with shm reads/writes replaced by pointer args).
//      - V19Engine wrapper forwarding methods (PIMPL).
//
// Supported gather modes: {v11, v11outer, v11outer_auto, v18, v18outer, v19}.
// V13/V14 stay on the IPC server (`density_scatter_ttnn_server` binary).

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include "v19_engine.h"

// TT-Metal
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/circular_buffer_constants.h>

// V11 helpers
#include "v11_tile_ownership.h"

// TTNN
#include "ttnn/device.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/memory_config/memory_config.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"

using tt::tt_metal::TensorLayout;
using tt::tt_metal::PageConfig;

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;
using CoreCoord    = tt::tt_metal::CoreCoord;
using CoreRange    = tt::tt_metal::CoreRange;
using CoreRangeSet = tt::tt_metal::CoreRangeSet;
using hrclock      = std::chrono::high_resolution_clock;

template <class T>
static double ms_since(T t0) {
    return std::chrono::duration<double, std::milli>(hrclock::now() - t0).count();
}

// ═══════════════════════════════════════════════════════════════════
// Utility helpers
// ═══════════════════════════════════════════════════════════════════

static std::string to_float_literal(float v) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%.9g", v);
    std::string s(buf);
    if (s.find('.')==std::string::npos &&
        s.find('e')==std::string::npos &&
        s.find('n')==std::string::npos) s += ".0";
    return s + "f";
}

static std::vector<CoreCoord> get_cores(const CoreCoord& grid, int n) {
    std::vector<CoreCoord> cores;
    for (uint32_t y=0; y<grid.y && (int)cores.size()<n; ++y)
        for (uint32_t x=0; x<grid.x && (int)cores.size()<n; ++x)
            cores.push_back({x,y});
    return cores;
}

static CoreRangeSet cores_to_crs(const std::vector<CoreCoord>& cv) {
    std::set<CoreRange> s;
    for (auto& c : cv) s.insert(CoreRange{c,c});
    return CoreRangeSet(s);
}

// ═══════════════════════════════════════════════════════════════════
// DCT matrix builders (matching Python TTNNFieldSolver exactly)
// ═══════════════════════════════════════════════════════════════════

// DCT-II: C[k,n] = (2/N) * cos(pi * k * (n+0.5) / N)
static std::vector<float> build_dct2(int N) {
    std::vector<float> m(N*N);
    for (int k=0; k<N; ++k)
        for (int n=0; n<N; ++n)
            m[k*N+n] = (float)((2.0/N) * std::cos(M_PI * k * (n+0.5) / N));
    return m;
}

// DCT-II transpose (DCT_N^T): m[n,k] = (2/N) * cos(pi*k*(n+0.5)/N)
static std::vector<float> build_dct2_t(int N) {
    auto src = build_dct2(N);
    std::vector<float> t(N*N);
    for (int k=0; k<N; ++k)
        for (int n=0; n<N; ++n)
            t[n*N+k] = src[k*N+n];
    return t;
}

// IDCT (DCT-III inverse): mat[k,0]=1, mat[k,n]=2*cos(pi*n*(k+0.5)/N) for n>0
static std::vector<float> build_idct(int N) {
    std::vector<float> m(N*N);
    for (int k=0; k<N; ++k) {
        m[k*N+0] = 1.0f;
        for (int n=1; n<N; ++n)
            m[k*N+n] = (float)(2.0 * std::cos(M_PI * n * (k+0.5) / N));
    }
    return m;
}

// IDCT transpose: mat[n,k] from build_idct
static std::vector<float> build_idct_t(int N) {
    auto src = build_idct(N);
    std::vector<float> t(N*N);
    for (int k=0; k<N; ++k)
        for (int n=0; n<N; ++n)
            t[n*N+k] = src[k*N+n];
    return t;
}

// IDXST: mat[k,n] = sin(pi*n*(k+0.5)/N)
static std::vector<float> build_idxst(int N) {
    std::vector<float> m(N*N);
    for (int k=0; k<N; ++k)
        for (int n=0; n<N; ++n)
            m[k*N+n] = (float)(std::sin(M_PI * n * (k+0.5) / N));
    return m;
}

// IDXST transpose
static std::vector<float> build_idxst_t(int N) {
    auto src = build_idxst(N);
    std::vector<float> t(N*N);
    for (int k=0; k<N; ++k)
        for (int n=0; n<N; ++n)
            t[n*N+k] = src[k*N+n];
    return t;
}

// ═══════════════════════════════════════════════════════════════════
// TTNN tensor helpers
// ═══════════════════════════════════════════════════════════════════

using TT = tt::tt_metal::Tensor;

static TT make_tt_tensor(const std::vector<float>& data, int rows, int cols,
                          MeshDevice* dev) {
    // CPU ROW_MAJOR → device ROW_MAJOR → device TILE.
    // Device-side tilize is dramatically faster than CPU tilize for large M*N.
    TensorLayout cpu_layout(
        ttnn::DataType::FLOAT32,
        PageConfig(ttnn::Layout::ROW_MAJOR),
        tt::tt_metal::MemoryConfig{});
    auto cpu_t = TT::from_vector<float>(
        data, ttnn::TensorSpec(ttnn::Shape{(uint32_t)rows, (uint32_t)cols}, cpu_layout));
    auto dev_rm   = cpu_t.to_device(dev);
    auto dev_tile = ttnn::to_layout(dev_rm, ttnn::Layout::TILE);
    return dev_tile;
}

static std::vector<float> tt_tensor_to_vec(const TT& t, int M, int N) {
    // Device TILE → device ROW_MAJOR → CPU ROW_MAJOR (device-side untilize).
    auto _t0 = hrclock::now();
    auto dev_rm = ttnn::to_layout(t, ttnn::Layout::ROW_MAJOR);
    auto _t1 = hrclock::now();
    auto cpu_t  = dev_rm.cpu();
    auto _t2 = hrclock::now();
    auto vec    = cpu_t.to_vector<float>();
    auto _t3 = hrclock::now();
    static thread_local uint64_t _bench_call_count = 0;
    if (getenv("TTNN_DL_BENCH") && (_bench_call_count++ % 50 == 0)) {
        double tl_ms = std::chrono::duration<double, std::milli>(_t1 - _t0).count();
        double cp_ms = std::chrono::duration<double, std::milli>(_t2 - _t1).count();
        double tv_ms = std::chrono::duration<double, std::milli>(_t3 - _t2).count();
        printf("[ttnn_dl] M=%d N=%d  to_layout=%.3f  cpu()=%.3f  to_vector=%.3f  total=%.3f\n",
               M, N, tl_ms, cp_ms, tv_ms, tl_ms + cp_ms + tv_ms);
        fflush(stdout);
    }
    // Trim to M×N (TTNN pads to multiples of 32)
    if ((int)vec.size() != M*N) {
        // Tensor was padded — extract just the first M rows, each N elements
        int padN = ((N + 31) / 32) * 32;
        std::vector<float> out(M*N);
        for (int r=0; r<M; ++r)
            std::copy(vec.begin() + r*padN, vec.begin() + r*padN + N, out.begin() + r*N);
        return out;
    }
    return vec;
}

// Wrap an existing MeshBuffer (already in TT DRAM, ROW_MAJOR fp32)
// as a ttnn::Tensor without any host-device data transfer.
// density_buf has page_size = N*sizeof(float), total = M*N*sizeof(float),
// which matches ROW_MAJOR [M,N] fp32 interleaved DRAM exactly.
static TT wrap_mesh_buf_as_tensor(
        std::shared_ptr<MeshBuffer> mesh_buf,
        int rows, int cols) {
    TensorLayout rm_layout(
        ttnn::DataType::FLOAT32,
        PageConfig(ttnn::Layout::ROW_MAJOR),
        tt::tt_metal::MemoryConfig{});
    TensorSpec spec(ttnn::Shape{(uint32_t)rows, (uint32_t)cols}, rm_layout);
    TensorTopology topo{};
    MeshTensor mt(mesh_buf, spec, topo);
    return TT(DeviceStorage(std::move(mt)));
}

// ═══════════════════════════════════════════════════════════════════
// TTNN DCT solver context (initialized once per (M, N, bsx, bsy))
// ═══════════════════════════════════════════════════════════════════

struct TTNNDCTSolver {
    int M, N;
    TT DCT_N_T_tt;   // N×N
    TT DCT_M_tt;     // M×M
    TT IDXST_M_tt;   // M×M  (used for field_x left-multiply: 2*IDXST_M)
    TT IDCT_N_T_tt;  // N×N  (used for field_x right-multiply)
    TT IDCT_M_tt;    // M×M  (used for field_y left-multiply: 2*IDCT_M)
    TT IDXST_N_T_tt; // N×N  (used for field_y right-multiply)
    TT wu_tt;        // M×N  eigenvalue weights for field_x
    TT wv_tt;        // M×N  eigenvalue weights for field_y

    // Zero-copy field path: keep the most recent field-x/field-y ROW_MAJOR
    // MeshBuffers alive across solve_device() calls so V21 EF can read from
    // them on-chip directly (skipping D2H + H2D of M*N*4 bytes per side).
    std::shared_ptr<MeshBuffer> latest_field_x_mb;
    std::shared_ptr<MeshBuffer> latest_field_y_mb;
    bool latest_field_valid = false;
    // When true, solve_device still produces the ROW_MAJOR chip tensors (so
    // latest_field_*_mb stay valid for V21 EF zero-copy) but SKIPS the host
    // EnqueueReadMeshBuffer — saves M*N*4*2 bytes of d2h per iter when the
    // backward will read from chip anyway.
    bool skip_field_d2h = false;

    void init(int M_, int N_, float bsx, float bsy, MeshDevice* dev) {
        M = M_; N = N_;
        // Build IDXST with 2× factor folded in (avoids scalar multiply)
        auto idxst_m = build_idxst(M);
        auto idct_m  = build_idct(M);
        for (auto& v : idxst_m) v *= 2.0f;
        for (auto& v : idct_m)  v *= 2.0f;

        DCT_N_T_tt   = make_tt_tensor(build_dct2_t(N), N, N, dev);
        DCT_M_tt     = make_tt_tensor(build_dct2(M),   M, M, dev);
        IDXST_M_tt   = make_tt_tensor(idxst_m,         M, M, dev);
        IDCT_N_T_tt  = make_tt_tensor(build_idct_t(N), N, N, dev);
        IDCT_M_tt    = make_tt_tensor(idct_m,          M, M, dev);
        IDXST_N_T_tt = make_tt_tensor(build_idxst_t(N),N, N, dev);

        // Eigenvalue weights (matching Python TTNNFieldSolver lines 111-122)
        std::vector<float> wu_v(M*N), wv_v(M*N);
        for (int u=0; u<M; ++u) {
            double wu = 2.0*M_PI*u / M;
            for (int v=0; v<N; ++v) {
                double wv = 2.0*M_PI*v / N * (bsx/bsy);
                double d  = wu*wu + wv*wv;
                if (d == 0.0) { wu_v[u*N+v] = wv_v[u*N+v] = 0.0f; continue; }
                double inv = 1.0 / d;
                wu_v[u*N+v] = (float)(wu * inv * 0.5);
                wv_v[u*N+v] = (float)(wv * inv * 0.5);
            }
        }
        wu_tt = make_tt_tensor(wu_v, M, N, dev);
        wv_tt = make_tt_tensor(wv_v, M, N, dev);
        printf("[ttnn_solver] Matrices uploaded for %dx%d bsx=%.4f bsy=%.4f\n",
               M, N, bsx, bsy);
        fflush(stdout);
    }

    // Returns field_x and field_y as flat float vectors (size M*N, row-major)
    void solve(const std::vector<float>& density_flat,
               std::vector<float>& field_x, std::vector<float>& field_y,
               MeshDevice* dev,
               double& upload_ms, double& compute_ms, double& download_ms) {
        auto ts = hrclock::now();
        auto rho_tt = make_tt_tensor(density_flat, M, N, dev);
        upload_ms = ms_since(ts);

        ts = hrclock::now();
        // 2D DCT-II: auv = DCT_M @ rho @ DCT_N^T
        auto temp  = ttnn::operations::matmul::matmul(rho_tt,   DCT_N_T_tt);
        auto auv   = ttnn::operations::matmul::matmul(DCT_M_tt, temp);

        // Eigenvalue scaling
        auto fx_auv = ttnn::multiply(auv, wu_tt);
        auto fy_auv = ttnn::multiply(auv, wv_tt);

        // IDXST_IDCT: field_x = (2*IDXST_M) @ fx_auv @ IDCT_N^T
        auto temp_x    = ttnn::operations::matmul::matmul(fx_auv,    IDCT_N_T_tt);
        auto field_x_t = ttnn::operations::matmul::matmul(IDXST_M_tt, temp_x);

        // IDCT_IDXST: field_y = (2*IDCT_M) @ fy_auv @ IDXST_N^T
        auto temp_y    = ttnn::operations::matmul::matmul(fy_auv,    IDXST_N_T_tt);
        auto field_y_t = ttnn::operations::matmul::matmul(IDCT_M_tt, temp_y);

        // Sync
        Finish(dev->mesh_command_queue());
        compute_ms = ms_since(ts);

        ts = hrclock::now();
        field_x = tt_tensor_to_vec(field_x_t, M, N);
        field_y = tt_tensor_to_vec(field_y_t, M, N);
        download_ms = ms_since(ts);
    }

    // Zero-copy variant: accepts a ROW_MAJOR device tensor (already on TT DRAM)
    // instead of a host vector.  Avoids the D2H + H2D round-trip.
    // Returns true if the field d2h was read DIRECTLY into fx_direct/fy_direct
    // (host-backward fast path — eliminates the caller-side field memcpy). When
    // fx_direct/fy_direct are null, or the grid is tile-padded, falls back to
    // filling field_x/field_y and returns false.
    bool solve_device(TT rho_rm,
                      std::vector<float>& field_x, std::vector<float>& field_y,
                      MeshDevice* dev,
                      double& tilize_ms, double& compute_ms, double& download_ms,
                      float* fx_direct = nullptr, float* fy_direct = nullptr) {
        auto ts = hrclock::now();
        auto rho_tt = ttnn::to_layout(rho_rm, ttnn::Layout::TILE);
        Finish(dev->mesh_command_queue());
        tilize_ms = ms_since(ts);

        // HiFi4 + fp32 dest accumulator = fp32 precision throughout the matmul
        // chain. Default LoFi/bf16 loses precision on the DCT and at bigblue3
        // scale drives DREAMPlace into a 5% HPWL drift / +34% iter regression.
        // Opt-in via TTNN_HIFI=1; default off so the existing bf16 path is the
        // shipping default.
        ttnn::DeviceComputeKernelConfig hifi_cfg;
        hifi_cfg.math_fidelity      = tt::tt_metal::MathFidelity::HiFi4;
        hifi_cfg.fp32_dest_acc_en   = true;
        hifi_cfg.math_approx_mode   = false;
        hifi_cfg.packer_l1_acc      = false;
        std::optional<const ttnn::DeviceComputeKernelConfig> cfg_opt =
            (std::getenv("TTNN_HIFI") && std::string(std::getenv("TTNN_HIFI")) == "1")
                ? std::optional<const ttnn::DeviceComputeKernelConfig>(hifi_cfg)
                : std::nullopt;
        static bool _logged_fidelity = false;
        if (!_logged_fidelity) {
            printf("[ttnn_solver] matmul fidelity: %s\n",
                   cfg_opt.has_value() ? "HiFi4+fp32_dest_acc" : "LoFi (default)");
            fflush(stdout);
            _logged_fidelity = true;
        }
        #define MM(A, B) ttnn::operations::matmul::matmul( \
            (A), (B), false, false, std::nullopt, std::nullopt, std::nullopt, std::nullopt, cfg_opt)

        ts = hrclock::now();
        auto temp  = MM(rho_tt,   DCT_N_T_tt);
        auto auv   = MM(DCT_M_tt, temp);

        auto fx_auv = ttnn::multiply(auv, wu_tt);
        auto fy_auv = ttnn::multiply(auv, wv_tt);

        auto temp_x    = MM(fx_auv,    IDCT_N_T_tt);
        auto field_x_t = MM(IDXST_M_tt, temp_x);

        auto temp_y    = MM(fy_auv,    IDXST_N_T_tt);
        auto field_y_t = MM(IDCT_M_tt, temp_y);
        #undef MM

        Finish(dev->mesh_command_queue());
        compute_ms = ms_since(ts);

        ts = hrclock::now();
        // ── Fast download path: bypass TTNN's slow cpu() (~3 GB/s observed).
        // 1. to_layout(ROW_MAJOR) untilizes onto a fresh DRAM buffer.
        // 2. Grab the underlying MeshBuffer via leak_ownership.
        // 3. EnqueueReadMeshBuffer reads at the raw ~7-8 GB/s PCIe rate.
        // 4. Pipeline both reads (first non-blocking) so DMAs can overlap.
        auto fx_rm = ttnn::to_layout(field_x_t, ttnn::Layout::ROW_MAJOR);
        auto fy_rm = ttnn::to_layout(field_y_t, ttnn::Layout::ROW_MAJOR);
        Finish(dev->mesh_command_queue());  // ensure both untilizes done
        auto mb_x = fx_rm.device_storage().get_mesh_buffer_leak_ownership();
        auto mb_y = fy_rm.device_storage().get_mesh_buffer_leak_ownership();
        // Zero-copy stash: keep these MeshBuffers alive so V21 EF can read
        // from them on-chip on the subsequent backward. The previous iter's
        // buffers get released here (shared_ptr replacement).
        latest_field_x_mb = mb_x;
        latest_field_y_mb = mb_y;
        latest_field_valid = true;
        bool did_direct = false;
        if (!skip_field_d2h) {
            const int padN = ((N + 31) / 32) * 32;
            const int padM = ((M + 31) / 32) * 32;
            if (fx_direct && fy_direct && padN == N && padM == M) {
                // ── Host-backward fast path: read the two field maps DIRECTLY
                // into the caller's output buffers (e.g. the numpy arrays the
                // CPU electric_force backward consumes). Eliminates the
                // redundant host→host field memcpy (fw_ms) entirely; the only
                // surviving cost is the unavoidable chip→host d2h.
                dev->mesh_command_queue().enqueue_read_mesh_buffer(fx_direct, mb_x, true);
                dev->mesh_command_queue().enqueue_read_mesh_buffer(fy_direct, mb_y, true);
                did_direct = true;
            } else {
                // Note: enqueue_read_mesh_buffer doesn't support non-blocking on
                // this version of tt-metal (use enqueue_read_shards for that).
                EnqueueReadMeshBuffer(dev->mesh_command_queue(), field_x, mb_x, true);
                EnqueueReadMeshBuffer(dev->mesh_command_queue(), field_y, mb_y, true);
                // Trim if padded (N not a multiple of 32). 2048 is, so no-op.
                if (padN != N) {
                    std::vector<float> tx(M * N), ty(M * N);
                    for (int r = 0; r < M; ++r) {
                        std::copy(field_x.begin() + r * padN, field_x.begin() + r * padN + N, tx.begin() + r * N);
                        std::copy(field_y.begin() + r * padN, field_y.begin() + r * padN + N, ty.begin() + r * N);
                    }
                    field_x = std::move(tx);
                    field_y = std::move(ty);
                }
            }
        }
        // else: field_x / field_y are left untouched by us. Caller is expected
        // to not consume them (V21 EF reads via latest_field_addrs() from chip).
        download_ms = ms_since(ts);
        return did_direct;
    }
};

// ═══════════════════════════════════════════════════════════════════
// IPC helpers
// ═══════════════════════════════════════════════════════════════════

static void flag_write(const std::string& path, const char* content) {
    std::string tmp = path + ".tmp";
    FILE* f = fopen(tmp.c_str(), "w");
    if (!f) { perror(("fopen " + tmp).c_str()); return; }
    fputs(content, f); fclose(f);
    rename(tmp.c_str(), path.c_str());
}

// ═══════════════════════════════════════════════════════════════════
// V19Engine — in-process engine class.
//
// `Impl` holds all the state that crosses the setup → per-iter boundary
// (the mesh device, MeshBuffers, MeshWorkloads, TTNN DCT solver,
// per-iter scratch vectors, V11 hist state). `Impl::Impl` is the port of
// the server's main() setup phase; `Impl::scatter` is the per-iter body.
// ═══════════════════════════════════════════════════════════════════

namespace v19 {

// Forward-declared in v19_engine.h. Defined here so it has access to all
// the TT-Metal / TTNN types pulled in by the includes above.
struct V19Engine::Impl {
    Impl(int M, int N, int NC_max,
         float xl, float yl, float xh, float yh,
         const std::string& gather_mode_arg);

    void set_initial_density(const float* id_normalized);

    void scatter(const float* px_in, const float* py_in,
                 const float* sx_in, const float* sy_in,
                 int32_t nc_actual_in,
                 float* density_out,
                 float* field_x_out, float* field_y_out,
                 ScatterTiming* timing_out);

    // ── Grid / config (set from ctor params) ──
    int    M_      = 0;
    int    N_      = 0;
    int    NC_max_ = 0;
    float  xl_=0, yl_=0, xh_=1e6f, yh_=1e6f;
    std::string gather_mode_;

    // ── Mode flags (set from GATHER_MODE) ──
    bool use_v11_=false, use_v11outer_=false, use_v18_=false,
         use_v18outer_=false, use_v19_=false;
    // V13/V14/etc are always false in this engine but kept for branch
    // compatibility with the ported setup body.
    bool use_v6_=false, use_v7_=false, use_v8_=false, use_v9_=false,
         use_v10_=false, use_v13_=false, use_v14_=false, use_v15_=false,
         use_v16_=false;

    // ── Derived sizing ──
    float    bsx_=0, bsy_=0, inv_ba_=0;
    int      nc_all_=0;
    uint32_t Mt_=0, M_tiles_=0, N_tiles_v11_=0;
    uint32_t soa_padded_=0;
    uint32_t hist_pgsz_v11_=0;

    // ── Device + queue (set in ctor; cq is a ptr because MeshCommandQueue&
    //   is non-rebindable). ──
    std::shared_ptr<MeshDevice> mesh_device_;
    MeshCommandQueue*           cq_ = nullptr;

    // ── DRAM buffers (V4 base + V11/V18/V19 state). ──
    // Names mirror the server's main() locals so the auto& aliases at the
    // top of scatter() can rebind without any body edits.
    std::shared_ptr<MeshBuffer> px_buf_, py_buf_, sx_buf_, sy_buf_;
    std::shared_ptr<MeshBuffer> contrib_buf_, density_buf_, strips_buf_;
    std::shared_ptr<MeshBuffer> tile_map_buf_, route_buf_, owned_lookup_buf_;
    std::shared_ptr<MeshBuffer> hist_buf_, shard_table_buf_, shard_reduce_buf_, drop_buf_;
    std::shared_ptr<MeshBuffer> v19_coord_buf_, v19_density_dram_;
    std::shared_ptr<MeshBuffer> v19_density_dram_outer_;
    std::shared_ptr<MeshBuffer> v31_route_buf_, v31_rcount_buf_;   // V31 forward-stash of (cell,bin,area)
    std::shared_ptr<MeshBuffer> v31_geom_buf_;   // per-cell geometry (128B/cell) for V29 prep-reuse
    std::vector<uint32_t> v31_first_tile_;   // per-core first tile (cbase = first_tile*1024)
    uint32_t v31_route_pg_ = 0, v31_max_tpc_ = 0, v31_nc_all_ = 0;
    uint32_t v31_geom_pg_ = 0;   // geom buffer page size (= soa_padded*128); 0 if off
    // V31_EF_GEOM: per-cell ratio (orig_area/clamped_area) for the density scatter. When the
    // forward uses the real clamped-size footprint (V31_EF_GEOM), the density must be ·ratio
    // (the orig workaround carried it implicitly). Buffer indexed by active cell (scatter order).
    std::shared_ptr<MeshBuffer> v31_ratio_buf_;
    std::vector<float> v31_ratio_data_;   // host staging (scatter order, padded)
    uint32_t v31_ratio_a_ = 0; bool v31_ef_geom_ = false, v31_ratio_uploaded_ = false;
    // V31 backward (built lazily on first compute_electric_force_v31)
    bool v31_bw_built_ = false;
    std::shared_ptr<MeshBuffer> v31_fx_buf_, v31_fy_buf_, v31_grad_buf_;
    uint32_t v31_field_pg_ = 0, v31_grad_pg_ = 0, v31_nbx_cap_ = 0, v31_slab_cells_ = 0;
    MeshWorkload v31_bw_wl_;
    MeshCoordinateRange v31_bw_dr_{MeshCoordinate{0,0}, MeshCoordinate{0,0}};
    KernelHandle v31_bw_k_ = 0;
    std::vector<float> v31_fxf_, v31_fyf_;
    std::vector<uint32_t> v31_gflat_, v31_rcount_host_;
    std::string kdir_;   // kernel source dir (mirror of constructor KDIR)
    uint32_t v19_density_slab_bins_outer_ = 0;
    uint32_t v19_density_dram_pgsz_outer_ = 0;

    // ── Workloads. ──
    MeshWorkload wl_scatter_, wl_gather_;
    MeshWorkload wl_v11_scatter_, wl_v11_accum_, wl_v11_hist_;
    MeshWorkload wl_v19_writeout_;

    // ── TTNN DCT solver. ──
    TTNNDCTSolver ttnn_solver_;
    bool          ttnn_solver_ready_ = false;

    // ── Per-iter scratch vectors (heap-allocated once in ctor). ──
    std::vector<float>    px_, py_, sx_, sy_;
    std::vector<float>    density_flat_, field_x_, field_y_;
    std::vector<uint32_t> v11_hist_data_, v11_global_count_;

    // ── V11 hist refresh state. ──
    bool     v11_dbg_first_ = true;
    uint64_t v11_iter_      = 0;

    // ── Initial-density add-back slot (set via set_initial_density). ──
    std::vector<float> initial_density_;
    bool initial_density_set_ = false;
};

// ═══════════════════════════════════════════════════════════════════
// V19Engine::Impl::Impl — ported from server_host.cpp's main() setup.
// Lines that originally read argv are replaced by ctor param mapping;
// IPC paths setup is stripped; everything else is unchanged.
// ═══════════════════════════════════════════════════════════════════

V19Engine::Impl::Impl(int M, int N, int NC_max,
                     float xl, float yl, float xh, float yh,
                     const std::string& gather_mode_arg)
    : M_(M), N_(N), NC_max_(NC_max),
      xl_(xl), yl_(yl), xh_(xh), yh_(yh),
      gather_mode_(gather_mode_arg)
{
    // Reject unsupported modes early.
    static const std::vector<std::string> kSupported = {
        "v11", "v11outer", "v11outer_auto", "v18", "v18outer", "v19"
    };
    bool supported = false;
    for (const auto& m : kSupported) if (gather_mode_arg == m) { supported = true; break; }
    if (!supported) {
        throw std::runtime_error(
            "V19Engine: unsupported gather_mode '" + gather_mode_arg +
            "'. Supported: v11, v11outer, v11outer_auto, v18, v18outer, v19.");
    }

    // Bind ctor params to the names the ported main() body expects.
    // (M, N, NC_max, xl, yl, xh, yh already match.)

    printf("[v19_engine] V19 in-process engine init\n");
    printf("[v19_engine] M=%d N=%d NC_max=%d\n", M, N, NC_max);
    fflush(stdout);

    // ── Gather mode: v13_fpu=tile-record + TRISC FPU matmul gather,
    //   v11=cell-centric tile-routed,
    //   v10=V6 scatter + bulk-async gather, v9=y-chunked scatter+SFPU gather,
    //   v8=SFPU (N≤512), v7=dense-scalar, v6=sparse ──
    // Override with env var GATHER_MODE=v6|v7|v8|v9|v10|v11|v11outer|v11outer_auto|v13_fpu|v14|v15|v16|auto
    // Rung 3: gather_mode comes from the ctor param (set by Python via
    // GATHER_MODE env var, but read in the wrapper, not here). Override
    // gather_mode_env so the existing dispatch below works unchanged.
    std::string gather_mode_env = gather_mode_arg;
    bool use_v16 = false;  // V16 Phase A: V15 kernels + hash-based tile ownership
    bool use_v15 = false;  // V15 hybrid: V13 scatter + NCRISC-parallel V15 gather
    bool use_v14 = false;  // V14 Architecture-A: writer-side FPU + reduce gather
    bool use_v13 = false;  // V13_fpu tile-record scatter + FPU matmul gather
    bool use_v11 = false;  // V11 cell-centric tile-routed (Phase 1: stub validator)
    bool use_v11outer = false;  // V11 Stage B': SFPU outer-product on scatter side
    bool use_v18 = false;  // V18: V11 + per-source hash-table pre-aggregation
    bool use_v18outer = false;  // V18-outer: V18 hash-agg + V11outer SFPU outer-product
    bool use_v19 = false;       // V19: direct L1 atomic-add gather (no separate gather kernel)
    bool use_v10 = false;  // V6 sparse scatter + V10 batched gather
    bool use_v9 = false;
    bool use_v8 = false;
    bool use_v7 = false;
    if (gather_mode_env == "v16") {
        // V16 Phase A: hash-based tile ownership (V15_HANDOFF §10.3 /
        // V16_PLAN.md §6.1 Path A). Reuses V15's gather kernels verbatim;
        // the only difference is that build_hash_ownership() decorrelates
        // tile_to_core from spatial position, so hot tiles get spread
        // across many cores instead of concentrated on one. This is the
        // smallest viable change that fixes V15's dense-2048 divergence.
        use_v16 = true;
        use_v15 = true;
        use_v13 = true;
    } else if (gather_mode_env == "v15") {
        // V15 hybrid: reuses V13's scatter program but installs a new gather
        // program that activates NCRISC as a parallel record-reader. Set
        // use_v13 too so the scatter side wires up correctly; the gather
        // branch below switches on use_v15.
        use_v15 = true;
        use_v13 = true;
    } else if (gather_mode_env == "v14") {
        use_v14 = true;
    } else if (gather_mode_env == "v13_fpu" || gather_mode_env == "v13") {
        use_v13 = true;
    } else if (gather_mode_env == "v11") {
        // V11 cell-centric tile-routed scatter+accumulate. Replaces V6 sparse
        // scatter and V10 bulk gather entirely.
        use_v11 = true;
    } else if (gather_mode_env == "v11outer") {
        // V11 Stage B': SFPU pre-computes 64 outer products per cell, so the
        // BRISC/NCRISC scatter side skips the inline ox*oy multiply. Uses
        // v4_outer_compute.cpp + v11_outer_scatter_*_dm.cpp. Everything else
        // (gather, hist, host dispatch shapes) is identical to V11.
        use_v11 = true;
        use_v11outer = true;
    } else if (gather_mode_env == "v11outer_auto") {
        // V11 with per-grid Stage B' routing. The 18-config sweep showed Stage
        // B' is a clean win at N>=2048 (mean -19.7% scatter) but regresses at
        // N=512 (+24%) due to SFPU dispatch overhead exceeding saved multiplies.
        // 1024 is mixed (small designs win, large lose). Threshold N>=2048 is
        // the only fully-safe routing per memory/v11_stage_bprime_sweep_outcome.
        use_v11 = true;
        use_v11outer = (N >= 2048);
        printf("[server] v11outer_auto: N=%d → %s\n",
               N, use_v11outer ? "Stage B' (v11outer)" : "V11 baseline");
    } else if (gather_mode_env == "v18") {
        // V18: V11 + per-source hash-table pre-aggregation on the scatter
        // side. Reuses V11's gather + tile_to_core ownership verbatim; only
        // the scatter kernel files differ. See docs/V18_HASHAGG_HANDOFF.md.
        use_v11 = true;
        use_v18 = true;
        use_v11outer = false;
    } else if (gather_mode_env == "v18outer") {
        // V18-outer: V18 hash-agg + V11outer SFPU outer-product compute.
        use_v11 = true;
        use_v18 = true;
        use_v18outer = true;
        use_v11outer = true;  // for v4_outer_compute selection
    } else if (gather_mode_env == "v19") {
        // V19: direct L1 atomic-add gather. Scatter kernel atomic-adds
        // fixed-point area into the owning core's L1 density slab; no
        // separate gather kernel. Phase 1 = no V18 dedup, direct atomics.
        use_v11 = true;
        use_v19 = true;
        use_v11outer = false;
        use_v18 = false;
    } else if (gather_mode_env == "v10") {
        use_v10 = true;
    } else if (gather_mode_env == "v9") {
        use_v9 = true;
    } else if (gather_mode_env == "v8") {
        use_v8 = ((uint32_t)N <= 512u);
        use_v9 = !use_v8;  // fall back to V9 (chunked dense + SFPU/scalar gather) for N>512
    } else if (gather_mode_env == "v7") {
        use_v7 = true;
    } else if (gather_mode_env == "v6") {
        // explicit v6
    } else {
        // auto: use V8 when N≤512, else V9 (chunked scatter) for any larger grid.
        use_v8 = ((uint32_t)N <= 512u);
        use_v9 = !use_v8;
    }
    const char* gmode_str = use_v16 ? "v16-hash-ownership"
                          : use_v15 ? "v15-hybrid"
                          : use_v14 ? "v14-arch-a"
                          : use_v13 ? "v13-fpu-matmul"
                          : use_v19 ? "v19-l1-atomic"
                          : use_v18outer ? "v18-outer-hashagg+stageB"
                          : use_v18 ? "v18-hashagg"
                          : use_v11outer ? "v11-outer-stage-b-prime"
                          : use_v11 ? "v11-tile-routed"
                          : use_v10 ? "v10-bulk"
                          : use_v9 ? "v9-chunked"
                          : use_v8 ? "v8-sfpu"
                          : use_v7 ? "v7-dense"
                          : "v6-sparse";
    printf("[v19_engine] gather_mode=%s\n", gmode_str);
    fflush(stdout);

    // Save mode flags into members so scatter() can branch on them.
    use_v11_      = use_v11;
    use_v11outer_ = use_v11outer;
    use_v18_      = use_v18;
    use_v18outer_ = use_v18outer;
    use_v19_      = use_v19;
    use_v7_       = use_v7;
    use_v8_       = use_v8;
    use_v9_       = use_v9;
    use_v10_      = use_v10;
    use_v13_      = use_v13;
    use_v14_      = use_v14;
    use_v15_      = use_v15;
    use_v16_      = use_v16;
    use_v6_ = !(use_v7 || use_v8 || use_v9 || use_v10 || use_v11 || use_v13 ||
                use_v14 || use_v15 || use_v16);

    std::string KDIR;
#ifndef DENSITY_KERNEL_DIR
    KDIR = std::string(getenv("TT_METAL_HOME") ? getenv("TT_METAL_HOME") : ".") +
           "/../experiments/density_scatter/tt_metal/kernels/";
#else
    KDIR = DENSITY_KERNEL_DIR;
#endif
    kdir_ = KDIR;

    // ── IPC paths setup removed (Rung 3 has no IPC layer). ──

    // ── Open device ───────────────────────────────────────────────
    mesh_device_ = MeshDevice::create_unit_mesh(0);
    auto& mesh_device = mesh_device_;     // alias so ported body uses local name
    printf("[v19_engine] TT device opened.\n"); fflush(stdout);

    cq_ = &mesh_device->mesh_command_queue();
    MeshCommandQueue& cq = *cq_;          // alias for ported body

    // ── Grid and sizing ───────────────────────────────────────────
    constexpr uint32_t MAX_OVERLAP       = 8u;
    constexpr uint32_t MAX_BINS_PER_CELL = 12u;
    constexpr uint32_t V4_HEADER_BYTES   = 512u;
    constexpr uint32_t TILE_ELEMS        = 1024u;
    constexpr uint32_t TILE_BYTES        = TILE_ELEMS * sizeof(float);

    float bsx=(xh-xl)/M, bsy=(yh-yl)/N;
    float inv_ba = 1.0f/(bsx*bsy);
    uint32_t Mt = (uint32_t)M / 32u;

    CoreCoord grid = mesh_device->compute_with_storage_grid_size();
    int nc_all = (int)(grid.x * grid.y);
    printf("[server] grid = %ux%u  nc_all = %d (odd=%d)\n",
           (unsigned)grid.x, (unsigned)grid.y, nc_all, nc_all & 1);
    fflush(stdout);

    auto all_ccs = get_cores(grid, nc_all);
    auto mt_ccs  = get_cores(grid, (int)Mt);
    CoreRangeSet all_crs = cores_to_crs(all_ccs);
    CoreRangeSet mt_crs  = cores_to_crs(mt_ccs);
    MeshCoordinateRange device_range(mesh_device->shape());

    uint32_t n_tiles_total = ((uint32_t)NC_max + TILE_ELEMS-1u) / TILE_ELEMS;
    uint32_t base_tpc      = n_tiles_total / (uint32_t)nc_all;
    uint32_t rem_tpc       = n_tiles_total % (uint32_t)nc_all;
    uint32_t max_tpc       = base_tpc + (rem_tpc>0 ? 1u : 0u);
    uint32_t max_contrib   = max_tpc * TILE_ELEMS * MAX_BINS_PER_CELL;
    // Cap max_contrib so V6 scatter L1 scratch fits in 1536 KB.
    // The kernel has overflow protection (if total < max_contrib), so capping is safe;
    // standard cells with 2*bsx clamping hit at most 3×3 = 9 bins so rarely overflow.
    // CBs 0–21 consume ~220 KB of L1 alongside CB_SCRATCH; subtract that overhead.
    // sc_scratch = V4_HEADER_BYTES + max_contrib*8 + max_sorted*8 + (128+M)*4
    //   where max_sorted = max_contrib + Mt*8
    //   → sc_scratch = 512 + 16*max_contrib + 64*Mt + (128+M)*4
    // Budget: 1536 KB − 220 KB (CB 0-21 overhead) = 1316 KB for sc_scratch
    {
        constexpr uint32_t L1_SC_BUDGET = (1536u - 220u) * 1024u;
        uint32_t overhead = V4_HEADER_BYTES + Mt * 64u + (128u + (uint32_t)M) * sizeof(uint32_t);
        uint32_t max_contrib_cap = (L1_SC_BUDGET - overhead) / 16u;
        if (max_contrib > max_contrib_cap) {
            printf("[server] WARNING: max_contrib %u > L1 cap %u; capping (some large-cell contribs may be dropped)\n",
                   max_contrib, max_contrib_cap);
            max_contrib = max_contrib_cap;
        }
    }
    uint32_t max_sorted    = max_contrib + Mt * 8u;
    uint32_t avg_per_bucket= max_contrib / Mt + 1u;
    uint32_t max_bucket    = std::max(512u, std::min(max_contrib, avg_per_bucket*4u));

    uint32_t part_base_gath = 32u;   // 32 x-cols per gather core
    uint32_t contrib_pgsz  = (V4_HEADER_BYTES + max_sorted*8u + 31u) & ~31u;
    uint32_t contrib_bytes = (uint32_t)nc_all * contrib_pgsz;
    uint32_t soa_padded    = n_tiles_total * TILE_ELEMS;
    uint32_t soa_bytes     = n_tiles_total * TILE_BYTES;
    uint32_t tile_pgsz     = TILE_BYTES;

    uint32_t density_pgsz  = (uint32_t)N * sizeof(float);
    uint32_t density_bytes = (uint32_t)M * density_pgsz;

    // V7/V8: dense strip buffer — each scatter core writes Mt strips of (32*N floats)
    // Page indexed as [src_idx * Mt + gc_idx], each page = strip_pgsz bytes
    // V7: column-major strips; V8: row-major strips (same size, different layout)
    uint32_t strip_pgsz  = 32u * (uint32_t)N * sizeof(float);
    uint32_t strips_bytes = (use_v7 || use_v8 || use_v9)
        ? ((uint32_t)nc_all * Mt * strip_pgsz)
        : 4u;  // dummy 4-byte placeholder when V6 is used

    // ── DRAM buffers ──────────────────────────────────────────────
    auto make_buf = [&](uint32_t sz, uint32_t pg) {
        DeviceLocalBufferConfig cfg{.page_size=pg,.buffer_type=BufferType::DRAM};
        ReplicatedBufferConfig  rcfg{.size=sz};
        return MeshBuffer::create(rcfg, cfg, mesh_device.get());
    };

    auto px_buf      = make_buf(soa_bytes,   tile_pgsz);
    auto py_buf      = make_buf(soa_bytes,   tile_pgsz);
    auto sx_buf      = make_buf(soa_bytes,   tile_pgsz);
    auto sy_buf      = make_buf(soa_bytes,   tile_pgsz);
    auto contrib_buf = make_buf(contrib_bytes, contrib_pgsz);
    auto density_buf = make_buf(density_bytes, density_pgsz);
    // V7/V8/V9 strips buffer (only allocated for dense modes; placeholder otherwise)
    auto strips_buf  = make_buf(strips_bytes, (use_v7 || use_v8 || use_v9) ? strip_pgsz : 4u);

    // ── CB helpers ────────────────────────────────────────────────
    auto make_cb_all = [&](Program& p, uint32_t idx, uint32_t n_tiles) {
        CreateCircularBuffer(p, all_crs,
            CircularBufferConfig(n_tiles*tile_pgsz, {{idx,tt::DataFormat::Float32}})
                .set_page_size(idx, tile_pgsz));
    };
    auto make_cb_mt  = [&](Program& p, uint32_t idx, uint32_t n_tiles) {
        CreateCircularBuffer(p, mt_crs,
            CircularBufferConfig(n_tiles*tile_pgsz, {{idx,tt::DataFormat::Float32}})
                .set_page_size(idx, tile_pgsz));
    };
    auto make_cb_mt_raw = [&](Program& p, uint32_t idx, uint32_t bytes) {
        CreateCircularBuffer(p, mt_crs,
            CircularBufferConfig(bytes, {{idx,tt::DataFormat::Float32}})
                .set_page_size(idx, bytes));
    };

    // ══════════════════════════════════════════════════════════════
    // PROGRAM 1: V4 Scatter (all cores)
    // ══════════════════════════════════════════════════════════════
    Program prog_scatter = CreateProgram();
    for (uint32_t i=0; i<4; ++i) make_cb_all(prog_scatter, i, 2);
    make_cb_all(prog_scatter, 4, 1); make_cb_all(prog_scatter, 5, 1);
    for (uint32_t j=0; j<MAX_OVERLAP; ++j) {
        make_cb_all(prog_scatter, 6+j, 1); make_cb_all(prog_scatter, 14+j, 1);
    }
    // V9 chunking: pick the smallest n_chunks (power of 2) such that the
    // per-chunk dense L1 buffer fits comfortably under L1_BUDGET.
    uint32_t v9_n_chunks = 1;
    uint32_t v9_chunk_height = (uint32_t)N;
    if (use_v9) {
        constexpr uint32_t L1_DENSE_BUDGET = 1024u * 1024u;  // ~1 MB for partial map
        uint32_t target_floats = L1_DENSE_BUDGET / sizeof(float);
        uint64_t per_chunk_floats = (uint64_t)Mt * 32ull * (uint64_t)N;
        while ((per_chunk_floats / v9_n_chunks) > target_floats && v9_n_chunks < 64u) {
            v9_n_chunks *= 2u;
        }
        // Make chunk_height divide nby exactly; round n_chunks up to nearest power-of-2 divisor of N.
        while (((uint32_t)N % v9_n_chunks) != 0u && v9_n_chunks < (uint32_t)N) v9_n_chunks <<= 1u;
        v9_chunk_height = (uint32_t)N / v9_n_chunks;
        printf("[server] V9: n_chunks=%u chunk_height=%u (per-core scratch ≈ %u KB)\n",
               v9_n_chunks, v9_chunk_height,
               (Mt * 32u * v9_chunk_height * 4u + (uint32_t)M * 4u) / 1024u);
        fflush(stdout);
    }

    uint32_t sc_scratch;
    if (use_v9) {
        // V9: partial[Mt][32][chunk_height] + bx2dest[M]
        sc_scratch = (Mt * 32u * v9_chunk_height * sizeof(float)
                      + (uint32_t)M * sizeof(uint32_t) + 31u) & ~31u;
    } else if (use_v7 || use_v8) {
        // V7/V8: partial_strips[Mt][32*N] floats + bx2dest[M] uint32
        // Row-major (V8) and column-major (V7) strips have same total size.
        sc_scratch = (Mt * 32u * (uint32_t)N * sizeof(float)
                      + (uint32_t)M * sizeof(uint32_t) + 31u) & ~31u;
    } else {
        // V6: header + unsorted + sorted + running/bx2dest
        sc_scratch = (V4_HEADER_BYTES + max_contrib*8u + max_sorted*8u
                      + (128u + (uint32_t)M)*sizeof(uint32_t) + 31u) & ~31u;
    }
    printf("[server] scatter L1 scratch = %u KB (limit 1536 KB)\n",
           sc_scratch / 1024u);
    if (sc_scratch >= 1536u * 1024u) {
        printf("[server] FATAL: scatter L1 scratch %u KB >= 1536 KB even after capping\n",
               sc_scratch / 1024u);
        fflush(stdout); std::exit(1);
    }
    fflush(stdout);
    CreateCircularBuffer(prog_scatter, all_crs,
        CircularBufferConfig(sc_scratch, {{24u,tt::DataFormat::Float32}})
            .set_page_size(24u, sc_scratch));

    std::map<std::string,std::string> v4_defs = {
        {"V4_BSX_F",      to_float_literal(bsx)},
        {"V4_BSY_F",      to_float_literal(bsy)},
        {"V4_INV_BSX_F",  to_float_literal(1.0f/bsx)},
        {"V4_INV_BSY_F",  to_float_literal(1.0f/bsy)},
        {"V4_XL_F",       to_float_literal(xl)},
        {"V4_YL_F",       to_float_literal(yl)},
        {"V4_MAX_OVERLAP", std::to_string(MAX_OVERLAP)},
    };
    std::vector<UnpackToDestMode> sc_unpack(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    for (int i=0; i<4; ++i) sc_unpack[i] = UnpackToDestMode::UnpackToDestFp32;

    std::string reader_kernel  = use_v9 ? "v9_reader_chunked.cpp"  : "v4_reader.cpp";
    std::string compute_kernel = use_v9 ? "v9_compute_chunked.cpp" : "v4_compute.cpp";
    auto rk = CreateKernel(prog_scatter, KDIR+reader_kernel, all_crs,
        DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
    auto ck = CreateKernel(prog_scatter, KDIR+compute_kernel, all_crs,
        ComputeConfig{.fp32_dest_acc_en=true,.unpack_to_dest_mode=sc_unpack,
                      .math_approx_mode=false,.defines=v4_defs});

    // V7/V8: column-major dense strips (V8 reuses V7 scatter)
    // V9: y-chunked column-major scatter
    // V6: sparse
    std::string ncrisc_kernel = use_v9 ? "v9_ncrisc_scatter_chunked.cpp"
                              : (use_v7 || use_v8) ? "v4_ncrisc_scatter_dense.cpp"
                              : "v4_ncrisc_scatter.cpp";
    auto sk = CreateKernel(prog_scatter, KDIR+ncrisc_kernel, all_crs,
        DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});

    uint32_t px_a=(uint32_t)px_buf->address(), py_a=(uint32_t)py_buf->address();
    uint32_t sx_a=(uint32_t)sx_buf->address(), sy_a=(uint32_t)sy_buf->address();
    uint32_t ca  =(uint32_t)contrib_buf->address();
    uint32_t sa  =(uint32_t)strips_buf->address();
    for (int c=0; c<nc_all; ++c) {
        auto cc = all_ccs[c];
        uint32_t my_n = base_tpc + ((uint32_t)c < rem_tpc ? 1u : 0u);
        uint32_t first = (uint32_t)c * base_tpc + std::min((uint32_t)c, rem_tpc);
        if (use_v9) {
            SetRuntimeArgs(prog_scatter, rk, cc,
                {px_a,py_a,sx_a,sy_a,tile_pgsz,first,my_n,v9_n_chunks});
            SetRuntimeArgs(prog_scatter, ck, cc, {my_n, v9_n_chunks});
            SetRuntimeArgs(prog_scatter, sk, cc,
                {sa, strip_pgsz, (uint32_t)c, my_n,
                 (uint32_t)M, (uint32_t)N, Mt, 32u, v9_n_chunks, v9_chunk_height});
        } else {
        SetRuntimeArgs(prog_scatter, rk, cc, {px_a,py_a,sx_a,sy_a,tile_pgsz,first,my_n});
        SetRuntimeArgs(prog_scatter, ck, cc, {my_n});
        if (use_v7 || use_v8) {
            // v4_ncrisc_scatter_dense args: strips_dram, strip_pgsz, my_page_id,
            //   n_tiles, nbx, nby, nc_gath, part_base(unused), 0, 0
            SetRuntimeArgs(prog_scatter, sk, cc,
                {sa, strip_pgsz, (uint32_t)c, my_n,
                 (uint32_t)M, (uint32_t)N, Mt, 32u, 0u, 0u});
        } else {
            SetRuntimeArgs(prog_scatter, sk, cc,
                {ca, contrib_pgsz, (uint32_t)c, my_n,
                 (uint32_t)M, (uint32_t)N, (uint32_t)Mt,
                 part_base_gath, 0u, max_contrib});
        }
        }  // end !use_v9 branch
    }
    MeshWorkload wl_scatter;
    wl_scatter.add_program(device_range, std::move(prog_scatter));

    // ══════════════════════════════════════════════════════════════
    // PROGRAM 2: Gather (Mt cores, 32 x-cols each)
    //   V6: sparse-pull (v6_gather_density_only)
    //   V7: dense-strip (v7_gather_dense)
    // ══════════════════════════════════════════════════════════════
    Program prog_gather = CreateProgram();
    uint32_t inv_ba_u32; std::memcpy(&inv_ba_u32, &inv_ba, 4);
    uint32_t da = (uint32_t)density_buf->address();

    // V9 dispatches to V8 SFPU gather if it fits L1, else falls back to V7 BRISC gather.
    bool v9_use_v8_gather = false;
    if (use_v9) {
        // V8 gather L1 estimate: 6 × N_TILES × 4096 + small overhead.
        // For N=1024 (N_TILES=32): 786 KB. For N=2048 (N_TILES=64): 1572 KB.
        // We squeeze N=2048 by single-buffering CB_RAW (CB_RAW = N_TILES instead of 2*N_TILES).
        uint32_t N_TILES_g = (uint32_t)N / 32u;
        uint32_t v8_l1_full   = (2u + 1u + 2u + 1u) * N_TILES_g * 4096u;  // 786KB @ 1024
        uint32_t v8_l1_compact= (1u + 1u + 2u + 1u) * N_TILES_g * 4096u;  // 1310KB @ 2048
        v9_use_v8_gather = (v8_l1_full <= 1024u * 1024u + 256u * 1024u) // 1.25MB margin
                       || (v8_l1_compact <= 1280u * 1024u);
    }
    bool effective_v8_gather = use_v8 || (use_v9 && v9_use_v8_gather);
    bool effective_v7_gather = use_v7 || (use_v9 && !v9_use_v8_gather);

    if (effective_v8_gather) {
        // V8: SFPU-accelerated gather via TRISC + BRISC split.
        // For V9 mode at N=2048, use single-buffer CB_RAW to fit L1.
        uint32_t N_TILES = (uint32_t)N / 32u;
        uint32_t cb_raw_factor = (use_v9 && N_TILES > 32u) ? 1u : 2u;

        // CB_RAW (c_0): double-buffered strip input (BRISC→TRISC) — single-buffer for v9 N=2048
        make_cb_mt(prog_gather, 0, cb_raw_factor * N_TILES);
        // CB_TILED (c_1): tilized intermediary for sources 1..nc_src-1
        make_cb_mt(prog_gather, 1, N_TILES);
        // CB_ACCUM (c_2): ping-pong accumulator (TRISC internal)
        make_cb_mt(prog_gather, 2, 2u * N_TILES);
        // CB_OUT_TILES (c_3): final accumulated tiles in tile-face format
        // (TRISC→BRISC). BRISC decodes face format and normalizes by inv_ba.
        make_cb_mt(prog_gather, 3, N_TILES);

        // BRISC DataMovement kernel: DMA strips in, write density out
        auto gk8_dm = CreateKernel(prog_gather, KDIR+"v8_gather_dm.cpp", mt_crs,
            DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,
                               .noc=NOC::RISCV_0_default});

        // TRISC Compute kernel: tilize + SFPU accumulate → tile-format output
        std::map<std::string,std::string> v8_defs = {
            {"V8_NC_SRC",   std::to_string((uint32_t)nc_all)},
            {"V8_N_TILES",  std::to_string(N_TILES)},
        };
        std::vector<UnpackToDestMode> v8_unpack(NUM_CIRCULAR_BUFFERS,
                                                UnpackToDestMode::Default);
        auto gk8_cmp = CreateKernel(prog_gather, KDIR+"v8_gather_compute.cpp", mt_crs,
            ComputeConfig{.fp32_dest_acc_en=true,
                          .unpack_to_dest_mode=v8_unpack,
                          .math_approx_mode=false,
                          .defines=v8_defs});

        for (uint32_t c = 0; c < Mt; ++c) {
            SetRuntimeArgs(prog_gather, gk8_dm, mt_ccs[c], {
                sa, strip_pgsz,         // strips buffer
                da, density_pgsz,       // density output
                c,                      // my_id (gather core index)
                Mt,                     // nc_gath
                (uint32_t)nc_all,       // nc_src (scatter cores)
                (uint32_t)N,            // nby
                N_TILES,                // nby/32
                inv_ba_u32,             // 1/(bsx*bsy) as float bits
            });
            // Compute kernel uses only JIT defines; no runtime args needed.
            SetRuntimeArgs(prog_gather, gk8_cmp, mt_ccs[c], {});
        }
    } else if (effective_v7_gather) {
        // V7: incoming strip + accum strip; no header buffer needed
        uint32_t v7_scratch = 2u * strip_pgsz + (uint32_t)M * sizeof(uint32_t);
        v7_scratch = (v7_scratch + 31u) & ~31u;
        make_cb_mt_raw(prog_gather, 24, v7_scratch);

        auto gk7 = CreateKernel(prog_gather, KDIR+"v7_gather_dense.cpp", mt_crs,
            DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,
                               .noc=NOC::RISCV_0_default});
        for (uint32_t c=0; c<Mt; ++c) {
            SetRuntimeArgs(prog_gather, gk7, mt_ccs[c], {
                sa, strip_pgsz,         // strips buffer
                da, density_pgsz,       // density output
                c, Mt,                  // my_id, nc_gath
                (uint32_t)nc_all,       // nc_src (scatter cores)
                (uint32_t)N,            // nby
                inv_ba_u32,             // 1/(bsx*bsy)
            });
        }
    } else if (use_v10) {
        // V10: V6 sparse scatter + bulk-async-read gather.
        // L1 layout: headers (n_src*512) + tuples (SRC_CHUNK*max_bucket*8) + accum (32*N*4)
        constexpr uint32_t V10_HEADER_BYTES = 512u;
        constexpr uint32_t V10_SRC_CHUNK    = 16u;
        uint32_t v10_scratch = (uint32_t)nc_all * V10_HEADER_BYTES
                             + V10_SRC_CHUNK * max_bucket * 8u
                             + 32u * (uint32_t)N * sizeof(float);
        v10_scratch = (v10_scratch + 31u) & ~31u;
        printf("[server] V10 gather L1 scratch = %u KB (limit 1536 KB)\n", v10_scratch / 1024u);
        if (v10_scratch >= 1536u * 1024u) {
            printf("[server] FATAL: V10 scratch %u KB exceeds L1 budget\n", v10_scratch/1024u);
            fflush(stdout); std::exit(1);
        }
        fflush(stdout);
        make_cb_mt_raw(prog_gather, 24, v10_scratch);

        auto gk10 = CreateKernel(prog_gather, KDIR+"v10_gather_density.cpp", mt_crs,
            DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,
                               .noc=NOC::RISCV_0_default});
        for (uint32_t c=0; c<Mt; ++c) {
            uint32_t col_start = c * 32u;
            SetRuntimeArgs(prog_gather, gk10, mt_ccs[c], {
                ca, contrib_pgsz,
                da, density_pgsz,
                c, col_start, 32u,
                (uint32_t)N,
                (uint32_t)nc_all,
                max_bucket,
                inv_ba_u32,
            });
        }
    } else {
        // V6: sparse pull gather
        uint32_t gath_scratch = (V4_HEADER_BYTES + max_bucket*8u + 31u) & ~31u;
        gath_scratch += 32u * (uint32_t)N * sizeof(float);
        gath_scratch  = (gath_scratch + 31u) & ~31u;
        make_cb_mt_raw(prog_gather, 24, gath_scratch);

        auto gk6 = CreateKernel(prog_gather, KDIR+"v6_gather_density_only.cpp", mt_crs,
            DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,
                               .noc=NOC::RISCV_0_default});
        for (uint32_t c=0; c<Mt; ++c) {
            uint32_t col_start = c * 32u;
            SetRuntimeArgs(prog_gather, gk6, mt_ccs[c], {
                ca, contrib_pgsz,
                da, density_pgsz,
                c, col_start, 32u,
                (uint32_t)N,
                (uint32_t)nc_all,
                max_bucket,
                inv_ba_u32,
            });
        }
    }

    MeshWorkload wl_gather;
    wl_gather.add_program(device_range, std::move(prog_gather));

    // ══════════════════════════════════════════════════════════════
    // PROGRAM 3 + 4 (V11 only): cell-centric tile-routed scatter+accum.
    //   prog_v11_scatter: v4_reader (BRISC) + v4_compute (TRISC) +
    //                     v11_scatter_dm (NCRISC) — routes tuples via
    //                     route_buf to tile owners.
    //   prog_v11_accum:   v11_accum_dm (BRISC) — reads route_buf[*][me],
    //                     accumulates owned tiles, writes density_buf.
    // ══════════════════════════════════════════════════════════════
    uint32_t M_tiles = (uint32_t)M / 32u;
    uint32_t N_tiles_v11 = (uint32_t)N / 32u;
    std::shared_ptr<MeshBuffer> tile_map_buf, route_buf, owned_lookup_buf, hist_buf, shard_table_buf, shard_reduce_buf, drop_buf;
    // V19: hoist to outer scope so the per-iter readback can see them.
    uint32_t v19_density_slab_bins_outer = 0u;
    uint32_t v19_density_dram_pgsz_outer = 0u;
    std::shared_ptr<MeshBuffer> v19_density_dram_outer;
    MeshWorkload wl_v19_writeout;
    bool v19_writeout_built = false;
    MeshWorkload wl_v11_scatter, wl_v11_accum, wl_v11_hist;
    // wl_v11_accum now performs the full gather (accum + reduce_a + reduce_bc)
    // in a single kernel using NOC semaphores for shard sync — keeping all 3
    // phases in one program saves 2 host Finish() barriers (~16ms).
    std::vector<uint16_t> tile_to_core_v11;
    std::vector<std::vector<uint32_t>> core_to_tiles_v11;
    uint32_t tile_map_bytes_v11 = 0;
    uint32_t tile_map_pgsz_v11  = 0;
    uint32_t route_pgsz_v11     = 0;
    uint32_t owned_lookup_pgsz_v11 = 0;
    uint32_t v11_max_per_page_tuples = 0;
    uint32_t hist_pgsz_v11 = 0;
    uint32_t shard_table_pgsz_v11 = 0;
    uint32_t shard_reduce_pgsz_v11 = 0;
    uint32_t v11_max_hot_tiles = 0;
    // Constants for sharding
    constexpr uint32_t SHARD_BYTES = 16u;       // bytes per tile entry: byte0=K, 1..7=alts, 8..11=hot_tile_seq, 12..15=pad
    // MAX_K (env-tunable, default 8). Hard cap on how many cores a hot tile
    // can be sharded across. Bumping requires SHARD_BYTES is wide enough:
    // 16 bytes = 1 K + 7 alt cores + 4 hot_seq + 4 pad → MAX_K hard limit is 8.
    // For larger K we'd need a wider entry; keep ≤ 8 unless SHARD_BYTES grows.
    auto env_uint = [](const char* name, uint32_t def) -> uint32_t {
        const char* s = getenv(name);
        if (s == nullptr || s[0] == '\0') return def;
        int v = atoi(s);
        return v <= 0 ? def : (uint32_t)v;
    };
    const uint32_t MAX_K = std::min(8u, env_uint("V11_MAX_K", 8u));
    // HOT_THRESHOLD env-tunable. Default is effectively infinite (no sharding):
    // The 2026-05-21 ON-vs-OFF sweep on all 6 2048-grid configs showed that the
    // K=8 sharding scheme is net negative on 5 of 6 configs (wall-time loss of
    // 4-30 sec per run) because the iter-0 shard_table becomes stale as cells
    // migrate, and the small fp32 reorder it causes pushes DREAMPlace's
    // convergence onto longer paths (sometimes to worse final HPWL too).
    // Set V11_HOT_THRESHOLD=5000 (the old default) to re-enable for designs
    // that benefit (e.g. bigblue3_2048 saves ~14s with sharding ON).
    // See memory/v11_gather_sharding_investigation.md for the full analysis.
    const uint32_t HOT_THRESHOLD = env_uint("V11_HOT_THRESHOLD", 100000000u);
    constexpr uint32_t TILE_BYTES_v11 = 32u * 32u * 4u;  // 4096 bytes per tile
    // CB slot headroom for shard slots (above n_owned_max). Same value used by
    // initial accum build and every periodic refresh so the JIT'd kernel binary
    // is identical and the cache hits.
    // Env-tunable (default 2*MAX_K=16). Each slot is 4KB; max useful is ~200
    // before L1 budget overflows.
    const uint32_t V11_CB_SLOT_HEADROOM = env_uint("V11_CB_SLOT_HEADROOM", 2u * MAX_K);

    // V18 hash-table sizing (see docs/V18_HASHAGG_HANDOFF.md). hash_bits=15 →
    // 32K slots × 8 bytes = 256 KB per RISC = 512 KB per Tensix core for the
    // two scatter RISCs. Clamp to [12, 16] (4K..64K slots) so slot indices fit
    // in uint16 for the dirty-list optimization. FATAL guard catches over-
    // budget configurations.
    const uint32_t V18_HASH_BITS = std::min(16u, std::max(12u, env_uint("V18_HASH_BITS", 16u)));
    // env_bool: explicit 0/1 parse (env_uint treats 0 as "unset", which is wrong for boolean toggles).
    auto env_bool = [](const char* name, bool def) -> bool {
        const char* s = getenv(name);
        if (s == nullptr || s[0] == '\0') return def;
        return s[0] != '0';  // "0" → false, anything else → true
    };
    // Default at hash_bits=16: bf16 area (saves L1) + dirty list (faster flush).
    // bigblue1_2048 verified: bf16+dirty fits 1024 KB/core L1 and is ~7% faster
    // than bf16+no_dirty. Smaller hash_bits keeps fp32+dirty (current).
    const uint32_t V18_BF16_AREA = env_bool("V18_BF16_AREA", V18_HASH_BITS >= 16) ? 1u : 0u;
    const uint32_t V18_NO_DIRTY  = env_bool("V18_NO_DIRTY",  false) ? 1u : 0u;
    // Dirty-list size: 0 if disabled, else 2 bytes/slot.
    const uint32_t v18_dirty_bytes_per_risc = V18_NO_DIRTY ? 0u : (1u << V18_HASH_BITS) * 2u;
    if (use_v18) {
        const uint32_t slot_bytes = V18_BF16_AREA ? 6u : 8u;
        const uint32_t v18_table_bytes_per_risc = (1u << V18_HASH_BITS) * slot_bytes;
        const uint32_t v18_total_per_risc = v18_table_bytes_per_risc + v18_dirty_bytes_per_risc;
        printf("[server] V18: hash_bits=%u slots=%u slot_bytes=%u table_per_risc=%u KB dirty_per_risc=%u KB total_per_core=%u KB bf16_area=%u no_dirty=%u\n",
               V18_HASH_BITS, 1u << V18_HASH_BITS, slot_bytes,
               v18_table_bytes_per_risc >> 10,
               v18_dirty_bytes_per_risc >> 10,
               (v18_total_per_risc * 2u) >> 10,
               V18_BF16_AREA, V18_NO_DIRTY);
        if (v18_total_per_risc * 2u > 1024u * 1024u) {
            printf("[server] FATAL: V18 L1 use (2 × %u KB) exceeds 1024 KB budget. "
                   "Lower V18_HASH_BITS (current=%u) or set V18_BF16_AREA=1 V18_NO_DIRTY=1.\n",
                   v18_total_per_risc >> 10, V18_HASH_BITS);
            fflush(stdout);
            std::exit(1);
        }
        fflush(stdout);
    }

    // Per-core shard information (populated in v11_dbg_first from real shard_table)
    struct PerCoreShardInfo {
        std::vector<uint32_t> primary_tile_ids;
        std::vector<uint8_t>  primary_K;
        std::vector<uint32_t> primary_hot_seq;
        struct ShardEntry { uint32_t tile_id; uint32_t hot_tile_seq; uint8_t shard_idx_in_K; };
        std::vector<ShardEntry> shard_entries;
    };
    std::vector<PerCoreShardInfo> per_core_v11;
    uint32_t v11_dense_offset_bytes = 0u;  // offset of dense from CB_SCRATCH base (set below)
    uint32_t n_owned_max = 0;              // hoisted: needed by the per-iter refresh path

    // ── V13_fpu declarations (parallel to V11) ───────────────────────────────
    // V13 emits 40-byte tile-records (≤4 per cell) instead of V11's 8-byte
    // bin-tuples (≤64 per cell). Gather uses a TRISC FPU matmul to compute
    // the 8×8 outer-product `ox[j]*oy[k]` per record. Density is bf16 in DRAM
    // and UNNORMALIZED (host must divide by bin_area on readback).
    constexpr uint32_t V13_MAX_OVERLAP    = 8u;
    constexpr uint32_t V13_PAGE_HDR_BYTES = 64u;
    constexpr uint32_t V13_RECORD_BYTES   = 40u;
    constexpr uint32_t V13_RECORDS_CAP    = 4096u;  // per (writer, receiver) page
    constexpr uint32_t V13_MAX_IN_FLIGHT  = 16u;
    constexpr uint32_t V13_SHARED_BYTES_  = 4u + 4u + 8u*4u + 8u*4u;  // ScatterShared
    constexpr uint32_t V13_OV_BUF_BYTES   = 128u;   // 128 B per RISC overflow ring
    constexpr uint32_t V13_K_BATCH          = 32u;
    constexpr uint32_t V13_N_INFLIGHT       = 4u;          // OPT-1: 4-tile push batching
    constexpr uint32_t V13_PUSH_BATCH       = V13_N_INFLIGHT * V13_K_BATCH;  // 128
    constexpr uint32_t V13_CB_OXY_DEPTH     = 8u;          // ≥ 2 × N_INFLIGHT for pipelining
    constexpr uint32_t V13_MAX_READ_RECORDS = 64u;
    constexpr uint32_t TILE_BF16_BYTES_v13 = 32u * 32u * 2u;
    std::shared_ptr<MeshBuffer> tile_map_buf_v13, route_buf_v13, owned_lookup_buf_v13,
                                overflow_buf_v13, density_buf_v13, v15_stats_buf,
                                v15_spill_buf;
    uint32_t v15_spill_pgsz = 0u;   // size of ONE spill chunk = bucket_cap × 40 B
    uint32_t v15_spill_chunks = 0u; // MAX_SPILL_CHUNKS per (receiver, owned_tile)
    MeshWorkload wl_v13_scatter, wl_v13_accum;
    std::vector<uint16_t> tile_to_core_v13;
    std::vector<std::vector<uint32_t>> core_to_tiles_v13;
    uint32_t tile_map_bytes_v13   = 0u;
    uint32_t tile_map_pgsz_v13    = 0u;
    uint32_t route_pgsz_v13       = 0u;
    uint32_t owned_lookup_pgsz_v13 = 0u;
    uint32_t overflow_pgsz_v13    = 256u;
    uint32_t density_pgsz_v13     = 0u;   // bf16 row, N*2 bytes
    uint32_t total_tiles_v13      = 0u;
    uint32_t M_tiles_v13          = (uint32_t)M / 32u;
    uint32_t N_tiles_v13          = (uint32_t)N / 32u;
    uint32_t n_owned_max_v13      = 0u;

    // ── V14 Architecture-A declarations ──────────────────────────────────────
    constexpr uint32_t V14_MAX_OVERLAP    = 8u;
    constexpr uint32_t V14_K_BATCH        = 32u;
    constexpr uint32_t V14_N_INFLIGHT     = 4u;
    constexpr uint32_t V14_PUSH_BATCH     = V14_N_INFLIGHT * V14_K_BATCH;  // 128
    constexpr uint32_t V14_CB_OXY_DEPTH   = 8u;
    constexpr uint32_t TILE_BF16_BYTES_v14 = 32u * 32u * 2u;
    std::shared_ptr<MeshBuffer> tile_map_buf_v14, partial_buf_v14, density_buf_v14;
    std::shared_ptr<MeshBuffer> v14_drop_buf;  // DROP-INSTRUMENTATION
    uint32_t v14_drop_buf_pgsz_outer = 0u;
    MeshWorkload wl_v14_scatter, wl_v14_reduce;
    std::vector<uint16_t> tile_to_core_v14;
    std::vector<std::vector<uint32_t>> core_to_tiles_v14;
    uint32_t tile_map_bytes_v14   = 0u;
    uint32_t tile_map_pgsz_v14    = 0u;
    uint32_t partial_pgsz_v14     = TILE_BF16_BYTES_v14;  // 2048 per page
    uint32_t density_pgsz_v14     = 0u;
    uint32_t total_tiles_v14      = 0u;
    uint32_t M_tiles_v14          = (uint32_t)M / 32u;
    uint32_t N_tiles_v14          = (uint32_t)N / 32u;
    uint32_t n_owned_max_v14      = 0u;

    if (use_v11) {
        // Build snake-fill ownership map on host
        v11::build_snake_fill_ownership(M_tiles, N_tiles_v11, (uint32_t)nc_all,
                                        tile_to_core_v11, core_to_tiles_v11);
        uint32_t total_tiles_v11 = M_tiles * N_tiles_v11;
        tile_map_bytes_v11 = total_tiles_v11 * sizeof(uint16_t);
        tile_map_pgsz_v11 = (tile_map_bytes_v11 + 31u) & ~31u;

        // route_buf page size: 64 (header) + max_per_page * 8 (tuples).
        // Header stays 64 bytes for Blackhole 64-byte cache-line alignment.
        // Tuples are now packed at 8 bytes each; scatter pads partial flushes
        // up to multiples of 4 tuples (32 bytes) so every NOC write is 32B
        // aligned. v11_max_per_page_tuples must stay a multiple of 4.
        // Per-page tuple budget split between NCRISC (first half) and BRISC
        // (second half). With writer-side sort+dedup in v11_scatter_*_dm.cpp's
        // flush_recv each (writer, receiver) pair emits at most one tuple
        // per unique bin its cells touched (≤ ~3000 typical; single RISC
        // ≤ ~2000). 4096 (= half_cap 2048 per RISC) covers that and keeps
        // accum L1 scratch under the 1.5 MB cap on grid=2048.
        v11_max_per_page_tuples = 4096u;
        route_pgsz_v11 = (64u + v11_max_per_page_tuples * 8u + 31u) & ~31u;
        // Step 5b: NCRISC and BRISC share each writer's page — NCRISC fills
        // first half [64, 64+max/2*8), BRISC fills second half. Page count
        // stays nc_all*nc_all (same as Step 4) so accum has only nc_all
        // chunks per RISC, avoiding the 2× writer overhead.
        uint64_t route_total = (uint64_t)nc_all * (uint64_t)nc_all * route_pgsz_v11;

        // Owned-lookup page: M_tiles*N_tiles uint16 (local_idx or 0xFFFF).
        owned_lookup_pgsz_v11 = (total_tiles_v11 * (uint32_t)sizeof(uint16_t) + 31u) & ~31u;
        uint32_t owned_lookup_total = (uint32_t)nc_all * owned_lookup_pgsz_v11;

        // Compute n_owned (max owned tiles per core) — outer-scoped above.
        n_owned_max = 0;
        for (auto& v : core_to_tiles_v11)
            if ((uint32_t)v.size() > n_owned_max) n_owned_max = (uint32_t)v.size();

        printf("[server] V11: %u tiles (%ux%u), %u cores, max_owned=%u\n",
               total_tiles_v11, M_tiles, N_tiles_v11, (uint32_t)nc_all, n_owned_max);
        printf("[server] V11: route_pgsz=%u (max %u tuples), route_total=%llu MB, "
               "owned_lookup_total=%u KB\n",
               route_pgsz_v11, v11_max_per_page_tuples,
               (unsigned long long)(route_total / (1024u * 1024u)),
               owned_lookup_total / 1024u);
        fflush(stdout);

        // Allocate DRAM buffers
        tile_map_buf      = make_buf(tile_map_pgsz_v11, tile_map_pgsz_v11);
        owned_lookup_buf  = make_buf(owned_lookup_total, owned_lookup_pgsz_v11);
        // Drop-counter instrumentation buffer: one uint32 per writer (NCRISC
        // writers [0..nc_all) + BRISC writers [nc_all..2*nc_all)). Each
        // 32-byte page holds a single uint32 at offset 0; rest is padding.
        // Kernel writes via noc_inline_dw_write at end of execution.
        drop_buf = make_buf(2u * (uint32_t)nc_all * 32u, 32u);

        // hist_buf: per-core histogram pages, total_tiles uint32 entries each.
        // Each writer dumps its local count map; host reduces to global.
        hist_pgsz_v11 = (total_tiles_v11 * (uint32_t)sizeof(uint32_t) + 31u) & ~31u;
        uint32_t hist_total = (uint32_t)nc_all * hist_pgsz_v11;
        hist_buf = make_buf(hist_total, hist_pgsz_v11);
        printf("[server] V11 hist_pgsz=%u, hist_total=%u KB\n",
               hist_pgsz_v11, hist_total / 1024u);

        // shard_table_buf: SHARD_BYTES per tile (K + 7 alts + hot_tile_seq + pad), 1 page total.
        shard_table_pgsz_v11 = (total_tiles_v11 * SHARD_BYTES + 31u) & ~31u;
        shard_table_buf = make_buf(shard_table_pgsz_v11, shard_table_pgsz_v11);
        printf("[server] V11 shard_table_pgsz=%u (%u tiles × %u B)\n",
               shard_table_pgsz_v11, total_tiles_v11, SHARD_BYTES);

        // Initialize trivial shard_table: K=1 for all tiles, no alts.
        std::vector<uint8_t> shard_init(shard_table_pgsz_v11, 0);
        for (uint32_t t = 0; t < total_tiles_v11; ++t) {
            shard_init[t * SHARD_BYTES] = 1;  // K=1
        }
        EnqueueWriteMeshBuffer(cq, shard_table_buf, shard_init, false);

        // shard_reduce_buf: pages indexed by hot_tile_seq * MAX_K + shard_idx.
        // Each page = TILE_BYTES (4096). Max hot tiles bounded by experimental
        // observation (adaptec1 had 4 hot tiles; bigblue2 may have 200+).
        // Provision generously: 256 hot tiles × 8 shards = 8 MB.
        v11_max_hot_tiles = 256u;
        shard_reduce_pgsz_v11 = TILE_BYTES_v11;
        uint32_t shard_reduce_total = v11_max_hot_tiles * MAX_K * shard_reduce_pgsz_v11;
        shard_reduce_buf = make_buf(shard_reduce_total, shard_reduce_pgsz_v11);
        printf("[server] V11 shard_reduce_buf: %u pages × %u B = %u KB\n",
               v11_max_hot_tiles * MAX_K, shard_reduce_pgsz_v11,
               shard_reduce_total / 1024u);

        // route_buf can exceed 4 GB at large grids; pgsz must fit but the total
        // is given to ReplicatedBufferConfig as a uint32. Cap and warn.
        if (route_total > (uint64_t)0xFFFFFFFFu) {
            printf("[server] FATAL: V11 route_buf %llu B > 4 GB (writer/reader page ladder too large)\n",
                   (unsigned long long)route_total);
            fflush(stdout); std::exit(1);
        }
        route_buf = make_buf((uint32_t)route_total, route_pgsz_v11);

        // Upload tile_to_core[] to DRAM
        std::vector<uint8_t> tile_map_upload(tile_map_pgsz_v11, 0);
        std::memcpy(tile_map_upload.data(), tile_to_core_v11.data(),
                    tile_map_bytes_v11);
        EnqueueWriteMeshBuffer(cq, tile_map_buf, tile_map_upload, false);

        // Build per-core owned_lookup[] pages: tile_idx → local_idx (or 0xFFFF).
        std::vector<uint8_t> owned_upload(owned_lookup_total, 0xFF);
        for (uint32_t c = 0; c < (uint32_t)nc_all; ++c) {
            uint16_t* page = reinterpret_cast<uint16_t*>(
                owned_upload.data() + (size_t)c * owned_lookup_pgsz_v11);
            // init all to 0xFFFF (already done by 0xFF fill)
            for (uint32_t local = 0; local < core_to_tiles_v11[c].size(); ++local) {
                uint32_t tile_idx = core_to_tiles_v11[c][local];
                page[tile_idx] = (uint16_t)local;
            }
        }
        EnqueueWriteMeshBuffer(cq, owned_lookup_buf, owned_upload, false);
        Finish(cq);

        // ── prog_v11_scatter: v4_reader + v4_compute + v11_scatter_dm ─────
        Program prog_v11_sc = CreateProgram();
        for (uint32_t i = 0; i < 4; ++i) make_cb_all(prog_v11_sc, i, 2);
        // Output CBs deepened 1→2 slots: lets TRISC pipeline tile N+1's 18 SFPU
        // passes while NCRISC is still routing tile N. Eliminates the OX0
        // back-pressure stall (see memory/v11_ox0_backpressure.md).
        // L1 cost: 18 CBs × 4096 B = 72 KB extra per core.
        make_cb_all(prog_v11_sc, 4, 2);
        make_cb_all(prog_v11_sc, 5, 2);
        for (uint32_t j = 0; j < MAX_OVERLAP; ++j) {
            make_cb_all(prog_v11_sc, 6 + j, 2);
            make_cb_all(prog_v11_sc, 14 + j, 2);
        }
        // Stage B': add 8 outer-product CBs c_32..c_39, each with 8 tiles
        // per batch (one per K value) × 2 buffering = 16 tile slots.
        // Skip c_22..c_31 to avoid conflict with c_24 (V11 CB_SCRATCH).
        if (use_v11outer) {
            for (uint32_t j = 0; j < MAX_OVERLAP; ++j) {
                make_cb_all(prog_v11_sc, 32 + j, 16);  // 8 K-tiles × 2 batches
            }
        }
        // V11 scatter scratch (matches kernel layout exactly).
        // NCRISC region: tile_to_core, staging_n[][], counts, offsets, shard_table, hdr_n.
        // Then ScatterShared, then BRISC region: staging_b[][], counts, offsets, hdr_b.
        constexpr uint32_t V11_MAX_IN_FLIGHT = 64u;
        constexpr uint32_t V11_TUPLE_BYTES   = 8u;  // packed V11Contrib
        constexpr uint32_t V11_HDR_BYTES     = 64u;
        constexpr uint32_t V11_SCATTER_SHARED_BYTES = 4u + 4u + 8u * 4u + 8u * 4u; // ~72 B
        const uint32_t v18_slot_bytes      = V18_BF16_AREA ? 6u : 8u;
        const uint32_t v18_table_bytes_per = use_v18 ? (1u << V18_HASH_BITS) * v18_slot_bytes : 0u;
        const uint32_t v18_dirty_bytes_per_layout = (use_v18 && !V18_NO_DIRTY) ? (1u << V18_HASH_BITS) * 2u : 0u;
        uint32_t shared_state_off_sc = 0u;
        uint32_t brisc_state_off_sc = 0u;
        uint32_t v18_hash_n_off_sc = 0u;
        uint32_t v18_dirty_n_off_sc = 0u;
        uint32_t v18_hash_b_off_sc = 0u;
        uint32_t v18_dirty_b_off_sc = 0u;
        uint32_t v11_sc_scratch = 0u;
        {
            uint32_t off = 0u;
            off += tile_map_bytes_v11;
            off  = (off + 7u) & ~7u;                                  // tile_to_core
            off += (uint32_t)nc_all * V11_MAX_IN_FLIGHT * V11_TUPLE_BYTES;  // staging_n
            off  = (off + 3u) & ~3u;
            off += (uint32_t)nc_all * 4u;                              // staging_count_n
            off += (uint32_t)nc_all * 4u;                              // dram_offset_n
            off  = (off + 63u) & ~63u;
            off += shard_table_pgsz_v11;                               // shard_table
            off  = (off + 63u) & ~63u;
            off += V11_HDR_BYTES;                                       // hdr_scratch_n
            off  = (off + 63u) & ~63u;
            if (use_v18) {
                v18_hash_n_off_sc = off;
                off += v18_table_bytes_per;                             // V18 NCRISC hash table
                off  = (off + 63u) & ~63u;
                if (!V18_NO_DIRTY) {
                    v18_dirty_n_off_sc = off;
                    off += v18_dirty_bytes_per_layout;                  // V18 NCRISC dirty list
                    off  = (off + 63u) & ~63u;
                }
            }
            shared_state_off_sc = off;
            off += V11_SCATTER_SHARED_BYTES;                            // ScatterShared
            off  = (off + 63u) & ~63u;
            brisc_state_off_sc = off;
            off += (uint32_t)nc_all * V11_MAX_IN_FLIGHT * V11_TUPLE_BYTES;  // staging_b
            off  = (off + 3u) & ~3u;
            off += (uint32_t)nc_all * 4u;                              // staging_count_b
            off += (uint32_t)nc_all * 4u;                              // dram_offset_b
            off  = (off + 63u) & ~63u;
            off += V11_HDR_BYTES;                                       // hdr_scratch_b
            off  = (off + 63u) & ~63u;
            if (use_v18) {
                v18_hash_b_off_sc = off;
                off += v18_table_bytes_per;                             // V18 BRISC hash table
                off  = (off + 63u) & ~63u;
                if (!V18_NO_DIRTY) {
                    v18_dirty_b_off_sc = off;
                    off += v18_dirty_bytes_per_layout;                  // V18 BRISC dirty list
                    off  = (off + 63u) & ~63u;
                }
            }
            off += 64u;                                                 // safety gap
            v11_sc_scratch = (off + 31u) & ~31u;
        }

        // V19 density slab layout: per-core slab of ⌈M*N/nc_all⌉ uint32 bins,
        // followed by a small noc_coords[nc_all] table. Placed AFTER existing
        // V11 scratch in CB_SCRATCH.
        uint32_t v19_density_l1_off  = 0u;
        uint32_t v19_density_slab_bins = 0u;
        uint32_t v19_coord_pgsz      = 0u;
        std::shared_ptr<MeshBuffer> v19_coord_buf;
        std::shared_ptr<MeshBuffer> v19_density_dram;
        uint32_t v19_density_dram_pgsz = 0u;
        if (use_v19) {
            v19_density_l1_off = v11_sc_scratch;
            uint32_t total_bins = (uint32_t)M * (uint32_t)N;
            // Pad slab_bins to multiple of 8 so density_slab_bytes is 32-byte
            // aligned. Without this, density_dram_pgsz = (bytes + 31) & ~31
            // would be larger than density_slab_bins*4, and the de-stride
            // readback would slip out of sync (the strided[] vector is sized
            // for unpadded bins so it consumes adjacent pages' padding bytes
            // as if they were valid bin values).
            uint32_t slab_bins_raw = (total_bins + (uint32_t)nc_all - 1u) / (uint32_t)nc_all;
            v19_density_slab_bins = (slab_bins_raw + 7u) & ~7u;
            uint32_t density_slab_bytes = v19_density_slab_bins * 4u;
            uint32_t coords_bytes = ((uint32_t)nc_all * 4u + 31u) & ~31u;
            v19_coord_pgsz = coords_bytes;
            v11_sc_scratch = ((v19_density_l1_off + density_slab_bytes + coords_bytes + 31u) & ~31u);
            printf("[server] V19: density_l1_off=%u slab_bins=%u (%u KB) coords_pgsz=%u\n",
                   v19_density_l1_off, v19_density_slab_bins,
                   density_slab_bytes >> 10, v19_coord_pgsz);
            fflush(stdout);
            // Allocate DRAM buffer for the worker_noc_coords table (one page).
            DeviceLocalBufferConfig cfg{.page_size = v19_coord_pgsz,
                                        .buffer_type = BufferType::DRAM};
            ReplicatedBufferConfig rcfg{.size = v19_coord_pgsz};
            v19_coord_buf = MeshBuffer::create(rcfg, cfg, mesh_device.get());
            // Populate it: packed uint32 per core = (noc_y << 16) | noc_x.
            // Pad to coord_pgsz / 4 entries.
            std::vector<uint32_t> coords((size_t)(v19_coord_pgsz / 4u), 0u);
            auto* dev0 = mesh_device->get_devices()[0];
            for (int c = 0; c < nc_all; ++c) {
                CoreCoord noc = dev0->worker_core_from_logical_core(all_ccs[c]);
                coords[c] = ((uint32_t)noc.y << 16) | (uint32_t)noc.x;
            }
            EnqueueWriteMeshBuffer(cq, v19_coord_buf, coords, false);
            Finish(cq);

            // V19 DRAM density staging: per-core page of slab_bins uint32.
            // Kernel writes its L1 slab here at end-of-scatter; host reads + de-strides.
            v19_density_dram_pgsz = ((density_slab_bytes + 31u) & ~31u);
            DeviceLocalBufferConfig dcfg{.page_size = v19_density_dram_pgsz,
                                         .buffer_type = BufferType::DRAM};
            ReplicatedBufferConfig drcfg{.size = (uint64_t)nc_all * v19_density_dram_pgsz};
            v19_density_dram = MeshBuffer::create(drcfg, dcfg, mesh_device.get());
            printf("[server] V19 DRAM staging: %u pages × %u B = %u KB\n",
                   nc_all, v19_density_dram_pgsz, (nc_all * v19_density_dram_pgsz) >> 10);
            fflush(stdout);
            // Publish to outer scope for the per-iter readback.
            v19_density_slab_bins_outer = v19_density_slab_bins;
            v19_density_dram_pgsz_outer = v19_density_dram_pgsz;
            v19_density_dram_outer      = v19_density_dram;

            // Build the V19 writeout workload — runs after scatter Finish()
            // to ensure all atomics have settled.
            // CRITICAL: must match the scatter program's CB layout exactly so
            // c_24's L1 base address (and therefore the density slab location)
            // is identical between scatter and writeout programs.
            Program prog_v19_wo = CreateProgram();
            for (uint32_t i = 0; i < 4; ++i) make_cb_all(prog_v19_wo, i, 2);
            make_cb_all(prog_v19_wo, 4, 2);
            make_cb_all(prog_v19_wo, 5, 2);
            for (uint32_t j = 0; j < MAX_OVERLAP; ++j) {
                make_cb_all(prog_v19_wo, 6 + j, 2);
                make_cb_all(prog_v19_wo, 14 + j, 2);
            }
            CreateCircularBuffer(prog_v19_wo, all_crs,
                CircularBufferConfig(v11_sc_scratch, {{24u, tt::DataFormat::Float32}})
                    .set_page_size(24u, v11_sc_scratch));
            auto rk_wo = CreateKernel(prog_v19_wo,
                KDIR + "v19_writeout_fp32_dm.cpp", all_crs,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                                   .noc = NOC::RISCV_0_default});
            CreateKernel(prog_v19_wo,
                KDIR + "v19_writeout_void_ncrisc.cpp", all_crs,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                                   .noc = NOC::RISCV_1_default});
            const uint32_t v19_scale_bits = env_uint("V19_SCALE_BITS", 20u);
            const uint32_t density_buf_pgsz =
                (uint32_t)((uint64_t)N * sizeof(float));
            for (int c = 0; c < nc_all; ++c) {
                SetRuntimeArgs(prog_v19_wo, rk_wo, all_ccs[c], {
                    v19_density_l1_off,
                    v19_density_slab_bins,
                    (uint32_t)density_buf->address(),  // write fp32 directly to density_buf
                    density_buf_pgsz,
                    (uint32_t)c,
                    (uint32_t)M,
                    (uint32_t)N,
                    v19_scale_bits,
                });
            }
            wl_v19_writeout.add_program(device_range, std::move(prog_v19_wo));
            v19_writeout_built = true;
        }

        printf("[server] V11 scatter scratch = %u KB (shared_off=%u, brisc_off=%u)\n",
               v11_sc_scratch / 1024u, shared_state_off_sc, brisc_state_off_sc);
        if (use_v18) {
            printf("[server] V18: hash_n_off=%u dirty_n_off=%u hash_b_off=%u dirty_b_off=%u table_per=%u KB dirty_per=%u KB\n",
                   v18_hash_n_off_sc, v18_dirty_n_off_sc,
                   v18_hash_b_off_sc, v18_dirty_b_off_sc,
                   v18_table_bytes_per >> 10, v18_dirty_bytes_per_layout >> 10);
            fflush(stdout);
        }
        if (v11_sc_scratch >= 1536u * 1024u) {
            printf("[server] FATAL: V11 scatter scratch %u KB exceeds L1 budget\n",
                   v11_sc_scratch / 1024u);
            fflush(stdout); std::exit(1);
        }
        fflush(stdout);
        CreateCircularBuffer(prog_v11_sc, all_crs,
            CircularBufferConfig(v11_sc_scratch, {{24u, tt::DataFormat::Float32}})
                .set_page_size(24u, v11_sc_scratch));

        std::vector<UnpackToDestMode> v11_unpack(NUM_CIRCULAR_BUFFERS,
                                                 UnpackToDestMode::Default);
        for (int i = 0; i < 4; ++i) v11_unpack[i] = UnpackToDestMode::UnpackToDestFp32;
        // V11-outer: the outer-product compute kernel ALSO unpacks from
        // c_6..c_21 (ox/oy intermediates) for the 64 multiplies. Those CBs
        // are Float32, so the unpacker must be in fp32-to-DST mode for them
        // too. Without this, the unpacker would do a bf16 conversion that
        // garbles the ox/oy values feeding the outer product.
        if (use_v11outer) {
            for (uint32_t j = 0; j < MAX_OVERLAP; ++j) {
                v11_unpack[6 + j]  = UnpackToDestMode::UnpackToDestFp32;
                v11_unpack[14 + j] = UnpackToDestMode::UnpackToDestFp32;
            }
        }

        // BRISC kernel does combined v4_reader + scatter_b (Step 5 parallel scatter).
        // V11-outer uses sibling files; V18 also uses sibling files (hash-agg);
        // V18-outer combines both (hash-agg with precomputed outer-product).
        const bool use_v31_stash_pre = use_v19 && (env_uint("V31_STASH", 0u) != 0u);
        const std::string brisc_kernel_file =
              use_v31_stash_pre ? "v31_scatter_b_stash_dm.cpp"
            : use_v19      ? "v19_scatter_b_dm.cpp"
            : use_v18outer ? "v18_outer_scatter_b_dm.cpp"
            : use_v18      ? "v18_scatter_b_dm.cpp"
            : use_v11outer ? "v11_outer_scatter_b_dm.cpp"
            :                "v11_scatter_b_dm.cpp";
        const std::string compute_kernel_file = use_v11outer
            ? "v4_outer_compute.cpp" : "v4_compute.cpp";
        const bool use_v31_stash = use_v19 && (env_uint("V31_STASH", 0u) != 0u);
        const std::string ncrisc_kernel_file =
              use_v31_stash? "v31_scatter_stash_dm.cpp"
            : use_v19      ? "v19_scatter_dm.cpp"
            : use_v18outer ? "v18_outer_scatter_dm.cpp"
            : use_v18      ? "v18_scatter_dm.cpp"
            : use_v11outer ? "v11_outer_scatter_dm.cpp"
            :                "v11_scatter_dm.cpp";
        auto rk_v11 = CreateKernel(prog_v11_sc, KDIR + brisc_kernel_file, all_crs,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                               .noc = NOC::RISCV_0_default});
        // G-PRESCALE: V11 scatter pre-multiplies ox and oy outputs by
        // sqrt(inv_bin_area) inside v4_compute (SFPU), so area = ox*oy is
        // already pre-scaled by inv_bin_area at scatter emission time. The
        // gather kernel's V11A-SCALE phase then becomes a no-op.
        std::map<std::string, std::string> v4_defs_v11 = v4_defs;
        v4_defs_v11["V4_OX_OY_SCALE_F"] = to_float_literal(std::sqrt(inv_ba));
        auto ck_v11 = CreateKernel(prog_v11_sc, KDIR + compute_kernel_file, all_crs,
            ComputeConfig{.fp32_dest_acc_en = true,
                          .unpack_to_dest_mode = v11_unpack,
                          .math_approx_mode = false,
                          .defines = v4_defs_v11});
        auto sk_v11 = CreateKernel(prog_v11_sc, KDIR + ncrisc_kernel_file, all_crs,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                               .noc = NOC::RISCV_1_default});

        // Step-5 inter-RISC sync semaphores. data_ready/brisc_done are direct-L1
        // counters bumped via volatile stores; tables_ready is a one-shot flag.
        uint32_t sc_data_ready_sem  = CreateSemaphore(prog_v11_sc, all_crs, 0u);
        uint32_t sc_brisc_done_sem  = CreateSemaphore(prog_v11_sc, all_crs, 0u);
        uint32_t sc_tables_ready_sem= CreateSemaphore(prog_v11_sc, all_crs, 0u);

        uint32_t tm_a = (uint32_t)tile_map_buf->address();
        uint32_t rt_a = (uint32_t)route_buf->address();

        // ── V31 forward-stash buffers + staging CB (gated; default OFF) ──
        uint32_t v31_route_pg = 0, v31_rcount_pg = 0, v31_route_a = 0, v31_rcount_a = 0;
        if (use_v31_stash) {
            // recs/cell cap. 512-grid spans ≤8 (→16 ok), but 2048-grid cells span up to
            // ~15×3 bins → a 512-cell tile-half produced 8462 recs (>16·512=8192) and the
            // c_25/c_26 staging CB overflowed (watcher NOC-overflow, wedged the device).
            // 32 → 16384/half cap (~2× the observed 8462); L1 staging 2×256 KB still fits
            // <1.5 MB. (Robust fix = chunk-flush the staging when full; this covers adaptec.)
            const uint32_t SPAN_CAP = 32u;
            uint32_t v31_cap = (max_tpc) * 1024u * SPAN_CAP;       // records per core
            v31_route_pg  = v31_cap * 16u;
            v31_rcount_pg = 64u;
            // 2·nc pages: [0,nc)=NCRISC half [0,512), [nc,2nc)=BRISC half [512,1024).
            const uint32_t SPAN_CAP_H = (SPAN_CAP/2u > 1u) ? (SPAN_CAP/2u) : 1u;  // per-half cap
            v31_route_buf_  = make_buf(2u*(uint32_t)nc_all * v31_route_pg,  v31_route_pg);
            v31_rcount_buf_ = make_buf(2u*(uint32_t)nc_all * v31_rcount_pg, v31_rcount_pg);
            v31_route_a  = (uint32_t)v31_route_buf_->address();
            v31_rcount_a = (uint32_t)v31_rcount_buf_->address();
            const uint32_t stage_bytes = 512u * SPAN_CAP * 16u;   // per-half tile: 512 cells × span
            CreateCircularBuffer(prog_v11_sc, all_crs,                              // NCRISC staging
                CircularBufferConfig(stage_bytes, {{25u, tt::DataFormat::Float32}}).set_page_size(25u, stage_bytes));
            CreateCircularBuffer(prog_v11_sc, all_crs,                              // BRISC staging
                CircularBufferConfig(stage_bytes, {{26u, tt::DataFormat::Float32}}).set_page_size(26u, stage_bytes));
            (void)SPAN_CAP_H;
            printf("[server] V31_STASH ON: route_pg=%u KB ×2nc, stage=%u KB (c_25,c_26)\n",
                   v31_route_pg>>10, stage_bytes>>10); fflush(stdout);
            v31_route_pg_=v31_route_pg; v31_max_tpc_=max_tpc; v31_nc_all_=(uint32_t)nc_all;
            v31_first_tile_.assign(nc_all,0u);
        }
        // ── V29 prep-reuse per-cell GEOMETRY stash (gated by V31_GEOM; needs V31_STASH) ──
        const bool use_v31_geom = use_v31_stash && (env_uint("V31_GEOM", 0u) != 0u);
        // V35_STASH64: emit a 64 B compact geom record (px[8]+py[4]) instead of 128 B → halves
        // the backward count/place DRAM read (forward+backward co-design). Valid when max_h<=4
        // (all live configs; the backward asserts it). The forward writes py[0..3] only.
        const uint32_t v31_geom64 = use_v31_geom && (env_uint("V35_STASH64", 0u) != 0u) ? 1u : 0u;
        uint32_t v31_geom_a = 0, v31_geom_pg = 0, v31_bin_area_bits = 0;
        // FCCS: stash px/py as int32 (×2^13) in the forward so the backward skips its
        // per-cell soft-float quantize (the dominant live wrapper overhead). V29 reads
        // float px/py → leave it 0 for V29 runs.
        const uint32_t v31_geom_int = (env_uint("V31_GEOM_INT", 0u) != 0u) ? 1u : 0u;
        if (use_v31_geom) {
            // Single flat-page buffer indexed by global active cell index (tile*1024+ci).
            v31_geom_pg = soa_padded * (v31_geom64 ? 64u : 128u);   // 64 B compact (V35_STASH64) or 128 B records
            v31_geom_buf_ = make_buf(v31_geom_pg, v31_geom_pg);
            v31_geom_a = (uint32_t)v31_geom_buf_->address();
            v31_geom_pg_ = v31_geom_pg;
            { float ba = (inv_ba > 0.0f) ? (1.0f / inv_ba) : 1.0f;
              std::memcpy(&v31_bin_area_bits, &ba, 4); }
            // V31_EF_GEOM: per-cell ratio buffer for the density scatter (·ratio).
            v31_ef_geom_ = (env_uint("V31_EF_GEOM", 0u) != 0u);
            if (v31_ef_geom_) {
                v31_ratio_buf_ = make_buf(soa_padded * 4u, soa_padded * 4u);
                v31_ratio_a_ = (uint32_t)v31_ratio_buf_->address();
                v31_ratio_uploaded_ = false;
                printf("[server] V31_EF_GEOM ON: per-cell ratio buf for density scatter (addr=%u)\n", v31_ratio_a_); fflush(stdout);
            }
            const uint32_t geom_stage = (TILE_ELEMS / 2u) * 128u;   // 512 recs/half × 128 B = 64 KB
            CreateCircularBuffer(prog_v11_sc, all_crs,                          // NCRISC geom staging
                CircularBufferConfig(geom_stage, {{27u, tt::DataFormat::Float32}}).set_page_size(27u, geom_stage));
            CreateCircularBuffer(prog_v11_sc, all_crs,                          // BRISC geom staging
                CircularBufferConfig(geom_stage, {{28u, tt::DataFormat::Float32}}).set_page_size(28u, geom_stage));
            printf("[server] V31_GEOM ON: geom_buf=%.1f MB (1 page), stage=%u KB (c_27,c_28)\n",
                   (double)v31_geom_pg/1048576.0, geom_stage>>10); fflush(stdout);
            if (v31_ef_geom_) {   // per-cell ratio tile (512 floats/half) for the density scatter
                const uint32_t ratio_stage = (TILE_ELEMS / 2u) * 4u;
                CreateCircularBuffer(prog_v11_sc, all_crs,                       // NCRISC ratio tile
                    CircularBufferConfig(ratio_stage, {{29u, tt::DataFormat::Float32}}).set_page_size(29u, ratio_stage));
                CreateCircularBuffer(prog_v11_sc, all_crs,                       // BRISC ratio tile
                    CircularBufferConfig(ratio_stage, {{30u, tt::DataFormat::Float32}}).set_page_size(30u, ratio_stage));
            }
        }

        for (int c = 0; c < nc_all; ++c) {
            auto cc = all_ccs[c];
            uint32_t my_n = base_tpc + ((uint32_t)c < rem_tpc ? 1u : 0u);
            uint32_t first = (uint32_t)c * base_tpc + std::min((uint32_t)c, rem_tpc);
            if (use_v31_stash) v31_first_tile_[c]=first;
            uint32_t st_a = (uint32_t)shard_table_buf->address();
            uint32_t drop_a = (uint32_t)drop_buf->address();

            // BRISC (combined reader + scatter_b). my_writer_id = my_core_id + nc_all.
            // V18 appends two extra args (24,25) for hash table offset+bits.
            std::vector<uint32_t> brisc_args = {
                px_a, py_a, sx_a, sy_a, tile_pgsz, first, my_n,    // 0..6 reader
                tile_map_bytes_v11,                                 // 7
                (uint32_t)c,                                        // 8 (unused on BRISC; placeholder)
                (uint32_t)nc_all,                                   // 9
                M_tiles, N_tiles_v11,                               // 10,11
                (uint32_t)M, (uint32_t)N,                           // 12,13
                rt_a, route_pgsz_v11, v11_max_per_page_tuples,      // 14..16
                (uint32_t)c,                                        // 17 my_writer_id (BRISC shares writer slot with NCRISC)
                sc_data_ready_sem, sc_brisc_done_sem,               // 18,19
                shared_state_off_sc, brisc_state_off_sc,            // 20,21
                sc_tables_ready_sem,                                // 22
                drop_a,                                             // 23 drop_dram (BRISC drop slot = c + nc_all)
            };
            if (use_v18) {
                brisc_args.push_back(v18_hash_b_off_sc);            // 24 hash table L1 offset
                brisc_args.push_back(V18_HASH_BITS);                // 25 hash bits (log2 slots)
                brisc_args.push_back(v18_dirty_b_off_sc);           // 26 dirty list L1 offset (unused if no_dirty)
                brisc_args.push_back(V18_BF16_AREA);                // 27 bf16 area flag
                brisc_args.push_back(V18_NO_DIRTY);                 // 28 no-dirty flag
            }
            if (use_v19) {
                brisc_args.push_back(v19_density_l1_off);           // 24 density slab L1 offset
                brisc_args.push_back(v19_density_slab_bins);        // 25 slab bin count
                brisc_args.push_back(env_uint("V19_SCALE_BITS", 20u));  // 26 scale_bits
                if (use_v31_stash) {
                    brisc_args.push_back((uint32_t)c);              // 27 my_core_idx
                    brisc_args.push_back(v31_route_a);              // 28 route_base
                    brisc_args.push_back(v31_route_pg);             // 29 route_pg
                    brisc_args.push_back(v31_rcount_a);             // 30 rcount_base
                    brisc_args.push_back(v31_rcount_pg);            // 31 rcount_pg
                    brisc_args.push_back(v31_geom_a);               // 32 geom_base (0 = off)
                    brisc_args.push_back(v31_geom_pg);              // 33 geom_pg
                    brisc_args.push_back(v31_bin_area_bits);        // 34 bin_area bits
                    brisc_args.push_back(v31_geom_int);             // 35 stash px/py as int32 (×2^13)
                    brisc_args.push_back(v31_ratio_a_);             // 36 per-cell ratio buf (0=off, V31_EF_GEOM)
                    brisc_args.push_back(v31_geom64);               // 37 emit 64 B compact geom record
                } else {
                    brisc_args.push_back(0u);                       // 27 dram_dst (BRISC: unused)
                    brisc_args.push_back(0u);                       // 28 dram_pgsz
                    brisc_args.push_back((uint32_t)c);              // 29 my_core_idx
                    brisc_args.push_back((uint32_t)nc_all);         // 30 nc_all
                }
            }
            SetRuntimeArgs(prog_v11_sc, rk_v11, cc, brisc_args);
            SetRuntimeArgs(prog_v11_sc, ck_v11, cc, {my_n});
            // NCRISC scatter. my_writer_id = my_core_id (existing slot).
            // V18 appends two extra args (23,24) for hash table offset+bits.
            std::vector<uint32_t> ncrisc_args = {
                tm_a, tile_map_pgsz_v11, tile_map_bytes_v11,        // 0..2
                (uint32_t)c, (uint32_t)nc_all,                      // 3,4
                M_tiles, N_tiles_v11,                               // 5,6
                (uint32_t)M, (uint32_t)N,                           // 7,8
                my_n,                                               // 9
                rt_a, route_pgsz_v11, v11_max_per_page_tuples,      // 10..12
                inv_ba_u32,                                         // 13
                st_a, shard_table_pgsz_v11,                         // 14,15
                (uint32_t)c,                                        // 16 my_writer_id (NCRISC slot)
                sc_data_ready_sem, sc_brisc_done_sem,               // 17,18
                shared_state_off_sc, brisc_state_off_sc,            // 19,20
                sc_tables_ready_sem,                                // 21
                drop_a,                                             // 22 drop_dram
            };
            if (use_v18) {
                ncrisc_args.push_back(v18_hash_n_off_sc);           // 23 hash table L1 offset
                ncrisc_args.push_back(V18_HASH_BITS);               // 24 hash bits (log2 slots)
                ncrisc_args.push_back(v18_dirty_n_off_sc);          // 25 dirty list L1 offset (unused if no_dirty)
                ncrisc_args.push_back(V18_BF16_AREA);               // 26 bf16 area flag
                ncrisc_args.push_back(V18_NO_DIRTY);                // 27 no-dirty flag
            }
            if (use_v19) {
                ncrisc_args.push_back(v19_density_l1_off);          // 23 density slab L1 offset
                ncrisc_args.push_back(v19_density_slab_bins);       // 24 slab bin count
                ncrisc_args.push_back((uint32_t)v19_coord_buf->address()); // 25 noc_coord DRAM addr
                ncrisc_args.push_back(v19_coord_pgsz);              // 26 noc_coord page size
                ncrisc_args.push_back(env_uint("V19_SCALE_BITS", 20u));  // 27 scale_bits
                ncrisc_args.push_back((uint32_t)v19_density_dram->address()); // 28 DRAM staging base
                ncrisc_args.push_back(v19_density_dram_pgsz);       // 29 page size per core
                ncrisc_args.push_back((uint32_t)c);                 // 30 my_core_idx
            }
            if (use_v31_stash) {
                ncrisc_args.push_back(first);                       // 31 first_tile
                ncrisc_args.push_back(v31_route_a);                 // 32 route_base
                ncrisc_args.push_back(v31_route_pg);                // 33 route_pg
                ncrisc_args.push_back(v31_rcount_a);                // 34 rcount_base
                ncrisc_args.push_back(v31_rcount_pg);               // 35 rcount_pg
                ncrisc_args.push_back(v31_geom_a);                  // 36 geom_base (0 = off)
                ncrisc_args.push_back(v31_geom_pg);                 // 37 geom_pg
                ncrisc_args.push_back(v31_bin_area_bits);           // 38 bin_area bits
                ncrisc_args.push_back(v31_geom_int);                // 39 stash px/py as int32 (×2^13)
                ncrisc_args.push_back(v31_ratio_a_);                // 40 per-cell ratio buf (0=off, V31_EF_GEOM)
                ncrisc_args.push_back(v31_geom64);                  // 41 emit 64 B compact geom record
            }
            SetRuntimeArgs(prog_v11_sc, sk_v11, cc, ncrisc_args);
        }
        // V19: broadcast worker noc_coords table as common runtime args.
        // Each scatter kernel reads via get_common_arg_val<uint32_t>(i) for
        // i in [0, nc_all). Replaces the prior noc_async_read of coord_buf,
        // which corrupted at 2048 grid (110 cores simultaneously reading
        // page 0 of a single-page DRAM buffer into a high-L1-offset
        // destination silently returned shifted data and stalled
        // noc_async_atomic_barrier).
        if (use_v19) {
            std::vector<uint32_t> v19_common_coords((size_t)nc_all, 0u);
            auto* dev0 = mesh_device->get_devices()[0];
            for (int c = 0; c < nc_all; ++c) {
                CoreCoord noc = dev0->worker_core_from_logical_core(all_ccs[c]);
                v19_common_coords[c] = ((uint32_t)noc.y << 16) | (uint32_t)noc.x;
            }
            // Only NCRISC reads common args (it populates L1 noc_coords once
            // per kernel launch; BRISC reads noc_coords from L1 after the
            // tables_ready_sem signals NCRISC has finished INIT).
            SetCommonRuntimeArgs(prog_v11_sc, sk_v11, v19_common_coords);
        }
        wl_v11_scatter.add_program(device_range, std::move(prog_v11_sc));

        // ── Compute accum CB scratch and dense_offset_bytes ───────────────
        // dense_offset_bytes is fixed (independent of n_owned) — it's the
        // sum of all regions before the dense buffer. Both accum and reduce
        // kernels mirror this layout to locate dense at the same L1 address.
        // Must match SRC_CHUNK in v11_accum_*_dm.cpp. Halved from 16 to 8 to
        // free L1 budget for max_per_writer = 4096 (avoids cap-hit tuple drops).
        constexpr uint32_t V11_SRC_CHUNK = 8u;
        uint32_t inbound_max = v11_max_per_page_tuples;
        {
            // Layout mirrors v11_accum_dm.cpp / v11_accum_n_dm.cpp exactly so
            // BRISC and NCRISC compute identical region offsets:
            //   owned_lookup, hdrs_b, hdrs_n, buf_b, buf_n, [128B gap], dense_b, dense_n, tmp_shard
            uint32_t off = 0u;
            off += (total_tiles_v11 * (uint32_t)sizeof(uint16_t) + 7u) & ~7u; // owned_lookup
            off  = (off + 63u) & ~63u;
            off += (uint32_t)nc_all * 64u;                      // inbound_hdrs_b
            off  = (off + 63u) & ~63u;
            off += (uint32_t)nc_all * 64u;                      // inbound_hdrs_n
            off  = (off + 63u) & ~63u;
            off += V11_SRC_CHUNK * inbound_max * 8u;            // inbound_buf_b (8B tuples)
            off  = (off + 63u) & ~63u;
            off += V11_SRC_CHUNK * inbound_max * 8u;            // inbound_buf_n (8B tuples)
            off  = (off + 63u) & ~63u;
            off += 128u;                                        // safety gap
            v11_dense_offset_bytes = off;
        }
        // Generous fixed CB budget. With BRISC+NCRISC parallel accum we need
        // 2x dense (one buffer per RISC) + 1 tmp_shard. Reused identically by
        // every refresh so the JIT'd kernel binary is always the same and the
        // JIT cache hits.
        uint32_t v11_ac_scratch = (v11_dense_offset_bytes
            + (2u * (n_owned_max + V11_CB_SLOT_HEADROOM) + 1u) * TILE_BYTES_v11
            + 31u) & ~31u;
        printf("[server] V11 accum scratch (initial) = %u KB (max_owned=%u, dense_off=%u)\n",
               v11_ac_scratch / 1024u, n_owned_max, v11_dense_offset_bytes);
        if (v11_ac_scratch >= 1536u * 1024u) {
            printf("[server] FATAL: V11 accum scratch %u KB exceeds L1 budget\n",
                   v11_ac_scratch / 1024u);
            fflush(stdout); std::exit(1);
        }
        fflush(stdout);

        // Initialize per_core_v11 with primary-only info (shard entries empty).
        // v11_dbg_first block rebuilds this with real shard info.
        per_core_v11.resize(nc_all);
        for (uint32_t c = 0; c < (uint32_t)nc_all; ++c) {
            per_core_v11[c] = {};
            per_core_v11[c].primary_tile_ids = core_to_tiles_v11[c];
            per_core_v11[c].primary_K.assign(core_to_tiles_v11[c].size(), 1u);
            per_core_v11[c].primary_hot_seq.assign(core_to_tiles_v11[c].size(), 0u);
        }

        // ── prog_v11_accum: merged accum + reduce (Phase A/B/C) in one kernel ─
        // Uses NOC semaphores for in-program shard→primary sync, so the
        // gather phase costs only ONE host Finish() barrier.
        Program prog_v11_ac = CreateProgram();
        CreateCircularBuffer(prog_v11_ac, all_crs,
            CircularBufferConfig(v11_ac_scratch, {{24u, tt::DataFormat::Float32}})
                .set_page_size(24u, v11_ac_scratch));

        auto ak_v11 = CreateKernel(prog_v11_ac, KDIR + "v11_accum_dm.cpp", all_crs,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                               .noc = NOC::RISCV_0_default});
        auto ak_v11_n = CreateKernel(prog_v11_ac, KDIR + "v11_accum_n_dm.cpp", all_crs,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                               .noc = NOC::RISCV_1_default});

        // BRISC↔NCRISC merge semaphore (initial cold build always has it).
        uint32_t merge_sem_id_init = CreateSemaphore(prog_v11_ac, all_crs, 0u);
        // G-PMERGE: two additional sems so BRISC and NCRISC can each do half
        // of dense_b += dense_n in parallel after both V11{A,N}-ACC are done.
        uint32_t brisc_acc_done_sem_init = CreateSemaphore(prog_v11_ac, all_crs, 0u);
        uint32_t ncrisc_half_merge_done_sem_init = CreateSemaphore(prog_v11_ac, all_crs, 0u);
        // Step 5b: route_buf has nc_all writers (NCRISC + BRISC share pages).
        // Accum splits writer reads in half: BRISC [0..nc_all/2), NCRISC [nc_all/2..nc_all).
        uint32_t nc_split_init = (uint32_t)nc_all / 2u;

        uint32_t ol_a = (uint32_t)owned_lookup_buf->address();
        uint32_t da_v11 = (uint32_t)density_buf->address();
        uint32_t srb_a_init = (uint32_t)shard_reduce_buf->address();
        for (int c = 0; c < nc_all; ++c) {
            auto cc = all_ccs[c];
            auto& info = per_core_v11[c];
            uint32_t n_primary = (uint32_t)info.primary_tile_ids.size();
            CoreCoord my_noc = mesh_device->worker_core_from_logical_core(cc);
            // Initial build: no shards, no hot primaries (rebuilt in v11_dbg_first).
            std::vector<uint32_t> args = {
                ol_a, owned_lookup_pgsz_v11,
                (uint32_t)c, (uint32_t)nc_all,
                M_tiles, N_tiles_v11,
                (uint32_t)M, (uint32_t)N,
                rt_a, route_pgsz_v11, v11_max_per_page_tuples,
                da_v11, density_pgsz, inv_ba_u32,
                n_primary,
                0u,                           // n_shard
                srb_a_init,                   // srb_dram
                shard_reduce_pgsz_v11,        // srb_pgsz
                MAX_K,
                0u,                           // n_primary_hot
                nc_split_init,
                merge_sem_id_init,
                (uint32_t)my_noc.x,
                (uint32_t)my_noc.y,
                brisc_acc_done_sem_init,           // arg 24 (G-PMERGE)
                ncrisc_half_merge_done_sem_init,   // arg 25 (G-PMERGE)
            };
            for (auto tid : info.primary_tile_ids) args.push_back(tid);
            // No hot quads, no shard quints on initial build.
            SetRuntimeArgs(prog_v11_ac, ak_v11,   cc, args);
            SetRuntimeArgs(prog_v11_ac, ak_v11_n, cc, args);
        }
        wl_v11_accum.add_program(device_range, std::move(prog_v11_ac));

        // ── prog_v11_hist: v4_reader + v4_compute + v11_histogram ─────────
        // Counts per-tile contributions per writer; output dumped to hist_buf.
        Program prog_v11_h = CreateProgram();
        for (uint32_t i = 0; i < 4; ++i) make_cb_all(prog_v11_h, i, 2);
        make_cb_all(prog_v11_h, 4, 1);
        make_cb_all(prog_v11_h, 5, 1);
        for (uint32_t j = 0; j < MAX_OVERLAP; ++j) {
            make_cb_all(prog_v11_h, 6 + j, 1);
            make_cb_all(prog_v11_h, 14 + j, 1);
        }
        // Hist scratch: total_tiles uint32 counters, 32-aligned
        uint32_t v11_h_scratch = (total_tiles_v11 * (uint32_t)sizeof(uint32_t) + 31u) & ~31u;
        if (v11_h_scratch >= 1536u * 1024u) {
            printf("[server] FATAL: V11 hist scratch %u KB exceeds L1 budget\n",
                   v11_h_scratch / 1024u);
            fflush(stdout); std::exit(1);
        }
        printf("[server] V11 hist scratch = %u KB\n", v11_h_scratch / 1024u);
        fflush(stdout);
        CreateCircularBuffer(prog_v11_h, all_crs,
            CircularBufferConfig(v11_h_scratch, {{24u, tt::DataFormat::Float32}})
                .set_page_size(24u, v11_h_scratch));

        std::vector<UnpackToDestMode> v11h_unpack(NUM_CIRCULAR_BUFFERS,
                                                  UnpackToDestMode::Default);
        for (int i = 0; i < 4; ++i) v11h_unpack[i] = UnpackToDestMode::UnpackToDestFp32;
        auto rk_h = CreateKernel(prog_v11_h, KDIR + "v4_reader.cpp", all_crs,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                               .noc = NOC::RISCV_0_default});
        auto ck_h = CreateKernel(prog_v11_h, KDIR + "v4_compute.cpp", all_crs,
            ComputeConfig{.fp32_dest_acc_en = true,
                          .unpack_to_dest_mode = v11h_unpack,
                          .math_approx_mode = false,
                          .defines = v4_defs_v11});
        auto hk = CreateKernel(prog_v11_h, KDIR + "v11_histogram.cpp", all_crs,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                               .noc = NOC::RISCV_1_default});

        uint32_t hist_a = (uint32_t)hist_buf->address();
        for (int c = 0; c < nc_all; ++c) {
            auto cc = all_ccs[c];
            uint32_t my_n = base_tpc + ((uint32_t)c < rem_tpc ? 1u : 0u);
            uint32_t first = (uint32_t)c * base_tpc + std::min((uint32_t)c, rem_tpc);
            SetRuntimeArgs(prog_v11_h, rk_h, cc,
                {px_a, py_a, sx_a, sy_a, tile_pgsz, first, my_n});
            SetRuntimeArgs(prog_v11_h, ck_h, cc, {my_n});
            SetRuntimeArgs(prog_v11_h, hk, cc, {
                (uint32_t)c,
                M_tiles, N_tiles_v11,
                (uint32_t)M, (uint32_t)N,
                my_n,
                hist_a, hist_pgsz_v11,
            });
        }
        wl_v11_hist.add_program(device_range, std::move(prog_v11_h));
    }

    // ══════════════════════════════════════════════════════════════
    // V13_fpu setup: tile-record scatter + TRISC FPU matmul accum.
    // Mirrors v13_full_smoke_host.cpp layout. Programs:
    //   prog_v13_scatter: v11_scatter_b → v13_scatter_brisc renamed actually;
    //                     real names: v13_scatter_brisc (BRISC) +
    //                     v4_compute (TRISC, shared w/ V11) +
    //                     v13_scatter_ncrisc (NCRISC).
    //   prog_v13_accum:   v13_accum_brisc_mt (BRISC) +
    //                     v13_accum_compute_mt (TRISC FPU matmul) +
    //                     v13_accum_ncrisc_void (NCRISC stub).
    // Density buffer is bf16, UNNORMALIZED — host divides by bin_area on D2H.
    // ══════════════════════════════════════════════════════════════
    if (use_v13) {
        if (use_v16) {
            // V16: hash-based ownership decorrelates tile_to_core from
            // spatial position. Hot tiles (concentrated near design center
            // in real ISPD2005 circuits) get spread across many cores,
            // eliminating the per-core hot-bucket overflow that causes
            // V15's dense-2048 divergence. See V16_PLAN.md §6.1 Path A.
            v11::build_hash_ownership(M_tiles_v13, N_tiles_v13, (uint32_t)nc_all,
                                      tile_to_core_v13, core_to_tiles_v13);
        } else {
            v11::build_snake_fill_ownership(M_tiles_v13, N_tiles_v13, (uint32_t)nc_all,
                                            tile_to_core_v13, core_to_tiles_v13);
        }
        total_tiles_v13 = M_tiles_v13 * N_tiles_v13;
        tile_map_bytes_v13 = total_tiles_v13 * sizeof(uint16_t);
        tile_map_pgsz_v13  = (tile_map_bytes_v13 + 31u) & ~31u;

        route_pgsz_v13 = (V13_PAGE_HDR_BYTES + V13_RECORDS_CAP * V13_RECORD_BYTES + 31u) & ~31u;
        uint64_t route_total = (uint64_t)nc_all * (uint64_t)nc_all * route_pgsz_v13;
        if (route_total > 0xFFFFFFFFull) {
            printf("[server] FATAL: V13 route_buf %llu B > 4 GB\n",
                   (unsigned long long)route_total);
            fflush(stdout); std::exit(1);
        }
        owned_lookup_pgsz_v13 = (total_tiles_v13 * (uint32_t)sizeof(uint16_t) + 31u) & ~31u;
        uint32_t owned_lookup_total = (uint32_t)nc_all * owned_lookup_pgsz_v13;
        uint32_t overflow_total = (uint32_t)nc_all * overflow_pgsz_v13;
        density_pgsz_v13 = (uint32_t)N * (uint32_t)sizeof(uint16_t);  // bf16 row
        uint32_t density_total_v13 = (uint32_t)M * density_pgsz_v13;

        n_owned_max_v13 = 0u;
        for (auto& v : core_to_tiles_v13)
            if ((uint32_t)v.size() > n_owned_max_v13) n_owned_max_v13 = (uint32_t)v.size();

        printf("[server] V13: %u tiles (%ux%u), max_owned=%u, route_pgsz=%u, "
               "route_total=%llu MB, density(bf16)=%u KB\n",
               total_tiles_v13, M_tiles_v13, N_tiles_v13, n_owned_max_v13,
               route_pgsz_v13, (unsigned long long)(route_total / (1024u*1024u)),
               density_total_v13 / 1024u);
        fflush(stdout);

        // Allocate V13 buffers.
        tile_map_buf_v13     = make_buf(tile_map_pgsz_v13, tile_map_pgsz_v13);
        owned_lookup_buf_v13 = make_buf(owned_lookup_total, owned_lookup_pgsz_v13);
        route_buf_v13        = make_buf((uint32_t)route_total, route_pgsz_v13);
        overflow_buf_v13     = make_buf(overflow_total, overflow_pgsz_v13);
        density_buf_v13      = make_buf(density_total_v13, density_pgsz_v13);
        // V15 stats buffer: per-core debug dump. 256 bytes per core (64 u32
        // slots, used for per-tile bucket counts and overflow indicators).
        v15_stats_buf        = make_buf((uint32_t)nc_all * 256u, 256u);
        // V15 SPILL buffer is allocated below after v15_bucket_cap is known.

        // Upload tile_to_core.
        {
            std::vector<uint8_t> up(tile_map_pgsz_v13, 0);
            std::memcpy(up.data(), tile_to_core_v13.data(), tile_map_bytes_v13);
            EnqueueWriteMeshBuffer(cq, tile_map_buf_v13, up, false);
        }
        // Upload owned_lookup per core.
        {
            std::vector<uint8_t> up(owned_lookup_total, 0xFF);
            for (uint32_t c = 0; c < (uint32_t)nc_all; ++c) {
                uint16_t* page = reinterpret_cast<uint16_t*>(
                    up.data() + (size_t)c * owned_lookup_pgsz_v13);
                for (uint32_t local = 0; local < core_to_tiles_v13[c].size(); ++local) {
                    page[core_to_tiles_v13[c][local]] = (uint16_t)local;
                }
            }
            EnqueueWriteMeshBuffer(cq, owned_lookup_buf_v13, up, false);
        }
        // Zero overflow + density (defensive).
        {
            std::vector<uint8_t> z(overflow_total, 0);
            EnqueueWriteMeshBuffer(cq, overflow_buf_v13, z, false);
        }
        {
            std::vector<uint8_t> z(density_total_v13, 0);
            EnqueueWriteMeshBuffer(cq, density_buf_v13, z, false);
        }
        Finish(cq);

        // ── PROGRAM 1: V13 scatter ─────────────────────────────────────────
        Program prog_v13_sc = CreateProgram();
        for (uint32_t i = 0; i < 4; ++i) make_cb_all(prog_v13_sc, i, 2);
        make_cb_all(prog_v13_sc, 4, 1); make_cb_all(prog_v13_sc, 5, 1);
        for (uint32_t j = 0; j < V13_MAX_OVERLAP; ++j) {
            make_cb_all(prog_v13_sc, 6  + j, 1);
            make_cb_all(prog_v13_sc, 14 + j, 1);
        }

        uint32_t sc_shared_off_v13 = 0u, sc_brisc_off_v13 = 0u, sc_scratch_v13 = 0u;
        {
            uint32_t off = 0u;
            off += tile_map_bytes_v13;                                 off = (off + 7u) & ~7u;
            off += (uint32_t)nc_all * V13_MAX_IN_FLIGHT * V13_RECORD_BYTES;
            off = (off + 3u) & ~3u;
            off += (uint32_t)nc_all * 4u;                              // staging_count_n
            off += (uint32_t)nc_all * 4u;                              // dram_offset_n
            off = (off + 63u) & ~63u;
            off += V13_PAGE_HDR_BYTES;                                  off = (off + 63u) & ~63u;
            sc_shared_off_v13 = off;
            off += V13_SHARED_BYTES_;                                   off = (off + 63u) & ~63u;
            sc_brisc_off_v13 = off;
            off += (uint32_t)nc_all * V13_MAX_IN_FLIGHT * V13_RECORD_BYTES;
            off = (off + 3u) & ~3u;
            off += (uint32_t)nc_all * 4u;                              // staging_count_b
            off += (uint32_t)nc_all * 4u;                              // dram_offset_b
            off = (off + 63u) & ~63u;
            off += V13_PAGE_HDR_BYTES;
            off += 64u;                                                 // safety gap
            sc_scratch_v13 = (off + 31u) & ~31u;
        }
        printf("[server] V13 scatter scratch = %u KB\n", sc_scratch_v13 / 1024u);
        if (sc_scratch_v13 >= 1536u * 1024u) {
            printf("[server] FATAL: V13 scatter scratch %u KB exceeds L1\n",
                   sc_scratch_v13 / 1024u);
            fflush(stdout); std::exit(1);
        }
        fflush(stdout);
        CreateCircularBuffer(prog_v13_sc, all_crs,
            CircularBufferConfig(sc_scratch_v13, {{24u, tt::DataFormat::Float32}})
                .set_page_size(24u, sc_scratch_v13));

        std::vector<UnpackToDestMode> v13_unpack(NUM_CIRCULAR_BUFFERS,
                                                  UnpackToDestMode::Default);
        for (int i = 0; i < 4; ++i) v13_unpack[i] = UnpackToDestMode::UnpackToDestFp32;

        auto sk_v13_b = CreateKernel(prog_v13_sc, KDIR + "v13_scatter_brisc.cpp", all_crs,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                               .noc = NOC::RISCV_0_default});
        auto ck_v13_sc = CreateKernel(prog_v13_sc, KDIR + "v4_compute.cpp", all_crs,
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4,
                          .fp32_dest_acc_en = true,
                          .unpack_to_dest_mode = v13_unpack,
                          .math_approx_mode = false,
                          .defines = v4_defs});
        auto sk_v13_n = CreateKernel(prog_v13_sc, KDIR + "v13_scatter_ncrisc.cpp", all_crs,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                               .noc = NOC::RISCV_1_default});

        uint32_t sc_data_ready_sem_v13   = CreateSemaphore(prog_v13_sc, all_crs, 0u);
        uint32_t sc_brisc_done_sem_v13   = CreateSemaphore(prog_v13_sc, all_crs, 0u);
        uint32_t sc_tables_ready_sem_v13 = CreateSemaphore(prog_v13_sc, all_crs, 0u);

        uint32_t tm_a_v13 = (uint32_t)tile_map_buf_v13->address();
        uint32_t rt_a_v13 = (uint32_t)route_buf_v13->address();
        uint32_t ov_a_v13 = (uint32_t)overflow_buf_v13->address();

        for (int c = 0; c < nc_all; ++c) {
            auto cc = all_ccs[c];
            uint32_t my_n  = base_tpc + ((uint32_t)c < rem_tpc ? 1u : 0u);
            uint32_t first = (uint32_t)c * base_tpc + std::min((uint32_t)c, rem_tpc);
            // BRISC: combined reader + scatter (cells [512..1024) per cell-tile).
            SetRuntimeArgs(prog_v13_sc, sk_v13_b, cc, {
                px_a, py_a, sx_a, sy_a, tile_pgsz, first, my_n,        // 0..6
                tile_map_bytes_v13,                                      // 7
                (uint32_t)c, (uint32_t)nc_all,                           // 8,9
                M_tiles_v13, N_tiles_v13,                                // 10,11
                (uint32_t)M, (uint32_t)N,                                // 12,13
                rt_a_v13, route_pgsz_v13, V13_RECORDS_CAP,               // 14..16
                (uint32_t)c,                                             // 17 my_writer_id
                sc_data_ready_sem_v13, sc_brisc_done_sem_v13,            // 18,19
                sc_shared_off_v13, sc_brisc_off_v13,                     // 20,21
                sc_tables_ready_sem_v13,                                 // 22
                ov_a_v13,                                                // 23 overflow_dram
            });
            SetRuntimeArgs(prog_v13_sc, ck_v13_sc, cc, {my_n});
            // NCRISC scatter (cells [0..512)).
            SetRuntimeArgs(prog_v13_sc, sk_v13_n, cc, {
                tm_a_v13, tile_map_pgsz_v13, tile_map_bytes_v13,         // 0..2
                (uint32_t)c, (uint32_t)nc_all,                           // 3,4
                M_tiles_v13, N_tiles_v13,                                // 5,6
                (uint32_t)M, (uint32_t)N,                                // 7,8
                my_n,                                                    // 9
                rt_a_v13, route_pgsz_v13, V13_RECORDS_CAP,               // 10..12
                (uint32_t)c,                                             // 13 my_writer_id
                sc_data_ready_sem_v13, sc_brisc_done_sem_v13,            // 14,15
                sc_shared_off_v13, sc_brisc_off_v13,                     // 16,17
                sc_tables_ready_sem_v13,                                 // 18
                ov_a_v13,                                                // 19 overflow_dram
            });
        }
        wl_v13_scatter.add_program(device_range, std::move(prog_v13_sc));

        // ── PROGRAM 2: V13 or V15 accumulate ───────────────────────────────
        Program prog_v13_ac = CreateProgram();
        // bf16 CBs.
        // V13: c_0 OX, c_1 OY, c_16 DENSE.
        // V15: also c_2 OX_N, c_3 OY_N — NCRISC parallel reader's CB pair.
        auto make_cb_bf16_v13 = [&](uint32_t idx, uint32_t n) {
            CreateCircularBuffer(prog_v13_ac, all_crs,
                CircularBufferConfig(n * TILE_BF16_BYTES_v13,
                                     {{idx, tt::DataFormat::Float16_b}})
                    .set_page_size(idx, TILE_BF16_BYTES_v13));
        };
        make_cb_bf16_v13(0,  V13_CB_OXY_DEPTH);   // CB_OX (shared between BRISC and TRISC)
        make_cb_bf16_v13(1,  V13_CB_OXY_DEPTH);   // CB_OY
        make_cb_bf16_v13(16, 2);                  // CB_DENSE

        // L1 scratch layout: V13 uses owned_lookup + hdrs + inbound_buf +
        // per_tile_staging[n_owned × PUSH_BATCH] + dense_row_major.
        // V15 uses owned_lookup + hdrs(BRISC) + inbound_buf(BRISC) +
        // staging(BRISC, single tile) + dense_row_major +
        // hdrs(NCRISC) + inbound_buf(NCRISC) + staging(NCRISC).
        uint32_t ac_scratch_v13 = 0u;
        uint32_t ncrisc_scratch_off_v15 = 0u;
        uint32_t v15_bucket_cap = 0u;
        uint32_t v15_ncrisc_bucket_off = 0u;  // absolute L1 offset of bucket_n
        bool v15_brisc_only = false;
        if (use_v15) {
            // V15 single-pass bucket-by-tile-id layout. BUCKET_CAP is sized
            // dynamically to fit L1 — small grids (n_owned small) get a HUGE
            // cap, large grids (n_owned large) get a smaller cap that's still
            // enough to hold the average + slack records per owned tile.
            //
            // Per-RISC fixed overhead (everything except buckets):
            //   owned_lookup   : total_tiles × 2 B
            //   hdrs           : nc_all × 64 B
            //   inbound_buf    : V13_MAX_READ_RECORDS × 40 B (= 2.5 KB)
            //   bucket_*_count : n_owned × 4 B
            //   staging        : V13_PUSH_BATCH × 40 B (= 5 KB)
            //   dense_row_major (BRISC only) : 2 KB
            //
            // Total L1 budget for accum = 1100 KB (rest reserved for CBs and FW).
            // Total cross-RISC bucket budget = (1100 KB - 2 * fixed_overhead).
            // BUCKET_CAP = (bucket_budget / 2 RISCs) / (n_owned × 40 B).
            uint32_t fixed_per_risc = 0u;
            fixed_per_risc += total_tiles_v13 * (uint32_t)sizeof(uint16_t);
            fixed_per_risc = (fixed_per_risc + 63u) & ~63u;
            fixed_per_risc += (uint32_t)nc_all * V13_PAGE_HDR_BYTES;
            fixed_per_risc = (fixed_per_risc + 63u) & ~63u;
            fixed_per_risc += V13_MAX_READ_RECORDS * V13_RECORD_BYTES;
            fixed_per_risc = (fixed_per_risc + 63u) & ~63u;
            fixed_per_risc += n_owned_max_v13 * (uint32_t)sizeof(uint32_t);
            fixed_per_risc = (fixed_per_risc + 63u) & ~63u;
            fixed_per_risc += V13_PUSH_BATCH * V13_RECORD_BYTES;
            fixed_per_risc = (fixed_per_risc + 63u) & ~63u;
            // BRISC has dense_row_major; pad both equally for simplicity.
            fixed_per_risc += TILE_BF16_BYTES_v13;
            fixed_per_risc = (fixed_per_risc + 63u) & ~63u;
            fixed_per_risc += 256u;  // safety

            uint32_t total_fixed = 2u * fixed_per_risc;
            // V15 scratch budget: 1.5 MB L1 minus CBs (~36 KB for 3 bf16 CBs)
            // and FW (~80 KB). 1350 KB verified to fit at all grids.
            uint32_t budget = 1350u * 1024u;
            uint32_t bucket_budget = (budget > total_fixed) ? (budget - total_fixed) : 0u;
            // Per RISC bucket budget. We give BRISC ALL the bucket budget for
            // small n_owned configs where the hot tile can hold > cap records
            // (denser circuits like adaptec3 / bigblue3). Dropping records
            // there breaks convergence. For larger n_owned (where records
            // spread thin across many owned tiles), 50/50 split keeps the
            // NCRISC parallel-reader win without overflow.
            // brisc_only=true at small n_owned (where BRISC has enough L1 to
            // hold the full bucket). Dual mode would halve pass 1 but doubles
            // pass 2 overhead, net slower at 512 grid.
            uint32_t per_risc_bucket;
            if (n_owned_max_v13 <= 4u) {
                per_risc_bucket = bucket_budget;
                v15_brisc_only = true;
            } else {
                per_risc_bucket = bucket_budget / 2u;
                v15_brisc_only = false;
            }
            uint32_t per_tile_bytes = n_owned_max_v13 * V13_RECORD_BYTES;
            uint32_t cap = (per_tile_bytes > 0u)
                           ? (per_risc_bucket / per_tile_bytes) : 4096u;
            if (cap < 64u)    cap = 64u;     // never let cap go too low
            if (cap > 16384u) cap = 16384u;  // hard upper bound (was 4096)
            // SPILL WRITES are CHUNKED to ≤4 KB inside the kernel, so a single
            // call no longer hits the noc_async_write large-size hang. Cap is
            // still limited to keep L1 bucket size manageable and DRAM spill_buf
            // bounded. 4096 was the pre-spill V15 default and worked well.
            constexpr uint32_t MAX_BUCKET_CAP = 4096u;
            if (cap > MAX_BUCKET_CAP) cap = MAX_BUCKET_CAP;
            if (const char* e = getenv("V15_CAP_OVERRIDE")) {
                uint32_t v = (uint32_t)atoi(e);
                if (v > 0u) cap = v;
            }
            v15_bucket_cap = cap;

            // Now compute the actual scratch layout.
            uint32_t off = 0u;
            // BRISC region
            off += total_tiles_v13 * (uint32_t)sizeof(uint16_t); off = (off + 63u) & ~63u; // owned_lookup
            off += (uint32_t)nc_all * V13_PAGE_HDR_BYTES;        off = (off + 63u) & ~63u; // hdrs
            off += V13_MAX_READ_RECORDS * V13_RECORD_BYTES;      off = (off + 63u) & ~63u; // inbound_buf
            off += n_owned_max_v13 * v15_bucket_cap * V13_RECORD_BYTES; off = (off + 63u) & ~63u; // bucket_b
            off += n_owned_max_v13 * sizeof(uint32_t);           off = (off + 63u) & ~63u; // bucket_b_count
            off += n_owned_max_v13 * sizeof(uint32_t);           off = (off + 63u) & ~63u; // spill_chunks_b
            off += V13_PUSH_BATCH * V13_RECORD_BYTES;            off = (off + 63u) & ~63u; // staging
            off += TILE_BF16_BYTES_v13;                          off = (off + 63u) & ~63u; // dense_row_major
            ncrisc_scratch_off_v15 = off;
            if (v15_brisc_only) {
                // No NCRISC L1 region in brisc-only mode (NCRISC just sets the
                // sem and exits). v15_ncrisc_bucket_off is unused in this mode
                // (BRISC kernel skips the bucket_n drain).
                v15_ncrisc_bucket_off = 0u;
                off += 512u;
            } else {
                // NCRISC region: owned_lookup_n, hdrs_n, inbound_buf_n,
                // bucket_n, bucket_n_count, spill_chunks_n.
                off += total_tiles_v13 * (uint32_t)sizeof(uint16_t); off = (off + 63u) & ~63u;
                off += (uint32_t)nc_all * V13_PAGE_HDR_BYTES;        off = (off + 63u) & ~63u;
                off += V13_MAX_READ_RECORDS * V13_RECORD_BYTES;      off = (off + 63u) & ~63u;
                v15_ncrisc_bucket_off = off;  // ← BRISC reads from here in pass 2
                off += n_owned_max_v13 * v15_bucket_cap * V13_RECORD_BYTES; off = (off + 63u) & ~63u;
                off += n_owned_max_v13 * sizeof(uint32_t);           off = (off + 63u) & ~63u; // bucket_n_count
                off += n_owned_max_v13 * sizeof(uint32_t);           off = (off + 63u) & ~63u; // spill_chunks_n
                off += 512u;
            }
            ac_scratch_v13 = (off + 31u) & ~31u;
            printf("[server] V15 bucket_cap=%u (n_owned_max=%u, total_fixed=%u KB)\n",
                   v15_bucket_cap, n_owned_max_v13, total_fixed / 1024u);
            fflush(stdout);

            // V15 DRAM spill buffer: when a per-(receiver, owned_tile) L1
            // bucket fills to bucket_cap, the kernel writes the bucket to
            // DRAM and resets the L1 count. Pass-2 drain reads spill chunks
            // back, then drains the L1 residual. This eliminates the silent
            // record drops that broke convergence on adaptec3 / bigblue3.
            //
            // Layout:
            //   spill_buf[my_recv_id × n_owned_max × MAX_SPILL_CHUNKS
            //             + local_tile_idx × MAX_SPILL_CHUNKS
            //             + chunk_idx]
            //   where each spill chunk = bucket_cap × 40 bytes.
            //
            // Adaptive sizing: at 512-grid with cap≈10K and n_owned≈3,
            // MAX_SPILL_CHUNKS=4 is plenty (largest hot tile ~25K records);
            // at 2048-grid with cap≈400 and n_owned≈38, MAX_SPILL_CHUNKS=8
            // (hot tiles can have ~3K records per RISC).
            // adaptec3_512 hot tiles have ~25K records. With cap=2048, we need
            // ≥16 chunks (= 32K capacity) to capture them all. At 2048-grid
            // (cap≈400, n_owned≈38), use 16 too — same DRAM cost.
            v15_spill_chunks = 16u;
            v15_spill_pgsz   = ((v15_bucket_cap * V13_RECORD_BYTES) + 31u) & ~31u;
            // 2× — BRISC and NCRISC each get a half of the spill_buf so they
            // never overwrite each other's spilled chunks. BRISC uses lower
            // half (offset 0); NCRISC uses upper half (offset nc_all × ...).
            uint64_t spill_total = 2ull * (uint64_t)nc_all
                                 * (uint64_t)n_owned_max_v13
                                 * (uint64_t)v15_spill_chunks
                                 * (uint64_t)v15_spill_pgsz;
            if (spill_total > 0xFFFFFFFFull) {
                printf("[server] FATAL: V15 spill_buf %llu B > 4 GB\n",
                       (unsigned long long)spill_total);
                fflush(stdout); std::exit(1);
            }
            v15_spill_buf = make_buf((uint32_t)spill_total, v15_spill_pgsz);
            printf("[server] V15 spill_buf: %llu MB (max_chunks=%u, page=%u B)\n",
                   (unsigned long long)(spill_total / (1024u*1024u)),
                   v15_spill_chunks, v15_spill_pgsz);
            fflush(stdout);
        } else {
            uint32_t off = 0u;
            off += total_tiles_v13 * (uint32_t)sizeof(uint16_t); off = (off + 63u) & ~63u;
            off += (uint32_t)nc_all * V13_PAGE_HDR_BYTES;        off = (off + 63u) & ~63u;
            off += V13_MAX_READ_RECORDS * V13_RECORD_BYTES;      off = (off + 63u) & ~63u;
            // OPT-1: per_tile_staging holds PUSH_BATCH (=128) records per tile
            off += n_owned_max_v13 * V13_PUSH_BATCH * V13_RECORD_BYTES; off = (off + 63u) & ~63u;
            off += n_owned_max_v13 * 4u;                          off = (off + 63u) & ~63u;
            off += TILE_BF16_BYTES_v13;                           off = (off + 63u) & ~63u;
            off += 256u;                                          // safety
            ac_scratch_v13 = (off + 31u) & ~31u;
        }
        printf("[server] %s accum scratch = %u KB (max_owned=%u)\n",
               use_v15 ? "V15" : "V13",
               ac_scratch_v13 / 1024u, n_owned_max_v13);
        if (ac_scratch_v13 >= 1536u * 1024u) {
            printf("[server] FATAL: %s accum scratch %u KB exceeds L1\n",
                   use_v15 ? "V15" : "V13",
                   ac_scratch_v13 / 1024u);
            fflush(stdout); std::exit(1);
        }
        fflush(stdout);
        CreateCircularBuffer(prog_v13_ac, all_crs,
            CircularBufferConfig(ac_scratch_v13, {{24u, tt::DataFormat::Float32}})
                .set_page_size(24u, ac_scratch_v13));

        // Pick kernel source files based on gather mode.
        const std::string brisc_kernel = use_v15 ? "v15_gather_brisc.cpp"
                                                  : "v13_accum_brisc_mt.cpp";
        const std::string ncrisc_kernel = use_v15 ? "v15_gather_ncrisc.cpp"
                                                   : "v13_accum_ncrisc_void.cpp";
        const std::string compute_kernel = use_v15 ? "v15_gather_compute.cpp"
                                                    : "v13_accum_compute_mt.cpp";

        // V15 spill is gated by compile-time define V15_SPILL_ENABLED. We only
        // enable it at small n_owned (≤16) where bucket overflow can happen but
        // total bucket loop iterations are few. At n_owned>16 (2048-grid configs),
        // records spread thin across many owned tiles so overflow is impossible
        // (cap=438 vs avg ~200 records/bucket), and emitting the flush code in
        // the binary causes a kernel-side deadlock — even when never executed.
        std::map<std::string, std::string> v15_brisc_defines;
        std::map<std::string, std::string> v15_ncrisc_defines;
        bool v15_force_no_spill = (getenv("V15_FORCE_NO_SPILL") &&
                                   std::string(getenv("V15_FORCE_NO_SPILL")) == "1");
        // Spill enabled only when n_owned ≤ 16. At n_owned > 16 (2048-grid)
        // the spill lambda's binary presence causes a kernel-side completion-
        // queue deadlock even when never executed — root cause unknown, likely
        // I-cache / NOC state interaction. Chunked-write fix only helps the
        // large-write hang (small-grid case), not the high-n_owned case.
        if (use_v15 && n_owned_max_v13 <= 16u && !v15_force_no_spill) {
            v15_brisc_defines["V15_SPILL_ENABLED"] = "1";
            v15_ncrisc_defines["V15_SPILL_ENABLED"] = "1";
        }
        if (use_v15) {
            printf("[server] V15 spill compile-flag: %s\n",
                   v15_brisc_defines.count("V15_SPILL_ENABLED") ? "ENABLED" : "DISABLED");
            fflush(stdout);
        }

        auto ak_v13_b = CreateKernel(prog_v13_ac, KDIR + brisc_kernel, all_crs,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                               .noc = NOC::RISCV_0_default,
                               .defines = v15_brisc_defines});
        auto ak_v13_n = CreateKernel(prog_v13_ac, KDIR + ncrisc_kernel, all_crs,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                               .noc = NOC::RISCV_1_default,
                               .defines = v15_ncrisc_defines});
        auto ak_v13_c = CreateKernel(prog_v13_ac, KDIR + compute_kernel, all_crs,
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4,
                          .fp32_dest_acc_en = true,
                          .math_approx_mode = false});

        uint32_t ol_a_v13 = (uint32_t)owned_lookup_buf_v13->address();
        uint32_t da_a_v13 = (uint32_t)density_buf_v13->address();

        // V15 needs an extra semaphore for NCRISC→BRISC bucketing-done signal.
        uint32_t v15_ncrisc_done_sem = 0u;
        if (use_v15) {
            v15_ncrisc_done_sem = CreateSemaphore(prog_v13_ac, all_crs, 0u);
        }

        for (int c = 0; c < nc_all; ++c) {
            auto cc = all_ccs[c];
            uint32_t n_owned = (uint32_t)core_to_tiles_v13[c].size();
            std::vector<uint32_t> args_brisc = {
                ol_a_v13, owned_lookup_pgsz_v13,
                (uint32_t)c, (uint32_t)nc_all,
                M_tiles_v13, N_tiles_v13,
                (uint32_t)M, (uint32_t)N,
                rt_a_v13, route_pgsz_v13, V13_RECORDS_CAP,
                da_a_v13, density_pgsz_v13,
                n_owned,
            };
            if (use_v15) {
                args_brisc.push_back(v15_bucket_cap);            // 14
                args_brisc.push_back(v15_ncrisc_done_sem);       // 15
                args_brisc.push_back(v15_ncrisc_bucket_off);     // 16
                args_brisc.push_back(v15_brisc_only ? 1u : 0u);  // 17
                args_brisc.push_back((uint32_t)v15_stats_buf->address());  // 18
                args_brisc.push_back((uint32_t)v15_spill_buf->address());  // 19
                args_brisc.push_back(v15_spill_pgsz);            // 20
                args_brisc.push_back(v15_spill_chunks);          // 21
                args_brisc.push_back(n_owned_max_v13);           // 22 — spill slot stride
            }
            for (uint32_t i = 0; i < n_owned; ++i) {
                args_brisc.push_back(core_to_tiles_v13[c][i]);
            }
            SetRuntimeArgs(prog_v13_ac, ak_v13_b, cc, args_brisc);

            if (use_v15) {
                // V15 NCRISC args:
                // 0..7: my_core_id, nc_all, M_tiles, N_tiles, route_dram,
                //       route_pgsz, records_cap, n_owned_tiles
                // 8: ncrisc_scratch_off
                // 9..10: owned_lookup_dram, owned_lookup_pgsz
                // 11: bucket_cap
                // 12: ncrisc_done_sem_id
                std::vector<uint32_t> args_ncrisc = {
                    (uint32_t)c, (uint32_t)nc_all,                              // 0..1
                    M_tiles_v13, N_tiles_v13,                                   // 2..3
                    rt_a_v13, route_pgsz_v13, V13_RECORDS_CAP,                  // 4..6
                    n_owned, ncrisc_scratch_off_v15,                            // 7..8
                    ol_a_v13, owned_lookup_pgsz_v13,                            // 9..10
                    v15_bucket_cap,                                             // 11
                    v15_ncrisc_done_sem,                                        // 12
                    v15_brisc_only ? 1u : 0u,                                   // 13
                    (uint32_t)v15_spill_buf->address(),                         // 14: spill_dram
                    v15_spill_pgsz,                                             // 15
                    v15_spill_chunks,                                           // 16
                    n_owned_max_v13,                                            // 17: spill slot stride
                };
                SetRuntimeArgs(prog_v13_ac, ak_v13_n, cc, args_ncrisc);
            } else {
                SetRuntimeArgs(prog_v13_ac, ak_v13_n, cc, {});
            }
            SetRuntimeArgs(prog_v13_ac, ak_v13_c, cc, {n_owned});
        }
        wl_v13_accum.add_program(device_range, std::move(prog_v13_ac));
    }

    // ══════════════════════════════════════════════════════════════════════
    // V14 Architecture-A: writer-side FPU scatter + simple reduce
    // ══════════════════════════════════════════════════════════════════════
    if (use_v14) {
        v11::build_snake_fill_ownership(M_tiles_v14, N_tiles_v14, (uint32_t)nc_all,
                                        tile_to_core_v14, core_to_tiles_v14);
        total_tiles_v14 = M_tiles_v14 * N_tiles_v14;
        tile_map_bytes_v14 = total_tiles_v14 * sizeof(uint16_t);
        tile_map_pgsz_v14  = (tile_map_bytes_v14 + 31u) & ~31u;

        density_pgsz_v14 = (uint32_t)N * (uint32_t)sizeof(uint16_t);  // bf16 row
        uint32_t density_total_v14 = (uint32_t)M * density_pgsz_v14;

        // partial_buf: nc_all × total_tiles pages of 2048 bytes each.
        uint64_t partial_total_v14 = (uint64_t)nc_all * (uint64_t)total_tiles_v14 * partial_pgsz_v14;

        n_owned_max_v14 = 0u;
        for (auto& v : core_to_tiles_v14)
            if ((uint32_t)v.size() > n_owned_max_v14) n_owned_max_v14 = (uint32_t)v.size();

        printf("[server] V14: %u tiles (%ux%u), max_owned=%u, partial_buf=%llu MB, "
               "density(bf16)=%u KB\n",
               total_tiles_v14, M_tiles_v14, N_tiles_v14, n_owned_max_v14,
               (unsigned long long)(partial_total_v14 / (1024u*1024u)),
               density_total_v14 / 1024u);
        fflush(stdout);

        // Allocate V14 buffers.
        tile_map_buf_v14  = make_buf(tile_map_pgsz_v14, tile_map_pgsz_v14);
        partial_buf_v14   = make_buf((uint32_t)partial_total_v14, partial_pgsz_v14);
        density_buf_v14   = make_buf(density_total_v14, density_pgsz_v14);

        // DROP-INSTRUMENTATION: 1 uint32 per writer (= nc_all uint32s total).
        // BRISC writes its bucket-overflow drop count here at end of each iter.
        v14_drop_buf_pgsz_outer = (uint32_t)nc_all * 4u;
        v14_drop_buf_pgsz_outer = (v14_drop_buf_pgsz_outer + 31u) & ~31u;
        v14_drop_buf = make_buf(v14_drop_buf_pgsz_outer, v14_drop_buf_pgsz_outer);

        // Upload tile_to_core.
        {
            std::vector<uint8_t> up(tile_map_pgsz_v14, 0);
            std::memcpy(up.data(), tile_to_core_v14.data(), tile_map_bytes_v14);
            EnqueueWriteMeshBuffer(cq, tile_map_buf_v14, up, false);
        }
        // Zero partial + density (defensive).
        {
            std::vector<uint8_t> z((size_t)partial_total_v14, 0);
            EnqueueWriteMeshBuffer(cq, partial_buf_v14, z, false);
        }
        {
            std::vector<uint8_t> z(density_total_v14, 0);
            EnqueueWriteMeshBuffer(cq, density_buf_v14, z, false);
        }
        Finish(cq);

        // ── PROGRAM 1: V14 scatter (SFPU + FPU) ─────────────────────────────
        Program prog_v14_sc = CreateProgram();

        // Phase 1 CBs: c_0..c_3 fp32 depth 2 (SFPU inputs)
        for (uint32_t i = 0; i < 4; ++i) make_cb_all(prog_v14_sc, i, 2);
        // c_4..c_5 fp32 depth 2 (bxl/byl outputs — double-buffered for SFPU pipelining)
        make_cb_all(prog_v14_sc, 4, 2);
        make_cb_all(prog_v14_sc, 5, 2);
        // c_6..c_21 fp32 depth 2 (ox[0..7], oy[0..7] — double-buffered)
        for (uint32_t j = 0; j < V14_MAX_OVERLAP; ++j) {
            make_cb_all(prog_v14_sc, 6 + j, 2);
            make_cb_all(prog_v14_sc, 14 + j, 2);
        }

        // Phase 2 CBs: c_22, c_23 bf16 depth V14_CB_OXY_DEPTH (OX_FPU, OY_FPU)
        auto make_cb_bf16_v14 = [&](Program& p, uint32_t idx, uint32_t n) {
            CreateCircularBuffer(p, all_crs,
                CircularBufferConfig(n * TILE_BF16_BYTES_v14,
                                     {{idx, tt::DataFormat::Float16_b}})
                    .set_page_size(idx, TILE_BF16_BYTES_v14));
        };
        make_cb_bf16_v14(prog_v14_sc, 22, V14_CB_OXY_DEPTH);
        make_cb_bf16_v14(prog_v14_sc, 23, V14_CB_OXY_DEPTH);
        // c_25 bf16 depth 2 (CB_PARTIAL)
        make_cb_bf16_v14(prog_v14_sc, 25, 2);

        // CB_SCRATCH (c_24): tile_to_core + tile_buckets + counts + dense_row_major + active_tile_list
        // BUCKET_CAP is computed dynamically so total scratch fits within 1100 KB L1 budget,
        // leaving ~436 KB for the 22 other CBs (212 KB) plus firmware/stack overhead.
        constexpr uint32_t V14_L1_SCRATCH_BUDGET = 1100u * 1024u;
        constexpr uint32_t V14_RECORD_BYTES_HOST  = 40u;
        uint32_t V14_BUCKET_CAP;
        {
            // Fixed overhead (not scaled by BUCKET_CAP)
            uint32_t fixed = 0u;
            fixed += tile_map_bytes_v14;              fixed = (fixed + 63u) & ~63u;
            fixed += total_tiles_v14 * 4u;            fixed = (fixed + 63u) & ~63u;  // counts
            fixed += TILE_BF16_BYTES_v14;             fixed = (fixed + 63u) & ~63u;  // dense_row_major
            fixed += total_tiles_v14 * 4u;            fixed = (fixed + 63u) & ~63u;  // active_tile_list
            fixed += 512u;                                                              // safety

            uint32_t bucket_bytes = (V14_L1_SCRATCH_BUDGET > fixed)
                                    ? (V14_L1_SCRATCH_BUDGET - fixed) : 0u;
            uint32_t cap = bucket_bytes / (total_tiles_v14 * V14_RECORD_BYTES_HOST);
            // Test: lift artificial 64-cap ceiling. With ceiling=256 the L1
            // budget naturally caps it. At 512-grid this allows ~109; at 1024
            // ~27 (already L1-limited); at 2048 ~6 (already L1-limited).
            const char* cap_env = std::getenv("V14_BUCKET_CAP_OVERRIDE");
            uint32_t cap_ceil = cap_env ? (uint32_t)std::atoi(cap_env) : 256u;
            V14_BUCKET_CAP = std::max(4u, std::min(cap_ceil, cap));
        }

        uint32_t sc_scratch_v14 = 0u;
        {
            uint32_t off = 0u;
            off += tile_map_bytes_v14;                                          off = (off + 63u) & ~63u;
            off += total_tiles_v14 * V14_BUCKET_CAP * V14_RECORD_BYTES_HOST;  off = (off + 63u) & ~63u;
            off += total_tiles_v14 * 4u;                                        off = (off + 63u) & ~63u;
            off += TILE_BF16_BYTES_v14;                                         off = (off + 63u) & ~63u;
            off += total_tiles_v14 * 4u;                                        off = (off + 63u) & ~63u;
            off += 512u;
            sc_scratch_v14 = (off + 31u) & ~31u;
        }
        printf("[server] V14 scatter scratch = %u KB  bucket_cap=%u\n",
               sc_scratch_v14 / 1024u, V14_BUCKET_CAP);
        if (sc_scratch_v14 >= 1200u * 1024u) {
            printf("[server] FATAL: V14 scatter scratch %u KB exceeds safe limit\n",
                   sc_scratch_v14 / 1024u);
            fflush(stdout); std::exit(1);
        }
        fflush(stdout);
        CreateCircularBuffer(prog_v14_sc, all_crs,
            CircularBufferConfig(sc_scratch_v14, {{24u, tt::DataFormat::Float32}})
                .set_page_size(24u, sc_scratch_v14));

        std::vector<UnpackToDestMode> v14_unpack(NUM_CIRCULAR_BUFFERS,
                                                  UnpackToDestMode::Default);
        for (int i = 0; i < 4; ++i) v14_unpack[i] = UnpackToDestMode::UnpackToDestFp32;

        // Pass dynamic BUCKET_CAP as a JIT define so the kernel recompiles per grid size.
        std::map<std::string, std::string> v14_brisc_defs = {
            {"V14_BUCKET_CAP", std::to_string(V14_BUCKET_CAP)},
        };
        auto sk_v14_b = CreateKernel(prog_v14_sc, KDIR + "v14_scatter_brisc.cpp", all_crs,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                               .noc = NOC::RISCV_0_default,
                               .defines = v14_brisc_defs});
        auto ck_v14_sc = CreateKernel(prog_v14_sc, KDIR + "v14_scatter_compute.cpp", all_crs,
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4,
                          .fp32_dest_acc_en = true,
                          .unpack_to_dest_mode = v14_unpack,
                          .math_approx_mode = false,
                          .defines = v4_defs});
        auto sk_v14_n = CreateKernel(prog_v14_sc, KDIR + "v14_scatter_ncrisc.cpp", all_crs,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                               .noc = NOC::RISCV_1_default});

        uint32_t sc_tables_ready_sem_v14 = CreateSemaphore(prog_v14_sc, all_crs, 0u);

        uint32_t tm_a_v14 = (uint32_t)tile_map_buf_v14->address();
        uint32_t pa_a_v14 = (uint32_t)partial_buf_v14->address();
        uint32_t da_drop_v14 = (uint32_t)v14_drop_buf->address();

        for (int c = 0; c < nc_all; ++c) {
            auto cc = all_ccs[c];
            uint32_t my_n  = base_tpc + ((uint32_t)c < rem_tpc ? 1u : 0u);
            uint32_t first = (uint32_t)c * base_tpc + std::min((uint32_t)c, rem_tpc);
            // BRISC
            SetRuntimeArgs(prog_v14_sc, sk_v14_b, cc, {
                px_a, py_a, sx_a, sy_a, tile_pgsz, first, my_n,        // 0..6
                tile_map_bytes_v14,                                      // 7
                (uint32_t)c,                                             // 8 (unused)
                (uint32_t)nc_all,                                        // 9
                M_tiles_v14, N_tiles_v14,                                // 10,11
                (uint32_t)M, (uint32_t)N,                                // 12,13
                total_tiles_v14,                                         // 14
                pa_a_v14, partial_pgsz_v14,                              // 15,16
                (uint32_t)c,                                             // 17 my_writer_id
                sc_tables_ready_sem_v14,                                 // 18
                da_drop_v14,                                             // 19 drop buf base
            });
            // TRISC: n_scatter_tiles (Phase 1 SFPU). Phase 2 uses sentinel-
            // driven loop — BRISC pushes ALL_DONE when no more tile groups.
            SetRuntimeArgs(prog_v14_sc, ck_v14_sc, cc, {my_n});
            // NCRISC
            SetRuntimeArgs(prog_v14_sc, sk_v14_n, cc, {
                tm_a_v14, tile_map_pgsz_v14, tile_map_bytes_v14,         // 0..2
                sc_tables_ready_sem_v14,                                 // 3
            });
        }
        wl_v14_scatter.add_program(device_range, std::move(prog_v14_sc));

        // ── PROGRAM 2: V14 reduce (FPU add_tiles) ──────────────────────────
        Program prog_v14_rd = CreateProgram();

        // CB_A (c_0): bf16 input tiles A from NCRISC reader, depth=2
        constexpr uint32_t V14_RD_CB_DEPTH = 2u;
        CreateCircularBuffer(prog_v14_rd, all_crs,
            CircularBufferConfig(V14_RD_CB_DEPTH * TILE_BF16_BYTES_v14,
                                 {{0u, tt::DataFormat::Float16_b}})
                .set_page_size(0u, TILE_BF16_BYTES_v14));
        // CB_B (c_1): bf16 input tiles B from NCRISC reader, depth=2
        CreateCircularBuffer(prog_v14_rd, all_crs,
            CircularBufferConfig(V14_RD_CB_DEPTH * TILE_BF16_BYTES_v14,
                                 {{1u, tt::DataFormat::Float16_b}})
                .set_page_size(1u, TILE_BF16_BYTES_v14));

        // CB_OUT (c_16): bf16 accumulated tile from TRISC, depth=1
        CreateCircularBuffer(prog_v14_rd, all_crs,
            CircularBufferConfig(1u * TILE_BF16_BYTES_v14,
                                 {{16u, tt::DataFormat::Float16_b}})
                .set_page_size(16u, TILE_BF16_BYTES_v14));

        // CB_SCRATCH (c_2): bf16 scratch for BRISC row-major conversion, depth=1
        // NOC DMA cannot read from stack/BSS L1, only from CB-allocated L1.
        CreateCircularBuffer(prog_v14_rd, all_crs,
            CircularBufferConfig(1u * TILE_BF16_BYTES_v14,
                                 {{2u, tt::DataFormat::Float16_b}})
                .set_page_size(2u, TILE_BF16_BYTES_v14));

        printf("[server] V14 reduce FPU: CB_A/B=%u KB each  CB_OUT=%u KB  CB_SCRATCH=%u KB  (max_owned=%u)\n",
               V14_RD_CB_DEPTH * TILE_BF16_BYTES_v14 / 1024u,
               TILE_BF16_BYTES_v14 / 1024u,
               TILE_BF16_BYTES_v14 / 1024u,
               n_owned_max_v14);
        fflush(stdout);

        // BRISC: density writer
        auto rk_v14_b = CreateKernel(prog_v14_rd, KDIR + "v14_reduce_brisc.cpp", all_crs,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                               .noc = NOC::RISCV_0_default});
        // NCRISC: partial tile reader
        auto rk_v14_n = CreateKernel(prog_v14_rd, KDIR + "v14_reduce_ncrisc.cpp", all_crs,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                               .noc = NOC::RISCV_1_default});
        // TRISC: FPU eltwise add accumulation
        auto rk_v14_c = CreateKernel(prog_v14_rd, KDIR + "v14_reduce_compute.cpp", all_crs,
            ComputeConfig{.math_fidelity = MathFidelity::LoFi,
                          .fp32_dest_acc_en = false,
                          .math_approx_mode = false});

        uint32_t da_a_v14 = (uint32_t)density_buf_v14->address();
        for (int c = 0; c < nc_all; ++c) {
            auto cc = all_ccs[c];
            uint32_t n_owned = (uint32_t)core_to_tiles_v14[c].size();

            // BRISC args: density_dram, density_pgsz, N_tiles, nbx, nby, n_owned, tile_ids...
            std::vector<uint32_t> brisc_args = {
                da_a_v14, density_pgsz_v14,
                N_tiles_v14,
                (uint32_t)M, (uint32_t)N,
                n_owned,
            };
            for (uint32_t i = 0; i < n_owned; ++i)
                brisc_args.push_back(core_to_tiles_v14[c][i]);
            SetRuntimeArgs(prog_v14_rd, rk_v14_b, cc, brisc_args);

            // NCRISC args: partial_dram, partial_pgsz, nc_all, n_owned, tile_ids...
            std::vector<uint32_t> ncrisc_args = {
                pa_a_v14, partial_pgsz_v14,
                (uint32_t)nc_all,
                n_owned,
            };
            for (uint32_t i = 0; i < n_owned; ++i)
                ncrisc_args.push_back(core_to_tiles_v14[c][i]);
            SetRuntimeArgs(prog_v14_rd, rk_v14_n, cc, ncrisc_args);

            // TRISC args: n_owned, nc_all
            SetRuntimeArgs(prog_v14_rd, rk_v14_c, cc, {n_owned, (uint32_t)nc_all});
        }
        wl_v14_reduce.add_program(device_range, std::move(prog_v14_rd));
    }

    // ── JIT compile ───────────────────────────────────────────────
    printf("[server] JIT compiling kernels...\n"); fflush(stdout);
    auto t_jit = hrclock::now();
    if (!use_v11 && !use_v13 && !use_v14) {
        EnqueueMeshWorkload(cq, wl_scatter, false); Finish(cq);
        EnqueueMeshWorkload(cq, wl_gather,  false); Finish(cq);
    }
    if (use_v11) {
        printf("[server]   V11 hist JIT...\n"); fflush(stdout);
        EnqueueMeshWorkload(cq, wl_v11_hist,    false); Finish(cq);
        if (use_v19) {
            // V19 scatter kernel running with garbage warmup DRAM data appears
            // to deadlock at 2048 grid (very long zero-init + atomic ops issued
            // against uninitialized cell positions). Skip warmup of wl_v11_scatter
            // and wl_v11_accum for V19; first real iter will trigger JIT with
            // valid Python data. V19 doesn't use wl_v11_accum at all in real iters.
            printf("[server]   V19: skipping V11 scatter / accum warmup (will JIT on first real iter)\n");
            fflush(stdout);
        } else {
            printf("[server]   V11 scatter JIT...\n"); fflush(stdout);
            EnqueueMeshWorkload(cq, wl_v11_scatter, false); Finish(cq);
            printf("[server]   V11 accum JIT...\n"); fflush(stdout);
            EnqueueMeshWorkload(cq, wl_v11_accum,   false); Finish(cq);
        }
        if (use_v19) {
            printf("[server]   V19 writeout JIT...\n"); fflush(stdout);
            EnqueueMeshWorkload(cq, wl_v19_writeout, false); Finish(cq);
        }
        printf("[server]   V11 (and V19) JIT done.\n"); fflush(stdout);
    }
    if (use_v13) {
        EnqueueMeshWorkload(cq, wl_v13_scatter, false); Finish(cq);
        EnqueueMeshWorkload(cq, wl_v13_accum,   false); Finish(cq);
    }
    if (use_v14) {
        printf("[server]   V14 scatter JIT...\n"); fflush(stdout);
        EnqueueMeshWorkload(cq, wl_v14_scatter, false); Finish(cq);
        printf("[server]   V14 scatter JIT done. Now V14 reduce JIT...\n"); fflush(stdout);
        EnqueueMeshWorkload(cq, wl_v14_reduce,  false); Finish(cq);
        printf("[server]   V14 reduce JIT done.\n"); fflush(stdout);
    }
    printf("[server] JIT done: %.1f ms\n", ms_since(t_jit)); fflush(stdout);

    // ── Upload DCT matrices as TTNN tensors ───────────────────────
    printf("[v19_engine] Uploading DCT matrices via TTNN...\n"); fflush(stdout);
    auto& ttnn_solver = ttnn_solver_;  // alias to member so init below writes through
    ttnn_solver.init(M, N, bsx, bsy, mesh_device.get());
    ttnn_solver_ready_ = true;
    printf("[v19_engine] TTNN solver ready.\n"); fflush(stdout);

    // ── Copy locals → members so the scatter() method can see them. ──
    // (Lambdas / locals declared in the setup body above die at end-of-scope;
    //  these members survive across scatter() calls.)
    soa_padded_     = soa_padded;
    Mt_             = Mt;
    nc_all_         = nc_all;
    M_tiles_        = M_tiles;
    N_tiles_v11_    = N_tiles_v11;
    hist_pgsz_v11_  = hist_pgsz_v11;
    bsx_            = bsx;
    bsy_            = bsy;
    inv_ba_         = inv_ba;

    // Buffers (shared_ptr — cheap to copy, refcount bump).
    px_buf_      = px_buf;
    py_buf_      = py_buf;
    sx_buf_      = sx_buf;
    sy_buf_      = sy_buf;
    contrib_buf_ = contrib_buf;
    density_buf_ = density_buf;
    strips_buf_  = strips_buf;
    tile_map_buf_     = tile_map_buf;
    route_buf_        = route_buf;
    owned_lookup_buf_ = owned_lookup_buf;
    hist_buf_         = hist_buf;
    shard_table_buf_  = shard_table_buf;
    shard_reduce_buf_ = shard_reduce_buf;
    drop_buf_         = drop_buf;
    v19_density_dram_outer_         = v19_density_dram_outer;
    v19_density_slab_bins_outer_    = v19_density_slab_bins_outer;
    v19_density_dram_pgsz_outer_    = v19_density_dram_pgsz_outer;

    // Workloads (MeshWorkload move-only).
    wl_v11_scatter_  = std::move(wl_v11_scatter);
    wl_v11_accum_    = std::move(wl_v11_accum);
    wl_v11_hist_     = std::move(wl_v11_hist);
    wl_v19_writeout_ = std::move(wl_v19_writeout);

    // Per-iter scratch vectors — sized once here, reused across scatter()s.
    px_.assign(soa_padded, 0.0f);
    py_.assign(soa_padded, 0.0f);
    sx_.assign(soa_padded, 0.0f);
    sy_.assign(soa_padded, 0.0f);
    density_flat_.assign((size_t)M * (size_t)N, 0.0f);
    field_x_.assign((size_t)M * (size_t)N, 0.0f);
    field_y_.assign((size_t)M * (size_t)N, 0.0f);

    if (use_v11) {
        v11_hist_data_.assign((size_t)nc_all * (hist_pgsz_v11 / 4u), 0u);
        v11_global_count_.assign((size_t)M_tiles * N_tiles_v11, 0u);
    }

    printf("[v19_engine] init complete.\n"); fflush(stdout);
}

// ═══════════════════════════════════════════════════════════════════
// V19Engine::Impl::set_initial_density — one-time copy of normalized
// initial density (folded into density_flat before TTNN DCT when
// CPU_DCT=0; ignored on CPU_DCT=1 path since the Python wrapper
// re-applies it after density readback).
// ═══════════════════════════════════════════════════════════════════
void V19Engine::Impl::set_initial_density(const float* id_normalized) {
    if (id_normalized == nullptr) return;
    initial_density_.assign(id_normalized,
                            id_normalized + (size_t)M_ * (size_t)N_);
    initial_density_set_ = true;
}

// ═══════════════════════════════════════════════════════════════════
// V19Engine::Impl::scatter — ported from server_host.cpp's while(true)
// loop body. shm reads/writes replaced by pointer args; the state-flag
// polling + SHM_STATE_DONE handshake is stripped (caller drives now).
// ═══════════════════════════════════════════════════════════════════
void V19Engine::Impl::scatter(const float* px_in, const float* py_in,
                              const float* sx_in, const float* sy_in,
                              int32_t nc_actual_in,
                              float* density_out,
                              float* field_x_out, float* field_y_out,
                              ScatterTiming* timing_out) {
    auto t_total_start = hrclock::now();

    // ── Bind member fields back to the names the ported loop body uses. ──
    auto& cq            = *cq_;
    auto& mesh_device   = mesh_device_;
    int   M             = M_;
    int   N             = N_;
    int   NC_max        = NC_max_;
    uint32_t soa_padded = soa_padded_;
    uint32_t Mt         = Mt_;
    int      nc_all     = nc_all_;
    uint32_t M_tiles    = M_tiles_;
    uint32_t N_tiles_v11= N_tiles_v11_;
    uint32_t hist_pgsz_v11 = hist_pgsz_v11_;
    float    bsx        = bsx_;
    float    bsy        = bsy_;
    float    inv_ba     = inv_ba_;
    bool     use_v6     = use_v6_;
    bool     use_v7     = use_v7_;
    bool     use_v8     = use_v8_;
    bool     use_v9     = use_v9_;
    bool     use_v10    = use_v10_;
    bool     use_v11    = use_v11_;
    bool     use_v11outer = use_v11outer_;
    bool     use_v13    = use_v13_;
    bool     use_v14    = use_v14_;
    bool     use_v15    = use_v15_;
    bool     use_v16    = use_v16_;
    bool     use_v18    = use_v18_;
    bool     use_v18outer = use_v18outer_;
    bool     use_v19    = use_v19_;
    auto&    px_buf     = px_buf_;
    auto&    py_buf     = py_buf_;
    auto&    sx_buf     = sx_buf_;
    auto&    sy_buf     = sy_buf_;
    auto&    contrib_buf= contrib_buf_;
    auto&    density_buf= density_buf_;
    auto&    strips_buf = strips_buf_;
    auto&    tile_map_buf    = tile_map_buf_;
    auto&    route_buf       = route_buf_;
    auto&    owned_lookup_buf= owned_lookup_buf_;
    auto&    hist_buf        = hist_buf_;
    auto&    shard_table_buf = shard_table_buf_;
    auto&    shard_reduce_buf= shard_reduce_buf_;
    auto&    drop_buf        = drop_buf_;
    auto&    v19_density_dram_outer = v19_density_dram_outer_;
    uint32_t v19_density_slab_bins_outer = v19_density_slab_bins_outer_;
    uint32_t v19_density_dram_pgsz_outer = v19_density_dram_pgsz_outer_;
    auto&    wl_v11_scatter  = wl_v11_scatter_;
    auto&    wl_v11_accum    = wl_v11_accum_;
    auto&    wl_v11_hist     = wl_v11_hist_;
    auto&    wl_v19_writeout = wl_v19_writeout_;
    auto&    ttnn_solver     = ttnn_solver_;

    // Stubs for modes this engine doesn't support. The original loop body
    // has `if (use_v13) {...}` / `if (use_v14) {...}` branches that name
    // these symbols; we hit those branches as dead code only (use_v13/14
    // are forced false at ctor time). The stubs let it compile.
    MeshWorkload wl_v13_scatter, wl_v13_accum;
    MeshWorkload wl_v14_scatter, wl_v14_reduce;
    std::shared_ptr<MeshBuffer> density_buf_v13;
    std::shared_ptr<MeshBuffer> density_buf_v14;
    std::shared_ptr<MeshBuffer> partial_buf_v14;
    std::shared_ptr<MeshBuffer> tile_map_buf_v13;
    std::shared_ptr<MeshBuffer> route_buf_v13;
    std::shared_ptr<MeshBuffer> owned_lookup_buf_v13;
    std::shared_ptr<MeshBuffer> v14_drop_buf;
    std::shared_ptr<MeshBuffer> v13_overflow_buf;
    std::shared_ptr<MeshBuffer> v13_stats_buf;
    std::shared_ptr<MeshBuffer> v13_tile_map_buf;
    std::shared_ptr<MeshBuffer> v15_stats_buf;
    uint32_t v13_owned_pgsz = 0u, v13_tile_map_pgsz = 0u, v13_route_pgsz = 0u;
    uint32_t v14_density_pgsz = 0u, v14_partial_pgsz = 0u;
    uint32_t v13_density_pgsz = 0u, v13_overflow_pgsz = 0u, v13_stats_pgsz = 0u;
    uint32_t v15_stats_pgsz = 0u;
    bool v13_use_v15_path = false;
    // V14 dead-branch stubs (use_v14=false so these are never read).
    uint32_t total_tiles_v14 = 0u, partial_pgsz_v14 = 0u, v14_drop_buf_pgsz_outer = 0u;

    // V6/V7/V8/V9/V10 dead-branch stubs (engine only supports V11-family).
    MeshWorkload wl_scatter, wl_gather;

    // Coords (members).
    float xl = xl_, yl = yl_, xh = xh_, yh = yh_;

    // V11 setup-time locals needed by debug-dump path inside scatter().
    // route_buf is captured as an Impl member; the page-size + tuple cap
    // are saved here so the EXPORT_ROUTE_BUF_PATH dump path can format
    // the buffer. Defaulting to 0 is safe — that path is opt-in via env.
    uint32_t route_pgsz_v11 = 0u;
    uint32_t v11_max_per_page_tuples = 0u;

    // Per-iter scratch (lives across iters as members).
    auto& px = px_;
    auto& py = py_;
    auto& sx = sx_;
    auto& sy = sy_;
    auto& density_flat = density_flat_;
    auto& field_x      = field_x_;
    auto& field_y      = field_y_;
    auto& v11_hist_data    = v11_hist_data_;
    auto& v11_global_count = v11_global_count_;
    bool& v11_dbg_first    = v11_dbg_first_;
    uint64_t& v11_iter     = v11_iter_;
    constexpr uint64_t V11_HIST_REFRESH_ITERS = 1000000u;

    // ── Replace shm pos reads with input pointer reads. ──
    int32_t nc_actual = std::min(nc_actual_in, NC_max);
    // Skip the host→host copy when the caller wrote directly into our own input
    // buffers (px_in_buf() etc.) — eliminates ~nc*16 bytes of memcpy per iter.
    if (px_in != px.data()) std::memcpy(px.data(), px_in, (size_t)nc_actual * sizeof(float));
    if (py_in != py.data()) std::memcpy(py.data(), py_in, (size_t)nc_actual * sizeof(float));
    if (sx_in != sx.data()) std::memcpy(sx.data(), sx_in, (size_t)nc_actual * sizeof(float));
    if (sy_in != sy.data()) std::memcpy(sy.data(), sy_in, (size_t)nc_actual * sizeof(float));
    // Tail beyond nc_actual stays 0 (vectors zero-init'd in ctor).
    {

            // ── Measure the untimed per-iter prologue (px/py/sx/sy host memcpy
            // at lines above + any setup) to attribute the "fold+oh" residual. ──
            {
                static const bool _prol_time =
                    getenv("FOLD_TIME") && std::string(getenv("FOLD_TIME")) == "1";
                if (_prol_time && (v11_iter_ % 50u) == 0u) {
                    double _pm = ms_since(t_total_start);
                    printf("[prologue] iter=%llu t_total_start->h2d = %.3f ms "
                           "(incl px/py/sx/sy host memcpy, nc=%d)\n",
                           (unsigned long long)v11_iter_, _pm, nc_actual);
                    fflush(stdout);
                }
            }

            // ── H2D: upload cell positions ────────────────────────────────────
            auto ts = hrclock::now();
            EnqueueWriteMeshBuffer(cq, px_buf, px, false);
            EnqueueWriteMeshBuffer(cq, py_buf, py, false);
            EnqueueWriteMeshBuffer(cq, sx_buf, sx, false);
            EnqueueWriteMeshBuffer(cq, sy_buf, sy, false);
            // V31_EF_GEOM: upload per-cell ratio ONCE (constant; density scatter ·ratio).
            if (v31_ef_geom_ && v31_ratio_buf_ && !v31_ratio_uploaded_ && !v31_ratio_data_.empty()) {
                std::vector<float> rr(soa_padded, 1.0f);   // pad cells → ratio 1 (kc=hc=0, no scatter)
                size_t ncp = std::min((size_t)soa_padded, v31_ratio_data_.size());
                std::memcpy(rr.data(), v31_ratio_data_.data(), ncp * sizeof(float));
                EnqueueWriteMeshBuffer(cq, v31_ratio_buf_, rr, false);
                v31_ratio_uploaded_ = true;
            }
            Finish(cq);
            double h2d_ms = ms_since(ts);

            // ── Scatter ───────────────────────────────────────────────────────
            // V11: cell-centric tile-routed; replaces V6 sparse + V10 gather.
            // Other modes: V6 scatter populates contrib_buf for downstream gather.
            double scatter_ms = 0.0;
            double gather_ms = 0.0;
            if (use_v11) {
                // Phase 3 hist + shard_table refresh: triggered on iter 0 and
                // every V11_HIST_REFRESH_ITERS iters thereafter so the
                // shard_table tracks the *current* hot tiles as cells migrate.
                double hist_ms = 0.0;
                bool v11_should_refresh =
                    v11_dbg_first || ((v11_iter % V11_HIST_REFRESH_ITERS) == 0u);
                // Rung 3: the refresh block is disabled at runtime
                // (V11_HIST_REFRESH_ITERS = 1,000,000) and references many
                // setup-only locals (PerCoreShardInfo, SHARD_BYTES,
                // tile_to_core_v11, core_to_tiles_v11, …) that don't survive
                // into scatter(). Compile it out for now; if we ever re-enable
                // the periodic refresh we'll need to promote those locals
                // into Impl members and remove the `#if 0` guard.
#if 0
                if (v11_should_refresh) {
                    auto ts_h = hrclock::now();
                    EnqueueMeshWorkload(cq, wl_v11_hist, false); Finish(cq);
                    hist_ms = ms_since(ts_h);
                    EnqueueReadMeshBuffer(cq, v11_hist_data, hist_buf, true);
                    uint32_t per_core_words = hist_pgsz_v11 / 4u;
                    uint32_t total_tiles_v11x = M_tiles * N_tiles_v11;
                    std::fill(v11_global_count.begin(), v11_global_count.end(), 0u);
                    for (uint32_t c = 0; c < (uint32_t)nc_all; ++c) {
                        const uint32_t* page = v11_hist_data.data() + (size_t)c * per_core_words;
                        for (uint32_t t = 0; t < total_tiles_v11x; ++t) {
                            v11_global_count[t] += page[t];
                        }
                    }
                    // Stats: per-core load (sum of owned tile counts) — this is
                    // what really matters for receiver-side load balance.
                    uint64_t total = 0;
                    uint32_t max_t = 0;
                    for (uint32_t t = 0; t < total_tiles_v11x; ++t) {
                        total += v11_global_count[t];
                        if (v11_global_count[t] > max_t) max_t = v11_global_count[t];
                    }
                    std::vector<uint64_t> per_core_load((size_t)nc_all, 0);
                    for (uint32_t c = 0; c < (uint32_t)nc_all; ++c) {
                        for (uint32_t tile_idx : core_to_tiles_v11[c]) {
                            per_core_load[c] += v11_global_count[tile_idx];
                        }
                    }
                    uint64_t min_load = UINT64_MAX, max_load = 0;
                    uint64_t sum_load = 0;
                    for (uint32_t c = 0; c < (uint32_t)nc_all; ++c) {
                        if (per_core_load[c] < min_load) min_load = per_core_load[c];
                        if (per_core_load[c] > max_load) max_load = per_core_load[c];
                        sum_load += per_core_load[c];
                    }
                    double avg_load = (double)sum_load / nc_all;
                    printf("[server] V11 hist: total=%llu, max_per_tile=%u, "
                           "per-core load min=%llu max=%llu avg=%.0f imbalance=%.2fx, "
                           "hist_ms=%.2f\n",
                           (unsigned long long)total, max_t,
                           (unsigned long long)min_load,
                           (unsigned long long)max_load, avg_load,
                           (double)max_load / avg_load, hist_ms);

                    // Histogram distribution dump: top-20 tile counts + decile cuts.
                    if (v11_dbg_first) {
                        std::vector<uint32_t> sorted_counts(v11_global_count);
                        std::sort(sorted_counts.begin(), sorted_counts.end(), std::greater<uint32_t>());
                        printf("[server] V11 top-20 per-tile counts: ");
                        for (uint32_t i = 0; i < 20 && i < sorted_counts.size(); ++i) {
                            printf("%u ", sorted_counts[i]);
                        }
                        printf("\n");
                        size_t n = sorted_counts.size();
                        printf("[server] V11 tile-count percentiles: p99=%u p95=%u p90=%u p75=%u p50=%u p25=%u\n",
                               sorted_counts[n/100], sorted_counts[n/20], sorted_counts[n/10],
                               sorted_counts[n/4], sorted_counts[n/2], sorted_counts[(3*n)/4]);
                        uint32_t over_thresholds[] = {10000, 5000, 2000, 1000, 500, 200, 100};
                        printf("[server] V11 tiles-above-threshold: ");
                        for (uint32_t T : over_thresholds) {
                            uint32_t count_over = 0;
                            for (uint32_t c : v11_global_count) if (c > T) count_over++;
                            printf(">%u=%u  ", T, count_over);
                        }
                        printf("\n");
                    }

                    // Compute shard table from this histogram
                    std::vector<uint8_t> shard_table_v11;
                    std::vector<uint32_t> per_core_shard_count_v11;
                    v11::build_shard_table(v11_global_count, tile_to_core_v11,
                                           (uint32_t)nc_all, HOT_THRESHOLD, MAX_K, SHARD_BYTES,
                                           shard_table_v11, per_core_shard_count_v11);
                    // Stats
                    uint32_t hot_count = 0, max_K_picked = 0;
                    for (uint32_t t = 0; t < total_tiles_v11x; ++t) {
                        uint8_t K = shard_table_v11[(size_t)t * SHARD_BYTES];
                        if (K > 1) hot_count++;
                        if (K > max_K_picked) max_K_picked = K;
                    }
                    uint32_t max_shards_per_core = 0, total_shards = 0;
                    for (uint32_t c = 0; c < (uint32_t)nc_all; ++c) {
                        if (per_core_shard_count_v11[c] > max_shards_per_core)
                            max_shards_per_core = per_core_shard_count_v11[c];
                        total_shards += per_core_shard_count_v11[c];
                    }
                    printf("[server] V11 shard_table: %u hot tiles, max_K=%u, "
                           "total shard slots=%u, max_per_core=%u\n",
                           hot_count, max_K_picked, total_shards, max_shards_per_core);

                    // Pad shard_table to upload size and upload.
                    std::vector<uint8_t> upload_buf(shard_table_pgsz_v11, 0);
                    std::memcpy(upload_buf.data(), shard_table_v11.data(),
                                std::min((size_t)shard_table_pgsz_v11, shard_table_v11.size()));
                    EnqueueWriteMeshBuffer(cq, shard_table_buf, upload_buf, false);
                    Finish(cq);

                    // Buffer addresses (recomputed here since they're out of the use_v11 scope)
                    const uint32_t ol_a   = (uint32_t)owned_lookup_buf->address();
                    const uint32_t rt_a   = (uint32_t)route_buf->address();
                    const uint32_t da_v11 = (uint32_t)density_buf->address();

                    // ── Step 2: build per_core_v11 with real shard info ──────────
                    per_core_v11.assign(nc_all, PerCoreShardInfo{});
                    for (uint32_t c = 0; c < (uint32_t)nc_all; ++c) {
                        for (uint32_t tile_idx : core_to_tiles_v11[c]) {
                            per_core_v11[c].primary_tile_ids.push_back(tile_idx);
                            const uint8_t* entry = shard_table_v11.data() + (size_t)tile_idx * SHARD_BYTES;
                            per_core_v11[c].primary_K.push_back(entry[0]);
                            per_core_v11[c].primary_hot_seq.push_back(
                                *reinterpret_cast<const uint32_t*>(entry + 8));
                        }
                    }
                    for (uint32_t t = 0; t < total_tiles_v11x; ++t) {
                        const uint8_t* entry = shard_table_v11.data() + (size_t)t * SHARD_BYTES;
                        uint8_t K = entry[0];
                        if (K < 2) continue;
                        uint32_t hot_seq = *reinterpret_cast<const uint32_t*>(entry + 8);
                        for (uint8_t shard_idx = 1; shard_idx < K; ++shard_idx) {
                            uint8_t alt = entry[shard_idx];
                            per_core_v11[alt].shard_entries.push_back({t, hot_seq, shard_idx});
                        }
                    }
                    uint32_t n_total_max_v11 = 0;
                    for (auto& info : per_core_v11) {
                        uint32_t n = (uint32_t)(info.primary_tile_ids.size() + info.shard_entries.size());
                        if (n > n_total_max_v11) n_total_max_v11 = n;
                    }
                    printf("[server] V11 per_core_v11 built: n_total_max=%u\n", n_total_max_v11);

                    // ── Step 3: update owned_lookup_buf with shard slots ─────────
                    {
                        uint32_t owned_lookup_total_v11 = (uint32_t)nc_all * owned_lookup_pgsz_v11;
                        std::vector<uint8_t> owned_upload_v2(owned_lookup_total_v11, 0xFF);
                        for (uint32_t c = 0; c < (uint32_t)nc_all; ++c) {
                            uint16_t* page = reinterpret_cast<uint16_t*>(
                                owned_upload_v2.data() + (size_t)c * owned_lookup_pgsz_v11);
                            auto& info = per_core_v11[c];
                            uint16_t local = 0;
                            for (uint32_t tid : info.primary_tile_ids) page[tid] = local++;
                            for (auto& sh : info.shard_entries)        page[sh.tile_id] = local++;
                        }
                        EnqueueWriteMeshBuffer(cq, owned_lookup_buf, owned_upload_v2, false);
                        Finish(cq);
                    }

                    // ── Step 5: rebuild wl_v11_accum with merged accum+reduce
                    {
                        // Use the SAME CB size every refresh (matches initial cold
                        // build) so the kernel JIT binary is identical and cache-hits.
                        if (n_total_max_v11 > n_owned_max + V11_CB_SLOT_HEADROOM) {
                            printf("[server] FATAL: n_total_max=%u exceeds CB budget "
                                   "(n_owned_max=%u + headroom=%u). Increase V11_CB_SLOT_HEADROOM.\n",
                                   n_total_max_v11, n_owned_max, V11_CB_SLOT_HEADROOM);
                            fflush(stdout); std::exit(1);
                        }
                        uint32_t v11_ac_scratch_real =
                            (v11_dense_offset_bytes
                             + (2u * (n_owned_max + V11_CB_SLOT_HEADROOM) + 1u) * TILE_BYTES_v11
                             + 31u) & ~31u;
                        if (v11_ac_scratch_real >= 1536u * 1024u) {
                            printf("[server] FATAL: V11 sharded accum scratch %u KB exceeds L1\n",
                                   v11_ac_scratch_real / 1024u);
                            fflush(stdout); std::exit(1);
                        }
                        MeshWorkload new_wl_accum;
                        Program new_prog_ac = CreateProgram();
                        CreateCircularBuffer(new_prog_ac, all_crs,
                            CircularBufferConfig(v11_ac_scratch_real, {{24u, tt::DataFormat::Float32}})
                                .set_page_size(24u, v11_ac_scratch_real));
                        auto ak_real = CreateKernel(new_prog_ac, KDIR + "v11_accum_dm.cpp", all_crs,
                            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                                               .noc = NOC::RISCV_0_default});
                        // Twin NCRISC kernel: accumulates writers [nc_split..nc_all)
                        // into dense_n, signals BRISC via merge_sem.
                        auto ak_real_n = CreateKernel(new_prog_ac, KDIR + "v11_accum_n_dm.cpp", all_crs,
                            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                                               .noc = NOC::RISCV_1_default});

                        // BRISC↔NCRISC merge semaphore (one per core).
                        uint32_t merge_sem_id = CreateSemaphore(new_prog_ac, all_crs, 0u);
                        // G-PMERGE: BRISC and NCRISC each do half the merge in parallel.
                        uint32_t brisc_acc_done_sem = CreateSemaphore(new_prog_ac, all_crs, 0u);
                        uint32_t ncrisc_half_merge_done_sem = CreateSemaphore(new_prog_ac, all_crs, 0u);
                        // Step 5b: route_buf has nc_all writers. Accum reads
                        // BRISC [0..nc_all/2), NCRISC [nc_all/2..nc_all).
                        uint32_t nc_split = (uint32_t)nc_all / 2u;

                        // Allocate H semaphore slots, where H = max hot primaries
                        // per core. Each hot primary on a core uses a different
                        // slot so concurrent shard signals don't interfere.
                        uint32_t H_max_hot = 0u;
                        for (auto& info : per_core_v11) {
                            uint32_t cnt = 0;
                            for (uint8_t K : info.primary_K) if (K >= 2u) cnt++;
                            if (cnt > H_max_hot) H_max_hot = cnt;
                        }
                        std::vector<uint32_t> sem_ids;
                        sem_ids.reserve(H_max_hot);
                        for (uint32_t s = 0; s < H_max_hot; ++s) {
                            sem_ids.push_back(CreateSemaphore(new_prog_ac, all_crs, 0u));
                        }
                        printf("[server] V11 allocated %u shard sem slot(s) + 1 merge_sem; nc_split=%u\n",
                               H_max_hot, nc_split);

                        // Build map: hot tile_id → sem_id used by its primary.
                        // The primary's i-th hot tile (in primary_tile_ids order)
                        // gets sem_ids[i]. All cores agree on this assignment
                        // because shard owners look up the same map entry.
                        std::vector<uint32_t> tile_to_sem(M_tiles * N_tiles_v11, 0u);
                        for (auto& info : per_core_v11) {
                            uint32_t slot = 0;
                            for (uint32_t i = 0; i < info.primary_tile_ids.size(); ++i) {
                                if (info.primary_K[i] >= 2u) {
                                    tile_to_sem[info.primary_tile_ids[i]] = sem_ids[slot];
                                    slot++;
                                }
                            }
                        }

                        uint32_t srb_a = (uint32_t)shard_reduce_buf->address();

                        for (int c = 0; c < nc_all; ++c) {
                            auto& info = per_core_v11[c];
                            uint32_t n_primary     = (uint32_t)info.primary_tile_ids.size();
                            uint32_t n_shard       = (uint32_t)info.shard_entries.size();
                            uint32_t n_primary_hot = 0;
                            for (uint8_t K : info.primary_K) if (K >= 2u) n_primary_hot++;
                            // This core's own NOC XY — needed by NCRISC to do the
                            // local-loopback semaphore inc that signals BRISC.
                            CoreCoord my_noc =
                                mesh_device->worker_core_from_logical_core(all_ccs[c]);
                            std::vector<uint32_t> args = {
                                ol_a, owned_lookup_pgsz_v11,
                                (uint32_t)c, (uint32_t)nc_all,
                                M_tiles, N_tiles_v11,
                                (uint32_t)M, (uint32_t)N,
                                rt_a, route_pgsz_v11, v11_max_per_page_tuples,
                                da_v11, density_pgsz, inv_ba_u32,
                                n_primary,
                                n_shard,
                                srb_a, shard_reduce_pgsz_v11,
                                MAX_K,
                                n_primary_hot,
                                nc_split,
                                merge_sem_id,
                                (uint32_t)my_noc.x,
                                (uint32_t)my_noc.y,
                                brisc_acc_done_sem,            // arg 24 (G-PMERGE)
                                ncrisc_half_merge_done_sem,    // arg 25 (G-PMERGE)
                            };
                            // primary tile IDs
                            for (auto tid : info.primary_tile_ids) args.push_back(tid);
                            // hot quads (local_idx, hot_seq, K, sem_id) — slot
                            // assignment matches tile_to_sem map.
                            for (uint32_t i = 0; i < n_primary; ++i) {
                                if (info.primary_K[i] >= 2u) {
                                    args.push_back(i);
                                    args.push_back(info.primary_hot_seq[i]);
                                    args.push_back((uint32_t)info.primary_K[i]);
                                    args.push_back(tile_to_sem[info.primary_tile_ids[i]]);
                                }
                            }
                            // shard quints (hot_seq, shard_idx, prim_x, prim_y, sem_id).
                            for (auto& sh : info.shard_entries) {
                                uint32_t prim_core = (uint32_t)tile_to_core_v11[sh.tile_id];
                                CoreCoord prim_noc =
                                    mesh_device->worker_core_from_logical_core(all_ccs[prim_core]);
                                args.push_back(sh.hot_tile_seq);
                                args.push_back((uint32_t)sh.shard_idx_in_K);
                                args.push_back((uint32_t)prim_noc.x);
                                args.push_back((uint32_t)prim_noc.y);
                                args.push_back(tile_to_sem[sh.tile_id]);
                            }
                            SetRuntimeArgs(new_prog_ac, ak_real,   all_ccs[c], args);
                            SetRuntimeArgs(new_prog_ac, ak_real_n, all_ccs[c], args);
                        }
                        new_wl_accum.add_program(device_range, std::move(new_prog_ac));
                        EnqueueMeshWorkload(cq, new_wl_accum, false); Finish(cq);  // JIT
                        wl_v11_accum = std::move(new_wl_accum);
                        printf("[server] V11 merged accum+reduce JIT done\n");
                    }

                    // (reduce_a / reduce_bc no longer built — merged into accum.)

                    fflush(stdout);
                    v11_dbg_first = false;
                }
#endif
                (void)v11_should_refresh;  // mark used (block compiled out)
                (void)hist_ms;             // mark used (block compiled out)

                ts = hrclock::now();
                EnqueueMeshWorkload(cq, wl_v11_scatter, false); Finish(cq);
                scatter_ms = ms_since(ts);

                // Drop-counter: read drop_buf (2*nc_all uint32s, one per
                // writer-RISC) and report max + sum if any drops happened.
                // Cost: ~7 KB DRAM read, <100 µs per iter.
                {
                    std::vector<uint32_t> drop_host(2u * (size_t)nc_all * 8u, 0);
                    EnqueueReadMeshBuffer(cq, drop_host, drop_buf, true);
                    uint64_t drop_sum = 0;
                    uint32_t drop_max = 0;
                    int max_slot = -1;
                    for (uint32_t s = 0; s < 2u * (uint32_t)nc_all; ++s) {
                        uint32_t v = drop_host[s * 8u];  // slot 0 of each 32-byte page
                        drop_sum += v;
                        if (v > drop_max) { drop_max = v; max_slot = (int)s; }
                    }
                    static uint64_t last_drop_print_iter = 0;
                    if (drop_sum > 0u && v11_iter - last_drop_print_iter >= 50u) {
                        printf("[server] DROP iter=%llu sum=%llu max=%u "
                               "(slot=%d %s c=%d)\n",
                               (unsigned long long)v11_iter,
                               (unsigned long long)drop_sum,
                               drop_max, max_slot,
                               (max_slot >= nc_all) ? "BRISC" : "NCRISC",
                               (max_slot >= nc_all) ? max_slot - nc_all : max_slot);
                        fflush(stdout);
                        last_drop_print_iter = v11_iter;
                    }
                }

                ts = hrclock::now();
                if (use_v19) {
                    // Writeout kernel converts uint32 → fp32 in L1 and writes
                    // directly to density_buf in DRAM (the TTNN DCT input).
                    // No host PCIe round-trip, no host-side conversion loop.
                    EnqueueMeshWorkload(cq, wl_v19_writeout, false);
                    Finish(cq);
                } else {
                    // Merged kernel does accum + reduce_a + reduce_bc in one program;
                    // shard sync is via NOC semaphores (1 host Finish() vs the old 3).
                    EnqueueMeshWorkload(cq, wl_v11_accum, false); Finish(cq);
                }
                gather_ms = ms_since(ts);
                v11_iter++;
            } else if (use_v14) {
                // V14: zero partial_buf before scatter (inactive tiles must be zero
                // for reduce correctness). Then scatter writes only active tiles.
                {
                    static std::vector<uint8_t> v14_zeros;
                    uint64_t psize = (uint64_t)nc_all * (uint64_t)total_tiles_v14 * partial_pgsz_v14;
                    if (v14_zeros.size() != (size_t)psize) v14_zeros.assign((size_t)psize, 0);
                    EnqueueWriteMeshBuffer(cq, partial_buf_v14, v14_zeros, false);
                    Finish(cq);
                }
                // DROP-INSTRUMENTATION: zero drop buf before scatter.
                {
                    std::vector<uint8_t> zd(v14_drop_buf_pgsz_outer, 0);
                    EnqueueWriteMeshBuffer(cq, v14_drop_buf, zd, false);
                    Finish(cq);
                }
                ts = hrclock::now();
                EnqueueMeshWorkload(cq, wl_v14_scatter, false); Finish(cq);
                scatter_ms = ms_since(ts);
                // DROP-INSTRUMENTATION: read drop counts every N iters, print stats.
                {
                    static uint32_t v14_drop_iter = 0;
                    v14_drop_iter++;
                    if (v14_drop_iter == 1u || v14_drop_iter == 50u ||
                        v14_drop_iter == 100u || v14_drop_iter == 200u ||
                        v14_drop_iter == 300u || v14_drop_iter == 400u ||
                        v14_drop_iter == 500u || v14_drop_iter == 600u) {
                        std::vector<uint32_t> drops((size_t)nc_all, 0);
                        EnqueueReadMeshBuffer(cq, drops, v14_drop_buf, true);
                        uint64_t total = 0; uint32_t mx = 0; int hot = 0;
                        for (int c = 0; c < nc_all; ++c) {
                            total += drops[c];
                            if (drops[c] > mx) { mx = drops[c]; hot = c; }
                        }
                        printf("[V14-DROPS iter=%u] total=%llu  max/core=%u (core %d)  avg/core=%llu\n",
                               v14_drop_iter, (unsigned long long)total, mx, hot,
                               (unsigned long long)(total / (uint64_t)nc_all));
                        fflush(stdout);
                    }
                }
                ts = hrclock::now();
                EnqueueMeshWorkload(cq, wl_v14_reduce, false); Finish(cq);
                gather_ms = ms_since(ts);
            } else if (use_v13) {
                ts = hrclock::now();
                EnqueueMeshWorkload(cq, wl_v13_scatter, false); Finish(cq);
                scatter_ms = ms_since(ts);
                ts = hrclock::now();
                EnqueueMeshWorkload(cq, wl_v13_accum, false); Finish(cq);
                gather_ms = ms_since(ts);
                // V15 stats dump at iter 1 — verifies per-tile bucket counts.
                if (use_v15) {
                    static int v15_stats_dumped = 0;
                    if (!v15_stats_dumped) {
                        std::vector<uint32_t> stats_host((size_t)nc_all * 64u, 0);
                        EnqueueReadMeshBuffer(cq, stats_host, v15_stats_buf, true);
                        printf("[server] V15-STATS (iter 1):\n");
                        for (int c = 0; c < nc_all; ++c) {
                            uint32_t base = (uint32_t)c * 64u;
                            uint32_t n_owned = stats_host[base + 0];
                            uint32_t cap = stats_host[base + 1];
                            uint32_t total = stats_host[base + 2];
                            if (n_owned == 0u || n_owned > 64u) continue;
                            printf("  core %3d: n_owned=%u cap=%u total=%u  bcounts:",
                                   c, n_owned, cap, total);
                            for (uint32_t t = 0; t < n_owned && t < 8u; ++t) {
                                printf(" t%u=%u", t, stats_host[base + 3 + t]);
                            }
                            printf("  tile_pushes:");
                            for (uint32_t t = 0; t < n_owned && t < 8u; ++t) {
                                printf(" t%u=%u", t, stats_host[base + 16 + t]);
                            }
                            printf("\n");
                            // First record of each bucket (slot 24+t*4 in stats_host u32 array,
                            // which corresponds to byte offset 96 + t*16 in DRAM).
                            for (uint32_t t = 0; t < n_owned && t < 4u; ++t) {
                                uint32_t base24 = base + 24u + t*4u;
                                uint32_t w0 = stats_host[base24 + 0];
                                uint32_t w1 = stats_host[base24 + 1];
                                uint32_t w2 = stats_host[base24 + 2];
                                uint32_t w3 = stats_host[base24 + 3];
                                int16_t bxl = (int16_t)(w0 & 0xFFFF);
                                int16_t byl = (int16_t)((w0 >> 16) & 0xFFFF);
                                uint16_t tid = (uint16_t)(w1 & 0xFFFF);
                                printf("    t%u rec0: bxl=%d byl=%d tile_id=%u ox[0:1]=0x%08x oy[0:1]=0x%08x\n",
                                       t, bxl, byl, tid, w2, w3);
                            }
                        }
                        fflush(stdout);
                        v15_stats_dumped = 1;
                    }
                }
            } else {
                ts = hrclock::now();
                EnqueueMeshWorkload(cq, wl_scatter, false); Finish(cq);
                scatter_ms = ms_since(ts);
                ts = hrclock::now();
                EnqueueMeshWorkload(cq, wl_gather, false); Finish(cq);
                gather_ms = ms_since(ts);
            }

            // ── D2H: density readback ─────────────────────────────────────────
            // CPU_DCT=0 (default): density stays on device — zero-copy wrap
            //   into a ttnn::Tensor, fold initial_density on-device, solve DCT.
            // CPU_DCT=1: density must reach the host for the Python DCT path.
            double density_d2h_ms = 0;
            double upload_ms = 0, compute_ms = 0, download_ms = 0;
            bool field_direct = false;  // true if solve_device read fields straight into field_x_out/field_y_out
            static const bool use_cpu_dct =
                (getenv("CPU_DCT") && std::string(getenv("CPU_DCT")) == "1");

            // initial_density_map: host vector (kept for CPU_DCT=1 fallback)
            // and device tensor (uploaded once, used for on-device ttnn::add).
            static std::vector<float> id_cache;
            static TT id_cache_tt;
            static bool id_cache_tt_ready = false;
            if (id_cache.empty() && initial_density_set_) {
                // Rung 3: source initial_density from the engine member
                // (set by the Python wrapper via set_initial_density()) instead
                // of the original shm slot.
                id_cache.assign(initial_density_.begin(),
                                initial_density_.end());
            }
            if (!use_cpu_dct && !id_cache_tt_ready) {
                // Upload as ROW_MAJOR (matching density_buf layout) so
                // ttnn::add doesn't need a layout conversion.
                TensorLayout id_layout(
                    ttnn::DataType::FLOAT32,
                    PageConfig(ttnn::Layout::ROW_MAJOR),
                    tt::tt_metal::MemoryConfig{});
                auto id_cpu = TT::from_vector<float>(
                    id_cache,
                    ttnn::TensorSpec(ttnn::Shape{(uint32_t)M, (uint32_t)N}, id_layout));
                id_cache_tt = id_cpu.to_device(mesh_device.get());
                id_cache_tt_ready = true;
                printf("[server] id_cache_tt uploaded once (%dx%d, ROW_MAJOR)\n", M, N);
                fflush(stdout);
            }

            if (use_cpu_dct) {
                // CPU DCT path: D2H is required — the Python client runs DCT
                ts = hrclock::now();
                if (use_v14) {
                    // V14 writes bf16 UNNORMALIZED density (same format as V13).
                    static std::vector<uint8_t> v14_bf16_bytes;
                    if (v14_bf16_bytes.size() != (size_t)M * N * 2u) {
                        v14_bf16_bytes.assign((size_t)M * N * 2u, 0);
                    }
                    EnqueueReadMeshBuffer(cq, v14_bf16_bytes, density_buf_v14, true);
                    density_d2h_ms = ms_since(ts);
                    const size_t MN = (size_t)M * N;
                    const uint16_t* src =
                        reinterpret_cast<const uint16_t*>(v14_bf16_bytes.data());
                    float* df = density_flat.data();
                    const float scale = 1.0f / ((float)bsx * (float)bsy);
                    for (size_t i = 0; i < MN; ++i) {
                        uint32_t bits = ((uint32_t)src[i]) << 16;
                        float f;
                        std::memcpy(&f, &bits, sizeof(f));
                        df[i] = f * scale;
                    }
                } else if (use_v13) {
                    // V13 writes bf16 UNNORMALIZED density. Read bf16 bytes,
                    // expand to fp32 in density_flat, then divide by bin_area
                    // (1/(bsx*bsy)) to match V11's normalized contract.
                    static std::vector<uint8_t> v13_bf16_bytes;
                    if (v13_bf16_bytes.size() != (size_t)M * N * 2u) {
                        v13_bf16_bytes.assign((size_t)M * N * 2u, 0);
                    }
                    EnqueueReadMeshBuffer(cq, v13_bf16_bytes, density_buf_v13, true);
                    density_d2h_ms = ms_since(ts);
                    const size_t MN = (size_t)M * N;
                    const uint16_t* src =
                        reinterpret_cast<const uint16_t*>(v13_bf16_bytes.data());
                    float* df = density_flat.data();
                    const float scale = 1.0f / ((float)bsx * (float)bsy);
                    for (size_t i = 0; i < MN; ++i) {
                        uint32_t bits = ((uint32_t)src[i]) << 16;
                        float f;
                        std::memcpy(&f, &bits, sizeof(f));
                        df[i] = f * scale;
                    }
                } else {
                    EnqueueReadMeshBuffer(cq, density_flat, density_buf, true);
                    density_d2h_ms = ms_since(ts);
                }

                // V19 debug: dump density on a chosen iter for V11-vs-V19 diff.
                {
                    static const uint32_t DUMP_ITER = []() -> uint32_t {
                        const char* s = getenv("DENSITY_DUMP_ITER");
                        return s ? (uint32_t)atoi(s) : 0u;
                    }();
                    static const char* DUMP_PATH = getenv("DENSITY_DUMP_PATH");
                    static uint32_t dump_count = 0;
                    if (DUMP_ITER != 0u && DUMP_PATH != nullptr) {
                        dump_count++;
                        if (dump_count == DUMP_ITER) {
                            FILE* fp = fopen(DUMP_PATH, "wb");
                            if (fp) {
                                fwrite(density_flat.data(), sizeof(float),
                                       density_flat.size(), fp);
                                fclose(fp);
                                printf("[server] density dumped iter=%u to %s "
                                       "(%zu floats)\n", DUMP_ITER, DUMP_PATH,
                                       density_flat.size());
                                fflush(stdout);
                            }
                        }
                    }
                }

                // NOTE: CPU_DCT=1 path does NOT fold initial_density_map on
                // the server — the Python client adds (initial_density_map /
                // bin_area) itself in scatter_ttnn_client.py before its CPU
                // DCT. Adding here too would double-count and break
                // convergence (V13 hits HPWL=169M, V11 has small bias).
                (void)id_cache;
            } else {
                if (use_v14) {
                    printf("[server] FATAL: GATHER_MODE=v14 requires CPU_DCT=1 "
                           "(bf16 density not wired into TT-DCT zero-copy path).\n");
                    fflush(stdout); std::exit(1);
                }
                if (use_v13) {
                    // V13's bf16 density_buf isn't compatible with the fp32
                    // zero-copy TTNN DCT path today. Use CPU_DCT=1 with V13.
                    printf("[server] FATAL: GATHER_MODE=v13_fpu requires CPU_DCT=1 "
                           "(bf16 density not wired into TT-DCT zero-copy path).\n");
                    fflush(stdout); std::exit(1);
                }
                // Zero-copy path: wrap density_buf as a TTNN tensor (no D2H),
                // fold initial_density on-device, and solve DCT directly.
                // The wrapper is created once and kept alive across iterations
                // so its destructor doesn't deallocate density_buf's DRAM.
                // The gather kernel overwrites density_buf's contents in-place
                // each iteration — the wrapper just provides TTNN metadata.
                static TT density_buf_tt;
                static bool density_buf_tt_ready = false;
                if (!density_buf_tt_ready) {
                    density_buf_tt = wrap_mesh_buf_as_tensor(density_buf, M, N);
                    density_buf_tt_ready = true;
                }
                auto _tfold = hrclock::now();
                auto rho_folded = ttnn::add(density_buf_tt, id_cache_tt);
                static const bool _fold_time =
                    getenv("FOLD_TIME") && std::string(getenv("FOLD_TIME")) == "1";
                if (_fold_time) {
                    Finish(cq);                       // force the add to complete to time it
                    double _fm = ms_since(_tfold);
                    if ((v11_iter_ % 50u) == 0u) {
                        printf("[fold_time] iter=%llu ttnn::add(fold)+Finish = %.3f ms\n",
                               (unsigned long long)v11_iter_, _fm);
                        fflush(stdout);
                    }
                }

                field_direct = ttnn_solver.solve_device(
                                         std::move(rho_folded), field_x, field_y,
                                         mesh_device.get(),
                                         upload_ms, compute_ms, download_ms,
                                         field_x_out, field_y_out);
            }

            // ── Optional debug dump ──────────────────────────────────────────
            // EXPORT_DENSITY_PATH dumps EVERY iter (overwrite-style; the
            // smoke test expects this single-iter behavior).
            // For CPU_DCT=0 the density_flat is stale — do a one-off D2H
            // only when the env var is actually set (debug/test only).
            // EXPORT_POS_PATH dumps cell positions at selected iters.
            {
                const char* dump_path = getenv("EXPORT_DENSITY_PATH");
                if (dump_path && dump_path[0]) {
                    if (!use_cpu_dct) {
                        EnqueueReadMeshBuffer(cq, density_flat, density_buf, true);
                    }
                    FILE* f = fopen(dump_path, "wb");
                    if (f) {
                        fwrite(density_flat.data(), sizeof(float),
                               (size_t)M * (size_t)N, f);
                        fclose(f);
                    }
                }
                static uint32_t dump_iter_count = 0;
                dump_iter_count++;
                if (dump_iter_count == 1u || dump_iter_count == 50u || dump_iter_count == 100u) {
                    const char* pos_path = getenv("EXPORT_POS_PATH");
                    if (pos_path && pos_path[0]) {
                        char path_with_iter[1024];
                        snprintf(path_with_iter, sizeof(path_with_iter),
                                 "%s.iter%u.bin", pos_path, dump_iter_count);
                        FILE* f = fopen(path_with_iter, "wb");
                        if (f) {
                            int32_t n = nc_actual;
                            int32_t Mi = M, Ni = N;
                            float xlf = xl, ylf = yl, xhf = xh, yhf = yh;
                            fwrite(&Mi, sizeof(int32_t), 1, f);
                            fwrite(&Ni, sizeof(int32_t), 1, f);
                            fwrite(&xlf, sizeof(float), 1, f);
                            fwrite(&ylf, sizeof(float), 1, f);
                            fwrite(&xhf, sizeof(float), 1, f);
                            fwrite(&yhf, sizeof(float), 1, f);
                            fwrite(&n, sizeof(int32_t), 1, f);
                            fwrite(px.data(), sizeof(float), (size_t)n, f);
                            fwrite(py.data(), sizeof(float), (size_t)n, f);
                            fwrite(sx.data(), sizeof(float), (size_t)n, f);
                            fwrite(sy.data(), sizeof(float), (size_t)n, f);
                            fclose(f);
                        }
                    }
                }
                const char* rb_path = getenv("EXPORT_ROUTE_BUF_PATH");
                if (use_v11 && dump_iter_count == 100u && rb_path && rb_path[0]) {
                    uint64_t rb_size = (uint64_t)nc_all * (uint64_t)nc_all * route_pgsz_v11;
                    std::vector<uint8_t> rb_host(rb_size);
                    EnqueueReadMeshBuffer(cq, rb_host, route_buf, true);
                    FILE* f = fopen(rb_path, "wb");
                    if (f) {
                        int32_t nc_int = nc_all;
                        uint32_t pgsz = route_pgsz_v11;
                        uint32_t maxt = v11_max_per_page_tuples;
                        fwrite(&nc_int, sizeof(int32_t), 1, f);
                        fwrite(&pgsz, sizeof(uint32_t), 1, f);
                        fwrite(&maxt, sizeof(uint32_t), 1, f);
                        fwrite(rb_host.data(), 1, rb_size, f);
                        fclose(f);
                        printf("[server] DEBUG: dumped route_buf (%llu MB) to %s\n",
                               (unsigned long long)(rb_size / (1024 * 1024)), rb_path);
                        fflush(stdout);
                    }
                }
            }

            // ── Write fields into caller-provided output buffers. ──
            //   When CPU_DCT=1 (default for Rung 3): density goes into
            //   density_out. If field_x_out aliases density_out (caller
            //   pre-allocated one buffer + aliased), the duplicate memcpy
            //   is skipped. If field_y_out is nullptr (caller has a
            //   pre-zeroed buffer it reuses), the memset is skipped.
            //   When CPU_DCT=0: density_out gets the chip density;
            //   field_x_out/field_y_out get the TTNN-DCT result. Null
            //   pointers skip the corresponding write.
            ts = hrclock::now();
            const size_t mn_bytes = (size_t)M * N * sizeof(float);
            if (density_out) {
                std::memcpy(density_out, density_flat.data(), mn_bytes);
            }
            if (use_cpu_dct) {
                if (field_x_out && field_x_out != density_out) {
                    std::memcpy(field_x_out, density_flat.data(), mn_bytes);
                }
                if (field_y_out) {
                    std::memset(field_y_out, 0, mn_bytes);
                }
            } else if (!field_direct) {
                // Fallback memcpy only when solve_device did NOT already read
                // the fields straight into field_x_out/field_y_out.
                if (field_x_out) std::memcpy(field_x_out, field_x.data(), mn_bytes);
                if (field_y_out) std::memcpy(field_y_out, field_y.data(), mn_bytes);
            }
            double fw_ms = ms_since(ts);

            double total_ms = h2d_ms + scatter_ms + gather_ms + density_d2h_ms
                            + upload_ms + compute_ms + download_ms + fw_ms;

            // ── Fill timing_out (replaces shm header write). ──
            if (timing_out) {
                *timing_out = {};
                timing_out->h2d_ms           = h2d_ms;
                timing_out->scatter_ms       = scatter_ms;
                timing_out->gather_ms        = gather_ms;
                timing_out->d2h_density_ms   = density_d2h_ms;
                timing_out->ttnn_upload_ms   = upload_ms;
                timing_out->ttnn_compute_ms  = compute_ms;
                timing_out->ttnn_download_ms = download_ms;
                timing_out->fw_ms            = fw_ms;
                timing_out->total_server_ms  = ms_since(t_total_start);
                timing_out->gather_mode = use_v13 ? 6u
                                       : use_v11 ? 5u
                                       : use_v10 ? 4u
                                       : use_v9  ? 3u
                                       : (use_v8 ? 2u : (use_v7 ? 1u : 0u));
            }

            // ── Progress log (every 50 iters). ──
            if ((v11_iter_ % 50u) == 0u) {
                printf("[v19_engine] iter=%llu h2d=%.3f sc=%.3f ga=%.3f "
                       "d2h=%.3f dct=%.3f fw=%.3f total=%.3f\n",
                       (unsigned long long)v11_iter_,
                       h2d_ms, scatter_ms, gather_ms, density_d2h_ms,
                       compute_ms, fw_ms, total_ms);
                fflush(stdout);
            }
    }  // closes the brace opened at end of scatter() prologue
}

// ═══════════════════════════════════════════════════════════════════
// V19Engine — public PIMPL wrapper.
// ═══════════════════════════════════════════════════════════════════

V19Engine::V19Engine(int M, int N, int NC_max,
                     float xl, float yl, float xh, float yh,
                     const std::string& gather_mode)
    : impl_(std::make_unique<Impl>(M, N, NC_max, xl, yl, xh, yh, gather_mode)) {}

V19Engine::~V19Engine() = default;

void V19Engine::set_initial_density(const float* id_normalized) {
    impl_->set_initial_density(id_normalized);
}

void V19Engine::scatter(const float* px, const float* py,
                        const float* sx, const float* sy,
                        int32_t nc_actual,
                        float* density_out,
                        float* field_x_out, float* field_y_out,
                        ScatterTiming* timing_out) {
    impl_->scatter(px, py, sx, sy, nc_actual,
                   density_out, field_x_out, field_y_out, timing_out);
}

int V19Engine::M() const noexcept      { return impl_->M_;      }
int V19Engine::N() const noexcept      { return impl_->N_;      }
int V19Engine::NC_max() const noexcept { return impl_->NC_max_; }
const std::string& V19Engine::gather_mode_str() const noexcept {
    return impl_->gather_mode_;
}
void* V19Engine::mesh_device_ptr() const noexcept {
    return static_cast<void*>(impl_->mesh_device_.get());
}

uint32_t V19Engine::latest_field_x_addr() const noexcept {
    if (!impl_->ttnn_solver_ready_ || !impl_->ttnn_solver_.latest_field_valid) return 0u;
    return (uint32_t)impl_->ttnn_solver_.latest_field_x_mb->address();
}
uint32_t V19Engine::latest_field_y_addr() const noexcept {
    if (!impl_->ttnn_solver_ready_ || !impl_->ttnn_solver_.latest_field_valid) return 0u;
    return (uint32_t)impl_->ttnn_solver_.latest_field_y_mb->address();
}
void V19Engine::set_skip_field_d2h(bool skip) noexcept {
    impl_->ttnn_solver_.skip_field_d2h = skip;
}
float*   V19Engine::px_in_buf() noexcept { return impl_->px_.data(); }
float*   V19Engine::py_in_buf() noexcept { return impl_->py_.data(); }
float*   V19Engine::sx_in_buf() noexcept { return impl_->sx_.data(); }
float*   V19Engine::sy_in_buf() noexcept { return impl_->sy_.data(); }
uint32_t V19Engine::soa_padded() const noexcept { return impl_->soa_padded_; }
uint32_t V19Engine::v31_route_addr()  const noexcept { return impl_->v31_route_buf_  ? (uint32_t)impl_->v31_route_buf_->address()  : 0u; }
uint32_t V19Engine::v31_rcount_addr() const noexcept { return impl_->v31_rcount_buf_ ? (uint32_t)impl_->v31_rcount_buf_->address() : 0u; }
uint32_t V19Engine::v31_geom_addr()   const noexcept { return impl_->v31_geom_buf_   ? (uint32_t)impl_->v31_geom_buf_->address()   : 0u; }
uint32_t V19Engine::v31_geom_pg()     const noexcept { return impl_->v31_geom_pg_; }
void V19Engine::set_ef_ratio(const float* r, int n) {   // V31_EF_GEOM: per-cell ratio (scatter order)
    impl_->v31_ratio_data_.assign(r, r + n);
    impl_->v31_ratio_uploaded_ = false;
}
std::vector<int32_t> V19Engine::read_geom() {   // DEBUG: read the whole V31_GEOM buffer (int32 words)
    auto& I = *impl_;
    if (!I.v31_geom_buf_ || I.v31_geom_pg_ == 0u) return {};
    std::vector<int32_t> v((size_t)I.v31_geom_pg_ / 4u, 0);
    EnqueueReadMeshBuffer(*I.cq_, v, I.v31_geom_buf_, true);
    return v;
}
uint32_t V19Engine::v31_route_pg()    const noexcept { return impl_->v31_route_pg_;  }
uint32_t V19Engine::v31_max_tpc()     const noexcept { return impl_->v31_max_tpc_;   }
uint32_t V19Engine::v31_nc_all()      const noexcept { return impl_->v31_nc_all_;    }
const uint32_t* V19Engine::v31_first_tile() const noexcept { return impl_->v31_first_tile_.empty() ? nullptr : impl_->v31_first_tile_.data(); }

void V19Engine::compute_electric_force_v31(
        const int32_t* sel, const float* ratio, int na, int nn,
        const float* fx, const float* fy, float* grad, double* tm) {
    auto& I = *impl_;
    const int M = I.M_, N = I.N_, nc = I.nc_all_;
    const size_t NB = (size_t)M * N;
    auto clk = []{ return std::chrono::high_resolution_clock::now(); };
    auto t0 = clk();

    // ── field: upload FLOAT directly (per-x-column page bx = column bx, N vals).
    //    Backward multiplies area·field in float → no scale, no int overflow. ──
    if (I.v31_fxf_.size() != NB) { I.v31_fxf_.resize(NB); I.v31_fyf_.resize(NB); }
    std::memcpy(I.v31_fxf_.data(), fx, NB * sizeof(float));
    std::memcpy(I.v31_fyf_.data(), fy, NB * sizeof(float));

    const uint32_t chunk = 2048u;
    const uint32_t col_pg = (uint32_t)N * 4u;
    auto grid = I.mesh_device_->compute_with_storage_grid_size();
    auto all_ccs = get_cores(grid, nc);
    CoreRangeSet all_crs = cores_to_crs(all_ccs);

    if (!I.v31_bw_built_) {
        auto make_buf = [&](uint64_t sz, uint32_t pg) {
            DeviceLocalBufferConfig c{.page_size=pg,.buffer_type=BufferType::DRAM};
            ReplicatedBufferConfig r{.size=sz};
            return MeshBuffer::create(r, c, I.mesh_device_.get());
        };
        I.v31_field_pg_ = col_pg;
        I.v31_slab_cells_ = I.v31_max_tpc_ * 1024u;
        I.v31_grad_pg_ = I.v31_slab_cells_ * 8u;
        I.v31_nbx_cap_ = std::min<uint32_t>((uint32_t)M / (uint32_t)nc + 48u, 100000u / (uint32_t)N);
        if (I.v31_nbx_cap_ < 1u) I.v31_nbx_cap_ = 1u;
        I.v31_fx_buf_   = make_buf((uint64_t)M * col_pg, col_pg);
        I.v31_fy_buf_   = make_buf((uint64_t)M * col_pg, col_pg);
        I.v31_grad_buf_ = make_buf((uint64_t)nc * I.v31_grad_pg_, I.v31_grad_pg_);
        I.v31_gflat_.assign((size_t)nc * I.v31_slab_cells_ * 2u, 0u);
        I.v31_rcount_host_.assign((size_t)2u * nc * 16u, 0u);  // 2nc pages (NCRISC + BRISC halves), 64B each
        using tt::CBIndex;
        Program p = CreateProgram();
        auto mkc = [&](uint32_t idx, uint32_t bytes){ CircularBufferConfig c(bytes,{{(CBIndex)idx,tt::DataFormat::Float32}}); c.set_page_size((CBIndex)idx,bytes); CreateCircularBuffer(p,all_crs,c); };
        mkc(24u, I.v31_nbx_cap_ * col_pg);                   // fx band
        mkc(25u, I.v31_nbx_cap_ * col_pg);                   // fy band
        mkc(26u, chunk*16u + 64u);                           // route chunk
        mkc(27u, I.v31_slab_cells_ * 8u + 64u);              // grad slab
        I.v31_bw_k_ = CreateKernel(p, I.kdir_ + "v31_backward.cpp", all_crs,
            DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
        const uint32_t ra = I.v31_route_buf_->address(), fxa = I.v31_fx_buf_->address(),
                       fya = I.v31_fy_buf_->address(), ga = I.v31_grad_buf_->address();
        for (int c = 0; c < nc; ++c)
            SetRuntimeArgs(p, I.v31_bw_k_, all_ccs[c],
                {(uint32_t)c, ra, I.v31_route_pg_, 0u, I.v31_slab_cells_, I.v31_nbx_cap_,
                 (uint32_t)M, (uint32_t)N, fxa, fya, col_pg, ga, I.v31_grad_pg_,
                 I.v31_first_tile_[c]*1024u, chunk, 0u, (uint32_t)nc});
        I.v31_bw_dr_ = MeshCoordinateRange{I.mesh_device_->shape()};
        I.v31_bw_wl_.add_program(I.v31_bw_dr_, std::move(p));
        I.v31_bw_built_ = true;
    }

    // h2d field
    auto th = clk();
    EnqueueWriteMeshBuffer(*I.cq_, I.v31_fx_buf_, I.v31_fxf_, false);
    EnqueueWriteMeshBuffer(*I.cq_, I.v31_fy_buf_, I.v31_fyf_, false);
    Finish(*I.cq_);
    if (tm) tm[0] = std::chrono::duration<double,std::milli>(clk()-th).count();

    // read per-core rcount + refresh args (rcount changes per iter as cells move)
    EnqueueReadMeshBuffer(*I.cq_, I.v31_rcount_host_, I.v31_rcount_buf_, true);
    Program& p = I.v31_bw_wl_.get_programs().at(I.v31_bw_dr_);
    const uint32_t ra = I.v31_route_buf_->address(), fxa = I.v31_fx_buf_->address(),
                   fya = I.v31_fy_buf_->address(), ga = I.v31_grad_buf_->address();
    for (int c = 0; c < nc; ++c) {
        uint32_t rcnt0 = I.v31_rcount_host_[(size_t)c * 16u];              // NCRISC half
        uint32_t rcnt1 = I.v31_rcount_host_[(size_t)(c + nc) * 16u];       // BRISC half
        SetRuntimeArgs(p, I.v31_bw_k_, all_ccs[c],
            {(uint32_t)c, ra, I.v31_route_pg_, rcnt0, I.v31_slab_cells_, I.v31_nbx_cap_,
             (uint32_t)M, (uint32_t)N, fxa, fya, col_pg, ga, I.v31_grad_pg_,
             I.v31_first_tile_[c]*1024u, chunk, rcnt1, (uint32_t)nc});
    }

    // run backward
    auto tg = clk();
    EnqueueMeshWorkload(*I.cq_, I.v31_bw_wl_, false);
    Finish(*I.cq_);
    if (tm) tm[1] = std::chrono::duration<double,std::milli>(clk()-tg).count();

    // d2h grad slab + map → grad_full[sel[i]] (raw +force; ratio + descale here)
    auto td = clk();
    EnqueueReadMeshBuffer(*I.cq_, I.v31_gflat_, I.v31_grad_buf_, true);
    // The forward's v4_compute pre-scales ox/oy by sqrt(inv_bin_area), so the
    // stashed area = px·py·inv_bin_area. The backward force needs px·py, so undo
    // the prescale by multiplying by bin_area (= 1/inv_ba_).
    const float binA = (I.inv_ba_ > 0.0f) ? (1.0f / I.inv_ba_) : 1.0f;
    const uint32_t* ft = I.v31_first_tile_.data();
    for (int c = 0; c < nc; ++c) {
        uint32_t cell_lo = ft[c] * 1024u;
        uint32_t cell_hi = (c + 1 < nc) ? ft[c+1] * 1024u : (uint32_t)na;
        if (cell_hi > (uint32_t)na) cell_hi = (uint32_t)na;
        const size_t pbase = (size_t)c * I.v31_slab_cells_ * 2u;
        const float* gsf = reinterpret_cast<const float*>(I.v31_gflat_.data());
        for (uint32_t i = cell_lo; i < cell_hi; ++i) {
            uint32_t loc = i - cell_lo;
            int g = sel[i];
            // NOTE: do NOT multiply by ratio here. The TT forward uses ORIG cell
            // sizes, so stashed overlap(orig) ≈ ratio·overlap(clamped) — ratio is
            // already baked in. CPU force = ratio·Σovlp(clamped)·f ≈ Σovlp(orig)·f.
            float rr = binA;                     // grad = bin_area · Σ (overlap_orig·inv_ba)·field
            (void)g; (void)ratio;
            grad[g]      = gsf[pbase + loc*2u]      * rr;
            grad[nn + g] = gsf[pbase + loc*2u + 1u] * rr;
        }
    }
    if (tm) { tm[2] = std::chrono::duration<double,std::milli>(clk()-td).count();
              tm[3] = std::chrono::duration<double,std::milli>(clk()-t0).count(); }
}

}  // namespace v19
