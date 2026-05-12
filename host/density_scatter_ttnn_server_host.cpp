// SPDX-License-Identifier: Apache-2.0
//
// Path 1: V4 Scatter + Gather-Density-Only + TTNN C++ DCT Field Solver
//
// Single process (single device) replaces the full V4-V9b pipeline:
//   1. V4 scatter  (110 Tensix cores, same as full pipeline)
//   2. gather-density-only  (Mt cores, accumulate + normalize → row-major density DRAM)
//   3. D2H density readback (~1 MB)
//   4. TTNN C++ matmul-based DCT solve (6 matmuls, same math as Python TTNNFieldSolver)
//   5. D2H field readback + write files
//
// IPC (same protocol as density_scatter_full_pipeline):
//   pos.bin     [int32 NC_actual][float32 px NC_max][py NC_max][sx NC_max][sy NC_max]
//   field_x.bin / field_y.bin  [float32 M*N]
//   ready.flag / go.flag / done.flag / quit.flag
//
// Usage:
//   density_scatter_ttnn_server M N NC_max ipc_dir [xl yl xh yh]

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
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

// ── POSIX shared-memory IPC ───────────────────────────────────────────────────
static constexpr uint32_t SHM_STATE_IDLE = 0u;
static constexpr uint32_t SHM_STATE_GO   = 1u;
static constexpr uint32_t SHM_STATE_DONE = 2u;
static constexpr uint32_t SHM_STATE_QUIT = 3u;
static constexpr size_t   SHM_HEADER_SIZE = 64u;  // one cache line
//
// Header layout (all little-endian):
//   [0:4]   state        (volatile uint32_t)
//   [4:8]   nc_actual    (int32_t)
//   [8:44]  timing[9]    (float: h2d scatter gather d2h_den upload compute download fw total)
//   [44:48] gather_mode  (uint32_t: 0=v6, 1=v7)
//   [48:64] padding
//
// Data (offset SHM_HEADER_SIZE):
//   px[soa_padded], py[soa_padded], sx[soa_padded], sy[soa_padded]
//   field_x[M*N], field_y[M*N]

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
    auto dev_rm = ttnn::to_layout(t, ttnn::Layout::ROW_MAJOR);
    auto cpu_t  = dev_rm.cpu();
    auto vec    = cpu_t.to_vector<float>();
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
// Main server
// ═══════════════════════════════════════════════════════════════════

int main(int argc, char* argv[]) {
    if (argc < 5) {
        fprintf(stderr,
            "Usage: %s M N NC_max ipc_dir [xl yl xh yh]\n",
            argv[0]);
        return 1;
    }
    int M      = atoi(argv[1]);
    int N      = atoi(argv[2]);
    int NC_max = atoi(argv[3]);
    std::string ipc_dir = argv[4];
    float xl = 0, yl = 0, xh = 1e6f, yh = 1e6f;
    if (argc >= 9) {
        xl=atof(argv[5]); yl=atof(argv[6]);
        xh=atof(argv[7]); yh=atof(argv[8]);
    }

    printf("[server] V4 Scatter + Gather-Density + TTNN DCT server\n");
    printf("[server] M=%d N=%d NC_max=%d ipc_dir=%s\n", M, N, NC_max, ipc_dir.c_str());
    fflush(stdout);

    // ── Gather mode: v11=cell-centric tile-routed (Phase 1 stub),
    //   v10=V6 scatter + bulk-async gather, v9=y-chunked scatter+SFPU gather,
    //   v8=SFPU (N≤512), v7=dense-scalar, v6=sparse ──
    // Override with env var GATHER_MODE=v6|v7|v8|v9|v10|v11|auto
    std::string gather_mode_env = getenv("GATHER_MODE") ? getenv("GATHER_MODE") : "auto";
    bool use_v11 = false;  // V11 cell-centric tile-routed (Phase 1: stub validator)
    bool use_v10 = false;  // V6 sparse scatter + V10 batched gather
    bool use_v9 = false;
    bool use_v8 = false;
    bool use_v7 = false;
    if (gather_mode_env == "v11") {
        // V11 cell-centric tile-routed scatter+accumulate. Replaces V6 sparse
        // scatter and V10 bulk gather entirely.
        use_v11 = true;
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
    const char* gmode_str = use_v11 ? "v11-tile-routed"
                          : use_v10 ? "v10-bulk"
                          : use_v9 ? "v9-chunked"
                          : use_v8 ? "v8-sfpu"
                          : use_v7 ? "v7-dense"
                          : "v6-sparse";
    printf("[server] gather_mode=%s\n", gmode_str);
    fflush(stdout);

    std::string KDIR;
#ifndef DENSITY_KERNEL_DIR
    KDIR = std::string(getenv("TT_METAL_HOME") ? getenv("TT_METAL_HOME") : ".") +
           "/../experiments/density_scatter/tt_metal/kernels/";
#else
    KDIR = DENSITY_KERNEL_DIR;
#endif

    std::string ready_flag = ipc_dir + "/ready.flag";
    std::string shm_path   = ipc_dir + "/scatter.shm";

    mkdir(ipc_dir.c_str(), 0777);
    unlink(ready_flag.c_str());

    // ── Open device ───────────────────────────────────────────────
    auto mesh_device = MeshDevice::create_unit_mesh(0);
    printf("[server] TT device opened.\n"); fflush(stdout);

    MeshCommandQueue& cq = mesh_device->mesh_command_queue();

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
    std::shared_ptr<MeshBuffer> tile_map_buf, route_buf, owned_lookup_buf, hist_buf, shard_table_buf, shard_reduce_buf;
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
    constexpr uint32_t MAX_K       = 8u;        // K cap
    constexpr uint32_t HOT_THRESHOLD = 5000u;   // tiles with count > this get K-way
    constexpr uint32_t TILE_BYTES_v11 = 32u * 32u * 4u;  // 4096 bytes per tile
    // CB slot headroom for shard slots (above n_owned_max). Same value used by
    // initial accum build and every periodic refresh so the JIT'd kernel binary
    // is identical and the cache hits.
    constexpr uint32_t V11_CB_SLOT_HEADROOM = 2u * MAX_K;

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
        make_cb_all(prog_v11_sc, 4, 1);
        make_cb_all(prog_v11_sc, 5, 1);
        for (uint32_t j = 0; j < MAX_OVERLAP; ++j) {
            make_cb_all(prog_v11_sc, 6 + j, 1);
            make_cb_all(prog_v11_sc, 14 + j, 1);
        }
        // V11 scatter scratch (matches kernel layout exactly).
        // NCRISC region: tile_to_core, staging_n[][], counts, offsets, shard_table, hdr_n.
        // Then ScatterShared, then BRISC region: staging_b[][], counts, offsets, hdr_b.
        constexpr uint32_t V11_MAX_IN_FLIGHT = 64u;
        constexpr uint32_t V11_TUPLE_BYTES   = 8u;  // packed V11Contrib
        constexpr uint32_t V11_HDR_BYTES     = 64u;
        constexpr uint32_t V11_SCATTER_SHARED_BYTES = 4u + 4u + 8u * 4u + 8u * 4u; // ~72 B
        uint32_t shared_state_off_sc = 0u;
        uint32_t brisc_state_off_sc = 0u;
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
            off += 64u;                                                 // safety gap
            v11_sc_scratch = (off + 31u) & ~31u;
        }
        printf("[server] V11 scatter scratch = %u KB (shared_off=%u, brisc_off=%u)\n",
               v11_sc_scratch / 1024u, shared_state_off_sc, brisc_state_off_sc);
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

        // BRISC kernel does combined v4_reader + scatter_b (Step 5 parallel scatter).
        auto rk_v11 = CreateKernel(prog_v11_sc, KDIR + "v11_scatter_b_dm.cpp", all_crs,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                               .noc = NOC::RISCV_0_default});
        auto ck_v11 = CreateKernel(prog_v11_sc, KDIR + "v4_compute.cpp", all_crs,
            ComputeConfig{.fp32_dest_acc_en = true,
                          .unpack_to_dest_mode = v11_unpack,
                          .math_approx_mode = false,
                          .defines = v4_defs});
        auto sk_v11 = CreateKernel(prog_v11_sc, KDIR + "v11_scatter_dm.cpp", all_crs,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                               .noc = NOC::RISCV_1_default});

        // Step-5 inter-RISC sync semaphores. data_ready/brisc_done are direct-L1
        // counters bumped via volatile stores; tables_ready is a one-shot flag.
        uint32_t sc_data_ready_sem  = CreateSemaphore(prog_v11_sc, all_crs, 0u);
        uint32_t sc_brisc_done_sem  = CreateSemaphore(prog_v11_sc, all_crs, 0u);
        uint32_t sc_tables_ready_sem= CreateSemaphore(prog_v11_sc, all_crs, 0u);

        uint32_t tm_a = (uint32_t)tile_map_buf->address();
        uint32_t rt_a = (uint32_t)route_buf->address();

        for (int c = 0; c < nc_all; ++c) {
            auto cc = all_ccs[c];
            uint32_t my_n = base_tpc + ((uint32_t)c < rem_tpc ? 1u : 0u);
            uint32_t first = (uint32_t)c * base_tpc + std::min((uint32_t)c, rem_tpc);
            uint32_t st_a = (uint32_t)shard_table_buf->address();

            // BRISC (combined reader + scatter_b). my_writer_id = my_core_id + nc_all.
            SetRuntimeArgs(prog_v11_sc, rk_v11, cc, {
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
            });
            SetRuntimeArgs(prog_v11_sc, ck_v11, cc, {my_n});
            // NCRISC scatter. my_writer_id = my_core_id (existing slot).
            SetRuntimeArgs(prog_v11_sc, sk_v11, cc, {
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
            });
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
                          .defines = v4_defs});
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

    // ── JIT compile ───────────────────────────────────────────────
    printf("[server] JIT compiling kernels...\n"); fflush(stdout);
    auto t_jit = hrclock::now();
    EnqueueMeshWorkload(cq, wl_scatter, false); Finish(cq);
    EnqueueMeshWorkload(cq, wl_gather,  false); Finish(cq);
    if (use_v11) {
        EnqueueMeshWorkload(cq, wl_v11_hist,    false); Finish(cq);
        EnqueueMeshWorkload(cq, wl_v11_scatter, false); Finish(cq);
        EnqueueMeshWorkload(cq, wl_v11_accum,   false); Finish(cq);
    }
    printf("[server] JIT done: %.1f ms\n", ms_since(t_jit)); fflush(stdout);

    // ── Upload DCT matrices as TTNN tensors ───────────────────────
    printf("[server] Uploading DCT matrices via TTNN...\n"); fflush(stdout);
    TTNNDCTSolver ttnn_solver;
    ttnn_solver.init(M, N, bsx, bsy, mesh_device.get());
    printf("[server] TTNN solver ready.\n"); fflush(stdout);

    // ── Open shared memory (Python creates + ftruncates before launching us) ──
    {
        int shm_fd = -1;
        // Poll briefly in case Python is still creating the file
        for (int i = 0; i < 100 && shm_fd < 0; ++i) {
            shm_fd = open(shm_path.c_str(), O_RDWR);
            if (shm_fd < 0) usleep(100000);  // 100 ms
        }
        if (shm_fd < 0) {
            perror(("[server] open shm: " + shm_path).c_str());
            return 1;
        }

        size_t pos_bytes   = (size_t)soa_padded * sizeof(float);
        size_t field_bytes = (size_t)M * N * sizeof(float);
        size_t shm_size    = SHM_HEADER_SIZE + 4 * pos_bytes + 2 * field_bytes;

        void* shm_ptr = mmap(nullptr, shm_size, PROT_READ | PROT_WRITE,
                             MAP_SHARED, shm_fd, 0);
        close(shm_fd);
        if (shm_ptr == MAP_FAILED) { perror("[server] mmap shm"); return 1; }

        volatile uint32_t* shm_state = (volatile uint32_t*)shm_ptr;
        volatile int32_t*  shm_nc    = (volatile int32_t*)((char*)shm_ptr + 4);
        const float* shm_px  = (const float*)((char*)shm_ptr + SHM_HEADER_SIZE);
        const float* shm_py  = shm_px + soa_padded;
        const float* shm_sx  = shm_py + soa_padded;
        const float* shm_sy  = shm_sx + soa_padded;
        float*       shm_fx  = (float*)((char*)shm_ptr + SHM_HEADER_SIZE + 4 * pos_bytes);
        float*       shm_fy  = shm_fx + M * N;

        // ── Signal READY (after shm is mapped so Python can immediately use it) ──
        flag_write(ready_flag, "READY\n");
        printf("[server] READY — polling shm state\n"); fflush(stdout);

        std::vector<float> px(soa_padded, 0.0f), py(soa_padded, 0.0f),
                           sx(soa_padded, 0.0f), sy(soa_padded, 0.0f);
        std::vector<float> density_flat(M * N);
        std::vector<float> field_x(M * N), field_y(M * N);

        // V11 hist data buffer (only used when use_v11)
        std::vector<uint32_t> v11_hist_data;
        std::vector<uint32_t> v11_global_count;
        if (use_v11) {
            v11_hist_data.assign((size_t)nc_all * (hist_pgsz_v11 / 4u), 0u);
            v11_global_count.assign((size_t)M_tiles * N_tiles_v11, 0u);
        }
        bool v11_dbg_first = true;
        // Periodic shard_table refresh: cell positions migrate during DREAMPlace
        // optimization, so iter-0 hot tiles go cold and new tiles become hot. Every
        // V11_HIST_REFRESH_ITERS iters we re-run the histogram and rebuild the
        // shard-aware accum workload. (Iter 0 also triggers via the modulo.)
        //
        // KNOWN ISSUE: each refresh creates a new Program with new semaphores,
        // and TT-Metal's per-core semaphore limit is 16. After ~4 refreshes
        // we hit the limit and crash. Disabled until we refactor the refresh
        // path to reuse the original Program's semaphores via SetRuntimeArgs only.
        // KNOWN ISSUE: TT-Metal's per-core semaphore limit is 16, and each
        // refresh creates a new Program with fresh semaphores. Even at
        // 400-iter cadence (≤ 3 refreshes for a 1000-iter run) we tripped
        // over an empty metrics file — likely some other resource leak,
        // not just semaphores. Set to 1_000_000 to disable.
        constexpr uint64_t V11_HIST_REFRESH_ITERS = 1000000u;
        uint64_t v11_iter = 0;

        while (true) {
            uint32_t st = *shm_state;
            if (st == SHM_STATE_QUIT) {
                printf("[server] QUIT received.\n"); fflush(stdout);
                break;
            }
            if (st != SHM_STATE_GO) { usleep(500); continue; }

            // ── Copy positions from shm into zero-padded local vectors ────────
            int32_t nc_actual = std::min((int32_t)*shm_nc, NC_max);
            std::memcpy(px.data(), shm_px, (size_t)nc_actual * sizeof(float));
            std::memcpy(py.data(), shm_py, (size_t)nc_actual * sizeof(float));
            std::memcpy(sx.data(), shm_sx, (size_t)nc_actual * sizeof(float));
            std::memcpy(sy.data(), shm_sy, (size_t)nc_actual * sizeof(float));
            // Tail beyond nc_actual stays 0 (vectors initialized once, nc fixed)

            // ── H2D: upload cell positions ────────────────────────────────────
            auto ts = hrclock::now();
            EnqueueWriteMeshBuffer(cq, px_buf, px, false);
            EnqueueWriteMeshBuffer(cq, py_buf, py, false);
            EnqueueWriteMeshBuffer(cq, sx_buf, sx, false);
            EnqueueWriteMeshBuffer(cq, sy_buf, sy, false);
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

                ts = hrclock::now();
                EnqueueMeshWorkload(cq, wl_v11_scatter, false); Finish(cq);
                scatter_ms = ms_since(ts);
                ts = hrclock::now();
                // Merged kernel does accum + reduce_a + reduce_bc in one program;
                // shard sync is via NOC semaphores (1 host Finish() vs the old 3).
                EnqueueMeshWorkload(cq, wl_v11_accum, false); Finish(cq);
                gather_ms = ms_since(ts);
                v11_iter++;
            } else {
                ts = hrclock::now();
                EnqueueMeshWorkload(cq, wl_scatter, false); Finish(cq);
                scatter_ms = ms_since(ts);
                ts = hrclock::now();
                EnqueueMeshWorkload(cq, wl_gather, false); Finish(cq);
                gather_ms = ms_since(ts);
            }

            // ── D2H: density readback ─────────────────────────────────────────
            ts = hrclock::now();
            EnqueueReadMeshBuffer(cq, density_flat, density_buf, true);
            double density_d2h_ms = ms_since(ts);

            // ── Optional debug dump ──────────────────────────────────────────
            // EXPORT_DENSITY_PATH dumps EVERY iter (overwrite-style; the
            // smoke test expects this single-iter behavior).
            // EXPORT_POS_PATH dumps cell positions ONLY at iter 100 (used
            // for the iter-100 density correctness comparison).
            {
                const char* dump_path = getenv("EXPORT_DENSITY_PATH");
                if (dump_path && dump_path[0]) {
                    FILE* f = fopen(dump_path, "wb");
                    if (f) {
                        fwrite(density_flat.data(), sizeof(float),
                               (size_t)M * (size_t)N, f);
                        fclose(f);
                    }
                }
                static uint32_t dump_iter_count = 0;
                dump_iter_count++;
                // Dump positions at multiple iters when EXPORT_POS_PATH is set.
                // Filename: <path>.iter<N>.bin — so we can diff which cells
                // are fixed (positions identical across iters) vs movable.
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
                // DEBUG: at iter 100, also dump route_buf (V11 scatter output)
                // to a file. Lets us check whether ghost-corner tuples come
                // from scatter (route_buf has them) or accum (route_buf clean).
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

            // ── TTNN DCT solve ────────────────────────────────────────────────
            double upload_ms = 0, compute_ms = 0, download_ms = 0;
            ttnn_solver.solve(density_flat, field_x, field_y,
                              mesh_device.get(),
                              upload_ms, compute_ms, download_ms);

            // ── Write fields directly into shm ────────────────────────────────
            ts = hrclock::now();
            std::memcpy(shm_fx, field_x.data(), (size_t)M * N * sizeof(float));
            std::memcpy(shm_fy, field_y.data(), (size_t)M * N * sizeof(float));
            double fw_ms = ms_since(ts);

            double total_ms = h2d_ms + scatter_ms + gather_ms + density_d2h_ms
                            + upload_ms + compute_ms + download_ms + fw_ms;

            // ── Write timing into shm header ──────────────────────────────────
            float* tf = (float*)((char*)shm_ptr + 8);
            tf[0] = (float)h2d_ms;
            tf[1] = (float)scatter_ms;
            tf[2] = (float)gather_ms;
            tf[3] = (float)density_d2h_ms;
            tf[4] = (float)upload_ms;
            tf[5] = (float)compute_ms;
            tf[6] = (float)download_ms;
            tf[7] = (float)fw_ms;
            tf[8] = (float)total_ms;
            ((uint32_t*)((char*)shm_ptr + 44))[0] = use_v11 ? 5u
                                                  : use_v10 ? 4u
                                                  : use_v9 ? 3u
                                                  : (use_v8 ? 2u : (use_v7 ? 1u : 0u));

            // ── Signal DONE ───────────────────────────────────────────────────
            *shm_state = SHM_STATE_DONE;

            printf("[server] done h2d=%.3f scatter=%.3f gather=%.3f d2h_den=%.3f "
                   "upload=%.3f compute=%.3f download=%.3f fw=%.3f total=%.3f "
                   "gather_mode=%s\n",
                   h2d_ms, scatter_ms, gather_ms, density_d2h_ms,
                   upload_ms, compute_ms, download_ms, fw_ms, total_ms,
                   gmode_str);
            fflush(stdout);
        }

        munmap(shm_ptr, shm_size);
    }

    return 0;
}
