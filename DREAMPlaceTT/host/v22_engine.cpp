// SPDX-License-Identifier: Apache-2.0
//
// V22 optimizer-on-chip engine. See v22_engine.h.
// Chains the four validated element-wise SFPU kernels through persistent DRAM
// buffers, sharing a caller-provided MeshDevice (no second hardware open).

#include "v22_engine.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tt_metal.hpp>

namespace v22 {

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;
using hrclock = std::chrono::high_resolution_clock;

template <class T> static double ms_since(T t0) {
    return std::chrono::duration<double, std::milli>(hrclock::now() - t0).count();
}

#ifndef DENSITY_KERNEL_DIR
#define DENSITY_KERNEL_DIR "kernels/"
#endif

static constexpr uint32_t TILE_ELEMS = 32u * 32u;
static constexpr uint32_t TILE_BYTES = TILE_ELEMS * 4u;

struct V22OptEngine::Impl {
    MeshDevice* mesh_device = nullptr;
    MeshCommandQueue* cq = nullptr;
    int num_nodes = 0;
    uint32_t n_elems = 0;          // 2 * num_nodes
    uint32_t n_tiles_total = 0;
    uint32_t padded_elems = 0;
    uint32_t tiles_per_core = 0;
    int nc_all = 0;
    CoreCoord grid;
    CoreRangeSet all_crs;
    MeshCoordinateRange device_range = MeshCoordinateRange{MeshCoordinate{0, 0}, MeshCoordinate{0, 0}};

    // Persistent DRAM buffers (all 2*nn fp32 unless noted)
    std::shared_ptr<MeshBuffer> b_wl, b_dg, b_v, b_uprev;     // step inputs
    std::shared_ptr<MeshBuffer> b_g1, b_g2;                   // combine→precond→nesterov grad
    std::shared_ptr<MeshBuffer> b_unew, b_vnew, b_vclamp;     // nesterov/clamp outputs
    std::shared_ptr<MeshBuffer> b_lo, b_hi, b_pw, b_area;     // constants (configure once)
    std::shared_ptr<MeshBuffer> s_dw, s_adw, s_alpha, s_coef; // 1-tile scalar broadcasts

    // Four programs (built once)
    bool built = false;
    MeshWorkload wl_combine, wl_precond, wl_nesterov, wl_clamp;
    // Phase-B resident variants: nesterov writes u in-place into b_uprev,
    // clamp writes pos in-place into b_v (so both stay resident across steps).
    bool built_res = false;
    MeshWorkload wl_nesterov_res, wl_clamp_res;

    // Step B: step-size reduction (ss/sy/yy over s,y).
    std::shared_ptr<MeshBuffer> b_s, b_y, b_ssout;   // s, y inputs + nc*3 partial tiles
    bool built_ss = false;
    MeshWorkload wl_stepsize;
    std::vector<int> ss_core_nt;                     // tiles/core (mask empty-core partials)

    // No-host line-search machinery (all reuse combine/nesterov/clamp kernels).
    std::shared_ptr<MeshBuffer> b_g_k;               // snapshot of preconditioned grad @ v_k
    std::shared_ptr<MeshBuffer> s_neg1, s_zero;      // combine scalars: diff (-1) and copy (0)
    bool built_nohost = false;
    MeshWorkload wl_snap_gk, wl_diff_v, wl_diff_g, wl_nesterov_trial, wl_commit_v, wl_commit_u;

    // Host staging (avoid per-iter alloc)
    std::vector<float> stage;        // padded_elems
    std::vector<float> scalar_tile;  // TILE_ELEMS

    Impl(void* md, int nn) : mesh_device(static_cast<MeshDevice*>(md)), num_nodes(nn) {
        if (!mesh_device) throw std::runtime_error("v22: null mesh_device");
        if (nn <= 0)      throw std::runtime_error("v22: num_nodes must be > 0");
        cq = &mesh_device->mesh_command_queue();
        n_elems = (uint32_t)nn * 2u;
        n_tiles_total = (n_elems + TILE_ELEMS - 1u) / TILE_ELEMS;
        padded_elems  = n_tiles_total * TILE_ELEMS;

        grid = mesh_device->compute_with_storage_grid_size();
        nc_all = (int)(grid.x * grid.y);
        all_crs = CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(grid.x - 1, grid.y - 1)));
        tiles_per_core = (n_tiles_total + (uint32_t)nc_all - 1u) / (uint32_t)nc_all;

        auto make_buf = [&](uint64_t bytes, uint32_t pg) {
            DeviceLocalBufferConfig cfg{.page_size = pg, .buffer_type = BufferType::DRAM};
            ReplicatedBufferConfig rcfg{.size = bytes};
            return MeshBuffer::create(rcfg, cfg, mesh_device);
        };
        const uint64_t io = (uint64_t)n_tiles_total * TILE_BYTES;
        b_wl = make_buf(io, TILE_BYTES);  b_dg   = make_buf(io, TILE_BYTES);
        b_v  = make_buf(io, TILE_BYTES);  b_uprev= make_buf(io, TILE_BYTES);
        b_g1 = make_buf(io, TILE_BYTES);  b_g2   = make_buf(io, TILE_BYTES);
        b_unew = make_buf(io, TILE_BYTES); b_vnew = make_buf(io, TILE_BYTES);
        b_vclamp = make_buf(io, TILE_BYTES);
        b_lo = make_buf(io, TILE_BYTES);  b_hi   = make_buf(io, TILE_BYTES);
        b_pw = make_buf(io, TILE_BYTES);  b_area = make_buf(io, TILE_BYTES);
        s_dw = make_buf(TILE_BYTES, TILE_BYTES); s_adw  = make_buf(TILE_BYTES, TILE_BYTES);
        s_alpha = make_buf(TILE_BYTES, TILE_BYTES); s_coef = make_buf(TILE_BYTES, TILE_BYTES);
        b_s = make_buf(io, TILE_BYTES); b_y = make_buf(io, TILE_BYTES);
        b_ssout = make_buf((uint64_t)nc_all * 3u * TILE_BYTES, TILE_BYTES);
        b_g_k = make_buf(io, TILE_BYTES);
        s_neg1 = make_buf(TILE_BYTES, TILE_BYTES); s_zero = make_buf(TILE_BYTES, TILE_BYTES);

        stage.assign(padded_elems, 0.0f);
        scalar_tile.assign(TILE_ELEMS, 0.0f);

        std::printf("[v22_engine] num_nodes=%d n_elems=%u tiles=%u nc=%d tiles/core=%u  DRAM≈%.1f MB\n",
                    nn, n_elems, n_tiles_total, nc_all, tiles_per_core,
                    13.0 * (double)io / 1048576.0);
        std::fflush(stdout);
    }

    // Upload an array (host layout [x(nn)|y(nn)]) into a buffer, INTERLEAVED on device
    // ([x0,y0,x1,y1,...]) + zero-padded. Interleaving matches the on-chip unsort's b_dg
    // layout; element-wise kernels are layout-agnostic so all operands share it.
    // NOTE: `stage` / `scalar_tile` are SHARED host buffers reused across calls.
    // step()/configure() upload several buffers back-to-back from them, so the
    // write MUST be blocking — a non-blocking write returns before the host buffer
    // is consumed, and the next upload() then overwrites it mid-DMA (a race that
    // corrupted the larger configs' g_k/v_k/scalars → first-step blow-up to ~1e34).
    void upload(std::shared_ptr<MeshBuffer>& buf, const float* src) {
        const uint32_t nn = (uint32_t)num_nodes;
        for (uint32_t i = 0; i < nn; ++i) { stage[2u*i] = src[i]; stage[2u*i+1u] = src[nn+i]; }
        if (padded_elems > n_elems)
            std::memset(stage.data() + n_elems, 0, (size_t)(padded_elems - n_elems) * sizeof(float));
        EnqueueWriteMeshBuffer(*cq, buf, stage, true);   // blocking: stage is reused next call
    }
    void upload_scalar(std::shared_ptr<MeshBuffer>& buf, float v) {
        std::fill(scalar_tile.begin(), scalar_tile.end(), v);
        EnqueueWriteMeshBuffer(*cq, buf, scalar_tile, true);   // blocking: scalar_tile reused
    }
    void download(std::shared_ptr<MeshBuffer>& buf, float* dst) {  // device interleaved -> host [x|y]
        const uint32_t nn = (uint32_t)num_nodes;
        EnqueueReadMeshBuffer(*cq, stage, buf, true);
        for (uint32_t i = 0; i < nn; ++i) { dst[i] = stage[2u*i]; dst[nn+i] = stage[2u*i+1u]; }
    }

    void make_cb(Program& prog, uint32_t idx, uint32_t n_slots) {
        CircularBufferConfig cfg(n_slots * TILE_BYTES, {{(CBIndex)idx, DataFormat::Float32}});
        cfg.set_page_size((CBIndex)idx, TILE_BYTES);
        CreateCircularBuffer(prog, all_crs, cfg);
    }

    // Generic single-output element-wise program (reader=v22_elt_reader,
    // writer=v22_elt_writer, compute=<file>). Buffer addrs are baked into args.
    Program build_elt(const char* compute_file, uint32_t n_streams, uint32_t n_scalars,
                      uint32_t in0, uint32_t in1, uint32_t in2,
                      uint32_t s0, uint32_t s1, uint32_t out) {
        Program prog = CreateProgram();
        make_cb(prog, (uint32_t)CBIndex::c_0, 2);
        make_cb(prog, (uint32_t)CBIndex::c_1, 2);
        make_cb(prog, (uint32_t)CBIndex::c_2, 2);
        make_cb(prog, (uint32_t)CBIndex::c_3, 1);
        make_cb(prog, (uint32_t)CBIndex::c_4, 1);
        make_cb(prog, (uint32_t)CBIndex::c_16, 2);

        auto rk = CreateKernel(prog, std::string(DENSITY_KERNEL_DIR) + "v22_elt_reader.cpp", all_crs,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        auto wk = CreateKernel(prog, std::string(DENSITY_KERNEL_DIR) + "v22_elt_writer.cpp", all_crs,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        std::vector<UnpackToDestMode> um(64, UnpackToDestMode::Default);
        for (uint32_t i = 0; i <= 4; ++i) um[i] = UnpackToDestMode::UnpackToDestFp32;
        auto ck = CreateKernel(prog, std::string(DENSITY_KERNEL_DIR) + compute_file, all_crs,
            ComputeConfig{.fp32_dest_acc_en = true, .unpack_to_dest_mode = um, .math_approx_mode = false});

        for (int c = 0; c < nc_all; ++c) {
            const uint32_t ts = std::min((uint32_t)c * tiles_per_core, n_tiles_total);
            const uint32_t te = std::min(ts + tiles_per_core, n_tiles_total);
            const uint32_t nt = te - ts;
            CoreCoord cc{(uint32_t)(c % grid.x), (uint32_t)(c / grid.x)};
            SetRuntimeArgs(prog, rk, cc, {nt, ts, n_streams, n_scalars, in0, in1, in2, s0, s1});
            SetRuntimeArgs(prog, wk, cc, {nt, ts, out});
            SetRuntimeArgs(prog, ck, cc, {nt});
        }
        return prog;
    }

    // Nesterov program (3 streams v/g/u + 2 scalars alpha/coef → 2 outputs u/v).
    Program build_nesterov(uint32_t addr_v, uint32_t addr_g, uint32_t addr_u,
                           uint32_t addr_alpha, uint32_t addr_coef,
                           uint32_t addr_uout, uint32_t addr_vout) {
        Program prog = CreateProgram();
        make_cb(prog, (uint32_t)CBIndex::c_0, 2);
        make_cb(prog, (uint32_t)CBIndex::c_1, 2);
        make_cb(prog, (uint32_t)CBIndex::c_2, 2);
        make_cb(prog, (uint32_t)CBIndex::c_3, 1);
        make_cb(prog, (uint32_t)CBIndex::c_4, 1);
        make_cb(prog, (uint32_t)CBIndex::c_16, 2);
        make_cb(prog, (uint32_t)CBIndex::c_17, 2);

        auto rk = CreateKernel(prog, std::string(DENSITY_KERNEL_DIR) + "v22_nesterov_reader.cpp", all_crs,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        auto wk = CreateKernel(prog, std::string(DENSITY_KERNEL_DIR) + "v22_nesterov_writer.cpp", all_crs,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        std::vector<UnpackToDestMode> um(64, UnpackToDestMode::Default);
        for (uint32_t i = 0; i <= 4; ++i) um[i] = UnpackToDestMode::UnpackToDestFp32;
        auto ck = CreateKernel(prog, std::string(DENSITY_KERNEL_DIR) + "v22_nesterov_compute.cpp", all_crs,
            ComputeConfig{.fp32_dest_acc_en = true, .unpack_to_dest_mode = um, .math_approx_mode = false});

        for (int c = 0; c < nc_all; ++c) {
            const uint32_t ts = std::min((uint32_t)c * tiles_per_core, n_tiles_total);
            const uint32_t te = std::min(ts + tiles_per_core, n_tiles_total);
            const uint32_t nt = te - ts;
            CoreCoord cc{(uint32_t)(c % grid.x), (uint32_t)(c / grid.x)};
            SetRuntimeArgs(prog, rk, cc, {nt, ts, addr_v, addr_g, addr_u, addr_alpha, addr_coef});
            SetRuntimeArgs(prog, wk, cc, {nt, ts, addr_uout, addr_vout});
            SetRuntimeArgs(prog, ck, cc, {nt});
        }
        return prog;
    }

    void build_all() {
        if (built) return;
        auto A = [](std::shared_ptr<MeshBuffer>& b) { return (uint32_t)b->address(); };
        // combine: in0=wl, in1=dg, scalar0=dw → out=g1
        wl_combine.add_program(device_range,
            build_elt("v22_gradcombine_compute.cpp", 2, 1, A(b_wl), A(b_dg), A(b_wl), A(s_dw), A(s_dw), A(b_g1)));
        // precond: in0=g1(grad), in1=pw, in2=area, scalar0=adw → out=g2
        wl_precond.add_program(device_range,
            build_elt("v22_precond_compute.cpp", 3, 1, A(b_g1), A(b_pw), A(b_area), A(s_adw), A(s_adw), A(b_g2)));
        // nesterov: v_k, g2, u_prev, alpha, coef → u_new, v_new
        wl_nesterov.add_program(device_range,
            build_nesterov(A(b_v), A(b_g2), A(b_uprev), A(s_alpha), A(s_coef), A(b_unew), A(b_vnew)));
        // clamp: in0=v_new(pos), in1=lo, in2=hi → out=v_clamp
        wl_clamp.add_program(device_range,
            build_elt("v22_clamp_compute.cpp", 3, 0, A(b_vnew), A(b_lo), A(b_hi), A(b_lo), A(b_lo), A(b_vclamp)));
        built = true;
    }

    // Step B: per-core lane-wise reduction (ss/sy/yy) over s,y. reader streams the
    // core's tile shard 3x (one per product); compute accumulates in DST; writer
    // packs 3 partial tiles per core. Mirrors v22_stepsize_microbench_host.cpp.
    void build_stepsize() {
        if (built_ss) return;
        Program prog = CreateProgram();
        make_cb(prog, (uint32_t)CBIndex::c_0, 4);
        make_cb(prog, (uint32_t)CBIndex::c_1, 4);
        make_cb(prog, (uint32_t)CBIndex::c_16, 4);
        auto rk = CreateKernel(prog, std::string(DENSITY_KERNEL_DIR) + "v22_stepsize_reader.cpp", all_crs,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        auto wk = CreateKernel(prog, std::string(DENSITY_KERNEL_DIR) + "v22_stepsize_writer.cpp", all_crs,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        std::vector<UnpackToDestMode> um(64, UnpackToDestMode::Default);
        um[0] = UnpackToDestMode::UnpackToDestFp32; um[1] = UnpackToDestMode::UnpackToDestFp32;
        auto ck = CreateKernel(prog, std::string(DENSITY_KERNEL_DIR) + "v22_stepsize_compute.cpp", all_crs,
            ComputeConfig{.fp32_dest_acc_en = true, .unpack_to_dest_mode = um, .math_approx_mode = false});
        const uint32_t addr_s = (uint32_t)b_s->address();
        const uint32_t addr_y = (uint32_t)b_y->address();
        const uint32_t addr_o = (uint32_t)b_ssout->address();
        ss_core_nt.assign(nc_all, 0);
        for (int c = 0; c < nc_all; ++c) {
            const uint32_t ts = std::min((uint32_t)c * tiles_per_core, n_tiles_total);
            const uint32_t te = std::min(ts + tiles_per_core, n_tiles_total);
            const uint32_t nt = te - ts;
            ss_core_nt[c] = (int)nt;
            CoreCoord cc{(uint32_t)(c % grid.x), (uint32_t)(c / grid.x)};
            SetRuntimeArgs(prog, rk, cc, {nt, ts, addr_s, addr_y});
            SetRuntimeArgs(prog, wk, cc, {(uint32_t)c * 3u, addr_o});
            SetRuntimeArgs(prog, ck, cc, {nt});
        }
        wl_stepsize.add_program(device_range, std::move(prog));
        built_ss = true;
    }

    void build_all_resident() {
        build_all();  // combine + precond reused as-is (b_wl,b_dg → b_g1 → b_g2)
        if (built_res) return;
        auto A = [](std::shared_ptr<MeshBuffer>& b) { return (uint32_t)b->address(); };
        // nesterov: v_k=b_v, g2, u_prev=b_uprev → u_new=b_uprev (in-place), v_new=b_vnew
        wl_nesterov_res.add_program(device_range,
            build_nesterov(A(b_v), A(b_g2), A(b_uprev), A(s_alpha), A(s_coef), A(b_uprev), A(b_vnew)));
        // clamp: v_new=b_vnew, lo, hi → b_v (in-place resident pos)
        wl_clamp_res.add_program(device_range,
            build_elt("v22_clamp_compute.cpp", 3, 0, A(b_vnew), A(b_lo), A(b_hi), A(b_lo), A(b_lo), A(b_v)));
        built_res = true;
    }

    // No-host line-search programs. combine doubles as: copy (out = a + 0*b) and
    // scalar-add diff (out = a + (-1)*b = a-b). nesterov_trial reads the g_k snapshot
    // and writes the UNCOMMITTED b_vnew/b_unew (clamp via build_all's wl_clamp → b_vclamp).
    void build_nohost() {
        build_all_resident();   // also gives build_all (combine/precond/clamp)
        build_stepsize();       // the ss/sy/yy reduction over b_s,b_y
        if (built_nohost) return;
        auto A = [](std::shared_ptr<MeshBuffer>& b) { return (uint32_t)b->address(); };
        upload_scalar(s_neg1, -1.0f);  upload_scalar(s_zero, 0.0f);  Finish(*cq);
        // snapshot g_k: b_g_k = b_g2 + 0*b_g2
        wl_snap_gk.add_program(device_range,
            build_elt("v22_gradcombine_compute.cpp", 2, 1, A(b_g2), A(b_g2), A(b_g2), A(s_zero), A(s_zero), A(b_g_k)));
        // dv = v_kp1 - v_k  (b_vclamp - b_v) → b_s
        wl_diff_v.add_program(device_range,
            build_elt("v22_gradcombine_compute.cpp", 2, 1, A(b_vclamp), A(b_v), A(b_vclamp), A(s_neg1), A(s_neg1), A(b_s)));
        // dg = g_kp1 - g_k  (b_g2 - b_g_k) → b_y
        wl_diff_g.add_program(device_range,
            build_elt("v22_gradcombine_compute.cpp", 2, 1, A(b_g2), A(b_g_k), A(b_g2), A(s_neg1), A(s_neg1), A(b_y)));
        // trial nesterov: v_k=b_v, g=b_g_k(snapshot), u=b_uprev → u_new=b_unew, v_new=b_vnew (uncommitted)
        wl_nesterov_trial.add_program(device_range,
            build_nesterov(A(b_v), A(b_g_k), A(b_uprev), A(s_alpha), A(s_coef), A(b_unew), A(b_vnew)));
        // commit: b_v <- b_vclamp ; b_uprev <- b_unew  (copy via combine +0*)
        wl_commit_v.add_program(device_range,
            build_elt("v22_gradcombine_compute.cpp", 2, 1, A(b_vclamp), A(b_vclamp), A(b_vclamp), A(s_zero), A(s_zero), A(b_v)));
        wl_commit_u.add_program(device_range,
            build_elt("v22_gradcombine_compute.cpp", 2, 1, A(b_unew), A(b_unew), A(b_unew), A(s_zero), A(s_zero), A(b_uprev)));
        built_nohost = true;
    }
};

V22OptEngine::V22OptEngine(void* md, int nn) : impl_(std::make_unique<Impl>(md, nn)) {}
V22OptEngine::~V22OptEngine() = default;
int V22OptEngine::num_nodes() const noexcept { return impl_->num_nodes; }

void V22OptEngine::configure(const float* lo, const float* hi,
                             const float* pin_w, const float* area) {
    impl_->upload(impl_->b_lo, lo);
    impl_->upload(impl_->b_hi, hi);
    impl_->upload(impl_->b_pw, pin_w);
    impl_->upload(impl_->b_area, area);
    Finish(*impl_->cq);
    impl_->build_all();
}

void V22OptEngine::prewarm() {
    impl_->build_all();
    // Dummy launch to JIT all four programs.
    std::vector<float> z(impl_->n_elems, 0.0f);
    OptTiming t;
    step(z.data(), z.data(), z.data(), z.data(), 0.0f, 0.0f, 0.0f, 1.0f, z.data(), z.data(), &t);
}

void V22OptEngine::step(const float* wl_grad, const float* density_grad,
                        const float* v_k, const float* u_prev,
                        float step_size, float coef, float density_weight, float precond_alpha,
                        float* u_new, float* v_new,
                        OptTiming* timing_out) {
    auto& I = *impl_;
    I.build_all();
    auto t0 = hrclock::now();
    I.upload(I.b_wl, wl_grad);
    I.upload(I.b_dg, density_grad);
    I.upload(I.b_v,  v_k);
    I.upload(I.b_uprev, u_prev);
    I.upload_scalar(I.s_dw,    density_weight);
    I.upload_scalar(I.s_adw,   precond_alpha * density_weight);
    I.upload_scalar(I.s_alpha, step_size);
    I.upload_scalar(I.s_coef,  coef);
    Finish(*I.cq);
    double h2d = ms_since(t0);

    auto t1 = hrclock::now();
    EnqueueMeshWorkload(*I.cq, I.wl_combine,  false);
    EnqueueMeshWorkload(*I.cq, I.wl_precond,  false);
    EnqueueMeshWorkload(*I.cq, I.wl_nesterov, false);
    EnqueueMeshWorkload(*I.cq, I.wl_clamp,    false);
    Finish(*I.cq);
    double comp = ms_since(t1);

    auto t2 = hrclock::now();
    I.download(I.b_unew,   u_new);
    I.download(I.b_vclamp, v_new);
    double d2h = ms_since(t2);

    if (timing_out) {
        timing_out->h2d_ms = h2d;
        timing_out->compute_ms = comp;
        timing_out->d2h_ms = d2h;
        timing_out->total_ms = ms_since(t0);
    }
}

// ── Phase B: chip-resident pos / u ──
void V22OptEngine::set_pos(const float* v_init) {
    auto& I = *impl_;
    I.build_all_resident();
    I.upload(I.b_v, v_init);       // resident pos = v_k
    I.upload(I.b_uprev, v_init);   // u_k = v_k clone (matches torch init)
    Finish(*I.cq);
}

void V22OptEngine::step_resident(const float* wl_grad, const float* density_grad,
                                 float step_size, float coef, float density_weight, float precond_alpha,
                                 OptTiming* timing_out) {
    auto& I = *impl_;
    I.build_all_resident();
    auto t0 = hrclock::now();
    I.upload(I.b_wl, wl_grad);
    I.upload(I.b_dg, density_grad);
    I.upload_scalar(I.s_dw,    density_weight);
    I.upload_scalar(I.s_adw,   precond_alpha * density_weight);
    I.upload_scalar(I.s_alpha, step_size);
    I.upload_scalar(I.s_coef,  coef);
    Finish(*I.cq);
    double h2d = ms_since(t0);

    auto t1 = hrclock::now();
    EnqueueMeshWorkload(*I.cq, I.wl_combine,      false);
    EnqueueMeshWorkload(*I.cq, I.wl_precond,      false);
    EnqueueMeshWorkload(*I.cq, I.wl_nesterov_res, false);  // u in-place → b_uprev
    EnqueueMeshWorkload(*I.cq, I.wl_clamp_res,    false);  // pos in-place → b_v
    Finish(*I.cq);
    double comp = ms_since(t1);

    if (timing_out) {
        timing_out->h2d_ms = h2d;
        timing_out->compute_ms = comp;
        timing_out->d2h_ms = 0.0;
        timing_out->total_ms = ms_since(t0);
    }
}

void V22OptEngine::get_pos(float* out) {
    impl_->download(impl_->b_v, out);
}

void V22OptEngine::stepsize_sums(const float* s, const float* y,
                                 double* ss, double* sy, double* yy) {
    auto& I = *impl_;
    I.build_stepsize();
    I.upload(I.b_s, s);   // interleave + zero-pad tail (pad=0 contributes 0 to the sums)
    I.upload(I.b_y, y);
    Finish(*I.cq);
    EnqueueMeshWorkload(*I.cq, I.wl_stepsize, false);
    Finish(*I.cq);
    std::vector<float> partials((size_t)I.nc_all * 3u * TILE_ELEMS, 0.0f);
    EnqueueReadMeshBuffer(*I.cq, partials, I.b_ssout, true);
    double a0 = 0.0, a1 = 0.0, a2 = 0.0;
    for (int c = 0; c < I.nc_all; ++c) {
        if (I.ss_core_nt[c] == 0) continue;  // empty-core partials are garbage
        const size_t base = (size_t)c * 3u * TILE_ELEMS;
        for (uint32_t i = 0; i < TILE_ELEMS; ++i) {
            a0 += partials[base + 0 * TILE_ELEMS + i];
            a1 += partials[base + 1 * TILE_ELEMS + i];
            a2 += partials[base + 2 * TILE_ELEMS + i];
        }
    }
    if (ss) *ss = a0;
    if (sy) *sy = a1;
    if (yy) *yy = a2;
}

void V22OptEngine::combine_precond_ondevice(const float* wl_grad,
                                            float density_weight, float precond_alpha) {
    auto& I = *impl_;
    I.build_all();                 // combine + precond programs
    I.upload(I.b_wl, wl_grad);     // raw wirelength grad (the allowed h2d)
    I.upload_scalar(I.s_dw,  -density_weight);                 // combine: g1 = wl + (-dw)*b_dg = wl - dw*force
    I.upload_scalar(I.s_adw, precond_alpha * density_weight);  // precond divisor: max(pw + a*dw*area, 1)
    Finish(*I.cq);
    EnqueueMeshWorkload(*I.cq, I.wl_combine, false);   // b_wl, b_dg(resident) -> b_g1
    EnqueueMeshWorkload(*I.cq, I.wl_precond, false);   // b_g1, b_pw, b_area -> b_g2
    Finish(*I.cq);
}

void V22OptEngine::get_precond_grad(float* out) {
    impl_->download(impl_->b_g2, out);
}

// ── No-host line-search loop ──
void V22OptEngine::snapshot_precond_grad() {
    auto& I = *impl_; I.build_nohost();
    EnqueueMeshWorkload(*I.cq, I.wl_snap_gk, false);   // b_g_k <- b_g2
    Finish(*I.cq);
}

void V22OptEngine::nesterov_clamp_trial(float step_size, float coef) {
    auto& I = *impl_; I.build_nohost();
    I.upload_scalar(I.s_alpha, step_size);
    I.upload_scalar(I.s_coef,  coef);
    Finish(*I.cq);
    EnqueueMeshWorkload(*I.cq, I.wl_nesterov_trial, false);  // b_v,b_g_k,b_uprev -> b_unew,b_vnew
    EnqueueMeshWorkload(*I.cq, I.wl_clamp,          false);  // b_vnew -> b_vclamp (v_k/u_k intact)
    Finish(*I.cq);
}

void V22OptEngine::linesearch_sums(double* ss, double* yy) {
    auto& I = *impl_; I.build_nohost();
    EnqueueMeshWorkload(*I.cq, I.wl_diff_v,   false);   // b_s = v_kp1 - v_k
    EnqueueMeshWorkload(*I.cq, I.wl_diff_g,   false);   // b_y = g_kp1 - g_k
    EnqueueMeshWorkload(*I.cq, I.wl_stepsize, false);   // partials of s·s, s·y, y·y
    Finish(*I.cq);
    std::vector<float> partials((size_t)I.nc_all * 3u * TILE_ELEMS, 0.0f);
    EnqueueReadMeshBuffer(*I.cq, partials, I.b_ssout, true);
    double a0 = 0.0, a2 = 0.0;   // a0 = Σ dv·dv, a2 = Σ dg·dg
    for (int c = 0; c < I.nc_all; ++c) {
        if (I.ss_core_nt[c] == 0) continue;
        const size_t base = (size_t)c * 3u * TILE_ELEMS;
        for (uint32_t i = 0; i < TILE_ELEMS; ++i) {
            a0 += partials[base + 0 * TILE_ELEMS + i];
            a2 += partials[base + 2 * TILE_ELEMS + i];
        }
    }
    if (ss) *ss = a0;
    if (yy) *yy = a2;
}

void V22OptEngine::commit_trial() {
    auto& I = *impl_; I.build_nohost();
    EnqueueMeshWorkload(*I.cq, I.wl_commit_v, false);   // b_v <- b_vclamp
    EnqueueMeshWorkload(*I.cq, I.wl_commit_u, false);   // b_uprev <- b_unew
    Finish(*I.cq);
}

void V22OptEngine::get_trial_pos(float* out) {
    impl_->download(impl_->b_vclamp, out);
}

// ── On-device density_grad: the on-chip unsort writes b_dg directly ──
uint32_t V22OptEngine::density_grad_address() const {
    return (uint32_t)impl_->b_dg->address();
}
uint32_t V22OptEngine::density_grad_num_tiles() const {
    return impl_->n_tiles_total;
}

void V22OptEngine::step_resident_ondevice_grad(const float* wl_grad,
                                               float step_size, float coef, float density_weight, float precond_alpha,
                                               OptTiming* timing_out) {
    auto& I = *impl_;
    I.build_all_resident();
    auto t0 = hrclock::now();
    I.upload(I.b_wl, wl_grad);            // wl_grad still host-supplied; density_grad already in b_dg (on-chip unsort)
    I.upload_scalar(I.s_dw,    density_weight);
    I.upload_scalar(I.s_adw,   precond_alpha * density_weight);
    I.upload_scalar(I.s_alpha, step_size);
    I.upload_scalar(I.s_coef,  coef);
    Finish(*I.cq);
    double h2d = ms_since(t0);

    auto t1 = hrclock::now();
    EnqueueMeshWorkload(*I.cq, I.wl_combine,      false);
    EnqueueMeshWorkload(*I.cq, I.wl_precond,      false);
    EnqueueMeshWorkload(*I.cq, I.wl_nesterov_res, false);  // u in-place → b_uprev
    EnqueueMeshWorkload(*I.cq, I.wl_clamp_res,    false);  // pos in-place → b_v
    Finish(*I.cq);
    double comp = ms_since(t1);

    if (timing_out) {
        timing_out->h2d_ms = h2d;
        timing_out->compute_ms = comp;
        timing_out->d2h_ms = 0.0;
        timing_out->total_ms = ms_since(t0);
    }
}

}  // namespace v22
