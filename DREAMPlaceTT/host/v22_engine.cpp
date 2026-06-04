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

        stage.assign(padded_elems, 0.0f);
        scalar_tile.assign(TILE_ELEMS, 0.0f);

        std::printf("[v22_engine] num_nodes=%d n_elems=%u tiles=%u nc=%d tiles/core=%u  DRAM≈%.1f MB\n",
                    nn, n_elems, n_tiles_total, nc_all, tiles_per_core,
                    13.0 * (double)io / 1048576.0);
        std::fflush(stdout);
    }

    // Upload an array (length 2*nn, may be < padded) into a buffer (zero-padded).
    void upload(std::shared_ptr<MeshBuffer>& buf, const float* src) {
        std::memcpy(stage.data(), src, (size_t)n_elems * sizeof(float));
        if (padded_elems > n_elems)
            std::memset(stage.data() + n_elems, 0, (size_t)(padded_elems - n_elems) * sizeof(float));
        EnqueueWriteMeshBuffer(*cq, buf, stage, false);
    }
    void upload_scalar(std::shared_ptr<MeshBuffer>& buf, float v) {
        std::fill(scalar_tile.begin(), scalar_tile.end(), v);
        EnqueueWriteMeshBuffer(*cq, buf, scalar_tile, false);
    }
    void download(std::shared_ptr<MeshBuffer>& buf, float* dst) {
        EnqueueReadMeshBuffer(*cq, stage, buf, true);
        std::memcpy(dst, stage.data(), (size_t)n_elems * sizeof(float));
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
    step(z.data(), z.data(), z.data(), z.data(), 0.0f, 0.0f, 0.0f, z.data(), z.data(), &t);
}

void V22OptEngine::step(const float* wl_grad, const float* density_grad,
                        const float* v_k, const float* u_prev,
                        float alpha, float coef, float density_weight,
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
    I.upload_scalar(I.s_adw,   alpha * density_weight);
    I.upload_scalar(I.s_alpha, alpha);
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
                                 float alpha, float coef, float density_weight,
                                 OptTiming* timing_out) {
    auto& I = *impl_;
    I.build_all_resident();
    auto t0 = hrclock::now();
    I.upload(I.b_wl, wl_grad);
    I.upload(I.b_dg, density_grad);
    I.upload_scalar(I.s_dw,    density_weight);
    I.upload_scalar(I.s_adw,   alpha * density_weight);
    I.upload_scalar(I.s_alpha, alpha);
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

}  // namespace v22
