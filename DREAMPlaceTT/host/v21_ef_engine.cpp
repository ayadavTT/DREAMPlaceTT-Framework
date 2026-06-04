// SPDX-License-Identifier: Apache-2.0
//
// V21 electric_force on-chip engine. See v21_ef_engine.h.
// Adapted from host/v21_electric_force_microbench_host.cpp:
//   - Uses v5 face-merge kernels (best correct config from session 2026-05-27).
//   - Reuses a caller-provided MeshDevice (no second hardware open).

#include "v21_ef_engine.h"

#include <cstdio>
#include <cstring>
#include <chrono>
#include <map>
#include <string>
#include <stdexcept>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tt_metal.hpp>

namespace v21ef {

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

struct V21EFEngine::Impl {
    MeshDevice* mesh_device = nullptr;
    MeshCommandQueue* cq = nullptr;

    int M = 0, N = 0;
    int num_nodes_max = 0;
    int num_nodes_configured_ = 0;
    float xl = 0.f, yl = 0.f, bsx = 1.f, bsy = 1.f;
    float inv_bsx = 1.f, inv_bsy = 1.f;

    // Geometry derived from grid
    uint32_t field_pg_bytes = 0;
    uint32_t field_total_pages = 0;
    uint32_t pos_pages = 0;
    uint32_t cpc = 0;             // cells per core (padded)
    uint32_t const_pg_bytes = 0;
    uint32_t grad_pg_bytes = 0;
    int nc_all = 0;
    CoreCoord grid;
    CoreRangeSet all_crs;
    std::vector<MeshCoordinateRange> all_ccs_unused;  // we use a single MeshCoordinateRange

    static constexpr uint32_t POS_PG_BYTES   = 4096u;
    // V23: full-tile 1024-cell batches (vs v5's 512).
    static constexpr uint32_t V21S_BATCH     = 1024u;
    // V23 adaptive bin-span bounds (computed from clamped sizes at configure;
    // stable across iters since sizes are fixed). Default 8 = V21 behaviour.
    uint32_t max_k_ = 8u, max_h_ = 8u;
    // Empirical limit: > 9 batches per launch silently corrupts V21 EF output
    // (documented in the microbench handoff). a1 designs have cpc=7 batches so
    // 16 "worked", but a2/a3/bb1/bb3 have 11-12 batches → must chunk at 8.
    static constexpr uint32_t MAX_BATCHES_PER_LAUNCH = 8u;
    uint32_t pos_floats_per_pg = POS_PG_BYTES / 4u;

    // DRAM buffers
    std::shared_ptr<MeshBuffer> field_x_buf, field_y_buf;
    std::shared_ptr<MeshBuffer> pos_buf, const_buf, grad_buf;

    // Cached programs (lazy-built on first compute)
    bool prog_built = false;
    MeshWorkload wl;
    MeshCoordinateRange device_range = MeshCoordinateRange{MeshCoordinate{0, 0}, MeshCoordinate{0, 0}};
    uint32_t sem_brisc_done_id = 0, sem_ncrisc_done_id = 0;
    uint32_t sem_brisc_fxy_id  = 0, sem_ncrisc_fxy_id  = 0;
    KernelHandle br_handle = 0, nc_handle = 1, compute_handle = 2;

    // Staging vectors (pre-allocated to avoid per-iter alloc)
    std::vector<float> field_x_padded, field_y_padded;
    std::vector<float> pos_staging;
    std::vector<uint32_t> const_pages;
    std::vector<float> grad_pages;
    std::vector<int32_t> all_ccs_indices;  // logical core indices

    // Stored selection: sel[i] is the index in the full DREAMPlace pos/const
    // array for the i-th cell that V21 EF processes (movable[0..nm) ∪
    // filler[num_phys..num_phys+nf)). Used by compute_from_full().
    std::vector<int32_t> sel_;
    int num_total_nodes_ = 0;

    Impl(void* mesh_device_ptr_,
         int M_, int N_, int num_nodes_max_,
         float xl_, float yl_, float bsx_, float bsy_)
        : mesh_device(static_cast<MeshDevice*>(mesh_device_ptr_)),
          M(M_), N(N_), num_nodes_max(num_nodes_max_),
          xl(xl_), yl(yl_), bsx(bsx_), bsy(bsy_)
    {
        if (!mesh_device) {
            throw std::runtime_error("v21ef: null mesh_device");
        }
        cq = &mesh_device->mesh_command_queue();
        inv_bsx = 1.0f / bsx;
        inv_bsy = 1.0f / bsy;

        grid = mesh_device->compute_with_storage_grid_size();
        all_crs = CoreRangeSet(CoreRange(CoreCoord(0, 0),
                                         CoreCoord(grid.x - 1, grid.y - 1)));
        nc_all = (int)(grid.x * grid.y);

        // field page: 1 row per page (N floats × 4 B)
        field_pg_bytes = (uint32_t)N * 4u;
        if (field_pg_bytes % 64u != 0u) {
            // Blackhole DRAM-read alignment = 64 B. Pad to 64.
            field_pg_bytes = (field_pg_bytes + 63u) & ~63u;
        }
        field_total_pages = (uint32_t)M;

        // Pos pages
        const uint32_t pos_total = (uint32_t)(num_nodes_max * 2);
        pos_pages = (pos_total + pos_floats_per_pg - 1u) / pos_floats_per_pg;

        // Cells per core (padded to V21S_BATCH)
        const uint32_t PAD_GRAN = (uint32_t)nc_all * V21S_BATCH;
        const uint32_t num_nodes_padded = (((uint32_t)num_nodes_max + PAD_GRAN - 1u) / PAD_GRAN) * PAD_GRAN;
        cpc = num_nodes_padded / (uint32_t)nc_all;
        if ((cpc % V21S_BATCH) != 0u) {
            throw std::runtime_error("v21ef: cpc not multiple of V21S_BATCH");
        }
        const_pg_bytes = cpc * 32u;
        grad_pg_bytes  = cpc * 8u;

        // Allocate DRAM buffers (one-time)
        auto make_buf = [&](uint64_t total_bytes, uint32_t pgsz) {
            DeviceLocalBufferConfig cfg{.page_size = pgsz, .buffer_type = BufferType::DRAM};
            ReplicatedBufferConfig rcfg{.size = total_bytes};
            return MeshBuffer::create(rcfg, cfg, mesh_device);
        };
        field_x_buf = make_buf((uint64_t)field_total_pages * field_pg_bytes, field_pg_bytes);
        field_y_buf = make_buf((uint64_t)field_total_pages * field_pg_bytes, field_pg_bytes);
        pos_buf     = make_buf((uint64_t)pos_pages * POS_PG_BYTES, POS_PG_BYTES);
        const_buf   = make_buf((uint64_t)nc_all * const_pg_bytes, const_pg_bytes);
        grad_buf    = make_buf((uint64_t)nc_all * grad_pg_bytes, grad_pg_bytes);

        // Pre-allocate host staging
        field_x_padded.assign((size_t)field_total_pages * (field_pg_bytes / 4u), 0.0f);
        field_y_padded.assign((size_t)field_total_pages * (field_pg_bytes / 4u), 0.0f);
        pos_staging.assign((size_t)pos_pages * pos_floats_per_pg, 0.0f);
        const_pages.assign((size_t)nc_all * (const_pg_bytes / 4u), 0u);
        grad_pages.assign((size_t)nc_all * (grad_pg_bytes / 4u), 0.0f);

        std::printf("[v21ef_engine] M=%d N=%d nc_all=%d cpc=%u  pos_pages=%u  field_pg=%u B\n",
                    M, N, nc_all, cpc, pos_pages, field_pg_bytes);
        std::printf("[v21ef_engine] DRAM: field=2× %.1f MB, pos=%.1f MB, const=%.1f MB, grad=%.1f MB\n",
                    (double)field_total_pages * field_pg_bytes / 1048576.0,
                    (double)pos_pages * POS_PG_BYTES / 1048576.0,
                    (double)nc_all * const_pg_bytes / 1048576.0,
                    (double)nc_all * grad_pg_bytes / 1048576.0);
        std::fflush(stdout);
    }

    // Build the V21 EF v5 face-merge program for `cells_end - cells_start` cells per core.
    Program build_program(uint32_t cells_start, uint32_t cells_end) {
        Program prog = CreateProgram();
        const uint32_t cb24_size = 512u * 1024u;  // V23 batch=1024 per-cell staging
        const uint32_t cb25_size = 16u * 1024u;
        {
            CircularBufferConfig cb24_cfg(cb24_size, {{CBIndex::c_24, DataFormat::Float32}});
            cb24_cfg.set_page_size(CBIndex::c_24, cb24_size);
            CreateCircularBuffer(prog, all_crs, cb24_cfg);
            CircularBufferConfig cb25_cfg(cb25_size, {{CBIndex::c_25, DataFormat::Float32}});
            cb25_cfg.set_page_size(CBIndex::c_25, cb25_size);
            CreateCircularBuffer(prog, all_crs, cb25_cfg);
        }

        constexpr uint32_t TILE_BYTES = 32u * 32u * 4u;
        auto make_cb = [&](uint32_t idx, uint32_t n_slots) {
            CircularBufferConfig cfg(n_slots * TILE_BYTES, {{idx, DataFormat::Float32}});
            cfg.set_page_size(idx, TILE_BYTES);
            CreateCircularBuffer(prog, all_crs, cfg);
        };
        // V23 (non-v5) CB layout: separate cb_fx (c_2) + cb_fy (c_3), adaptive slots.
        make_cb((uint32_t)CBIndex::c_0,  max_k_);          // cb_px (one per k)
        make_cb((uint32_t)CBIndex::c_1,  max_h_);          // cb_py (one per h)
        make_cb((uint32_t)CBIndex::c_2,  max_h_);          // cb_fx — streamed (1 group)
        make_cb((uint32_t)CBIndex::c_3,  max_k_ * max_h_); // cb_fy — full run-ahead
        make_cb((uint32_t)CBIndex::c_4,  4);   // cb_neg_ratio
        make_cb((uint32_t)CBIndex::c_5,  4);   // cb_acc
        make_cb((uint32_t)CBIndex::c_16, 4);   // cb_gx_out
        make_cb((uint32_t)CBIndex::c_17, 4);   // cb_gy_out

        std::map<std::string, std::string> dm_defines;
        std::map<std::string, std::string> compute_defines;
        dm_defines["V21_USE_SFPU"] = "1";
        compute_defines["V21_USE_SFPU"] = "1";
        // V23 defines (full-tile + adaptive bounds + boundary-only prep).
        dm_defines["V21S_BATCH_OVERRIDE"]  = std::to_string(V21S_BATCH);
        dm_defines["V21S_MAX_K"]           = std::to_string(max_k_);
        dm_defines["V21S_MAX_H"]           = std::to_string(max_h_);
        dm_defines["V21_PREP_FAST"]        = "1";
        compute_defines["V21S_MAX_K"]      = std::to_string(max_k_);
        compute_defines["V21S_MAX_H"]      = std::to_string(max_h_);

        auto br_k = CreateKernel(prog,
            std::string(DENSITY_KERNEL_DIR) + "v21_ef_brisc.cpp",
            all_crs,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                               .noc = NOC::RISCV_0_default,
                               .defines = dm_defines});
        auto nc_k = CreateKernel(prog,
            std::string(DENSITY_KERNEL_DIR) + "v21_ef_ncrisc.cpp",
            all_crs,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                               .noc = NOC::RISCV_1_default,
                               .defines = dm_defines});

        sem_brisc_done_id  = CreateSemaphore(prog, all_crs, 0u);
        sem_ncrisc_done_id = CreateSemaphore(prog, all_crs, 0u);

        std::vector<UnpackToDestMode> unpack_modes(64, UnpackToDestMode::Default);
        unpack_modes[(uint32_t)CBIndex::c_0] = UnpackToDestMode::UnpackToDestFp32;
        unpack_modes[(uint32_t)CBIndex::c_1] = UnpackToDestMode::UnpackToDestFp32;
        unpack_modes[(uint32_t)CBIndex::c_2] = UnpackToDestMode::UnpackToDestFp32;
        unpack_modes[(uint32_t)CBIndex::c_3] = UnpackToDestMode::UnpackToDestFp32;
        unpack_modes[(uint32_t)CBIndex::c_4] = UnpackToDestMode::UnpackToDestFp32;
        unpack_modes[(uint32_t)CBIndex::c_5] = UnpackToDestMode::UnpackToDestFp32;

        auto compute_k = CreateKernel(prog,
            std::string(DENSITY_KERNEL_DIR) + "v21_ef_compute.cpp",
            all_crs,
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = true,
                .unpack_to_dest_mode = unpack_modes,
                .math_approx_mode = false,
                .defines = compute_defines,
            });

        // Runtime args per core. Kernels were created in BRISC, NCRISC, COMPUTE
        // order → handles 0, 1, 2.
        br_handle = br_k;
        nc_handle = nc_k;
        compute_handle = compute_k;

        const uint32_t field_x_base = (uint32_t)field_x_buf->address();
        const uint32_t field_y_base = (uint32_t)field_y_buf->address();
        const uint32_t pos_base     = (uint32_t)pos_buf->address();
        const uint32_t const_base   = (uint32_t)const_buf->address();
        const uint32_t grad_base    = (uint32_t)grad_buf->address();

        union { uint32_t u; float f; } cvt;
        auto f2u = [&](float fv) { cvt.f = fv; return cvt.u; };

        for (int c = 0; c < nc_all; ++c) {
            const uint32_t cells_start_global = (uint32_t)c * cpc;
            std::vector<uint32_t> common_tail = {
                cells_start_global,
                (uint32_t)num_nodes_configured_,
                (uint32_t)M, (uint32_t)N,
                field_x_base, field_y_base, field_pg_bytes,
                pos_base,
                const_base, const_pg_bytes,
                grad_base, grad_pg_bytes,
                f2u(xl), f2u(yl), f2u(bsx), f2u(bsy), f2u(inv_bsx), f2u(inv_bsy),
            };
            std::vector<uint32_t> br_args = {(uint32_t)c, cells_start, cells_end};
            std::vector<uint32_t> nc_args = {(uint32_t)c, cells_start, cells_end};
            br_args.insert(br_args.end(), common_tail.begin(), common_tail.end());
            nc_args.insert(nc_args.end(), common_tail.begin(), common_tail.end());
            br_args.push_back(sem_brisc_done_id);
            br_args.push_back(sem_ncrisc_done_id);
            nc_args.push_back(sem_brisc_done_id);
            nc_args.push_back(sem_ncrisc_done_id);

            CoreCoord cc{(uint32_t)(c % grid.x), (uint32_t)(c / grid.x)};
            SetRuntimeArgs(prog, br_k, cc, br_args);
            SetRuntimeArgs(prog, nc_k, cc, nc_args);
            const uint32_t n_batches = (cells_end - cells_start) / V21S_BATCH;
            SetRuntimeArgs(prog, compute_k, cc, {n_batches});
        }
        return prog;
    }

    // Override field addresses for the zero-copy chip-field path.
    // 0 means use the engine's own field_x_buf / field_y_buf.
    uint32_t override_fx_addr = 0;
    uint32_t override_fy_addr = 0;

    void update_args(uint32_t cells_start, uint32_t cells_end) {
        Program& prog_ref = wl.get_programs().at(device_range);
        const uint32_t field_x_base = override_fx_addr
            ? override_fx_addr : (uint32_t)field_x_buf->address();
        const uint32_t field_y_base = override_fy_addr
            ? override_fy_addr : (uint32_t)field_y_buf->address();
        const uint32_t pos_base     = (uint32_t)pos_buf->address();
        const uint32_t const_base   = (uint32_t)const_buf->address();
        const uint32_t grad_base    = (uint32_t)grad_buf->address();
        union { uint32_t u; float f; } cvt;
        auto f2u = [&](float fv) { cvt.f = fv; return cvt.u; };

        for (int c = 0; c < nc_all; ++c) {
            const uint32_t cells_start_global = (uint32_t)c * cpc;
            std::vector<uint32_t> common_tail = {
                cells_start_global,
                (uint32_t)num_nodes_configured_,
                (uint32_t)M, (uint32_t)N,
                field_x_base, field_y_base, field_pg_bytes,
                pos_base, const_base, const_pg_bytes,
                grad_base, grad_pg_bytes,
                f2u(xl), f2u(yl), f2u(bsx), f2u(bsy), f2u(inv_bsx), f2u(inv_bsy),
            };
            std::vector<uint32_t> br_args = {(uint32_t)c, cells_start, cells_end};
            std::vector<uint32_t> nc_args = {(uint32_t)c, cells_start, cells_end};
            br_args.insert(br_args.end(), common_tail.begin(), common_tail.end());
            nc_args.insert(nc_args.end(), common_tail.begin(), common_tail.end());
            br_args.push_back(sem_brisc_done_id);
            br_args.push_back(sem_ncrisc_done_id);
            nc_args.push_back(sem_brisc_done_id);
            nc_args.push_back(sem_ncrisc_done_id);

            CoreCoord cc{(uint32_t)(c % grid.x), (uint32_t)(c / grid.x)};
            SetRuntimeArgs(prog_ref, br_handle, cc, br_args);
            SetRuntimeArgs(prog_ref, nc_handle, cc, nc_args);
            const uint32_t n_batches = (cells_end - cells_start) / V21S_BATCH;
            SetRuntimeArgs(prog_ref, compute_handle, cc, {n_batches});
        }
    }
};

V21EFEngine::V21EFEngine(void* mesh_device,
                         int M, int N, int num_nodes_max,
                         float xl, float yl, float bsx, float bsy)
    : impl_(std::make_unique<Impl>(mesh_device, M, N, num_nodes_max, xl, yl, bsx, bsy)) {}

void V21EFEngine::prewarm() {
    if (impl_->prog_built) return;
    impl_->device_range = MeshCoordinateRange{impl_->mesh_device->shape()};
    const uint32_t chunk_cells = Impl::MAX_BATCHES_PER_LAUNCH * Impl::V21S_BATCH;
    impl_->wl.add_program(impl_->device_range,
                          impl_->build_program(0u, std::min(chunk_cells, impl_->cpc)));
    impl_->prog_built = true;
    // Dummy launch to JIT-compile kernels eagerly. Buffers may contain
    // uninitialized data — that's fine; we discard the output.
    auto t0 = hrclock::now();
    EnqueueMeshWorkload(*impl_->cq, impl_->wl, false);
    Finish(*impl_->cq);
    std::printf("[v21ef_engine] prewarm done in %.1f ms\n", ms_since(t0));
    std::fflush(stdout);
}

void V21EFEngine::configure_with_sel(
    const float* ox_full, const float* oy_full,
    const float* nsx_full, const float* nsy_full,
    const float* ratio_full,
    const int32_t* sel, int num_active,
    int num_total_nodes) {
    if (num_active > impl_->num_nodes_max) {
        throw std::invalid_argument(
            "configure_with_sel: num_active > num_nodes_max");
    }
    impl_->num_nodes_configured_ = num_active;
    impl_->num_total_nodes_ = num_total_nodes;
    impl_->sel_.assign(sel, sel + num_active);

    // V23 adaptive bounds: max possible bin span = floor(size/bs)+2 (covers worst
    // alignment), clamped to [1,8]. Pos-independent (cancels in the span), so stable.
    {
        uint32_t mk = 1u, mh = 1u;
        for (int i = 0; i < num_active; ++i) {
            const int32_t g = sel[i];
            int kc = (int)(nsx_full[g] * impl_->inv_bsx) + 2; if (kc < 1) kc = 1; if (kc > 8) kc = 8;
            int hc = (int)(nsy_full[g] * impl_->inv_bsy) + 2; if (hc < 1) hc = 1; if (hc > 8) hc = 8;
            if ((uint32_t)kc > mk) mk = (uint32_t)kc;
            if ((uint32_t)hc > mh) mh = (uint32_t)hc;
        }
        impl_->max_k_ = mk; impl_->max_h_ = mh;
        std::printf("[v21ef_engine] V23 adaptive: max_k=%u max_h=%u (work %u/64)\n",
                    mk, mh, mk * mh);
        std::fflush(stdout);
    }

    // Pack const_pages using the sel-gathered values.
    std::fill(impl_->const_pages.begin(), impl_->const_pages.end(), 0u);
    const uint32_t cpc = impl_->cpc;
    for (int i = 0; i < num_active; ++i) {
        const int32_t g_full = sel[i];
        const uint32_t core = (uint32_t)i / cpc;
        const uint32_t intra = (uint32_t)i % cpc;
        const size_t base = (size_t)core * (impl_->const_pg_bytes / 4u) + intra * 8u;
        std::memcpy(&impl_->const_pages[base + 0], &ox_full[g_full],    4);
        std::memcpy(&impl_->const_pages[base + 1], &oy_full[g_full],    4);
        std::memcpy(&impl_->const_pages[base + 2], &nsx_full[g_full],   4);
        std::memcpy(&impl_->const_pages[base + 3], &nsy_full[g_full],   4);
        std::memcpy(&impl_->const_pages[base + 4], &ratio_full[g_full], 4);
    }
    EnqueueWriteMeshBuffer(*impl_->cq, impl_->const_buf, impl_->const_pages, false);
    Finish(*impl_->cq);
}

void V21EFEngine::compute_from_full(
    const float* pos_full,
    const float* fx_full, const float* fy_full,
    float* grad_full,
    EFTiming* timing_out) {
    if (impl_->sel_.empty()) {
        throw std::runtime_error(
            "compute_from_full: configure_with_sel() not called");
    }
    auto t_total = hrclock::now();
    EFTiming t{};
    const int num_active = impl_->num_nodes_configured_;
    const int nn = impl_->num_total_nodes_;
    const int M = impl_->M, N = impl_->N;
    const uint32_t cpc = impl_->cpc;

    // ── Gather pos: pos_staging[i] = pos_full[sel[i]] (x),
    //               pos_staging[num_active + i] = pos_full[nn + sel[i]] (y).
    //    Tail padding is left at zero (V21 cell-loop guards skip invalid cells).
    std::fill(impl_->pos_staging.begin(), impl_->pos_staging.end(), 0.0f);
    for (int i = 0; i < num_active; ++i) {
        const int32_t g = impl_->sel_[i];
        impl_->pos_staging[i]              = pos_full[g];
        impl_->pos_staging[num_active + i] = pos_full[nn + g];
    }

    // ── Pack field_x / field_y into staging (row-pad if field_pg_bytes != N*4) ──
    const uint32_t row_floats = impl_->field_pg_bytes / 4u;
    if ((uint32_t)N == row_floats) {
        std::memcpy(impl_->field_x_padded.data(), fx_full, sizeof(float) * M * N);
        std::memcpy(impl_->field_y_padded.data(), fy_full, sizeof(float) * M * N);
    } else {
        for (int r = 0; r < M; ++r) {
            std::memcpy(impl_->field_x_padded.data() + (size_t)r * row_floats,
                        fx_full + (size_t)r * N, sizeof(float) * N);
            std::memcpy(impl_->field_y_padded.data() + (size_t)r * row_floats,
                        fy_full + (size_t)r * N, sizeof(float) * N);
        }
    }

    // ── H2D ──
    auto t_h2d = hrclock::now();
    EnqueueWriteMeshBuffer(*impl_->cq, impl_->field_x_buf, impl_->field_x_padded, false);
    EnqueueWriteMeshBuffer(*impl_->cq, impl_->field_y_buf, impl_->field_y_padded, false);
    EnqueueWriteMeshBuffer(*impl_->cq, impl_->pos_buf,     impl_->pos_staging,    false);
    Finish(*impl_->cq);
    t.h2d_field_ms = ms_since(t_h2d);
    t.h2d_pos_ms = 0.0;  // bundled into h2d_field_ms

    // ── Lazy build program if prewarm() wasn't called (should be no-op normally) ──
    if (!impl_->prog_built) {
        impl_->device_range = MeshCoordinateRange{impl_->mesh_device->shape()};
        const uint32_t chunk_cells = Impl::MAX_BATCHES_PER_LAUNCH * Impl::V21S_BATCH;
        impl_->wl.add_program(impl_->device_range,
                              impl_->build_program(0u, std::min(chunk_cells, cpc)));
        impl_->prog_built = true;
        EnqueueMeshWorkload(*impl_->cq, impl_->wl, false);
        Finish(*impl_->cq);
    }

    // ── Run all chunks ──
    auto t_compute = hrclock::now();
    const uint32_t chunk_cells = Impl::MAX_BATCHES_PER_LAUNCH * Impl::V21S_BATCH;
    const uint32_t n_chunks = (cpc + chunk_cells - 1u) / chunk_cells;
    for (uint32_t chunk = 0; chunk < n_chunks; ++chunk) {
        const uint32_t cs = chunk * chunk_cells;
        const uint32_t ce = std::min(cs + chunk_cells, cpc);
        impl_->update_args(cs, ce);
        EnqueueMeshWorkload(*impl_->cq, impl_->wl, false);
        Finish(*impl_->cq);
    }
    t.launches = n_chunks;
    t.compute_ms = ms_since(t_compute);

    // ── D2H + scatter back to full layout ──
    auto t_d2h = hrclock::now();
    EnqueueReadMeshBuffer(*impl_->cq, impl_->grad_pages, impl_->grad_buf, true);
    t.d2h_grad_ms = ms_since(t_d2h);

    // Scatter grad_pages → grad_full. The V21 kernel ALREADY bakes in the
    // negation (gxy_out = acc * neg_ratio, neg_ratio = -ratio), so grad_pages
    // is already the correct -force gradient (microbench-validated bit-exact
    // vs CPU's `-electric_force`). Do NOT negate again here — doing so flips
    // the density gradient sign and prevents convergence (root cause of the
    // adaptec2/3, bigblue1/3 divergence). Copy through verbatim.
    const uint32_t grad_floats_per_page = impl_->grad_pg_bytes / 4u;
    for (int i = 0; i < num_active; ++i) {
        const uint32_t core = (uint32_t)i / cpc;
        const uint32_t intra = (uint32_t)i % cpc;
        const size_t base = (size_t)core * grad_floats_per_page + (size_t)intra * 2u;
        const int32_t g = impl_->sel_[i];
        grad_full[g]      = impl_->grad_pages[base + 0];
        grad_full[nn + g] = impl_->grad_pages[base + 1];
    }

    t.total_ms = ms_since(t_total);
    if (timing_out) *timing_out = t;
}

V21EFEngine::~V21EFEngine() = default;

int V21EFEngine::num_nodes_configured() const noexcept {
    return impl_->num_nodes_configured_;
}

void V21EFEngine::configure_constants(const float* ox, const float* oy,
                                       const float* nsx, const float* nsy,
                                       const float* ratio,
                                       int num_nodes) {
    if (num_nodes > impl_->num_nodes_max) {
        throw std::invalid_argument(
            "v21ef configure_constants: num_nodes > num_nodes_max");
    }
    impl_->num_nodes_configured_ = num_nodes;

    // Pack const_pages: per-core page of cpc cells × 32 B. Slots after num_nodes
    // remain zero so V21's bin loop produces gx=gy=0 (no contribution).
    std::fill(impl_->const_pages.begin(), impl_->const_pages.end(), 0u);
    const uint32_t cpc = impl_->cpc;
    for (int g = 0; g < num_nodes; ++g) {
        const uint32_t core = (uint32_t)g / cpc;
        const uint32_t intra = (uint32_t)g % cpc;
        const size_t base = (size_t)core * (impl_->const_pg_bytes / 4u) + intra * 8u;
        std::memcpy(&impl_->const_pages[base + 0], &ox[g],    4);
        std::memcpy(&impl_->const_pages[base + 1], &oy[g],    4);
        std::memcpy(&impl_->const_pages[base + 2], &nsx[g],   4);
        std::memcpy(&impl_->const_pages[base + 3], &nsy[g],   4);
        std::memcpy(&impl_->const_pages[base + 4], &ratio[g], 4);
    }
    // Upload const_buf once.
    EnqueueWriteMeshBuffer(*impl_->cq, impl_->const_buf, impl_->const_pages, false);
    Finish(*impl_->cq);
}

void V21EFEngine::compute(const float* pos,
                           const float* field_x, const float* field_y,
                           float* grad_x, float* grad_y,
                           EFTiming* timing_out) {
    if (impl_->num_nodes_configured_ == 0) {
        throw std::runtime_error("v21ef compute: configure_constants() not called");
    }
    auto t_total = hrclock::now();
    EFTiming t{};

    const int M = impl_->M, N = impl_->N;
    const int num_nodes = impl_->num_nodes_configured_;
    const uint32_t cpc = impl_->cpc;
    const uint32_t total_cells = (uint32_t)impl_->nc_all * cpc;

    // ── Pack pos into staging (pos layout: [x0..x_{N-1} | y0..y_{N-1}], padded) ──
    {
        std::fill(impl_->pos_staging.begin(), impl_->pos_staging.end(), 0.0f);
        std::memcpy(impl_->pos_staging.data(), pos, sizeof(float) * num_nodes);              // x slice
        std::memcpy(impl_->pos_staging.data() + num_nodes, pos + num_nodes,
                    sizeof(float) * num_nodes);                                              // y slice
    }

    // ── Pack field_x / field_y into staging (pad rows to field_pg_bytes/4 floats) ──
    const uint32_t row_floats = impl_->field_pg_bytes / 4u;
    if ((uint32_t)N == row_floats) {
        std::memcpy(impl_->field_x_padded.data(), field_x, sizeof(float) * M * N);
        std::memcpy(impl_->field_y_padded.data(), field_y, sizeof(float) * M * N);
    } else {
        std::fill(impl_->field_x_padded.begin(), impl_->field_x_padded.end(), 0.0f);
        std::fill(impl_->field_y_padded.begin(), impl_->field_y_padded.end(), 0.0f);
        for (int r = 0; r < M; ++r) {
            std::memcpy(impl_->field_x_padded.data() + (size_t)r * row_floats,
                        field_x + (size_t)r * N, sizeof(float) * N);
            std::memcpy(impl_->field_y_padded.data() + (size_t)r * row_floats,
                        field_y + (size_t)r * N, sizeof(float) * N);
        }
    }

    // ── H2D uploads ──
    auto t_h2d = hrclock::now();
    EnqueueWriteMeshBuffer(*impl_->cq, impl_->field_x_buf, impl_->field_x_padded, false);
    EnqueueWriteMeshBuffer(*impl_->cq, impl_->field_y_buf, impl_->field_y_padded, false);
    Finish(*impl_->cq);
    t.h2d_field_ms = ms_since(t_h2d);

    t_h2d = hrclock::now();
    EnqueueWriteMeshBuffer(*impl_->cq, impl_->pos_buf, impl_->pos_staging, false);
    Finish(*impl_->cq);
    t.h2d_pos_ms = ms_since(t_h2d);

    // ── Lazy build program on first call ──
    const uint32_t chunk_cells = Impl::MAX_BATCHES_PER_LAUNCH * Impl::V21S_BATCH;
    const uint32_t n_chunks = (cpc + chunk_cells - 1u) / chunk_cells;
    if (!impl_->prog_built) {
        impl_->device_range = MeshCoordinateRange{impl_->mesh_device->shape()};
        impl_->wl.add_program(impl_->device_range,
                              impl_->build_program(0u, std::min(chunk_cells, cpc)));
        impl_->prog_built = true;
        // Warm-up: empty enqueue to JIT-compile kernels (large cost is here).
        EnqueueMeshWorkload(*impl_->cq, impl_->wl, false);
        Finish(*impl_->cq);
    }

    // ── Run all chunks ──
    auto t_compute = hrclock::now();
    for (uint32_t chunk = 0; chunk < n_chunks; ++chunk) {
        const uint32_t cs = chunk * chunk_cells;
        const uint32_t ce = std::min(cs + chunk_cells, cpc);
        impl_->update_args(cs, ce);
        EnqueueMeshWorkload(*impl_->cq, impl_->wl, false);
        Finish(*impl_->cq);
    }
    t.launches = n_chunks;
    t.compute_ms = ms_since(t_compute);

    // ── D2H grad readback ──
    auto t_d2h = hrclock::now();
    EnqueueReadMeshBuffer(*impl_->cq, impl_->grad_pages, impl_->grad_buf, true);
    t.d2h_grad_ms = ms_since(t_d2h);

    // ── Unpack grad_pages → grad_x[], grad_y[]. Per core, the page has
    //    pairs (gx0, gy0, gx1, gy1, ...). ──
    const uint32_t grad_floats_per_page = impl_->grad_pg_bytes / 4u;
    for (int g = 0; g < num_nodes; ++g) {
        const uint32_t core = (uint32_t)g / cpc;
        const uint32_t intra = (uint32_t)g % cpc;
        const size_t base = (size_t)core * grad_floats_per_page + (size_t)intra * 2u;
        grad_x[g] = impl_->grad_pages[base + 0];
        grad_y[g] = impl_->grad_pages[base + 1];
    }
    (void)total_cells;

    t.total_ms = ms_since(t_total);
    if (timing_out) *timing_out = t;
}

void V21EFEngine::compute_from_full_chip_fields(
    const float* pos_full,
    uint32_t fx_chip_addr,
    uint32_t fy_chip_addr,
    float* grad_full,
    EFTiming* timing_out) {
    if (impl_->sel_.empty()) {
        throw std::runtime_error(
            "compute_from_full_chip_fields: configure_with_sel() not called");
    }
    if (fx_chip_addr == 0 || fy_chip_addr == 0) {
        throw std::invalid_argument(
            "compute_from_full_chip_fields: field addresses must be non-zero");
    }
    auto t_total = hrclock::now();
    EFTiming t{};
    const int num_active = impl_->num_nodes_configured_;
    const int nn = impl_->num_total_nodes_;
    const uint32_t cpc = impl_->cpc;

    // Pos gather (same as compute_from_full).
    std::fill(impl_->pos_staging.begin(), impl_->pos_staging.end(), 0.0f);
    for (int i = 0; i < num_active; ++i) {
        const int32_t g = impl_->sel_[i];
        impl_->pos_staging[i]              = pos_full[g];
        impl_->pos_staging[num_active + i] = pos_full[nn + g];
    }

    // ── H2D pos only (no field upload — that's the zero-copy win) ──
    auto t_h2d = hrclock::now();
    EnqueueWriteMeshBuffer(*impl_->cq, impl_->pos_buf, impl_->pos_staging, false);
    Finish(*impl_->cq);
    t.h2d_pos_ms = ms_since(t_h2d);
    t.h2d_field_ms = 0.0;  // skipped

    // Lazy build / prewarm
    if (!impl_->prog_built) {
        impl_->device_range = MeshCoordinateRange{impl_->mesh_device->shape()};
        const uint32_t chunk_cells = Impl::MAX_BATCHES_PER_LAUNCH * Impl::V21S_BATCH;
        impl_->wl.add_program(impl_->device_range,
                              impl_->build_program(0u, std::min(chunk_cells, cpc)));
        impl_->prog_built = true;
        EnqueueMeshWorkload(*impl_->cq, impl_->wl, false);
        Finish(*impl_->cq);
    }

    // ── Override the field DRAM addresses for this call ──
    impl_->override_fx_addr = fx_chip_addr;
    impl_->override_fy_addr = fy_chip_addr;

    auto t_compute = hrclock::now();
    const uint32_t chunk_cells = Impl::MAX_BATCHES_PER_LAUNCH * Impl::V21S_BATCH;
    const uint32_t n_chunks = (cpc + chunk_cells - 1u) / chunk_cells;
    for (uint32_t chunk = 0; chunk < n_chunks; ++chunk) {
        const uint32_t cs = chunk * chunk_cells;
        const uint32_t ce = std::min(cs + chunk_cells, cpc);
        impl_->update_args(cs, ce);
        EnqueueMeshWorkload(*impl_->cq, impl_->wl, false);
        Finish(*impl_->cq);
    }
    t.launches = n_chunks;
    t.compute_ms = ms_since(t_compute);

    // Reset overrides so next compute_from_full() (non-chip) call still works.
    impl_->override_fx_addr = 0;
    impl_->override_fy_addr = 0;

    auto t_d2h = hrclock::now();
    EnqueueReadMeshBuffer(*impl_->cq, impl_->grad_pages, impl_->grad_buf, true);
    t.d2h_grad_ms = ms_since(t_d2h);

    // grad_pages already carries the correct -force sign (kernel negates via
    // neg_ratio). Copy verbatim — do NOT double-negate. See compute_from_full.
    const uint32_t grad_floats_per_page = impl_->grad_pg_bytes / 4u;
    for (int i = 0; i < num_active; ++i) {
        const uint32_t core = (uint32_t)i / cpc;
        const uint32_t intra = (uint32_t)i % cpc;
        const size_t base = (size_t)core * grad_floats_per_page + (size_t)intra * 2u;
        const int32_t g = impl_->sel_[i];
        grad_full[g]      = impl_->grad_pages[base + 0];
        grad_full[nn + g] = impl_->grad_pages[base + 1];
    }

    t.total_ms = ms_since(t_total);
    if (timing_out) *timing_out = t;
}

}  // namespace v21ef
