// SPDX-License-Identifier: Apache-2.0
//
// V19 microbench host — sweeps grid size, cell count, and cell distribution
// to characterize V19's atomic-add scatter under controlled conditions.
//
// What it does:
//   1. Generates synthetic (bx, by, area_fixed) cell triplets in host RAM
//      per chosen distribution.
//   2. Uploads triplets to a DRAM staging buffer (paged per core).
//   3. Launches scatter program (NCRISC kernel emits one atomic per cell to
//      the owning core's L1 density slab).
//   4. Finish() — ensures all atomics across all cores have ACK'd.
//   5. Launches writeout program (BRISC streams its L1 slab to DRAM).
//   6. Reads back DRAM, de-strides, validates atomic sums against host model.
//
// Tracy zones surfaced (in profile_log_device.csv after TT_METAL_DEVICE_PROFILER=1):
//   V19MB-INIT          per-core init time (zero slab + load coords)
//   V19MB-CELLS-READ    DRAM-to-L1 cell triplet load
//   V19MB-EMIT-BATCH    1024 atomic emits per zone (back-pressure indicator)
//   V19MB-BARRIER       wait for all outgoing atomics
//   V19-WRITEOUT        DRAM streamout (mirrors production v19_writeout_dm.cpp)
//
// CLI:
//   v19_microbench_host <grid> <n_cells> <dist>
//     grid     : 512 | 1024 | 2048
//     n_cells  : total synthetic cells (split round-robin across 110 cores)
//     dist     : cluster | uniform | hotcold
//
// Use scripts/run_v19_microbench.sh for sweep.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/allocator.hpp>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

using hrclock = std::chrono::high_resolution_clock;
static double ms_since(const hrclock::time_point& t0) {
    return std::chrono::duration<double, std::milli>(hrclock::now() - t0).count();
}

// 3 u32 per cell: (bx, by, area_fixed)
static constexpr uint32_t CELL_U32 = 3u;
static constexpr uint32_t CELL_BYTES = CELL_U32 * 4u;

// Cells packed densely in DRAM, paged at this size (multiple of 32 B for noc).
// 12 KB / 12 B per cell = 1024 cells per page.
static constexpr uint32_t CELL_PAGE_BYTES = 12u * 1024u;
static constexpr uint32_t CELLS_PER_PAGE  = CELL_PAGE_BYTES / CELL_BYTES;

static constexpr uint32_t SCALE_BITS = 20u;
static constexpr float    SCALE_F    = (float)(1u << SCALE_BITS);

static void gen_cells(const std::string& dist,
                      uint32_t nbx, uint32_t nby,
                      uint32_t n_cells,
                      std::vector<uint32_t>& out_packed) {
    out_packed.assign((size_t)n_cells * CELL_U32, 0u);
    std::mt19937 rng(0xC0FFEE);  // deterministic per run
    auto pack = [&](uint32_t i, uint32_t bx, uint32_t by, float area) {
        uint32_t area_fixed = (uint32_t)(area * SCALE_F);
        if (area_fixed == 0u) area_fixed = 1u;  // ensure non-zero emit
        out_packed[i * CELL_U32 + 0] = bx;
        out_packed[i * CELL_U32 + 1] = by;
        out_packed[i * CELL_U32 + 2] = area_fixed;
    };

    if (dist == "origin") {
        // All atomics target bin (0,0) ONLY. owner_idx=0, local_idx=0 → low L1
        // address on core 0. Diagnostic for L1-address-range hypothesis.
        std::uniform_real_distribution<float> u01(0.05f, 0.45f);
        for (uint32_t i = 0; i < n_cells; ++i) {
            pack(i, 0u, 0u, u01(rng));
        }
    } else if (dist == "low_local") {
        // Atomics target bins with local_idx < 8 (low end of density slab).
        // bin_global must be in {0,1,...,7} * nc_all + owner. Spread across owners.
        std::uniform_real_distribution<float> u01(0.05f, 0.45f);
        for (uint32_t i = 0; i < n_cells; ++i) {
            uint32_t owner_idx = i % 110;
            uint32_t local_idx = (i / 110) % 8;  // local 0..7
            uint32_t bin_global = local_idx * 110u + owner_idx;
            uint32_t bx = bin_global / nby;
            uint32_t by = bin_global % nby;
            pack(i, bx, by, u01(rng));
        }
    } else if (dist == "high_local") {
        // Atomics target bins with local_idx near max (high end of density slab).
        // Only meaningful for 2048 grid. Diagnoses if high-byte-offset atomics fail.
        uint32_t bins_per_core_max = (nbx * nby + 109u) / 110u;
        uint32_t local_hi_start = (bins_per_core_max >= 8u) ? (bins_per_core_max - 8u) : 0u;
        std::uniform_real_distribution<float> u01(0.05f, 0.45f);
        for (uint32_t i = 0; i < n_cells; ++i) {
            uint32_t owner_idx = i % 110u;
            uint32_t local_idx = local_hi_start + ((i / 110u) % 8u);
            uint32_t bin_global = local_idx * 110u + owner_idx;
            if (bin_global >= nbx * nby) {
                pack(i, 0u, 0u, 0.0f);  // skip out-of-range
                continue;
            }
            uint32_t bx = bin_global / nby;
            uint32_t by = bin_global % nby;
            pack(i, bx, by, u01(rng));
        }
    } else if (dist == "cluster") {
        // 4 hot bins in a tight corner; all cells split evenly among them.
        // Stresses owner-side L1 atomicity + bandwidth, NOT spread.
        std::array<std::pair<uint32_t, uint32_t>, 4> hot = {{
            {0u,         0u},
            {0u,         (uint32_t)(nby / 2u)},
            {(uint32_t)(nbx / 2u), 0u},
            {(uint32_t)(nbx / 2u), (uint32_t)(nby / 2u)},
        }};
        std::uniform_real_distribution<float> u01(0.05f, 0.45f);
        for (uint32_t i = 0; i < n_cells; ++i) {
            auto [bx, by] = hot[i & 3u];
            pack(i, bx, by, u01(rng));
        }
    } else if (dist == "uniform") {
        // Cells touch random bins across the full grid.
        std::uniform_int_distribution<uint32_t> bx_dist(0u, nbx - 1u);
        std::uniform_int_distribution<uint32_t> by_dist(0u, nby - 1u);
        std::uniform_real_distribution<float> u01(0.05f, 0.45f);
        for (uint32_t i = 0; i < n_cells; ++i) {
            pack(i, bx_dist(rng), by_dist(rng), u01(rng));
        }
    } else if (dist == "hotcold") {
        // 80% land in a 4-bin hot set, 20% spread randomly.
        std::array<std::pair<uint32_t, uint32_t>, 4> hot = {{
            {0u,         0u},
            {0u,         (uint32_t)(nby / 2u)},
            {(uint32_t)(nbx / 2u), 0u},
            {(uint32_t)(nbx / 2u), (uint32_t)(nby / 2u)},
        }};
        std::uniform_int_distribution<uint32_t> bx_dist(0u, nbx - 1u);
        std::uniform_int_distribution<uint32_t> by_dist(0u, nby - 1u);
        std::uniform_real_distribution<float> u01(0.05f, 0.45f);
        std::uniform_int_distribution<uint32_t> hot_pick(0u, 9u);  // 80/20
        for (uint32_t i = 0; i < n_cells; ++i) {
            if (hot_pick(rng) < 8u) {
                auto [bx, by] = hot[i & 3u];
                pack(i, bx, by, u01(rng));
            } else {
                pack(i, bx_dist(rng), by_dist(rng), u01(rng));
            }
        }
    } else {
        fprintf(stderr, "[v19mb] unknown dist '%s' (cluster|uniform|hotcold)\n", dist.c_str());
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <grid> <n_cells> <dist> [mode]\n", argv[0]);
        fprintf(stderr, "  grid: 512|1024|2048\n");
        fprintf(stderr, "  dist: cluster|uniform|hotcold|origin|low_local|high_local\n");
        fprintf(stderr, "  mode: split (default — strided staging + host de-stride)\n");
        fprintf(stderr, "        combined (emit+writeout in 1 kernel, strided)\n");
        fprintf(stderr, "        dense (Variant B — per-bin direct write, NoC-alignment limited)\n");
        fprintf(stderr, "        block (Variant A — block partition ownership, dens_buf is\n");
        fprintf(stderr, "               row-major already, NO de-stride needed)\n");
        return 1;
    }
    uint32_t grid      = (uint32_t)std::atoi(argv[1]);
    uint32_t n_cells   = (uint32_t)std::atoi(argv[2]);
    std::string dist   = argv[3];
    std::string mode   = (argc >= 5) ? argv[4] : "split";

    fprintf(stderr, "[v19mb] grid=%u n_cells=%u dist=%s mode=%s\n",
            grid, n_cells, dist.c_str(), mode.c_str());

    auto mesh_device = MeshDevice::create(MeshDeviceConfig{MeshShape{1, 1}});
    auto& cq = mesh_device->mesh_command_queue();
    CoreCoord grid_cc = mesh_device->compute_with_storage_grid_size();
    int nc_all = (int)(grid_cc.x * grid_cc.y);
    fprintf(stderr, "[v19mb] grid=%ux%u nc_all=%d\n",
            (unsigned)grid_cc.x, (unsigned)grid_cc.y, nc_all);

    // Per-core logical cores list
    std::vector<CoreCoord> all_ccs;
    for (uint32_t y = 0; y < grid_cc.y; ++y)
        for (uint32_t x = 0; x < grid_cc.x; ++x)
            all_ccs.push_back({x, y});
    std::set<CoreRange> crs_set;
    for (auto cc : all_ccs) crs_set.insert(CoreRange{cc, cc});
    CoreRangeSet all_crs(crs_set);
    MeshCoordinateRange device_range(mesh_device->shape());

    // Per-core cell splits
    uint32_t base_cells = n_cells / (uint32_t)nc_all;
    uint32_t rem_cells  = n_cells % (uint32_t)nc_all;
    uint32_t max_cells_per_core = base_cells + (rem_cells > 0 ? 1u : 0u);
    uint32_t max_pages_per_core = (max_cells_per_core + CELLS_PER_PAGE - 1u) / CELLS_PER_PAGE;
    uint32_t cells_l1_bytes     = max_pages_per_core * CELL_PAGE_BYTES;

    fprintf(stderr, "[v19mb] per-core cells: base=%u rem=%u max=%u pages=%u (%u KB L1)\n",
            base_cells, rem_cells, max_cells_per_core, max_pages_per_core,
            cells_l1_bytes >> 10);

    // Generate ALL cells in host
    std::vector<uint32_t> host_cells;
    gen_cells(dist, grid, grid, n_cells, host_cells);

    // Re-pack: per-core contiguous (round-robin into core groups).
    // Cell i in flat array → owner core (i % nc_all), slot (i / nc_all) inside that core.
    // Then within each core, pad up to max_pages_per_core full pages with zero cells.
    std::vector<uint32_t> dram_cells((size_t)nc_all * max_pages_per_core * (CELL_PAGE_BYTES / 4u), 0u);
    for (uint32_t i = 0; i < n_cells; ++i) {
        uint32_t owner = i % (uint32_t)nc_all;
        uint32_t slot  = i / (uint32_t)nc_all;
        uint32_t core_off_u32 = owner * max_pages_per_core * (CELL_PAGE_BYTES / 4u) + slot * CELL_U32;
        dram_cells[core_off_u32 + 0] = host_cells[i * CELL_U32 + 0];
        dram_cells[core_off_u32 + 1] = host_cells[i * CELL_U32 + 1];
        dram_cells[core_off_u32 + 2] = host_cells[i * CELL_U32 + 2];
    }

    // Allocate ALL DRAM buffers first (creation order), then write to them.
    // Doing writes AFTER all allocations defends against potential allocator
    // side-effects (e.g., later allocation perturbing earlier buffer data).
    uint32_t total_pages = (uint32_t)nc_all * max_pages_per_core;
    DeviceLocalBufferConfig cell_cfg{
        .page_size = CELL_PAGE_BYTES,
        .buffer_type = BufferType::DRAM};
    ReplicatedBufferConfig cell_rcfg{.size = (uint64_t)total_pages * CELL_PAGE_BYTES};
    auto cell_buf = MeshBuffer::create(cell_rcfg, cell_cfg, mesh_device.get());

    uint32_t coord_pgsz = ((uint32_t)nc_all * 4u + 31u) & ~31u;
    DeviceLocalBufferConfig coord_cfg{
        .page_size = coord_pgsz,
        .buffer_type = BufferType::DRAM};
    ReplicatedBufferConfig coord_rcfg{.size = coord_pgsz};
    auto coord_buf = MeshBuffer::create(coord_rcfg, coord_cfg, mesh_device.get());
    std::vector<uint32_t> coords((size_t)(coord_pgsz / 4u), 0u);
    {
        auto* dev0 = mesh_device->get_devices()[0];
        for (int c = 0; c < nc_all; ++c) {
            CoreCoord noc = dev0->worker_core_from_logical_core(all_ccs[c]);
            coords[c] = ((uint32_t)noc.y << 16) | (uint32_t)noc.x;
        }
    }

    // L1 density slab math. ALIGNMENT NOTE: density_slab_bins must be padded
    // so density_slab_bytes is 32-byte aligned, otherwise the DRAM page size
    // gets rounded up by host and the strided readback / de-stride math goes
    // out of sync (every bin reads from a misaligned location).  Pad bins
    // upward to the nearest multiple of 8 (= 32 bytes).
    //
    // L1 layout: cells_l1 + coords at LOW offsets (DRAM-read destinations),
    // density slab at HIGH offset (only receives writes — local stores AND
    // incoming atomic increments — proven safe at high offsets).
    //
    // Why this matters: at 2048 grid the density slab is ~152 KB. If we
    // put it FIRST, then cells_l1 lives at L1 offset > 150 KB, and the
    // kernel's noc_async_read_page into that high L1 destination silently
    // returns garbage (same class of bug as the V19 noc_async_read of
    // coord_buf at 2048). Moving DRAM reads to LOW L1 offset sidesteps it.
    uint32_t total_bins = grid * grid;
    uint32_t density_slab_bins_raw = (total_bins + (uint32_t)nc_all - 1u) / (uint32_t)nc_all;
    uint32_t density_slab_bins = (density_slab_bins_raw + 7u) & ~7u;
    uint32_t density_slab_bytes = density_slab_bins * 4u;
    uint32_t cells_l1_off   = 0u;
    uint32_t coords_off     = ((cells_l1_off + cells_l1_bytes) + 31u) & ~31u;
    uint32_t density_l1_off = ((coords_off + coord_pgsz) + 31u) & ~31u;
    // +256 bytes for dense writeout scratch ring (8 slots × 32 B each).
    uint32_t cb24_size      = ((density_l1_off + density_slab_bytes + 256u) + 31u) & ~31u;

    fprintf(stderr, "[v19mb] L1 layout: density=%u-%u  coords=%u-%u  cells=%u-%u  total_c24=%u KB\n",
            density_l1_off, density_l1_off + density_slab_bytes,
            coords_off, coords_off + coord_pgsz,
            cells_l1_off, cells_l1_off + cells_l1_bytes,
            cb24_size >> 10);

    if (cb24_size > 1497u * 1024u) {
        fprintf(stderr, "[v19mb] FATAL: L1 (%u KB) exceeds Blackhole budget (1497 KB)\n",
                cb24_size >> 10);
        return 1;
    }

    // DRAM density staging (one page per core, per V19 production).
    uint32_t density_dram_pgsz = (density_slab_bytes + 31u) & ~31u;
    DeviceLocalBufferConfig dens_cfg{
        .page_size = density_dram_pgsz,
        .buffer_type = BufferType::DRAM};
    ReplicatedBufferConfig dens_rcfg{.size = (uint64_t)nc_all * density_dram_pgsz};
    auto dens_buf = MeshBuffer::create(dens_rcfg, dens_cfg, mesh_device.get());

    // Variant B: dense row-major DRAM density map (M*N×u32). Pages of 4 KB
    // (1024 bins each); reading back gives row-major bin_global order with
    // no de-stride. ALLOCATED ONLY IN dense MODE to avoid perturbing the
    // DRAM bank layout for `split` (which has known-good addresses).
    uint32_t dense_dram_pgsz = 4096u;  // 1024 bins per page
    uint32_t total_bins_global = grid * grid;
    uint64_t dense_bytes = (uint64_t)total_bins_global * 4ull;
    uint64_t dense_bytes_padded = ((dense_bytes + dense_dram_pgsz - 1u) /
                                   dense_dram_pgsz) * dense_dram_pgsz;
    std::shared_ptr<MeshBuffer> dense_buf;
    if (mode == "dense") {
        DeviceLocalBufferConfig dense_cfg{
            .page_size = dense_dram_pgsz,
            .buffer_type = BufferType::DRAM};
        ReplicatedBufferConfig dense_rcfg{.size = dense_bytes_padded};
        dense_buf = MeshBuffer::create(dense_rcfg, dense_cfg, mesh_device.get());
    }

    // Diagnostic DRAM buffer: 16 u32 per core (page_size = 64 B).
    uint32_t diag_pgsz = 64u;  // 16 u32 = 64 B (32-aligned)
    DeviceLocalBufferConfig diag_cfg{
        .page_size = diag_pgsz,
        .buffer_type = BufferType::DRAM};
    ReplicatedBufferConfig diag_rcfg{.size = (uint64_t)nc_all * diag_pgsz};
    auto diag_buf = MeshBuffer::create(diag_rcfg, diag_cfg, mesh_device.get());

    // Now do the actual DRAM uploads (AFTER all allocations are settled).
    EnqueueWriteMeshBuffer(cq, cell_buf, dram_cells, false);
    EnqueueWriteMeshBuffer(cq, coord_buf, coords, false);
    // Zero the dense buffer so unwritten bins (or out-of-bounds writes) read as 0.
    if (mode == "dense" && dense_buf) {
        std::vector<uint32_t> dense_zeros((size_t)(dense_bytes_padded / 4u), 0u);
        EnqueueWriteMeshBuffer(cq, dense_buf, dense_zeros, false);
    }

    Finish(cq);  // ensure all uploads done before kernel JIT

    // Sanity: re-print buffer addresses now that all 3 are allocated.
    fprintf(stderr, "[v19mb] AFTER all alloc + write: cell=0x%08x coord=0x%08x dens=0x%08x\n",
            (uint32_t)cell_buf->address(),
            (uint32_t)coord_buf->address(),
            (uint32_t)dens_buf->address());

    // DIAGNOSTIC: read coord_buf back to host and verify it matches what we wrote.
    {
        std::vector<uint32_t> readback(coord_pgsz / 4u, 0u);
        EnqueueReadMeshBuffer(cq, readback, coord_buf, true);
        fprintf(stderr, "[v19mb] coord_buf readback: rb[0]=0x%08x (want=0x%08x) rb[1]=0x%08x rb[10]=0x%08x rb[109]=0x%08x\n",
                readback[0], coords[0], readback[1], readback[10], readback[109]);
    }
    {
        std::vector<uint32_t> rb_cells(CELL_PAGE_BYTES / 4u, 0u);  // read first page only
        // Read page 0 from cell_buf
        // EnqueueReadMeshBuffer reads entire buffer; we want just page 0. Easier:
        // just dump the first 6 u32 of dram_cells we computed, then re-read same
        // amount via a small temp buffer.
        fprintf(stderr, "[v19mb] cell_buf host source: page0[0..5]={%u %u %u  %u %u %u}\n",
                dram_cells[0], dram_cells[1], dram_cells[2],
                dram_cells[3], dram_cells[4], dram_cells[5]);
    }

    // Helper: allocate a uniform CB at a given index with N tile slots of fp32.
    // Mirrors production V19: scatter + writeout programs allocate the same
    // CB pre-c_24 sizes so c_24's L1 base address is identical in both programs.
    // (CB allocator places CBs sequentially; if kernels of different binary
    // sizes shift L1_UNRESERVED_BASE between programs, identical pre-CB layout
    // still makes c_24 land at the same offset within the program-local
    // allocation region.)
    constexpr uint32_t TILE_BYTES_FP32 = 1024u * 4u;
    auto make_cb_uniform = [&](Program& prog, uint32_t idx, uint32_t n_slots) {
        CreateCircularBuffer(prog, all_crs,
            CircularBufferConfig(n_slots * TILE_BYTES_FP32,
                                 {{idx, tt::DataFormat::Float32}})
                .set_page_size(idx, TILE_BYTES_FP32));
    };

    // ── Zero-init program ──
    // Runs BEFORE scatter so all 110 cores have a zeroed density slab AT THE
    // SAME TIME (separated by Finish(cq) from scatter). This eliminates the
    // cross-core race where a slow core's in-kernel zero loop can overwrite
    // atomics that already landed from a faster-starting peer — observed as
    // sum_actual=0 in the 2048-grid microbench when zeroing was in scatter.
    Program prog_zero = CreateProgram();
    for (uint32_t i = 0; i < 4u; ++i) make_cb_uniform(prog_zero, i, 2u);
    make_cb_uniform(prog_zero, 4u, 2u);
    make_cb_uniform(prog_zero, 5u, 2u);
    for (uint32_t j = 0; j < 8u; ++j) {
        make_cb_uniform(prog_zero, 6u + j, 2u);
        make_cb_uniform(prog_zero, 14u + j, 2u);
    }
    CreateCircularBuffer(prog_zero, all_crs,
        CircularBufferConfig(cb24_size, {{24u, tt::DataFormat::Float32}})
            .set_page_size(24u, cb24_size));
    auto zk = CreateKernel(prog_zero,
        "kernels/v19_mb_zero_ncrisc.cpp", all_crs,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                           .noc = NOC::RISCV_1_default});
    CreateKernel(prog_zero,
        "kernels/v19_mb_void_brisc.cpp", all_crs,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                           .noc = NOC::RISCV_0_default});
    for (int c = 0; c < nc_all; ++c) {
        SetRuntimeArgs(prog_zero, zk, all_ccs[c], {
            density_l1_off,
            density_slab_bins,
        });
    }
    MeshWorkload wl_zero;
    wl_zero.add_program(device_range, std::move(prog_zero));

    // ── Scatter program ──
    Program prog_sc = CreateProgram();
    // Pre-c_24 CBs: 22 total (c_0..c_3, c_4, c_5, c_6..c_13, c_14..c_21) all
    // 2-tile fp32. Matches production V19 scatter+writeout layout.
    for (uint32_t i = 0; i < 4u; ++i) make_cb_uniform(prog_sc, i, 2u);
    make_cb_uniform(prog_sc, 4u, 2u);
    make_cb_uniform(prog_sc, 5u, 2u);
    for (uint32_t j = 0; j < 8u; ++j) {
        make_cb_uniform(prog_sc, 6u + j, 2u);
        make_cb_uniform(prog_sc, 14u + j, 2u);
    }
    CreateCircularBuffer(prog_sc, all_crs,
        CircularBufferConfig(cb24_size, {{24u, tt::DataFormat::Float32}})
            .set_page_size(24u, cb24_size));
    const std::string scatter_kernel_path =
        (mode == "combined") ? "kernels/v19_mb_combined_ncrisc.cpp"
      : (mode == "block")    ? "kernels/v19_mb_scatter_block_ncrisc.cpp"
                             : "kernels/v19_mb_scatter_ncrisc.cpp";
    auto sk = CreateKernel(prog_sc,
        scatter_kernel_path, all_crs,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                           .noc = NOC::RISCV_1_default});
    CreateKernel(prog_sc,
        "kernels/v19_mb_void_brisc.cpp", all_crs,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                           .noc = NOC::RISCV_0_default});
    for (int c = 0; c < nc_all; ++c) {
        uint32_t my_n = base_cells + ((uint32_t)c < rem_cells ? 1u : 0u);
        uint32_t my_pages = (my_n + CELLS_PER_PAGE - 1u) / CELLS_PER_PAGE;
        uint32_t first_page = (uint32_t)c * max_pages_per_core;
        std::vector<uint32_t> args = {
            (uint32_t)c,                              // 0 my_core_idx
            my_n,                                     // 1 n_cells
            (uint32_t)cell_buf->address(),            // 2 cell_dram_addr
            CELL_PAGE_BYTES,                          // 3 cell_dram_pgsz
            my_pages,                                 // 4 n_pages
            first_page,                               // 5 first_page_id
            density_l1_off,                           // 6
            density_slab_bins,                        // 7
            (uint32_t)coord_buf->address(),           // 8
            coord_pgsz,                               // 9
            cells_l1_off,                             // 10
            grid,                                     // 11 nbx
            grid,                                     // 12 nby
            (uint32_t)nc_all,                         // 13
        };
        if (mode == "combined") {
            args.push_back((uint32_t)dens_buf->address());    // 14
            args.push_back(density_dram_pgsz);                // 15
            args.push_back((uint32_t)diag_buf->address());    // 16 diag dram
            args.push_back(diag_pgsz);                        // 17 diag pgsz
            args.push_back(coords_off);                       // 18 explicit coords L1 offset
        } else {
            // split-mode scatter kernel reads coords_off at arg 14
            args.push_back(coords_off);                       // 14
        }
        SetRuntimeArgs(prog_sc, sk, all_ccs[c], args);
    }
    // Coords passed as COMMON runtime args (broadcast to all cores).
    // Kernel reads via get_common_arg_val<uint32_t>(i) for i in [0, nc_all).
    SetCommonRuntimeArgs(prog_sc, sk, coords);
    MeshWorkload wl_sc;
    wl_sc.add_program(device_range, std::move(prog_sc));

    // ── Writeout program (mirrors production V19 CB layout!) ──
    // We use a simpler CB layout here (just c_24) — since this microbench
    // doesn't share scatter/writeout across CB-allocation-order-dependent
    // calls, we can just allocate a single c_24 with the SAME size.
    Program prog_wo = CreateProgram();
    // Mirror scatter's pre-c_24 CB layout exactly (Bug 2 fix from V19 handoff
    // doc — without this, c_24's L1 base differs between scatter and writeout
    // programs and the writeout reads from a wrong address, returning zeros).
    for (uint32_t i = 0; i < 4u; ++i) make_cb_uniform(prog_wo, i, 2u);
    make_cb_uniform(prog_wo, 4u, 2u);
    make_cb_uniform(prog_wo, 5u, 2u);
    for (uint32_t j = 0; j < 8u; ++j) {
        make_cb_uniform(prog_wo, 6u + j, 2u);
        make_cb_uniform(prog_wo, 14u + j, 2u);
    }
    CreateCircularBuffer(prog_wo, all_crs,
        CircularBufferConfig(cb24_size, {{24u, tt::DataFormat::Float32}})
            .set_page_size(24u, cb24_size));
    const std::string writeout_kernel = (mode == "dense")
        ? "kernels/v19_mb_writeout_dense_brisc.cpp"
        : "kernels/v19_writeout_dm.cpp";
    auto wo_brisc = CreateKernel(prog_wo,
        writeout_kernel, all_crs,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                           .noc = NOC::RISCV_0_default});
    CreateKernel(prog_wo,
        "kernels/v19_writeout_void_ncrisc.cpp", all_crs,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                           .noc = NOC::RISCV_1_default});
    for (int c = 0; c < nc_all; ++c) {
        if (mode == "dense") {
            // Variant B writeout args: each core writes its bins directly into
            // the row-major dense_buf at bin_global = local*nc_all + c.
            SetRuntimeArgs(prog_wo, wo_brisc, all_ccs[c], {
                density_l1_off,
                density_slab_bins,
                (uint32_t)dense_buf->address(),
                dense_dram_pgsz,
                (uint32_t)c,
                (uint32_t)nc_all,
                total_bins_global,
            });
        } else {
            SetRuntimeArgs(prog_wo, wo_brisc, all_ccs[c], {
                density_l1_off,
                density_slab_bins,
                (uint32_t)dens_buf->address(),
                density_dram_pgsz,
                (uint32_t)c,
            });
        }
    }
    MeshWorkload wl_wo;
    wl_wo.add_program(device_range, std::move(prog_wo));

    // ── Zero-init (single Finish() barrier ensures all cores done before any
    //    atomic emit starts).
    fprintf(stderr, "[v19mb] launching zero-init...\n");
    auto t_zero = hrclock::now();
    EnqueueMeshWorkload(cq, wl_zero, false);
    Finish(cq);
    double zero_ms = ms_since(t_zero);
    fprintf(stderr, "[v19mb] zero-init done in %.2f ms\n", zero_ms);

    // ── JIT warmup is OPTIONAL here. We'll just run the real workload.
    fprintf(stderr, "[v19mb] launching scatter...\n");
    auto t0 = hrclock::now();
    EnqueueMeshWorkload(cq, wl_sc, false);
    Finish(cq);
    double scatter_ms = ms_since(t0);
    fprintf(stderr, "[v19mb] scatter done in %.2f ms\n", scatter_ms);

    double writeout_ms = 0.0;
    if (mode == "split" || mode == "dense" || mode == "block") {
        fprintf(stderr, "[v19mb] launching writeout (mode=%s)...\n", mode.c_str());
        t0 = hrclock::now();
        EnqueueMeshWorkload(cq, wl_wo, false);
        Finish(cq);
        writeout_ms = ms_since(t0);
        fprintf(stderr, "[v19mb] writeout done in %.2f ms\n", writeout_ms);
    } else {
        fprintf(stderr, "[v19mb] mode=combined → writeout fused into scatter kernel\n");
    }

    // ── Validate against host model
    std::vector<uint64_t> expected_sum((size_t)total_bins, 0ull);
    for (uint32_t i = 0; i < n_cells; ++i) {
        uint32_t bx = host_cells[i * CELL_U32 + 0];
        uint32_t by = host_cells[i * CELL_U32 + 1];
        uint32_t af = host_cells[i * CELL_U32 + 2];
        if (bx >= grid || by >= grid) continue;
        if (af == 0u) continue;
        uint32_t bg = bx * grid + by;
        expected_sum[bg] += af;
    }

    uint64_t mismatches = 0;
    uint64_t total_actual = 0, total_expected = 0;
    uint32_t max_diff = 0;
    double read_ms = 0.0;
    std::vector<uint32_t> strided;  // for diag dump (combined mode)

    if (mode == "dense") {
        // Variant B: read row-major dense buffer directly. No de-stride.
        fprintf(stderr, "[v19mb] reading dense DRAM density map...\n");
        t0 = hrclock::now();
        std::vector<uint32_t> dense_buf_host((size_t)(dense_bytes_padded / 4u), 0u);
        EnqueueReadMeshBuffer(cq, dense_buf_host, dense_buf, true);
        read_ms = ms_since(t0);
        fprintf(stderr, "[v19mb] dense_buf_host[0..6] = {%u %u %u %u %u %u %u} expected[0] = %u\n",
                dense_buf_host[0], dense_buf_host[1], dense_buf_host[2],
                dense_buf_host[3], dense_buf_host[4], dense_buf_host[5],
                dense_buf_host[6],
                (uint32_t)(expected_sum[0] & 0xFFFFFFFFu));
        uint32_t printed = 0;
        for (uint32_t bg = 0; bg < total_bins; ++bg) {
            uint32_t got  = dense_buf_host[bg];
            uint32_t want = (uint32_t)(expected_sum[bg] & 0xFFFFFFFFu);
            total_actual   += got;
            total_expected += want;
            if (got != want) {
                ++mismatches;
                uint32_t d = got > want ? got - want : want - got;
                if (d > max_diff) max_diff = d;
                if (printed < 6) {
                    fprintf(stderr, "[v19mb] MISMATCH bg=%u got=%u want=%u (page=%u, off_u32=%u)\n",
                            bg, got, want, bg / 1024u, bg % 1024u);
                    ++printed;
                }
            }
        }
    } else if (mode == "block") {
        // BLOCK partition: dens_buf is already row-major bin_global. The
        // existing per-core-page writeout writes core c's slab (= bins
        // c*slab_bins..(c+1)*slab_bins-1) to page c of dens_buf, which the
        // host read concatenates in page order → bins 0, 1, 2, ... directly.
        fprintf(stderr, "[v19mb] reading DRAM density (block — no de-stride)...\n");
        t0 = hrclock::now();
        strided.assign((size_t)nc_all * density_slab_bins, 0u);
        EnqueueReadMeshBuffer(cq, strided, dens_buf, true);
        read_ms = ms_since(t0);
        for (uint32_t bg = 0; bg < total_bins; ++bg) {
            uint32_t got  = strided[bg];   // direct index — block layout is row-major
            uint32_t want = (uint32_t)(expected_sum[bg] & 0xFFFFFFFFu);
            total_actual   += got;
            total_expected += want;
            if (got != want) {
                ++mismatches;
                uint32_t d = got > want ? got - want : want - got;
                if (d > max_diff) max_diff = d;
            }
        }
    } else {
        // split / combined: strided staging + host de-stride.
        fprintf(stderr, "[v19mb] reading DRAM density staging...\n");
        t0 = hrclock::now();
        strided.assign((size_t)nc_all * density_slab_bins, 0u);
        EnqueueReadMeshBuffer(cq, strided, dens_buf, true);
        read_ms = ms_since(t0);
        for (uint32_t bg = 0; bg < total_bins; ++bg) {
            uint32_t owner = bg % (uint32_t)nc_all;
            uint32_t local = bg / (uint32_t)nc_all;
            uint32_t got   = strided[(size_t)owner * density_slab_bins + local];
            uint32_t want  = (uint32_t)(expected_sum[bg] & 0xFFFFFFFFu);
            total_actual   += got;
            total_expected += want;
            if (got != want) {
                ++mismatches;
                uint32_t d = got > want ? got - want : want - got;
                if (d > max_diff) max_diff = d;
            }
        }
    }

    // DIAGNOSTIC: check the marker slot at density_l1[density_slab_bins-1].
    // If kernel mode=combined was used and L1<->DRAM round-trip is intact, the
    // marker should be 0xDEADBEEF on every core. If it's 0, even local stores
    // are being lost between scatter end and writeout / DRAM read.
    if (mode == "combined") {
        // Read diagnostic buffer
        std::vector<uint32_t> diag_data((size_t)nc_all * (diag_pgsz / 4u), 0u);
        EnqueueReadMeshBuffer(cq, diag_data, diag_buf, true);

        fprintf(stderr, "[v19mb]   host args:  cell=0x%08x  coord=0x%08x  dens=0x%08x  diag=0x%08x\n",
                (uint32_t)cell_buf->address(),
                (uint32_t)coord_buf->address(),
                (uint32_t)dens_buf->address(),
                (uint32_t)diag_buf->address());
        fprintf(stderr, "[v19mb]   per-core diag dump (via separate DRAM buf, 4 sample cores):\n");
        for (int c : {0, 1, 50, 109}) {
            uint32_t* d = &diag_data[(size_t)c * 16u];
            fprintf(stderr,
                    "[v19mb]   c%-3d  density_l1=0x%08x  coords[0]=0x%08x  noc_coord_dram=0x%08x\n"
                    "         cell0=(bx=%u, by=%u, af=0x%08x)  my_idx=%u  n_cells=%u\n"
                    "         cell_dram_addr=0x%08x cell_pgsz=%u first_page=%u n_pages=%u\n"
                    "         coords[my]=0x%08x  cells_l1[3]=%u  cells_l1_addr=0x%08x  sent=0x%08x\n",
                    c, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
                    d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15]);
        }
    }

    fprintf(stderr, "[v19mb] === RESULT grid=%u cells=%u dist=%s ===\n",
            grid, n_cells, dist.c_str());
    fprintf(stderr, "[v19mb]   scatter:  %.3f ms  ({writeout: %.3f ms, dram_read: %.3f ms})\n",
            scatter_ms, writeout_ms, read_ms);
    fprintf(stderr, "[v19mb]   sum_actual=%llu  sum_expected=%llu (low32 wrapped)\n",
            (unsigned long long)total_actual, (unsigned long long)total_expected);
    fprintf(stderr, "[v19mb]   mismatches=%llu  max_diff=%u  %s\n",
            (unsigned long long)mismatches, max_diff,
            (mismatches == 0) ? "✓ OK" : "✗ FAIL");

    // CSV-style result line on stdout for easy parsing
    printf("RESULT,%u,%u,%s,%.3f,%.3f,%.3f,%llu,%llu,%llu\n",
           grid, n_cells, dist.c_str(),
           scatter_ms, writeout_ms, read_ms,
           (unsigned long long)mismatches,
           (unsigned long long)total_actual,
           (unsigned long long)total_expected);
    fflush(stdout);

    mesh_device->close();
    return mismatches == 0 ? 0 : 2;
}
