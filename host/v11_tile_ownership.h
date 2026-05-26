// SPDX-License-Identifier: Apache-2.0
//
// V11 tile-ownership map builder.
//
// V11 partitions the M×N density grid into 32×32 spatial tiles and assigns
// each tile to one Tensix core that "owns" it (accumulates contributions
// from all 120 cell processors that target that tile).
//
// Snake-fill traversal: visit row 0 left→right, row 1 right→left, row 2
// left→right, etc. Spatially-adjacent tiles map to consecutive linear
// indices, which co-locates them on near-neighbor cores and keeps NOC
// distances small for the inevitable inter-core routing.
//
// Usage:
//   std::vector<uint16_t> tile_to_core;
//   std::vector<std::vector<uint32_t>> core_to_tiles;
//   build_snake_fill_ownership(M_tiles, N_tiles, nc_all,
//                              tile_to_core, core_to_tiles);
//   // tile_to_core[tile_x * N_tiles + tile_y] = owner core linear id
//   // core_to_tiles[core_id] = list of tile linear indices owned by that core

#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace v11 {

// Build snake-fill ownership map.
//   M_tiles × N_tiles total tiles, partitioned across nc_all cores.
//   Each core gets ceil(M_tiles*N_tiles / nc_all) or floor(...) tiles
//   (within ±1 of the average).
//
// Output:
//   tile_to_core[tile_x * N_tiles + tile_y] = owner core id (uint16_t).
//   core_to_tiles[core_id] = list of tile linear indices (tile_x * N_tiles + tile_y).
inline void build_snake_fill_ownership(
    uint32_t M_tiles,
    uint32_t N_tiles,
    uint32_t nc_all,
    std::vector<uint16_t>& tile_to_core,
    std::vector<std::vector<uint32_t>>& core_to_tiles)
{
    const uint32_t total_tiles = M_tiles * N_tiles;
    tile_to_core.assign(total_tiles, 0);
    core_to_tiles.assign(nc_all, std::vector<uint32_t>{});

    // Build the snake-fill linear traversal: order[k] = tile linear index
    // visited at step k. Row 0 LTR, row 1 RTL, row 2 LTR, ...
    std::vector<uint32_t> order(total_tiles);
    uint32_t k = 0;
    for (uint32_t tx = 0; tx < M_tiles; ++tx) {
        if ((tx & 1u) == 0u) {
            for (uint32_t ty = 0; ty < N_tiles; ++ty)
                order[k++] = tx * N_tiles + ty;
        } else {
            for (uint32_t ty = N_tiles; ty-- > 0;)
                order[k++] = tx * N_tiles + ty;
        }
    }

    // Assign each step to a core. Even split: base = total/nc, rem = total%nc.
    // First `rem` cores get base+1 tiles, the rest get base.
    const uint32_t base = total_tiles / nc_all;
    const uint32_t rem  = total_tiles % nc_all;

    uint32_t step = 0;
    for (uint32_t c = 0; c < nc_all; ++c) {
        uint32_t my_n = base + (c < rem ? 1u : 0u);
        for (uint32_t i = 0; i < my_n; ++i) {
            uint32_t tile_idx = order[step++];
            tile_to_core[tile_idx] = (uint16_t)c;
            core_to_tiles[c].push_back(tile_idx);
        }
    }
}

// Convenience: return max tiles owned by any single core (= ceil(total/nc_all)).
inline uint32_t max_tiles_per_core(uint32_t M_tiles, uint32_t N_tiles, uint32_t nc_all) {
    const uint32_t total = M_tiles * N_tiles;
    return (total + nc_all - 1u) / nc_all;
}

// ── V16 hash-based ownership ───────────────────────────────────────────────
//
// Snake-fill assigns spatially-adjacent tiles to the same core. Real
// circuits (ISPD2005 adaptec3, bigblue3, etc.) have hot tiles 119/135/136
// clustered near the design center; under snake-fill these all land on one
// core, which then either spills to DRAM (V15 with cost) or drops mass
// (V15 without spill ⇒ DIVERGED).
//
// Hash-based ownership decorrelates ownership from spatial position. Each
// tile gets assigned to a pseudo-random core via a mixing hash, so hot
// tiles 119/135/136 land on three different cores. The per-(core, tile)
// bucket count for the hottest tile is reduced from ~25K records (all on
// one core) to ~25K/110 ≈ 230 records per core on average, well within
// the V15 cap=4096. No spill ever needed.
//
// We use the same `tile_to_core` / `core_to_tiles` output structure as
// snake-fill so it's a drop-in replacement for V15 host code. Each core
// still owns exactly base or base+1 tiles (perfectly balanced count).
//
// Hash: SplitMix64-style integer mixing on the tile linear index.
// Deterministic across runs (no randomness), so density-write DRAM layout
// is reproducible — only ownership changes, not where each tile's bytes
// land in DRAM.
//
// Reference: V16_PLAN.md §6.1 "(§10.3) Hash-based tile ownership" and
// V15_HANDOFF.md §7.5 "Path A".
inline void build_hash_ownership(
    uint32_t M_tiles,
    uint32_t N_tiles,
    uint32_t nc_all,
    std::vector<uint16_t>& tile_to_core,
    std::vector<std::vector<uint32_t>>& core_to_tiles)
{
    const uint32_t total_tiles = M_tiles * N_tiles;
    tile_to_core.assign(total_tiles, 0);
    core_to_tiles.assign(nc_all, std::vector<uint32_t>{});

    // Build a permutation of [0, total_tiles) sorted by a mixed hash of
    // each tile's linear index. SplitMix64-style mixer (deterministic).
    std::vector<std::pair<uint64_t, uint32_t>> hashed;
    hashed.reserve(total_tiles);
    for (uint32_t t = 0; t < total_tiles; ++t) {
        uint64_t x = (uint64_t)t + 0x9E3779B97F4A7C15ULL;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        x =  x ^ (x >> 31);
        hashed.emplace_back(x, t);
    }
    std::sort(hashed.begin(), hashed.end(),
              [](const auto& a, const auto& b){ return a.first < b.first; });

    // Assign each tile (in the hash-permuted order) to cores round-robin
    // so each core gets base or base+1 tiles. First `rem` cores get +1.
    const uint32_t base = total_tiles / nc_all;
    const uint32_t rem  = total_tiles % nc_all;
    uint32_t step = 0;
    for (uint32_t c = 0; c < nc_all; ++c) {
        uint32_t my_n = base + (c < rem ? 1u : 0u);
        for (uint32_t i = 0; i < my_n; ++i) {
            uint32_t tile_idx = hashed[step++].second;
            tile_to_core[tile_idx] = (uint16_t)c;
            core_to_tiles[c].push_back(tile_idx);
        }
    }
}

// ── Hot-tile sharding ──────────────────────────────────────────────────────
//
// shard_table layout (one entry per tile, packed into a flat byte array):
//   byte 0: K (shard count, 1..MAX_K). K=1 means no sharding.
//   bytes 1..K-1: K-1 alt owner core IDs (uint8 each)
//   bytes K..SHARD_BYTES-1: padding (zero)
//
// Routing: for tuple emit i targeting tile T, owner =
//   (i % K == 0) ? tile_to_core[T] : alts[(i % K) - 1]
//
// Caller-allocated `shard_table` must be sized total_tiles * SHARD_BYTES.
// `per_core_shard_count` (size nc_all) is filled with how many shards each
// core hosts; used by host to size accum kernel's shard slots.
inline void build_shard_table(
    const std::vector<uint32_t>& global_count,   // size = total_tiles
    const std::vector<uint16_t>& tile_to_core,   // size = total_tiles
    uint32_t nc_all,
    uint32_t hot_threshold,
    uint32_t max_k,
    uint32_t shard_bytes,
    std::vector<uint8_t>& shard_table,
    std::vector<uint32_t>& per_core_shard_count)
{
    const uint32_t total_tiles = (uint32_t)global_count.size();
    shard_table.assign((size_t)total_tiles * shard_bytes, 0);
    per_core_shard_count.assign(nc_all, 0);

    // Init all tiles to K=1 (no sharding).
    for (uint32_t t = 0; t < total_tiles; ++t) {
        shard_table[(size_t)t * shard_bytes] = 1;
    }

    // Walk hot tiles in decreasing-count order; for each, pick K-1 alts as
    // the cores currently hosting the fewest shards (greedy load-balance).
    std::vector<uint32_t> hot_tiles;
    for (uint32_t t = 0; t < total_tiles; ++t) {
        if (global_count[t] > hot_threshold) hot_tiles.push_back(t);
    }
    std::sort(hot_tiles.begin(), hot_tiles.end(),
              [&](uint32_t a, uint32_t b) { return global_count[a] > global_count[b]; });

    for (uint32_t t : hot_tiles) {
        uint32_t cnt = global_count[t];
        uint32_t K = (cnt + hot_threshold - 1) / hot_threshold;
        if (K > max_k) K = max_k;
        if (K < 2) continue;
        uint8_t primary = (uint8_t)tile_to_core[t];

        // Pick K-1 alts: cores with lowest per_core_shard_count, excluding primary.
        std::vector<uint32_t> candidates(nc_all);
        for (uint32_t c = 0; c < nc_all; ++c) candidates[c] = c;
        std::sort(candidates.begin(), candidates.end(),
                  [&](uint32_t a, uint32_t b) {
                      return per_core_shard_count[a] < per_core_shard_count[b];
                  });

        uint8_t* entry = shard_table.data() + (size_t)t * shard_bytes;
        entry[0] = (uint8_t)K;
        uint32_t alts_picked = 0;
        for (uint32_t i = 0; i < nc_all && alts_picked < K - 1u; ++i) {
            uint32_t c = candidates[i];
            if (c == (uint32_t)primary) continue;
            entry[1 + alts_picked] = (uint8_t)c;
            per_core_shard_count[c]++;
            alts_picked++;
        }
    }

    // Assign compact hot_tile_seq index for shard_reduce_buf page addressing.
    // Pages are indexed hot_tile_seq * MAX_K + shard_idx (0 = primary slot).
    uint32_t seq = 0;
    for (uint32_t t : hot_tiles) {
        uint8_t* entry = shard_table.data() + (size_t)t * shard_bytes;
        if (entry[0] >= 2) {
            uint32_t* slot = reinterpret_cast<uint32_t*>(entry + 8);
            *slot = seq++;
        }
    }
}

}  // namespace v11

// ────────────────────────────────────────────────────────────────────────────
// V12 tile-ownership helpers.
//
// V12 reuses the same snake-fill ownership algorithm as V11, but adds
// route_buf sizing helpers and a per-core tile-arg vector builder.
// ────────────────────────────────────────────────────────────────────────────
namespace v12 {

// Build snake-fill tile ownership (identical algorithm to v11, exposed
// separately so V12 host code can call it without the V11 namespace).
inline void build_snake_fill_ownership(
    uint32_t M_tiles,
    uint32_t N_tiles,
    uint32_t nc_all,
    std::vector<uint16_t>& tile_to_core,
    std::vector<std::vector<uint32_t>>& core_to_tiles)
{
    v11::build_snake_fill_ownership(M_tiles, N_tiles, nc_all,
                                    tile_to_core, core_to_tiles);
}

// Compute route_buf page size and max vectors per half-page.
//
// Each route page is written by one (writer, receiver) pair.
// Layout:  [32B header] [max_half × 80B NCRISC vectors] [max_half × 80B BRISC vectors]
//
// max_half is chosen so a writer's worst-case contribution to one receiver
// fits within max_half vectors. Worst case: all nc_cells land in the same
// receiver tile — but in practice each cell maps to at most 1×1=1 tile per
// scatter direction, so the average is ~nc_cells_per_writer/total_tiles.
//
// We use a generous upper bound:
//   max_half = ceil(nc_cells_per_writer * MAX_BINS_PER_CELL / total_tiles)
//            with a minimum floor of MIN_HALF_FLOOR.
//
// Returns:
//   max_overlaps_half  — max V12Overlap vectors per NCRISC/BRISC half
//   route_pgsz         — total page size in bytes (must be ≤ 4 GB total buf)
inline void compute_route_pgsz(
    uint32_t nc_max,
    uint32_t nc_all,
    uint32_t M_tiles,
    uint32_t N_tiles,
    uint32_t max_bins_per_cell,    // typically 8 (MAX_OVERLAP^2 / avg tiles)
    uint32_t& max_overlaps_half,
    uint32_t& route_pgsz)
{
    constexpr uint32_t V12_OVERLAP_BYTES  = 80u;
    // 64 bytes: bytes 0..31 NCRISC region, bytes 32..63 BRISC region.
    // Keeps both 32-byte NOC writes non-overlapping (race-free).
    constexpr uint32_t V12_HDR_BYTES      = 64u;
    constexpr uint32_t MIN_HALF_FLOOR     = 64u;

    uint32_t total_tiles = M_tiles * N_tiles;
    uint32_t nc_cells_per_writer = (nc_max + nc_all - 1u) / nc_all;

    // Per (writer, receiver) pair: expected vectors from one writer to one tile.
    // Each cell overlaps at most max_bins_per_cell tiles; the expected number
    // targeting any specific tile is max_bins_per_cell / total_tiles * nc_cells_per_writer.
    // We add a 4× safety margin.
    uint64_t raw = ((uint64_t)nc_cells_per_writer * max_bins_per_cell * 4u
                    + total_tiles - 1u) / total_tiles;
    if (raw < MIN_HALF_FLOOR) raw = MIN_HALF_FLOOR;
    // Cap to avoid oversized buffers: if it's still unreasonably large, warn.
    if (raw > 65536u) raw = 65536u;

    max_overlaps_half = (uint32_t)raw;

    // Page = 64B header + 2 × max_half × 80B, rounded to 32B.
    uint32_t data_bytes = 2u * max_overlaps_half * V12_OVERLAP_BYTES;
    route_pgsz = (V12_HDR_BYTES + data_bytes + 31u) & ~31u;
}

// Total route_buf size in bytes.
// route_buf has nc_all × nc_all pages (each writer × each receiver).
inline uint64_t route_buf_total_bytes(uint32_t nc_all, uint32_t route_pgsz) {
    return (uint64_t)nc_all * (uint64_t)nc_all * (uint64_t)route_pgsz;
}

// V12 L1 scratch size computation.
// Mirrors the offsets in v12_brisc_combined.cpp and v12_ncrisc_combined.cpp.
// Returns the required CB_SCRATCH size in bytes for a single core.
inline uint32_t v12_l1_scratch_bytes(
    uint32_t M_tiles,
    uint32_t N_tiles,
    uint32_t nc_all)
{
    constexpr uint32_t TILE_BYTES   = 1024u * sizeof(float);
    constexpr uint32_t TILE_BF16    = 32u * 32u * sizeof(uint16_t);
    constexpr uint32_t OV_BYTES     = 80u;
    constexpr uint32_t B_STAGE_CAP  = 4u;
    constexpr uint32_t N_STAGE_CAP  = 4u;
    constexpr uint32_t OV_STAGE_CAP = 8u;
    constexpr uint32_t HDR_BYTES    = 64u;  // must match V12_HDR_BYTES in kernels

    // Align to 16B for NOC transfer requirements (same as host arg calculation).
    uint32_t tile_map_bytes  = (M_tiles * N_tiles * (uint32_t)sizeof(uint16_t) + 15u) & ~15u;
    uint32_t noc_table_bytes = nc_all * 2u * (uint32_t)sizeof(uint32_t);
    uint32_t off = 0u;

    // tile_to_core
    off += tile_map_bytes; off = (off + 63u) & ~63u;
    // NOC coord table (appended right after tile_to_core)
    off += noc_table_bytes; off = (off + 63u) & ~63u;
    // cell px/py/sx/sy
    off += 4u * TILE_BYTES; off = (off + 63u) & ~63u;
    // BRISC scatter staging + counts + offsets
    off += nc_all * B_STAGE_CAP * OV_BYTES;
    off  = (off + 3u) & ~3u;
    off += nc_all * 2u * (uint32_t)sizeof(uint32_t);
    off  = (off + 63u) & ~63u;
    off += HDR_BYTES; off = (off + 63u) & ~63u;
    // NCRISC scatter staging + counts + offsets + private cell buffers
    off += nc_all * N_STAGE_CAP * OV_BYTES;
    off  = (off + 3u) & ~3u;
    off += nc_all * 2u * (uint32_t)sizeof(uint32_t);
    off  = (off + 63u) & ~63u;
    off += HDR_BYTES; off = (off + 63u) & ~63u;
    off += 4u * TILE_BYTES; off = (off + 63u) & ~63u;
    // Phase 2: BRISC inbound headers + ov_stage + ox/oy scratch
    off += nc_all * HDR_BYTES; off = (off + 63u) & ~63u;
    off += OV_STAGE_CAP * OV_BYTES; off = (off + 63u) & ~63u;
    off += 2u * TILE_BF16; off = (off + 63u) & ~63u;
    // Phase 2: NCRISC density tile buffer
    off += TILE_BYTES; off = (off + 63u) & ~63u;

    return off;
}

}  // namespace v12
