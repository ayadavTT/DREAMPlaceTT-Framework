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
