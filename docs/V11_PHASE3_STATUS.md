# V11 Phase 3 (Hot-Tile Sharding) — Pause Status

## Where the code is RIGHT NOW

**Functional and accurate** at all configs that V11 passes. Sharding
infrastructure (histogram + shard_table builder + scatter shard-table
read) is in place but **shard routing is safety-disabled** — scatter
hard-codes `owner = primary` until the accum K-way reduce is implemented.

```bash
# Smoke test confirms accuracy preserved at 2048²×1.5M:
GATHER_MODE=v11 /localdev/ayadav/tt-work/TTPort/DREAMPlace/dp_env/bin/python3 \
  /localdev/ayadav/tt-work/TTPort/experiments/density_scatter/tt_metal/v11_phase2_smoke.py \
  --container <container> --grid 2048 --cells 1500000
# → rel_L2 = 0.32%, PASS, scatter=24.9ms, gather=15.5ms
```

## What was learned in this session

### Single-iter benchmarks are misleading
V11 single-iter at 2048²×211K: scatter+gather = 8.5 ms (~3.6× CPU 8T).
But E2E DREAMPlace adaptec1_2048 (NC_max=370K): scatter+gather = 36.3 ms
(0.78× CPU 8T). The DREAMPlace fillers + per-call fixed overhead change
the math entirely. Benchmark E2E from the start.

### CPU 8T baselines (Gate 0)
- adaptec1_2048: median scatter+gather = **28.42 ms**
- bigblue2_2048: median scatter+gather = **42.37 ms**

### V11 E2E results
- adaptec1_2048: 36.3 ms (0.78× CPU)
- bigblue2_2048: 46.9 ms (0.90× CPU)

### Profile data identified the real bottleneck
V11A-ACC zone has **14× variance** between mean and max per chunk. Gather
time = max-core total ACC time across 7 chunks. This is **load
imbalance**, not algorithm cost.

### Histogram pre-pass on real DREAMPlace adaptec1
- max_per_tile = **755,526 contribs**
- per-core load imbalance = **38.67×**
- 4 hot tiles, K=8 each, 28 shard slots needed
- Confirms sharding is the right direction

### What I tried that didn't help
| Optimization | Result |
|--------------|--------|
| Pre-multiply by inv_ba in scatter | Reverted (scatter +9 ms regression) |
| Fused SCALE+WRITE in accum | Reverted (per-write barriers serialized NOC) |
| Bulk header read (1 barrier vs 7) | Negligible improvement in E2E |
| Tile-based SFPU rewrite (V12) | Designed but not built — scatter side
  produces SPARSE per-tile contributions (~9/1024 cells per tile), so
  SFPU vectorization wastes 99% of lanes. SFPU only wins if dense
  partial tiles get filled by many cells per tile, which happens at
  bigblue2 but not adaptec1. |

### What I tried that helped
| Optimization | Saved | Status |
|--------------|-------|--------|
| 2-zero-streak break in V11-ROUTE inner loop | ~5 ms scatter at 1.5M | Kept |
| 64-byte cache-line alignment for headers | (correctness fix, not perf) | Kept |
| 32-byte tuples (16 B payload + pad) | (correctness fix) | Kept |
| 128-byte safety gap before dense buffer | (correctness fix) | Kept |

## Files added this session

```
experiments/density_scatter/tt_metal/host/v11_tile_ownership.h
  - build_snake_fill_ownership()
  - build_shard_table()  ← NEW: greedy K-way assignment

experiments/density_scatter/tt_metal/kernels/v11_scatter_dm.cpp
  - Loads tile_to_core AND shard_table from DRAM
  - Routing currently uses primary only (safety)

experiments/density_scatter/tt_metal/kernels/v11_accum_dm.cpp
  - Bulk-header read (1 barrier vs N)
  - 64-byte cache-line aligned, 32-byte tuples

experiments/density_scatter/tt_metal/kernels/v11_histogram.cpp
  - Counts per-tile contribs, dumps to per-core DRAM page

experiments/density_scatter/tt_metal/host/density_scatter_ttnn_server_host.cpp
  - GATHER_MODE=v11 dispatch (gmode_str="v11-tile-routed")
  - Allocates tile_map_buf, route_buf, owned_lookup_buf, hist_buf,
    shard_table_buf, shard_reduce_buf
  - Builds prog_v11_scatter, prog_v11_accum, prog_v11_hist
  - First-iter histogram + shard_table compute + log + upload

experiments/density_scatter/tt_metal/v11_phase2_smoke.py
  - Standalone single-iter smoke harness (rel_L2 vs CPU triangle ref)

experiments/density_scatter/V11_PHASE2_STATUS.md
experiments/density_scatter/V11_PHASE3_STATUS.md  (this file)
experiments/density_scatter/V12_DESIGN.md           (deferred V12 design)

integration/scatter_ttnn_client.py
  - Added "5: v11" to gather_mode decoder dict
```

## How to resume

### Step 1: Re-enable shard routing in scatter (5 min)

Find this block in `kernels/v11_scatter_dm.cpp`:
```cpp
// SAFETY: shard routing disabled until accum K-way reduce ...
uint32_t owner = primary;
(void)shard_table;
(void)emit_counter;
```
Replace with the round-robin block (which exists in git history if needed):
```cpp
const uint8_t* sh = shard_table + map_idx * 16u;
uint32_t K = (uint32_t)sh[0];
uint32_t owner = (K <= 1u) ? primary
                            : ((emit_counter % K) == 0 ? primary
                                                       : (uint32_t)sh[emit_counter % K]);
emit_counter++;
```

### Step 2: Extend host owned_lookup with shard slots (~1-2 hours)

In `host/density_scatter_ttnn_server_host.cpp`, after `build_shard_table()`
call, build per-core lists:
```cpp
struct CoreShardInfo {
    std::vector<uint32_t> primary_tiles;  // existing snake-fill list
    std::vector<uint8_t>  primary_K;       // 1 (cold) or 2..8 (hot)
    std::vector<std::tuple<uint32_t, uint32_t, uint32_t>>
        shard_slots;  // (tile_id, hot_tile_seq, shard_idx_in_K)
};
std::vector<CoreShardInfo> per_core(nc_all);

// Walk shard_table: for each hot tile T with K>1 and alts list,
// for shard_idx in 1..K-1:
//     per_core[alts[shard_idx-1]].shard_slots.push_back(
//         (T, hot_tile_seq[T], shard_idx));
```

Update `owned_lookup_buf` upload so each core's page marks BOTH primary
tiles (slot 0..n_primary-1) AND shard tiles (slot n_primary..end).

### Step 3: Modify v11_accum_dm.cpp to handle shard slots (~1 hour)

The kernel mostly works — owned_lookup just has more entries now. Main
change: drop the SCALE step and density-write at the end (those move to
the new reduce kernel).

### Step 4: Author kernels/v11_reduce_dm.cpp (~2 hours)

```cpp
// Phase A: shard slots write dense to shard_reduce_buf
//   for s in own_shard_slots:
//     dram_offset = s.hot_tile_seq * MAX_K * TILE_BYTES
//                 + s.shard_idx * TILE_BYTES
//     noc_async_write(dense[s.local_idx], dram_offset, TILE_BYTES);
//   noc_async_write_barrier();
//
// Phase B: primaries read shards from shard_reduce_buf, BRISC-add
//   for p in own_primary_tiles where K > 1:
//     for shard_idx in 1..K-1:
//       noc_async_read(dram_offset(p, shard_idx), tmp_l1, TILE_BYTES);
//     barrier();
//     for shard_idx in 1..K-1:
//       for i in 0..1023: dense[p.local_idx][i] += tmp_l1[shard_idx][i];
//
// Phase C: scale + density write (moved here from accum)
```

### Step 5: Add prog_v11_reduce in host + wire into per-iter loop (~1 hour)

```cpp
EnqueueMeshWorkload(cq, wl_v11_scatter, false); Finish(cq);
EnqueueMeshWorkload(cq, wl_v11_accum,   false); Finish(cq);
// Sharding adds:
EnqueueMeshWorkload(cq, wl_v11_reduce,  false); Finish(cq);
```

### Step 6: Validate (~2-4 hours)

1. Smoke test at synthetic 1.5M (no hot tiles → reduce is no-op,
   should match V11 today).
2. Smoke test at synthetic with **artificially-clustered** cells (e.g.,
   all in 5% of die). Validate that hot tiles are correctly handled.
3. E2E DREAMPlace adaptec1 + bigblue2 (real data, clustered).
4. Compare scatter+gather median to CPU 8T baseline.

### Step 7: Add caching (~1 hour)

Currently histogram runs every iter (~13 ms overhead). Cache shard_table
for N=25 iters per plan §3.10. Track iter counter on host.

## Open questions to consider before resuming

1. **Will sharding actually hit 3× at adaptec1?** Math: 755K/8 = 95K per
   core. At ~100 ns per tuple, that's 9.5 ms minimum on the slowest
   core. Plus normal owned tiles and reduce overhead. Realistic
   ceiling: ~12-15 ms gather (vs current 26 ms). Close to target but
   not guaranteed.

2. **Should we revisit V12 (tile-based SFPU) for bigblue2?** If
   sharding gets adaptec1 to 3× but bigblue2 still bottlenecks, the
   tile-based approach may help bigblue2 (where SFPU lanes are
   actually utilized due to higher cell density per tile).

3. **Is there a simpler path?** With Phase 5 + sharding falling short,
   the IMPLEMENTATION_LOG.md mentions a "tiled matmul" approach (10-32×
   speedup via SFPU matmul). That's a much bigger architectural pivot
   but could make adaptec1 trivially fast.

## Reproducing baselines + V11 numbers

```bash
# CPU 8T baselines
OMP_NUM_THREADS=8 /localdev/ayadav/tt-work/TTPort/DREAMPlace/dp_env/bin/python3 \
  /localdev/ayadav/tt-work/TTPort/integration/run_dreamplace.py --device cpu \
  --benchmark /localdev/ayadav/tt-work/TTPort/DREAMPlace/test/ispd2005/adaptec1_2048.json \
  --results-dir /localdev/ayadav/tt-work/TTPort/experiments/density_scatter/results

# V11 E2E
GATHER_MODE=v11 OMP_NUM_THREADS=8 /localdev/ayadav/tt-work/TTPort/DREAMPlace/dp_env/bin/python3 \
  /localdev/ayadav/tt-work/TTPort/integration/run_dreamplace.py --device scatter_ttnn \
  --container <ayadav-container> \
  --benchmark /localdev/ayadav/tt-work/TTPort/DREAMPlace/test/ispd2005/adaptec1_2048.json \
  --results-dir /localdev/ayadav/tt-work/TTPort/experiments/density_scatter/results

# Single-iter accuracy + perf snapshot
GATHER_MODE=v11 /localdev/ayadav/tt-work/TTPort/DREAMPlace/dp_env/bin/python3 \
  /localdev/ayadav/tt-work/TTPort/experiments/density_scatter/tt_metal/v11_phase2_smoke.py \
  --container <ayadav-container> --grid 2048 --cells 1500000
```

## Build commands

```bash
# Container must be running (your bh-38-special-ayadav-for-reservation-XXXXX)
docker ps  # find container name
docker start <container>  # if exited

docker exec -w /localdev/ayadav/tt-work/TTPort/experiments/density_scatter/tt_metal/build_ttnn \
  <container> bash -c 'cmake --build . --target density_scatter_ttnn_server -j$(nproc)'
```

## Bottom line

**adaptec1_2048**: V11 single-iter is fast (3.6× CPU) but DREAMPlace E2E
with fillers is slow (0.78×). Real bottleneck is single-tile imbalance
(755K contribs in one tile). Sharding implementation is the right next
step but takes 6-12 more hours.

**bigblue2_2048**: V11 single-iter is roughly even with CPU (1×). E2E
similar (0.90×). Likely benefits from sharding less dramatically than
adaptec1. May ultimately need a tile-based SFPU rewrite (V12) for 3×.
