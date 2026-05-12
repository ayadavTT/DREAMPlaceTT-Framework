# V11 Phase 3 (Hot-Tile Sharding) — Agent Handoff

**Audience:** an agent (or engineer) picking this up cold. Read top-to-bottom.
This document is self-contained — every code path, file, line number, and
algorithm needed to finish Phase 3 is here.

---

## TL;DR — Goal of Phase 3

V11 scatter+gather kernel time at the bigblue2_2048 / adaptec1_2048
benchmarks is currently **slower than CPU 8T** because the V11 accumulator
suffers from extreme load imbalance: at adaptec1_2048, **one tile receives
755,526 contribs** out of 2.16M total (**38× per-core imbalance**). The
slowest receiver's accumulation time dominates total gather wall-time.

Phase 3 fixes this by **K-way replicating "hot" tiles** across multiple
owners. The scatter side round-robins contribs among the K owners; an
extra **reduce phase** sums the partial dense tiles back at the primary.

**Target metric:** V11 scatter+gather median ≥ 3× CPU 8T (kernel time
only — no host h2d/d2h, no DCT, no IPC).

| Benchmark | CPU 8T | V11 today | Phase 3 target |
|-----------|--------|-----------|----------------|
| adaptec1_2048 (NC=370K) | 28.4 ms | 36.3 ms (0.78×) | ≤ 9.5 ms |
| bigblue2_2048 (NC=1.5M) | 42.4 ms | 46.9 ms (0.90×) | ≤ 14.1 ms |

(Numbers above are **median across iterations** of full DREAMPlace E2E,
measured kernel-only. CPU 8T = `OMP_NUM_THREADS=8` baseline.)

---

## What's already built (DON'T re-do this)

### File: `experiments/density_scatter/tt_metal/host/v11_tile_ownership.h`

- `build_snake_fill_ownership(M_tiles, N_tiles, nc_all, &tile_to_core, &core_to_tiles)` — original primary partition. Unchanged.
- `build_shard_table(global_count, tile_to_core, nc_all, hot_threshold, max_k, shard_bytes, &shard_table, &per_core_shard_count)` — **NEW**. Greedy K-way assignment. Sorts hot tiles by count desc, picks K-1 alts per hot tile from cores with the lowest current shard count.
  - `shard_table[T*16+0]` = K (1 if cold, 2..MAX_K=8 if hot)
  - `shard_table[T*16+1..7]` = up to 7 alt owner uint8 IDs
  - `shard_table[T*16+8..11]` = (reserved for `hot_tile_seq` uint32; not yet populated — see "Step 1 fix" below)
  - `shard_table[T*16+12..15]` = pad
  - `per_core_shard_count[c]` = number of shard slots assigned to core c

### File: `experiments/density_scatter/tt_metal/kernels/v11_histogram.cpp`

Histogram pre-pass. NCRISC kernel that mirrors `v11_scatter_dm.cpp`'s
walk over the 8x8 (j, k) overlap grid, but instead of routing tuples,
just increments `local_count[tile_id]++`. Writes its local count array
to a per-core DRAM page at the end.

Works correctly. Used at first iter to capture global histogram.

### File: `experiments/density_scatter/tt_metal/kernels/v11_scatter_dm.cpp`

Loads `tile_to_core[]` AND `shard_table[]` from DRAM at startup. Round-
robin routing logic is **WRITTEN BUT DISABLED** — currently uses
`owner = primary` always (safety: alt owners' accum doesn't yet handle
shard slots, so contribs to alts would be silently dropped).

Look for the marked block:
```cpp
// SAFETY: shard routing disabled until accum K-way reduce
// is implemented. Otherwise contribs to alt owners get
// dropped (alts have no owned_lookup entry for the tile).
// Re-enable once Phase 3.4 (accum changes + reduce program)
// lands.
uint32_t owner = primary;
(void)shard_table;
(void)emit_counter;
```

The disabled round-robin code (apply during Step 7):
```cpp
const uint8_t* sh = shard_table + map_idx * 16u;
uint32_t K = (uint32_t)sh[0];
uint32_t owner = (K <= 1u) ? primary
                            : ((emit_counter % K) == 0 ? primary
                                                       : (uint32_t)sh[emit_counter % K]);
emit_counter++;
```

### File: `experiments/density_scatter/tt_metal/kernels/v11_accum_dm.cpp`

Standard V11 accumulator. Reads tuples from `route_buf[*][me]` per chunk,
accumulates into local dense buffer indexed by `owned_lookup[tile_idx]`,
multiplies by inv_bin_area (SCALE step), writes density_buf via NOC.
**Only handles primary tile slots today.** Needs extension (see Step 3).

### File: `experiments/density_scatter/tt_metal/host/density_scatter_ttnn_server_host.cpp`

Allocates and uses these V11/Phase 3 buffers (search for `_v11`):
- `tile_map_buf` — primary tile_to_core (uint16 per tile, 1 page total)
- `route_buf` — `nc_all × nc_all` page array, per-(writer, reader) staging
- `owned_lookup_buf` — per-core: tile_id → local_idx (uint16 per tile)
- `hist_buf` — per-core histogram counts (uint32 per tile)
- `shard_table_buf` — single page, 16 bytes per tile (see structure above)
- `shard_reduce_buf` — **allocated but unused**, sized 256 hot tiles × 8 shards × 4 KB = 8 MB. Pages indexed `hot_tile_seq * MAX_K + shard_idx`.

Per-iter execution (use_v11 branch):
```cpp
if (v11_dbg_first) {
    EnqueueMeshWorkload(cq, wl_v11_hist, false); Finish(cq);
    // ... read hist_buf, build global_count, build shard_table
    // ... upload shard_table to shard_table_buf
}
EnqueueMeshWorkload(cq, wl_v11_scatter, false); Finish(cq);
EnqueueMeshWorkload(cq, wl_v11_accum,   false); Finish(cq);
```

Hist + shard_table compute happens ONLY on first iter today (for
debug/analysis). Real production code will need to refresh every N iters
(see Step 8).

---

## What still needs to be done

### Step 1 — Compute `hot_tile_seq` (compact index for hot tiles)

**Why:** the reduce phase indexes `shard_reduce_buf` pages by
`hot_tile_seq * MAX_K + shard_idx`. We need a per-tile mapping from the
tile's global index to its compact hot-tile-only index.

**Where to add:** in `build_shard_table()` (inside `v11_tile_ownership.h`).
After identifying hot tiles, walk them in the same order they were sorted
by count and assign `hot_tile_seq = 0, 1, 2, ...`. Store at
`shard_table[T*16+8..11]` as a uint32.

**Code sketch (inside `build_shard_table`, end of function):**
```cpp
uint32_t seq = 0;
for (uint32_t t : hot_tiles) {
    uint8_t* entry = shard_table.data() + (size_t)t * shard_bytes;
    if (entry[0] >= 2) {
        // store hot_tile_seq at bytes 8..11 (little-endian uint32)
        uint32_t* slot = reinterpret_cast<uint32_t*>(entry + 8);
        *slot = seq++;
    }
}
```

After this step the host knows total hot tiles count for sizing assertions.

### Step 2 — Build per-core "shard info" lists on host (~1 hour)

**Why:** the new accum + reduce kernels need per-core knowledge of:
- Which tiles this core owns as **shard** (vs primary), and at what local_idx
- For each owned shard tile: which `(hot_tile_seq, shard_idx_in_K)` it represents
- Which of this core's primary tiles are "hot" (K > 1) and need reduce
- For each hot primary tile: K and `hot_tile_seq`

**Where to add:** in `host/density_scatter_ttnn_server_host.cpp`, right
after the `build_shard_table()` call inside the `if (v11_dbg_first)` block.

**Data structures (define near top of `if (use_v11)` block):**
```cpp
struct PerCoreShardInfo {
    // primary tiles (existing, but augment with K)
    std::vector<uint32_t> primary_tile_ids;   // tile_idx
    std::vector<uint8_t>  primary_K;          // K=1 cold, 2..8 hot
    std::vector<uint32_t> primary_hot_seq;    // valid only when K>1
    // shard tiles (NEW)
    struct ShardEntry {
        uint32_t tile_id;
        uint32_t hot_tile_seq;
        uint8_t  shard_idx_in_K;  // 1..K-1
    };
    std::vector<ShardEntry> shard_entries;
};
std::vector<PerCoreShardInfo> per_core_v11((size_t)nc_all);
```

**Population logic:**
```cpp
// Initialize primary_tile_ids/K from snake-fill (existing core_to_tiles_v11)
for (uint32_t c = 0; c < (uint32_t)nc_all; ++c) {
    for (uint32_t tile_idx : core_to_tiles_v11[c]) {
        per_core_v11[c].primary_tile_ids.push_back(tile_idx);
        const uint8_t* entry = shard_table_v11.data() + (size_t)tile_idx * SHARD_BYTES;
        per_core_v11[c].primary_K.push_back(entry[0]);
        per_core_v11[c].primary_hot_seq.push_back(
            *reinterpret_cast<const uint32_t*>(entry + 8));
    }
}

// Walk shard_table: for each hot tile, push shard entries into alts' per-core lists
for (uint32_t t = 0; t < total_tiles_v11x; ++t) {
    const uint8_t* entry = shard_table_v11.data() + (size_t)t * SHARD_BYTES;
    uint8_t K = entry[0];
    if (K < 2) continue;
    uint32_t hot_seq = *reinterpret_cast<const uint32_t*>(entry + 8);
    for (uint8_t shard_idx = 1; shard_idx < K; ++shard_idx) {
        uint8_t alt_owner = entry[shard_idx];
        per_core_v11[alt_owner].shard_entries.push_back(
            {t, hot_seq, shard_idx});
    }
}
```

### Step 3 — Update `owned_lookup_buf` to include shard slots (~30 min)

**Why:** the accum kernel uses `owned_lookup[tile_id] → local_idx` to
decide where to accumulate. After Step 2, each core may "own" tiles
both as primary AND as shard. Both need entries in owned_lookup.

**Where:** in the per-iter (or per-refresh) flow, regenerate the upload
buffer for `owned_lookup_buf`:

```cpp
std::vector<uint8_t> owned_upload_v2(owned_lookup_total, 0xFF);
for (uint32_t c = 0; c < (uint32_t)nc_all; ++c) {
    uint16_t* page = reinterpret_cast<uint16_t*>(
        owned_upload_v2.data() + (size_t)c * owned_lookup_pgsz_v11);
    auto& info = per_core_v11[c];
    uint16_t local = 0;
    for (uint32_t tile_idx : info.primary_tile_ids) {
        page[tile_idx] = local++;
    }
    for (auto& sh : info.shard_entries) {
        page[sh.tile_id] = local++;
    }
}
EnqueueWriteMeshBuffer(cq, owned_lookup_buf, owned_upload_v2, false);
Finish(cq);
```

### Step 4 — Modify `v11_accum_dm.cpp` to handle shard slots (~1-2 hours)

**Why:** dense buffer now needs slots for `n_primary + n_shard` tiles.
The kernel mostly works as-is — just ensure `n_owned` runtime arg
includes both kinds, and that owned_tile_indices runtime args list
primary tiles first, then shard tiles.

**Critical: REMOVE the SCALE step and density-write at end of accum kernel.**
Both move to the new reduce kernel. Without this change, primary cores
would write incomplete (un-summed) density values.

Find and delete these blocks at the end of `v11_accum_dm.cpp`:
```cpp
// ── Step 3: scale by inv_bin_area ────────────────────────────────────
{ DeviceZoneScopedN("V11A-SCALE"); ... }

// ── Step 4: write owned tiles to density_buf ─────────────────────────
{ DeviceZoneScopedN("V11A-DENSITY-WRITE"); ... }
```

(Keep the per-chunk read+accumulate logic intact.)

### Step 5 — Update accum runtime args + L1 sizing (~30 min)

In `host/density_scatter_ttnn_server_host.cpp`, adjust accum runtime
args to include both primary AND shard tile_ids. The existing pattern:

```cpp
std::vector<uint32_t> args = {
    ol_a, owned_lookup_pgsz_v11,
    (uint32_t)c, (uint32_t)nc_all,
    M_tiles, N_tiles_v11,
    (uint32_t)M, (uint32_t)N,
    rt_a, route_pgsz_v11, v11_max_per_page_tuples,
    da_v11, density_pgsz, inv_ba_u32,
    n_owned,  // ← change to: n_primary + n_shard
};
for (uint32_t i = 0; i < n_owned; ++i) {
    args.push_back(core_to_tiles_v11[c][i]);
}
```

becomes:
```cpp
auto& info = per_core_v11[c];
uint32_t n_total = info.primary_tile_ids.size() + info.shard_entries.size();
std::vector<uint32_t> args = {... existing ..., n_total};
for (uint32_t tile_idx : info.primary_tile_ids) args.push_back(tile_idx);
for (auto& sh : info.shard_entries)  args.push_back(sh.tile_id);
```

L1 dense buffer in accum kernel needs sizing for `max(n_primary + n_shard)`.
Update host's `v11_ac_scratch` sizing accordingly:
```cpp
uint32_t n_total_max = 0;
for (auto& info : per_core_v11) {
    n_total_max = std::max(n_total_max,
        (uint32_t)(info.primary_tile_ids.size() + info.shard_entries.size()));
}
v11_ac_scratch += n_total_max * 32u * 32u * sizeof(float); // dense
```

### Step 6 — Author `kernels/v11_reduce_dm.cpp` (~2-3 hours)

**Purpose:** runs after accum on every Tensix core. Phase A: shards
write their local dense to `shard_reduce_buf`. Phase B (host barrier
ensures sync): primary owners of hot tiles read K-1 shards from DRAM
and BRISC-add them. Phase C: scale + density write.

**Runtime args:**
```
0:  shard_reduce_dram_addr
1:  shard_reduce_pgsz   (= TILE_BYTES = 4096)
2:  density_dram_addr
3:  density_pgsz        (= nby * 4)
4:  inv_bin_area_u32
5:  M_tiles
6:  N_tiles
7:  nbx
8:  nby
9:  max_K               (= 8)
10: n_primary           (number of primary tiles)
11: n_primary_hot       (number of primary tiles with K>1)
12: n_shard             (number of shard slots at this core)
13..13+n_primary-1:           primary tile_ids (for density write)
13+n_primary..+n_primary_hot-1: per-hot-primary triples
                              (tile_id, hot_tile_seq, K) interleaved (3 args each)
... and shard entries: (tile_id, hot_tile_seq, shard_idx) triples
```

(Exact arg encoding is implementer's choice; document it in kernel header.)

**Pseudocode:**
```cpp
// Need access to dense buffer left over by accum kernel.
// IMPORTANT: dense lives in CB_SCRATCH; the reduce kernel must use
// the SAME L1 layout as the accum kernel did. Easiest path: declare
// dense_layout offsets identically in both kernels. Or have accum
// write dense to DRAM at end and reduce read it back (simpler but +DRAM I/O).

// Phase A: shards write
{
    InterleavedAddrGen<true> rgen{shard_reduce_dram, shard_reduce_pgsz};
    for (each shard_entry: shard_entries) {
        uint32_t local_idx = n_primary + entry_index;  // shard slots after primary
        uint32_t page = entry.hot_tile_seq * max_K + entry.shard_idx;
        noc_async_write(dense + local_idx*1024, rgen.get_noc_addr(page), 4096);
    }
    noc_async_write_barrier();
}

// Phase B: primaries read shards + BRISC add
{
    for (each primary_hot_tile: hot_primaries) {
        uint32_t primary_local = primary_tile_local_idx;  // 0..n_primary-1
        for (uint32_t s = 1; s < K; ++s) {
            uint32_t page = hot_tile_seq * max_K + s;
            noc_async_read(rgen.get_noc_addr(page), tmp_shard, 4096);
            noc_async_read_barrier();
            for (uint32_t i = 0; i < 1024; ++i) {
                dense[primary_local*1024 + i] += tmp_shard[i];
            }
        }
    }
}

// Phase C: scale + density write (same as removed accum code)
{
    for (uint32_t i = 0; i < n_primary*1024; ++i) dense[i] *= inv_bin_area;
    InterleavedAddrGen<true> dgen{density_dram, density_pgsz};
    for (each primary tile p) {
        for (uint32_t bxw = 0; bxw < 32; ++bxw) {
            // partial-page write: 32 floats (one row of tile, in by direction)
            uint64_t dst = dgen.get_noc_addr(tile_x*32 + bxw)
                         + (uint64_t)tile_y * 32u * 4u;
            noc_async_write(dense + p_local*1024 + bxw*32, dst, 32*4);
        }
    }
    noc_async_write_barrier();
}
```

**L1 cost:** existing dense (n_primary+n_shard tiles × 4 KB) + 1 tmp_shard (4 KB).
For 2048² with up to 38 primary + 8 shard = 46 × 4 KB = 184 KB + 4 KB tmp = 188 KB.

### Step 7 — Build `prog_v11_reduce` in host + wire per-iter (~1 hour)

In `host/density_scatter_ttnn_server_host.cpp`, add a third program after
`prog_v11_accum`:

```cpp
Program prog_v11_re = CreateProgram();
// Use the SAME CB_SCRATCH that accum used (same size, so dense layout matches)
CreateCircularBuffer(prog_v11_re, all_crs,
    CircularBufferConfig(v11_ac_scratch + 4096, {{24u, tt::DataFormat::Float32}})
        .set_page_size(24u, v11_ac_scratch + 4096));  // +4KB for tmp_shard

auto rk = CreateKernel(prog_v11_re, KDIR + "v11_reduce_dm.cpp", all_crs,
    DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,
                       .noc=NOC::RISCV_0_default});

uint32_t srb_a = (uint32_t)shard_reduce_buf->address();
for (int c = 0; c < nc_all; ++c) {
    auto& info = per_core_v11[c];
    std::vector<uint32_t> args = {
        srb_a, shard_reduce_pgsz_v11,
        da_v11, density_pgsz, inv_ba_u32,
        M_tiles, N_tiles_v11, (uint32_t)M, (uint32_t)N,
        MAX_K,
        (uint32_t)info.primary_tile_ids.size(),
        // count of hot primaries
        // count of shard entries
    };
    // append primary_tile_ids
    // append (tile_id, hot_seq, K) for each hot primary
    // append (tile_id, hot_seq, shard_idx) for each shard entry
    SetRuntimeArgs(prog_v11_re, rk, all_ccs[c], args);
}
wl_v11_reduce.add_program(device_range, std::move(prog_v11_re));
```

JIT compile + per-iter:
```cpp
EnqueueMeshWorkload(cq, wl_v11_scatter, false); Finish(cq);
EnqueueMeshWorkload(cq, wl_v11_accum,   false); Finish(cq);
EnqueueMeshWorkload(cq, wl_v11_reduce,  false); Finish(cq);  // NEW
```

### Step 8 — Re-enable shard routing in scatter kernel (5 min)

After Step 7's accum + reduce changes work, replace the safety block in
`v11_scatter_dm.cpp` with the proper round-robin (code in "What's already
built" section above).

### Step 9 — Validate (~2-4 hours)

In order:

1. **Smoke at synthetic 1.5M (no hot tiles)** — `v11_phase2_smoke.py` at
   2048²×1.5M. shard_table will be trivial (K=1). Reduce phase should
   be a no-op. rel_L2 should still pass (current: 0.32%).
2. **Smoke at synthetic with manually-clustered cells** — modify the
   smoke script's RNG to put 80% of cells in 5% of die area. Verify
   shard_table identifies hot tiles correctly. rel_L2 should pass.
3. **E2E DREAMPlace adaptec1_2048** — full run. Check final HPWL is
   reasonable (similar to CPU). Measure scatter+gather median.
4. **E2E DREAMPlace bigblue2_2048** — same. This is the hardest case.

### Step 10 — Hist refresh caching (~1 hour, optional but important)

Currently histogram runs ONLY on first iter (because each run costs
~13 ms which doubles per-iter cost). For correct DREAMPlace behavior
across all iters (cells move per iter, hot tiles drift), refresh the
histogram + shard_table every N iters (plan §3.10 recommends N=25).

In the per-iter loop:
```cpp
static uint32_t v11_iter = 0;
if (v11_iter == 0 || (v11_iter % 25) == 0) {
    // run hist, recompute shard_table, re-upload owned_lookup, re-set runtime args
}
v11_iter++;
```

Per-iter cost amortized: 13 ms / 25 = 0.5 ms — within budget.

---

## Realistic performance ceiling

Even with all of Phase 3 done correctly:

- **adaptec1_2048**: max single tile = 755K contribs. K=8 sharding gives
  per-shard owner ~95K contribs ≈ 9.5 ms accumulator work on the slowest
  core. Plus normal owned tiles (~125K typical). Ceiling: ~12-15 ms gather.
  Combined with ~10 ms scatter → **~22-25 ms scatter+gather**, vs CPU 28 ms.
  **Realistic 1.1-1.3× speedup. Not 3×.**
- **bigblue2_2048**: max tile probably ~5-10K contribs (less peaked).
  K=2-4 sharding likely sufficient. Combined target ~14-18 ms.
  **Realistic 2.5-3× speedup. May hit target.**

**If adaptec1 doesn't hit 3× after Phase 3, consider:**
- Reducing per-core overhead (V11A-SCALE on SFPU, fused programs)
- Switching to a tile-based matmul scatter (see V12_TILED_MATMUL_DESIGN.md)
- Parallelizing accumulator across BRISC + NCRISC + TRISC

---

## Pitfalls & gotchas (learned from this session)

### NOC alignment requires 64-byte cache-line care
- Adjacent 32-byte L1 destinations share a 64-byte cache line on Blackhole
- Parallel NOC writes to the same line race → garbage in odd-indexed slot
- **Fix already applied:** all V11 buffers use 64-byte page headers + 32-byte
  tuples, with 64-byte alignment between adjacent L1 regions
- 128-byte safety gap between `inbound_buf` and `dense` — keep it!

### Per-iter histogram is expensive (13 ms at 2048²)
- Don't run every iter — cache for ~25 iters
- Synthetic uniform cells produce no hot tiles; only test with real
  DREAMPlace data or artificially-clustered synthetic

### V4 SFPU floor() precision drops ~14 cells per 100K
- Small accuracy hit at low cell counts. The "2-zero-streak break"
  optimization in `v11_scatter_dm.cpp` accepts this; tested and
  verified equivalent to pure-continue at ≥300K cells.

### `n_owned` runtime arg must include BOTH primary AND shard tiles
- The accum kernel iterates `for (i = 0; i < n_owned; ++i)` to dump
  density to DRAM — if shards aren't included it'll crash
- This is the most likely source of bugs during Step 5

### Cross-core sync via `Finish(cq)` is the simplest pattern
- The plan suggests semaphores for tighter pipelining, but Finish()
  between programs is much easier to debug. Use it for V0; only
  optimize to semaphores if you measure that the launch overhead is
  the bottleneck.

---

## Code reproduction (for the new agent's first sanity check)

```bash
# Container must be running (your bh-38-special-ayadav-for-reservation-XXXXX)
docker ps  # find container name
docker start <container>  # if exited

# Build
docker exec -w /localdev/ayadav/tt-work/TTPort/experiments/density_scatter/tt_metal/build_ttnn \
  <container> bash -c 'cmake --build . --target density_scatter_ttnn_server -j$(nproc)'

# Quick smoke test (V11 today, before Phase 3 changes)
GATHER_MODE=v11 /localdev/ayadav/tt-work/TTPort/DREAMPlace/dp_env/bin/python3 \
  /localdev/ayadav/tt-work/TTPort/experiments/density_scatter/tt_metal/v11_phase2_smoke.py \
  --container <container> --grid 2048 --cells 1500000
# Expected: rel_L2 ~ 0.003, scatter ~25 ms, gather ~15 ms, PASS

# E2E DREAMPlace (V11 today)
GATHER_MODE=v11 OMP_NUM_THREADS=8 /localdev/ayadav/tt-work/TTPort/DREAMPlace/dp_env/bin/python3 \
  /localdev/ayadav/tt-work/TTPort/integration/run_dreamplace.py --device scatter_ttnn \
  --container <container> \
  --benchmark /localdev/ayadav/tt-work/TTPort/DREAMPlace/test/ispd2005/adaptec1_2048.json \
  --results-dir /localdev/ayadav/tt-work/TTPort/experiments/density_scatter/results
# Look at the resulting CSV: median scatter_ms + gather_ms ≈ 36 ms

# CPU 8T baseline (for comparison)
OMP_NUM_THREADS=8 /localdev/ayadav/tt-work/TTPort/DREAMPlace/dp_env/bin/python3 \
  /localdev/ayadav/tt-work/TTPort/integration/run_dreamplace.py --device cpu \
  --benchmark /localdev/ayadav/tt-work/TTPort/DREAMPlace/test/ispd2005/adaptec1_2048.json \
  --results-dir /localdev/ayadav/tt-work/TTPort/experiments/density_scatter/results
# CSV's scatter_ms = scatter+gather combined on CPU, median ≈ 28 ms
```

After your changes, the same E2E run on adaptec1 should give a much
lower scatter+gather median.

---

## Key file inventory

```
experiments/density_scatter/
├── FAST_SCATTER_GATHER_PLAN.md      Original V11 design (read for context)
├── V11_PHASE2_STATUS.md             Phase 2 results (V11 baseline)
├── V11_PHASE3_HANDOFF.md            ★ THIS FILE
├── V12_DESIGN.md                    Deferred V12 tile-based design
├── V12_TILED_MATMUL_DESIGN.md       Alt approach: tile-based matmul scatter
└── tt_metal/
    ├── kernels/
    │   ├── v4_compute.cpp           SFPU per-cell geometry (REUSE)
    │   ├── v4_reader.cpp            SoA cell reader (REUSE)
    │   ├── v11_scatter_dm.cpp       V11 scatter NCRISC (modify Step 8)
    │   ├── v11_accum_dm.cpp         V11 accum BRISC (modify Step 4)
    │   ├── v11_histogram.cpp        Phase 3 hist (DONE, REUSE)
    │   └── v11_reduce_dm.cpp        ★ TO CREATE in Step 6
    ├── host/
    │   ├── v11_tile_ownership.h     Snake-fill + shard_table (modify Step 1)
    │   └── density_scatter_ttnn_server_host.cpp
    │                                 Main host (modify Steps 2,3,5,7,8,10)
    └── v11_phase2_smoke.py          Single-iter smoke test (REUSE for validation)

integration/
└── scatter_ttnn_client.py           Decoder dict has v11=5 (DONE)
```

Estimated total effort: **6-12 hours** of focused implementation +
validation. Checkpoint after each Step in this doc.

## 2026-05-08 status & post-implementation findings

Phase 3 (Steps 1-9) was implemented and validated end-to-end. Steps 1-9
(scatter routing, accum K-aware, two-program reduce, build_shard_table,
runtime args, etc.) all work; rel_L2 ≤ 0.003 on synthetic 1.5M, no hangs
on adaptec1_2048 / bigblue2_2048. **Step 10 (hist refresh) is still
deferred and is now the critical missing piece — see findings below.**

### Pitfall 1: `Finish()` overhead is NOT 8-10ms — it's <1ms

The 3-program reduce path (accum + reduce_a + reduce_bc, separated by host
`Finish()`s) was assumed to cost ~16-20 ms in extra Finish overhead. We
merged the three kernels into one using NOC semaphores
(`noc_semaphore_inc` + `noc_semaphore_wait`, see `v11_accum_dm.cpp`). The
merge was correct (passes smoke + E2E) but yielded **no speedup** — gather
went from 26.1 ms → 26.4 ms. So each `Finish()` costs only ~0.1 ms on
Blackhole at this scale; do not bother chasing it.

The merged design is still cleaner architecturally and removes 2 host
round-trips per iter. It's left in place. The semaphore pattern (one
sem_id slot per hot tile per core, allocated as multiple
`CreateSemaphore()` calls) is a useful template for future cross-core
in-program sync.

### Pitfall 2: shard_table is iter-0-stale → Phase 3 doesn't help

Profiler (`TT_METAL_DEVICE_PROFILER=1`, CSV at
`<TT_METAL_HOME>/generated/profiler/.logs/profile_log_device.csv`) on
adaptec1 shows:
- **`V11A-ACC` zone is the bottleneck**: ~21 ms on the slowest core (1,7),
  vs <2 ms median.
- All other zones combined sum to <4 ms.

The slowest core (1,7) is not getting its hot tile sharded, even though 4
tiles ARE being sharded with K=8. Why: the histogram is run only at iter
0 with random initial cell positions, but DREAMPlace cells migrate during
optimization. By iter 30+, the actual hot tiles have moved. Phase 3
shards the iter-0 hot tiles (now cold) and leaves the actual hot tiles
unsharded.

**Implication:** Phase 3 sharding without Step 10 (periodic hist refresh)
is dead weight on real DREAMPlace runs. End-to-end results (median
scatter+gather kernel time, full DREAMPlace iters):
- adaptec1 V11 = 38 ms vs CPU 28 ms → 0.74×
- bigblue2 V11 = 50 ms vs CPU 42 ms → 0.84×

(Both worse than CPU; both worse than Phase 2 V11 on uniform synthetic
data.)

### Pitfall 3: snake-fill can put 2 hot primaries on one core

`build_shard_table` picks primary owners from `tile_to_core` (snake-fill).
Adjacent tiles often go to the same core, and on adaptec1 two hot tiles
landed on core 53. The original kernel design assumed ≤1 hot primary per
core (single semaphore). Fix: allocate H semaphore slots where H = max
hot primaries per core; map each hot tile to a sem_id slot in a
core-local order. The current `v11_accum_dm.cpp` does this.

### Recommended next steps

1. **Step 10 (hist refresh)** is now THE priority. Approach: every N iters
   (e.g., 25), enqueue `wl_v11_hist`, read back, recompute shard_table,
   update owned_lookup_buf, and re-set runtime args on the existing
   `wl_v11_accum` (no JIT — the kernel binary doesn't need changing,
   only the runtime arg block + DRAM lookup buffer). Generously
   pre-allocate the CB at startup so any `n_total_max` fits without
   resize.
2. **If Step 10 alone isn't enough**, attack `V11A-ACC` directly:
   manual-unroll the inner loop, drop redundant bounds checks (owned_lookup
   already encodes ownership), or restructure dense layout for better cache
   reuse. Multiple BRISC NOC ports could parallelize tuple reads with
   accumulate, but that's a bigger rewrite.
3. **Lowering HOT_THRESHOLD** (e.g., 5000 → 1000) speculatively shards
   more tiles. Cheap to try but increases shard_reduce_buf pressure and
   per-core max_K. Validate accuracy and CB sizing.

### Profiler usage notes

- Build is already Tracy-enabled, so just set `TT_METAL_DEVICE_PROFILER=1`
  on any run. Output CSV at the path above. `run host ID` column is `0`
  for kernel zones (not unique per call), so derive per-call timings by
  counting events and dividing.
- The Blackhole device clock is 1.35 GHz; cycles ÷ 1350 = µs.
- `DeviceZoneScopedN("NAME")` in kernels emits ZONE_START/ZONE_END events
  paired by core+RISC. See `/tmp/profile_v11.py` (helper script that
  aggregates per-zone max-across-cores per kernel call) for an example.
