# V18 HASHAGG — Scatter-side Pre-Aggregation Handoff

**Status**: design complete, implementation pending
**Owner**: TBD (incoming agent)
**Created**: 2026-05-21
**Predecessor work**: V11 baseline (shipped), V11outer / Stage B' (shipped 2026-05-20)

---

## 0. TL;DR — What you're building

A new V11 variant called **V18** (gather mode flag: `GATHER_MODE=v18`) that adds **per-source hash-table aggregation** on the scatter side. Every (bx, by, area) tuple is summed into a hash table keyed by (bx, by) inside the producing Tensix core **before** it is written to the DRAM route_buf. The receiver gather kernels stay unchanged.

**Expected outcome on adaptec1_2048**:
- Tuples emitted: **4.65M → ~2.4M (~50% reduction)**
- Heavy-call gather time: 20 ms → 5-7 ms per call
- Total per-run wall-time: **−2 to −4 seconds saved**
- HPWL drift: within ±0.5% (DREAMPlace tolerance)
- 18/18 configs must still converge

**Why it works**: V11 emits 1 tuple per (cell, bin) overlap, then routes 4.65M tuples to receivers. Hot bins receive ~1330 tuples each from across the chip. If each scatter source first sums local duplicates per (bx, by), receivers see at most ~220 tuples per hot bin (110 sources × 2 RISCs) — a 6× reduction at the bottleneck. Total tuple count drops ~50% because of the heavy hot-tile contributions.

---

## 1. Required reading (in order)

Before writing any code, read these. They contain the actual rationale and prior outcomes; this doc only summarizes.

1. **`docs/V11_PHASE3_HANDOFF.md`** — V11 architecture (scatter+gather flow, route_buf layout, tile_to_core ownership)
2. **`docs/V11_V13_GATHER_HANDOFF.md`** — original V11 gather design + variants explored
3. **Memory note** (Claude context): `~/.claude/projects/.../memory/v11_gather_sharding_investigation.md` — why K=8 sharding was disabled by default 2026-05-21; the convergence-path issue
4. **Memory note**: `~/.claude/projects/.../memory/v11_stage_bprime_sweep_outcome.md` — Stage B' (v11outer) sweep methodology and results template; use the same approach for V18
5. **Memory note**: `~/.claude/projects/.../memory/v11_gather_skew_diagnostic.md` — the diagnostic tool and per-config skew measurements that motivated this work
6. **Source code**: `kernels/v11_scatter_b_dm.cpp` (BRISC, ~360 lines) — this is the file you'll copy and modify
7. **Source code**: `kernels/v11_scatter_dm.cpp` (NCRISC) — sibling file, same routing structure as BRISC
8. **Source code**: `kernels/v11_accum_dm.cpp` / `v11_accum_n_dm.cpp` — gather side, DO NOT MODIFY. V18 reuses them verbatim.

---

## 2. Context — Why hash aggregation, and what was already tried

### What we already know about gather skew

The histogram dump on adaptec1_2048 (run `GATHER_MODE=v11` with logging) showed:

```
V11 hist: total=4654334 tuples, max_per_tile=1362381
Top-20 per-tile counts:
  1362381  603099  483705  215306  769  728  722  721  714 712 704 704 703 700 697 696 695 692 691 691
                                   ^ "plateau" at ~500-700
4 mega-tiles                       4092 cold tiles

tiles >5000 = 4   tiles >1000 = 4   tiles >500 = 1688   tiles >100 = 4096
```

The distribution is brutally **bimodal**: 4 mega-tiles (carrying 57% of all tuples) and 4092 cold tiles each with ~500-700 tuples.

### What didn't work

- **K=8 hot-tile sharding** (the existing `build_shard_table()` machinery): only catches the 4 mega-tiles. SHARD-SUM has serial DRAM reads + scalar BRISC adds. As cells migrate during DREAMPlace, the iter-0 shard table goes stale and the small fp32 reorder it causes pushes DREAMPlace onto longer convergence paths. **Net: −86 ms gather but +5-30 sec wall time on 5 of 6 large configs**. Disabled by default 2026-05-21 (default `HOT_THRESHOLD=100,000,000`).
- **Periodic shard refresh**: blocked by TT-Metal sem-leak in the refresh path (every refresh allocates fresh semaphores, exceeding the 16-per-core limit after ~4 refreshes).
- **Hot-tile-only dense pre-agg** (Approach 3 hot-tile variant): would deliver ~80% of the win at 10% of the L1 cost. Less flexible than hash-agg because it requires upfront hot-tile identification via histogram and degrades as hot tiles migrate.

### Why hash-agg is the chosen path

- **Adaptive**: doesn't need upfront knowledge of hot tiles. Adapts to wherever density is concentrated each iter.
- **Graceful**: if a bin is touched only once, hash adds an entry and emits it — no penalty, no correctness risk.
- **No staleness**: the table is rebuilt every scatter call, so no "outdated state" problem like sharding had.
- **L1 fits comfortably**: ~256 KB per RISC for a 32K-slot table.
- **Reduces TOTAL tuple count**, not just balance — the gather kernel does less work overall, not the same work spread differently.

---

## 3. Architecture — the V18 design

### What changes vs V11

| Component | V11 | V18 |
|-----------|-----|-----|
| BRISC scatter (`v11_scatter_b_dm.cpp`) | Routes each tuple immediately | **Hashes into local table; flushes at end** |
| NCRISC scatter (`v11_scatter_dm.cpp`) | Routes each tuple immediately | **Hashes into local table; flushes at end** |
| Compute kernel (`v4_compute.cpp`) | unchanged | **unchanged** |
| Gather (`v11_accum_dm.cpp`, `v11_accum_n_dm.cpp`) | unchanged | **unchanged** — same tuple format on the wire |
| Host (`density_scatter_ttnn_server_host.cpp`) | unchanged | Add `GATHER_MODE=v18` branch + L1 budget guard |

**Critical invariant**: the tuple format on the route_buf wire stays exactly the same as V11 (header + sorted (bx, by, area) entries). The receiver gather has no idea pre-aggregation happened. This is what makes the change low-risk.

### Per-source flow (V18 BRISC, simplified)

```
For each cell in this source's batch:
    Compute ox/oy/bx/by/area as in V11
    For each (j, k) overlap:
        bx = bxl + j; by = byl + k
        area = ox[j] * oy[k]
        key = (bx << 16) | by         // pack into uint32
        slot = find_or_insert(hash_table, key)
        hash_table[slot].area += area
        hash_table[slot].key = key

After all cells processed (end-of-batch flush):
    For each non-empty slot in hash_table:
        bx = slot.key >> 16; by = slot.key & 0xFFFF
        tile_idx = (bx >> 5) * N_tiles + (by >> 5)   // 32-bin tiles
        owner = tile_to_core[tile_idx]
        Append (bx, by, area) to staging[owner]
        If staging[owner] full: flush to route_buf[me][owner]
    Final flush all non-empty staging buffers

Reset hash_table for next call
```

NCRISC does the symmetric thing on its half of cells, in its own separate hash table. Each source-core thus emits two independent compacted streams (BRISC + NCRISC), at most 2 tuples per (bx, by) per source.

---

## 4. Hash table design (key decisions)

### Key, value, layout

- **Key**: `uint32_t` = `(bx << 16) | by`. Both `bx` and `by` fit in 16 bits for grids up to 65536 (we use 2048, plenty of room).
- **Empty sentinel**: `0xFFFFFFFF`. Valid keys can never collide with this since both bx,by < 65535.
- **Value**: `float` (4 bytes) for the accumulated area.
- **Slot size**: 8 bytes (`uint32_t key + float area`).

### Sizing

Per-source unique-bin count estimate (from histogram math):
- Worst case (all cells touch unique bins): ~42K unique bins per source
- Realistic (with mega-tile fan-in): ~21K unique bins per source after dedup

Use a table sized to keep load factor < 0.5 to bound linear-probe length:
- **32768 slots × 8 bytes = 256 KB per RISC**
- Load factor at 21K entries = 0.64 (acceptable for linear probing)
- Worst case at 42K entries = 1.28 (table full — would need 65K slots)

**Recommended**: start with 32K slots = 256 KB. Add an env-tunable size (`V18_HASH_BITS`, default 15 → 32K slots) so we can experiment.

Two separate tables per Tensix core (one for BRISC, one for NCRISC): **512 KB per core total**. L1 is 1 MB per core, so there's room for other buffers but it's tight. See section 5.

### Collision handling

**Linear probing.** Simpler than chaining (no pointers, no allocation), cache-friendly:

```cpp
inline uint32_t find_or_insert(uint32_t* keys, float* areas, uint32_t key, uint32_t mask) {
    uint32_t h = hash32(key) & mask;
    while (true) {
        uint32_t k = keys[h];
        if (k == key) return h;                  // hit
        if (k == 0xFFFFFFFF) {                   // empty slot — insert
            keys[h] = key;
            areas[h] = 0.0f;
            return h;
        }
        h = (h + 1) & mask;                      // probe forward
    }
}
```

Hash function: use a simple fast mixer (avoid expensive multiplies):
```cpp
inline uint32_t hash32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x85ebca6b;
    x ^= x >> 13;
    return x;
}
```

### Reset between scatter calls

After flushing, the table must be cleared for the next call. Two options:
- **Track inserted slots in a small `dirty[]` list**, walk and clear only those. ~4 µs for 21K entries.
- **memset the whole `keys[]` array to 0xFF**: 32K slots × 4 bytes = 128 KB clear. At ~30 GB/s L1 bandwidth = ~4 µs. Equivalent cost, simpler code.

**Recommended**: memset approach. Cleaner and avoids the "dirty list" data structure.

---

## 5. L1 budget — critical to validate before building

V11 already uses substantial L1 per core. Adding 512 KB hash tables (2× 256 KB) is risky. Run this math first:

### V11 scatter L1 usage today (per core, approximate)

```
tile_map (tile_to_core):                  ~8 KB
shard_table copy:                         64 KB  (now ~unused with sharding off)
cell px/py/sx/sy temp:                    16 KB
BRISC scatter staging + counts + offsets: ~30 KB
NCRISC scatter staging + counts:          ~30 KB
ox/oy buffers (2 tiles bf16):              4 KB
Other scratch:                            ~30 KB
TOTAL V11 scatter use:                   ~180 KB
```

V18 ADDS:
```
BRISC hash table:    256 KB
NCRISC hash table:   256 KB
TOTAL V18 ADDITION:  512 KB
```

**Grand total V18 scatter**: ~700 KB out of 1024 KB L1. **Tight but feasible**.

Knobs if you run out:
- Reduce hash table to 16K slots (128 KB each, 256 KB total) — accept higher collision rate
- Use BF16 area (2 bytes instead of 4) — table slot becomes 6 bytes, 32K × 6 = 192 KB each. **Note**: user has empirically verified BF16 area accumulation does not hurt DREAMPlace convergence (from their low-precision DREAMPlace experiments). This is your fallback if 256K is too tight.
- Drop shard_table copy if sharding stays disabled (~64 KB freed)

**Action item**: before writing kernel code, modify `v12_l1_scratch_bytes()` or write a new helper to print V18's L1 usage at host startup and FATAL if > 1 MB.

---

## 6. Implementation plan (in order)

### Phase 1: kernel + host scaffolding

1. **Create new kernel files** as copies of V11 originals:
   - `kernels/v18_scatter_b_dm.cpp` ← copy of `v11_scatter_b_dm.cpp`
   - `kernels/v18_scatter_dm.cpp` ← copy of `v11_scatter_dm.cpp`
   - **Do not modify** `v11_*.cpp` files (preserve V11 baseline).

2. **Add hash table to each kernel**:
   - Reserve L1 space (in CB_SCRATCH or a dedicated CB index) for `uint32_t keys[32768]` + `float areas[32768]`
   - At kernel start, memset keys to 0xFF
   - Replace the tuple-emission section (where V11 calls `flush_recv` per tuple) with hash-table insert
   - At end-of-batch, walk the table and flush per-owner staging buffers (reuse V11's existing flush logic)

3. **Add `GATHER_MODE=v18` branch to host**:
   - In `host/density_scatter_ttnn_server_host.cpp` around line 432 (next to `v11outer` branch)
   - Set `use_v18 = true; use_v11 = true; use_v11outer = false`
   - Use ternary to pick `v18_scatter_b_dm.cpp` / `v18_scatter_dm.cpp` kernel file
   - L1 budget check (see section 5)

4. **Add env-var forwarding** in `integration/scatter_ttnn_client.py:354+`:
   - `V18_HASH_BITS` (default 15 → 32K slots)
   - `V18_BF16_AREA` (default 0; if 1, use BF16 area for tighter L1)

5. **Build**:
   ```bash
   docker exec -w /localdev/ayadav/tt-work/TTPort/DREAMPlaceTT-Framework \
     bh-38-special-ayadav-for-reservation-75063 \
     bash -c "TT_METAL_HOME=\$(pwd)/tt-metal bash scripts/build_server.sh -j 16"
   ```

### Phase 2: smoke test on adaptec1_512 (smallest, cheapest)

```bash
TT_METAL_HOME=$(pwd)/tt-metal CPU_DCT=1 GATHER_MODE=v18 \
  DREAMPlace/dp_env/bin/python3 integration/run_dreamplace.py \
    --device scatter_ttnn --container bh-38-special-ayadav-for-reservation-75063 \
    --benchmark benchmarks/configs/sweep_adaptec1_512.json \
    --results-dir /tmp/v18_smoke_512
```

**Success criteria**:
- No FATAL or crash
- HPWL within ±2% of V11 baseline (V11 at adaptec1_512: HPWL=70,332,032)
- scatter_ms doesn't explode (acceptable: up to 2× V11's scatter)
- gather_ms unchanged or smaller

### Phase 3: bigger config + Tracy profile (adaptec1_2048)

```bash
TT_METAL_HOME=$(pwd)/tt-metal CPU_DCT=1 GATHER_MODE=v18 \
  TT_METAL_DEVICE_PROFILER=1 \
  DREAMPlace/dp_env/bin/python3 integration/run_dreamplace.py \
    --device scatter_ttnn --container bh-38-special-ayadav-for-reservation-75063 \
    --benchmark benchmarks/configs/sweep_adaptec1_2048.json \
    --results-dir /tmp/v18_2048
```

Analyze with existing tools:
```bash
DREAMPlace/dp_env/bin/python3 tools/v11_shard_diag.py \
  --profile /tmp/v18_2048/profile_log_device.csv \
  --label "V18 hashagg @ adaptec1_2048"
```

**Look for**:
- V11A-ACC time on critical core should drop (less work because fewer tuples)
- Total per-call gather time on heavy iters should drop 30-50%
- Scatter wall on heavy iters may rise but less than gather drops

### Phase 4: 18-config sweep + comparison

Use `scripts/run_v11outer_sweep.sh` as a template. Modify to run V11 baseline + V18:

```bash
# pseudocode
for cfg in $CONFIGS; do
  run_one v11 $cfg     # already-known baseline
  run_one v18 $cfg     # new path
done
```

Run `tools/v11outer_compare.py` adapted for V11 vs V18.

**Success criteria for full sweep**:
- **HPWL convergence**: 18/18 within ±2% of V11 baseline; no diverged runs (HPWL ≠ -inf, overflow < 0.10)
- **Gather speedup**: ≥30% reduction in `gather_ms_median` on heavy configs (adaptec1_2048, adaptec2_2048, bigblue1_2048, bigblue3_2048)
- **End-to-end wall time**: net negative across all 18 configs; significant wins on 2048 configs
- **No regression on 512 configs**: scatter overhead from hash table should be small enough that 512 configs are within noise

---

## 7. Critical gotchas (read all of these)

### 7.1 Env-var forwarding bug

The biggest landmine: **`integration/scatter_ttnn_client.py` has an env-var allowlist** at line ~354 (`docker exec -e ...`). Any new env var must be added there or it silently won't reach the server. We hit this 2026-05-21 with `V11_HOT_THRESHOLD` — the entire prior sharding sweep was invalid because the env var was never propagating.

**Always verify**: `strings host/build/density_scatter_ttnn_server | grep V18` shows your env-var strings AND a run logs the actual values you set (add a startup printf with the parsed env-var values).

### 7.2 L1 budget — fail loudly, don't silently corrupt

V11 already has a FATAL at:
```cpp
if (n_total_max_v11 > n_owned_max + V11_CB_SLOT_HEADROOM) {
    printf("[server] FATAL: ...");
    fflush(stdout); std::exit(1);
}
```

Add the same for V18. If hash table + V11 buffers exceed 1024 KB:
```cpp
if (v18_l1_total > 1024u * 1024u) {
    printf("[server] FATAL: V18 L1 use %u KB exceeds 1 MB\n", v18_l1_total/1024);
    std::exit(1);
}
```

### 7.3 BF16 area drift validation

If you switch to BF16 area for tighter L1, validate carefully:
- Each `area = ox * oy` is fp32 from compute
- Cast to bf16 BEFORE summing in hash table
- Each per-source partial sum has ~12 additions max (avg 6) — bf16 mantissa loss per add is tiny
- HPWL should still match within 2%

User has prior empirical evidence this works (their low-precision DREAMPlace version). Cite when committing.

### 7.4 Sharding interaction — disable explicitly

Current default has sharding OFF (HOT_THRESHOLD=100M), but the routing code still consults `shard_table[]` and computes K=1 paths. **For V18, just remove the shard-table consultation entirely** in the scatter kernel — it's dead code that adds L1 cost and routing-loop cost.

Specifically: in the V18 kernel, the routing decision is purely `owner = tile_to_core[tile_idx]`. No K, no alts, no shard_idx_in_K, no emit_counter.

### 7.5 V11 emit_counter — remove it

`v11_scatter_b_dm.cpp` has `uint32_t emit_counter = 0;` and uses it for sharding parity. **V18 doesn't need this** — remove it. Each tuple goes through the hash table; the "which owner" decision happens at flush time from tile_to_core, not from tuple-index parity.

### 7.6 Per-tuple cost — keep it low

Linear-probe hash insert is ~5-15 ns/tuple (a few L1 loads). V11's current per-tuple flush is ~30 ns (write to staging + occasional barrier). So V18 scatter side should be SLIGHTLY faster per tuple, not slower.

If your hash table feels slow, the usual culprits:
- Hash function too expensive (avoid 64-bit multiplies on BRISC)
- Table size too small → long probe chains
- Cache misses (table > L1 cache region) — should not happen at 256 KB

### 7.7 Flush order doesn't matter

After hash table fills, you walk it linearly and emit tuples in arbitrary order. **This is fine** — the receiver gather kernel doesn't care about tuple order; it just accumulates each (bx, by, area) into the right dense bin.

V11 already does sort + dedup in `flush_recv()` (`v11_scatter_b_dm.cpp:flush_recv`). For V18 you can keep the sort+dedup (cheap, ~few µs, and ensures DRAM-write coalescing) or skip it (gather doesn't need it). **Recommend: keep sort+dedup** for DRAM write efficiency.

### 7.8 Two RISCs, two tables

BRISC and NCRISC scatter handle DIFFERENT cells (split by cell index range). They do not share data. **Each gets its own hash table in a separate L1 region.** Do not try to share — fp32 atomicity across Tensix RISCs is not guaranteed.

After flush, each emits to its own writer-row in route_buf. Receiver core reads from both BRISC writer-row and NCRISC writer-row of each source. V11's existing gather (BRISC accum reads NCRISC's stuff via V11A-MERGE, etc.) handles this correctly.

### 7.9 HPWL tolerance is what really matters

V18 will produce slightly different HPWL than V11 because (a) BF16 (if used) and (b) different float addition orders. Acceptable drift is ±2% — same tolerance V11outer met (it was bit-exact, but DREAMPlace's tolerance is much looser).

**Use the same convergence criteria as the V11outer sweep**:
- HPWL > 0 (not -inf or NaN)
- Final overflow < 0.10

A 2-3% HPWL improvement is also fine — sometimes the reorder lands DREAMPlace in a better basin (we saw this with sharding-off where adaptec2_2048 got 2.5% better HPWL).

### 7.10 Don't break V11 / v11outer

V18 must be a SIBLING, not a replacement. Existing modes (`GATHER_MODE=v11`, `v11outer`, `v11outer_auto`) must keep working unchanged. This is the same constraint as Stage B' — see `kernels/v11_outer_scatter_b_dm.cpp` for the sibling-file pattern.

---

## 8. Existing tools the new agent should know about

### Diagnostic
- `tools/gather_skew_diag.py` — per-core gather busy time + skew ratio
- `tools/v11_shard_diag.py` — per-zone (V11A-ACC, V11N-ACC, V11A-MERGE, etc.) decomposition
- `tools/plot_v11_scatter_timeline.py` — per-RISC Gantt timeline (use for V18 scatter visualization too)
- `tools/plot_v11_accum_slow_core_timeline.py` — accum/gather timeline

### Comparison
- `tools/v11outer_compare.py` — sweep comparison report; adapt for V11 vs V18
- `scripts/run_v11outer_sweep.sh` — 18-config sweep script template

### Histogram dump
At iter 0, the host now logs:
```
V11 top-20 per-tile counts: ...
V11 tile-count percentiles: p99=X p95=X p90=X p75=X p50=X p25=X
V11 tiles-above-threshold: >10000=X  >5000=X  >2000=X  ...
```
Use this to understand the tile distribution for any new design under test.

---

## 9. Convergence targets (acceptance criteria)

Run the full 18-config sweep with V11 baseline (current `GATHER_MODE=v11` with sharding off) and V18. All these must hold for shipping V18:

| Config | V11 baseline HPWL | V18 HPWL tolerance | V11 baseline gather_ms | V18 target gather_ms |
|--------|-------------------|---------------------|------------------------|----------------------|
| adaptec1_512 | 70,332,032 | within ±2% | 2.85 | ≤ 2.85 (acceptable: small regression) |
| adaptec1_1024 | 70,855,544 | ±2% | 5.95 | ≤ 5.95 |
| adaptec1_2048 | 71,329,584 | ±2% | 7.14 | **≤ 4.0 (~45% reduction target)** |
| adaptec2_2048 | 82,903,552 | ±2% | 9.86 | ≤ 6.0 |
| adaptec3_2048 | 188,280,000 | ±2% | 9.54 | ≤ 6.5 |
| bigblue1_2048 | 88,488,416 | ±2% | 10.36 | ≤ 6.0 |
| bigblue2_2048 | 132,991,488 | ±2% | 10.40 | ≤ 7.0 |
| bigblue3_2048 | 291,774,784 | ±2% | 12.91 | ≤ 8.0 |
| (all 512/1024 configs) | (see V11 baseline) | ±2% | (V11 baseline) | within ±10% |

**Mandatory**:
- 18/18 configs must converge (no diverged HPWL=-inf or overflow > 0.10)
- HPWL drift can be in either direction (sometimes better)
- Total end-to-end wall_time per config: V18 ≤ V11 baseline + 5% (i.e., no significant regression on any config); ideally ≥ 10% faster on 2048 configs

---

## 10. Test infrastructure

### Hardware
- **Container**: `bh-38-special-ayadav-for-reservation-75063` (Blackhole BH-38, exclusive to this work)
- **Chip ownership check**: before running `tt-smi -r`, verify no other containers exist with `docker ps`. Only reset if you own the chip.
- **JIT cache**: `/root/.cache/tt-metal-cache` (inside container). Don't clear unless explicitly recovering from a hang.

### Environment
- `TT_METAL_HOME=$(pwd)/tt-metal` (always — the framework has its own tt-metal vs outer tt-metal; outer doesn't work; see memory note `tt_metal_home_abi`)
- `CPU_DCT=1` (required for V11 family — bypasses density add-back issue; see memory note `cpu_dct_required_for_v11`)
- `GATHER_MODE=v18`
- `TT_METAL_DEVICE_PROFILER=1` (for Tracy capture during analysis only; adds ~5% runtime)

### Reset
```bash
docker exec bh-38-special-ayadav-for-reservation-75063 bash -c "tt-smi -r"
# Only when truly stuck — see ownership check above
```

---

## 11. Decision tree if you hit problems

### "Build fails"
- Check `docker exec ... whoami` returns root (container build needs root)
- Check `TT_METAL_HOME` points to the framework's tt-metal, NOT outer
- Permissions on `host/build/CMakeCache.txt` — sometimes need root rebuild after host-side cmake

### "Kernel JIT fails or crashes"
- Print L1 budget at startup before running first iter
- Check `printf` of hash table size matches your expectation
- Look for `noc_async_*_barrier` calls — missing barriers cause silent data corruption
- Tracy CSV won't write if kernel SEGVs — check `dmesg` and `journalctl`

### "HPWL diverges (becomes -inf or extreme)"
- Most likely: hash collision logic has a bug; same key produces different slots → some tuples drop
- Quickest debug: disable hash table (always insert at slot 0 — basically V11 baseline) → if HPWL still diverges, problem isn't the hash table; if HPWL converges, the hash logic is broken
- Add per-emit assertion: `assert(staging_count[owner] <= MAX_IN_FLIGHT)` (V11 already has this; V18 should keep it)
- Check that `memset(keys, 0xFF, ...)` runs every scatter call (forgot reset → tuples accumulate across calls → catastrophe)

### "Gather time doesn't drop"
- Tracy V11A-ACC time per core: should drop ~50% on heavy iters. If not dropping, either fewer tuples aren't actually being emitted (hash table bug) OR the gather kernel has another bottleneck (header read, density write).
- Count emitted tuples per scatter call — print `total_emitted` per source and verify it's ~50% of V11 baseline count.

### "Scatter time explodes"
- Hash table probing chain too long — increase table size or check hash function
- L1 cache misses if table is too big — 256 KB is right at the boundary
- BF16 cast overhead per tuple (if BF16 enabled) — should be ~1 ns per cast

### "Specific config regresses"
- Compare its histogram dump (top-20 per-tile counts) — designs with very uniform distributions (e.g., 4096 cold tiles all ~500 tuples) gain less because there are fewer duplicates to merge
- This is expected for very-cold configs (e.g., adaptec3_512). Don't try to fix; document it.

---

## 12. Recommended order of operations

Week 1:
- [ ] Day 1-2: Read all required docs + memory notes. Run V11 baseline on adaptec1_2048 and reproduce the histogram + Tracy timeline.
- [ ] Day 2-3: Implement Phase 1 (kernel scaffolding + host branch). Build successfully but don't run yet.
- [ ] Day 3-4: Phase 2 smoke test on adaptec1_512. Debug crash / HPWL divergence until convergent.
- [ ] Day 4-5: Phase 3 on adaptec1_2048. Capture Tracy. Verify per-zone reduction.

Week 2:
- [ ] Day 1-2: Phase 4 18-config sweep. Generate comparison table.
- [ ] Day 3: BF16 area variant + sweep. Decide which to ship.
- [ ] Day 4-5: Documentation + memory notes update. Write a `V18_HASHAGG_OUTCOME.md` summarizing the sweep results in the same format as the V11outer outcome memory.

If at any point HPWL won't converge, escalate to user before deep-debugging — the user's empirical knowledge of DREAMPlace convergence behavior is valuable.

---

## 13. Files you'll touch (summary)

**New files (create)**:
- `kernels/v18_scatter_b_dm.cpp`
- `kernels/v18_scatter_dm.cpp`
- `docs/V18_HASHAGG_OUTCOME.md` (at end, summarizing results)
- `scripts/run_v18_sweep.sh` (copy of `run_v11outer_sweep.sh`, adapted)

**Modified files**:
- `host/density_scatter_ttnn_server_host.cpp` — add `GATHER_MODE=v18` branch + L1 budget guard
- `integration/scatter_ttnn_client.py` — forward `V18_HASH_BITS`, `V18_BF16_AREA` env vars

**Read-only (do NOT modify)**:
- `kernels/v11_*.cpp` — V11 baseline must keep working
- `kernels/v4_compute.cpp` — compute kernel reused as-is
- `kernels/v11_accum_dm.cpp`, `v11_accum_n_dm.cpp` — gather reused as-is

**Reference (read for understanding)**:
- `host/v11_tile_ownership.h` — tile_to_core builders
- `host/CMakeLists.txt` — make sure new kernel files don't need a build-system change (kernels are JIT-compiled, just need to exist on disk in `kernels/` dir)

---

## 14. Context you'd otherwise miss

- **Why we care about this**: the chip is Blackhole BH-38, DREAMPlace is the workload, the goal is to beat V11's wall time on 18 ISPD2005 benchmarks. We have a 2× target over V11 on dense-2048 configs and 3× over CPU. Stage B' (v11outer) shipped 2026-05-20 with ~−4% sc+ga via per-grid routing. V18 hashagg is the next move targeting the gather side.
- **User's working style**: doesn't want time estimates ("how long will this take"), wants results. Communicate progress in concrete terms ("kernel compiles, HPWL converges on adaptec1_512, see comparison table") not estimates.
- **Sharding is DEAD**: the K=8 sharding code path is still in the codebase but disabled by default (HOT_THRESHOLD=100M). V18 should NOT integrate with it. The shard_table consultation in scatter routing should be removed entirely from V18's kernel files.
- **Don't break V11**: this is a hard constraint. Sibling files only, never modify V11 originals.

---

## 15. Open questions / decisions to make during implementation

1. **Hash table size**: 32K (256 KB) vs 16K (128 KB)? Empirically test with a small sweep early.
2. **BF16 vs fp32 area**: ship with fp32 by default; verify BF16 works as a fallback if L1 budget needs it.
3. **Sort + dedup before route_buf write**: keep V11's existing `flush_recv` sort+dedup (cheap and good for DRAM coalescing) or skip it (gather doesn't need it)? Recommended: keep.
4. **NCRISC hash table**: implement in V1 or defer to V2? Recommended: V1 (full BRISC + NCRISC) for clean comparison. The complexity is symmetric — if you wrote the BRISC version, NCRISC is just a copy.
5. **Threshold for "this hash table is too full"**: if load factor exceeds 0.7, log a warning. Indicates need for larger table.

Document your decisions in `V18_HASHAGG_OUTCOME.md` at the end so the next-next agent knows what was tried.

---

*Last updated: 2026-05-21. Created by Claude Opus 4.7 after V11 sharding investigation + hash-agg design discussion with user.*
