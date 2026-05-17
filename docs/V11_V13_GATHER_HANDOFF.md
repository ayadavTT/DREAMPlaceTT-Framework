# Gather-Speedup Handoff — V11 Tail-Variance, and Evaluating V13

**Status as of 2026-05-17.** Owner of next iteration: TBA.

**Goal for the next iteration:** *speed up the scatter + gather runtime on TT hardware* to be **at least 3× faster than the CPU 8T density baseline** (the true CPU production target — see §6.7). Nothing else (h2d, d2h, DCT, host code) is in scope unless it is on the scatter+gather critical path.

Current status against the 8T target: V11 (post G-PRESCALE + G-PMERGE) is **1.15× – 2.2× slower** than CPU 8T density across the 18 configs. To hit 3× faster, the kernel needs to be **3.4× – 6.6× faster than today**. This is not closeable with incremental optimization; the next agent should evaluate V13 or a V11/V13 hybrid (see §5 and §8).

Two paths the next person should evaluate:
1. **Continue squeezing V11** — its kernel-level critical path is at ~3.7 ms / iter at grid 2048; the unsolved problem is per-iter *variance* (median 7 ms, max 24 ms gather at grid 2048).
2. **Evaluate V13 (or V13_fpu) scatter+gather** — currently slower than V11 at grid 512 but architected to scale better. Decide whether V13 alone, or a V11/V13 hybrid, can beat current V11.

This document is self-contained for those two goals. Read it before touching kernels.

---

## 0. The handoff document that started this session

When I started I read **`docs/V13_PERF_HANDOFF.md` (2026-05-15)** end-to-end. It covers:
- What the density-scatter op does (math + DREAMPlace contract)
- V11 and V13_fpu architecture overviews
- The convergence bug that made adaptec1_512 work (send **original** cell sizes to TT, not clamped)
- CPU/V11/V13_fpu benchmark numbers at that time
- Per-stage timing analysis
- A ranked optimization plan

Read `docs/V13_PERF_HANDOFF.md` first if you have not. This handoff is a *companion* — it documents what changed since 2026-05-15 and the new bottlenecks.

I also pulled context from `docs/V11_PHASE3_HANDOFF.md` (V11 phase 3 structure) and from this project's auto-memory at `~/.claude/projects/-localdev-ayadav-tt-work-TTPort-DREAMPlaceTT-Framework/memory/`. Key memory entries:
- `v11_sweep_config.md` — env + host fixes required to reproduce results
- `v11_a2_perf_win.md` — scatter wins from earlier (writer-side sort/dedup removal, max_per_page tuning)
- `v11_g_prescale_win.md` — this session
- `v11_g_pmerge_win.md` — this session
- `cpu_dct_required_for_v11.md` — historical; superseded by the new TT-DCT fix below

---

## 1. The op (one paragraph reminder)

DREAMPlace calls `ElectricPotentialFunction.forward` ~600–900 times per benchmark. Inside it: `density_map = scatter(cells → bins)`; then `field_x, field_y = DCT(density_map)`. We replace the density_map step (and now optionally the DCT) on TT. The exact math the kernel must implement is in `DREAMPlace/dreamplace/ops/electric_potential/src/electric_density_map.cpp:255-309`. See §1 of `docs/V13_PERF_HANDOFF.md` for the full equation.

**In scope for next person: scatter + gather only.** Other phases (h2d/d2h, DCT, host orchestration) are off-limits unless directly affecting scatter+gather.

---

## 2. What changed this session

### 2.1 Optimizations landed in source (all on V11 path)

| ID | What | File(s) | Gather impact @ adaptec1_2048 |
|---|---|---|---:|
| **G-PRESCALE** | Pre-multiply `ox`/`oy` by `sqrt(inv_bin_area)` inside `v4_compute` SFPU. Removes the `V11A-SCALE` phase from the BRISC gather kernel. | `kernels/v4_compute.cpp`, `kernels/v11_accum_dm.cpp`, `host/density_scatter_ttnn_server_host.cpp` | 10.34 → 7.81 ms (**−25%**) |
| **G-PMERGE** | BRISC and NCRISC each merge half of `dense_b += dense_n` in parallel. Adds 2 sems (`brisc_acc_done`, `ncrisc_half_merge_done`). Reduces critical path. | `kernels/v11_accum_dm.cpp`, `kernels/v11_accum_n_dm.cpp`, `host/density_scatter_ttnn_server_host.cpp` | 7.81 → 7.05 ms (**−10%** additional) |
| **CPU_DCT bypass** | Server's `CPU_DCT=1` mode skips TTNN DCT and writes raw density to `shm_fx`. Required for compatibility with the client's CPU-DCT path. | `host/density_scatter_ttnn_server_host.cpp:1693` | — |
| **TT DCT initial_density fold** | Client uploads `initial_density_map / bin_area` into a new shm slot; server adds it to `density_flat` before each `TTNNDCTSolver.solve()`. Enables `CPU_DCT=0` (full TT-side DCT) without HPWL divergence to 155 M. | `host/density_scatter_ttnn_server_host.cpp:1263, 1278, 1696`, `integration/scatter_ttnn_client.py:233, 263, 791` | — |
| **Legacy-warmup skip** | Wrap the `wl_scatter`/`wl_gather` warmup `EnqueueMeshWorkload` in `if (!use_v11)`. The legacy kernel `v4_ncrisc_scatter.cpp` is not in this repo; without the guard the server aborts at JIT. | `host/density_scatter_ttnn_server_host.cpp:1219` | — |

### 2.2 Optimizations tried and reverted

| ID | Hypothesis | Result | Why |
|---|---|---|---|
| **G-SHARD-PARALLEL** | Fire K-1 shard reads concurrently → cut DRAM-read latency in `V11A-SHARD-SUM` | gather **−0.2 to −0.8 %** | DRAM reads are not the bottleneck. `V11A-SHARD-SUM`'s 770 µs is dominated by the *wait* on K-1 shard-owner semaphores; reads + adds are cheap. |
| **G-NOFINISH** | Skip the inter-program `Finish()` between scatter and accum dispatch to save host overhead | gather wash, total wall_time **+3 %** (regression) | `Finish()` is ~50 µs, not the 1.5 ms I'd estimated. The 3.4 ms gap between kernel-busy and wall-clock is in fast-dispatch + NOC routing, not host-side Finish. |
| **G-quick (unroll V11A-MERGE)** | Manual 8-way unroll + `__restrict__` to expose ILP | gather wash | Compiler already pipelines; the merge is L1-latency-bound, not ILP-bound. |

### 2.3 Cumulative result

| Config | V11 baseline (2026-05-15) | After this session | Δ |
|---|---:|---:|---:|
| adaptec1_512 gather | ~3.14 ms | 2.82 ms | −10 % |
| adaptec1_2048 gather | 10.34 ms | 7.05 ms | **−32 %** |

HPWL preserved to within 0.03 % of CPU baseline across all 18 configs.

---

## 3. The current state of V11

### 3.1 Median timing per config (post-G-PMERGE)

Numbers below are from the most recent 18-config sweep, results dir `results/v11_perf_sweep_tracy_gpmerge/`. Full CSV: `results/v11_perf_sweep_tracy_gpmerge/v11_summary.csv`.

| design | grid | scatter med (ms) | gather med (ms) | HPWL (M) | iters |
|---|---|---:|---:|---:|---:|
| adaptec1 | 512  | 1.95 | 2.82 | 70.33 | 628 |
| adaptec1 | 1024 | 3.05 | 5.95 | 70.86 | 654 |
| adaptec1 | 2048 | 6.13 | 7.05 | 71.33 | 672 |
| adaptec2 | 512  | 2.36 | 3.12 | 78.68 | 631 |
| adaptec2 | 1024 | 3.25 | 5.18 | 79.26 | 657 |
| adaptec2 | 2048 | 5.65 | 9.83 | 82.90 | 712 |
| adaptec3 | 512  | 4.62 | 3.03 | 186.4 | 674 |
| adaptec3 | 1024 | 5.48 | 5.03 | 185.9 | 685 |
| adaptec3 | 2048 | 8.23 | 9.57 | 188.3 | 752 |
| bigblue1 | 512  | 2.71 | 3.67 | 87.40 | 695 |
| bigblue1 | 1024 | 4.10 | 3.66 | 117.1 | 999 |
| bigblue1 | 2048 | 8.00 | 10.34 | 88.49 | 790 |
| bigblue2 | 512  | 5.27 | 2.86 | 131.4 | 659 |
| bigblue2 | 1024 | 6.57 | 5.97 | 131.3 | 678 |
| bigblue2 | 2048 | 10.53 | 10.40 | 133.0 | 722 |
| bigblue3 | 512  | 6.37 | 4.69 | 293.6 | 750 |
| bigblue3 | 1024 | 7.29 | 9.78 | 290.1 | 787 |
| bigblue3 | 2048 | 10.26 | 12.90 | 291.8 | 822 |

### 3.2 Tracy zone breakdown (BRISC critical path) @ adaptec1_2048

After G-PRESCALE + G-PMERGE, mean event durations on BRISC (warmup-10 dropped, ~9.6 K iters × n_cores events):

| Zone | mean µs / event | Notes |
|---|---:|---|
| **V11A-ACC** (per chunk, ×7 chunks/iter) | **318** | Per-tuple `dense[idx] += a`. ~80 cycles/tuple — FP-soft on BRISC RISC-V. |
| **V11A-SHARD-SUM** (hot cores only) | **768** | Wait-bound on K-1 shard-owner sem signals. Reads + adds are small. |
| **V11A-MERGE** (upper half) | **718** | Was 1413 µs before G-PMERGE. Now BRISC does upper half while NCRISC does lower half in parallel. |
| V11A-ZERO | 142 | Naive `dense[i] = 0` loop. |
| V11A-HDR-READ-ALL | 52 | One bulk read per chunk. |
| V11A-DENSITY-WRITE | 45 | Final DRAM write of owned tiles. |
| V11A-DATA-READ (per chunk) | 6 | Already small. |

NCRISC parallel work (not on critical path):
- V11N-ACC × 7 chunks = 248 µs/event each
- V11N-MERGE-HALF = 1166 µs (most of this is *wait* for BRISC's V11A-ACC to finish, then ~700 µs of actual merge work)

**Sum BRISC busy (non-hot cores): ~3.0 ms.**
**Sum BRISC busy (hot cores, ~14 % of cores): ~3.7 ms.**
**Gather wall-clock (median): 7.05 ms.**

The ~3.4 ms gap between BRISC busy and wall-clock is host dispatch + NOC routing of args + barrier, **not** host-side `Finish()` (verified by G-NOFINISH experiment).

---

## 3.5 Silent tuple drops at the per-page cap — IMPORTANT

After this session, the scatter writers run with `v11_max_per_page_tuples = 4096` → split as 2048 per RISC per `(writer, receiver)` page. **The drop path is real and exists** in `kernels/v11_scatter_dm.cpp:230-242` and `kernels/v11_scatter_b_dm.cpp:175-188`:

```cpp
uint32_t already = dram_offset_tuples[recv];
if (already >= ncrisc_cap_tuples) {
    total_drops += cnt;
    staging_count[recv] = 0u;
    return;                              // entire flush silently dropped
}
if (already + cnt > ncrisc_cap_tuples) {
    uint32_t kept = ncrisc_cap_tuples - already;
    total_drops += cnt - kept;
    cnt = kept;                          // partial drop
}
```

### 3.5.1 Drop-counter instrumentation (in source now)

This session added a drop counter that is *kept in the codebase* for the next agent:

- **Allocation** (host): `host/density_scatter_ttnn_server_host.cpp:866` allocates `drop_buf` — `2*nc_all` 32-byte pages in DRAM, one slot per writer-RISC.
- **Per-RISC slot assignment**: NCRISC writes its drop total to `drop_buf[my_writer_id]` (= `c`); BRISC writes to `drop_buf[my_writer_id + nc_all]` (= `c + nc_all`). The slot offset is hardcoded in `kernels/v11_scatter_b_dm.cpp` end-of-kernel.
- **Kernel write** (NCRISC `v11_scatter_dm.cpp` end-of-kernel; BRISC mirror): single `noc_inline_dw_write` of `total_drops` to the DRAM slot.
- **Host read** (`host/density_scatter_ttnn_server_host.cpp:1633`): after the per-iter scatter `Finish()`, host calls `EnqueueReadMeshBuffer(cq, drop_host, drop_buf, true)` (~7 KB read), sums and finds max, prints `[server] DROP iter=N sum=X max=Y (slot=Z BRISC|NCRISC c=K)` when total > 0 (throttled to one print per 50 iters).
- **Cost**: ~50 µs/iter for the read + sum. Acceptable.

### 3.5.2 Measured drops on adaptec1_2048 (CPU_DCT=1, GATHER_MODE=v11)

| iter | total drops | max single writer-RISC | comment |
|---:|---:|---|---|
| 50  | 824 | NCRISC c=4 (336) | start of density ramp |
| 100 | **126,968** | NCRISC c=2 (2908) | peak |
| 150 | 29,376 | NCRISC c=14 (1408) | settling |
| 200 | 6,356 | NCRISC c=10 (992) | declining |
| 250 | 408 | NCRISC c=10 (312) | near zero |
| 300+ | ~0 | — | placement stable |

Final HPWL = 71.33 M (**identical** to non-instrumented baseline). The drop counter measurement does NOT alter results.

### 3.5.3 Why convergence still works despite massive drops (peak ~58 % of tuples)

1. **Drops are biased toward already-saturated hot tiles.** A drop fires only when a (writer, receiver) pair has already exceeded 2048 tuples on a hot tile — these are already-high-density regions. The field solve treats them as repellent; under-counting density there clamps the repulsion magnitude but preserves its sign.
2. **Drops are transient.** They peak during DREAMPlace's density-penalty ramp (iter 50–200), then decay to ~0 once placement stabilizes. Only ~150 iters out of 672 are meaningfully affected.
3. **DREAMPlace averages gradients.** Adam-like optimizer absorbs noisy single-iter gradients. Bias on hot-tile gradients during the ramp is corrected on subsequent iters when those regions have spread out.
4. **Dropped tuples are typically marginal contributions.** Tuples late in a flush are often edge-overlap (small `ox*oy`). The big-area contributions arrive first and are committed before the cap is hit.
5. **HPWL is geometric, not density-driven.** Final HPWL depends on cell positions, which depend on the *trajectory* of gradient updates, not any single bad gradient.

### 3.5.4 Open hypothesis — drops may explain part of the gather variance

Iters with massive drops do *less downstream work* (less data for V11A-DATA-READ and V11A-ACC to process), so they may be the *faster* iters and iters with no drops are the *slower* ones. The bimodal distribution (median 7 ms vs max 24 ms at adaptec1_2048) could be partly drop-driven.

**Experiment for the next agent**: correlate per-iter `gather_ms` (from `results/.../tt/sweep_*_iters.csv`) with per-iter drop count (you'd need to log drops every iter into a CSV, not just one print every 50 iters). If the slow iters are the ones with no drops, that confirms the hypothesis. Code change: add a CSV writer in the host's drop-read block.

### 3.5.5 Reproducing the drop measurement

```bash
cd /localdev/ayadav/tt-work/TTPort/DREAMPlaceTT-Framework

# Pick any config — drops are most visible at larger grids
CONTAINER=bh-38-special-ayadav-for-reservation-75032 \
CPU_DCT=1 \
GATHER_MODE=v11 \
RESULTS_DIR=results/g_drop_run \
stdbuf -oL DREAMPlace/dp_env/bin/python3 integration/run_dreamplace.py \
    --device scatter_ttnn \
    --container bh-38-special-ayadav-for-reservation-75032 \
    --benchmark benchmarks/configs/sweep_adaptec1_2048.json \
    --results-dir results/g_drop_run 2>&1 | grep -E "DROP|HPWL"
```

Look for `[server] DROP iter=N sum=X max=Y (slot=Z RISC c=K)` lines. **Important**: the wrapper `scripts/run_sweep.sh` strips this output via its `tail -3` filter — invoke `integration/run_dreamplace.py` directly as shown.

### 3.5.6 Next experiments to consider

1. **Raise the cap and re-measure.** Set `v11_max_per_page_tuples = 8192` (`host/density_scatter_ttnn_server_host.cpp:838`). Route_buf grows from 378 MB to 756 MB at adaptec1_2048 — still under the 4 GB cap. Re-run the sweep; check whether HPWL changes (it shouldn't, given drops don't break convergence) but whether **gather variance drops** (the test of the variance hypothesis above).
   - **2026-05-17 attempt — both pg=8192 and pg=6144 are NOT viable as a simple knob.** With pg=8192 the accum L1 footprint exceeds the 1.5 MB cap (`TT_THROW: Statically allocated circular buffers ... grow to 1628544 B which is beyond max L1 size of 1572864 B`). With pg=6144 L1 fits but **gather goes from 7 ms → 25 ms** (3.5×) because `V11A-DATA-READ` reads `max_per_writer * 8` bytes per page regardless of actual `cnt`, so doubling the cap doubles the DRAM read per chunk. Drops did fall ~94% as hoped (iter 124 had 7,592 drops vs pg=4k iter 100's 126,968), but the run failed to converge in 128 iters (overflow stuck at 0.91). **Conclusion: raising the cap alone is a net loss — the test of "drops cause variance" is inconclusive from this angle.** To run the experiment correctly you must first do G4 (read only `cnt × 8` bytes per page in V11A-DATA-READ) so a bigger cap doesn't penalize the per-iter steady state. See `host/density_scatter_ttnn_server_host.cpp:838` for the cap and `kernels/v11_accum_dm.cpp:223-225` for the read-the-whole-page logic.
2. **Sweep drops across all 18 configs.** The current numbers are only for adaptec1_2048. Some designs (bigblue1 was the highest-variance) may show entirely different drop profiles. Pattern: smaller grids = more cell clustering per tile = more drops; bigger designs (more cells) = more drops per writer.
3. **Tile-bucket the scatter output (G3 from `~/.claude/plans/your-job-is-to-spicy-key.md`).** Bucketing tuples by destination tile inside `flush_recv` means each (writer, receiver, tile) gets its own cap rather than sharing one with sibling tiles — could eliminate the cap pressure entirely. Bigger refactor.

---

## 4. The open problem — gather variance

This is the headline open issue. **The slow tail-iter gather is 3–6× the median**:

| config | median ms | p99 ms | max ms | tail ratio |
|---|---:|---:|---:|---:|
| adaptec1_512  | 2.89 | 8.04 | 8.20 | 2.8× |
| adaptec1_2048 | 7.01 | 23.18 | 23.50 | 3.4× |
| **bigblue1_1024** | **3.61** | **23.78** | **23.83** | **6.6×** |
| bigblue3_2048 | 12.71 | 24.69 | 24.72 | 1.9× |

Source: `results/v11_perf_sweep_tracy_gpmerge/tt/sweep_<design>_<grid>_scatter_ttnn_iters.csv` (column `gather_ms`, one row per iter). **p99 ≈ max** in most configs, meaning the slow iters are a sustained population, not a single outlier.

### 4.1 Root causes (ranked by leverage)

1. **Load imbalance from cell clustering** (biggest cause).
   - Snake-fill tile ownership (`host/v11_tile_ownership.h::build_snake_fill_ownership`) gives each core a *spatially adjacent* block of tiles.
   - As DREAMPlace clusters cells, the cores owning those clustered regions get a disproportionate share of tuples. All other cores idle waiting on those cores at MERGE / SHARD-SUM sync points.
   - **Fix idea**: replace snake-fill with hash-based ownership where each core gets a *spatially scattered* set of tiles. Each core then sees a uniform random sample of cell density regardless of where cells cluster. Local change.

2. **Stale hot-tile partition.**
   - `V11_HIST_REFRESH_ITERS = 1000000` (`host/density_scatter_ttnn_server_host.cpp:1315`). The shard_table is built once at iter 0 (when cells are uniformly placed → no hot tiles → K=1 for everything) and **never updated**.
   - As cells cluster, the actually-hot tiles emerge but never get K-way sharded → one core does all the work for that tile every iter.
   - **Fix idea**: set `V11_HIST_REFRESH_ITERS = 50` or 100. Each refresh re-runs the histogram and rebuilds the accum program (~50–200 ms one-shot, JIT-cached after first). Periodic re-balancing.

3. **BRISC/NCRISC + cross-core sync stalls.**
   - V11A-MERGE waits on V11N-ACC-done from the same core.
   - V11A-SHARD-SUM waits on K-1 sem signals from *other cores* (shard owners).
   - If any one core falls behind (DRAM/NOC contention burst), the slowest-core-this-iter becomes the critical path for every core.
   - Partly attacked by G-PMERGE. Further reduction requires eliminating MERGE entirely (full **G2**: partition owned tiles between BRISC and NCRISC). My analysis suggests G2 may not give net win on top of G-PMERGE because per-RISC tuple reads double; needs careful prototyping.

4. **Tuple drops may be the *fourth* variance source.** See §3.5.4. Iters with massive drops do less downstream work; the slow tail-iters may be the *un-dropped* ones. Test this by correlating `gather_ms` with per-iter drop count.

---

## 5. Why V13 is worth evaluating

V13 (and the variant V13_fpu) was designed to address V11's per-tuple cost on BRISC RISC-V:
- V11 emits per-bin tuples → 80 cycles per `dense[idx] += a` on BRISC (FP-soft path, see §3.2).
- V13 emits **records** (cell-shaped, not bin-shaped) and uses **FPU matmul** on TRISC for the accumulate. Hardware FP — orders of magnitude faster per tuple, but with overhead in record routing.

The 2026-05-15 numbers (`docs/V13_PERF_HANDOFF.md` §4) showed V13_fpu was slower than V11 at grid 512 but scaled better at grid 2048. Since this session we have not re-measured V13_fpu after G-PRESCALE + G-PMERGE landed on V11. **A fresh apples-to-apples sweep of V13_fpu is the next person's first task.**

### 5.1 Hypotheses to test

H1. *V13_fpu's per-bin cost is structurally lower than V11's V11A-ACC.* If true, V13_fpu beats post-G-PMERGE V11 at grid 1024 and 2048.

H2. *V13_fpu has lower iter-to-iter variance* because the matmul gather is data-volume-bounded, not control-flow-bounded. If true, p99/median ratio is much smaller than V11's 3–6×.

H3. *A hybrid is possible*: V11 scatter (efficient at moderate cell density) + V13 gather (TRISC matmul). Worth designing only if H1 is partially true.

H4. *V13 gather has its own hot-tile imbalance.* The record-routed scatter pushes work to receivers by tile. Same load-imbalance pattern as V11 likely applies. Test with the same designs that show worst V11 variance (bigblue1_1024).

### 5.2 Suggested experimental path

1. Run V13_fpu on the 18-config sweep with current main-line code. Use the same env in §6, but `GATHER_MODE=v13_fpu`. Capture Tracy.
2. Compute the same per-config gather mean/median/p99/max table as §4.
3. Compare to V11 G-PMERGE. Identify where V13 wins, where it loses.
4. If V13 wins at large grids: investigate why V11 loses there (probably the per-tuple cost or the imbalance), and whether the V13 mechanism can be brought into V11 or vice versa.
5. If V13 loses everywhere: extract specific micro-features (e.g., is there a packing trick in V13's scatter that V11 could adopt?).

---

## 6. How to reproduce this work — env + commands

### 6.1 Environment requirements

- Container: `bh-38-special-ayadav-for-reservation-75032` (must be running; `docker start <name>` to restart if it has exited; the TT device occasionally drops the container, that is normal).
- Host repo root: `/localdev/ayadav/tt-work/TTPort/DREAMPlaceTT-Framework`
- `TT_METAL_HOME` should be the in-framework path, **not** the outer tt-metal. There is an ABI mismatch — see auto-memory `tt_metal_home_abi.md`. The scripts default to `$FW_ROOT/tt-metal` which is correct.
- Python: `DREAMPlace/dp_env/bin/python3` (handled by the scripts).

### 6.2 Building the server

```bash
cd /localdev/ayadav/tt-work/TTPort/DREAMPlaceTT-Framework

docker exec -w "$(pwd)" bh-38-special-ayadav-for-reservation-75032 \
    bash -c 'TT_METAL_HOME=$(pwd)/tt-metal bash scripts/build_server.sh -j 16'
```

Builds `host/build/density_scatter_ttnn_server`. **Must run inside the docker container** because it links against the in-container TT-Metal libs.

### 6.3 Single-config run (canonical)

```bash
# adaptec1 at grid 512, CPU DCT (client-side DCT, server skips TTNN DCT)
CONTAINER=bh-38-special-ayadav-for-reservation-75032 \
CPU_DCT=1 \
GATHER_MODE=v11 \
RESULTS_DIR=results/my_run \
bash scripts/run_sweep.sh sweep_adaptec1_512
```

Output: `results/my_run/sweep_adaptec1_512_scatter_ttnn_metrics.json` (look for `hpwl`, `overflow`, `n_ep_calls`, `scatter_ttnn_{scatter,gather}_ms_median`).
Per-iter log: `results/my_run/sweep_adaptec1_512_scatter_ttnn_iters.csv` (per-iter gather_ms is in column 8).

Expected on a healthy run: HPWL ≈ 70.33 M, overflow ≈ 0.069, n_ep_calls ≈ 626–630, gather median ≈ 2.82 ms.

`scripts/run_sweep.sh` line 57 has `2>&1 | tail -3` which hides per-iter stdout. Drop that filter (or call `integration/run_dreamplace.py` directly) if you want to see per-iter `[server] done ...` lines as they happen.

### 6.4 Full TT-side run (CPU_DCT=0)

DCT runs on TT device end-to-end:

```bash
CONTAINER=bh-38-special-ayadav-for-reservation-75032 \
CPU_DCT=0 \
GATHER_MODE=v11 \
RESULTS_DIR=results/tt_dct_run \
bash scripts/run_sweep.sh sweep_adaptec1_512
```

Expected HPWL ≈ 70.35 M (within 0.03 % of CPU_DCT=1).

### 6.5 18-config sweep with Tracy device profiler

```bash
CONTAINER=bh-38-special-ayadav-for-reservation-75032 \
RESULTS_DIR=results/v11_perf_sweep_tracy_NEW \
bash scripts/run_v11_perf_sweep_tracy.sh
```

Runs all 6 designs × 3 grids with `TT_METAL_DEVICE_PROFILER=1`. ~20–30 min total. Outputs:
- `results/<dir>/v11_summary.csv` — one row per config with all medians/means
- `results/<dir>/tt/sweep_<design>_<grid>_scatter_ttnn_metrics.json` — full metrics
- `results/<dir>/tt/sweep_<design>_<grid>_scatter_ttnn_iters.csv` — per-iter
- `results/<dir>/tt_profile/<design>_<grid>/profile_log_device.csv` — Tracy raw (~300 MB / config, 5.6 GB total)
- `results/<dir>/tt/sweep_<design>_<grid>.log` — server stdout/stderr for that run

After: process Tracy CSVs into zone aggregates:

```bash
python3 tools/process_v11_profile.py \
    --root   results/v11_perf_sweep_tracy_NEW/tt_profile \
    --output results/v11_perf_sweep_tracy_NEW/profile_analysis
```

Outputs per-config `<design>_<grid>_zones.csv` plus a cross-config `v11_profile_rollup.csv`.

### 6.6 The CPU baseline — and the gap V11 has to close

**The CPU code saturates at 8 threads on this machine.** This was discovered 2026-05-17: an OMP=8 sweep is consistently *equal to or faster than* the OMP=16 sweep across all 18 configs. The workload is memory-bandwidth-bound; adding hyperthreads past 8 only adds cache-line contention and NUMA noise. So **CPU 8T is the real production baseline**, not 16T.

Run the CPU baseline:
```bash
CONTAINER=bh-38-special-ayadav-for-reservation-75032 \
  BACKENDS=cpu \
  OMP_NUM_THREADS=8 \
  RESULTS_DIR=results/perf_sweep_cpu8t \
  bash scripts/run_perf_sweep.sh
```
Outputs at `results/perf_sweep_cpu8t/cpu/sweep_<design>_<grid>_cpu_metrics.json`.
The relevant key is `cpu_density_median_ms` — that is the CPU **scatter + gather** runtime in milliseconds per iter. TT's `scatter_ms + gather_ms` competes with this number directly.

**CPU 8T density baseline (median ms per iter):**

| design   | 512  | 1024 | 2048  |
|---       | ---: | ---: | ---:  |
| adaptec1 | 3.37 | 4.10 | 10.67 |
| adaptec2 | 5.33 | 6.45 | 10.82 |
| adaptec3 | 12.51 | 13.49 | 16.37 |
| bigblue1 | 5.82 | 6.82 | 13.51 |
| bigblue2 | 12.94 | 14.00 | 17.34 |
| bigblue3 | 19.08 | 23.10 | 20.16 |

**Current V11 (post G-PMERGE) scatter+gather (median ms per iter), and ratio vs 8T:**

| design   | grid | V11 sc+ga | CPU 8T density | V11 / CPU8T |
|---       | ---: | ---: | ---: | ---: |
| adaptec1 | 512  | 4.77 | 3.37 | 1.42× slower |
| adaptec1 | 1024 | 9.00 | 4.10 | 2.20× slower |
| adaptec1 | 2048 | 13.18 | 10.67 | 1.24× slower |
| adaptec2 | 512  | 5.48 | 5.33 | 1.03× slower |
| adaptec2 | 1024 | 8.43 | 6.45 | 1.31× slower |
| adaptec2 | 2048 | 15.48 | 10.82 | 1.43× slower |
| adaptec3 | 512  | 7.65 | 12.51 | **0.61× (TT 1.6× faster)** |
| adaptec3 | 1024 | 10.51 | 13.49 | **0.78× (TT 1.3× faster)** |
| adaptec3 | 2048 | 17.80 | 16.37 | 1.09× slower |
| bigblue1 | 512  | 6.38 | 5.82 | 1.10× slower |
| bigblue1 | 1024 | 7.76 | 6.82 | 1.14× slower |
| bigblue1 | 2048 | 18.34 | 13.51 | 1.36× slower |
| bigblue2 | 512  | 8.13 | 12.94 | **0.63× (TT 1.6× faster)** |
| bigblue2 | 1024 | 12.54 | 14.00 | **0.90× (TT 1.1× faster)** |
| bigblue2 | 2048 | 20.93 | 17.34 | 1.21× slower |
| bigblue3 | 512  | 11.06 | 19.08 | **0.58× (TT 1.7× faster)** |
| bigblue3 | 1024 | 17.07 | 23.10 | **0.74× (TT 1.4× faster)** |
| bigblue3 | 2048 | 23.16 | 20.16 | 1.15× slower |

TT wins today are concentrated on **dense designs at small grids** (adaptec3_512, bigblue2_512, bigblue3_512). Those are exactly the cases where CPU's per-cell loop has the worst cache behavior — many cells per bin, high contention on atomic-add. TT loses on **sparse designs at large grids** (adaptec1_*, adaptec2_2048) because CPU vectorizes well there and TT pays heavy fixed overhead per iter (host dispatch + NOC routing of 110 cores).

### The 3× target

The original goal: **TT scatter+gather ≤ 1/3 of CPU 8T density**. To hit it, each config above needs the V11 sc+ga column to drop by **3.4× – 6.6×** (the inverse of the current ratio). Specifically:

- adaptec1_2048: 13.18 → ≤ 3.56 ms (-73 %)
- adaptec3_2048: 17.80 → ≤ 5.46 ms (-69 %)
- bigblue1_2048: 18.34 → ≤ 4.50 ms (-75 %)
- bigblue3_2048: 23.16 → ≤ 6.72 ms (-71 %)

For comparison, the largest single optimization this session (G-PRESCALE) cut adaptec1_2048 gather by 25 %. Each subsequent optimization gave 5–10 %. **You will not get to 3× with kernel micro-optimization.** Plan accordingly:

- The structural bottleneck on V11 is the per-tuple FP-soft add on BRISC (§3.2). At ~80 cycles per tuple × ~2K tuples per receiver per iter, V11A-ACC is fundamentally limited.
- TRISC SFPU can do fp32 in hardware — if you can move the accumulate to TRISC, the per-tuple cost drops by ~5–10×.
- V13 already uses TRISC matmul gather. **Evaluating V13 vs V11 at the post-G-PMERGE state is the highest-leverage open question.**
- If V13 wins (especially at the configs where V11 currently loses, e.g., adaptec1_2048): drop V11 and finish V13.
- If V13 loses everywhere: build a V11/V13 hybrid that uses V11's scatter at small grids and V13's gather at large ones; or move V11's V11A-ACC inner loop to a TRISC kernel.

### 6.7 If the container is dead

Symptom: `docker exec <container> echo hi` returns `Error response from daemon: container ... is not running`. Or a sweep hangs at ~10 min with `n_ep_calls=0`.

Fix:
```bash
docker start bh-38-special-ayadav-for-reservation-75032
```

This was authorized once this session; the next agent should ask the user before doing it unless you have a written authorization. **Do not `tt-smi -r`** without explicit user authorization — that resets the chip.

---

## 7. Critical files

### 7.1 Source code

| Path | What |
|---|---|
| `host/density_scatter_ttnn_server_host.cpp` | C++ server: shm IPC, all program/buffer setup, all the new fixes from §2.1 |
| `host/v11_tile_ownership.h` | Snake-fill ownership + shard_table builder. Modify this for hash-based ownership in §4.1 fix 1. |
| `kernels/v4_compute.cpp` | TRISC SFPU compute: bin index + overlap. G-PRESCALE pre-multiply lives here. |
| `kernels/v4_reader.cpp` | BRISC reader for the scatter front-end (shared with V11 hist program). |
| `kernels/v11_scatter_b_dm.cpp`, `kernels/v11_scatter_dm.cpp` | V11 scatter: BRISC half + NCRISC half. Step-5 parallel scatter with shared writer page. Drop-counter writes to `drop_buf` slot at end of kernel (§3.5). |
| `kernels/v11_accum_dm.cpp` | V11 BRISC accum (gather). All the gather-side phases: ZERO, LOOKUP, HDR-READ, DATA-READ, ACC, MERGE-upper, SHARD-WRITE, SHARD-SUM, DENSITY-WRITE. |
| `kernels/v11_accum_n_dm.cpp` | V11 NCRISC accum mirror. V11N-ACC, V11N-MERGE-HALF (lower half), signals. |
| `kernels/v11_histogram.cpp` | One-shot histogram program — used at iter 0 to build shard_table. |
| `kernels/v13_scatter_brisc.cpp`, `kernels/v13_scatter_ncrisc.cpp` | V13 scatter (untouched this session). |
| `kernels/v13_accum_brisc.cpp`, `kernels/v13_accum_ncrisc.cpp`, `kernels/v13_accum_compute.cpp` | V13 accum (matmul-based). |
| `integration/scatter_ttnn_client.py` | Python client. Lines 224–290 set up shm (4 pos + 3 field slots), lines 786–795 upload `initial_density_map`. Lines 800–830 are the CPU_DCT fallback. |
| `integration/run_dreamplace.py` | Wraps DREAMPlace's `nlplace.py`, attaches `ElectricPotentialFunction.forward` hook, collects per-iter timings. |

### 7.2 Scripts

| Path | What |
|---|---|
| `scripts/build_server.sh` | Build the C++ server. Must run inside container. |
| `scripts/run_sweep.sh` | Single-design wrapper. **Note**: line 57 `2>&1 \| tail -3` hides per-iter output. |
| `scripts/run_v11_perf_sweep_tracy.sh` | 18-config sweep + Tracy. Used for headline numbers. |
| `scripts/run_perf_sweep.sh` | 6×3×2 CPU/TT sweep (kept for CPU baseline comparisons). |
| `scripts/run_smoke.sh` | Synthetic correctness check (does *not* go through DREAMPlace). |

### 7.3 Tracy profile data

**Per-config raw Tracy device-profiler CSVs** (huge — ~300 MB each):
```
results/v11_perf_sweep_tracy_gpmerge/tt_profile/<design>_<grid>/profile_log_device.csv
```
6 designs × 3 grids = 18 files, total ~5.6 GB.

**Per-zone aggregates** (small, what you actually read):
```
results/v11_perf_sweep_tracy_gpmerge/profile_analysis/<design>_<grid>_zones.csv
results/v11_perf_sweep_tracy_gpmerge/profile_analysis/v11_profile_rollup.csv
```

`tools/process_v11_profile.py` reads the raw CSVs and emits the aggregates.

### 7.4 Per-iter timing CSVs (one row per DREAMPlace EP-forward call)

```
results/v11_perf_sweep_tracy_gpmerge/tt/sweep_<design>_<grid>_scatter_ttnn_iters.csv
```

Columns: `iter, ep_ms, pos_write_ms, kernel_wait_ms, field_read_ms, h2d_ms, scatter_ms, gather_ms, d2h_density_ms, ttnn_upload_ms, ttnn_compute_ms, ttnn_download_ms, fw_ms, total_server_ms, total_client_ms, gather_mode, cpu_dct_ms, ep_total_ms, hpwl, overflow`.

For the variance analysis in §4 use the `gather_ms` column.

### 7.5 Summary CSVs (one row per config)

```
results/v11_perf_sweep_tracy_gpmerge/v11_summary.csv      # post-G-PMERGE (CURRENT)
results/v11_perf_sweep_tracy_gprescale/v11_summary.csv    # post-G-PRESCALE only
```

The schema is `design,grid,n_ep_calls,hpwl,overflow,wall_time_s,scatter_ms_median,gather_ms_median,cpu_dct_ms_median,h2d_ms_median,d2h_density_ms_median,fw_ms_median,ep_total_ms_median,total_server_ms_median,...mean,...`.

### 7.6 Other docs in `docs/`

| File | When to read |
|---|---|
| `docs/V13_PERF_HANDOFF.md` | **READ FIRST.** Starting handoff this session was built on. |
| `docs/V11_PHASE3_HANDOFF.md` | Earlier V11 design notes. |
| `docs/V11_PHASE2_STATUS.md`, `docs/V11_PHASE3_STATUS.md` | V11 phase histories. |
| `docs/DREAMPLACE_TT_ARCHITECTURE.md` | How the Python client/server split works. |
| `docs/PROFILING.md` | How to enable Tracy / TT_METAL_DEVICE_PROFILER. |
| `docs/RUNTIME_BREAKDOWN.md` | Where iter time goes at a high level. |
| `docs/KNOWN_ISSUES.md` | Things that bit prior agents. |

---

## 8. Suggested first week of work

**Target to beat: CPU 8T density (§6.6). V11 today is 1.15–2.2× slower than 8T. Goal is V11/V13 ≤ 1/3 of 8T → 3.4–6.6× speedup needed.** Incremental kernel tuning will not get you there; the structural lever is moving the per-tuple accumulate off BRISC RISC-V (FP-soft, ~80 cycles/tuple) to TRISC SFPU (hardware fp32).

In priority order:

1. **Re-baseline V13_fpu on the current main against CPU 8T.** Same 18-config Tracy sweep, `GATHER_MODE=v13_fpu`. Produce a summary CSV. Compute the V13/CPU8T ratio table like §6.6. This is one command + one analysis script — *do this first*. The decisive question: **does V13 gather close the 3× gap at any config?**
2. **Per-iter variance plot for V13_fpu.** Same analysis as §4. Hypothesis: V13's matmul gather is data-bounded → tighter distribution than V11. Confirm or refute.
3. **Decide V11 vs V13 vs hybrid**, based on (1) and (2):
   - **V13 wins clearly at 2048** → drop V11, finish V13 (clean up, productize, run with TT-DCT enabled).
   - **V13 wins on some configs, V11 on others** → build a hybrid. The most natural split: V13 scatter+gather for grid ≥ 1024 (where TT pays the dispatch cost once and benefits from matmul), V11 for grid 512 (where dispatch is a bigger fraction). Switch is in the host's mode selector.
   - **V13 loses everywhere** → only path forward is a major V11 redesign: move V11A-ACC's per-tuple accumulate to a TRISC kernel that uses SFPU. That is a substantial refactor (new compute kernel, change of L1 CB topology, new BRISC↔TRISC handshake).
4. **Test the "drops → variance" hypothesis on V11.** See §3.5.4–§3.5.6. Pre-requisite is **G4** (read only `cnt × 8` bytes per page in `V11A-DATA-READ`) — without it, raising the per-page cap penalizes the steady state. Then raise cap to 8192 and re-sweep.
5. **If V11 stays alive in the pipeline**, also try:
   - `V11_HIST_REFRESH_ITERS = 50` (one-line change at `host/density_scatter_ttnn_server_host.cpp:1315`) — adapts shard_table to evolving cell density; predicted to reduce tail.
   - Hash-based tile ownership in `host/v11_tile_ownership.h` — collapses load-imbalance tail.
   - G2 (partition owned tiles between BRISC and NCRISC), documented in `~/.claude/plans/your-job-is-to-spicy-key.md`. My analysis on top of G-PMERGE was net-wash but the assumptions depend on per-tuple cost — re-evaluate after pulling work to TRISC.

**Hard deliverable for next iteration**: a sweep where at least one of `(V13, V11, V11/V13 hybrid)` achieves *median scatter+gather ≤ CPU 8T density / 3* on **at least one** config in {adaptec1, bigblue1} at every grid. If none of the configs reach 3×, the deliverable is a written, evidence-based recommendation on whether to (a) accept TT cannot beat 8T 3× on this workload, (b) escalate to a TRISC SFPU port, or (c) require a major architecture change (e.g., persistent-kernel V11 with sub-iter pipelining).

---

## 9. Watch-outs from this session

- **Don't trust the auto-memory entry `cpu_dct_required_for_v11.md` blindly** — it predates the TT-DCT fix and is now misleading. The TT-DCT path (`CPU_DCT=0`) works correctly; the new entry `v11_sweep_config.md` is authoritative.
- **Kernel files referenced by the host but missing from `kernels/`**: `v4_ncrisc_scatter.cpp`, `v4_ncrisc_scatter_dense.cpp`, `v8_gather_*.cpp`, `v9_*.cpp`, `v10_*.cpp`. These are legacy paths; the warmup-skip fix (§2.1) routes around them. If you re-enable any non-V11 gather mode, you'll hit JIT failure and need to source the file from `/localdev/ayadav/tt-work/TTPort/DREAMPlaceTT/server/kernels/` (the *previous* framework) or skip its `CreateKernel`.
- **`scripts/run_sweep.sh:57`** has the `2>&1 | tail -3` filter — it hides per-iter output. Either patch the script or call `integration/run_dreamplace.py` directly to see iter logs.
- **Container exits** under stress — when this happens, sweeps return `n_ep_calls=0` with 606 s wall time (the client's ready-flag timeout). Just `docker start <container>` and retry. Do not `tt-smi -r` without authorization.
- **Per-iter gather variance is real** and depends on cell distribution — DO NOT report only mean or only median. The median tells you the easy case; the p99/max tells you what the user actually feels.

---

## Glossary

| Term | Meaning |
|---|---|
| **bin** | One cell of the density grid. Grid 512 means 512×512 bins. |
| **tile** | 32×32 = 1024 bins, the TT-Metal native unit. |
| **K** | Number of cores sharing one hot tile in K-way sharding. Capped at `MAX_K = 8`. |
| **route_buf** | DRAM-banked buffer where scatter writes per-(writer, receiver) tuple pages. Sized `nc_all × nc_all × route_pgsz`. |
| **shard_reduce_buf** | DRAM staging for K-way shard reduction. Pages indexed by `hot_tile_seq × MAX_K + shard_idx`. |
| **owned_lookup** | Per-core map `tile_idx → local_idx_in_dense`, or `0xFFFF` for tiles this core doesn't own. |
| **V11A-***, **V11N-***, **V4C-***, **V4R-*** | Tracy zone prefixes for BRISC accum, NCRISC accum, TRISC compute, BRISC reader. |
| **G-PRESCALE**, **G-PMERGE** | Optimization landmarks from this session. See §2.1. |
| **CPU_DCT=1** | Server writes raw density to `shm_fx`; Python client runs CPU DCT. |
| **CPU_DCT=0** | Server runs full TTNN DCT after folding `initial_density_map` from `shm_id`. |
