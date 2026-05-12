# Performance Model — Measured & Theoretical Costs

This document gives you the numbers needed to back-of-envelope a kernel's wall
time **before** you build it, then verify with TT-Tracy after.

Every row is tagged for provenance:

- **(M-Tracy)** — measured this project, captured in
  `results/tracy_smoke_2026-05-12/profile_v11.txt`. Reproducible via the recipe in § 3.
- **(M-host)** — measured this project from server log lines / sweep CSVs.
- **(E)** — estimate; verify before relying on it for a design decision.

If you measure a number that contradicts an entry, **update this doc.**

---

## 1. Measured reference data — STEADY-STATE

> ⚠️ **Steady-state is critical**. The first measurements were taken on a 1-iter
> smoke fixture and were optimistic by 2–3× for several zones. The numbers
> below come from a **real multi-iter sweep** with the first 20 invocations
> per (core, zone) filtered out. A side-by-side comparison vs the smoke
> numbers is in § 1.5.

All numbers below come from this Tracy run:
- **Hardware**: 1× Blackhole BH-38 SKU; 110 worker Tensix cores; **AICLK = 1350 MHz** (read from CSV header).
- **Workload**: `sweep_adaptec1_2048` (ISPD-2005 adaptec1, 370k cells, grid 2048×2048, 759 EP calls). Run converges (`hpwl=72.5M, overflow=0.069`).
- **Build**: V11 sort+dedup variant, `v11_max_per_page_tuples=4096`, `SRC_CHUNK=8`.
- **Captured**: 2026-05-12 with `TT_METAL_DEVICE_PROFILER=1`; `--skip-warmup 20`. Full output in `results/tracy_sweep_adaptec1_2048_2026-05-12/profile_v11.txt`.

### 1.1 Per-zone kernel times (steady-state, median & max across 110 cores)

| Zone | RISC | Median µs (M-Tracy) | Max µs (M-Tracy) | max/median | Note |
|---|---|---:|---:|---:|---|
| `BRISC-KERNEL` (full kernel) | BRISC   | 2268 | 2992 | 1.32 | Whole-kernel time per iter. |
| `NCRISC-KERNEL` (full kernel) | NCRISC | 2090 | 2629 | 1.26 | |
| `TRISC-KERNEL` (compute, all 3) | TRISC_0/1/2 | 1509 | 1669 | 1.11 | SFPU compute. |
| **`V11A-ACC`** (BRISC accum inner loop) | BRISC | **334** | **1534** | **4.60** | **The hot zone; still 4.6× imbalance after warmup.** Lower than smoke's 8.66× because real workload's K-way sharding actually fires. |
| **`V11N-ACC`** (NCRISC accum inner loop) | NCRISC | **250** | **1274** | **5.10** | |
| `V11A-MERGE` (`dense[i] += dense_n[i]`) | BRISC | 1474 | 2072 | 1.41 | Per-elem ~38 ns (51 cycles per RMW). **Workload-independent**. |
| `V11A-SCALE` (`dense[i] *= inv_ba`) | BRISC | 2539 | 2763 | 1.09 | Per-elem ~65 ns (88 cycles per mul-store). |
| `V11A-ZERO` (write-only `dense[i]=0`) | BRISC | 140 | 148 | 1.05 | Per-elem ~3.6 ns (5 cycles per store + loop). **Workload-independent**. |
| `V11N-ZERO` | NCRISC | 140 | 148 | 1.05 | Same. |
| `V11-ROUTE` (NCRISC scatter [0..512)) | NCRISC | **5517** | 5665 | 1.03 | **3× slower than smoke fixture (workload-dependent).** Well balanced. |
| `V11B-ROUTE` (BRISC scatter [512..1024)) | BRISC | **3178** | 3267 | 1.03 | **2.6× slower than smoke.** Well balanced. |
| `V11A-DATA-READ` (256 KB NOC_0 read) | BRISC | 4.99 | 13.28 | 2.66 | 256 KB / 13.28 µs = **19 GB/s (steady-state slowest-core NOC_0)** |
| `V11N-DATA-READ` (256 KB NOC_1 read) | NCRISC | 7.23 | 16.94 | 2.34 | 256 KB / 16.94 µs = **15 GB/s (steady-state slowest-core NOC_1)** |
| `V11A-HDR-READ-ALL` (read 7 KB of headers) | BRISC | 52 | 116 | 2.23 | Bulk-read header strip. |
| `V11N-HDR-READ` | NCRISC | 42 | 127 | 3.04 | |
| `V11-MAP-LOAD` (DRAM → L1: tile_to_core + shard_table) | NCRISC | 78 | 139 | 1.79 | One-shot per launch. |
| `V11A-DENSITY-WRITE` (small NOC writes) | BRISC | 44 | 45 | 1.03 | Many small 128-B writes; header-bound. |
| `V11A-LOOKUP-LOAD` (one DRAM page) | BRISC | 2.0 | 3.7 | 1.83 | One-shot owned_lookup load. |
| `V11N-LOOKUP-LOAD` (one DRAM page) | NCRISC | 2.3 | 4.0 | 1.70 | Same. |
| `V4C-SFPU-TILE` (SFPU bin geometry, per cell-tile) | TRISC_0/1/2 | **4008** | 4099 | 1.02 | **4× slower than smoke fixture** (real cell sizes hit more SFPU passes). |
| `V11B-FINAL-FLUSH` (scatter cleanup) | BRISC | 1025 | 1165 | 1.14 | Most of remaining flush work. |
| `V11-FINAL-FLUSH` | NCRISC | 734 | 880 | 1.20 | |
| `V11-CB-WAIT` (NCRISC waiting on TRISC) | NCRISC | 39 | 39 | 1.01 | Tiny — TRISC keeps up with NCRISC well. |

### 1.2 Derived per-element / per-cycle costs (steady-state, from § 1.1)

Assuming `n_total ≈ 38` dense slots on grid 2048 (= 38912 floats per dense buffer):

| Operation | µs per elem | Cycles per elem (@ 1.35 GHz) | Math |
|---|---:|---:|---|
| L1 write-only `dense[i] = 0` (M-Tracy) | 3.6 ns | **~5 cycles** | 140 µs / 38912 elems |
| L1 RMW `dense[i] += dense_n[i]` (M-Tracy) | 38 ns | **~51 cycles** | 1474 µs / 38912 elems |
| L1 RMW `dense[i] *= s` (M-Tracy) | 65 ns | **~88 cycles** | 2539 µs / 38912 elems |
| SFPU complex tile-pass (v4_compute) (M-Tracy) | 220 µs / pass | **~300K cycles / pass** | 4008 µs / 18 passes per tile |
| NOC_0 read 256 KB (slowest core, M-Tracy) | — | — | 13.28 µs → **19 GB/s** |
| NOC_1 read 256 KB (slowest core, M-Tracy) | — | — | 16.94 µs → **15 GB/s** |

The ~51 cycles for an RMW is dominated by in-order pipeline + load-use latency
(2 loads × 3 cycles + add + store + index arithmetic + loop branch). The
SCALE multiplier being slower (88 cycles) than MERGE (51 cycles) reflects fewer
overlappable loads — only 1 load vs 2 in MERGE, so less work hides the load
latency.

### 1.5 ⚠️ Smoke fixture vs steady-state — what changed

The earlier (smoke) numbers were misleadingly optimistic for the workload-dependent zones. Here's the honest side-by-side:

| Zone | Smoke (1-iter, synthetic 1.5M) | Steady-state (sweep_adaptec1_2048) | Change | Why |
|---|---:|---:|---:|---|
| V11A-ACC median | 264 µs | **334 µs** | +27 % | Real workload has more tuples/iter |
| V11A-ACC max | 2285 µs | **1534 µs** | −33 % | Real workload's sharding fires; smoke had pathological concentration |
| V11A-ACC max/median | 8.66× | **4.60×** | imbalance is REAL but smaller in practice |
| V11N-ACC max/median | 8.54× | **5.10×** | same effect |
| V11A-MERGE median | 1486 µs | **1474 µs** | 0 % | Workload-independent (just iterates dense slots) ✓ |
| V11A-SCALE median | 2195 µs | **2539 µs** | +16 % | Likely scheduling jitter |
| V11A-ZERO median | 140 µs | **140 µs** | 0 % | Workload-independent ✓ |
| V11-ROUTE (scatter) median | 1819 µs | **5517 µs** | **+203 %** | Real cell-size distribution → way more tuples per cell |
| V11B-ROUTE (scatter) median | 1226 µs | **3178 µs** | **+159 %** | Same reason |
| V4C-SFPU-TILE (compute) | 1019 µs | **4008 µs** | **+293 %** | Real cells trigger more SFPU passes (more overlap, bigger sub-cells) |
| V11A-DATA-READ median | 1.84 µs | **4.99 µs** | +171 % | DRAM bank contention at steady-state |
| V11N-DATA-READ median | 3.31 µs | **7.23 µs** | +118 % | Same |
| V11A-DATA-READ max (slowest-core NOC_0 BW) | 4.07 µs / 63 GB/s | **13.28 µs / 19 GB/s** | **−70 % BW** | The big one: smoke saw ~3× too much NOC throughput |
| V11N-DATA-READ max (slowest-core NOC_1 BW) | 7.45 µs / 34 GB/s | **16.94 µs / 15 GB/s** | **−56 % BW** | |
| Slowest core ID | (1, 2) | (1, 7) | different! | Which core is hot depends on cell distribution. |

**Key takeaways:**

1. **Architectural constants are stable**: `V11A-ZERO` (5 cycles/store), `V11A-MERGE` (51 cycles/RMW), L1 buffer sizes — all matched within ~1 %. These are safe to use in design predictions regardless of fixture.

2. **Workload-dependent zones differ a lot**: scatter (`V11-ROUTE`, `V11B-ROUTE`) and SFPU compute (`V4C-SFPU-TILE`) were 2–4× faster in smoke than in real. **Always re-measure on a real benchmark before claiming a perf win.**

3. **NOC bandwidth was the most-misleading number**: smoke said 63 GB/s on NOC_0, steady-state shows **19 GB/s**. The 3× gap reflects DRAM bank contention at sustained throughput. Design budgets should use the steady-state number.

4. **The load-imbalance signature is REAL but smaller in practice**: smoke showed 8.66× imbalance on V11A-ACC, steady-state shows 4.60×. K-way sharding actually does something on the real workload. But 4.6× is still too high — we want ≤ 1.5× (constraint § 1.3).

### 1.3 NOC bandwidth (slowest-core, what bounds wall time)

| Channel | Slowest-core BW (M-Tracy) | Median-core BW (M-Tracy) | Per transfer |
|---|---:|---:|---|
| NOC_0 (BRISC, V11A-DATA-READ) | **63 GB/s** | 137 GB/s | 256 KB per read |
| NOC_1 (NCRISC, V11N-DATA-READ) | **34 GB/s** | 77 GB/s | 256 KB per read |

The slowest-core number is what your kernel design is bounded by. Use the
**slowest-core** column for predictions.

### 1.4 Total per-iter wall time (steady-state, host side)

From `[server] done` log lines in this smoke run + steady-state sweep data:

| Phase | Time (M-host) | Note |
|---|---:|---|
| `h2d` (host → DRAM) | 2.5 ms | px/py/sx/sy SoA upload |
| `scatter` (V11) | 45.6 ms | smoke had sort+dedup overhead; production median 16–22 ms |
| `gather` (V11 accum) | 11.0 ms | This is `Finish(cq)` wall time for the accum program |
| `d2h_density` | 2.2 ms | density readback |
| `compute` (TTNN DCT) | 92.6 ms | inflated by profiler overhead in this run; production median ~3 ms |
| `download` | 91.2 ms | inflated by profiler |
| `fw` (host force-from-potential) | 20.9 ms | DREAMPlace CPU-side gradient step |
| **Total** | **274 ms** | with profiler on; production ~38 ms median |

Numbers from `results/sweep_v2_final` (no profiler) for **steady-state production**:

| Benchmark | scatter+gather wall (M-host) | CPU 8T baseline | Ratio |
|---|---:|---:|---:|
| sweep_adaptec1_2048 | ~22 ms | ~28 ms | **0.78× CPU (slower)** |
| sweep_adaptec1_512  | ~10 ms | ~8 ms  | 1.25× |
| sweep_bigblue3_2048 | ~33 ms | ~48 ms | **0.69× CPU (slower)** |
| sweep_bigblue3_512  | ~25 ms | ~25 ms | parity |

We are at or below CPU on scatter+gather wall today. The **3× CPU goal**
(targets in § 6) requires a 4–5× reduction in scatter+gather time.

---

## 2. Per-unit capabilities (with measured throughput)

### 2.1 BRISC / NCRISC scalar (RISC-V RV32IMC, 1.35 GHz)

| Operation | Throughput |
|---|---|
| Peak scalar ops (one core) | 1.35 G ops/s (E, from clock) |
| Effective `dense[i] += a` RMW | **270 M RMW/s/core (M-Tracy)** — 1 op per 51 cycles |
| Effective `dense[i] = 0` store | **270 M store/s/core (M-Tracy)** — 1 store per 5 cycles |
| Effective `dense[i] *= s` mul-store | **155 M mul-store/s/core (M-Tracy)** — 1 per 88 cycles |
| Load-use latency | ~3 cycles (E) — derived from above + pipeline knowledge |
| 2 independent NOC channels per core (NOC_0 on BRISC, NOC_1 on NCRISC) | parallel use → both at peak simultaneously |

**Practical implication**: the V11A-ACC bottleneck (scalar `dense[i] += area`)
is ~5× slower than naïve "1 op per cycle" expectations. The dense
buffer is in L1, so loads are L1-cached; the cost is **pipeline + index
arithmetic** not memory latency.

### 2.2 TRISC SFPU (vector compute, 1.35 GHz)

The SFPU is the SIMD unit on TRISC. We use it for `v4_compute.cpp` (bin
geometry: `floor`, `correct_bin_idx`, overlap multipliers).

| Metric | Throughput |
|---|---|
| Peak: 128 lanes × 1.35 GHz | ~170 G lane-ops/s/core (E, peak math) |
| Effective for `v4_compute` (18 outputs per cell-tile, **steady-state**) | **~1 cell-tile / 4008 µs/core (M-Tracy)** ≈ **4.5 G ops/s/core for our workload** |
| SFPU tile-pass cycles (per 1024-elem output, steady-state) | ~300K cycles (M-Tracy) |
| Branch (`v_if … v_endif`) overhead | ~4 cycles per predicated block (E) |

**Practical**: real SFPU throughput on actual benchmarks (steady-state) is
**~40× below peak math** for our complex bin-geometry mix. Smoke test
(1019 µs/tile) was 4× faster than real (4008 µs/tile) — don't trust smoke
SFPU numbers for design budgets.

### 2.3 TRISC FPU (matmul + eltwise tile-grain)

Used by the TTNN DCT phase, NOT by the V11 scatter/accum kernels.

| Metric | Value |
|---|---|
| Peak matmul (per Tensix) | ~1 tile/cycle = 32K MACs/cycle for bfp16 (E) |
| Per-tile DCT matmul time | Not measured per-zone here; full DCT was ~3 ms steady-state |

If you're proposing an SFPU/FPU-based accumulator, **measure first** —
the eltwise add throughput in particular is implementation-dependent.

### 2.4 L1 SRAM (per-core, 1.5 MB hard cap)

| Metric | Value |
|---|---|
| L1 scalar write (just store, no read) | **5 cycles/element (M-Tracy)** including index + loop |
| L1 scalar RMW (load + add + store) | **51 cycles/element (M-Tracy)** |
| L1 capacity | 1.5 MB (hardware fixed) |
| Cross-RISC coherence | Implicit (write-through). Tested: no fences needed (KNOWN_ISSUES #5). |
| Float atomic add | Not available (constraint table § 4 #7) |

### 2.5 NOC (Network-on-Chip) — STEADY-STATE

| Metric | Slowest-core (bound) | Median-core | Source |
|---|---:|---:|---|
| NOC_0 (BRISC) read 256 KB | **19 GB/s** | 51 GB/s | M-Tracy steady-state (V11A-DATA-READ) |
| NOC_1 (NCRISC) read 256 KB | **15 GB/s** | 35 GB/s | M-Tracy steady-state (V11N-DATA-READ) |
| NOC_0 + NOC_1 in parallel | ~34 GB/s/core (slowest) | ~86 GB/s/core (median) | Sum of above |
| `noc_semaphore_inc` round trip | ~50–100 cycles | ~50 cycles | E |
| NOC write overshoot (alignment penalty) | up to 16 bytes if size ≢ 0 mod 32 | same | KNOWN_ISSUES #3 |
| Small-write (< 256 B) overhead | header-dominated; ~1 µs each | same | derived from V11A-DENSITY-WRITE (38 ns/write × 1 µs setup) |

> Smoke fixture showed 63 GB/s on NOC_0 — that was misleadingly high.
> Steady-state with DRAM bank contention is **3× lower**.

### 2.6 DRAM

| Metric | Value | Source |
|---|---|---|
| First-byte latency | ~600 ns ≈ 800 cycles | E |
| Bulk read 256 KB via NOC, steady-state | 7.2 µs (~35 GB/s, NCRISC NOC_1) | M-Tracy |
| Bulk read 256 KB via NOC, slowest core | 16.9 µs (~15 GB/s) | M-Tracy |
| Bulk write per page | ~1 µs/page typical | E |
| Aggregate chip DRAM throughput | ~50–80 GB/s sustained | E (estimate revised downward from earlier 150 GB/s; sweep saw ~35 GB/s per NOC channel) |
| `InterleavedAddrGen` page-per-bank striping | Round-robin across DRAM banks | TT-Metal SDK |

### 2.7 Host ↔ device dispatch

| Operation | Cost (M-host) |
|---|---:|
| `EnqueueMeshWorkload(cq, wl, false); Finish(cq);` | **3–5 ms per pair** (M-host; estimate) |
| JIT compile (cold cache) | ~60 s one-time per kernel-source change |
| Container restart + chip reset | ~15 s total (`tt-smi -r 1` + container up) |
| `DeviceZoneScopedN` overhead | ~100 cycles per zone enter+exit (E) |

V11 has **two** Programs per iter (scatter, accum), so host dispatch adds
~6–10 ms / iter. Halving Programs cuts ~3–5 ms/iter — worth doing if
otherwise easy.

---

## 3. Reproducing these measurements (TT-Tracy recipe) — STEADY-STATE

```bash
# 1. Reset chip (after container restart)
bash scripts/reset_chip.sh

# 2. Run a real multi-iter sweep with profiler enabled (NOT smoke).
#    The smoke test runs 1 iter and is contaminated by cold-cache effects.
TT_METAL_DEVICE_PROFILER=1 bash scripts/run_sweep.sh sweep_adaptec1_2048

# 3. Pull the CSV out of the container (large: ~300 MB for 759 iters)
docker cp $CONTAINER:$TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv /tmp/

# 4. Aggregate per zone with warmup filtering.
#    --skip-warmup 20 drops the first 20 invocations per (core, zone),
#    which is approximately the first iter's worth of intra-iter zones.
python tools/profile_v11.py /tmp/profile_log_device.csv --skip-warmup 20

# 5. Per-core distribution for the imbalanced zone
python tools/profile_v11.py /tmp/profile_log_device.csv --zone V11A-ACC --per-core --skip-warmup 20
```

The expected steady-state output for `sweep_adaptec1_2048` is in
`results/tracy_sweep_adaptec1_2048_2026-05-12/profile_v11.txt`. Numbers within
~10–20 % across runs (scheduling jitter) are normal.

> **DON'T use the smoke harness (`run_smoke.sh`) for steady-state numbers.**
> It runs one EP call, so all measurements include first-iter cold-cache
> effects. The smoke fixture gave median V11-ROUTE = 1819 µs; real
> steady-state is 5517 µs (3× difference). Always use a multi-iter sweep.

---

## 4. Calculation framework — predict before you build

Use this template:

```
T_iter = max over 110 cores of (
    T_NOC_in     // DRAM/NOC inbound bytes ÷ slowest-core BW (§ 2.5)
  + T_NOC_out    // outbound bytes ÷ slowest-core BW
  + T_scalar     // RV32 ops × cycles-per-op (§ 2.1) ÷ 1.35 GHz
  + T_SFPU       // SFPU passes × ~75K cycles/pass (§ 2.2) ÷ 1.35 GHz
  + T_FPU        // matmul tiles × ~1 cycle/tile (§ 2.3) ÷ 1.35 GHz
  + T_sync       // 100 cycles/sema_inc (§ 2.5), CB plumbing
) + T_dispatch   // n_Programs × 4 ms (§ 2.7)
```

**The `max over 110 cores` is what bounds the iter**, not the average. If your
design has 1 slow core at 100 ms and 109 fast ones at 10 ms, your iter is
100 ms.

### Worked example — V11A-ACC inner loop (adaptec1_2048, today)

Inputs (steady-state sweep_adaptec1_2048):
- Tuples landing on slowest receiver ≈ 40,000 (estimate; max-core does ~5× more work than median; median pair is ~8000 tuples)
- L1 RMW = 51 cycles (§ 2.1, M-Tracy steady-state)

Prediction: `40k × 51 / 1.35 GHz ≈ 1.5 ms per ACC invocation on slowest core`.

Measured: `V11A-ACC max core = 1.53 ms per invocation` (steady-state). **Prediction matches within ~2%.** ✓

### Worked example — balanced V11A-ACC after re-balancing

If a re-balance brings max-core tuples to 1.5× median (target):
- Steady-state median V11A-ACC: 334 µs ≈ 8000 tuples × 51 / 1.35 GHz = 302 µs ✓
- Target max-core after rebalance: `1.5 × 334 = 500 µs`
- Today: max-core = 1534 µs → **3× speedup possible on slowest core** by re-balancing alone.

---

## 5. The bar: beat CPU at scatter+gather

CPU 8T (M-host, from `results/sweep_a1_bb3_report/summary.csv`):

| Benchmark | nc | grid | CPU 8T scatter+gather (M-host) | 3× speedup target |
|---|---:|---:|---:|---:|
| sweep_adaptec1_2048 | 370K | 2048² | **28 ms** | **≤ 9.5 ms** |
| sweep_bigblue3_2048 | 2.1M | 2048² | **48 ms** | **≤ 16 ms** |
| sweep_adaptec1_512  | 370K | 512²  | **8 ms** | **≤ 2.7 ms** |
| sweep_bigblue3_512  | 2.1M | 512²  | **25 ms** | **≤ 8.5 ms** |

V11 today (M-host, with sort+dedup):

| Benchmark | TT scatter+gather (M-host) | vs target | vs CPU |
|---|---:|---:|---:|
| sweep_adaptec1_2048 | ~22 ms | 2.3× over | 0.78× CPU |
| sweep_bigblue3_2048 | ~33 ms | 2.1× over | 0.69× CPU |
| sweep_adaptec1_512  | ~10 ms | 3.7× over | 1.25× CPU |
| sweep_bigblue3_512  | ~25 ms | 2.9× over | parity |

**We are NOT yet beating CPU on scatter+gather.** A new V11 variant should
clear 3× CPU on at least 2 of 4 benchmarks; below that, the engineering
cost doesn't pay back.

---

## 6. Pre-build checklist for a new variant

Fill in this table for your proposed design **before coding**:

| Question | Your answer |
|---|---|
| DRAM bytes in per iter (per-core, slowest) | … |
| DRAM bytes out per iter (per-core, slowest) | … |
| → NOC-bound floor (§ 2.5: bytes ÷ 34–63 GB/s) | … ms |
| Scalar ops (RV32) per iter (per-core, slowest) | … M ops |
| → Scalar-bound floor (§ 2.1: at 1 op per 51 cycles for RMW) | … ms |
| SFPU passes per iter (per-core, slowest) | … |
| → SFPU-bound floor (§ 2.2: 75K cycles/pass) | … ms |
| Programs per iter (each = 4 ms dispatch) | …× 4 ms = … ms |
| L1 footprint per core (peak) | … KB ≤ 1500 KB? |
| Estimated `max_core/median_core` ratio | … (must be ≤ 3×) |
| **Predicted total iter wall time** | **… ms** |
| **3× CPU target on hardest benchmark (bigblue3_2048)** | **≤ 16 ms** |
| **Predicted speedup vs CPU on bigblue3_2048** | **…×** |

If predicted speedup < 3× on at least 2 of 4 benchmarks, **don't build it
yet** — revise the design.

---

## 7. Provenance & verification

All (M-Tracy) rows in §1.1, §2.1, §2.2, §2.4, §2.5, §2.6 are sourced to
`results/tracy_sweep_adaptec1_2048_2026-05-12/profile_v11.txt` — a real
multi-iter steady-state sweep with `--skip-warmup 20` filtering. The earlier
smoke-test artifact (`results/tracy_smoke_2026-05-12/profile_v11.txt`) is
kept for the side-by-side comparison in §1.5 but is **not** the source of
truth for performance budgets.

(M-host) rows are sourced to:
- The server's `[server] done h2d=… scatter=… gather=…` log lines.
- `results/sweep_a1_bb3_report/summary.csv` (CPU baselines).
- `results/README.md` § "sweep_v2_final" (current V11 measurements).

(E) rows are educated guesses — flag them when you find them wrong.

To regenerate the steady-state Tracy reference:
```bash
bash scripts/reset_chip.sh
TT_METAL_DEVICE_PROFILER=1 bash scripts/run_sweep.sh sweep_adaptec1_2048
docker cp $CONTAINER:$TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv /tmp/
python tools/profile_v11.py /tmp/profile_log_device.csv --skip-warmup 20 \
    > results/tracy_sweep_adaptec1_2048_$(date -I)/profile_v11.txt
```

If you re-run on a different SKU or after kernel changes, please commit the
updated aggregation to a new `results/tracy_*` subdirectory rather than
overwriting.
