# DREAMPlace + Tenstorrent Blackhole: Runtime Breakdown

**Benchmark:** adaptec1 (ISPD 2005)  
**Grid:** 512 × 512 bins  
**Cells:** ~370,971 movable + filler (NC_max allocated: 556,456)  
**Hardware:** Tenstorrent Blackhole p150b (110 Tensix cores, 24 GB GDDR6)  
**Host:** bh-38, 8-thread CPU, 48 EP forward calls per run  
**Date:** 2026-05-03

---

## 1. Architecture Overview

DREAMPlace's electric-potential forward pass (the most expensive per-iteration operation) is split across two compute domains:

```
Host CPU (Python / DREAMPlace)
  │
  ├─ Nesterov optimizer, wirelength, backward pass  ← always CPU
  │
  └─ ElectricPotentialFunction.forward()
       │
       │  [CPU mode]  native DREAMPlace OpenMP scatter + FFTW3 DCT
       │
       └─ [TT mode]   IPC → Docker container → C++ server → TT kernels
                           ↑
            pos.bin (host→server)   field_x/y.bin (server→host)
```

### TT Pipeline (7 Programs, all on Blackhole Tensix cores)

| Step | Kernel | Cores | Operation |
|------|--------|-------|-----------|
| V4  | `v4_reader.cpp` + `v4_compute.cpp` + `v4_ncrisc_scatter.cpp` | 110 Tensix | Scatter: accumulate triangle density from NC cells into M×N density map |
| V6  | `v6_gather_reader.cpp` + `v6_dct_compute.cpp` | Mt = M/32 | Gather density rows + yDCT: `temp = rho @ DCT_N^T` |
| V7  | `v6_dct_compute.cpp` (reused) | Mt | xDCT: `auv = DCT_M @ temp` |
| V8a | `v6_dct_compute.cpp` + `v_scale_tiled_reader.cpp` | Mt | Scale + row_x: `mid_x = (auv .* wu) @ IDXST_N^T` |
| V8b | `v6_dct_compute.cpp` + `v_scale_tiled_reader.cpp` | Mt | Scale + row_y: `mid_y = (auv .* wv) @ IDXCT_N^T` |
| V9a | `v6_dct_compute.cpp` + `v_tiled_matmul_reader.cpp` | Mt | col_x: `field_x = IDXCT_M @ mid_x` |
| V9b | `v6_dct_compute.cpp` + `v_tiled_matmul_reader.cpp` | Mt | col_y: `field_y = IDXST_M @ mid_y` |

**V4 Scatter detail:** Each Tensix core handles a column strip of the density grid. BRISC reads cell positions from L1 (loaded by NCRISC DMA from DRAM), computes triangle overlap area using SFPU, and NCRISC scatter-adds contributions to the appropriate grid bins in DRAM. All 110 cores operate in parallel with no synchronisation.

**V6–V9b DCT detail:** The spectral field solve decomposes into 6 matrix multiplications implementing a DCT-II → Poisson solve → IDXST/IDCT pipeline. Each program uses Mt = 16 Tensix cores (for M=512), executing tiled 32×32 FP32 matmuls. The DCT matrices (DCT_M, DCT_N, IDXST_M, IDCT_M, etc.) are pre-loaded into DRAM once at startup.

---

## 2. IPC Protocol (Host ↔ Docker Container)

```
Host                              Docker (C++ server)
────                              ──────────────────
write pos.bin  ──────────────→   read pos.bin (H2D)
write go.flag  ──────────────→   V4 Scatter kernel
                                 V6 Gather+yDCT kernel
                                 V7 xDCT kernel
                                 V8a scale+row_x kernel
                                 V8b scale+row_y kernel
                                 V9a col_x kernel
                                 V9b col_y kernel (D2H)
                                 write field_x.bin, field_y.bin
                ←────────────    write done.flag (with timings)
read done.flag
parse timings
read field_x/y.bin
```

**pos.bin layout:**  `int32 NC | float32[NC] px | float32[NC] py | float32[NC] sx | float32[NC] sy`  
**Field map layout:** `float32[512×512]` row-major, one file each for x/y components

---

## 3. CPU-Only Baseline (adaptec1, 512×512, 48 EP calls)

All operations run on 8-thread host CPU using OpenMP (scatter) and FFTW3 FFT (DCT).

| Operation | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | % of EP |
|-----------|-----------|-------------|----------|----------|---------|
| **Total EP** | **30.69** | **31.74** | 12.95 | 45.81 | 100% |
| Scatter (OpenMP density map) | 27.68 | 28.66 | 9.97 | 42.74 | ~90% |
| DCT (FFTW3 2D FFT-based) | 2.67 | 2.71 | 2.15 | 3.06 | ~9% |
| Other (overhead) | 0.33 | 0.33 | 0.24 | 0.56 | ~1% |

**Total wall time (30 global placement iterations + legalization):** 9.55 s

The EP call count (48) exceeds the configured 30 iterations because DREAMPlace adds ~18 calls during filler initialization and sub-iteration steps.

The CPU scatter time varies significantly (10–43ms) because early iterations have random cell placement (many overlap pairs), while later iterations have more spread-out placements (fewer active bins).

---

## 4. CPU + TT Blackhole (adaptec1, 512×512, 48 EP calls)

The TT server JIT-compiles 19 kernel programs on the first run (~18.8 s, once per cold kernel-cache). Subsequent runs reuse the compiled cache. All timing below is from the **second run (warm JIT cache)** and excludes the first EP call (which absorbs JIT time).

### 4.1 Server-Side Breakdown (inside Docker, measured by C++ server)

| Operation | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | % of server compute |
|-----------|-----------|-------------|----------|----------|---------------------|
| **Server compute total** | **94.95** | **95.27** | 77.07 | 116.05 | 100% |
| H2D (read pos.bin from filesystem) | 0.77 | 0.75 | 0.72 | 0.91 | ~0.8% |
| **V4 Scatter (TT kernel)** | **7.06** | **7.06** | 6.93 | 7.20 | ~7.4% |
| **DCT (V6–V9b, 6 TT matmul programs)** | **80.10** | **80.97** | 61.66 | 101.99 | ~84.4% |
| D2H (read computed fields from DRAM) | 2.11 | 2.07 | 1.95 | 2.48 | ~2.2% |
| Field write (write field_x/y.bin to filesystem) | 4.72 | 4.68 | 4.46 | 6.12 | ~5.0% |

### 4.2 Host-Side Breakdown (measured by Python client)

| Operation | Mean (ms) | Median (ms) | Min (ms) | Max (ms) |
|-----------|-----------|-------------|----------|----------|
| pos.bin write (host→filesystem) | 6.21 | 6.05 | 5.89 | 21.10 |
| go.flag write (IPC signal) | 0.23 | 0.22 | 0.21 | 0.31 |
| kernel_wait (poll until done.flag) | 97.66 | 98.67 | 78.48 | 121.23 |
| field_read (read field_x/y.bin) | 0.69 | 0.69 | 0.58 | 0.74 |
| IPC overhead (signaling latency) | 2.91 | 2.90 | 2.28 | 3.26 |


| **Total client EP time** | **104.91** | **105.61** | 88.24 | 128.47 |

> **IPC overhead** = `kernel_wait_ms − (h2d + scatter + dct + d2h)`. Captures flag-file polling latency, filesystem cache flush delays, and OS scheduling jitter.

### 4.3 Full Round-Trip Waterfall (median values)

```
Host writes pos.bin           ██████  6.05 ms
Host writes go.flag           █  0.22 ms
  Server: H2D (read pos.bin)   █  0.75 ms
  Server: V4 Scatter           ████████  7.06 ms
  Server: V6 Gather+yDCT       ████████████████████████████████████████████ ~20 ms
  Server: V7 xDCT              ████████████████████████ ~12 ms
  Server: V8a scale+row_x      ████████████████████████ ~12 ms
  Server: V8b scale+row_y      ████████████████████████ ~12 ms
  Server: V9a col_x            ████████████████████████ ~12 ms
  Server: V9b col_y            ████████████████████████ ~12 ms
                               [V6–V9b total: ~81 ms]
  Server: D2H (write fields)   ███  2.07 ms
  Server: write field_x/y.bin  █████  4.68 ms
Host reads done.flag          (included in kernel_wait, IPC OH: ~2.9 ms)
Host reads field_x/y.bin      █  0.69 ms
──────────────────────────────────────────────────────
Total EP round-trip                                   105.61 ms
```

**Total wall time (30 iterations + initialization):** 14.44 s

---

## 5. CPU vs TT Comparison

| Operation | CPU (ms) | TT (ms) | Ratio | Winner |
|-----------|----------|---------|-------|--------|
| Scatter | 28.66 | 7.06 | **4.06× faster** | **TT** |
| DCT field solve | 2.71 | 80.97 | **29.9× slower** | CPU |
| Total EP | 31.74 | 105.61 | **3.33× slower** | CPU |
| Wall time (30 iters) | 9.55 s | 14.44 s | 1.51× slower | CPU |

### Per-category cost breakdown (median, TT run)

```
                   CPU mode     TT mode
                   ─────────    ───────────────────────────────────────────────────────
pos.bin write      —            ████   6.05 ms
H2D                —            █  0.75 ms
Scatter            ████████████████████████████  28.66 ms   ████ 7.06 ms  (4× speedup)
DCT V6-V9b         ██  2.71 ms  ████████████████████████████████████████████████  80.97 ms
D2H                —            █  2.07 ms
Field write        —            ███  4.68 ms
Field read         —            █  0.69 ms
IPC overhead       —            ██  2.90 ms
─────────────────────────────────────────────────────────
Total EP           ████████████████████  31.74 ms   ████████████████████████████████████  105.61 ms
```

---

## 6. Per-Kernel Analysis

### V4 Scatter (the win)
- **TT: 7.06ms vs CPU: 28.66ms → 4.06× speedup**
- 110 Tensix cores process cell→bin contributions in parallel (column-strip partitioning)
- Each core handles `N/110` columns; NCRISC DMA reads cell batches from DRAM while BRISC computes triangle overlaps via SFPU
- Consistent timing (6.93–7.20ms range) indicates compute-bound, not memory-bound
- Scales well: the scatter is the most parallelizable operation and TT's data-flow architecture suits it

### V6–V9b DCT Field Solve (the bottleneck)
- **TT: 80.97ms vs CPU: 2.71ms → 29.9× slower**
- Implements DCT as 6 sequential matrix multiplications: each program must complete before the next starts
- For 512×512: each matmul is 512×512 @ 512×512, executed in 32×32 tiles on Mt=16 cores
- CPU alternative (FFTW3) uses an `O(N² log N)` FFT algorithm with highly optimized SIMD; the matmul approach is `O(N³)` per axis without tiling overlap benefits
- DCT time varies with iteration (61–102ms) — early iterations have denser maps that cause more active tiles in V6 gather; later iterations converge to sparser maps

### H2D Transfer
- **0.75ms** to read 370K × 4 floats × 4 bytes = **~5.9 MB** from shared filesystem into DRAM
- Bottlenecked by filesystem cache flush on the host side (pos.bin write = 6.05ms dominates)
- Combined host-write + H2D = 6.8ms total data ingestion overhead

### D2H Transfer
- **2.07ms** to move 2 × 512×512 × 4 bytes = **~2.1 MB** from DRAM
- Plus **4.68ms** server-side file write and **0.69ms** host-side file read
- Combined outbound data path = 7.44ms

### IPC Signaling Overhead
- **2.90ms** of polling latency from flag-file based synchronization
- Includes: OS scheduler wake-up delay, filesystem metadata flush, 500µs sleep intervals in the polling loop
- Could be reduced to ~0.1ms with POSIX named semaphores or shared memory signaling

---

## 7. Observations and Bottleneck Analysis

### Why the TT run is slower end-to-end

The total TT EP time (105.61ms) is 3.3× slower than CPU (31.74ms) because:

1. **DCT dominates at 77% of TT server time** (80.97ms vs 7.06ms scatter). The 6-matmul DCT implementation is computationally expensive compared to FFTW3's FFT-based approach.

2. **IPC overhead is non-trivial:** pos.bin write (6.05ms) + field files write+read (5.37ms) + IPC signaling (2.90ms) = **14.3ms per iteration** that has no CPU equivalent.

3. **Sequential matmul programs:** V6→V7→V8a→V8b→V9a→V9b run sequentially with a `Finish(cq)` barrier between each. There is no pipeline overlap.

### Where TT is genuinely faster

- **Scatter is 4× faster** on TT (7ms vs 29ms). For a design with more cells or a future version with larger grids where scatter dominates, TT hardware would provide real speedup.

### Theoretical break-even

If DCT were as fast on TT as on CPU (2.71ms), the TT pipeline would be:
`0.75 + 7.06 + 2.71 + 2.07 + 4.68 + 0.69 + 2.90 = 20.86ms` per EP call — **1.52× faster than CPU**.

Adding the fixed IPC overhead (14.3ms) currently prevents this; a shared-memory IPC implementation would bring it to ~6.5ms total overhead.

---

## 8. Key Numbers at a Glance

```
Benchmark:   adaptec1 (ISPD 2005), 512×512 grid, 370K cells
Hardware:    Blackhole p150b, 110 Tensix cores

                    CPU-only    CPU + TT     Speedup
                    ────────    ────────     ───────
EP median / iter:   31.74 ms   105.61 ms    0.30× (3.3× slower)
Wall time (30 it):  9.55 s      14.44 s     0.66× (1.5× slower)

Sub-operation breakdown (TT mode, median):
  pos.bin write (host):    6.05 ms   ← IPC cost, shared filesystem
  H2D (server reads):      0.75 ms   ← DRAM DMA ingestion
  V4 Scatter (TT):         7.06 ms   ← 4× faster than CPU scatter
  DCT V6 Gather+yDCT:     ~20 ms    ← (estimated, part of 80.97ms total)
  DCT V7 xDCT:            ~12 ms    ← (estimated)
  DCT V8a scale+row_x:    ~12 ms    ← (estimated)
  DCT V8b scale+row_y:    ~12 ms    ← (estimated)
  DCT V9a col_x:          ~12 ms    ← (estimated)
  DCT V9b col_y:          ~12 ms    ← (estimated)
  DCT total (V6–V9b):     80.97 ms  ← 30× slower than CPU DCT
  D2H (server writes):     2.07 ms   ← field maps to DRAM
  Field file write:        4.68 ms   ← server writes 2×1MB files
  Field file read (host):  0.69 ms   ← host reads field_x/y.bin
  IPC overhead:            2.90 ms   ← flag-file polling latency
  ─────────────────────────────
  Total EP (client):      105.61 ms
```

---

## 9. Methodology

- Runs used `integration/run_dreamplace.py` with timing hooks patching `ElectricPotentialFunction.forward`
- CPU scatter timed by patching `ElectricDensityMapFunction.forward`; DCT timed by wrapping `dct2/idct_idxst/idxst_idct` objects
- TT timings from C++ server's `done.flag` payload: `OK h2d=X scatter=Y dct=Z d2h=W fw=V`
- IPC overhead = `client kernel_wait_ms − (h2d + scatter + dct + d2h)` (server-reported)
- All per-iteration data saved to CSV; medians reported here to exclude JIT-cold outliers
- First TT run: JIT compiles 19 programs (~18.8s one-time cost); second run (warm cache) used for all medians above
- CPU `ep_mean` varies across iterations (10–46ms) because early random placement creates denser maps requiring more scatter work
