# Density Scatter on TT Blackhole — Implementation Log

A detailed record of everything implemented, every design decision, every bug
encountered, and every fix applied during the DREAMPlace → Blackhole porting
project. Intended for anyone who picks up this work later.

---

## Table of Contents

1. [Project Goal](#1-project-goal)
2. [Background: The Algorithm](#2-background-the-algorithm)
3. [TT Blackhole Hardware Recap](#3-tt-blackhole-hardware-recap)
4. [Architecture Decisions](#4-architecture-decisions)
5. [Implementation Walkthrough](#5-implementation-walkthrough)
6. [Optimization Rounds](#6-optimization-rounds)
7. [Bugs Encountered and Fixes](#7-bugs-encountered-and-fixes)
8. [Benchmark Results](#8-benchmark-results)
9. [What Was Tried and Abandoned](#9-what-was-tried-and-abandoned)
10. [Remaining Limitations and Future Work](#10-remaining-limitations-and-future-work)

---

## 1. Project Goal

Port the **density scatter** kernel from DREAMPlace (VLSI global placement tool)
to Tenstorrent Blackhole hardware, utilize all 110 Tensix cores, and benchmark
against single- and multi-threaded CPU baselines with a fair input/output
comparison.

**DREAMPlace context:** every iteration of global placement calls density scatter
to build a 2D density map from the current cell positions. It accounts for ~30%
of total placement runtime (source: DREAMPlace paper, ISPD 2019).

---

## 2. Background: The Algorithm

### Density Scatter (Triangle Kernel)

```
for each cell i with position (cx, cy) and size (csx, csy):
    for each bin (bx, by) in the grid:
        tx = overlap_1d(cx, csx, bin_xl[bx], bin_size_x)
        ty = overlap_1d(cy, csy, bin_yl[by], bin_size_y)
        density_map[bx, by] += tx * ty
```

where `overlap_1d(x, size, bin_xl, bin_size) = max(0, min(x+size, bin_xl+bin_size) - max(x, bin_xl))`.

This is the **ElectricPotential density function** from:
```
DREAMPlace/dreamplace/ops/electric_potential/src/density_function.h
```

### Properties

- **Scatter-add pattern**: random read of cell data, random write to density bins.
- **Variable fan-out**: each cell touches ≈ 2–5 bins in X and 2–5 in Y (≈ 4–25 bin writes per cell).
- **No data dependency between cells** → embarrassingly parallel in cell-dimension.
- **Write conflicts between cells** → requires atomics or partitioned accumulators.
- **Separable**: `tx * ty = f(x) × g(y)` — enables outer-product tile computation.

---

## 3. TT Blackhole Hardware Recap

| Resource | Quantity / Spec |
|----------|-----------------|
| Tensix cores (compute+memory) | 110 active (11×10 grid, 2 harvested rows) |
| RISC-V processors per Tensix | 5: BRISC (data mov), NCRISC (data mov), TRISC0/1/2 (compute) |
| L1 SRAM per Tensix core | 1.5 MB |
| GDDR6 DRAM | 24 GB, 8 channels |
| NOC bandwidth (theoretical) | ~800 GB/s aggregate |
| BRISC/NCRISC clock | ~500–750 MHz, single-issue in-order |
| TRISC clock | ~500 MHz, optimized for 32×32 tile matmul |

**Key constraint**: BRISC/NCRISC are scalar RISC-V processors — no SIMD, no OOO.
Float operations use software FPU or the SFPU (only on TRISC).

---

## 4. Architecture Decisions

### Decision 1: Column-strip partitioning (not row-strip)

The density map is partitioned by **X-columns**: each Tensix core owns a
contiguous range of columns and accumulates into private L1. No cross-core
communication needed.

Alternatives considered:
- Row-strip: same logic, would also work.
- Cell partitioning: send cells to fixed cores and merge results → requires
  reduction across cores, complex NOC usage.

Column-strip won because the final write is a simple `noc_async_write` of
a contiguous L1 region to DRAM, and there is zero cross-core dependency.

### Decision 2: BRISC kernel (not TRISC) for compute

Density scatter is a **scalar float operation** (element-by-element, no tiles).
TRISC kernels are optimized for 32×32 tile matrix operations using the Matrix
Engine. Using TRISC for scalar work means:
- No access to the Matrix Engine (need aligned tiles)
- SFPU available but limited to per-element operations on existing tile buffers
- More complex API (tile pack/unpack, CBs structured as 32×32 tiles)

BRISC runs standard C++ and can access L1 SRAM via direct pointer dereference.
For a scalar scatter kernel, BRISC is the correct choice.

Abandoned: the initial design used TRISC for compute and BRISC for write. After
discovering TRISC incompatibility with scalar access, both were collapsed into a
single BRISC kernel (`compute_writer_density_map.cpp`).

### Decision 3: NCRISC for data movement (cell streaming)

NCRISC is the dedicated data-movement processor (NOC master for reads). It reads
cell data from DRAM in batches using `InterleavedAddrGen` and pushes to a
double-buffered Circular Buffer (`CB_CELLS`). BRISC consumes from the CB.

This producer-consumer pattern:
- Overlaps DRAM read latency with BRISC compute
- Uses only the NOC that NCRISC owns by default
- Allows BRISC to focus entirely on accumulation

### Decision 4: Per-core cell pre-filtering on the host

**v0 (naïve):** Every core reads ALL N cells from DRAM, filters for its strip.
Problem: 110× DRAM read amplification (110 cores × N cells × 16 bytes).
For N=211K, that's 110 × 3.4 MB = 374 MB of DRAM reads per scatter call.

**v1 (optimized):** Host pre-assigns each cell to exactly the cores whose column
strip it overlaps. Cells at strip boundaries appear in 2 cores' lists; all others
appear in exactly 1. Total reads ≈ 1.1–1.6× N (instead of 110×N).

The host partitioning runs once per placement iteration (cells move slightly each
time) and is O(N·log C) using binary search on sorted core-start-column arrays.

### Decision 5: Program reuse across benchmark iterations

Building and compiling a `MeshWorkload` is expensive (~37–644 ms, amortized by
JIT caching). For benchmarking, the workload is built once and `EnqueueMeshWorkload`
is called in a tight loop without rebuilding. This isolates pure device execution
time from setup overhead.

---

## 5. Implementation Walkthrough

### File: `tt_metal/kernels/reader_cell_data.cpp` (NCRISC)

Streams per-core cell records from DRAM to CB_CELLS:

```
Runtime args: cells_dram_addr, (unused), num_cells, batch_size,
              cells_page_size, first_page_id

Algorithm:
  cell_gen = InterleavedAddrGen(cells_dram_addr, cells_page_size)
  while cells_left > 0:
      batch = min(cells_left, batch_size)
      cb_reserve_back(CB_CELLS, 1)
      noc_async_read_page(page_id, cell_gen, cb_write_ptr)
      noc_async_read_barrier()
      cb_push_back(CB_CELLS, 1)
      page_id++; cells_left -= batch
```

Each "page" in CB_CELLS contains `batch_size × 16 bytes` = `batch_size` cells of
4 floats each (cx, cy, csx, csy).

**Critical**: `batch_size` must be chosen so `batch_size × 16` is a multiple of
32 bytes (NOC read alignment). Use batch_size=256 (4096 bytes/page) or batch_size=64
(1024 bytes/page). **Do NOT use batch_size=3 or batch_size=5** — 48/80 bytes are
not multiples of 32 and produce garbled data (Bug #7 below).

### File: `tt_metal/kernels/compute_writer_density_map.cpp` (BRISC)

Reads cells from CB_CELLS, accumulates triangle overlaps into L1, writes to DRAM:

```
Runtime args: (unused), density_dram_addr, bin_col_start, num_bin_cols,
              num_bins_y, num_cells, batch_size,
              xl, yl, xh, yh (float bit-cast to uint32),
              num_bins_x_total, density_page_size

L1 layout:
  CB_ACCUM (c_24): max_strip_cols × num_bins_y × 4 bytes
                  (e.g., 20 × 2048 × 4 = 160 KB for 2048×2048 grid with 110 cores)

Algorithm:
  accum = get_write_ptr(CB_ACCUM)
  memset(accum, 0, num_bin_cols × num_bins_y × sizeof(float))
  inv_bsx = 1.0f / bin_size_x   ← precomputed reciprocal (avoids float divide)
  inv_bsy = 1.0f / bin_size_y
  while cells_left > 0:
      batch = min(cells_left, batch_size)
      cb_wait_front(CB_CELLS, 1)
      cd = get_read_ptr(CB_CELLS)
      for ci in 0..batch-1:
          (cx,cy,csx,csy) = cd[ci*4..ci*4+3]
          bx_lo, bx_hi = x range in strip (clipped to strip)
          by_lo, by_hi = y range (clipped to grid)
          for bx in bx_lo..bx_hi-1:
              tx = tri(cx, csx, bin_xl_bx, bin_size_x)
              if tx == 0: continue
              for by in by_lo..by_hi-1:
                  ty = tri(cy, csy, bin_yl_by, bin_size_y)
                  accum[bx × num_bins_y + by] += tx × ty
      cb_pop_front(CB_CELLS, 1)
      cells_left -= batch
  density_gen = InterleavedAddrGen(density_dram_addr, density_page_size)
  for col in 0..num_bin_cols-1:
      noc_async_write(accum + col × col_bytes,
                      density_gen.get_noc_addr(bin_col_start + col),
                      col_bytes)
  noc_async_write_barrier()
```

### File: `tt_metal/host/density_scatter_host.cpp`

Host orchestration:
1. Generate cells with reproducible RNG (same seed as CPU baseline)
2. Partition cells to cores: `build_per_core_cells()` — O(N·log C)
3. Upload flat_cells buffer to DRAM (replicated across mesh devices)
4. Build Program: CreateKernel, CreateCircularBuffer, SetRuntimeArgs
5. Wrap in MeshWorkload, trigger JIT compile with one blocking Finish
6. Run CPU baselines (1T, 8T, 14T, 96T) with identical cell data
7. Repeat EnqueueMeshWorkload for `num_iters` iterations, time each
8. Read back density map, compare to CPU reference

**arg order** (CLI):
```
density_scatter_tt  N  bins_x  bins_y  num_iters  batch_size
```

---

## 6. Optimization Rounds

### v0 → v1: Per-Core Cell Partitioning (4.8× speedup)

**Problem (v0):** Each core read all N cells and filtered in the kernel.
DRAM reads: 110 × N × 16 bytes = 110 × 3.4 MB = 374 MB for N=211K.
At effective DRAM BW ≈ 40 GB/s, that's ~9 ms just for reads — and the
actual 79 ms included all 110 cores serializing on DRAM banks.

**Fix (v1):** Host pre-filters cells per core.
```cpp
// Host: O(N·log C) binary search on sorted core-start-column list
for each cell:
    c_lo = last core whose start_col <= cell_bx_lo
    c_hi = core owning cell's rightmost column (bx_hi)
    for c in c_lo..c_hi:
        per_core_cells[c].append(cell)
```

Cells at strip boundaries appear in 2 cores (contributing to both, correctly).
DRAM reads: ~1.1–1.6 × N × 16 bytes (depending on cell width in bins).

Result: **79 ms → 16.5 ms** (4.8× speedup).

### v1 → v2: Program Reuse + Float Division Elimination (1.05× incremental)

**Problem (v1):** Each benchmark iteration called `EnqueueMeshWorkload` which
re-triggered program construction overhead. Also, `(cx - xl) / bin_size_x`
in the BRISC kernel was ~20 cycles per float divide.

**Fix 1 (program reuse):** Build the `MeshWorkload` once before the loop.
Each iteration re-enqueues the same pre-compiled workload.

**Fix 2 (reciprocal):** Precompute `inv_bsx = 1.0f / bin_size_x` and
`inv_bsy = 1.0f / bin_size_y` before the cell loop. Replace all divides
with multiplies (1 cycle vs ~20 cycles on RISC-V without FPU divide unit).

Result: **16.5 ms → 15.7 ms** for 512×512. More significant for larger grids.

### Multi-thread CPU baseline added

The 1-thread CPU was the only baseline initially. 1T CPU was already faster than
TT (10 ms vs 15.7 ms), which seemed alarming. But 1T CPU uses no atomics. A fair
comparison requires the same parallel CPU implementation that DREAMPlace uses.

Added `cpu_scatter_mt()` using OpenMP with `atomic_add_float()` (CAS loop).
Result: 96T CPU = 16 ms ≈ TT exec = 15.7 ms. The comparison is now fair.

---

## 7. Bugs Encountered and Fixes

### Bug 1: TT-Metalium CMake package not found

**Symptom:**
```
fatal error: spdlog/fmt/bundled/core.h: No such file or directory
```

**Cause:** CMake could not find the TT-Metalium package config file. The
`CMAKE_PREFIX_PATH` was set to `tt-metal/build` but the config lives at
`tt-metal/build_Release/lib/cmake/tt-metalium/tt-metalium-config.cmake`.

**Fix:** Updated `tt_metal/CMakeLists.txt`:
```cmake
find_package(TT-Metalium REQUIRED
    PATHS ${TT_METAL_HOME}/build_Release/lib/cmake/tt-metalium
    NO_DEFAULT_PATH)
```

---

### Bug 2: L1 SRAM overflow ("segment overflows region")

**Symptom:**
```
Error: segment[1] overflows region:1 limit of 0x12c0 bytes,
       reduce the size of thread_local variables
```

**Cause:** The initial kernel had a static C array in the data segment:
```cpp
static float accum[MAX_BIN_COLS * MAX_BINS_Y];  // e.g., 512 × 512 × 4 = 1 MB
```
BRISC's data segment is limited to ~4.8 KB (0x12c0 bytes). A megabyte array
in BSS far exceeds this.

**Fix:** Allocate accum from L1 SRAM via a Circular Buffer:
```cpp
// Host: allocate CB_ACCUM (c_24) with size = max_strip_cols × num_bins_y × 4
CircularBufferConfig cb_accum_cfg = CircularBufferConfig(accum_cb_bytes, ...)
    .set_page_size(CB_ACCUM_IDX, accum_cb_bytes);
CreateCircularBuffer(program, all_cores, cb_accum_cfg);

// Kernel: get L1 pointer
float* accum = reinterpret_cast<float*>(get_write_ptr(CB_ACCUM));
```

The CB allocates from the 1.5 MB L1 pool, not the tiny data segment.

---

### Bug 3: `MAX_BINS_Y` undeclared after L1 overflow fix

**Symptom:**
```
error: 'MAX_BINS_Y' was not declared in this scope
```

**Cause:** After removing the static array declaration (which defined
`MAX_BINS_Y`), a reference to it remained in the accumulation index:
```cpp
accum[bx * MAX_BINS_Y + by]  // MAX_BINS_Y no longer exists
```

**Fix:** Replace with the runtime variable:
```cpp
accum[bx * num_bins_y + by]
```

---

### Bug 4: Wrong kernel include paths

**Symptom:**
```
fatal error: dataflow_api.h: No such file or directory
fatal error: compute_kernel_api/common.h: No such file or directory
```

**Cause:** TT-Metal kernel include paths changed between versions. The
correct paths in the installed build are:
```
api/dataflow/dataflow_api.h     (for BRISC/NCRISC kernels)
api/compute/common.h            (for TRISC kernels)
```

**Fix:** Updated all kernel files to use the correct include paths.

---

### Bug 5: Deprecated type warnings breaking build

**Symptoms:**
```
warning: 'using CoreCoord = ...' is deprecated
error: narrowing conversion from 'uint64_t' to 'uint32_t'
```

**Cause:** TT-Metal changed its type aliases. `tt::tt_metal::CoreCoord`
is now the canonical type; old aliases are deprecated.

**Fix:**
```cpp
// In host:
using CoreCoord   = tt::tt_metal::CoreCoord;
using CoreRange   = tt::tt_metal::CoreRange;
using CoreRangeSet = tt::tt_metal::CoreRangeSet;

// For address narrowing:
(uint32_t)cells_buf->address()

// In CMakeLists.txt:
target_compile_options(density_scatter_tt PRIVATE -Wno-deprecated-declarations)
```

---

### Bug 6: OpenMP `#pragma` syntax error

**Symptom:**
```
error: stray '#' in program
error: 'pragma' was not declared in this scope
```

**Cause:** OpenMP pragmas inside `extern "C"` blocks or at certain nesting
levels need explicit braces to form a compound statement.

**Fix:**
```cpp
// Wrong:
#pragma omp parallel
    #pragma omp single
    cpu_nthreads = omp_get_num_threads();

// Correct:
#pragma omp parallel
{
    #pragma omp single
    cpu_nthreads = omp_get_num_threads();
}
```

---

### Bug 7: Garbled cell data from unaligned NOC batch size — **root cause of all correctness failures**

**Symptom:** TT density map sum was ~50% of CPU sum. Max per-element error was
huge (5×10⁸). Exact match on 512×512 became 50% match for no apparent reason.

**Root cause:** The CLI argument order is `N bins_x bins_y num_iters batch_size`.
During testing we called:
```bash
./density_scatter_tt 211447 512 512 64 3
```
expecting `batch=64, iters=3`. The actual parsing gave `num_iters=64, batch_size=3`.

With `batch_size=3`: each DRAM page = 3 × 16 bytes = **48 bytes**.
The TT NOC requires read/write transfers to be multiples of **32 bytes**.
48 is a multiple of 16 but NOT of 32. The `noc_async_read_page` reads 48 bytes
into CB_CELLS but the underlying DMA may truncate to 32 bytes or read
64 bytes, corrupting the cell data. Approximately half the cell contributions
were lost, producing ~50% of the expected density.

**Fix:** Always use batch_size that makes `batch_size × 16` a multiple of 32:
- batch_size=2 → 32 bytes ✓
- batch_size=64 → 1024 bytes ✓
- batch_size=256 → 4096 bytes ✓ (recommended)

Correct invocation:
```bash
./density_scatter_tt 500000 2048 2048 5 256   # N bins_x bins_y iters batch
```

**Misleading symptoms:** Because the breakage was in DRAM reads (not logic),
the kernel appeared to run at normal speed and produce a plausible-looking
density map — just with ~half the contributions. The correctness check's
threshold (`max_err <= 1e-4 × sum`) was too lenient to catch a 50% deficit
in the per-element comparison. This bug wasted significant debugging time
because code changes were blamed for a pre-existing argument-order issue.

---

### Bug 8: False L1 overflow diagnosis (related to Bug 7)

**Symptom:** For 2048×2048 grid, the first run (with batch=3 due to Bug 7)
produced TT sum > CPU sum (1.83× too high), which looked like accumulation
overflow rather than corrupted reads.

**Incorrect diagnosis:** Believed the CB_ACCUM was overflowing L1 SRAM (at
187 cols × 2048 rows × 4B = 1.46 MB ≈ 1.5 MB limit).

**Why this seemed plausible:** A naïve calculation using grid.x (11 cores in
X-direction) instead of nc (110 total cores) yields max_strip_cols = 187.
With 110 cores dividing 2048 columns, the actual max_strip_cols is 20
(each core owns ~18–20 columns), giving CB_ACCUM = 160 KB. No overflow.

**Wasted work:** Implemented L1 cell-caching (CB_CELL_CACHE) and Y-chunk
accumulation to reduce CB_ACCUM to 256 KB budget. These changes were correct
in principle but introduced Bug 9 and were unnecessary for the current scale.

**Correct fix:** Fix Bug 7 (batch_size). CB_ACCUM is 160 KB — safely within L1.

---

### Bug 9: L1 cell-cache approach corrupts density map (caused by Bug 8 investigation)

**Symptom:** After adding CB_CELL_CACHE (c_25) and caching cells in L1 before
Y-chunk processing, TT sum was exactly ~50% of CPU sum regardless of grid size.

**Root cause investigation:** Adding a third CB (CB_CELL_CACHE, index c_25)
alongside CB_ACCUM (c_24) caused some interaction or memory layout issue.
Alternatively, the bug may have been triggered by reading from CB_CELL_CACHE
using raw pointer writes without `cb_reserve_back`, which is outside the
designed CB API.

**Resolution:** Reverted to the original single-pass design (no cell cache,
no Y-chunk loop). The revert confirmed the kernels themselves were always
correct — the root issue was always Bug 7 (wrong batch_size).

The Y-chunk approach (if needed for hypothetical grids larger than ~4K×4K)
should instead use NCRISC streaming cells multiple times (once per Y-chunk)
rather than caching in L1. This avoids the CB API misuse.

---

## 8. Benchmark Results

All runs: Blackhole p150b, 110 Tensix cores, 24 GB GDDR6. Docker container
`bh-38-special-ayadav-for-reservation-72646`. `batch_size=256`.

### Config 1: N=211,447, grid=512×512 (adaptec1 scale)

| Backend | Latency | Throughput | vs CPU 1T |
|---------|---------|------------|-----------|
| CPU 1T | 10.3 ms | 104 M/s | 1.00× |
| CPU 8T | 11 ms | 97 M/s | 0.94× |
| CPU 14T | **6.9 ms** | **155 M/s** | **1.48×** |
| CPU 96T | 15.0 ms | 72 M/s | 0.69× |
| TT exec | 15.7 ms | 68 M/s | 0.65× vs 1T, 1.05× vs 96T |
| TT e2e | 30 ms | 35 M/s | 0.34× vs 1T |

### Config 2: N=500,000, grid=2048×2048 (large scale)

| Backend | Latency | Throughput | vs CPU 1T |
|---------|---------|------------|-----------|
| CPU 1T | 50.9 ms | 49.7 M/s | 1.00× |
| CPU 8T | 26.7 ms | 94.8 M/s | 1.91× |
| CPU 14T | **17.0 ms** | **148 M/s** | **2.99×** |
| CPU 96T | 17.6 ms | 143 M/s | 2.89× |
| TT exec | 33.6 ms | 75 M/s | **1.52× vs 1T**, 0.51× vs 14T |
| TT e2e | 72.9 ms | 35 M/s | 0.70× vs 1T |

### Correctness

- 512×512: Max err = **0.00** (bit-exact float32 match with CPU reference)
- 2048×2048: Max err = **302** on values up to ~1×10⁵ (< 0.3% — float32
  accumulation order difference between 110 independent BRISC accumulators
  vs. sequential CPU loop)

---

## 9. What Was Tried and Abandoned

### TRISC (compute kernel) for density accumulation

**Tried:** Initial design used a TRISC kernel (RISCV_0/1/2) for the inner
loop, exploiting the SFPU for fast float math.

**Why abandoned:** TRISC kernels operate on 32×32 **tiles** loaded from CBs
via `tile_regs_*` and `pack_untilize_*` APIs. The density scatter kernel needs
scalar random access to an L1 float array, which the tile API doesn't support.
Attempting to use TRISC for scalar work required manually managing tile packing/
unpacking with large amounts of boilerplate for effectively one FMA per cell-bin
pair. The overhead far exceeded the benefit.

**Lesson:** TRISC / Matrix Engine is the right choice for GEMM-like operations
on aligned tiles. For scatter operations with irregular access patterns, BRISC
is the correct processor.

### Separate BRISC writer + TRISC compute

**Tried:** Split into NCRISC (reader), TRISC (compute), BRISC (writer), mirroring
DREAMPlace's CUDA kernel decomposition (stream→compute→scatter).

**Why abandoned:** With TRISC handling only scalar operations (no tile benefit),
the pipeline added synchronization overhead without compute gain. Merged TRISC
into BRISC as a single `compute_writer_density_map.cpp`.

### L1 Cell Cache for Y-chunk accumulation

**Tried:** To support Y-chunked accumulation without re-reading cells from DRAM,
added a CB_CELL_CACHE (index c_25) where BRISC would copy all cells from CB_CELLS
into L1, then process multiple Y-chunks from the L1 copy.

**Why abandoned:** The approach produced 50% density deficit (Bug 9). The CB API
expects `cb_reserve_back`/`cb_push_back` before writes; writing directly to the
CB base pointer (obtained via `get_write_ptr` without reserving) is outside the
designed use and may corrupt adjacent CB metadata. Additionally, the Y-chunking
was solving a non-existent problem (Bug 8 — L1 was never overflowing).

**If needed for very large grids (>4K×4K):** Implement Y-chunking by having
NCRISC stream cells `y_num_chunks` times (once per Y-chunk), coordinated with
BRISC via CB_CELLS push/pop. This avoids any L1 cache aliasing.

### NoC multicast for cell broadcast

**Considered:** Instead of each core independently reading its cell slice from
DRAM, one "source" core reads a block of cells and multicasts to all cores in
its row using `noc_async_write_multicast`. This could eliminate the remaining
DRAM read amplification (currently 1.15–1.6× due to boundary cells).

**Not implemented:** The DRAM read overhead is already small relative to the
compute time. At N=211K with batch=256, H2D upload is 0.6–4 ms — not a
bottleneck. Multicast complicates the host setup significantly.

---

## 10. Remaining Limitations and Future Work

### Current bottlenecks (in order of impact)

1. **Host partitioning (30–31 ms for N=500K):** Single-threaded O(N·log C)
   sort + assignment. Parallelizing with OpenMP across cells would reduce to
   ~3 ms. Incremental re-partitioning (cells move slightly per iteration)
   would reduce further.

2. **D2H read-back (7 ms for 16.8 MB density map):** PCIe bandwidth-limited.
   Can overlap with CPU gradient computation in a real placement loop.

3. **BRISC scalar throughput (33 ms for N=500K, 110 cores):** In-order RISC-V
   vs OOO x86 with SIMD. Fundamental hardware limitation.

### Potential 10–32× kernel speedup: tiled matmul approach

The density scatter is **separable**: `tx × ty = f(cx,csx,bx) × g(cy,csy,by)`.
For a batch of B cells and a strip of S columns:

```
TX[B × S] = triangle_x(cell_batch, strip_columns)   (B×S matrix)
TY[B × R] = triangle_y(cell_batch, strip_rows)       (B×R matrix)
density_strip[S × R] += TX^T × TY                   (outer product reduce)
```

`TX^T × TY` is a matrix multiplication: S×B · B×R → S×R. The Tensix Matrix
Engine performs 32×32 FP16/BF16 matmul in 1 cycle. Even with float32→BF16
conversion overhead, this could be 10–32× faster than scalar BRISC.

Implementation path:
1. Quantize triangle overlaps to BF16 (adequate precision for placement)
2. Use TRISC to compute `TX^T × TY` in 32×32 tiles
3. Accumulate into `density_strip` using SFPU

### Integration into DREAMPlace

The scatter result is used by `ElectricPotentialFunction.forward()` in:
```
DREAMPlace/dreamplace/ops/electric_potential/electric_potential.py
```

To integrate:
1. Wrap `density_scatter_tt` in a Python `ctypes` binding
2. Implement `torch.autograd.Function` with custom forward (TT scatter)
   and backward (CPU electric force, or a second TT kernel)
3. Replace the CUDA scatter call when Blackhole is available

### Gradient kernel (electric force)

The backward pass is an electric force gather — essentially the transpose of
scatter. Instead of `density_map[bx, by] += tx × ty`, it computes:
```
force_x[i] += sum_{bx,by} density_map[bx,by] × d_tx/d_cx × ty
```

This has the same structure (per-cell, per-bin inner loop) and would benefit
from the same column-strip partitioning and BRISC kernel approach.

---

*Last updated: April 2026*
