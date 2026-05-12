# Scatter + DCT Field Solve: Architecture Options

This document describes the current state of TT-Metal acceleration for DREAMPlace's electric potential step and the viable paths to combine the V4 scatter kernel with a fast DCT solver.

---

## Background: The Electric Potential Pipeline

DREAMPlace's `ElectricPotentialFunction.forward()` runs two stages each iteration:

1. **Density scatter** (`ElectricDensityMapFunction`): Maps cell positions to a 2-D density grid (M×N bins). This is the computationally heavy scatter step — each cell contributes a bilinear weight to up to 4 bins.
2. **DCT field solve**: Solves the Poisson equation on the density grid using DCT to produce electric field vectors (field_x, field_y). The CPU path uses FFTW3.

For adaptec1 at 512×512:
- CPU scatter: ~30 ms/iter
- CPU DCT (FFTW3): ~2.7 ms/iter
- CPU total EP: ~33 ms/iter

---

## Implemented Modes

### Mode: `cpu`
Pure CPU baseline. Uses DREAMPlace's built-in FFTW3 DCT solver. Reference for correctness and runtime.

### Mode: `tt` (V4–V9b full pipeline)
C++ server (`density_scatter_full_pipeline`) running inside Docker. Replaces **both** scatter and DCT with TT-Metal kernels:
- V4 kernels: BRISC scatter contributions → DRAM
- V6 gather_reader: gather + normalize + tilize + first matmul DCT
- V7–V9: remaining 5 matmuls + pointwise scale + field extract

Measured: V6 DCT phase = ~81 ms (all 6 matmuls + overhead). This is **slower** than CPU FFTW3 (2.7 ms) due to heavy NOC traffic in the distributed matmul layout.

### Mode: `ttnn_dct` (CPU scatter + TTNN field solve)
Python worker (`ttnn_dct_worker.py`) running inside Docker. Keeps CPU scatter unchanged, replaces only the DCT solve with `TTNNFieldSolver` (6 dense matmuls on Tensix using TTNN Python API).

Measured on adaptec1_short (512×512, 48 iters):
| Component | Time |
|---|---|
| CPU scatter | 30.3 ms |
| Tensix compute (6 matmuls) | **0.52 ms** |
| TT upload (density → device) | 0.52 ms |
| TT download (fields → CPU) | 0.90 ms |
| Field file write (IPC) | 1.91 ms |
| IPC polling overhead | 1.51 ms |
| Density write (IPC) | 1.18 ms |
| Field read (IPC) | 0.77 ms |
| **Full round-trip** | **7.47 ms** |

CPU DCT baseline: 2.73 ms.

Key insight: **Tensix compute is 5.3× faster than CPU DCT** (0.52 ms vs 2.73 ms), but file-based IPC adds ~6.9 ms overhead, making end-to-end 2.7× slower. The IPC bottleneck is file writes (two 1 MB field files = 1.91 ms) and polling latency (1 ms sleep intervals = 1.51 ms).

---

## Why TTNN DCT Beats V6 C++ DCT

TTNN's `TTNNFieldSolver` uses the same 6-matmul decomposition as V6–V9, but executes on a single Tensix core using the high-level TTNN matmul op, which maps efficiently to the BFP8/BFP16 matrix engine. V6 distributes across 110 cores with significant inter-core NOC traffic for tile passing — the communication overhead dominates.

- V6 C++ DCT: ~81 ms
- TTNN Python DCT: **0.52 ms** (156× faster)

The scatter step is unaffected by either DCT approach.

---

## Path Options

### Path 1 (Recommended): Unified C++ Binary — V4 Scatter + Gather-Density + TTNN C++ DCT

**Architecture**: A single C++ server binary that:
1. Runs V4 scatter kernels to accumulate contributions in DRAM (unchanged)
2. Runs a new `v6_gather_density_only` BRISC kernel to produce a row-major normalized density buffer in DRAM
3. Downloads density to CPU host memory (one D2H read)
4. Re-uploads density as a TTNN tile tensor and runs `TTNNFieldSolver` via TTNN C++ API (`_ttnncpp.so`)
5. Downloads field_x/field_y back to CPU and sends to DREAMPlace client

**Why unified binary**: The TT device can only be opened by **one process at a time**. A subsequent open (after close) takes ~120 ms — far too expensive for per-iteration handoff. Both V4 scatter and TTNN must run in the same process.

**Estimated speedup over `ttnn_dct` mode**: Eliminates file-based IPC (~3.6 ms savings on density_write + field_read + field_write). The scatter result is already in DRAM so no extra density write over IPC. Only D2H/H2D transfers for density (1 MB) and field (2 MB) remain, plus TTNN compute (0.52 ms).

**New kernel**: `v6_gather_density_only.cpp` — adapted from `v2_gather_density.cpp` (pure gather, row-major DRAM write) with normalization added (÷ inv_bin_area, same as V6 Phase 2a). No TRISC/NCRISC needed, BRISC only.

**Build**: New CMake target in `experiments/density_scatter/tt_metal/CMakeLists.txt`, linked against `_ttnncpp.so` + `libtt_metal.so` + `libtt_stl.so`, compiled with `clang++-20 -std=c++2a -DSPDLOG_FMT_EXTERNAL`.

**Status**: In progress.

---

### Path 2: Shared Memory IPC (Python Worker, Drop File I/O)

Replace file-based IPC in the current `ttnn_dct` mode with POSIX shared memory (`mmap`/`shm_open`). The Python worker polls a shared-memory flag byte (busy-wait or `usleep(10)`) instead of file existence.

**Expected improvement**: Eliminates the 1.91 ms field file write and the 1.51 ms polling latency (replaces with ~0.1 ms busy-poll overhead). Density write would also become a `memcpy` (~0.1 ms for 1 MB) instead of `np.tofile` (1.18 ms).

**Estimated round-trip**: ~0.52 ms compute + ~0.1 ms memcpy + ~0.2 ms D2H + ~0.1 ms signal = ~1 ms total, vs. 2.73 ms CPU DCT → **net speedup**.

**Complexity**: Medium. Requires Python `mmap` + C-level shared memory, plus careful synchronization (double-buffering or epoch counter). Docker `/dev/shm` must be sized appropriately (`--shm-size`).

**Status**: Not implemented.

---

### Path 3: C++ Pybind11 Bridge (Zero-Copy, Single Process)

Expose `TTNNFieldSolver` as a Python extension module (`ttnn_poisson_pybind.so`) that DREAMPlace imports directly. No separate worker process, no IPC at all. The density tensor stays in PyTorch/CPU memory; the pybind layer calls TTNN C++ API inline.

**Expected improvement**: Zero IPC overhead. Full round-trip = upload (0.52 ms) + compute (0.52 ms) + download (0.90 ms) = **~2 ms**, vs. 2.73 ms CPU DCT → modest speedup.

**Complexity**: High. Requires matching Python ABI (pybind11 must be compiled against the exact Python version DREAMPlace uses), and the TT device can only be opened once — must coordinate with any other TT-Metal usage in the same process.

**Status**: Not implemented.

---

## Summary Table

| Mode | Scatter | DCT Compute | IPC Overhead | Total EP (est.) | Status |
|---|---|---|---|---|---|
| `cpu` | CPU 30 ms | CPU FFTW3 2.7 ms | — | ~33 ms | Done |
| `tt` (V4-V9) | TT-Metal | TT-Metal 81 ms | file ~2 ms | ~83 ms | Done |
| `ttnn_dct` | CPU 30 ms | Tensix 0.52 ms | file 6.9 ms | ~39 ms | Done |
| **Path 1** | TT-Metal | Tensix TTNN C++ | D2H/H2D ~2 ms | **~35 ms** | In progress |
| Path 2 | CPU 30 ms | Tensix 0.52 ms | shm ~0.3 ms | **~31 ms** | Not started |
| Path 3 | CPU 30 ms | Tensix inline | none | **~31 ms** | Not started |

Path 1 targets replacing the full V4-V9 pipeline (83 ms DCT) with TTNN (0.52 ms) while keeping scatter on TT-Metal. Paths 2/3 target the `ttnn_dct` IPC bottleneck.

---

## Files

| File | Role |
|---|---|
| `integration/ttnn_dct_client.py` | Host-side IPC client for `ttnn_dct` mode |
| `integration/ttnn_dct_worker.py` | Docker-side TTNN DCT worker for `ttnn_dct` mode |
| `integration/run_dreamplace.py` | DREAMPlace runner with `--device` flag |
| `field_solver/ttnn_poisson_solver.py` | `TTNNFieldSolver` — Python TTNN 6-matmul DCT |
| `TTPort/dreamplace_ref/.../ttnn_poisson_solver.py` | Same solver, packaged for Docker worker |
| `experiments/density_scatter/tt_metal/kernels/v4_*.cpp` | V4 scatter kernels |
| `experiments/density_scatter/tt_metal/kernels/v2_gather_density.cpp` | Reference for Path 1 gather-density kernel |
| `experiments/density_scatter/tt_metal/kernels/v6_gather_reader.cpp` | V6 gather+DCT (reference for normalize step) |
| `experiments/density_scatter/tt_metal/host/density_scatter_full_pipeline_host.cpp` | V4-V9 full pipeline host (Path 1 base) |
