# DREAMPlace × TT-Metal Integration — Path 1: V4 Scatter + TTNN C++ DCT

This document describes Path 1 of the TT-Metal acceleration of DREAMPlace's
electric potential (density + field) computation.  Everything here is live,
compiled, and producing results against the adaptec1 benchmark.

---

## What is the electric potential kernel?

DREAMPlace's global placer uses a differentiable density model.  Every
optimizer iteration calls `ElectricPotentialFunction.forward`, which:

1. **Density scatter** — maps each movable cell's (position, size) to a
   fractional contribution over the bins it overlaps in the M×N placement grid.
   Result: a float32 density map ρ(u, v) of shape (M, N).
2. **Normalize** — divide by bin area (bsx × bsy).
3. **DCT field solve** — compute the electrostatic-force field by:
   - 2-D DCT-II of ρ → spectral coefficients A(u, v)
   - Pointwise multiply by eigenvalue weights (W_u, W_v)
   - Inverse transforms (IDXST × IDCT and IDCT × IDXST) → field_x, field_y
4. The backward pass bilinear-interpolates field_x/field_y at each cell centre
   to get density-force gradients.

On a pure-CPU DREAMPlace run (adaptec1, 512×512 bins, ~371 k movable cells)
each forward call takes **~36 ms** (scatter ~33 ms + DCT ~2.7 ms).

---

## Path 1 Architecture

Path 1 replaces **both** the scatter and the DCT with TT-Metal acceleration,
running in a single C++ server process that owns the device for the entire
DREAMPlace run:

```
Host process (dp_env Python, DREAMPlace optimizer)
  │
  │  patch_dreamplace() replaces ElectricPotentialFunction.forward
  │
  │  Per-iteration:
  │   1. Write pos.bin  ──────────────────────────────────────────────────┐
  │   2. Write go.flag  ──────────────────────────────────────────────────┤ file-based
  │   3. Poll done.flag ◄──────────────────────────────────────────────────┤ IPC
  │   4. Read field_x/y.bin ◄─────────────────────────────────────────────┘
  │
  └─ returns zero scalar to DREAMPlace (backward reads ctx.field_map_x/y)

Docker container (density_scatter_ttnn_server C++ binary)
  │
  ├─ Startup (once):
  │   • Open Blackhole device (MeshDevice)
  │   • JIT-compile V4 scatter + V6 gather kernels (2 Programs)
  │   • Upload 8 DCT/IDCT/IDXST matrices as TTNN device tensors
  │   • Write ready.flag
  │
  └─ Per-iteration:
      • Read pos.bin → H2D upload (px, py, sx, sy SoA)
      • Run V4 scatter (110 Tensix cores)
      • Run V6 gather  (16 Tensix cores)
      • D2H density readback (~1 MB float32)
      • TTNN 6-matmul DCT solve → field_x, field_y
      • Write field_x.bin, field_y.bin
      • Write done.flag with per-stage timing
```

---

## Files

| File | Role |
|---|---|
| `experiments/density_scatter/tt_metal/host/density_scatter_ttnn_server_host.cpp` | C++ server: device init, kernel dispatch, TTNN DCT, IPC loop (598 lines) |
| `experiments/density_scatter/tt_metal/kernels/v4_reader.cpp` | RISCV_0 reader: loads px/py/sx/sy tiles from DRAM SoA into CBs |
| `experiments/density_scatter/tt_metal/kernels/v4_compute.cpp` | TRISC SFPU: computes bxl/byl and 1-D overlap integrals (8+8 CBs out) (268 lines) |
| `experiments/density_scatter/tt_metal/kernels/v4_ncrisc_scatter.cpp` | RISCV_1: reads SFPU output CBs, counting-sorts by gather-core destination, writes contribution page to DRAM (216 lines) |
| `experiments/density_scatter/tt_metal/kernels/v6_gather_density_only.cpp` | BRISC: reads all scatter pages, accumulates density strip, normalises, writes to DRAM (119 lines) |
| `experiments/density_scatter/tt_metal/CMakeLists.txt` | Builds `density_scatter_ttnn_server` against TT::Metalium + `_ttnncpp.so` |
| `integration/scatter_ttnn_client.py` | Host-side IPC client + `patch_dreamplace()` (343 lines) |
| `integration/scatter_ttnn_worker.py` | Docker exec wrapper: sets env, execs the binary (54 lines) |
| `integration/run_dreamplace.py` | Top-level runner: `--device scatter_ttnn` wires everything together |

---

## V4 Scatter Kernel (3 co-operating RISC-V cores per Tensix)

The V4 scatter runs on all 110 available Tensix compute cores simultaneously.
Cells are pre-sorted on the host by area (largest first) then interleaved
round-robin across 1024-element tiles for load balance.  Each core owns
`ceil(NC_max/1024) / 110 ≈ 5` tiles (≈5120 cells).

**RISCV_0 reader (`v4_reader.cpp`):**
Reads the SoA position arrays (px, py, sx, sy) from DRAM in 1024-element tiles
and pushes into CBs 0–3 for the TRISC engine.

**TRISC SFPU compute (`v4_compute.cpp`):**
For each 1024-element tile, runs 18 SFPU passes over 256-element DST
register faces:

- Pass 1 — `face_bxl`: bxl = max(0, ⌊(cx − xl) / bsx⌋), with a
  correction step that checks exact bin boundaries to compensate for SFPU
  reciprocal-multiply ULP rounding.
- Pass 2 — `face_byl`: same for Y axis.
- Passes 3–10 — `face_overlap_x<J>` for J=0..7: computes the 1-D X overlap
  of cell cx±csx/2 with bin bxl+J:
  ```
  overlap_x[J] = max(0, min(cx+csx/2, xl+(bxl+J+1)*bsx) − max(cx−csx/2, xl+(bxl+J)*bsx))
  ```
- Passes 11–18 — `face_overlap_y<K>` for K=0..7: same for Y.

Each pass writes one tile-sized output CB.  The kernel uses `fp32_dest_acc_en`
so all arithmetic stays in float32 throughout the SFPU pipeline.

**RISCV_1 NCRISC scatter (`v4_ncrisc_scatter.cpp`):**
Waits for all 18 output CBs from TRISC.  For each cell in the tile:
- Reads bxl (int), byl (int), overlap_x[0..7], overlap_y[0..7]
- Iterates J=0..7, K=0..7; breaks on first zero overlap (early exit for small/zero-size cells)
- For each valid (bx, by) pair: records a `V4Contrib{bx, by, area=ox[J]*oy[K]}`
- Applies bounds check: skips if bx ≥ nbx or by ≥ nby

After all tiles: counting-sorts the contributions by destination gather core
(using a bx→core lookup built at startup), writes the sorted contribution page
to DRAM (header + aligned buckets per gather core).

**Sizes and limits (512×512 grid, 110 scatter cores, 16 gather cores):**
- NC_max = nc_actual × 1.5 (auto-sized at first call)
- Max tiles per scatter core = ceil(NC_max / 1024) / 110 ≈ 5
- L1 scratch per core = ~964 KB (< 1.5 MB Blackhole L1)
- DRAM contribution buffer = 110 × 499 KB ≈ 52 MB

---

## V6 Gather-Density Kernel

16 gather cores each own 32 x-columns of the 512-column density grid.  This
kernel replaces the gather+density+normalize steps of the earlier V2/V5 split
pipeline with a single BRISC kernel that leaves the result directly in DRAM as
a row-major float32 array ready for TTNN.

**Per-iteration work:**
1. For each of the 110 scatter core contribution pages:
   - Read 512-byte header (NOC async read) → find offset range for this gather
     core's bucket
   - If bucket is non-empty: stream contributions in `max_bucket`-sized chunks
     from DRAM into L1
   - For each `Contrib{bx, by, area}`: accumulate into L1 float array
     `accum[col_local × nby + by]`
2. Multiply every element of `accum` by `1/(bsx × bsy)`.
3. Write 32 normalized columns back to the density DRAM buffer (one NOC write
   per column).

The gather is the current **bottleneck**: it performs 110 header reads +
variable-sized data reads (total ~50 MB DRAM traffic) with a scattered write
pattern, taking **76 ms median** (decreasing from ~94 ms early to ~59 ms late
as cells cluster and contribution counts fall).

---

## TTNN C++ DCT Solver

Implemented in `density_scatter_ttnn_server_host.cpp` as `TTNNDCTSolver`.
Initialised once per (M, N, bsx, bsy) tuple; all 8 matrices live on device
for the full DREAMPlace run.

**Mathematics** (matching Python `TTNNFieldSolver` exactly):

```
Auv  = DCT_M  @ rho @ DCT_N^T           # 2-D DCT-II → spectral coefficients

# Eigenvalue weighting
wu(u,v) = π·u/M / ((π·u/M)² + (π·v/N · bsx/bsy)²) × 0.5
wv(u,v) = π·v/N·bsx/bsy / ... × 0.5

fx_auv = Auv ⊙ Wu                        # pointwise multiply
fy_auv = Auv ⊙ Wv

field_x = (2·IDXST_M) @ fx_auv @ IDCT_N^T   # IDXST_IDCT transform
field_y = (2·IDCT_M)  @ fy_auv @ IDXST_N^T  # IDCT_IDXST transform
```

**6 matmuls total** (all 512×512 × 512×512 float32 tile-layout):
`rho @ DCT_N^T`, `DCT_M @ temp`, `fx_auv @ IDCT_N^T`, `IDXST_M @ temp_x`,
`fy_auv @ IDXST_N^T`, `IDCT_M @ temp_y`.

TTNN pads tensors to multiples of 32; the host trims the output back to M×N
during D2H readback.

**Steady-state timing:** ~0.71 ms compute (after first-call JIT warmup of
~17 ms).

---

## IPC Protocol

All communication uses files in a shared directory (bind-mounted at
`/localdev` in Docker):

```
pos.bin       [int32   nc_actual          ]
              [float32 px[nc_actual]      ]  ← sorted by area, interleaved across tiles
              [float32 py[nc_actual]      ]
              [float32 sx[nc_actual]      ]
              [float32 sy[nc_actual]      ]

field_x.bin   [float32 field_x[M*N]      ]  ← row-major
field_y.bin   [float32 field_y[M*N]      ]

done.flag     "OK h2d=X scatter=Y gather=Z d2h_den=W upload=A compute=B download=C fw=D total=E\n"

ready.flag    "READY\n"  (written once by server after JIT + TTNN init)
go.flag       "GO\n"     (written by client each iteration)
quit.flag     "QUIT\n"   (written by client at teardown)
```

**Cell sort:** the host Python pre-sorts cells by area (largest first) and
interleaves them across 1024-element tiles so that each scatter core receives
a balanced range of cell sizes.  The sort indices and sorted sx/sy arrays are
cached after the first call and reused for all subsequent iterations (cell sizes
do not change during global placement).

**Known race condition (fixed):** A stale `ready.flag` left by a previous
server session would cause `ScatterTTNNClient.start()` to return immediately,
before the new server had completed its startup cleanup (which includes
`unlink(go_flag)`).  The client would then write `go.flag` for the first
request, and the new server's startup would silently delete it, causing the
server to poll indefinitely while the client timed out at 120 s.

Fix: `start()` removes `ready.flag` before launching the subprocess, so the
client only proceeds once the new server has written a fresh flag.

---

## Cell-Count Padding

The server pre-allocates DRAM buffers for `NC_max = nc_actual × 1.5` cells.
The client writes only `nc_actual` positions; the server zero-pads the
remaining `NC_max − nc_actual` entries.  Zero-padded cells have position (0, 0)
and size (0, 0).  The V4 SFPU kernel handles them correctly: `bxl = max(0, …)
= 0` and `overlap_x[0] = max(0, min(0, bin_right) − max(0, xl)) = 0` (since
xl > 0 in any real benchmark), so no contribution is generated and no scatter
core produces garbage entries.

---

## Building

```bash
cd experiments/density_scatter/tt_metal
mkdir -p build_ttnn && cd build_ttnn
cmake .. \
  -DCMAKE_CXX_COMPILER=clang++-20 \
  -DTT_METAL_HOME=/localdev/ayadav/tt-work/TTPort/tt-metal
make density_scatter_ttnn_server -j$(nproc)
```

Requires `_ttnncpp.so` from a TT-Metal build that includes TTNN C++ headers.
The CMake file skips the target silently if `_ttnncpp.so` is not found.

**Kernel header compatibility:** the V4/V6 kernels use `#if __has_include`
guards so the same source compiles against both the newer path layout
(`api/compute/common.h`, `api/dataflow/dataflow_api.h`) and the older layout
(`compute_kernel_api/common.h`, `dataflow_api.h`).

**`NUM_CIRCULAR_BUFFERS`:** the `sc_unpack` vector (passed to `ComputeConfig`)
must have exactly `NUM_CIRCULAR_BUFFERS` elements.  The server includes
`<tt-metalium/circular_buffer_constants.h>` to get this value at compile time
(64 for Blackhole/ayadav tree, 32 for Wormhole/aagarwal tree) rather than
hardcoding it.

---

## Running

```bash
# CPU baseline
python3 integration/run_dreamplace.py \
    --device cpu \
    --benchmark DREAMPlace/test/ispd2005/adaptec1_short.json

# Path 1 (V4 scatter + TTNN C++ DCT)
python3 integration/run_dreamplace.py \
    --device scatter_ttnn \
    --benchmark DREAMPlace/test/ispd2005/adaptec1_short.json \
    --container bh-38-special-ayadav-for-reservation-73401
```

Outputs:
- `results/<benchmark>_<device>_metrics.json` — aggregate timing stats
- `results/<benchmark>_<device>_iters.csv` — per-iteration breakdown

---

## Measured Performance — adaptec1_short (512×512 bins, ~371 k cells, 48 iterations)

| Stage | CPU | scatter_ttnn | Δ |
|---|---|---|---|
| **EP call (end-to-end)** | **36.2 ms** | **106.4 ms** | 2.9× slower |
| Density scatter | 32.8 ms | **6.9 ms** | **4.7× faster** |
| DCT field solve | 2.7 ms | **0.71 ms** | **3.9× faster** |
| Gather (v6) | — | 76.2 ms | ← **bottleneck** |
| H2D cell positions | — | 0.72 ms | — |
| D2H density map | — | 0.22 ms | — |
| IPC overhead | — | ~10 ms | pos.bin write + polling |

All median values, warmup iterations excluded.  Gather time decreases over the
run as cells cluster (from ~94 ms at iteration 1 to ~59 ms by iteration 48).
First call is ~1.3 s due to TTNN JIT warmup (~17 ms compute → ~0.71 ms
thereafter).

**Bottleneck analysis:** the gather kernel reads ~52 MB of DRAM
(110 scatter-core contribution pages) per iteration, with a random write
pattern into the accumulation buffer.  This is far below peak Blackhole DRAM
bandwidth.  The V4 scatter and TTNN DCT individually outperform their CPU
equivalents; the gather is what currently makes the overall pipeline slower
than CPU.

**What would bring parity or speedup:** reducing gather DRAM traffic by
(a) coalescing scatter contributions before writing, or (b) moving the density
accumulation into a multi-core reduction rather than having each gather core
scan all 110 scatter pages independently.

---

## Accuracy

The TTNN DCT field solve produces bit-identical results to the Python
`TTNNFieldSolver`, which was separately validated against DREAMPlace's
reference CPU DCT (correlation > 0.99 at 512×512).

End-to-end placement quality (HPWL, overflow convergence) requires a full
DREAMPlace run to convergence.  The adaptec1_short benchmark (30-iteration
global + legalization) completes in **14.0 s wall time** with scatter_ttnn
vs **9.7 s** for the CPU baseline — consistent with the 2.9× per-iteration EP
slowdown.
