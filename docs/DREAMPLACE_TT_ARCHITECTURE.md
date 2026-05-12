# DREAMPlace on Tenstorrent Hardware: Architecture and Kernel Reference

How cell positions flow from a Python placement optimizer into TT-Metal kernels on
Blackhole, through a DCT field solver, and back as electric force fields — every
stage documented with data layouts and kernel internals.

---

## 1. System Overview

DREAMPlace is an analytical placement engine. Its inner loop calls an
**Electric Potential (EP) forward pass** each iteration: given current cell positions,
compute the density map and solve Poisson's equation to obtain per-cell force fields.
On CPU this is scatter + FFTW3. On Tenstorrent Blackhole the same math runs via:

1. **V4 Scatter** — Tensix kernels across all cores accumulate cell-to-bin contributions
2. **Gather-Density** — Mt cores reduce contributions to a normalized density map
3. **TTNN DCT** — six TTNN matmuls solve the Poisson equation on-device
4. **IPC** — Python host and C++ server share data through a memory-mapped file

The C++ server (`density_scatter_ttnn_server`) runs inside a Docker container that
has the Blackhole device mounted. The Python DREAMPlace process runs on the host.
They communicate through a POSIX mmap of an NFS-backed file — on the same physical
host the Linux page cache is shared, so reads and writes from either side are
effectively zero-copy.

---

## 2. End-to-End Pipeline

```
HOST  (Python process, dp_env)          DOCKER  (C++ server, Blackhole device open)
══════════════════════════════          ══════════════════════════════════════════════

DREAMPlace Nesterov optimizer
  │  iter N: cell pos tensor
  │
  ├─ ElectricPotentialFunction.forward
  │    (monkey-patched by scatter_ttnn_client.py)
  │
  ├─ write px/py/sx/sy → scatter.shm          ─────── shared mmap (ipc_shm/scatter.shm) ───────►
  │    (~0.45 ms, zero-copy into mapped pages)
  │
  ├─ shm_state ← GO                           ─────── cache-line write visible to server ───────►
  │                                                                                          │
  │  [spin-poll shm_state]                                                          poll shm_state == GO
  │                                                                                          │
  │                                                                            ├─ memcpy positions
  │                                                                            │    shm → local vectors
  │                                                                            │
  │                                                                            ├─ H2D  (~0.6 ms)
  │                                                                            │    EnqueueWriteMeshBuffer
  │                                                                            │    px/py/sx/sy → DRAM
  │                                                                            │
  │                                                                            ├─ V4 Scatter  (~6–19 ms)
  │                                                                            │    all Tensix cores
  │                                                                            │    BRISC / TRISC / NCRISC
  │                                                                            │
  │                                                                            ├─ Gather  (~33–75 ms)
  │                                                                            │    ├─ V6 sparse  (M×N > 512²)
  │                                                                            │    └─ V7 dense   (M×N ≤ 512²)
  │                                                                            │
  │                                                                            ├─ D2H density  (~0.2 ms)
  │                                                                            │    EnqueueReadMeshBuffer
  │                                                                            │    density[M×N] ← DRAM
  │                                                                            │
  │                                                                            ├─ TTNN DCT  (~0.7–2.4 ms)
  │                                                                            │    6 matmuls on-device
  │                                                                            │    → field_x[M×N], field_y[M×N]
  │                                                                            │
  │                                                                            ├─ write fields → scatter.shm
  │                                                                            │    (~0.18 ms)
  │                                                                            │
  │                                                                            ├─ write 9 timing floats
  │                                                                            │    into shm header
  │                                                                            │
  │  [spin-poll exits]                         ◄────── shm_state ← DONE ──────┘
  │
  ├─ read field_x/field_y from shm  (~0.34 ms)
  │    .copy() → heap tensors (2 × M×N × float32)
  │
  └─ return to DREAMPlace backward pass
       → per-cell force gradients
       → optimizer step
  │
  iter N+1 ──────────────────────────────────────────────────────────────────────────────────►
```

---

## 3. IPC: Shared Memory Layout

`ipc_shm/scatter.shm` is created by the Python client with `ftruncate`, then
mmapped by both sides. The server opens it after JIT compiles and writes
`ready.flag` to signal readiness.

```
Offset  Size    Field           Description
──────  ──────  ──────────────  ─────────────────────────────────────────────
0       4 B     state           uint32: 0=IDLE  1=GO  2=DONE  3=QUIT
4       4 B     nc_actual       int32: number of valid cells this iteration
8       36 B    timing[9]       float32 ×9: h2d, scatter, gather, d2h_density,
                                  ttnn_upload, ttnn_compute, ttnn_download,
                                  field_write, total  (server → client)
44      4 B     gather_mode     uint32: 0=v6  1=v7
48      16 B    padding         (completes 64-byte cache line)
─── Data region ─────────────────────────────────────────────────────────────
64      soa_padded×4 B  px[]    float32, soa_padded = ceil(nc_max/1024)*1024
+1×     soa_padded×4 B  py[]
+2×     soa_padded×4 B  sx[]   half-widths (clamped)
+3×     soa_padded×4 B  sy[]   half-heights (clamped)
+4×     M×N×4 B         field_x[]  row-major float32, written by server
+5×     M×N×4 B         field_y[]  row-major float32, written by server
```

**State machine:**

```
  Python                         C++ Server
    │                                │
    │──── write pos into shm ────►   │
    │──── state = GO ─────────────►  │
    │                                │── read nc_actual, memcpy positions
    │                                │── run scatter / gather / DCT
    │                                │── write fields into shm
    │                                │── write timing into header
    │◄─── state = DONE ─────────────│
    │──── read fields (.copy()) ──►  │
    │──── state = IDLE ───────────►  │
    │                             (poll for next GO)
```

---

## 4. V4 Scatter Kernel

### Role

Distribute `nc_total` cells across all Tensix cores. Each core:
1. Reads a tile of 1024 cells from DRAM (positions + half-sizes) — BRISC
2. Computes which density bins the cell overlaps and by how much (SFPU) — TRISC
3. Writes contributions to DRAM — NCRISC, in one of two modes:
   - **V6 path** (`v4_ncrisc_scatter.cpp`): counting-sort contributions by destination
     gather core, write a sparse sorted page to `contrib_buf[my_page]`
   - **V7 path** (`v4_ncrisc_scatter_dense.cpp`): **no sorting** — accumulate directly
     into per-gather-core dense 32×N strips, write strips to `strips_buf`

The counting sort only exists in the V6 path. V7 eliminates it entirely by writing
pre-sorted dense strips during scatter, so gather can do sequential reads instead
of random-access pulls from sparse pages.

### Core assignment

```
nc_total cells → ceil(nc_total / 1024) tiles
                → distributed across all nc_all Tensix cores

core c gets:
  first_tile = c * base_tpc + min(c, rem_tpc)
  n_tiles    = base_tpc + (c < rem_tpc ? 1 : 0)
```

### Per-core processor roles

```
Tensix Core c
════════════════════════════════════════════════════════════════════════════

  BRISC  (RISCV_0) — v4_reader.cpp
  │
  └── for each tile t in first_tile .. first_tile + n_tiles - 1:
        │
        ├─ noc_async_read_page(t, px_dram → CB_0)   ┐
        ├─ noc_async_read_page(t, py_dram → CB_1)   │  4 reads issued
        ├─ noc_async_read_page(t, sx_dram → CB_2)   │  back-to-back
        ├─ noc_async_read_page(t, sy_dram → CB_3)   ┘
        ├─ noc_async_read_barrier()                     ← one barrier per tile
        └─ cb_push_back × 4                             → signals TRISC

  TRISC  (compute) — v4_compute.cpp                    [runs concurrently with BRISC]
  │
  └── for each tile t:
        │
        ├─ cb_wait_front(CB_0..3)                       ← waits for BRISC push
        │
        ├─ 18 SFPU passes over 1024 cells (8-wide per face):
        │    Pass  1:  bxl = floor((px − xl) / bsx)   → CB_4
        │    Pass  2:  byl = floor((py − yl) / bsy)   → CB_5
        │    Pass  3–10:  overlap_x[j], j=0..7        → CB_6..CB_13
        │    Pass 11–18:  overlap_y[k], k=0..7        → CB_14..CB_21
        │
        └─ cb_pop_front(CB_0..3),  cb_push_back(CB_4..21) → signals NCRISC

  NCRISC  (RISCV_1) — v4_ncrisc_scatter.cpp     (V6: counting-sort path)
                    — v4_ncrisc_scatter_dense.cpp (V7: dense-strip path)
  │
  └── for each tile t:
        │
        ├─ cb_wait_front(CB_4..21)                      ← waits for TRISC push
        │
        ├─ [V6]  for each of 1024 cells × up to 8×8 bin overlaps:
        │          form (bx, by, area) → unsorted[]
        │        counting-sort unsorted[] by dest gather core
        │        noc_async_write(header + sorted data → contrib_buf[my_page])
        │        noc_async_write_barrier()
        │
        └─ [V7]  for each of 1024 cells × up to 8×8 bin overlaps:
                   form (bx, by, area)
                   partial_strips[bx/32][bx%32][by] += area   (direct accum, no sort)
                 (after all tiles) noc_async_write(strips → strips_buf)
                 noc_async_write_barrier()
```

### SFPU bin-index and overlap computation (v4_compute.cpp)

For each of the 1024 cells in the tile (processed 8-wide per SFPU face):

```
bxl = max(0, floor((px - xl) × inv_bsx))       ← corrected for FP rounding
byl = max(0, floor((py - yl) × inv_bsy))

For j in 0..MAX_OVERLAP-1:
  bin_left  = xl + (bxl + j) × bsx
  bin_right = bin_left + bsx
  overlap_x[j] = max(0, min(px+sx, bin_right) - max(px, bin_left))

For k in 0..MAX_OVERLAP-1:
  overlap_y[k] = max(0, min(py+sy, bin_top) - max(py, bin_bottom))
```

Produces 18 output tiles per input tile. The `correct_bin_idx` correction catches
±1 ULP errors from the SFPU reciprocal multiply.

### NCRISC scatter (V6 path): v4_ncrisc_scatter.cpp

```
L1 scratch layout (CB_SCRATCH):
  [0 .. 512 B)                header[nc_all+1]  — per-dest bucket counts
  [512 .. 512+max_contrib×8)  unsorted[]        — (bx, by, area) structs
  [.. +max_sorted×8)          sorted[]          — sorted contributions
  [.. +128×4)                 running[]         — prefix-sum working copy
  [.. +M×4)                   bx2dest[]         — precomputed column→core map

Steps per kernel invocation:
  1. Wait for 18 CB tiles from TRISC
  2. For each of 1024 cells × MAX_OVERLAP² bins:
       form (bx, by, area), append to unsorted[]
  3. Count: header[bx2dest[bx]]++ for each contribution
  4. Prefix-sum with 8-element alignment padding
  5. Scatter unsorted → sorted by destination
  6. NOC-write header (512 B) + sorted data to DRAM contrib_buf[my_page]
```

### NCRISC scatter (V7 path): v4_ncrisc_scatter_dense.cpp

```
Instead of a sparse sorted page, accumulates contributions directly
into 32-column density strips:

L1 scratch:
  partial_strips[Mt][32][N]  — one 32×N accumulator per gather core

For each tile: form (bx, by, area) as V6, but immediately:
  strip_idx = bx / 32            → which gather core owns this col
  local_col = bx % 32
  partial_strips[strip_idx][local_col][by] += area

After all tiles: NOC-write strips[my_id][gather_core] to DRAM strips_buf
  (nc_all × Mt pages of 32×N floats each)
```

---

## 5. Gather Kernels

Two gather kernels exist. The server auto-selects based on grid size:
- **V7 dense-strip**: grid ≤ 512×512 (strip fits in L1)
- **V6 sparse pull**: grid > 512×512

Both run on `Mt = M / 32` cores. Each gather core owns 32 x-columns of the density
grid and writes `32 × N` float32 density values to DRAM.

### 5a. V6 Sparse Gather (v6_gather_density_only.cpp)

```
Gather Core c  (owns cols col_start .. col_start+31)
════════════════════════════════════════════════════════════════════════════

  INIT
  ────
  accum[32×N] ← 0

  ┌── for s = 0 .. nc_src-1  (nc_src ≈ 110 scatter cores) ──────────────────┐
  │                                                                           │
  │   noc_async_read(contrib_buf[s].page, hdr_buf, 512 B)                    │
  │   ░░░░░░░░░░░░░░░░░░ BARRIER 1 ░░░░░░░░░░░░░░░░░░░░░░  ← stall ~500 ns │ ← bottleneck
  │                                                                           │
  │   off_start = hdr_buf[my_id]                                              │
  │   off_end   = hdr_buf[my_id + 1]                                          │
  │   count     = off_end − off_start                                         │
  │                                                                           │
  │   if count == 0 ──────────────────────────────────────────────────────►  │
  │                  (skip to next s)                                         │
  │   ┌── while done < count ──────────────────────────────────────────────┐ │
  │   │   chunk = min(count − done, max_bucket)                            │ │
  │   │   noc_async_read(page + HEADER + done×8, cb_buf, chunk×8 B)       │ │
  │   │   ░░░░░░░░░░░░░░░░ BARRIER 2 ░░░░░░░░░░░░░  ← stall ~500 ns      │ │ ← bottleneck
  │   │   for i in 0..chunk:                                               │ │
  │   │     accum[cb_buf[i].bx − col_start][cb_buf[i].by] += cb_buf[i].area│ │
  │   │   done += chunk                                                    │ │
  │   └────────────────────────────────────────────────────────────────────┘ │
  └───────────────────────────────────────────────────────────────────────────┘

  FINALIZE
  ────────
  accum[:] *= inv_bin_area
  noc_async_write(accum col 0..31 → density_buf)
  noc_async_write_barrier()
```

**Per-source timeline (serial stalls):**

```
  s=0:  │─issue hdr─│░░░BARRIER 1░░░│─check offset─│─issue data─│░░░BARRIER 2░░░│─accumulate─│
  s=1:  │─issue hdr─│░░░BARRIER 1░░░│─check offset─│─issue data─│░░░BARRIER 2░░░│─accumulate─│
  s=2:  │─issue hdr─│░░░BARRIER 1░░░│─check offset─│─issue data─│░░░BARRIER 2░░░│─accumulate─│
  ...   (repeated for all 110 sources)

  ░ = core stalled, waiting for DRAM round-trip (~200–500 ns each)

  Minimum stall: 110 sources × 2 barriers × 200 ns = 44 μs
  Actual measured: 33–78 ms  (inner chunk loop multiplies barrier count
                               when each source's bucket exceeds max_bucket)
```

### 5b. V7 Dense-Strip Gather (v7_gather_dense.cpp)

```
Gather Core c  (owns cols c×32 .. (c+1)×32-1)
════════════════════════════════════════════════════════════════════════════

  INIT
  ────
  accum[32×N] ← 0

  ┌── for s = 0 .. nc_src-1  (nc_src ≈ 110 scatter cores) ──────────────────┐
  │                                                                           │
  │   noc_async_read_page(strips_buf[s×nc_gath + my_id], incoming, 32×N×4 B)│
  │   ░░░░░░░░░░░░░░░░░░░░ BARRIER ░░░░░░░░░░░░░░░░░░░░░░  ← stall ~500 ns │ ← bottleneck
  │                                                                           │
  │   for i in 0..32×N:  accum[i] += incoming[i]                             │
  │                                                                           │
  └───────────────────────────────────────────────────────────────────────────┘

  FINALIZE
  ────────
  accum[:] *= inv_bin_area
  noc_async_write(accum col 0..31 → density_buf)
  noc_async_write_barrier()
```

**Per-source timeline (serial stalls):**

```
  s=0:  │─issue strip read (64 KB)─│░░░░░░░BARRIER░░░░░░░│─element-wise add─│
  s=1:  │─issue strip read (64 KB)─│░░░░░░░BARRIER░░░░░░░│─element-wise add─│
  s=2:  │─issue strip read (64 KB)─│░░░░░░░BARRIER░░░░░░░│─element-wise add─│
  ...   (repeated for all 110 sources)

  ░ = core stalled, waiting for DRAM round-trip (~200–500 ns each)

  Minimum stall: 110 sources × 1 barrier × 200 ns = 22 μs
  Actual measured: 67–75 ms  (64 KB strip read at limited bandwidth adds
                               transfer time on top of the round-trip latency)

  V7 halves the barrier count vs V6 (1 per source instead of 2),
  but the serial stall chain remains.
```

V7 eliminates the two-read-per-source pattern of V6 by precomputing dense strips
during scatter (NCRISC writes directly to the right gather core's strip). Each
source is now one large sequential read (32×N×4 bytes) rather than a sparse header
+ variable-length data pair. But each read still has its own barrier — the serial
latency chain of 110 barriers remains.

**V7 is only used for 512×512** because strip size = 32 × 512 × 4 = 64 KB per
gather core, and Mt = 16 gather cores × 110 source strips = 110 MB total strips
buffer. For 1024×1024 the strips buffer would be 440 MB, exceeding practical limits.

### Gather mode comparison

```
                   V6 Sparse               V7 Dense-Strip
─────────────────  ─────────────────────   ──────────────────────────────
Scatter output     contrib_buf:            strips_buf:
                   per-core pages with     pre-sorted dense 32×N strips
                   (bx, by, area) sorted   per gather core per scatter core
                   by dest gather core

Gather reads       2 NOC reads per src:    1 NOC read per src:
per source         header (512 B) +        full strip (32×N×4 B = 64 KB)
                   data (variable)

Gather barriers    2 × nc_src              1 × nc_src

Bandwidth          Low (sparse, ~useful    High (always 64 KB/strip even
                   bytes only)             if strip is mostly zero)

Grid size          Any                     ≤ 512×512 (L1 constraint)

Steady-state       Benefits from cell      Flat across all iterations
timing             convergence (sparse
                   contributions → faster)
```

---

## 6. TTNN DCT Field Solver

After gather produces a normalized density map `ρ[M×N]`, the Poisson equation
`∇²φ = −ρ` is solved in frequency space using Discrete Cosine Transforms.
Six TTNN matmul operations implement the full 2D DCT → scale → inverse DCT pipeline.

### Mathematical pipeline

```
Input: ρ[M×N]  (normalized density map, float32)

Step 1: 2D DCT-II
  temp = ρ  @ DCT_N^T          # [M×N] @ [N×N]
  auv  = DCT_M @ temp           # [M×M] @ [M×N]
  auv is the 2D DCT-II spectrum of ρ

Step 2: Eigenvalue scaling (element-wise, not matmul)
  wu[u,v]  = πu / M / (wu² + wv²)   × 0.5
  wv[u,v]  = πv / N / (wu² + wv²)   × 0.5  × (bsx/bsy)
  fx_auv   = auv × wu               # [M×N] element-wise
  fy_auv   = auv × wv               # [M×N] element-wise

Step 3a: Inverse transform for field_x  (IDXST_IDCT)
  temp_x   = fx_auv @ IDCT_N^T      # [M×N] @ [N×N]
  field_x  = (2×IDXST_M) @ temp_x  # [M×M] @ [M×N]

Step 3b: Inverse transform for field_y  (IDCT_IDXST)
  temp_y   = fy_auv @ IDXST_N^T    # [M×N] @ [N×N]
  field_y  = (2×IDCT_M)  @ temp_y  # [M×M] @ [M×N]

Output: field_x[M×N], field_y[M×N]  (electric force field, float32)
```

Six matmuls total (the two element-wise multiplies are TTNN `multiply`, not `matmul`).
All weight matrices (DCT_N_T, DCT_M, IDXST_M, IDCT_N_T, IDCT_M, IDXST_N_T, wu, wv)
are precomputed once at server startup and held in device DRAM as TTNN tensors.

### DCT matrix sizes and TTNN timing

| Grid | Matrix dims | TTNN compute time |
|------|------------|------------------|
| 512×512 | 512×512 each | ~0.6–0.7 ms |
| 1024×1024 | 1024×1024 each | ~0.9 ms |
| 2048×2048 | 2048×2048 each | ~2.3–2.4 ms |

TTNN pads tensors to multiples of 32. The `tt_tensor_to_vec` helper strips the
padding rows/columns after D2H to return exactly `M×N` elements.

---

## 7. DRAM Buffer Layout

All buffers are `MeshBuffer` (replicated, device-local, DRAM type):

```
Buffer          Pages                   Page size           Total (adaptec1, 512×512)
─────────────   ──────────────────────  ──────────────────  ─────────────────────────
px_buf          n_tiles pages           4 KB (1024 floats)  363 pages × 4 KB = 1.5 MB
py_buf          n_tiles pages           4 KB                1.5 MB
sx_buf          n_tiles pages           4 KB                1.5 MB
sy_buf          n_tiles pages           4 KB                1.5 MB
contrib_buf     nc_all pages            contrib_pgsz        ~110 × variable KB
density_buf     M pages (x-columns)     N × 4 B             512 × 2 KB = 1 MB
strips_buf (V7) nc_all × Mt pages       32×N×4 B            110×16 × 64 KB = 109 MB
                (only allocated for V7)
```

---

## 8. Server Startup Sequence

```
docker exec density_scatter_ttnn_server  M N NC_max ipc_dir [xl yl xh yh]
════════════════════════════════════════════════════════════════════════════

  │
  ├─ MeshDevice::create_unit_mesh(0)
  │    open Blackhole device, query compute_with_storage_grid_size()
  │
  ├─ Compute sizing
  │    n_tiles    = ceil(NC_max / 1024)
  │    base_tpc   = n_tiles / nc_all       (tiles per core, base)
  │    max_contrib = base_tpc × 1024 × MAX_BINS_PER_CELL
  │                  → L1-capped to fit scatter scratch in 1316 KB
  │    soa_padded = n_tiles × 1024
  │    contrib_pgsz, strip_pgsz, density_pgsz
  │
  ├─ Allocate DRAM buffers  (MeshBuffer, replicated)
  │    px_buf / py_buf / sx_buf / sy_buf   — soa_padded × 4 B each
  │    contrib_buf                          — nc_all × contrib_pgsz  (V6)
  │    strips_buf                           — nc_all × Mt × strip_pgsz  (V7 only)
  │    density_buf                          — M × N × 4 B
  │
  ├─ Build Program 1: V4 Scatter  (all nc_all cores)
  │    ├─ BRISC  v4_reader.cpp             — streams 4 SoA tiles from DRAM → CBs
  │    ├─ TRISC  v4_compute.cpp            — 18 SFPU passes per tile
  │    └─ NCRISC v4_ncrisc_scatter.cpp     (V6: counting-sort → contrib page)
  │            or v4_ncrisc_scatter_dense.cpp  (V7: accumulate → dense strips)
  │         SetRuntimeArgs for all nc_all cores
  │
  ├─ Build Program 2: Gather  (Mt = M/32 cores)
  │    ├─ V6: BRISC  v6_gather_density_only.cpp  — sparse NOC pulls → density
  │    └─ V7: BRISC  v7_gather_dense.cpp          — strip reads → density
  │         SetRuntimeArgs for Mt cores
  │
  ├─ JIT compile  (~60 s cold, cached on subsequent runs)
  │    EnqueueMeshWorkload(wl_scatter) + Finish()
  │    EnqueueMeshWorkload(wl_gather)  + Finish()
  │
  ├─ TTNNDCTSolver::init()
  │    Build 8 matrices on CPU  (DCT_N_T, DCT_M, IDXST_M, IDCT_N_T,
  │                               IDCT_M, IDXST_N_T, wu, wv)
  │    Upload all matrices to device as TTNN TILE tensors
  │
  ├─ Open scatter.shm  (mmap, O_RDWR)
  │    map header + px/py/sx/sy + field_x/field_y views into process address space
  │
  ├─ Write ready.flag  → Python client unblocks
  │
  └─ Main loop: poll shm_state  (see §3 state machine)
```

---

## 9. Per-Iteration Timing Breakdown

Each EP call from DREAMPlace triggers one full pass through the server:

```
Client side (Python):                   Server side (C++):
  pos_write_ms    ~0.45 ms  ──────►       [mmap write, zero-copy]
  [set state=GO]            ──────►       h2d_ms          ~0.6 ms
                                          scatter_ms      ~6 ms  (adaptec1)
                                                          ~19 ms (bigblue2)
                                          gather_ms       ~66 ms (V7, 512²)
                                                          ~33 ms (V6, 1024²)
                                          d2h_density_ms  ~0.2 ms
                                          ttnn_upload_ms  ~0.6 ms
                                          ttnn_compute_ms ~0.7 ms
                                          ttnn_download_ms ~1.1 ms
                                          fw_ms (shm write) ~0.18 ms
  field_read_ms   ~0.34 ms  ◄──────       [mmap read + .copy()]
  [kernel_wait_ms wraps the entire server-side time above]

Total EP call (adaptec1, 512×512, mmap IPC): ~82 ms median
```

---

## 10. Key Files

| File | Role |
|------|------|
| `integration/scatter_ttnn_client.py` | Host-side IPC client, mmap lifecycle, DREAMPlace monkey-patch |
| `integration/run_dreamplace.py` | Top-level runner: device setup, benchmark loading, metrics collection |
| `experiments/density_scatter/tt_metal/host/density_scatter_ttnn_server_host.cpp` | C++ server: device open, kernel build, server loop, TTNN DCT |
| `experiments/density_scatter/tt_metal/kernels/v4_reader.cpp` | BRISC: streams 4 SoA tiles from DRAM into CBs |
| `experiments/density_scatter/tt_metal/kernels/v4_compute.cpp` | TRISC: SFPU bin-index + overlap computation (18 passes/tile) |
| `experiments/density_scatter/tt_metal/kernels/v4_ncrisc_scatter.cpp` | NCRISC V6: counting-sort contributions → DRAM contrib page |
| `experiments/density_scatter/tt_metal/kernels/v4_ncrisc_scatter_dense.cpp` | NCRISC V7: accumulate into dense strips → DRAM strips buffer |
| `experiments/density_scatter/tt_metal/kernels/v6_gather_density_only.cpp` | BRISC: sparse NOC pulls from contrib pages → density |
| `experiments/density_scatter/tt_metal/kernels/v7_gather_dense.cpp` | BRISC: sequential strip reads → density (512×512 only) |
| `BENCHMARKS.md` | Measured timings, benchmark configs, run instructions |
| `DENSITY_PLANNING_APPROACHES.md` | Algorithmic proposals for gather speedup |
