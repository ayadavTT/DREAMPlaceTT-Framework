# TT-Metal Kernels (V11 + supporting V4)

The V11 pipeline runs across all 110 Tensix cores on Blackhole. Each Tensix has three RISC cores (BRISC, NCRISC, TRISC compute) and a 1.5 MB L1 SRAM bank. This directory holds the kernel sources the server JIT-compiles at startup.

## Pipeline flow (per iteration)

```
  per-cell tile in DRAM      one cell-tile = 1024 cells
  ────────────────────────
       ▼
  v4_reader (BRISC)          reads px, py, sx, sy SoA tiles into CBs c_0..c_3
       ▼
  v4_compute (TRISC SFPU)    bxl, byl, overlap_x[0..7], overlap_y[0..7] → c_4..c_21
       ▼
  v11_scatter_dm (NCRISC)    cells [0..512) — route tuples to owners
  v11_scatter_b_dm (BRISC)   cells [512..1024) — route + also acts as reader/scatter for its half
       ▼   tuples in route_buf DRAM
  v11_accum_dm (BRISC)       reads route_buf[0..nc/2) — accumulate to dense_b
  v11_accum_n_dm (NCRISC)    reads route_buf[nc/2..nc) — accumulate to dense_n
       ▼   merge dense_n into dense_b in L1
  (BRISC continues) Phase A: shard-write hot-tile partials to shard_reduce_buf
                    Phase B: hot primary reads its K shards, sums them
                    Phase C: scale by inv_bin_area + write density tile to DRAM
       ▼
  TTNN C++ DCT (6 matmuls)   density → potential
       ▼
  v_gather (host-side post-DCT) — read field back to host
```

## File-by-file

| Kernel | RISC | What it does |
|---|---|---|
| `v4_reader.cpp` | BRISC | Streams `px/py/sx/sy` SoA tiles from DRAM into CBs c_0..c_3 (one cell-tile = 1024 cells = 4 KB per channel). Reused unchanged from V4. |
| `v4_compute.cpp` | TRISC SFPU | For each cell: compute integer bin index (`bxl`, `byl`) and per-axis overlap arrays `overlap_x[0..7]`, `overlap_y[0..7]`. Pushes 18 output tiles to CBs c_4..c_21. Uses SFPU floor + a 1-ULP bin-edge correction. |
| `v11_scatter_dm.cpp` | NCRISC | Walks cells [0..512) of each cell-tile, forms `(bx, by, area)` tuples for the 8×8 overlap footprint, routes each to its owner core via the `tile_to_core[]` map, stages in L1, NOC-writes batched flushes to `route_buf[my_writer_id, recv]` DRAM pages. **Has writer-side sort+dedup** (insertion sort by `(bx<<16|by)`, run-length-combine, pad to multiple of 4) before each flush. |
| `v11_scatter_b_dm.cpp` | BRISC | Twin of v11_scatter_dm.cpp for cells [512..1024). Also acts as the DRAM reader on its NOC channel. Coordinates with NCRISC via two L1 semaphores (`data_ready`, `brisc_done`). |
| `v11_histogram.cpp` | NCRISC | Phase 3a "hot-tile prepass": runs the same SFPU front-end as scatter but counts per-tile contribution counts to a per-core hist buffer in DRAM. Host reduces hist across cores to identify the K-way "hot" tiles. |
| `v11_accum_dm.cpp` | BRISC | Combined Phase 2 + 3 accumulator on the receive side. Reads `route_buf[*][me]` in SRC_CHUNK=8 batches, hash-on-receive `dense[map_idx → local_slot]`. Then merges NCRISC's dense_n half. Then runs hot-tile shard-write (Phase A), shard-sum (Phase B), scale (Phase C), density-write to DRAM. |
| `v11_accum_n_dm.cpp` | NCRISC | Twin of v11_accum_dm.cpp for the other half of writers `[nc_split..nc_all)`. Zeros dense_n, accumulates its half of writers, then signals BRISC via a per-core L1 semaphore (`merge_sem`) to release the merge. |
| `v11_reduce_a_dm.cpp` | BRISC | (Phase 3 legacy split) Shard owners write their dense slot to shard_reduce_buf DRAM page and signal the primary. Inlined into `v11_accum_dm.cpp` in current pipeline. Kept for reference. |
| `v11_reduce_bc_dm.cpp` | BRISC | (Phase 3 legacy split) Primary reads K-1 shard pages back, sums them into its prim_slot, then runs scale + write. Inlined into `v11_accum_dm.cpp` in current pipeline. Kept for reference. |

## L1 layout (V11 accum, on each core)

Each accumulating core's L1 (CB_SCRATCH = c_24) holds:
```
[owned_lookup ][ hdrs_b ][ hdrs_n ][ buf_b ][ buf_n ][128 B gap][ dense_b ][ dense_n ][ tmp_shard ]
   M*N*2 B      nc_all*64  nc_all*64  16*max  16*max              n_total*    n_total*    1024
                bytes      bytes      *8 each *8 each              1024*4      1024*4      *4
```
For grid 2048 with `n_total ≤ 19` slots × 4 KB ≈ 76 KB dense, this fits in 1.5 MB. The mirrored layout calculation lives in `host/density_scatter_ttnn_server_host.cpp:1052` (`v11_dense_offset_bytes`).

## Tuple format

`V11Contrib` is **8 bytes** (`uint16 bx, uint16 by, float area`). Each NOC write is rounded to a multiple of 4 tuples (= 32 bytes) for Blackhole 32-byte alignment; the padding tuples carry `area = 0` and are skipped by the accumulator's `if (a == 0.0f) continue;` check.

## Per-page header layout (route_buf)

Header is 64 bytes per (writer, receiver) page:
- Bytes  0..31 = NCRISC scatter header (`cnt_n` at byte 0, rest zero — set by NCRISC scatter at `V11-HDR-WRITE`)
- Bytes 32..63 = BRISC scatter header (`cnt_b` at byte 32, rest zero — set by BRISC scatter at `V11B-HDR-WRITE`)

The accumulator reads both `cnt_n` (`hdr[0]`) and `cnt_b` (`hdr[8]`) to know how many tuples to iterate from each half.

## Cap (`v11_max_per_page_tuples`)

Per-page tuple budget split half / half between NCRISC scatter (first half) and BRISC scatter (second half). Currently **4096** (= 2048 per RISC). Beyond this, tuples are silently dropped — this is the dominant correctness issue for dense placements; see `docs/KNOWN_ISSUES.md` #1.

## Where to start when modifying

- **Front-end (SFPU geometry)**: `v4_compute.cpp`. Be careful with the bin-edge correction (`correct_bin_idx`), the SFPU floor has 1-ULP error.
- **Scatter (tuple routing)**: `v11_scatter_dm.cpp` / `v11_scatter_b_dm.cpp`. Inner loop walks `8×8` overlap grid per cell. Sort+dedup `flush_recv` is hot — keep cost bounded.
- **Accumulator (tuple → dense)**: `v11_accum_dm.cpp` `V11A-ACC` zone. The `dense[local * 1024 + bxw*32 + byw] += a;` inner loop is currently the slowest zone (~21 ms on the slowest core). See `docs/V11_PHASE3_HANDOFF.md` Steps 1 (sharding refresh), 3 (loop micro-optimization), 4 (BRISC+NCRISC parallel — already done).

## Build / dispatch

The server reads kernel source files directly at startup (no link step needed). Modify a kernel, restart the server (no rebuild of the server itself unless the host-side `SetRuntimeArgs` list changes). Kernel source path is baked into the server binary via the `DENSITY_KERNEL_DIR` preprocessor define at CMake time (`host/CMakeLists.txt`).
