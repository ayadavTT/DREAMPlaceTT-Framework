# Density Map Scatter+Gather: Acceleration Approaches

Algorithmic and hardware-level options for speeding up density map construction
(scatter + gather) on Tenstorrent Blackhole. Goals: reduce per-EP-call gather time,
keep work partitioning cell-centric and non-stationary-safe, avoid bin-centric
schemes that degrade when hot spots move.

---

## 0. Diagnosis: What Is Slow and Why

From measured benchmarks (2026-05-04, nc_max = nc_actual):

| Phase | adaptec1 1024×1024 | bigblue2 1024×1024 |
|-------|-------------------|-------------------|
| Scatter | ~5.9 ms | ~18.8 ms |
| Gather (V6, steady-state) | ~33 ms | ~78 ms |
| DCT (TTNN) | ~0.9 ms | ~0.9 ms |

**Scatter** is already fast and cell-centric. It tiles cells into groups of 1024,
accumulates contributions in L1 private buffers, and scales linearly with `nc_total`.
No urgent problem here.

**Gather dominates at 50–80% of total EP time.** The stated bottleneck
(`BENCHMARKS.md`) is: _"110 reads per gather core, each requiring a round-trip
barrier."_ This is the sequential-barrier anti-pattern:

```cpp
// Current (suspected) pattern — one barrier per read:
for i in 0..N_reads:
    noc_async_read(addr[i], l1_dst[i], 4)
    noc_async_read_barrier()   // waits for THIS one read only
    use(l1_dst[i])             // then proceeds
```

Each `noc_async_read_barrier()` call pays one full NOC round-trip latency to DRAM
(~200–500 ns). 110 sequential round-trips = 22–55 μs of stall time per gather-core
invocation, accumulated across many rounds per EP call. The hardware supports many
outstanding NOC transactions simultaneously; the software is serializing them.

---

## 1. Fix V6: Async Batch Reads

**Change the barrier placement, not the algorithm.** When all bin addresses for a
batch of cells are known before any read is issued (they always are, since cell
positions are in L1), batch all reads and issue one barrier for the entire batch:

```cpp
// Proposed — one barrier for the entire batch:
for each cell in batch:
    noc_async_read(field_addr(cell.bx, cell.by), l1_buf[i], 4)

noc_async_read_barrier()   // covers ALL N in-flight reads

for each cell in batch:
    use(l1_buf[i])
```

The NOC can hold many outstanding requests simultaneously. One barrier at the end
drains them all. Buffer requirement: 110 reads × 4 bytes = 440 bytes of L1 — trivial.

### Expected impact

| Pattern | Effective time for 110 reads |
|---------|------------------------------|
| Sequential barriers | 110 × 500 ns = 55 μs |
| Batch barrier | ~1.1 μs (issue) + ~0.5 μs (drain) = 1.6 μs |

If V6 is genuinely doing per-read barriers, this single change could reduce gather
from 33–78 ms to a few ms with no algorithmic change and no data-layout change.

### Tradeoffs

- Correctness: identical to current.
- Requires all target addresses computed upfront in the batch — already true.
- No change to data layout or kernel structure.

---

## 2. V8: Sort-Then-Tile-Prefetch

If batched reads are not enough — or if the bottleneck shifts to DRAM bandwidth
rather than latency — the architectural fix is to **eliminate random DRAM access
during gather entirely** by sorting cells into spatial tiles and DMA-prefetching
each tile's field data as a bulk sequential read.

### Data layout

```
Grid:     M×N bins       (e.g., 1024×1024)
Tile:     T×T bins       (e.g., 16×16 = 256 bins per tile)
Tiles:    (M/T)×(N/T)    (e.g., 64×64 = 4096 tiles for 1024×1024)

Per-tile field data:
  field_x:  T×T × 4 bytes = 1 KB
  field_y:  T×T × 4 bytes = 1 KB
  Total:    2 KB per tile
```

2 KB fits trivially in L1. Even 32×32 tiles (4 KB/tile) do.

### Per-iteration algorithm

**Phase A — Tile assignment** (O(nc), parallel across cores):

```cpp
for each cell i:
    tile_x[i] = (int)(px[i] / (T * bin_w));
    tile_y[i] = (int)(py[i] / (T * bin_h));
    key[i]    = tile_y[i] * (M/T) + tile_x[i];   // flat tile index
```

**Phase B — Sort by tile** (O(nc), counting/radix sort):

- Each core handles `nc / n_cores` cells and builds a local histogram over tile indices.
- One round of NOC-based prefix-sum gives global tile offsets.
- Each core scatter-writes its cells to the globally sorted array.

For `B = 4096` tiles and `nc ≤ 1.5M`, a single-pass counting sort suffices. The
prefix-sum requires one round of NOC exchange across cores.

**Phase C — Tile-parallel gather** (O(nc), embarrassingly parallel across tiles):

```cpp
for each tile T assigned to this core:
    // One bulk DMA for both field channels — sequential, not random
    noc_async_read(field_x_base + tile_offset(T), l1_fx, T*T*4)
    noc_async_read(field_y_base + tile_offset(T), l1_fy, T*T*4)
    noc_async_read_barrier()

    for each cell c in tile T (contiguous after sort):
        local_bx = bx(c) % T;
        local_by = by(c) % T;
        fx[c] = bilinear(l1_fx, local_bx, local_by, frac_x(c));
        fy[c] = bilinear(l1_fy, local_bx, local_by, frac_y(c));
```

### Cell distribution

| Design | nc_total | Tiles (16×16, 1024×1024) | Avg cells/tile |
|--------|----------|--------------------------|----------------|
| adaptec1 | 370,971 | 4,096 | 91 |
| bigblue2 | 1,473,813 | 4,096 | 360 |

With 126 Tensix cores: ~33 tiles/core for bigblue2.

### NOC traffic analysis

| Phase | NOC pattern | Total bytes |
|-------|-------------|-------------|
| Phase B prefix-sum | log2(126) rounds × small exchange | < 1 MB |
| Phase C tile DMA | 4096 × 2 KB = sequential bulk reads | 8 MB |
| Per-cell during gather | Zero | 0 |

Compare to V6: `1.4M cells × 4 reads × barrier per read → serial NOC latency chains`.
V8 replaces every per-cell random-access NOC read with a bulk DMA, which the NCRISC
DMA engine sustains at near-peak DRAM bandwidth.

At 100 GB/s DRAM bandwidth: 8 MB / 100 GB/s = 80 μs upper bound. With pipeline
overlap (Phase C compute while NCRISC fetches next tile), effective gather time
should be 5–20 ms for bigblue2.

### Non-stationarity

Cells move every iteration → tile assignments recomputed each call. This is correct
behavior: the sort is O(nc) regardless of where cells land. No assumption about
"busy regions" anywhere. Hot spots moving just means different tiles get more cells —
the per-tile work scales naturally with occupancy.

---

## 3. Double-Buffer DMA + SFPU Pipeline

Once tiles are the unit of work (either from V8 or from a V6 improvement), overlap
DMA prefetch with SFPU compute using double buffering:

```
NCRISC (DMA engine)          BRISC / SFPU (compute)
────────────────────         ──────────────────────────
prefetch tile[0] → buf_A
wait (barrier)
                             process buf_A (tile[0])  →  prefetch tile[1] → buf_B
                             wait on buf_B flag
                             process buf_B (tile[1])  →  prefetch tile[2] → buf_A
                             ...
```

NCRISC and BRISC are independent processors sharing L1 via address-based partitioning.
Coordination is a flag word in L1 (a single-cycle load, not a NOC operation):

```cpp
// NCRISC side:
noc_async_read(tile_addr, ping_buf, 2048);
noc_async_read_barrier();
l1_ready_flag[ping] = 1;                // signal BRISC via L1

// BRISC side:
while (!l1_ready_flag[ping]) {}         // spin on L1, not NOC
process(ping_buf);
l1_ready_flag[ping] = 0;
```

This hides DMA latency completely behind compute when tiles are large enough that
compute time ≥ DMA time. Effective gather time approaches `max(DMA_time, compute_time)`
instead of `DMA_time + compute_time`.

---

## 4. SFPU-Vectorized Force Accumulation

Once field data is in L1 (from tile prefetch or strip prefetch), the per-cell
force computation is:

```
fx = (1-wx)(1-wy)*fx00 + wx(1-wy)*fx10 + (1-wx)wy*fx01 + wx*wy*fx11
fy = (1-wx)(1-wy)*fy00 + wx(1-wy)*fy10 + (1-wx)wy*fy01 + wx*wy*fy11
```

SFPU operates on 16-wide float32 datums. With SoA cell layout — `px[16], py[16],
frac_x[16], frac_y[16]` — 16 cells can be processed in one SFPU pass:

- `(1 - frac_x)` and `frac_x` computed as vector subtract
- 4 L1 lookups × 16 cells = 64 L1 reads per SFPU pass (fast: 1–2 cycle L1 latency)
- 8 multiplies + 6 adds per output element, 2 outputs per cell → ~14 FLOP/cell

At 16-wide SFPU, arithmetic intensity is ~0.875 FLOP/cycle/lane. The bottleneck
remains DMA (memory-bound), not arithmetic. SFPU vectorization provides a modest
1.2–1.5× improvement by reducing loop overhead and enabling pipelined multiply-adds.

The SoA layout required here is the same layout scatter already produces for the
sorted cell arrays in V8.

---

## 5. Approximate Coarse-Grid Gather

**Observation**: The electric field is the output of a DCT-based Poisson solver
applied to a smoothed density map. It is spatially smooth by construction —
sub-bin-scale variation is suppressed. The field does not have meaningful information
at the per-bin level beyond what a coarser representation provides.

**Proposal**: After DCT, apply an average-pool with factor K before gather:

| Grid | Downsample K | Coarse size | Field bytes (both channels) | Fits in single-core L1? |
|------|-------------|-------------|----------------------------|-------------------------|
| 512×512 | 8 | 64×64 | 32 KB | Yes |
| 1024×1024 | 8 | 128×128 | 128 KB | Yes |
| 2048×2048 | 16 | 128×128 | 128 KB | Yes |

With the entire coarse field in L1:
- Each core loads its 128 KB field once at EP-call start (one DMA, ~1 μs)
- Gather is then pure L1 lookups for all nc cells — zero DRAM traffic during gather
- No NOC reads during force computation at all

### Accuracy tradeoff

Bilinear interpolation on the coarse field introduces error proportional to
`K × bin_size × field_curvature`. During global placement (iterations 0–70 of 100),
the field is smooth and convergence is dominated by gradient direction, not magnitude
precision. A 8× downsampling is unlikely to affect HPWL outcome.

During late-stage global placement and legalization, switch back to the fine-grained
field. This can be triggered by overflow threshold (e.g., switch at overflow < 0.3).

### Implementation cost

Low: one average-pool kernel applied to DCT output before writing field files. No
change to scatter, no change to kernel launch. The coarse field can live in a
dedicated L1 region broadcast to all gather cores.

---

## 6. Scatter: Tile-Privatized Reduction

Scatter is fast today (~5–19 ms). This section applies if it becomes a bottleneck
at larger grid sizes or cell counts.

**Current pattern**: Each core accumulates contributions for its cell range across
the full M×N density grid, writing a sparse partial map to DRAM. A separate reduce
kernel sums all partial maps. For 2048×2048: 16 MB partial map × 126 cores = 2 GB
of DRAM writes in the worst case.

**Tile-privatized alternative**: Each core is responsible for a spatial tile of the
density grid, not a range of cells. Cell-to-core assignment follows the same tile
sort as V8 gather. Each core:
1. Receives only cells that land in its tiles (from Phase B sort).
2. Accumulates contributions locally in L1.
3. Writes one tile of the density grid to DRAM (no cross-core reduction needed).

This eliminates the reduce phase entirely. DRAM writes are exactly `M×N×4 = 16 MB`
total for 2048×2048 — no redundancy.

**Coupling with V8 gather**: Scatter and gather share the same tile assignment from
Phase B. The sort is computed once per EP call and reused for both passes.

---

## 7. Summary: Tradeoff Matrix

| Approach | Gather speedup | Correctness | Non-stationarity | Implementation cost |
|----------|---------------|-------------|-----------------|---------------------|
| Async batch reads (V6 fix) | 5–20× | Exact | None required | Low — barrier placement only |
| V8 Sort+Tile-Prefetch | 3–10× | Exact | None required | Medium — sort kernel + tile DMA |
| Double-buffer DMA+compute | 1.5–2× additional | Exact | None required | Medium — NCRISC/BRISC split |
| SFPU 16-wide vectorization | 1.2–1.5× | Exact | None required | Low — SoA layout + intrinsics |
| Coarse-grid approximate gather | 10–50× | Approx (early iters) | None required | Low — average-pool at DCT output |
| Tile-privatized scatter | 1.5× scatter | Exact | None required | High — coupled with gather routing |

All approaches scale with `nc_total` (fixed per design), not with the density of
occupied bins. Cell distribution across the grid changing every iteration does not
affect correctness or load balance beyond the O(nc) sort cost.

---

## 8. Recommended Sequence

1. **Audit V6 for per-read barriers.** Inspect the gather kernel source and confirm
   whether `noc_async_read_barrier()` is called once per read or once per batch.
   If per-read, move the barrier to after the full batch. This is the highest
   ROI change: no algorithmic modification, no data-layout change, potentially
   10–20× gather reduction for free.

2. **Implement V8 Sort+Tile-Prefetch** if barrier batching is insufficient.
   The sort phase is O(nc) and parallelizable across all cores; the tile-DMA phase
   replaces all random NOC access with sequential bulk reads. Projected gather time:
   5–20 ms for bigblue2 at 1024×1024 (down from 78 ms).

3. **Add double-buffer DMA+compute** on top of V8. This hides per-tile DMA latency
   behind SFPU compute at no correctness cost.

4. **Enable coarse-grid gather** for global-placement iterations (overflow > 0.3)
   where field smoothness guarantees approximation validity. Switch to fine-grained
   field for final iterations. This is an independent option that stacks with any
   of the above.
