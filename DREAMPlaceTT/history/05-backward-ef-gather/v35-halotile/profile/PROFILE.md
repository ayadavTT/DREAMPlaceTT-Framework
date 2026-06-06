# V35 backward — per-zone profile interpretation (for the next-gen kernel author)

**Captured:** 2026-06-06, `v35live` harness on real bh silicon, `TT_METAL_DEVICE_PROFILER=1`
(clean `CloseDevice` dump → `profile_log_device.csv`). Raw per-zone table: [`zones.txt`](zones.txt).
Config: N=M=2048, 210904 cells, 130 cores, W=37, 56 tiles, max_k=6/max_h=3, uniform (cluster=0).
Harness stage wall-clock: **count 0.63 / plan 0.25 / place 1.33 / gather 1.92 ms** (TOTAL 4.13).

## How to read the numbers
The table is **per-instance medians** (one core, one invocation). The two passes (`V35-COUNT`,
`V35-PLACE`) and the gather's band loads run **once per core per iter**; the inner zones
(`*-GATHER`, `V35-PREP`) run `max_k·max_h·n_batches` times per iter, so their *per-iter sum*
is what matters, not the single-instance median.

## Where the backward actually spends time (strengths / weaknesses)

| zone | kernel | per-instance | what it is | verdict |
|---|---|---|---|---|
| **V35-PLACE** | v35_place | **863 µs** | PASS-2: scatter each cell into the one tile-grouped buffer | **heaviest single pass** — the grouping, not the math |
| **V35-LOADBAND** | v35_gather_brisc | **403 µs** | NoC read of the fx field band (page-by-page) into L1 | **2nd heaviest** — the field DMA, once per worker/iter |
| **V35-COUNT** | v35_count | **379 µs** | PASS-1: per-tile cell counts for the plan | heavy for a counting pass |
| V35-PREP | v35_gather_brisc | 135 µs ×batches | build px/ratio/base_idx per batch + drain oidx | moderate, per-batch |
| V28N-LOADBAND | v28_ncrisc | 145 µs | NoC read of the fy field band | parallel to fx loadband |
| V35-LOADCHUNK | v35_gather_brisc | 59 µs | stream the cell records for the slice | small |
| **V35-GATHER / V28N-GATHER** | gather/v28_ncrisc | **22 µs** ×(k·h·batch) | the actual SFPU `Σ px·py·f` per (k,h) | **fast — the compute is NOT the bottleneck** |
| V28N-DRAIN | v28_ncrisc | 2.5 µs | write gx/gy tile to DRAM | negligible |

**Takeaways for the next generation:**
1. **The compute is already cheap.** `*-GATHER` (the SFPU multiply-accumulate, the thing V35
   exists to do fast) is ~22 µs/instance — do **not** spend effort micro-optimizing the SFPU sum.
2. **The grouping dominates.** `V35-PLACE` (863 µs) + `V35-COUNT` (379 µs) = the count→plan→place
   machinery that makes the gather clustering-immune. This is the biggest on-device backward
   target — a cheaper grouping (e.g. fuse count+place, or a routing scheme that avoids the
   full PASS-2 scatter) would move the needle most.
3. **The field band load is the 2nd target.** `V35-LOADBAND` (403 µs fx + 145 µs fy). It's
   page-by-page from the page-interleaved field (required for correctness — a contiguous read
   gave NaNs). A future agent could explore a field layout / DMA pattern that lets one larger
   coalesced read replace the per-page reads, or overlap the band load with compute.
4. **Remember the host residual.** This profile is on-device only. The dominant *backward*
   cost in production is still the **host `d2h` grad-unsort** (`grad[sel]`, up to ~44 ms on
   bigblue3_2048 — see `../../../PIPELINE.md` §4), which the on-chip optimizer fusion (v22)
   would remove. On-device, place+loadband are the levers; end-to-end, d2h is.

## Reproduce / extend
```
cd DREAMPlaceTT/history/harness && bash run_harnesses.sh build       # builds v35live etc.
TT_METAL_DEVICE_PROFILER=1 ./build/v35live                            # clean close → CSV dump
python3 ../../../harness/parse_zones.py <TT_METAL_HOME>/generated/profiler/.logs/profile_log_device.csv
```
(Profiling is ~per-zone overhead-light on the harness; do NOT use the production run for this —
the in-process engine doesn't `CloseDevice`, so the CSV never flushes. Forward kernels lack a
standalone harness; their per-*stage* timing is in `../../../PIPELINE.md` §4.)
