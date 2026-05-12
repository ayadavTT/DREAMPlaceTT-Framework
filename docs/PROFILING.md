# Profiling DREAMPlace-TT with TT-Metal Device Profiler (TT-Tracy)

The V11 kernels are instrumented with `DeviceZoneScopedN(...)` markers around every meaningful phase. Enabling the TT-Metal device profiler captures per-core start/end cycle counts for every zone, on every iteration. You can then:

- Identify the **slowest core** in each phase (load imbalance — the dominant V11 issue).
- See which kernel phase actually consumes the time (scatter vs accum vs DCT).
- Compare two runs side-by-side to confirm an optimization moved the needle.

## TL;DR — one-time profile

```bash
# Inside the container (the profiler writes to a path inside the container's FS).
TT_METAL_DEVICE_PROFILER=1 bash scripts/run_sweep.sh sweep_adaptec1_2048

# CSV ends up here, by default:
docker exec $CONTAINER ls -lh /localdev/$USER/tt-work/TTPort/tt-metal/generated/profiler/.logs/

# Aggregate per-zone (max-across-cores, which is what bounds the iteration).
docker cp $CONTAINER:/localdev/$USER/tt-work/TTPort/tt-metal/generated/profiler/.logs/profile_log_device.csv /tmp/
python tools/profile_v11.py /tmp/profile_log_device.csv
```

## What the profiler captures

The TT-Metal device profiler (sometimes called "TT-Tracy") instruments every kernel that includes `tools/profiler/kernel_profiler.hpp` and uses one of:

- `DeviceZoneScopedN("MY-ZONE-NAME")` — a scoped RAII zone; records ZONE_START at construction, ZONE_END at destruction.
- The implicit `KERNEL` zone — start/end of the whole kernel main.

For each event the profiler writes a row to `.logs/profile_log_device.csv` containing: chip_id, core_x, core_y, RISC (BRISC/NCRISC/TRISC_0..2), cycle, zone_name, ZONE_START or ZONE_END. The first row of the CSV is a header carrying `CHIP_FREQ:<MHz>` for converting cycles → µs.

## Enable the profiler

There's already a hook in `integration/scatter_ttnn_client.py`:
```python
"TT_METAL_DEVICE_PROFILER": os.environ.get("TT_METAL_DEVICE_PROFILER", "0"),
```
So setting the env var on the host propagates into the container. From the framework root:
```bash
TT_METAL_DEVICE_PROFILER=1 bash scripts/run_sweep.sh sweep_adaptec1_2048
```
For a much shorter run, the smoke test already runs only 1 iter:
```bash
TT_METAL_DEVICE_PROFILER=1 bash scripts/run_smoke.sh
```

> **Build cost**: profiling instrumentation is compiled in by default. **No rebuild needed** when toggling `TT_METAL_DEVICE_PROFILER`. The runtime overhead is small (a few hundred cycles per zone).

## Where the CSV lands

The TT-Metal profiler writes to its own logs directory inside the container, **not** to the framework's `results/`:

```
$TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv
```

To pull it onto the host:
```bash
docker cp $CONTAINER:$TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv /tmp/
```

## DeviceZoneScopedN markers in the V11 pipeline

Every V11 kernel uses these zone names. Use them as a mental map:

| Kernel | RISC | Zone | What's measured |
|---|---|---|---|
| `v11_scatter_dm` | NCRISC | `V11-MAP-LOAD` | Loading `tile_to_core[]` and `shard_table[]` from DRAM (one-shot per launch). |
| `v11_scatter_dm` | NCRISC | `V11-CB-WAIT` | Waiting for TRISC SFPU output (`cb_wait_front`). Hot wait → bottleneck is compute. |
| `v11_scatter_dm` | NCRISC | `V11-ROUTE` | Walking cells [0..512), forming tuples, staging in L1. Per cell-tile. |
| `v11_scatter_dm` | NCRISC | `V11-FINAL-FLUSH` | Flushing remaining staging buffers to DRAM. |
| `v11_scatter_dm` | NCRISC | `V11-HDR-WRITE` | Writing 32-byte cnt_n headers per (writer, receiver) page. |
| `v11_scatter_b_dm` | BRISC | `V11B-DRAM-READ` | DRAM load of `px/py/sx/sy` cell-tile for this iter. |
| `v11_scatter_b_dm` | BRISC | `V11B-ROUTE` | Walks cells [512..1024) — twin of `V11-ROUTE`. |
| `v11_scatter_b_dm` | BRISC | `V11B-FINAL-FLUSH` | Same as NCRISC's, for BRISC's half. |
| `v11_scatter_b_dm` | BRISC | `V11B-HDR-WRITE` | cnt_b header bytes 32..63. |
| `v11_accum_dm` | BRISC | `V11A-LOOKUP-LOAD` | One-shot DRAM read of this core's `owned_lookup` page. |
| `v11_accum_dm` | BRISC | `V11A-ZERO` | Zero `dense[]` and `dense_n[]` slots in L1. |
| `v11_accum_dm` | BRISC | `V11A-HDR-READ-ALL` | Bulk NOC-read of all writer pages' headers. |
| `v11_accum_dm` | BRISC | `V11A-DATA-READ` | NOC-read of writer pages' tuple regions, in `SRC_CHUNK=8` batches. |
| `v11_accum_dm` | BRISC | **`V11A-ACC`** | **The hot inner loop** — `dense[local*1024 + bxw*32 + byw] += area`. Typically 50–80 % of gather time on the slowest core. |
| `v11_accum_dm` | BRISC | `V11A-MERGE` | After NCRISC twin signals, `dense[i] += dense_n[i]`. |
| `v11_accum_dm` | BRISC | `V11A-SHARD-WRITE` | (Phase A) Shard owners write to `shard_reduce_buf` and ping the primary. |
| `v11_accum_dm` | BRISC | `V11A-SHARD-SUM` | (Phase B) Hot primary waits for K-1 signals, sums K shard pages into prim_slot. |
| `v11_accum_dm` | BRISC | `V11A-SCALE` | Multiply primary dense slots by `inv_bin_area`. |
| `v11_accum_dm` | BRISC | `V11A-DENSITY-WRITE` | 32-byte NOC writes of primary tiles to `density_buf` DRAM. |
| `v11_accum_n_dm` | NCRISC | `V11N-LOOKUP-LOAD` / `V11N-ZERO` / `V11N-HDR-READ` / `V11N-DATA-READ` / **`V11N-ACC`** / `V11N-SIGNAL` | Twin of BRISC accum on the other half of writers. |

The two ACC zones (`V11A-ACC` and `V11N-ACC`) are the slowest. Optimizations should target those.

## Identifying load imbalance — the V11 problem signature

V11 cores all run the same accumulate kernel, but each core's tuple count varies because some tiles are hotter than others. The **slowest core** sets the iteration's wall time. To see this:

```bash
python tools/profile_v11.py /tmp/profile_log_device.csv --zone V11A-ACC
```
Output (example):
```
Zone V11A-ACC (BRISC):
  cores      : 110
  median (µs):  1846
  max    (µs): 21430     ← slowest core
  ratio max/median: 11.6×
```

When the `max/median` ratio is > 5×, you have load imbalance. The fix is either:
- Lower `HOT_THRESHOLD` so more tiles get K-way sharded.
- Enable periodic histogram refresh so the shard table tracks current hot tiles (currently disabled — see `docs/KNOWN_ISSUES.md` #2).

## TT-Tracy GUI (interactive timeline)

Tenstorrent ships a Tracy fork that opens the same CSV in a Tracy-style timeline viewer. To use it:

```bash
# Inside the container, the Tracy binary is usually at:
$TT_METAL_HOME/build_Release/tools/profiler/tt_tracy

# But on a Tracy GUI machine (with X11), run:
$TT_METAL_HOME/build_Release/tools/profiler/tt_tracy /tmp/profile_log_device.csv
```
Refer to the upstream TT-Metal docs for the GUI specifically:
https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/kernel_print.html (and the profiler docs section).

For most performance bug-hunts the per-zone CSV aggregator (`tools/profile_v11.py` below) is faster than firing up the GUI.

## Per-zone analysis script

See `tools/profile_v11.py` — aggregates the profiler CSV into a per-zone, per-RISC summary and flags the worst core in each zone. Sample usage:

```bash
# Default: all zones, sorted by max-core time descending
python tools/profile_v11.py /tmp/profile_log_device.csv

# Filter to one zone
python tools/profile_v11.py /tmp/profile_log_device.csv --zone V11A-ACC

# Per-core histogram for one zone (find the slowest cores)
python tools/profile_v11.py /tmp/profile_log_device.csv --zone V11A-ACC --per-core
```

## Caveats & pitfalls

- **Warmup iterations**: the first 3–4 EP calls are JIT-cache-cold or dispatch-cold; their per-zone times are 2–10× the steady-state. Drop them or filter to `iteration ≥ 4`.
- **Profiler buffer size**: large benchmarks (bigblue3_2048, 900+ iters) generate a few hundred MB of CSV. If you see truncated output, the per-core profiler ring buffer overflowed — restrict to a single iter via the smoke test for finer-grained zones.
- **Empty CSV**: `TT_METAL_DEVICE_PROFILER=1` must be set **before the server JIT-compiles**. If you set it after a `ready.flag` was already produced, kernels were built without the marker — kill the server and re-launch.
- **Cycle → time conversion**: read `CHIP_FREQ:1350MHz` (Blackhole) from the first CSV header line; older docs sometimes say 1000 MHz which is wrong for BH.

## What "looking at this profile" should answer

When debugging perf, check these in order:

1. **Smoke test passes?** → if no, it's correctness, not perf. Stop.
2. **Median V11A-ACC time / max V11A-ACC time** → if max ≫ median, load imbalance.
3. **`V11-CB-WAIT` time** → if high, SFPU compute or BRISC reader is the bottleneck.
4. **`V11A-DATA-READ` time** → if high, NOC bandwidth is the bottleneck; the 32-byte → 8-byte tuple packing was supposed to fix this.
5. **`V11A-MERGE` time vs `V11N-ACC` time** → if MERGE is high, BRISC is waiting on NCRISC ACC; rebalance the writer split.

See `docs/V11_PHASE3_HANDOFF.md` for the historical context on each of those zones.
