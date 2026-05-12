# Grid-Size Benchmark Experiments

End-to-end instructions for running DREAMPlace scatter/gather/DCT benchmarks across
grid sizes (512×512, 1024×1024, 2048×2048) on two ISPD 2005 designs, comparing
Tenstorrent Blackhole TT-Metal against a CPU baseline.

---

## Designs and Cell Counts

| Design   | Movable cells | Filler cells | nc_total  |
|----------|--------------|-------------|-----------|
| adaptec1 | 211,447      | 159,524     | 370,971   |
| bigblue2 | 534,782      | 939,031     | 1,473,813 |

`nc_total` is fixed for the lifetime of a PlaceDB run. The server allocates DRAM
buffers sized to `nc_total`; the scatter kernel iterates exactly `ceil(nc_total/1024)`
tiles, so getting this right matters for performance.

---

## Prerequisites

### 1. DREAMPlace Python environment

`run_dreamplace.py` self-activates `DREAMPlace/dp_env`. No manual `source activate`
needed. If the venv is missing, build DREAMPlace first:

```bash
cd /localdev/ayadav/tt-work/TTPort/DREAMPlace
mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
make -j$(nproc)
make install
```

### 2. Server binary (required for TT runs)

The C++ scatter+DCT server must be compiled inside Docker (needs TT-Metal headers):

```bash
docker exec <container> bash -c "
  cd /localdev/ayadav/tt-work/TTPort/experiments/density_scatter/tt_metal
  mkdir -p build_ttnn && cd build_ttnn
  cmake .. -DCMAKE_CXX_COMPILER=clang++-20 \
           -DTT_METAL_HOME=/localdev/aagarwal/tt-metal
  make density_scatter_ttnn_server -j$(nproc)
"
```

Binary lands at:
`experiments/density_scatter/tt_metal/build_ttnn/density_scatter_ttnn_server`

Rebuild any time `density_scatter_ttnn_server_host.cpp` or a kernel `.cpp` changes.

### 3. Running Docker container

A container with the Blackhole device mounted must be running. Find it:

```bash
docker ps --format '{{.Names}}' | grep -E 'bh-|blackhole|special'
```

`run_dreamplace.py` and `run_grid_benchmarks.sh` both auto-detect the first match.
Pass `--container <name>` to override.

### 4. Disk space / cache redirect

If `/home` is full (Errno 28), redirect caches before running:

```bash
export MPLCONFIGDIR=/localdev/ayadav/mpl_cache
mkdir -p $MPLCONFIGDIR
```

---

## Benchmark JSON Files

One JSON per design × grid, under `DREAMPlace/test/ispd2005/`:

```
adaptec1_512.json    adaptec1_1024.json    adaptec1_2048.json
bigblue2_512.json    bigblue2_1024.json    bigblue2_2048.json
```

Key fields common to all:

```json
{
    "aux_input": "benchmarks/ispd2005/<design>/<design>.aux",
    "gpu": 0,
    "num_bins_x": <512|1024|2048>,
    "num_bins_y": <512|1024|2048>,
    "global_place_stages": [{
        "num_bins_x": <512|1024|2048>,
        "num_bins_y": <512|1024|2048>,
        "iteration": 100,
        "learning_rate": 0.01,
        "wirelength": "weighted_average",
        "optimizer": "nesterov",
        "Llambda_density_weight_iteration": 1,
        "Lsub_iteration": 1
    }],
    "stop_overflow": 0.07,
    "dtype": "float32",
    "num_threads": 8,
    "deterministic_flag": 1,
    "random_seed": 1000
}
```

**100 iterations** is required for steady-state measurement. V6 sparse gather
starts at ~200ms on a cold run and converges to its steady-state value (30–80ms
depending on design+grid) over ~80–100 iterations as placement converges and
sparse contributions decrease.

---

## Running a Single Benchmark

### CPU baseline

```bash
cd /localdev/ayadav/tt-work/TTPort
python3 integration/run_dreamplace.py \
    --device cpu \
    --benchmark DREAMPlace/test/ispd2005/adaptec1_512.json \
    --results-dir DREAMPlace/install/results
```

### TT scatter+DCT (Path 1: V4 scatter + TTNN C++ DCT)

```bash
cd /localdev/ayadav/tt-work/TTPort
python3 integration/run_dreamplace.py \
    --device scatter_ttnn \
    --benchmark DREAMPlace/test/ispd2005/adaptec1_512.json \
    --results-dir DREAMPlace/install/results \
    [--container <name>]          # optional: auto-detected if omitted
    [--ipc-dir /tmp/ipc_bench]    # optional: default ipc_scatter_ttnn/
```

On first run, the server JIT-compiles TT-Metal kernels (~60s). Subsequent runs
use the compiled cache and start in a few seconds.

### Gather mode override

The server selects gather mode automatically:
- **V7 dense-strip** for 512×512 (fits in L1)
- **V6 sparse** for 1024×1024 and 2048×2048

To force a mode:

```bash
GATHER_MODE=v6 python3 integration/run_dreamplace.py --device scatter_ttnn ...
GATHER_MODE=v7 python3 integration/run_dreamplace.py --device scatter_ttnn ...
GATHER_MODE=auto python3 integration/run_dreamplace.py --device scatter_ttnn ...  # default
```

`GATHER_MODE` is propagated into the Docker container automatically.

---

## Running All Grid Benchmarks

`tools/run_grid_benchmarks.sh` runs CPU + TT for all 6 design×grid combinations:

```bash
cd /localdev/ayadav/tt-work/TTPort
./tools/run_grid_benchmarks.sh [--container <name>] [--skip-cpu] [--skip-tt]
```

Options:
- `--container <name>` — override auto-detection
- `--skip-cpu` — skip CPU baseline runs (useful if already collected)
- `--skip-tt` — skip TT runs

Logs are written per-run:
```
DREAMPlace/install/results/<design>_<grid>_cpu.log
DREAMPlace/install/results/<design>_<grid>_scatter_ttnn_auto.log
```

The script prints a summary table at the end. Full results are in
`DREAMPlace/install/results/grid_benchmark_summary.txt`.

---

## Output Files

Each run writes two files to `--results-dir`:

| File | Contents |
|------|----------|
| `<design>_<grid>_<device>_metrics.json` | Median/mean of all timing fields, HPWL, overflow, wall time |
| `<design>_<grid>_<device>_iters.csv` | Per-EP-call row: `iter, ep_ms, scatter_ms, gather_ms, ttnn_compute_ms, ...` |

### Key fields in metrics.json

| Field | Meaning |
|-------|---------|
| `ep_median_ms` | Median EP forward-pass wall time (host-side) |
| `scatter_ttnn_scatter_ms_median` | TT scatter kernel time (server-reported) |
| `scatter_ttnn_gather_ms_median` | TT gather kernel time (server-reported) |
| `scatter_ttnn_ttnn_compute_ms_median` | TTNN DCT compute time (6 matmuls) |
| `scatter_ttnn_h2d_ms_median` | Host→DRAM transfer (server reads pos.bin) |
| `scatter_ttnn_fw_ms_median` | Server writes field_x/y.bin |
| `scatter_ttnn_gather_mode` | `v6` or `v7` — which gather kernel ran |
| `cpu_scatter_ms_median` | CPU scatter time (when `--device cpu`) |
| `cpu_dct_ms_median` | CPU DCT time (when `--device cpu`) |

---

## Interpreting Results

### Steady-state vs. global median

For V6 sparse gather (1024×1024 and 2048×2048), gather time starts high (~200ms
at random init) and converges as cells cluster. The **global median** over 100
iterations includes the warm-up period and understates the true steady-state speed.

Use the **last-10-iteration median** from the CSV for a steady-state figure:

```python
import csv, numpy as np
rows = list(csv.DictReader(open("results/bigblue2_1024_scatter_ttnn_iters.csv")))
gather_ss = np.median([float(r['gather_ms']) for r in rows[-10:]])
scatter_ss = np.median([float(r['scatter_ms']) for r in rows[-10:]])
```

V7 dense-strip gather (512×512) is flat across all iterations — the global median
is already the steady-state value.

### Scatter time scales with nc_total

Scatter iterates `ceil(nc_total / 1024)` tiles per call. Tiles beyond valid cells
contain zero-size placeholders but are still processed. With `nc_max = nc_actual`
(the current setting), no wasted tiles are included.

| Design | nc_total | Tiles | Scatter (ss) |
|--------|----------|-------|-------------|
| adaptec1 | 370,971 | 363 | ~5.9ms |
| bigblue2 | 1,473,813 | 1,440 | ~18–19ms |

### DCT time scales with grid area

TTNN DCT runs 3 passes of 2 matmuls each (512 or 1024 or 2048 dimension):

| Grid | DCT time |
|------|---------|
| 512×512 | ~0.7ms |
| 1024×1024 | ~0.9ms |
| 2048×2048 | ~2.4ms |

### IPC overhead

`scatter_ttnn` uses POSIX mmap on an NFS-backed shared file (`ipc_shm/scatter.shm`).
Host Python and the containerized C++ server map the same file; because both run on the
same physical host they share the kernel page cache, making position writes and field reads
effectively zero-copy.

The earlier file-based IPC (`pos.bin` write + flag rename + `field_x/y.bin` read) added
7–30ms of syscall overhead per EP call. The mmap approach reduces this to <1ms:

| Metric | File-based IPC | mmap IPC | Speedup |
|--------|---------------|----------|---------|
| pos write (adaptec1_512) | 5.58ms | 0.45ms | 12.4× |
| field write (server-side) | 1.82ms | 0.18ms | 10.1× |
| field read (client-side) | 0.34ms | 0.34ms | 1.0× |
| total client IPC | ~7.7ms | ~0.97ms | 7.9× |

`field_read_ms` is unchanged because the client still calls `.copy()` on the mmap region
to transfer 2MB (512×512×2 float32) to a heap tensor before DREAMPlace's backward pass
can hold a reference to it.

---

## Benchmark Results (2026-05-04, nc_max = nc_actual)

Hardware: Tenstorrent Blackhole p150b, container `bh-38-special-snadkarni-for-reservation-73413`.
All times are **last-10-iteration medians** (steady-state).

| Design | Grid | scatter (TT) | gather (TT, ss) | DCT (TT) | EP (TT, ss) | EP (CPU) | TT/CPU |
|--------|------|-------------|----------------|---------|------------|---------|--------|
| adaptec1 | 512 | 5.9ms | 67.7ms (V7) | 0.6ms | 90.9ms | 8.9ms | 10.2× slower |
| adaptec1 | 1024 | 5.9ms | 33.5ms (V6) | 0.9ms | 69.0ms | 16.1ms | 4.3× slower |
| adaptec1 | 2048 | 10.1ms | 47.1ms (V6) | 2.3ms | 139ms | 57.7ms | 2.4× slower |
| bigblue2 | 512 | 18.0ms | 75.3ms (V7) | 0.8ms | 145ms | 39.6ms | 3.7× slower |
| bigblue2 | 1024 | 18.8ms | 77.7ms (V6) | 0.9ms | 160ms | 36.5ms | 4.4× slower |
| bigblue2 | 2048 | 19.1ms | 63.9ms (V6) | 2.4ms | 199ms | 76.1ms | 2.6× slower |

**EP breakdown is: ~IPC overhead + scatter + gather + DCT.** Gather dominates at
50–80% of total EP time in all cases. The bottleneck is the sequential NOC-read+barrier
pattern in V6 gather (110 reads per gather core, each requiring a round-trip barrier).

### Phase-by-phase CPU vs TT comparison (2026-05-04, nc_max = nc_actual)

All times are **last-10-iteration medians** (steady-state).
CPU scatter on TT has no separate gather pass — density accumulation is folded into
scatter on CPU, so the CPU gather column does not exist.
TT DCT total = compute + upload + download.

| Design | Grid | nc_max | CPU scatter | TT scatter | TT gather | CPU DCT | TT DCT compute | TT DCT upload+download |
|--------|------|-------:|------------:|-----------:|----------:|--------:|---------------:|-----------------------:|
| adaptec1 | 512×512 | 370,971 | 5.81 ms | 5.86 ms | 67.71 ms (V7) | 2.35 ms | 0.64 ms | 1.43 ms |
| adaptec1 | 1024×1024 | 370,971 | 6.51 ms | 5.85 ms | 33.54 ms (V6) | 7.41 ms | 0.88 ms | 5.46 ms |
| adaptec1 | 2048×2048 | 370,971 | 21.25 ms | 10.12 ms | 47.10 ms (V6) | 32.82 ms | 2.34 ms | 24.30 ms |
| bigblue2 | 512×512 | 1,473,813 | 34.87 ms | 17.99 ms | 75.33 ms (V7) | 2.76 ms | 0.77 ms | 1.64 ms |
| bigblue2 | 1024×1024 | 1,473,813 | 27.11 ms | 18.79 ms | 77.71 ms (V6) | 8.89 ms | 0.87 ms | 5.18 ms |
| bigblue2 | 2048×2048 | 1,473,813 | 38.88 ms | 19.06 ms | 63.88 ms (V6) | 31.54 ms | 2.35 ms | 26.44 ms |

**Scatter**: TT is competitive or faster at all grid sizes. At 2048×2048, TT scatter
is ~2× faster than CPU for both designs (scatter scales with nc_total on TT, with
grid area on CPU).

**DCT compute**: TT wins decisively — 3–14× faster than CPU depending on grid size
(0.64 ms vs 2.35 ms at 512×512, 2.34 ms vs 32.82 ms at 2048×2048). However, TTNN
tensor upload+download cost grows faster than compute: at 2048×2048 the transfer
overhead (24–26 ms) is ~10× the compute time, making total TT DCT (26–29 ms) only
marginally better than CPU (31–33 ms) at that size.

**Gather**: No CPU equivalent — the entire gather time is TT overhead introduced by
the two-program scatter+gather split. Eliminating or reducing gather latency is the
primary optimization target.

### Effect of nc_max = nc_actual fix

Correcting `nc_max` from `nc_actual × 1.5` to `nc_actual` reduced tile count by 33%,
with the following scatter improvement:

| Design | scatter before | scatter after | Δ |
|--------|---------------|--------------|---|
| adaptec1 (all grids) | 7.3ms | 5.9ms | −19% |
| bigblue2 (all grids) | 22–26ms | 18–19ms | −16% to −30% |

Scatter time does not scale fully linearly with tile count due to fixed per-batch
overhead, giving less than the theoretical 33% reduction.

### Effect of mmap IPC (2026-05-04)

Replacing file-based IPC with POSIX mmap on an NFS-backed shared file eliminated ~7ms of
per-EP syscall overhead. Measured on adaptec1_short (512×512, nc_max = 370,971, V7 gather,
48 EP calls), container `bh-38-special-snadkarni-for-reservation-73590`:

| Metric | File-based IPC | mmap IPC | Δ |
|--------|---------------|----------|---|
| pos_write_ms_median | 5.58ms | 0.45ms | −92% |
| fw_ms_median (server field write) | 1.82ms | 0.18ms | −90% |
| field_read_ms_median | 0.34ms | 0.34ms | 0% |
| scatter_ms_median | 6.93ms | 5.99ms | −14% |
| gather_ms_median | 76.21ms | 66.14ms | −13% |
| ttnn_compute_ms_median | 0.71ms | 0.66ms | −7% |
| total_server_ms_median | 91.15ms | 75.57ms | −17% |
| total_client_ms_median | 100.97ms | 77.78ms | −23% |
| ep_median_ms (DREAMPlace) | 106.37ms | 81.81ms | −23% |

The scatter and gather improvements are secondary effects: fewer wasted nanoseconds
between the host and server allow the server to start the next iteration sooner, reducing
queuing latency.

---

## Troubleshooting

**`assert` / L1 overflow in server log:**
The scatter kernel's L1 contribution buffer is capped dynamically. If the cap fires:
```
[server] WARNING: max_contrib 172032 > L1 cap 83968; capping
```
this is expected for bigblue2 at all grid sizes. Cells with many bin overlaps
may have their later contributions dropped, which slightly degrades density accuracy
but does not crash.

**IOMMU address mismatch / NOC error:**
```
Expected NOC address: 0x1000000000000000, but got 0x1000000040000000
```
Reset the device:
```bash
docker exec <container> tt-smi --reset
```
Then rerun.

**`ready.flag` timeout (>300s):**
The worker waits up to 5 minutes for JIT compilation. If it times out, check the
server log (printed via `[scatter_ttnn_srv]` prefix) for compilation errors. Cold
JIT is expected to take ~60s; longer suggests a broken build or missing kernel files.

**`gather_mode: unknown` in metrics.json:**
This is a parsing bug in older client code — the `gather_mode=v6` string in
`done.flag` was being silently dropped. Fixed in `scatter_ttnn_client.py` (the
`ValueError` branch now stores strings instead of discarding them). Re-running
produces the correct `v6`/`v7` value.
