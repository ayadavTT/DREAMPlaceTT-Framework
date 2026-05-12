# Reproducing DREAMPlace-TT Results

Step-by-step guide to build, run, and reproduce the results in [results/README.md](results/README.md).

## Hardware / OS requirements

- Tenstorrent Blackhole card (we tested on a "BH-38" SKU; 110 Tensix cores; 1.5 MB L1 per core; 4 GB+ DRAM)
- Linux host (Ubuntu 20.04+)
- Docker with the TT-Metal SDK image
- ≥ 32 GB RAM, ≥ 50 GB free disk for builds + benchmarks

## 1. Get a Docker container with the TT-Metal SDK

Your TT setup procedure may differ; in our environment the IT team provisions a per-user container, e.g. `bh-38-special-<user>-for-reservation-<id>`. Verify with:

```bash
docker ps --format "{{.Names}}" | grep bh-38
```

The container has `tt-metal/` bind-mounted (typically at `/localdev/.../tt-metal`) and `tt-smi` available.

> If you don't have a container yet: see the TT-Metal docs at https://github.com/tenstorrent/tt-metal for setup. The framework requires `_ttnncpp.so` and `libtt_stl.so` from a Release build.

Export the container name once:
```bash
export CONTAINER=bh-38-special-<user>-for-reservation-<id>
```

## 2. Clone (or symlink) external trees

The framework expects two large external trees at the top level:

```
DREAMPlaceTT-Framework/
├── tt-metal/      # TT-Metal SDK (~10 GB)
└── DREAMPlace/    # DREAMPlace placer (~5 GB) — incl. its dp_env Python venv
```

If you got this framework from GitHub:
```bash
git submodule update --init --recursive
```

If you got it via the local TTPort workspace, the symlinks should already be set up:
```bash
ls -la tt-metal/ DREAMPlace/  # should print symlinks
```

Otherwise, manually:
```bash
ln -s /path/to/tt-metal tt-metal
ln -s /path/to/DREAMPlace DREAMPlace
```

## 3. Build TT-Metal (one-time, inside the container)

The Release build with TTNN C++ enabled is required:
```bash
docker exec $CONTAINER bash -lc \
  "cd /path/to/tt-metal && bash build_metal.sh --build-programming-examples"
```
This produces `tt-metal/build_Release/lib/_ttnncpp.so` and the CMake package files we need.

## 4. Build DREAMPlace (one-time, on the host)

```bash
cd DREAMPlace
# Follow upstream DREAMPlace build instructions to produce dp_env (Python venv)
# and install/dreamplace/Placer.py etc. Typically:
python3 -m venv dp_env
source dp_env/bin/activate
pip install -r requirements.txt
mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
make -j$(nproc) install
```

The framework only needs the `dp_env` Python interpreter and the installed `dreamplace` Python module (it monkey-patches `dreamplace.EvalMetrics` and `ElectricPotentialFunction.forward`).

## 5. Build the framework's TT server

Inside the Docker container:
```bash
docker exec $CONTAINER bash -lc \
  "cd /path/to/DREAMPlaceTT-Framework && bash scripts/build_server.sh"
```
Output: `host/build/density_scatter_ttnn_server`.

## 6. Smoke test

A single-iter correctness check (synthetic 1.5M cells, grid 2048; passes if `rel_L2 < 1%`):

```bash
bash scripts/run_smoke.sh
```
Expected output (last few lines):
```
[v11_smoke] RESULT: rel_L2 = 3.177e-03  max_abs = 1.000e+00  ref_max = 8.924e+00
[v11_smoke] PASS (rel_L2 < 1%)
```

If you get an IOMMU sysmem error (`Expected NOC address: 0x..., but got 0x...`), reset the chip:
```bash
bash scripts/reset_chip.sh
```

## 7. Reproduce the sweep

All 4 benchmarks (~10–15 minutes total):
```bash
bash scripts/run_sweep.sh
```
Single benchmark:
```bash
bash scripts/run_sweep.sh sweep_adaptec1_2048
```

Results land in `results/sweep_latest/` (or `$RESULTS_DIR`):
- `<bench>_scatter_ttnn_metrics.json` — final HPWL, overflow, timing
- `<bench>_scatter_ttnn_iters.csv` — per-iter trace
- `<bench>_scatter_ttnn_evalmetrics.csv` — DREAMPlace eval metrics

## 8. Compare against CPU baseline

Run CPU separately with the same benchmark JSON (uses DREAMPlace's native CPU path):
```bash
$DREAMPLACE_PYTHON integration/run_dreamplace.py \
    --device cpu \
    --benchmark benchmarks/configs/sweep_adaptec1_2048.json \
    --results-dir results/cpu_baseline
```

## 9. Profile (optional)

To trace per-zone timings on TT cores using the TT-Metal device profiler:

```bash
TT_METAL_DEVICE_PROFILER=1 bash scripts/run_sweep.sh sweep_adaptec1_2048
docker cp $CONTAINER:$TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv /tmp/
python tools/profile_v11.py /tmp/profile_log_device.csv
python tools/profile_v11.py /tmp/profile_log_device.csv --zone V11A-ACC --per-core
```

Full details, zone-name index, and load-imbalance debugging recipes in
[docs/PROFILING.md](docs/PROFILING.md).

## Expected results

See [results/README.md](results/README.md) for the full numeric table.

The cycle to remember:
- **smoke test** = pass/fail correctness check, takes seconds.
- **single benchmark** = ~1–4 minutes wall, depending on grid+cell count.
- **full sweep** = ~10–15 minutes.

## Common pitfalls

- **"Timeout waiting for ready.flag"** — server crashed at startup. Usually the IOMMU sysmem error after a container restart. Run `scripts/reset_chip.sh`.
- **`n_ep_calls=0` in metrics** — server died mid-run. Check `docker exec $CONTAINER dmesg` and re-run with `TT_METAL_LOGGER_LEVEL=DEBUG`.
- **`FATAL: V11 accum scratch ... exceeds L1 budget`** — you raised `v11_max_per_page_tuples` too high. Keep it ≤ 4096 with `SRC_CHUNK=8`, or rebalance L1 by lowering `V11_CB_SLOT_HEADROOM`.
- **`/home` full (Errno 28)** — DREAMPlace writes to `~/.cache`. The `/home` is often a small quota volume; redirect cache to a bigger disk.

For more pitfalls + workarounds see [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md).
