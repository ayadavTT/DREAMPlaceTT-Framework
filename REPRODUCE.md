# Reproducing DREAMPlace-TT Results

Step-by-step guide to build, run, and reproduce the results in [results/README.md](results/README.md).

## Hardware / OS requirements

- Tenstorrent Blackhole card (we tested on a "BH-38" SKU; 110 Tensix cores; 1.5 MB L1 per core; 4 GB+ DRAM)
- Linux host (Ubuntu 22.04+)
- Docker with the TT-Metal SDK image (root inside the container)
- ≥ 64 GB RAM, ≥ 50 GB free disk for builds + benchmarks

## 1. Get a Docker container with the TT-Metal SDK

Your TT setup procedure may differ; in our environment the IT team provisions a per-user container, e.g. `bh-38-special-ayadav-for-reservation-74318`. Verify with:

```bash
docker ps --format "{{.Names}}" | grep bh-
```

The container has `tt-metal/` bind-mounted (typically at `/localdev/.../tt-metal`) and `tt-smi` available.

> If you don't have a container yet: see the TT-Metal docs at https://github.com/tenstorrent/tt-metal for setup. The framework requires `_ttnncpp.so` and `libtt_stl.so` from a Release build.

Export the container name once:
```bash
export CONTAINER=bh-38-special-<user>-for-reservation-<id>
```

## 2. Clone the framework + init submodules

`tt-metal/` and `DREAMPlace/` are git submodules pinned to the commits we test against (see `.gitmodules`). Clone with `--recursive` or init after cloning:

```bash
git clone --recursive <this-repo-url> DREAMPlaceTT-Framework
cd DREAMPlaceTT-Framework

# Or, if you cloned without --recursive:
git submodule update --init --recursive
```

Pinned commits (for reference):
- `tt-metal`: `67c740484dd0c17d5d7c5125645f385da3a7629d` (v0.71.0-dev20260512-38)
- `DREAMPlace`: `37214b40fe3837cc7d392c7d6092ccd6ff04a02c` (4.3.0-20)

## 3. Install Python 3.8 inside the container

DREAMPlace's compiled extensions (`electric_potential_cpp.cpython-38-*.so`, `density_potential_cpp.cpython-38-*.so`, etc.) are built against Python 3.8 ABI. Ubuntu 22.04 (which ships in the TT-Metal container) defaults to Python 3.10, so install 3.8 via the deadsnakes PPA:

```bash
docker exec $CONTAINER bash -lc '
  apt-get install -y software-properties-common
  add-apt-repository -y ppa:deadsnakes/ppa
  apt-get update
  apt-get install -y python3.8 python3.8-dev python3.8-venv
'
# verify:
docker exec $CONTAINER /usr/bin/python3.8 --version  # → "Python 3.8.20"
```

> This step is required for `--device scatter_ttnn` (IPC server is C++-only but the Python orchestrator runs DREAMPlace) and for `--device scatter_ttnn_inprocess` (the pybind11 module itself is built against Python 3.8).

## 4. Build TT-Metal (one-time, inside the container)

The Release build with TTNN C++ enabled is required:
```bash
docker exec $CONTAINER bash -lc \
  "cd /path/to/DREAMPlaceTT-Framework/tt-metal && bash build_metal.sh --build-programming-examples"
```
This produces `tt-metal/build_Release/lib/_ttnncpp.so` and the CMake package files we need. Build time: 15–30 min on a fast machine.

The framework's `host/CMakeLists.txt` searches for `tt-metal/build_Release` first, then `tt-metal/build`.

## 5. Build DREAMPlace + apply the torch 2.x compat patch (one-time, on the host)

```bash
cd DREAMPlace

# Apply the torch 2.x compat shim (resolves at::ceil / std::ceil ambiguity).
# Without this, the build fails on `electric_density_map.cpp` and a few other
# files when compiled against torch 2.x + GCC 11+.
git apply ../patches/dreamplace-torch2x-compat.patch

# Create the Python 3.8 venv DREAMPlace will use.
/usr/bin/python3.8 -m venv dp_env
source dp_env/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt
# torch 2.x + pybind11 must match; the framework was tested with torch==2.4.1+cpu.

mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install -DPython_EXECUTABLE=$(which python)
make -j$(nproc) install
cd ../..
```

After this:
- `DREAMPlace/dp_env/bin/python3` is the Python 3.8 interpreter the framework calls into.
- `DREAMPlace/install/dreamplace/` has the patched DREAMPlace + the compiled `electric_potential_cpp.so` etc.

## 6. Get the ISPD2005 benchmark inputs

The JSON configs in `benchmarks/configs/` reference inputs in `benchmarks/ispd2005/<design>/<design>.aux`. These are the standard placement benchmarks (`.nodes`, `.nets`, `.pl`, `.scl`, `.wts`, `.aux`) and are **not** distributed with this repo.

```bash
mkdir -p benchmarks/ispd2005
cd benchmarks/ispd2005
# Download the ISPD2005 placement-contest benchmark suite. Several mirrors host it;
# upstream DREAMPlace's docs reference one. Example layout you need:
#   benchmarks/ispd2005/adaptec1/adaptec1.aux  (+ .nodes, .nets, .pl, .scl, .wts)
#   benchmarks/ispd2005/adaptec2/...
#   benchmarks/ispd2005/bigblue1/...
#   benchmarks/ispd2005/bigblue3/...
```

For just the smoke test, you only need `adaptec1`. To run the full 18-config sweep you need `adaptec1`, `adaptec2`, `adaptec3`, `bigblue1`, `bigblue2`, `bigblue3`.

## 7. Build the framework's TT binaries

Inside the Docker container:
```bash
docker exec $CONTAINER bash -lc \
  "cd /path/to/DREAMPlaceTT-Framework && bash scripts/build_server.sh"
```

Output:
- `host/build/density_scatter_ttnn_server` — IPC server binary (path used by `--device scatter_ttnn` and `_direct`)
- `host/build/v19_engine.cpython-38-*.so` — pybind11 module (path used by `--device scatter_ttnn_inprocess`)
- A handful of `v1*_*_smoke_host` microbench / smoke targets

The build script auto-detects `DREAMPlace/dp_env/lib/python3.8/site-packages/pybind11` and `DREAMPlace/dp_env/bin/python3`; if either is missing it skips the `v19_engine` target with a non-fatal warning. Re-run after building DREAMPlace if you forgot to set up `dp_env` first.

## 8. Smoke test (IPC server path)

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

## 9. Run DREAMPlace on TT — V19 in-process (production fast path)

The V19 pybind11 in-process engine is the current fastest path. It eliminates the per-iter IPC + JSON-over-shm overhead of `--device scatter_ttnn` by importing the TT engine directly into the DREAMPlace Python process.

```bash
export TT_METAL_HOME=$PWD/tt-metal
export TT_METAL_RUNTIME_ROOT=$PWD/tt-metal
export GATHER_MODE=v19         # required — selects V19's L1-atomic gather
export CPU_DCT=1               # required — keeps DREAMPlace's add-back path active

DREAMPlace/dp_env/bin/python3 integration/run_dreamplace.py \
    --device scatter_ttnn_inprocess \
    --benchmark benchmarks/configs/sweep_adaptec1_2048.json \
    --container $CONTAINER \
    --results-dir results/v19_inprocess
```

Why each env var matters:
- `TT_METAL_HOME` / `TT_METAL_RUNTIME_ROOT` — TT-Metal looks here for the kernel-build asset directory; mismatching layouts between the framework's `tt-metal/` and the outer SDK install break ABI (see memory note).
- `GATHER_MODE=v19` — without it, `host/v19_engine.cpp` defaults to V8/V9 auto-route, which references legacy kernel files no longer in the tree.
- `CPU_DCT=1` — without it, DREAMPlace's `initial_density_map` add-back is bypassed and HPWL diverges (see `docs/KNOWN_ISSUES.md`).

Other available devices on `integration/run_dreamplace.py`:
- `cpu` — DREAMPlace's native CPU path (16-thread OMP baseline).
- `scatter_ttnn` — legacy IPC path (`density_scatter_ttnn_server` via docker exec + /dev/shm).
- `scatter_ttnn_direct` — IPC path, direct binary launch (no docker exec).
- `scatter_ttnn_inprocess` — V19 pybind11 in-process (recommended).

## 10. Reproduce the sweep

All configs in `benchmarks/configs/sweep_*.json` (~15–25 minutes wall-clock total):
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

## 11. Compare against CPU baseline

Run CPU separately with the same benchmark JSON:
```bash
DREAMPlace/dp_env/bin/python3 integration/run_dreamplace.py \
    --device cpu \
    --benchmark benchmarks/configs/sweep_adaptec1_2048.json \
    --num-threads 16 \
    --results-dir results/cpu_baseline
```

## 12. Profile (optional)

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
- **full sweep** = ~15–25 minutes.

## Common pitfalls

- **"Timeout waiting for ready.flag"** — IPC server crashed at startup. Usually the IOMMU sysmem error after a container restart. Run `scripts/reset_chip.sh`.
- **"Waiting for lock 'CHIP_IN_USE_<N>_PCIe'"** — another user (or a stale process of yours) holds the chip lock. `ps -ef | grep tracy` to find it. Don't kill another user's process on a shared reservation.
- **`ImportError: libtorch_python.so: cannot open shared object file`** — happens when `electric_potential_cpp` is imported before `torch`. Always `import torch` first in any script that uses DREAMPlace's compiled extensions.
- **`FATAL: V11 accum scratch ... exceeds L1 budget`** — you raised `v11_max_per_page_tuples` too high. Keep it ≤ 4096 with `SRC_CHUNK=8`, or rebalance L1 by lowering `V11_CB_SLOT_HEADROOM`.
- **`/home` full (Errno 28)** — DREAMPlace writes to `~/.cache`. The `/home` is often a small quota volume; redirect cache to a bigger disk.
- **HPWL diverges to 155M on adaptec1_512** — you forgot `CPU_DCT=1`. The V11/V19 engines assume DREAMPlace's host-side add-back is enabled.

For more pitfalls + workarounds see [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md).
