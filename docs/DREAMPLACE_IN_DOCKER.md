# Running DREAMPlace Inside the TT-Metal Docker Container

This document covers how to run DREAMPlace's CPU and TT-accelerated modes from
within the Blackhole Docker container.  Use this when you want to benchmark
or profile a TT-Metal kernel from a single `docker exec` invocation rather
than from the host.

---

## Why run inside Docker?

The TT-Metal device is owned by the container process.  For the `scatter_ttnn`
and `tt` modes the server binary already runs inside Docker; only the DREAMPlace
optimizer (client) runs on the host.  Running both inside the same container:
- eliminates the cross-process IPC hop (pos.bin / flag files)
- simplifies profiling (one process tree, one address space)
- allows direct TT device access without network IPC

For the **CPU baseline**, host and container performance are essentially identical
(within ~2% across all iterations, <5% in late steady-state) — Docker alone adds
no measurable overhead.

---

## Prerequisites

### 1. Container must be running

```bash
docker ps | grep bh-38   # confirm container is up
docker start bh-38-special-ayadav-for-reservation-73489   # start if stopped
```

The container name follows the pattern `bh-38-special-<user>-for-reservation-<id>`.

### 2. /localdev is bind-mounted

The container mounts `/localdev` from the host, so all code, venvs, and
benchmarks at `/localdev/ayadav/tt-work/TTPort/` are already visible inside
the container — no copying needed.

---

## One-time setup: install Python 3.8 inside the container

The DREAMPlace virtual environment (`dp_env`) was built against Python 3.8.
Ubuntu 22.04 containers ship with Python 3.10 only.  Install Python 3.8 once
per fresh container image:

```bash
docker exec bh-38-special-ayadav-for-reservation-73489 bash -c "
  DEBIAN_FRONTEND=noninteractive
  add-apt-repository -y ppa:deadsnakes/ppa
  apt-get install -y python3.8 python3.8-venv python3.8-dev
"
```

> **Persistence note:** This installation lives in the container's overlay
> filesystem.  It survives `docker stop` / `docker start` cycles but is lost
> if the container is deleted and recreated from scratch with `docker run`.
> Re-run the block above for any new container.  Consider adding Python 3.8
> to the base image to avoid this step permanently.

Verify the install:

```bash
docker exec bh-38-special-ayadav-for-reservation-73489 \
  /localdev/ayadav/tt-work/TTPort/DREAMPlace/dp_env/bin/python3.8 -c "
    import sys, torch
    sys.path.insert(0, '/localdev/ayadav/tt-work/TTPort/DREAMPlace/install')
    import dreamplace.Placer
    print('Python:', sys.version[:6])
    print('torch:', torch.__version__)
    print('DREAMPlace: OK')
  "
```

Expected output:
```
Python: 3.8.20
torch: 2.4.1+cpu
DREAMPlace: OK
```

---

## Running DREAMPlace

All modes use the same `run_dreamplace.py` entry point.  The script
auto-detects the dp_env and re-execs under Python 3.8 if needed.

### CPU baseline (no TT device)

```bash
docker exec bh-38-special-ayadav-for-reservation-73489 bash -c "
  cd /localdev/ayadav/tt-work/TTPort/integration
  /localdev/ayadav/tt-work/TTPort/DREAMPlace/dp_env/bin/python3.8 run_dreamplace.py \
    --device cpu \
    --benchmark /localdev/ayadav/tt-work/TTPort/DREAMPlace/test/ispd2005/adaptec1_short.json \
    --results-dir /localdev/ayadav/tt-work/TTPort/results/my_run
"
```

Expected runtime: ~9–10 s wall-time, ep_median ~33 ms (identical to host).

### scatter_ttnn mode (V4 scatter + TTNN C++ DCT)

```bash
docker exec bh-38-special-ayadav-for-reservation-73489 bash -c "
  cd /localdev/ayadav/tt-work/TTPort/integration
  /localdev/ayadav/tt-work/TTPort/DREAMPlace/dp_env/bin/python3.8 run_dreamplace.py \
    --device scatter_ttnn \
    --benchmark /localdev/ayadav/tt-work/TTPort/DREAMPlace/test/ispd2005/adaptec1_short.json \
    --container bh-38-special-ayadav-for-reservation-73489 \
    --results-dir /localdev/ayadav/tt-work/TTPort/results/my_run
"
```

> The `--container` flag names the container that runs the C++ server binary.
> When running DREAMPlace itself inside the container, this is the **same**
> container name.

### Full TT pipeline (V4 scatter + V6–V9b DCT matmuls)

```bash
docker exec bh-38-special-ayadav-for-reservation-73489 bash -c "
  cd /localdev/ayadav/tt-work/TTPort/integration
  /localdev/ayadav/tt-work/TTPort/DREAMPlace/dp_env/bin/python3.8 run_dreamplace.py \
    --device tt \
    --benchmark /localdev/ayadav/tt-work/TTPort/DREAMPlace/test/ispd2005/adaptec1_short.json \
    --container bh-38-special-ayadav-for-reservation-73489 \
    --results-dir /localdev/ayadav/tt-work/TTPort/results/my_run
"
```

---

## Available benchmarks

| File | Grid | Cells | Iters | Wall (CPU) |
|------|------|-------|-------|------------|
| `DREAMPlace/test/ispd2005/adaptec1_short.json` | 512×512 | ~371k | 48 EP | ~10 s |
| `DREAMPlaceTT/benchmarks/ispd2005/adaptec1.json` | 512×512 | ~371k | full | ~60 s |

The `adaptec1_short` JSON runs only the global placement stage
(30 configured iterations, ~48 actual EP calls due to filler steps) and is the
standard benchmark used for all timing measurements in this project.

---

## Output files

`run_dreamplace.py` writes two files per run into `--results-dir`:

| File | Contents |
|------|----------|
| `<benchmark>_<device>_metrics.json` | wall time, ep_median, ep_mean, sub-op medians |
| `<benchmark>_<device>_iters.csv` | per-iteration ep_ms, scatter_ms, dct_ms |

For `scatter_ttnn` mode the metrics JSON also contains per-sub-op server
timings: `h2d_ms`, `scatter_ms`, `gather_ms`, `ttnn_compute_ms`, etc.

---

## Performance reference (adaptec1_short, 512×512, 48 EP calls)

Measured on Blackhole p150b, 110 Tensix, container
`bh-38-special-ayadav-for-reservation-73489`, 2026-05-04.

| Mode | EP median | Wall time | vs CPU |
|------|-----------|-----------|--------|
| **CPU (host)** | 33.0 ms | 9.7 s | 1.0× |
| **CPU (Docker, no TT device)** | 32.9 ms | 9.6 s | **0.997×** |
| scatter_ttnn (Docker, TT device open) | 90.0 ms | 21.2 s | 2.75× slower |
| full TT pipeline (Docker, TT device open) | 108.4 ms | ~40 s | 3.32× slower |

Key observations:
- **Docker adds no overhead** for pure-CPU DREAMPlace (within 2% noise floor).
- The slowdown seen in TT modes comes from the TT device being open, not from
  Docker itself.  TT-Metal's ~200 polling threads compete with DREAMPlace's
  OMP/MKL threads for CPU bandwidth.
- scatter_ttnn bottleneck: gather kernel (66 ms, 85% of server time).
- Full TT bottleneck: V6–V9b DCT matmuls (80 ms, 28× slower than CPU FFTW3 DCT).

---

## Troubleshooting

**`python3.8: command not found` inside container**
Python 3.8 was not installed or the container was recreated.  Re-run the
one-time setup block above.

**`No module named 'dreamplace'`**
The script must `cd` to the integration directory first — the re-exec logic
uses `os.path.dirname(__file__)` to locate `dp_env`.  Always run as:
```bash
cd /localdev/ayadav/tt-work/TTPort/integration
/localdev/ayadav/tt-work/TTPort/DREAMPlace/dp_env/bin/python3.8 run_dreamplace.py ...
```

**`Timeout waiting for scatter_ttnn server ready.flag`**
JIT compilation takes ~3–20 s depending on firmware cache state.  The
pre-compiled firmware cache lives at:
```
/localdev/ayadav/tt-work/TTPort/tt-metal/tt_metal/pre-compiled/
```
If the cache is cold (first run, or cache cleared), allow up to 60 s.

**Container stopped after a `density_scatter_full_pipeline` run**
The full pipeline binary calls `exit()` after completion, which can terminate
the container process tree.  Restart with `docker start <container>`.
