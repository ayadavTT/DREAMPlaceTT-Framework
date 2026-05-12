# DREAMPlace-TT Framework Architecture

This document describes how the framework's pieces fit together. For the per-iteration kernel pipeline (V11), see [docs/DREAMPLACE_TT_ARCHITECTURE.md](docs/DREAMPLACE_TT_ARCHITECTURE.md) and [docs/V11_PHASE3_HANDOFF.md](docs/V11_PHASE3_HANDOFF.md).

## Process model

```
                       Host process                           Docker container
                       ────────────                           ────────────────
  user runs DREAMPlace ─► run_dreamplace.py ─► dreamplace.Placer
                                                  │
                                                  ▼ (monkey-patched at import)
                                       ElectricPotentialFunction.forward
                                                  │
                                                  ▼
                                          ScatterTTNNClient
                                          (integration/scatter_ttnn_client.py)
                                                  │
                                                  │  POSIX shmem (px, py, sx, sy, density, fx, fy)
                                                  │  state byte: 0=idle, 1=GO, 2=QUIT
                                                  ▼
                                                  ├──── docker exec ────► scatter_ttnn_worker.py
                                                  │                        │
                                                  │                        ▼
                                                  │                density_scatter_ttnn_server
                                                  │                (host/density_scatter_ttnn_server_host.cpp)
                                                  │                        │
                                                  │                        ▼
                                                  │                  Tensix kernels (110 cores)
                                                  │                  ────────────────────────
                                                  │                  v4_compute (TRISC SFPU)
                                                  │                  v11_scatter_dm (NCRISC) ┐
                                                  │                  v11_scatter_b_dm (BRISC)┘
                                                  │                  v11_accum_dm (BRISC) ┐
                                                  │                  v11_accum_n_dm (NCRISC)
                                                  │                  TTNN matmuls (TRISC) — DCT solve
                                                  ◄────────────────── density + fx + fy
```

**Key design choices**
- **Out-of-process server**: one long-lived TT server per DREAMPlace run, so JIT compilation amortizes across iters.
- **POSIX shmem for state**: tiny, zero-copy host↔server.
- **Server JIT-compiles kernels** on startup from `kernels/` (path baked in at compile time via `DENSITY_KERNEL_DIR`).
- **DREAMPlace is unmodified**; we monkey-patch two import-time hooks (`EvalMetrics.evaluate`, `ElectricPotentialFunction.forward`) from `run_dreamplace.py`.

## Directory map

```
DREAMPlaceTT-Framework/
├── kernels/                         9 TT-Metal kernel sources (V11 + supporting V4)
├── host/
│   ├── density_scatter_ttnn_server_host.cpp   ~1700 LOC — the V11 pipeline server
│   ├── v11_tile_ownership.h          snake-fill tile→core mapping + hot-tile sharding
│   └── CMakeLists.txt                build recipe (clang-20 + TT-Metalium + TTNN)
├── integration/
│   ├── run_dreamplace.py             top-level Python launcher
│   ├── scatter_ttnn_client.py        host-side IPC client + DREAMPlace monkey-patch
│   ├── scatter_ttnn_worker.py        docker-side exec wrapper
│   └── process_tt_profile.py         post-process TT-Metal device-profiler CSVs
├── scripts/
│   ├── build_server.sh               cmake + make inside container
│   ├── run_smoke.sh                  1-iter correctness check
│   ├── run_sweep.sh                  end-to-end on all 4 benchmarks
│   └── reset_chip.sh                 tt-smi -r 1 (chip reset)
├── benchmarks/configs/               6 DREAMPlace JSON configs (sweep_*, adaptec1_*)
├── tools/
│   ├── v11_phase2_smoke.py           smoke harness (referenced by run_smoke.sh)
│   ├── v11_phase1_smoke.py           older phase 1 smoke
│   └── cpu_vs_tt_density.py          diagnostic: CPU reference vs TT density diff
├── results/                          baseline + post-fix sweep summaries
├── docs/                             design + status + handoff documents
├── tt-metal -> ../tt-metal/          (symlink) TT-Metal SDK
└── DREAMPlace -> ../DREAMPlace/      (symlink) DREAMPlace placer
```

## Build pipeline

1. **TT-Metal** (one-time, in container):
   `cd tt-metal && bash build_metal.sh --build-programming-examples`
   Produces `build_Release/lib/_ttnncpp.so` + cmake package.

2. **DREAMPlace** (one-time, on host):
   Standard DREAMPlace build → `dp_env/` venv + `install/dreamplace/`.

3. **Framework server** (every code change):
   `bash scripts/build_server.sh`
   Builds `host/build/density_scatter_ttnn_server` via cmake. Uses `TT_METAL_HOME` env var to find TT-Metal.

The server **JIT-compiles** kernels on startup (reads `kernels/*.cpp` directly), so kernel changes do NOT require a server rebuild. Kernel source path baked at build time as `DENSITY_KERNEL_DIR`.

## Runtime IPC

`scatter_ttnn_client.py` creates a POSIX shared-memory region:
- Header bytes 0..3 = state (`0` idle, `1` GO, `2` QUIT)
- 4..7 = nc (live cell count for this iter)
- 8..15 = nc_max (allocated capacity)
- 16..onwards = SoA arrays: `px[nc_max], py[nc_max], sx[nc_max], sy[nc_max], density[M*N], fx[M*N], fy[M*N]`

The server polls the state byte. On `GO` it:
1. Reads positions, runs scatter → accum → DCT → gather kernels.
2. Writes density (for debugging) + force field.
3. Sets state back to `0`.

A second file `ready.flag` is touched by the server once after JIT compile completes; the client `start()` blocks until that exists.

## What `monkey-patches` does DREAMPlace need?

`scatter_ttnn_client.patch_dreamplace()` runs once at import time:
1. Replaces `dreamplace.electric_potential.ElectricPotentialFunction.forward` to call `ScatterTTNNClient.run_one_iter()` instead of CPU/CUDA.
2. Patches `dreamplace.EvalMetrics.EvalMetrics.evaluate` (bare `EvalMetrics` import — DREAMPlace's own import path) so we can record per-iter HPWL/overflow into `<bench>_scatter_ttnn_evalmetrics.csv`.

Everything else in DREAMPlace is untouched.

## Adding a new kernel variant

1. Drop `kernels/v11_yourname.cpp`.
2. Modify `host/density_scatter_ttnn_server_host.cpp` to:
   - `CreateKernel(prog, KDIR + "v11_yourname.cpp", ...)`
   - Pass runtime args via `SetRuntimeArgs(...)`
3. Rebuild: `bash scripts/build_server.sh`.
4. Smoke test: `bash scripts/run_smoke.sh`.
5. End-to-end: `bash scripts/run_sweep.sh sweep_adaptec1_2048`.

If your kernel changes the L1 layout, mirror the layout calc in `host/density_scatter_ttnn_server_host.cpp`'s `v11_dense_offset_bytes` block.

## See also

- [kernels/README.md](kernels/README.md) — design notes per kernel
- [docs/V11_PHASE3_HANDOFF.md](docs/V11_PHASE3_HANDOFF.md) — outstanding perf work (V11A-ACC, sharding refresh, dense accumulator)
- [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) — bugs we hit and how we (or didn't) fix them
