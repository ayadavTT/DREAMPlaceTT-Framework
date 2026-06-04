# Kernel testbench framework

A **robust, isolated validation harness for every kernel** in this history. Each
kernel gets its **own standalone test file** in its folder
(`<stage>/<kernel>/test/<kernel>_test.cpp`) — there is deliberately **no single
mega-harness** wiring kernels as `--mode` switches. The only shared code is this
`testbench/` library that the per-kernel tests `#include`.

## What every kernel test does
For each config in the sweep it:
1. **Generates a synthetic workload** via `synth.hpp` — cells with `(px,py,sx,sy)`
   in bin units (and a field map for backward-EF kernels), parameterized by:
   - **scale** = a real design's cell count (`design_profiles.hpp`): adaptec1/2/3, bigblue1/2/3,
   - **grid** ∈ {512, 1024, 2048},
   - **distribution** ∈ {uniform, cluster (one dense blob), clouds (several blobs)}.
2. **Runs the kernel in isolation** on the chip (its own TT-Metal program — reuse
   the kernel's prior host harness as the starting point).
3. **Checks correctness** vs the fp64 CPU reference (`synth::ref_density` for
   forward; a stage-appropriate reference for backward/optimizer) using
   `synth::rel_l2` (pass if `rel_l2 < tol`, default 1e-2 for bf16 paths).
4. **Measures performance** and (with the profiler on) emits the per-zone Tracy table.
5. **Writes the report into the kernel's `profile/` folder** (`report.hpp`):
   `results.csv` (one row/config) + `REPORT.md` (summary + per-zone Tracy + headline).

## Sweep matrix
6 designs × 3 grids × 3 distributions = **54 configs** per kernel (a `--quick`
subset of 1 design × 3 grids × 3 distributions = 9 for fast iteration).

## Shared library (this folder)
- `design_profiles.hpp` — real ISPD2005 cell counts, grid list, `Dist` enum.
- `synth.hpp` — `gen_workload()`, `overlap_of()`, `ref_density()`, `gen_field()`, `rel_l2()`.
- `report.hpp` — `Report`/`Row` → `results.csv` + summary.

## Per-kernel layout (the pattern every kernel follows)
```
<stage>/<kernel>/
├── KERNEL.md            # the doc
├── src/                 # kernel .cpp/.h (also the JIT source; DENSITY_KERNEL_DIR points here)
├── test/
│   ├── <kernel>_test.cpp   # the standalone harness (separate per kernel)
│   └── run.sh              # build (CMake target) + run sweep + profiler + process zones
└── profile/
    ├── results.csv         # correctness + perf, one row per config
    ├── REPORT.md           # human summary + headline schema + per-zone Tracy
    ├── profile_log_device.csv   # raw Tracy device dump (per representative config)
    └── zones.txt           # processed per-zone table (tools/profile_v11.py)
```

## Build / run
A single `history/CMakeLists.txt` adds **one target per kernel test** (separate
binaries from separate sources — not a mode switch). `run.sh` in each kernel
folder builds its target, runs the sweep, and (when `TT_METAL_DEVICE_PROFILER=1`)
copies + processes the device CSV into `profile/`.

```bash
# one kernel, quick sweep:
bash <stage>/<kernel>/test/run.sh --quick
# one kernel, full 54-config sweep + Tracy:
TT_METAL_DEVICE_PROFILER=1 bash <stage>/<kernel>/test/run.sh
```

## Why isolated (not via DREAMPlace)
Running each kernel standalone against synthetic workloads lets us measure its
true cost/correctness across problem sizes and distributions **without** the
DREAMPlace optimizer loop in the way — and the distribution axis (cluster/clouds
vs uniform) is exactly what stresses scatter atomic-contention and backward
load-balance, which is where these kernels differ.
```
