# V19 — Rung 3: in-process pybind11 extension to eliminate IPC overhead

**Date**: 2026-05-22
**Status**: design + scope estimate; **not started**
**Predecessor work**: V19 block-partition + fp32-on-chip writeout is **shipped and converging on all 18 configs** (see `docs/V19_L1_ATOMIC_HANDOFF.md` and memory note `v19_block_fp32_outcome.md`).

This document is a complete handoff so the next agent can pick up the Rung 3 optimization. Read it end-to-end before touching code.

---

## 0. TL;DR

Today V19's chip kernel is fast — but every iteration still pays **2-12 ms of IPC overhead** going from DREAMPlace's Python process to the TT server process (`density_scatter_ttnn_server`, a separate C++ binary launched via `docker exec`). The IPC is shared-memory with `/dev/shm`, but it costs:

1. **`pos_write`** (0.5-4 ms/iter): Python `memcpy` of 4 position arrays into shm
2. **`field_read`** (0.3-4 ms/iter): Python `memcpy` of field result out of shm
3. **`ipc_overhead`** (1-7 ms/iter): docker-exec handshake, state-flag poll, `Finish(cq)` waits triggered cross-process

Across the 18-config sweep at ~700 iter/run, that's **80-150 seconds of cumulative wall time** spent on software-architecture overhead the chip never sees.

**Rung 3 = in-process integration**: refactor the C++ server into a pybind11 extension callable directly from DREAMPlace's Python. No shm, no docker exec, no IPC — `engine.scatter(px, py, sx, sy)` is a function call.

Expected wins per iter:
- Eliminate all of `pos_write` + `field_read` + `ipc_overhead` = the entire **IPC+Python bucket**
- Replace it with the unavoidable minimum: `h2d` (positions → TT DRAM via PCIe DMA, already counted) + `d2h` (density → CPU RAM, already counted)

---

## 1. Required reading (in order)

1. **`docs/V19_L1_ATOMIC_HANDOFF.md`** — original V19 atomic-add gather design
2. **Memory note**: `~/.claude/projects/.../memory/v19_block_fp32_outcome.md` — what V19 looks like today (block ownership + fp32-on-chip writeout)
3. **`host/density_scatter_ttnn_server_host.cpp`** (the file you're about to refactor) — currently the binary's `main(int argc, char** argv)`
4. **`integration/scatter_ttnn_client.py`** — Python side of today's IPC; will become a thin pybind11 wrapper
5. **`integration/run_dreamplace.py`** — entry point; understand how `--device scatter_ttnn` plugs into DREAMPlace's `ElectricPotentialFunction.forward`

---

## 2. Current architecture (what to refactor)

### 2.1 Today

```
                 host (Python)                          docker container (C++)
┌──────────────────────────────────────┐     ┌──────────────────────────────────────┐
│ DREAMPlace iteration N:              │     │ density_scatter_ttnn_server          │
│   ElectricPotentialFunction.forward  │     │  (a binary launched via docker exec) │
│     ├ scatter_ttnn_client.scatter()  │     │                                       │
│     │   ├ memcpy px→shm, py→shm,..   │ ←──→│ ┌── while(true) ────────────────┐   │
│     │   ├ shm_state = GO              │     │ │  read shm_state                │   │
│     │   ├ wait for shm_state=DONE     │     │ │  memcpy shm→px,py,sx,sy        │   │
│     │   ├ memcpy field→torch tensors  │     │ │  h2d, scatter, gather          │   │
│     │   └ return field                │     │ │  d2h, fw                       │   │
│     ├ DCT (CPU, CPU_DCT=1)           │     │ │  shm_state = DONE              │   │
│     ├ field calc                      │     │ └────────────────────────────────┘   │
│     └ return potential                │     │                                       │
└──────────────────────────────────────┘     └──────────────────────────────────────┘
```

### 2.2 Target

```
                                 host (Python)
┌────────────────────────────────────────────────────────────────────────────┐
│ DREAMPlace iteration N:                                                     │
│   ElectricPotentialFunction.forward                                         │
│     ├ density, fx, fy = v19_engine.scatter(px, py, sx, sy, nc_actual)       │
│     │     # ↑ pybind11 call into C++; positions pass by numpy buffer        │
│     │     # ↑ C++ DMAs positions to TT DRAM, runs kernels, DMAs density back│
│     │     # ↑ density/field returned as numpy arrays (zero-copy)            │
│     ├ DCT (CPU)                                                             │
│     ├ field calc                                                            │
│     └ return potential                                                      │
└────────────────────────────────────────────────────────────────────────────┘
```

The TT chip stays in the container. The DREAMPlace process either runs **inside** the container too (simplest), or links the extension `.so` from the container's filesystem (LD_LIBRARY_PATH gymnastics).

---

## 3. Refactor plan (concrete step-by-step)

### Step 1 — Extract `V19Engine` class from `density_scatter_ttnn_server_host.cpp` (3 hours)

Current `main()` is one giant function with two phases:
- **Setup phase** (lines 374-2615 today, ~2240 lines): mesh device open, ~20 DRAM `MeshBuffer` allocations, 3 `MeshWorkload` JITs (hist / scatter / writeout), CB layouts, runtime args, FATAL budget checks, etc.
- **Per-iter loop** (lines 2617-3333, ~715 lines): polls `shm_state`, copies positions in, runs programs, copies density out, repeats.

Refactor into:

```cpp
// host/v19_engine.h
class V19Engine {
public:
    V19Engine(int M, int N, int NC_max,
              float xl, float yl, float xh, float yh,
              const std::string& gather_mode = "v19");
    ~V19Engine();

    // Run one scatter iteration. Inputs: cell positions (host arrays of length nc_actual).
    // Outputs: density [M*N fp32], field_x [M*N fp32], field_y [M*N fp32]
    // Returned as raw pointers + length; caller (pybind11) wraps in numpy arrays.
    struct ScatterResult {
        const float* density;
        const float* field_x;
        const float* field_y;
        // per-iter timing breakdown
        double h2d_ms, scatter_ms, gather_ms, fw_ms, total_ms;
    };
    ScatterResult scatter(const float* px, const float* py,
                          const float* sx, const float* sy,
                          int32_t nc_actual);

private:
    // All the today-local variables become member fields:
    int M_, N_, NC_max_;
    std::shared_ptr<MeshDevice> mesh_device_;
    MeshCommandQueue cq_;
    std::shared_ptr<MeshBuffer> px_buf_, py_buf_, sx_buf_, sy_buf_;
    // ... ~20 MeshBuffer fields ...
    MeshWorkload wl_v11_scatter_, wl_v11_accum_, wl_v11_hist_, wl_v19_writeout_;
    // ... internal state: iter counter, hist refresh logic, etc. ...

    // Pre-allocated output buffers (reused across iters; safe because Python copies them out)
    std::vector<float> out_density_, out_fx_, out_fy_;
};
```

The constructor does **everything** today's `main()` does before the `while(true)` loop. The `scatter()` method does **one iteration** of the loop body.

Subtleties:
- The current loop has refresh-every-N-iters logic (`v11_should_refresh = (v11_iter % V11_HIST_REFRESH_ITERS == 0)`). Move iter counter into member state.
- All `static` variables in the loop body (e.g., `static const uint32_t SCALE_BITS = env_uint(...)`) need to be either constants in the class or class members read once in ctor.
- `Finish(cq)` is fine in the method — it blocks until chip work is done, which is what Python expects synchronously.

### Step 2 — pybind11 binding (1 hour)

```cpp
// host/v19_engine_pybind.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "v19_engine.h"
namespace py = pybind11;

PYBIND11_MODULE(v19_engine, m) {
    py::class_<V19Engine>(m, "V19Engine")
        .def(py::init<int, int, int, float, float, float, float, std::string>(),
             py::arg("M"), py::arg("N"), py::arg("NC_max"),
             py::arg("xl"), py::arg("yl"), py::arg("xh"), py::arg("yh"),
             py::arg("gather_mode") = "v19")
        .def("scatter", [](V19Engine& eng,
                            py::array_t<float, py::array::c_style> px,
                            py::array_t<float, py::array::c_style> py_arr,
                            py::array_t<float, py::array::c_style> sx,
                            py::array_t<float, py::array::c_style> sy,
                            int32_t nc_actual) {
            auto r = eng.scatter(px.data(), py_arr.data(), sx.data(), sy.data(), nc_actual);
            // Wrap result pointers in numpy arrays (zero-copy view; ENGINE owns the buffer)
            int total = ... // M*N
            py::capsule own_density(r.density, [](void*) {});  // we own; don't free
            auto den_arr = py::array_t<float>({total}, {sizeof(float)},
                                              r.density, own_density);
            // ditto field_x, field_y
            return py::make_tuple(den_arr, fx_arr, fy_arr,
                                  py::dict("h2d_ms"_a = r.h2d_ms,
                                           "scatter_ms"_a = r.scatter_ms,
                                           "gather_ms"_a = r.gather_ms,
                                           "fw_ms"_a = r.fw_ms,
                                           "total_ms"_a = r.total_ms));
        });
}
```

### Step 3 — CMake build target for the extension (1 hour)

```cmake
# host/CMakeLists.txt — append
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(v19_engine MODULE
    v19_engine_pybind.cpp
    # Plus all the existing density_scatter_ttnn_server_host.cpp dependencies,
    # but extracted into v19_engine.cpp (no main()).
)
target_link_libraries(v19_engine PUBLIC TT::Metalium)
target_link_libraries(v19_engine PRIVATE
    "${TT_BUILD_USED}/lib/_ttnncpp.so"
    "${TT_BUILD_USED}/lib/libtt_stl.so")
target_include_directories(v19_engine PRIVATE ${TT_TTNN_INCLUDES})
target_compile_definitions(v19_engine PRIVATE
    "DENSITY_KERNEL_DIR=\"${_FW_ROOT}/kernels/\"")
```

Note: pybind11 is shipped with PyTorch and standalone via pip. Use the standalone one (`pip install pybind11`).

### Step 4 — Build inside the container (1 hour)

The tt-metal libs live in the container. The simplest setup:
- `docker exec` into container interactively
- Build the extension there: `cmake --build build --target v19_engine -j 16`
- Resulting `.so` lives at `host/build/v19_engine*.so`

The extension links against `_ttnncpp.so` and `libtt_stl.so` which are in `<TT_METAL_HOME>/build_Release/lib`. The DREAMPlace process needs `LD_LIBRARY_PATH` to find them at import time.

### Step 5 — Run DREAMPlace inside the container (1-2 hours)

The simplest path: bind-mount the DREAMPlace install + repo into the container, and run `DREAMPlace/dp_env/bin/python3 integration/run_dreamplace.py ...` **from inside** the container. That way:
- The Python process has direct access to the extension `.so`
- `LD_LIBRARY_PATH` is set automatically by the container
- No docker exec needed at all

Steps:
1. The container already has the repo at `/localdev/ayadav/tt-work/TTPort/DREAMPlaceTT-Framework`
2. DREAMPlace's venv is at `/localdev/ayadav/tt-work/TTPort/DREAMPlace/dp_env`
3. Both should already be mounted (verify with `docker exec <container> ls /localdev/...`)
4. Run `docker exec -e LD_LIBRARY_PATH=... <container> /localdev/.../dp_env/bin/python3 ...`

Alternative: install tt-metal's runtime libs on the host (matching the container's build) and run DREAMPlace on the host as today. More fragile but means no docker exec round-trip.

### Step 6 — Rewrite `scatter_ttnn_client.py` as a thin wrapper (0.5 hours)

Replace today's IPC-based client with:

```python
# integration/scatter_ttnn_client.py (Rung 3 version)
import v19_engine  # built in step 4

class ScatterTTNNClient:
    def __init__(self, M, N, nc_max, xl, yl, xh, yh, container_ignored, ipc_dir_ignored):
        self._engine = v19_engine.V19Engine(M, N, nc_max, xl, yl, xh, yh, "v19")

    def scatter(self, px, py, sx, sy, nc_actual):
        density, fx, fy, timings = self._engine.scatter(px, py, sx, sy, nc_actual)
        self._timings.append(timings)  # for medians/CSV
        return density, fx, fy
```

The patch into `ElectricDensityMapFunction.forward` stays the same.

### Step 7 — Test (1-2 hours)

Smoke first: run V19 in-process on `sweep_adaptec1_512`, confirm HPWL matches the previous V19 run.

Then sweep all 18 configs and compare with the existing V19-block + fp32 numbers. Targets:
- HPWL drift within ±1% of out-of-process V19 (should be bit-exact, actually, since chip work is unchanged)
- `pos_write`, `field_read`, `ipc_overhead` should drop to **0** in the metrics CSV
- `total_server_ms` should roughly match in-process `total_ms`
- ep_median should drop by the full IPC+Python bucket (~3-12 ms per iter depending on config)

---

## 4. Realistic time budget

| Step | Time | Risk |
|---|---|---|
| 1. Refactor into V19Engine class | 2-3 h | Medium — lots of state to encapsulate cleanly |
| 2. pybind11 bindings | 0.5-1 h | Low — straightforward |
| 3. CMake build target | 0.5-1 h | Medium — tt-metal CMake quirks |
| 4. Build in container | 1 h | Medium — pybind11 finding python headers |
| 5. Run DREAMPlace in container | 1-2 h | Medium — bind mounts, env vars |
| 6. Rewrite client.py | 0.5 h | Low |
| 7. Test + debug | 1-3 h | Variable |

**Total: 7-12 hours** of focused work. Plan for a fresh session.

---

## 5. Gotchas & things you'd otherwise miss

### 5.1 The ipc_dir and container args become dead

Today's `scatter_ttnn_client.py` takes `container`, `ipc_dir`, `num_cells` — all become unused. Don't remove the parameters from `patch_dreamplace()`; just ignore them for backwards compatibility (existing tests pass `--container ...` on the CLI).

### 5.2 Output ownership

The engine owns the output buffers (density, fx, fy) and reuses them across iterations. **DREAMPlace must copy them into torch tensors before the next `scatter()` call**, or its tensors will silently mutate when the engine overwrites the buffer next iter.

Cleanest fix: in pybind11, copy into freshly-allocated numpy arrays each call:
```cpp
auto den_arr = py::array_t<float>({total});
std::memcpy(den_arr.mutable_data(), r.density, total * sizeof(float));
```

Adds 1-2 ms/iter back for the memcpy. Or use `torch.tensor.share_memory_()` semantics if you want true zero-copy (more fragile).

### 5.3 Per-iter timing CSV

`scatter_ttnn_client.py` writes per-iter timings to a CSV today. Keep that mechanism — the pybind11 call returns timings as a dict; append it to `self._timings` like today.

### 5.4 Common runtime args

The current scatter program uses `SetCommonRuntimeArgs` for the 110-entry `noc_coords` table (see memory `v19_works_all_grids.md`). Don't accidentally drop that during refactor; it's the fix that made V19 work at 2048.

### 5.5 The `density_buf` is the output

The fp32-on-chip writeout writes directly to `density_buf` (an `MeshBuffer` allocated up front). For Rung 3, after `Finish(cq)` you read it via `EnqueueReadMeshBuffer(cq, output_vec, density_buf, true)`. That's how the density gets back to the host. **Don't try to bypass this** — TTNN's DCT (when CPU_DCT=0) consumes density_buf in-place.

### 5.6 The hist program runs occasionally

V11's `wl_v11_hist` is enqueued only on iter 0 + every `V11_HIST_REFRESH_ITERS`. V19 doesn't actually need hist for correctness (it's a stale optimization from sharding mode), but the program is still built. You can either keep building it (1-time cost) or strip it from V19 mode. Stripping is cleaner but riskier — leave for follow-up.

### 5.7 Don't break the V11/V18 modes

The same server binary handles `GATHER_MODE=v11`, `v11outer`, `v18`, `v19` etc. Don't accidentally break V11 baseline when refactoring. The cleanest approach is to keep `density_scatter_ttnn_server_host.cpp` as the binary entry point (for V11/V18 sweeps) **and ALSO** export the same logic as a class via `v19_engine.h`. Use composition: the binary's `main()` calls into the same engine class.

---

## 6. Where the wins actually live

From the 18-config breakdown (memory `v19_block_fp32_outcome.md`):

| Bucket | Range across 18 configs | Will Rung 3 eliminate? |
|---|---|---|
| chip (sc+ga) | 2.5-20.5 ms | No — already optimal |
| server host (h2d + d2h + fw) | 0.9-8.3 ms | No — PCIe physics |
| **IPC+Python (pw+fr+ipc)** | **2.4-12.4 ms** | **YES — the target** |
| cpu_dct (host) | 1.9-17.7 ms | No — separate problem (TTNN DCT) |
| python_remainder | 3.9-52.2 ms | Partially — copy elimination |

Rung 3 attacks bucket #3 directly. Bucket #5 (python_remainder) is fuzzy — some of it might also shrink (`pos_write` is part of it from DREAMPlace's POV), but most is the field calc / initial_density add-back inside `ElectricPotentialFunction.forward`. **Don't claim 100% bucket-5 elimination — measure.**

---

## 7. Files you'll touch

**New files**:
- `host/v19_engine.h` — class declaration
- `host/v19_engine.cpp` — class implementation (extracted from server_host)
- `host/v19_engine_pybind.cpp` — pybind11 bindings
- Possibly `integration/v19_engine_setup.py` — convenience installer/loader

**Modified files**:
- `host/density_scatter_ttnn_server_host.cpp` — `main()` becomes a thin wrapper around `V19Engine` (keeps the binary working for V11/V18 sweeps)
- `host/CMakeLists.txt` — add `v19_engine` pybind module target
- `integration/scatter_ttnn_client.py` — rewrite to use the extension
- `scripts/build_server.sh` — add the new target

**Read-only references** (don't modify but read for understanding):
- `kernels/v19_scatter_dm.cpp`, `v19_scatter_b_dm.cpp`, `v19_writeout_fp32_dm.cpp` — the kernel side, unchanged
- `integration/run_dreamplace.py` — entry point, mostly unchanged
- `docs/V19_L1_ATOMIC_HANDOFF.md` — original V19 design context

---

## 8. Acceptance criteria

Rung 3 ships when:

1. **`v19_engine` extension builds clean** inside the container, links against tt-metal libs
2. **`import v19_engine` works** in DREAMPlace's Python env
3. **Sweep_adaptec1_512 V19 wall** drops by ~5-15% (the IPC+Python savings, depending on config)
4. **HPWL matches** the out-of-process V19 within ±0.5% on at least 6 configs (adaptec1/2/3 + bigblue1/2/3 at 512)
5. **`pos_write`, `field_read`, `ipc_overhead` are zero or absent** in the per-iter timing CSV
6. **18-config full sweep converges** with HPWL drift within ±2% of out-of-process V19

If 4-5 fail, debug. If 6 fails on 1-2 configs but otherwise OK, document and ship anyway (the `bigblue1_1024_metastability` problem is unrelated to Rung 3).

---

## 9. Decision tree if things go wrong

### "pybind11 build fails to find Python headers"
- The container probably has python-dev installed. `apt-get install python3-dev` if not.
- pybind11 finds Python via `PYTHON_EXECUTABLE` cmake var. Set it to DREAMPlace's venv python.

### "Linking fails: `_ttnncpp.so: undefined symbol`"
- The extension must link against the same ttnn libs the binary links against. Inspect with `ldd host/build/density_scatter_ttnn_server` and replicate.

### "Import works but Engine() crashes"
- Same symptoms as a bad server bootup — chip device fails to open, mesh wrong size. Compare against the working binary's startup. Check `TT_METAL_HOME` env var is set.

### "Works but gives wrong HPWL"
- Likely a copy-out / buffer-reuse bug (see §5.2). Verify by adding `np.copy()` on the Python side immediately after `scatter()` returns.

### "Slower than expected"
- Measure the per-iter timing dict. If `total_ms` is close to the old `total_server_ms` but Python iter is slower, the extension is fine and the wall regression is elsewhere (cpu_dct, python_remainder).

---

## 10. Open follow-ups (for after Rung 3 lands)

- **TTNN DCT on chip**: today we run `CPU_DCT=1` because TTNN DCT had its own issues. At adaptec3_2048 the CPU DCT is ~17 ms/iter — same magnitude as the IPC+Python bucket Rung 3 eliminates. Worth revisiting.
- **`python_remainder` bucket**: instrument `ElectricPotentialFunction.forward` directly to see what specifically is in there (field calc, initial_density add-back, etc.).
- **bigblue1_1024 metastability**: separate DREAMPlace trajectory issue, see memory note.

---

## 11. Reproduction recipe (today's V19 state, as baseline before Rung 3)

To reproduce the V19-block + fp32 numbers from the 18-config sweep (so you can compare Rung 3 against them):

```bash
TT_METAL_HOME=$(pwd)/tt-metal CPU_DCT=1 GATHER_MODE=v19 V19_SCALE_BITS=20 \
  DREAMPlace/dp_env/bin/python3 integration/run_dreamplace.py \
    --device scatter_ttnn \
    --container bh-38-special-ayadav-for-reservation-75661 \
    --benchmark benchmarks/configs/sweep_adaptec1_512.json \
    --results-dir /tmp/v19_baseline
```

Expect per-iter median: scatter ~2.3 ms, gather ~0.24 ms, ep_median ~12 ms, wall ~24 s, HPWL 70,373,920. See `memory/v19_block_fp32_outcome.md` for all 18 configs' expected numbers.

---

*Last updated 2026-05-22. Created by Claude Opus 4.7 as a handoff for the Rung 3 in-process integration.*
