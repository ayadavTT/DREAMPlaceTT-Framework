# host/ — TT-Metal C++ host code

Host programs that build/JIT-compile kernels and orchestrate the device. The **production
density pipeline** is the in-process pybind engine `v19_engine`; the rest are the legacy
IPC server, the microbench harness hosts, and alternative-backward engine variants.

> **Pipeline flow, per-stage cost, and the optimization frontier are in
> [`../PIPELINE.md`](../PIPELINE.md).** This file is the host-dir inventory.

## Production engine — `v19_engine` (the shipping path)
Built as a **pybind11 module** that DREAMPlace imports in-process (no IPC server):
- **`v19_engine.cpp`** (+ `v19_engine.h`) — forward engine: h2d → V19 L1-atomic scatter
  (+V31 geom stash) → V11 accumulate → V19 writeout → fold fixed-macro `initial_density`
  → TTNN DCT (`solve_device`) → electric field kept on-chip (`latest_field_addrs`).
- **`v19_engine_pybind.cpp`** — the pybind surface: `scatter_from_pos`, direct-input
  buffers, `set_skip_field_d2h`, `configure_electric_force_v35`,
  `compute_electric_force_v35_chip`.
- **`v35_ef_engine.cpp`** (+ `.h`) — V35 backward: count→(host plan)→place→gather→
  grad-unsort (reads the on-chip field + V31 stash; the only host round-trip is the final
  `grad[sel]` scatter).

Build it:
```bash
cd host/build
cmake .. -DCMAKE_CXX_COMPILER=clang++-20 -DTT_METAL_HOME=$TT_METAL_HOME   # first time
ninja v19_engine        # produces v19_engine.cpython-*.so that DREAMPlace imports
```
Kernels are JIT-loaded from `../kernels/` by path (`DENSITY_KERNEL_DIR`) — editing a
kernel needs only a cache clear, not an engine rebuild. Rebuild the `.so` only when the
engine `.cpp`/`.h` or its `CreateKernel`/`SetRuntimeArgs` wiring changes.

(Note: `host/CMakeLists.txt` also still builds the legacy `density_scatter_ttnn_server`,
and an in-comment label calls `v19_engine` "forward-only" — stale; it now links the V35
backward engine too.)

## Other host programs in this dir (not the production path)
- **`density_scatter_ttnn_server_host.cpp`** — the **legacy IPC server** (the old
  architecture: POSIX-shmem IPC, polls a GO byte). Still builds; superseded by the
  in-process `v19_engine`. Kept for reference.
- **Microbench harness hosts** — `v25_ef_l1gather_host`, `v26_host`, `v27_host`, `v28_host`,
  `v29_host`, `v30_host`, `fccs_host`, `v32_regroup_host`, `v31_backward_host`,
  `atomic_bench_host`, `mcast_bw_host`, `v11op_bench_host`, `v19_microbench_host`,
  `v35_host`, `v35live_host`, `v20_mb_permute_host`, `v21_electric_force_microbench_host`,
  `v22_*_microbench_host`, `v32_e2e_host`. These are built + run by the **history harness**
  (`../history/harness/CMakeLists.txt` + `run_harnesses.sh`), which loads kernels from its
  own vendored `../history/harness/kernels/`. Use them to test a kernel in isolation.
- **Alternative-backward engines** — `v21_ef_engine.cpp`, `v29_ef_engine.cpp`,
  `v30_ef_engine.cpp`, `fccs_ef_engine.cpp`, `v22_engine.cpp` — engine variants exercised
  by the harness / earlier experiments (V21 is the exact backward prewarmed in production).
- **`v11_tile_ownership.h`** — tile→core (snake-fill) + hot-tile sharding helpers.

## Modifying the production engine
- **Pass a runtime arg to a kernel:** update `SetRuntimeArgs(prog, kernel_id, cc, {...})`
  and the kernel's `get_arg_val<uint32_t>(...)`.
- **Add a kernel:** put the `.cpp` in `../kernels/`, then `CreateKernel(... DENSITY_KERNEL_DIR
  + "name.cpp" ...)` + its `CircularBufferConfig` + args in `v19_engine.cpp` (forward) or
  `v35_ef_engine.cpp` (backward); rebuild the `.so`.
- After any change: rebuild + quick `adaptec1_512` run (expect overflow ≈ 0.069) to confirm
  convergence. Profile **watcher-off**.
