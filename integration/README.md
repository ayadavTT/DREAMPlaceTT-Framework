# integration/

Python layer that lets stock DREAMPlace use the TT server for its electric-potential step.

## Files

| File | Role |
|---|---|
| `run_dreamplace.py` | CLI entry point. Loads a benchmark JSON, sets up monkey-patches, runs DREAMPlace's `Placer`, writes `<bench>_<device>_metrics.json` + `<bench>_<device>_iters.csv` + `<bench>_<device>_evalmetrics.csv`. |
| `scatter_ttnn_client.py` | Host-side IPC client. Spawns the server inside Docker via `scatter_ttnn_worker.py`, owns the POSIX shmem region, monkey-patches `dreamplace.electric_potential.ElectricPotentialFunction.forward` to route through the server. |
| `scatter_ttnn_worker.py` | Docker-side exec wrapper. Sets `TT_METAL_HOME`, `LD_LIBRARY_PATH`, `ARCH_NAME=blackhole`, and `os.execve`s the `density_scatter_ttnn_server` binary. |
| `process_tt_profile.py` | Post-processes a TT-Metal device-profiler CSV into per-kernel per-zone aggregated timing. |

## How DREAMPlace gets patched

`scatter_ttnn_client.patch_dreamplace()` runs once, on first import.

1. Patches `dreamplace.electric_potential.ElectricPotentialFunction.forward` so each iter calls `ScatterTTNNClient.run_one_iter(px, py, sx, sy)` instead of computing on CPU/GPU. The client copies positions into shmem, sets `state = GO`, polls until `state = idle`, then returns `fx`, `fy` as torch tensors.
2. Patches the bare `import EvalMetrics` module (NOT `dreamplace.EvalMetrics` — DREAMPlace uses a bare import internally). `EvalMetrics.EvalMetrics.evaluate()` is wrapped to also push `(hpwl, overflow)` to a list captured in `_iter_timings`, written out as CSV at the end.

Caveat: if you `import dreamplace.EvalMetrics` from the patching site, that's a different object than the one DREAMPlace itself uses. **Use the bare import.**

## Env vars

| Var | Purpose |
|---|---|
| `DREAMPLACE_TT_SERVER_BINARY` | Override path to the `density_scatter_ttnn_server` binary. Default: `$framework/host/build/density_scatter_ttnn_server`. |
| `TT_METAL_HOME` | TT-Metal SDK root (passed to server). Default: `$framework/tt-metal`. |
| `GATHER_MODE` | `v11` (default) or `v8-sfpu`. Selects the kernel pipeline. |
| `OMP_NUM_THREADS` | For DREAMPlace's CPU side (HPWL eval, position update). Default 8. |
| `EXPORT_POS_PATH` | If set, server dumps `(M, N, xl..yh, nc, px[], py[], sx[], sy[])` at iters 1, 50, 100 → `$path.iter1.bin`, etc. |
| `EXPORT_DENSITY_PATH` | If set, server overwrite-dumps the per-iter density (`float[M*N]`) every iter. |
| `TT_METAL_DEVICE_PROFILER` | `1` to enable TT-Metal device profiler. Run `process_tt_profile.py` to aggregate the CSV afterwards. |
| `TT_METAL_LOGGER_LEVEL` | `WARNING` / `DEBUG` to control TT-Metal log verbosity. |

## Command-line shapes

```bash
# CPU baseline (no docker container needed)
python integration/run_dreamplace.py \
    --device cpu \
    --benchmark benchmarks/configs/sweep_adaptec1_2048.json \
    --results-dir results/cpu_baseline

# TT pipeline
GATHER_MODE=v11 python integration/run_dreamplace.py \
    --device scatter_ttnn \
    --container $CONTAINER \
    --benchmark benchmarks/configs/sweep_adaptec1_2048.json \
    --results-dir results/tt_sweep
```

## What's NOT in this folder

We removed these alternate-path files from the framework:
- `tt_ep_client.py` / `tt_ep_worker.py` — older FIFO-based prototype.
- `ttnn_dct_client.py` / `ttnn_dct_worker.py` — standalone DCT solver (subset of the V11 pipeline).
- `tt_density_direct.py` — in-process C++ extension; faster but more fragile, not used in current flow.

If you need them, copy from the upstream TTPort workspace.
