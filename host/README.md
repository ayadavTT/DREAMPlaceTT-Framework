# host/

C++ server that orchestrates the V11 pipeline. Runs inside the TT-Metal Docker container.

## Files

- **`density_scatter_ttnn_server_host.cpp`** (~1700 LOC) — the server. Sets up DRAM buffers, builds and JIT-compiles kernel programs, owns the POSIX shmem IPC, dispatches one iteration per `state==GO` byte, returns `density / fx / fy` to the client.
- **`v11_tile_ownership.h`** — pure-header utilities for tile→core mapping (snake-fill) and hot-tile sharding (`build_shard_table`). Used by the server.
- **`CMakeLists.txt`** — find-package TT-Metalium, link `_ttnncpp.so` + `libtt_stl.so`, produce `density_scatter_ttnn_server`.

## Build

```bash
bash scripts/build_server.sh
# or manually:
mkdir -p build && cd build
cmake .. -DCMAKE_CXX_COMPILER=clang++-20 -DTT_METAL_HOME=$TT_METAL_HOME
make -j$(nproc)
```

Output: `host/build/density_scatter_ttnn_server`. Run inside the Docker container.

## Run signature

The server is normally launched by `integration/scatter_ttnn_worker.py` (which is exec'd inside the container by `scatter_ttnn_client.py`). Direct invocation:

```
density_scatter_ttnn_server <M> <N> <NC_max> <ipc_dir> [xl yl xh yh]
```

It JIT-compiles kernels at startup, then polls `<ipc_dir>/scatter.shm` for state changes.

## Key constants (top of `density_scatter_ttnn_server_host.cpp`)

| Name | Value | Notes |
|---|---|---|
| `v11_max_per_page_tuples` | 4096 | Per (writer, receiver) page tuple cap. Split half/half between NCRISC and BRISC scatter. |
| `MAX_OVERLAP` | 8 | Max bins one cell can overlap per axis (so 8×8 footprint). |
| `MAX_K` | 8 | Max-way sharding for hot tiles. |
| `HOT_THRESHOLD` | 5000 | Tiles with > N cells overlapping become hot at iter 0. |
| `V11_CB_SLOT_HEADROOM` | 2*MAX_K = 16 | Reserve dense slots beyond `n_owned_max` for hot-tile shards. |
| `V11_HIST_REFRESH_ITERS` | 1_000_000 | Histogram refresh disabled — see `docs/KNOWN_ISSUES.md` #2. |

## Modifying the server

Common changes:
- **Pass an extra runtime arg to a kernel**: update `SetRuntimeArgs(prog, kernel_id, cc, {...})` and the kernel's `get_arg_val<uint32_t>(...)` reads.
- **Add a new kernel**: `CreateKernel(prog, KDIR + "v11_yourname.cpp", all_crs, DataMovementConfig{...})`. Then `wl.add_program(...)`.
- **Resize an L1 region**: update both the kernel's offset arithmetic AND the host's `v11_dense_offset_bytes` mirror calculation (line ~1052).

After any change: `bash scripts/build_server.sh && bash scripts/run_smoke.sh`.
