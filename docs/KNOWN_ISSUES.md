# Known Issues + Workarounds

Catalog of issues we've hit while bringing DREAMPlace up on TT, with whether a fix exists and where in the code.

## 1. Convergence still fails on grid-512 benchmarks

**Symptom**: `sweep_adaptec1_512` and `sweep_bigblue3_512` finish DREAMPlace with `overflow ≈ 0.3–0.6` and HPWL 2–3× CPU baseline. Grid-2048 versions converge fine.

**Root cause**: per-(writer, receiver) tuple count exceeds the `v11_max_per_page_tuples / 2` cap on hot pairs. Grid 512 has 256 tiles ÷ 110 cores ≈ 3 owned tiles per receiver → tuples concentrate on few receivers; grid 2048 has ~38 owned tiles per receiver → load is naturally distributed.

**Partial fix applied (Option B)**:
- `v11_max_per_page_tuples` 2048 → 4096
- `SRC_CHUNK` 16 → 8 (keeps accum L1 budget OK on grid 2048)
- Insertion sort + run-length-combine in scatter `flush_recv` (combines duplicates within each 64-tuple batch before NOC write)
This brings density `rel_L2` at iter 100 from 15.7 % → 1.6 % and reduces HPWL gap by ~10 %, but isn't enough to clear the 0.07 overflow threshold on grid 512.

**Open**: the next step is one of:
- **Fix periodic histogram refresh** so the shard table tracks migrating hot tiles (see #2 below). The mechanism exists in `host/density_scatter_ttnn_server_host.cpp` (`V11_HIST_REFRESH_ITERS`) but is disabled because of a semaphore leak.
- **Hash-table dedup** in scatter staging to catch cross-batch duplicates (we tried this and it regressed bigblue3_512 because each cap-hit drops a larger combined-mass spill — see the git history / memory file). Would need a cap-aware spill policy.
- **Per-receiver dense accumulator** on the writer (Option A in the original plan). Eliminates the cap problem completely but doesn't fit L1: 110 × 12 KB per RISC = 1.3 MB, and we have ~1.4 MB for staging.

## 2. Periodic histogram refresh crashes (semaphore leak)

**Symptom**: setting `V11_HIST_REFRESH_ITERS` to anything ≤ 1000 causes the server to die silently after a few refreshes; `n_ep_calls=0` in the metrics JSON.

**Root cause**: each refresh in the `v11_dbg_first` block at `host/density_scatter_ttnn_server_host.cpp:1466` creates a new `Program` with fresh semaphores (one merge_sem + up to H_max_hot shard sems per core). TT-Metal's per-core semaphore limit is 16, so ~4 refreshes exhaust the pool and subsequent kernel launches deadlock.

**Workaround**: refresh is disabled (`V11_HIST_REFRESH_ITERS = 1000000`).

**Fix**: refactor the refresh path to **reuse** the original `Program`'s semaphores via `SetRuntimeArgs` only — pre-allocate the max H_max_hot semaphore slots in the initial program build, then update kernel runtime args on each refresh without recreating the program.

## 3. IOMMU sysmem mapping error after container restart

**Symptom**:
```
UMD | error | Expected NOC address: 0x1000000000000000, but got 0x1000000040000000
terminate called after throwing an instance of 'std::runtime_error'
```
appears on first kernel launch after the Docker container is restarted.

**Fix**: chip reset via `tt-smi -r 1`.
```bash
bash scripts/reset_chip.sh
```

## 4. `/home` filesystem full (Errno 28)

**Symptom**:
```
OSError: [Errno 28] No space left on device: '~/.cache/matplotlib/...'
```

**Root cause**: `/home` is often a small (10 GB) network volume.

**Fix**: redirect Python/matplotlib/pip caches to a bigger disk:
```bash
export XDG_CACHE_HOME=/localdev/<user>/.cache
export PIP_CACHE_DIR=/localdev/<user>/.cache/pip
export MPLCONFIGDIR=/localdev/<user>/.cache/matplotlib
```

## 5. Cross-RISC L1 hand-off — fences and cache invalidates

**Hypothesis we tried and dropped**: `asm("fence rw, rw")` between NCRISC's CPU stores and BRISC's reads on shared L1; we also tried `invalidate_l1_cache()` before reads.

**Outcome**: **zero effect on convergence numbers** across multiple sweeps. The hardware on Blackhole apparently handles cross-RISC visibility correctly without explicit fences in our access pattern. **Do not waste time on this** before verifying cap-hit drops (#1).

Detail saved in `memory/feedback_ncrisc_fence.md` in the project memory directory.

## 6. `dbg_pos_*.bin` / `dbg_dens_*.bin` files

The server has `EXPORT_POS_PATH` and `EXPORT_DENSITY_PATH` environment variables that, when set, dump per-iter position vectors (at iter 1, 50, 100) and overwrite-style density. Used by `tools/cpu_vs_tt_density.py` to diff TT vs a CPU reference computed from the same iter's positions. The dump writes only when the env var is set, so it's free if you don't enable it.

## 7. Kernel JIT cache stale after kernel edit

**Symptom**: you edit a kernel `.cpp` file, restart the server, but observe old behavior.

**Root cause**: TT-Metal caches JIT'd kernels by content hash. If your edit's hash matches a cached entry (unlikely but possible after revert), the old binary is used. More commonly: the kernel file path baked into the server binary (`DENSITY_KERNEL_DIR`) points elsewhere than where you edited.

**Fix**:
- Verify `DENSITY_KERNEL_DIR` (set in `host/CMakeLists.txt`) points to your `kernels/` folder.
- If you've truly hit a cache collision: `rm -rf ~/.cache/tt-metal` (or the equivalent inside the container).

## 8. Performance: scatter is ~46 ms (vs ~16 ms baseline) with sort dedup

**Cause**: insertion sort O(n²) at MAX_IN_FLIGHT=64 costs ~20 ms across many flushes. Acceptable for correctness but not optimal.

**Open**: replace with radix sort (O(n)) or hash-table dedup (~1.3 µs per insert avg). We benchmarked hash-table at 23 ms scatter (faster) but its correctness on bigblue3_512 was worse (see #1). Needs more work.
