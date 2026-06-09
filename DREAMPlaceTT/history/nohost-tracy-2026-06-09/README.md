# TT device-zone profiling — no-host pipeline (2026-06-09)

Profiling of the no-host density+optimizer pipeline (`V22_NOHOST=drive`), branch
`density-codesign-compact-stash64-dual`. See `../../NOHOST_OPTIMIZER.md` for the full
results + the per-config timing table.

## Files here
- **`per_config_stage_timing.txt`** — the **per-iteration host-wall cost breakdown** for all
  18 configs (median ms/iter, warmup-4 dropped): forward `h2d / scatter / DCT / total` and
  backward `count / plan / place / gather / unsort / total`. This is the substantive
  per-stage timing ("how long scatter, dct, backward took"). Regenerate with
  `tools/nohost_timing.sh` (greps `[FWD]`/`[BWD-V35]` from the `.nhsw_*.log` run logs).
- **`our_device_zones.txt`** — the 68 `DeviceZoneScopedN(...)` device zones our kernels
  register (proof the live kernels are Tracy-instrumented): e.g. `V35-COUNT/PLACE/LOADBAND/
  GATHER` (backward), `GB-LOOP/GB-FXFILL` (gather BRISC), `CMP-MATH/CMP-LOOP` (v28 compute),
  `NB-LOOP/NB-FYFILL` (v28 ncrisc), `V11H-COUNT/WRITE` (histogram), `V19-FP-CONVERT/WRITE`
  (writeout), `V19N/V19B-ATOMIC` (forward stash).
- **`zone_src_locations.log`** — the raw tt-metal zone-registration dump (zone name → source
  file:line for every instrumented zone, ours + tt-metal firmware).

## Device-zone *timing* CSV (the `profile_log_device.csv`) — capture procedure

The device profiler IS active and our kernels ARE instrumented (zones above). However, the
current `tt-metal/build_Release` runs the **real-time profiler** ("[Real-time profiler] Device
1 sync complete...") which *streams* zone timings to a live Tracy client rather than writing the
post-mortem `profile_log_device.csv`. So a connected Tracy client (or a post-mortem-profiler
rebuild) is needed to dump the per-zone *timing table*. The hooks are in place:
`engine.dump_profiler()` (→ `ReadMeshDeviceProfilerResults`) is called per-iter when
`TT_PROFILE_DUMP=1`, and the run sets `TT_METAL_DEVICE_PROFILER=1`:

```bash
# (post-mortem profiler build of tt-metal required for the CSV; the real-time build streams to Tracy)
export TT_METAL_DEVICE_PROFILER=1 TT_PROFILE_DUMP=1 V22_NOHOST=drive DPTT_DEVID=1
python3 integration/run_dreamplace.py --device scatter_ttnn_inprocess \
  --benchmark benchmarks/configs/prof_adaptec1_2048.json   # 20-iter short config for profiling
# → tt-metal/generated/profiler/.logs/profile_log_device.csv  (then: integration/process_tt_profile.py)
```

Prior post-mortem device-zone captures for the V35 backward kernels live in
`results/tracy_smoke_2026-05-12/` and `results/tracy_sweep_adaptec1_2048_2026-05-12/`.
