# V19 scatter — profile

## Headline (machine-readable)
```yaml
kernel: v19-scatter
stage: forward-scatter
config: adaptec1_512        # 211K cells, 512x512 grid
cells: 370971
metric_host_ms: 2.33        # density construction (scatter+accum), median/iter, host timing hook
vs_cpu_x: 2.1               # avg across 18-config sweep (CPU 16T density construction)
vs_cpu_range: "1.27-3.16x"
critical_path: "per-cell NoC atomic-add loop (BRISC+NCRISC); NoC/atomic traffic bound, not SFPU"
load_imbalance: "cell-distribution driven (clusters -> hot owner cores)"
per_zone_tracy: PENDING     # device CSV not captured; regenerate (recipe below)
```

## Evidence on hand
- `host_iters_adaptec1_512.csv` — per-iter host-level timing (the `scatter_ms` column is this kernel's density-construction cost). From the validated `2026-06-02` adaptec1_512 run.
- Headline cross-config numbers from the 18-config sweep ([[v19_block_fp32_outcome]]).

## Per-zone Tracy — PENDING (regeneration recipe)
The kernel is instrumented with `DeviceZoneScopedN` zones (`V19N-INIT`, `V19N-CB-WAIT`,
`V19N-ATOMIC`, `V19B-DRAM-READ`, `V19B-ATOMIC`). To capture the exact per-zone device profile:

```bash
# inside the TT container, from the DREAMPlaceTT folder
export TT_METAL_HOME=$PWD/tt-metal TT_METAL_RUNTIME_ROOT=$PWD/tt-metal
export GATHER_MODE=v19
TT_METAL_DEVICE_PROFILER=1 \
  DREAMPlace/dp_env/bin/python3 integration/run_dreamplace.py \
    --device scatter_ttnn_inprocess \
    --benchmark benchmarks/configs/sweep_adaptec1_2048.json \  # 2048 = clearest zones
    --results-dir results/prof_v19 --profile
# raw CSV lands at $TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv
cp $TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv ./profile_log_device.csv
# process into the per-zone table:
python tools/profile_v11.py profile_log_device.csv > zones.txt
```
Then drop `profile_log_device.csv` + `zones.txt` here and update the `per_zone_tracy` field above
with the critical zone + its median/max µs and max/median imbalance ratio.
```
