# DREAMPlace-TT Sweep Results

Two snapshots are stored here:

## `sweep_a1_bb3_report/` — Baseline (May 11, 2026)

Pre-fix results from the V11 pipeline **before** the per-page tuple cap raise + sort+dedup fix. CPU baselines + TT runs across the 4 target benchmarks. Includes per-iter HPWL/overflow plots, the summary CSV, and the markdown report. Source: `experiments/density_scatter/results/sweep_a1_bb3/report/`.

| Design | G | CPU HPWL | TT HPWL | TT/CPU | CPU ovf | TT ovf | CPU iters | TT iters |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| adaptec1 | 512 | 7.028e+07 | 1.793e+08 | **+155%** | 0.07 | 0.35 | 629 | 628 |
| adaptec1 | 2048 | 7.128e+07 | 7.360e+07 | +3.26% | 0.07 | 0.07 | 618 | 771 |
| bigblue3 | 512 | 2.998e+08 | 9.923e+08 | **+231%** | 0.07 | 0.55 | 705 | 898 |
| bigblue3 | 2048 | 2.916e+08 | 8.023e+08 | **+175%** | 0.07 | 0.44 | 826 | 888 |

## `sweep_v2_final/` — Post-fix (May 11, 2026)

After the Option B fix:
- `v11_max_per_page_tuples`: 2048 → 4096
- `SRC_CHUNK`: 16 → 8 (kept L1 budget OK on grid 2048)
- Insertion sort + run-length-combine in scatter `flush_recv` (MAX_IN_FLIGHT=64)

Smoke test `rel_L2 = 3.177e-03` (unchanged from no-dedup baseline).

| Design | G | CPU HPWL | TT HPWL | TT/CPU | CPU ovf | TT ovf | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| adaptec1 | 512  | 70.3M | 164.6M | +134% | 0.07 | 0.337 | ❌ not converged |
| adaptec1 | 2048 | 71.3M | 72.5M  | **+1.6%** | 0.07 | **0.069** | ✅ converged |
| bigblue3 | 512  | 300M  | 870M   | +190% | 0.07 | 0.570 | ❌ not converged |
| bigblue3 | 2048 | 292M  | 720M   | +146% | 0.07 | 0.412 | ❌ not converged |

**Summary**: All 4 benchmarks improved from baseline. adaptec1_2048 fully converges. Grid-512 benchmarks still need further work — the per-page tuple cap is still being hit on the hottest (writer, receiver) pairs because grid 512 concentrates tuples onto only ~3 owned tiles per receiver (vs ~38 on grid 2048).

See `docs/KNOWN_ISSUES.md` for next steps (sharding refresh, per-receiver dense accumulator, etc.).

## Reproducing

```bash
bash scripts/run_sweep.sh                          # all 4 benchmarks
bash scripts/run_sweep.sh sweep_adaptec1_2048      # single benchmark
```

Outputs land in `$RESULTS_DIR` (default `./results/sweep_latest/`).
