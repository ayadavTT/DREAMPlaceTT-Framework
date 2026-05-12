# tools/

Diagnostic + benchmarking helpers. Not part of the runtime path; use as needed.

| Tool | Purpose |
|---|---|
| `v11_phase2_smoke.py` | Single-iter correctness check. Generates synthetic 1.5M cells on a 2048×2048 grid, runs scatter→accum→DCT once via the server, compares TT density vs a CPU reference. Pass criterion: `rel_L2 < 1%`. Invoked by `scripts/run_smoke.sh`. |
| `v11_phase1_smoke.py` | Older phase-1 smoke (scatter+accum only, no DCT). Kept for regression tracking. |
| `cpu_vs_tt_density.py` | Given a position dump (`EXPORT_POS_PATH`) and a TT density dump (`EXPORT_DENSITY_PATH`), compute the CPU reference density from the positions and diff. Reports `rel_L2`, mass deficit, top-10 worst bins. Used to localize density bias during convergence debugging. |
| `profile_v11.py` | Aggregates `profile_log_device.csv` (from `TT_METAL_DEVICE_PROFILER=1`) into a per-(zone, RISC) summary, with the slowest-core time per zone highlighted. See [docs/PROFILING.md](../docs/PROFILING.md) for full usage. |

## Typical debug workflow

```bash
# 1. Run with dumps enabled (server writes pos at iter 1/50/100, density every iter).
EXPORT_POS_PATH=/localdev/$USER/dbg/pos.bin \
EXPORT_DENSITY_PATH=/localdev/$USER/dbg/dens.bin \
bash scripts/run_sweep.sh sweep_adaptec1_512

# 2. Diff iter-100 TT density vs CPU computed from iter-100 positions.
python tools/cpu_vs_tt_density.py \
    /localdev/$USER/dbg/pos.bin.iter100.bin \
    /localdev/$USER/dbg/dens.bin

# Output:
#   Global: CPU sum=8.5368e+05  TT sum=8.4968e+05
#   Mass deficit: -4.0410e+03  (-0.47%)
#   rel_L2 = 0.0165
#   Top 10 worst bins (by |delta|):
#     bx   by         cpu         tt      delta
#    214  207      649.61     592.15     -57.46
#    ...
```

If `rel_L2 < 0.005` (= smoke baseline) at iter 100, the per-iter math is fine; investigate convergence further (`docs/KNOWN_ISSUES.md` #1). If `rel_L2 ≫ 0.005`, the scatter or accum has a bug — start with the worst-bin coordinates.
