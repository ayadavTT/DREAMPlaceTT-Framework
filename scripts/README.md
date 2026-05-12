# scripts/

Top-level shell wrappers for the common framework workflows.

| Script | Purpose |
|---|---|
| `build_server.sh` | CMake + make the TT server inside the Docker container. Honors `$TT_METAL_HOME`. |
| `run_smoke.sh` | Single-iter correctness check on synthetic data (rel_L2 < 1% = pass). |
| `run_sweep.sh` | End-to-end on the 4 benchmark configs (or a subset). Writes per-iter CSVs + metrics JSON. |
| `reset_chip.sh` | `tt-smi -r 1` chip reset. Needed after container restarts. |

All scripts use these env vars:
- `CONTAINER` — Docker container name (required, except `build_server.sh`).
- `PYTHON` — Python interpreter (default: `$framework/DREAMPlace/dp_env/bin/python3`).
- `TT_METAL_HOME` — TT-Metal SDK root (default: `$framework/tt-metal`).
- `RESULTS_DIR` — output dir for sweep (default: `$framework/results/sweep_latest/`).

See [REPRODUCE.md](../REPRODUCE.md) for the full workflow.
