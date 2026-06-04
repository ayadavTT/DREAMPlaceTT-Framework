# Kernel history — a reference library for building new TT kernels

This folder is a curated, systematic record of every density/placement kernel we
built, tried, and tested on Tenstorrent Blackhole for the DREAMPlace port. It
exists so a future agent (human or AI) can **read one folder and understand a
proven approach end-to-end** — its mechanism, its measured cost, the tricks that
made it work, and the pitfalls that killed earlier variants — and reuse that to
design/test a new kernel faster.

## How to use this library
1. **Start at `MANIFEST.csv`** — one row per kernel: stage, status, headline cost,
   vs-CPU speedup, whether a profile is attached, and memory refs. Grep it to find
   "what's the best known approach for stage X and what did it cost."
2. **`LEADERBOARD.md`** ranks kernels within each stage by cost/effectiveness with
   the one-line trick + when-to-use.
3. **Open the kernel's folder** (`<stage>/<kernel>/`):
   - `KERNEL.md` — the full systematic doc (8-section schema; see `TEMPLATE/KERNEL.md`).
   - `src/` — the kernel source files, verbatim.
   - `profile/` — `profile_log_device.csv` (raw Tracy device profiler), `zones.txt`
     (processed per-zone table), and `PROFILE.md` (headline schema + interpretation).
4. **Building a new kernel?** Copy `TEMPLATE/KERNEL.md`, find the closest prior
   approach here, reuse its `src/` as a starting point, and read its §6 (gotchas)
   and §7 (lessons) before you start — they encode hard-won Blackhole-specific traps.

## Layout (grouped by pipeline stage)
```
01-forward-scatter/      cell -> bin area spreading (build density map)
02-forward-gather-accum/ accumulate/reduce partials + on-chip density writeout
03-forward-geom-stash/   forward stashes per-cell geometry for a prep-free backward
04-backward-bucketing/   group cells by field region (regroup / bin-owner / band)
05-backward-ef-gather/   sample field at cell positions, weighted -> electric-force grad
06-backward-grad-writeout/ grad emit / readout
07-optimizer-nesterov/   chip-resident Nesterov/BB elementwise update (clamp/precond/...)
08-dct-solve/            NOT a kernel — the on-chip TTNN matmul DCT recipe (host code)
99-infra-probes/         microbenches/probes that informed design (lessons, not pipeline kernels)
```

## Status legend
- ⭐ **SHIPPED** — in a production/live path.
- ✓ **VALIDATED** — correct (bit-exact / rel_l2 / HPWL-converges) in microbench or core; may not be the perf winner.
- ◐ **SUPERSEDED** — worked as an intermediate, replaced by a later approach.
- ✗ **REFUTED** — built + tested, turned out a dead-end (kept for the lesson).
- 🔧 **INFRA** — probe/microbench/diagnostic, not a pipeline kernel.

## A note on profiles
Many recent kernels have only host-level timing saved, not per-zone device Tracy
CSVs (those need a `TT_METAL_DEVICE_PROFILER=1` build + chip run). Where the raw
CSV is missing, `profile/PROFILE.md` carries the measured headline + bottleneck
finding (from the run logs / memory) and an exact **regeneration recipe**. Each
kernel folder is profiled at a **canonical config for its stage** so the
LEADERBOARD numbers are comparable (forward: `adaptec1_2048`; backward-EF: the EF
microbench 512/1024/2048 spans).

## DCT
The forward DCT/IDXST/IDCT solve is **not a Tensix kernel** — it's the
`TTNNDCTSolver` matmul chain in `host/v19_engine.cpp`. `08-dct-solve/` documents
that recipe (host-code excerpt + the bf16-vs-HiFi4 precision note), with no `src/`.

---

## Measured results (real silicon — bh-38, 2026-06-03)

This library is backed by **real on-chip measurements**, not just memory. See
`LEADERBOARD.md` for the cross-kernel cost/effectiveness table, and each kernel's
`profile/` (`results*.csv` for forward, `run.log` + `REPORT.md` for backward).

**Captured with real data (11 kernels):**
- Forward scatter: **v19** (54/54 PASS, 1.45–6.5 ms, distribution-insensitive),
  **v11** (48/54 — fails bigblue3-clustered: route_buf cap overflow), v18 (crashes there).
- Backward EF: **v25** (0.422 ms ≈17× CPU), **v28** (1.676 ms, all-grids correct),
  **fccs** (1.136 ms), v26 (chip 0.783 ms), v27 (0.793 ms), v30 (0.696 ms),
  v29 (10.25 ms, standalone correctness CHECK).

**Pending a clean device** (the v31_backward standalone wedges the chip; a device
reset was blocked this session): v32_regroup, atomic-bench, mcast-bw, v11op-bench
(timed out on a wedged device) — re-run via `harness/run_harnesses.sh`.
**File-based** (need `--inputs` .bin): v21_ef, v22_{nesterov,elt,stepsize} —
offline-validated vs torch. **Per-zone Tracy** (`zones.txt`): the build has
`ENABLE_TRACY=ON`; capture pending a `dump_profiler()` pass on a clean device.

## Reproducing the tests (self-contained)
```bash
# forward family (engine-driven), per kernel:
bash 01-forward-scatter/v19-scatter/test/run.sh            # or --quick
# backward/infra family (standalone harnesses), all at once:
bash harness/run_harnesses.sh
```
Both use only the in-folder `tt-metal`, `kernels/`, and `DREAMPlace/dp_env`.
`harness/run_harnesses.sh` deliberately **excludes v31_backward** (it wedges the
device standalone — run it only via the live engine).
