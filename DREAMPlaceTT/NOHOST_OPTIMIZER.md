# DREAMPlace density step + optimizer — fully on Tenstorrent Blackhole ("no host in the loop")

**Status:** ✅ DONE + validated on all 18 benchmark configs (2026-06-09). Branch `density-codesign-compact-stash64-dual`. Device user ayadav@tenstorrent.com.

This is the end state of the on-chip-optimizer effort (supersedes the "NOT YET DONE" list in `ONCHIP_OPTIMIZER_HANDOFF.md`): the DREAMPlace **density forward, density-gradient backward, gradient unsort, AND the optimizer position update** all run on Blackhole, with the **density gradient never round-tripping to the host**.

---

## 1. What "no host in the loop" means here (and the one allowed transfer)

Per iteration, everything *we* compute is on-device:

```
FORWARD  (TT): pos → V19 uint32 scatter (+V31 geom/route stash) → TTNN DCT → field (resident)
BACKWARD (TT): V35 count → place → gather  →  on-chip unsort → density grad RESIDENT in b_dg
OPTIMIZER(TT, V22): combine(raw_wl + dw·b_dg) → precond → nesterov → clamp ;
                    line-search step size α = sqrt(Σ(Δpos)²/Σ(Δgrad)²) ALSO on-device
POSITIONS:    resident in V22's b_v / b_vclamp, updated on-device
```

- **The density gradient never touches the host** — the on-chip unsort writes it straight into V22's `b_dg`, and V22 consumes it on-device. The CPU unsort (and its ~36 ms d2h) is elided (`skip_cpu_unsort`).
- **The only host transfers are the explicitly-allowed ones** (per the handoff §1): the **wirelength gradient** `raw_wl` (DREAMPlace's own operator — a data transfer, not "our" processing) is uploaded h2d; positions are mirrored to host **only** for DREAMPlace's wirelength operator + the HPWL/overflow metrics; per-iter scalars (α, coef, density_weight) are host scalars.
- The positions are **resident on-device** (V22 updates them there). They are also copied to host each iteration *solely* because DREAMPlace's wirelength operator is a CPU op that needs host pos — removing that would require porting the wirelength gradient to TT (out of scope; see §6).

---

## 2. Result: 16/16 non-deferred configs converge under no-host drive

`.verifymatrix_nohost.sh` (`V22_NOHOST=drive`), all on attempt 1, **zero retries** (fully stable):

| config | overflow | config | overflow | config | overflow |
|---|---|---|---|---|---|
| adaptec1_512  | 0.0694 | adaptec2_512  | 0.0696 | adaptec3_512  | 0.0660 |
| adaptec1_1024 | 0.0690 | adaptec2_1024 | 0.0696 | adaptec3_1024 | 0.0692 |
| adaptec1_2048 | 0.0689 | adaptec2_2048 | 0.0692 | **adaptec3_2048** | **0.0819 (deferred)** |
| bigblue1_512  | 0.0675 | bigblue2_512  | 0.0661 | bigblue3_512  | 0.0693 |
| bigblue1_1024 | 0.0697 | bigblue2_1024 | 0.0694 | bigblue3_1024 | 0.0692 |
| bigblue1_2048 | 0.0690 | **bigblue2_2048** | **0.0748 (deferred)** | bigblue3_2048 | 0.0695 |

The **2 deferred configs** (adaptec3_2048, bigblue2_2048) stall at ~0.07–0.08 on the **production/host** path too (handoff §8) — our no-host results match (bigblue2_2048 is even slightly better than the ~0.082 host baseline). They are **not** no-host regressions. The other **16 converge ≤ 0.07**, bit-comparably to production.

---

## 3. Per-iteration cost breakdown (host wall-clock, median ms/iter, warmup-4 dropped)

From the 18 no-host run logs (`.nhsw_<config>.log`; `FWD_TIMING`/`V35_TIMING` forced on in the production lock). **Forward** stages + **backward** (V35) stages. `BWD d2h` here is the **on-chip unsort device time** (single-RISC), NOT a host transfer.

| config | FWD h2d | scatter | DCT | FWD total | BWD count | plan | place | gather | unsort | BWD total | iters |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| adaptec1_512  | 0.64 | 4.72 | 0.97 | 7.54 | 0.49 | 0.24 | 1.24 | 2.05 | 1.45 | 5.48 | 598 |
| adaptec1_1024 | 0.68 | 5.92 | 1.03 | 9.29 | 0.53 | 0.26 | 1.09 | 2.04 | 1.70 | 5.64 | 665 |
| adaptec1_2048 | 0.80 | 12.20 | 2.79 | 19.95 | 0.54 | 0.47 | 1.11 | 3.27 | 2.03 | 7.44 | 682 |
| adaptec2_512  | 1.07 | 7.70 | 0.98 | 11.13 | 0.76 | 0.26 | 1.94 | 2.96 | 2.14 | 8.06 | 587 |
| adaptec2_1024 | 0.94 | 8.52 | 0.99 | 12.10 | 0.79 | 0.27 | 1.83 | 2.87 | 2.11 | 7.88 | 636 |
| adaptec2_2048 | 0.93 | 13.07 | 2.80 | 20.74 | 0.81 | 0.43 | 1.85 | 3.41 | 2.68 | 9.17 | 745 |
| adaptec3_512  | 2.99 | 16.43 | 1.08 | 22.06 | 1.65 | 0.29 | 4.35 | 5.40 | 4.69 | 16.39 | 662 |
| adaptec3_1024 | 2.37 | 16.56 | 1.10 | 21.89 | 1.66 | 0.30 | 4.28 | 5.60 | 4.61 | 16.46 | 670 |
| adaptec3_2048 | 2.48 | 19.25 | 2.81 | 28.86 | 1.70 | 0.47 | 4.59 | 6.19 | 5.88 | 18.84 | 1115 |
| bigblue1_512  | 0.98 | 7.67 | 0.91 | 10.77 | 0.79 | 0.25 | 2.05 | 2.98 | 2.28 | 8.35 | 638 |
| bigblue1_1024 | 1.26 | 9.45 | 1.12 | 13.62 | 0.82 | 0.28 | 1.99 | 2.79 | 2.26 | 8.15 | 923 |
| bigblue1_2048 | 1.00 | 18.71 | 2.79 | 26.64 | 0.84 | 0.43 | 1.99 | 4.54 | 2.83 | 10.63 | 767 |
| bigblue2_512  | 2.92 | 18.06 | 0.97 | 23.26 | 1.73 | 0.28 | 4.61 | 5.77 | 4.99 | 17.33 | 627 |
| bigblue2_1024 | 2.16 | 19.57 | 1.09 | 24.63 | 1.79 | 0.30 | 4.85 | 6.38 | 4.99 | 18.30 | 682 |
| bigblue2_2048 | 2.09 | 23.33 | 2.79 | 32.44 | 1.80 | 0.37 | 4.83 | 6.55 | 6.23 | 19.76 | 903 |
| bigblue3_512  | 3.03 | 23.70 | 1.14 | 29.15 | 2.38 | 0.29 | 6.37 | 7.43 | 6.86 | 23.34 | 682 |
| bigblue3_1024 | 2.83 | 26.54 | 1.14 | 32.38 | 2.43 | 0.30 | 6.76 | 8.45 | 6.82 | 24.72 | 720 |
| bigblue3_2048 | 2.81 | 32.79 | 2.77 | 42.50 | 2.44 | 0.43 | 6.76 | 10.03 | 8.66 | 28.29 | 861 |

`DCT` = the TTNN DCT compute (`DCT(up/comp/dn)` comp term). `FWD total` = forward server wall. The V22 optimizer step (combine/precond/nesterov/clamp + the α reduction) is the per-iter on-device optimizer overhead, ~6 SFPU launches. Regenerate this table with `tools/nohost_timing.sh` (greps `[FWD]`/`[BWD-V35]` from each `.nhsw_*.log`); a saved copy + the device-zone details are in `history/nohost-tracy-2026-06-09/`.

**Observations:** forward `scatter` dominates the forward (memory-bound uint32 scatter); backward `gather`+`place`+`unsort` dominate the backward. The on-chip `unsort` is single-RISC (the dualization lever, ~2×, is documented in `onchip-unsort-64b-reliable`). The no-host loop is somewhat slower per-iter than the host path (extra V22 launches + single-RISC unsort) but converges everywhere.

---

## 4. Live no-host pipeline + the kernels it uses

Auto-selected config (from the `[v35]` line): `grouped=64B(COMPACT) source=64B dual_risc=YES bucket=YES max_k=8 max_h=4`. Forward: `V31_STASH`+`V31_GEOM`+`V19_SKIP_CELL_SORT`.

| stage | kernels (in `DREAMPlaceTT/kernels/`) |
|---|---|
| forward scatter + stash | `v11_histogram.cpp`, `v11_accum_dm.cpp`, `v11_accum_n_dm.cpp`, `v19_writeout_fp32_dm.cpp`, `v19_writeout_void_ncrisc.cpp`, `v31_scatter_stash_dm.cpp`, `v31_scatter_b_stash_dm.cpp` |
| DCT | TTNN ops (no custom kernel) |
| backward (V35, 64B-compact + stash64 + dual + bucket) | `v35_count64_bucket_n.cpp`, `v35_place_s64_bucket_n.cpp`, `v35_gather_brisc_compact.cpp` (+ `v28_compute_pbf.cpp`/`v28_ncrisc_pbf.cpp`), `v35_onchip_unsort.cpp` |
| optimizer (V22) | `v22_gradcombine_compute.cpp`, `v22_precond_compute.cpp`, `v22_nesterov_compute.cpp`, `v22_clamp_compute.cpp`, `v22_stepsize_compute.cpp` + `v22_elt_reader/writer.cpp`, `v22_nesterov_reader/writer.cpp`, `v22_stepsize_reader/writer.cpp` |

The V22 line-search machinery reuses these kernels: `gradcombine` doubles as a scalar-add (the `Δpos`/`Δgrad` diffs) and a copy (the g_k snapshot + the commit); `stepsize` does the `Σ(Δpos)²`/`Σ(Δgrad)²` reduction. **No new kernels were added for the no-host loop.** A snapshot of the live kernel set is preserved under `DREAMPlaceTT/history/nohost-prod-kernels-2026-06-09/`.

**Device-zone (Tracy) instrumentation.** The live kernels carry `DeviceZoneScopedN(...)` markers (68 zones — e.g. `V35-COUNT/PLACE/LOADBAND/GATHER`, `GB-LOOP/GB-FXFILL`, `CMP-MATH`, `NB-LOOP`, `V11H-COUNT`, `V19-FP-CONVERT/WRITE`, `V19N/V19B-ATOMIC`; listed in `history/nohost-tracy-2026-06-09/our_device_zones.txt`). The device profiler runs (`TT_METAL_DEVICE_PROFILER=1`, profiler starts + syncs on device 1, zones registered). **NOTE:** the current `tt-metal/build_Release` uses the *real-time* profiler (streams to a live Tracy client), so the post-mortem per-zone `profile_log_device.csv` requires a connected Tracy client or a post-mortem-profiler rebuild — see `history/nohost-tracy-2026-06-09/README.md` for the capture procedure. The **per-stage host-wall breakdown (§3)** is the captured per-iteration cost data.

---

## 5. How to run / reproduce

```bash
C=bh-37-special-ayadav-for-reservation-82536; F=/localdev/ayadav/DREAMPlaceTT-Framework
docker exec "$C" bash -lc "
  export TT_METAL_HOME=$F/tt-metal TT_METAL_RUNTIME_ROOT=$F/tt-metal ARCH_NAME=blackhole TT_METAL_CACHE=/dev/shm/ttc_nohost
  export DREAMPLACE_DIR=$F/DREAMPlace/install PYTHONPATH=$F/DREAMPlace/install:$F/DREAMPlaceTT/host/build:$F/integration
  export DPTT_DEVID=1 V22_NOHOST=drive
  $F/DREAMPlace/dp_env/bin/python3 $F/integration/run_dreamplace.py \
    --device scatter_ttnn_inprocess --benchmark $F/benchmarks/configs/sweep_adaptec1_512.json \
    --results-dir /tmp/nh 2>&1 | tail -40"
```

- **Env flags** (all gated; production is the default with none set):
  - `V22_NOHOST=drive` — the full no-host loop (grad resident, on-device optimizer + α).
  - `V22_NOHOST=shadow_grad` — validate V22's on-device combine+precond vs host g_k (non-driving).
  - `V22_OPT=shadow|drive` — the earlier *with-host* drive (V22 does only the position update fed the host gradient; kept for comparison).
  - `V35_ONCHIP_UNSORT=1` — validate the on-chip unsort bit-exact vs the CPU unsort (correctness gate, no optimizer).
- **Full 18-config sweep:** `F/.verifymatrix_nohost.sh` (per-config container restart + fresh cache + retry-on-blowup; writes `.verifymatrix_nohost_summary.txt`).
- Build the pybind after editing `host/*.cpp`/`kernels/*.cpp`: `cd DREAMPlaceTT/host/build && ninja v19_engine` (Ninja, not make). Kernels JIT at runtime — use a fresh `TT_METAL_CACHE`.

---

## 6. Deferred / optional (with rationale)

- **Forward reads resident pos (handoff Step C):** would *not* remove the pos round-trip — DREAMPlace's host wirelength operator needs host pos regardless — it only saves the density-forward's ~1 ms/iter `px/py` upload, via an invasive change to the locked production forward. Low value; deferred. (A reusable interleaved-pos permute could use `history/99-infra-probes/v20-permute/`.) Revisit only if the wirelength gradient is also ported to TT.
- **Perf:** the in-engine on-chip unsort is single-RISC (dualize → ~2×, see `onchip-unsort-64b-reliable`); the no-host loop runs ~6 extra SFPU launches/iter. Convergence is unaffected.
- **2-config stall (adaptec3_2048, bigblue2_2048):** pre-existing, upstream of this work (handoff §8) — present on the CPU-unsort/host path too.
