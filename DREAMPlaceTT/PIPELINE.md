# DREAMPlaceTT — Production Density Pipeline (forward + backward)

> **Audience:** an engineer/agent optimizing the TT-Metal density forward/backward
> on Blackhole — either trimming wasted cost or building next-gen kernels.
> **This file is the live map.** The per-kernel deep-dives are in
> `history/<stage>/<version>/KERNEL.md`; this file ties them into the *current*
> shipping pipeline, gives the measured cost breakdown, and points at the
> optimization frontier.
>
> Last verified: 2026-06-06 (commit on `main` after the V35 convergence fix +
> kernel-dir prune). If the engine/client changed since, re-verify against the
> source paths cited below.

---

## 0. TL;DR — what ships today

DREAMPlace's `ElectricPotential` forward+backward run **fully on the TT device**
via an **in-process pybind engine** (no IPC server). One placement iteration:

```
 forward  : cell pos → V19 L1-atomic scatter (+V31 geom stash) → density
            → fold fixed-macro initial_density → TTNN DCT → electric field (on chip)
 backward : field (on chip) + V31 stash → V35 halo-tile gather → per-cell force → grad
```

- **Converges** to DREAMPlace's overflow threshold: 16/18 sweep configs ≤ 0.07
  (the 2 misses — adaptec3_2048 0.081 / bigblue2_2048 0.082 — are *physically*
  borderline; even the exact fp64 CPU backward can't reach 0.07 for them).
- **Activated by:** `--device scatter_ttnn_inprocess` with **no env flags** — the
  production lock in `integration/scatter_ttnn_client_inprocess.py::patch_dreamplace()`
  forces the path (`GATHER_MODE=v19`, `V21_EF/V35_EF/V31_STASH/V31_GEOM/V31_EF_GEOM/
  V19_SKIP_CELL_SORT/FWD_TIMING/V35_TIMING=1`) and disconnects all alternatives.
- **Grid:** 13×10 = 130 Tensix worker cores on Blackhole p150b; cells batched 1024/tile.

---

## 1. Where the code lives (current, not historical)

| Component | File | Role |
|---|---|---|
| Forward engine | `host/v19_engine.cpp` (+ `.h`) | scatter→accum→writeout→fold→DCT; owns DRAM/L1 buffers, JIT-compiles kernels |
| Forward pybind | `host/v19_engine_pybind.cpp` | `scatter_from_pos`, direct-input buffers, skip-field-d2h; builds the `.so` DREAMPlace imports |
| Backward engine | `host/v35_ef_engine.cpp` (+ `.h`) | V35 count→plan→place→gather→grad-unsort |
| Integration / lock | `integration/scatter_ttnn_client_inprocess.py` | patches `ElectricPotentialFunction.forward/backward`; the production lock; per-stage timing |
| Forward EP hook | `integration/scatter_ttnn_client.py` | forward-hook timing (`[fwd_full]`) |
| Live kernels | `kernels/*.cpp` (28 .cpp + `v21_ef_common.h`) | the JIT-loaded set — see §3 |
| Kernel archive | `history/<stage>/<version>/{KERNEL.md,src,profile}` | every past + present kernel, documented |
| Microbench harness | `history/harness/` (`run_harnesses.sh`, vendored `kernels/`) | isolated per-kernel test rig |

Build the engine `.so`: `cd host/build && ninja v19_engine` (needs `TT_METAL_HOME`).
Kernels are **JIT-compiled at runtime** from `kernels/` by path string — no rebuild
to change a kernel, just clear the cache (`rm -rf $TT_METAL_CACHE`).

---

## 2. The pipeline, stage by stage

### Forward — `V19Engine::scatter()` (`host/v19_engine.cpp`)
1. **h2d** — `EnqueueWriteMeshBuffer` px/py/sx/sy (cell centers + clamped sizes) → device; `Finish`. Direct-input buffers avoid a host memcpy (`use_direct_in_`).
2. **histogram** (one-time, iter 0) — `wl_v11_hist` (`v11_histogram.cpp`): per-tile cell counts → shard table for hot-tile reduction. Refresh is disabled (`V11_HIST_REFRESH_ITERS=1e6`).
3. **scatter (+stash)** — `wl_v11_scatter`: per cell, `v4_reader` loads pos, `v4_compute` (SFPU) computes the bin `(bxl,byl)` + clamped triangle overlaps `ox[0..7]/oy[0..7]`; then **`v31_scatter_stash_dm`** (NCRISC, cells 0–511) / **`v31_scatter_b_stash_dm`** (BRISC, cells 512–1023) **atomic-add** the density into per-bin L1 slabs (uint32 fixed-point, `V19_SCALE_BITS=20`) **and** write a 128-B per-cell geometry stash (gidx, bxl_g, byl, kc, hc, `word5=ratio_c·bin_area`, px[8], py[8]) for the V35 backward.
4. **accumulate** — `wl_v11_accum` (`v11_accum_dm` BRISC + `v11_accum_n_dm` NCRISC): merge the per-core density halves, hot-tile shard-reduce.
5. **writeout** — `wl_v19_writeout` (`v19_writeout_fp32_dm` BRISC + `v19_writeout_void_ncrisc` NCRISC void): scale by `1/(bsx·bsy)` and write the fp32 density map to DRAM.
6. **fold** — add the fixed-macro `initial_density` (`set_initial_density`, normalized `/bin_area`, uploaded once as `id_cache_tt`) via `ttnn::add` before the DCT.
7. **DCT** — `solve_device()` (TTNN C++ matmuls, ROW_MAJOR→TILE→matmuls→ROW_MAJOR): density → potential → **electric field** `field_x/field_y`, kept **on chip** (`latest_field_addrs()` exposes the row-major DRAM addresses; `skip_field_d2h=True` so it is NOT downloaded — V35 reads it on-chip).

### Backward — `compute_electric_force_v35_chip()` (`host/v35_ef_engine.cpp`)
Reads the on-chip field + the V31 stash, host-free except the final grad-unsort:
1. **count** — `v35_count.cpp`: per-tile cell counts (for the grouping plan).
2. **plan** (host) — read counts, balance tiles across the 130 workers, write `tile_base`/`srcpref`, set gather runtime args.
3. **place** — `v35_place.cpp`: scatter each cell into a single tile-grouped buffer (PASS 2), so each gather worker owns a contiguous slice of one tile → one resident field band.
4. **gather** — `v35_gather_brisc.cpp` (BRISC: loads the fx band **page-by-page** from the page-interleaved field, builds px/ratio/base_idx, drains oidx to DRAM) + `v28_ncrisc.cpp` (NCRISC: loads the fy band, drains gx/gy) + `v28_compute.cpp` (TRISC SFPU: `gx=ratio·Σ px[k]·py[h]·fx[k,h]`, `gy` likewise).
5. **d2h grad-unsort** (host) — read `gradx/grady/oidx`, scatter `grad[sel[oidx]] += gx/gy`. **This host readback is the dominant backward residual** (see §4).

`compute_from_chip` produces +force; the client negates to −force and writes `out[sel]`.

> **V21 vs V35.** The lock sets `V21_EF=1` (patches the backward, prewarms the exact
> on-chip `v21_ef_brisc/ncrisc/compute`, cos 1.0 vs CPU) **and** `V35_EF=1` (routes the
> actual gradient to the faster V35 tile-grouped gather). So `v21_ef_*` are compiled
> but V35 computes the force. V21 is the slower **exact** backward, useful as the
> ground-truth oracle (run `DBG_V35_EF=0 DBG_V21_EF_USE_CPU_GRAD=1` for TT-fwd + CPU-bwd).

---

## 3. The 28 live kernels (in `kernels/`)

**Forward (9):** `v4_reader`, `v4_compute`, `v11_histogram`, `v31_scatter_stash_dm`,
`v31_scatter_b_stash_dm`, `v11_accum_dm`, `v11_accum_n_dm`, `v19_writeout_fp32_dm`,
`v19_writeout_void_ncrisc`. (DCT uses TTNN's built-in tilize/matmul/untilize/eltwise
kernels — not in this dir.)
**Backward V35 (5):** `v35_count`, `v35_place`, `v35_gather_brisc`, `v28_ncrisc`, `v28_compute`.
**Backward V21 exact-fallback (3, prewarmed):** `v21_ef_brisc`, `v21_ef_ncrisc`, `v21_ef_compute` (+ `v21_ef_common.h`).
**On-chip optimizer (11, not yet wired — kept for the planned Nesterov fusion):** `v22_*`.

Per-kernel deep-dives (mechanism, gotchas, lineage): `history/<stage>/<version>/KERNEL.md`.
The shipping kernels map to: forward scatter+stash → `history/03-forward-geom-stash/v31-stash`
and `history/01-forward-scatter/{v4-base,v19-scatter}`; accum → `history/02-forward-gather-accum`;
V35 backward → `history/05-backward-ef-gather/{v35-halotile,v28-ef-multibin,v21-ef}`;
DCT → `history/08-dct-solve`; optimizer → `history/07-optimizer-nesterov`.

---

## 4. Measured cost + the optimization frontier

Per-iteration density step, **watcher-off** (`TT_METAL_WATCHER` adds 1.5–2.6×; always
profile with it OFF). Stages sum to the totals. Numbers from the 2026-06-06 sweep.

**Forward EP_TOTAL (ms): scatter-dominated.** scatter is 55–74% of the forward and
scales with cell count. h2d / writeout / dct / fold are small and roughly flat.

| config | h2d | **scatter** | writeout | dct | field_d2h | fold | EP_TOT |
|---|---|---|---|---|---|---|---|
| adaptec1_512 | 0.59 | **5.51** | 0.23 | 1.03 | 0.18 | 0.74 | 9.45 |
| adaptec1_2048 | 0.70 | **13.13** | 2.77 | 2.81 | 0.41 | 0.76 | 21.9 |
| bigblue3_2048 | 2.78 | **36.51** | 2.61 | 2.85 | 0.41 | 0.74 | 50.4 |

**Backward TOTAL (ms): place + gather + d2h-dominated.** The host **grad-unsort d2h**
is the biggest single cost on heavy configs (grows with cell count → 44 ms on
bigblue3_2048) because the gradient is read back to host and scattered `grad[sel]`.

| config | count | plan | place | gather | **d2h** | TOTAL |
|---|---|---|---|---|---|---|
| adaptec1_512 | 1.06 | 0.26 | 2.64 | 2.73 | **3.75** | 11.0 |
| adaptec1_2048 | 1.06 | 0.26 | 2.54 | 4.16 | **5.86** | 14.6 |
| bigblue3_2048 | 5.50 | 0.34 | 14.6 | 16.1 | **44.2** | 83.6 |

### The frontier (highest-value targets)
1. **Backward `d2h` grad-unsort (biggest residual).** The whole point of V35 is on-chip
   gather, yet the gradient still round-trips to host for the `grad[sel[oidx]]` scatter.
   Eliminating it needs **on-chip optimizer fusion** (keep grad on device, feed the v22
   Nesterov kernels directly) — the v22 kernels exist for exactly this.
2. **Forward `scatter` (dominates forward).** L1 atomic-add over 130 cores; cost scales
   with cells. Levers: better tile routing / load balance, fewer atomics, or a
   compute-side (SFPU outer-product) variant (see `history/99-infra-probes/v11op-bench`).
3. **Backward `place` + `gather`.** The halo-tile grouping (place) + the per-(k,h) SFPU
   sum (gather). `gather` grows with the bin-span; the `MAXKH=8` cap truncates the widest
   dense-center cells (a small under-count — raising it needs a wider stash record).
4. **DCT (`dct≈2.8 ms`)** is already small (TTNN matmuls, `Finish`-bracketed) — low ROI.

`fold` and `field_d2h` are near-zero (the field stays on chip via `skip_field_d2h`);
do **not** re-introduce the host field download.

---

## 5. Developing a new kernel (isolation → production)

- **Test in isolation:** add the kernel to `history/harness/kernels/`, add a host `.cpp`
  (copy an existing `host/<x>_host.cpp` as a template) + a `HARNESSES` line in
  `history/harness/CMakeLists.txt`, then `bash history/harness/run_harnesses.sh`
  (output → each kernel's `history/<stage>/<version>/profile/run.log`, Tracy zones).
  The harness is **self-contained** (its own vendored `kernels/`), decoupled from the
  production dir.
- **Promote to production:** copy the `.cpp` into `kernels/`; wire a `CreateKernel(...)` +
  `CircularBufferConfig` + `SetRuntimeArgs` into `host/v19_engine.cpp` (forward) or
  `host/v35_ef_engine.cpp` (backward); rebuild the `.so`. (Kernels are wired by the host
  program in TT-Metal — there is no config-only toggle.)
- **Always:** profile **watcher-off**; verify convergence with a quick `adaptec1_512` run
  (expect overflow ≈ 0.069) before trusting a change.

---

## 6. Hard-won gotchas (read before optimizing)
- **NoC atomics:** use `noc_semaphore_inc`, never raw `noc_atomic_*`; blind COUNT is exact at any contention, fetch-add *return value* is only correct at ≤2-way (`history/99-infra-probes/atomic-bench`). ⇒ count-prefix regroup, not atomic-slot.
- **Page-interleaved field:** read it **page-by-page** (`get_noc_addr(page)`), never one contiguous block — pages are striped across DRAM banks (this bug gave NaN gradients).
- **CB flow-control:** every reserved CB must be consumed, or the producer dead-locks (the V35 `CB_OIDX` deadlock). A vestigial push with too few slots wedges the gather.
- **Per-cell ratio:** the stashed px/py are *ratio-free* clamped overlaps; the per-cell `ratio_c` must ride in `word5=ratio_c·bin_area`, else movable cells over-count by `1/ratio` (grad cos 0.93, not 1.0).
- **Blackhole dispatch:** a lone-NCRISC program hangs dispatch — pair with a void BRISC. The 8-chip box is an ethernet fabric; a wedged dispatch needs a full `tt-smi -r 0 1 2 3 4 5 6 7`, not a single-board reset.
- **Watcher:** `TT_METAL_WATCHER` is for hang-debugging only; it inflates on-device time 1.5–2.6× — OFF for any timing.
