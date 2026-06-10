# Handoff: make the DREAMPlace forward+backward EXTREMELY fast on Tenstorrent Blackhole

**Last updated:** 2026-06-09 · **Branch:** `density-codesign-compact-stash64-dual` · **Device user:** ayadav@tenstorrent.com

> Your mission, in one sentence: **drive down the per-iteration runtime of the DREAMPlace density
> forward + backward on Blackhole by co-designing the forward and backward kernels together —
> aggressively, creatively, and from first principles about what the hardware is actually good at.**

Read this top-to-bottom. Then read, in order: `DREAMPlaceTT/PIPELINE.md` (the authoritative
pipeline map), `DREAMPlaceTT/history/README.md` + `LEADERBOARD.md` + `MANIFEST.csv` (the kernel
library), `DREAMPlaceTT/history/nohost-tracy-2026-06-09/PROFILE_ANALYSIS.md` (where the time goes
today), `docs/BLACKHOLE_HW_SPEC.md` + `docs/PERFORMANCE_MODEL.md`, and the persistent memory at
`/home/ayadav/.claude/projects/-localdev-ayadav-DREAMPlaceTT-Framework/memory/MEMORY.md`.

---

## 0. Mindset (read this first — it's the point of the role)

The correctness battle is **won**: the full density step + optimizer runs on-device, no host in the
loop, converges on 16/16 non-deferred configs (`NOHOST_OPTIMIZER.md`). Your job is **not** to add
features — it is to make it **fast**. That means:

- **Be bold. Reimplement entire stages or kernels from scratch if that's the right answer.** Nothing
  in `kernels/` is sacred. The current kernels are *robust and correct*, not *optimal*. If a clean
  re-architecture of scatter, or gather, or the whole forward→backward data path is faster, do it.
- **Co-optimize forward and backward as ONE problem.** This is the single most important lever and
  the explicit ask. The forward decides the *data layout* the backward consumes. The backward cost
  is largely set by *how the forward stashed its geometry/field*. Don't optimize a kernel in
  isolation — ask "what should the forward emit so the backward is trivial?" and vice-versa. The
  codebase already proves this works (V31 geom-stash made the backward prep-free; the 64B
  source-stash codesign made forward scatter *faster* AND backward count/place cheaper — see
  `stash64-codesign`, `compact-record-validated`).
- **Think about the hardware, not the algorithm.** The fastest kernel is the one that keeps the
  *strongest unit* saturated. Ask: which unit is this op's natural home — the SFPU (vector math),
  the FPU/matmul engine (dense linear algebra), the NoC (data movement / atomics), the dataflow
  RISCs (address generation / scatter)? Then **transform the problem to fit that unit** — without
  being wasteful (don't pad a sparse problem into a dense matmul if 95% is zeros; don't burn SFPU
  lanes on data movement).
- **Repurpose aggressively.** The biggest wins here came from using an API for something it wasn't
  "for": `noc_semaphore_inc` as a distributed float/int accumulator for the scatter (instead of a
  reduction tree); the matmul engine as the DCT solver; L1 as a sharded scatter target to dodge the
  DRAM 64-B cacheline RMW race. Hunt for more of these. Read the TT-Metal API surface asking "what
  else can this primitive do?"
- **Mine the history.** Every prior kernel generation is archived with its mechanism, its measured
  cost, its Tracy device-zone profile, and its gotchas/lessons. Several *fast* designs were shelved
  for correctness/robustness reasons (e.g. v25 EF gather at **17× CPU**, v11 hash-scatter) — their
  *speed traits* may be reusable now that the surrounding pipeline is different. Read their
  `profile/` and `KERNEL.md §7 (lessons)`.

---

## 1. Where the time goes TODAY (your targets)

Per-iteration host-wall breakdown, the largest design `bigblue3` (full table:
`DREAMPlaceTT/history/nohost-tracy-2026-06-09/PROFILE_ANALYSIS.md`; regen with `tools/nohost_timing.sh`).
Steps sum to each op's total; ms/iter.

**Forward (density EP op):**

| step | bb3_512 | bb3_1024 | bb3_2048 | what it is | bound by |
|---|--:|--:|--:|---|---|
| **scatter** | **23.4** | **26.5** | **32.8** | uint32 atomic density scatter (cell→bins) | DRAM atomics / cell count — **#1 cost** |
| writeout | 0.2 | 0.7 | 2.5 | fp32 density tile writeout | DRAM write |
| dct | 1.1 | 1.6 | 2.8 | TTNN matmul DCT solve | matmul (cheap, well-tuned) |
| dct_up / field_d2h | 0.3 | 0.4 | 0.9 | density→TTNN tensor / field readback | small transfers |
| fold+resid | 0.9 | 1.0 | 0.8 | on-device ttnn::add of initial density (untimed) | device |
| field_wrap | 4.0 | 6.0 | 4.0 | **HOST px/py permute** (`px[i]=pos[ix[i]]+ox[i]`) | host — movable on-device (V20 permute kernel exists) |
| h2d / prep / ctx | 2.8 | 4.6 | 2.8 | px/py upload + numpy prep + autograd ctx | transfer/host |
| **FWD total** | **32.6** | **41.0** | **46.6** | | |

**Backward (density EP op):**

| step | bb3_512 | bb3_1024 | bb3_2048 | what it is | bound by |
|---|--:|--:|--:|---|---|
| **gather** | **7.4** | **8.7** | **10.2** | SFPU multi-bin electric-force sample | **field-fill DATA MOVEMENT (~90%), not SFPU math** |
| **place** | **6.4** | **6.8** | **6.8** | counting-sort + bulk DRAM writes (cell→tile-group) | **DRAM WRITE-bound (~49% scattered writes)** |
| **unsort** (`d2h`) | **6.8** | **6.8** | **8.5** | on-chip unsort (grad→node-order, 3 device passes) | **SINGLE-RISC** (dualize → ~2×) |
| count | 2.4 | 2.4 | 2.4 | per-core tile histogram | DRAM read |
| plan / pybind / scale / zero | 3.2 | 3.5 | 3.0 | host metadata + marshalling + grad scale | host (small) |
| **BWD total** | **26.4** | **28.3** | **31.1** | | |

**Density step (fwd+bwd): 59 / 69 / 78 ms.** Plus the uninstrumented per-iter remainder (DreamPlace
wirelength op, the V22 optimizer's ~8 SFPU launches, metrics) — instrument these if you want the true
iteration wall (see PROFILE_ANALYSIS §caveats).

**The headline insight that should drive you:** the dominant stages are **data-movement-bound, not
compute-bound.** Forward scatter is DRAM atomics. Gather's SFPU math is ~1.5 µs but is *starved* by
~22 µs of field-fill per group (`gather-place-finegrain-profile`). Place is DRAM-write-bound
(`place-deepdive-profile`). So the wins are in **moving less data, moving it in friendlier shapes,
and keeping the right unit saturated** — almost never in "faster math."

---

## 2. The co-optimization thesis (the core ask)

**The forward and backward share one artifact: the per-cell geometry/field data.** Today the forward
(`v31_scatter_stash_dm`) stashes a 128-B-per-cell geometry record + the density map; the backward
re-reads that stash, re-derives each cell's bin footprint, *re-loads the field band per group*
(the gather field-fill that dominates), and counts/places/gathers/unsorts. **Most of the backward's
cost is re-deriving and re-fetching things the forward already had in hand.**

Co-optimization questions to attack (non-exhaustive — be creative):
- **What format should the forward emit so the backward never re-loads the field?** The gather is
  ~90% field-fill. If the forward (which already touches the field via the DCT) emitted the field
  *already gathered/grouped per cell*, or in a tile layout the backward streams once, the gather
  collapses. (`field-cast`/FCCS is one prior attempt — clustering-immune, bucketing-free; read it.)
- **Can count + place be folded into the forward?** `place-deepdive-profile` notes: "fold grouping
  into the forward → place negligible." The forward already visits every cell; if it emitted cells
  *already grouped by tile/field-region*, the backward's count+place (≈9 ms at bb3) largely vanishes.
- **Can the unsort be eliminated by emitting node-order from the start?** The unsort exists only
  because the gather output is in gather/slot order. If the forward stash recorded node identity in
  a layout that lets the gather write node-order directly (or the optimizer consume slot-order), the
  3-pass on-chip unsort (≈8 ms) disappears. (See `onchip-unsort-64b-reliable` for why it's structured
  as it is — the 64-B cacheline race, the L1-shard trick, the sel-duplicate accumulate.)
- **Source-stash codesign already won once:** the 64-B source stash made the forward scatter
  *−15% (faster!)* AND backward count −46% / place −30%, bit-exact (`stash64-codesign`,
  `compact-record-validated`). That is the template — find the next such joint record format.

---

## 3. The hardware — and the doctrine of "transform the problem to fit the strongest unit"

Blackhole p150b: 13×10 = **130 Tensix cores**, each with **BRISC + NCRISC** (dataflow RISCs, run the
two NoCs in parallel), **3× TRISC** (the compute pipe: unpack/math/pack feeding the **SFPU** vector
engine + the **FPU/matmul** engine), and a **1.5 MB L1** bank. Full spec: `docs/BLACKHOLE_HW_SPEC.md`;
cost model: `docs/PERFORMANCE_MODEL.md`.

Unit strengths and how each op might be re-cast onto them:
- **FPU / matmul engine** — by far the highest FLOP/s. It's *idle* in scatter/gather/place/unsort
  today. **Can any of these be re-expressed as a matmul?** The DCT already is (TTNN matmul,
  cheap+well-tuned — proof the engine is fast here). A density scatter is `cell × bin` incidence ×
  area → that's a sparse matmul; a gather is `field[bin] · weight[cell,bin]` → a sparse matvec. The
  trap is sparsity (don't densify a 99%-zero incidence). But block-structured / banded formulations,
  or batching the *dense inner* footprint (each cell touches ≤ k×h ≤ 8×8 bins) as small dense tiles,
  could put real work on the matmul engine. Investigate `docs/V13_FPU_UNLEASH_HANDOFF.md` (a prior
  FPU-matmul scatter that converged) and `history/99-infra-probes/v13-*`.
- **SFPU (vector)** — great for the per-element optimizer (V22) and the EF weighting math. But it's
  *starved* in gather (data-movement-bound). Make it SFPU-friendly only when you can keep its lanes
  fed — i.e., when the data is already tiled and resident. `docs/V21A_SFPU_HANDOFF.md` profiled the
  EF SFPU port; read it for what's actually fast vs starved.
- **NoC + dataflow RISCs** — the data-movement workhorses, and where the dominant cost actually is.
  `noc_semaphore_inc` as an accumulator (the scatter's atomic-add) is the canonical "repurpose a
  primitive" win — `history/99-infra-probes/atomic-bench` measured it. Levers: **dual-RISC** (both
  NoCs issuing — proven ~2× for issue-bound ops like the L1 unsort, only ~15% for DRAM-bandwidth-bound
  ones — `dual-risc-count-place`, `onchip-unsort-64b-reliable`), multicast (`mcast-bw` probe),
  64-B-aligned coalesced access (the DRAM cacheline rule — sub-64B scattered writes RMW-race and
  silently drop data), and L1-resident sharding to dodge DRAM entirely.
- **L1 (1.5 MB/core)** — the fast scratchpad. The unsort scatters into L1 node-shards (16-B-aligned,
  no RMW race) instead of DRAM. Field bands are held L1-resident in the gather. Ask: what else can be
  made L1-resident to kill a DRAM round-trip? (Watch the L1 budget at bb3 scale — CBs + shards coexist.)
- **Format conversion as a weapon.** Several wins came from *changing the data representation* to suit
  the hardware: uint32 fixed-point density (atomic-add friendly) instead of fp32; the 64-B grouped
  record (one coalesced read instead of scattered fields); interleaved vs tile layouts. **Reshaping
  the input/output to exploit a unit's natural access pattern is often bigger than any kernel tweak.**

---

## 4. Proven innovation patterns in THIS codebase (your inspiration set)

These already shipped and are measured — study *why* they were fast and look for the next one:
- **`noc_semaphore_inc` distributed accumulation** → the scatter's atomic density add (no reduction
  tree). `history/99-infra-probes/atomic-bench`.
- **Matmul-as-DCT** → the DCT solve is TTNN tilize→matmul→untilize; ~0.7–1.1 ms, beats CPU.
  `history/08-dct-solve`.
- **64-B source-stash forward/backward codesign** → forward scatter faster + backward cheaper,
  bit-exact. `stash64-codesign`, `compact-record-validated`.
- **L1-target unsort (dodge the DRAM 64-B cacheline RMW race)** → 9.5 ms dual-RISC vs 40 ms CPU,
  bit-exact; virtual coords are NoC-independent so dual-RISC "just works".
  `onchip-unsort-64b-reliable`.
- **Field-cast (FCCS) gather** → clustering-immune, bucketing-free EF. `history/05-backward-ef-gather`.
- **Halo-tile field band** → the gather loads a W+max_k halo so most cells read their full footprint
  from one band. (Its field-fill is still the bottleneck — the next target.)
- **Dual-RISC count/place + per-batch footprint** → ~15–30%, bit-exact. `dual-risc-count-place`,
  `lever-c-per-batch-footprint-validated`.

---

## 5. How to use the history archive + Tracy profiles (do this constantly)

`DREAMPlaceTT/history/` is a **systematic kernel library** built exactly for you:
1. **`MANIFEST.csv`** — grep for "best known approach for stage X + its cost + does it have a profile".
2. **`LEADERBOARD.md`** — per-stage ranking with the one-line trick + when-to-use. (e.g. v25 EF
   gather = **0.42 ms ≈ 17× CPU** but 4-corner; v28 = correct-at-all-grids default; v27-bucket =
   0.79 ms cell-level atomic-free.) **Several fast designs were shelved for correctness/robustness,
   not speed — re-examine whether their speed trait is now reusable.**
3. **`<stage>/<kernel>/KERNEL.md`** — 8-section doc; read **§6 gotchas** and **§7 lessons** before
   reusing — they encode Blackhole traps (alignment, cacheline race, CB-wrap, route_buf overflow).
4. **`<stage>/<kernel>/profile/`** — `profile_log_device.csv` (raw Tracy device zones), `zones.txt`
   (per-zone table), `PROFILE.md` (interpretation). **This is the per-op compute-vs-data-movement
   ground truth.** Use it to confirm *which sub-zone* dominates before you optimize (e.g. the gather's
   `GB-FXFILL` field-fill vs `GB-LOOP` math).
5. **`99-infra-probes/`** — microbenches that informed design: `atomic-bench` (semaphore accum),
   `mcast-bw` (multicast bandwidth), `v20-permute` (chip-side pos permute — *the kernel to wire in to
   kill the host `field_wrap`*), `v13-fpu-*` (FPU-matmul scatter), `v16-probes`.

**To capture a FRESH Tracy device-zone profile** (the current build streams real-time Tracy; the
post-mortem CSV needs the procedure in `DREAMPlaceTT/history/nohost-tracy-2026-06-09/README.md`): the
live kernels are already instrumented (`DeviceZoneScopedN`, 68 zones listed in `our_device_zones.txt`).
Use `integration/process_tt_profile.py` to summarize. Build tt-metal post-mortem-profiler-enabled if
you need the per-zone ns table, then run a short config (`benchmarks/configs/prof_adaptec1_2048.json`,
20 iters) with `TT_METAL_DEVICE_PROFILER=1 TT_PROFILE_DUMP=1`.

---

## 6. Research mandate (spend real time here before writing kernels)

You are explicitly asked to **research widely and steal good ideas**:
- **Read the TT-Metal API docs and headers** (`tt-metal/tt_metal/...`, `dataflow_api.h`,
  `compute_kernel_api/...`, the SFPU/`sfpi` headers under `tt-sfpu/`). For every primitive ask "what
  *else* could this do for scatter/gather/place/unsort?" (semaphores, async NoC, multicast, gather/
  scatter DMA, tilize/untilize, reduce, the matmul LLKs).
- **Read fast, production-grade tt-metal kernels** — the matmul, reduce, all-gather/CCL, and
  data-movement examples in `tt-metal/` are heavily tuned. Read *how* they keep the engine/NoC
  saturated (double-buffering, CB sizing, mcast, work decomposition) and **map those techniques onto
  our ops.** The all-gather/reduce-scatter patterns are directly relevant to the unsort and the
  density accumulate.
- **Relate everything back to DREAMPlace's ops.** scatter = sparse cell→bin area accumulation;
  gather = field sampling weighted by cell footprint; place = group-by-tile; unsort = inverse
  permutation/scatter; DCT = matmul. For each, find the closest well-optimized tt-metal primitive and
  ask whether a reformulation lets us use it.
- **Read `docs/PERFORMANCE_MODEL.md`, `docs/RUNTIME_BREAKDOWN.md`, `docs/PROFILING.md`,
  `docs/DENSITY_PLANNING_APPROACHES.md`** — they already reason about roofline / where the model says
  time *should* go vs where it goes.

---

## 7. Creative seed directions (NOT prescriptive — starting points to expand on)

- **Forward scatter (#1 cost):** it's DRAM-atomic-bound. Can it be L1-resident (per-core density
  shard) + a cheap reduce, dodging DRAM atomics? Can the matmul engine do the cell→bin area spreading
  for the dense ≤8×8 footprint? Can `noc_semaphore`-accum be coalesced to 64-B lines?
- **Gather (field-fill-bound):** make the field *come to the cells* in a layout that's a single
  streamed read (field-stationary / pre-grouped by the forward), so the SFPU stops starving. Or fold
  the gather into the DCT's untilize (the field is already in tiles there).
- **Place + count (DRAM-write-bound):** fold grouping into the forward (the forward already iterates
  cells in an order — emit them tile-grouped). Double-buffer the scattered writes.
- **Unsort (single-RISC):** dualize it (the proven ~2×); or eliminate it by emitting node-order from
  the gather; or have the optimizer consume slot-order directly (V22 is layout-agnostic).
- **Optimizer (V22, ~8 launches/iter):** fuse the combine→precond→nesterov→clamp + the diffs +
  stepsize into 1–2 launches (dispatch-bound, trivial compute).
- **field_wrap (host):** wire the `v20-permute` kernel into the forward so px/py are built on-device
  (kills the 4–6 ms host permute). Convergence-safe (bit-identical px/py) — see `NOHOST_OPTIMIZER.md §6`.

---

## 8. Build / run / profile / validate (the recipe + the device gotchas)

**Container/device:** host `bh-37`, container `bh-37-special-ayadav-for-reservation-82536` (docker,
shared multi-tenant — other tenants have their own boards; do NOT `tt-smi -r` without checking).
**Use `DPTT_DEVID=1` = PCI Dev 3 = the healthy board** (Dev 2 = logical 0 = FAULTY). `F=` the repo root.

**Build the pybind** (after editing `host/*.cpp`; Ninja, not make):
```bash
C=bh-37-special-ayadav-for-reservation-82536
docker exec "$C" bash -lc 'export TT_METAL_HOME=/localdev/ayadav/DREAMPlaceTT-Framework/tt-metal \
  TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME ARCH_NAME=blackhole; \
  cd /localdev/ayadav/DREAMPlaceTT-Framework/DREAMPlaceTT/host/build && ninja v19_engine'
```
Kernels (`kernels/*.cpp`) are **JIT-compiled at runtime** — no rebuild for a kernel-only edit, but use
a **fresh `TT_METAL_CACHE=/dev/shm/<unique>`** or the JIT serves a stale cached kernel.

**Run one config** (no-host loop = the current end state):
```bash
F=/localdev/ayadav/DREAMPlaceTT-Framework
docker exec "$C" bash -lc "
  export TT_METAL_HOME=$F/tt-metal TT_METAL_RUNTIME_ROOT=$F/tt-metal ARCH_NAME=blackhole TT_METAL_CACHE=/dev/shm/ttc
  export DREAMPLACE_DIR=$F/DREAMPlace/install PYTHONPATH=$F/DREAMPlace/install:$F/DREAMPlaceTT/host/build:$F/integration
  export DPTT_DEVID=1 V22_NOHOST=drive
  $F/DREAMPlace/dp_env/bin/python3 $F/integration/run_dreamplace.py --device scatter_ttnn_inprocess \
    --benchmark $F/benchmarks/configs/sweep_bigblue3_2048.json --results-dir /tmp/r 2>&1 | tail -40"
```
`FWD_TIMING`/`V35_TIMING` are forced on by the production lock → every run emits the `[fwd_full]` /
`[v35_timing]` / `[bwd_full]` per-stage lines (the breakdown in §1). **For pure speed iteration**,
the isolated micro-harnesses (`DREAMPlaceTT/history/harness/`, `ninja <target>`, e.g. `v35dual`,
`v35compact`, `v25_ef_l1gather`, `v22e2e`) run a single stage on synthetic data with device-median
timing + correctness check — much faster than the full DREAMPlace loop. Build/run via the harness
CMake (Ninja). **Watcher:** `TT_METAL_WATCHER=10` ON when debugging a hang/wedge (`tt-smi -r 3` +
`docker restart` recovers a soft-wedge); OFF for timing (it inflates device time 1.5–2.6×).

**Per-config timing table:** `tools/nohost_timing.sh` (greps the `.nhsw_*.log` run logs).

**Sweep all 18:** `.verifymatrix_nohost.sh` (per-config container restart + retry; `V22_NOHOST=drive`).

## 9. Guardrails — you MUST NOT break these
- **Convergence is the contract.** Any kernel change must still converge the **16 non-deferred
  configs** to overflow ≤ ~0.07 (`.verifymatrix_nohost.sh`). `adaptec3_2048` & `bigblue2_2048` are
  pre-existing deferred stalls (upstream, not your kernels — see `ONCHIP_OPTIMIZER_HANDOFF.md §8`).
- **Validate bit-exact first, in isolation** (the harness gives rel_l2 / nbad vs a CPU reference)
  before wiring into the live loop. Many fast designs *silently* corrupted data (the 64-B cacheline
  race dropped ~half the writes while looking 8× faster — `onchip-unsort-64b-reliable`). A fast wrong
  kernel is worthless; prove correctness, then chase speed.
- **The `max_contrib > L1 cap` warning is a RED HERRING** (legacy V6/V9, fires on passing designs).
- **Always run with `TT_METAL_WATCHER` when debugging**; the board soft-wedges and a wedge is only
  diagnosable with the watcher log.
- **Measure on a clean, uncontended board** (other tenants share the host) and report
  device-median, watcher-off numbers.

---

## 10. Suggested first moves
1. Reproduce the §1 profile on bigblue3 (run + `tools/nohost_timing.sh`) so you trust the baseline.
2. Read `LEADERBOARD.md` + the `profile/` of the current scatter (`01-forward-scatter`), gather
   (`05-backward-ef-gather`), place (`04-backward-bucketing`), unsort, and the `stash64`/`compact`
   codesign notes — internalize *why* each is the cost it is.
3. Pick the biggest co-optimization lever (forward emits backward-friendly grouped/field data) and
   prototype it in an isolated harness against a CPU reference before touching the live pipeline.
4. Quick wins to bank early: wire `v20-permute` (kills host `field_wrap`), dualize the on-chip unsort
   (~2×), fuse the V22 launches.

Be bold, be creative, measure everything, never break convergence.
