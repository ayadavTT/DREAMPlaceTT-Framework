# V35 Density-Backward on Tenstorrent — Handoff for a Fresh Agent

> **You are picking this up cold, with none of the prior chat history or memories.**
> This document is self-contained. Read it top to bottom before touching anything.
> **First action on a new board: run the "Is the board even alive?" check in §7 — do
> not skip it.** The previous environment was blocked by a board/firmware problem,
> not by this code.

---

## 1. The goal (what we are trying to achieve)

Run **DREAMPlace** (an analytical VLSI global placer) with its **density forward AND
backward** steps executing on **Tenstorrent Blackhole** hardware, such that:

1. It **converges** — the placement quality (HPWL / overflow per iteration) matches the
   CPU reference within noise.
2. It **beats the CPU** (16-thread) on the density components: **scatter, gather, DCT,
   and the backward (electric force)**, per the per-iteration baseline in
   `results/cpu16t_baseline_breakdown.csv`.
3. **No host involvement inside the per-iteration loop** (everything on-chip) — host
   round-trips are "residual" costs to optimize later, but the *compute* must be on TT.

The **forward** (chip scatter + chip DCT) already runs on TT and beats CPU. The open
problem this work targets is the **density backward (electric force)**, and the kernel
built for it is **V35**. The immediate deliverable the new agent should produce:
**live V35 convergence (HPWL) + per-op forward/backward-vs-CPU timing across the 18
configs** (adaptec1/2/3 × bigblue1/2/3 × {512,1024,2048}).

---

## 2. The math (density backward = electric force)

Per movable cell `c`, the density gradient ("electric force") is a small windowed,
overlap-weighted gather over the field maps `fx,fy`:

```
gx[c] = ratio[c] · Σ_{k<kc} Σ_{h<hc}  px[c,k] · py[c,h] · fx[(bxl[c]+k)·NBY + (byl[c]+h)]
gy[c] = ratio[c] · Σ_{k<kc} Σ_{h<hc}  px[c,k] · py[c,h] · fy[(bxl[c]+k)·NBY + (byl[c]+h)]
```

- `fx,fy` = electric field, two **column-major** `NBX×NBY` maps (the chip-DCT output).
  Index `[col·NBY + row]`, where `col = bxl` (x-bin), `row = byl` (y-bin).
- `px[c,·], py[c,·]` = the cell's fractional overlap with each bin column/row (≤8 each),
  computed in the forward (smooth "triangle" overlap).
- `bxl,byl` = first bin the cell touches; `kc,hc` = how many bins it spans (≤8).
- `ratio` = original_area / stretched(clamped)_area (small cells are stretched to
  ≥ bin·√2 for smooth density; ratio conserves charge). `ratio = node_area /
  (node_size_x_clamped · node_size_y_clamped)`.

**Why it's hard on TT:** the CPU has a shared cache, so any of 16 threads reads any
field bin for free and processes cells in natural order with **zero data movement**.
TT has only per-core L1 (no shared cache), so cells and the field bins they need do
*not* co-locate under any fixed partition — historically every approach paid a
"reconciliation tax" (move field to cells, or cells to field) that the CPU never pays
and that scales with `cells × span`. See `docs/DENSITY_BACKWARD_FAILURES.md`.

---

## 3. The V35 kernel — "Forward-Grouped Halo-Tile" backward

**One-line idea:** partition the field into **fixed-width column tiles** sized so one
tile (+ a `max_k` halo) always fits L1; assign cores to tiles **in proportion to each
tile's cell count**; compute the gather on the **SFPU**, 32 cells per instruction.

This clears all four taxes that killed the predecessors (V21/V28/V29/V30/FCCS — see
`DREAMPlaceTT/history/05-backward-ef-gather/` for each one's KERNEL.md + measured death):

| property | how V35 gets it |
|---|---|
| no NoC-per-bin latency | field tile is **L1-resident** → direct L1 load |
| cells stay **whole** (no record explosion) | the **halo** keeps a cell's full k×h window inside one tile |
| **no streaming** (FCCS's death) | the whole tile is resident, not streamed → **no sort needed** |
| **clustering-immune** | cores allocated ∝ cell count (greedy makespan); hot tiles get many helper cores, each replicating that tile's *small* band |
| **no `nc²` route_buf / OOM** (V29's death) | single-level grouping, O(ncells) memory |
| **MAC on SFPU** (FCCS's int-BRISC death) | reuses the validated `v28_compute.cpp` (32 cells/op) |

**The fully on-chip pipeline (no host per-cell work):**

```
PASS 1  v35_count        per-core histogram of its stash slice by tile = bxl/W
        (host plan)      read counts (O(nc·ntiles) ints) → tile_base[] + per-source
                         prefix[] + greedy proportional core allocation  ← tiny metadata
PASS 2  v35_place        local counting-sort each chunk by tile, BULK-write 128 B
                         records into one tile-grouped buffer (count-prefix, no atomics)
PASS 3  v35_gather_brisc gather field from the L1-resident halo'd tile, feed v28_compute
        + v28_compute    (SFPU Σ px·py·f ·ratio), v28_ncrisc drains gx/gy + writes oidx
        (writeback)      grad → grad_full[sel[oidx]] (accumulate; sub-cells share a node)
```

Inputs are **prep-free**: geometry comes from the forward's **V31_GEOM stash**
(128 B/cell: `[0]gidx [1]bxl_g [2]byl [3]kc [4]hc [5]ratio [6..13]px[8] [14..21]py[8]`,
floats); the field comes from the **chip-DCT output already in DRAM**
(`engine.latest_field_addrs()`).

---

## 4. What is BUILT and VALIDATED (and what is not)

### Validated on real silicon (standalone microbenches, synthetic cells) ✅
Both bit-exact vs the CPU multi-bin reference (`rel_l2 ≈ 7e-8`, `nbad=0`) at **all grids
and uniform / moderate / extreme clustering**:

- **Gather + balance** (`DREAMPlaceTT/host/v35_host.cpp`, target `v35`): beats CPU on
  **17/18** configs (parity at adaptec1_512), avg ~2×, up to **3.6× @ adaptec3_2048**,
  clustering-immune. Table: `DREAMPlaceTT/history/05-backward-ef-gather/v35-halotile/profile/sweep_18config.csv`.
- **Fully on-chip grouping** (`DREAMPlaceTT/host/v35live_host.cpp`, target `v35live`):
  the complete count→plan→place→gather. Bit-exact. **Beats CPU on 4/6 @2048**
  (a2 1.11×, a3 1.52×, bb1 1.27×, bb2 1.36×), parity a1_2048, loses at 512/1024 (CPU
  backward is cheap there so the fixed grouping overhead dominates) and bb3_2048 (1.1M:
  the place-sort + count scale with cells). Table:
  `.../v35-halotile/profile/onchip_grouping_live.csv`. The current limiter is the
  **on-chip grouping soft-compute** (count reads the full 128 B stash; place's local
  counting-sort copy), NOT the gather — see "next steps".

### Built but NOT yet run in the live DREAMPlace loop ⚠️
- **Live engine** `host/v35_ef_engine.{h,cpp}` (also copied into `DREAMPlaceTT/host/`):
  persistent-buffer port of the pipeline; reads the forward stash + chip-DCT field;
  returns grad. **Compiles clean.**
- **pybind**: `configure_electric_force_v35` + `compute_electric_force_v35_chip` in
  `host/v19_engine_pybind.cpp` (and `DREAMPlaceTT/host/...`).
- **client branch**: `V35_EF=1` in `integration/scatter_ttnn_client_inprocess.py`.
- **Convergence = bit-exactness so far**, i.e. *inferred*, **not yet observed live**.
  (FCCS, which uses the same forward-stash path with a coarser int grad, *does* converge
  live to HPWL 60.87M on adaptec1_512 — strong corroboration that V35 will.)

---

## 5. THE BLOCKER from the previous environment (read this — it cost a full session)

On the previous board, **the raw tt-metal program path hung — not just V35, but even
tt-metal's own hello-world example.** Diagnosis (with `gdb` on the hung process):
the host dispatched a program (`EnqueueMeshWorkload → finish`) and the completion-queue
reader thread waited forever in `completion_queue_wait_front` — **the device accepted
the dispatch but never signalled completion** (wedged dispatch). It survived `tt-smi -r`
twice. Root cause: the board firmware was **19.8.0** (put there by a co-tenant tt-xla
reservation), while the tt-metal build supports **≤19.5.0**; the early runs that worked
predated the firmware bump.

**Implication for you:** before assuming anything is broken in V35, confirm the board
runs *any* tt-metal kernel (§7). If even hello-world hangs, it's the board/firmware, not
this code — get a board whose firmware matches your tt-metal, or a tt-metal matching the
firmware. (Note: a *driver reload* `rmmod/modprobe tenstorrent` or a power cycle clears a
wedged dispatch that `tt-smi -r` cannot — but needs host root.)

---

## 6. Where everything lives (file map)

| what | path |
|---|---|
| V35 kernels | `DREAMPlaceTT/kernels/v35_count.cpp`, `v35_place.cpp`, `v35_gather_brisc.cpp` (+ reuses `v28_compute.cpp`, `v28_ncrisc.cpp`); copies also in framework `kernels/` |
| Standalone gather microbench | `DREAMPlaceTT/host/v35_host.cpp` (target `v35`) |
| Standalone on-chip-grouping microbench | `DREAMPlaceTT/host/v35live_host.cpp` (target `v35live`) |
| Standalone harness build | `DREAMPlaceTT/history/harness/{CMakeLists.txt,run_harnesses.sh}` |
| Live engine | `host/v35_ef_engine.{h,cpp}` and `DREAMPlaceTT/host/v35_ef_engine.{h,cpp}` |
| Live pybind | `host/v19_engine_pybind.cpp` (methods `*_electric_force_v35*`) |
| Live client branch | `integration/scatter_ttnn_client_inprocess.py` (`V35_EF` / `v35`) |
| Live DREAMPlace driver | `integration/run_dreamplace.py` |
| Benchmark configs | `benchmarks/configs/<design>_<grid>.json` |
| CPU baseline (the bar to beat) | `results/cpu16t_baseline_breakdown.csv` |
| Kernel "history library" (every prior approach, measured) | `DREAMPlaceTT/history/` (start at `LEADERBOARD.md`, `MANIFEST.csv`) |
| Why prior approaches failed / the design rationale | `docs/DENSITY_BACKWARD_FAILURES.md`, `docs/DENSITY_BACKWARD_NEXTGEN_DESIGN.md` |
| V35 KERNEL.md (8-section doc) | `DREAMPlaceTT/history/05-backward-ef-gather/v35-halotile/KERNEL.md` |

`tt-metal/` and `DREAMPlace/` are **git submodules** — after cloning, run
`git submodule update --init --recursive` (or use the ones in your reservation
container). `dp_env/` (a python3.8 venv) and the vendored copies under `DREAMPlaceTT/`
are **not** in git (too large) — see `REPRODUCE.md` to recreate them.

---

## 7. How to run (do these in order)

Set once (adjust paths to your clone + your reservation's tt-metal):
```bash
F=/path/to/DREAMPlaceTT-Framework
export TT_METAL_HOME=$F/tt-metal TT_METAL_RUNTIME_ROOT=$F/tt-metal
export TT_METAL_CACHE=/dev/shm/ttcache     # fast RAM cache → fast cold JIT
```

### 7a. IS THE BOARD ALIVE? (mandatory first check)
Run any prebuilt tt-metal example and confirm it COMPLETES (not just opens the device):
```bash
timeout 180 $TT_METAL_HOME/build_Release/ttnn/examples/example_lab_eltwise_binary
```
- Completes with a result → board is healthy, proceed.
- Hangs after "Device 0 sync complete" → **board/firmware is wedged or mismatched**
  (the §5 problem). Check `tt-smi -s | grep fw_bundle`; you need firmware compatible
  with your tt-metal. Fix the board before anything else.

### 7b. Prove V35 is correct + fast (standalone, no DREAMPlace needed)
```bash
cd $F/DREAMPlaceTT/history/harness && bash run_harnesses.sh build   # builds targets
B=$F/DREAMPlaceTT/history/harness/build
$B/v35     2048 2048 210904 6 3 0      # gather+balance: expect rel_l2 ~7e-8, "OK"
$B/v35live 2048 2048 210904 6 3 0      # full on-chip grouping: expect "OK"
# args: <NBX> <NBY> <ncells> <max_k> <max_h> <cluster:0=unif,1=mod,2=extreme>
```

### 7c. Live DREAMPlace with V35 backward (the deliverable)
Build the engine (python3.8 / dp_env):
```bash
cd $F/host/build && CXX=clang++-20 cmake .. -DCMAKE_CXX_COMPILER=clang++-20 \
  -DTT_METAL_HOME=$TT_METAL_HOME \
  -Dpybind11_DIR=$F/DREAMPlace/dp_env/lib/python3.8/site-packages/pybind11/share/cmake/pybind11 \
  -DPython3_EXECUTABLE=$F/DREAMPlace/dp_env/bin/python3
make v19_engine -j32
```
Run (adaptec1_512 first — reference HPWL ≈ 60.87M):
```bash
cd $F/DREAMPlace/install
V21_EF=1 V35_EF=1 V31_STASH=1 V31_GEOM=1 V31_EF_GEOM=1 V19_SKIP_CELL_SORT=1 V35_TIMING=1 \
  DREAMPLACE_DIR=$F/DREAMPlace/install \
  PYTHONPATH=$F/DREAMPlace/install:$F/host/build:$F/integration \
  $F/DREAMPlace/dp_env/bin/python3 $F/integration/run_dreamplace.py \
  --device scatter_ttnn_inprocess --benchmark $F/benchmarks/configs/adaptec1_512.json \
  --results-dir /tmp/v35_a1_512
```
- **Convergence:** look for `HPWL = ...` at the end (≈ 60.87M for adaptec1_512) and the
  per-iter `[evalmetrics]` overflow trend.
- **Backward timing:** `V35_TIMING=1` prints `[v35_timing] count=.. plan=.. place=..
  gather=.. d2h=.. total=..` every 20 iters. Compare `total` (and the components) to the
  adaptec1_512 backward row of `results/cpu16t_baseline_breakdown.csv`.
- Then sweep all 18 `benchmarks/configs/*` and build the V35-vs-CPU table.

**Critical env flags:** `V21_EF=1` installs the backward-patch hook; `V35_EF=1` selects
V35 inside it (you need BOTH). `V31_STASH=1 V31_GEOM=1 V31_EF_GEOM=1` make the forward
produce the per-cell geometry stash V35 reads. **Do NOT set `V31_GEOM_INT=1`** — V35
reads **float** px/py (that flag stashes int32; it's for FCCS). `V19_SKIP_CELL_SORT=1`
keeps stash index == active cell index.

---

## 8. Gotchas (hard-won — will save you hours)

- **Blackhole DRAM reads need 64-B aligned src+size** (writes 16-B). V35 records are
  **128 B** precisely so every record offset is 64-B aligned; 96 B records silently
  returned garbage for odd slices. Keep them 128 B.
- **JIT cache:** any rebuild of the engine changes the kernel-cache hash → the next run
  cold-compiles *all* forward kernels (slow). Use `TT_METAL_CACHE=/dev/shm/...` and
  **never SIGKILL a run mid-JIT** (corrupts/half-fills the cache). Give live runs a
  generous timeout (cold JIT + 100 iters can exceed 130 s).
- **Two-point logic for V35 vs FCCS:** V35 ⇒ `V31_GEOM=1` *without* `V31_GEOM_INT`.
- **grad accumulation:** big cells are sub-tiled into ≤8-bin sub-cells in the forward, so
  several sub-cells map to one node — the writeback **accumulates** `grad[sel[oidx]]`
  (engine zeroes the active entries first). `subcell_parent()` gives the sub-cell→parent
  map; configure with `sel[subcell_parent()]`.
- **Never `tt-smi -r` on a shared board without explicit owner permission** — it resets
  all PCI devices and kicks every session on that board.
- **`max_k/max_h` must cover the ORIG span** (orig = clamped + 2·|offset|), capped at 8
  — under-sizing drops bins; over-sizing balloons the gather (each extra k/h is a full
  1024-cell field-gather pass).

---

## 9. Next steps for you (in priority order)

1. **§7a board check**, then **§7b** to confirm V35 is correct on your board.
2. **§7c**: run V35 live on adaptec1_512 → confirm **HPWL convergence** (≈60.87M) and
   capture the `[v35_timing]` backward breakdown.
3. **Sweep all 18 configs**; build the V35-vs-CPU table (convergence + per-op timing).
   Compare against `results/cpu16t_baseline_breakdown.csv`.
4. **Optimize the grouping** (the live limiter, NOT the gather):
   - `count` reads the full 128 B stash just for `bxl` → have the forward also stash a
     compact `bxl[]` array (4 B/cell) → ~5× faster count.
   - `place`'s local counting-sort 32-word L1 copy dominates at high cell counts →
     speed it up or move to a wider copy.
   These two are what make V35 lose at 512/1024 and bb3_2048; fixing them should make it
   beat CPU broadly.
5. Longer term: fuse the on-chip optimizer (V22 Nesterov/clamp, already built — see
   `history/07-optimizer-nesterov/`) so grad never round-trips the host.

---

## 10. The numbers to expect (standalone, real silicon — your target)

V35 backward (prep+compute) vs CPU-16T per-iter backward (ms); >1 ⇒ V35 wins:

| grid | a1 | a2 | a3 | bb1 | bb2 | bb3 |
|---|---|---|---|---|---|---|
| 512  | ~par | 1.2× | 2.0× | 1.3× | 2.1× | 2.0× | (gather-only; full pipeline loses here) |
| 1024 | 1.3× | 1.9× | 2.4× | 1.9× | 2.3× | 2.0× | (gather-only) |
| 2048 | 1.0–2.1× | 2.3× | 3.6× | 2.8× | 3.4× | 2.3× | (gather-only) |

Full **on-chip** pipeline (count+place+gather) @2048 beats CPU on a2/a3/bb1/bb2, parity
a1, loses bb3 (see §4). The kernel/gather is solved; the grouping is the optimization
frontier. **Convergence: bit-exact grad ⇒ expected to match CPU HPWL exactly — confirm
it live, that's deliverable #2.**

---

*Generated at end of the build/validation session. Everything above the §5 blocker is
done and reproducible; §7c (live convergence + timing) is the one thing that needs a
healthy board to finish.*
