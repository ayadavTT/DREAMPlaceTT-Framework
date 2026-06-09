# Handoff: On-chip Density Step + Optimizer on Tenstorrent Blackhole

**Last updated:** 2026-06-08 · **Branch:** `density-codesign-compact-stash64-dual` · **Device user:** ayadav@tenstorrent.com

> Read this top-to-bottom before touching code. Then read the persistent memory at
> `/home/ayadav/.claude/projects/-localdev-ayadav-DREAMPlaceTT-Framework/memory/MEMORY.md`
> (one-line index) and especially `onchip-unsort-64b-reliable.md` (the deep notes this doc summarizes).

---

## 1. The Goal

Run the DREAMPlace **density step on Tenstorrent Blackhole (p150b)** end-to-end with **no host in the loop** — i.e. the density forward (scatter→DCT→field), the backward density gradient, the **unsort** (gather-order → node-order grad), and the **optimizer position update** all execute on-device, with the gradient and positions never round-tripping to the host for *processing*. The objective is to **beat, or get as close as possible to, the CPU's forward+backward+optimizer per-iteration time**, while preserving DREAMPlace convergence.

Concretely, the end state is: **forward (TT) → backward density grad (TT) → on-chip unsort → V22 optimizer (combine→precond→nesterov→clamp) → updated positions resident on-device → forward reads them directly**, replacing DREAMPlace's CPU `optimizer.step()`. Success = **all 18 benchmark configs converge** (overflow ≤ ~0.07) bit-comparably to the CPU path.

(Allowed host transfers: the **wirelength gradient** is DREAMPlace's own operator — uploading it to the device is a data transfer, not "processing". DREAMPlace's metrics/density-weight schedule remain DREAMPlace's. Everything *we* compute — unsort, combine, precond, nesterov, clamp, BB stepsize — must be on-device.)

---

## 2. Current State (what works, what's left)

### DONE + VALIDATED
- **Production forward + V35 backward** is the live pipeline (uint32 scatter + TTNN DCT + V35 count→place→gather→**CPU unsort**). Converges on **16/18** configs. (The 2 failures — `adaptec3_2048`, `bigblue2_2048` — are a **separate, deferred** issue; see §8.)
- **On-chip unsort → V22 `b_dg` layout: COMPLETE, bit-exact, fully on-device.** `kernels/v35_onchip_unsort.cpp` + the `V35_ONCHIP_UNSORT`-gated block in `host/v35_ef_engine.cpp` produce V22's interleaved `density_grad` buffer (`b_dg`) on-device from the live gather output, validated **0 mismatches vs the CPU unsort at every iteration**, both resolutions (`adaptec1_512` maxsub=1; `adaptec1_2048` maxsub=3). No host node-mapping or unsort processing. **Gated off by default → production untouched.**
- **V22 optimizer engine** (`host/v22_engine.{h,cpp}`): the four ePlace SFPU kernels (combine→precond→nesterov→clamp), made **interleaved-internal**, with `step_resident_ondevice_grad()` (consumes `b_dg`, no host grad upload) + `density_grad_address()`/`density_grad_num_tiles()`. **e2e-validated** by `v22e2e` harness: `unsort → b_dg → V22 → get_pos` matches a host Nesterov reference to fp32 (max_rel 3.6e-5, 0 out-of-tol @8K/131K/371K).

### ✅ DONE 2026-06-09 — the remaining integration is COMPLETE (see `DREAMPlaceTT/NOHOST_OPTIMIZER.md`)
The "NOT YET DONE" items below are all done; the full no-host loop converges on 16/16 non-deferred configs. Superseded by `DREAMPlaceTT/NOHOST_OPTIMIZER.md` (architecture + 18-config results + per-config timing). Summary:
- **V22 IS wired into the live DREAMPlace loop** — built into the `v19_engine` pybind; driven via `V22_NOHOST=drive` (or `V22_OPT=shadow|drive` for the with-host variant).
- **Step-size (α) IS computed on-device** — the line-search `α=sqrt(Σ(Δpos)²/Σ(Δgrad)²)` reduction runs on device via the V22 stepsize kernel over resident pos/grad diffs. (BB itself diverges in this DreamPlace setup → the no-host loop correctly uses the nobb LINE SEARCH, not step_bb.)
- **The density gradient never touches host** — on-chip unsort writes it into V22's `b_dg`; V22 does combine→precond→nesterov→clamp on-device. CPU unsort elided.
- **The Nesterov optimizer IS replaced** — `step_nobb`/`step_bb` patched to the V22 no-host line search (`_install_v22_optimizer` / `_v22_nohost_step_nobb` in `integration/scatter_ttnn_client_inprocess.py`).
- **18-config convergence VALIDATED** — `.verifymatrix_nohost.sh`: 16/16 non-deferred converge ≤0.07 (attempt 1, 0 retries); the 2 deferred (adaptec3_2048, bigblue2_2048) match the host baseline (§8).
- **Resident pos → forward (handoff §7 Step C): DEFERRED** — would not remove the pos round-trip (DreamPlace's host wirelength operator needs host pos anyway); only saves the density-forward px/py h2d via an invasive locked-forward change. Low value; see NOHOST_OPTIMIZER.md §6.
- A key bug found+fixed along the way: a shared-host-staging-buffer race in the V22 upload path (intermittent large-config blow-ups) → made the writes blocking.

---

## 3. Hardware, Container, How to Build & Run

### Device / container
- **Host:** bh-37, 8× Blackhole p150b on a shared multi-tenant host.
- **Container:** `bh-37-special-ayadav-for-reservation-82536` (docker). All builds/runs happen *inside* it via `docker exec`.
- **Use `DPTT_DEVID=1`** (live pybind) / `V35_DEVID=1` (harness) = logical chip 1 = **PCI Dev 3 (healthy)**. **PCI Dev 2 = logical chip 0 = FAULTY** (wrong numerics, hpwl=-inf) — never use it.
- `F=/localdev/ayadav/DREAMPlaceTT-Framework` (repo root, mounted into the container at the same path).

### Build the production pybind (after editing `host/*.cpp` or `kernels/*.cpp`)
```bash
C=bh-37-special-ayadav-for-reservation-82536
docker exec "$C" bash -lc '
  export TT_METAL_HOME=/localdev/ayadav/DREAMPlaceTT-Framework/tt-metal
  export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME ARCH_NAME=blackhole
  cd /localdev/ayadav/DREAMPlaceTT-Framework/DREAMPlaceTT/host/build && ninja v19_engine'
```
> The build system is **Ninja, not Make** (both prod `host/build` and `history/harness/build`). `make` will fail.
> Kernels (`kernels/*.cpp`) are **JIT-compiled at runtime** — no host rebuild needed for a kernel-only change, BUT use a **fresh `TT_METAL_CACHE`** dir or the JIT may serve a stale cached kernel.

### Run one benchmark config (live pipeline)
```bash
C=bh-37-special-ayadav-for-reservation-82536
F=/localdev/ayadav/DREAMPlaceTT-Framework
docker exec "$C" bash -lc "
  export TT_METAL_HOME=$F/tt-metal TT_METAL_RUNTIME_ROOT=$F/tt-metal ARCH_NAME=blackhole TT_METAL_CACHE=/dev/shm/ttcache
  export DREAMPLACE_DIR=$F/DREAMPlace/install PYTHONPATH=$F/DREAMPlace/install:$F/DREAMPlaceTT/host/build:$F/integration
  export DPTT_DEVID=1
  ps -eo pid,cmd | grep dp_env/bin/python3 | grep -v grep | awk '{print \$1}' | xargs -r kill -9; sleep 1
  timeout 300 $F/DREAMPlace/dp_env/bin/python3 $F/integration/run_dreamplace.py \
    --device scatter_ttnn_inprocess --benchmark $F/benchmarks/configs/sweep_adaptec1_512.json \
    --results-dir /tmp/run_a1 2>&1 | tail -40"
```
Convergence signal: a final `Overflow = <value>` line (≤ ~0.07 = converged).

### Run all 18 configs (convergence sweep)
`F/.verifymatrix.sh` runs the 18 configs (adaptec1/2/3 × bigblue1/2/3 × 512/1024/2048) with **no V35_* envs** (shipped defaults) and writes `F/.verifymatrix_summary.txt` (overflow + per-stage timing per config). **Watcher OFF** in that script (timing). Edit it to add `V35_ONCHIP_UNSORT=1` etc. when validating the on-chip path.

### Harness builds/runs (isolated microbenches)
```bash
docker exec "$C" bash -lc '
  export TT_METAL_HOME=/localdev/ayadav/DREAMPlaceTT-Framework/tt-metal TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME
  cd /localdev/ayadav/DREAMPlaceTT-Framework/DREAMPlaceTT/history/harness/build
  ninja v22e2e           # build a harness target
  V35_DEVID=1 TT_METAL_WATCHER=10 ./v22e2e 131072'
```
Useful harnesses: `v22e2e` (unsort→V22→pos e2e), `v35unsortl1full` (unsort→b_dg standalone), `v35unsortl1` (L1-scatter reliability/cost).

### Diagnostic env flags (live pipeline)
- `V35_ONCHIP_UNSORT=1` — run the on-chip unsort each backward iter and **validate it bit-exact vs the CPU unsort** (prints `[v35_onchip_unsort] vs CPU unsort: mismatches=N/M ... OK`). This is your correctness gate.
- `V35_ONCHIP_DBG=1` — dump the first ~10 mismatching nodes (g, k, h, exp vs got).
- `V35_COLLIDE=1` — collision/footprint structure (slots/node, sel-duplicate stats). **Note:** the per-iter `V35_ONCHIP_UNSORT` compare is ground truth; `V35_COLLIDE` is a one-shot snapshot and once misled us.
- `V35_FOOTPRINT=1` — per-cell (k,h) histogram + CDF (one-shot).
- `TT_METAL_WATCHER=10` — device watcher (ON when debugging a hang/wedge; **OFF for timing** — it inflates on-device time 1.5–2.6×).

---

## 4. Architecture / Data Flow

```
FORWARD (TT):  pos → scatter(uint32 density) → writeout(fp32) → TTNN DCT → field (resident on device)
BACKWARD (TT, v35_ef_engine.cpp compute_from_chip):
   count → plan(host, cheap) → place → gather  →  per-slot (gx,gy) in gxb/gyb + oidx in oibb (device, page-per-core)
   CPU unsort (CURRENT):  d2h gxb/gyb/oibb → grad[sel[oidx]] += (gx,gy)  → grad to DREAMPlace
   ON-CHIP unsort (NEW, gated V35_ONCHIP_UNSORT): scatter gxb/gyb→b_dg interleaved on-device (see §5)
OPTIMIZER (CURRENT = DREAMPlace CPU):  obj_and_grad_fn (combine wl+density, precond) → step_bb (BB stepsize, nesterov, clamp) → new pos
OPTIMIZER (TARGET = V22 on-device):  combine→precond→nesterov→clamp consuming b_dg + raw wl_grad, pos resident
```

Key facts about the gather output (what the unsort consumes):
- `gxb`, `gyb`, `oibb` are **page-per-core** MeshBuffers (page `c` = core `c`'s slots, stride `I.mc` floats; valid slots = `I.g_n[c]`).
- `oidx[slot]` indexes into **`sel`** (the active-node array). `node = sel[oidx]`. **`sel` is NOT injective** — a large cell is split into multiple *consecutive* active entries (`oi`) sharing one node; the CPU sums their partial gradients (`grad[sel[oi]] += `). The on-chip unsort must replicate this accumulate (it does — see §5).
- `sel` is **constant across iterations** (the active-node set never changes — only positions/grouping do). So it's uploaded once.

---

## 5. The On-chip Unsort (the validated piece) — how it works

**Files:** `kernels/v35_onchip_unsort.cpp` (kernel) + `host/v35_ef_engine.cpp` (the `if(getenv("V35_ONCHIP_UNSORT"))` block in `compute_from_chip`, ~line 334).

**b_dg layout (= V22's `density_grad`):** interleaved `[gx0,gy0,gx1,gy1,...]`, length `2*num_nodes`, tile-interleaved DRAM (page = one 1024-fp32 tile, `noc_async_write_tile`).

**3 device passes (same program, shard CB persists in L1 across launches; `Finish` between passes = the cross-core barrier):**
- **phase 2 ZERO:** each core zeroes its own L1 shard.
- **phase 0 SCATTER:** each core reads its gather slots (`gxb/gyb/oidx` page c) + decodes `node`/`sub` from the **resident packed `sel`** (`sel_u32[oi] = (sub<<24)|node`), and writes each slot's `[gx,gy,0,0]` (16B, 16B-aligned) into `shard[(node_local*maxsub + sub)*16]` on the **owning** core (`owner = node/epc`).
- **phase 1 FLUSH:** each owner core **sums the `maxsub` sub-slots per node** (float add — works on the dataflow RISC), compacts 16B→8B interleaved, and bulk-writes its whole tiles to `b_dg`.

**Why the design is the way it is (hard-won):**
- DRAM scattered writes need **full 64B cacheline** writes (sub-64B concurrent writes RMW-race and silently drop ~half) → so the unsort targets **L1** (SRAM, no cacheline RMW), then flushes contiguous tiles to DRAM.
- L1 NoC writes require **16B-aligned src+dst** → each node slot is ≥16B-aligned, write a full 16B.
- `node = sel[oidx]` is resolved **on-device** via a 64B-aligned read of the resident `sel` (page=64B, `sel[oi]=page(oi>>4)[oi&15]`), chunked (CHUNK=256) to overlap latency. Per-slot 4B random `sel` reads are illegal (64B read alignment).
- **`sel`-duplicate accumulate** (the wide/big cells): pack the per-`oi` sub-index (rank among consecutive same-node entries) into `sel`'s top byte; route each partial to a distinct sub-slot; flush sums. `maxsub` = max node multiplicity (computed from `sel` at build; 3 @adaptec1_2048). Shard = `epc*maxsub*16` bytes.
- Blackhole **virtual coords are NoC-independent** (`worker_core_from_logical_core`), so the same coord table works for both NoCs (dual-RISC).

**Validated cost (standalone harness, dual-RISC, bb3 2.04M):** ~10.5 ms (scatter 9.6 + flush 0.9) for the 1-write interleaved version — ~4× better than the 40 ms CPU d2h+unsort it replaces. (The in-engine on-chip path is single-RISC currently; dualize when wiring for perf.)

**Kernel arg order** (`SetRuntimeArgs`): `phase, my_core, g_n, gx_base, gy_base, oi_base, out_pg, coord_base, coord_pg, nc, epc, two_nn, my_tile0, my_ntiles, bdg_base, bdg_pg(4096), na, sel_base, maxsub`.

**Open scaling caveat:** at bb3, `maxsub` may be larger and the shard (`epc*maxsub*16`) coexists in L1 with the gather's field CB → may overflow. Mitigation when you get there: chunk the `gx/gy/oidx` CBs, or use a primary-shard + small overflow-buffer instead of `maxsub`-sized shard.

---

## 6. Key Files

| File | What |
|---|---|
| `DREAMPlaceTT/host/v35_ef_engine.cpp` | V35 backward (count/place/gather) + **on-chip unsort block** (`V35_ONCHIP_UNSORT`) + diagnostics (`V35_COLLIDE`, `V35_FOOTPRINT`, `V35_ONCHIP_DBG`). The CPU unsort is at the `grad[sel[oi]] += ` loop. |
| `DREAMPlaceTT/kernels/v35_onchip_unsort.cpp` | The on-chip unsort kernel (3-phase). **JIT-compiled.** |
| `DREAMPlaceTT/host/v22_engine.{h,cpp}` | V22 optimizer engine (interleaved-internal). `configure/set_pos/step_resident/step_resident_ondevice_grad/get_pos/density_grad_address`. **Not in any production build target yet** (only the `v22e2e` harness compiles it in). |
| `DREAMPlaceTT/kernels/v22_*.cpp` (11) | V22 SFPU kernels (combine/precond/nesterov/clamp + readers/writers). Also vendored to `history/harness/kernels/`. |
| `DREAMPlaceTT/host/v22e2e_host.cpp` | e2e harness: unsort→b_dg→V22→pos vs host Nesterov. |
| `DREAMPlaceTT/history/harness/kernels/v35_unsort_l1_full.cpp` | standalone unsort→b_dg kernel (1-write interleaved, dual-RISC; where the cost numbers come from). |
| `DREAMPlaceTT/host/v19_engine.cpp` | forward engine + `DPTT_DEVID`; the forward density scatter is `use_v19_` (NOT the legacy V6/V9 `max_contrib` path — that warning is a red herring, see §8). |
| `integration/scatter_ttnn_client_inprocess.py` | the integration client: patches the autograd backward → V35 (`compute_electric_force_v35_chip`, ~line 789); locks the production config. **This is where the optimizer hook will go.** |
| `DREAMPlace/install/dreamplace/NonLinearPlace.py` | DREAMPlace opt loop; optimizer created ~line 170 (`NesterovAcceleratedGradientOptimizer`, `obj_and_grad_fn=model.obj_and_grad_fn`). `install/` is what runs (PYTHONPATH). |
| `DREAMPlace/install/dreamplace/NesterovAcceleratedGradientOptimizer.py` | `step_bb`: BB stepsize (`s·s, s·y, y·y`) + nesterov (`u=v-α·g; v=u+coef(u-u_prev)`) + clamp (`constraint_fn`). **This is what V22 must replace.** |
| `F/.verifymatrix.sh` | 18-config convergence sweep driver. |
| memory dir | `…/memory/MEMORY.md` (index) + `onchip-unsort-64b-reliable.md` (deep notes), `forward-l1cap-drops-largecells.md` (the refuted hypothesis + the deferred 2-config stall), `v35-fwd-bwd-stage-timing.md`, `always-run-with-tt-watcher.md` (device recovery + IOMMU note). |

---

## 7. How to Proceed (the remaining integration, in order)

**Step A — V22 in-loop, fed on-device, validated numerically on ONE config first.**
Wire so that, each backward iteration: the on-chip unsort fills `b_dg` (done) → V22 runs `combine(raw wl_grad + dw·b_dg) → precond → nesterov → clamp` on-device. You must **bypass DREAMPlace's host grad-combine+precond** (`obj_and_grad_fn` currently returns the already-combined, preconditioned grad) and instead feed V22 the **raw** `wl_grad` (h2d transfer is fine — it's DREAMPlace's WL operator) + `b_dg` + the precond constants (`pin_w`, `area` → `V22.configure` once) + scalars (`alpha`, `coef`, `density_weight`). **First validate** V22's per-step output reproduces DREAMPlace's `v_kp1` to fp32 on one config (extend the `v22e2e`-style check into the loop) before trusting it.
- `coef` is a pure scalar recurrence (`a_kp1=(1+sqrt(4a²+1))/2; coef=(a_k-1)/a_kp1`) — fine to keep as a host scalar (not "processing").
- `density_weight` is DREAMPlace's schedule (host scalar) — keep.

**Step B — BB stepsize on-device.** `alpha = sqrt(s·s)/... ` needs `s_k=v_k−v_{k−1}` (pos diff) and `y_k=g_k−g_{k−1}` (grad diff) dot products over the full `2*num_nodes` vector. With pos+grad resident, write a small **on-device reduction kernel** for `s·s, s·y, y·y` → scalars → `alpha`. (Going to host for this would d2h the whole grad — exactly what we're avoiding.)

**Step C — resident pos → forward.** Keep V22's updated pos resident (`b_v`) and have the forward scatter read it directly (V20-style permute) so positions never round-trip. Until then, `get_pos` → DREAMPlace `p.data` is a transfer-only bridge that works but isn't "no host".

**Step D — replace `step_bb`.** Subclass/patch `NesterovAcceleratedGradientOptimizer` (or intercept in the client) so the per-element update routes through V22 (A) with `alpha` from (B), matching `step_bb`'s math + the density-weight schedule. Watch the constraint/clamp bounds (`lo/hi` = `xl/yl`, `xh-nsx/yh-nsy`; fixed nodes lo=-BIG, hi=+BIG).

**Step E — validate 18 configs.** Run `.verifymatrix.sh` (with the on-chip optimizer enabled) on a **stable, uncontended** board; require overflow ≤ ~0.07 on the 16 that already converge (the other 2 are deferred, §8). Compare per-iter overflow trajectory vs the CPU path.

**Suggested first action for the next agent:** turn on `V35_ONCHIP_UNSORT=1` and confirm `mismatches=0` on a couple configs (sanity that the checkpoint still holds), then start Step A as code (it needs the device only for the final numeric check, so most of it can be written first).

---

## 8. Deferred / Known Issues & Gotchas

- **2-config convergence stall (DEFERRED, separate from this work):** `adaptec3_2048` (overflow 0.081) and `bigblue2_2048` (0.082) don't reach 0.07 on the **current CPU-unsort** pipeline. The `[server] WARNING: max_contrib > L1 cap` is a **RED HERRING** — it's from the legacy V6/V9 scatter (not the active `use_v19_` path), is cell-count-driven, and fires on PASSING designs too (incl. the largest `bigblue3_2048`). Cause unknown. Next step when revisited: a **CPU-baseline run** (does pure DREAMPlace also stop ~0.08 on these at _2048? → design difficulty, not a TT bug) before assuming a defect. The on-chip unsort faithfully reproduces whatever the gather produces, so this is upstream of our work. (See `forward-l1cap-drops-largecells.md`.)
- **Backward (k,h) truncation:** the gather window is capped at `MAX_KH=8`; cells with k>8 (true max ~15; 0.001–0.5% of cells) lose outer columns. Pre-existing in forward (V21/V23 MAX_KH=8) + backward. Small; does NOT correlate with the 2 stalls. Note: this is the *same* set of wide cells that the unsort's sel-duplicate accumulate handles.
- **Device wedge:** soft-wedges at JIT/launch with the device idle. Recovery = **`tt-smi -r 3`** (PCI Dev 3 = our board) **+ `docker restart`**; a container restart *alone* does NOT clear it. `tt-smi -r N` indexes by **PCI Dev ID**, not logical chip (logical 1 → PCI 3; verify mapping via the UMD "Opening local chip ids/PCIe ids" log).
- **IOMMU multi-tenant contention:** `UMD error: Expected NOC address 0x1000000000000000, but got 0x1000000040000000` (off by exactly 1GB) = **another tenant** (e.g. `scardoza` running `tests/nightly/t3000/ccl/test_minimal_all_gather_async.py`) holds the IOMMU sysmem base; our full-cluster init then fails. **Not a board wedge — `tt-smi -r` does NOT fix it.** Check `ps -eo pid,etime,cmd | grep -iE 'scardoza|all_gather'` and **wait** for their run to finish. `TT_METAL_VISIBLE_DEVICES` is NOT honored to isolate.
- **Container instability:** `bh-37-special-ayadav-for-reservation-82536` exits/crashes intermittently → `docker start <C>; sleep 14`. If the container was renewed and lacks python3.8: `add-apt-repository -y ppa:deadsnakes/ppa && apt-get install -y python3.8 python3.8-dev python3.8-venv`, then rebuild `v19_engine`.
- **JIT cache:** kernel-only edits with an unchanged host can serve a STALE cached kernel ("ninja: no work to do"). Use a fresh `TT_METAL_CACHE=/dev/shm/<unique>` to force recompile.
- **Watcher:** ON (`TT_METAL_WATCHER=10`) only when debugging a hang/wedge; OFF for any timing measurement.
- **Always run with the watcher when debugging** so a hang is diagnosable (`generated/watcher/watcher.log`).

---

## 9. One-paragraph summary for a hurried reader

The on-chip unsort — turning the V35 gather output into V22's interleaved `b_dg` density-gradient buffer entirely on-device, including the tricky `sel`-duplicate accumulate for big cells — is **done and validated bit-exact** (`V35_ONCHIP_UNSORT=1`, gated off in production). V22's optimizer kernels are validated standalone. **What remains is wiring V22 to drive DREAMPlace's optimization loop** (feed it raw `wl_grad` + on-device `b_dg`, compute the BB stepsize on-device, keep pos resident, replace `step_bb`) **and validating convergence on the 18 configs** on a stable board. Start by confirming the unsort checkpoint (`V35_ONCHIP_UNSORT=1` → 0 mismatches), then implement Step A (§7).
