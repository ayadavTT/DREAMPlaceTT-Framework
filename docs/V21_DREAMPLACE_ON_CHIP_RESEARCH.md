# V21 — Porting DREAMPlace's optimizer onto TT: research + roadmap

**Date**: 2026-05-24
**Status**: research only — no code yet
**Goal**: identify the minimum-viable set of DREAMPlace components to port onto TT so that **`pos` lives on chip permanently** and **the host fused loop is structurally impossible** (not just optimized).

This is a precursor to V20+, not a replacement. V20 (chip-side permute kernel) is still the right immediate next step. V21 is the longer arc.

---

## 0. Where we are today

| What runs where | Status |
|---|---|
| Density-map construction (cell positions → bin densities) | **On chip** (V19 scatter + writeout) ✓ |
| DCT solve (density → field maps Fx, Fy) | **On chip** (TTNN matmul chain in `solve_device`) ✓ |
| Field maps consumed by `electric_force` (density gradient) | **On host** ❌ |
| Wirelength + gradient (`weighted_average_wirelength`) | **On host** ❌ |
| Nesterov optimizer step (position update) | **On host** ❌ |
| Constraint clamp (keep cells in [xl, xh] × [yl, yh]) | **On host** ❌ |
| HPWL eval, density-weight update, gamma update | **On host** ❌ |
| **pos array** lives in | **Host DRAM** (DREAMPlace's torch tensor) |

The fused loop exists *only because* `pos` lives on host and must be reshipped to chip every iter in the chip's preferred layout. If pos were chip-resident, there'd be nothing to fuse.

---

## 1. DREAMPlace's per-iter pipeline (from source, ~30 ms on CPU 16T)

From `dreamplace/NonLinearPlace.py` outer loop calling `NesterovAcceleratedGradientOptimizer.step()` (`dreamplace/NesterovAcceleratedGradientOptimizer.py:168` — `step_bb`):

```python
# Pseudocode of the per-iter work
def step_bb():
    # 1. Compute objective + gradient at current position v_k
    obj_k, g_k = obj_and_grad_fn(v_k)        # ← this is the chip+host path today

    # 2. Compute objective + gradient at v_k_1 (previous reference)
    obj_k_1, g_k_1 = obj_and_grad_fn(v_k_1)  # ← TWO evaluations per iter!

    # 3. Barzilai-Borwein step size from (v_k - v_k_1) and (g_k - g_k_1)
    s_k = v_k - v_k_1
    y_k = g_k - g_k_1
    step_size = (s_k . y_k) / (y_k . y_k)    # dot products + scalar

    # 4. Position update (one Nesterov step)
    u_kp1 = v_k - step_size * g_k
    v_kp1 = u_kp1 + coef * (u_kp1 - u_k)     # acceleration term

    # 5. Constraint clamp
    constraint_fn(v_kp1)                      # element-wise clamp to [xl, xh]

    # 6. Save state for next iter
    v_k_1 = v_k;  v_k = v_kp1;  u_k = u_kp1;  a_k = a_kp1
```

And `obj_and_grad_fn` (`dreamplace/PlaceObj.py:394`):

```python
def obj_and_grad_fn(self, pos):
    obj = obj_fn(pos)                        # wirelength + λ·density_penalty
    obj.backward()                            # PyTorch autograd → fills pos.grad
    precondition_op(pos.grad, density_weight, ...)
    return obj, pos.grad
```

Which calls `obj_fn`:

```python
def obj_fn(self, pos):
    self.wirelength = wirelength_op(pos)       # ~1700 LoC C++/CUDA kernel
    self.density    = density_op(pos)           # ElectricPotentialFunction.forward — already ports to TT
    return wirelength + density_weight * density
```

PyTorch's autograd then runs the backward of each op:
- `wirelength.backward()` → wirelength gradient kernel (`weighted_average_wirelength_atomic_kernel`)
- `density.backward()` → `electric_force` kernel (consumes saved field maps + cell positions → grad)

Combined into `pos.grad`. Then preconditioner scales the gradient. Then optimizer uses it.

---

## 2. The four chip-eligible components (in roughly increasing port complexity)

### Component A — Electric Force (density gradient)

**What it is**: backward pass of the electric-potential density operator. Reads `field_map_x`, `field_map_y` (M×N on chip already) plus per-cell positions/sizes, produces a per-cell gradient (one (gx, gy) per movable+filler cell).

**Source**: `dreamplace/ops/electric_potential/src/electric_force_cuda_kernel.cu` (~155 lines).

**Math**: for each cell c with bin range [bx_lo..bx_hi] × [by_lo..by_hi]:
```
gx[c] = sum over bins (b_x, b_y) overlapping cell c of:
          field_x[b_x, b_y] · overlap_area(c, b_x, b_y) / cell_area[c]
gy[c] = similar with field_y
```

**Why it's a natural fit for TT**:
- Same per-cell bin-walking pattern as V19 scatter (just read instead of write).
- Field maps are *already on chip* (output of DCT). No d2h needed.
- Output is one `(gx, gy)` per cell — 2× per-cell data, smaller than the density map.

**Replaces**: the host's `density.backward()` call + the d2h of field maps (currently 4.6 ms via fast TTNN download) + the per-cell electric_force kernel running on CPU.

**Estimated effort**: ~1-2 weeks. Pattern is structurally identical to V19 scatter (per-cell × per-bin loop with atomics). Reuse much of the V19 host wiring.

**Estimated win** *(2026-05-26: now measured, see §11)*: original estimate was ~6–10 ms/iter on bb3 from eliminating field d2h + on-host electric_force + autograd glue. The naive port measures **slower than CPU 16T at every scale we tested** (0.30-0.39×). The per-cell × per-bin shape is wrong for this hardware; landing the projected win requires SFPU vectorization + tiled field loads, not a straight transliteration. See §11 for the measured numbers and §4-revised for revised phase ordering.

---

### Component B — Constraint clamp

**What it is**: `constraint_fn(v_kp1)` — element-wise `max(xl, min(xh, pos))` on each position component.

**Source**: implemented as a couple of `torch.clamp` calls in DREAMPlace's BasicPlace flow.

**Math**: trivial. Per cell: `pos[c] = clamp(pos[c], xl, xh)` (for x; symmetric for y).

**Why trivial on TT**: tiny SFPU broadcast op. Can be folded into any kernel that produces pos.

**Estimated effort**: hours (it's literally one kernel pass over `pos`).

**Estimated win**: <1 ms/iter; bundled into the optimizer kernel naturally.

---

### Component C — Nesterov BB optimizer step

**What it is**: the actual position update math from `step_bb()`.

**Source**: `dreamplace/NesterovAcceleratedGradientOptimizer.py:168-240`.

**Math (per iter)**:
```
1. s_k = v_k - v_k_1                       # vector subtract, len 2*nc
2. y_k = g_k - g_k_1                       # vector subtract, len 2*nc
3. bb_step = (s_k · y_k) / (y_k · y_k)     # two dot products → 2 scalars
4. a_kp1   = (1 + sqrt(4*a_k^2 + 1)) / 2   # scalar
5. coef    = (a_k - 1) / a_kp1              # scalar
6. u_kp1   = v_k - bb_step * g_k           # vector AXPY
7. v_kp1   = u_kp1 + coef * (u_kp1 - u_k)  # vector AXPY
8. clamp(v_kp1)                            # element-wise (Component B)
```

**Why it's tractable on TT**:
- All vector ops are element-wise — perfectly parallel across 110 cores.
- Dot products are reductions — one tree-sum kernel; TT-Metal has primitives.
- Scalar arithmetic on host or in a single core.
- No atomics needed.

**Estimated effort**: 1 week. Need:
  - Vector AXPY kernel (~50 LoC)
  - Dot-product / reduction kernel (TT-Metal probably has built-ins; if not ~100 LoC)
  - Glue to chain them

**Estimated win**: ~3-5 ms/iter directly + eliminates the per-iter host↔chip pos round-trip (~10 ms in current TT path).

---

### Component D — Wirelength + gradient (HARD)

**What it is**: weighted-average wirelength for net-based wirelength metric (DREAMPlace's primary wirelength model).

**Source**: `dreamplace/ops/weighted_average_wirelength/src/*` (~1700 lines C++/CUDA across 10 files).

**Math (per net)**: for each net N with pins P_1..P_k:
```
WAWL_x(N) = sum_i (x_i * exp(x_i/γ))  / sum_i exp(x_i/γ)
          - sum_i (x_i * exp(-x_i/γ)) / sum_i exp(-x_i/γ)
```
γ is the smoothness parameter (updated periodically per `update_gamma_op`).

The atomic variant uses per-pin atomics for the sum-reductions. Output: per-cell wirelength gradient.

**Why it's hard on TT**:
- Net structure is irregular — each net has a different number of pins (from 2 to thousands).
- Atomic reductions on chip can be made to work (V19 atomic-add is precedent) but the pin→net mapping is non-trivial.
- Two `exp()` passes per pin — TT SFPU can do exp but precision and throughput need tuning.

**Estimated effort**: 4-8 weeks. Largest single kernel in DREAMPlace.

**Estimated win**: ~5-10 ms/iter (replaces host wirelength + eliminates wirelength-grad d2h↔h2d).

**Pragmatic fallback**: leave wirelength on host. Per iter, d2h `pos` (8 MB) → host computes wirelength + grad → h2d wirelength_grad (8 MB) → chip sums with density grad. Round-trip adds ~5 ms/iter but unblocks pos-on-chip.

---

## 3. What `obj_and_grad_fn` needs from chip to "stop touching host" per iter

DREAMPlace's optimizer needs only **one thing** from each `obj_and_grad_fn` call to take a step: `g_k` (gradient w.r.t. pos). The objective scalar is for logging / line search / convergence checks; it can be sampled less frequently.

So the minimum chip-resident state to eliminate the fused loop is:
- **pos** (`v_k`, `u_k`, `v_k_1` arrays — 3 × 2 × nc floats)
- **gradient g** (combined density_grad + wirelength_grad — 2 × nc floats)
- **field maps Fx, Fy** (2 × M × N floats — already chip-resident as TTNN tensors)
- **Nesterov scalars** `a_k`, `alpha_k`, etc. (constant per design)

Total: ~30–60 MB of chip DRAM for big designs. Fine — chip has 16 GB.

---

## 4. Minimum-viable port to eliminate the fused loop

If the goal is *just* "eliminate the fused loop," the cheapest path is:

### Minimal path (Components A + C, leave B trivial, leave D on host)

```
       per iter, in this minimum-viable design:
       ┌─────────────────────────────────────────────────────────┐
       │ CHIP:                                                    │
       │   scatter (V19) → density_buf                            │
       │   TTNN DCT      → field_x, field_y                       │
       │   electric_force kernel (Component A)                    │
       │      reads field_x/y + pos[] (chip-resident)             │
       │      → density_grad (2 × nc floats)                      │
       │   ─── d2h: pos (8 MB) ───────────────────────────────┐  │
       └──────────────────────────────────────────────────────│──┘
                                                              │
       ┌──────────────────────────────────────────────────────▼──┐
       │ HOST:                                                    │
       │   wirelength_op(pos) + wirelength.backward()             │
       │   → wirelength_grad (2 × nc floats)                      │
       │   ─── h2d: wirelength_grad (8 MB) ──────────────────┐   │
       └─────────────────────────────────────────────────────│───┘
                                                             │
       ┌─────────────────────────────────────────────────────▼───┐
       │ CHIP (continued):                                        │
       │   sum_grad = density_grad + wirelength_grad              │
       │   precondition(sum_grad)                                 │
       │   nesterov_step(pos, sum_grad)  (Component C)            │
       │   constraint_clamp(pos)         (Component B)            │
       │                                                          │
       │   pos LIVES HERE permanently. No fused loop ever.        │
       └──────────────────────────────────────────────────────────┘
```

Per-iter cost estimate (bb3_2048):
- chip work (scatter + DCT + electric_force + optimizer): ~30 ms
- d2h pos: ~2 ms
- host wirelength: ~5 ms
- h2d wirelength_grad: ~2 ms
- **Total: ~39 ms** (vs current 61 ms = ~1.6× speedup at the per-iter level)

The fused loop is *structurally gone* because pos never returns to the host except for the small wirelength-only d2h.

### Full path (Components A + B + C + D)

Add Component D (wirelength on chip). Eliminates the d2h(pos)/h2d(grad) round-trip. Per-iter drops to ~32 ms (1.65×). Bigger savings come from headache: 4–8 weeks of kernel work.

---

## 5. Recommended phase order

| Phase | What ships | Effort | Cumulative win on bb3 |
|---|---|---|---|
| **V20** | Chip-side permute (eliminates host fused loop; pos still host-resident) | 6-10 hr | −16 ms/iter |
| **V21a** | Electric force on chip + d2h(field_x/y) eliminated | 1-2 wk | additional −6 to −10 ms |
| **V21b** | Nesterov + clamp on chip (pos becomes chip-resident) | 1 wk | additional −3 to −5 ms |
| **V21c** | Wirelength on chip (eliminates pos round-trip) | 4-8 wk | additional −5 to −8 ms |

After V20 + V21a + V21b: pos lives on chip permanently. Fused loop is structurally impossible. Estimated bb3 per-iter: ~40-45 ms (vs CPU 53 → ~1.2×).

After V21c: full chip residency. Estimated bb3 per-iter: ~30-35 ms → 1.6× CPU. To get to **2×** you'd also need TTNN DCT in fp32 (separate open work).

---

## 6. Reusable assets from V19 work

A lot of V19's plumbing maps directly:

- **Per-cell × per-bin loop pattern** in V19 scatter is identical to what `electric_force` needs (just read field instead of atomic-write density).
- **Cell-sort permutation** (`ix[]`, `iy[]`, `ox[]`, `oy[]`) can be reused identically for electric_force.
- **Per-core L1 layout, NoC alignment quirks** (32-B alignment, 96 KB write cap) — all well-trodden.
- **TTNN tensor download fast path** (via `get_mesh_buffer_leak_ownership()` + `EnqueueReadMeshBuffer`) — already proven to give 6.9 GB/s.
- **In-process pybind11 module** (`v19_engine`) — V21 stages live alongside the existing engine; just extend `scatter()` to `scatter_and_grad()` or similar.

---

## 7. Concrete file map (what to touch in DREAMPlace)

| Stage | DREAMPlace file (read-only, for reference) | TT file to create |
|---|---|---|
| Electric force | `dreamplace/ops/electric_potential/src/electric_force_cuda_kernel.cu` (155 LoC) | `kernels/v21_electric_force_dm.cpp` (~200 LoC) + host JIT in `v19_engine.cpp` |
| Nesterov step | `dreamplace/NesterovAcceleratedGradientOptimizer.py:168-240` | `kernels/v21_nesterov_step_dm.cpp` (~80 LoC) + dot-product reduction kernel |
| Constraint clamp | called inside the optimizer's `constraint_fn` | inline in Nesterov kernel, ~10 lines |
| Wirelength | `dreamplace/ops/weighted_average_wirelength/src/weighted_average_wirelength_atomic_cuda_kernel.cu` (~155 LoC) | `kernels/v21_wawl_dm.cpp` (~400 LoC, complex) |

Host-side glue:
- Extend `host/v19_engine.cpp` (the in-process engine class) with new MeshBuffers for `pos`, `g_density`, `g_wirelength`, `nesterov_state`.
- Extend pybind11 wrapper with `do_iter(...)` that runs the whole per-iter chip pipeline and returns only what the host needs (HPWL stats, optionally pos for monitoring).
- Replace the patch hook's monkey-patching of `ElectricPotentialFunction.forward` with a higher-level hook that replaces `NesterovAcceleratedGradientOptimizer.step` itself.

---

## 8. Open questions / risks

1. **Wirelength precision on TT SFPU**: the `exp()` calls in WAWL need careful precision tuning. TT SFPU's `exp` accuracy is lower than CPU IEEE. May affect convergence trajectory. Mitigation: validate HPWL bit-equal CPU on a benchmark set before/after Component D.

2. **Multi-net atomics**: WAWL needs per-net reductions. Net counts go to ~10⁶ at bigblue3 scale. Memory note: V18 hash-aggregation got close to this pattern; can it be reused?

3. **Determinism**: DREAMPlace's `deterministic_flag` requires fixed-point atomics. V19 already does that (FixedPoint atomics with `V19_SCALE_BITS`). Extend to electric_force + wirelength.

4. **HPWL is computed every 50-100 iters for the density-weight update**. That requires a small d2h every so often — fine, not in the hot path.

5. **PyTorch autograd integration**: DREAMPlace uses PyTorch's autograd, which records a graph during `forward` and replays it during `backward`. Replacing the whole `obj_and_grad_fn` with a single chip kernel means *bypassing autograd entirely*. The monkey-patch lands at a higher level (`NesterovAcceleratedGradientOptimizer.step` itself), and we expose only the position update — no torch tensor graph.

6. **Periodic operations**: DREAMPlace also runs `update_gamma_op`, `update_density_weight_op`, fence-region ops, etc. These can stay on host (they run every Lgamma_step iterations — typically every 50-100 iters, so amortized cost ≈ 0).

---

## 8.5 V20 accuracy-bug status (carry-forward from V20 work)

The V20 microbench had a 75% mismatch rate. **Root cause has been identified** (parallel session): Blackhole `noc_async_read` from DRAM requires **64-byte aligned source + size**, not 32-byte as on Wormhole. Reads with 32-aligned-but-not-64-aligned addresses silently return wrong data. Memory note: `blackhole_dram_read_align_64.md`.

Fix landed in the refactored V20 kernel pair:
- `kernels/v19_mb_permute_common.h` — `V20_POS_READ_ALIGN = 64`
- `kernels/v19_mb_permute_ncrisc.cpp` (NoC 1 handles second half of cells)
- `kernels/v19_mb_permute_brisc.cpp` (NoC 0 handles first half)

When V21 begins, the V20 microbench should be re-tested first (build + run on adaptec1_2048 and bigblue3_2048 sizes) to confirm:
1. **Accuracy: 0/N mismatches** (vs prior 75%).
2. **Speed: still ≤ 10 ms at bigblue3 scale** (the per-byte bandwidth doesn't change with alignment; some headroom may be lost due to 2× reads-per-cell on the BRISC/NCRISC split).

Any V21 kernel (electric_force, Nesterov, wirelength) that does `noc_async_read` from DRAM must use 64-byte alignment from the start.

## 9. What to do first (when picking up V21)

1. **Read this doc** and `docs/V20_CHIP_PERMUTE_HANDOFF.md` end to end.
2. **Ship V20 first** (the chip-permute kernel). V21 builds on the same engine + per-core paging infrastructure V20 establishes.
3. **Prototype Component A (electric_force) standalone** with a microbench similar to `host/v19_microbench_host.cpp`. Synthetic field maps + synthetic positions, verify gradient bit-identical to CPU's `electric_force_cpp`. The math is the most validated part of the system — perfect target for the first real V21 milestone.
4. **Then Component B + C (clamp + Nesterov)** — these are small and let us prove the "pos lives on chip" claim end-to-end on at least one config (run DREAMPlace for N iters with `pos` only sampled at iter end for HPWL).
5. **Defer Component D (wirelength)** until A + B + C are shipping. Wirelength is the largest single piece; doing it last lets us focus on the convergence-critical pieces first.

The fused loop is gone after step 4. From there, **the entire DREAMPlace inner loop runs without the host touching `pos`**, modulo the wirelength round-trip if Component D isn't yet shipped.

---

## 11. V21a measured results (2026-05-26)

**Status**: kernel CORRECT, perf NOT YET A WIN.

Standalone microbench at `host/v21_electric_force_microbench_host.cpp` (TT side) + `integration/v21_ef_microbench.py` (Python orchestrator). Synthetic inputs (numpy seed 12345), CPU baseline calls DREAMPlace's production `electric_potential_cpp.electric_force` at `OMP_NUM_THREADS=16`. TT timing is median of 5 launches after JIT warmup.

| Workload | Cells | TT kernel (median) | CPU 16T (median) | Speedup | Mismatches | Max diff |
|---|---:|---:|---:|---:|---:|---:|
| Tiny (32×32) | 80 | 0.108 ms | 0.057 ms | **0.54×** | 0 / 80 | 0 |
| a1_2048 size | 540 K | 27.56 ms | 10.89 ms | **0.39×** | 0 / 540 000 | 0 |
| bb3_2048 size | 1.1 M | 55.92 ms | 16.70 ms | **0.30×** | 0 / 1.1 M | 0 |

### Correctness
**Bit-exact** at every tested scale (max_abs_diff = 0, max_rel_diff = 0). Matches DREAMPlace's production CPU `electric_force_cpp.electric_force` exactly. Per-cell × per-bin math, bin-range clamping (+1 then clamp), `triangle_density_function` semantics, ratio scaling, and the negation-baked-in convention from `electric_potential.py:235` are all correct.

### Performance: slower than CPU and the gap widens with scale
- TT: 27.6 → 55.9 ms at 2.04× more cells = **2.03× wall** — linear (DRAM-bound).
- CPU: 10.9 → 16.7 ms at 2.04× more cells = **1.53× wall** — sub-linear (L3 cache helps).

So as designs grow, the chip falls further behind CPU. This is the **opposite** of what we need.

### What we tried and what didn't move the needle
**Grouped-barrier optimization** (`V21_FIELD_GROUP=4`): cells issue field reads in groups of 4 with a single shared barrier (vs one barrier per cell). Bit-exact preserved. Wall-clock impact: 27.56 → 28.60 ms — within measurement noise, **no win**. → Barriers are not the bottleneck.

### Where the time actually goes (TT-Tracy device profile, a1_2048)

Profiled with `TT_METAL_DEVICE_PROFILER=1`. Per-batch breakdown (1 batch = 64 cells = 16 groups of 4):

| Zone | Per-group µs | Per-batch µs | % of total |
|---|---:|---:|---:|
| **V21MB-EF-MATH** (scalar per-cell × per-bin compute) | **41.3** | **661** | **89.5%** |
| V21MB-EF-PREP (bin-range arithmetic + DRAM read issue) | 4.2 | 67 | 9.1% |
| V21MB-EF-READ-POS (1× per batch, batched) | — | 11.5 | 1.5% |
| V21MB-EF-WAIT (`noc_async_read_barrier`) | 0.33 | 5.3 | 0.7% |
| V21MB-EF-READ-CONST, WRITE-OUT | — | <1 | <0.1% |

**Key revelation**: `noc_async_read_barrier()` is 0.7% — i.e. DRAM reads are essentially free. By the time the kernel reaches the barrier, the field-row reads are already done. The grouped-barrier change earlier was correctly bit-exact but couldn't possibly help because barriers weren't the cost.

**The real bottleneck is scalar BRISC/NCRISC floating-point math.** Per-cell compute is ~10 µs (41.3 µs ÷ 4 cells/group). For ~25 (k, h) pairs × ~12 float ops each ≈ ~300 ops per cell, that's ~33 ns/op = **~43 cycles per float op at 1.3 GHz**. This is consistent with the BRISC/NCRISC data-movement RISCs not having hardware FPUs — float ops compile to soft-float library calls.

CPU 16T native FMAs at ~3.5 GHz with vectorization → ~50× faster per op. That explains the entire 3× wall-clock gap and then some (CPU also has L3 cache for the field map).

### What this means for optimization paths

| Path | Targets | Will it help? |
|---|---|:---:|
| 1. Move inner loop to TRISC/SFPU (vectorized float) | the 90% scalar-math bottleneck | **Yes — this is the only meaningful path** |
| 2. Tiled field loads | DRAM read cost | **No** — DRAM is already 0.7% |
| 3. Interleaved field_x/field_y storage | DRAM op count | **No** — same reason |
| 4. Enable BRISC hardware float (if compiler flag exists) | the 90% scalar-math bottleneck | **Worth investigating** — could be a quick 10-30× without a kernel rewrite |

The right next step is to verify whether the BRISC/NCRISC compile flags actually disable hardware float (TT-Metal kernels are JIT-built; the build invocations are visible). If hardware float is off and can be turned on, that's a cheap win. If not, we need the TRISC/SFPU rewrite — substantial but well-scoped: move the per-cell × per-bin loop into a compute kernel, reshape per-cell metadata into 32-lane tiles, and use the existing SFPU patterns from V11outer / V13 as a template.

### Recommendation
**Treat the simple per-cell port as a feasibility/correctness checkpoint, not the production kernel.** Before integrating into `v19_engine.cpp` for the host fused-loop elimination, invest the rewrite for (1) or (2). Alternatively, accept that V21a may never pay off standalone and revisit only after V21b (Nesterov on chip) makes `pos` chip-resident — at which point we can elide the per-iter cell-sort and pos-d2h work that costs more than electric_force itself anyway.

### Reproducing these numbers
```bash
export TT_METAL_HOME=$PWD/tt-metal
export TT_METAL_RUNTIME_ROOT=$PWD/tt-metal
DREAMPlace/dp_env/bin/python3 integration/v21_ef_microbench.py \
    --M 2048 --N 2048 --num-nodes 540000
DREAMPlace/dp_env/bin/python3 integration/v21_ef_microbench.py \
    --M 2048 --N 2048 --num-nodes 1100000
```
The microbench source: `host/v21_electric_force_microbench_host.cpp`, `kernels/v21_ef_{common.h,brisc.cpp,ncrisc.cpp}`, `integration/v21_ef_microbench.py`. CMake target: `v21_ef_microbench` (currently added to `host/CMakeLists.txt`).

---

## 10. Memory notes / handoffs this builds on

- `docs/V19_RUNG3_INPROCESS_HANDOFF.md` — in-process pybind11 engine (delivered)
- `docs/V20_CHIP_PERMUTE_HANDOFF.md` — chip-side permute kernel (designed, microbench-validated; accuracy bug remains)
- `memory/v20_chip_permute_handoff.md` — current state of V20 work
- `memory/v19_block_fp32_outcome.md` — V19 scatter + writeout design
- `memory/v17_ttnn_gather_outcome.md` — failed experiments using stock TTNN ops; informs why custom kernels remain the path
- `dreamplace/NonLinearPlace.py` — the loop you're replacing
- `dreamplace/NesterovAcceleratedGradientOptimizer.py` — the optimizer to port
- `dreamplace/PlaceObj.py:obj_and_grad_fn` — the entry point we monkey-patch

---

*Last updated 2026-05-24. Background research only — no code yet. V21 follows V20.*
