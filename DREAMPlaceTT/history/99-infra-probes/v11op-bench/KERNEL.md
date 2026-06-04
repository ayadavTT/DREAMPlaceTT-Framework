# v11op-bench — SFPU outer-product microbench (INFRA)

> **TL;DR:** Measures the SFPU `ox ⊗ oy` outer product (the 64 (j,k) area
> multiplies per cell) in isolation: **12.4 ns/cell**, ~12–15× faster than the
> BRISC scalar multiplies (~150–180 ns/cell). Justified Stage B′ — but also
> showed routing (not the multiply) dominates V11, so the realistic win was
> 5–10%, shipped as `v11outer_auto`.

---

## 1. Identity
- **Stage:** infra — SFPU outer-product microbench
- **Status:** 🔧 INFRA (→ shipped as the V11/V18 outer-scatter variant).
- **Source files:** `src/v11op_bench_brisc.cpp` (75), `v11op_bench_compute.cpp` (147, SFPU), `v11op_bench_ncrisc.cpp` (13).
- **Activated by:** `host/v11op_bench_host.cpp` (target `v11op_bench_host`).

## 2. Problem & contract
Time the per-cell 8×8 `ox⊗oy` outer product on the SFPU vs BRISC scalar. Inputs: cell count. Output: ns/cell.

## 3. Mechanism / findings
- **SFPU outer product: 12.4 ns/cell** for all 64 (j,k) multiplies — 12–15× the BRISC scalar path (~150–180 ns/cell).
- **But:** routing dominates V11's per-cell BRISC time (~85%); the multiplies are only ~15%. So full Stage-B′ integration yields **5–10%**, not 3× — the "mul-bound" framing was wrong; V11 is **route-bound**.
- **Shipped outcome:** B′ helps only at large grids (frees route cycles) → `GATHER_MODE=v11outer_auto` routes B′ by grid size.

## 4. Performance (measured — TT)
- 12.4 ns/cell SFPU outer product. Realistic V11 sc+ga improvement from B′: −4.1% aggregate (per-grid routed).

## 5–7. Correctness / gotchas / when
- The lesson: **profile before optimizing** — the SFPU multiply was 12× faster but only 15% of the cost; the real bottleneck was NCRISC routing. Don't optimize the wrong stage.

## 8. Provenance
- **Memory:** [[v11_stage_bprime_outer_product]], [[v11_stage_bprime_sweep_outcome]], [[v11_ox0_backpressure]]
- **Host:** `host/v11op_bench_host.cpp`
