# V12 scatter — intermediate scatter (V11 → V13 bridge)

> **TL;DR:** An intermediate scatter architecture between V11 (route/scalar) and
> V13 (FPU matmul), including "combined" BRISC/NCRISC variants. Superseded;
> retained for the architectural record.

---

## 1. Identity
- **Stage:** forward — density scatter (intermediate)
- **Status:** ◐ SUPERSEDED (intermediate step; replaced by V13/V18/V19).
- **Lineage:** V11 → **V12** → V13_fpu. Explored combined-RISC scatter before the FPU-matmul direction.
- **Source files:** `src/v12_scatter_brisc.cpp` (436), `v12_scatter_ncrisc.cpp` (360), `v12_brisc_combined.cpp` (563), `v12_ncrisc_combined.cpp` (420), and `v12_accum_*` for its accumulate.
- **Activated by:** legacy `GATHER_MODE=v12` (IPC server).

## 2. Problem & contract
Density scatter (same `Σ ox·oy → D` contract), with "combined" kernel variants that fuse reader+scatter on a single RISC. Inputs: v4_compute overlaps. Output: density.

## 3. Mechanism (how it works)
- The `*_combined` variants fuse the reader and scatter into one BRISC/NCRISC kernel (fewer kernels, different L1/pipelining tradeoff) vs the split V11 layout.
- **THE TRICK / lesson:** combined-RISC fusion was an intermediate experiment; V13's FPU-matmul and V19's atomic-add proved more effective, so V12 was retired.

## 4. Performance (measured — TT)
- Intermediate; no convergence/perf-win memory. Superseded.

## 5. Correctness
- Functional intermediate (no specific convergence record retained).

## 6. Gotchas / pitfalls
- IPC-server legacy path; superseded — not maintained against later engine changes.

## 7. When to use / avoid
- **Avoid for production.** Reference value: combined reader+scatter RISC fusion as an architectural data point between V11 and V13.

## 8. Provenance
- **Host:** `host/v19_engine.cpp` (legacy v12 path). Tools: `tools/v12_accuracy_test.py`, `tools/v12_perf_benchmark.py`.
