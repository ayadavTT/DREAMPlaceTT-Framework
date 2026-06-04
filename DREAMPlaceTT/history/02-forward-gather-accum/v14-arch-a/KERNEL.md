# V14 (Arch-A) — reduce + scatter (intermediate gather architecture)

> **TL;DR:** An intermediate "Architecture A" forward (separate reduce + scatter
> kernels) explored between V13 and V15. Functional on the IPC server but
> superseded — kept for the architectural record.

---

## 1. Identity
- **Stage:** forward — density reduce + scatter (intermediate)
- **Status:** ◐ SUPERSEDED (functional on IPC server; replaced by V15-spill / V19-atomic).
- **Lineage:** V13 → **V14 Arch-A** → V15 (spill) → V19 (atomic). An architecture-exploration step.
- **Source files:** `src/v14_reduce_brisc.cpp` (91), `v14_reduce_compute.cpp` (57), `v14_reduce_ncrisc.cpp` (61), `v14_scatter_brisc.cpp` (412), `v14_scatter_compute.cpp` (345), `v14_scatter_ncrisc.cpp` (50).
- **Activated by:** `GATHER_MODE=v14` (IPC server).

## 2. Problem & contract
Density scatter + a separate reduce stage (the "Arch-A" split). Inputs: v4_compute overlaps. Output: density map. Stays on the IPC server (not the in-process zero-copy path).

## 3. Mechanism (how it works)
- Separate `reduce` (brisc/compute/ncrisc) and `scatter` (brisc/compute/ncrisc) kernel sets — the architecture explored whether splitting reduce from scatter helped pipelining.
- **THE TRICK / lesson:** the split-stage approach didn't beat the merged V11 accum+reduce or V19's atomic-add; it's an architectural dead-end retained for comparison.

## 4. Performance (measured — TT)
- Functional; superseded on speed. Has saved device profiles (v14 = 16 result dirs) for the architectural comparison.

## 5. Correctness
- Functional on the IPC server (no convergence-win memory; treated as superseded).

## 6. Gotchas / pitfalls
- IPC-server only; `CPU_DCT=1`. More kernel launches than the merged V11/V19 paths (more host round-trips).

## 7. When to use / avoid
- **Avoid for production** — V19 atomic-add superseded it. Reference value: the reduce/scatter split architecture.

## 8. Provenance
- **Handoff:** `docs/V14_ARCH_A_HANDOFF.md`
- **Host:** `host/v19_engine.cpp` (v14 path)
