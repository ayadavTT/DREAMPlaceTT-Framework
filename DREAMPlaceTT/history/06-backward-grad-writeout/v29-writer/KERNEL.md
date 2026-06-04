# grad writeout — per-cell gradient emit (v29_writer / v30_readout)

> **TL;DR:** The small "emit the computed gradient" tails of the backward
> pipelines. `v29_writer` writes per-cell grad out of the V29 gather; `v30_readout`
> converts the V30 fixed-point atomic-summed grad to float and reads it out. Most
> backward kernels emit grad inline; this folder collects the standalone emit
> stages + the grad-scatter pattern.

---

## 1. Identity
- **Stage:** backward — gradient scatter / writeout
- **Status:** ✓ VALIDATED (tails of the V29/V30 pipelines).
- **Source files:** `src/v29_writer.cpp` (37), `src/v30_readout.cpp` (18).
- **Activated by:** within the V29 (`host/v29_host.cpp`) and V30 (`host/v30_host.cpp`) pipelines.

## 2. Problem & contract
Emit the per-cell `(grad_x, grad_y)` produced by the EF gather into the output buffer (`grad_full[sel]`), with no negation (the host/caller negates to match `electric_force` sign). **Inputs:** per-cell grad (fp32 from V29, or fixed-point accumulator from V30). **Output:** grad array indexed by `sel`. **Invariants:** for V30, convert fixed-point → float on readout; grad scatter uses `noc_semaphore_inc` for cross-core accumulation (never raw atomics).

## 3. Mechanism (how it works)
- **v29_writer:** streams the gathered per-cell grad to DRAM (simple per-cell store; the work was in the gather).
- **v30_readout:** the V30 field-stationary backward accumulates grad in a **fixed-point** L1 buffer via atomics; readout converts to float (`×INV_SCALE`) and emits.
- **Grad-scatter pattern (general):** when a cell's grad must accumulate across cores (sub-cell parents, shared owners), use `noc_semaphore_inc` fixed-point accumulation — the same blind-count-is-exact rule as the forward atomics ([[atomic_fetchadd_contention_deadlock]]).

## 4–7. Performance / correctness / gotchas / when
- Cheap relative to the gather; not a bottleneck. Fixed-point scale must avoid grad quantization (V30 ~1.6e-4). Most EF kernels (V25/V28/FCCS) emit grad inline and don't need a separate writer.

## 8. Provenance
- **Memory:** [[v29_full_pipeline]], [[v30_clustering_verdict]], [[v32_e2e_backward_chain]], [[atomic_fetchadd_contention_deadlock]]
- **Host:** `host/v29_host.cpp`, `host/v30_host.cpp`
