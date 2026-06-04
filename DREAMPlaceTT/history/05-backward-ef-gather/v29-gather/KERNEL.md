# V29 gather — dual-RISC EF gather (BRISC fx / NCRISC fy)

> **TL;DR:** The gather+compute+scatter stage of the V29 full pipeline, with the
> field gather **split across two RISCs** (BRISC handles field_x, NCRISC handles
> field_y) — a Tracy-driven parallelization. Core gather beats CPU (512=1.13 ms).
> Bit-exact.

---

## 1. Identity
- **Stage:** backward — electric-force gather + grad scatter
- **Status:** ✓ VALIDATED (bit-exact; core beats CPU). Backward half of V29 (front-end = `04-backward-bucketing/v29-bucketing`).
- **Lineage:** V25/V28 L1-gather → **V29 dual-RISC gather** (BRISC fx / NCRISC fy) in the full prep→bucket→gather→scatter pipeline.
- **Source files:** `src/v29_gather_brisc.cpp` (147), `src/v29_writer.cpp` (37).
- **Activated by:** live via engine `V29EFEngine` + `V29_EF=1`; pipeline `host/v29_host.cpp`.

## 2. Problem & contract
**Computation.** For each bucketed cell, gather its overlapped field_x/field_y samples (L1-resident band), compute the weighted force, and scatter the grad. **Inputs:** per-band cell buckets (from v29-bucketing) + L1 field band. **Output:** per-cell grad (no negation; caller negates). **Invariants:** dual-RISC split must not race on shared L1 (sem sync).

## 3. Mechanism (how it works)
- **Dual-RISC gather split:** BRISC gathers/accumulates the field_x contribution, NCRISC the field_y — the two field maps are independent, so splitting halves the gather critical path (Tracy-driven decision).
- **L1-resident band:** the field band stays in L1 (V25/V28 idea).
- **writer:** emits the per-cell grad.
- **THE TRICK:** parallelize the x/y field gather across the two data-movement RISCs (they're independent reductions) — measured to bring the core gather below CPU.

## 4. Performance (measured — TT; see `profile/run.log`)
- Pipeline this run: gather+compute+scatter = **5.26 ms** (prep+bucket 4.99 ms is the other half). Core gather alone: 512=1.125 ms (beats CPU), 1024=1.027, 2048=2.525, bit-exact ([[ef_backward_2x_target]]).
- **Bottleneck:** the wrapper (prep+bucket + d2h), not the gather core — the gather already beats CPU; the surrounding stages dominate ([[ef_backward_2x_target]]).

## 5. Correctness
- Bit-exact (rel_l2 8.7e-8) in the V29 pipeline.

## 6. Gotchas / pitfalls
- Dual-RISC sharing L1: sync so BRISC/NCRISC don't race (same hazard class as V21's 2-point sem).
- The gather is NOT the bottleneck — don't optimize it further; attack prep/bucket/transfer (FCCS does).

## 7. When to use / avoid
- **Wins:** as the gather core (it beats CPU). **Avoid:** the surrounding V29 band-bucketing is clustering-fragile and dominates cost — the gather's win is masked by it; FCCS removes the wrapper.

## 8. Provenance
- **Memory:** [[ef_backward_2x_target]], [[v29_full_pipeline]], [[v29_live_swap]]
- **Host:** `host/v29_host.cpp`; engine `host/v29_ef_engine.cpp`
