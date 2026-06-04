# V22 K5 — Barzilai-Borwein step-size (elementwise SFPU reductions)

> **TL;DR:** Computes the Barzilai-Borwein adaptive step size `α = <Δpos,Δpos> /
> <Δpos,Δgrad>` on chip — two dot-product reductions over the resident pos/grad
> deltas — so the line-search step is computed without a host round-trip. K5,
> the last of the V22 chip optimizer kernels (K1–K5 now complete).

---

## 1. Identity
- **Stage:** optimizer — step-size (BB) reduction
- **Status:** ✓ VALIDATED (microbench, rel < 2e-7 vs torch); completes V22 K1–K5.
- **Lineage:** K1 nesterov, K2–K4 elt, **K5 stepsize** = the full chip-resident optimizer. K6 was "scatter-reader integration," not a new kernel; wirelength (Comp D) is the remaining host op.
- **Source files:** `src/v22_stepsize_compute.cpp` (110), `src/v22_stepsize_reader.cpp` (39), `src/v22_stepsize_writer.cpp` (27).
- **Activated by:** `host/v22_stepsize_microbench_host.cpp` (target `v22_stepsize_microbench`, file-based `--inputs/--output`).

## 2. Problem & contract
**Computation.** BB step: `α_k = (Δx·Δx)/(Δx·Δg)` where `Δx = x_k − x_{k-1}`, `Δg = g_k − g_{k-1}` — two inner products over the `2·num_nodes` vector, then a scalar divide. **Inputs:** resident `Δx, Δg`. **Output:** scalar `α` (used to scale the next gradient step). Replaces DREAMPlace's host-side BB line-search step computation.

## 3. Mechanism (how it works)
- **Reduction:** the compute kernel accumulates the two dot products `<Δx,Δx>` and `<Δx,Δg>` across tiles (SFPU multiply + reduce), then forms `α` via the inline-NR reciprocal (same divide-avoidance as K4).
- **THE TRICK:** keep the BB step reduction on chip so the optimizer's adaptive step needs no second forward eval or host round-trip — the resident `Δx,Δg` are already there from K1.

## 4. Performance (measured — TT; see `profile/run.log` — file-based harness)
- 0.06–0.17 ms, rel < 2e-7 vs torch ([[v22_k5_stepsize]]). (Standalone run needs `--inputs` .bin; validated offline.)

## 5. Correctness
- rel < 2e-7 vs torch BB reference at all scales. PASS.

## 6. Gotchas / pitfalls
- Reduction order / fp accumulation: use fp32 accumulate for the dot products (bf16 accum loses precision on large vectors).
- Divide via inline NR reciprocal (no SFPI divide), like K4.
- Harness is **file-based** (`--inputs/--output`) — reproduce by dumping torch `Δx,Δg` to .bin; not a no-arg run.

## 7. When to use / avoid
- **Wins:** in the chip-resident optimizer to compute the BB step without host involvement. Standalone it's a two-dot-product microbench.

## 8. Provenance
- **Memory:** [[v22_k5_stepsize]], [[v22_chip_optimizer_engine]]
- **Handoff:** `docs/V22_CHIP_RESIDENT_POS_HANDOFF.md`
- **Host:** `host/v22_stepsize_microbench_host.cpp`; engine `host/v22_engine.cpp`
