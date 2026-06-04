# V22 K2/K3/K4 — clamp, grad-combine, precondition (elementwise SFPU)

> **TL;DR:** Three elementwise SFPU kernels of the chip-resident optimizer:
> **K2 clamp** (bound pos to the placement region), **K3 grad-combine**
> (`g = wl_grad + density_weight·density_grad`), **K4 precondition** (divide by
> per-node `pin_weight + density_weight·area`, via inline Newton-Raphson
> reciprocal — no SFPI divide). All validated rel_l2 < 1e-5 vs torch.

---

## 1. Identity
- **Stage:** optimizer — elementwise (clamp / combine / precondition)
- **Status:** ✓ VALIDATED (microbench, rel_l2 < 1e-5 all scales); part of `V22_CHIP_OPT=1`.
- **Lineage:** companions to K1 (`v22-nesterov`) and K5 (`v22-stepsize`) in the V22 chip-resident optimizer.
- **Source files:** `src/v22_clamp_compute.cpp` (83, K2), `src/v22_gradcombine_compute.cpp` (76, K3), `src/v22_precond_compute.cpp` (120, K4), shared `src/v22_elt_reader.cpp` (63), `src/v22_elt_writer.cpp` (29).
- **Activated by:** `host/v22_elt_microbench_host.cpp --op {clamp|combine|precond}` (target `v22_elt_microbench`).

## 2. Problem & contract
- **K3 grad-combine:** `g_k = wl_grad + density_weight · density_grad` (elementwise over 2·num_nodes).
- **K4 precondition:** `g_k /= (pin_weight + density_weight · area)` per node — the DREAMPlace preconditioner.
- **K2 clamp:** `pos = clamp(pos, lo, hi)` to keep cells in the placement region.
- **Inputs:** resident `pos`/`grad` + per-node constants (lo,hi,pin_w,area) + scalar `density_weight`. **Output:** in-place.

## 3. Mechanism (how it works)
- Shared `v22_elt_reader`/`v22_elt_writer` stream tiles; each `--op` selects the compute kernel (K2/K3/K4). All run on the SFPU over the resident buffers.
- **K4 inline NR reciprocal:** the SFPU has no cheap divide via SFPI, so K4 computes `1/(pin_w + dw·area)` with an **inline Newton-Raphson reciprocal** then multiplies — the key implementation trick.
- **THE TRICK:** keep the whole gradient post-processing (combine → precond → clamp) on chip around the resident pos, so it composes with K1 (nesterov) + the EF backward into a fully on-chip optimizer iteration.

## 4. Performance (measured — TT; see `profile/run.log`)
- Elementwise, sub-ms each; validated rel_l2 < 1e-5 ([[v22_k2k3k4_elt_microbench]]).

## 5. Correctness
- rel_l2 < 1e-5 vs torch for all three ops at all scales. PASS. (K4 NR reciprocal is accurate to bf16/fp32 tolerance.)

## 6. Gotchas / pitfalls
- **K4:** use inline NR reciprocal, not an SFPI divide (which isn't available/cheap on this SFPU path).
- Gate on rel_l2 (FMA 1-ULP), not per-element.
- The `density_weight` scalar must be the current-iter value (it ramps during placement).

## 7. When to use / avoid
- **Wins:** in the chip-resident optimizer (V22_CHIP_OPT) to eliminate host-side gradient post-processing. Standalone they're trivial elementwise ops.

## 8. Provenance
- **Memory:** [[v22_k2k3k4_elt_microbench]], [[v22_chip_optimizer_engine]]
- **Handoff:** `docs/V22_CHIP_RESIDENT_POS_HANDOFF.md`
- **Host:** `host/v22_elt_microbench_host.cpp`; engine `host/v22_engine.cpp`
