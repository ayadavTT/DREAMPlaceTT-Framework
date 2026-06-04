# V22 K1 — Nesterov u/v momentum update (chip-resident optimizer)

> **TL;DR:** The Nesterov-accelerated-gradient `u`/`v` momentum update done
> elementwise on the SFPU, on a **chip-resident** position buffer — so pos never
> round-trips to host between iterations. K1 of the V22 chip optimizer (K1–K5).
> Validated rel_l2 < 1e-5 vs torch at all scales.

---

## 1. Identity
- **Stage:** optimizer — Nesterov/BB elementwise update
- **Status:** ✓ VALIDATED (microbench, rel_l2 < 1e-5 vs torch); part of the `V22_CHIP_OPT=1` chip-resident optimizer engine.
- **Lineage:** V20/V21 kept pos on host (per-iter d2h/h2d) → **V22 (chip-resident pos + on-chip optimizer K1–K5)** so the "fused loop" becomes structurally impossible. K1 = the Nesterov step.
- **Source files:** `src/v22_nesterov_compute.cpp` (125, SFPU), `src/v22_nesterov_reader.cpp` (62), `src/v22_nesterov_writer.cpp` (38).
- **Activated by:** `host/v22_nesterov_microbench_host.cpp` (target `v22_nesterov_microbench`); live via `V22OptEngine` + `V22_CHIP_OPT=1`.

## 2. Problem & contract
**Computation.** Nesterov accelerated gradient: given preconditioned gradient `g_k` and momentum `a_k`, update `u_{k+1} = v_k − α·g_k`, `a_{k+1} = (1+√(4a_k²+1))/2`, `v_{k+1} = u_{k+1} + (a_k−1)/a_{k+1}·(u_{k+1}−u_k)`. All elementwise over the `2·num_nodes` position vector. **Inputs:** resident `u,v` buffers + `g_k` + scalar coefs. **Output:** updated `u,v` in place (chip-resident).

## 3. Mechanism (how it works)
- **Reader/compute/writer split:** reader streams `u_k,v_k,g_k` tiles into CBs; compute (SFPU) does the elementwise FMA update; writer stores back to the resident buffers.
- **Chip-resident pos:** the position/momentum buffers live in chip DRAM across iterations — the whole point is that the optimizer step happens on chip, so pos is never downloaded mid-loop. Combined with K2–K5 + the EF backward, a full optimizer iteration can stay on chip.
- **THE TRICK:** it's a plain elementwise FMA, but keeping it on the resident buffer is what removes the per-iter host fused-loop (the dominant host cost — see [[tt_vs_cpu_forward_sidebyside]]).

## 4. Performance (measured — TT; see `profile/run.log`)
- Tiny (elementwise over 2·num_nodes): sub-ms; gate on rel_l2 not per-element (FMA gives 1-ULP `v_{k+1}` diffs, not a bug) ([[v22_k1_nesterov_microbench]]).

## 5. Correctness
- rel_l2 < 1e-5 vs torch reference at all scales. PASS.

## 6. Gotchas / pitfalls
- **Gate on rel_l2, not per-element equality** — the fused FMA produces 1-ULP differences from torch's separate mul+add; that's expected, not a bug.
- The momentum scalar recurrence (`a_k`) must match DREAMPlace's NesterovAcceleratedGradientOptimizer exactly.

## 7. When to use / avoid
- **Wins when:** you keep pos chip-resident (V22_CHIP_OPT) so the optimizer step + clamp + precond all run on chip → no host fused loop. **Avoid:** standalone it's trivial; its value is only in the resident-pos pipeline.

## 8. Provenance
- **Memory:** [[v22_k1_nesterov_microbench]], [[v22_chip_optimizer_engine]], [[v22_host_backward_direct_d2h]], [[v22_movable_relabel]]
- **Handoff:** `docs/V22_CHIP_RESIDENT_POS_HANDOFF.md`
- **Host:** `host/v22_nesterov_microbench_host.cpp`; engine `host/v22_engine.cpp`
