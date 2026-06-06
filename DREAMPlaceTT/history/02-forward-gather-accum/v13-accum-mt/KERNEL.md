# V13 accum (multithreaded FPU) — matmul-style density accumulate

> **TL;DR:** The accumulate stage for the V13_fpu forward — uses the **FPU
> matmul engine** (bf16×bf16 → fp32 DST) with a custom paired-unpack LLK to
> accumulate density, multithreaded across the 3 TRISCs. Converges (the fix was
> applying CPU offset/ratio to TT cell sizes). Companion to `v13-fpu-scatter`.

---

## 1. Identity
- **Stage:** forward — density accumulate (FPU matmul)
- **Status:** ✓ VALIDATED (converges within 0.13% of CPU @512 after the offset fix).
- **Lineage:** V11 scalar accum → **V13_fpu (FPU matmul accum)** — uses the matmul/reduce engine instead of scalar adds. The `_mt` files are the shipped multithreaded versions (non-`_mt` were the single-thread bring-up).
- **Source files:** `src/v13_accum_brisc_mt.cpp` (349), `src/v13_accum_compute_mt.cpp` (138, TRISC), `src/v13_accum_ncrisc_void.cpp` (15), `src/v13_llk_math_kaccum.h` (89, custom LLK). Single-thread bring-up predecessors (archived alongside): `src/v13_accum_brisc.cpp`, `src/v13_accum_compute.cpp`, `src/v13_accum_ncrisc.cpp`.
- **Activated by:** `GATHER_MODE=v13_fpu` (IPC server path).

## 2. Problem & contract
**Computation.** Accumulate per-cell density contributions using the FPU (matmul/reduce) rather than scalar SFPU/BRISC adds. **Inputs:** scatter tiles (bf16). **Output:** density map (fp32 DST → bf16/fp32). **Invariants:** kernel writes **unnormalized** density; host multiplies by `1/(bsx·bsy)` to match the V11/V12 contract; CPU offset/ratio must be applied to TT cell sizes (send orig = clamped + 2·offset).

## 3. Mechanism (how it works)
- **FPU matmul accumulate:** uses `v13_llk_math_kaccum.h` — a custom math LLK for k-accumulation — with a paired-unpack (`v13_llk_unpack_paired.h`) so bf16×bf16 products accumulate into the fp32 DST register across tiles. Runs `_mt` (multithreaded across TRISC_0/1/2).
- **THE TRICK:** offload the density accumulate onto the **FPU** (the matmul/reduce engine, otherwise idle in a scatter-heavy workload) instead of scalar adds. The convergence fix was applying the CPU's offset/ratio to the TT cell sizes (send orig sizes) so the density matches CPU.

## 4. Performance (measured — TT)
- Converges within 0.13% of CPU @512. Rich device-profile history exists (v13 has the most saved profiles). FPU-accumulate is a distinct engine-utilization approach; see `profile/`.

## 5. Correctness
- adaptec1_512 converges within 0.13% of CPU after the offset/ratio fix ([[project_v13_fpu_convergence]]); density normalization handled host-side ([[v13_fpu_density_normalization]]).

## 6. Gotchas / pitfalls
- Kernel writes **unnormalized** density — host MUST multiply by `1/(bsx·bsy)` (the V11/V12 contract) ([[v13_fpu_density_normalization]]).
- Send **orig** cell sizes (clamped + 2·offset) — missing the CPU offset/ratio was the original divergence root cause ([[project_v13_fpu_convergence]]).
- `CPU_DCT=1` required for V13 (it stays on the IPC server, not the in-process zero-copy DCT path).

## 7. When to use / avoid
- **Wins:** to exploit the otherwise-idle FPU/matmul engine for density accumulate. **Avoid:** V19 atomic-add is simpler and the production path; V13 is the FPU-engine experiment (IPC-server only).

## 8. Provenance
- **Memory:** [[project_v13_fpu_convergence]], [[v13_fpu_density_normalization]]
- **Handoff:** `docs/V13_FPU_UNLEASH_HANDOFF.md`, `docs/V13_FPU_SPEEDUP_PLAN.md`
- **Host:** `host/v19_engine.cpp` (v13 path)
