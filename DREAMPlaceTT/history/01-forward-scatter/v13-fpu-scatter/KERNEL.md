# V13_fpu scatter — FPU matmul-based density scatter

> **TL;DR:** A density scatter that uses the **FPU matmul engine** (bf16×bf16 →
> fp32) to deposit cell areas, rather than the SFPU/atomic approaches — exploiting
> the otherwise-idle matmul unit in a scatter-heavy workload. Converges within
> 0.13% of CPU after applying CPU offset/ratio to cell sizes. Pairs with
> `02-forward-gather-accum/v13-accum-mt`.

---

## 1. Identity
- **Stage:** forward — density scatter (FPU matmul)
- **Status:** ✓ VALIDATED (converges; IPC-server path). `GATHER_MODE=v13_fpu`.
- **Lineage:** V11 (route/scalar) / V12 (intermediate) → **V13_fpu (FPU matmul)** — distinct from V18/V19 which stay on SFPU/atomic.
- **Source files:** `src/v13_scatter_brisc.cpp` (398), `src/v13_scatter_ncrisc.cpp` (351).
- **Activated by:** `GATHER_MODE=v13_fpu` (IPC server; V13 stays on the server, not in-process).

## 2. Problem & contract
**Computation.** Deposit `ox·oy` area into bins using bf16×bf16→fp32 FPU matmul/reduce. **Inputs:** v4_compute overlaps (bf16). **Output:** density partials → v13-accum. **Invariants:** kernel writes unnormalized density (host ×1/(bsx·bsy)); send orig cell sizes (clamped + 2·offset); `CPU_DCT=1`.

## 3. Mechanism (how it works)
- Forms the per-cell `ox ⊗ oy` outer product and accumulates it into the density tiles using the FPU matmul path (LoFi/bf16 with fp32 DST accumulate).
- **THE TRICK:** use the **matmul engine** for the area deposit — in a scatter workload the FPU is otherwise idle, so this is "free" compute. The accumulate companion (`v13-accum-mt`) continues on the FPU.
- Convergence required applying the CPU offset/ratio to the TT cell sizes (the [[project_v13_fpu_convergence]] fix).

## 4. Performance (measured — TT)
- Converges within 0.13% of CPU @512. Has the richest saved device-profile history of any kernel family (v13 = 56 result dirs). FPU-engine utilization is the differentiator; see `profile/`.

## 5. Correctness
- adaptec1_512 converges within 0.13% of CPU after the offset/ratio fix ([[project_v13_fpu_convergence]]).

## 6. Gotchas / pitfalls
- Unnormalized output → host normalizes ([[v13_fpu_density_normalization]]).
- Send orig sizes (clamped + 2·offset) — the convergence fix.
- bf16 FPU precision: LoFi loses precision (same class as the chip-DCT bf16 note); fp32 DST accumulate mitigates.

## 7. When to use / avoid
- **Wins:** to exploit the FPU/matmul engine for density (engine-diversity experiment). **Avoid:** V19 atomic-add is the simpler production path; V13 is IPC-server-only and more complex.

## 8. Provenance
- **Memory:** [[project_v13_fpu_convergence]], [[v13_fpu_density_normalization]]
- **Handoff:** `docs/V13_FPU_UNLEASH_HANDOFF.md`, `docs/V13_PERF_HANDOFF.md`
- **Host:** `host/v13_full_smoke_host.cpp`, `host/v19_engine.cpp` (v13 path)
