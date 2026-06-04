# DCT solve — on-chip Poisson field solve (TTNN matmul, NOT a kernel)

> **TL;DR:** The forward DCT/IDXST/IDCT field solve is **not a Tensix kernel** —
> it's a chain of **TTNN matmuls** in `host/v19_engine.cpp` (`TTNNDCTSolver`).
> 2D DCT-II → frequency `wu/wv` multiplies → IDXST/IDCT inverse → `field_x,
> field_y`. Runs entirely on chip (default), beats CPU DCT (~0.8 ms vs ~2.1 ms
> @512). Documented here as a host-code recipe (no `src/`).

---

## 1. Identity
- **Stage:** forward — DCT/Poisson field solve
- **Status:** ⭐ SHIPPED (the default on-chip DCT; `CPU_DCT` unset). **No kernel file** — TTNN matmul chain in host code.
- **Lineage:** CPU DCT (`CPU_DCT=1`) → **on-chip TTNN DCT (`TTNNDCTSolver::solve_device`)** as the default.
- **Source:** `host/v19_engine.cpp` — `TTNNDCTSolver` struct + `solve_device()` (DCT matrix builders `build_dct2/idct/idxst` + the matmul chain).
- **Activated by:** default (do not set `CPU_DCT`). `TTNN_HIFI=1` for fp32-precision matmuls.

## 2. Problem & contract
**Computation.** Solve the electrostatic field from the density `ρ`: `auv = DCT_M @ ρ @ DCT_Nᵀ`; `field_x = (2·IDXST_M) @ (auv·wu) @ IDCT_Nᵀ`; `field_y = (2·IDCT_M) @ (auv·wv) @ IDXST_Nᵀ`. **Inputs:** density `[M,N]` (chip-resident from the scatter) + the precomputed DCT/IDCT/IDXST matrices + `wu,wv` frequency weights. **Output:** `field_x, field_y [M,N]`. **Invariants:** `initial_density_map` add-back folded on-device; field kept chip-resident for zero-copy backward.

## 3. Mechanism (how it works)
- **Matrix builders** (`build_dct2`, `build_dct2_t`, `build_idct`, `build_idxst`, …) create the N×N / M×M transform matrices once per `(M,N)`; `2×` factors folded into IDXST/IDCT to avoid a scalar multiply.
- **`solve_device`:** the 6-matmul chain above via `ttnn::operations::matmul::matmul`, with a fast `to_layout(ROW_MAJOR)` + `EnqueueReadMeshBuffer` download (bypasses TTNN's slow `cpu()` path), and a zero-copy variant that keeps the field MeshBuffers alive for the EF backward.
- **THE TRICK:** the whole Poisson solve is dense linear algebra → map it onto the **FPU matmul engine** via TTNN. Keep density resident so there's no h2d, and keep the field resident so the backward can read it without d2h.

## 4. Performance (measured — TT)
- chip DCT **0.72–1.08 ms @512** (measured, [[dreamplacett_validated_512]]) vs CPU DCT ~2.07 ms; scales well (CPU DCT ~17.8 ms @2048 vs chip ~2–3 ms).
- **Precision:** default LoFi/bf16 matmuls drift ~5% HPWL / +34% iter at bigblue3/2048 — set `TTNN_HIFI=1` (HiFi4 + fp32 dest-accumulate) for parity there.

## 5. Correctness
- Converges (54/54 forward sweep + the live HPWL parity). bf16 fine at small/medium grid; HiFi4 for large.

## 6. Gotchas / pitfalls
- bf16 matmul precision at 2048 → `TTNN_HIFI=1`.
- The `initial_density_map` add-back must be folded on-device for `CPU_DCT=0` convergence (it is).
- Use the fast `EnqueueReadMeshBuffer` download path, not TTNN `cpu()` (~3 GB/s vs ~7–8 GB/s).

## 7. When to use / avoid
- **Wins:** always, as the default forward DCT — beats CPU and keeps the field on chip for the backward. **Avoid:** `CPU_DCT=1` only as a debug escape hatch.

## 8. Provenance
- **Memory:** [[dreamplacett_validated_512]], [[dreamplacett_standalone_folder]], [[cpu_dct_required_for_v11]], [[tt_vs_cpu_forward_sidebyside]]
- **Host:** `host/v19_engine.cpp` (`TTNNDCTSolver`)
