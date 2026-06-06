# V12 accum — sentinel-protocol FPU density accumulate (Phase 2)

> **TL;DR:** The Phase-2 density accumulate for the V12 forward — BRISC streams
> per-cell (OX,OY) bf16 tile pairs to the TRISC, which accumulates `Σ ox⊗oy`
> on the **FPU matmul engine** (bf16×bf16 → fp32 DST). Uses a **sentinel-based,
> deadlock-free** producer/consumer protocol. Superseded by V13_fpu (cleaner LLK)
> and ultimately by the V19 L1-atomic scatter that ships.

---

## 1. Identity
- **Stage:** forward — density accumulate (FPU matmul, Phase 2)
- **Status:** SUPERSEDED (→ V13_fpu accum → V19 L1-atomic).
- **Lineage:** V11 scalar accum → **V12 (FPU matmul + sentinel protocol)** → V13_fpu accum (`v13-accum-mt`) → V19 direct L1 atomic-add.
- **Source files:** `src/v12_accum_brisc.cpp` (282, RISCV_0), `src/v12_accum_compute.cpp` (98, TRISC), `src/v12_accum_ncrisc.cpp` (RISCV_1).
- **Activated by:** `GATHER_MODE=v12` (legacy IPC server path; not reachable under the production lock).

## 2. Problem & contract
- **Computation:** accumulate per-cell density `Σ ox⊗oy` (outer products of the x/y triangle overlaps) into the per-bin density map, using the FPU matmul/reduce path instead of scalar adds.
- **Inputs:** Phase-1 scatter tiles (bf16 OX/OY pairs) via circular buffers; runs **after** the Phase-1 scatter barrier (the coordinator increment + go-signal handshake from `v12_scatter_brisc.cpp`).
- **Output:** unnormalized density (fp32 DST → bf16/fp32); host applies `1/(bsx·bsy)`.
- **Invariants:** two-phase barrier ordering (scatter must complete before accumulate); BRISC is the CB producer, TRISC the consumer.

## 3. Mechanism
- **RISC division:** BRISC packs/streams (CB_OX, CB_OY) pairs for real contributions then pushes **one sentinel** to signal end-of-stream; TRISC consumes pairs, matmul-accumulates into DST, and stops on the sentinel; NCRISC is the writer/void.
- **THE TRICK:** the **sentinel terminator** makes the producer/consumer loop deadlock-free without a precomputed count — the consumer doesn't need to know how many pairs are coming.

## 4. Performance (measured — TT)
- FPU-matmul accumulate; superseded before a full per-zone profile. The FPU-accum idea was carried into V13_fpu (`v13-accum-mt`, profiled there).

## 5. Correctness
- Bring-up variant; the FPU-accum approach was validated in its successor V13_fpu (converges within 0.13% of CPU @512 after the offset/ratio fix).

## 6. Gotchas / pitfalls
- Requires the Phase-1 scatter barrier to have fired first (ordering bug = reads uninitialized scatter tiles).
- bf16 matmul accumulation loses precision vs the uint32 fixed-point that V19 ships.

## 7. When to use / avoid (the lesson)
- **Wins when:** exploring FPU-matmul density accumulate. **Avoid when:** shipping — V19's direct L1 atomic-add is simpler and more accurate. **Lesson kept:** the sentinel-terminated CB stream (deadlock-free, count-free).

## 8. Provenance
- **Lineage doc:** companion to `01-forward-scatter/v12-scatter`; succeeded by `02-forward-gather-accum/v13-accum-mt`.
- **Test:** archived (legacy IPC `GATHER_MODE=v12`); no standalone harness host extant.
