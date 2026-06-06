# v13-smoke-bench — V13 Phase-0 bring-up smokes & microbenches (INFRA)

> **TL;DR:** The isolated bring-up probes written while standing up the V13_fpu
> matmul forward: an **FPU matmul smoke** (A·B on one tile), a **multicast smoke**
> (canonical BH handshake), an **OX/OY pack microbench** (BRISC tile-packing cost),
> and a **void BRISC** pairing kernel. Pure infra — they validated primitives /
> measured costs, never shipped.

---

## 1. Identity
- **Stage:** infra — V13 bring-up smokes + microbenches
- **Status:** 🔧 INFRA (primitives validated → folded into V13_fpu; not shipped standalone).
- **Source files:**
  - `src/v13_matmul_smoke_compute.cpp` (49, TRISC), `src/v13_matmul_smoke_reader.cpp`, `src/v13_matmul_smoke_writer.cpp` — FPU `A·B` smoke: reads two bf16 32×32 tiles (A from CB_IN0, B from CB_IN1), computes A·B on the FPU (fp32 DST via `fp32_dest_acc_en`), writes the result tile.
  - `src/v13_mcast_smoke_dm.cpp` (204, NCRISC) — multicast smoke with the canonical handshake (bootstrap fan-in barrier on core-0, then NoC multicast), modeled on `tt_metal/programming_examples/contributed/multicast/`.
  - `src/v13_pack_bench_brisc.cpp` (140), `src/v13_pack_bench_ncrisc.cpp` — OX/OY packing microbench: times packing K=32 synthetic cell records into a 32×32 bf16 OX tile + OY tile in TT-Metal tile/face layout.
  - `src/v13_void_brisc.cpp` (16) — minimal no-op BRISC that exits immediately; paired with an NCRISC kernel so every core has both DM RISCs running (lone-NCRISC programs hang dispatch on Blackhole).
- **Activated by:** standalone bring-up (Phase 0a/0b); `v13_void_brisc` is still used live as the BRISC filler by the `fccs` and `mcast_bw` harnesses.

## 2. Problem & contract
- **matmul smoke:** validate the FPU bf16×bf16→fp32 matmul path on a single tile before using it for V13 density accumulate.
- **mcast smoke:** validate the multicast bootstrap-barrier + NoC-multicast handshake on Blackhole.
- **pack bench:** measure BRISC's cost to pack cell overlaps into bf16 OX/OY tiles (the per-cell prep cost).
- **void BRISC:** give a core a valid BRISC kernel so a NCRISC-only program doesn't wedge dispatch.

## 3. Mechanism
- **matmul smoke:** reader → CB_IN0/CB_IN1 (bf16 tiles); TRISC `matmul_tiles` → DST (fp32); writer drains DST → DRAM.
- **mcast smoke:** core-0 fan-in rendezvous, then sender multicasts a payload to the receiver grid; barrier semantics matched to the canonical example.
- **pack bench:** BRISC loops K records, writes face-interleaved bf16, bracketed by device timestamps.
- **THE TRICK (Blackhole):** every core needs **both** DM RISCs occupied — pairing a void BRISC with a lone NCRISC kernel avoids the dispatch hang.

## 4. Performance (measured — TT)
- Microbenches report device-timestamped zones (matmul tile latency, pack ns/cell). Numbers are bring-up references; the realized costs were re-measured in the shipping variants (see `v11op-bench` for the SFPU outer-product cost).

## 5. Correctness
- Smokes: pass/fail (matmul result tile vs reference; multicast payload received). Benches: timing only (no correctness contract).

## 6. Gotchas / pitfalls
- Lone-NCRISC programs hang dispatch on Blackhole → always pair with `v13_void_brisc`.
- matmul smoke needs `fp32_dest_acc_en` in the host ComputeConfig or DST overflows in bf16.

## 7. When to use / avoid (the lesson)
- **Wins when:** isolating a single primitive (matmul / multicast / packing) during bring-up. **Avoid when:** measuring end-to-end (use the full-kernel harnesses). **Lesson kept:** the void-BRISC pairing requirement + the canonical multicast handshake, both reused throughout.

## 8. Provenance
- **Lineage:** Phase-0 bring-up for the V13_fpu forward (`01-forward-scatter/v13-fpu-scatter`, `02-forward-gather-accum/v13-accum-mt`).
- **Test:** `v13_void_brisc` runs live in the `fccs` / `mcast_bw` harness targets; the matmul/mcast/pack smokes ran via ad-hoc Phase-0 hosts (not retained) — kept here as documented references.
