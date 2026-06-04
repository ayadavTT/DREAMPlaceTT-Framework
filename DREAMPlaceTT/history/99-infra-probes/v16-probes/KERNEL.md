# v16-probes — Blackhole DST layout + SrcB-transpose investigation (INFRA/REFUTED)

> **TL;DR:** Probes that investigated whether a matmul-unpack SrcB-transpose
> ("Option A") could accelerate the V16 hybrid. **Refuted on Blackhole** — but
> they produced two durable hardware facts: (1) the SFPU `dst_reg[i]` layout is
> 4 rows × 8 cols (col-parity interleaved), NOT 2×16; (2) `THCON_SEC1` haloize
> is not honored for matmul unpacks on Blackhole (SEC0 works). Kept for the
> hardware lessons.

---

## 1. Identity
- **Stage:** infra — SFPU/matmul-unpack probes (not a pipeline kernel)
- **Status:** ✗ REFUTED (Option A dead on Blackhole) / 🔧 INFRA (hardware facts).
- **Source files:** `src/v16_dst_probe_*` (DST register layout), `src/v16_sec1_probe_*` (SEC1 haloize), `src/v16_sfpu_bench_*` (SFPU pack bench).
- **Activated by:** `host/v16_dst_probe_host.cpp`, `host/v16_sec1_probe_host.cpp`, `host/v16_sfpu_bench_host.cpp`.

## 2. Problem & contract
Probe whether (a) per-lane SFPU packs can match a 2×16 DST assumption, and (b) the SrcB matmul-unpack transpose (haloize) works — to decide if "Option A" (SrcB-transpose LLK adapter) could speed the V16 hybrid.

## 3. Mechanism / findings
- **DST register layout:** `dst_reg[i]` covers **4 rows × 8 cols (col-parity interleaved)**, NOT the 2×16 the V16-HYBRID handoff assumed. Per-lane writes via `vConstTileId` predicate cost the same per instruction → SFPU pack speedup vs BRISC drops from 2× to ~1.2×.
- **SrcB-transpose (Option A):** `THCON_SEC1_REG2_Haloize_mode` is defined in cfg_defines but **silicon doesn't honor it for matmul unpacks** (probe: 0/1024 lanes differ vs baseline); SEC0 haloize DOES work (sanity 1024/1024). ⇒ Option A would land on **Quasar** (first-class SrcB transpose), not Blackhole.
- **Lesson:** verify HW capability with a probe before building on a datasheet bit; the DST layout fact is reused by any SFPU-pack work.

## 4–7. Performance / correctness / gotchas / when
- No pipeline use (refuted). The hardware facts (DST layout, SEC1 not honored) are the value. SFPU pack ceiling ~1.2× on Blackhole.

## 8. Provenance
- **Memory:** [[v16_dst_register_layout]], [[v16_option_a_outcome]]
- **Handoff:** `docs/V16_HYBRID_HANDOFF.md`, `docs/V16_OPTION_A_INVESTIGATION.md`
- **Host:** `host/v16_*_probe_host.cpp`
