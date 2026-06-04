# Kernel leaderboard — measured cost & effectiveness by stage

Real-silicon numbers from the isolated testbenches (bh-38, 2026-06-03), grouped
by pipeline stage. Use this to pick "the best known approach for stage X and what
it cost." `★` = recommended default. Times are device/kernel medians.
"M" = memory-only (not re-measured this session; needs clean-device rerun).

---

## Forward — density-map construction (scatter)
| kernel | measured | correctness | notes |
|---|---|---|---|
| ★ **v19-scatter** | 1.45 / 2.0 / 3.2 / 3.4 / 6.5 ms (adaptec1/bb1/adaptec3/bb2/bb3) | 54/54 PASS, rel_l2 ≤2.4e-5 | cell-bound, **distribution-insensitive**, no per-page cap |
| v11-scatter | 1.16–5.1 ms (route only) | **48/54** — FAILS bigblue3-clustered | route_buf cap overflows on hot tiles (the reason V19 won) |
| v18-scatter | — | crashes bigblue3-clustered | hash pre-agg; route overflow → hard crash at scale |
| v13-fpu-scatter | M (converges 0.13%) | converges | FPU-matmul engine variant (IPC-server) |
**Verdict:** v19 atomic-add is the robust production scatter. v11/v18's route_buf cap is a hard scaling limit under clustering — *measured*.

## Forward — DCT solve
| kernel | measured | notes |
|---|---|---|
| ★ **dct-solve** (TTNN matmul) | 0.72–1.08 ms @512 (vs CPU ~2.07 ms) | on-chip default; `TTNN_HIFI=1` for fp32 parity at 2048 |

## Backward — electric-force gather (the headline race vs CPU EF ~5–7 ms)
| kernel | measured | correctness | notes |
|---|---|---|---|
| ★ **v25-ef-l1gather** | **0.422 ms (≈17× CPU)** | rel_l2 8.6e-5, nbad 0 | L1-resident direct-load + int fixed-point. 4-corner → small/med grid |
| ★ **v28-ef-multibin** | 1.676 ms | rel_l2 7.0e-8, nbad 0 | correct at ALL grids (multi-bin); the safe default |
| **fccs** | 1.136 ms | rel_l2 2.3e-4, nbad 0 | field-cast, clustering-immune, no bucketing; live-grafted |
| v26-ef | chip 0.783 ms (host grouping 26 ms!) | rel_l2 4.6e-8 | SFPU fp32 4-corner; grouping dominates → use V27 bucketing |
| v30-backward | 0.696 ms | rel_l2 1.6e-4 | field-stationary + fixed-point atomic grad |
| v29-gather | 10.25 ms (prep+bucket 4.99 + gather 5.26) | **rel_l2=nan standalone (CHECK)** | band-bucket clustering fragility (matches [[v29_live_swap]]) |
| v21-ef | M (~17 ms @2048) | bit-exact | first chip EF; producer-bound; the correctness reference |
| v31-backward | live converges | — | ⚠ standalone harness wedges the device — run live only |
**Verdict:** v25 is fastest (≈17× CPU) but 4-corner; **v28 is the correct-at-all-grids default**; FCCS is the clustering-immune, bucketing-free design.

## Backward — cell bucketing / regroup
| kernel | measured | notes |
|---|---|---|
| ★ **v27-bucket** | 0.793 ms (vs CPU 5–7 ms) | cell-level, atomic-free, DRAM-resident |
| v32-regroup | M (0.57/2.55/5.84 ms) | count-prefix, clustering-immune; needs clean-device rerun |
| v30-grouping | M (imbalance 1.25–1.54×) | adaptive bin-owner slab |
| v29-bucketing | 4.99 ms (prep dominates) | use bucket_only + forward stash to skip soft-float prep |

## Optimizer — Nesterov/BB (chip-resident)
| kernel | measured | notes |
|---|---|---|
| v22-nesterov / v22-elt / v22-stepsize | M (sub-ms, rel < 1e-5 vs torch) | file-based harnesses (`--inputs`); offline-validated K1–K5 |

## Infra / probes (lessons)
| probe | finding |
|---|---|
| atomic-bench | blind `noc_semaphore_inc` COUNT exact at any contention; fetch-add slot double-reads (≤2-way only) → use count-prefix |
| mcast-bw | ~13.5 GB/s field-cast → FCCS viable; NOC0/RISCV_0 + receivers-ready-first handshake |
| v11op-bench | SFPU outer product 12.4 ns/cell, but V11 is route-bound (15% mul) → profile before optimizing |
| v16-probes | Blackhole DST = 4×8 col-parity (not 2×16); SEC1 haloize not honored (SrcB-transpose dead → Quasar) |

---

### Cross-cutting lessons for a new kernel
1. **Atomic-add (blind count) beats route+accum** for scatter/grad — no per-page cap, distribution-insensitive (v19 vs v11).
2. **Make the field/data L1-resident** and gather by direct load; do MAC in **integer fixed-point on BRISC** (no FPU there) — v25's 17×.
3. **Bucketing is the fragile/expensive step** — either make it adaptive/count-prefix (v30/v32) or avoid it via **field-cast multicast** (FCCS).
4. **Reuse forward geometry** (V31 stash) — the backward's soft-float prep is redundant.
5. **Profile before optimizing** — the bottleneck is usually routing/producers/host, not the SFPU (v11op, v21).
6. **Blackhole HW traps:** 64-B DRAM read align; `noc_async_write` >~96 KB silently corrupts; multicast on NOC0 + handshake; DST 4×8 layout.
