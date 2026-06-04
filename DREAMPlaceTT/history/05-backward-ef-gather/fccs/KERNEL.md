# FCCS — Field-Cast, Cell-Stationary electric-force backward

> **TL;DR:** Multicast the (small) field map to **all cores** (an explicit
> shared-cache fill over the NoC), keep cells **balanced and stationary** on
> their cores (no bucketing, no route_buf, no record unsort), and compute the
> per-cell force locally with an **integer MAC**. Kills all four prior backward
> taxes (prep, bucket, route, unsort), is clustering-immune, and beats CPU
> backward on 17/18 configs. Wired live (`FCCS_EF=1`).

---

## 1. Identity
- **Stage:** backward — electric-force gather (field-cast, cell-stationary)
- **Status:** ⭐ SHIPPED live (`FCCS_EF=1`); beats CPU backward 17/18 configs; converges (adaptec1_512 HPWL 60.88M = ref).
- **Lineage:** V21 (recompute) → V25/V28 (L1-gather) → V29/V30/V32 (bucketing approaches) → **FCCS (field-cast, no bucketing)** — the next-gen design that drops the grouping stage entirely.
- **Source files:** `src/fccs_dm.cpp` (188, main compute), `src/fccs_gather_brisc.cpp` (215), `src/fccs_live_dm.cpp` / `fccs_live_stream_dm.cpp` (live graft), `fccs_quant_*` (int quantize/compute/reader/writer), `fccs_deinterleave_dm.cpp`, `fccs_geom_dm.cpp`, `fccs_sort_dm.cpp`.
- **Activated by:** standalone microbench `host/fccs_host.cpp` (target `fccs`); live via engine `FCCSEFEngine` + `FCCS_EF=1`.

## 2. Problem & contract
**Computation.** EF backward (same math). **Inputs:** field maps (multicast to all cores) + the forward's `V31_GEOM` per-cell geometry stash (so no host geometry). **Output:** raw force into `grad_full[sel]`. **Invariants:** field-cast bandwidth is the make-or-break (measured ~13.5 GB/s, fine); multicast must use NOC0/RISCV_0 with receivers-ready-first handshake; int compute uses forward-ordering (no sort).

## 3. Mechanism (how it works)
- **Field-cast (multicast):** broadcast the small field map to **every** core's L1 via NoC multicast — an explicit shared-cache fill. This replaces per-core field gather/bucketing: every core has the whole field locally.
- **Cell-stationary, balanced:** cells stay where the forward put them, split into **balanced slices** across cores (no bucket/route_buf, no per-bin owner). This is what makes it **clustering-immune** — the band/bin-owner approaches (V29/V30) suffered load imbalance under clustering; FCCS doesn't bucket at all.
- **Local int MAC:** each core computes its cells' force against the local (cast) field in **integer fixed-point**, reading the forward `V31_GEOM` stash so there's no geometry recompute or host transfer.

**THE TRICK:** invert the data movement — instead of moving *cells* to the field region (bucketing, the fragile/imbalanced step), move the (small) *field* to all cells (multicast). Combined with cell-stationary balance + forward-geometry reuse, it removes prep + bucket + route + unsort simultaneously.

## 4. Performance (measured — TT; see `profile/`)
- Microbench 1.17 ms; beats/parity CPU at 512+1024; **beats CPU backward 17/18 configs** (only bigblue3_2048/1.1M is 1.13× behind) after forward-ordering + int32 compute ([[fccs_kernel_built]]).
- Field-cast BW ~13.5 GB/s (0.16/0.62/2.45 ms @ 2/8/32 MB for 512/1024/2048) — the make-or-break passed ([[multicast_hangs_bh38_fw1980]]).
- Live: converges (HPWL 60.88M); but live wall 49 ms shared with V29 dispatch overhead (microbench=1.17 ms) — field still host-uploaded in the live graft ([[fccs_live_graft_design]]).

## 5. Correctness
- rel_l2 2.3e-4 (int quantize); accurate + clustering-immune (512 uniform==clustered 1.7 ms). Live converges to ref HPWL.

## 6. Gotchas / pitfalls
- **Multicast hang**: must use NOC0 + RISCV_0 with a **receivers-ready-first** handshake (canonical contributed/multicast pattern); mcast on NOC1 with NOC0 coords hangs ([[multicast_hangs_bh38_fw1980]]).
- Per-tile done-fanin / NoC contention at 2048 was the original slowdown — fixed by forward-ordering (removed the sort tax) + int32 compute.
- Reads the forward `V31_GEOM` stash → forward must run with `V31_STASH=1 V31_GEOM=1`.

## 7. When to use / avoid
- **Wins when:** the field fits the multicast budget (all grids do at ~13.5 GB/s) and you want a clustering-immune, bucketing-free backward. This is the recommended next-gen EF backward.
- **Avoid:** if multicast BW were the bottleneck (it isn't here) or for a one-shot where the forward stash isn't available.

## 8. Provenance
- **Memory:** [[fccs_nextgen_backward_design]], [[multicast_hangs_bh38_fw1980]], [[fccs_kernel_built]], [[fccs_live_graft_design]]
- **Handoff:** `docs/DENSITY_BACKWARD_NEXTGEN_DESIGN.md`
- **Host:** `host/fccs_host.cpp`; engine `host/fccs_ef_engine.cpp`
