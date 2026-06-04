# V19 writeout — fp32-on-chip density emit to row-major DRAM

> **TL;DR:** Converts each core's uint32 fixed-point density slab to fp32 **on
> chip** and writes it per-row straight into the row-major `density_buf` DRAM
> (the same buffer TTNN-DCT consumes) — eliminating the ~9 ms host PCIe
> round-trip + uint32→fp32 CPU loop that used to dominate V19's gather at 2048.
> Block-partition ownership makes this a contiguous per-row write (no host
> de-stride).

---

## 1. Identity
- **Stage:** forward — density writeout (final emit)
- **Status:** ⭐ SHIPPED (the V19 writeout; production forward). Pairs with `01-forward-scatter/v19-scatter`.
- **Lineage:** in-kernel writeout (raced atomics) → **separate fp32 writeout kernel** (correct) → block-partition + fp32-on-chip (no host de-stride).
- **Source files:** `src/v19_writeout_fp32_dm.cpp` (99, BRISC), `src/v19_writeout_void_ncrisc.cpp` (8), `src/v19_writeout_dm.cpp` (49, uint32 variant).
- **Activated by:** `GATHER_MODE=v19` (the `prog_v19_wo` program in `host/v19_engine.cpp`).

## 2. Problem & contract
**Computation.** Each core owns global bins `[c·slab_bins, (c+1)·slab_bins)` (block partition). Convert its L1 slab `uint32 fixed-point → fp32 = u·INV_SCALE`, then write it into `density_buf[M,N]` (row-major, page = N·4 = one row). **Inputs:** the L1 density slab (post-scatter) + `density_buf` addr + `scale_bits`. **Output:** row-major fp32 density in DRAM. **Invariants:** `slab_bins` multiple of 8 (host pads) so per-row col offsets are 16-B aligned; CB alloc order must match the scatter program so c_24 lands at the same L1 base.

## 3. Mechanism (how it works)
1. **In-place uint32→fp32 convert** of the slab (the uint32 form is done once scatter finished). 2. Walk the core's bin range row-by-row: first/last partial rows + full middle rows, issuing **one `noc_async_write` per row** of `cols·4` bytes. 3. Single barrier (disjoint L1 + disjoint DRAM regions → no hazards).
- **THE TRICK:** block-partition ownership means a core's slab maps to **contiguous rows** of the row-major DRAM density → direct per-row writes, no host de-stride. Combined with on-chip fp32 convert, the ~9 ms/iter host work at 2048 goes to ~0.

## 4. Performance (measured — TT)
- Eliminated the ~9 ms (of ~10 ms) host work at 2048 grid ([[v19_block_fp32_outcome]]). The writeout itself is bandwidth-bound row writes; folded into the V19 forward total measured in `01-forward-scatter/v19-scatter` (no separate per-zone capture yet).

## 5. Correctness
- Bit-stable (the separate-kernel split fixed the 4–8 ULP cross-run races from in-kernel writeout). Part of the 54/54 V19 convergence.

## 6. Gotchas / pitfalls
- **Must be a separate kernel** from scatter (so all atomics land first) — in-kernel writeout races.
- **CB allocation order must match** the scatter program (else c_24 base differs → reads wrong L1).
- `slab_bins` multiple of 8 for alignment.

## 7. When to use / avoid
- **Wins:** always, for V19 — it's what makes V19 host-cheap at scale. **Avoid:** N/A (it's the V19 emit).

## 8. Provenance
- **Memory:** [[v19_block_fp32_outcome]], [[v19_works_all_grids]]
- **Host:** `host/v19_engine.cpp` (`prog_v19_wo`)
