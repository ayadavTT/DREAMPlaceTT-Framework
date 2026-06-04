# v20-permute — chip-permute / block-partition microbench kernels (INFRA)

> **TL;DR:** The V19 "microbench" (`v19_mb_*`) kernels: a simplified atomic-scatter
> (one atomic per cell) + block-partition + dense-writeout variants used to
> characterize V19's atomic scatter under controlled distributions, and to
> explore a chip-side permute (V20). The V20 fastpath shipped via host OMP; the
> chip-permute kernel stayed research.

---

## 1. Identity
- **Stage:** infra — V19 atomic-scatter microbench + V20 chip-permute experiments
- **Status:** ◐/🔧 (microbench harness for V19; V20 chip-permute is research — host OMP fastpath shipped instead).
- **Source files:** `src/v19_mb_scatter_ncrisc.cpp`, `v19_mb_scatter_block_ncrisc.cpp`, `v19_mb_combined_ncrisc.cpp`, `v19_mb_writeout_dense_brisc.cpp`, `v19_mb_zero_ncrisc.cpp`, `v19_mb_void_brisc.cpp`, `v19_mb_permute_{brisc,ncrisc}.cpp`, `v19_mb_permute_common.h`.
- **Activated by:** `host/v19_microbench_host.cpp` (target `v19_microbench_host`), `host/v20_mb_permute_host.cpp` (target `v20_mb_permute_host`).

## 2. Problem & contract
Characterize V19's atomic-add scatter in isolation: synthetic `(bx,by,area)` triplets (one atomic per cell) across distributions (cluster/uniform/hotcold/origin/low_local/high_local) × grids × cell counts × modes (split/combined/dense/block). Validates atomic sums vs a host model; surfaces Tracy zones (`V19MB-EMIT-BATCH`, `V19MB-BARRIER`, …).

## 3. Mechanism / findings
- **Distribution probes** isolated where atomics fail/slow: e.g. `high_local` diagnosed high-L1-offset atomic correctness at 2048; `origin`/`low_local` probed L1-address-range hypotheses.
- **Block-partition mode** (`v19_mb_scatter_block_ncrisc`) validated that contiguous-block ownership lets writeout dump row-major DRAM directly (no host de-stride) at ~+1 ms contention cost — the basis for the production V19 writeout.
- **V20 chip-permute** explored moving the per-iter cell permute on-chip; the shipped V20 used a **host OMP fastpath** instead (chip-permute remained research, see handoff).

## 4–7. Performance / correctness / gotchas / when
- These are the controlled-experiment kernels behind V19's design decisions (block partition, alignment, distribution robustness). The production V19 scatter (`01-forward-scatter/v19-scatter`) is the real kernel; this is its characterization harness.
- Note: this microbench feeds single-bin `(bx,by,area)` triplets, so it tests the atomic mechanism, not the full j,k overlap loop (that's the production kernel).

## 8. Provenance
- **Memory:** [[v20_chip_permute_handoff]], [[v19_block_fp32_outcome]], [[blackhole_dram_read_align_64]]
- **Handoff:** `docs/V20_CHIP_PERMUTE_HANDOFF.md`
- **Host:** `host/v19_microbench_host.cpp`, `host/v20_mb_permute_host.cpp`
