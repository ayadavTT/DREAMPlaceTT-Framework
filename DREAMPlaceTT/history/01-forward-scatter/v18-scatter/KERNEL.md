# V18 scatter — hash-table pre-aggregation density construction

> **TL;DR:** V11's tile-ownership route, but with an on-chip **hash table** that
> pre-aggregates a cell's repeated contributions to the same bin *before*
> routing, collapsing the gather tail. Correct at all grids; reduces gather
> 17–20% but scatter rises and host `kernel_wait` absorbs the chip savings — so
> it was **not a wall-time win** and V19 superseded it. Kept as a validated
> reference for the pre-aggregation idea.

---

## 1. Identity
- **Stage:** forward — density-map construction (scatter + hash pre-agg + route)
- **Status:** ✓ VALIDATED (`GATHER_MODE=v18` / `v18outer`); correct, not a wall win
- **Lineage:** V11 (route+accum) → **V18 (hash pre-agg)** → V19 (L1 atomic-add). V18 was the "aggregate-before-route" experiment between V11 and V19.
- **Source files:** `src/v18_scatter_dm.cpp` (543 ln, NCRISC), `src/v18_scatter_b_dm.cpp` (469 ln, BRISC), `src/v18_outer_scatter_dm.cpp` / `src/v18_outer_scatter_b_dm.cpp` (Stage-B′ outer-product variant).
- **Activated by:** `GATHER_MODE=v18` (or `v18outer`). Built in-process by `host/v19_engine.cpp` (shares the V11 setup path).

## 2. Problem & contract
**Computation.** Identical density forward (`Σ ox·oy → D[M,N]`). V18 differs from V11 only in the route stage: before flushing `(bx,by,area)` tuples to `route_buf`, it **accumulates duplicate (bx,by) hits into a per-core hash table**, so each bin is routed once with its summed area instead of many times.

**Inputs/outputs:** same CB contract as V11/V19 (`bxl,byl,ox[8],oy[8]` from v4_compute; `route_buf` DRAM pages consumed by the accumulate stage). Adds an L1 hash-table region per core.

**Invariants:** hash table must fit L1; collisions fall back to direct emit; same `route_buf` page format as V11.

## 3. Mechanism (how it works)
**Pre-aggregation.** As the scatter loop walks each cell's 8×8 neighborhood, instead of immediately staging every `(bx,by,area)` it hashes `(bx,by)` into a small per-core L1 hash table and does `table[h].area += area`. Periodically (or at flush) the table's nonzero entries are emitted to `route_buf`. This collapses the many-cells-into-one-bin pattern (dense clusters) into far fewer routed tuples → the accumulate stage reads less.

**RISC division of labor:** same BRISC(reader+half)/NCRISC(half) split + same-core handshake as V11. The extra work is the hash insert/aggregate in the inner loop.

**THE TRICK:** move the summation *upstream* of the NoC route. V11 sums on the receiver (accumulate stage); V18 sums partially on the sender (hash). For clustered density, this cuts route_buf traffic and the gather tail dramatically (max iter dropped 11→6 ms at adaptec1_512).

## 4. Performance (measured — TT)
- See `profile/results*.csv` once swept. Memory baseline ([[v18_phase3_outcome]]):
  - adaptec1_512: chip sc+ga −17% mean / −23% median (gather tail collapses), HPWL +0.03% — but **wall 27.8→28.2 s flat** (host `kernel_wait` absorbed the chip savings).
  - adaptec1_2048: gather mean −20%, **scatter +29%**, sc+ga median +6.6%, wall 63≈63 s.
- **Critical path:** the hash insert adds NCRISC work (scatter +29% at 2048), and host kernel_wait dominates so chip savings don't reach wall time.

## 5. Correctness
- HPWL +0.03% vs V11/CPU at 512 (converges). Isolated synthetic: density rel_l2 (see report).

## 6. Gotchas / pitfalls
- Hash table sizing vs L1 budget; collision fallback must stay correct (else dropped mass).
- The win is **distribution-dependent** — clusters benefit (lots of same-bin hits), uniform barely.
- Chip-level gains can be invisible at wall time when host `kernel_wait` / IPC dominate — measure wall, not just chip.

## 7. When to use / avoid
- **Wins when:** density is highly clustered and the accumulate/gather tail is the bottleneck, and you're not host-bound.
- **Avoid:** if host overhead dominates (then chip savings don't surface) or for uniform density. V19's atomic-add achieves the same "aggregate at the owner" effect more cheaply → preferred.

## 8. Provenance
- **Memory:** [[v18_phase3_outcome]]
- **Handoff:** `docs/V18_HASHAGG_HANDOFF.md`
- **Host:** `host/v19_engine.cpp` (V11/V18 shared setup)
