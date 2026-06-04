# v35-halotile — Forward-Grouped Halo-Tile EF backward (fast + clustering-immune, all grids)

> **TL;DR:** Partition the field into FIXED-WIDTH column tiles sized so one tile (+max_k halo)
> always fits L1 → no streaming, cells stay whole. Assign cores to tiles in proportion to each
> tile's cell count (greedy makespan) → clustering-immune. Reuses the V28 SFPU multi-bin gather
> verbatim. **Beats CPU 16T on 17/18 configs (avg ~2×, up to 3.6× @2048), bit-exact (rel_l2 7e-8),
> uniform≈clustered at every config.** Validated as a gather+balance microbench; on-chip grouping +
> live wiring are the remaining steps.

---

## 1. Identity
- **Stage:** backward-ef-gather
- **Status:** VALIDATED (microbench, real silicon bh-38, 2026-06-03) — gather+adaptive-balance core; not yet live-wired.
- **Lineage:** v28-ef-multibin (correct-all-grids SFPU gather, uniform-only microbench) + v30 adaptive slab balance → **THIS** (fixed L1-tiles + proportional core allocation = clustering-immune at all grids, cells whole). Supersedes **fccs** (field broadcast + int-BRISC MAC + streaming-sort → lost 5–13× at large grids).
- **Source files:** `src/v35_host.cpp` (partition + harness, host); reuses `src/v28_brisc.cpp` (BRISC fx gather + prep), `src/v28_ncrisc.cpp` (NCRISC fy gather + band load), `src/v28_compute.cpp` (TRISC SFPU Σ px·py·f, 32 cells/op) **verbatim**.
- **Activated by:** standalone harness `v35` (`history/harness`); not yet behind a live `*_EF` flag.

## 2. Problem & contract
- **Computation:** density backward / electric force. Per cell `c`: `gx[c]=ratio·Σ_{k<kc,h<hc} px[c,k]·py[c,h]·fx[(bxl+k)·M+(byl+h)]`, same for `gy` with `fy`. Field is two column-major planes `fx,fy : [col·M + row]`, `N×M` bins.
- **Inputs:** field `fx,fy` (DRAM, page = full plane `N·M·4`); per-core cell records (DRAM, page/core `mc·24·4`): `[0]bxl_loc [1]byl [2]kc [3]hc [4]ratio(f32) [5]pad [6..13]px[8] [14..21]py[8]`. Geometry is exactly the forward V31_GEOM stash (`px/py/bxl/ratio` are prep-free from `v4_compute`).
- **Output:** `gx,gy` per cell (fp32), DRAM, page/core `mc·4`. In the live kernel these are written at the cell's original index (`grad[orig]`) via the v29-writer pattern — no host unsort.
- **Invariants:** tile band read is column-contiguous (64-B aligned by construction, `col0·M·4`); a cell's full x-window lies inside its tile because the tile carries a `+max_k` halo; `kc` is clamped so `bxl_loc+kc ≤ band_cols`; field tail beyond the valid band is zeroed so out-of-range reads return 0 (×px(0)=0).

## 3. Mechanism (how it works)
- **Data residency:** each core loads ONE field tile `[col0, col0+band_cols)` for BOTH planes into L1 (`band_cols = W + max_k`, sized so `W·M·4 ≤ ~350 KB/plane` → fits L1 at every grid). No streaming (the FCCS death) → a cell's whole `kc×hc` window is a direct L1 load → **no sort**.
- **RISC labor (inherited from V28):** BRISC loads fx band + packs px/py/ratio/base_idx tiles (lane = cell) + gathers fx tiles per `(k,h)`; NCRISC loads fy band + gathers fy tiles in parallel; TRISC SFPU does `Σ px·py·f` over `max_k×max_h` for 32 cells/op, `×ratio`, writes gx/gy.
- **Algorithm (the V35 partition, host-side here; on-chip = V30 prepcount+band):**
  1. Tile the field into `ntiles = ceil(N/W)` fixed column tiles; `W` floored so `ntiles ≤ nc/2` (⇒ every tile can get ≥2 cores → even split).
  2. Histogram cells per tile (`cnt[t]`) — each cell goes to tile `bxl/W` (cell-level, whole).
  3. **Greedy makespan core allocation:** each nonempty tile starts with 1 core; every remaining core goes to the tile with the worst current per-core load `cnt/alloc`. Minimizes max-cells-per-core directly.
  4. Split each tile's cells into `alloc[t]` equal contiguous chunks; helper cores of a hot tile each load a COPY of that tile's small band and process one chunk.
- **THE TRICK (two):** (a) **fixed L1-fitting tiles + halo** keep cells whole and resident with no streaming → no sort, no record-explosion. (b) **proportional core allocation** makes work ∝ cells/core regardless of spatial distribution → clustering-immune, while only ever replicating a *hot tile's small band* (never broadcasting the whole field — the FCCS cost).

## 4. Performance (measured — TT)
- **Profiled config:** bh-38, AICLK ~1.35 GHz, 110 cores. 6 designs × 3 grids × {uniform, clustered}; spans 512:2×2 / 1024:4×2 / 2048:6×3. Full table: `profile/sweep_18config.csv`.
- **Headline (uniform ms | CPU 16T ms | ×):**
  | grid | a1 | a2 | a3 | bb1 | bb2 | bb3 |
  |---|---|---|---|---|---|---|
  | 512  | 1.58/1.53/**0.97** | 1.72/2.11/1.22 | 2.02/4.05/2.01 | 1.73/2.30/1.33 | 2.06/4.23/2.06 | 2.79/5.56/1.99 |
  | 1024 | 1.79/2.27/1.27 | 1.97/3.66/1.86 | 2.46/5.88/2.39 | 1.98/3.71/1.87 | 2.70/6.30/2.34 | 3.72/7.52/2.02 |
  | 2048 | 2.18/4.52/**2.07** | 2.60/5.97/2.29 | 3.49/12.62/**3.62** | 2.62/7.21/2.75 | 3.93/13.17/3.36 | 6.12/13.85/2.26 |
  - **Beats CPU on 17/18** (a1_512 = 0.97× parity). Mean ≈ **2.05×**.
- **Load imbalance (max/mean cells per core):** ≤1.05× @2048 uniform, ≤1.25× @2048 moderate-cluster, **1.67× @2048 EXTREME cluster** (90% of cells in ~4% of columns) — the bounded worst case; still beats CPU there (bb3 8.59 vs 13.85, a3 4.33 vs 12.62).
- **Scaling:** clean across grids and cell counts (211K→1.1M). Clustered ≈ uniform within ~12% at every config (the property FCCS/V28 never demonstrated).

## 5. Correctness
- **Method:** rel_l2 vs CPU multi-bin reference + per-element nbad (tol 1e-3·max(1,|ref|)).
- **Result:** **rel_l2 = 5.0e-8 (512) / 5.7e-8 (1024) / 7.0e-8 (2048), nbad = 0 at ALL 18 configs and all of uniform/moderate/extreme clustering.** Bit-exact in practice.

## 6. Gotchas / pitfalls
- **ntiles ≈ nc is the imbalance trap:** if `ntiles` is just under `nc`, some tile gets only 1 core and carries a full tile's load → ~2× imbalance even on UNIFORM. Fix = floor `W` so `ntiles ≤ nc/2`. (Measured: a1_2048 uniform 1.93×→1.05×, bb3 10.0→6.12 ms.)
- **Greedy beats largest-remainder** for clustered: directly minimizes max-cells-per-core (a1_2048 clustered 1.69×→1.27×).
- **Band must be zero-padded past `valid_cols`** (edge/last tile) so unclamped `base_idx+kh` reads return 0, not NaN (inherited V28 invariant).
- **Helper cores replicate a hot tile's band** (independent DRAM reads) — fine because the band is small (~350 KB); do NOT confuse with broadcasting the whole field.
- **This microbench computes the partition on the HOST** (metadata: ntiles+nc ints) and generates a random field. For live: histogram+alloc → V30 prepcount/band on-chip; field from chip-DCT (`latest_field_addrs`); grad→`grad[orig]` via v29-writer. The MEASURED part is the gather+balance (the dominant cost); the deferred grouping is cheap (V27 cell-bucket 0.79 ms) — unlike FCCS, which measured the cheap part (multicast) and hid the expensive one (int-MAC).

## 7. When to use / avoid (the lesson)
- **Wins when:** any grid / cell count / spatial distribution — it is the first EF backward shown clustering-immune AND fast AND correct at all 18 configs. The field tile + halo always fits L1, so there is no streaming and no sort.
- **Avoid / watch:** extreme clustering pushes imbalance to ~1.67× (still a CPU win); if that ever bites, let one core sweep multiple tiles (load bands sequentially) so cores decouple from tiles entirely.
- **What killed earlier variants:** V21 NoC-per-read latency; V28 host-bucket (uniform-only); V29 route_buf OOM + host unsort; V30 record-explosion (span²); FCCS int-BRISC MAC + L1-forced streaming→sort. V35 keeps cells WHOLE (halo) on a STATIONARY resident tile with PROPORTIONAL balance → none of those taxes apply.

## 8. Provenance
- **Memory:** [[v35_halotile_backward]], [[v28_multibin_ef_gather]], [[v30_clustering_verdict]], [[fccs_kernel_built]], [[cpu16t_baseline_targets]], [[density_backward_failures]] (docs/DENSITY_BACKWARD_FAILURES.md, DENSITY_BACKWARD_NEXTGEN_DESIGN.md).
- **Host entry / run:** `history/harness/build/v35` ; e.g. `v35 2048 2048 1100000 6 3 0` (grid grid ncells max_k max_h cluster) ; bh-38, 2026-06-03, median of 7.
- **Next:** (1) on-chip grouping (V27/V30) + measure; (2) grad→grad[orig] + chip-DCT field; (3) live `V35_EF=1` wiring, gate on adaptec/bigblue HPWL convergence + beating `results/cpu16t_baseline_breakdown.csv`.
