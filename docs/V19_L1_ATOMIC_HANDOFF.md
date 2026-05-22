# V18 hash-aggregation + V19 L1-atomic gather — handoff

Date: 2026-05-22 (single session by agent A)
Branch: main (uncommitted; see "Files changed" below)

This document is a complete handoff so another agent can pick up V19 (and the V18 ideas it builds on) and finish testing it across all 18 sweep configs. Read it end-to-end before touching code.

---

## Where things stand

| Path | Status | Bench result |
|---|---|---|
| V18 hash-agg (`GATHER_MODE=v18`) | **shipped** earlier in the week (per memory `v18_phase3_outcome.md`) | adaptec1_512 chip sc+ga −17% mean, HPWL +0.03%; **wall flat 27.8→28.2 s** because host `kernel_wait` absorbs chip savings. adaptec1_2048 gather −20%, scatter +29%, wall flat. |
| V18-outer (V18 hash + V11outer SFPU outer-product) | prototyped, sibling kernels exist | not a clean win at 512 (gather goes up because outer-product CBs steal L1) — kept as a knob, not the default. |
| V19 L1-atomic (`GATHER_MODE=v19`) | **working on adaptec1_512, blocked on 2048** | 512: HPWL 70,373,920 (+0.06 % vs V11 70,332,032), bit-exact deterministic across runs, wall 24.5–26.7 s vs V11 27.8 s. 2048: kernel hangs at JIT-warmup of V11_scatter program (which runs V19 scatter kernel for the first time); see "Open issues". |
| Atomic microbench (`host/build/atomic_bench_host`) | written + characterized | L1 `noc_semaphore_inc` works (~15–534 ns / op, contention-scaled). DRAM atomic NOT supported (hangs). Fixed-point fp32 pattern validated. |

Memory notes added in this session:
- `memory/v19_l1_atomic_converges.md` — the two bugs that made V19 work on 512.
- (`memory/v18_phase3_outcome.md` and earlier V18 notes are already in MEMORY.md.)

---

## V18 — recap (read this first; V19 reuses concepts)

The full design is in `docs/V18_HASHAGG_HANDOFF.md`. One-paragraph recap:

**Idea**: V11's gather is dominated by 4 mega-tiles on 2048 grids that receive 600k–1.36M tuples each. Pre-aggregate per-source: each scatter kernel (BRISC + NCRISC) keeps an L1 hash table keyed on `(bx<<16)|by` with `area` accumulator; on end-of-batch it flushes only the unique entries via the existing V11 route_buf. Wire format identical, gather kernels untouched.

### V18 vs V11 — all 18 sweep configs (2026-05-21 sweep data)

Both **V11 and V18 converge on all 18/18 configs** (overflow < 0.07 on every run). V18 saves wall time on 15/18 configs (mean −5.1 s/config). Chip sc+ga is essentially flat (mean +0.15 ms): wall savings come from DREAMPlace needing slightly fewer iterations when fed V18's cleaner (no silent-drop) density.

**Exact V18 configuration used in every row of the table** (confirmed by `grep "V18:" run.log` on each result dir):
```
V18_HASH_BITS = 16      (65536 slots per RISC)
V18_BF16_AREA = 1       (auto-default at hb>=16; 6-byte slots = 4-byte key + 2-byte bf16 area)
V18_NO_DIRTY  = 0       (dirty list ENABLED — flush walks only touched slots, not full table)
```
L1 footprint per core: 1024 KB total for V18 hash tables — `table_per_risc=384 KB` (65536 × 6) + `dirty_per_risc=128 KB`, times 2 RISCs. `scripts/run_v18_sweep.sh` does not override these env vars, so the sweep uses the host's auto-default for `V18_BF16_AREA` (= 1 when `V18_HASH_BITS >= 16`). hb=15 + fp32 area + dirty list also works on 17/18 configs but historically hung on at least one bigblue 2048 config until the mid-batch drain + 64-probe limit fix landed; hb=16 + bf16-area is the "always-safe" config the operator picked for the converged sweep.

| config | V11 wall | V18 wall | Δ wall | V11 sc+ga med | V18 sc+ga med | Δ sc+ga | V18 conv | V18 HPWL drift |
|---|---:|---:|---:|---:|---:|---:|:---:|---:|
| adaptec1_512  | 26.6 s | 27.5 s | +0.9 s | 4.76 ms | 5.53 ms | +0.77 ms | ✓ | +0.03 % |
| adaptec1_1024 | 41.5 s | 38.5 s | **−3.0 s** | 8.95 ms | 8.75 ms | −0.20 ms | ✓ | +0.11 % |
| adaptec1_2048 | 66.7 s | 65.3 s | −1.4 s | 13.24 ms | 14.93 ms | +1.69 ms | ✓ | −0.07 % |
| adaptec2_512  | 38.0 s | 34.8 s | **−3.2 s** | 5.38 ms | 5.26 ms | −0.12 ms | ✓ | +0.13 % |
| adaptec2_1024 | 44.0 s | 41.0 s | **−3.0 s** | 8.32 ms | 7.69 ms | −0.63 ms | ✓ | +0.39 % |
| adaptec2_2048 | 82.6 s | 81.4 s | −1.2 s | 15.42 ms | 14.06 ms | **−1.36 ms** | ✓ | −1.24 % |
| adaptec3_512  | 70.9 s | 57.2 s | **−13.7 s** | 7.35 ms | 7.76 ms | +0.41 ms | ✓ | +0.02 % |
| adaptec3_1024 | 80.7 s | 72.5 s | **−8.2 s** | 10.22 ms | 10.32 ms | +0.10 ms | ✓ | +0.04 % |
| adaptec3_2048 | 148.7 s | 117.0 s | **−31.7 s** | 17.48 ms | 18.31 ms | +0.83 ms | ✓ | +0.15 % |
| bigblue1_512  | 40.2 s | 39.7 s | −0.5 s | 6.27 ms | 7.06 ms | +0.79 ms | ✓ | −0.05 % |
| bigblue1_1024 | 69.7 s | 66.1 s | −3.6 s | 7.67 ms | 8.60 ms | +0.93 ms | ✓ | −2.87 % |
| bigblue1_2048 | 93.6 s | 101.7 s | **+8.1 s** | 18.27 ms | 19.75 ms | +1.48 ms | ✓ | −0.03 % |
| bigblue2_512  | 80.7 s | 69.8 s | **−10.9 s** | 7.83 ms | 8.08 ms | +0.25 ms | ✓ | −0.06 % |
| bigblue2_1024 | 87.6 s | 76.2 s | **−11.4 s** | 12.21 ms | 11.87 ms | −0.34 ms | ✓ | +0.34 % |
| bigblue2_2048 | 131.5 s | 127.9 s | −3.6 s | 20.63 ms | 21.22 ms | +0.59 ms | ✓ | +0.05 % |
| bigblue3_512  | 127.9 s | 125.7 s | −2.2 s | 10.54 ms | 10.80 ms | +0.26 ms | ✓ | −0.06 % |
| bigblue3_1024 | 139.7 s | 143.0 s | +3.3 s | 16.61 ms | 13.88 ms | **−2.73 ms** | ✓ | −0.02 % |
| bigblue3_2048 | 174.0 s | 167.2 s | −6.8 s | 22.74 ms | 22.64 ms | −0.10 ms | ✓ | +0.03 % |

**Aggregate stats**:
- Convergence: V11 18/18, V18 18/18 (every overflow < 0.07; HPWL drift |Δ| < 3 % everywhere, mostly < 0.5 %).
- Wall: V18 better on 15/18, mean Δ = **−5.11 s/config**. Biggest wins: adaptec3_2048 (−31.7 s), adaptec3_512 (−13.7 s), bigblue2_1024 (−11.4 s), bigblue2_512 (−10.9 s), adaptec3_1024 (−8.2 s). Only loss: bigblue1_2048 (+8.1 s).
- Chip sc+ga: V18 better on 7/18, mean Δ = +0.15 ms (essentially flat). The wall win is *not* a chip win — V18's tighter density just makes DREAMPlace settle slightly faster.

### V18-outer (V18 hash + V11outer SFPU outer-product) — 5 configs only

Only ran on the five large 2048-grid configs where V11outer was already beneficial: adaptec2/3_2048, bigblue1/2/3_2048. All 5 converged (HPWL drift < 0.7 %). Not generally a wall-time win — V11outer alone usually beats V18-outer, since the hash table competes with the outer-product CBs for L1.

### V18 result sources (for reproducibility)

| Sweep timestamp dir | Configs it covers |
|---|---|
| `results/v18_sweep_20260521_1523/` | adaptec1/2/3 + bigblue1 (×3 grids each) — 12 configs |
| `results/v18_sweep_20260521_1717/` | bigblue2 set (×3 grids) — 3 configs |
| `results/v18_sweep_20260521_1800_finish/` | bigblue3 set + adaptec1_512 re-run — 4 configs |
| `results/v11outer_sweep_20260520_2204/` | V11 canonical reference baseline — 18 configs |
| `results/v18outer_2048_1959/` | V18-outer — 5 configs (2048 only) |

Table reproduction: `/tmp/v18_table.py` (uses `v11outer_sweep_20260520_2204` as the V11 reference so it doesn't accidentally pin to a corrupted V11 re-run from inside a later V18 sweep dir).


**Kernels added**:
- `kernels/v18_scatter_b_dm.cpp` (BRISC), `kernels/v18_scatter_dm.cpp` (NCRISC) — sibling V11 copies with the hash-table insert + dirty list + mid-batch drain
- `kernels/v18_outer_scatter_b_dm.cpp`, `kernels/v18_outer_scatter_dm.cpp` — combined V11outer outer-product SFPU + V18 hash dedup

**Knobs**:
- `V18_HASH_BITS` (default 15 → 32 K slots; 16 needs no-dirty + bf16-area to fit L1)
- `V18_BF16_AREA` (default 0; 1 stores area as bf16 → 6-byte slots)
- `V18_NO_DIRTY` (default 0; 1 skips the dirty-list and walks the full table on flush)

**Bugs fixed (already in main)**:
1. **bigblue1_2048 hang** — hash table at hb=15 filled completely on hot bins → infinite probe loop. Fix: mid-batch drain when dirty count exceeds threshold + 64-probe limit per insert.
2. **L1 overflow at hb=16 + fp32 + dirty** for V18outer — 1.98 MB > 1.5 MB budget. Fix: defaulted to no-dirty + bf16 area when hb≥16.
3. **`env_uint` treats `"0"` as unset**, so `V18_NO_DIRTY=0` was using the default `=1`. Fix: added `env_bool` lambda using `s[0] != '0'`.
4. **`V18_NO_DIRTY` not in client docker-exec allowlist** in `integration/scatter_ttnn_client.py` — env var silently dropped. Fix: added it (along with `V19_SCALE_BITS`, `DENSITY_DUMP_*`).

**Open V18 problem**: chip sc+ga drops 17 % but `kernel_wait` on the host eats the saving. Profiling on Tracy showed the host-side dispatcher is the bottleneck; until the dispatcher is fixed, V18 isn't a wall-time win on this codebase.

---

## V19 — the architecture

V19's hypothesis: skip the gather phase entirely by doing the bin reduction *during scatter*, using L1 atomic increments.

### Wire format and ownership

There is no `route_buf`. Density bins are partitioned strided across all 110 cores:
```
bin_global = bx * nby + by      // 0 … nbx*nby - 1
owner_idx  = bin_global % nc_all // which core owns this bin
local_idx  = bin_global / nc_all // its slot inside that core's L1 slab
```
Each core holds an L1 slab of `ceil(nbx*nby / nc_all)` `uint32` slots (fixed-point density). The slab lives in `CB_SCRATCH` (`c_24`) at offset `density_l1_off` (computed by host so it lands after the existing V11 scratch).

### The atomic op

`noc_semaphore_inc(target_l1_noc_addr, value_uint32)` — a hardware-level "L1 ACC AT" packet. It atomically adds `value` to the 32-bit word at the target L1 address (any core's L1, addressed via NoC). Determined empirically: integer add is bit-exact regardless of NoC arrival order. DRAM is NOT supported (verified by hang in microbench).

Each (j,k) overlap inside a cell emits:
```cpp
float    area_fp32  = ox_j * oy_k;                       // already pre-scaled by inv_bin_area (G-PRESCALE)
uint32_t area_fixed = (uint32_t)(area_fp32 * (1<<SCALE)); // SCALE = V19_SCALE_BITS (default 20)
if (area_fixed == 0u) continue;                          // drops contributions below 1 ULP of fixed-point
uint32_t bin_global = bx_val*nby + by_val;
uint32_t owner_idx  = bin_global % nc_all;
uint32_t local_idx  = bin_global / nc_all;
uint32_t packed_xy  = noc_coords[owner_idx];
uint32_t owner_x    = packed_xy & 0xFFFFu;
uint32_t owner_y    = packed_xy >> 16;
uint64_t target = get_noc_addr(owner_x, owner_y, (uint32_t)density_l1 + local_idx*4u);
noc_semaphore_inc(target, area_fixed);
```

`noc_coords[]` is a small per-core lookup table loaded once at kernel entry from a 1-page DRAM buffer. Packed `(noc_y<<16)|noc_x`.

### Two-kernel split (critical)

Scatter and writeout must be **separate programs**, run sequentially:

```
EnqueueMeshWorkload(wl_v11_scatter);   Finish(cq);   // V11 scatter program runs V19 scatter kernel
EnqueueMeshWorkload(wl_v19_writeout);  Finish(cq);   // ↑ all atomics across all 110 cores have landed
EnqueueReadMeshBuffer(v19_density_dram);             // strided readback
de-stride into v19_dense, EnqueueWriteMeshBuffer(density_buf)
```

Why two kernels: `noc_async_atomic_barrier()` waits only on this core's *outgoing* atomics, not on *incoming* ones from other cores. Doing the DRAM writeout at the end of the scatter kernel races with still-arriving atomics → tt_density.sum is non-deterministic and HPWL non-reproducible. Splitting into a 2nd kernel uses the host `Finish(cq)` boundary as the cross-core barrier (every core's outgoing atomics flushed at kernel exit).

### Files

| File | Role |
|---|---|
| `kernels/v19_scatter_dm.cpp` | NCRISC scatter — zeros own L1 density slab, loads `noc_coords[]`, signals tables_ready, then per-tile emits atomic-incs |
| `kernels/v19_scatter_b_dm.cpp` | BRISC scatter — V11-style DRAM reader + atomic emitter for j-overlaps owned by BRISC |
| `kernels/v19_writeout_dm.cpp` | BRISC-only writeout kernel — reads own L1 slab, chunked `noc_async_write` to DRAM staging (one page per core) |
| `kernels/v19_writeout_void_ncrisc.cpp` | empty NCRISC pair for the writeout program |
| `kernels/atomic_bench_brisc.cpp` / `..._ncrisc.cpp` + `host/atomic_bench_host.cpp` | microbench, NOT in the production path |

### Host integration (in `host/density_scatter_ttnn_server_host.cpp`)

- `gather_mode_env == "v19"` sets `use_v11=true; use_v19=true; use_v11outer=false; use_v18=false;`. V19 reuses the V11 scatter program (same MeshWorkload), just with V19 kernels.
- V19 setup block (≈ lines 1285–1363): computes `v19_density_l1_off`, allocates `v19_coord_buf` (one DRAM page of 110 packed `(noc_y<<16)|noc_x`) and `v19_density_dram` (110 strided pages of slab bytes), and builds `wl_v19_writeout` as a separate `MeshWorkload`.
- The writeout program's CB layout MUST mirror the scatter program's CBs c_0…c_21 with the same sizes BEFORE c_24, otherwise `get_write_ptr(CB_SCRATCH)` returns a different L1 base in writeout vs scatter → writeout reads zeros.
- Per-iter dispatch (≈ line 2916): after scatter `Finish`, enqueue writeout + `Finish`, then read the strided DRAM buffer, de-stride on host into `density_buf`, write that back. (De-stride is currently host-side; on-chip de-stride is a Phase 2 optimization.)
- New env vars (also added to the client allowlist in `integration/scatter_ttnn_client.py`):
  - `V19_SCALE_BITS` (default 20) — fixed-point shift; bigger = finer resolution but smaller dynamic range before uint32 overflow.
  - `DENSITY_DUMP_ITER` + `DENSITY_DUMP_PATH` — dumps `density_flat` to a binary file on the given iter for diff testing.

---

## The four V19 bugs we found this session (all fixed)

These took the bulk of the session. The first three you can read about in `memory/v19_l1_atomic_converges.md`; the fourth was discovered after that memory was written.

### Bug 1 — non-deterministic atomics

**Symptom**: two back-to-back V19 runs on adaptec1_512 produced different HPWL (96 M vs 104 M at SCALE=2^18; later 96 M vs 104 M at 2^20).

**Root cause**: the NCRISC scatter kernel originally ended with `noc_async_atomic_barrier()` then wrote its own L1 slab to DRAM. The barrier only waits on THIS core's outgoing atomics, not on incoming ones. Other cores' atomic-incs were still landing on this core's L1 while it was being DRAM-streamed out.

**Fix**: moved the DRAM writeout into a separate kernel (`v19_writeout_dm.cpp` + paired void NCRISC). The host `Finish(cq)` between scatter and writeout serves as the cross-core barrier — every core's kernel exit flushes its outgoing atomics.

### Bug 2 — CB layout mismatch between scatter and writeout programs

**Symptom**: after applying Bug-1 fix, V19 was deterministic but HPWL plateaued at 79.79 M (vs V11 70.34 M) with overflow stuck at 0.81. `density_flat.sum()` was identically zero on the host audit — DREAMPlace was running only on `initial_density_map`, no movable density.

**Root cause**: the writeout program initially created only `c_24`. The scatter program creates 22 CBs (`c_0..c_3` of 2 tiles fp32 each, then `c_4..c_5`, then `c_6..c_13` and `c_14..c_21`) BEFORE `c_24`. CB L1 base addresses are allocated sequentially, so c_24's L1 base differed between the two programs. `get_write_ptr(CB_SCRATCH)` returned different addresses → writeout read zeros from the wrong location.

**Fix**: writeout program now creates all 22 CBs first with the same sizes, then `c_24`. See `make_cb_all(prog_v19_wo, …)` loop in host (lines ~1334–1340).

**Diagnostic for any future occurrence**: print `tt_density.sum()` on the host. If it's identically zero across all iters while HPWL still moves (because `initial_density_map` is added back on the Python side), the writeout is reading the wrong L1 region.

### Bug 3 — `noc_async_write` > ~96 KB silent corruption

**Symptom**: V19 on adaptec1_2048 hung the chip on its very first run. Subsequent runs (even V11 baseline) misbehaved until container restart.

**Root cause**: at 2048 grid the density slab is ~149 KB per core. The original writeout did a single `noc_async_write(L1, dram, 152544)`. This exceeds the safe limit on Blackhole and silently corrupts NoC state (same gotcha noted in `memory/v15_spill_pgsz_bug.md`: V15 spill > ~96 KB corrupts).

**Fix**: chunked the writeout into pieces of ≤ 64 KB:
```cpp
constexpr uint32_t MAX_CHUNK_BYTES = 64u * 1024u;
for (uint32_t off = 0; off < total_bytes; off += MAX_CHUNK_BYTES) {
    uint32_t chunk = total_bytes - off;
    if (chunk > MAX_CHUNK_BYTES) chunk = MAX_CHUNK_BYTES;
    noc_async_write(l1_addr + off, page_base + (uint64_t)off, chunk);
}
noc_async_write_barrier();
```

### Bug 4 — unaligned `coords_off`

**Symptom**: V19 on 2048 STILL hung at JIT-warmup of the V11 scatter program after Bug 3's fix. Same chip-state-poisoning behavior as before.

**Root cause**: in the V19 NCRISC and BRISC kernels, the offset of the `noc_coords[]` table inside `CB_SCRATCH` was computed as:
```cpp
const uint32_t coords_off = density_l1_off + density_slab_bins * 4u;
```
For 512 grid: `density_slab_bins=2384` → `2384*4=9536` mod 32 = 0. Coincidentally 32-aligned. Worked.
For 2048 grid: `density_slab_bins=38131` → `38131*4=152524` mod 32 = 12. **Unaligned**. `noc_async_read` into an L1 destination that isn't 32-byte aligned silently hangs on Blackhole.

**Fix**: align in both kernels:
```cpp
const uint32_t coords_off =
    (density_l1_off + density_slab_bins * 4u + 31u) & ~31u;
```
Host already allocates enough padding (the `v11_sc_scratch` round-up to 32 covered it).

---

## What's left — V19 on 2048 still hangs after the four fixes

This is the open challenge for the next agent.

### Current symptom (2026-05-22 evening)

After Bugs 1–4 are all fixed in the tree, running V19 on `sweep_adaptec1_2048.json` reaches the JIT-warmup phase and prints:
```
[server]   V11 hist JIT...
[server]   V11 scatter JIT...
```
…then never gets to `V11 accum JIT...`. The chip is hung. JIT itself completes (ELFs are written to `~/.cache/tt-metal-cache/.../v19_scatter_*/.../*.elf` shortly after the print) — the hang is **on-chip kernel execution**, not compilation.

After the hang, even V11 baseline on 2048 misbehaves: it runs but gather time inflates from ~7 ms to ~23 ms, and HPWL diverges around iter 400 to ~1.1 B (normal: 70 M). Container restart is needed to clear it.

### What we tried

- `tt-smi -r 0` (multiple times) — clears firmware but doesn't fully purge stale NoC packets/L1 state.
- 30 s settling wait after reset — not enough.
- Container restart (`docker restart bh-38-special-…`) — clears stale resources; chip is reusable for V11 baseline (mostly).
- V19 on 512 still works after all four fixes — the alignment fix is a no-op for 512 (2384 was already aligned). So the kernel + host changes are not breaking 512.

### Likely remaining bug (untested hypothesis)

The fact that the kernel hangs **on first launch** (warmup, before any Python data has arrived) and at exactly the same point on every retry strongly suggests another shape-dependent bug in the V19 kernels that only fires at the 2048 sizes:

- `n_tiles_per_core` at 2048 is 3–4, vs 1–2 at 512. More tiles → more iterations of the per-tile loop. The CB allocation is 2 slots; if a sync semaphore is mis-counted you'd deadlock on tile 3 (not tile 1).
- `density_slab_bins=38131` at 2048 vs 2384 at 512. The zero-init loop is 16× longer. If there's any unsigned-overflow or L1 boundary check tied to `density_slab_bins`, it kicks in at 2048 only.
- `nbx*nby` = 4 194 304 at 2048 vs 262 144 at 512. The `bin_global` calculation fits in u32 either way (max bin index 4 M < 2^32), but the `% nc_all` and `/ nc_all` are non-trivial division ops — if there's a u32-rollover risk it'd happen here.

Suggested debugging path (in priority order):
1. **Skip the JIT-warmup of `wl_v11_scatter` when `use_v19=true`** — let the first real iter trigger compile+run. This will tell us whether the hang is in the warmup-specific zero-data case or universal. The print prints are already in place (search for `V11 hist JIT...` in host); just gate the `EnqueueMeshWorkload(wl_v11_scatter)` JIT call on `!use_v19`.
2. **Add `DeviceZoneScopedN` around every section of the V19 NCRISC kernel** (V19N-INIT, V19N-CB-WAIT, V19N-ATOMIC) and Tracy-profile the warmup to see which section hangs. The zones are already in the kernel; you just need to run with `TT_METAL_DEVICE_PROFILER=1` and tools/v19_*.py extraction.
3. **Test V19 with `density_slab_bins` artificially clamped to ≤ 2400** by setting `nc_all` higher for the slab calc only. If V19 then converges at 2048, the bug is in the large-slab handling.
4. **Try `n_tiles_per_core=0` warmup** — force my_n=0 on the warmup launch via SetRuntimeArgs override. If that also hangs, the bug is in the kernel's Step-0 init (zero loop or noc_coords read). If it doesn't, the bug is in the per-tile loop.

### One more hypothesis: the device may itself have been left in a bad state

The user noted toward the end of the session: "the device was getting rebooted so it is possible that there was some corruption." If the chip is mid-recovery when V19 2048 launches, the symptoms (V11 baseline divergence, gather 3× slower, V19 hang on first kernel) could be downstream of that, not a V19 bug. **Before assuming a V19 kernel bug, restart the docker container fully, then run V11 baseline 2048 first and confirm HPWL≈70 M and gather≈7 ms**. If V11 isn't healthy, fix the chip state first; only then is the V19 2048 hang diagnostic.

---

## Files changed in this session (uncommitted)

```
M REPRODUCE.md                                  (small)
M benchmarks/configs/*.json                     (sweep config plumbing — most are touch-only)
M docs/V11_V13_GATHER_HANDOFF.md                (memory updates)
M docs/V13_FPU_UNLEASH_HANDOFF.md               (memory updates)
M host/CMakeLists.txt                           (atomic_bench_host target)
M host/density_scatter_ttnn_server_host.cpp     (V19 setup, writeout MeshWorkload, JIT prints, density-dump hooks)
M host/v11_tile_ownership.h                     (small)
M integration/run_dreamplace.py                 (small)
M integration/scatter_ttnn_client.py            (V19_SCALE_BITS + DENSITY_DUMP_* allowlist; density_audit moved BEFORE initial_density_map addback)
M kernels/v11_accum_*.cpp                       (G-PMERGE residual)
M kernels/v13_*.cpp                             (unrelated)
M scripts/build_server.sh                       (small)
M tools/v11_phase2_smoke.py                     (small)
?? kernels/v18_scatter_b_dm.cpp                 (V18, shipped)
?? kernels/v18_scatter_dm.cpp                   (V18, shipped)
?? kernels/v18_outer_scatter_b_dm.cpp           (V18-outer)
?? kernels/v18_outer_scatter_dm.cpp             (V18-outer)
?? kernels/v19_scatter_b_dm.cpp                 (V19 BRISC)
?? kernels/v19_scatter_dm.cpp                   (V19 NCRISC)
?? kernels/v19_writeout_dm.cpp                  (V19 writeout BRISC)
?? kernels/v19_writeout_void_ncrisc.cpp         (V19 writeout NCRISC pair)
?? kernels/atomic_bench_brisc.cpp               (microbench)
?? kernels/atomic_bench_ncrisc.cpp              (microbench)
?? host/atomic_bench_host.cpp                   (microbench)
... (many config + tool files, all small)
```

The atomic microbench and V18 sibling kernels are independent of V19 and can be committed separately.

---

## Reproduction recipes

### V19 on adaptec1_512 (known good)
```bash
TT_METAL_HOME=$(pwd)/tt-metal CPU_DCT=1 GATHER_MODE=v19 V19_SCALE_BITS=20 \
  DREAMPlace/dp_env/bin/python3 integration/run_dreamplace.py \
    --device scatter_ttnn \
    --container bh-38-special-ayadav-for-reservation-75063 \
    --benchmark benchmarks/configs/sweep_adaptec1_512.json \
    --results-dir /tmp/v19_512
```
Expect: HPWL = 70,373,920 (bit-exact across runs), wall 24–27 s, scatter ≈ 2.3 ms, gather ≈ 2.7 ms.

### V19 on adaptec1_2048 (currently hangs at JIT-warmup of V11 scatter)
Same command, swap `sweep_adaptec1_512.json` → `sweep_adaptec1_2048.json`.

If the chip is misbehaving:
```bash
docker exec bh-38-special-ayadav-for-reservation-75063 bash -c "ps -ef | grep density_scatter_ttnn_server | grep -v grep | grep -v defunct | awk '{print \$2}' | xargs -r kill -KILL"
docker restart bh-38-special-ayadav-for-reservation-75063
docker exec bh-38-special-ayadav-for-reservation-75063 bash -c "tt-smi -r 0"
sleep 30
# Then run V11 baseline first to confirm chip health (HPWL≈70M, gather≈7ms expected)
```

### Density correctness audit
```bash
TT_METAL_HOME=$(pwd)/tt-metal CPU_DCT=1 GATHER_MODE=v19 V19_SCALE_BITS=20 \
  DENSITY_AUDIT=1 DENSITY_AUDIT_MAX_ITERS=20 \
  DREAMPlace/dp_env/bin/python3 integration/run_dreamplace.py …
```
Inspect `results/density_audit/density_audit.csv` — `mass_deficit_pct` should be < 5 %, `rel_l2_pct` < 10 %. The audit was moved earlier in the Python client so the TT density is compared *before* `initial_density_map` is added (apples-to-apples vs the CPU `_cpu_density_fp32` reference).

---

## Goals for the next agent

1. **Get V19 working on adaptec1_2048** (the canonical V11-weakness target — 4 hot mega-tiles in gather). Use the debugging path in "Likely remaining bug" above.
2. **Sweep V19 across the 18 configs** once 2048 is healthy. Compare against V11 (baseline `results/v11/`) and V11outer (`results/v11outer_*/`) for HPWL drift and wall time. Aim for ±2 % HPWL drift and ≤ V11 wall on every config.
3. **Move host-side de-stride on-chip** (Phase 2). Currently the host pays ~2.25 ms per iter to de-stride `v19_strided[]` into `v19_dense[]`. A small reduce kernel that runs on-chip after writeout (or merged into writeout) would eliminate this.
4. **Investigate why `kernel_wait` on the host dominates V18's runtime** — same bottleneck likely limits V19's wall-time win on 2048. May need to coalesce more work per Finish, or switch to non-blocking enqueues.
5. **Consider committing the changes** — the working state for 512 is committable; the 2048 work is best kept in a branch until the hang is understood.
