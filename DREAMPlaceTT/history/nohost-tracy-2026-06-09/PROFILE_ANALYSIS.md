# No-host pipeline profile — where the time goes (all 18 configs)

Per-iteration cost, decomposed into **actual on-device operations (OPS)**, **data movement**,
and **reducible host overhead**, from the no-host run logs (`.nhsw_*.log`, `V22_NOHOST=drive`).
All values ms/iter (the fine-grained `[fwd_full]`/`[v35_timing]` steady-state iter).

- **OPS** = forward `scatter + writeout + DCT` + backward `count + place + gather + unsort`.
- **move** = `h2d + dct_up + field_d2h` (necessary transfers).
- **host** = `field_wrap + fold+resid + ctx + plan` (host-serial framework work — the reducible part).
- **iter** = forward `EP_TOTAL` + backward total. **Excludes the V22 optimizer step** (~8 SFPU
  launches/iter, not separately logged — mostly launch dispatch, not compute).

| config | scatter | DCT | writeout | gather | place | unsort | count | **OPS** | move | host | iter |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| adaptec1_512 | 4.7 | 1.1 | 0.2 | 2.1 | 1.2 | 1.4 | 0.5 | **11.1 (80%)** | 0.9 | 2.5 (18%) | 13.9 |
| adaptec1_1024 | 5.9 | 1.1 | 0.7 | 2.0 | 1.1 | 1.7 | 0.5 | **13.1 (81%)** | 0.9 | 2.7 (17%) | 16.1 |
| adaptec1_2048 | 12.2 | 2.8 | 2.7 | 3.1 | 1.1 | 2.1 | 0.6 | **24.6 (86%)** | 1.6 | 3.2 (11%) | 28.6 |
| adaptec2_512 | 7.6 | 1.0 | 0.2 | 3.0 | 1.9 | 2.2 | 0.8 | **16.7 (79%)** | 1.7 | 3.5 (17%) | 21.1 |
| adaptec2_1024 | 8.5 | 1.0 | 0.7 | 2.9 | 1.8 | 2.1 | 0.8 | **17.9 (84%)** | 1.2 | 2.6 (13%) | 21.1 |
| adaptec2_2048 | 13.1 | 2.7 | 2.6 | 3.5 | 1.9 | 2.6 | 0.8 | **27.2 (87%)** | 1.7 | 3.1 (10%) | 31.3 |
| adaptec3_512 | 16.4 | 1.1 | 0.3 | 5.4 | 4.3 | 4.7 | 1.6 | **33.8 (80%)** | 3.2 | 6.2 (15%) | 42.5 |
| adaptec3_1024 | 16.4 | 1.0 | 0.7 | 5.6 | 4.3 | 4.6 | 1.6 | **34.3 (84%)** | 2.4 | 4.6 (11%) | 40.7 |
| adaptec3_2048 | 19.3 | 2.8 | 2.6 | 6.1 | 4.5 | 5.8 | 1.7 | **42.8 (84%)** | 3.3 | 5.3 (11%) | 50.7 |
| bigblue1_512 | 7.7 | 0.9 | 0.3 | 3.0 | 2.0 | 2.3 | 0.8 | **17.0 (82%)** | 1.3 | 3.1 (15%) | 20.9 |
| bigblue1_1024 | 9.6 | 1.2 | 0.7 | 2.8 | 2.0 | 2.3 | 0.8 | **19.4 (82%)** | 1.5 | 3.5 (15%) | 23.8 |
| bigblue1_2048 | 18.8 | 2.8 | 2.7 | 4.2 | 2.0 | 2.7 | 0.8 | **34.1 (88%)** | 1.8 | 3.4 (9%) | 38.7 |
| bigblue2_512 | 17.9 | 1.1 | 0.2 | 5.6 | 4.6 | 5.0 | 1.7 | **36.1 (82%)** | 2.8 | 5.6 (13%) | 43.8 |
| bigblue2_1024 | 19.5 | 1.4 | 0.8 | 6.4 | 4.9 | 5.0 | 1.8 | **39.8 (76%)** | 4.2 | 9.1 (18%) | 52.1 |
| bigblue2_2048 | 23.2 | 2.8 | 2.7 | 6.3 | 4.9 | 6.2 | 1.8 | **47.8 (87%)** | 2.9 | 4.9 (9%) | 54.9 |
| bigblue3_512 | 23.4 | 1.1 | 0.2 | 7.4 | 6.4 | 6.8 | 2.4 | **47.8 (85%)** | 3.0 | 6.0 (11%) | 56.0 |
| bigblue3_1024 | 26.5 | 1.6 | 0.7 | 8.7 | 6.8 | 6.8 | 2.4 | **53.7 (81%)** | 4.9 | 8.5 (13%) | 66.1 |
| bigblue3_2048 | 32.8 | 2.8 | 2.5 | 10.2 | 6.8 | 8.5 | 2.4 | **66.0 (88%)** | 3.6 | 6.1 (8%) | 74.9 |

## Headline findings

- **80–88% of each iteration is on-device operations; 8–18% is host overhead; ~5–9% is data movement.** So most time is "real work" at the *stage* level — the inefficiency is *within* the ops (below), plus a reducible host slice.
- **Forward `scatter` is the single dominant cost: 34–48% of the whole iteration** at every scale. It's an atomic uint32 density scatter — DRAM/atomic-bound, not arithmetic.
- **Backward `gather + place + unsort` = 22–37%** of the iteration, growing with cell count.
- **`DCT` is cheap (1–3 ms, ~3–10%)** and **`count` is small (0.5–2.4 ms)**.

## Actual operation vs "unnecessary"/reducible time

The stage time is ≥80% "ops", but those ops are themselves largely **data-movement-bound, not compute** (from prior device-zone Tracy + `gather-place-finegrain-profile` / `place-deepdive-profile`):

| item | cost | how much is actual compute | reducible-time lever |
|---|---|---|---|
| forward `scatter` | 34–48% of iter | low — DRAM atomics dominate | the #1 target; memory/atomic-bound scatter |
| backward `gather` | 5–14% of iter | **~7%** (SFPU math ~1.5 µs/(k,h) starved by ~22 µs field-fill) | **cut (k,h) footprint + speed the indexed field-fill — NOT the SFPU** |
| backward `place` | 3–9% of iter | low — DRAM-WRITE-bound (~49% scattered writes, 27% read, 21% sort) | double-buffer the writes (~−49%) / fold grouping into the forward |
| on-chip `unsort` | 1.4–8.5 ms | data-movement (L1 scatter + flush) | **single-RISC → dualize ≈ 2×** (bit-exact, documented) |
| host `field_wrap` | 0.8–4.1 ms (10–15% of fwd) | none — host framework | field stays on-device in no-host; this host wrapping should be near-zero → investigate/eliminate |
| host `fold+resid`, `plan` | ~0.7–0.9, 0.25–0.47 | none — host-serial | trim |
| V22 optimizer step | not logged | trivial element-wise | ~8 launches/iter → fuse into fewer (dispatch-bound) |

**Biggest wins, ranked:** (1) the forward atomic scatter (largest absolute, every config); (2) the gather field-fill (it's ~90% data movement, not math); (3) dualize the on-chip unsort (~2×); (4) the `place` DRAM writes; (5) the host `field_wrap` (~10–15% of forward, should be ~0 in no-host).

## Caveats (precision)

- The **within-op compute-vs-movement split** (gather ~7% compute, place write-bound) is from prior post-mortem Tracy captures (`results/tracy_*`, May 12) + the memory notes — **not** freshly measured on this no-host run (the current `build_Release` streams Tracy live, no post-mortem CSV).
- The **V22 optimizer step** is not in the iter total (not instrumented). It is ~8 SFPU launches over `2·num_nodes` element-wise data — small compute, mostly launch dispatch.
- Regenerate the stage table with `tools/nohost_timing.sh`.
