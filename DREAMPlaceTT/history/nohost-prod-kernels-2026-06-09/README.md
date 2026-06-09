# Production kernel snapshot — no-host density+optimizer pipeline (2026-06-09)

A copy of the live `DREAMPlaceTT/kernels/` set (42 .cpp + 1 .h) as used by the
**no-host** pipeline that converges 16/16 non-deferred configs (`V22_NOHOST=drive`;
see `DREAMPlaceTT/NOHOST_OPTIMIZER.md` for results + the per-config timing table).

Branch `density-codesign-compact-stash64-dual`. Auto-selected backward config:
`grouped=64B(COMPACT) source=64B dual_risc=YES bucket=YES max_k=8 max_h=4`.
Forward: `V31_STASH` + `V31_GEOM` + `V19_SKIP_CELL_SORT`.

## Kernels actually JIT-compiled in the live no-host pipeline

| stage | kernels |
|---|---|
| **forward** scatter + geom/route stash | `v11_histogram.cpp`, `v11_accum_dm.cpp`, `v11_accum_n_dm.cpp`, `v19_writeout_fp32_dm.cpp`, `v19_writeout_void_ncrisc.cpp`, `v31_scatter_stash_dm.cpp`, `v31_scatter_b_stash_dm.cpp` |
| **DCT** | TTNN ops (no custom kernel) |
| **backward** V35 (count→place→gather) | `v35_count64_bucket_n.cpp`, `v35_place_s64_bucket_n.cpp`, `v35_gather_brisc_compact.cpp`, `v28_compute_pbf.cpp`, `v28_ncrisc_pbf.cpp` |
| **on-chip unsort** (density grad → b_dg, no host) | `v35_onchip_unsort.cpp` |
| **optimizer** V22 (combine→precond→nesterov→clamp + α reduction) | `v22_gradcombine_compute.cpp`, `v22_precond_compute.cpp`, `v22_nesterov_compute.cpp`, `v22_clamp_compute.cpp`, `v22_stepsize_compute.cpp`, `v22_elt_reader.cpp`, `v22_elt_writer.cpp`, `v22_nesterov_reader.cpp`, `v22_nesterov_writer.cpp`, `v22_stepsize_reader.cpp`, `v22_stepsize_writer.cpp` |

The V22 no-host line search reuses these kernels with rewired buffers — `gradcombine`
doubles as a scalar-add (Δpos/Δgrad diffs) and a copy (g_k snapshot + commit);
`stepsize` does the `Σ(Δpos)²`/`Σ(Δgrad)²` reduction. **No new kernels were added.**

## The rest (kept in the prod dir, NOT in the live no-host path)

Alternative/auto-select variants (`v35_count.cpp`, `v35_count64.cpp`, `v35_count64_bucket.cpp`,
`v35_count64_n.cpp`, `v35_place.cpp`, `v35_place_compact.cpp`, `v35_place_s64*.cpp`,
`v35_gather_brisc.cpp`, `v35_gather_brisc_pbf.cpp`, `v28_compute.cpp`, `v28_ncrisc.cpp`) — the
engine auto-selects among these by problem size; the live set above is what bb-scale configs pick.
`v21_ef_*` = the V21 full-layout EF backward (reference path). `v4_*` = the V4 base scatter.

## Tracy device-zone profiles + per-stage timing

See `../nohost-tracy-2026-06-09/`: the per-config host-wall per-stage timing
(`per_config_stage_timing.txt`), the 68 instrumented device zones these kernels register
(`our_device_zones.txt`, e.g. `V35-LOADBAND`, `GB-LOOP`), and the raw zone registration
(`zone_src_locations.log`). The device profiler runs; the post-mortem per-zone timing CSV
needs a Tracy client / post-mortem rebuild (procedure in that README).
