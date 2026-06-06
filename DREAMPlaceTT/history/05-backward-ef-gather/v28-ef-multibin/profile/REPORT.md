# v28-ef-multibin — measured run (real silicon, 2026-06-03)

From the isolated harness:
```
[v28] N=2048 M=2048 ncells=210904 nc=110 cpc_col=19 band_cols=25 max_k=6 max_h=3 band=200KB mc=2048
[v28] chip MEDIAN=1.676 ms  (multi-bin, CPU EF ~5-7 ms)
[v28] accuracy: max_abs=1.53986e-06 rel_l2=6.98594e-08 nbad=0  OK
```

Full stdout: `run.log`.

**Per-zone Tracy CAPTURED 2026-06-06** (the "pending" capture is done). The shipped V35
backward reuses `v28_ncrisc` (NCRISC fy band + drain) + `v28_compute` (SFPU sum); its device
zones — `V28N-LOADBAND` ≈145 µs, `V28N-GATHER` ≈22 µs/instance, `V28N-DRAIN` ≈2.5 µs — were
captured via the `v35live` harness (clean-device profiler dump) and live in
`../v35-halotile/profile/{zones.txt,PROFILE.md}` (alongside the v35_count/place/gather zones).
`v28_compute` has no `DeviceZoneScopedN` markers (pure SFPU) so it has no sub-zones; its cost
shows up inside the gather's `*-GATHER` zone (the multiply-accumulate is cheap).
