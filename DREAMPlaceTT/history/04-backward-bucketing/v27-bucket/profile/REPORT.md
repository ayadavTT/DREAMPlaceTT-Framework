# v27-bucket — measured run (real silicon, 2026-06-03)

From the isolated harness:
```
[v27] H=2048 ncells=540000 nc=110 rpc=19 cpc=4912  max/(src,band)=71 cap=122  route_buf=34 MB
[v27] bucketing scatter MEDIAN=0.793 ms  (vs CPU EF ~5-7 ms; this is the backward grouping)
[v27] validate: bands_mismatch=0 cap_overflow=0  OK
```

Full stdout: `run.log`. Per-zone Tracy pending (ENABLE_TRACY=ON; needs dump_profiler on clean device).
