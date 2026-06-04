# v30-backward — measured run (real silicon, 2026-06-03)

From the isolated harness:
```
[v30] grid=512 ncells=451000 span=2x2 nc=110 slab_bins=2384  workers=110 target=20508 max_worker=17088 max_nslabs=1 imbal=1.04x clustered=0
[v30] device MEDIAN=0.696 ms   rel_l2=1.597e-04  FAIL
```

Full stdout: `run.log`. Per-zone Tracy pending (ENABLE_TRACY=ON; needs dump_profiler on clean device).
