# fccs — measured run (real silicon, 2026-06-03)

From the isolated harness:
```
[fccs] grid=512 ncells=200000 span=2x2 clustered=0 nc=110 cpc=1835 W=128 Tx=4 tiles=8 tile=256KB
[fccs] device MEDIAN=1.136 ms  rel_l2=2.256e-04 nbad=0  OK
```

Full stdout: `run.log`. Per-zone Tracy pending (ENABLE_TRACY=ON; needs dump_profiler on clean device).
