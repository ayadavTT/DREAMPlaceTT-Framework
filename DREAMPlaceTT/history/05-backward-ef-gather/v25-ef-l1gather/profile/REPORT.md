# v25-ef-l1gather — measured run (real silicon, 2026-06-03)

From the isolated harness:
```
[v25] H=2048 W=2048 ncells=540000  nc=110 band_rows=20 band=320 KB/core max_cells=4910
[v25] warmup...
[v25] warmup=626.13 ms
[v25] kernel runs: 0.383 0.401 0.417 0.422 0.428 0.432 0.486 
[v25] kernel MEDIAN = 0.422 ms   (CPU EF baseline ~7.3 ms)
```

Full stdout: `run.log`. Per-zone Tracy pending (ENABLE_TRACY=ON; needs dump_profiler on clean device).
