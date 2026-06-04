# v29-gather — measured run (real silicon, 2026-06-03)

From the isolated harness:
```
[v29] launching P1 (prep+bucket)...
[v29] P1 done. launching P2 (gather+compute+writer)...
[v29] P2 done.
[v29] N=2048 M=2048 ncells=210904 nc=110 cpc_x=19 band_cols=27 max_k=7 max_h=4 cap=67 route=99MB
[v29] prep+bucket=4.989 ms  gather+compute+scatter=5.259 ms  TOTAL=10.248 ms  (CPU EF ~5-7 ms, V21 12.9 ms)
```

Full stdout: `run.log`. Per-zone Tracy pending (ENABLE_TRACY=ON; needs dump_profiler on clean device).
