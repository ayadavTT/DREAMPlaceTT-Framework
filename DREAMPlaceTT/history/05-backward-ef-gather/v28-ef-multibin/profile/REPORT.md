# v28-ef-multibin — measured run (real silicon, 2026-06-03)

From the isolated harness:
```
[v28] N=2048 M=2048 ncells=210904 nc=110 cpc_col=19 band_cols=25 max_k=6 max_h=3 band=200KB mc=2048
[v28] chip MEDIAN=1.676 ms  (multi-bin, CPU EF ~5-7 ms)
[v28] accuracy: max_abs=1.53986e-06 rel_l2=6.98594e-08 nbad=0  OK
```

Full stdout: `run.log`. Per-zone Tracy pending (ENABLE_TRACY=ON; needs dump_profiler on clean device).
