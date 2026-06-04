# v11op-bench — measured run (real silicon, 2026-06-03)

From the isolated harness:
```
[v11op-bench] n_batches=1000 (each batch = 1024 cells × 64 outer products on SFPU)
[bh-38-special-ayadav-for-reservation-78385:00404] *** Process received signal ***
[bh-38-special-ayadav-for-reservation-78385:00404] Signal: Aborted (6)
[bh-38-special-ayadav-for-reservation-78385:00404] Signal code:  (-6)
[bh-38-special-ayadav-for-reservation-78385:00404] [ 0] /lib/x86_64-linux-gnu/libc.so.6(+0x42520)[0x7f7653c6d520]
```

Full stdout: `run.log`. Per-zone Tracy pending (ENABLE_TRACY=ON; needs dump_profiler on clean device).
