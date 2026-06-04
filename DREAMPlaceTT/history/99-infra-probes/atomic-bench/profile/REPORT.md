# atomic-bench — measured run (real silicon, 2026-06-03)

From the isolated harness:
```
[bh-38-special-ayadav-for-reservation-78385:00186] *** Process received signal ***
[bh-38-special-ayadav-for-reservation-78385:00186] Signal: Aborted (6)
[bh-38-special-ayadav-for-reservation-78385:00186] Signal code:  (-6)
[bh-38-special-ayadav-for-reservation-78385:00186] [ 0] /lib/x86_64-linux-gnu/libc.so.6(+0x42520)[0x7f1415b1f520]
[bh-38-special-ayadav-for-reservation-78385:00186] [ 1] /lib/x86_64-linux-gnu/libc.so.6(pthread_kill+0x12c)[0x7f1415b739fc]
```

Full stdout: `run.log`. Per-zone Tracy pending (ENABLE_TRACY=ON; needs dump_profiler on clean device).
