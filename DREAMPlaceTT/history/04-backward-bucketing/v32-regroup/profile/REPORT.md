# v32-regroup — measured run (real silicon, 2026-06-03)

From the isolated harness:
```
[bh-38-special-ayadav-for-reservation-78385:00047] *** Process received signal ***
[bh-38-special-ayadav-for-reservation-78385:00047] Signal: Aborted (6)
[bh-38-special-ayadav-for-reservation-78385:00047] Signal code:  (-6)
[bh-38-special-ayadav-for-reservation-78385:00047] [ 0] /lib/x86_64-linux-gnu/libc.so.6(+0x42520)[0x7f21eb8a9520]
[bh-38-special-ayadav-for-reservation-78385:00047] [ 1] /lib/x86_64-linux-gnu/libc.so.6(pthread_kill+0x12c)[0x7f21eb8fd9fc]
```

Full stdout: `run.log`. Per-zone Tracy pending (ENABLE_TRACY=ON; needs dump_profiler on clean device).
