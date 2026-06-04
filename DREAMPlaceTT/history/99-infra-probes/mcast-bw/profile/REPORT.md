# mcast-bw — measured run (real silicon, 2026-06-03)

From the isolated harness:
```
[bh-38-special-ayadav-for-reservation-78385:00295] *** Process received signal ***
[bh-38-special-ayadav-for-reservation-78385:00295] Signal: Aborted (6)
[bh-38-special-ayadav-for-reservation-78385:00295] Signal code:  (-6)
[bh-38-special-ayadav-for-reservation-78385:00295] [ 0] /lib/x86_64-linux-gnu/libc.so.6(+0x42520)[0x7fc1c5dff520]
[bh-38-special-ayadav-for-reservation-78385:00295] [ 1] /lib/x86_64-linux-gnu/libc.so.6(pthread_kill+0x12c)[0x7fc1c5e539fc]
```

Full stdout: `run.log`. Per-zone Tracy pending (ENABLE_TRACY=ON; needs dump_profiler on clean device).
