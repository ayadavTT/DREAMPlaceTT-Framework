# v26-ef — measured run (real silicon, 2026-06-03)

From the isolated harness:
```
[v26] grouping split: field-shard(artifact)=13.163 ms  cell-bucket(real)=13.149 ms
[v26] H=2048 W=2048 ncells=540000 nc=110 band=320KB mc=6144 nbat_max=6  GROUPING(host,fp32)=26.313 ms
[v26] chip(3-kernel)=0.783 ms  grouping(host)=26.313 ms  TOTAL=27.096 ms  (CPU EF ~5-7 ms)
[v26] accuracy: max_abs=2.13851e-07 rel_l2=4.56586e-08 nbad=0
```

Full stdout: `run.log`. Per-zone Tracy pending (ENABLE_TRACY=ON; needs dump_profiler on clean device).
