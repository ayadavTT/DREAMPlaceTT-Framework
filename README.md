# DREAMPlace-TT

End-to-end framework for running DREAMPlace's analytic placer on Tenstorrent **Blackhole** hardware. The expensive electric-potential step (per-iter density scatter → Poisson via DCT → field gather) is offloaded to a TT-Metal kernel pipeline; the rest of DREAMPlace runs unchanged on the host.

```
host (CPU)                            Tenstorrent Blackhole (110 Tensix cores)
─────────                             ─────────────────────────────────────────
DREAMPlace (unmodified)           ┌─► v4_compute   (TRISC SFPU bin geometry)
   │                              │
   │  px, py, sx, sy via shmem    │   v11_scatter_dm  (NCRISC) ┐ writer half
   ├─────────────────────────────►├─► v11_scatter_b   (BRISC)  ┘ + reader
   │                              │      ↓ tuples via NOC to owner cores
   │                              │   v11_accum_dm    (BRISC) ─┐ accumulator
   │                              │   v11_accum_n_dm  (NCRISC) ┘ + merge
   │                              │      ↓ density buf
   │                              │   TTNN DCT (6 matmuls)
   │   field_x, field_y           │      ↓ field_x, field_y
   ◄──────────────────────────────┴───
```

## Status snapshot

| Benchmark | TT HPWL | CPU HPWL | Overflow | Converged? |
|---|---:|---:|---:|:---:|
| sweep_adaptec1_2048 | 72.5M | 71.3M (+1.6%) | **0.069** | ✅ |
| sweep_adaptec1_512  | 164.6M | 70.3M | 0.337 | ❌ |
| sweep_bigblue3_2048 | 720M | 292M | 0.412 | ❌ |
| sweep_bigblue3_512  | 870M | 300M | 0.570 | ❌ |

Grid-2048 converges. Grid-512 still hits the per-page tuple cap on the hottest writer-receiver pairs; see [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) and [docs/V11_PHASE3_HANDOFF.md](docs/V11_PHASE3_HANDOFF.md).

## Layout

```
DREAMPlaceTT-Framework/
├── README.md              this file
├── REPRODUCE.md           step-by-step reproduction
├── ARCHITECTURE.md        framework + V11 pipeline design
├── kernels/               TT-Metal kernels (V11 + supporting V4)
├── host/                  C++ server + CMakeLists
├── integration/           Python ↔ DREAMPlace glue
├── benchmarks/configs/    DREAMPlace JSON configs we sweep
├── scripts/               build_server.sh, run_smoke.sh, run_sweep.sh, reset_chip.sh
├── tools/                 smoke harnesses + diagnostic scripts
├── results/               summary of sweep results (baseline + post-fix)
└── docs/
    ├── KNOWN_ISSUES.md
    ├── DREAMPLACE_TT_ARCHITECTURE.md
    ├── V11_PHASE3_HANDOFF.md
    ├── BLACKHOLE_HW_SPEC.md
    └── ...
```

External dependencies (large, **NOT in the repo**; managed as git submodules in the GitHub push):
```
tt-metal/        # https://github.com/tenstorrent/tt-metal — pin to the version listed in REPRODUCE.md
DREAMPlace/      # https://github.com/limbo018/DREAMPlace — pin + apply our tiny monkey-patch (see integration/)
```
In this local checkout they're set up as **symlinks** to existing trees one directory up. Replace with submodules before pushing.

## Quick start (one-line, assuming everything is already built)

```bash
export CONTAINER=bh-38-special-ayadav-for-reservation-74318   # your tt-metal container name
bash scripts/run_smoke.sh                                     # 1-iter correctness check, ~30 s
bash scripts/run_sweep.sh                                     # all 4 benchmarks, ~10 min
```

## First-time setup

See [REPRODUCE.md](REPRODUCE.md) for full step-by-step instructions covering:

1. Get a TT Blackhole machine + Docker container with tt-metal SDK
2. Clone TT-Metal and DREAMPlace (or `git submodule update --init` on the GitHub repo)
3. Build the C++ server inside the container
4. Run smoke test + sweep
5. Reproduce the result tables

## What's the actual workload?

DREAMPlace places ~100k–2M chip cells inside a rectangular die. Each iteration computes an **electric potential** over a `M × N` density grid (typically 512×512 or 2048×2048) and uses the gradient as a force to push cells toward better placements. That step is ~70 % of total runtime; it's what we accelerate on TT.

Pipeline mapped to TT:
1. **Scatter**: spread each cell's overlap area to bins it touches → per-bin density (V11 cell-centric tile-routed scatter using both BRISC and NCRISC on each Tensix core).
2. **Solve**: density → potential via 2D DCT and an analytical multiplier (TTNN C++ DCT, 6 matmuls).
3. **Gather**: read potential field back to host (per-cell forces are computed CPU-side).

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design.

## Top-level documents

| Doc | Read it when… |
|---|---|
| [REPRODUCE.md](REPRODUCE.md) | …you want to actually run this on your machine. |
| [ARCHITECTURE.md](ARCHITECTURE.md) | …you want to understand how the framework is wired. |
| [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) | …something doesn't work. Or you're picking up the convergence work. |
| [docs/V11_PHASE3_HANDOFF.md](docs/V11_PHASE3_HANDOFF.md) | …you're optimizing V11 kernel perf further. |
| [docs/PROFILING.md](docs/PROFILING.md) | …you want per-zone timing breakdown (TT-Tracy device profiler). |
| [docs/CONSTRAINTS.md](docs/CONSTRAINTS.md) | …you're proposing a new V11 variant — the design contract and Tensix capability reference. |
| [docs/PERFORMANCE_MODEL.md](docs/PERFORMANCE_MODEL.md) | …you want per-unit latency / NOC bandwidth / DRAM cost numbers for back-of-envelope kernel-design predictions, and the TT-Tracy recipe to verify them. |
| [docs/DREAMPLACE_TT_ARCHITECTURE.md](docs/DREAMPLACE_TT_ARCHITECTURE.md) | …you want the full design rationale + decision log. |
| [docs/BLACKHOLE_HW_SPEC.md](docs/BLACKHOLE_HW_SPEC.md) | …you need TT Blackhole hardware specs. |
| [results/README.md](results/README.md) | …you want the actual numbers, by benchmark. |
| [kernels/README.md](kernels/README.md) | …you're editing a kernel. |

## License

Our framework code: see [LICENSE](LICENSE) (Apache 2.0).
DREAMPlace: see upstream license.
TT-Metal: see upstream license.
