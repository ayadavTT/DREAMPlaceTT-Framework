# benchmarks/

DREAMPlace JSON configs for the benchmarks we sweep on TT.

## configs/

The 4 benchmark configurations our sweep targets, plus a couple of references.

| File | Design | Grid | iteration cap |
|---|---|---|---|
| `sweep_adaptec1_512.json` | ISPD-2005 adaptec1 (~370k cells) | 512×512 | 1000 |
| `sweep_adaptec1_2048.json` | adaptec1 | 2048×2048 | 1000 |
| `sweep_bigblue3_512.json` | ISPD-2005 bigblue3 (~2.1M cells) | 512×512 | 1000 |
| `sweep_bigblue3_2048.json` | bigblue3 | 2048×2048 | 1000 |
| `adaptec1_512.json` | adaptec1 (short variant) | 512×512 | 100 |
| `adaptec1_2048.json` | adaptec1 (short variant) | 2048×2048 | 100 |

`iteration` is the per-stage iteration cap. The `sweep_*` variants are tuned for convergence (1000-iter); the bare `<bench>_<grid>.json` are short variants useful for quick smoke checks.

DREAMPlace's other global-place params (`density_weight=8e-05`, `gamma=4.0`, `random_seed=1000`, `target_density=1.0`) are identical across configs and match the upstream ISPD-2005 setup — the only thing we vary in `sweep_*` is the iteration count.

## Input data (bookshelf files)

The actual netlist + node + macro data lives outside this framework, alongside DREAMPlace itself:
```
DREAMPlace/benchmarks/ispd2005/adaptec1/
DREAMPlace/benchmarks/ispd2005/bigblue3/
```
Configs reference these via the `aux_input` field, e.g. `"aux_input": "benchmarks/ispd2005/adaptec1/adaptec1.aux"`. DREAMPlace resolves these relative to its working directory.

The ISPD-2005 dataset is freely available; if you cloned DREAMPlace via the framework's symlink, the benchmarks should be in place. Otherwise: see the upstream DREAMPlace README.

## Adding a new benchmark

Just drop a JSON into `configs/`. The 4 default `run_sweep.sh` benchmarks are baked into the script — extend the `BENCHES` array there, or pass benchmark names on the command line.
