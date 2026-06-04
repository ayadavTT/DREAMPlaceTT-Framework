#!/usr/bin/env python3
"""Isolated testbench for the V19 forward density-scatter kernels
(v19_scatter_dm.cpp + v19_scatter_b_dm.cpp + v19_writeout_fp32_dm.cpp).

Runs the REAL production scatter program via the in-process V19 engine
(v4_compute -> v19_scatter_dm -> v19_writeout), with the DCT solve DISABLED
(CPU_DCT=1) so we isolate the density-construction kernels. Sweeps real design
scales x grids x spatial distributions, checks density correctness vs the exact
fp64 CPU reference, and records per-config kernel time.

This is a standalone file for THIS kernel only (no mode-switch mega-harness).
Run via test/run.sh (which also captures the per-zone Tracy profile).
"""
import os, sys, csv, time, argparse, statistics
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_KDIR = os.path.dirname(_HERE)                 # .../v18-scatter
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))  # DREAMPlaceTT
sys.path.insert(0, os.path.join(_ROOT, "history", "testbench"))
sys.path.insert(0, os.path.join(_ROOT, "host", "build"))   # v19_engine.so
import synth                                                # noqa: E402

os.environ.setdefault("CPU_DCT", "1")          # isolate scatter: no on-chip DCT
os.environ.setdefault("GATHER_MODE", "v18")
TOL = 1e-2                                       # bf16/fixed-point path bar


def run(grid: int, design: str, n_iters: int = 4):
    # ONE MeshDevice + ONE design per process: NC_max == this design's cell count
    # so the scatter processes exactly this design's tiles (timing must not be
    # contaminated by a larger NC_max). run.sh invokes this once per (design,grid).
    import v19_engine
    rows = []
    nc = synth.DESIGNS[design]
    print(f"[v18-test] building engine grid={grid} design={design} NC_max={nc} ...", flush=True)
    eng = v19_engine.V19Engine(grid, grid, nc, 0.0, 0.0, float(grid), float(grid), "v18")
    if True:
        for dist in synth.DISTS:
            px, py, sx, sy = synth.gen_workload(nc, grid, dist, seed=1234)
            px = np.ascontiguousarray(px); py = np.ascontiguousarray(py)
            sx = np.ascontiguousarray(sx); sy = np.ascontiguousarray(sy)
            dens = None; ms = []
            for it in range(n_iters):
                t0 = time.perf_counter()
                density, _fx, _fy, timing = eng.scatter(px, py, sx, sy, int(nc))
                dt = (time.perf_counter() - t0) * 1000.0
                if it >= 2:
                    ms.append(timing.get("scatter_ms", dt))
                dens = density
            kernel_ms = statistics.median(ms) if ms else float("nan")
            t0 = time.perf_counter()
            ref = synth.ref_density(px, py, sx, sy, grid)
            ref_ms = (time.perf_counter() - t0) * 1000.0
            r = synth.rel_l2(np.asarray(dens), ref)
            ok = r < TOL
            rows.append(dict(design=design, grid=grid, dist=dist, n_cells=nc,
                             rel_l2=r, ok=ok, kernel_ms=kernel_ms, ref_ms=ref_ms))
            print(f"[v18-test] {design:9s} grid={grid:5d} {dist:8s} nc={nc:8d} "
                  f"rel_l2={r:.2e} {'OK' if ok else 'FAIL'} kernel={kernel_ms:.3f}ms",
                  flush=True)
    return rows


def write_reports(rows, quick):
    pdir = os.path.join(_KDIR, "profile")
    os.makedirs(pdir, exist_ok=True)
    suffix = "_quick" if quick else ""
    csvp = os.path.join(pdir, f"results{suffix}.csv")
    # merge with any existing rows (other grids from prior invocations)
    merged = {}
    if os.path.exists(csvp):
        with open(csvp) as f:
            for d in csv.DictReader(f):
                merged[(d["design"], d["grid"], d["dist"])] = d
    for r in rows:
        merged[(r["design"], str(r["grid"]), r["dist"])] = dict(
            kernel="v18-scatter", design=r["design"], grid=r["grid"], dist=r["dist"],
            n_cells=r["n_cells"], rel_l2=f"{r['rel_l2']:.3e}", **{"pass": int(r["ok"])},
            kernel_ms=f"{r['kernel_ms']:.4f}", ref_ms=f"{r['ref_ms']:.4f}")
    order = sorted(merged.values(), key=lambda d: (d["design"], int(d["grid"]), d["dist"]))
    cols = ["kernel", "design", "grid", "dist", "n_cells", "rel_l2", "pass", "kernel_ms", "ref_ms"]
    with open(csvp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for d in order:
            w.writerow({k: d.get(k, "") for k in cols})
    rows = [dict(design=d["design"], grid=int(d["grid"]), dist=d["dist"], n_cells=d["n_cells"],
                 rel_l2=float(d["rel_l2"]), ok=int(d["pass"]) == 1,
                 kernel_ms=float(d["kernel_ms"])) for d in order]
    npass = sum(1 for r in rows if r["ok"])
    md = os.path.join(pdir, f"REPORT{suffix}.md")
    with open(md, "w") as f:
        f.write(f"# v18-scatter — measured report{' (quick)' if quick else ''}\n\n")
        f.write(f"Isolated run of the production V19 scatter program (v4_compute -> "
                f"v19_scatter_dm -> v19_writeout_fp32) via the in-process engine, CPU_DCT=1 "
                f"(no DCT). Correctness = density rel_l2 vs exact fp64 box-overlap reference; "
                f"pass if rel_l2 < {TOL}.\n\n")
        f.write(f"**{npass}/{len(rows)} configs PASS**\n\n")
        f.write("| design | grid | dist | n_cells | rel_l2 | pass | scatter ms |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['design']} | {r['grid']} | {r['dist']} | {r['n_cells']} | "
                    f"{r['rel_l2']:.2e} | {'OK' if r['ok'] else 'FAIL'} | {r['kernel_ms']:.3f} |\n")
        f.write("\n> Per-zone TT Tracy table: see `zones.txt` (generated by run.sh with "
                "TT_METAL_DEVICE_PROFILER=1).\n")
    print(f"[v18-test] wrote {csvp}\n[v18-test] wrote {md}  ({npass}/{len(rows)} pass)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="mark report as quick (single design)")
    ap.add_argument("--grid", type=int, required=True, help="single grid (one MeshDevice per process)")
    ap.add_argument("--design", required=True, help="single design (NC_max == its cell count)")
    a = ap.parse_args()
    rows = run(a.grid, a.design)
    write_reports(rows, a.quick)
