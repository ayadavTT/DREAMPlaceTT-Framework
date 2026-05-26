#!/usr/bin/env python3
"""
V21a electric_force microbench — Python orchestrator.

Two-process design: this script synthesizes inputs, writes them to a binary,
invokes the TT-side `v21_ef_microbench` binary, then runs DREAMPlace's
production `electric_potential_cpp.electric_force(...)` on the same inputs
and compares results. The CPU baseline is the literal production code path
(not a re-implementation) so semantics can't drift.

Usage:
    python integration/v21_ef_microbench.py --M 32 --N 32 --num-nodes 80
    python integration/v21_ef_microbench.py --M 2048 --N 2048 --num-nodes 540000
    python integration/v21_ef_microbench.py --M 2048 --N 2048 --num-nodes 2040000

The script self-reexecs under DREAMPlace/dp_env so torch + electric_potential_cpp
import cleanly (same pattern as integration/run_dreamplace.py).
"""

from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter

_THIS_DIR = Path(__file__).resolve().parent
_ROOT     = _THIS_DIR.parent
_DREAMPLACE_DIR = Path(os.environ.get("DREAMPLACE_DIR", _ROOT / "DREAMPlace" / "install"))
_VENV_ROOT      = _DREAMPLACE_DIR.parent / "dp_env"
_DP_ENV_PYTHON  = _VENV_ROOT / "bin" / "python3"


def _reexec_with_dp_env_if_needed() -> None:
    if not _DP_ENV_PYTHON.is_file():
        return
    if os.path.realpath(sys.prefix) == os.path.realpath(str(_VENV_ROOT)):
        return
    os.environ["VIRTUAL_ENV"] = str(_VENV_ROOT)
    bindir = str(_VENV_ROOT / "bin")
    os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
    os.execv(str(_DP_ENV_PYTHON), [str(_DP_ENV_PYTHON)] + sys.argv)


# ── Binary format constants (must match v21_electric_force_microbench_host.cpp) ──
_IN_MAGIC  = 0x56323121
_OUT_MAGIC = 0x56323122


def synthesize(M: int, N: int, num_nodes: int, seed: int = 12345):
    """Build synthetic inputs for the microbench. Returns a dict of numpy fp32 arrays."""
    import numpy as np
    rng = np.random.default_rng(seed)
    # Grid is [xl, xh) × [yl, yh) with bin size bsx, bsy.
    xl, yl = 0.0, 0.0
    bsx, bsy = 1.0, 1.0
    xh = xl + M * bsx
    yh = yl + N * bsy
    # Keep cells away from grid boundary so bin coverage stays ≤ V21_MAX_K_RANGE (=8).
    # Max node size is 4*bsx; we shrink the pos range by 4 bins on each side.
    margin = 4.0
    pos_x = rng.uniform(xl + margin, xh - margin, size=num_nodes).astype(np.float32)
    pos_y = rng.uniform(yl + margin, yh - margin, size=num_nodes).astype(np.float32)
    pos = np.concatenate([pos_x, pos_y]).astype(np.float32)

    field_x = rng.uniform(-1.0, 1.0, size=(M, N)).astype(np.float32)
    field_y = rng.uniform(-1.0, 1.0, size=(M, N)).astype(np.float32)

    # Sizes: clamped lies in [sqrt(2)*bsx, 4*bsx]; orig is a bit smaller; offset
    # = (orig - clamped) / 2; ratio = orig_area / clamped_area.
    import math
    nsx = rng.uniform(math.sqrt(2.0) * bsx, 4.0 * bsx, size=num_nodes).astype(np.float32)
    nsy = rng.uniform(math.sqrt(2.0) * bsy, 4.0 * bsy, size=num_nodes).astype(np.float32)
    orig_x = rng.uniform(0.5, 1.0, size=num_nodes).astype(np.float32) * nsx
    orig_y = rng.uniform(0.5, 1.0, size=num_nodes).astype(np.float32) * nsy
    ox = ((orig_x - nsx) / 2.0).astype(np.float32)
    oy = ((orig_y - nsy) / 2.0).astype(np.float32)
    ratio = ((orig_x * orig_y) / (nsx * nsy)).astype(np.float32)

    return {
        "M": M, "N": N, "num_nodes": num_nodes,
        "xl": xl, "yl": yl, "xh": xh, "yh": yh, "bsx": bsx, "bsy": bsy,
        "pos": pos, "field_x": field_x, "field_y": field_y,
        "ox": ox, "oy": oy, "nsx": nsx, "nsy": nsy, "ratio": ratio,
    }


def write_inputs(path: str, d: dict) -> None:
    """Pack inputs into the binary format the C++ binary expects."""
    with open(path, "wb") as f:
        f.write(struct.pack(
            "<IIIIffff",
            _IN_MAGIC, d["M"], d["N"], d["num_nodes"],
            d["xl"], d["yl"], d["bsx"], d["bsy"],
        ))
        f.write(d["field_x"].tobytes(order="C"))
        f.write(d["field_y"].tobytes(order="C"))
        f.write(d["pos"].tobytes(order="C"))
        f.write(d["ox"].tobytes(order="C"))
        f.write(d["oy"].tobytes(order="C"))
        f.write(d["nsx"].tobytes(order="C"))
        f.write(d["nsy"].tobytes(order="C"))
        f.write(d["ratio"].tobytes(order="C"))


def read_output(path: str):
    """Parse the binary the C++ side wrote. Returns (grad_x, grad_y, kernel_ms)."""
    import numpy as np
    with open(path, "rb") as f:
        header = f.read(12)
        magic, num_nodes, kms_x1000 = struct.unpack("<III", header)
        if magic != _OUT_MAGIC:
            raise RuntimeError(f"Bad output magic 0x{magic:08x}")
        grad_x = np.frombuffer(f.read(num_nodes * 4), dtype=np.float32).copy()
        grad_y = np.frombuffer(f.read(num_nodes * 4), dtype=np.float32).copy()
    return grad_x, grad_y, kms_x1000 / 1000.0


def cpu_reference(d: dict, num_threads: int = 16):
    """Run DREAMPlace's production electric_force on the same inputs. Returns (ref_gx, ref_gy, median_ms)."""
    # OMP_NUM_THREADS controls the OpenMP team inside electric_potential_cpp.
    os.environ["OMP_NUM_THREADS"] = str(num_threads)

    # Make DREAMPlace importable.
    dp = os.path.realpath(str(_DREAMPLACE_DIR))
    for p in [dp, os.path.join(dp, "dreamplace")]:
        if p not in sys.path:
            sys.path.insert(0, p)

    import numpy as np
    import torch
    torch.set_num_threads(num_threads)
    from dreamplace.ops.electric_potential import electric_potential_cpp

    M = d["M"]; N = d["N"]; nn = d["num_nodes"]
    xl = d["xl"]; yl = d["yl"]; xh = d["xh"]; yh = d["yh"]
    bsx = d["bsx"]; bsy = d["bsy"]

    field_x_t = torch.from_numpy(d["field_x"].copy()).float()
    field_y_t = torch.from_numpy(d["field_y"].copy()).float()
    pos_t     = torch.from_numpy(d["pos"].copy()).float()
    ox_t      = torch.from_numpy(d["ox"].copy()).float()
    oy_t      = torch.from_numpy(d["oy"].copy()).float()
    nsx_t     = torch.from_numpy(d["nsx"].copy()).float()
    nsy_t     = torch.from_numpy(d["nsy"].copy()).float()
    ratio_t   = torch.from_numpy(d["ratio"].copy()).float()
    # bin_center_x = xl + (i + 0.5) * bsx for i in [0, M)
    bin_center_x_t = torch.from_numpy(
        (xl + (np.arange(M, dtype=np.float32) + 0.5) * bsx)
    ).float()
    bin_center_y_t = torch.from_numpy(
        (yl + (np.arange(N, dtype=np.float32) + 0.5) * bsy)
    ).float()
    # grad_pos: per electric_potential.py:235, the C++ multiplies grad_out by
    # grad_pos before returning. Setting grad_pos = 1 makes the result equal to
    # the per-cell gx (so we can compare apples-to-apples with the chip).
    grad_pos = torch.ones(2 * nn, dtype=torch.float32)

    # num_movable_impacted_bins_x/y, num_filler_impacted_bins_x/y: passed in but
    # unused inside the launcher (the launcher computes bin range itself per cell).
    # Set to a reasonable value.
    impacted = 8

    # Treat all cells as movable, no fillers, no fixed.
    num_movable_nodes = nn
    num_filler_nodes  = 0

    timings_ms = []
    result = None
    n_iters = 5
    for _ in range(n_iters):
        t0 = perf_counter()
        result = electric_potential_cpp.electric_force(
            grad_pos, M, N,
            impacted, impacted, impacted, impacted,
            field_x_t.view([-1]), field_y_t.view([-1]), pos_t,
            nsx_t, nsy_t,
            ox_t, oy_t, ratio_t,
            bin_center_x_t, bin_center_y_t,
            float(xl), float(yl), float(xh), float(yh), float(bsx), float(bsy),
            num_movable_nodes, num_filler_nodes,
        )
        timings_ms.append((perf_counter() - t0) * 1000.0)

    timings_ms.sort()
    median_ms = timings_ms[n_iters // 2]
    # Apply the `-` from electric_potential.py:235.
    output = -result
    ref_gx = output[:nn].detach().numpy().copy()
    ref_gy = output[nn:].detach().numpy().copy()
    print(f"[v21ef-py] CPU 16T runs ms: {['%.2f'%m for m in timings_ms]} → median {median_ms:.2f}")
    return ref_gx, ref_gy, median_ms


def diff_report(ref_gx, ref_gy, got_gx, got_gy, rel_tol: float = 1e-4, abs_tol: float = 1e-5):
    """Return (n_mismatches, max_abs_diff, max_rel_diff, first_bad_idx, first_bad_payload_str)."""
    import numpy as np
    abs_dx = np.abs(ref_gx - got_gx)
    abs_dy = np.abs(ref_gy - got_gy)
    scale_x = np.maximum(1.0, np.abs(ref_gx))
    scale_y = np.maximum(1.0, np.abs(ref_gy))
    rel_dx = abs_dx / scale_x
    rel_dy = abs_dy / scale_y
    rel_max = np.maximum(rel_dx, rel_dy)
    bad = rel_max > rel_tol
    n = int(bad.sum())
    max_abs = float(max(abs_dx.max(), abs_dy.max()))
    max_rel = float(rel_max.max())
    first_bad = -1
    payload = ""
    if n > 0:
        first_bad = int(np.argmax(bad))
        payload = (f"c={first_bad} ref_gx={ref_gx[first_bad]:.6g} got_gx={got_gx[first_bad]:.6g} "
                   f"ref_gy={ref_gy[first_bad]:.6g} got_gy={got_gy[first_bad]:.6g}")
    return n, max_abs, max_rel, first_bad, payload


def main():
    _reexec_with_dp_env_if_needed()
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--num-nodes", type=int, required=True)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--num-threads", type=int, default=16,
                    help="OMP threads for the CPU baseline (default: 16)")
    ap.add_argument("--binary", default=str(_ROOT / "host" / "build" / "v21_ef_microbench"),
                    help="Path to v21_ef_microbench binary")
    ap.add_argument("--rel-tol", type=float, default=1e-4)
    ap.add_argument("--keep-tmp", action="store_true",
                    help="Don't delete the inputs/output binary tmp files")
    args = ap.parse_args()

    if not os.path.isfile(args.binary):
        print(f"FATAL: binary not found at {args.binary}", file=sys.stderr)
        print("Build it first:  bash scripts/build_server.sh", file=sys.stderr)
        sys.exit(1)

    print(f"[v21ef-py] M={args.M} N={args.N} num_nodes={args.num_nodes}")

    # ── Synthesize ──
    d = synthesize(args.M, args.N, args.num_nodes, seed=args.seed)

    # ── Stage tmp files ──
    tmpdir = Path(tempfile.mkdtemp(prefix="v21_ef_"))
    inputs_path = tmpdir / "inputs.bin"
    output_path = tmpdir / "output.bin"
    print(f"[v21ef-py] inputs  → {inputs_path}")
    print(f"[v21ef-py] outputs ← {output_path}")
    write_inputs(str(inputs_path), d)

    # ── Run TT binary ──
    print(f"[v21ef-py] running {args.binary}")
    proc = subprocess.run(
        [args.binary, "--inputs", str(inputs_path), "--output", str(output_path)],
        check=False, capture_output=True, text=True,
    )
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    if proc.returncode != 0:
        print(f"FATAL: binary returned {proc.returncode}", file=sys.stderr)
        sys.exit(proc.returncode)

    # ── Read TT output ──
    got_gx, got_gy, kernel_ms = read_output(str(output_path))

    # ── CPU reference ──
    ref_gx, ref_gy, cpu_ms = cpu_reference(d, num_threads=args.num_threads)

    # ── Diff ──
    n_bad, max_abs, max_rel, first_bad, payload = diff_report(
        ref_gx, ref_gy, got_gx, got_gy, rel_tol=args.rel_tol)

    speedup = (cpu_ms / kernel_ms) if kernel_ms > 0 else float("nan")
    print("=" * 60)
    print(f"V21a EF MICROBENCH  M={args.M} N={args.N} num_nodes={args.num_nodes}")
    print(f"  TT kernel median   : {kernel_ms:.3f} ms")
    print(f"  CPU 16T median     : {cpu_ms:.3f} ms")
    print(f"  speedup (CPU/TT)   : {speedup:.2f}×")
    print(f"  mismatches         : {n_bad} / {args.num_nodes}  (rel_tol={args.rel_tol})")
    print(f"  max abs diff       : {max_abs:.6g}")
    print(f"  max rel diff       : {max_rel:.6g}")
    if n_bad > 0:
        print(f"  first bad          : {payload}")
        print("  ✗ FAIL")
        rc = 2
    else:
        print("  ✓ accuracy OK")
        rc = 0
    print("=" * 60)
    # Machine-readable summary line
    print(f"V21A_RESULT,{args.M},{args.N},{args.num_nodes},"
          f"{kernel_ms:.3f},{cpu_ms:.3f},{speedup:.3f},"
          f"{n_bad},{max_abs:.6g},{max_rel:.6g}")

    if not args.keep_tmp:
        try:
            inputs_path.unlink()
            output_path.unlink()
            tmpdir.rmdir()
        except OSError:
            pass

    sys.exit(rc)


if __name__ == "__main__":
    main()
