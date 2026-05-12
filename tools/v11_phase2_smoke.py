#!/usr/bin/env python3
"""V11 Phase 2 smoke test.

Runs V11 (cell-centric tile-routed) scatter+accum on a small synthetic
workload, dumps the resulting density map, and compares against the CPU
triangle-density reference. Pass: rel_L2 < 1%.

Usage:
  python3 v11_phase2_smoke.py --container <name> [--grid 512] [--cells 100000]
"""

import argparse
import os
import sys
import tempfile
import time

import numpy as np


def cpu_triangle_density(px, py, sx, sy, xl, yl, xh, yh, M, N):
    """Reference CPU triangle scatter — port from v8_accuracy_test."""
    bsx32 = np.float32((xh - xl) / M)
    bsy32 = np.float32((yh - yl) / N)
    inv_bsx32 = np.float32(1.0 / float(bsx32))
    inv_bsy32 = np.float32(1.0 / float(bsy32))
    bsx = float(bsx32); bsy = float(bsy32)
    inv_ba = 1.0 / (bsx * bsy)

    px32 = px.astype(np.float32)
    py32 = py.astype(np.float32)
    sx32 = sx.astype(np.float32)
    sy32 = sy.astype(np.float32)
    px = px32.astype(np.float64)
    py = py32.astype(np.float64)
    sx = sx32.astype(np.float64)
    sy = sy32.astype(np.float64)

    max_sx_bins = int(np.ceil(float(sx.max()) / bsx)) + 2
    max_sy_bins = int(np.ceil(float(sy.max()) / bsy)) + 2
    K = max(2, max_sx_bins)
    H = max(2, max_sy_bins)

    chunk = max(1, 25_000_000 // (K * H))
    density_flat = np.zeros(M * N, dtype=np.float64)

    n = len(px)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        nx = px[s:e]; ny = py[s:e]
        snx = sx[s:e]; sny = sy[s:e]
        nx32 = px32[s:e]; ny32 = py32[s:e]
        snx32 = sx32[s:e]; sny32 = sy32[s:e]

        # Float32 floor of bin index
        bxl_f = np.floor((nx32 - np.float32(xl)) * inv_bsx32).astype(np.int32)
        byl_f = np.floor((ny32 - np.float32(yl)) * inv_bsy32).astype(np.int32)
        bxl_f = np.maximum(bxl_f, 0)
        byl_f = np.maximum(byl_f, 0)

        # Boundary correction (mirror v4_compute)
        bl_x = np.float32(xl) + bxl_f.astype(np.float32) * bsx32
        bxl_f = np.where(bl_x > nx32, bxl_f - 1, bxl_f)
        bxl_f = np.maximum(bxl_f, 0)
        bl_x_next = np.float32(xl) + (bxl_f + 1).astype(np.float32) * bsx32
        bxl_f = np.where(bl_x_next <= nx32, bxl_f + 1, bxl_f)

        bl_y = np.float32(yl) + byl_f.astype(np.float32) * bsy32
        byl_f = np.where(bl_y > ny32, byl_f - 1, byl_f)
        byl_f = np.maximum(byl_f, 0)
        bl_y_next = np.float32(yl) + (byl_f + 1).astype(np.float32) * bsy32
        byl_f = np.where(bl_y_next <= ny32, byl_f + 1, byl_f)

        for j in range(K):
            bx = bxl_f + j
            valid_x = (bx >= 0) & (bx < M)
            bx_left = xl + bx * bsx
            bx_right = bx_left + bsx
            ox = np.minimum(nx + snx, bx_right) - np.maximum(nx, bx_left)
            ox = np.where(valid_x & (ox > 0), ox, 0.0)
            for k in range(H):
                by = byl_f + k
                valid_y = (by >= 0) & (by < N)
                by_left = yl + by * bsy
                by_right = by_left + bsy
                oy = np.minimum(ny + sny, by_right) - np.maximum(ny, by_left)
                oy = np.where(valid_y & (oy > 0), oy, 0.0)
                area = ox * oy
                mask = area > 0
                idx = bx[mask].astype(np.int64) * N + by[mask].astype(np.int64)
                np.add.at(density_flat, idx, area[mask])

    density_flat *= inv_ba
    return density_flat.reshape(M, N).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", required=True)
    ap.add_argument("--grid", type=int, default=512)
    ap.add_argument("--cells", type=int, default=100000)
    ap.add_argument("--gather-mode", default="v11")
    args = ap.parse_args()

    sys.path.insert(0, "/localdev/ayadav/tt-work/TTPort/integration")
    import scatter_ttnn_client as stc

    M = N = args.grid
    nc = args.cells
    xl, yl, xh, yh = 0.0, 0.0, 1.0e6, 1.0e6

    rng = np.random.default_rng(42)
    avg = (xh - xl) / M * 1.5
    px = rng.uniform(xl, xh - avg, size=nc).astype(np.float32)
    py = rng.uniform(yl, yh - avg, size=nc).astype(np.float32)
    sx = rng.uniform(avg * 0.5, avg * 2.0, size=nc).astype(np.float32)
    sy = rng.uniform(avg * 0.5, avg * 2.0, size=nc).astype(np.float32)

    base = "/localdev/ayadav/tt-work/TTPort/ipc_shm"
    os.makedirs(base, exist_ok=True)
    ipc_dir = tempfile.mkdtemp(prefix="v11p2_", dir=base)
    dump_path = os.path.join(base, f"v11p2_density_{M}_{nc}.bin")

    os.environ["GATHER_MODE"] = args.gather_mode
    os.environ["EXPORT_DENSITY_PATH"] = dump_path
    if os.path.exists(dump_path):
        os.remove(dump_path)

    print(f"\n[v11_smoke] grid={M}x{N} cells={nc} mode={args.gather_mode} "
          f"container={args.container}")

    print(f"[v11_smoke] computing CPU reference density...")
    t0 = time.time()
    ref = cpu_triangle_density(px, py, sx, sy, xl, yl, xh, yh, M, N)
    cpu_secs = time.time() - t0
    print(f"[v11_smoke] CPU reference computed in {cpu_secs:.1f}s "
          f"(sum={float(ref.sum()):.6e}, max={float(ref.max()):.4e})")

    client = stc.ScatterTTNNClient(
        container=args.container,
        ipc_dir=ipc_dir,
        num_bins_x=M, num_bins_y=N,
        nc_max=nc,
        xl=xl, yl=yl, xh=xh, yh=yh,
        direct=False,
    )

    try:
        client.start(M=M, N=N, nc_actual=nc, xl=xl, yl=yl, xh=xh, yh=yh)
        t0 = time.time()
        fx, fy, timing = client.call(px, py, sx, sy)
        elapsed = (time.time() - t0) * 1000
        print(f"[v11_smoke] EP call returned in {elapsed:.1f} ms; "
              f"gather_mode={timing.get('gather_mode')}, "
              f"scatter={timing.get('scatter_ms'):.2f} ms, "
              f"gather={timing.get('gather_ms'):.2f} ms")

        if not os.path.exists(dump_path):
            print(f"[v11_smoke] FAIL: density dump not found at {dump_path}")
            return 1

        with open(dump_path, "rb") as f:
            buf = f.read()
        tt = np.frombuffer(buf, dtype=np.float32).copy().reshape(M, N)

        diff = (tt.astype(np.float64) - ref.astype(np.float64))
        l2_diff = float(np.sqrt(np.sum(diff * diff)))
        l2_ref = float(np.sqrt(np.sum(ref.astype(np.float64) ** 2)))
        rel_l2 = l2_diff / l2_ref if l2_ref > 0 else float("inf")
        max_abs = float(np.max(np.abs(diff)))
        ref_max = float(np.max(np.abs(ref)))

        print(f"[v11_smoke] RESULT: rel_L2 = {rel_l2:.3e}  "
              f"max_abs = {max_abs:.3e}  ref_max = {ref_max:.3e}")
        print(f"[v11_smoke] sums: tt={float(tt.sum()):.6e}  ref={float(ref.sum()):.6e}  "
              f"ratio={float(tt.sum())/float(ref.sum()):.6f}")
        if rel_l2 < 0.01:
            print(f"[v11_smoke] PASS (rel_L2 < 1%)")
            return 0
        else:
            print(f"[v11_smoke] FAIL (rel_L2 >= 1%)")
            return 1
    finally:
        try: client.stop()
        except Exception: pass
        try:
            for f in os.listdir(ipc_dir): os.remove(os.path.join(ipc_dir, f))
            os.rmdir(ipc_dir)
        except Exception: pass


if __name__ == "__main__":
    sys.exit(main())
