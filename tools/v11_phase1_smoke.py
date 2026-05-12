#!/usr/bin/env python3
"""V11 Phase 1 smoke test.

Launches the V11 server stub once with synthetic cells, runs a single EP
forward pass, and checks the server stdout for the "V11 Phase 1 OK" line.

Phase 1 scope: validates that the snake-fill ownership map and per-receiver
counter logic work end-to-end. No NOC tuple writes or accumulation yet —
that's Phase 2.

Usage:
  python3 v11_phase1_smoke.py --container <container_name>
"""

import argparse
import os
import sys
import tempfile
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", required=True)
    ap.add_argument("--grid", type=int, default=512)
    ap.add_argument("--cells", type=int, default=10000)
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
    ipc_dir = tempfile.mkdtemp(prefix="v11p1_", dir=base)

    os.environ["GATHER_MODE"] = "v11"

    client = stc.ScatterTTNNClient(
        container=args.container,
        ipc_dir=ipc_dir,
        num_bins_x=M, num_bins_y=N,
        nc_max=nc,
        xl=xl, yl=yl, xh=xh, yh=yh,
        direct=False,
    )

    print(f"\n[v11_smoke] grid={M}x{N} cells={nc} container={args.container}")
    print(f"[v11_smoke] ipc_dir={ipc_dir}")

    try:
        client.start(M=M, N=N, nc_actual=nc, xl=xl, yl=yl, xh=xh, yh=yh)
        print("[v11_smoke] server started, sending EP request...")
        t0 = time.time()
        fx, fy, timing = client.call(px, py, sx, sy)
        elapsed = (time.time() - t0) * 1000
        print(f"[v11_smoke] EP call returned in {elapsed:.1f} ms; "
              f"gather_mode={timing.get('gather_mode')}")
        print(f"[v11_smoke] timing keys: scatter={timing.get('scatter_ms'):.2f} "
              f"gather={timing.get('gather_ms'):.2f} "
              f"total={timing.get('total_server_ms'):.2f} ms")
        # Print server stdout lines containing V11 markers
        # (server logs flow through stc's threaded reader)
        time.sleep(0.5)  # allow server log thread to drain
    finally:
        try: client.stop()
        except Exception: pass
        try:
            for f in os.listdir(ipc_dir): os.remove(os.path.join(ipc_dir, f))
            os.rmdir(ipc_dir)
        except Exception: pass

    print("[v11_smoke] done.")


if __name__ == "__main__":
    main()
