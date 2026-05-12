#!/usr/bin/env python3
"""
cpu_vs_tt_density.py — Diagnostic: compute CPU reference density from a
position dump and diff it against the TT-produced density at the same iter.

Useful for identifying mass deficits and per-bin biases when DREAMPlace
convergence on TT deviates from CPU baseline.

Usage:
    python cpu_vs_tt_density.py <pos_dump.bin.iter100.bin> <tt_density.bin>

The position dump format (binary):
    int32  M
    int32  N
    float  xl, yl, xh, yh
    int32  nc
    float[nc] px
    float[nc] py
    float[nc] sx
    float[nc] sy

The TT density file is just `float[M*N]` (row-major, bx-then-by).

Set these via env vars from the server when running DREAMPlace:
    EXPORT_POS_PATH=$IPC_DIR/dbg_pos.bin
    EXPORT_DENSITY_PATH=$IPC_DIR/dbg_dens.bin
The server writes pos at iter 1, 50, 100 and density every iter.
"""
import struct, sys
import numpy as np


def load_positions(path):
    with open(path, "rb") as f:
        M, N = struct.unpack("<ii", f.read(8))
        xl, yl, xh, yh = struct.unpack("<ffff", f.read(16))
        nc = struct.unpack("<i", f.read(4))[0]
        px = np.fromfile(f, dtype=np.float32, count=nc)
        py = np.fromfile(f, dtype=np.float32, count=nc)
        sx = np.fromfile(f, dtype=np.float32, count=nc)
        sy = np.fromfile(f, dtype=np.float32, count=nc)
    return M, N, xl, yl, xh, yh, px, py, sx, sy


def cpu_density(M, N, xl, yl, xh, yh, px, py, sx, sy):
    bsx = (xh - xl) / M
    bsy = (yh - yl) / N
    dens = np.zeros((M, N), dtype=np.float64)
    nc = len(px)
    for i in range(nc):
        if i % 200000 == 0:
            print(f"  CPU density progress: {i}/{nc}", flush=True)
        cx, cy = px[i], py[i]
        csx, csy = sx[i], sy[i]
        bxl = int(max(0, (cx - xl) // bsx))
        bxh = int(min(M - 1, (cx + csx - xl - 1e-9) // bsx))
        byl = int(max(0, (cy - yl) // bsy))
        byh = int(min(N - 1, (cy + csy - yl - 1e-9) // bsy))
        for bx in range(bxl, bxh + 1):
            if bx < 0 or bx >= M:
                continue
            bin_xl = xl + bx * bsx
            bin_xh = bin_xl + bsx
            ox = max(0.0, min(cx + csx, bin_xh) - max(cx, bin_xl))
            if ox <= 0:
                continue
            for by in range(byl, byh + 1):
                if by < 0 or by >= N:
                    continue
                bin_yl = yl + by * bsy
                bin_yh = bin_yl + bsy
                oy = max(0.0, min(cy + csy, bin_yh) - max(cy, bin_yl))
                if oy > 0:
                    dens[bx, by] += ox * oy
    dens *= 1.0 / (bsx * bsy)
    return dens


def main():
    if len(sys.argv) != 3:
        sys.exit(f"usage: {sys.argv[0]} <pos_dump> <tt_density>")
    pos_path, tt_path = sys.argv[1:]

    M, N, xl, yl, xh, yh, px, py, sx, sy = load_positions(pos_path)
    print(f"Grid {M}x{N}, nc={len(px)}, region [{xl},{yl}]..[{xh},{yh}]")

    print("Computing CPU reference...")
    cpu = cpu_density(M, N, xl, yl, xh, yh, px, py, sx, sy)
    tt = np.fromfile(tt_path, dtype=np.float32, count=M * N).reshape(M, N).astype(np.float64)

    delta = tt - cpu
    rel_l2 = np.linalg.norm(delta) / np.linalg.norm(cpu)
    pct_mass = 100 * delta.sum() / cpu.sum()
    print(f"\nGlobal: CPU sum={cpu.sum():.4e}  TT sum={tt.sum():.4e}")
    print(f"Mass deficit: {delta.sum():+.4e}  ({pct_mass:+.2f}%)")
    print(f"rel_L2 = {rel_l2:.4f}")
    print(f"|delta|: max={np.abs(delta).max():.4f}  mean={np.abs(delta).mean():.4f}")

    print("\nTop 10 worst bins (by |delta|):")
    idx = np.unravel_index(np.argsort(np.abs(delta).ravel())[::-1][:10], delta.shape)
    print(f"  {'bx':>4} {'by':>4} {'cpu':>10} {'tt':>10} {'delta':>10}")
    for i in range(10):
        bx, by = int(idx[0][i]), int(idx[1][i])
        print(f"  {bx:4d} {by:4d} {cpu[bx,by]:10.2f} {tt[bx,by]:10.2f} {delta[bx,by]:+10.2f}")


if __name__ == "__main__":
    main()
