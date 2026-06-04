"""synth.py — Python mirror of synth.hpp for the engine-driven (forward) kernel
testbenches. Generates DREAMPlace-like synthetic workloads (cells in bin units)
across real design scales, grids, and spatial distributions, plus the exact
fp64 CPU density reference (box-overlap, matching v4_compute)."""
import numpy as np

# Real ISPD2005 movable-node counts (from <design>.nodes headers).
DESIGNS = {
    "adaptec1":  211447, "adaptec2": 255023, "adaptec3":  451650,
    "bigblue1":  278164, "bigblue2": 557866, "bigblue3": 1096812,
}
GRIDS = [512, 1024, 2048]
DISTS = ["uniform", "cluster", "clouds"]


def gen_workload(n_cells, grid, dist, seed=0):
    """Return px,py,sx,sy float32 arrays (length n_cells) in bin units."""
    rng = np.random.default_rng(seed)
    # sizes: 95% sub-bin std cells, 5% multi-bin macros
    macro = rng.random(n_cells) < 0.05
    sx = np.where(macro, rng.uniform(1.5, 8.0, n_cells), rng.uniform(0.10, 0.80, n_cells)).astype(np.float32)
    sy = np.where(macro, rng.uniform(1.5, 8.0, n_cells), rng.uniform(0.10, 0.80, n_cells)).astype(np.float32)
    G = float(grid)
    if dist == "uniform":
        px = rng.uniform(0, G, n_cells).astype(np.float32)
        py = rng.uniform(0, G, n_cells).astype(np.float32)
    elif dist == "cluster":
        sd = G * 0.08
        px = (G * 0.5 + rng.normal(0, sd, n_cells)).astype(np.float32)
        py = (G * 0.5 + rng.normal(0, sd, n_cells)).astype(np.float32)
    elif dist == "clouds":
        K, sd = 8, G * 0.04
        cx = rng.uniform(0, G, K); cy = rng.uniform(0, G, K)
        k = rng.integers(0, K, n_cells)
        px = (cx[k] + rng.normal(0, sd, n_cells)).astype(np.float32)
        py = (cy[k] + rng.normal(0, sd, n_cells)).astype(np.float32)
    else:
        raise ValueError(f"unknown dist {dist}")
    # clamp so cell fits in [0, grid)
    px = np.clip(px, 0.0, np.maximum(G - sx, 0.0)).astype(np.float32)
    py = np.clip(py, 0.0, np.maximum(G - sy, 0.0)).astype(np.float32)
    return px, py, sx, sy


def ref_density(px, py, sx, sy, grid, chunk=200000):
    """Exact fp64 density map flat [grid*grid] (bin_global = bx*grid+by),
    box-overlap model matching v4_compute. area deposited = ox*oy.
    Vectorized + chunked np.bincount (fast, bounded RAM even at bigblue3)."""
    j = np.arange(8)
    D = np.zeros(grid * grid, dtype=np.float64)
    n = len(px)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        pxc = px[s:e].astype(np.float64); pyc = py[s:e].astype(np.float64)
        sxc = sx[s:e].astype(np.float64); syc = sy[s:e].astype(np.float64)
        bxl = np.floor(pxc).astype(np.int64); byl = np.floor(pyc).astype(np.int64)
        binx = bxl[:, None] + j[None, :]                       # (m,8)
        lo = np.maximum(binx.astype(np.float64), pxc[:, None])
        hi = np.minimum(binx.astype(np.float64) + 1.0, (pxc + sxc)[:, None])
        ox = np.maximum(0.0, hi - lo)
        biny = byl[:, None] + j[None, :]
        lo = np.maximum(biny.astype(np.float64), pyc[:, None])
        hi = np.minimum(biny.astype(np.float64) + 1.0, (pyc + syc)[:, None])
        oy = np.maximum(0.0, hi - lo)
        area = ox[:, :, None] * oy[:, None, :]                 # (m,8,8)
        bx = np.broadcast_to(binx[:, :, None], area.shape)
        by = np.broadcast_to(biny[:, None, :], area.shape)
        valid = (area > 0) & (bx >= 0) & (bx < grid) & (by >= 0) & (by < grid)
        bg = (bx * grid + by)[valid].ravel()
        w = area[valid].ravel()
        if bg.size:
            D += np.bincount(bg, weights=w, minlength=grid * grid)
    return D


def rel_l2(got, ref):
    got = np.asarray(got, dtype=np.float64).ravel()
    ref = np.asarray(ref, dtype=np.float64).ravel()
    n = min(got.size, ref.size)
    d = got[:n] - ref[:n]
    den = np.sqrt(np.sum(ref[:n] ** 2))
    return float(np.sqrt(np.sum(d ** 2)) / den) if den > 0 else float(np.sqrt(np.sum(d ** 2)))
