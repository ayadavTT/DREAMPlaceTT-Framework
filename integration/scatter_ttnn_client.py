"""
scatter_ttnn_client.py — host-side DREAMPlace integration for Path 1.

Monkey-patches ElectricPotentialFunction.forward to send cell positions to
the C++ density_scatter_ttnn_server binary.

IPC: POSIX shared memory (mmap of a file in ipc_dir).
  scatter.shm — header (64 B) + px/py/sx/sy[soa_padded] + field_x/y[M*N]
  ready.flag  — written once by server after JIT+init (startup sync)

Header layout (64 bytes, little-endian):
  [0:4]   state        (u32: 0=idle 1=go 2=done 3=quit)
  [4:8]   nc_actual    (i32)
  [8:44]  timing[9]    (f32: h2d scatter gather d2h_den upload compute download fw total)
  [44:48] gather_mode  (u32: 0=v6 1=v7 2=v8)
  [48:64] padding

Usage:
    import scatter_ttnn_client
    scatter_ttnn_client.patch_dreamplace(
        container="",            # unused when direct=True
        ipc_dir="/dev/shm/tt_scatter_ipc",
        num_cells=0,             # 0 = auto
        direct=True,             # launch server binary directly (no docker exec)
    )
"""

import atexit
import mmap
import os
import struct
import subprocess
import sys
import threading
import time
from typing import Optional

import numpy as np
import torch


# ── Per-iter density audit (TT vs CPU) ────────────────────────────────────────
# Active when DENSITY_AUDIT env var is set. Writes per-iter rel_L2 and
# max_abs_err to <DENSITY_AUDIT_DIR>/density_audit.csv, and per-iter cell
# positions to <DIR>/pos_iter{N}.npz for the first DENSITY_AUDIT_MAX_ITERS
# (default 700). Compares TT density against a pure-fp32 CPU reference
# computed on the SAME positions, so any deviation is attributable to the
# TT pipeline alone.
_audit_iter = 0
_audit_csv = None
_audit_dir = None
_audit_max_iters = 700


def _cpu_density_fp32(px, py, sx, sy, M, N, xl, yl, bsx, bsy):
    """Vectorized pure-fp32 density (matches DREAMPlace CPU electric_density)."""
    inv_bsx = np.float32(1.0 / bsx)
    inv_bsy = np.float32(1.0 / bsy)
    bxl = np.maximum(np.floor((px - xl) * inv_bsx).astype(np.int32), 0)
    byl = np.maximum(np.floor((py - yl) * inv_bsy).astype(np.int32), 0)
    j_arr = np.arange(8, dtype=np.int32)
    bx_grid = bxl[:, None] + j_arr[None, :]  # (n, 8)
    by_grid = byl[:, None] + j_arr[None, :]
    bx_left = xl + bx_grid.astype(np.float32) * np.float32(bsx)
    bx_right = bx_left + np.float32(bsx)
    by_left = yl + by_grid.astype(np.float32) * np.float32(bsy)
    by_right = by_left + np.float32(bsy)
    ox = np.maximum(0.0,
                    np.minimum(px[:, None] + sx[:, None], bx_right)
                    - np.maximum(px[:, None], bx_left)).astype(np.float32)
    oy = np.maximum(0.0,
                    np.minimum(py[:, None] + sy[:, None], by_right)
                    - np.maximum(py[:, None], by_left)).astype(np.float32)
    bx_ok = (bx_grid >= 0) & (bx_grid < M)
    by_ok = (by_grid >= 0) & (by_grid < N)
    ox = np.where(bx_ok, ox, np.float32(0.0))
    oy = np.where(by_ok, oy, np.float32(0.0))
    valid_cell = (sx > 0) & (sy > 0)
    ox = np.where(valid_cell[:, None], ox, np.float32(0.0))
    oy = np.where(valid_cell[:, None], oy, np.float32(0.0))

    density_flat = np.zeros(M * N, dtype=np.float64)
    # 64 bincount calls; each scatters one (j, k) outer-product contribution.
    for j in range(8):
        bx_j = bx_grid[:, j]
        for k in range(8):
            by_k = by_grid[:, k]
            v = ox[:, j] * oy[:, k]
            mask = bx_ok[:, j] & by_ok[:, k]
            bins = np.where(mask, bx_j * N + by_k, 0).astype(np.intp)
            wts = np.where(mask, v, 0.0).astype(np.float64)
            density_flat += np.bincount(bins, weights=wts, minlength=M * N)
    return density_flat.reshape(M, N).astype(np.float32)


def _density_audit(px, py, sx, sy, tt_density, M, N, xl, yl, xh, yh):
    """Compare TT density to fp32 CPU reference; emit per-iter row to CSV."""
    global _audit_iter, _audit_csv, _audit_dir, _audit_max_iters
    if _audit_csv is None:
        _audit_dir = os.environ.get("DENSITY_AUDIT_DIR",
                                    "/localdev/ayadav/tt-work/TTPort/DREAMPlaceTT-Framework/results/density_audit")
        try:
            _audit_max_iters = int(os.environ.get("DENSITY_AUDIT_MAX_ITERS", "700"))
        except ValueError:
            _audit_max_iters = 700
        os.makedirs(_audit_dir, exist_ok=True)
        _audit_csv = open(os.path.join(_audit_dir, "density_audit.csv"), "w")
        _audit_csv.write("iter,rel_l2_pct,max_abs_err,max_abs_err_rel_to_max_bin_pct,"
                         "mass_deficit_pct,tt_sum,cpu_sum,tt_max,cpu_max,n_real\n")
        _audit_csv.flush()
        print(f"[density_audit] writing CSV+positions to {_audit_dir} "
              f"(positions for first {_audit_max_iters} iters)", flush=True)

    bsx = (xh - xl) / M
    bsy = (yh - yl) / N
    inv_ba = 1.0 / (bsx * bsy)
    n = len(px)
    n_real = int(((sx > 0) & (sy > 0)).sum())

    cpu = _cpu_density_fp32(px.astype(np.float32), py.astype(np.float32),
                            sx.astype(np.float32), sy.astype(np.float32),
                            M, N, np.float32(xl), np.float32(yl), bsx, bsy)
    # Server normalizes density by inv_ba before writing to shm; apply same on CPU side.
    cpu = cpu * np.float32(inv_ba)
    diff = tt_density - cpu
    l2_ref = float(np.linalg.norm(cpu.flatten()))
    rel_l2 = float(np.linalg.norm(diff.flatten()) / max(l2_ref, 1e-12)) * 100.0
    max_abs = float(np.abs(diff).max())
    max_bin = float(cpu.max())
    max_abs_rel = max_abs / max(max_bin, 1e-12) * 100.0
    mass_def = (1.0 - float(tt_density.sum()) / max(float(cpu.sum()), 1e-12)) * 100.0

    _audit_csv.write(f"{_audit_iter},{rel_l2:.6f},{max_abs:.6e},{max_abs_rel:.4f},"
                     f"{mass_def:.6f},{tt_density.sum():.4e},{cpu.sum():.4e},"
                     f"{tt_density.max():.4f},{max_bin:.4f},{n_real}\n")
    _audit_csv.flush()

    if _audit_iter < _audit_max_iters:
        np.savez_compressed(os.path.join(_audit_dir, f"pos_iter{_audit_iter:04d}.npz"),
                            px=px.astype(np.float32), py=py.astype(np.float32),
                            sx=sx.astype(np.float32), sy=sy.astype(np.float32),
                            M=M, N=N, xl=xl, yl=yl, xh=xh, yh=yh)

    if _audit_iter < 5 or (_audit_iter % 50 == 0):
        print(f"[density_audit] iter {_audit_iter}: rel_L2={rel_l2:.4f}% "
              f"max_abs={max_abs:.3e} (={max_abs_rel:.3f}% of cpu_max {max_bin:.2f}) "
              f"mass_def={mass_def:.4f}%", flush=True)

    _audit_iter += 1

# ── Shared-memory header constants ────────────────────────────────────────────
_SHM_STATE_IDLE = 0
_SHM_STATE_GO   = 1
_SHM_STATE_DONE = 2
_SHM_STATE_QUIT = 3
_SHM_HEADER_SIZE = 64       # one cache line
# struct offsets within header
_OFF_STATE    = 0            # uint32
_OFF_NC       = 4            # int32
_OFF_TIMINGS  = 8            # 9× float32  → bytes 8..44
_OFF_GMODE    = 44           # uint32
# struct format for reading timing after DONE (excludes padding)
_HDR_TIMING_FMT = '<IifffffffffI'   # state nc h2d sc ga d2h ul co dl fw tot gm


_client: Optional["ScatterTTNNClient"] = None


_THIS_DIR = os.path.dirname(os.path.realpath(__file__))
_ROOT     = os.path.dirname(_THIS_DIR)
# DREAMPLACE_TT_SERVER_BINARY env var overrides the default (used by CI).
# Default = $framework_root/host/build/density_scatter_ttnn_server
_SERVER_BINARY = os.environ.get(
    "DREAMPLACE_TT_SERVER_BINARY",
    os.path.join(_ROOT, "host", "build", "density_scatter_ttnn_server"),
)
# TT_METAL_HOME env var should be set by the user (Docker bind-mount).
# Default falls back to ../tt-metal/ relative to the framework root.
_TT_METAL_HOME = os.environ.get(
    "TT_METAL_HOME",
    os.path.join(_ROOT, "tt-metal"),
)


class ScatterTTNNClient:
    def __init__(self, container: str, ipc_dir: str,
                 num_bins_x: int = 0, num_bins_y: int = 0,
                 nc_max: int = 0,
                 xl: float = 0.0, yl: float = 0.0,
                 xh: float = 1e6, yh: float = 1e6,
                 direct: bool = False):
        self.container = container
        self.ipc_dir   = ipc_dir
        self.direct    = direct
        self.M = num_bins_x
        self.N = num_bins_y
        self.nc_max = nc_max
        self.xl, self.yl, self.xh, self.yh = xl, yl, xh, yh

        self.ready_flag = os.path.join(ipc_dir, "ready.flag")
        self.shm_path   = os.path.join(ipc_dir, "scatter.shm")

        self._proc  = None
        self._ready = False
        self._timings: list = []

        # Shared-memory state (set up in start())
        self._shm_fd: int = -1
        self._mm:     Optional[mmap.mmap] = None
        self._shm_px: Optional[np.ndarray] = None  # writable views into shm
        self._shm_py: Optional[np.ndarray] = None
        self._shm_sx: Optional[np.ndarray] = None
        self._shm_sy: Optional[np.ndarray] = None
        self._shm_fx: Optional[np.ndarray] = None  # field_x[M,N]
        self._shm_fy: Optional[np.ndarray] = None  # field_y[M,N]
        self._soa_padded: int = 0

        self._sort_idx:  Optional[np.ndarray] = None
        self._sx_sorted: Optional[np.ndarray] = None
        self._sy_sorted: Optional[np.ndarray] = None

    # ── Shared-memory lifecycle ───────────────────────────────────────────────

    def _create_shm(self) -> None:
        """Create and mmap the shared memory file; set up numpy array views."""
        soa_padded   = ((self.nc_max + 1023) // 1024) * 1024
        pos_bytes    = soa_padded * 4          # float32
        field_bytes  = self.M * self.N * 4
        # Layout adds one extra field-sized slot (M*N float32) for
        # initial_density_map_normalized so the server can fold fixed-terminal
        # density into density_flat BEFORE the TT DCT. Required for TT-DCT
        # convergence (when CPU_DCT=0). The slot is written once by the client.
        total_size   = _SHM_HEADER_SIZE + 4 * pos_bytes + 3 * field_bytes

        self._shm_fd = os.open(self.shm_path,
                               os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o666)
        os.ftruncate(self._shm_fd, total_size)

        self._mm = mmap.mmap(self._shm_fd, total_size,
                             access=mmap.ACCESS_WRITE)
        self._soa_padded = soa_padded

        # Writable numpy views directly into the mmap region
        off = _SHM_HEADER_SIZE
        self._shm_px = np.ndarray((soa_padded,), dtype=np.float32,
                                  buffer=self._mm, offset=off)
        off += pos_bytes
        self._shm_py = np.ndarray((soa_padded,), dtype=np.float32,
                                  buffer=self._mm, offset=off)
        off += pos_bytes
        self._shm_sx = np.ndarray((soa_padded,), dtype=np.float32,
                                  buffer=self._mm, offset=off)
        off += pos_bytes
        self._shm_sy = np.ndarray((soa_padded,), dtype=np.float32,
                                  buffer=self._mm, offset=off)
        off += pos_bytes
        self._shm_fx = np.ndarray((self.M, self.N), dtype=np.float32,
                                  buffer=self._mm, offset=off)
        off += field_bytes
        self._shm_fy = np.ndarray((self.M, self.N), dtype=np.float32,
                                  buffer=self._mm, offset=off)
        off += field_bytes
        # initial_density_map_normalized (= initial_density_map / bin_area).
        # Written once by the client; consumed by the server before the TT DCT.
        self._shm_id = np.ndarray((self.M, self.N), dtype=np.float32,
                                  buffer=self._mm, offset=off)

        # Zero the header state word so server starts in IDLE
        struct.pack_into('<I', self._mm, _OFF_STATE, _SHM_STATE_IDLE)
        print(f"[scatter_ttnn] shm created: {self.shm_path} "
              f"({total_size // 1024} KB, soa_padded={soa_padded})", flush=True)

    def _close_shm(self) -> None:
        if self._mm is not None:
            try:
                self._mm.close()
            except Exception:
                pass
            self._mm = None
        if self._shm_fd >= 0:
            try:
                os.close(self._shm_fd)
            except Exception:
                pass
            self._shm_fd = -1
        try:
            os.unlink(self.shm_path)
        except FileNotFoundError:
            pass

    # ── Server lifecycle ──────────────────────────────────────────────────────

    def start(self, M: int = 0, N: int = 0, nc_actual: int = 0,
              xl: float = 0.0, yl: float = 0.0,
              xh: float = 1e6, yh: float = 1e6) -> None:
        if self._ready:
            return

        if M: self.M = M
        if N: self.N = N
        if xl or yl or xh != 1e6 or yh != 1e6:
            self.xl, self.yl, self.xh, self.yh = xl, yl, xh, yh

        if self.nc_max == 0:
            if nc_actual == 0:
                raise ValueError("nc_max=0 and no nc_actual supplied")
            self.nc_max = nc_actual
            print(f"[scatter_ttnn] nc_max={self.nc_max} (= nc_actual)", flush=True)

        # Create shm BEFORE launching the server so it can open() it right away
        os.makedirs(self.ipc_dir, exist_ok=True)
        try:
            os.remove(self.ready_flag)
        except FileNotFoundError:
            pass
        self._create_shm()

        gather_mode = os.environ.get("GATHER_MODE", "auto")
        server_args = [
            str(self.M), str(self.N), str(self.nc_max), self.ipc_dir,
            str(self.xl), str(self.yl), str(self.xh), str(self.yh),
        ]

        if self.direct:
            if not os.path.exists(_SERVER_BINARY):
                raise FileNotFoundError(
                    f"[scatter_ttnn] Server binary not found: {_SERVER_BINARY}\n"
                    "Build with: cd experiments/density_scatter/tt_metal && "
                    "cmake --build build_ttnn --target density_scatter_ttnn_server"
                )
            env = {
                **os.environ,
                "TT_METAL_HOME":            _TT_METAL_HOME,
                "TT_METAL_RUNTIME_ROOT":    _TT_METAL_HOME,
                "ARCH_NAME":                os.environ.get("ARCH_NAME", "blackhole"),
                "GATHER_MODE":              gather_mode,
                "LD_LIBRARY_PATH":          (
                    os.path.join(_TT_METAL_HOME, "build_Release", "lib") + ":"
                    + os.environ.get("LD_LIBRARY_PATH", "")
                ),
                "TT_METAL_LOGGER_LEVEL":    os.environ.get("TT_METAL_LOGGER_LEVEL", "WARNING"),
                "TT_METAL_DEVICE_PROFILER": os.environ.get("TT_METAL_DEVICE_PROFILER", "0"),
            }
            cmd = [_SERVER_BINARY] + server_args
            print(f"[scatter_ttnn] Launching server (direct): {_SERVER_BINARY}", flush=True)
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, env=env)
        else:
            # Launch C++ binary inside the container via docker exec.
            # ipc_dir must be on a shared filesystem (NFS) visible from both host and container.
            # Both sides mmap ipc_dir/scatter.shm — same page cache on the same physical host.
            tt_lib = os.path.join(_TT_METAL_HOME, "build_Release", "lib")
            cmd = [
                "docker", "exec",
                "-e", f"GATHER_MODE={gather_mode}",
                "-e", f"EXPORT_DENSITY_PATH={os.environ.get('EXPORT_DENSITY_PATH', '')}",
                "-e", f"EXPORT_POS_PATH={os.environ.get('EXPORT_POS_PATH', '')}",
                "-e", f"EXPORT_ROUTE_BUF_PATH={os.environ.get('EXPORT_ROUTE_BUF_PATH', '')}",
                "-e", f"TT_METAL_HOME={_TT_METAL_HOME}",
                "-e", f"TT_METAL_RUNTIME_ROOT={_TT_METAL_HOME}",
                "-e", f"ARCH_NAME={os.environ.get('ARCH_NAME', 'blackhole')}",
                "-e", f"TT_METAL_LOGGER_LEVEL={os.environ.get('TT_METAL_LOGGER_LEVEL', 'WARNING')}",
                "-e", f"TT_METAL_DEVICE_PROFILER={os.environ.get('TT_METAL_DEVICE_PROFILER', '0')}",
                "-e", f"CPU_DCT={os.environ.get('CPU_DCT', '')}",
                "-e", f"LD_LIBRARY_PATH={tt_lib}",
                self.container,
                _SERVER_BINARY,
            ] + server_args
            print(f"[scatter_ttnn] Launching server via docker exec: "
                  f"container={self.container}", flush=True)
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        def _log():
            for line in self._proc.stdout:
                print(f"[scatter_ttnn_srv] {line}", end="", flush=True)
        threading.Thread(target=_log, daemon=True).start()

        print("[scatter_ttnn] Waiting for ready.flag (JIT ~60 s on cold cache)…",
              flush=True)
        deadline = time.monotonic() + 600
        while not os.path.exists(self.ready_flag):
            if time.monotonic() > deadline:
                raise TimeoutError("Timeout waiting for scatter_ttnn server ready.flag")
            time.sleep(0.1)
        print("[scatter_ttnn] Server READY.", flush=True)

        self._ready = True
        atexit.register(self.stop)

    def stop(self) -> None:
        if not self._ready:
            return
        # Signal QUIT via shm state word
        if self._mm is not None:
            try:
                struct.pack_into('<I', self._mm, _OFF_STATE, _SHM_STATE_QUIT)
            except Exception:
                pass
        if self._proc:
            try:
                self._proc.wait(timeout=20)
            except Exception:
                pass
        self._close_shm()
        self._ready = False

    # ── Per-iteration compute call ────────────────────────────────────────────

    def call(self,
             px: np.ndarray, py: np.ndarray,
             sx: np.ndarray, sy: np.ndarray) -> tuple:
        nc = len(px)

        # One-time cell sort for load balancing (largest cells → lowest tile indices)
        if self._sort_idx is None or len(self._sort_idx) != nc:
            area_rank = np.argsort(sx * sy)[::-1].copy()
            n_tiles   = (nc + 1023) // 1024
            ranks     = np.arange(nc)
            dest      = (ranks % n_tiles) * 1024 + (ranks // n_tiles)
            sort_idx  = np.full(nc, area_rank[0], dtype=np.int64)
            valid     = dest < nc
            sort_idx[dest[valid]] = area_rank[valid]
            # BUG FIX: cells whose rank has dest >= nc never get placed in
            # sort_idx, so their original index is permanently lost from the
            # scatter. For adaptec1_512 with nc=370,971 and n_tiles=363, that
            # silently dropped 738 cells (0.199% of the scatter set) every iter
            # — matching the observed 0.19% mass deficit and explaining why
            # V13_fpu's adaptec1_512 trajectory drifts from CPU.
            # Fix: place the cells whose ranks had dest>=nc at the unfilled
            # slots (positions in [0,nc) not covered by dest[valid]).
            invalid_ranks = np.flatnonzero(~valid)
            unfilled_positions = np.setdiff1d(np.arange(nc),
                                               dest[valid], assume_unique=False)
            assert len(invalid_ranks) == len(unfilled_positions), \
                f"sort fix: {len(invalid_ranks)} dropped ranks vs " \
                f"{len(unfilled_positions)} unfilled positions"
            sort_idx[unfilled_positions] = area_rank[invalid_ranks]
            print(f"[scatter_ttnn] sort fix: recovered {len(invalid_ranks)} "
                  f"dropped cells (was {len(invalid_ranks)*100/nc:.3f}% mass loss)",
                  flush=True)
            self._sort_idx  = sort_idx
            # All positions now point to a unique original cell. invalid_mask
            # is empty.
            self._invalid_mask = np.zeros(nc, dtype=bool)
            self._sx_sorted = sx[sort_idx].astype(np.float32)
            self._sy_sorted = sy[sort_idx].astype(np.float32)

        idx  = self._sort_idx
        px_s = px[idx].astype(np.float32)
        py_s = py[idx].astype(np.float32)

        # Write positions directly into shm (zero-copy into mapped memory)
        t0 = time.perf_counter()
        self._shm_px[:nc] = px_s
        self._shm_py[:nc] = py_s
        self._shm_sx[:nc] = self._sx_sorted
        self._shm_sy[:nc] = self._sy_sorted
        pos_write_ms = (time.perf_counter() - t0) * 1000

        # Signal GO: write nc_actual then set state (ordering matters on non-TSO arches)
        struct.pack_into('<i', self._mm, _OFF_NC,    nc)
        struct.pack_into('<I', self._mm, _OFF_STATE, _SHM_STATE_GO)

        # Wait for DONE
        t0 = time.perf_counter()
        deadline = time.monotonic() + 120
        while True:
            st = struct.unpack_from('<I', self._mm, _OFF_STATE)[0]
            if st == _SHM_STATE_DONE:
                break
            if time.monotonic() > deadline:
                raise TimeoutError("[scatter_ttnn] Timeout waiting for shm DONE")
            time.sleep(0.0005)
        kernel_wait_ms = (time.perf_counter() - t0) * 1000

        # Read timing from header
        (_, _, h2d, scatter, gather, d2h_den,
         upload, compute, download, fw, total, gm_int) = \
            struct.unpack_from(_HDR_TIMING_FMT, self._mm, 0)

        # Reset state to IDLE so server can accept next request
        struct.pack_into('<I', self._mm, _OFF_STATE, _SHM_STATE_IDLE)

        # Copy fields out of shm (server may overwrite next iteration)
        t0 = time.perf_counter()
        field_x = torch.from_numpy(self._shm_fx.copy())
        field_y = torch.from_numpy(self._shm_fy.copy())
        field_read_ms = (time.perf_counter() - t0) * 1000

        timing = {
            "pos_write_ms":    pos_write_ms,
            "kernel_wait_ms":  kernel_wait_ms,
            "field_read_ms":   field_read_ms,
            "h2d_ms":          float(h2d),
            "scatter_ms":      float(scatter),
            "gather_ms":       float(gather),
            "d2h_density_ms":  float(d2h_den),
            "ttnn_upload_ms":  float(upload),
            "ttnn_compute_ms": float(compute),
            "ttnn_download_ms":float(download),
            "fw_ms":           float(fw),
            "total_server_ms": float(total),
            "total_client_ms": pos_write_ms + kernel_wait_ms + field_read_ms,
            "gather_mode":     {0: "v6", 1: "v7", 2: "v8", 3: "v9", 4: "v10", 5: "v11"}.get(int(gm_int), "v6"),
        }
        self._timings.append(timing)
        return field_x, field_y, timing

    def timing_summary(self, n_warmup: int = 4) -> dict:
        timings = self._timings[n_warmup:] if len(self._timings) > n_warmup else self._timings
        if not timings:
            timings = self._timings
        if not timings:
            return {}
        keys = list(timings[0].keys())
        result = {}
        for k in keys:
            vals = [t[k] for t in timings if k in t]
            if not vals:
                continue
            if isinstance(vals[0], (int, float)):
                result[f"{k}_mean"]   = float(np.mean(vals))
                result[f"{k}_median"] = float(np.median(vals))
            else:
                # non-numeric field (e.g. gather_mode string): report most common value
                result[k] = max(set(vals), key=vals.count)
        result["n_iters"] = len(self._timings)
        result["n_timed"] = len(timings)
        return result


def patch_dreamplace(container: str, ipc_dir: str, num_cells: int = 0,
                     direct: bool = False) -> None:
    global _client
    _client = ScatterTTNNClient(container=container, ipc_dir=ipc_dir, nc_max=num_cells,
                                direct=direct)
    os.makedirs(ipc_dir, exist_ok=True)

    # Save the original CPU forward so we can fall back to it after a configurable
    # iter count (env var V13_USE_CPU_AFTER_ITER). Used to test "run TT for the
    # iters where it's bit-accurate, switch to CPU before the trajectory drifts."
    try:
        import dreamplace.ops.electric_potential.electric_potential as _ep_mod_save
        _orig_ep_forward = (_ep_mod_save.ElectricPotentialFunction.forward.__func__
                            if hasattr(_ep_mod_save.ElectricPotentialFunction.forward, "__func__")
                            else _ep_mod_save.ElectricPotentialFunction.forward)
    except Exception:
        _orig_ep_forward = None
    _iter_counter = [0]

    @staticmethod
    def _scatter_ttnn_forward(
        ctx, pos,
        node_size_x_clamped, node_size_y_clamped,
        offset_x, offset_y, ratio,
        bin_center_x, bin_center_y,
        initial_density_map, target_density,
        xl, yl, xh, yh,
        bin_size_x, bin_size_y,
        num_movable_nodes, num_filler_nodes,
        padding, padding_mask,
        num_bins_x, num_bins_y,
        num_movable_impacted_bins_x, num_movable_impacted_bins_y,
        num_filler_impacted_bins_x, num_filler_impacted_bins_y,
        deterministic_flag, sorted_node_map,
        exact_expkM=None, exact_expkN=None,
        inv_wu2_plus_wv2=None,
        wu_by_wu2_plus_wv2_half=None,
        wv_by_wu2_plus_wv2_half=None,
        dct2=None, idct2=None, idct_idxst=None, idxst_idct=None,
        fast_mode=True,
    ):
        import time as _time

        # Fall back to original CPU forward after V13_USE_CPU_AFTER_ITER iters.
        # The diagnostic test: TT density is bit-accurate early but the trajectory
        # drifts from CPU due to bf16 noise compounding; if we cut over to CPU
        # before the drift becomes catastrophic, does DREAMPlace converge?
        _iter_counter[0] += 1
        try:
            _cpu_after = int(os.environ.get("V13_USE_CPU_AFTER_ITER", "0"))
        except ValueError:
            _cpu_after = 0
        if _cpu_after > 0 and _iter_counter[0] > _cpu_after and _orig_ep_forward is not None:
            if _iter_counter[0] == _cpu_after + 1:
                print(f"[scatter_ttnn] iter {_iter_counter[0]}: cutting over to CPU forward "
                      f"(V13_USE_CPU_AFTER_ITER={_cpu_after})", flush=True)
            return _orig_ep_forward(
                ctx, pos,
                node_size_x_clamped, node_size_y_clamped,
                offset_x, offset_y, ratio,
                bin_center_x, bin_center_y,
                initial_density_map, target_density,
                xl, yl, xh, yh, bin_size_x, bin_size_y,
                num_movable_nodes, num_filler_nodes,
                padding, padding_mask, num_bins_x, num_bins_y,
                num_movable_impacted_bins_x, num_movable_impacted_bins_y,
                num_filler_impacted_bins_x, num_filler_impacted_bins_y,
                deterministic_flag, sorted_node_map,
                exact_expkM, exact_expkN,
                inv_wu2_plus_wv2, wu_by_wu2_plus_wv2_half, wv_by_wu2_plus_wv2_half,
                dct2, idct2, idct_idxst, idxst_idct, fast_mode)

        t_total = _time.perf_counter()

        num_nodes = pos.shape[0] // 2
        nc = int(num_movable_nodes) + int(num_filler_nodes)

        # Raw position + size for movable+filler cells.
        # DREAMPlace pos layout is [movable | fixed_terminals | filler] — fixed
        # terminals are interleaved between movable and filler ranges. So
        # pos[:nc] WOULD INCORRECTLY include fixed terminals (scattering them
        # as if they were movable; they're already in initial_density_map →
        # double count) and miss the last num_terminals filler cells. Build
        # the correct scatter slice explicitly.
        #
        # CRITICAL: CPU DREAMPlace uses CLAMPED size (>= bin*sqrt2) with a
        # `+ offset_x` position shift AND a per-cell `ratio = orig_area / clamped_area`
        # post-multiplier on the density (electric_density_map.cpp:277-301). Our V13
        # kernel applies neither — it sends raw pos with clamped size. At grid 512
        # ALL movable cells are stretched (median ratio=0.179) → TT density is
        # ~5.6x larger per cell than CPU → density signal hugely amplified →
        # divergence. Fix: send ORIGINAL cell sizes with ORIGINAL positions.
        # This preserves per-cell mass exactly (orig_area) without kernel changes.
        # orig_x = clamped_x + 2*offset_x  (since offset_x = (orig - clamped)/2).
        import numpy as _np
        n_mov = int(num_movable_nodes)
        n_fil = int(num_filler_nodes)
        # Recover original sizes from clamped + 2*offset (= orig)
        orig_sx_all = (node_size_x_clamped + 2.0 * offset_x).detach().cpu().numpy()
        orig_sy_all = (node_size_y_clamped + 2.0 * offset_y).detach().cpu().numpy()
        px_full = _np.concatenate([
            pos[:n_mov].detach().cpu().numpy(),
            pos[num_nodes - n_fil:num_nodes].detach().cpu().numpy(),
        ])
        py_full = _np.concatenate([
            pos[num_nodes:num_nodes + n_mov].detach().cpu().numpy(),
            pos[2 * num_nodes - n_fil:2 * num_nodes].detach().cpu().numpy(),
        ])
        sx_full = _np.concatenate([
            orig_sx_all[:n_mov],
            orig_sx_all[num_nodes - n_fil:num_nodes],
        ])
        sy_full = _np.concatenate([
            orig_sy_all[:n_mov],
            orig_sy_all[num_nodes - n_fil:num_nodes],
        ])

        # One-time scatter-set audit: verify we're NOT including fixed terminals.
        # Fixed terminals in adaptec1 are macros with sizes much larger than
        # standard cells / fillers — if any sx/sy here exceeds the largest
        # filler/movable size we expect, terminals leaked in.
        if not hasattr(_client, "_scatter_set_audited"):
            _client._scatter_set_audited = True
            n_total = num_nodes
            n_term = n_total - n_mov - n_fil
            full_sx_clamped = node_size_x_clamped.detach().cpu().numpy()
            full_sy_clamped = node_size_y_clamped.detach().cpu().numpy()
            term_sx = full_sx_clamped[n_mov:n_mov + n_term]
            term_sy = full_sy_clamped[n_mov:n_mov + n_term]
            mov_sx = full_sx_clamped[:n_mov]
            mov_sy = full_sy_clamped[:n_mov]
            fil_sx = full_sx_clamped[n_total - n_fil:n_total]
            fil_sy = full_sy_clamped[n_total - n_fil:n_total]
            print(f"[scatter_ttnn] SCATTER-SET AUDIT (one-time):", flush=True)
            print(f"  layout: total={n_total}, movable={n_mov}, fixed_terminals={n_term}, "
                  f"filler={n_fil}", flush=True)
            print(f"  Sent to TT: nc={len(sx_full)} cells "
                  f"(should be {n_mov} + {n_fil} = {n_mov + n_fil}, NOT including {n_term} terminals)",
                  flush=True)
            print(f"  movable sx: min={mov_sx.min():.2f} median={float(_np.median(mov_sx)):.2f} max={mov_sx.max():.2f}",
                  flush=True)
            print(f"  movable sy: min={mov_sy.min():.2f} median={float(_np.median(mov_sy)):.2f} max={mov_sy.max():.2f}",
                  flush=True)
            print(f"  filler  sx: min={fil_sx.min():.2f} median={float(_np.median(fil_sx)):.2f} max={fil_sx.max():.2f}",
                  flush=True)
            print(f"  filler  sy: min={fil_sy.min():.2f} median={float(_np.median(fil_sy)):.2f} max={fil_sy.max():.2f}",
                  flush=True)
            if n_term > 0:
                print(f"  TERMINAL sx: min={term_sx.min():.2f} median={float(_np.median(term_sx)):.2f} max={term_sx.max():.2f}",
                      flush=True)
                print(f"  TERMINAL sy: min={term_sy.min():.2f} median={float(_np.median(term_sy)):.2f} max={term_sy.max():.2f}",
                      flush=True)
            print(f"  Sent_sx: min={sx_full.min():.2f} median={float(_np.median(sx_full)):.2f} max={sx_full.max():.2f}",
                  flush=True)
            print(f"  Sent_sy: min={sy_full.min():.2f} median={float(_np.median(sy_full)):.2f} max={sy_full.max():.2f}",
                  flush=True)
            sent_max = max(float(sx_full.max()), float(sy_full.max()))
            term_max = max(float(term_sx.max()) if n_term > 0 else 0.0,
                            float(term_sy.max()) if n_term > 0 else 0.0)
            mvfil_max = max(float(mov_sx.max()), float(mov_sy.max()),
                             float(fil_sx.max()), float(fil_sy.max()))
            if sent_max > mvfil_max + 1.0:
                print(f"  ❌ FAIL: sent set has cell {sent_max:.2f} larger than any "
                      f"movable/filler ({mvfil_max:.2f}) — terminals leaked in", flush=True)
            else:
                print(f"  ✅ PASS: sent max ({sent_max:.2f}) within movable/filler bound "
                      f"({mvfil_max:.2f}); terminals (max {term_max:.2f}) NOT included",
                      flush=True)

        # Big-cell sub-tiling: V11's scatter kernel walks at most 8 bins per
        # cell in each direction; cells spanning >8 bins lose most of their
        # density. To fix without kernel changes, split each big cell into a
        # grid of sub-cells, each sized ≤7×bsx wide so they're guaranteed to
        # span ≤8 bins regardless of alignment (a 7×bsx cell starting mid-bin
        # covers 8 bins; a 8×bsx cell mid-bin would cover 9).
        # Layout (which cells split, how many subs each) is fixed by cell
        # shape, so we compute it once at first call and cache it.
        import numpy as _np
        if not hasattr(_client, "_subcell_layout") or _client._subcell_layout is None:
            bsx_f = (xh - xl) / int(num_bins_x)
            bsy_f = (yh - yl) / int(num_bins_y)
            V11_SAFE = 7  # sub-cell spans ≤7 full bin widths, max 8 bins covered
            nx = _np.maximum(1, _np.ceil(sx_full / (V11_SAFE * bsx_f))).astype(_np.int32)
            ny = _np.maximum(1, _np.ceil(sy_full / (V11_SAFE * bsy_f))).astype(_np.int32)
            big_mask = (nx > 1) | (ny > 1)
            big_idx = _np.flatnonzero(big_mask).astype(_np.int32)
            small_idx = _np.flatnonzero(~big_mask).astype(_np.int32)
            n_big = int(big_idx.size)
            n_sub_total = int((nx[big_mask] * ny[big_mask]).sum())
            new_nc = int(small_idx.size) + n_sub_total
            print(f"[scatter_ttnn] subcell: {n_big} big cells → {n_sub_total} sub-cells; "
                  f"nc {nc} → {new_nc} (Δ={new_nc-nc:+d}, +{(new_nc-nc)*100/max(1,nc):.1f}%)",
                  flush=True)
            # Pre-build the sub-cell index mapping. For each big cell k with
            # nx[k]*ny[k] sub-cells, store (i_offset_array, j_offset_array,
            # parent_idx_array) flattened.
            parent = _np.repeat(big_idx, nx[big_mask] * ny[big_mask])
            # i,j offsets within each cell's grid (flat: 0..nx*ny-1).
            # We use a clever trick: for each big cell with grid (kx,ky),
            # we want ij[0..kx*ky] = [(0,0),(0,1),...,(0,ky-1),(1,0),...].
            iflat = _np.empty(n_sub_total, dtype=_np.int32)
            jflat = _np.empty(n_sub_total, dtype=_np.int32)
            cur = 0
            for k in range(n_big):
                kx = int(nx[big_idx[k]]); ky = int(ny[big_idx[k]])
                cnt = kx * ky
                iflat[cur:cur+cnt] = _np.repeat(_np.arange(kx), ky)
                jflat[cur:cur+cnt] = _np.tile(_np.arange(ky), kx)
                cur += cnt
            # Sub-cell sizes: sub_dx = sx[big] / nx[big]. Per-parent.
            sub_dx_per_parent = (sx_full[big_idx] / nx[big_idx]).astype(_np.float32)
            sub_dy_per_parent = (sy_full[big_idx] / ny[big_idx]).astype(_np.float32)
            # Broadcast to flat per-sub arrays.
            parent_in_big = _np.repeat(_np.arange(n_big, dtype=_np.int32),
                                        nx[big_mask] * ny[big_mask])
            sub_dx_flat = sub_dx_per_parent[parent_in_big]
            sub_dy_flat = sub_dy_per_parent[parent_in_big]
            _client._subcell_layout = {
                "big_idx": big_idx,
                "small_idx": small_idx,
                "parent": parent,        # original cell index per sub-cell
                "iflat": iflat,
                "jflat": jflat,
                "sub_dx_flat": sub_dx_flat,
                "sub_dy_flat": sub_dy_flat,
                "new_nc": new_nc,
            }
        L = _client._subcell_layout

        if L["big_idx"].size > 0:
            parent = L["parent"]
            sub_dx_flat = L["sub_dx_flat"]
            sub_dy_flat = L["sub_dy_flat"]
            # Sub-cell positions: parent_position + i*sub_dx, parent_position + j*sub_dy.
            sub_px = px_full[parent] + L["iflat"] * sub_dx_flat
            sub_py = py_full[parent] + L["jflat"] * sub_dy_flat
            # Build final arrays: small cells first, then sub-cells.
            px = _np.concatenate([px_full[L["small_idx"]], sub_px]).astype(_np.float32)
            py = _np.concatenate([py_full[L["small_idx"]], sub_py]).astype(_np.float32)
            sx = _np.concatenate([sx_full[L["small_idx"]], sub_dx_flat]).astype(_np.float32)
            sy = _np.concatenate([sy_full[L["small_idx"]], sub_dy_flat]).astype(_np.float32)
            nc_send = int(px.size)
        else:
            px, py, sx, sy = px_full, py_full, sx_full, sy_full
            nc_send = nc

        if not _client._ready:
            _client.xl, _client.yl = float(xl), float(yl)
            _client.xh, _client.yh = float(xh), float(yh)
            _client.M, _client.N   = int(num_bins_x), int(num_bins_y)
            _client.start(nc_actual=nc_send)

        # Write initial_density_map (already normalized by bin_area) into the
        # shm slot ONCE. The server reads it and folds it into density_flat
        # before the TT DCT — required for convergence when CPU_DCT=0 because
        # the V11 scatter pipeline only emits movable+filler density. The
        # CPU_DCT=1 path still does its own add-back below.
        if not getattr(_client, "_id_uploaded", False):
            bin_area_local = float(bin_size_x) * float(bin_size_y)
            _id_np = (initial_density_map / bin_area_local).detach().cpu().numpy().astype(_np.float32)
            _client._shm_id[:] = _id_np
            _client._id_uploaded = True

        field_x, field_y, timing = _client.call(px, py, sx, sy)

        # CPU_DCT diagnostic mode: the server wrote the bf16-decoded fp32
        # density into the fx slot (fy is zero). Run the DCT/IDCT chain on
        # CPU in fp32 using the operators DREAMPlace itself uses on CPU,
        # so we can isolate the TTNN bf16 DCT precision from the kernel
        # density precision.
        if os.environ.get("CPU_DCT") == "1":
            import time as _t_cpu_dct
            t_cd = _t_cpu_dct.perf_counter()
            density_map = field_x.to(pos.dtype)  # TT scatter density (movable+filler), normalized by inv_ba

            # CRITICAL FIX: V13_fpu server does NOT apply initial_density_map.
            # It only scatters movable+filler. The fixed-cell terminal density
            # is in DREAMPlace's `initial_density_map` (unnormalized) and must
            # be added here, normalized to match TT density's bin-area units.
            # Without this, fixed terminal density is missing from the field
            # solve → terminal regions don't repel cells → divergence at 512.
            bin_area = float(bin_size_x) * float(bin_size_y)
            density_map = density_map + (initial_density_map / bin_area).to(pos.dtype)

            if os.environ.get("DENSITY_AUDIT") == "1":
                tt_dens_np = density_map.detach().cpu().numpy()
                _density_audit(px, py, sx, sy, tt_dens_np,
                                int(num_bins_x), int(num_bins_y),
                                float(xl), float(yl), float(xh), float(yh))

            auv = dct2.forward(density_map)
            auv_wu = auv.mul(wu_by_wu2_plus_wv2_half)
            auv_wv = auv.mul(wv_by_wu2_plus_wv2_half)
            field_x = idxst_idct.forward(auv_wu)
            field_y = idct_idxst.forward(auv_wv)
            timing["cpu_dct_ms"] = (_t_cpu_dct.perf_counter() - t_cd) * 1000

        ctx.field_map_x = field_x.to(pos.dtype)
        ctx.field_map_y = field_y.to(pos.dtype)
        ctx.pos                = pos
        ctx.node_size_x_clamped = node_size_x_clamped
        ctx.node_size_y_clamped = node_size_y_clamped
        ctx.offset_x           = offset_x
        ctx.offset_y           = offset_y
        ctx.ratio              = ratio
        ctx.bin_center_x       = bin_center_x
        ctx.bin_center_y       = bin_center_y
        ctx.target_density     = target_density
        ctx.xl = xl;  ctx.yl = yl;  ctx.xh = xh;  ctx.yh = yh
        ctx.bin_size_x         = bin_size_x
        ctx.bin_size_y         = bin_size_y
        ctx.num_movable_nodes  = num_movable_nodes
        ctx.num_filler_nodes   = num_filler_nodes
        ctx.padding            = padding
        ctx.num_bins_x         = num_bins_x
        ctx.num_bins_y         = num_bins_y
        ctx.num_movable_impacted_bins_x = num_movable_impacted_bins_x
        ctx.num_movable_impacted_bins_y = num_movable_impacted_bins_y
        ctx.num_filler_impacted_bins_x  = num_filler_impacted_bins_x
        ctx.num_filler_impacted_bins_y  = num_filler_impacted_bins_y
        ctx.deterministic_flag = deterministic_flag
        ctx.sorted_node_map    = sorted_node_map

        timing["ep_total_ms"] = (_time.perf_counter() - t_total) * 1000
        _client._timings[-1] = timing  # replace with complete entry

        return torch.zeros(1, dtype=pos.dtype, device=pos.device)

    try:
        import dreamplace.ops.electric_potential.electric_potential as ep_mod
        ep_mod.ElectricPotentialFunction.forward = _scatter_ttnn_forward
        print("[scatter_ttnn_client] Patched ElectricPotentialFunction.forward → "
              "TT-Metal V4 scatter + TTNN C++ DCT", flush=True)
    except ImportError as e:
        raise ImportError(
            f"[scatter_ttnn_client] Could not import DREAMPlace electric_potential: {e}"
        ) from e


def get_timing_summary() -> dict:
    if _client is None:
        return {}
    return _client.timing_summary()


def teardown() -> None:
    if _client is not None:
        _client.stop()
