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
        total_size   = _SHM_HEADER_SIZE + 4 * pos_bytes + 2 * field_bytes

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
        deadline = time.monotonic() + 300
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
            self._sort_idx  = sort_idx
            # Positions in sort_idx that were never written by the dest mapping
            # default to area_rank[0] — meaning the largest cell would be
            # replicated at those slots. Mark them invalid (size=0 → zero
            # contribution from the triangle scatter) to avoid amplifying that
            # one cell. Without this, density bins covered by the largest cell
            # see a ~(N_unwritten + 1)× spike.
            self._invalid_mask = ~np.isin(np.arange(nc), dest[valid])
            self._sx_sorted = sx[sort_idx].astype(np.float32)
            self._sy_sorted = sy[sort_idx].astype(np.float32)
            self._sx_sorted[self._invalid_mask] = 0.0
            self._sy_sorted[self._invalid_mask] = 0.0

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

        t_total = _time.perf_counter()

        num_nodes = pos.shape[0] // 2
        nc = int(num_movable_nodes) + int(num_filler_nodes)

        # Raw position + size for movable+filler cells.
        px_full = pos[:nc].detach().cpu().numpy()
        py_full = pos[num_nodes:num_nodes+nc].detach().cpu().numpy()
        sx_full = node_size_x_clamped[:nc].detach().cpu().numpy()
        sy_full = node_size_y_clamped[:nc].detach().cpu().numpy()

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

        field_x, field_y, timing = _client.call(px, py, sx, sy)

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
