# SPDX-License-Identifier: Apache-2.0
"""
V19 Rung 3 — in-process client.

Drop-in replacement for ``ScatterTTNNClient`` (the IPC-based client in
``scatter_ttnn_client.py``) that talks to the chip via a pybind11
extension instead of POSIX shared memory + ``docker exec``.

Public API matches ``ScatterTTNNClient`` exactly so the same
``patch_dreamplace`` hook works against either:

    __init__(container, ipc_dir, num_bins_x, num_bins_y, nc_max,
             xl, yl, xh, yh, direct=False)
    start(M, N, nc_actual, xl, yl, xh, yh)
    call(px, py, sx, sy) -> (field_x_torch, field_y_torch, timing_dict)
    stop()
    timing_summary(n_warmup=4) -> dict
    _timings: list
    _shm_id: writable numpy array of shape (M, N), float32

The ``container``, ``ipc_dir``, and ``direct`` arguments are accepted for
signature compatibility but ignored — there is no IPC layer in this client.

When CPU_DCT=1 (the default for Rung 3 today): the engine returns density
in the ``field_x`` slot and zeros in ``field_y``, mirroring the IPC
server's behavior at ``density_scatter_ttnn_server_host.cpp:3303-3309``.
The Python ``patch_dreamplace`` hook downstream runs the CPU DCT on the
returned density.
"""

import os
import time
from typing import Optional

import numpy as np
import torch


class ScatterTTNNClientInProcess:
    def __init__(self,
                 container: str = "",         # ignored (signature compat)
                 ipc_dir: str = "",           # ignored
                 num_bins_x: int = 0,
                 num_bins_y: int = 0,
                 nc_max: int = 0,
                 xl: float = 0.0, yl: float = 0.0,
                 xh: float = 1e6, yh: float = 1e6,
                 direct: bool = False):       # ignored
        self.container = container
        self.ipc_dir   = ipc_dir
        self.direct    = direct
        self.M = num_bins_x
        self.N = num_bins_y
        self.nc_max = nc_max
        self.xl, self.yl, self.xh, self.yh = xl, yl, xh, yh

        self._engine = None
        self._ready  = False
        self._timings: list = []

        # Public attribute matching IPC client: initial-density slot the
        # patch hook writes to before the first call(). Populated in start()
        # once we know M/N. Wrapped in a small "trigger on write" via the
        # _IDArrayProxy below so set_initial_density() gets called.
        self._shm_id: Optional[np.ndarray] = None

        # Cell-sort state (identical to scatter_ttnn_client.py:436-467).
        self._sort_idx:    Optional[np.ndarray] = None
        self._invalid_mask: Optional[np.ndarray] = None
        self._sx_sorted:   Optional[np.ndarray] = None
        self._sy_sorted:   Optional[np.ndarray] = None

        # _id_uploaded: bool flag — patch hook sets True after writing into
        # _shm_id (see scatter_ttnn_client.py:810-814). We honor it: on the
        # next call() we forward _shm_id into the engine if the flag flips.
        self._id_uploaded = False
        self._id_uploaded_to_engine = False

    # ── Lifecycle ─────────────────────────────────────────────────────────

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
            print(f"[scatter_ttnn_inprocess] nc_max={self.nc_max} (= nc_actual)",
                  flush=True)

        gather_mode = os.environ.get("GATHER_MODE", "v19")

        # Import lazily so the module can be tested without a chip if
        # v19_engine.so isn't yet built. Helpful error otherwise.
        try:
            import v19_engine  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Could not import `v19_engine` pybind11 extension. Build it with:\n"
                "  cd host/build && cmake --build . --target v19_engine\n"
                "Then make sure PYTHONPATH includes host/build (or copy the .so "
                "next to this client).\n"
                f"Original error: {e}"
            )

        print(f"[scatter_ttnn_inprocess] Constructing V19Engine "
              f"(M={self.M}, N={self.N}, NC_max={self.nc_max}, "
              f"gather_mode={gather_mode})...", flush=True)
        self._engine = v19_engine.V19Engine(
            self.M, self.N, self.nc_max,
            float(self.xl), float(self.yl), float(self.xh), float(self.yh),
            gather_mode,
        )

        # Initial-density slot — patch hook writes into this before first
        # call(). Allocated as a plain numpy array; we forward it into the
        # engine via set_initial_density() on the next call() once
        # _id_uploaded flips True.
        self._shm_id = np.zeros((self.M, self.N), dtype=np.float32)
        self._ready  = True
        print("[scatter_ttnn_inprocess] Engine ready.", flush=True)

    def stop(self) -> None:
        # PIMPL dtor releases the mesh device — just drop the reference.
        self._engine = None
        self._ready  = False

    # ── C++-fast-path: configure cells once, scatter from pos per iter ───
    # These replace the slower call(px, py, sx, sy) Python pipeline. After
    # configure_cells(), the patch hook's only per-iter work is producing a
    # `pos` numpy view; everything else (cell extract, big-cell sub-tile,
    # cell sort, h2d/chip/d2h) is one C++ call.

    def configure_cells(self, orig_sx_all, orig_sy_all,
                        n_movable, n_filler, num_nodes,
                        bin_size_x, bin_size_y):
        if self._engine is None:
            raise RuntimeError("configure_cells: call start() first")
        self._engine.configure_cells(
            orig_sx_all, orig_sy_all,
            int(n_movable), int(n_filler), int(num_nodes),
            float(bin_size_x), float(bin_size_y),
        )
        # set_initial_density may be called before/after configure_cells.

    def call_from_pos(self, pos_np: np.ndarray) -> tuple:
        """Per-iter fast path. `pos_np` is the full 2*num_nodes float32 view
        of DREAMPlace's pos tensor; the C++ engine extracts movable+filler,
        applies sub-cell layout, sorts, and dispatches the chip work."""
        if not self._ready or self._engine is None:
            raise RuntimeError("call_from_pos: before start()")
        # Flush initial-density upload on the first call (mirrors the IPC path).
        if self._id_uploaded and not self._id_uploaded_to_engine:
            self._engine.set_initial_density(self._shm_id)
            self._id_uploaded_to_engine = True

        t0 = time.perf_counter()
        density, fx, fy, timing = self._engine.scatter_from_pos(pos_np)
        total_client_ms = (time.perf_counter() - t0) * 1000

        out_fx = torch.from_numpy(fx)
        out_fy = torch.from_numpy(fy)
        timing["total_client_ms"] = total_client_ms
        self._timings.append(timing)
        return out_fx, out_fy, timing

    # ── Per-iteration compute call ────────────────────────────────────────

    def call(self,
             px: np.ndarray, py: np.ndarray,
             sx: np.ndarray, sy: np.ndarray) -> tuple:
        if not self._ready or self._engine is None:
            raise RuntimeError("ScatterTTNNClientInProcess.call() before start()")

        nc = len(px)

        # ── One-time cell sort (verbatim from scatter_ttnn_client.py:436-467) ──
        # Largest cells → lowest tile indices for V11 load balancing. The fix
        # for the dropped-cells bug (cells whose rank had dest>=nc) is included.
        if self._sort_idx is None or len(self._sort_idx) != nc:
            area_rank = np.argsort(sx * sy)[::-1].copy()
            n_tiles   = (nc + 1023) // 1024
            ranks     = np.arange(nc)
            dest      = (ranks % n_tiles) * 1024 + (ranks // n_tiles)
            sort_idx  = np.full(nc, area_rank[0], dtype=np.int64)
            valid     = dest < nc
            sort_idx[dest[valid]] = area_rank[valid]
            invalid_ranks      = np.flatnonzero(~valid)
            unfilled_positions = np.setdiff1d(np.arange(nc), dest[valid],
                                              assume_unique=False)
            assert len(invalid_ranks) == len(unfilled_positions), \
                f"sort fix: {len(invalid_ranks)} dropped ranks vs " \
                f"{len(unfilled_positions)} unfilled positions"
            sort_idx[unfilled_positions] = area_rank[invalid_ranks]
            print(f"[scatter_ttnn_inprocess] sort fix: recovered "
                  f"{len(invalid_ranks)} cells "
                  f"(was {len(invalid_ranks)*100/max(nc,1):.3f}% mass loss)",
                  flush=True)
            self._sort_idx     = sort_idx
            self._invalid_mask = np.zeros(nc, dtype=bool)
            self._sx_sorted    = sx[sort_idx].astype(np.float32)
            self._sy_sorted    = sy[sort_idx].astype(np.float32)

        idx  = self._sort_idx
        # Fancy-indexing path. np.take with out= into a reused buffer was
        # tried and ended up SLOWER on bigblue3 (~50 ms regression) — the
        # cache-line eviction cost of reusing the same memory dominates the
        # malloc savings. Fresh-alloc fancy index wins on large nc.
        px_s = np.ascontiguousarray(px[idx], dtype=np.float32)
        py_s = np.ascontiguousarray(py[idx], dtype=np.float32)

        # Forward initial_density into the engine on first call after upload.
        # The patch hook writes self._shm_id then flips _id_uploaded=True
        # (see scatter_ttnn_client.py:810-814). Honor the same flag.
        if self._id_uploaded and not self._id_uploaded_to_engine:
            self._engine.set_initial_density(self._shm_id)
            self._id_uploaded_to_engine = True

        # ── The call. No IPC, no polling. ──
        t0 = time.perf_counter()
        density, fx, fy, timing = self._engine.scatter(
            px_s, py_s, self._sx_sorted, self._sy_sorted, int(nc)
        )
        total_client_ms = (time.perf_counter() - t0) * 1000

        # Mirror the IPC server's slot overloading at server_host.cpp:3303-3309:
        # when CPU_DCT=1, the "field_x" slot actually contains density and
        # "field_y" is zero. The patch hook downstream interprets it as such.
        #
        # Buffers are REUSED across calls (zero-copy views of engine-owned
        # numpy arrays). For CPU_DCT=1, fx==density (same Python object) and
        # fy is the pre-zeroed constant buffer. DREAMPlace consumes both
        # before the next forward, so reuse is safe (see v19_engine_pybind.cpp
        # comment).
        out_fx = torch.from_numpy(fx)
        out_fy = torch.from_numpy(fy)

        # Augment the timing dict with the actually-measured client wall.
        # IPC-bucket keys stay 0.0 (the engine's pybind layer sets them).
        timing["total_client_ms"] = total_client_ms
        self._timings.append(timing)
        return out_fx, out_fy, timing

    # ── Stats (identical to IPC client) ───────────────────────────────────

    def timing_summary(self, n_warmup: int = 4) -> dict:
        timings = self._timings[n_warmup:] if len(self._timings) > n_warmup else self._timings
        if not timings:
            timings = self._timings
        if not timings:
            return {}
        keys = list(timings[0].keys())
        result: dict = {}
        for k in keys:
            vals = [t[k] for t in timings if k in t]
            if not vals:
                continue
            if isinstance(vals[0], (int, float)):
                result[f"{k}_mean"]   = float(np.mean(vals))
                result[f"{k}_median"] = float(np.median(vals))
            else:
                result[k] = max(set(vals), key=vals.count)
        result["n_iters"] = len(self._timings)
        result["n_timed"] = len(timings)
        return result


# ── patch_dreamplace shim ────────────────────────────────────────────────
#
# Mirrors the public function in scatter_ttnn_client.py:553 but injects an
# in-process client instead of the IPC one. Defers to scatter_ttnn_client
# for the actual ElectricPotentialFunction.forward replacement (so the two
# code paths stay in sync as that hook evolves).

_client: Optional[ScatterTTNNClientInProcess] = None


def patch_dreamplace(container: str = "", ipc_dir: str = "",
                     num_cells: int = 0, direct: bool = False) -> None:
    """Patch DREAMPlace to call the in-process engine for the density forward.

    Signature matches scatter_ttnn_client.patch_dreamplace so run_dreamplace.py
    can drop us in. Reuses that module's _scatter_ttnn_forward by swapping
    its module-level `_client` reference to our in-process client.
    """
    import scatter_ttnn_client as _ipc_mod  # the existing client module

    global _client
    _client = ScatterTTNNClientInProcess(
        container=container, ipc_dir=ipc_dir, nc_max=num_cells, direct=direct,
    )
    # Hijack the IPC module's _client slot so its _scatter_ttnn_forward hook
    # (which references _ipc_mod._client) uses ours. This keeps the
    # patch-hook body in one place.
    _ipc_mod._client = _client

    # Defer to the IPC module's patch_dreamplace for the actual forward
    # monkeypatch — but skip its client construction by stashing ours first.
    # We can't call _ipc_mod.patch_dreamplace() because that would overwrite
    # _ipc_mod._client. So inline the rest: monkeypatch the forward hook.
    try:
        import dreamplace.ops.electric_potential.electric_potential as ep_mod
        # Pull the staticmethod the IPC module installs. We rely on
        # _ipc_mod.patch_dreamplace having a reference to the inner function;
        # easier: call _ipc_mod.patch_dreamplace once to install the hook, then
        # re-overwrite _ipc_mod._client back to ours.
    except ImportError as e:
        raise ImportError(f"DREAMPlace not importable: {e}")

    # Install the IPC hook (it monkeypatches ElectricPotentialFunction.forward).
    # The hook calls _ipc_mod._client.call(...), which is now our in-process
    # client. The IPC client constructed inside patch_dreamplace is unused —
    # we overwrite _ipc_mod._client right after.
    _ipc_mod.patch_dreamplace(container=container, ipc_dir=ipc_dir or "/tmp/_unused_ipc",
                              num_cells=num_cells, direct=direct)
    _ipc_mod._client = _client
    print("[scatter_ttnn_inprocess] Hijacked _ipc_mod._client → in-process engine",
          flush=True)
