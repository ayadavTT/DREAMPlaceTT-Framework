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
                        bin_size_x, bin_size_y,
                        offx_all=None, offy_all=None, ratio_all=None):
        if self._engine is None:
            raise RuntimeError("configure_cells: call start() first")
        import numpy as _np
        if offx_all is None: offx_all = _np.empty(0, _np.float32)
        if offy_all is None: offy_all = _np.empty(0, _np.float32)
        if ratio_all is None: ratio_all = _np.empty(0, _np.float32)
        self._engine.configure_cells(
            orig_sx_all, orig_sy_all,
            int(n_movable), int(n_filler), int(num_nodes),
            float(bin_size_x), float(bin_size_y),
            offx_all, offy_all, ratio_all,
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
        if os.environ.get("V35_DENSITY_CHK") == "1":
            self._last_density = np.array(density, copy=True)

        out_fx = torch.from_numpy(fx)
        out_fy = torch.from_numpy(fy)
        timing["total_client_ms"] = total_client_ms
        self._timings.append(timing)
        if os.environ.get("FWD_TIMING") == "1":
            _fn = getattr(self, "_fwd_dbg_n", 0); self._fwd_dbg_n = _fn + 1
            if _fn % 50 == 0:
                _dct = (timing.get('ttnn_upload_ms',0)+timing.get('ttnn_compute_ms',0)
                        +timing.get('ttnn_download_ms',0))
                print(f"[fwd_timing] iter~{_fn}: h2d={timing.get('h2d_ms',0):.3f} "
                      f"scatter={timing.get('scatter_ms',0):.3f} "
                      f"writeout={timing.get('gather_ms',0):.3f} DCT={_dct:.3f} "
                      f"server={timing.get('total_server_ms',0):.3f}", flush=True)
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
        if os.environ.get("FWD_TIMING") == "1":
            _fn = getattr(self, "_fwd_dbg_n", 0); self._fwd_dbg_n = _fn + 1
            if _fn % 50 == 0:
                _dct = (timing.get('ttnn_upload_ms',0)+timing.get('ttnn_compute_ms',0)
                        +timing.get('ttnn_download_ms',0))
                print(f"[fwd_timing] iter~{_fn}: h2d={timing.get('h2d_ms',0):.3f} "
                      f"scatter={timing.get('scatter_ms',0):.3f} "
                      f"writeout={timing.get('gather_ms',0):.3f} DCT={_dct:.3f} "
                      f"server={timing.get('total_server_ms',0):.3f}", flush=True)
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
    # ── SOLIDIFIED PRODUCTION PIPELINE: V19 scatter → TT DCT → V35 backward ──
    # Lock the production configuration at the entry point so the density
    # pipeline can ONLY run this path. Alternative forward modes (v6..v18),
    # alternative backwards (v21/v29/v30/v31/fccs/cpu-grad), the CPU-side DCT,
    # and every diagnostic check are disconnected by forcing/stripping their env
    # gates here (their code remains in-tree but is now unreachable). Per-stage
    # timing ([fwd_full]/[bwd_full]) is forced ON so it is ALWAYS recorded.
    # Kept-optional knobs (not forced): TTNN_HIFI (DCT precision), TT_PROFILE_DUMP.
    os.environ["GATHER_MODE"] = "v19"                       # forward scatter = V19 only
    for _f in ("V21_EF", "V35_EF", "V31_STASH", "V31_GEOM", "V31_EF_GEOM",
               "V19_SKIP_CELL_SORT", "FWD_TIMING", "V35_TIMING"):
        os.environ[_f] = "1"                               # production path + timing always-on
    for _f in ("CPU_DCT", "V29_EF", "V30_EF", "V31_EF", "FCCS_EF",
               "V21_EF_USE_CPU_GRAD", "V21_EF_ZERO_COPY", "FCCS_HOST_FIELD",
               "FCCS_TIMING", "V29_TIMING", "V35_DENSITY_CHK", "V35_GATHER_CHK",
               "V35_GEOM_SCAN", "V35_STASH_CHK", "V35_VS_CPU", "V35_VS_V21",
               "V35_ANOM", "V35_DIAG", "V35_DROP", "V21_EF_DIAG", "DENSITY_AUDIT",
               "FCCS_DBG_PROD", "FCCS_GRAD_DIAG", "FOLD_TIME", "TTNN_DL_BENCH",
               "DENSITY_DUMP_ITER", "DENSITY_DUMP_PATH", "EXPORT_DENSITY_PATH",
               "EXPORT_POS_PATH", "EXPORT_ROUTE_BUF_PATH", "V14_BUCKET_CAP_OVERRIDE",
               "V15_CAP_OVERRIDE", "V15_FORCE_NO_SPILL"):
        os.environ.pop(_f, None)                           # disconnect alt modes + diagnostics
    # DEBUG override (off by default): DBG_<NAME>=<val> forces env NAME=val through the
    # lock — for isolation tests (e.g. DBG_V35_EF=0 DBG_V21_EF_USE_CPU_GRAD=1 = TT fwd + CPU bwd).
    for _f in ("V35_EF", "V21_EF_USE_CPU_GRAD", "V21_EF_ZERO_COPY",
               "V35_VS_CPU", "V35_DIAG", "V35_DROP",
               "V35_GEOM_SCAN", "V35_STASH_CHK", "V35_VS_V21", "V35_ANOM"):
        _dbg = os.environ.get("DBG_" + _f)
        if _dbg is not None:
            os.environ[_f] = _dbg
    print("[scatter_ttnn_inprocess] PRODUCTION PIPELINE LOCKED: V19 scatter + TT DCT "
          "+ V35 backward; timing always-on; alt modes/diagnostics disconnected", flush=True)

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

    # ── V21 EF backward patch (opt-in via V21_EF=1) ────────────────────────
    if os.environ.get("V21_EF") == "1":
        _install_v21_ef_backward()
        # NOTE: the V19 d2h-skip is enabled inside the backward's configure-once
        # block (the engine is constructed lazily in start(), so _client._engine
        # is None here at patch time).

    # ── V22 optimizer-on-chip patch (opt-in via V22_OPT=shadow|drive) ───────
    # Routes the Nesterov position update (u_kp1/v_kp1/clamp) through the V22
    # SFPU kernels so positions are updated ON the TT device. shadow = validate
    # vs the host step_bb without driving; drive = let V22 produce the update.
    if (os.environ.get("V22_OPT", "").lower() in ("shadow", "drive")
            or os.environ.get("V22_NOHOST", "").lower() in ("shadow_grad", "drive")):
        _install_v22_optimizer()


def _install_v21_ef_backward() -> None:
    """Replace ElectricPotentialFunction.backward with one that calls the
    V21 EF on-chip kernel instead of electric_potential_cpp.electric_force.

    The first invocation lazily calls V19Engine.configure_electric_force() with
    the per-cell constants captured into ctx by the forward hook (cf.
    scatter_ttnn_client.py:659-689). After that, each backward issues
    one chip launch and downloads grad_x/grad_y.
    """
    import dreamplace.ops.electric_potential.electric_potential as ep_mod
    import scatter_ttnn_client as _ipc_mod

    _state = {"configured": False}

    def _v21_ef_backward(ctx, grad_pos):
        import numpy as _np
        import torch as _torch
        client = _ipc_mod._client
        engine = client._engine

        # ── One-time configure: pass full-layout arrays + a sel index map.
        # Engine handles the gather/scatter in C++ on each compute_from_full().
        if not _state["configured"]:
            # Engine expects float32 numpy. ctx tensors are torch CPU tensors
            # — .numpy() is zero-copy iff dtype/layout match. Force float32.
            ox_full     = ctx.offset_x.detach().contiguous().to(_torch.float32).numpy()
            oy_full     = ctx.offset_y.detach().contiguous().to(_torch.float32).numpy()
            nsx_full    = ctx.node_size_x_clamped.detach().contiguous().to(_torch.float32).numpy()
            nsy_full    = ctx.node_size_y_clamped.detach().contiguous().to(_torch.float32).numpy()
            ratio_full  = ctx.ratio.detach().contiguous().to(_torch.float32).numpy()
            n_mov  = int(ctx.num_movable_nodes)
            n_fil  = int(ctx.num_filler_nodes)
            nn_full = int(ctx.pos.shape[0]) // 2
            num_phys = int(ratio_full.shape[0]) - n_fil
            if n_fil > 0:
                sel = _np.concatenate([
                    _np.arange(n_mov, dtype=_np.int32),
                    _np.arange(num_phys, num_phys + n_fil, dtype=_np.int32),
                ])
            else:
                sel = _np.arange(n_mov, dtype=_np.int32)
            engine.configure_electric_force_full(
                ox_full, oy_full, nsx_full, nsy_full, ratio_full,
                sel, nn_full,
                float(ctx.xl), float(ctx.yl),
                float(ctx.bin_size_x), float(ctx.bin_size_y),
            )
            # ── V29 backward (L1-resident multi-bin gather) — opt-in via V29_EF=1 ──
            _state["v29"] = os.environ.get("V29_EF") == "1"
            if _state["v29"]:
                engine.configure_electric_force_v29(
                    ox_full, oy_full, nsx_full, nsy_full, ratio_full,
                    sel, nn_full,
                    float(ctx.xl), float(ctx.yl),
                    float(ctx.bin_size_x), float(ctx.bin_size_y),
                )
                print("[v21_ef_backward] V29_EF=1: backward → V29 L1-resident gather", flush=True)
            # ── V30 field-stationary backward — opt-in via V30_EF=1 ──
            _state["v30"] = os.environ.get("V30_EF") == "1"
            if _state["v30"]:
                engine.configure_electric_force_v30(
                    ox_full, oy_full, nsx_full, nsy_full, ratio_full,
                    sel, nn_full,
                    float(ctx.xl), float(ctx.yl),
                    float(ctx.bin_size_x), float(ctx.bin_size_y),
                )
                print("[v21_ef_backward] V30_EF=1: backward → V30 field-stationary gather", flush=True)
            # ── FCCS field-cast backward (reads forward stash) — opt-in via FCCS_EF=1 ──
            # Requires the forward to run with V31_STASH=1 V31_GEOM=1 (stashes per-cell
            # px/py geometry). Field-cast multicast + balanced cell slices + in-L1 sort.
            _state["fccs"] = os.environ.get("FCCS_EF") == "1"
            if _state["fccs"]:
                # Big cells are sub-tiled into ≤8-bin sub-cells in the forward
                # (configure_cells), so the stash has new_nc > nc records. The FCCS
                # backward MUST configure with the SAME new_nc sub-cell list (else
                # na=nc misaligns with the new_nc stash → garbage grad at 1024/2048).
                # subcell_parent maps each sub-cell→parent active-cell; sel maps that→node.
                _subpar = engine.subcell_parent()
                _sel_sub = sel[_subpar] if _subpar.size > 0 else sel
                _state["fccs_sel_sub"] = _sel_sub
                engine.configure_electric_force_fccs(
                    ox_full, oy_full, nsx_full, nsy_full, ratio_full,
                    _sel_sub, nn_full,
                    float(ctx.xl), float(ctx.yl),
                    float(ctx.bin_size_x), float(ctx.bin_size_y),
                )
                print(f"[v21_ef_backward] FCCS sub-cell map: nc(sel)={len(sel)} "
                      f"new_nc(sub)={len(_sel_sub)} (Δ=+{len(_sel_sub)-len(sel)})", flush=True)
                print("[v21_ef_backward] FCCS_EF=1: backward → FCCS field-cast gather", flush=True)
            # ── V35 fully-on-chip Forward-Grouped Halo-Tile backward — opt-in via V35_EF=1 ──
            # Same forward stash + sub-cell map as FCCS; on-chip count→plan→place→gather.
            _state["v35"] = os.environ.get("V35_EF") == "1"
            if _state["v35"]:
                _subpar = engine.subcell_parent()
                _sel_sub = sel[_subpar] if _subpar.size > 0 else sel
                _state["v35_sel_sub"] = _sel_sub
                engine.configure_electric_force_v35(
                    ox_full, oy_full, nsx_full, nsy_full, ratio_full,
                    _sel_sub, nn_full,
                    float(ctx.xl), float(ctx.yl),
                    float(ctx.bin_size_x), float(ctx.bin_size_y),
                )
                print(f"[v21_ef_backward] V35 sub-cell map: nc(sel)={len(sel)} new_nc(sub)={len(_sel_sub)}", flush=True)
                print("[v21_ef_backward] V35_EF=1: backward → V35 on-chip halo-tile gather", flush=True)
                # V35 reads the field on-chip (via latest_field_addrs); the forward's
                # host field download → ctx.field_map is vestigial. Skip it so the
                # forward keeps the field on device (saves the M*N*4*2 d2h per iter).
                try:
                    if os.environ.get("DBG_NO_SKIP_FIELD") == "1":
                        # keep the host field map populated so V35_VS_CPU/V21 grad
                        # diagnostics have a valid reference (debug only).
                        print("[v21_ef_backward] V35: field d2h KEPT (DBG_NO_SKIP_FIELD)", flush=True)
                    else:
                        engine.set_skip_field_d2h(True)
                        print("[v21_ef_backward] V35: forward field d2h SKIPPED (field read on-chip)", flush=True)
                except Exception as e:
                    print(f"[v21_ef_backward] WARNING set_skip_field_d2h (V35): {e}", flush=True)
            # ── V31 no-host backward (reads forward stash) — opt-in via V31_EF=1 ──
            # Requires the forward to run with V31_STASH=1 (stashes px·py) and
            # V19_SKIP_CELL_SORT=1 (so scatter index == active cell index).
            _state["v31"] = os.environ.get("V31_EF") == "1"
            _state["v31_ratio_full"] = ratio_full
            _state["v31_sel"] = sel
            _state["v31_nn"] = nn_full
            if _state["v31"]:
                print("[v21_ef_backward] V31_EF=1: backward → V31 no-host stash-reuse gather", flush=True)
            # ── DIAGNOSTIC: max bin span vs V21's MAX_KH=8 limit ──
            try:
                bsx = float(ctx.bin_size_x); bsy = float(ctx.bin_size_y)
                nsx_sel = nsx_full[sel]; nsy_sel = nsy_full[sel]
                # bins a cell spans ≈ floor(ns/bin)+1 (matches kernel bin_xh-bin_xl)
                kspan = (_np.floor(nsx_sel / bsx).astype(_np.int32) + 1)
                hspan = (_np.floor(nsy_sel / bsy).astype(_np.int32) + 1)
                n_over_k = int((kspan > 8).sum()); n_over_h = int((hspan > 8).sum())
                print(f"[v21_ef_backward] bin-span DIAG: max_k={int(kspan.max())} "
                      f"max_h={int(hspan.max())} | cells>8bins: k={n_over_k} h={n_over_h} "
                      f"({100.0*max(n_over_k,n_over_h)/len(sel):.2f}%)  [V21 MAX_KH=8]",
                      flush=True)
            except Exception as _e:
                print(f"[v21_ef_backward] bin-span DIAG failed: {_e}", flush=True)

            _state["configured"] = True
            _state["num_active"] = int(sel.shape[0])
            _state["nn_full"] = nn_full
            _state["zero_copy_fields"] = os.environ.get("V21_EF_ZERO_COPY") == "1"
            # Now that the engine exists, enable the V19 d2h-skip so the forward
            # stops downloading field maps (V21 reads them from chip directly).
            if _state["zero_copy_fields"]:
                try:
                    engine.set_skip_field_d2h(True)
                    print("[v21_ef_backward] V19 field d2h SKIPPED (zero-copy from chip)",
                          flush=True)
                except Exception as e:
                    print(f"[v21_ef_backward] WARNING set_skip_field_d2h: {e}", flush=True)
            # Pre-allocate output buffer once, reuse each iter.
            _state["out_torch"] = _torch.zeros(2 * nn_full, dtype=_torch.float32)
            print(f"[v21_ef_backward] configured: num_active={int(sel.shape[0])} "
                  f"(movable={n_mov} filler={n_fil}) out_buf=2×{nn_full} fp32  "
                  f"zero_copy_fields={_state['zero_copy_fields']}",
                  flush=True)

        # ── Per-iter: zero-copy views into torch tensors, call C++ engine.
        # ctx.pos / field_map_* are CPU torch tensors. .numpy() is zero-copy
        # iff contiguous + matching dtype. We force float32 view via .to() ONLY
        # if the dtype isn't already float32 (cheap when it is).
        _bt0 = time.perf_counter()   # backward per-iter wall-clock start
        pos_t = ctx.pos
        if pos_t.dtype != _torch.float32:
            pos_t = pos_t.to(_torch.float32)
        pos_np = pos_t.detach().contiguous().numpy()
        _bt_pos = time.perf_counter()

        fx_t = ctx.field_map_x.detach().contiguous().view(-1)
        if fx_t.dtype != _torch.float32: fx_t = fx_t.to(_torch.float32)
        fx_np = fx_t.numpy()
        fy_t = ctx.field_map_y.detach().contiguous().view(-1)
        if fy_t.dtype != _torch.float32: fy_t = fy_t.to(_torch.float32)
        fy_np = fy_t.numpy()
        _bt_field = time.perf_counter()

        out_torch = _state["out_torch"]
        out_torch.zero_()
        out_np = out_torch.numpy()  # zero-copy view
        _bt_prep = time.perf_counter()   # pos copy + field copy + out-zero done

        # ── DIAGNOSTIC: dump raw V21 vs CPU force at iter V21_EF_DIAG_ITER ──
        # Uses grad_pos=ones so we compare raw FORCE (no upstream-grad scaling).
        # Dumps to .npy for offline analysis (sign, bias, localization).
        _state["diag_count"] = _state.get("diag_count", 0) + 1
        _diag_iter = int(os.environ.get("V21_EF_DIAG_ITER", "0"))
        if os.environ.get("V21_EF_DIAG") == "1" and _state["diag_count"] == _diag_iter + 1:
            try:
                import numpy as _dnp
                import dreamplace.ops.electric_potential.electric_potential as _epm
                ones = _torch.ones_like(grad_pos)
                # CPU raw electric_force (NO leading negation, grad_pos=1) → +force_cpu
                cpu_raw = _epm.electric_potential_cpp.electric_force(
                    ones, ctx.num_bins_x, ctx.num_bins_y,
                    ctx.num_movable_impacted_bins_x, ctx.num_movable_impacted_bins_y,
                    ctx.num_filler_impacted_bins_x, ctx.num_filler_impacted_bins_y,
                    ctx.field_map_x.view([-1]), ctx.field_map_y.view([-1]), ctx.pos,
                    ctx.node_size_x_clamped, ctx.node_size_y_clamped,
                    ctx.offset_x, ctx.offset_y, ctx.ratio,
                    ctx.bin_center_x, ctx.bin_center_y, ctx.xl, ctx.yl, ctx.xh, ctx.yh,
                    ctx.bin_size_x, ctx.bin_size_y, ctx.num_movable_nodes,
                    ctx.num_filler_nodes).detach().cpu().numpy().astype(_dnp.float64)
                # V21/V31 raw kernel output into out_np (raw +force, no negation).
                out_torch.zero_()
                if _state.get("v31"):
                    engine.compute_electric_force_v31(_state["v31_sel"], _state["v31_ratio_full"],
                        fx_np, fy_np, out_np, _state["v31_nn"])
                else:
                    _ = engine.compute_electric_force_full(pos_np, fx_np, fy_np, out_np)
                v21_raw = out_np.astype(_dnp.float64).copy()
                it = _state['diag_count'] - 1
                pfx = os.environ.get("V21_EF_DIAG_PREFIX", "/tmp/v21diag")
                _dnp.save(f"{pfx}_cpu_it{it}.npy", cpu_raw)
                _dnp.save(f"{pfx}_v21_it{it}.npy", v21_raw)
                # Quick stats: sign correlation, bias, localization.
                nn = int(ctx.pos.shape[0]) // 2
                nmov = int(ctx.num_movable_nodes); nfil = int(ctx.num_filler_nodes)
                nphys = (v21_raw.shape[0]//2) - nfil if False else nn - nfil
                # Compare on movable x-slice [0,nmov) where both should be nonzero.
                cx = cpu_raw[:nmov]; vx = v21_raw[:nmov]
                # sign relationship
                dot = float(_dnp.dot(cx, vx)); cn=float(_dnp.dot(cx,cx)); vn=float(_dnp.dot(vx,vx))
                print(f"[V21_EF_DIAG] it={it} movable-x: cpu_norm={cn**.5:.4e} v21_norm={vn**.5:.4e} "
                      f"cos={dot/((cn*vn)**.5+1e-30):.4f} "
                      f"ratio_v21/cpu(median)={_dnp.median(_dnp.abs(vx)/(_dnp.abs(cx)+1e-30)):.4f}",
                      flush=True)
                # localization: per-1000-cell-block rel error
                rel_block=[]
                for b in range(0, nmov, max(1,nmov//10)):
                    e=b; f=min(b+max(1,nmov//10),nmov)
                    num=_dnp.linalg.norm(vx[e:f]-cx[e:f]); den=_dnp.linalg.norm(cx[e:f])+1e-30
                    rel_block.append(num/den)
                print(f"[V21_EF_DIAG] it={it} per-block rel err: "+
                      " ".join(f"{r:.3f}" for r in rel_block), flush=True)
            except Exception as _e:
                import traceback; traceback.print_exc()
                print(f"[V21_EF_DIAG] failed: {_e}", flush=True)

        # Zero-copy field path (opt-in): skip the d2h+h2d of field maps.
        # V19's DCT keeps the latest field tensors on chip; we read them
        # directly via their DRAM addresses.
        if _state.get("fccs"):
            # FCCS field-cast backward: reads the forward V31_GEOM stash + field,
            # writes +force into out_np[sel]; negate to -force (CPU -electric_force).
            # Prefer the chip-resident field (host-free, no field h2d) when available.
            _fx_addr, _fy_addr = engine.latest_field_addrs()
            if _fx_addr != 0 and _fy_addr != 0 and os.environ.get("FCCS_HOST_FIELD") != "1":
                ef_timing = engine.compute_electric_force_fccs_chip(pos_np, _fx_addr, _fy_addr, out_np)
            else:
                ef_timing = engine.compute_electric_force_fccs(pos_np, fx_np, fy_np, out_np)
            if os.environ.get("FCCS_TIMING") == "1":
                _n = getattr(client, "_fccs_dbg_n", 0); client._fccs_dbg_n = _n + 1
                if _n % 20 == 0:
                    print(f"[fccs_timing] iter~{_n}: quant={ef_timing.get('quant_ms',0):.3f} "
                          f"run={ef_timing.get('run_ms',0):.3f} h2d={ef_timing.get('h2d_ms',0):.3f} "
                          f"d2h={ef_timing.get('d2h_ms',0):.3f} total={ef_timing.get('total_ms',0):.3f} ms", flush=True)
            # ── one-shot gradient diff vs CPU at several iters (find the systematic error) ──
            if os.environ.get("FCCS_GRAD_DIAG") == "1":
                _gn = getattr(client, "_fccs_gdiag", 0); client._fccs_gdiag = _gn + 1
                if _gn in (0, 1, 2, 5, 50, 150, 250, 350):
                    try:
                        import numpy as _np
                        import torch as _torch
                        import dreamplace.ops.electric_potential.electric_potential as _epd
                        # RAW force (grad_pos=1) — FCCS out_np is raw (client applies grad_pos later),
                        # so compare apples-to-apples, not against the density-weighted CPU force.
                        cpu_pf = _epd.electric_potential_cpp.electric_force(
                            _torch.ones_like(grad_pos), ctx.num_bins_x, ctx.num_bins_y,
                            ctx.num_movable_impacted_bins_x, ctx.num_movable_impacted_bins_y,
                            ctx.num_filler_impacted_bins_x, ctx.num_filler_impacted_bins_y,
                            ctx.field_map_x.view([-1]), ctx.field_map_y.view([-1]), ctx.pos,
                            ctx.node_size_x_clamped, ctx.node_size_y_clamped,
                            ctx.offset_x, ctx.offset_y, ctx.ratio,
                            ctx.bin_center_x, ctx.bin_center_y, ctx.xl, ctx.yl, ctx.xh, ctx.yh,
                            ctx.bin_size_x, ctx.bin_size_y, ctx.num_movable_nodes, ctx.num_filler_nodes)
                        cpu = cpu_pf.detach().cpu().numpy().reshape(-1).astype(_np.float64)   # +force
                        f = _np.asarray(out_np).reshape(-1).astype(_np.float64)               # FCCS +force
                        sel = _np.asarray(_state["v31_sel"]); nnf = int(_state["v31_nn"])
                        idx = _np.concatenate([sel, nnf + sel])
                        ff = f[idx]; cc = cpu[idx]
                        rel = _np.linalg.norm(ff-cc)/(_np.linalg.norm(cc)+1e-30)
                        ratio = _np.abs(ff).sum()/(_np.abs(cc).sum()+1e-30)
                        mfx = float(_np.abs(_np.asarray(fx_np)).max()); mfy = float(_np.abs(_np.asarray(fy_np)).max())
                        nmov = int(ctx.num_movable_nodes)
                        relm = _np.linalg.norm(f[sel[:nmov]]-cpu[sel[:nmov]])/(_np.linalg.norm(cpu[sel[:nmov]])+1e-30)
                        relf = (_np.linalg.norm(f[sel[nmov:]]-cpu[sel[nmov:]])/(_np.linalg.norm(cpu[sel[nmov:]])+1e-30)) if len(sel)>nmov else -1
                        ovf = (f[idx]>2.6e5).sum()
                        nsx = _np.asarray(ctx.node_size_x_clamped.detach().cpu().numpy()); nsy = _np.asarray(ctx.node_size_y_clamped.detach().cpu().numpy())
                        bsx = float(ctx.bin_size_x); bsy = float(ctx.bin_size_y)
                        ksp = (_np.floor(nsx[:nmov]/bsx)+2).astype(int); hsp = (_np.floor(nsy[:nmov]/bsy)+2).astype(int)
                        # worst movable cell
                        d = _np.abs(f[sel[:nmov]]-cpu[sel[:nmov]]); wi = int(d.argmax())
                        # RATIO check: does f·ctx.ratio match cpu? (DREAMPlace gx*=ratio)
                        rat = _np.asarray(ctx.ratio.detach().cpu().numpy()).reshape(-1).astype(_np.float64)
                        fr = f[idx]*_np.concatenate([rat[sel], rat[sel]])
                        rel_wr = _np.linalg.norm(fr-cc)/(_np.linalg.norm(cc)+1e-30)
                        frm = f[sel[:nmov]]*rat[sel[:nmov]]; frf = f[sel[nmov:]]*rat[sel[nmov:]]
                        relm_wr = _np.linalg.norm(frm-cpu[sel[:nmov]])/(_np.linalg.norm(cpu[sel[:nmov]])+1e-30)
                        relf_wr = _np.linalg.norm(frf-cpu[sel[nmov:]])/(_np.linalg.norm(cpu[sel[nmov:]])+1e-30)
                        print(f"   RATIO-CHK sel[{wi}]={int(sel[wi])} ratio={rat[sel[wi]]:.4f} f={f[sel[wi]]:.3g} f*r={f[sel[wi]]*rat[sel[wi]]:.3g} cpu={cpu[sel[wi]]:.3g} | "
                              f"rel(f*r)={rel_wr:.3e}(was {rel:.3e}) relm(f*r)={relm_wr:.3e} relf(f*r)={relf_wr:.3e} | "
                              f"ratio[mov] min/med/max={rat[sel[:nmov]].min():.3f}/{_np.median(rat[sel[:nmov]]):.3f}/{rat[sel[:nmov]].max():.3f} "
                              f"ratio[fil] med={_np.median(rat[sel[nmov:]]):.3f}", flush=True)
                        # GEOMETRY check: stashed bxl/px vs CPU triangle_density_function(pos+offset, node_size_clamped)
                        try:
                            geom2 = _np.asarray(engine.read_geom())
                            if geom2.size:
                                g2 = geom2.reshape(-1,32); c = int(sel[wi])
                                # POST-SPLIT: map node c -> its active (sub-)cell index/indices via fccs_sel_sub.
                                ssub = _np.asarray(_state.get("fccs_sel_sub", sel))
                                ais = _np.where(ssub == c)[0]
                                fmx = ctx.field_map_x.detach().cpu().numpy(); fmy = ctx.field_map_y.detach().cpu().numpy()
                                NBX=int(ctx.num_bins_x); NBY=int(ctx.num_bins_y)
                                rat_c = float(rat[c]); man_gx=0.0
                                lines=[]
                                for ca in ais.tolist():
                                    rec=g2[ca]; bxl=int(rec[1]); byl=int(rec[2]); kc=int(rec[3]); hc=int(rec[4])
                                    px=[float(rec[6+j])/8192.0 for j in range(kc)]; py=[float(rec[14+j])/8192.0 for j in range(hc)]
                                    sub_gx=0.0
                                    for k in range(kc):
                                        cx=bxl+k
                                        for h in range(hc):
                                            cy=byl+h
                                            fv = fmx[cx,cy] if (0<=cx<NBX and 0<=cy<NBY) else 0.0
                                            sub_gx += px[k]*py[h]*float(fv)
                                    man_gx += sub_gx
                                    lines.append(f"act{ca}: bxl={bxl} byl={byl} kc={kc} hc={hc}")
                                # manual gx (stash geom · field · ratio) vs FCCS f vs CPU
                                man_gx *= rat_c
                                print(f"   GEOM/FIELD-TRACE c={c} #sub={len(ais)} [{'; '.join(lines)}] | "
                                      f"manual(stash·field·ratio)={man_gx:.4g}  FCCS f={f[c]:.4g}  CPU={cpu[c]:.4g}  "
                                      f"(manual==CPU? geom+field OK→FCCS-gather bug; manual==FCCS? stash/field bug)", flush=True)
                        except Exception as _eg:
                            import traceback; print(f"   GEOM/FIELD-TRACE failed: {_eg}\n{traceback.format_exc()[:400]}", flush=True)
                        print(f"[fccs_grad_diag] iter~{_gn}: rel_l2={rel:.3e} |fccs|/|cpu|={ratio:.4f} "
                              f"rel_mov={relm:.3e} rel_fil={relf:.3e} max|fx|={mfx:.1f} | "
                              f"mov span max_k={int(ksp.max())} max_h={int(hsp.max())} kc>8={int((ksp>8).sum())} hc>8={int((hsp>8).sum())} | "
                              f"worst_mov[{wi}] kc~{int(ksp[wi])} hc~{int(hsp[wi])} fccs={f[wi]:.3g} cpu={cpu[wi]:.3g}", flush=True)
                        try:
                            geom = _np.asarray(engine.read_geom())
                            if geom.size >= 32*(wi+1):
                                g = geom.reshape(-1, 32)
                                nmis = int((g[:nmov,0] != _np.arange(nmov)).sum())
                                rec = g[wi]
                                pxv = rec[6:6+max(1,int(rec[3]))].astype(_np.float64)/8192.0
                                pyv = rec[14:14+max(1,int(rec[4]))].astype(_np.float64)/8192.0
                                print(f"   geom[{wi}]: orig={int(rec[0])} bxl={int(rec[1])} byl={int(rec[2])} kc={int(rec[3])} hc={int(rec[4])} | "
                                      f"movable orig!=idx: {nmis}/{nmov} | px={_np.round(pxv,4)} py={_np.round(pyv,4)}", flush=True)
                                # host-replicate the FCCS gx from stashed geom + field; compare device vs CPU
                                bxl=int(rec[1]); byl=int(rec[2]); kc=int(rec[3]); hc=int(rec[4])
                                fxm = _np.asarray(fx_np).reshape(int(ctx.num_bins_x), int(ctx.num_bins_y))
                                gxh = 0.0
                                for _k in range(kc):
                                    for _h in range(hc):
                                        gxh += (rec[6+_k]/8192.0)*(rec[14+_h]/8192.0)*float(fxm[bxl+_k, byl+_h])
                                gxh *= bsx*bsy
                                print(f"   gx_hostreplica={gxh:.4g}  device={f[wi]:.4g}  cpu={cpu[wi]:.4g}  "
                                      f"(field@bins={[round(float(fxm[bxl+_k,byl+_h]),2) for _k in range(kc) for _h in range(hc)]})", flush=True)
                                try:
                                    fxb = _np.asarray(engine.read_fccs_fxb()); graw = _np.asarray(engine.read_fccs_grad_raw())
                                    if os.environ.get("FCCS_DBG_PROD")=="1" and fxb.size and graw.size>=4:
                                        # producer bufp probe: wt=2 band = fxb[256*NBY ..]; offsets 255/2814/60000
                                        b0=256*int(ctx.num_bins_y)
                                        print(f"   PROD-PROBE plane={int(graw[3])} | off255: prod={int(graw[0])} fxb={int(fxb[b0+255])} | "
                                              f"off2814: prod={int(graw[1])} fxb={int(fxb[b0+2814])} | off60000: prod={int(graw[2])} fxb={int(fxb[b0+60000])}", flush=True)
                                    NBY=int(ctx.num_bins_y); FSc=256.0  # 2^FS, FS=8
                                    exp_fi=[int(round(float(fxm[bxl+_k,byl+_h])*FSc)) for _k in range(kc) for _h in range(hc)]
                                    got_fi=[int(fxb[(bxl+_k)*NBY+(byl+_h)]) for _k in range(kc) for _h in range(hc)] if fxb.size else []
                                    accx=0
                                    if fxb.size:
                                        for _k in range(kc):
                                            for _h in range(hc):
                                                accx += int(rec[6+_k])*int(rec[14+_h])*int(fxb[(bxl+_k)*NBY+(byl+_h)])
                                    wf = int(graw[wi*2]) if graw.size>wi*2 else None       # worker k0h0 field
                                    wbxl = int(graw[wi*2+1]) if graw.size>wi*2+1 else None  # worker's bxl
                                    fxb_exp = int(fxb[bxl*NBY+byl]) if fxb.size else None
                                    print(f"   DBG4 cell {wi}: worker_bxl={wbxl} vs stash_bxl={bxl} | "
                                          f"worker_k0h0_field={wf} vs fxb[{bxl},{byl}]={fxb_exp}", flush=True)
                                except Exception as _e3:
                                    print(f"   field/graw check failed: {_e3}", flush=True)
                        except Exception as _e2:
                            print(f"   geom read failed: {_e2}", flush=True)
                    except Exception as _e:
                        print(f"[fccs_grad_diag] failed: {_e}", flush=True)
            out_np *= -1.0
        elif _state.get("v35"):
            # V35 fully-on-chip Forward-Grouped Halo-Tile backward: reads the forward
            # V31_GEOM stash + chip-DCT field (host-free), on-chip count→plan→place→
            # gather, writes +force into out_np[sel]; negate to -force.
            _fx_addr, _fy_addr = engine.latest_field_addrs()
            ef_timing = engine.compute_electric_force_v35_chip(pos_np, _fx_addr, _fy_addr, out_np)
            out_np *= -1.0
            # Always accumulate backward stage timing + register an end-of-run summary.
            _eft = getattr(client, "_ef_timings", None)
            if _eft is None:
                _eft = []; client._ef_timings = _eft
                import atexit as _atexit
                def _stage_summary(_cl=client):
                    import numpy as _n2
                    def _med(rows, k):
                        vs = [r[k] for r in rows if k in r]
                        return float(_n2.median(vs)) if vs else 0.0
                    fwd = getattr(_cl, "_timings", [])[4:] or getattr(_cl, "_timings", [])
                    bwd = getattr(_cl, "_ef_timings", [])[4:] or getattr(_cl, "_ef_timings", [])
                    print("\n==== TT per-iteration stage timing (median ms, warmup-4 dropped) ====", flush=True)
                    if fwd:
                        print(f"[FWD]  h2d={_med(fwd,'h2d_ms'):.3f}  scatter={_med(fwd,'scatter_ms'):.3f}  "
                              f"DCT(up/comp/dn)={_med(fwd,'ttnn_upload_ms'):.3f}/{_med(fwd,'ttnn_compute_ms'):.3f}/{_med(fwd,'ttnn_download_ms'):.3f}  "
                              f"fw_total={_med(fwd,'fw_ms'):.3f}  server={_med(fwd,'total_server_ms'):.3f}  client={_med(fwd,'total_client_ms'):.3f}  (n={len(fwd)})", flush=True)
                    if bwd:
                        print(f"[BWD-V35]  count={_med(bwd,'count_ms'):.3f}  plan={_med(bwd,'plan_ms'):.3f}  "
                              f"place={_med(bwd,'place_ms'):.3f}  gather={_med(bwd,'gather_ms'):.3f}  "
                              f"d2h={_med(bwd,'d2h_ms'):.3f}  bw_total={_med(bwd,'total_ms'):.3f}  (n={len(bwd)})", flush=True)
                    if fwd and bwd:
                        print(f"[TOTAL density-step/iter] ~{_med(fwd,'total_server_ms')+_med(bwd,'total_ms'):.3f} ms (fw_server + bw_total)", flush=True)
                _atexit.register(_stage_summary)
            _eft.append(ef_timing)
            if os.environ.get("V35_VS_V21") == "1":
                _it = _state.get("vv2_it", 0); _state["vv2_it"] = _it + 1
                if _it == 20:
                    try:
                        v35 = out_np.astype(_np.float64).copy()
                        v21 = _np.zeros_like(out_np)
                        engine.compute_electric_force_full_chip(pos_np, _fx_addr, _fy_addr, v21)  # exact TT bwd, same field
                        v21 = -v21.astype(_np.float64)   # full path returns -grad already; match v35 (-force) → negate
                        nn = v35.shape[0]//2; nm = int(ctx.num_movable_nodes)
                        for tag,lo,hi in [("mov-x",0,nm),("mov-y",nn,nn+nm)]:
                            a=v35[lo:hi]; b=v21[lo:hi]; na_=float((a*a).sum())**.5; nb_=float((b*b).sum())**.5
                            cos=float((a*b).sum())/(na_*nb_+1e-30); rel=float(((a-b)**2).sum())**.5/(nb_+1e-30)
                            print(f"[v35_vs_v21] {tag}: |v35|={na_:.4e} |v21|={nb_:.4e} ratio={na_/(nb_+1e-30):.4f} cos={cos:.5f} relL2={rel:.4f}", flush=True)
                        d = _np.abs(v35[:nn]-v21[:nn]); worst = _np.argsort(d)[-5:][::-1]
                        g = engine.read_geom(); gf = g.view(_np.float32); ncell=g.shape[0]//32
                        gid = g[0::32][:ncell]   # active cell -> node (oidx); but here gidx=active idx
                        # build node->list of active cells (via sel = _state v35_sel_sub)
                        sel = _state.get("v35_sel_sub")
                        import collections as _c; node2cells=_c.defaultdict(list)
                        if sel is not None:
                            for ci in range(ncell):
                                node2cells[int(sel[int(g[ci*32+0])]) if int(g[ci*32+0])<len(sel) else -1].append(ci)
                        for ni in worst:
                            cs = node2cells.get(int(ni), [])
                            print(f"   node {ni}: v35=({v35[ni]:.2e},{v35[nn+ni]:.2e}) v21=({v21[ni]:.2e},{v21[nn+ni]:.2e}) #subcells={len(cs)}", flush=True)
                            for ci in cs[:3]:
                                b=ci*32
                                print(f"      cell{ci}: bxl={g[b+1]} byl={g[b+2]} kc={g[b+3]} hc={g[b+4]} w5={gf[b+5]:.2f} px={[round(float(gf[b+6+j]),2) for j in range(min(int(g[b+3]),4))]}", flush=True)
                    except Exception as _e:
                        import traceback; print(f"[v35_vs_v21] failed: {_e}\n{traceback.format_exc()}", flush=True)
            if os.environ.get("V35_ANOM") == "1":
                _it = _state.get("an_it", 0); _state["an_it"] = _it + 1
                if _it == 1:
                    try:
                        g = engine.read_geom(); gf = g.view(_np.float32)
                        ncell = g.shape[0] // 32
                        NBX = int(ctx.num_bins_x); NBY = int(ctx.num_bins_y)
                        bxl = g[1::32][:ncell]; byl = g[2::32][:ncell]; kc = g[3::32][:ncell]; hc = g[4::32][:ncell]
                        w5 = gf[5::32][:ncell]
                        v = out_np.astype(_np.float64); nn = v.shape[0]//2
                        mx = _np.abs(v); top = _np.argsort(mx)[-5:][::-1]
                        print(f"[v35_anom] ncell={ncell} max|grad|={mx.max():.3e}  bxl>=NBX:{int((bxl>=NBX).sum())} byl>=NBY:{int((byl>=NBY).sum())} "
                              f"kc>8:{int((kc>8).sum())} hc>8:{int((hc>8).sum())} w5<=0:{int((w5<=0).sum())} "
                              f"w5_max={w5.max():.3e} w5_nan:{int(_np.isnan(w5).sum())} grad_nan:{int(_np.isnan(v).sum())}", flush=True)
                        for ni in top:
                            print(f"   node {ni}: grad={v[ni]:.3e}", flush=True)
                    except Exception as _e:
                        import traceback; print(f"[v35_anom] failed: {_e}\n{traceback.format_exc()}", flush=True)
            if os.environ.get("V35_STASH_CHK") == "1":
                _it = _state.get("sc_it", 0); _state["sc_it"] = _it + 1
                if _it == 20:
                    try:
                        g = engine.read_geom(); gf = g.view(_np.float32)
                        P = ctx.pos.detach().cpu().numpy(); NNn = P.shape[0]//2
                        NSX = ctx.node_size_x_clamped.detach().cpu().numpy()
                        NSY = ctx.node_size_y_clamped.detach().cpu().numpy()
                        OX = ctx.offset_x.detach().cpu().numpy(); OY = ctx.offset_y.detach().cpu().numpy()
                        xl=float(ctx.xl); yl=float(ctx.yl); bsx=float(ctx.bin_size_x); bsy=float(ctx.bin_size_y)
                        def dp_ov(nx, ns, lo, bs):  # DREAMPlace bins + triangle overlaps
                            bxl=int((nx-lo)/bs); bxh=int((nx+ns-lo)/bs)+1
                            ov=[];
                            for k in range(bxl,bxh):
                                bl=lo+k*bs; ov.append(min(nx+ns,bl+bs)-max(nx,bl))
                            return bxl, ov
                        for rr in range(6):
                            gi=g[rr*32+0]; b=rr*32
                            nx=P[gi]+OX[gi]; ny=P[NNn+gi]+OY[gi]
                            dbxl,dpx = dp_ov(nx,NSX[gi],xl,bsx); dbyl,dpy = dp_ov(ny,NSY[gi],yl,bsy)
                            sk=g[b+3]; sh=g[b+4]
                            print(f"cell{gi}: STASH bxl={g[b+1]} byl={g[b+2]} kc={sk} hc={sh} px={[round(float(gf[b+6+j]),3) for j in range(min(sk,4))]} py={[round(float(gf[b+14+j]),3) for j in range(min(sh,4))]}", flush=True)
                            print(f"         DRMPL bxl={dbxl} byl={dbyl} kc={len(dpx)} hc={len(dpy)} px={[round(v,3) for v in dpx[:4]]} py={[round(v,3) for v in dpy[:4]]} nsx_cl={NSX[gi]:.1f} bsx={bsx:.1f}", flush=True)
                    except Exception as _e:
                        import traceback; print(f"[stash_chk] failed: {_e}\n{traceback.format_exc()}", flush=True)
            if os.environ.get("V35_GEOM_SCAN") == "1":
                _it = _state.get("gs_it", 0); _state["gs_it"] = _it + 1
                if _it in (1, 20):
                    try:
                        g = engine.read_geom(); gf = g.view(_np.float32); ncell = g.shape[0]//32
                        G = g.reshape(ncell, 32); GF = gf.reshape(ncell, 32)
                        P = ctx.pos.detach().cpu().numpy(); NNn = P.shape[0]//2
                        NSX = ctx.node_size_x_clamped.detach().cpu().numpy()
                        NSY = ctx.node_size_y_clamped.detach().cpu().numpy()
                        OX = ctx.offset_x.detach().cpu().numpy(); OY = ctx.offset_y.detach().cpu().numpy()
                        xl=float(ctx.xl); yl=float(ctx.yl); bsx=float(ctx.bin_size_x); bsy=float(ctx.bin_size_y)
                        NBX=int(ctx.num_bins_x); NBY=int(ctx.num_bins_y)
                        s_bxl=G[:,1].astype(_np.int64); s_byl=G[:,2].astype(_np.int64)
                        s_kc =G[:,3].astype(_np.int64); s_hc =G[:,4].astype(_np.int64)
                        oidx = G[:,0].astype(_np.int64)    # stash word-0 = active cell index
                        sel = _state.get("v35_sel_sub")    # active-cell index -> node index
                        sel = _np.asarray(sel).astype(_np.int64) if sel is not None else None
                        if sel is not None:
                            okv = (oidx>=0)&(oidx<len(sel))
                            node = _np.where(okv, sel[_np.where(okv,oidx,0)], -1)
                        else:
                            node = oidx; okv = (oidx>=0)&(oidx<NNn)
                        valid = okv & (node>=0) & (node<NNn)
                        gid = _np.where(valid, node, 0)    # node index (masked by `act`)
                        nx = P[gid]+OX[gid]; ny = P[NNn+gid]+OY[gid]
                        d_bxl=_np.floor((nx-xl)/bsx).astype(_np.int64)
                        d_bxh=_np.floor((nx+NSX[gid]-xl)/bsx).astype(_np.int64)+1
                        d_byl=_np.floor((ny-yl)/bsy).astype(_np.int64)
                        d_byh=_np.floor((ny+NSY[gid]-yl)/bsy).astype(_np.int64)+1
                        d_bxl=_np.clip(d_bxl,0,NBX-1); d_bxh=_np.clip(d_bxh,0,NBX)
                        d_byl=_np.clip(d_byl,0,NBY-1); d_byh=_np.clip(d_byh,0,NBY)
                        d_kc=d_bxh-d_bxl; d_hc=d_byh-d_byl
                        act=valid&(s_kc>0)&(s_hc>0); na=int(act.sum())
                        print(f"[geom_scan] iter{_it} ncell={ncell} active={na} pad/invalid={int((~valid).sum())} (NBX={NBX} bsx={bsx:.2f})", flush=True)
                        for nm,sv,dv in [("bxl",s_bxl,d_bxl),("byl",s_byl,d_byl),("kc",s_kc,d_kc),("hc",s_hc,d_hc)]:
                            d=(sv-dv)[act]; mm=int((d!=0).sum())
                            u,c=_np.unique(d,return_counts=True)
                            hist={int(k):int(v) for k,v in zip(u.tolist(),c.tolist()) if k!=0}
                            print(f"  {nm}: mismatch={mm}/{na} ({100.0*mm/max(na,1):.1f}%)  (stash-dp) diffs: {dict(sorted(hist.items(), key=lambda kv:-kv[1])[:6])}", flush=True)
                        scale=(bsx*bsy)**0.5
                        sumpx=GF[:,6:14].sum(1)*scale; sumpy=GF[:,14:22].sum(1)*scale
                        rx=sumpx[act]/_np.maximum(NSX[gid][act],1e-9); ry=sumpy[act]/_np.maximum(NSY[gid][act],1e-9)
                        print(f"  sum(px)*scale / NSX_clamped: median={_np.median(rx):.4f} mean={_np.mean(rx):.4f} (1.0=correct, <1 near edges)", flush=True)
                        print(f"  sum(py)*scale / NSY_clamped: median={_np.median(ry):.4f} mean={_np.mean(ry):.4f}", flush=True)
                        # dense-center cells (bins near NBX/2) — the ones V35 over-counts
                        cen=act & (_np.abs(s_bxl-NBX//2)<8) & (_np.abs(s_byl-NBY//2)<8)
                        print(f"  dense-center cells (|bxl-{NBX//2}|<8): {int(cen.sum())}  kc mismatch there={int(((s_kc-d_kc)[cen]!=0).sum())}  px-ratio median={_np.median((sumpx[cen]/_np.maximum(NSX[gid][cen],1e-9))) if cen.sum() else float('nan'):.4f}", flush=True)
                    except Exception as _e:
                        import traceback; print(f"[geom_scan] failed: {_e}\n{traceback.format_exc()}", flush=True)
            if os.environ.get("V35_GATHER_CHK") == "1":
                _it = _state.get("gck_it", 0); _state["gck_it"] = _it + 1
                if _it in (1,3,5,7,8,9,10,15,20,100):
                    try:
                        g = engine.read_geom(); gf = g.view(_np.float32); ncell=g.shape[0]//32
                        G=g.reshape(ncell,32); GF=gf.reshape(ncell,32)
                        NBX=int(ctx.num_bins_x); NBY=int(ctx.num_bins_y)
                        sel=_np.asarray(_state["v35_sel_sub"]).astype(_np.int64)
                        oidx=G[:,0].astype(_np.int64)
                        okv=(oidx>=0)&(oidx<len(sel)); node=_np.where(okv,sel[_np.where(okv,oidx,0)],-1)
                        s_bxl=G[:,1].astype(_np.int64); s_byl=G[:,2].astype(_np.int64)
                        s_kc=G[:,3].astype(_np.int64); s_hc=G[:,4].astype(_np.int64)
                        w5=GF[:,5].astype(_np.float64); PX=GF[:,6:14].astype(_np.float64); PY=GF[:,14:22].astype(_np.float64)
                        fxf=ctx.field_map_x.detach().cpu().numpy().reshape(-1).astype(_np.float64)
                        fyf=ctx.field_map_y.detach().cpu().numpy().reshape(-1).astype(_np.float64)
                        v=out_np.astype(_np.float64); nn=v.shape[0]//2
                        cnt=_np.bincount(node[okv&(node>=0)], minlength=nn)
                        sub=_np.where(okv&(node>=0)&(s_kc>0)&(s_hc>0)
                                      &(_np.abs(s_bxl-NBX//2)<16)&(_np.abs(s_byl-NBY//2)<16))[0]
                        # FULL vectorized python gather over ALL cells, accumulated to nodes
                        gx=_np.zeros(ncell); gy=_np.zeros(ncell)
                        for k in range(8):
                            kk=(k<s_kc)&(s_bxl+k<NBX)&(s_bxl+k>=0)
                            for h in range(8):
                                m=kk&(h<s_hc)&(s_byl+h<NBY)&(s_byl+h>=0)
                                fi=_np.where(m,(s_bxl+k)*NBY+(s_byl+h),0)
                                gx+=_np.where(m,PX[:,k]*PY[:,h]*fxf[fi],0.0)
                                gy+=_np.where(m,PX[:,k]*PY[:,h]*fyf[fi],0.0)
                        gx*=w5; gy*=w5
                        m2=okv&(node>=0)
                        py_gx=_np.bincount(node[m2],weights=gx[m2],minlength=nn)
                        py_gy=_np.bincount(node[m2],weights=gy[m2],minlength=nn)
                        # V35 out_np = -force ; python gx = +force → compare against -py_gx
                        nm=int(ctx.num_movable_nodes)
                        for tag,lo,hi,pyv,v35v in [("mov-x",0,nm,-py_gx,v[:nn]),("mov-y",0,nm,-py_gy,v[nn:])]:
                            a=v35v[lo:hi]; b=pyv[lo:hi]
                            na_=float((a*a).sum())**.5; nb_=float((b*b).sum())**.5
                            cos=float((a*b).sum())/(na_*nb_+1e-30); rel=float(((a-b)**2).sum())**.5/(nb_+1e-30)
                            print(f"[gather_chk FULL it={_it}] {tag}: |v35|={na_:.4e} |py|={nb_:.4e} ratio={na_/(nb_+1e-30):.4f} cos={cos:.6f} relL2={rel:.5f}", flush=True)
                        # locate worst-deviating nodes (where V35 != python gather)
                        d=_np.abs(v[:nm]-(-py_gx[:nm])); worst=_np.argsort(d)[-6:][::-1]
                        for ni in worst:
                            print(f"   node {ni}: v35={v[ni]:.3e} py={-py_gx[ni]:.3e} #records={int(cnt[ni])}", flush=True)
                        # ── V35 vs CPU electric_force (raw, grad_pos=ones) — the grad that drives convergence ──
                        try:
                            import torch as _t, dreamplace.ops.electric_potential.electric_potential as _epc
                            cpu=(-_epc.electric_potential_cpp.electric_force(
                                _t.ones_like(grad_pos), ctx.num_bins_x, ctx.num_bins_y,
                                ctx.num_movable_impacted_bins_x, ctx.num_movable_impacted_bins_y,
                                ctx.num_filler_impacted_bins_x, ctx.num_filler_impacted_bins_y,
                                ctx.field_map_x.view([-1]), ctx.field_map_y.view([-1]), ctx.pos,
                                ctx.node_size_x_clamped, ctx.node_size_y_clamped,
                                ctx.offset_x, ctx.offset_y, ctx.ratio,
                                ctx.bin_center_x, ctx.bin_center_y, ctx.xl, ctx.yl, ctx.xh, ctx.yh,
                                ctx.bin_size_x, ctx.bin_size_y, ctx.num_movable_nodes,
                                ctx.num_filler_nodes)).detach().cpu().numpy().astype(_np.float64)
                            for tag,lo,hi in [("mov-x",0,nm),("mov-y",nn,nn+nm),("fil/rest-x",nm,nn),("fil/rest-y",nn+nm,2*nn)]:
                                a=v[lo:hi]; b=cpu[lo:hi]; na_=float((a*a).sum())**.5; nb_=float((b*b).sum())**.5
                                cosv=float((a*b).sum())/(na_*nb_+1e-30); rel=float(((a-b)**2).sum())**.5/(nb_+1e-30)
                                print(f"   V35 vs CPU(raw) {tag}: ratio={na_/(nb_+1e-30):.5f} cos={cosv:.6f} relL2={rel:.5f} |v35|={na_:.3e} |cpu|={nb_:.3e}", flush=True)
                            # WORST per-element V35-vs-CPU deviations (real corruption vs uniform fp32?)
                            dvc=_np.abs(v-cpu); wvc=_np.argsort(dvc)[-8:][::-1]
                            for idx in wvc:
                                nidx=int(idx%nn); comp='x' if idx<nn else 'y'
                                region='mov' if nidx<nm else 'fil/fix'
                                print(f"   WORST v35-vs-cpu node{nidx}.{comp} ({region}): v35={v[idx]:.4e} cpu={cpu[idx]:.4e} absdiff={dvc[idx]:.4e}", flush=True)
                            # split by records-per-node (cnt): single vs multi-record (sub-cells)
                            cm=cnt[:nm]
                            uq,uc=_np.unique(cm,return_counts=True)
                            print(f"   movable node record-count histogram (cnt:#nodes): {dict((int(k),int(v)) for k,v in zip(uq.tolist(),uc.tolist()))}", flush=True)
                            for nmlbl,msk in [("cnt==1",cm==1),("cnt>=2",cm>=2),("cnt==0",cm==0)]:
                                if msk.sum()==0: continue
                                a=v[:nm][msk]; b=cpu[:nm][msk]; na_=float((a*a).sum())**.5; nb_=float((b*b).sum())**.5
                                print(f"      {nmlbl} ({int(msk.sum())} nodes): V35/CPU ratio={na_/(nb_+1e-30):.4f} |V35|={na_:.3e} |CPU|={nb_:.3e}", flush=True)
                            # python(stash) vs CPU(raw): isolates stash-geometry vs DREAMPlace-geometry
                            pa=-py_gx[:nm]; pb=cpu[:nm]; pna=float((pa*pa).sum())**.5; pnb=float((pb*pb).sum())**.5
                            print(f"   python(stash) vs CPU(raw) mov-x: ratio={pna/(pnb+1e-30):.5f} cos={float((pa*pb).sum())/(pna*pnb+1e-30):.6f} relL2={float(((pa-pb)**2).sum())**.5/(pnb+1e-30):.5f}", flush=True)
                            # ── CRUCIAL: the ACTUAL final grad — out_np·grad_pos (full-V35) vs -electric_force(grad_pos) (v35kern_cpugrad) ──
                            gp=grad_pos.detach().cpu().numpy().astype(_np.float64)
                            print(f"   grad_pos: shape={tuple(grad_pos.shape)} min={gp.min():.4e} max={gp.max():.4e} mean={gp.mean():.4e} (scalar? {bool(_np.allclose(gp,gp.flat[0]))})", flush=True)
                            cpug=(-_epc.electric_potential_cpp.electric_force(
                                grad_pos, ctx.num_bins_x, ctx.num_bins_y,
                                ctx.num_movable_impacted_bins_x, ctx.num_movable_impacted_bins_y,
                                ctx.num_filler_impacted_bins_x, ctx.num_filler_impacted_bins_y,
                                ctx.field_map_x.view([-1]), ctx.field_map_y.view([-1]), ctx.pos,
                                ctx.node_size_x_clamped, ctx.node_size_y_clamped, ctx.offset_x, ctx.offset_y, ctx.ratio,
                                ctx.bin_center_x, ctx.bin_center_y, ctx.xl, ctx.yl, ctx.xh, ctx.yh,
                                ctx.bin_size_x, ctx.bin_size_y, ctx.num_movable_nodes, ctx.num_filler_nodes)).detach().cpu().numpy().astype(_np.float64)
                            outf = (v*gp) if gp.shape==v.shape else (v*float(gp.flat[0]))   # full-V35 final = out_np·grad_pos
                            for tag,lo,hi in [("mov-x",0,nm),("mov-y",nn,nn+nm)]:
                                a=outf[lo:hi]; b=cpug[lo:hi]; na_=float((a*a).sum())**.5; nb_=float((b*b).sum())**.5
                                print(f"   FINAL out_np·grad_pos vs -EF(grad_pos) {tag}: ratio={na_/(nb_+1e-30):.5f} cos={float((a*b).sum())/(na_*nb_+1e-30):.6f} relL2={float(((a-b)**2).sum())**.5/(nb_+1e-30):.5f}", flush=True)
                        except Exception as _e2:
                            import traceback; print(f"   [cpu-cmp] failed: {_e2}", flush=True)
                        print(f"   (FULL: ratio/cos ≈1.0 ⇒ V35 grad == correct electric force; deviation ⇒ found the bug)", flush=True)
                        # ── per-cell: WORST-mismatching cnt==1 cells, stash vs DREAMPlace ──
                        try:
                            mvalid=okv&(node>=0)&(node<nm)
                            rec_of_node=_np.full(nm,-1,dtype=_np.int64)
                            rec_of_node[node[mvalid]]=_np.where(mvalid)[0]
                            if 'cpu' in dir():
                                mism=_np.abs((-py_gx[:nm])-cpu[:nm]); mism[cnt[:nm]!=1]=-1.0
                                worstn=_np.argsort(mism)[-6:][::-1]
                                cand=[int(rec_of_node[nd]) for nd in worstn if rec_of_node[nd]>=0]
                            else:
                                cand=[]
                            P=ctx.pos.detach().cpu().numpy(); NNn=P.shape[0]//2
                            NSX=ctx.node_size_x_clamped.detach().cpu().numpy(); NSY=ctx.node_size_y_clamped.detach().cpu().numpy()
                            OX=ctx.offset_x.detach().cpu().numpy(); OY=ctx.offset_y.detach().cpu().numpy()
                            RAT=ctx.ratio.detach().cpu().numpy()
                            xl=float(ctx.xl); yl=float(ctx.yl); bsx=float(ctx.bin_size_x); bsy=float(ctx.bin_size_y); ba=bsx*bsy
                            sc=ba**0.5
                            def boxov(nx,ns,lo,bs,b0,kc):
                                return [max(0.0,min(nx+ns,lo+(b0+k+1)*bs)-max(nx,lo+(b0+k)*bs)) for k in range(kc)]
                            shown=0
                            for c in cand[:6]:
                                nd=int(node[c]); kc=int(s_kc[c]); hc=int(s_hc[c]); bx=int(s_bxl[c]); by=int(s_byl[c])
                                pxs=[round(float(PX[c,k]*sc),3) for k in range(min(kc,5))]
                                pys=[round(float(PY[c,h]*sc),3) for h in range(min(hc,5))]
                                nx=P[nd]+OX[nd]; ny=P[NNn+nd]+OY[nd]
                                dpx=[round(x,3) for x in boxov(nx,NSX[nd],xl,bsx,bx,min(kc,5))]
                                dpy=[round(x,3) for x in boxov(ny,NSY[nd],yl,bsy,by,min(hc,5))]
                                w5c=float(GF[c,5]); rstash=w5c/ba
                                _pyf = -py_gx[nd] if nd<len(py_gx) else float('nan')
                                _cpf = cpu[nd] if 'cpu' in dir() and nd<len(cpu) else float('nan')
                                print(f"   cell node{nd}: STASH px={pxs} py={pys} ratio={rstash:.4f} | DRMPL px={dpx} py={dpy} ratio={float(RAT[nd]):.4f}  ||  py_gather={_pyf:.3e} cpu={_cpf:.3e} ratio={_pyf/(_cpf+1e-30):.3f}", flush=True)
                                shown+=1
                        except Exception as _e3:
                            import traceback; print(f"   [px-chk] failed: {_e3}\n{traceback.format_exc()}", flush=True)
                    except Exception as _e:
                        import traceback; print(f"[gather_chk] failed: {_e}\n{traceback.format_exc()}", flush=True)
            if os.environ.get("V35_DENSITY_CHK") == "1":
                _it = _state.get("dc_it", 0); _state["dc_it"] = _it + 1
                if _it in (1, 3, 5, 10, 20):
                    try:
                        ttd = getattr(client, "_last_density", None)
                        g = engine.read_geom(); gf = g.view(_np.float32); ncell=g.shape[0]//32
                        G=g.reshape(ncell,32); GF=gf.reshape(ncell,32)
                        NBX=int(ctx.num_bins_x); NBY=int(ctx.num_bins_y)
                        ba=float(ctx.bin_size_x)*float(ctx.bin_size_y)
                        s_bxl=G[:,1].astype(_np.int64); s_byl=G[:,2].astype(_np.int64)
                        s_kc=G[:,3].astype(_np.int64); s_hc=G[:,4].astype(_np.int64)
                        w5=GF[:,5].astype(_np.float64); PX=GF[:,6:14].astype(_np.float64); PY=GF[:,14:22].astype(_np.float64)
                        NB=NBX*NBY
                        dens_r=_np.zeros(NB); dens_nr=_np.zeros(NB)   # with-ratio / no-ratio
                        for k in range(8):
                            kk=(k<s_kc)&(s_bxl+k<NBX)&(s_bxl+k>=0)
                            for h in range(8):
                                m=kk&(h<s_hc)&(s_byl+h<NBY)&(s_byl+h>=0)
                                bi=_np.where(m,(s_bxl+k)*NBY+(s_byl+h),0)
                                dens_r += _np.bincount(bi[m], weights=(w5*PX[:,k]*PY[:,h])[m], minlength=NB)
                                dens_nr+= _np.bincount(bi[m], weights=(PX[:,k]*PY[:,h]*ba)[m], minlength=NB)
                        print(f"[density_chk it={_it}] python density: max(with-ratio)={dens_r.max():.4g} max(no-ratio)={dens_nr.max():.4g} sum_r={dens_r.sum():.4g}", flush=True)
                        if ttd is None:
                            print("   [density_chk] TT density NOT captured (need it in call_from_pos)", flush=True)
                        else:
                            tt=_np.asarray(ttd).reshape(-1).astype(_np.float64)
                            print(f"   TT density: shape={_np.asarray(ttd).shape} max={tt.max():.4g} sum={tt.sum():.4g} min={tt.min():.4g}", flush=True)
                            # UNIT-INDEPENDENT overflow test: TT/ideal ratio at HOT bins vs COLD(median) bins.
                            # If no overflow, ratio is a CONSTANT (just a unit scale). If hot bins wrap/saturate,
                            # hot-ratio << cold-ratio. Use dens_r (ideal with-ratio density).
                            for nm,dd in [("with-ratio",dens_r),("no-ratio",dens_nr)]:
                                if tt.shape[0]!=dd.shape[0]: continue
                                nz=_np.where(dd>1e-9)[0]
                                if nz.size==0: continue
                                order=nz[_np.argsort(dd[nz])]
                                rr=tt/_np.maximum(dd,1e-12)
                                hot=order[-200:]; mid=order[len(order)//2-100:len(order)//2+100]; cold=order[:200]
                                print(f"   vs {nm}: TT/ideal ratio — HOT(top200) med={_np.median(rr[hot]):.5f}  MID med={_np.median(rr[mid]):.5f}  COLD(low200) med={_np.median(rr[cold]):.5f}", flush=True)
                                print(f"        max ideal dens={dd.max():.4g} @bin{int(order[-1])}: TT there={tt[order[-1]]:.4g} (ideal {dd[order[-1]]:.4g})  ⇒ HOT/COLD ratio drop = {(_np.median(rr[hot])/(_np.median(rr[cold])+1e-30)):.4f} (1.0=no overflow, <1=overflow/wrap)", flush=True)
                            # fixed-point ceiling check
                            import math as _m
                            sb=int(os.environ.get("V19_SCALE_BITS","20"))
                            ceil_val=(2.0**32)/(2.0**sb)
                            print(f"   V19 fixed-point ceiling (scale_bits={sb}) ≈ {ceil_val:.1f} density units; python max(no-ratio)={dens_nr.max():.4g}", flush=True)
                    except Exception as _e:
                        import traceback; print(f"[density_chk] failed: {_e}\n{traceback.format_exc()}", flush=True)
            if os.environ.get("V35_VS_CPU") == "1":
                _it = _state.get("vc_it", 0); _state["vc_it"] = _it + 1
                if _it in (1, 20, 50):
                    try:
                        import dreamplace.ops.electric_potential.electric_potential as _epm9
                        _cpu = (-_epm9.electric_potential_cpp.electric_force(
                            grad_pos, ctx.num_bins_x, ctx.num_bins_y,
                            ctx.num_movable_impacted_bins_x, ctx.num_movable_impacted_bins_y,
                            ctx.num_filler_impacted_bins_x, ctx.num_filler_impacted_bins_y,
                            ctx.field_map_x.view([-1]), ctx.field_map_y.view([-1]), ctx.pos,
                            ctx.node_size_x_clamped, ctx.node_size_y_clamped,
                            ctx.offset_x, ctx.offset_y, ctx.ratio,
                            ctx.bin_center_x, ctx.bin_center_y, ctx.xl, ctx.yl, ctx.xh, ctx.yh,
                            ctx.bin_size_x, ctx.bin_size_y, ctx.num_movable_nodes,
                            ctx.num_filler_nodes)).detach().cpu().numpy().astype(_np.float64)
                        v = out_np.astype(_np.float64); nn = v.shape[0]//2; nm = int(ctx.num_movable_nodes)
                        def _s(lo, hi, tag):
                            a = v[lo:hi]; b = _cpu[lo:hi]
                            na_ = float((a*a).sum())**.5; nb_ = float((b*b).sum())**.5
                            cosv = float((a*b).sum())/(na_*nb_+1e-30)
                            rel = float(((a-b)**2).sum())**.5/(nb_+1e-30)
                            print(f"[v35_vs_cpu it={_it}] {tag}: |v35|={na_:.4e} |cpu|={nb_:.4e} ratio={na_/(nb_+1e-30):.4f} cos={cosv:.5f} relL2={rel:.4f}", flush=True)
                        _s(0, nm, "mov-x"); _s(nn, nn+nm, "mov-y"); _s(nm, nn, "fil-x")
                    except Exception as _e:
                        import traceback; print(f"[v35_vs_cpu] failed: {_e}\n{traceback.format_exc()}", flush=True)
            if os.environ.get("V35_TIMING") == "1":
                _n = getattr(client, "_v35_dbg_n", 0); client._v35_dbg_n = _n + 1
                if _n % 20 == 0:
                    print(f"[v35_timing] iter~{_n}: count={ef_timing.get('count_ms',0):.3f} "
                          f"plan={ef_timing.get('plan_ms',0):.3f} place={ef_timing.get('place_ms',0):.3f} "
                          f"gather={ef_timing.get('gather_ms',0):.3f} d2h={ef_timing.get('d2h_ms',0):.3f} "
                          f"total={ef_timing.get('total_ms',0):.3f} ms", flush=True)
        elif _state.get("v31"):
            # V31 no-host backward: reads the forward stash + field, writes +force
            # into out_np[sel]; negate to -force (matches CPU -electric_force).
            ef_timing = engine.compute_electric_force_v31(
                _state["v31_sel"], _state["v31_ratio_full"], fx_np, fy_np, out_np, _state["v31_nn"])
            out_np *= -1.0
        elif _state.get("v30"):
            # V30 field-stationary backward. Returns +force into out_np[sel];
            # negate to -force (matches V21 neg_ratio / CPU -electric_force).
            ef_timing = engine.compute_electric_force_v30(pos_np, fx_np, fy_np, out_np)
            out_np *= -1.0
        elif _state.get("v29"):
            # V29 L1-resident multi-bin gather backward. Returns +force into
            # out_np[sel]; negate to -force (matches V21 neg_ratio / CPU -electric_force).
            ef_timing = engine.compute_electric_force_v29(pos_np, fx_np, fy_np, out_np)
            if os.environ.get("V29_TIMING") == "1":
                _n = getattr(client, "_v29_dbg_n", 0); client._v29_dbg_n = _n + 1
                if _n % 20 == 0:
                    print(f"[v29_timing] iter~{_n}: bucket={ef_timing.get('prep_ms',0):.3f} "
                          f"gather={ef_timing.get('gather_ms',0):.3f} h2d={ef_timing.get('h2d_ms',0):.3f} "
                          f"d2h={ef_timing.get('d2h_ms',0):.3f} total={ef_timing.get('total_ms',0):.3f} ms", flush=True)
            out_np *= -1.0
        elif _state.get("zero_copy_fields", False):
            fx_addr, fy_addr = engine.latest_field_addrs()
            if fx_addr != 0 and fy_addr != 0:
                ef_timing = engine.compute_electric_force_full_chip(
                    pos_np, fx_addr, fy_addr, out_np)
            else:
                # Fallback (shouldn't happen after first scatter).
                ef_timing = engine.compute_electric_force_full(pos_np, fx_np, fy_np, out_np)
        else:
            ef_timing = engine.compute_electric_force_full(pos_np, fx_np, fy_np, out_np)

        # ── DECISIVE TEST: V21_EF_USE_CPU_GRAD=1 runs V21 kernels (above) but
        # returns the CPU electric_force gradient. If the run still diverges,
        # V21 kernel EXECUTION corrupts shared V19 state (not the gradient).
        if os.environ.get("V21_EF_USE_CPU_GRAD") == "1":
            import dreamplace.ops.electric_potential.electric_potential as _epm2
            cpu_g = -_epm2.electric_potential_cpp.electric_force(
                grad_pos, ctx.num_bins_x, ctx.num_bins_y,
                ctx.num_movable_impacted_bins_x, ctx.num_movable_impacted_bins_y,
                ctx.num_filler_impacted_bins_x, ctx.num_filler_impacted_bins_y,
                ctx.field_map_x.view([-1]), ctx.field_map_y.view([-1]), ctx.pos,
                ctx.node_size_x_clamped, ctx.node_size_y_clamped,
                ctx.offset_x, ctx.offset_y, ctx.ratio,
                ctx.bin_center_x, ctx.bin_center_y, ctx.xl, ctx.yl, ctx.xh, ctx.yh,
                ctx.bin_size_x, ctx.bin_size_y, ctx.num_movable_nodes,
                ctx.num_filler_nodes)
            return tuple([cpu_g] + [None] * 39)

        _bt_eng = time.perf_counter()   # engine (V35 gather etc.) finished above
        # Final scaling by grad_pos. Cast to ctx.pos.dtype only if needed.
        if out_torch.dtype != ctx.pos.dtype:
            output = out_torch.to(ctx.pos.dtype)
        else:
            output = out_torch
        output = output.mul(grad_pos)
        _bt_end = time.perf_counter()

        # ── Fine-grained backward timing: every stage incl. host copies, so the
        # pieces SUM to the per-iter backward wall (TOTAL). Gated by V35_TIMING.
        if os.environ.get("V35_TIMING") == "1":
            _bn = getattr(client, "_bwd_dbg_n", 0); client._bwd_dbg_n = _bn + 1
            if _bn % 20 == 0:
                _et = ef_timing if isinstance(ef_timing, dict) else {}
                pos_cp = (_bt_pos   - _bt0)     * 1000.0
                fld_cp = (_bt_field - _bt_pos)  * 1000.0
                zero_t = (_bt_prep  - _bt_field)* 1000.0
                eng_wall = (_bt_eng - _bt_prep) * 1000.0
                scale_t = (_bt_end  - _bt_eng)  * 1000.0
                total   = (_bt_end  - _bt0)     * 1000.0
                eng_sum = (_et.get('count_ms',0)+_et.get('plan_ms',0)+_et.get('place_ms',0)
                           +_et.get('gather_ms',0)+_et.get('d2h_ms',0))
                eng_resid = eng_wall - eng_sum   # pybind marshalling + untimed engine
                print(f"[bwd_full] iter~{_bn}: pos_cpy={pos_cp:.3f} field_cpy={fld_cp:.3f} "
                      f"out_zero={zero_t:.3f} | engine[count={_et.get('count_ms',0):.3f} "
                      f"plan={_et.get('plan_ms',0):.3f} place={_et.get('place_ms',0):.3f} "
                      f"gather={_et.get('gather_ms',0):.3f} d2h={_et.get('d2h_ms',0):.3f} "
                      f"pybind+resid={eng_resid:.3f}] eng_wall={eng_wall:.3f} "
                      f"scale={scale_t:.3f} TOTAL={total:.3f} ms", flush=True)

        try:
            if client._timings:
                last = client._timings[-1]
                for k, v in ef_timing.items():
                    last[f"v21_ef_{k}"] = float(v) if not isinstance(v, str) else v
        except Exception:
            pass
        # Per-iter device-profiler flush (read+clear so the on-chip buffer never
        # overflows) — captures this iter's forward+backward zones to the CSV.
        if os.environ.get("TT_PROFILE_DUMP") == "1":
            try: engine.dump_profiler()
            except Exception: pass
        return tuple([output] + [None] * 39)

    ep_mod.ElectricPotentialFunction.backward = _v21_ef_backward
    print("[scatter_ttnn_inprocess] V21_EF=1: Patched ElectricPotentialFunction.backward → "
          "V21 on-chip kernel (compute_from_full, prewarmed)", flush=True)


def _install_v22_optimizer() -> None:
    """Route the Nesterov optimizer's per-element position update through the V22
    on-chip kernels (combine→precond→nesterov→clamp), so positions are updated ON
    the TT device. Patches BOTH step_nobb (the DEFAULT path: use_bb=0, line search)
    and step_bb, replacing ONLY the update site
        u_kp1 = v_k - alpha*g_k ; v_kp1 = clamp(u_kp1 + coef*(u_kp1 - u_k))
    with a V22 call; the surrounding state machine / line search is preserved verbatim.

    V22_OPT:
      "shadow" : compute the host update AND V22's, compare, USE the host result
                 (zero convergence risk — validation only).
      "drive"  : V22 produces u_kp1/v_kp1; the device update drives placement.

    v1 milestone: feed V22 the post-precond g_k with an IDENTITY preconditioner
    (pin_w = area = 0 → divisor max(0,1)=1) and density_grad = 0, so combine+precond
    are exact no-ops and nesterov+clamp bit-reproduce the host update. (Real on-device
    combine+precond fed by raw_wl + the on-chip-unsort b_dg is the later no-host upgrade.)
    """
    import numpy as _np
    import torch as _torch
    import scatter_ttnn_client as _ipc_mod
    import NesterovAcceleratedGradientOptimizer as _nag

    _mode = os.environ.get("V22_OPT", "").lower()           # "shadow" | "drive"
    _ss_mode = os.environ.get("V22_STEPSIZE", "").lower()   # "" | "shadow" | "device" (Step B)
    _nohost = os.environ.get("V22_NOHOST", "").lower()      # "" | "shadow_grad" | "drive" (no-host grad)
    _cmp_every = int(os.environ.get("V22_OPT_CMP_EVERY", "20"))
    _st = {"configured": False, "nn": 0, "step": 0,
           "u_max": 0.0, "v_max": 0.0, "v_relmax": 0.0, "a_relmax": 0.0}

    def _ensure_configured(opt, v_k):
        if _st["configured"]:
            return
        engine  = _ipc_mod._client._engine
        model   = opt.obj_and_grad_fn.__self__
        placedb = model.placedb
        dc      = model.data_collections
        nn   = int(v_k.numel()) // 2
        nmov = int(placedb.num_movable_nodes)
        nfil = int(placedb.num_filler_nodes)
        nsx = dc.node_size_x.detach().cpu().to(_torch.float32).contiguous().numpy()
        nsy = dc.node_size_y.detach().cpu().to(_torch.float32).contiguous().numpy()
        xl, yl = float(placedb.xl), float(placedb.yl)
        xh, yh = float(placedb.xh), float(placedb.yh)
        BIG = _np.float32(1e30)
        # move_boundary clamps movable + filler (NOT fixed): x∈[xl, xh-nsx], y∈[yl, yh-nsy].
        movfil = _np.zeros(nn, dtype=bool); movfil[:nmov] = True
        if nfil > 0:
            movfil[nn - nfil:] = True
        lo = _np.empty(2 * nn, dtype=_np.float32)
        hi = _np.empty(2 * nn, dtype=_np.float32)
        lo[:nn] = _np.where(movfil, _np.float32(xl), -BIG)
        hi[:nn] = _np.where(movfil, (float(xh) - nsx).astype(_np.float32), BIG)
        lo[nn:] = _np.where(movfil, _np.float32(yl), -BIG)
        hi[nn:] = _np.where(movfil, (float(yh) - nsy).astype(_np.float32), BIG)
        if _nohost:
            # No-host: V22 does the REAL precond on device, so it needs the real
            # constants. precond_div = max(pin_w + alpha*dw*area, 1) per node, both halves.
            pw_node = model.op_collections.pws_op(dc.net_weights).detach().cpu().to(_torch.float32).numpy()
            ar_node = dc.node_areas.detach().cpu().to(_torch.float32).numpy()
            pin_w = _np.empty(2 * nn, dtype=_np.float32); pin_w[:nn] = pw_node; pin_w[nn:] = pw_node
            area  = _np.empty(2 * nn, dtype=_np.float32); area[:nn]  = ar_node; area[nn:]  = ar_node
            _st["precond_div_node"] = None  # filled per-iter (depends on dw)
            _st["pw_node"] = pw_node; _st["ar_node"] = ar_node
        else:
            pin_w = _np.zeros(2 * nn, dtype=_np.float32)        # identity precond (with-host drive)
            area  = _np.zeros(2 * nn, dtype=_np.float32)
        engine.v22_configure(lo, hi, pin_w, area, nn)
        if _nohost:
            # Route the V35 backward's on-chip unsort into V22's b_dg (grad resident).
            # skip_cpu_unsort=True for drive (no host grad at all); False for shadow.
            # The V35 engine is created lazily on the first backward — if it isn't up
            # yet (drive calls this BEFORE the first eval), do a warmup eval to build it.
            try:
                engine.v22_wire_density_grad(_nohost == "drive")
            except Exception:
                opt.obj_and_grad_fn(v_k)                 # warmup: build the V35 backward engine
                engine.v22_wire_density_grad(_nohost == "drive")
            if _nohost == "drive":
                engine.v22_set_pos(v_k.detach().cpu().to(_torch.float32).contiguous().numpy())
        _st["configured"] = True; _st["nn"] = nn
        print(f"[v22_opt] configured: mode={_mode} nohost={_nohost or 'off'} nn={nn} "
              f"movable={nmov} filler={nfil} xl/yl/xh/yh={xl:.1f}/{yl:.1f}/{xh:.1f}/{yh:.1f} "
              f"({'REAL precond + b_dg resident' if _nohost else 'identity precond'})", flush=True)

    def _v22_update(opt, g_k, v_k, u_k, alpha, coef):
        """Return (u_kp1, v_kp1_clamped) as torch CPU tensors, computed on device."""
        engine = _ipc_mod._client._engine
        model = opt.obj_and_grad_fn.__self__
        try:
            dw = float(model.density_weight.flatten()[0])
        except Exception:
            dw = float(model.density_weight)
        nn = _st["nn"]
        g_np = g_k.detach().cpu().to(_torch.float32).contiguous().numpy()
        v_np = v_k.detach().cpu().to(_torch.float32).contiguous().numpy()
        u_np = u_k.detach().cpu().to(_torch.float32).contiguous().numpy()
        dg0  = _np.zeros(2 * nn, dtype=_np.float32)
        u_new, v_new, _td = engine.v22_step(
            g_np, dg0, v_np, u_np, float(alpha), float(coef), dw, 1.0)
        return _torch.from_numpy(u_new), _torch.from_numpy(v_new)

    def _record_cmp(tag, u_dev, u_ref, v_dev, v_ref):
        du = float((u_dev - u_ref.detach().cpu()).abs().max())
        dv = float((v_dev - v_ref.detach().cpu()).abs().max())
        vscale = float(v_ref.detach().cpu().abs().max()) + 1e-30
        _st["u_max"] = max(_st["u_max"], du)
        _st["v_max"] = max(_st["v_max"], dv)
        _st["v_relmax"] = max(_st["v_relmax"], dv / vscale)
        if (_st["step"] % _cmp_every) == 0:
            print(f"[v22_opt shadow:{tag}] step={_st['step']:4d}  "
                  f"max|Δu|={du:.3e} max|Δv|={dv:.3e} (rel {dv/vscale:.2e})  "
                  f"running max: u={_st['u_max']:.3e} v={_st['v_max']:.3e} "
                  f"vrel={_st['v_relmax']:.2e}", flush=True)

    # ── step_nobb (DEFAULT for our configs): line-search Nesterov ──
    def _v22_step_nobb(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            obj_and_grad_fn = self.obj_and_grad_fn
            constraint_fn   = self.constraint_fn
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                if not group['u_k']:
                    group['u_k'].append(p.data.clone())
                    group['v_k'].append(p)
                    obj, grad = obj_and_grad_fn(group['v_k'][i])
                    group['g_k'].append(grad.data.clone())
                    group['obj_k'].append(obj.data.clone())
                u_k = group['u_k'][i]; v_k = group['v_k'][i]
                g_k = group['g_k'][i]; obj_k = group['obj_k'][i]
                if not group['a_k']:
                    group['a_k'].append(_torch.ones(1, dtype=g_k.dtype, device=g_k.device))
                    group['v_k_1'].append(_torch.autograd.Variable(_torch.zeros_like(v_k), requires_grad=True))
                    group['v_k_1'][i].data.copy_(group['v_k'][i] - group['lr'] * g_k)
                    obj, grad = obj_and_grad_fn(group['v_k_1'][i])
                    group['g_k_1'].append(grad.data)
                    group['obj_k_1'].append(obj.data.clone())
                a_k = group['a_k'][i]; v_k_1 = group['v_k_1'][i]
                g_k_1 = group['g_k_1'][i]; obj_k_1 = group['obj_k_1'][i]
                if not group['alpha_k']:
                    group['alpha_k'].append((v_k - v_k_1).norm(p=2) / (g_k - g_k_1).norm(p=2))
                alpha_k = group['alpha_k'][i]
                if group['v_kp1'][i] is None:
                    group['v_kp1'][i] = _torch.autograd.Variable(_torch.zeros_like(v_k), requires_grad=True)
                v_kp1 = group['v_kp1'][i]
                a_kp1 = (1 + (4 * a_k.pow(2) + 1).sqrt()) / 2
                coef = (a_k - 1) / a_kp1
                _ensure_configured(self, v_k)
                # ── No-host grad SHADOW: validate V22's on-device combine+precond
                #    (raw_wl + resident b_dg) reproduces DreamPlace's host g_k. b_dg holds
                #    grad@v_k for steps>=1 (skip step 0: the BB init's v_k_1 eval leaves
                #    b_dg at v_k_1). raw_wl via a separate WL-op backward. ──
                if _nohost == "shadow_grad" and _st["step"] >= 1:
                    engine = _ipc_mod._client._engine
                    model = self.obj_and_grad_fn.__self__
                    try:
                        dw = float(model.density_weight.flatten()[0])
                    except Exception:
                        dw = float(model.density_weight)
                    if v_k.grad is not None:
                        v_k.grad.zero_()
                    wl = model.op_collections.wirelength_op(v_k)
                    wl.backward()
                    raw_wl = v_k.grad.detach().cpu().to(_torch.float32).contiguous().numpy().copy()
                    engine.v22_combine_precond(raw_wl, dw, 1.0)
                    g_dev = _np.asarray(engine.v22_get_precond_grad(), dtype=_np.float32)
                    g_host = g_k.detach().cpu().to(_torch.float32).contiguous().numpy()
                    nn = _st["nn"]
                    dgr = _np.abs(g_dev[:2*nn] - g_host[:2*nn]); sc = float(_np.abs(g_host).max()) + 1e-30
                    grel = float(dgr.max()) / sc
                    _st["a_relmax"] = max(_st["a_relmax"], grel)
                    if (_st["step"] % _cmp_every) == 0:
                        print(f"[v22_nohost shadow_grad] step={_st['step']:4d} dw={dw:.4e} "
                              f"max|Δg|={float(dgr.max()):.3e} relΔ={grel:.3e} "
                              f"(running max {_st['a_relmax']:.3e})", flush=True)
                alpha_kp1 = 0
                backtrack_cnt = 0
                max_backtrack_cnt = 10
                u_kp1 = None
                while True:
                    if _mode == "drive":
                        u_kp1, v_dev = _v22_update(self, g_k, v_k, u_k, alpha_k, coef)
                        v_kp1.data.copy_(v_dev)
                    else:
                        u_kp1 = v_k - alpha_k * g_k
                        v_kp1.data.copy_(u_kp1 + coef * (u_kp1 - u_k))
                        constraint_fn(v_kp1)
                        u_dev, v_dev = _v22_update(self, g_k, v_k, u_k, alpha_k, coef)
                        _record_cmp("nobb", u_dev, u_kp1, v_dev, v_kp1)
                    f_kp1, g_kp1 = obj_and_grad_fn(v_kp1)
                    # Nesterov line-search step: alpha = sqrt(Σ(Δv)² / Σ(Δg)²).
                    alpha_kp1 = _torch.sqrt(_torch.sum((v_kp1.data - v_k.data) ** 2) /
                                            _torch.sum((g_kp1.data - g_k.data) ** 2))
                    # ── Step B: on-device reduction for the step size ──
                    if _ss_mode in ("shadow", "device"):
                        engine = _ipc_mod._client._engine
                        s_np = (v_kp1.data - v_k.data).detach().cpu().to(_torch.float32).contiguous().numpy()
                        y_np = (g_kp1.data - g_k.data).detach().cpu().to(_torch.float32).contiguous().numpy()
                        ss, sy, yy = engine.v22_stepsize_sums(s_np, y_np)
                        a_dev = float(_np.sqrt(ss / yy)) if yy > 0 else float(alpha_kp1)
                        a_host = float(alpha_kp1)
                        arel = abs(a_dev - a_host) / (abs(a_host) + 1e-30)
                        _st["a_relmax"] = max(_st["a_relmax"], arel)
                        if (_st["step"] % _cmp_every) == 0:
                            print(f"[v22_opt stepsize:{_ss_mode}] step={_st['step']:4d} "
                                  f"alpha host={a_host:.6e} dev={a_dev:.6e} relΔ={arel:.2e} "
                                  f"(running max {_st['a_relmax']:.2e})", flush=True)
                        if _ss_mode == "device":
                            alpha_kp1 = _torch.tensor(a_dev, dtype=alpha_kp1.dtype, device=alpha_kp1.device)
                    backtrack_cnt += 1
                    group['obj_eval_count'] += 1
                    if alpha_kp1 > 0.95 * alpha_k or backtrack_cnt >= max_backtrack_cnt:
                        alpha_k.data.copy_(alpha_kp1.data)
                        break
                    else:
                        alpha_k.data.copy_(alpha_kp1.data)
                v_k_1.data.copy_(v_k.data)
                g_k_1.data.copy_(g_k.data)
                obj_k_1.data.copy_(obj_k.data)
                u_k.data.copy_(u_kp1.data)
                v_k.data.copy_(v_kp1.data)
                g_k.data.copy_(g_kp1.data)
                obj_k.data.copy_(f_kp1.data)
                a_k.data.copy_(a_kp1.data)
                _st["step"] += 1
        return loss

    # ── step_bb: Barzilai-Borwein Nesterov (only if use_bb=1) ──
    def _v22_step_bb(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            obj_and_grad_fn = self.obj_and_grad_fn
            constraint_fn   = self.constraint_fn
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                if not group['u_k']:
                    group['u_k'].append(p.data.clone())
                    group['v_k'].append(p)
                u_k = group['u_k'][i]; v_k = group['v_k'][i]
                obj_k, g_k = obj_and_grad_fn(v_k)
                if not group['obj_k']:
                    group['obj_k'].append(None)
                group['obj_k'][i] = obj_k.data.clone()
                if not group['a_k']:
                    group['a_k'].append(_torch.ones(1, dtype=g_k.dtype, device=g_k.device))
                    group['v_k_1'].append(_torch.autograd.Variable(_torch.zeros_like(v_k), requires_grad=True))
                    group['v_k_1'][i].data.copy_(group['v_k'][i] - group['lr'] * g_k)
                a_k = group['a_k'][i]; v_k_1 = group['v_k_1'][i]
                obj_k_1, g_k_1 = obj_and_grad_fn(v_k_1)
                if not group['obj_k_1']:
                    group['obj_k_1'].append(None)
                group['obj_k_1'][i] = obj_k_1.data.clone()
                if group['v_kp1'][i] is None:
                    group['v_kp1'][i] = _torch.autograd.Variable(_torch.zeros_like(v_k), requires_grad=True)
                v_kp1 = group['v_kp1'][i]
                if not group['alpha_k']:
                    group['alpha_k'].append((v_k - v_k_1).norm(p=2) / (g_k - g_k_1).norm(p=2))
                alpha_k = group['alpha_k'][i]
                a_kp1 = (1 + (4 * a_k.pow(2) + 1).sqrt()) / 2
                coef = (a_k - 1) / a_kp1
                with _torch.no_grad():
                    s_k = (v_k - v_k_1)
                    y_k = (g_k - g_k_1)
                    bb_short_step_size = (s_k.dot(y_k) / y_k.dot(y_k)).data
                    lip_step_size      = (s_k.norm(p=2) / y_k.norm(p=2)).data
                    step_size = bb_short_step_size if bb_short_step_size > 0 else min(lip_step_size, alpha_k)
                _ensure_configured(self, v_k)
                if _mode == "drive":
                    u_kp1, v_dev = _v22_update(self, g_k, v_k, u_k, step_size, coef)
                    v_kp1.data.copy_(v_dev)
                else:
                    u_kp1 = v_k - step_size * g_k
                    v_kp1.data.copy_(u_kp1 + coef * (u_kp1 - u_k))
                    constraint_fn(v_kp1)
                    u_dev, v_dev = _v22_update(self, g_k, v_k, u_k, step_size, coef)
                    _record_cmp("bb", u_dev, u_kp1, v_dev, v_kp1)
                group['obj_eval_count'] += 1
                v_k_1.data.copy_(v_k.data)
                alpha_k.data.copy_(step_size.data)
                u_k.data.copy_(u_kp1.data)
                v_k.data.copy_(v_kp1.data)
                a_k.data.copy_(a_kp1.data)
                _st["step"] += 1
        return loss

    # ── No-host DRIVE line search: gradient resident in b_dg, on-device step size ──
    # Replaces step_nobb's update entirely. Per step: V22 combine+precond(raw_wl + b_dg)
    # -> g_k (snapshot); line search uses V22 nesterov/clamp into uncommitted buffers,
    # the trial forward+backward (b_dg@v_kp1), V22 combine+precond -> g_kp1, and an
    # on-device reduction for alpha = sqrt(Σ(Δpos)²/Σ(Δgrad)²); commit on accept.
    # raw_wl = obj_and_grad_fn's (precond, density-zeroed) result un-preconditioned.
    def _v22_nohost_step_nobb(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        engine = _ipc_mod._client._engine
        for group in self.param_groups:
            obj_and_grad_fn = self.obj_and_grad_fn
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                v_k = p
                model = self.obj_and_grad_fn.__self__
                _ensure_configured(self, v_k)   # v22_configure + warmup + wire(skip_cpu) + set_pos (once)
                nn = _st["nn"]
                try:
                    dw = float(model.density_weight.flatten()[0])
                except Exception:
                    dw = float(model.density_weight)
                pdiv_node = _np.maximum(_st["pw_node"] + 1.0 * dw * _st["ar_node"], 1.0).astype(_np.float32)
                pdiv = _np.concatenate([pdiv_node, pdiv_node])   # 2*nn, both halves
                if not group['a_k']:
                    group['a_k'].append(_torch.ones(1, dtype=v_k.dtype, device=v_k.device))
                a_k = group['a_k'][i]
                if not group['alpha_k']:
                    lr0 = float(group['lr']) if float(group['lr']) > 0 else 1e-3
                    group['alpha_k'].append(_torch.tensor(lr0, dtype=v_k.dtype, device=v_k.device))
                alpha_k = group['alpha_k'][i]
                a_kp1 = (1 + (4 * a_k.pow(2) + 1).sqrt()) / 2
                coef = float((a_k - 1) / a_kp1)
                # ── grad @ v_k: CARRIED from the previous step's last (accepted) trial —
                #    raw_wl@v_k + the resident b_dg@v_k are already on hand, so NO re-eval
                #    here (halves the density evals vs re-evaluating). The first step does
                #    one init eval. raw_wl is dw-independent (un-preconditioned WL grad). ──
                if _st.get("raw_wl_carried") is None:
                    obj_k, g_ret = obj_and_grad_fn(v_k)   # init eval @ initial pos -> b_dg@v_k + raw_wl
                    _st["raw_wl_carried"] = g_ret.detach().cpu().to(_torch.float32).contiguous().numpy() * pdiv
                raw_wl = _st["raw_wl_carried"]
                engine.v22_combine_precond(raw_wl, dw, 1.0)   # b_g2 = g_k = precond(raw_wl + dw*b_dg)
                engine.v22_snapshot_g()                        # b_g_k = g_k
                alpha = float(alpha_k)
                backtrack = 0
                while True:
                    engine.v22_nesterov_clamp_trial(alpha, coef)        # b_vclamp=v_kp1, b_unew=u_kp1
                    v_kp1 = engine.v22_get_trial_pos()                  # 2*nn numpy
                    v_k.data.copy_(_torch.from_numpy(v_kp1))            # set p for the forward
                    obj_kp1, g_ret1 = obj_and_grad_fn(v_k)             # forward@v_kp1 -> b_dg@v_kp1
                    raw_wl1 = g_ret1.detach().cpu().to(_torch.float32).contiguous().numpy() * pdiv
                    engine.v22_combine_precond(raw_wl1, dw, 1.0)       # b_g2 = g_kp1
                    ss, yy = engine.v22_linesearch_sums()              # Σ(v_kp1-v_k)², Σ(g_kp1-g_k)²
                    alpha_kp1 = float(_np.sqrt(ss / yy)) if yy > 0 else alpha
                    backtrack += 1
                    group['obj_eval_count'] += 1
                    if alpha_kp1 > 0.95 * alpha or backtrack >= 10:
                        alpha = alpha_kp1
                        break
                    alpha = alpha_kp1
                _st["raw_wl_carried"] = raw_wl1   # raw_wl@(accepted v_kp1) = next step's v_k; b_dg already resident
                engine.v22_commit_trial()    # b_v <- v_kp1, b_uprev <- u_kp1 (resident)
                alpha_k.data.copy_(_torch.tensor(alpha, dtype=alpha_k.dtype, device=alpha_k.device))
                a_k.data.copy_(a_kp1.data)
                # NonLinearPlace reads optimizer.param_groups[0]["obj_k_1"][0] as the
                # reported objective ("nesterov already computed the next step's obj").
                if not group['obj_k_1']:
                    group['obj_k_1'].append(None)
                group['obj_k_1'][0] = obj_kp1.data.clone()
                if (_st["step"] % _cmp_every) == 0:
                    print(f"[v22_nohost drive] step={_st['step']:4d} dw={dw:.4e} alpha={alpha:.4e} "
                          f"backtrack={backtrack} ss={ss:.3e} yy={yy:.3e}", flush=True)
                _st["step"] += 1
        return loss

    if _nohost == "drive":
        _nag.NesterovAcceleratedGradientOptimizer.step_nobb = _v22_nohost_step_nobb
        _nag.NesterovAcceleratedGradientOptimizer.step_bb   = _v22_nohost_step_nobb
        print("[v22_opt] patched step_nobb/step_bb -> V22 NO-HOST line search "
              "(grad resident in b_dg, on-device alpha)", flush=True)
    else:
        _nag.NesterovAcceleratedGradientOptimizer.step_nobb = _v22_step_nobb
        _nag.NesterovAcceleratedGradientOptimizer.step_bb   = _v22_step_bb
        print(f"[v22_opt] patched NesterovAcceleratedGradientOptimizer.step_nobb + step_bb "
              f"(mode={_mode}); active path follows use_bb", flush=True)
