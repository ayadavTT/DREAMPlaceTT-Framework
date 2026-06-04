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

    # ── V21 EF backward patch (opt-in via V21_EF=1) ────────────────────────
    if os.environ.get("V21_EF") == "1":
        _install_v21_ef_backward()
        # NOTE: the V19 d2h-skip is enabled inside the backward's configure-once
        # block (the engine is constructed lazily in start(), so _client._engine
        # is None here at patch time).


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
        pos_t = ctx.pos
        if pos_t.dtype != _torch.float32:
            pos_t = pos_t.to(_torch.float32)
        pos_np = pos_t.detach().contiguous().numpy()

        fx_t = ctx.field_map_x.detach().contiguous().view(-1)
        if fx_t.dtype != _torch.float32: fx_t = fx_t.to(_torch.float32)
        fx_np = fx_t.numpy()
        fy_t = ctx.field_map_y.detach().contiguous().view(-1)
        if fy_t.dtype != _torch.float32: fy_t = fy_t.to(_torch.float32)
        fy_np = fy_t.numpy()

        out_torch = _state["out_torch"]
        out_torch.zero_()
        out_np = out_torch.numpy()  # zero-copy view

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

        # Final scaling by grad_pos. Cast to ctx.pos.dtype only if needed.
        if out_torch.dtype != ctx.pos.dtype:
            output = out_torch.to(ctx.pos.dtype)
        else:
            output = out_torch
        output = output.mul(grad_pos)

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
