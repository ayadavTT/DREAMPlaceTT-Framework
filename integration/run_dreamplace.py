#!/usr/bin/env python3
"""
run_dreamplace.py — Run DREAMPlace with selectable compute device.

Usage
-----
    # Full CPU (baseline — identical to running DREAMPlace normally)
    python3 run_dreamplace.py \\
        --device cpu \\
        --benchmark /path/to/adaptec1.json

    # CPU + TT device (only density scatter + DCT run on Blackhole)
    python3 run_dreamplace.py \\
        --device tt \\
        --benchmark /path/to/adaptec1.json \\
        --container bh-38-special-ayadav-for-reservation-73919 \\
        [--ipc-dir /tmp/tt_ep_ipc] \\
        [--nc-max 0]         # 0 = auto-detect (NC_actual × 1.5)

Output
------
  results/<benchmark>_<device>_metrics.json   HPWL, overflow, wall-time
  results/<benchmark>_<device>_tt_timings.csv scatter_ms, dct_ms per iteration
    (tt mode only)

The die bounds (xl/yl/xh/yh) and bin counts are read directly from the
running DREAMPlace forward pass — no manual entry required.
"""

import argparse
import json
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.realpath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)  # TTPort repository root
_DREAMPLACE_DIR = os.path.join(_ROOT, "DREAMPlace", "install")
_INTEGRATION_DIR = _THIS_DIR
_VENV_ROOT = os.path.join(_ROOT, "DREAMPlace", "dp_env")
_DP_ENV_PYTHON = os.path.join(_VENV_ROOT, "bin", "python3")


def _reexec_with_dp_env_if_needed() -> None:
    """Use DREAMPlace/dp_env so torch resolves without ``source activate``.

    Detect the venv with ``sys.prefix`` (not ``realpath(sys.executable)``): the
    venv's ``python3`` often symlinks to the system interpreter, so both paths
    resolve to the same binary and an executable-only check would skip re-exec.
    """
    if not os.path.isfile(_DP_ENV_PYTHON):
        return
    if os.path.realpath(sys.prefix) == os.path.realpath(_VENV_ROOT):
        return
    os.environ["VIRTUAL_ENV"] = _VENV_ROOT
    bindir = os.path.join(_VENV_ROOT, "bin")
    os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
    os.execv(_DP_ENV_PYTHON, [_DP_ENV_PYTHON] + sys.argv)


# ── parse args ────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="Run DREAMPlace with CPU or TT device for electric potential.")
    ap.add_argument("--device",
                    choices=["cpu", "tt", "ttnn_dct", "scatter_ttnn", "scatter_ttnn_direct"],
                    default="cpu",
                    help="cpu = pure CPU (baseline); tt = CPU+TT (V4-V9b kernels); "
                         "ttnn_dct = CPU scatter + TTNN DCT field solve; "
                         "scatter_ttnn = TT-Metal V4 scatter + TTNN C++ DCT via IPC; "
                         "scatter_ttnn_direct = same pipeline, direct binary launch + /dev/shm IPC (no docker exec)")
    ap.add_argument("--benchmark", required=True,
                    help="Path to DREAMPlace JSON parameter file")
    ap.add_argument("--container", default=None,
                    help="Docker container name running TT-Metal "
                         "(required for --device tt; auto-detect if omitted)")
    ap.add_argument("--ipc-dir",
                    default=None,
                    help="Shared directory for IPC files "
                         "(must be visible to both host and Docker; "
                         "default: <repo-root>/ipc)")
    ap.add_argument("--nc-max", type=int, default=0,
                    help="Maximum cell count for TT device allocation (0 = auto)")
    ap.add_argument("--results-dir", default=None,
                    help="Directory for output JSON/CSV (default: <repo-root>/results)")
    ap.add_argument("--profile", action="store_true", default=False,
                    help="Enable TT-Metal device profiler (TT mode only); "
                         "writes per-kernel per-core CSV to <results-dir>/tt_profile/")
    return ap.parse_args()


# ── auto-detect running TT Docker container ───────────────────────────────────

def _detect_container() -> str:
    import subprocess
    result = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}"],
        capture_output=True, text=True)
    lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
    # Pick first container whose name contains "bh" or "blackhole" or "special"
    for name in lines:
        if any(kw in name.lower() for kw in ("bh-", "blackhole", "special")):
            return name
    if lines:
        return lines[0]
    raise RuntimeError("No running Docker container found. "
                       "Start the TT container first, or pass --container.")


# ── inject DREAMPlace onto sys.path ──────────────────────────────────────────

def _setup_dreamplace_path() -> None:
    dp = os.path.realpath(_DREAMPLACE_DIR)
    for p in [dp, os.path.join(dp, "dreamplace")]:
        if p not in sys.path:
            sys.path.insert(0, p)
    # Also add the integration dir
    if _INTEGRATION_DIR not in sys.path:
        sys.path.insert(0, _INTEGRATION_DIR)


# ── metric extraction hook ────────────────────────────────────────────────────

_metrics: dict = {}
_iter_timings: list = []   # list of dicts, one per EP forward call
_eval_metrics_log: list = []  # per-iter {hpwl, overflow} from EvalMetrics.evaluate

# ── CPU sub-op timing buffers (filled by the patcher below) ─────────────────
_cpu_scatter_buf = [0.0]   # [0] = last scatter_ms
_cpu_dct_buf     = [0.0]   # [0] = accumulated dct_ms for current forward


def _install_timing_hooks(device: str = "cpu") -> None:
    """
    Patch ElectricPotentialFunction.forward to record per-iteration timing.
    In CPU mode: also patches ElectricDensityMapFunction.forward + DCT2/IDXST_IDCT/IDCT_IDXST
    so scatter_ms and dct_ms are measured independently for every EP call.
    """
    try:
        import dreamplace.ops.electric_potential.electric_potential as ep_mod
        import time as _time

        if device == "cpu":
            # ── Patch scatter (ElectricDensityMapFunction.forward is a staticmethod) ──
            _orig_dm = ep_mod.ElectricDensityMapFunction.forward
            @staticmethod
            def _timed_dm(ctx_or_pos, *args, **kwargs):
                t0 = _time.perf_counter()
                result = _orig_dm(ctx_or_pos, *args, **kwargs)
                _cpu_scatter_buf[0] = (_time.perf_counter() - t0) * 1000
                return result
            ep_mod.ElectricDensityMapFunction.forward = _timed_dm
            print("[run_dreamplace] Installed CPU scatter sub-op timing.", flush=True)

        # ── Patch top-level EP forward ─────────────────────────────────────
        _orig_ep = ep_mod.ElectricPotentialFunction.forward.__func__ \
            if hasattr(ep_mod.ElectricPotentialFunction.forward, "__func__") \
            else ep_mod.ElectricPotentialFunction.forward

        @staticmethod
        def _timed_forward(ctx, pos,
                           node_size_x_clamped, node_size_y_clamped,
                           offset_x, offset_y, ratio,
                           bin_center_x, bin_center_y,
                           initial_density_map, target_density,
                           xl, yl, xh, yh, bin_size_x, bin_size_y,
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
                           fast_mode=True):
            _cpu_dct_buf[0] = 0.0  # reset before each EP forward

            if device == "cpu" and dct2 is not None:
                # Wrap dct2 and idxst/idct calls to time DCT stage
                # We replace them with timing wrappers scoped to this call
                import numpy as _np
                class _TimedDCT:
                    def __init__(self, inner):
                        self._inner = inner
                    def forward(self, x):
                        t0 = _time.perf_counter()
                        r = self._inner.forward(x)
                        _cpu_dct_buf[0] += (_time.perf_counter() - t0) * 1000
                        return r

                dct2_w     = _TimedDCT(dct2)      if dct2      else None
                idct2_w    = _TimedDCT(idct2)     if idct2     else None
                idct_idxst_w = _TimedDCT(idct_idxst) if idct_idxst else None
                idxst_idct_w = _TimedDCT(idxst_idct) if idxst_idct else None

                t0 = _time.perf_counter()
                result = _orig_ep(ctx, pos,
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
                    inv_wu2_plus_wv2, wu_by_wu2_plus_wv2_half,
                    wv_by_wu2_plus_wv2_half,
                    dct2_w, idct2_w, idct_idxst_w, idxst_idct_w, fast_mode)
                ep_ms = (_time.perf_counter() - t0) * 1000
            else:
                t0 = _time.perf_counter()
                result = _orig_ep(ctx, pos,
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
                    inv_wu2_plus_wv2, wu_by_wu2_plus_wv2_half,
                    wv_by_wu2_plus_wv2_half,
                    dct2, idct2, idct_idxst, idxst_idct, fast_mode)
                ep_ms = (_time.perf_counter() - t0) * 1000

            entry = {"iter": len(_iter_timings), "ep_ms": ep_ms}
            if device == "cpu":
                entry["scatter_ms"] = _cpu_scatter_buf[0]
                entry["dct_ms"]     = _cpu_dct_buf[0]
                entry["other_ms"]   = max(0.0, ep_ms - _cpu_scatter_buf[0] - _cpu_dct_buf[0])
            elif device == "tt":
                try:
                    import tt_ep_client as _ttc
                    if _ttc._client and _ttc._client._timings:
                        entry.update(_ttc._client._timings[-1])
                except Exception:
                    pass
            elif device == "ttnn_dct":
                try:
                    import ttnn_dct_client as _tdc
                    if _tdc._client and _tdc._client._timings:
                        entry.update(_tdc._client._timings[-1])
                except Exception:
                    pass
            elif device == "scatter_ttnn_direct":
                try:
                    import scatter_ttnn_client as _stc
                    if _stc._client and _stc._client._timings:
                        entry.update(_stc._client._timings[-1])
                except Exception:
                    pass
            _iter_timings.append(entry)

            if len(_iter_timings) % 10 == 0:
                n = len(_iter_timings)
                if device == "cpu":
                    print(f"  [timing] iter {n:4d}  "
                          f"ep={ep_ms:.1f} ms  "
                          f"(scatter={_cpu_scatter_buf[0]:.1f}  "
                          f"dct={_cpu_dct_buf[0]:.1f})", flush=True)
                elif device == "ttnn_dct":
                    sc   = entry.get("scatter_cpu_ms",  float("nan"))
                    comp = entry.get("ttnn_compute_ms", float("nan"))
                    tot  = entry.get("ttnn_total_ms",   ep_ms)
                    ipc  = entry.get("ipc_overhead_ms", float("nan"))
                    print(f"  [timing] iter {n:4d}  "
                          f"ep={ep_ms:.1f} ms  "
                          f"(scatter_cpu={sc:.1f}  ttnn_compute={comp:.1f}  "
                          f"ttnn_total={tot:.1f}  ipc_oh={ipc:.1f})",
                          flush=True)
                elif device == "scatter_ttnn_direct":
                    srv = entry.get("total_server_ms", float("nan"))
                    sc  = entry.get("scatter_ms",      float("nan"))
                    gat = entry.get("gather_ms",       float("nan"))
                    dct = entry.get("ttnn_compute_ms", float("nan"))
                    pw  = entry.get("pos_write_ms",    float("nan"))
                    fr  = entry.get("field_read_ms",   float("nan"))
                    print(f"  [timing] iter {n:4d}  "
                          f"ep={ep_ms:.1f} ms  "
                          f"(server={srv:.1f}  scatter={sc:.1f}  "
                          f"gather={gat:.1f}  ttnn={dct:.1f}  "
                          f"pos_write={pw:.1f}  field_read={fr:.1f})",
                          flush=True)
                else:
                    sc  = entry.get("scatter_ms", float("nan"))
                    dct = entry.get("dct_ms",     float("nan"))
                    h2d = entry.get("h2d_ms",     float("nan"))
                    d2h = entry.get("d2h_ms",     float("nan"))
                    ipc = entry.get("ipc_overhead_ms", float("nan"))
                    total = entry.get("total_client_ms", ep_ms)
                    print(f"  [timing] iter {n:4d}  "
                          f"total={total:.1f} ms  "
                          f"(scatter={sc:.1f}  dct={dct:.1f}  "
                          f"h2d={h2d:.1f}  d2h={d2h:.1f}  ipc_oh={ipc:.1f})",
                          flush=True)
            return result

        ep_mod.ElectricPotentialFunction.forward = _timed_forward
        print(f"[run_dreamplace] Installed per-iteration EP timing hook (device={device}).",
              flush=True)
    except Exception as e:
        print(f"[run_dreamplace] Warning: EP timing hook failed: {e}", flush=True)

    try:
        import dreamplace.NonLinearPlace as nlp_mod
        _orig_call = nlp_mod.NonLinearPlace.__call__

        def _patched_call(self, params, placedb, *a, **kw):
            result = _orig_call(self, params, placedb, *a, **kw)
            if result:
                last = result[-1]
                try:
                    _metrics["hpwl"]     = float(last.hpwl)
                    _metrics["overflow"] = float(last.overflow)
                except Exception:
                    pass
            return result

        nlp_mod.NonLinearPlace.__call__ = _patched_call
    except Exception as e:
        print(f"[run_dreamplace] Warning: metrics hook failed: {e}", flush=True)

    # Per-iteration HPWL / overflow capture: monkey-patch EvalMetrics.evaluate
    # to record each iter's metrics. DREAMPlace's NonLinearPlace.py does a
    # bare `import EvalMetrics` (not `from dreamplace import EvalMetrics`),
    # so we patch BOTH module bindings to be safe.
    try:
        import EvalMetrics as em_mod  # the bare-import module DREAMPlace uses
        try:
            import dreamplace.EvalMetrics as em_mod_pkg  # also the package alias
            if em_mod_pkg is not em_mod:
                em_mod_pkg.EvalMetrics = em_mod.EvalMetrics  # unify if separate
        except Exception:
            pass
        _orig_eval = em_mod.EvalMetrics.evaluate

        def _patched_eval(self, *a, **kw):
            r = _orig_eval(self, *a, **kw)
            try:
                hpwl = float(self.hpwl) if hasattr(self, "hpwl") else float("nan")
            except Exception:
                hpwl = float("nan")
            try:
                ov = self.overflow
                if hasattr(ov, "numel") and ov.numel() > 1:
                    ov = float(ov.max())
                else:
                    ov = float(ov) if not isinstance(ov, float) else ov
            except Exception:
                ov = float("nan")
            _eval_metrics_log.append({"hpwl": hpwl, "overflow": ov})
            if len(_eval_metrics_log) <= 3 or len(_eval_metrics_log) % 100 == 0:
                print(f"  [evalmetrics] iter {len(_eval_metrics_log)}: hpwl={hpwl:.4e} overflow={ov:.4f}", flush=True)
            return r

        em_mod.EvalMetrics.evaluate = _patched_eval
        print(f"[run_dreamplace] Installed per-iteration HPWL/overflow hook on {em_mod.__file__}", flush=True)
        print(f"[run_dreamplace]   class id={id(em_mod.EvalMetrics)} method id={id(em_mod.EvalMetrics.evaluate)}", flush=True)
    except Exception as e:
        print(f"[run_dreamplace] Warning: EvalMetrics hook failed: {e}", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    _setup_dreamplace_path()

    benchmark_name = os.path.splitext(os.path.basename(args.benchmark))[0]
    results_dir = args.results_dir or os.path.join(_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    # ISPD JSONs use aux_input relative to DREAMPlace/install (see DREAMPlaceTT).
    benchmark_json = os.path.realpath(os.path.abspath(args.benchmark))
    _dp_install = os.path.realpath(_DREAMPLACE_DIR)
    os.chdir(_dp_install)

    print(f"\n{'='*60}", flush=True)
    print(f"  DREAMPlace — device={args.device}  benchmark={benchmark_name}",
          flush=True)
    print(f"{'='*60}\n", flush=True)

    # ── TT device setup ────────────────────────────────────────────────────────
    if args.device == "tt":
        container = args.container or _detect_container()
        print(f"[run_dreamplace] TT container: {container}", flush=True)

        ipc_dir = args.ipc_dir or os.path.join(_ROOT, "ipc")
        os.makedirs(ipc_dir, exist_ok=True)
        print(f"[run_dreamplace] IPC dir: {ipc_dir}", flush=True)

        profile_dir = None
        if args.profile:
            profile_dir = os.path.join(
                results_dir, "tt_profile", benchmark_name)
            os.makedirs(os.path.join(profile_dir, ".logs"), exist_ok=True)
            print(f"[run_dreamplace] Device profiler ON → {profile_dir}", flush=True)

        import tt_ep_client
        # Patch with lazy die-bound discovery (xl/yl/xh/yh filled on first call)
        tt_ep_client.patch_dreamplace(
            container=container,
            ipc_dir=ipc_dir,
            num_cells=args.nc_max,  # 0 = auto
            profile=args.profile,
            profile_dir=profile_dir,
        )

    # ── scatter_ttnn device setup (Path 1: TT-Metal V4 + TTNN C++ DCT) ──────────
    if args.device == "scatter_ttnn":
        container = args.container or _detect_container()
        print(f"[run_dreamplace] scatter_ttnn container: {container}", flush=True)
        # Default: NFS path (shared between host Python and container C++ server).
        # Both sides mmap ipc_dir/scatter.shm — same page cache on same physical host.
        ipc_dir = args.ipc_dir or os.path.join(_ROOT, "ipc_shm")
        os.makedirs(ipc_dir, exist_ok=True)
        print(f"[run_dreamplace] scatter_ttnn IPC dir: {ipc_dir}", flush=True)
        import scatter_ttnn_client
        scatter_ttnn_client.patch_dreamplace(
            container=container,
            ipc_dir=ipc_dir,
            num_cells=args.nc_max,
        )

    # ── scatter_ttnn_direct device setup (direct binary launch + /dev/shm IPC) ───
    if args.device == "scatter_ttnn_direct":
        ipc_dir = args.ipc_dir or "/dev/shm/tt_scatter_ipc"
        os.makedirs(ipc_dir, exist_ok=True)
        print(f"[run_dreamplace] scatter_ttnn_direct IPC dir: {ipc_dir} (direct launch)",
              flush=True)
        import scatter_ttnn_client
        scatter_ttnn_client.patch_dreamplace(
            container="",
            ipc_dir=ipc_dir,
            num_cells=args.nc_max,
            direct=True,
        )

    # ── ttnn_dct device setup ──────────────────────────────────────────────────
    if args.device == "ttnn_dct":
        container = args.container or _detect_container()
        print(f"[run_dreamplace] TTNN DCT container: {container}", flush=True)
        ipc_dir = args.ipc_dir or os.path.join(_ROOT, "ipc_ttnn_dct")
        os.makedirs(ipc_dir, exist_ok=True)
        print(f"[run_dreamplace] TTNN DCT IPC dir: {ipc_dir}", flush=True)
        import ttnn_dct_client
        ttnn_dct_client.patch_dreamplace(container=container, ipc_dir=ipc_dir)

    # ── Install timing hooks ───────────────────────────────────────────────────
    _install_timing_hooks(device=args.device)

    # ── Run DREAMPlace ─────────────────────────────────────────────────────────
    import Params
    import dreamplace.Placer as Placer

    params = Params.Params()
    params.load(benchmark_json)
    params.gpu = 0  # force CPU-only (no CUDA) regardless of JSON setting

    t_start = time.perf_counter()
    try:
        Placer.place(params, 0.01)
    except (SystemExit, FileNotFoundError, IOError, OSError):
        pass  # DREAMPlace raises SystemExit or file errors after placement
    wall_time_s = time.perf_counter() - t_start

    # ── Collect and save results ───────────────────────────────────────────────
    _metrics["wall_time_s"]  = wall_time_s
    _metrics["device"]       = args.device
    _metrics["benchmark"]    = benchmark_name
    _metrics["n_ep_calls"]   = len(_iter_timings)

    if _iter_timings:
        ep_times = [t["ep_ms"] for t in _iter_timings]
        _metrics["ep_mean_ms"]   = float(sum(ep_times) / len(ep_times))
        _metrics["ep_median_ms"] = float(sorted(ep_times)[len(ep_times)//2])
        _metrics["ep_total_ms"]  = float(sum(ep_times))

    if args.device == "scatter_ttnn_direct":
        import scatter_ttnn_client as _stc
        timing = _stc.get_timing_summary()
        _metrics.update({f"scatter_ttnn_direct_{k}": v for k, v in timing.items()})
        _stc.teardown()
        if _stc._client and _stc._client._timings:
            for i, entry in enumerate(_stc._client._timings):
                if i < len(_iter_timings):
                    _iter_timings[i].update(entry)

    if args.device == "scatter_ttnn":
        import scatter_ttnn_client as _stc
        timing = _stc.get_timing_summary()
        _metrics.update({f"scatter_ttnn_{k}": v for k, v in timing.items()})
        _stc.teardown()
        if _stc._client and _stc._client._timings:
            for i, entry in enumerate(_stc._client._timings):
                if i < len(_iter_timings):
                    _iter_timings[i].update(entry)

    if args.device == "ttnn_dct":
        import ttnn_dct_client as _tdc
        timing = _tdc.get_timing_summary()
        _metrics.update({f"ttnn_dct_{k}": v for k, v in timing.items()})
        _tdc.teardown()
        if _tdc._client and _tdc._client._timings:
            for i, td_entry in enumerate(_tdc._client._timings):
                if i < len(_iter_timings):
                    _iter_timings[i].update(td_entry)

    if args.device == "tt":
        import tt_ep_client as _ttc
        timing = _ttc.get_timing_summary()
        _metrics.update({f"tt_{k}": v for k, v in timing.items()})
        _ttc.teardown()

        # ── Device profiler post-processing ───────────────────────────────
        if args.profile and profile_dir:
            raw_csv = os.path.join(profile_dir, ".logs", "profile_log_device.csv")
            if os.path.exists(raw_csv):
                try:
                    proc_script = os.path.join(_INTEGRATION_DIR, "process_tt_profile.py")
                    import subprocess
                    r = subprocess.run(
                        [sys.executable, proc_script,
                         "--input",  raw_csv,
                         "--output", profile_dir,
                         "--name",   benchmark_name],
                        capture_output=True, text=True)
                    if r.returncode == 0:
                        print(r.stdout, end="", flush=True)
                    else:
                        print(f"[run_dreamplace] Profiler post-proc warning:\n{r.stderr}",
                              flush=True)
                except Exception as e:
                    print(f"[run_dreamplace] Profiler post-proc error: {e}", flush=True)
            else:
                print(f"[run_dreamplace] Warning: profiler CSV not found at {raw_csv}",
                      flush=True)

        if _ttc._client and _ttc._client._timings:
            for i, tt_entry in enumerate(_ttc._client._timings):
                if i < len(_iter_timings):
                    _iter_timings[i].update(tt_entry)

        if _iter_timings:
            import statistics as _stat
            for key in ("scatter_ms", "dct_ms", "h2d_ms", "d2h_ms",
                        "total_client_ms", "ipc_overhead_ms",
                        "pos_write_ms", "field_read_ms"):
                vals = [r[key] for r in _iter_timings if key in r]
                if vals:
                    _metrics.setdefault(f"tt_{key}_mean",   float(sum(vals)/len(vals)))
                    _metrics.setdefault(f"tt_{key}_median", float(_stat.median(vals)))

    # Save per-iteration CSV (all columns)
    if _iter_timings:
        # Merge in per-iter HPWL/overflow from the EvalMetrics hook.
        # EvalMetrics may fire more often than the EP forward (e.g. during
        # density-weight updates), so align by truncation to the shorter list.
        for i, row in enumerate(_iter_timings):
            if i < len(_eval_metrics_log):
                row["hpwl"] = _eval_metrics_log[i]["hpwl"]
                row["overflow"] = _eval_metrics_log[i]["overflow"]
        csv_path = os.path.join(results_dir, f"{benchmark_name}_{args.device}_iters.csv")
        keys = list(dict.fromkeys(k for row in _iter_timings for k in row))  # ordered union
        with open(csv_path, "w") as f:
            f.write(",".join(keys) + "\n")
            for row in _iter_timings:
                f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")
        print(f"[run_dreamplace] Per-iteration CSV → {csv_path}", flush=True)
    if _eval_metrics_log:
        # Also save the raw per-EvalMetrics log (may have more samples than EP forward).
        csv_path2 = os.path.join(results_dir, f"{benchmark_name}_{args.device}_evalmetrics.csv")
        with open(csv_path2, "w") as f:
            f.write("idx,hpwl,overflow\n")
            for i, em in enumerate(_eval_metrics_log):
                f.write(f"{i},{em['hpwl']},{em['overflow']}\n")
        # Final values from the last EvalMetrics also populate metrics if missing.
        last = _eval_metrics_log[-1]
        _metrics.setdefault("hpwl", last["hpwl"])
        _metrics.setdefault("overflow", last["overflow"])
        _metrics["n_evalmetrics"] = len(_eval_metrics_log)

    metrics_path = os.path.join(
        results_dir, f"{benchmark_name}_{args.device}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(_metrics, f, indent=2)

    print(f"\n[run_dreamplace] Done in {wall_time_s:.1f} s", flush=True)
    print(f"[run_dreamplace] HPWL     = {_metrics.get('hpwl', 'N/A')}", flush=True)
    print(f"[run_dreamplace] Overflow = {_metrics.get('overflow', 'N/A')}", flush=True)
    def _fmt(key):
        v = _metrics.get(key)
        return f"{v:.2f}" if isinstance(v, (int, float)) else "N/A"

    if args.device == "scatter_ttnn_direct":
        print(f"[run_dreamplace] pos_write  median = {_fmt('scatter_ttnn_direct_pos_write_ms_median')} ms/iter",
              flush=True)
        print(f"[run_dreamplace] field_read median = {_fmt('scatter_ttnn_direct_field_read_ms_median')} ms/iter",
              flush=True)
        print(f"[run_dreamplace] TT H2D    median = {_fmt('scatter_ttnn_direct_h2d_ms_median')} ms/iter",
              flush=True)
        print(f"[run_dreamplace] TT scatter median = {_fmt('scatter_ttnn_direct_scatter_ms_median')} ms/iter",
              flush=True)
        print(f"[run_dreamplace] TT gather  median = {_fmt('scatter_ttnn_direct_gather_ms_median')} ms/iter",
              flush=True)
        print(f"[run_dreamplace] TTNN DCT   median = {_fmt('scatter_ttnn_direct_ttnn_compute_ms_median')} ms/iter",
              flush=True)
        print(f"[run_dreamplace] Server total median = {_fmt('scatter_ttnn_direct_total_server_ms_median')} ms/iter",
              flush=True)
        print(f"[run_dreamplace] EP total   median = {_fmt('ep_median_ms')} ms/iter",
              flush=True)

    if args.device == "tt":
        print(f"[run_dreamplace] TT scatter median = {_fmt('tt_scatter_ms_median')} ms/iter",
              flush=True)
        print(f"[run_dreamplace] TT DCT    median = {_fmt('tt_dct_ms_median')} ms/iter",
              flush=True)
        print(f"[run_dreamplace] TT EP total median = {_fmt('tt_total_client_ms_median')} ms/iter",
              flush=True)
    elif args.device == "ttnn_dct":
        print(f"[run_dreamplace] CPU scatter median    = "
              f"{_fmt('ttnn_dct_scatter_cpu_ms_median')} ms/iter", flush=True)
        print(f"[run_dreamplace] TTNN upload  median   = "
              f"{_fmt('ttnn_dct_ttnn_upload_ms_median')} ms/iter", flush=True)
        print(f"[run_dreamplace] TTNN compute median   = "
              f"{_fmt('ttnn_dct_ttnn_compute_ms_median')} ms/iter", flush=True)
        print(f"[run_dreamplace] TTNN download median  = "
              f"{_fmt('ttnn_dct_ttnn_download_ms_median')} ms/iter", flush=True)
        print(f"[run_dreamplace] TTNN fw-write median  = "
              f"{_fmt('ttnn_dct_ttnn_fw_ms_median')} ms/iter", flush=True)
        print(f"[run_dreamplace] IPC overhead median   = "
              f"{_fmt('ttnn_dct_ipc_overhead_ms_median')} ms/iter", flush=True)
        print(f"[run_dreamplace] TTNN DCT total median = "
              f"{_fmt('ttnn_dct_ttnn_total_ms_median')} ms/iter", flush=True)
        print(f"[run_dreamplace] EP total median       = "
              f"{_fmt('ep_median_ms')} ms/iter", flush=True)
    print(f"[run_dreamplace] Metrics → {metrics_path}", flush=True)


if __name__ == "__main__":
    _reexec_with_dp_env_if_needed()
    main()
