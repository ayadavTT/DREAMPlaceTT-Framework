#!/usr/bin/env python3
"""
process_tt_profile.py — Post-process TT-Metal device profiler CSV.

Reads profile_log_device.csv (written by TT_METAL_DEVICE_PROFILER=1) and
produces a per-kernel, per-RISC, per-core timing breakdown for the
DREAMPlace full-pipeline programs (V4 scatter + V6–V9b DCT).

Usage:
    python3 process_tt_profile.py \\
        --input  <profile_dir>/.logs/profile_log_device.csv \\
        --output <profile_dir> \\
        --name   <benchmark_name>

Outputs (in --output dir):
    tt_profile_raw_kernels.csv      per-(program, core, risc) kernel duration
    tt_profile_program_summary.csv  per-program averages across all cores
    tt_profile_report.txt           human-readable summary printed to stdout
"""

import argparse
import csv
import os
import statistics
from pathlib import Path

# ── Pipeline program order (matches density_scatter_full_pipeline_host.cpp) ──
PROGRAM_NAMES = ["V4-Scatter", "V6-Gather+yDCT", "V7-xDCT",
                 "V8a-scale+row_x", "V8b-scale+row_y",
                 "V9a-col_x", "V9b-col_y"]
N_PROGRAMS = len(PROGRAM_NAMES)

# Blackhole AICLK (read from CSV header; fallback)
FALLBACK_FREQ_MHZ = 1350


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True, help="profile_log_device.csv path")
    ap.add_argument("--output", required=True, help="Output directory for reports")
    ap.add_argument("--name",   default="run",  help="Benchmark name for filenames")
    return ap.parse_args()


def read_freq_from_header(path: str) -> int:
    """Extract CHIP_FREQ[MHz] from the first header line."""
    with open(path) as f:
        first = f.readline()
    for token in first.split(","):
        if "CHIP_FREQ" in token and "MHz" in token:
            try:
                return int(token.split(":")[-1].strip().rstrip("]").strip())
            except ValueError:
                pass
    return FALLBACK_FREQ_MHZ


def load_kernel_events(path: str):
    """
    Read KERNEL ZONE_START / ZONE_END events.
    Returns list of dicts with keys:
        core_x, core_y, risc, cycles, event_type
    Only BRISC-KERNEL, NCRISC-KERNEL, TRISC-KERNEL zones are kept.
    """
    events = []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # ARCH/FREQ header
        next(reader)  # column names
        for row in reader:
            if len(row) < 12:
                continue
            zone_name  = row[10].strip()
            event_type = row[11].strip()
            if "KERNEL" not in zone_name:
                continue
            if event_type not in ("ZONE_START", "ZONE_END"):
                continue
            try:
                events.append({
                    "core_x":     int(row[1]),
                    "core_y":     int(row[2]),
                    "risc":       row[3].strip(),
                    "cycles":     int(row[5]),
                    "zone_name":  zone_name,
                    "event_type": event_type,
                })
            except (ValueError, IndexError):
                continue
    return events


def pair_starts_ends(events):
    """
    For each (core_x, core_y, risc) group, pair ZONE_START and ZONE_END
    events in timestamp order. Returns list of
    {core_x, core_y, risc, start_cycles, end_cycles, duration_cycles}.
    """
    from collections import defaultdict
    grouped = defaultdict(lambda: {"starts": [], "ends": []})
    for e in events:
        key = (e["core_x"], e["core_y"], e["risc"])
        if e["event_type"] == "ZONE_START":
            grouped[key]["starts"].append(e["cycles"])
        else:
            grouped[key]["ends"].append(e["cycles"])

    paired = []
    for (cx, cy, risc), data in grouped.items():
        starts = sorted(data["starts"])
        ends   = sorted(data["ends"])
        for s, e in zip(starts, ends):
            if e >= s:
                paired.append({
                    "core_x":         cx,
                    "core_y":         cy,
                    "risc":           risc,
                    "start_cycles":   s,
                    "end_cycles":     e,
                    "duration_cycles": e - s,
                })
    return paired


def assign_programs(paired_events, freq_mhz: int):
    """
    Assign pipeline program labels per (core_x, core_y, risc) by sorting each
    core's kernel pairs in time order and assigning cyclically.

    The 7 pipeline programs run sequentially on the server, but different cores
    participate in different subsets:
      - All 110 cores run V4-Scatter (1 program/iter).
      - A smaller subset (~16 cores) runs all 7 programs/iter.

    We detect programs_per_iter for each core as (n_pairs // N_iters), where
    N_iters is derived from the core with the most pairs (max_pairs // N_PROGRAMS).

    Returns the same list with added fields:
        program_idx, program_name, iteration, duration_us
    """
    if not paired_events:
        return paired_events

    from collections import defaultdict

    by_core_risc = defaultdict(list)
    for e in paired_events:
        by_core_risc[(e["core_x"], e["core_y"], e["risc"])].append(e)

    # Infer total iterations from the core that runs all N_PROGRAMS
    max_pairs = max(len(v) for v in by_core_risc.values())
    N_iters = max(1, max_pairs // N_PROGRAMS)

    result = []
    for (_cx, _cy, _risc), events in by_core_risc.items():
        events_sorted = sorted(events, key=lambda e: e["start_cycles"])
        n = len(events_sorted)
        progs_per_iter = max(1, n // N_iters)
        for i, e in enumerate(events_sorted):
            iter_num  = i // progs_per_iter
            prog_idx  = i % progs_per_iter
            e["program_raw_idx"] = iter_num * N_PROGRAMS + prog_idx
            e["program_idx"]     = prog_idx
            e["program_name"]    = PROGRAM_NAMES[prog_idx]
            e["iteration"]       = iter_num
            e["duration_us"]     = e["duration_cycles"] / freq_mhz
            result.append(e)

    return result


def summarise(paired_events, freq_mhz: int):
    """
    Return two tables:
        per_program: list of {program_name, risc, n_cores, n_iters,
                               mean_us, median_us, min_us, max_us,
                               mean_ms, median_ms}
        per_core:    same but with core_x, core_y for raw CSV
    """
    from collections import defaultdict

    # Group by (program_name, risc)
    by_prog_risc = defaultdict(list)
    for e in paired_events:
        by_prog_risc[(e["program_name"], e["risc"])].append(e["duration_us"])

    # Group by (program_name, risc, core_x, core_y)
    by_prog_risc_core = defaultdict(list)
    for e in paired_events:
        key = (e["program_name"], e["risc"], e["core_x"], e["core_y"])
        by_prog_risc_core[key].append(e["duration_us"])

    per_program = []
    for (prog, risc), durations in sorted(by_prog_risc.items(),
                                           key=lambda x: (PROGRAM_NAMES.index(x[0][0])
                                                          if x[0][0] in PROGRAM_NAMES else 99,
                                                          x[0][1])):
        n_iters = max(1, len(durations) // len({k for k in by_prog_risc_core
                                                 if k[0] == prog and k[1] == risc})) \
                  if by_prog_risc_core else 1
        per_program.append({
            "program_name": prog,
            "risc":         risc,
            "n_cores":      len({k for k in by_prog_risc_core
                                  if k[0] == prog and k[1] == risc}),
            "n_samples":    len(durations),
            "mean_us":      statistics.mean(durations),
            "median_us":    statistics.median(durations),
            "min_us":       min(durations),
            "max_us":       max(durations),
            "mean_ms":      statistics.mean(durations) / 1000,
            "median_ms":    statistics.median(durations) / 1000,
        })

    per_core = []
    for (prog, risc, cx, cy), durations in sorted(by_prog_risc_core.items()):
        per_core.append({
            "program_name": prog,
            "risc":         risc,
            "core_x":       cx,
            "core_y":       cy,
            "n_samples":    len(durations),
            "mean_us":      statistics.mean(durations),
            "median_us":    statistics.median(durations),
            "min_us":       min(durations),
            "max_us":       max(durations),
        })

    return per_program, per_core


def print_report(per_program, freq_mhz, name, n_warmup_iters=4):
    """Print human-readable summary to stdout."""
    W = 88
    print(f"\n{'═'*W}")
    print(f"  TT-Metal Device Profiler Report — {name}")
    print(f"  Blackhole AICLK: {freq_mhz} MHz  |  Warmup iters excluded from summary")
    print(f"{'─'*W}")

    # Per-program summary (one row per program, columns = RISC types present)
    risc_order = ["BRISC", "NCRISC", "TRISC_0", "TRISC_1", "TRISC_2"]
    prog_risc = {}
    for row in per_program:
        prog_risc.setdefault(row["program_name"], {})[row["risc"]] = row

    print(f"\n  {'Program':<22}  {'RISC':<10}  {'Cores':>6}  "
          f"{'Mean (µs)':>10}  {'Median (µs)':>11}  {'Min':>7}  {'Max':>7}  {'Mean (ms)':>9}")
    print(f"  {'─'*22}  {'─'*10}  {'─'*6}  {'─'*10}  {'─'*11}  {'─'*7}  {'─'*7}  {'─'*9}")

    for prog in PROGRAM_NAMES:
        if prog not in prog_risc:
            continue
        first = True
        for risc in risc_order:
            if risc not in prog_risc[prog]:
                continue
            r = prog_risc[prog][risc]
            prefix = prog if first else ""
            print(f"  {prefix:<22}  {risc:<10}  {r['n_cores']:>6}  "
                  f"{r['mean_us']:>10.1f}  {r['median_us']:>11.1f}  "
                  f"{r['min_us']:>7.1f}  {r['max_us']:>7.1f}  {r['mean_ms']:>9.4f}")
            first = False

    # Per-program total kernel time (sum across all active RISC types, per core)
    print(f"\n  {'Program':<22}  {'Active RISCs':<14}  "
          f"{'Wall time (mean ms)':>20}  Note")
    print(f"  {'─'*22}  {'─'*14}  {'─'*20}  {'─'*30}")
    for prog in PROGRAM_NAMES:
        if prog not in prog_risc:
            continue
        riscs_present = [r for r in risc_order if r in prog_risc[prog]]
        # Wall time ≈ max over RISC types (they run in parallel on same core)
        wall_ms = max(prog_risc[prog][r]["mean_ms"] for r in riscs_present)
        note = ("scatter" if prog == "V4-Scatter"
                else "matmul" if "DCT" in prog or "col" in prog or "row" in prog
                else "scale")
        print(f"  {prog:<22}  {', '.join(r.replace('TRISC_','T') for r in riscs_present):<14}  "
              f"{wall_ms:>20.4f}  {note}")

    print(f"\n{'═'*W}")


def save_csvs(per_program, per_core, output_dir, name):
    prog_path = os.path.join(output_dir, f"{name}_tt_profile_program_summary.csv")
    core_path = os.path.join(output_dir, f"{name}_tt_profile_per_core.csv")

    prog_fields = ["program_name", "risc", "n_cores", "n_samples",
                   "mean_us", "median_us", "min_us", "max_us", "mean_ms", "median_ms"]
    with open(prog_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=prog_fields)
        w.writeheader(); w.writerows(per_program)
    print(f"[profiler] Program summary → {prog_path}", flush=True)

    core_fields = ["program_name", "risc", "core_x", "core_y",
                   "n_samples", "mean_us", "median_us", "min_us", "max_us"]
    with open(core_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=core_fields)
        w.writeheader(); w.writerows(per_core)
    print(f"[profiler] Per-core CSV    → {core_path}", flush=True)

    return prog_path, core_path


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    print(f"[profiler] Reading {args.input} ...", flush=True)
    freq_mhz = read_freq_from_header(args.input)
    print(f"[profiler] Device clock: {freq_mhz} MHz", flush=True)

    events = load_kernel_events(args.input)
    print(f"[profiler] Loaded {len(events)} KERNEL events", flush=True)

    paired = pair_starts_ends(events)
    print(f"[profiler] Paired {len(paired)} kernel executions", flush=True)

    paired = assign_programs(paired, freq_mhz)

    # Count iterations detected
    if paired:
        n_iters = max(e["iteration"] for e in paired) + 1
        print(f"[profiler] Detected {n_iters} pipeline passes "
              f"({n_iters - 4} timed + 4 warmup assumed)", flush=True)

        # Exclude warmup passes (first 4 pipeline passes per program)
        paired = [e for e in paired if e["iteration"] >= 4]
        print(f"[profiler] Kept {len(paired)} events after warmup exclusion", flush=True)

    per_program, per_core = summarise(paired, freq_mhz)
    print_report(per_program, freq_mhz, args.name)
    save_csvs(per_program, per_core, args.output, args.name)


if __name__ == "__main__":
    main()
