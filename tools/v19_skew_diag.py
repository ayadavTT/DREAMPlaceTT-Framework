#!/usr/bin/env python3
"""V19 per-core workload skew diagnostic.

Reads a profile_log_device.csv from a V19 run with TT_METAL_DEVICE_PROFILER=1
and computes, per core, the time spent in each V19 kernel zone. Reports
median, max, max/median ratio per zone — the imbalance factor across cores.

V19 zone names (from kernels):
  V19N-INIT       : NCRISC scatter init (zero density slab + load coords)
  V19N-CB-WAIT    : NCRISC waiting for compute kernel CB outputs
  V19N-ATOMIC     : NCRISC emitting atomic increments + barrier
  V19B-DRAM-READ  : BRISC reading px/py/sx/sy from DRAM
  V19B-ATOMIC     : BRISC emitting atomic increments
  V19-FP-CONVERT  : Writeout in-place uint32->fp32 conversion
  V19-FP-WRITE    : Writeout per-row noc_async_write to density_buf

Usage:
  python tools/v19_skew_diag.py <profile_log_device.csv> [--label TAG] [--zones a,b,...]
"""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

V19_ZONES = [
    "V19N-INIT", "V19N-CB-WAIT", "V19N-ATOMIC",
    "V19B-DRAM-READ", "V19B-ATOMIC",
    "V19-FP-CONVERT", "V19-FP-WRITE",
]


def parse_intervals(path: Path):
    """Yield (cx, cy, risc, zone, start_cycles, end_cycles, freq_mhz)."""
    with path.open() as f:
        rd = csv.reader(f)
        header_line = next(rd, None)
        # Per-arch header conventions; rely on column positions used by V11 tool.
        next(rd, None)
        for row in rd:
            if not row or row[0].startswith("#"):
                continue
            try:
                # Columns: PCIeSlot, deviceID, core_x, core_y, risc, run_id,
                # run_host_id, zone_phase, zone_name, source_line, source_file,
                # stat_value, timer_value, type, num_marker, run_freq_mhz, ...
                cx = int(row[2])
                cy = int(row[3])
                risc = row[4].strip()
                phase = row[7].strip()
                zone = row[8].strip()
                cycles = int(row[12])
                freq_mhz = float(row[15]) if len(row) > 15 and row[15] else 1350.0
            except (ValueError, IndexError):
                continue
            if zone not in V19_ZONES:
                continue
            yield cx, cy, risc, zone, phase, cycles, freq_mhz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile", type=Path)
    ap.add_argument("--label", default="")
    ap.add_argument("--zones", default=",".join(V19_ZONES),
                    help="comma-separated zone names to report")
    args = ap.parse_args()

    selected_zones = set(args.zones.split(","))
    # Per (core, zone): list of (start_cy, end_cy) intervals
    intervals = defaultdict(list)
    freq_mhz = 1350.0
    # Reconstruct intervals from begin/end pairs sharing run_host_id (simplistic
    # approach: just pair successive begin/end events per core/zone)
    open_starts = defaultdict(list)
    for cx, cy, risc, zone, phase, cy_value, fmhz in parse_intervals(args.profile):
        freq_mhz = fmhz
        key = (cx, cy, risc, zone)
        if "begin" in phase.lower() or phase.lower() == "zone_start":
            open_starts[key].append(cy_value)
        elif "end" in phase.lower() or phase.lower() == "zone_end":
            if open_starts[key]:
                start = open_starts[key].pop(0)
                intervals[key].append((start, cy_value))

    if not intervals:
        # Fallback: csv has a single-cycle "duration" column. Treat each row
        # as a complete sample if no begin/end pairing was found.
        for cx, cy, risc, zone, phase, cy_value, fmhz in parse_intervals(args.profile):
            freq_mhz = fmhz
            intervals[(cx, cy, risc, zone)].append((0, cy_value))

    ns_per_cycle = 1000.0 / freq_mhz  # ns
    print(f"\n=== V19 per-core skew {('[' + args.label + ']') if args.label else ''} "
          f"(freq={freq_mhz:.1f} MHz) ===\n")
    print(f"{'zone':<18} {'cores':>5} {'med µs':>10} {'p90 µs':>10} {'max µs':>10} {'max/med':>8} {'total µs':>12}")
    for zone in V19_ZONES:
        if zone not in selected_zones:
            continue
        per_core = defaultdict(int)  # (cx,cy) -> total cycles in zone
        for (cx, cy, risc, z), ivs in intervals.items():
            if z != zone:
                continue
            for s, e in ivs:
                per_core[(cx, cy)] += max(0, e - s)
        if not per_core:
            continue
        vals_us = [v * ns_per_cycle / 1000.0 for v in per_core.values() if v > 0]
        if not vals_us:
            continue
        vals_us.sort()
        med = statistics.median(vals_us)
        mx = vals_us[-1]
        p90 = vals_us[max(0, int(0.9 * len(vals_us)) - 1)]
        total = sum(vals_us)
        ratio = (mx / med) if med > 0 else float("inf")
        print(f"{zone:<18} {len(vals_us):>5} {med:>10.2f} {p90:>10.2f} {mx:>10.2f} {ratio:>8.2f} {total:>12.2f}")
    print()


if __name__ == "__main__":
    main()
