#!/usr/bin/env python3
"""
profile_v11.py — Aggregate the TT-Metal device profiler CSV into a per-zone
per-RISC summary, with emphasis on the slowest-core time per zone (which is
what bounds each iteration on V11).

The DeviceZoneScopedN markers in v11_*.cpp emit a ZONE_START / ZONE_END pair
per (core, RISC, zone, kernel-invocation). We pair them up, convert cycles to
µs using the device clock from the CSV header, then aggregate per (zone, RISC).

Usage:
    python tools/profile_v11.py <profile_log_device.csv>
    python tools/profile_v11.py <csv> --zone V11A-ACC
    python tools/profile_v11.py <csv> --zone V11A-ACC --per-core
    python tools/profile_v11.py <csv> --skip-warmup 4

Common workflows:
    # Identify the worst zone:
    python tools/profile_v11.py /tmp/profile_log_device.csv
    # Focus on V11A-ACC (the typical V11 bottleneck):
    python tools/profile_v11.py /tmp/profile_log_device.csv --zone V11A-ACC --per-core

See docs/PROFILING.md for full context.
"""

from __future__ import annotations
import argparse
import csv
import statistics
import sys
from collections import defaultdict
from typing import Dict, List, Tuple


# All V11 zone names in pipeline order. Used only for sort ordering of output.
ZONE_ORDER = [
    "V11-MAP-LOAD", "V11-CB-WAIT", "V11-ROUTE", "V11-FINAL-FLUSH", "V11-HDR-WRITE",
    "V11B-DRAM-READ", "V11B-ROUTE", "V11B-FINAL-FLUSH", "V11B-HDR-WRITE",
    "V11A-LOOKUP-LOAD", "V11A-ZERO", "V11A-HDR-READ-ALL", "V11A-DATA-READ",
    "V11A-ACC", "V11A-MERGE",
    "V11A-SHARD-WRITE", "V11A-SHARD-SUM", "V11A-SCALE", "V11A-DENSITY-WRITE",
    "V11N-LOOKUP-LOAD", "V11N-ZERO", "V11N-HDR-READ", "V11N-DATA-READ",
    "V11N-ACC", "V11N-SIGNAL",
    # Histogram kernel
    "V11H-LOOKUP-LOAD", "V11H-COUNT", "V11H-WRITE",
]
ZONE_PRIORITY = {z: i for i, z in enumerate(ZONE_ORDER)}

FALLBACK_FREQ_MHZ = 1350  # Blackhole AICLK fallback


def parse_args():
    ap = argparse.ArgumentParser(
        description="Per-zone aggregator for the TT-Metal device profiler CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("csv", help="profile_log_device.csv path")
    ap.add_argument("--zone", help="Filter to a single zone name (e.g. V11A-ACC)")
    ap.add_argument("--per-core", action="store_true",
                    help="When filtering to one zone, print the per-core breakdown")
    ap.add_argument("--skip-warmup", type=int, default=4,
                    help="Drop the first N kernel invocations per (core, RISC) "
                         "as JIT/dispatch-cold warmup (default: 4)")
    ap.add_argument("--sort", choices=["max", "median", "name"], default="max",
                    help="Sort zones by this column (default: max)")
    return ap.parse_args()


def read_freq_mhz(path: str) -> int:
    with open(path) as f:
        first = f.readline()
    for token in first.split(","):
        if "CHIP_FREQ" in token and "MHz" in token:
            try:
                return int(token.split(":")[-1].strip().rstrip("]").strip())
            except ValueError:
                pass
    return FALLBACK_FREQ_MHZ


def load_events(path: str) -> List[dict]:
    """Read ZONE_START / ZONE_END events from the profiler CSV."""
    events = []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader, None)  # ARCH/FREQ header
        next(reader, None)  # column names
        for row in reader:
            if len(row) < 12:
                continue
            zone_name = row[10].strip()
            event_type = row[11].strip()
            if event_type not in ("ZONE_START", "ZONE_END"):
                continue
            try:
                events.append({
                    "core_x": int(row[1]),
                    "core_y": int(row[2]),
                    "risc": row[3].strip(),
                    "cycles": int(row[5]),
                    "zone_name": zone_name,
                    "event_type": event_type,
                })
            except (ValueError, IndexError):
                continue
    return events


def pair_events(events: List[dict]) -> List[dict]:
    """For each (core_x, core_y, risc, zone), pair ZONE_START/END in cycle order."""
    by_key = defaultdict(lambda: {"starts": [], "ends": []})
    for e in events:
        key = (e["core_x"], e["core_y"], e["risc"], e["zone_name"])
        if e["event_type"] == "ZONE_START":
            by_key[key]["starts"].append(e["cycles"])
        else:
            by_key[key]["ends"].append(e["cycles"])

    pairs = []
    for (cx, cy, risc, zone), data in by_key.items():
        starts = sorted(data["starts"])
        ends = sorted(data["ends"])
        for invocation, (s, e) in enumerate(zip(starts, ends)):
            if e >= s:
                pairs.append({
                    "core_x": cx, "core_y": cy, "risc": risc, "zone": zone,
                    "invocation": invocation, "cycles": e - s,
                })
    return pairs


def skip_warmup(pairs: List[dict], n: int) -> List[dict]:
    """Drop the first n invocations per (core, risc, zone)."""
    return [p for p in pairs if p["invocation"] >= n]


def aggregate_per_zone(pairs: List[dict], freq_mhz: int) -> Dict[Tuple[str, str], dict]:
    """Aggregate per (zone, risc) across all cores and invocations."""
    by_zr_core: Dict[Tuple[str, str], Dict[Tuple[int, int], List[int]]] = defaultdict(
        lambda: defaultdict(list))
    for p in pairs:
        zr = (p["zone"], p["risc"])
        cc = (p["core_x"], p["core_y"])
        by_zr_core[zr][cc].append(p["cycles"])

    summary: Dict[Tuple[str, str], dict] = {}
    for zr, core_to_cycles in by_zr_core.items():
        # Per-core mean (averages out per-iter noise), in µs.
        per_core_mean_us = {
            cc: (sum(cycles) / len(cycles)) / freq_mhz
            for cc, cycles in core_to_cycles.items()
        }
        means = list(per_core_mean_us.values())
        all_invocations_us = [c / freq_mhz for cs in core_to_cycles.values() for c in cs]
        summary[zr] = {
            "n_cores": len(per_core_mean_us),
            "n_invocations": len(all_invocations_us),
            "median_us": statistics.median(means),
            "mean_us": statistics.mean(means),
            "max_us": max(means),
            "min_us": min(means),
            "max_core": max(per_core_mean_us, key=per_core_mean_us.get),
            "per_core_mean_us": per_core_mean_us,
        }
    return summary


def print_summary(summary: dict, sort_by: str):
    """Print the per-zone summary table."""
    rows = list(summary.items())
    if sort_by == "max":
        rows.sort(key=lambda r: r[1]["max_us"], reverse=True)
    elif sort_by == "median":
        rows.sort(key=lambda r: r[1]["median_us"], reverse=True)
    else:
        rows.sort(key=lambda r: (ZONE_PRIORITY.get(r[0][0], 999), r[0][1]))

    hdr_zone = "Zone"; hdr_risc = "RISC"
    print(f"\n  {hdr_zone:<22} {hdr_risc:<8}  {'cores':>5} {'invokes':>8} "
          f"{'median(µs)':>11} {'max(µs)':>10} {'max/med':>8} {'slowest core':>15}")
    print(f"  {'─'*22} {'─'*8}  {'─'*5} {'─'*8} {'─'*11} {'─'*10} {'─'*8} {'─'*15}")
    for (zone, risc), s in rows:
        ratio = s["max_us"] / s["median_us"] if s["median_us"] > 0 else float("inf")
        cx, cy = s["max_core"]
        slow = f"({cx},{cy})"
        flag = ""
        if ratio > 5.0 and s["max_us"] > 100:
            flag = "  ← imbalance"
        print(f"  {zone:<22} {risc:<8}  {s['n_cores']:>5} {s['n_invocations']:>8} "
              f"{s['median_us']:>11.2f} {s['max_us']:>10.2f} {ratio:>8.2f} "
              f"{slow:>15}{flag}")


def print_per_core(summary: dict, zone: str):
    """Print the per-core mean for one zone, sorted by mean descending."""
    matched = [((zone, risc), s) for (z, risc), s in summary.items() if z == zone]
    if not matched:
        print(f"No data for zone '{zone}'.")
        return
    for (zone, risc), s in matched:
        print(f"\n  Zone {zone} on {risc}: {s['n_cores']} cores")
        per_core = sorted(s["per_core_mean_us"].items(),
                          key=lambda kv: kv[1], reverse=True)
        print(f"    {'rank':>5}  {'core(x,y)':>10}  {'mean(µs)':>10}")
        for i, ((cx, cy), us) in enumerate(per_core[:20]):
            print(f"    {i:>5}  ({cx:>3},{cy:>3})  {us:>10.2f}")
        if len(per_core) > 20:
            print(f"    ... and {len(per_core) - 20} more cores")
        print(f"    median: {s['median_us']:.2f} µs  |  max: {s['max_us']:.2f} µs")


def main():
    args = parse_args()
    freq = read_freq_mhz(args.csv)
    print(f"[profile_v11] CSV: {args.csv}", flush=True)
    print(f"[profile_v11] Device clock: {freq} MHz", flush=True)

    events = load_events(args.csv)
    print(f"[profile_v11] Loaded {len(events)} START/END events", flush=True)
    if not events:
        sys.exit("[profile_v11] No events found — did you set TT_METAL_DEVICE_PROFILER=1?")

    pairs = pair_events(events)
    print(f"[profile_v11] Paired into {len(pairs)} zone invocations", flush=True)

    if args.skip_warmup:
        pairs = skip_warmup(pairs, args.skip_warmup)
        print(f"[profile_v11] After dropping first {args.skip_warmup} invocations "
              f"per (core, zone): {len(pairs)} remain", flush=True)

    summary = aggregate_per_zone(pairs, freq)

    if args.zone:
        zones = {z for (z, _) in summary.keys()}
        if args.zone not in zones:
            print(f"[profile_v11] zone '{args.zone}' not found. Available zones:")
            for z in sorted(zones, key=lambda z: ZONE_PRIORITY.get(z, 999)):
                print(f"  - {z}")
            sys.exit(1)
        filtered = {k: v for k, v in summary.items() if k[0] == args.zone}
        print_summary(filtered, args.sort)
        if args.per_core:
            print_per_core(filtered, args.zone)
    else:
        print_summary(summary, args.sort)
        print("\n[profile_v11] Hint: re-run with --zone <ZONE> --per-core to see "
              "per-core distribution for a specific zone.")


if __name__ == "__main__":
    main()
