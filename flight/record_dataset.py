#!/usr/bin/env python3
"""
Fly the benchmark route under a given weather condition and record a
synchronised LiDAR + stereo + IMU + ground-truth dataset to disk.

One recording per condition; every SLAM method is then replayed against the
same bytes, so differences in the results come from the methods rather than
from run-to-run variation in the sim.

Because AirSim weather does not touch physics, the same waypoint script
produces near-identical trajectories in all five conditions -- the sensor
stream is the only variable.

Usage:
    # one condition
    ./airsim_venv/bin/python flight/record_dataset.py --condition clear

    # the whole benchmark grid back to back (~30-60 min of wall clock)
    ./airsim_venv/bin/python flight/record_dataset.py --condition all

    # if tools/probe_setup.py says lockstep stepping misbehaves on this build
    ./airsim_venv/bin/python flight/record_dataset.py --condition clear --free-running

Run tools/probe_setup.py first. Recording several GB against a mis-declared
sensor block is a slow way to find out settings.json needs a restart.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import airsim  # noqa: E402

from slam import config as cfg  # noqa: E402
from slam import weather as weather_mod  # noqa: E402
from slam.recorder import DatasetRecorder  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Record a SLAM benchmark dataset from AirSim.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--condition", default="clear",
                   help=f"one of {cfg.BENCHMARK_ORDER}, or 'all'")
    p.add_argument("--out", type=Path, default=cfg.DATASET_ROOT,
                   help="dataset root directory")
    p.add_argument("--free-running", action="store_true",
                   help="wall-clock capture instead of simPause lockstep")
    p.add_argument("--depth-gt", action="store_true",
                   help="also record ground-truth depth (evaluation only; large)")
    p.add_argument("--no-stereo", action="store_true",
                   help="LiDAR + IMU only -- much smaller and faster")
    p.add_argument("--overwrite", action="store_true",
                   help="delete an existing dataset directory instead of refusing")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    names = cfg.BENCHMARK_ORDER if args.condition == "all" else [args.condition]
    conditions = [weather_mod.get(n) for n in names]   # validate before connecting

    targets = []
    for c in conditions:
        d = args.out / c.name
        if d.exists() and any(d.iterdir()):
            if not args.overwrite:
                print(f"error: {d} already exists and is not empty "
                      f"(pass --overwrite to replace)", file=sys.stderr)
                return 1
            shutil.rmtree(d)
        targets.append((c, d))

    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("Connected to AirSim.")

    all_stats = []
    try:
        for condition, out_dir in targets:
            print("\n" + "=" * 72)
            print(f"RECORDING  {condition.name}  ->  {out_dir}")
            print("=" * 72)

            # Reset between conditions so every run starts from the same pose;
            # otherwise the second dataset begins wherever the first one landed.
            client.reset()
            client.confirmConnection()

            rec = DatasetRecorder(
                client=client,
                out_dir=out_dir,
                condition=condition,
                lockstep=not args.free_running,
                record_depth_gt=args.depth_gt,
                stereo=not args.no_stereo,
            )
            stats = rec.record()
            all_stats.append((condition.name, stats, out_dir))
            print("\n" + stats.summary())
            for w in stats.warnings:
                print(f"  WARNING: {w}")
    finally:
        try:
            weather_mod.reset(client)
            client.simPause(False)
        except Exception:
            pass

    print("\n" + "=" * 72)
    print("RECORDING COMPLETE")
    print("=" * 72)
    for name, stats, out_dir in all_stats:
        size_gb = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file()) / 1e9
        print(f"  {name:<12} {stats.lidar_scans:5d} scans  "
              f"{stats.stereo_frames:5d} stereo  {size_gb:5.2f} GB  {out_dir}")

    bad = [n for n, s, _ in all_stats if s.lidar_scans < 50]
    if bad:
        print(f"\n  WARNING: suspiciously short recordings: {bad}")

    stuck = [n for n, s, _ in all_stats if s.stuck]
    if stuck:
        print(f"\n  WARNING: aborted on a stuck drone: {stuck}")
        print("  The route flies a fixed circuit with no obstacle avoidance, so it "
              "must clear\n  everything in the environment. Raise "
              "slam.config.ROUTE_ALTITUDE (currently "
              f"{abs(cfg.ROUTE_ALTITUDE):.0f} m AGL) or move the circuit, then "
              "re-record\n  with --overwrite. Do not benchmark against a truncated "
              "recording.")

    if bad or stuck:
        return 1

    print("\nNext: tools/verify_dataset.py to sanity-check what was captured.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
