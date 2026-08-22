#!/usr/bin/env python3
"""Score a ROS SLAM trajectory against the same ground truth the Python SLAM uses.

This is the join between the two halves of the study:

* ``slam/lidar_slam.py`` and ``slam/stereo_slam.py`` run on the HOST, in
  ``airsim_venv``, and are scored by ``tools/run_benchmark.py``.
* rtabmap runs in the ROS container and writes its estimate to TUM via
  ``airsim_swarm_bridge``'s ``traj_recorder`` node.

Both are then scored **by the same code** -- ``slam/evaluate.py::evaluate_trajectory``
-- against the same ``datasets/<condition>/groundtruth.txt``. That is what makes
the two ATE numbers comparable rather than merely adjacent; a second
implementation of ATE would differ in alignment, association and units, and the
comparison would be meaningless.

Run this on the HOST (it needs scipy/open3d from ``airsim_venv``, which the ROS
container deliberately does not carry):

    ./airsim_venv/bin/python tools/compare_ros_slam.py --condition clear
    ./airsim_venv/bin/python tools/compare_ros_slam.py --all --out results/ros_vs_python.csv

Producing the ROS side first:

    ./scripts/ros_enter.sh ros2 run airsim_swarm_bridge dataset_to_rosbag \
        datasets/clear --out bags/clear --skip-stereo
    ./scripts/ros_enter.sh ros2 launch airsim_swarm_bridge slam_replay.launch.py \
        bag:=bags/clear record_trajectory:=true \
        trajectory_out:=results/ros/clear_rtabmap.txt

**Weather caveat, carried forward:** AirSim weather is a rendering effect only.
Camera streams in a rain/fog recording really are degraded; the LiDAR is a
raycast and its scans are identical to ``clear``. Any LiDAR-side weather result
here -- for rtabmap exactly as for ``slam/`` -- comes from the modelled
degradation in ``slam/degradation.py``, applied by ``dataset_to_rosbag --degraded``.
The ``lidar_degraded`` column records which. Never report one without the other.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np                                    # noqa: E402

from slam import config as cfg                        # noqa: E402
from slam.evaluate import evaluate_trajectory         # noqa: E402
from slam.geometry import read_tum                    # noqa: E402

DEFAULT_ROS_DIR = REPO_ROOT / "results" / "ros"


def load_ground_truth(condition: str, dataset_root: Path):
    """Ground truth for a condition, straight from the recording."""
    gt = dataset_root / condition / "groundtruth.txt"
    if not gt.exists():
        raise FileNotFoundError(
            f"no ground truth at {gt}. Record it first:\n"
            f"    ./airsim_venv/bin/python flight/record_dataset.py --condition {condition}")
    return read_tum(gt)


def find_ros_trajectories(ros_dir: Path, condition: str) -> dict[str, Path]:
    """ROS estimates for one condition, keyed by a method label.

    Convention: ``<condition>_<method>.txt``, e.g. ``clear_rtabmap.txt`` or
    ``fog_heavy_rtabmap_degraded.txt``.
    """
    out: dict[str, Path] = {}
    for p in sorted(ros_dir.glob(f"{condition}_*.txt")):
        out[p.stem[len(condition) + 1:]] = p
    return out


def validate(est_path: Path, est_t: np.ndarray) -> None:
    """Reject a trajectory whose timestamps are not strictly increasing.

    This is not paranoia. Two ``traj_recorder`` nodes pointed at one output file
    -- easy to do by launching a replay twice, or leaving one running -- append
    into it concurrently and produce a file that still parses, still has the
    right pose count, and still yields a plausible-looking ATE. The only visible
    symptom is that ``evaluate_trajectory`` interpolates ground truth along a
    zig-zagging time axis and reports a trajectory length several times the real
    one. Better to refuse than to publish that number.
    """
    if len(est_t) < 3:
        raise ValueError(f"{est_path.name}: {len(est_t)} poses, need at least 3")

    d = np.diff(est_t)
    backward = int((d < 0).sum())
    duplicate = len(est_t) - len(np.unique(est_t))
    if backward or duplicate:
        raise ValueError(
            f"{est_path.name}: timestamps are not strictly increasing "
            f"({backward} backward jumps, {duplicate} duplicate stamps). "
            f"This is what a file written by TWO traj_recorder nodes at once "
            f"looks like -- check nothing was left running from an earlier "
            f"launch, then re-record."
        )


def score(est_path: Path, gt_t, gt_T, align: bool) -> dict:
    est_t, est_T = read_tum(est_path)
    validate(est_path, est_t)
    m = evaluate_trajectory(est_t, est_T, gt_t, gt_T, align=align)

    # A second guard on the result: the associated ground-truth path cannot be
    # meaningfully longer than the recording it came from.
    gt_len = float(np.linalg.norm(np.diff(np.asarray(gt_T)[:, :3, 3], axis=0), axis=1).sum())
    if m.trajectory_length > 1.5 * gt_len:
        raise ValueError(
            f"{est_path.name}: associated ground truth measures "
            f"{m.trajectory_length:.0f} m against a {gt_len:.0f} m recording -- "
            f"the time association is scrambled, so the metrics are meaningless"
        )
    return asdict(m)


def python_side_rows(condition: str) -> list[dict]:
    """The matching rows already computed by tools/run_benchmark.py, if present."""
    p = REPO_ROOT / "results" / "metrics.json"
    if not p.exists():
        return []
    try:
        rows = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    return [r for r in rows if r.get("condition") == condition]


def warn_length_mismatch(rows: list[dict]) -> None:
    """Shout if the estimators were not scored over comparable flights.

    ATE is not normalised by distance, so a method evaluated over 12 m and one
    evaluated over 800 m produce numbers that look comparable and are not. This
    bites in practice: ``results/metrics.json`` may have been produced by a
    ``run_benchmark.py`` invocation with ``--max-frames`` or an interrupted
    recording, and its rows will happily line up next to a full-length ROS run.
    """
    per_source: dict[str, list[float]] = {}
    for r in rows:
        length = r.get("trajectory_length")
        if length and length == length:              # not NaN
            per_source.setdefault(r.get("source", "?"), []).append(float(length))
    if len(per_source) < 2:
        return

    longest = max(max(v) for v in per_source.values())
    shortest = min(min(v) for v in per_source.values())
    if longest > 2.0 * shortest:
        print("\n" + "!" * 72)
        print("WARNING: these estimators were NOT scored over comparable flights.")
        for src, lens in sorted(per_source.items()):
            print(f"    {src:8s} {min(lens):8.1f} - {max(lens):8.1f} m")
        print("  ATE is an absolute distance, not normalised by path length, so these\n"
              "  ATE values are NOT comparable. Re-run the shorter side over the full\n"
              "  route before putting them in one table -- for the Python side that is\n"
              "      ./airsim_venv/bin/python tools/run_benchmark.py\n"
              "  without any frame limit. drift_pct and rpe_trans_pct_10m are\n"
              "  length-normalised and remain meaningful in the meantime.")
        print("!" * 72)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", action="append", default=None,
                    help="weather condition (repeatable); default: all recorded")
    ap.add_argument("--all", action="store_true", help="every condition in slam/config.py")
    ap.add_argument("--ros-dir", type=Path, default=DEFAULT_ROS_DIR,
                    help=f"where traj_recorder wrote its TUM files (default: {DEFAULT_ROS_DIR})")
    ap.add_argument("--dataset-root", type=Path, default=cfg.DATASET_ROOT)
    ap.add_argument("--out", type=Path, default=None, help="write a CSV here as well")
    ap.add_argument("--no-align", action="store_true",
                    help="skip Umeyama alignment. Odometry starts at its own origin, "
                         "so alignment is normally required; use this only to check "
                         "an estimate that is already in the world frame.")
    args = ap.parse_args(argv)

    conditions = args.condition
    if args.all or not conditions:
        conditions = [c for c in cfg.WEATHER_CONDITIONS
                      if (args.dataset_root / c).is_dir()]
    if not conditions:
        print(f"no recorded datasets under {args.dataset_root}", file=sys.stderr)
        return 1

    rows: list[dict] = []
    missing: list[str] = []

    for cond in conditions:
        try:
            gt_t, gt_T = load_ground_truth(cond, args.dataset_root)
        except FileNotFoundError as exc:
            print(f"skip {cond}: {exc}", file=sys.stderr)
            continue

        found = find_ros_trajectories(args.ros_dir, cond)
        if not found:
            missing.append(cond)

        print(f"\n=== {cond} ===")
        print(f"  ground truth: {len(gt_t)} poses, "
              f"{np.linalg.norm(np.diff(gt_T[:, :3, 3], axis=0), axis=1).sum():.0f} m flown")

        for method, path in found.items():
            try:
                m = score(path, gt_t, gt_T, align=not args.no_align)
            except (ValueError, OSError) as exc:
                print(f"  {method:22s} FAILED: {exc}")
                continue
            # 'ros:' prefix keeps these distinguishable from run_benchmark.py's
            # own rows when the two tables are concatenated.
            rows.append(dict(m, condition=cond, method=f"ros:{method}",
                             source="ros", lidar_degraded="degraded" in method))
            print(f"  {method:22s} ATE {m['ate_rmse']:7.3f} m rmse | "
                  f"RPE {m['rpe_trans_pct_10m']:6.2f}%/10m | "
                  f"drift {m['drift_pct']:5.2f}% of {m['trajectory_length']:.0f} m"
                  + ("  DIVERGED" if m["diverged"] else ""))

        for r in python_side_rows(cond):
            label = f"{r['method']}{'+degraded' if r.get('lidar_degraded') else ''}"
            rows.append(dict(r, method=f"py:{label}", source="python"))
            print(f"  py:{label:19s} ATE {r['ate_rmse']:7.3f} m rmse | "
                  f"RPE {r.get('rpe_trans_pct_10m', float('nan')):6.2f}%/10m")

    if missing:
        print(f"\nno ROS trajectories found for: {', '.join(missing)}", file=sys.stderr)
        print(f"expected files like {args.ros_dir}/<condition>_rtabmap.txt -- "
              f"run slam_replay.launch.py with record_trajectory:=true", file=sys.stderr)

    if not rows:
        return 1

    warn_length_mismatch(rows)

    print("\nNOTE: AirSim weather is a RENDERING effect only. Camera degradation is\n"
          "      simulated; LiDAR degradation is MODELLED by slam/degradation.py and\n"
          "      only present where lidar_degraded is true.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for r in rows for k in r})
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
