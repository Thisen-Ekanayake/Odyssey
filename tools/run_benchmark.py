#!/usr/bin/env python3
"""
The benchmark: every SLAM method against every weather condition, offline.

Runs the cross product of

    {5 weather conditions} x {lidar_inertial, stereo_inertial} x {raw, degraded}

over the recorded datasets, writing a metrics CSV plus figures. Needs no
simulator -- it reads only what ``flight/record_dataset.py`` put on disk, so it
can be re-run freely as the algorithms change.

The ``raw`` / ``degraded`` axis is the honest treatment of a simulator
limitation. AirSim's weather degrades the camera stream but not the LiDAR, so:

  * ``raw``      -- exactly what the simulator produced. LiDAR is flat across
                    conditions; cameras genuinely degrade.
  * ``degraded`` -- ``slam.degradation`` applied to the LiDAR at the same
                    severity. This is a model, and it is labelled as one.

Reporting both means the simulator's limits are part of the finding rather than
a hidden flaw in it.

Usage:
    ./airsim_venv/bin/python tools/run_benchmark.py
    ./airsim_venv/bin/python tools/run_benchmark.py --methods lidar --conditions clear
    ./airsim_venv/bin/python tools/run_benchmark.py --max-frames 300   # quick pass
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import open3d as o3d  # noqa: E402

from slam import config as cfg  # noqa: E402
from slam import degradation as deg  # noqa: E402
from slam.evaluate import (evaluate_map, evaluate_trajectory,  # noqa: E402
                           reference_map_from_ground_truth)
from slam.lidar_slam import LidarInertialSLAM  # noqa: E402
from slam.source import DatasetSource  # noqa: E402
from slam.stereo_slam import StereoInertialSLAM  # noqa: E402

METHODS = ("lidar", "stereo")


def dataset_supports(dataset: Path, method: str) -> tuple[bool, str]:
    """Does this dataset carry the streams ``method`` needs?

    Checked up front so a dataset recorded with --no-stereo reports "no stereo
    frames" rather than running a method against nothing and reporting
    DIVERGED, which reads as an algorithm failure instead of missing input.
    """
    try:
        probe = DatasetSource(dataset, load_lidar=True, load_stereo=True, max_frames=1)
    except Exception as exc:
        return False, f"unreadable: {exc}"
    if method == "lidar" and not probe.has_lidar:
        return False, "no LiDAR scans in this dataset"
    if method == "stereo" and not probe.has_stereo:
        return False, "no stereo frames in this dataset (recorded with --no-stereo?)"
    return True, ""


def build(method: str, loops: bool, use_ba: bool = False):
    if method == "lidar":
        return LidarInertialSLAM(verbose=False, use_loop_closure=loops)
    if method == "stereo":
        # BA is off by default; see StereoInertialSLAM.__init__ for the measured
        # reason. --stereo-ba turns it back on for comparison runs.
        return StereoInertialSLAM(verbose=False, use_loop_closure=loops, use_ba=use_ba)
    raise ValueError(f"unknown method {method!r}")


def run_one(dataset: Path, method: str, condition: str, degrade: bool,
            loops: bool, max_frames: int | None, reference_map,
            seed: int = 42, use_ba: bool = False) -> dict:
    """One (dataset, method, degradation) cell of the grid.

    The RNG is re-seeded *per cell*, not once per process. Open3D's RANSAC
    (used in loop-closure verification) draws from a global stream, so seeding
    only at startup would make every cell depend on how many cells ran before
    it -- two cells with identical input would then produce different numbers,
    and the grid would not be reproducible or re-orderable.
    """
    o3d.utility.random.seed(seed)

    lidar_tf = None
    if degrade and method == "lidar":
        model = deg.for_condition(cfg.WEATHER_CONDITIONS[condition], seed=0)
        lidar_tf = model

    src = DatasetSource(
        dataset,
        load_lidar=(method == "lidar"),
        load_stereo=(method == "stereo"),
        lidar_transform=lidar_tf,
        max_frames=max_frames,
    )

    slam = build(method, loops, use_ba)
    t0 = time.perf_counter()
    res = slam.run(src, progress_every=0)
    wall = time.perf_counter() - t0

    traj = evaluate_trajectory(res.times, res.poses, res.gt_times, res.gt_poses)

    row: dict = {
        "condition": condition,
        "method": method,
        "lidar_degraded": degrade and method == "lidar",
        "loop_closure": loops,
        "stereo_ba": use_ba and method == "stereo",
        "severity": cfg.WEATHER_CONDITIONS[condition].severity,
        "wall_seconds": round(wall, 1),
        **{k: (round(v, 5) if isinstance(v, float) else v)
           for k, v in traj.as_dict().items()},
        **{f"stat_{k}": (round(v, 5) if isinstance(v, float) else v)
           for k, v in res.stats.items()},
        **{f"time_{k}": round(v, 3) for k, v in res.timings.items()},
    }

    if reference_map is not None and len(res.map_points):
        mm = evaluate_map(res.map_points, reference_map)
        row.update({f"map_{k}": (round(v, 5) if isinstance(v, float) else v)
                    for k, v in mm.as_dict().items()})

    if degrade and method == "lidar":
        row["alpha"] = round(lidar_tf.alpha, 6)
        row["effective_range_m"] = round(lidar_tf.effective_range(), 2)

    return row, res


def make_figures(rows: list[dict], out_dir: Path) -> None:
    """Degradation curves and a summary table image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    order = {c: i for i, c in enumerate(cfg.BENCHMARK_ORDER)}

    def series(method, degraded):
        pts = [r for r in rows
               if r["method"] == method and bool(r["lidar_degraded"]) == degraded
               and not r.get("diverged", False)]
        pts.sort(key=lambda r: order.get(r["condition"], 99))
        return pts

    # -- ATE / RPE per condition ------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    variants = [
        ("lidar", False, "LiDAR-inertial (raw sim)", "tab:blue", "-o"),
        ("lidar", True, "LiDAR-inertial (modelled degradation)", "tab:cyan", "--s"),
        ("stereo", False, "Stereo-inertial", "tab:orange", "-^"),
    ]
    for method, degraded, label, colour, style in variants:
        pts = series(method, degraded)
        if not pts:
            continue
        x = [order.get(p["condition"], 99) for p in pts]
        axes[0].plot(x, [p["ate_rmse"] for p in pts], style, color=colour, label=label)
        axes[1].plot(x, [p["rpe_trans_pct_10m"] for p in pts], style, color=colour,
                     label=label)

    for ax, ylab, title in (
        (axes[0], "ATE RMSE (m)", "Absolute trajectory error"),
        (axes[1], "RPE (% per 10 m)", "Local drift"),
    ):
        ax.set_xticks(range(len(cfg.BENCHMARK_ORDER)))
        ax.set_xticklabels(cfg.BENCHMARK_ORDER, rotation=20, ha="right")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "degradation_curves.png", dpi=140)
    plt.close(fig)

    # -- robustness --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.2))
    width = 0.26
    for i, (method, degraded, label, colour, _) in enumerate(variants):
        pts = series(method, degraded)
        if not pts:
            continue
        x = np.array([order.get(p["condition"], 99) for p in pts], dtype=float)
        ax.bar(x + (i - 1) * width,
               [100 * p.get("stat_tracking_failure_rate", 0) for p in pts],
               width, label=label, color=colour)
    ax.set_xticks(range(len(cfg.BENCHMARK_ORDER)))
    ax.set_xticklabels(cfg.BENCHMARK_ORDER, rotation=20, ha="right")
    ax.set_ylabel("tracking failures (% of frames)")
    ax.set_title("Front-end robustness")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "robustness.png", dpi=140)
    plt.close(fig)

    print(f"  figures -> {out_dir}/degradation_curves.png, {out_dir}/robustness.png")


def plot_trajectories(results: dict, out_dir: Path) -> None:
    """Top-down estimated-vs-truth overlay per condition."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = [c for c in cfg.BENCHMARK_ORDER if any(k[0] == c for k in results)]
    if not conds:
        return
    fig, axes = plt.subplots(1, len(conds), figsize=(4.2 * len(conds), 4.4),
                             squeeze=False)
    for ax, cond in zip(axes[0], conds):
        drawn = False
        for (c, method, degraded), res in results.items():
            if c != cond or len(res.poses) == 0:
                continue
            if not drawn and len(res.gt_poses):
                g = res.gt_poses[:, :3, 3]
                ax.plot(g[:, 1], g[:, 0], "k-", lw=2.2, alpha=0.45, label="ground truth")
                drawn = True
            p = res.poses[:, :3, 3]
            lbl = f"{method}{' (degraded)' if degraded else ''}"
            ax.plot(p[:, 1], p[:, 0], lw=1.2, label=lbl)
        ax.set_title(cond)
        ax.set_xlabel("East / y (m)")
        ax.set_ylabel("North / x (m)")
        ax.axis("equal")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "trajectories.png", dpi=140)
    plt.close(fig)
    print(f"  figures -> {out_dir}/trajectories.png")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", type=Path, default=cfg.DATASET_ROOT)
    ap.add_argument("--out", type=Path, default=cfg.RESULTS_ROOT)
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    ap.add_argument("--conditions", nargs="+", default=cfg.BENCHMARK_ORDER)
    ap.add_argument("--max-frames", type=int, default=None,
                    help="truncate each dataset (quick smoke run)")
    ap.add_argument("--no-loop-closure", action="store_true")
    ap.add_argument("--no-degraded", action="store_true",
                    help="skip the modelled-LiDAR-degradation arm")
    ap.add_argument("--no-map-metrics", action="store_true",
                    help="skip cloud-to-cloud map scoring (slow on big maps)")
    ap.add_argument("--stereo-ba", action="store_true",
                    help="enable windowed bundle adjustment in the stereo front end "
                         "(off by default: measured no ATE benefit, +55%% runtime)")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed applied per grid cell (loop-closure RANSAC)")
    ap.add_argument("--repeats", type=int, default=1,
                    help="repeat each cell with consecutive seeds and report the "
                         "spread; loop closure is stochastic, so a single run can "
                         "be a lucky or unlucky draw")
    args = ap.parse_args()

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    available = [c for c in args.conditions if (args.datasets / c).is_dir()]
    if not available:
        print(f"no datasets under {args.datasets}. Run flight/record_dataset.py first.",
              file=sys.stderr)
        return 1
    print(f"datasets: {available}")

    # Reference map: clear-condition scans placed by ground-truth poses, i.e.
    # the best map this sensor could build with perfect localisation.
    reference_map = None
    if not args.no_map_metrics and "clear" in available and "lidar" in args.methods:
        print("building reference map from the clear dataset (ground-truth poses)...")
        reference_map = reference_map_from_ground_truth(
            DatasetSource(args.datasets / "clear", load_stereo=False,
                          max_frames=args.max_frames))
        print(f"  reference map: {len(reference_map)} points")

    loops = not args.no_loop_closure
    degrade_opts = [False] if args.no_degraded else [False, True]

    rows, results, skipped = [], {}, []
    for cond in available:
        for method in args.methods:
            for degrade in degrade_opts:
                # Degradation only applies to the LiDAR; a second identical
                # stereo run would just be wasted time.
                if degrade and method != "lidar":
                    continue
                tag = f"{cond}/{method}{'+degraded' if degrade else ''}"
                supported, why = dataset_supports(args.datasets / cond, method)
                if not supported:
                    print(f"\n>>> {tag}\n    SKIPPED -- {why}")
                    skipped.append((tag, why))
                    continue
                print(f"\n>>> {tag}")
                try:
                    for rep in range(args.repeats):
                        seed = args.seed + rep
                        row, res = run_one(args.datasets / cond, method, cond, degrade,
                                           loops, args.max_frames, reference_map,
                                           seed, args.stereo_ba)
                        row["seed"] = seed
                        rows.append(row)
                        if rep == 0:
                            results[(cond, method, degrade)] = res
                        print(f"    [seed {seed}] {res.summary()}")
                        print(f"    [seed {seed}] ATE {row['ate_rmse']:.3f} m rmse, "
                              f"RPE {row['rpe_trans_pct_10m']:.2f}%/10 m"
                              + ("   [DIVERGED]" if row.get("diverged") else ""))
                except Exception as exc:
                    print(f"    FAILED: {type(exc).__name__}: {exc}")
                    traceback.print_exc(limit=3)
                    rows.append({"condition": cond, "method": method,
                                 "lidar_degraded": degrade, "error": str(exc)[:200]})

    args.out.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    csv_path = args.out / "metrics.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nmetrics -> {csv_path}")

    with open(args.out / "metrics.json", "w") as fh:
        json.dump(rows, fh, indent=2, default=str)

    ok = [r for r in rows if "error" not in r]
    if ok:
        make_figures(ok, args.out)
        plot_trajectories(results, args.out)

    print("\n" + "=" * 88)
    print(f"{'condition':<12} {'method':<8} {'degr':<5} {'ATE rmse':>9} "
          f"{'RPE %/10m':>10} {'fail %':>7} {'map rmse':>9}")
    print("=" * 88)
    for r in ok:
        print(f"{r['condition']:<12} {r['method']:<8} "
              f"{str(bool(r['lidar_degraded'])):<5} {r.get('ate_rmse', float('nan')):9.3f} "
              f"{r.get('rpe_trans_pct_10m', float('nan')):10.2f} "
              f"{100 * r.get('stat_tracking_failure_rate', 0):7.1f} "
              f"{r.get('map_rmse', float('nan')):9.3f}"
              + ("  DIVERGED" if r.get("diverged") else ""))
    print("=" * 88)
    if skipped:
        print("\nskipped cells:")
        for tag, why in skipped:
            print(f"  {tag}: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
