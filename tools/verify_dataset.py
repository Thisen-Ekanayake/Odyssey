#!/usr/bin/env python3
"""
Sanity-check a recorded dataset before spending hours running SLAM on it.

Catches the failure modes that are cheap to detect now and expensive to
discover halfway through a benchmark: dropped frames, non-monotonic or
duplicated timestamps, an IMU that disagrees with the ground-truth attitude,
a route that never closed its loop, and stereo pairs with no parallax.

It also reports the cross-condition comparison the study depends on: whether
the camera stream genuinely differs between weather conditions (it should) and
whether the LiDAR does (it should NOT -- that gap is what slam/degradation.py
is built to fill).

Usage:
    ./airsim_venv/bin/python tools/verify_dataset.py datasets/clear
    ./airsim_venv/bin/python tools/verify_dataset.py --all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy.spatial.transform import Rotation  # noqa: E402

from slam import config as cfg  # noqa: E402
from slam.source import DatasetSource  # noqa: E402


def _check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  --  {detail}" if detail else ""))
    return ok


def verify(path: Path) -> tuple[int, dict]:
    print(f"\n{'=' * 72}\n{path}\n{'=' * 72}")
    src = DatasetSource(path, load_stereo=True, load_lidar=True)
    fails = 0

    n_lidar, n_cam = len(src._lidar_ns), len(src._cam_ns)
    print(f"  condition={src.condition}  frames={len(src)}  "
          f"lidar={n_lidar}  stereo={n_cam}  imu={len(src._imu_t)}")

    fails += not _check("has LiDAR scans", n_lidar > 50, f"{n_lidar}")
    fails += not _check("has IMU samples", len(src._imu_t) > 500, f"{len(src._imu_t)}")
    fails += not _check("has ground truth", len(src._gt_t) > 500, f"{len(src._gt_t)}")

    # -- timestamps --------------------------------------------------------
    if len(src._imu_t) > 1:
        d = np.diff(src._imu_t)
        fails += not _check("IMU timestamps strictly increasing", bool((d > 0).all()),
                            f"{int((d <= 0).sum())} non-increasing steps")
        want = 1.0 / cfg.IMU_RATE_HZ
        fails += not _check(
            "IMU rate near nominal",
            abs(np.median(d) - want) < 0.5 * want,
            f"median dt {np.median(d) * 1000:.2f} ms (nominal {want * 1000:.1f} ms), "
            f"max gap {d.max() * 1000:.1f} ms")

    if n_lidar > 1:
        lt = np.array(src._lidar_ns) / 1e9
        d = np.diff(lt)
        fails += not _check("no duplicate LiDAR stamps", len(set(src._lidar_ns)) == n_lidar,
                            f"{n_lidar - len(set(src._lidar_ns))} duplicates")
        # A gap far above nominal means capture stalled and the map will have a hole.
        want = cfg.SENSOR_DECIMATION / cfg.IMU_RATE_HZ
        fails += not _check("no large LiDAR gaps", float(d.max()) < 10 * want,
                            f"max gap {d.max() * 1000:.0f} ms (nominal {want * 1000:.0f} ms)")

    # -- trajectory --------------------------------------------------------
    gt_t, gt_T = src.ground_truth()
    stats: dict = {}
    if len(gt_T) > 10:
        xyz = gt_T[:, :3, 3]
        length = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
        closure = float(np.linalg.norm(xyz[-1, :2] - xyz[0, :2]))
        stats["length_m"] = length
        stats["closure_m"] = closure

        expected = 2 * (cfg.ROUTE_LENGTH_X + cfg.ROUTE_LENGTH_Y) * cfg.ROUTE_LAPS
        fails += not _check("flew most of the planned route", length > 0.7 * expected,
                            f"{length:.0f} m flown vs {expected:.0f} m planned")
        # Without a revisit there is nothing for loop closure to close, and the
        # whole back end goes untested.
        fails += not _check("route closes the loop", closure < 15.0,
                            f"end is {closure:.1f} m from start")

        alt = xyz[:, 2]
        fails += not _check("held the survey altitude",
                            float(np.std(alt)) < 3.0,
                            f"z mean {alt.mean():.1f} m, sd {alt.std():.2f} m "
                            f"(target {cfg.ROUTE_ALTITUDE:.1f})")

    # -- IMU vs ground-truth attitude --------------------------------------
    # Integrating gyro over a short window must track the GT attitude change.
    # If the two disagree, the IMU axes or the frame convention are wrong, and
    # every motion prior downstream would be quietly broken.
    if len(gt_T) > 200 and len(src._imu_t) > 200:
        errs = []
        for start in np.linspace(50, len(src._imu_t) - 150, 8).astype(int):
            i0, i1 = start, start + 100          # ~1 s at 100 Hz
            t0, t1 = src._imu_t[i0], src._imu_t[i1]
            j0 = int(np.argmin(np.abs(gt_t - t0)))
            j1 = int(np.argmin(np.abs(gt_t - t1)))
            if j1 <= j0:
                continue
            R_gt = gt_T[j0][:3, :3].T @ gt_T[j1][:3, :3]

            R_imu = np.eye(3)
            for k in range(i0, i1):
                dt = src._imu_t[k + 1] - src._imu_t[k]
                R_imu = R_imu @ Rotation.from_rotvec(src._imu_gyro[k] * dt).as_matrix()

            errs.append(np.degrees(
                np.linalg.norm(Rotation.from_matrix(R_gt.T @ R_imu).as_rotvec())))
        if errs:
            med = float(np.median(errs))
            stats["imu_attitude_err_deg"] = med
            fails += not _check("gyro integration tracks GT attitude (1 s windows)",
                                med < 5.0, f"median error {med:.2f} deg over 8 windows")

    # -- stereo ------------------------------------------------------------
    if n_cam:
        import cv2
        frame = next((f for f in src if f.stereo is not None), None)
        if frame is None:
            fails += not _check("stereo frames decode", False, "none decoded")
        else:
            L, R = frame.stereo.left, frame.stereo.right
            ok = L.shape == (cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, 3)
            fails += not _check("stereo resolution", ok, f"{L.shape}")
            diff = float(np.abs(L.astype(np.int16) - R.astype(np.int16)).mean())
            fails += not _check("stereo pair has parallax", diff > 1.0,
                                f"mean |L-R| = {diff:.2f}")
            g = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY)
            stats["image_mean"] = float(g.mean())
            stats["image_contrast"] = float(g.std())
            print(f"        left image: mean {g.mean():.1f}, contrast (sd) {g.std():.1f}")

    # -- lidar content -----------------------------------------------------
    if n_lidar:
        counts, ranges = [], []
        for ns in src._lidar_ns[::max(1, n_lidar // 40)]:
            p = np.load(path / "lidar" / f"{ns}.npy")
            counts.append(len(p))
            if len(p):
                ranges.append(float(np.linalg.norm(p, axis=1).max()))
        stats["lidar_pts_mean"] = float(np.mean(counts))
        stats["lidar_max_range"] = float(np.max(ranges)) if ranges else 0.0
        fails += not _check("LiDAR scans are populated", np.mean(counts) > 1000,
                            f"mean {np.mean(counts):.0f} pts/scan")
        fails += not _check("LiDAR points within configured Range",
                            stats["lidar_max_range"] <= cfg.LIDAR_RANGE * 1.05,
                            f"max {stats['lidar_max_range']:.1f} m "
                            f"(Range={cfg.LIDAR_RANGE:.0f} m)")

    print(f"\n  -> {'OK' if fails == 0 else f'{fails} CHECK(S) FAILED'}")
    return fails, stats


def compare(results: dict[str, dict]) -> None:
    """Cross-condition summary: cameras should differ, LiDAR should not."""
    if len(results) < 2 or "clear" not in results:
        return

    print(f"\n{'=' * 72}\nCROSS-CONDITION COMPARISON (vs. clear)\n{'=' * 72}")
    base = results["clear"]
    print(f"  {'condition':<12} {'img mean':>9} {'contrast':>9} "
          f"{'lidar pts':>10} {'max range':>10}")
    for name in cfg.BENCHMARK_ORDER:
        s = results.get(name)
        if not s:
            continue
        print(f"  {name:<12} {s.get('image_mean', 0):9.1f} {s.get('image_contrast', 0):9.1f} "
              f"{s.get('lidar_pts_mean', 0):10.0f} {s.get('lidar_max_range', 0):10.1f}")

    cam_changed = any(
        abs(results[n].get("image_contrast", 0) - base.get("image_contrast", 0)) > 1.0
        for n in results if n != "clear"
    )
    lidar_changed = any(
        abs(results[n].get("lidar_pts_mean", 0) - base.get("lidar_pts_mean", 0))
        > 0.05 * max(base.get("lidar_pts_mean", 1), 1)
        for n in results if n != "clear"
    )

    print()
    _check("cameras are degraded by weather (expected: yes)", cam_changed,
           "image contrast varies across conditions")
    _check("LiDAR is NOT degraded by weather (expected: yes, unchanged)",
           not lidar_changed,
           "point counts are flat across conditions -- this is the simulator "
           "limitation slam/degradation.py models")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", type=Path, help="dataset directories")
    ap.add_argument("--all", action="store_true",
                    help=f"verify every condition under {cfg.DATASET_ROOT}")
    args = ap.parse_args()

    paths = list(args.paths)
    if args.all or not paths:
        paths = [cfg.DATASET_ROOT / n for n in cfg.BENCHMARK_ORDER
                 if (cfg.DATASET_ROOT / n).is_dir()]
    if not paths:
        print(f"no datasets found under {cfg.DATASET_ROOT}; "
              f"run flight/record_dataset.py first", file=sys.stderr)
        return 1

    total, results = 0, {}
    for p in paths:
        try:
            fails, stats = verify(p)
            total += fails
            results[p.name] = stats
        except Exception as exc:
            print(f"  ERROR reading {p}: {type(exc).__name__}: {exc}")
            total += 1

    compare(results)
    print(f"\n{'All datasets OK.' if total == 0 else f'{total} check(s) failed.'}\n")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
