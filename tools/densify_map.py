#!/usr/bin/env python3
"""
Simple K-nearest-neighbor interpolation to fill sparse gaps in a
point-cloud map produced by flight/square_capture_map.py.

This is NOT real surface reconstruction (no normals, no meshing, no
knowledge of the true surface) -- it just: for every point, finds its K
nearest neighbors, and for any neighbor farther away than --gap, inserts the
midpoint between them. That thickens sparse regions (e.g. the vertical gaps
between a 16-channel LiDAR's scan rings) with plausible in-between points.
Good enough to make a map look less "stripey"; not a substitute for more
LiDAR channels or a real reconstruction method.

Reads slam/config.py's REPO_ROOT only (read-only import, no other file
touched). Writes a new map_densified.pcd next to the original map.pcd --
never overwrites the source.

Run:
    ./airsim_venv/bin/python tools/densify_map.py                        # latest run
    ./airsim_venv/bin/python tools/densify_map.py --dir datasets_square/20260807_210222
    ./airsim_venv/bin/python tools/densify_map.py --k 8 --gap 0.8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slam.config import REPO_ROOT  # noqa: E402

OUT_ROOT = REPO_ROOT / "datasets_square"


def latest_run_dir() -> Path:
    runs = sorted(p for p in OUT_ROOT.iterdir() if p.is_dir()) if OUT_ROOT.exists() else []
    if not runs:
        raise SystemExit(f"no runs found under {OUT_ROOT} -- run "
                          f"flight/square_capture_map.py first")
    return runs[-1]


def knn_interpolate(points: np.ndarray, k: int, gap: float, max_gap: float) -> np.ndarray:
    """Midpoint of every (point, neighbor) pair in (gap, max_gap].

    The upper bound matters at the edge of the scanned area: out there the
    k-th nearest neighbor can be far away simply because there's no data
    beyond the boundary, not because of a real gap in the surface. Without
    a cap those pairs get bridged too, showing up as spurious streaks
    radiating outward from the map's edge.
    """
    tree = cKDTree(points)
    dists, idxs = tree.query(points, k=k + 1)   # column 0 is the point itself
    dists, idxs = dists[:, 1:], idxs[:, 1:]

    mask = (dists > gap) & (dists <= max_gap)
    src = np.repeat(np.arange(len(points)), k)[mask.ravel()]
    dst = idxs.ravel()[mask.ravel()]

    return (points[src] + points[dst]) / 2.0


def save_topdown_preview(out_path: Path, points: np.ndarray, added_mask: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    north, east, up = points[:, 0], points[:, 1], -points[:, 2]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(east[added_mask], north[added_mask], c="orangered",
              s=0.8, linewidths=0, alpha=0.6, label="interpolated", zorder=1)
    ax.scatter(east[~added_mask], north[~added_mask], c=up[~added_mask],
              cmap="turbo", s=0.5, linewidths=0, label="original", zorder=2)
    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")
    ax.set_title(f"Densified map -- {added_mask.sum()} interpolated / {len(points)} total")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", markerscale=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Preview written to {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, default=None,
                   help="datasets_square/<timestamp> run directory (default: latest)")
    p.add_argument("--k", type=int, default=6,
                   help="neighbors to consider per point")
    p.add_argument("--gap", type=float, default=0.6,
                   help="only interpolate between points farther apart than this (m)")
    p.add_argument("--max-gap", type=float, default=None,
                   help="and no farther apart than this (default: 5x --gap); "
                        "caps spurious bridging across real edges/boundaries")
    p.add_argument("--voxel", type=float, default=0.3,
                   help="voxel-downsample the merged cloud afterward (0 to skip)")
    p.add_argument("--no-preview", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.dir or latest_run_dir()
    map_path = run_dir / "map.pcd"
    if not map_path.exists():
        raise SystemExit(f"{map_path} not found")

    pcd = o3d.io.read_point_cloud(str(map_path))
    points = np.asarray(pcd.points)
    print(f"Loaded {len(points)} points from {map_path}")

    max_gap = args.max_gap if args.max_gap is not None else args.gap * 5
    print(f"KNN interpolation: k={args.k}, {args.gap} < gap <= {max_gap} m ...")
    new_points = knn_interpolate(points, args.k, args.gap, max_gap)
    print(f"  generated {len(new_points)} interpolated points")

    merged = np.vstack([points, new_points])
    added_mask = np.zeros(len(merged), dtype=bool)
    added_mask[len(points):] = True

    out_pcd = o3d.geometry.PointCloud()
    out_pcd.points = o3d.utility.Vector3dVector(merged)
    if args.voxel > 0:
        # voxel_down_sample doesn't preserve a point-index mapping, so keep
        # the un-downsampled merged array for the preview's original/added
        # split and only downsample what actually gets saved to disk.
        out_pcd = out_pcd.voxel_down_sample(args.voxel)
        print(f"  voxel-downsampled to {len(out_pcd.points)} points")

    out_path = run_dir / "map_densified.pcd"
    o3d.io.write_point_cloud(str(out_path), out_pcd)
    print(f"Saved {out_path} ({len(out_pcd.points)} points, was {len(points)})")

    if not args.no_preview:
        save_topdown_preview(run_dir / "map_densified_topdown.png", merged, added_mask)

    return 0


if __name__ == "__main__":
    sys.exit(main())
