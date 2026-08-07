#!/usr/bin/env python3
"""
Open an interactive 3D Open3D window over a point-cloud map produced by
flight/square_capture_map.py -- reads only the saved files
(datasets_square/<timestamp>/map.pcd + lidar_poses.txt), no AirSim
connection needed.

Standard Open3D orbit viewer (draw_geometries), not the live Filament/GUI
split-window used by flight/lidar_viz.py or flight/slam_live.py -- this is a
static map, so the simple interactive viewer is all it needs:
    left-drag   orbit
    right-drag / scroll   pan / zoom
    'h'         Open3D's own keybinding help overlay

Points are colored by height (turbo colormap, same as
square_capture_map.py's map_topdown.png, so the two views read consistently)
-- red/dark = ground, blue/white = high. The recorded flight path is drawn on
top as a bright-red line so the map reads in context.

Run:
    ./airsim_venv/bin/python tools/view_square_map.py                 # latest run
    ./airsim_venv/bin/python tools/view_square_map.py --dir datasets_square/20260807_210222
    ./airsim_venv/bin/python tools/view_square_map.py --no-trajectory --point-size 3
    ./airsim_venv/bin/python tools/view_square_map.py --densified      # tools/densify_map.py's output
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Force GLFW onto XWayland -- must be set before open3d is imported (same
# gotcha as flight/lidar_viz.py / flight/slam_live.py).
os.environ.setdefault("DISPLAY", ":1")
os.environ["XDG_SESSION_TYPE"] = "x11"

import matplotlib.cm as cm  # noqa: E402
import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slam.config import REPO_ROOT  # noqa: E402
from slam.geometry import read_tum  # noqa: E402

OUT_ROOT = REPO_ROOT / "datasets_square"
TRAJECTORY_COLOR = (1.0, 0.15, 0.15)


def latest_run_dir() -> Path:
    runs = sorted(p for p in OUT_ROOT.iterdir() if p.is_dir()) if OUT_ROOT.exists() else []
    if not runs:
        raise SystemExit(f"no runs found under {OUT_ROOT} -- run "
                          f"flight/square_capture_map.py first")
    return runs[-1]


def colorize_by_height(points: np.ndarray) -> np.ndarray:
    """turbo colormap over NED height (up = -z), matching map_topdown.png."""
    up = -points[:, 2]
    lo, hi = float(up.min()), float(up.max())
    t = (up - lo) / ((hi - lo) or 1e-6)
    return cm.turbo(t)[:, :3]


def load_map(map_path: Path, voxel: float | None) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(map_path))
    if voxel:
        pcd = pcd.voxel_down_sample(voxel)
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        raise SystemExit(f"{map_path} has no points")
    pcd.colors = o3d.utility.Vector3dVector(colorize_by_height(pts))
    return pcd


def load_trajectory(poses_path: Path) -> o3d.geometry.LineSet | None:
    if not poses_path.exists():
        return None
    _, poses = read_tum(poses_path)
    if len(poses) < 2:
        return None
    positions = poses[:, :3, 3]
    lines = [[i, i + 1] for i in range(len(positions) - 1)]
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(positions),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ls.colors = o3d.utility.Vector3dVector([TRAJECTORY_COLOR] * len(lines))
    return ls


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, default=None,
                   help="datasets_square/<timestamp> run directory (default: latest)")
    p.add_argument("--voxel", type=float, default=None,
                   help="extra voxel downsample before display (map.pcd is already "
                        "downsampled at capture time; use this only if the window is slow)")
    p.add_argument("--point-size", type=float, default=2.0)
    p.add_argument("--densified", action="store_true",
                   help="load map_densified.pcd (tools/densify_map.py's output) "
                        "instead of map.pcd")
    p.add_argument("--no-trajectory", action="store_true",
                   help="don't overlay the recorded flight path")
    p.add_argument("--no-frame", action="store_true",
                   help="don't draw the origin coordinate frame")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.dir or latest_run_dir()
    map_path = run_dir / ("map_densified.pcd" if args.densified else "map.pcd")
    if not map_path.exists():
        hint = (" -- run tools/densify_map.py first" if args.densified else "")
        raise SystemExit(f"{map_path} not found{hint}")

    print(f"Loading {map_path} ...")
    pcd = load_map(map_path, args.voxel)
    print(f"  {len(pcd.points)} points")

    geometries = [pcd]

    if not args.no_trajectory:
        traj = load_trajectory(run_dir / "lidar_poses.txt")
        if traj is not None:
            geometries.append(traj)
            print(f"  + flight path ({len(traj.lines)} segments)")

    if not args.no_frame:
        span = float(np.linalg.norm(np.ptp(np.asarray(pcd.points), axis=0)))
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=max(2.0, span * 0.05))
        geometries.append(frame)

    print("Opening interactive viewer (left-drag orbit, scroll zoom, 'h' for help)...")
    # The draw_geometries() convenience wrapper has no point-size knob, so use
    # the explicit Visualizer to actually apply --point-size / background.
    vis = o3d.visualization.Visualizer()
    label = f"{run_dir.name}{' (densified)' if args.densified else ''}"
    vis.create_window(window_name=f"Square-circuit LiDAR map -- {label}",
                      width=1280, height=800)
    for g in geometries:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.point_size = args.point_size
    opt.background_color = np.array([0.05, 0.05, 0.05])
    vis.run()
    vis.destroy_window()
    return 0


if __name__ == "__main__":
    sys.exit(main())
