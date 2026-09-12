#!/usr/bin/env python3
"""
Step 3: spawn a car on a road and drive it through the environment using only
the panoptic map -- staying on "road" voxels and clear of every "thing".

The sim is in Multirotor mode, so the car is not a physics vehicle: it is a
static car mesh spawned with simSpawnObject and moved every tick with
simSetObjectPose along a route planned on the map (planner.RoadGrid: A* over
road cells eroded by the car's half width, obstacles dilated by the same).
Every pose is re-checked against the map's road/obstacle grids and the run
reports how many ticks were off-road or in collision (expected: 0).

    ./airsim_venv/bin/python panoptic/drive_car.py                    # latest map, auto route
    ./airsim_venv/bin/python panoptic/drive_car.py --start 10 -40 --goal 120 90
    ./airsim_venv/bin/python panoptic/drive_car.py --dry-run          # plan + Open3D only, no sim
    ./airsim_venv/bin/python panoptic/drive_car.py --chase            # + CarCam window (cv2)
    ./airsim_venv/bin/python panoptic/drive_car.py --asset /Game/.../Car_03  --yaw-offset 90

The Open3D window shows the class-coloured map with the planned route (white)
and the car (yellow box) moving along it; --chase adds the sim's external
"CarCam" trailing the car. Needs the sim booted with PROFILE=panoptic (the
CarCam lives in settings.panoptic.json) -- the drone just sits parked.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("DISPLAY", ":1")           # GLFW onto XWayland, before open3d import
os.environ["XDG_SESSION_TYPE"] = "x11"

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg  # noqa: E402
from fuse import class_colors, latest_dataset  # noqa: E402
from planner import PurePursuit, RoadGrid  # noqa: E402

CAR_NAME = "PanopticCar"
ASSET_KEYWORDS = ["car", "sedan", "suv", "vehicle"]


def ned_to_display(p: np.ndarray) -> np.ndarray:
    """NED -> (east, north, up) so Open3D's default view is a normal map."""
    p = np.asarray(p, dtype=np.float64)
    return np.stack([p[..., 1], p[..., 0], -p[..., 2]], axis=-1)


class MapView:
    def __init__(self, m: dict, route: np.ndarray, grid: RoadGrid):
        import open3d as o3d
        self.o3d = o3d
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("Panoptic map - car drive", 1280, 800)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(ned_to_display(m["xyz"]))
        pc.colors = o3d.utility.Vector3dVector(class_colors(m["class_id"]))
        self.vis.add_geometry(pc)
        z = np.array([grid.z_at(x, y) - 0.3 for x, y in route])
        pts = ned_to_display(np.column_stack([route, z]))
        ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(pts),
                                  o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(pts) - 1)]))
        ls.paint_uniform_color([1, 1, 1])
        self.vis.add_geometry(ls)
        self.car = o3d.geometry.TriangleMesh.create_box(cfg.CAR_LENGTH, cfg.CAR_WIDTH, 1.5)
        self.car.translate([-cfg.CAR_LENGTH / 2, -cfg.CAR_WIDTH / 2, -1.5])  # centre, sits on z=0 (NED up = -z)
        self._template = np.asarray(self.car.vertices).copy()
        self.car.paint_uniform_color([1.0, 0.85, 0.1])
        self.vis.add_geometry(self.car)
        opt = self.vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.array([0.05, 0.05, 0.05])
        opt.line_width = 4.0

    def update(self, x, y, z, yaw) -> bool:
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        v = self._template @ R.T + np.array([x, y, z])
        self.car.vertices = self.o3d.utility.Vector3dVector(ned_to_display(v))
        self.vis.update_geometry(self.car)
        alive = self.vis.poll_events()
        self.vis.update_renderer()
        return alive

    def close(self):
        self.vis.destroy_window()


def pick_asset(client, forced: str | None) -> str:
    if forced:
        return forced
    assets = client.simListAssets()
    cands = sorted({a for a in assets if any(k in a.lower() for k in ASSET_KEYWORDS)})
    if not cands:
        sys.exit("no car-like asset in this level (simListAssets); pass --asset")
    print(f"  car asset: {cands[0]}" + (f"  ({len(cands) - 1} other candidates, --asset to pick)" if len(cands) > 1 else ""))
    return cands[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=None)
    ap.add_argument("--start", type=float, nargs=2, metavar=("X", "Y"), help="world NED")
    ap.add_argument("--goal", type=float, nargs=2, metavar=("X", "Y"))
    ap.add_argument("--speed", type=float, default=cfg.CAR_SPEED)
    ap.add_argument("--asset", default=None)
    ap.add_argument("--yaw-offset", type=float, default=0.0, help="deg, if the mesh's nose is not +X")
    ap.add_argument("--z-offset", type=float, default=0.0, help="m, if the mesh pivot is not at its base")
    ap.add_argument("--realtime", type=float, default=1.0, help="playback speed factor")
    ap.add_argument("--chase", action="store_true", help="show the CarCam feed (cv2 window)")
    ap.add_argument("--no-viz", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="no sim: plan and animate on the map only")
    args = ap.parse_args()

    ds = args.dataset or latest_dataset()
    mp = ds / "panoptic_map.npz"
    if not mp.exists():
        sys.exit(f"{mp} missing; run panoptic/fuse.py first")
    m = dict(np.load(mp))
    print(f"map {mp}: {len(m['xyz'])} voxels")
    grid = RoadGrid(m)
    print(f"  road cells {int(grid.road.sum())}, drivable {int(grid.drivable.sum())}, "
          f"obstacle cells {int(grid.obstacle.sum())}")
    if not grid.drivable.any():
        sys.exit("no drivable road in the map (is 'road' matched by config.CLASS_KEYWORDS?)")

    if args.start and args.goal:
        s_xy, g_xy = tuple(args.start), tuple(args.goal)
    else:
        a, b = grid.farthest_pair()
        s_xy, g_xy = grid.to_xy(*a), grid.to_xy(*b)
        print(f"  auto route: ({s_xy[0]:.0f},{s_xy[1]:.0f}) -> ({g_xy[0]:.0f},{g_xy[1]:.0f})")
    route = grid.route(s_xy, g_xy)
    length = float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum())
    print(f"  route: {len(route)} points, {length:.0f} m, ~{length / args.speed:.0f} s")

    client = None
    if not args.dry_run:
        import airsim
        client = airsim.MultirotorClient()
        client.confirmConnection()
        asset = pick_asset(client, args.asset)
        # UE4 FATALS (segfault, whole sim gone) on spawning a name that already
        # exists, and simListSceneObjects did not reliably show the previous
        # run's car -- so destroy unconditionally AND use a per-run name.
        for old in client.simListSceneObjects(f"{CAR_NAME}.*"):
            client.simDestroyObject(old)
        car_name = f"{CAR_NAME}_{int(time.time())}"
        x0, y0 = route[0]
        z0 = grid.z_at(x0, y0) - grid.cell / 2 + args.z_offset
        pose = airsim.Pose(airsim.Vector3r(x0, y0, z0), airsim.to_quaternion(0, 0, 0))
        name = client.simSpawnObject(car_name, asset, pose, airsim.Vector3r(1, 1, 1), physics_enabled=False)
        print(f"  spawned {name} at ({x0:.1f}, {y0:.1f}, {z0:.1f})")

    if client is None:
        car_name = None
    view = None if args.no_viz else MapView(m, route, grid)
    pp = PurePursuit(route, speed=args.speed)
    off_road = collisions = ticks = 0
    t_wall = time.time()
    try:
        while not pp.done:
            x, y, yaw = pp.step()
            ticks += 1
            on_road, free = grid.check_pose(x, y, yaw)
            off_road += not on_road
            collisions += not free
            z = grid.z_at(x, y) - grid.cell / 2
            if client is not None:
                import airsim
                q = airsim.to_quaternion(0, 0, yaw + math.radians(args.yaw_offset))
                client.simSetObjectPose(car_name, airsim.Pose(airsim.Vector3r(x, y, z + args.z_offset), q), True)
                if args.chase:
                    _chase(client, x, y, z, yaw)
            if view is not None and not view.update(x, y, z, yaw):
                print("\n  viewer closed")
                break
            if ticks % 20 == 0:
                print(f"\r  t={ticks * cfg.CAR_DT:6.1f} s  pos ({x:7.1f},{y:7.1f})  "
                      f"off-road {off_road}  collision {collisions}", end="", flush=True)
            target = t_wall + ticks * cfg.CAR_DT / args.realtime
            time.sleep(max(0.0, target - time.time()))
    finally:
        if view is not None:
            view.close()
    print(f"\ndone: {ticks} ticks, {ticks * cfg.CAR_DT:.0f} s of driving; "
          f"off-road ticks {off_road}, collision ticks {collisions}"
          + ("  -- OK" if off_road == 0 and collisions == 0 else "  -- VIOLATIONS"))


def _chase(client, x, y, z, yaw):
    import airsim
    import cv2
    back = 9.0
    cx, cy, cz = x - back * math.cos(yaw), y - back * math.sin(yaw), z - 3.5
    client.simSetCameraPose(cfg.CAR_CAM, airsim.Pose(airsim.Vector3r(cx, cy, cz),
                                                     airsim.to_quaternion(math.radians(-15), 0, yaw)),
                            external=True)
    resp = client.simGetImages([airsim.ImageRequest(cfg.CAR_CAM, airsim.ImageType.Scene, False, False)],
                               external=True)[0]
    if resp.image_data_uint8:
        from labels import _rgb
        cv2.imshow("CarCam", np.ascontiguousarray(_rgb(resp)))
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
