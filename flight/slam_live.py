#!/usr/bin/env python3
"""
Live SLAM against the running simulator, with an Open3D viewer.

Three panels: the drone's camera feed, the map SLAM is building from its OWN
estimated poses, and the estimated trajectory overlaid on ground truth.

The contrast with flight/lidar_viz.py is the point of the demo. That script
places every scan using the simulator's reported sensor pose -- perfect
localisation, so it shows the best map the sensor could ever produce. This one
places scans using poses it estimates itself, so the two side by side show
exactly what SLAM has to recover.

Ground truth is drawn for comparison only. It is never fed to the estimator.

Usage (after ./scripts/run_swarm.sh AirSimNH is up and the API is answering):
    ./airsim_venv/bin/python flight/slam_live.py
    ./airsim_venv/bin/python flight/slam_live.py --method stereo
    ./airsim_venv/bin/python flight/slam_live.py --weather fog_heavy

Close the window to land and exit. Weather is always reset on the way out.
"""
from __future__ import annotations

import os
# Force GLFW onto XWayland — must be set before open3d is imported.
os.environ.setdefault("DISPLAY", ":1")
os.environ["XDG_SESSION_TYPE"] = "x11"

import argparse  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402
import open3d.visualization.gui as gui  # type: ignore # noqa: E402
import open3d.visualization.rendering as rendering  # type: ignore # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import airsim  # noqa: E402

from slam import config as cfg  # noqa: E402
from slam import weather as weather_mod  # noqa: E402
from slam.lidar_slam import LidarInertialSLAM  # noqa: E402
from slam.source import LiveSource  # noqa: E402
from slam.stereo_slam import StereoInertialSLAM  # noqa: E402

RATE_HZ = 10.0
MAP_REFRESH_EVERY = 5        # SLAM frames between map redraws

# Rate for the display-only camera panel when SLAM itself doesn't already
# consume images (--method lidar). Deliberately its own thread/connection,
# polling far slower than RATE_HZ: simGetImages over the RPC link is much
# slower than getLidarData/getImuData, and fetching a 1280x720 frame inside
# the LiDAR/IMU polling loop stalls it long enough that the map visibly stops
# growing while the drone keeps flying. Decoupling it fixes that.
CAMERA_DISPLAY_HZ = 3.0

# Display-only downsample. The SLAM map itself stays at cfg.MAP_VOXEL_SIZE (0.4 m)
# because registration and the map metrics want that resolution -- but drawing it
# raw over a whole circuit is a solid wall of points with no visible structure.
MAP_DISPLAY_VOXEL = 0.9      # m
MAP_MAX_POINTS = 400_000     # hard cap; random-thinned above this

# Height ramp, low z (= HIGH altitude, NED z is down) -> high z (= ground).
# Multi-hue on purpose: the old two-stop blue<->red ramp passed through black in
# the middle, so mid-height structure vanished into the dark background.
HEIGHT_RAMP = np.array([
    [0.20, 0.25, 0.95],   # blue    -- highest
    [0.00, 0.80, 0.95],   # cyan
    [0.20, 0.90, 0.35],   # green
    [1.00, 0.85, 0.10],   # yellow
    [1.00, 0.25, 0.10],   # red     -- ground
])


def height_range(points: np.ndarray) -> tuple[float, float]:
    """Robust z extent: 2nd/98th percentile, so outliers don't eat the ramp."""
    lo, hi = np.percentile(points[:, 2], [2.0, 98.0])
    return float(lo), float(hi)


def colorize_by_height(points: np.ndarray,
                       z_range: tuple[float, float] | None = None) -> np.ndarray:
    """Blue (high) -> cyan -> green -> yellow -> red (ground).

    Same convention as flight/lidar_viz.py so the two viewers are comparable.
    NED means z is DOWN, hence the inverted mapping.
    """
    if len(points) == 0:
        return np.zeros((0, 3))
    lo, hi = height_range(points) if z_range is None else z_range
    if hi - lo < 1e-3:
        hi = lo + 1e-3
    t = np.clip((points[:, 2] - lo) / (hi - lo), 0.0, 1.0)

    f = t * (len(HEIGHT_RAMP) - 1)
    i = np.clip(f.astype(int), 0, len(HEIGHT_RAMP) - 2)
    w = (f - i)[:, None]
    return HEIGHT_RAMP[i] * (1.0 - w) + HEIGHT_RAMP[i + 1] * w


class SlamViewer:
    def __init__(self, method: str, condition_name: str,
                 display_voxel: float = MAP_DISPLAY_VOXEL,
                 point_size: float = 2.0) -> None:
        self.method = method
        self.condition_name = condition_name
        self.display_voxel = display_voxel
        self._running = True
        self._lock = threading.Lock()
        # Latched colour scale. Recomputing min/max per redraw made the whole
        # map change colour every time a new extreme appeared; this only ever
        # widens, so it settles once the environment's z extent is covered.
        self._z_range: tuple[float, float] | None = None

        app = gui.Application.instance
        app.initialize()
        self.win = app.create_window(
            f"AirSim SLAM — {method}-inertial ({condition_name})", 1700, 760)

        self.image_widget = gui.ImageWidget(o3d.geometry.Image(
            np.zeros((cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, 3), dtype=np.uint8)))
        self.win.add_child(self.image_widget)

        self.map_widget = self._make_scene([-40, -40, -30], [180, 140, 10])
        self.traj_widget = self._make_scene([-40, -40, -30], [180, 140, 10])
        self.win.add_child(self.map_widget)
        self.win.add_child(self.traj_widget)

        self.pt_mat = rendering.MaterialRecord()
        self.pt_mat.shader = "defaultUnlit"
        self.pt_mat.point_size = point_size * self.win.scaling

        self.line_mat = rendering.MaterialRecord()
        self.line_mat.shader = "unlitLine"
        self.line_mat.line_width = 2.5 * self.win.scaling

        self.map_widget.scene.add_geometry("map", o3d.geometry.PointCloud(), self.pt_mat)
        for name in ("est", "gt"):
            self.traj_widget.scene.add_geometry(
                name, o3d.geometry.LineSet(), self.line_mat)

        self.win.set_on_layout(self._on_layout)
        self.win.set_on_close(self._on_close)

    def _make_scene(self, lo, hi):
        w = gui.SceneWidget()
        w.scene = rendering.Open3DScene(self.win.renderer)
        w.scene.set_background([0.05, 0.05, 0.05, 1.0])
        bounds = o3d.geometry.AxisAlignedBoundingBox(lo, hi)
        w.setup_camera(60.0, bounds, bounds.get_center())
        frame_mat = rendering.MaterialRecord()
        frame_mat.shader = "defaultLit"
        w.scene.add_geometry("origin",
                             o3d.geometry.TriangleMesh.create_coordinate_frame(size=4.0),
                             frame_mat)
        return w

    def _on_layout(self, _):
        r = self.win.content_rect
        third = r.width // 3
        self.image_widget.frame = gui.Rect(r.x, r.y, third, r.height)
        self.map_widget.frame = gui.Rect(r.x + third, r.y, third, r.height)
        self.traj_widget.frame = gui.Rect(r.x + 2 * third, r.y,
                                          r.width - 2 * third, r.height)

    def _on_close(self):
        self._running = False
        return True

    @property
    def running(self) -> bool:
        return self._running

    # -- updates (posted to the GUI thread) --------------------------------

    def update_image(self, bgr: np.ndarray) -> None:
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        img = o3d.geometry.Image(rgb)
        gui.Application.instance.post_to_main_thread(
            self.win, lambda: self.image_widget.update_image(img))

    def update_map(self, points: np.ndarray) -> None:
        if len(points) == 0:
            return
        pts = np.asarray(points, dtype=np.float64)

        # Colour scale comes from the FULL map (before thinning) and only widens.
        lo, hi = height_range(pts)
        if self._z_range is not None:
            lo, hi = min(lo, self._z_range[0]), max(hi, self._z_range[1])
        self._z_range = (lo, hi)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        if self.display_voxel > 0:
            pcd = pcd.voxel_down_sample(self.display_voxel)
            pts = np.asarray(pcd.points)
        if len(pts) > MAP_MAX_POINTS:
            keep = np.random.default_rng(0).choice(len(pts), MAP_MAX_POINTS, replace=False)
            pts = pts[keep]
            pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(colorize_by_height(pts, self._z_range))

        def _apply():
            self.map_widget.scene.remove_geometry("map")
            self.map_widget.scene.add_geometry("map", pcd, self.pt_mat)
        gui.Application.instance.post_to_main_thread(self.win, _apply)

    def update_trajectories(self, est: np.ndarray, gt: np.ndarray) -> None:
        def line(pts, colour):
            ls = o3d.geometry.LineSet()
            if len(pts) < 2:
                return ls
            ls.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
            idx = np.stack([np.arange(len(pts) - 1), np.arange(1, len(pts))], axis=1)
            ls.lines = o3d.utility.Vector2iVector(idx)
            ls.colors = o3d.utility.Vector3dVector(np.tile(colour, (len(idx), 1)))
            return ls

        est_ls = line(est, [1.0, 0.35, 0.0])     # orange: estimated
        gt_ls = line(gt, [0.2, 0.9, 0.3])        # green: ground truth

        def _apply():
            self.traj_widget.scene.remove_geometry("est")
            self.traj_widget.scene.remove_geometry("gt")
            self.traj_widget.scene.add_geometry("gt", gt_ls, self.line_mat)
            self.traj_widget.scene.add_geometry("est", est_ls, self.line_mat)
        gui.Application.instance.post_to_main_thread(self.win, _apply)

    def run(self):
        gui.Application.instance.run()


def slam_thread(viewer: SlamViewer, method: str) -> None:
    """Owns its own AirSim connection: msgpack-rpc's IOLoop is not thread-safe."""
    client = airsim.MultirotorClient()
    client.confirmConnection()

    source = LiveSource(client, rate_hz=RATE_HZ,
                        use_lidar=(method == "lidar"),
                        use_stereo=(method == "stereo"))
    slam = (LidarInertialSLAM(verbose=False) if method == "lidar"
            else StereoInertialSLAM(verbose=False))

    est_path: list[np.ndarray] = []
    gt_path: list[np.ndarray] = []
    n = 0

    for frame in source:
        if not viewer.running:
            source.stop()
            break

        T = slam.process(frame)
        est_path.append(T[:3, 3].copy())
        if frame.gt_pose is not None:
            gt_path.append(frame.gt_pose[:3, 3].copy())

        if frame.stereo is not None:
            viewer.update_image(frame.stereo.left)

        n += 1
        if n % MAP_REFRESH_EVERY == 0:
            pts = (slam.map.points if method == "lidar" else slam._sparse_map())
            viewer.update_map(pts)
            viewer.update_trajectories(np.array(est_path),
                                       np.array(gt_path) if gt_path else np.empty((0, 3)))

            err = (np.linalg.norm(est_path[-1] - gt_path[-1])
                   if gt_path else float("nan"))
            kf = len(slam.backend.poses)
            print(f"\r  frames {n:5d}  keyframes {kf:4d}  "
                  f"loops {len(slam.backend.loops):3d}  "
                  f"fails {slam.n_failures:3d}  |est-gt| {err:6.2f} m",
                  end="", flush=True)


def camera_thread(viewer: SlamViewer, method: str) -> None:
    """Own connection: msgpack-rpc's IOLoop is not thread-safe, and this must
    never share the SLAM thread's client -- that would put a slow simGetImages
    call back on the critical path that feeds the map (see slam_thread).

    Only needed for --method lidar: --method stereo already gets its image
    for free from slam_thread's own stereo frames.
    """
    if method != "lidar":
        return
    client = airsim.MultirotorClient()
    client.confirmConnection()
    period = 1.0 / CAMERA_DISPLAY_HZ

    while viewer.running:
        tick = time.time()
        resp = client.simGetImages([
            airsim.ImageRequest(cfg.CAM_LEFT, airsim.ImageType.Scene, False, False),
        ], vehicle_name=cfg.DRONE)
        if resp:
            want = resp[0].width * resp[0].height * 3
            if resp[0].width and resp[0].height and len(resp[0].image_data_uint8) == want:
                img = np.frombuffer(resp[0].image_data_uint8, np.uint8).reshape(
                    resp[0].height, resp[0].width, 3)
                viewer.update_image(img)
        time.sleep(max(0.0, period - (time.time() - tick)))


def fly_route(client: airsim.MultirotorClient, viewer: SlamViewer) -> None:
    """Fly the same circuit the benchmark records, so the demo matches the study."""
    wps = cfg.route_waypoints()
    path = [airsim.Vector3r(*w) for w in wps]
    time.sleep(2.0)
    print(f"\n-> flying the {len(wps)}-waypoint circuit "
          f"({cfg.ROUTE_LAPS} laps, close the window to land)")
    client.moveOnPathAsync(
        path, cfg.ROUTE_SPEED,
        drivetrain=airsim.DrivetrainType.ForwardOnly,
        yaw_mode=airsim.YawMode(False, 0),
        vehicle_name=cfg.DRONE)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="lidar", choices=["lidar", "stereo"])
    ap.add_argument("--weather", default="clear", choices=cfg.BENCHMARK_ORDER)
    ap.add_argument("--map-voxel", type=float, default=MAP_DISPLAY_VOXEL,
                    help="display-only downsample for the map panel, in m "
                         f"(default {MAP_DISPLAY_VOXEL}; 0 = draw every point). "
                         "Does not affect the SLAM map itself.")
    ap.add_argument("--point-size", type=float, default=2.0,
                    help="rendered point size in the map panel (default 2.0)")
    args = ap.parse_args()

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    o3d.utility.random.seed(42)

    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("Connected to AirSim.")

    condition = weather_mod.get(args.weather)
    weather_mod.apply(client, condition)
    print(f"Weather: {condition.name} — {condition.description or 'baseline'}")
    if args.method == "lidar" and condition.airsim_params:
        print("  NOTE: AirSim weather does not affect LiDAR returns. This run shows "
              "the raw simulator behaviour;\n        slam/degradation.py models the "
              "LiDAR degradation in the offline benchmark.")

    client.enableApiControl(True, cfg.DRONE)
    client.armDisarm(True, cfg.DRONE)
    print("Taking off...")
    client.takeoffAsync(vehicle_name=cfg.DRONE).join()
    client.moveToPositionAsync(0, 0, cfg.ROUTE_ALTITUDE, 5.0,
                               vehicle_name=cfg.DRONE).join()
    client.hoverAsync(vehicle_name=cfg.DRONE).join()

    viewer = SlamViewer(args.method, condition.name,
                        display_voxel=args.map_voxel, point_size=args.point_size)
    threading.Thread(target=slam_thread, args=(viewer, args.method), daemon=True).start()
    threading.Thread(target=camera_thread, args=(viewer, args.method), daemon=True).start()
    threading.Thread(target=fly_route, args=(client, viewer), daemon=True).start()

    try:
        viewer.run()
    finally:
        print("\nLanding and clearing weather...")
        try:
            weather_mod.reset(client)
            client.cancelLastTask(vehicle_name=cfg.DRONE)
            client.landAsync(vehicle_name=cfg.DRONE).join()
            client.armDisarm(False, cfg.DRONE)
            client.enableApiControl(False, cfg.DRONE)
        except Exception as exc:
            print(f"  cleanup warning: {exc}")
        print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
