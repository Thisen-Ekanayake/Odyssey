#!/usr/bin/env python3
"""
Single-drone LiDAR mapping visualizer for AirSim Drone1.
Split window: live 3rd-person chase-cam feed (left) + accumulated LiDAR
point-cloud map (right), both GPU-rendered via Open3D's Filament backend.

Points are colored blue (high) -> red (ground) by a fixed NED-Z range and
accumulated into a persistent, voxel-downsampled map (not just the current
scan window).

The chase camera is an AirSim "ExternalCamera" (declared in settings.json as
"ChaseCam") that this script repositions every tick to trail behind the
drone's direction of travel -- not attached to the vehicle body, so its
framing tracks actual velocity rather than the drone's nose.

Flight uses a single moveOnPathAsync call over the whole waypoint pattern:
AirSim fits a pure-pursuit path follower with a velocity-derived lookahead,
so it rounds each corner instead of stopping and reversing direction at every
leg. The earlier chained-moveToPositionAsync approach caused a periodic
tilt/lurch (replanning a fresh trajectory at each waypoint, same root cause
already fixed in swarm_circle.py); a hand-rolled per-waypoint P-controller
fixed the tilt but was still visibly jerky at corners -- moveOnPathAsync's
built-in cornering (same technique as AirSim's own drone_survey example)
fixes that too.

Run after the Blocks/AirSimNH/etc. sim is up (restart required after the
settings.json change that added ChaseCam):
    ./airsim_venv/bin/python flight/lidar_viz.py

Close the window to land and exit.
"""
import os
# Force GLFW onto XWayland — must be set before open3d is imported.
os.environ.setdefault("DISPLAY", ":1")
os.environ["XDG_SESSION_TYPE"] = "x11"

import math
import time
import threading
import numpy as np
import airsim
import open3d as o3d
import open3d.visualization.gui as gui  # type: ignore
import open3d.visualization.rendering as rendering  # type: ignore

DRONE = "Drone1"
ALTITUDE = -8.0   # NED: negative = up (8 m)
UPDATE_HZ = 10     # LiDAR poll rate

# Flight speed for the survey path (m/s)
MAX_SPEED = 3.0

# Flight pattern: (label, dx, dy) relative offsets in NED (forward=+X, right=+Y)
FLIGHT_PATTERN = [
    ("forward  5 m",  ( 5,  0)),
    ("backward 10 m", (-10, 0)),
    ("left     5 m",  ( 0, -5)),
    ("right    5 m",  ( 0, 10)),
]

# Absolute waypoints (world/local NED, spawn = origin) derived from the pattern above
WAYPOINTS = [(0.0, 0.0)]
for _, (_dx, _dy) in FLIGHT_PATTERN:
    px, py = WAYPOINTS[-1]
    WAYPOINTS.append((px + _dx, py + _dy))

# Fixed altitude range used for map coloring, so color stays stable as the map grows
Z_COLOR_RANGE = (-12.0, 0.0)   # NED z: -12 (high) -> 0 (ground)
VOXEL_SIZE = 0.3               # m; keeps the accumulated map's point count bounded

# Chase camera: an AirSim ExternalCamera (see "ChaseCam" in settings.json),
# repositioned every tick to trail behind the drone's direction of travel.
CAM_NAME     = "ChaseCam"
CAM_HZ       = 15      # camera pose + frame poll rate
CHASE_DIST   = 8.0     # m behind the drone (along its direction of travel)
CHASE_HEIGHT = 4.0     # m above the drone (NED: camera z = drone z - CHASE_HEIGHT)
MIN_SPEED_FOR_HEADING = 0.3   # m/s; below this, keep the last heading (avoid spin while hovering)


def colorize_by_height(points: np.ndarray) -> np.ndarray:
    """Blue (high altitude) -> red (ground), using a fixed z range for a stable map."""
    z = points[:, 2]
    zmin, zmax = Z_COLOR_RANGE
    t = np.clip((z - zmin) / (zmax - zmin), 0.0, 1.0)
    return np.stack([t, np.zeros_like(t), 1.0 - t], axis=1)


class LidarViewer:
    def __init__(self):
        self._running = True

        # --- Filament / GUI setup (runs on main thread) ---
        app = gui.Application.instance
        app.initialize()

        self.win = app.create_window(
            "AirSim — Chase Cam + LiDAR Map (GPU / Filament/Vulkan)", 1600, 720
        )

        # Left panel: live chase-cam feed
        self.image_widget = gui.ImageWidget(o3d.geometry.Image(
            np.zeros((720, 1280, 3), dtype=np.uint8)
        ))
        self.win.add_child(self.image_widget)

        # Right panel: LiDAR map scene
        self.widget = gui.SceneWidget()
        self.widget.scene = rendering.Open3DScene(self.win.renderer)
        self.widget.scene.set_background([0.05, 0.05, 0.05, 1.0])
        self.win.add_child(self.widget)

        # Point cloud material
        self.mat = rendering.MaterialRecord()
        self.mat.shader = "defaultUnlit"
        self.mat.point_size = 2.5 * self.win.scaling

        # Seed geometry so the name exists for later remove calls
        empty = o3d.geometry.PointCloud()
        self.widget.scene.add_geometry("map", empty, self.mat)

        # Coordinate frame at origin
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
        frame_mat = rendering.MaterialRecord()
        frame_mat.shader = "defaultLit"
        self.widget.scene.add_geometry("frame", frame, frame_mat)

        # Camera framed around the full planned flight path
        xs = [w[0] for w in WAYPOINTS]
        ys = [w[1] for w in WAYPOINTS]
        margin = 15.0
        bounds = o3d.geometry.AxisAlignedBoundingBox(
            [min(xs) - margin, min(ys) - margin, -15],
            [max(xs) + margin, max(ys) + margin, 5],
        )
        self.widget.setup_camera(60.0, bounds, bounds.get_center())

        self.win.set_on_layout(self._on_layout)
        self.win.set_on_close(self._on_close)

        # Start background pollers: LiDAR map accumulation + chase-cam feed
        threading.Thread(target=self._poll_lidar, daemon=True).start()
        threading.Thread(target=self._poll_chase_cam, daemon=True).start()

    # ---- callbacks -------------------------------------------------------

    def _on_layout(self, _):
        r = self.win.content_rect
        half_w = r.width // 2
        self.image_widget.frame = gui.Rect(r.x, r.y, half_w, r.height)
        self.widget.frame = gui.Rect(r.x + half_w, r.y, r.width - half_w, r.height)

    def _on_close(self):
        self._running = False
        return True   # allow close

    # ---- background thread: poll LiDAR + accumulate persistent map -------

    def _poll_lidar(self):
        # msgpackrpc uses a Tornado IOLoop that is not thread-safe.
        # Each thread must own its own client connection.
        poll_client = airsim.MultirotorClient()
        poll_client.confirmConnection()

        map_pts = np.empty((0, 3), dtype=np.float64)
        map_colors = np.empty((0, 3), dtype=np.float64)

        while self._running:
            data = poll_client.getLidarData(lidar_name="LidarSensor1", vehicle_name=DRONE)

            if len(data.point_cloud) >= 3:
                pts = np.array(data.point_cloud, dtype=np.float64).reshape(-1, 3)

                # Shift from drone-body frame to world NED frame so scans overlay correctly
                p = poll_client.getMultirotorState(vehicle_name=DRONE).kinematics_estimated.position
                pts += np.array([p.x_val, p.y_val, p.z_val])
                colors = colorize_by_height(pts)

                map_pts = np.vstack([map_pts, pts])
                map_colors = np.vstack([map_colors, colors])

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(map_pts)
                pcd.colors = o3d.utility.Vector3dVector(map_colors)
                pcd = pcd.voxel_down_sample(VOXEL_SIZE)
                map_pts = np.asarray(pcd.points)
                map_colors = np.asarray(pcd.colors)

                def _update(p=pcd):
                    self.widget.scene.remove_geometry("map")
                    self.widget.scene.add_geometry("map", p, self.mat)

                gui.Application.instance.post_to_main_thread(self.win, _update)
                print(f"\r[LiDAR map] {len(map_pts):7d} points", end="", flush=True)

            time.sleep(1.0 / UPDATE_HZ)

    # ---- background thread: reposition + poll the external chase camera --

    def _poll_chase_cam(self):
        cam_client = airsim.MultirotorClient()
        cam_client.confirmConnection()

        heading = np.array([-1.0, 0.0])  # default trailing direction until the drone moves
        reported_ok = False
        last_err = None

        while self._running:
            try:
                state = cam_client.getMultirotorState(vehicle_name=DRONE)
                pos = state.kinematics_estimated.position
                vel = state.kinematics_estimated.linear_velocity

                speed = math.hypot(vel.x_val, vel.y_val)
                if speed > MIN_SPEED_FOR_HEADING:
                    heading = np.array([vel.x_val, vel.y_val]) / speed

                cam_x = pos.x_val - heading[0] * CHASE_DIST
                cam_y = pos.y_val - heading[1] * CHASE_DIST
                cam_z = pos.z_val - CHASE_HEIGHT

                # Look-at orientation toward the drone (derived from airsim.to_quaternion's
                # actual rotation convention: positive pitch tilts the view up, not down).
                dx, dy, dz = pos.x_val - cam_x, pos.y_val - cam_y, pos.z_val - cam_z
                yaw = math.atan2(dy, dx)
                pitch = -math.atan2(dz, math.hypot(dx, dy))
                pose = airsim.Pose(
                    airsim.Vector3r(cam_x, cam_y, cam_z),
                    airsim.to_quaternion(pitch, 0, yaw),
                )
                cam_client.simSetCameraPose(CAM_NAME, pose, external=True)

                resp = cam_client.simGetImages(
                    [airsim.ImageRequest(CAM_NAME, airsim.ImageType.Scene, False, False)],
                    external=True,
                )[0]

                expected_bytes = resp.width * resp.height * 3
                if resp.width > 0 and resp.height > 0 and len(resp.image_data_uint8) == expected_bytes:
                    frame = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
                    frame = frame.reshape(resp.height, resp.width, 3)
                    rgb = np.ascontiguousarray(frame[:, :, ::-1])  # BGR -> RGB
                    img = o3d.geometry.Image(rgb)

                    def _update(im=img):
                        self.image_widget.update_image(im)

                    gui.Application.instance.post_to_main_thread(self.win, _update)

                    if not reported_ok:
                        print(f"\n[ChaseCam] receiving {resp.width}x{resp.height} frames "
                              f"(mean brightness {rgb.mean():.0f}/255)")
                        reported_ok = True
                elif not reported_ok:
                    print(f"\r[ChaseCam] waiting for a valid frame "
                          f"(got {resp.width}x{resp.height}, {len(resp.image_data_uint8)} bytes)...",
                          end="", flush=True)

            except Exception as e:
                if str(e) != last_err:
                    print(f"\n[ChaseCam] error: {e}")
                    last_err = str(e)

            time.sleep(1.0 / CAM_HZ)

    def run(self):
        gui.Application.instance.run()


def follow_waypoints(client: airsim.MultirotorClient, viewer: "LidarViewer"):
    """
    Fly the whole pattern in a single moveOnPathAsync call. AirSim fits a
    pure-pursuit path follower with a velocity-derived lookahead through all
    waypoints, so it rounds each corner instead of stopping and reversing
    direction at every leg -- a hand-rolled per-waypoint P-controller (the
    previous approach here) still jerks at each corner even with continuous
    velocity commands. Lookahead formula and ForwardOnly drivetrain (nose
    points along the direction of travel) match AirSim's own drone_survey
    example, which uses this exact combination for smooth camera footage.
    """
    time.sleep(1.5)  # let viewer settle before moving

    path = [airsim.Vector3r(tx, ty, ALTITUDE) for tx, ty in WAYPOINTS[1:]]
    lookahead = MAX_SPEED + MAX_SPEED / 2

    print(f"\n-> flying full pattern ({len(path)} waypoints) at {MAX_SPEED:.1f} m/s...")
    client.moveOnPathAsync(
        path, MAX_SPEED,
        drivetrain=airsim.DrivetrainType.ForwardOnly,
        yaw_mode=airsim.YawMode(False, 0),
        lookahead=lookahead,
        adaptive_lookahead=1,
        vehicle_name=DRONE,
    ).join()

    client.hoverAsync(vehicle_name=DRONE)
    print("\nPattern complete — hovering until window closed.")


def main():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("Connected to AirSim.")

    client.enableApiControl(True, DRONE)
    client.armDisarm(True, DRONE)

    print("Taking off...")
    client.takeoffAsync(vehicle_name=DRONE).join()
    client.moveToPositionAsync(0, 0, ALTITUDE, 3.0, vehicle_name=DRONE).join()
    print(f"Hovering at {abs(ALTITUDE):.0f} m. Opening chase-cam + LiDAR map viewer (close window to quit)...\n")

    viewer = LidarViewer()

    threading.Thread(target=follow_waypoints, args=(client, viewer), daemon=True).start()

    try:
        viewer.run()   # blocks until window is closed
    finally:
        print("\nLanding...")
        client.landAsync(vehicle_name=DRONE).join()
        client.armDisarm(False, DRONE)
        client.enableApiControl(False, DRONE)
        print("Done.")


if __name__ == "__main__":
    main()
