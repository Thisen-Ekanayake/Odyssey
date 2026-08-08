#!/usr/bin/env python3
"""
5-window live view of the 4-drone convergence maneuver (swarm_converge.py):
one chase-cam feed per drone (Drone1..Drone4) plus one shared LiDAR map
window showing all 4 drones' scans merged and color-coded by drone.

Each drone needs its OWN external chase camera to show 4 simultaneous
views -- settings.json declares ChaseCam/ChaseCam2/ChaseCam3/ChaseCam4,
one per drone, each repositioned every tick to trail its own drone (same
technique as lidar_viz.py's single ChaseCam, just x4).

LiDAR points arrive in each drone's own local NED frame (DataFrame:
"SensorLocalFrame" in settings.json), so placing them on a SHARED map needs
that drone's spawn offset added on top of its sensor pose -- the same
world_x/world_y = local + spawn convention swarm_comms.SwarmPositions uses.
Points are colored placed by the SIMULATOR's own sensor pose (perfect
localization, no SLAM), same baseline lidar_viz.py uses for one drone.

The flight itself is delegated to swarm_converge.run() in a background
thread; this script only adds the 5-window visualization on top of it.

Run after the sim is up (restart required after the settings.json change
that added ChaseCam2-4):
    ./airsim_venv/bin/python flight/swarm_converge_viz.py

Close any window to land the swarm and exit.
"""
import os
# Force GLFW onto XWayland -- must be set before open3d is imported.
os.environ.setdefault("DISPLAY", ":1")
os.environ["XDG_SESSION_TYPE"] = "x11"

import math
import sys
import time
import threading
from pathlib import Path

import numpy as np
import airsim
import open3d as o3d
import open3d.visualization.gui as gui  # type: ignore
import open3d.visualization.rendering as rendering  # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slam.geometry import airsim_pose_to_matrix  # noqa: E402

from swarm_comms import DRONES, SPAWNS
import swarm_converge

UPDATE_HZ = 10       # LiDAR poll rate
VOXEL_SIZE = 1.0      # m; keeps the accumulated map's point count/density down

# name -> (drone, external camera name), one chase cam per drone.
CHASE_CAMS = {
    "Drone1": "ChaseCam",
    "Drone2": "ChaseCam2",
    "Drone3": "ChaseCam3",
    "Drone4": "ChaseCam4",
}
CAM_HZ = 15
CHASE_DIST = 8.0       # m behind the drone
CHASE_HEIGHT = 4.0     # m above the drone
MIN_SPEED_FOR_HEADING = 0.3

# Distinct color per drone in the merged map (RGB, 0-1).
DRONE_COLOR = {
    "Drone1": (1.0, 0.3, 0.3),   # red
    "Drone2": (0.3, 1.0, 0.3),   # green
    "Drone3": (0.3, 0.5, 1.0),   # blue
    "Drone4": (1.0, 1.0, 0.3),   # yellow
}

CAM_WIN_SIZE = (640, 360)
MAP_WIN_SIZE = (900, 700)


class SwarmViewer:
    def __init__(self):
        self._running = True

        app = gui.Application.instance
        app.initialize()

        # --- one small window per drone's chase cam ---
        self.image_widgets = {}
        self.cam_windows = {}
        for drone in DRONES:
            win = app.create_window(
                f"AirSim — {drone} chase cam", *CAM_WIN_SIZE
            )
            widget = gui.ImageWidget(o3d.geometry.Image(
                np.zeros((CAM_WIN_SIZE[1], CAM_WIN_SIZE[0], 3), dtype=np.uint8)
            ))
            win.add_child(widget)
            win.set_on_layout(self._make_fill_layout(win, widget))
            win.set_on_close(self._on_close)
            self.image_widgets[drone] = widget
            self.cam_windows[drone] = win

        # --- one shared window for the merged LiDAR map ---
        self.map_win = app.create_window("AirSim — Swarm LiDAR Map", *MAP_WIN_SIZE)
        self.map_widget = gui.SceneWidget()
        self.map_widget.scene = rendering.Open3DScene(self.map_win.renderer)
        self.map_widget.scene.set_background([0.05, 0.05, 0.05, 1.0])
        self.map_win.add_child(self.map_widget)
        self.map_win.set_on_layout(self._make_fill_layout(self.map_win, self.map_widget))
        self.map_win.set_on_close(self._on_close)

        self.mat = rendering.MaterialRecord()
        self.mat.shader = "defaultUnlit"
        self.mat.point_size = 2.0 * self.map_win.scaling

        empty = o3d.geometry.PointCloud()
        self.map_widget.scene.add_geometry("map", empty, self.mat)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
        frame_mat = rendering.MaterialRecord()
        frame_mat.shader = "defaultLit"
        self.map_widget.scene.add_geometry("frame", frame, frame_mat)

        # Frame the camera around the whole 4-corner spread so the shrinking
        # formation stays in view start to finish.
        xs = [SPAWNS[d][0] for d in DRONES]
        ys = [SPAWNS[d][1] for d in DRONES]
        margin = 20.0
        bounds = o3d.geometry.AxisAlignedBoundingBox(
            [min(xs) - margin, min(ys) - margin, -35.0],
            [max(xs) + margin, max(ys) + margin, 5.0],
        )
        self.map_widget.setup_camera(60.0, bounds, bounds.get_center())

        # Background pollers: one LiDAR-merge thread, one per-drone chase-cam thread.
        threading.Thread(target=self._poll_lidar, daemon=True).start()
        for drone, cam in CHASE_CAMS.items():
            threading.Thread(target=self._poll_chase_cam, args=(drone, cam), daemon=True).start()

    # ---- window plumbing --------------------------------------------------

    def _make_fill_layout(self, win, widget):
        def _layout(_):
            r = win.content_rect
            widget.frame = gui.Rect(r.x, r.y, r.width, r.height)
        return _layout

    def _on_close(self):
        self._running = False
        return True

    def quit(self):
        def _do_quit():
            gui.Application.instance.quit()
        try:
            gui.Application.instance.post_to_main_thread(self.map_win, _do_quit)
        except Exception:
            pass

    # ---- background thread: poll all 4 drones' LiDAR, merge into one map --

    def _poll_lidar(self):
        # msgpackrpc's tornado IOLoop isn't thread-safe -- own client per thread.
        poll_client = airsim.MultirotorClient()
        poll_client.confirmConnection()

        map_pts = {d: np.empty((0, 3), dtype=np.float64) for d in DRONES}

        while self._running:
            for drone in DRONES:
                data = poll_client.getLidarData(lidar_name="LidarSensor1", vehicle_name=drone)
                if len(data.point_cloud) < 3:
                    continue
                pts = np.array(data.point_cloud, dtype=np.float64).reshape(-1, 3)

                # Sensor pose is in THIS vehicle's own local NED frame; add its
                # spawn offset to land in the shared world frame (Z origin is
                # common across vehicles, only X/Y differ -- swarm_comms.py's
                # world_x/world_y convention).
                T = airsim_pose_to_matrix(data.pose)
                pts = pts @ T[:3, :3].T + T[:3, 3]
                sx, sy = SPAWNS[drone]
                pts[:, 0] += sx
                pts[:, 1] += sy

                merged = np.vstack([map_pts[drone], pts])
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(merged)
                pcd = pcd.voxel_down_sample(VOXEL_SIZE)
                map_pts[drone] = np.asarray(pcd.points)

            def _update():
                combined = o3d.geometry.PointCloud()
                for drone in DRONES:
                    n = len(map_pts[drone])
                    if n == 0:
                        continue
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(map_pts[drone])
                    pcd.colors = o3d.utility.Vector3dVector(
                        np.tile(DRONE_COLOR[drone], (n, 1)))
                    combined += pcd
                self.map_widget.scene.remove_geometry("map")
                self.map_widget.scene.add_geometry("map", combined, self.mat)

            gui.Application.instance.post_to_main_thread(self.map_win, _update)
            total = sum(len(v) for v in map_pts.values())
            print(f"\r[LiDAR map] {total:7d} points merged from {len(DRONES)} drones", end="", flush=True)

            time.sleep(1.0 / UPDATE_HZ)

    # ---- background thread: reposition + poll one drone's chase camera ----

    def _poll_chase_cam(self, drone: str, cam_name: str):
        cam_client = airsim.MultirotorClient()
        cam_client.confirmConnection()

        heading = np.array([-1.0, 0.0])
        reported_ok = False
        last_err = None

        while self._running:
            try:
                state = cam_client.getMultirotorState(vehicle_name=drone)
                pos = state.kinematics_estimated.position
                vel = state.kinematics_estimated.linear_velocity

                speed = math.hypot(vel.x_val, vel.y_val)
                if speed > MIN_SPEED_FOR_HEADING:
                    heading = np.array([vel.x_val, vel.y_val]) / speed

                cam_x = pos.x_val - heading[0] * CHASE_DIST
                cam_y = pos.y_val - heading[1] * CHASE_DIST
                cam_z = pos.z_val - CHASE_HEIGHT

                dx, dy, dz = pos.x_val - cam_x, pos.y_val - cam_y, pos.z_val - cam_z
                yaw = math.atan2(dy, dx)
                pitch = -math.atan2(dz, math.hypot(dx, dy))
                pose = airsim.Pose(
                    airsim.Vector3r(cam_x, cam_y, cam_z),
                    airsim.to_quaternion(pitch, 0, yaw),
                )
                cam_client.simSetCameraPose(cam_name, pose, external=True)

                resp = cam_client.simGetImages(
                    [airsim.ImageRequest(cam_name, airsim.ImageType.Scene, False, False)],
                    external=True,
                )[0]

                expected_bytes = resp.width * resp.height * 3
                if resp.width > 0 and resp.height > 0 and len(resp.image_data_uint8) == expected_bytes:
                    frame = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
                    frame = frame.reshape(resp.height, resp.width, 3)
                    rgb = np.ascontiguousarray(frame[:, :, ::-1])  # BGR -> RGB
                    img = o3d.geometry.Image(rgb)

                    widget = self.image_widgets[drone]
                    cam_win = self.cam_windows[drone]

                    def _update(im=img, w=widget):
                        w.update_image(im)

                    gui.Application.instance.post_to_main_thread(cam_win, _update)

                    if not reported_ok:
                        print(f"\n[{cam_name}] {drone}: receiving {resp.width}x{resp.height} frames")
                        reported_ok = True
                elif not reported_ok:
                    print(f"\r[{cam_name}] {drone}: waiting for a valid frame "
                          f"(got {resp.width}x{resp.height}, {len(resp.image_data_uint8)} bytes)...",
                          end="", flush=True)

            except Exception as e:
                if str(e) != last_err:
                    print(f"\n[{cam_name}] {drone}: error: {e}")
                    last_err = str(e)

            time.sleep(1.0 / CAM_HZ)

    def run(self):
        gui.Application.instance.run()


def main():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("Connected to AirSim.")

    viewer = SwarmViewer()

    def _fly():
        swarm_converge.run(client)
        viewer.quit()

    threading.Thread(target=_fly, daemon=True).start()

    viewer.run()   # blocks until a window is closed or the maneuver finishes


if __name__ == "__main__":
    main()
