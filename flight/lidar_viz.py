#!/usr/bin/env python3
"""
Live LiDAR point-cloud visualizer for AirSim Drone1.
Uses Open3D's Filament renderer (Vulkan / GPU) via the gui module.
Points colored blue (high) → red (low) by NED Z value.

Run after the Blocks sim is up:
    ./airsim_venv/bin/python lidar_viz.py

Close the window to land and exit.
"""
import os
# Force GLFW onto XWayland — must be set before open3d is imported.
os.environ.setdefault("DISPLAY", ":1")
os.environ["XDG_SESSION_TYPE"] = "x11"

import time
import threading
import numpy as np
import airsim
import open3d as o3d
import open3d.visualization.gui as gui  # type: ignore
import open3d.visualization.rendering as rendering  # type: ignore

DRONE = "Drone1"
ALTITUDE = -8.0   # NED: negative = up (8 m)
SPEED = 3.0       # m/s
UPDATE_HZ = 10    # LiDAR poll rate

# Flight pattern: (label, x, y) relative offsets in NED (forward=+X, right=+Y)
FLIGHT_PATTERN = [
    ("forward  5 m", ( 5,  0)),
    ("backward 10 m", (-10,  0)),
    ("left     5 m", ( 0, -5)),
    ("right    5 m", ( 0, 10)),
]


def colorize_by_height(points: np.ndarray) -> np.ndarray:
    """Blue (high altitude / negative NED-Z) → red (ground / positive NED-Z)."""
    z = points[:, 2]
    z_min, z_max = z.min(), z.max()
    t = np.zeros(len(z)) if z_max == z_min else (z - z_min) / (z_max - z_min)
    return np.stack([t, np.zeros_like(t), 1.0 - t], axis=1)


class LidarViewer:
    def __init__(self, client: airsim.MultirotorClient):
        self.client = client
        self._running = True

        # --- Filament / GUI setup (runs on main thread) ---
        app = gui.Application.instance
        app.initialize()

        self.win = app.create_window("AirSim LiDAR — GPU (Filament/Vulkan)", 1280, 720)
        self.widget = gui.SceneWidget()
        self.widget.scene = rendering.Open3DScene(self.win.renderer)
        self.widget.scene.set_background([0.05, 0.05, 0.05, 1.0])
        self.win.add_child(self.widget)

        # Point cloud material
        self.mat = rendering.MaterialRecord()
        self.mat.shader = "defaultUnlit"
        self.mat.point_size = 3.0 * self.win.scaling

        # Seed geometry so the name exists for later remove calls
        empty = o3d.geometry.PointCloud()
        self.widget.scene.add_geometry("lidar", empty, self.mat)

        # Coordinate frame at origin
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
        frame_mat = rendering.MaterialRecord()
        frame_mat.shader = "defaultLit"
        self.widget.scene.add_geometry("frame", frame, frame_mat)

        # Camera — look at a 100 m cube around origin
        bounds = o3d.geometry.AxisAlignedBoundingBox([-50, -50, -50], [50, 50, 50])
        self.widget.setup_camera(60.0, bounds, [0, 0, 0])

        self.win.set_on_layout(self._on_layout)
        self.win.set_on_close(self._on_close)

        # Start background poller
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    # ---- callbacks -------------------------------------------------------

    def _on_layout(self, _):
        self.widget.frame = self.win.content_rect

    def _on_close(self):
        self._running = False
        return True   # allow close

    # ---- background thread -----------------------------------------------

    def _poll_loop(self):
        # msgpackrpc uses a Tornado IOLoop that is not thread-safe.
        # Each thread must own its own client connection.
        poll_client = airsim.MultirotorClient()
        poll_client.confirmConnection()

        first = True
        while self._running:
            data = poll_client.getLidarData(lidar_name="LidarSensor1", vehicle_name=DRONE)

            if len(data.point_cloud) >= 3:
                pts = np.array(data.point_cloud, dtype=np.float64).reshape(-1, 3)
                colors = colorize_by_height(pts)

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts)
                pcd.colors = o3d.utility.Vector3dVector(colors)

                do_fit = first
                first = False

                def _update(p=pcd, fit=do_fit):
                    self.widget.scene.remove_geometry("lidar")
                    self.widget.scene.add_geometry("lidar", p, self.mat)
                    if fit:
                        b = p.get_axis_aligned_bounding_box()
                        self.widget.setup_camera(60.0, b, b.get_center())

                gui.Application.instance.post_to_main_thread(self.win, _update)
                print(f"\r[LiDAR] {len(pts):5d} points", end="", flush=True)

            time.sleep(1.0 / UPDATE_HZ)

    def run(self):
        gui.Application.instance.run()


def flight_pattern(client: airsim.MultirotorClient, viewer: "LidarViewer"):
    """Run the movement sequence in a background thread."""
    time.sleep(1.5)  # let viewer settle before moving

    x, y = 0.0, 0.0
    for label, (dx, dy) in FLIGHT_PATTERN:
        if not viewer._running:
            break
        x += dx
        y += dy
        print(f"\n-> {label}  →  ({x:.0f}, {y:.0f}) m")
        client.moveToPositionAsync(x, y, ALTITUDE, SPEED, vehicle_name=DRONE).join()
        time.sleep(0.5)

    print("\nPattern complete — hovering until window closed.")


def main():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("Connected to AirSim.")

    client.enableApiControl(True, DRONE)
    client.armDisarm(True, DRONE)

    print("Taking off...")
    client.takeoffAsync(vehicle_name=DRONE).join()
    client.moveToPositionAsync(0, 0, ALTITUDE, SPEED, vehicle_name=DRONE).join()
    print(f"Hovering at {abs(ALTITUDE):.0f} m. Opening LiDAR viewer (close window to quit)...\n")

    viewer = LidarViewer(client)

    threading.Thread(target=flight_pattern, args=(client, viewer), daemon=True).start()

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
