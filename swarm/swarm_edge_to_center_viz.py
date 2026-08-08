#!/usr/bin/env python3
"""
Live view of the swarm_edge_to_center.py maneuver, across 6 windows: one
chase-cam window per drone (4), one Open3D window with the formation
quadrilateral segmented into 4 small squares (perimeter edges +
center-to-midpoint dividers, same construction as swarm_lines_viz.py) with a
colored dot marking each drone's LIVE position on top of it, and one shared
Open3D LiDAR map merging all 4 drones' scans -- while swarm_edge_to_center.run()
drives the actual flight in a background thread.

The segmentation LINES are fixed target geometry, computed once from each
drone's position right after the climb to cruise altitude -- the same
one-time computation swarm_edge_to_center.py itself does before flying.
Recomputing them every tick (the way swarm_lines_viz.py does for its
always-hovering swarm) would draw a moving target that no longer matches
what the drones are actually flying toward once they leave their corners.
Only the drone-position DOTS move, tracking each drone every tick via
Open3DScene.set_geometry_transform (cheaper than rebuilding the mesh).

Each drone needs its OWN external chase camera to show 4 simultaneous
views -- settings.json declares ChaseCam/ChaseCam2/ChaseCam3/ChaseCam4, one
per drone, each repositioned every tick to trail its own drone (same
technique as swarm_converge_viz.py's chase-cam windows).

The LiDAR map is the same merge swarm_converge_viz.py builds: each drone's
LidarSensor1 returns points in that drone's own local NED frame (DataFrame:
"SensorLocalFrame" in settings.json), so placing them on one shared map needs
that drone's spawn offset added on top of its sensor pose -- the same
world_x/world_y = local + offset convention swarm_comms.SwarmPositions uses.
Points are placed by the SIMULATOR's own sensor pose (perfect localization,
no SLAM). Each drone keeps a distinct base hue, shaded dark-to-light by
height (NED z) so vertical structure is visible at a glance (see
_height_gradient_colors / HEIGHT_GRADIENT_Z, values copied from
swarm_converge_viz.py since both scripts fly the same ALTITUDE).

Run after the sim is up:
    ./airsim_venv/bin/python swarm/swarm_edge_to_center_viz.py

Closing any window only stops the viewer -- the flight itself (and its own
end-of-maneuver landing) runs independently on its own client and keeps
going to completion regardless.
"""
import os
# Force GLFW onto XWayland -- must be set before open3d is imported.
os.environ.setdefault("DISPLAY", ":1")
os.environ["XDG_SESSION_TYPE"] = "x11"

import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import airsim
import open3d as o3d
import open3d.visualization.gui as gui  # type: ignore
import open3d.visualization.rendering as rendering  # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slam.geometry import airsim_pose_to_matrix  # noqa: E402
from tools import scene_rotation  # noqa: E402
from tools import window_recorder  # noqa: E402,F401

from swarm_comms import DRONES, SPAWNS, SwarmPositions
from swarm_lines_viz import (
    PERIMETER_COLOR, SEGMENT_COLOR, PERIMETER_LINES, SEGMENT_LINES,
    _segment_intersection, _midpoint,
)
import swarm_edge_to_center

UPDATE_HZ = 10
WIN_SIZE = (900, 700)
LINE_WIDTH = 3.0
DOT_RADIUS = 3.0   # m -- sized to read clearly at this formation's ~100s-of-m scale

DRONE_COLOR = {
    "Drone1": (1.0, 0.3, 0.3),   # red
    "Drone2": (0.3, 1.0, 0.3),   # green
    "Drone3": (0.3, 0.5, 1.0),   # blue
    "Drone4": (1.0, 1.0, 0.3),   # yellow
}

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
CAM_WIN_SIZE = (640, 360)

# --- shared LiDAR map (same construction/tuning as swarm_converge_viz.py) ---
MAP_WIN_SIZE = (900, 700)
VOXEL_SIZE = 1.5   # m; keeps the accumulated map's point count/density down

HEIGHT_GRADIENT_Z = (1.0, -28.0)   # (dark-end z, bright-end z), meters NED
HEIGHT_GRADIENT_DARK = 0.28        # multiply base color by this at the dark end
HEIGHT_GRADIENT_LIGHT = 0.85       # blend fraction toward white at the bright end


def _height_gradient_colors(base_color, z: np.ndarray) -> np.ndarray:
    """(N,3) RGB array tinting `base_color` from dark (low) to light (high)
    by NED height `z` (N,), so vertical structure reads at a glance while
    each drone's points stay recognizably its own hue."""
    base = np.asarray(base_color, dtype=np.float64)
    z_dark, z_bright = HEIGHT_GRADIENT_Z
    t = (z - z_dark) / (z_bright - z_dark)
    t = np.clip(t, 0.0, 1.0)[:, None]
    dark = base * HEIGHT_GRADIENT_DARK
    light = base + (1.0 - base) * HEIGHT_GRADIENT_LIGHT
    return dark + t * (light - dark)


class EdgeToCenterViewer:
    def __init__(self):
        self._running = True

        app = gui.Application.instance
        app.initialize()

        # --- one small window per drone's chase cam ---
        self.image_widgets = {}
        self.cam_windows = {}
        for drone in DRONES:
            cam_win = app.create_window(f"AirSim — {drone} chase cam", *CAM_WIN_SIZE)
            widget = gui.ImageWidget(o3d.geometry.Image(
                np.zeros((CAM_WIN_SIZE[1], CAM_WIN_SIZE[0], 3), dtype=np.uint8)
            ))
            cam_win.add_child(widget)
            cam_win.set_on_layout(self._make_fill_layout(cam_win, widget))
            cam_win.set_on_close(self._on_close)
            self.image_widgets[drone] = widget
            self.cam_windows[drone] = cam_win

        self.win = app.create_window("AirSim — Swarm Edge-to-Center", *WIN_SIZE)
        self.widget = gui.SceneWidget()
        self.widget.scene = rendering.Open3DScene(self.win.renderer)
        self.widget.scene.set_background([0.05, 0.05, 0.05, 1.0])
        self.win.add_child(self.widget)
        self.win.set_on_close(self._on_close)
        self.win.set_on_layout(self._layout)

        self.line_mat = rendering.MaterialRecord()
        self.line_mat.shader = "unlitLine"
        self.line_mat.line_width = LINE_WIDTH * self.win.scaling

        self.dot_mat = rendering.MaterialRecord()
        self.dot_mat.shader = "defaultLit"

        empty = o3d.geometry.LineSet()
        self.widget.scene.add_geometry("lines", empty, self.line_mat)

        for d in DRONES:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=DOT_RADIUS)
            sphere.compute_vertex_normals()
            sphere.paint_uniform_color(DRONE_COLOR[d])
            self.widget.scene.add_geometry(f"dot_{d}", sphere, self.dot_mat)

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
        frame_mat = rendering.MaterialRecord()
        frame_mat.shader = "defaultLit"
        self.widget.scene.add_geometry("frame", frame, frame_mat)

        xs = [SPAWNS[d][0] for d in DRONES]
        ys = [SPAWNS[d][1] for d in DRONES]
        margin = 20.0
        bounds = o3d.geometry.AxisAlignedBoundingBox(
            [min(xs) - margin, min(ys) - margin, -35.0],
            [max(xs) + margin, max(ys) + margin, 5.0],
        )
        self.widget.setup_camera(60.0, bounds, bounds.get_center())

        # --- one shared window for the merged LiDAR map ---
        self.map_win = app.create_window("AirSim — Swarm LiDAR Map", *MAP_WIN_SIZE)
        self.map_widget = gui.SceneWidget()
        self.map_widget.scene = rendering.Open3DScene(self.map_win.renderer)
        self.map_widget.scene.set_background([0.05, 0.05, 0.05, 1.0])
        self.map_win.add_child(self.map_widget)
        self.map_win.set_on_close(self._on_close)
        self.map_win.set_on_layout(self._map_layout)

        self.map_mat = rendering.MaterialRecord()
        self.map_mat.shader = "defaultUnlit"
        self.map_mat.point_size = 2.0 * self.map_win.scaling

        empty_map = o3d.geometry.PointCloud()
        self.map_widget.scene.add_geometry("map", empty_map, self.map_mat)
        map_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
        self.map_widget.scene.add_geometry("frame", map_frame, frame_mat)
        self.map_widget.setup_camera(60.0, bounds, bounds.get_center())
        # Rotation control on the LiDAR map only -- self.widget (the
        # edge-to-center segmentation lines) is computed geometry, not
        # sensor data.
        self.rotation = scene_rotation.RotationPanel(
            self.map_win, self.map_widget, bounds, is_running=lambda: self._running)

        threading.Thread(target=self._poll, daemon=True).start()
        threading.Thread(target=self._poll_lidar, daemon=True).start()
        for drone, cam in CHASE_CAMS.items():
            threading.Thread(target=self._poll_chase_cam, args=(drone, cam), daemon=True).start()

    def _make_fill_layout(self, win, widget):
        def _layout(_):
            r = win.content_rect
            widget.frame = gui.Rect(r.x, r.y, r.width, r.height)
        return _layout

    def _layout(self, _):
        r = self.win.content_rect
        self.widget.frame = gui.Rect(r.x, r.y, r.width, r.height)

    def _map_layout(self, _):
        r = self.map_win.content_rect
        self.map_widget.frame = gui.Rect(r.x, r.y, r.width, r.height)
        self.rotation.layout(self.map_widget.frame)

    def _on_close(self):
        self._running = False
        return True

    def _poll(self):
        # msgpackrpc's tornado IOLoop isn't thread-safe -- own client per thread.
        poll_client = airsim.MultirotorClient()
        poll_client.confirmConnection()
        swarm = SwarmPositions(poll_client)

        target_drawn = False

        while self._running:
            positions = swarm.refresh()
            world = {
                d: (positions[d].world_x, positions[d].world_y,
                    poll_client.getMultirotorState(vehicle_name=d).kinematics_estimated.position.z_val)
                for d in DRONES
            }

            # Draw the 4-square segmentation once, from wherever the drones
            # are the first time we see them settled (post-climb) -- this is
            # the fixed target swarm_edge_to_center.py itself flies toward.
            if not target_drawn:
                cx = sum(p[0] for p in world.values()) / len(world)
                cy = sum(p[1] for p in world.values()) / len(world)
                order = sorted(DRONES, key=lambda d: math.atan2(world[d][1] - cy, world[d][0] - cx))
                corners = [world[d] for d in order]
                center = _segment_intersection(corners[0], corners[2], corners[1], corners[3])
                midpoints = [_midpoint(corners[i], corners[(i + 1) % 4]) for i in range(4)]
                line_points = corners + [center] + midpoints
                target_drawn = True

                def _draw_target(line_points=line_points):
                    lines = o3d.geometry.LineSet()
                    lines.points = o3d.utility.Vector3dVector(line_points)
                    lines.lines = o3d.utility.Vector2iVector(PERIMETER_LINES + SEGMENT_LINES)
                    colors = [PERIMETER_COLOR] * len(PERIMETER_LINES) + [SEGMENT_COLOR] * len(SEGMENT_LINES)
                    lines.colors = o3d.utility.Vector3dVector(colors)
                    self.widget.scene.remove_geometry("lines")
                    self.widget.scene.add_geometry("lines", lines, self.line_mat)

                gui.Application.instance.post_to_main_thread(self.win, _draw_target)

            def _move_dots(world=world):
                for d in DRONES:
                    x, y, z = world[d]
                    T = np.eye(4)
                    T[:3, 3] = [x, y, z]
                    self.widget.scene.set_geometry_transform(f"dot_{d}", T)

            gui.Application.instance.post_to_main_thread(self.win, _move_dots)
            time.sleep(1.0 / UPDATE_HZ)

    # ---- background thread: poll all 4 drones' LiDAR, merge into one map --

    def _poll_lidar(self):
        # msgpackrpc's tornado IOLoop isn't thread-safe -- own client per thread.
        poll_client = airsim.MultirotorClient()
        poll_client.confirmConnection()

        # Calibrated per-drone local->world offset (see SwarmPositions.offset
        # in swarm_comms.py) -- raw local (x, y) is NOT simply zero at that
        # drone's own spawn for non-Drone1 vehicles in this setup, so this
        # snapshots each drone's actual reading now and corrects against its
        # known spawn -- otherwise every drone's map lands in the same spot.
        calibration = SwarmPositions(poll_client)
        offsets = {d: calibration.offset(d) for d in DRONES}

        map_pts = {d: np.empty((0, 3), dtype=np.float64) for d in DRONES}

        while self._running:
            for drone in DRONES:
                data = poll_client.getLidarData(lidar_name="LidarSensor1", vehicle_name=drone)
                if len(data.point_cloud) < 3:
                    continue
                pts = np.array(data.point_cloud, dtype=np.float64).reshape(-1, 3)

                T = airsim_pose_to_matrix(data.pose)
                pts = pts @ T[:3, :3].T + T[:3, 3]
                dx, dy = offsets[drone]
                pts[:, 0] += dx
                pts[:, 1] += dy

                merged = np.vstack([map_pts[drone], pts])
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(merged)
                pcd = pcd.voxel_down_sample(VOXEL_SIZE)
                map_pts[drone] = np.asarray(pcd.points)

            def _update():
                combined = o3d.geometry.PointCloud()
                for drone in DRONES:
                    drone_pts = map_pts[drone]
                    if len(drone_pts) == 0:
                        continue
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(drone_pts)
                    pcd.colors = o3d.utility.Vector3dVector(
                        _height_gradient_colors(DRONE_COLOR[drone], drone_pts[:, 2]))
                    combined += pcd
                self.map_widget.scene.remove_geometry("map")
                self.map_widget.scene.add_geometry("map", combined, self.map_mat)

            gui.Application.instance.post_to_main_thread(self.map_win, _update)
            time.sleep(1.0 / UPDATE_HZ)

    # ---- background thread: reposition + poll one drone's chase camera ----

    def _poll_chase_cam(self, drone: str, cam_name: str):
        cam_client = airsim.MultirotorClient()
        cam_client.confirmConnection()
        # getMultirotorState().position is LOCAL to this vehicle, but
        # simSetCameraPose(..., external=True) places the camera in the
        # shared WORLD frame -- same local->world offset correction as
        # swarm_edge_to_center.py's moveOnPathAsync waypoints and this
        # viewer's own _poll() dot tracker. Without it every chase cam lands
        # near the same wrong spot regardless of where its drone actually is.
        calibration = SwarmPositions(cam_client)
        off_x, off_y = calibration.offset(drone)

        heading = np.array([-1.0, 0.0])
        reported_ok = False
        last_err = None

        while self._running:
            try:
                state = cam_client.getMultirotorState(vehicle_name=drone)
                pos = state.kinematics_estimated.position
                vel = state.kinematics_estimated.linear_velocity
                world_x = pos.x_val + off_x
                world_y = pos.y_val + off_y

                speed = math.hypot(vel.x_val, vel.y_val)
                if speed > MIN_SPEED_FOR_HEADING:
                    heading = np.array([vel.x_val, vel.y_val]) / speed

                cam_x = world_x - heading[0] * CHASE_DIST
                cam_y = world_y - heading[1] * CHASE_DIST
                cam_z = pos.z_val - CHASE_HEIGHT

                dx, dy, dz = world_x - cam_x, world_y - cam_y, pos.z_val - cam_z
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

    def quit(self):
        def _do_quit():
            gui.Application.instance.quit()
        try:
            gui.Application.instance.post_to_main_thread(self.win, _do_quit)
        except Exception:
            pass

    def run(self):
        gui.Application.instance.run()


def main():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("Connected to AirSim.")

    viewer = EdgeToCenterViewer()

    def _fly():
        swarm_edge_to_center.run(client)
        viewer.quit()

    threading.Thread(target=_fly, daemon=True).start()

    viewer.run()   # blocks until a window is closed or the maneuver finishes


if __name__ == "__main__":
    main()
