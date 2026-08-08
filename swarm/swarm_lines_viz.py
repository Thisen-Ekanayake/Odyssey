#!/usr/bin/env python3
"""
Takes off the swarm (Drone1-4) and shows a live top-down-ish 3D view, in its
own window, of the drones' formation quadrilateral segmented into 4 small
squares: the 4 perimeter edges (drone-to-drone around the formation, ordered
by angle around the centroid so the loop doesn't self-cross) plus a line from
the diagonal intersection (the two corner-to-corner diagonals are computed
but not drawn -- drawing them would cut each square into two triangles
instead) to the midpoint of each perimeter edge. No drone models or LiDAR,
just the lines, redrawn each tick from each drone's live position (same
world-frame convention as swarm_comms.SwarmPositions / swarm_converge_viz.py:
local position + that drone's calibrated spawn offset).

Run after the sim is up:
    ./airsim_venv/bin/python swarm/swarm_lines_viz.py

Close the window to land the swarm and exit.
"""
import os
# Force GLFW onto XWayland -- must be set before open3d is imported.
os.environ.setdefault("DISPLAY", ":1")
os.environ["XDG_SESSION_TYPE"] = "x11"

import math
import threading
import time

import numpy as np
import airsim
import open3d as o3d
import open3d.visualization.gui as gui  # type: ignore
import open3d.visualization.rendering as rendering  # type: ignore

from swarm_comms import DRONES, SPAWNS, SwarmPositions

UPDATE_HZ = 10
PERIMETER_COLOR = (0.2, 0.9, 1.0)   # cyan -- the 4 outer edges
SEGMENT_COLOR = (1.0, 0.7, 0.15)    # orange -- center-to-midpoint dividers
LINE_WIDTH = 3.0
WIN_SIZE = (900, 700)

# points layout per tick: [4 corners in perimeter order, center, 4 edge midpoints]
PERIMETER_LINES = [[0, 1], [1, 2], [2, 3], [3, 0]]
SEGMENT_LINES = [[4, 5], [4, 6], [4, 7], [4, 8]]


def _segment_intersection(p1, p2, p3, p4):
    """Point where segment p1-p2 crosses segment p3-p4 ((x,y,z) each);
    falls back to the average of all 4 points if they're ~parallel."""
    x1, y1, _ = p1
    x2, y2, _ = p2
    x3, y3, _ = p3
    x4, y4, _ = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-9:
        pts = np.array([p1, p2, p3, p4])
        return tuple(pts.mean(axis=0))
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return (
        x1 + t * (x2 - x1),
        y1 + t * (y2 - y1),
        p1[2] + t * (p2[2] - p1[2]),
    )


def _midpoint(p1, p2):
    return tuple((np.array(p1) + np.array(p2)) / 2.0)


class LinesViewer:
    def __init__(self):
        self._running = True

        app = gui.Application.instance
        app.initialize()

        self.win = app.create_window("AirSim — Swarm 4-Square Segmentation", *WIN_SIZE)
        self.widget = gui.SceneWidget()
        self.widget.scene = rendering.Open3DScene(self.win.renderer)
        self.widget.scene.set_background([0.05, 0.05, 0.05, 1.0])
        self.win.add_child(self.widget)
        self.win.set_on_close(self._on_close)
        self.win.set_on_layout(self._layout)

        self.mat = rendering.MaterialRecord()
        self.mat.shader = "unlitLine"
        self.mat.line_width = LINE_WIDTH * self.win.scaling

        empty = o3d.geometry.LineSet()
        self.widget.scene.add_geometry("lines", empty, self.mat)
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

        threading.Thread(target=self._poll_lines, daemon=True).start()

    def _layout(self, _):
        r = self.win.content_rect
        self.widget.frame = gui.Rect(r.x, r.y, r.width, r.height)

    def _on_close(self):
        self._running = False
        return True

    def _poll_lines(self):
        # msgpackrpc's tornado IOLoop isn't thread-safe -- own client per thread.
        poll_client = airsim.MultirotorClient()
        poll_client.confirmConnection()
        swarm = SwarmPositions(poll_client)

        while self._running:
            positions = swarm.refresh()
            world = {
                d: (positions[d].world_x, positions[d].world_y,
                    poll_client.getMultirotorState(vehicle_name=d).kinematics_estimated.position.z_val)
                for d in DRONES
            }

            # Order corners by angle around the centroid so the perimeter
            # loop traces the formation's actual outline instead of
            # whatever order DRONES happens to list them in (which
            # self-crosses into a bowtie for this swarm's spawn layout).
            cx = sum(p[0] for p in world.values()) / len(world)
            cy = sum(p[1] for p in world.values()) / len(world)
            order = sorted(DRONES, key=lambda d: math.atan2(world[d][1] - cy, world[d][0] - cx))
            corners = [world[d] for d in order]

            center = _segment_intersection(corners[0], corners[2], corners[1], corners[3])
            midpoints = [_midpoint(corners[i], corners[(i + 1) % 4]) for i in range(4)]
            points = corners + [center] + midpoints

            def _update(points=points):
                lines = o3d.geometry.LineSet()
                lines.points = o3d.utility.Vector3dVector(points)
                lines.lines = o3d.utility.Vector2iVector(PERIMETER_LINES + SEGMENT_LINES)
                colors = [PERIMETER_COLOR] * len(PERIMETER_LINES) + [SEGMENT_COLOR] * len(SEGMENT_LINES)
                lines.colors = o3d.utility.Vector3dVector(colors)
                self.widget.scene.remove_geometry("lines")
                self.widget.scene.add_geometry("lines", lines, self.mat)

            gui.Application.instance.post_to_main_thread(self.win, _update)
            time.sleep(1.0 / UPDATE_HZ)

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
    print(f"Connected. Swarm: {', '.join(DRONES)}")

    for d in DRONES:
        client.enableApiControl(True, d)
        client.armDisarm(True, d)

    print("Taking off...")
    for f in [client.takeoffAsync(vehicle_name=d) for d in DRONES]:
        f.join()

    viewer = LinesViewer()
    viewer.run()   # blocks until the window is closed

    print("Landing...")
    for f in [client.landAsync(vehicle_name=d) for d in DRONES]:
        f.join()
    for d in DRONES:
        client.armDisarm(False, d)
        client.enableApiControl(False, d)
    print("Done.")


if __name__ == "__main__":
    main()
