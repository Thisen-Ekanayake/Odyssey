#!/usr/bin/env python3
"""
Live view, in its own Open3D window, of the swarm_edge_to_center.py
maneuver: the formation quadrilateral segmented into 4 small squares
(perimeter edges + center-to-midpoint dividers, same construction as
swarm_lines_viz.py) with a colored dot marking each drone's LIVE position on
top of it, while swarm_edge_to_center.run() drives the actual flight in a
background thread.

The segmentation LINES are fixed target geometry, computed once from each
drone's position right after the climb to cruise altitude -- the same
one-time computation swarm_edge_to_center.py itself does before flying.
Recomputing them every tick (the way swarm_lines_viz.py does for its
always-hovering swarm) would draw a moving target that no longer matches
what the drones are actually flying toward once they leave their corners.
Only the drone-position DOTS move, tracking each drone every tick via
Open3DScene.set_geometry_transform (cheaper than rebuilding the mesh).

Run after the sim is up:
    ./airsim_venv/bin/python flight/swarm_edge_to_center_viz.py

Closing the window only stops the viewer -- the flight itself (and its own
end-of-maneuver landing) runs independently on its own client and keeps
going to completion regardless.
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


class EdgeToCenterViewer:
    def __init__(self):
        self._running = True

        app = gui.Application.instance
        app.initialize()

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

        threading.Thread(target=self._poll, daemon=True).start()

    def _layout(self, _):
        r = self.win.content_rect
        self.widget.frame = gui.Rect(r.x, r.y, r.width, r.height)

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
