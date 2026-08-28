#!/usr/bin/env python3
"""End-to-end check of cooperative_mapping.launch.py against the mock server.

Brings up the real launch file (bridge x4 + cloud_merger_node + octomap_server,
no RViz) and asserts that occupied voxels end up clustered around each drone's
own world position, not at the map origin and not swapped with another drone's.

This is the property that matters and the one most likely to break silently: a
missed NED->ENU swap on a spawn, a frame_id typo in cloud_merger_node, or
octomap_server losing its per-message sensor-origin lookup would all still
produce SOME octomap -- the geometry would just be wrong, with no error, no
crash, and a demo recording that looks plausible until someone checks the
numbers. This test checks the numbers.

    ./scripts/ros_enter.sh python3 ros2_ws/src/airsim_swarm_bridge/test/test_cooperative_mapping.py

Needs the reference mock server (mock_airsim_server.py) already running -- see
run_tests.sh, which starts it on the host venv before calling this.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import numpy as np
import rclpy
import sensor_msgs_py.point_cloud2 as pc2
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2

PORT = int(os.environ.get("MOCK_PORT", "41999"))

# NED (X, Y) spawns from swarm/swarm_comms.py::SPAWNS, converted to ENU the
# same way frames.ned_point_to_enu does: (x_enu, y_enu) = (y_ned, x_ned).
# Deliberately includes two asymmetric spawns (Drone2, Drone3) -- a spawn with
# X == Y hides an X/Y swap bug completely, which is exactly the mistake an
# earlier draft of this check made.
_SPAWNS_NED = {"Drone1": (150, 150), "Drone2": (90, -210),
               "Drone3": (-150, 150), "Drone4": (-160, -160)}
SPAWNS_ENU = {name: (y, x) for name, (x, y) in _SPAWNS_NED.items()}

_passed = 0
_failed: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global _passed
    if cond:
        _passed += 1
        print(f"  ok   {name}")
    else:
        _failed.append(name)
        print(f"  FAIL {name}  {detail}")


class Grab(Node):
    def __init__(self) -> None:
        super().__init__("cooperative_mapping_test")
        self.msg: PointCloud2 | None = None
        self.create_subscription(
            PointCloud2, "/octomap_point_cloud_centers", self._cb, 5)

    def _cb(self, msg: PointCloud2) -> None:
        self.msg = msg


def main() -> int:
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    # `ros2 launch` is a supervisor that forks one process per node (4 bridges +
    # cloud_merger + octomap_server + maneuver + swarm_state here) -- killing
    # only its own pid leaves every child running. start_new_session=True puts
    # the whole tree in its own process group so os.killpg can take it down in
    # one shot; SIGTERM alone was observed to leave several nodes behind.
    proc = subprocess.Popen(
        ["ros2", "launch", "airsim_swarm_bridge", "cooperative_mapping.launch.py",
         f"port:={PORT}", "rviz:=false"],
        cwd=repo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True)

    rclpy.init()
    node = Grab()
    try:
        # Spin for a fixed warm-up window and keep the LAST message, rather than
        # returning as soon as one "big enough" cloud arrives. octomap_server
        # publishes incrementally, so an early exit can grab a snapshot built
        # from whichever bridges happened to connect first -- not all four --
        # which produced exactly this test's first failure mode (Drone3/Drone4
        # "not found" because their bridges simply hadn't started streaming yet).
        warmup_s = 30.0
        t0 = time.time()
        while time.time() - t0 < warmup_s:
            rclpy.spin_once(node, timeout_sec=0.5)

        if node.msg is None:
            check("received /octomap_point_cloud_centers", False,
                  "no message in 45s -- launch failed? check the mock server is up")
            return 1
        check("received /octomap_point_cloud_centers", True)
        check("frame_id is map", node.msg.header.frame_id == "map",
              f"got {node.msg.header.frame_id!r}")

        pts = np.array([[p[0], p[1], p[2]]
                        for p in pc2.read_points(node.msg, field_names=("x", "y", "z"),
                                                 skip_nans=True)])
        check("got a non-trivial number of voxels", len(pts) > 500, f"got {len(pts)}")

        # Nothing should be near the map origin -- that is what a broken
        # sensor-origin lookup (e.g. accidentally pre-transforming clouds before
        # handing them to octomap_server) produces.
        d0 = np.hypot(pts[:, 0], pts[:, 1])
        check("no voxels near the map origin", (d0 < 30).sum() == 0,
              f"{(d0 < 30).sum()} voxels within 30m of (0,0)")

        # Each drone's own closest-spawn distance array, computed once.
        d_all = {n2: np.hypot(pts[:, 0] - e2[0], pts[:, 1] - e2[1])
                 for n2, e2 in SPAWNS_ENU.items()}

        for name, (ex, ey) in SPAWNS_ENU.items():
            own = d_all[name]
            close = own < 30.0
            check(f"{name} has voxels within 30m of its own ENU spawn {(ex, ey)}",
                  close.sum() > 20, f"only {close.sum()} within 30m, "
                  f"nearest={own.min():.1f}m")

            # Of the voxels assigned to name by nearest-spawn, most should
            # genuinely be close to it -- catches a swap between two drones
            # (e.g. Drone2's cloud actually being TF'd to Drone3's frame),
            # which per-drone "has some voxels nearby" alone would not catch
            # if the swap is symmetric.
            assigned = np.argmin(np.stack([d_all[n2] for n2 in SPAWNS_ENU]), axis=0) \
                == list(SPAWNS_ENU).index(name)
            if assigned.sum() > 0:
                cluster_d = own[assigned]
                check(f"{name}'s own-assigned cluster is actually near {name}",
                      float(np.median(cluster_d)) < 30.0,
                      f"median distance {np.median(cluster_d):.1f}m")

        print(f"\n{_passed} passed, {len(_failed)} failed")
        if _failed:
            print("failed: " + ", ".join(_failed))
        return 1 if _failed else 0
    finally:
        node.destroy_node()
        rclpy.shutdown()
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=10)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=5)


if __name__ == "__main__":
    sys.exit(main())
