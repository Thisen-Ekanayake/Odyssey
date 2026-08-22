#!/usr/bin/env python3
"""Numerical check of the AirSim(NED/FRD) <-> ROS(ENU/FLU) conversions.

A frame error here is silent: the map still looks like a map, ICP still
converges, and the mistake only shows up as a mirrored world nobody notices.
So rather than trusting the algebra, this compares the quaternion path against
the matrix path -- ``R(q_out)`` must equal ``R_ENU_NED @ R(q_in) @ R_FRD_FLU``
for random rotations -- and pins down the handful of cases with an obvious
physical answer (down is up, right is left, north is +y).

    ./scripts/ros_enter.sh python3 ros2_ws/src/airsim_swarm_bridge/test/test_frames.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from airsim_swarm_bridge.frames import (            # noqa: E402
    R_ENU_NED, R_FLU_FRD, FrameNames, frd_points_to_flu, frd_to_flu,
    ned_point_to_enu, ned_quat_to_enu, quat_multiply, quat_normalize,
)

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


def R_of(q: np.ndarray) -> np.ndarray:
    """(x, y, z, w) -> 3x3 rotation matrix."""
    x, y, z, w = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def main() -> int:
    print("matrix sanity")
    check("R_ENU_NED is a rotation (det=+1)", abs(np.linalg.det(R_ENU_NED) - 1) < 1e-12,
          f"det={np.linalg.det(R_ENU_NED)}")
    check("R_FLU_FRD is a rotation (det=+1)", abs(np.linalg.det(R_FLU_FRD) - 1) < 1e-12)
    check("R_ENU_NED is its own inverse", np.allclose(R_ENU_NED @ R_ENU_NED, np.eye(3)))
    check("R_FLU_FRD is its own inverse", np.allclose(R_FLU_FRD @ R_FLU_FRD, np.eye(3)))

    print("\npositions: the physical cases")
    # AirSim altitude is NEGATIVE up. A drone 5 m in the air is z = -5 in NED and
    # must publish as z = +5 in ENU -- this is the single most likely bug.
    check("5 m up (NED z=-5) -> ENU z=+5", ned_point_to_enu(0, 0, -5.0) == (0.0, 0.0, 5.0),
          f"got {ned_point_to_enu(0, 0, -5.0)}")
    check("on the ground stays at 0", ned_point_to_enu(0, 0, 0.0) == (0.0, 0.0, 0.0))
    check("10 m north -> ENU +y", ned_point_to_enu(10.0, 0, 0) == (0.0, 10.0, 0.0))
    check("10 m east  -> ENU +x", ned_point_to_enu(0, 10.0, 0) == (10.0, 0.0, 0.0))
    # Drone1 spawns at NED (150, 150); ENU keeps it on the same diagonal.
    check("Drone1 spawn (150,150,0)", ned_point_to_enu(150.0, 150.0, 0.0) == (150.0, 150.0, 0.0))
    # Drone2 at NED (90, -210) -> ENU (-210, 90): the sign must move with the axis.
    check("Drone2 spawn (90,-210,0) -> (-210,90,0)",
          ned_point_to_enu(90.0, -210.0, 0.0) == (-210.0, 90.0, 0.0),
          f"got {ned_point_to_enu(90.0, -210.0, 0.0)}")

    print("\nbody vectors: FRD -> FLU")
    check("forward is unchanged", frd_to_flu(1, 0, 0) == (1.0, 0.0, 0.0))
    check("right becomes -left", frd_to_flu(0, 1, 0) == (0.0, -1.0, 0.0))
    check("down becomes -up", frd_to_flu(0, 0, 1) == (0.0, 0.0, -1.0))
    # The LiDAR sits 10 cm ABOVE the body origin: z=-0.1 in FRD, +0.1 in FLU.
    check("lidar mount z=-0.1 (FRD) -> +0.1 (FLU)", frd_to_flu(0, 0, -0.1)[2] == 0.1)

    print("\nlidar clouds")
    pts = np.array([[1, 2, 3], [-4, 5, -6]], dtype=np.float32)
    got = frd_points_to_flu(pts)
    check("cloud flip matches the scalar helper",
          np.allclose(got, [[1, -2, -3], [-4, -5, 6]]), f"got {got.tolist()}")
    check("cloud stays float32", got.dtype == np.float32)
    check("flat input is reshaped to (N,3)",
          frd_points_to_flu(np.arange(9, dtype=np.float32)).shape == (3, 3))
    check("input is not mutated", np.allclose(pts, [[1, 2, 3], [-4, 5, -6]]))

    print("\norientation: quaternion path == matrix path")
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(500):
        q_in = quat_normalize(rng.normal(size=4))
        q_out = ned_quat_to_enu(*q_in)
        lhs = R_of(q_out)
        rhs = R_ENU_NED @ R_of(q_in) @ R_FLU_FRD
        worst = max(worst, float(np.abs(lhs - rhs).max()))
    check("500 random rotations agree with the matrix composition", worst < 1e-9,
          f"worst elementwise error {worst:.2e}")

    # Identity in AirSim = nose north, level. In ENU that is nose along +y,
    # i.e. yaw = +90 degrees about z.
    q_id = ned_quat_to_enu(0.0, 0.0, 0.0, 1.0)
    fwd = R_of(q_id) @ np.array([1.0, 0.0, 0.0])
    check("identity NED attitude points along ENU +y (north)",
          np.allclose(fwd, [0.0, 1.0, 0.0], atol=1e-9), f"forward={fwd}")
    up = R_of(q_id) @ np.array([0.0, 0.0, 1.0])
    check("body up maps to world up", np.allclose(up, [0.0, 0.0, 1.0], atol=1e-9),
          f"up={up}")

    # A 90-degree NED yaw (turn to face east) must come out as ENU yaw about +z.
    s, c = np.sin(np.pi / 4), np.cos(np.pi / 4)
    q_east = ned_quat_to_enu(0.0, 0.0, s, c)
    fwd_e = R_of(q_east) @ np.array([1.0, 0.0, 0.0])
    check("NED yaw +90 (facing east) -> ENU +x",
          np.allclose(fwd_e, [1.0, 0.0, 0.0], atol=1e-9), f"forward={fwd_e}")

    check("output quaternions stay unit",
          abs(np.linalg.norm(ned_quat_to_enu(0.1, 0.2, 0.3, 0.4)) - 1) < 1e-12)
    check("degenerate all-zero quaternion falls back to identity",
          np.allclose(quat_normalize(np.zeros(4)), [0, 0, 0, 1]))

    print("\nframe naming")
    f = FrameNames("Drone3")
    check("namespace is lowercased", f.ns == "drone3")
    check("base_link id", f.base_link == "drone3/base_link")
    check("optical frame id", f.cam_left_optical == "drone3/stereo_left_optical")
    check("map is shared", FrameNames.MAP == "map")

    print(f"\n{_passed} passed, {len(_failed)} failed")
    if _failed:
        print("failed: " + ", ".join(_failed))
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
