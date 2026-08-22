#!/usr/bin/env python3
"""Records an odometry topic to a TUM trajectory file.

This is the hand-off between the two halves of the comparison. rtabmap estimates
a trajectory inside the container; ``slam/evaluate.py`` computes ATE/RPE on the
host. Writing TUM -- the format ``slam/geometry.py::write_tum`` already produces
and ``read_tum`` already consumes -- means ``tools/compare_ros_slam.py`` can score
a ROS estimate with the *same* code that scores ``slam/lidar_slam.py``, so the
numbers are comparable rather than merely adjacent.

    timestamp tx ty tz qx qy qz qw

Poses are written in whatever frame the subscribed topic uses, with one
deliberate exception: ``--to-ned`` converts back from ROS ENU into the AirSim NED
frame the recorded ground truth is in. That is on by default, because the point
of the file is to be compared against ``datasets/<condition>/groundtruth.txt``.

    ros2 run airsim_swarm_bridge traj_recorder --ros-args \
        -p topic:=/odom -p out:=results/ros/clear_rtabmap.txt
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import rclpy
from rclpy.executors import ExternalShutdownException
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from .frames import enu_point_to_ned, quat_multiply, quat_normalize

# Inverse of the world/body pair used on the way out (see frames.py). Both
# rotations are their own inverse, so the same quaternions come back the other way.
_R = 1.0 / math.sqrt(2.0)
_Q_NED_ENU = (_R, _R, 0.0, 0.0)
_Q_FLU_FRD = (1.0, 0.0, 0.0, 0.0)


class TrajectoryRecorder(Node):

    def __init__(self) -> None:
        super().__init__("traj_recorder")

        self.declare_parameter("topic", "/odom")
        self.declare_parameter("out", "results/ros/trajectory.txt")
        self.declare_parameter("to_ned", True)
        self.declare_parameter("header", "")
        self.declare_parameter("flush_every", 50)

        g = lambda n: self.get_parameter(n).value            # noqa: E731
        self.to_ned: bool = bool(g("to_ned"))
        self._flush_every = max(1, int(g("flush_every")))
        self._n = 0

        out = Path(g("out"))
        if not out.is_absolute():
            from .repo import repo_root
            out = repo_root() / out
        out.parent.mkdir(parents=True, exist_ok=True)
        self._path = out

        # Written incrementally rather than at shutdown: these runs are long, and
        # a crash three minutes from the end should not cost the whole trajectory.
        self._fh = open(out, "w", buffering=1)
        for line in (g("header") or "").strip().splitlines():
            self._fh.write(f"# {line}\n")
        self._fh.write(f"# source topic: {g('topic')}\n")
        self._fh.write(f"# frame: {'AirSim NED (converted from ROS ENU)' if self.to_ned else 'ROS ENU'}\n")
        self._fh.write("# timestamp tx ty tz qx qy qz qw\n")

        self.create_subscription(
            Odometry, str(g("topic")), self._on_odom,
            QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                       history=QoSHistoryPolicy.KEEP_LAST, depth=50))
        self.get_logger().info(f"recording {g('topic')} -> {out}")

    def _on_odom(self, msg: Odometry) -> None:
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation

        if self.to_ned:
            x, y, z = enu_point_to_ned(p.x, p.y, p.z)
            # q_ned_frd = q_NED_ENU * q_enu_flu * q_FLU_FRD
            q = quat_normalize(quat_multiply(
                quat_multiply(_Q_NED_ENU, quat_normalize((o.x, o.y, o.z, o.w))),
                _Q_FLU_FRD))
            qx, qy, qz, qw = q
        else:
            x, y, z = p.x, p.y, p.z
            qx, qy, qz, qw = o.x, o.y, o.z, o.w

        self._fh.write(f"{t:.9f} {x:.6f} {y:.6f} {z:.6f} "
                       f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")
        self._n += 1
        if self._n % self._flush_every == 0:
            self._fh.flush()
            os.fsync(self._fh.fileno())

    def destroy_node(self) -> bool:
        try:
            self._fh.flush()
            self._fh.close()
            self.get_logger().info(f"wrote {self._n} poses to {self._path}")
        except Exception:                                   # noqa: BLE001
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = TrajectoryRecorder()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, SystemExit):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
