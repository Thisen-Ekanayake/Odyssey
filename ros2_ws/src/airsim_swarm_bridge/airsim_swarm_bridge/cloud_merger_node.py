#!/usr/bin/env python3
"""Relays every drone's LiDAR cloud onto one shared topic, for a real merged map.

    /drone1/lidar/points  \\
    /drone2/lidar/points   \\___ /swarm/merged_points ___ octomap_server ___ one
    /drone3/lidar/points   /                                                 shared
    /drone4/lidar/points  /                                                  OctoMap

This is a **relay, not a transform**: each message is republished with its
original ``header.frame_id`` (``droneN/lidar_link``) untouched. That is
deliberate, not a shortcut -- ``octomap_server`` already does its own per-message
``tf2_ros::MessageFilter`` lookup and uses the resulting transform's translation
as the ray-tracing sensor origin (confirmed against the installed binary: it
links ``tf2_ros::MessageFilter<PointCloud2>`` and ``pcl_ros::transformPointCloud``,
the standard octomap_server pattern). Pre-transforming to ``map`` in Python would
throw that away -- every ray would then appear to originate from the map origin
instead of each drone's actual LiDAR position, and free-space carving would be
wrong even though occupied points would still land in roughly the right place.

Four independently-moving, correctly-TF'd frame ids arriving on one topic is no
different to octomap_server than one frame whose pose changes over time, which
is the case it is built for -- so no synchronisation across drones is needed
either. Points simply accumulate into the shared tree as each drone's sweep
arrives, at whatever rate it arrives.

Ground truth pose is what places each cloud in `map` (via the TF chain
`bridge_node` already publishes), not a second estimator -- this is a
cooperative MAPPING demo, not a cooperative SLAM one. Contrast
`slam_lidar.launch.py`, where `icp_odometry` estimates the pose instead.
"""
from __future__ import annotations

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud2

SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=5,
)


class CloudMergerNode(Node):

    def __init__(self) -> None:
        super().__init__("cloud_merger")

        self.declare_parameter("drones", ["Drone1", "Drone2", "Drone3", "Drone4"])
        self.declare_parameter("output_topic", "/swarm/merged_points")

        drones = [d for d in self.get_parameter("drones").value if d]
        out_topic = self.get_parameter("output_topic").value

        self.pub = self.create_publisher(PointCloud2, out_topic, SENSOR_QOS)
        self._count = {d: 0 for d in drones}

        for d in drones:
            topic = f"/{d.lower()}/lidar/points"
            # A per-drone closure default (d=d) so each callback relays its own
            # count, not whichever `d` the loop happened to end on.
            self.create_subscription(
                PointCloud2, topic,
                lambda msg, d=d: self._relay(d, msg), SENSOR_QOS)
            self.get_logger().info(f"relaying {topic} -> {out_topic}")

        self.create_timer(5.0, self._report)

    def _relay(self, drone: str, msg: PointCloud2) -> None:
        # Republish unchanged: same frame_id, same stamp. octomap_server's own
        # TF lookup is what places these points and determines the ray origin --
        # see the module docstring for why that must not be pre-empted here.
        self.pub.publish(msg)
        self._count[drone] += 1

    def _report(self) -> None:
        total = sum(self._count.values())
        if total == 0:
            self.get_logger().warning(
                "no clouds relayed in the last 5s -- is bridge.launch.py running "
                "with lidar:=true for these drones?")
            return
        parts = ", ".join(f"{d}={n}" for d, n in self._count.items())
        self.get_logger().info(f"relayed {total} clouds/5s ({parts})")
        for d in self._count:
            self._count[d] = 0


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = CloudMergerNode()
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
