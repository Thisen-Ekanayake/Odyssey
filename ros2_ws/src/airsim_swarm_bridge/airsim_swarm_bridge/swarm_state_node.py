#!/usr/bin/env python3
"""Publishes the swarm's shared position picture and the formation geometry.

Two outputs, both in the shared ``map`` (ENU) frame:

``/swarm/state`` (:class:`airsim_swarm_msgs.msg.SwarmState`)
    Every drone's world position, GNSS fix and velocity, plus the centroid and
    the full pairwise-distance matrix. This is the ROS-native form of what
    ``swarm/swarm_comms.py::SwarmPositions.refresh()`` already returns, and it is
    produced by calling that class rather than by re-deriving it -- including its
    local->world offset calibration, which exists because AirSim reports
    multi-vehicle positions in each vehicle's own reference frame.

``/swarm/formation_markers`` (``visualization_msgs/MarkerArray``)
    The formation quadrilateral cut into four squares: the four perimeter edges
    plus a line from the diagonal intersection to each edge's midpoint. This is
    the same construction ``swarm/swarm_lines_viz.py`` draws in Open3D, corner
    ordering included -- sorting corners by angle around the centroid is what
    stops the perimeter from self-crossing into a bowtie for this spawn layout.
    The corner-to-corner diagonals are computed (the centre point needs them) but
    deliberately not drawn, since drawing them would cut each square into triangles.

One RPC connection, owned by this node alone.
"""
from __future__ import annotations

import math

import rclpy
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import Point
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from airsim_swarm_msgs.msg import DroneState, SwarmState

from . import frames
from .airsim_api import MultirotorClient
from .rpc import RpcError

PERIMETER_COLOR = ColorRGBA(r=0.20, g=0.75, b=1.00, a=1.0)   # cyan
SEGMENT_COLOR = ColorRGBA(r=1.00, g=0.70, b=0.15, a=1.0)     # orange
# Per-drone colours, matching the LiDAR map colouring in the viz launch.
DRONE_COLORS = [
    ColorRGBA(r=0.95, g=0.26, b=0.21, a=1.0),   # red
    ColorRGBA(r=0.30, g=0.69, b=0.31, a=1.0),   # green
    ColorRGBA(r=0.25, g=0.46, b=0.95, a=1.0),   # blue
    ColorRGBA(r=1.00, g=0.76, b=0.03, a=1.0),   # amber
]


class SwarmStateNode(Node):

    def __init__(self) -> None:
        super().__init__("swarm_state")

        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 41451)
        self.declare_parameter("rate", 10.0)
        self.declare_parameter("drones", [""])
        self.declare_parameter("marker_lifetime", 0.0)

        g = lambda n: self.get_parameter(n).value            # noqa: E731

        from .repo import import_swarm_module

        self._comms = import_swarm_module("swarm_comms")
        drones = [d for d in (g("drones") or []) if d]
        self.drones: list[str] = drones or list(self._comms.DRONES)

        self.client = MultirotorClient(g("host"), int(g("port")))
        self.get_logger().info(f"connecting to AirSim for {len(self.drones)} drones")
        self.client.confirmConnection(120.0)

        # SwarmPositions snapshots each vehicle's raw local reading at construction
        # and uses it as this run's zero point, so it must be built before the
        # swarm moves -- the same constraint the standalone scripts work under.
        self.swarm = self._comms.SwarmPositions(self.client, self.drones)
        self.get_logger().info(
            "calibrated local->world offsets: "
            + ", ".join(f"{d}=({self.swarm.offset(d)[0]:+.1f},{self.swarm.offset(d)[1]:+.1f})"
                        for d in self.drones))

        self.pub_state = self.create_publisher(SwarmState, "/swarm/state", 10)
        self.pub_markers = self.create_publisher(MarkerArray, "/swarm/formation_markers", 1)

        self._lifetime = float(g("marker_lifetime"))
        self.create_timer(1.0 / max(0.1, float(g("rate"))), self._tick)

    # -- main loop ---------------------------------------------------------

    def _tick(self) -> None:
        try:
            positions = self.swarm.refresh()
            # refresh() gives world x/y; z has to come from the kinematics, and the
            # formation drawing needs it so the quadrilateral sits at flight altitude.
            zs = {d: self.client.getMultirotorState(d).kinematics_estimated.position.z_val
                  for d in self.drones}
            states = {d: self.client.getMultirotorState(d) for d in self.drones}
        except (RpcError, OSError) as exc:
            self.get_logger().warning(f"swarm poll failed ({exc})", throttle_duration_sec=5.0)
            return

        stamp = self.get_clock().now().to_msg()

        # World NED -> ENU for every drone, once.
        enu: dict[str, tuple[float, float, float]] = {}
        for d in self.drones:
            p = positions[d]
            enu[d] = frames.ned_point_to_enu(p.world_x, p.world_y, zs[d])

        self._publish_state(stamp, positions, enu, states)
        self._publish_markers(stamp, enu)

    def _publish_state(self, stamp, positions, enu, states) -> None:
        msg = SwarmState()
        msg.header.stamp = stamp
        msg.header.frame_id = frames.FrameNames.MAP

        for d in self.drones:
            k = states[d].kinematics_estimated
            ds = DroneState()
            ds.vehicle_name = d
            ds.pose.position.x, ds.pose.position.y, ds.pose.position.z = enu[d]
            q = frames.ned_quat_to_enu(k.orientation.x_val, k.orientation.y_val,
                                       k.orientation.z_val, k.orientation.w_val)
            (ds.pose.orientation.x, ds.pose.orientation.y,
             ds.pose.orientation.z, ds.pose.orientation.w) = (
                float(q[0]), float(q[1]), float(q[2]), float(q[3]))
            vx, vy, vz = frames.frd_to_flu(k.linear_velocity.x_val,
                                           k.linear_velocity.y_val,
                                           k.linear_velocity.z_val)
            ds.velocity.x, ds.velocity.y, ds.velocity.z = vx, vy, vz
            ds.latitude = positions[d].latitude
            ds.longitude = positions[d].longitude
            ds.altitude = positions[d].altitude
            ds.armed = bool(getattr(states[d], "landed_state", 0))
            ds.has_collided = bool(states[d].collision.has_collided)
            msg.drones.append(ds)

        n = len(self.drones)
        msg.centroid = Point(
            x=sum(enu[d][0] for d in self.drones) / n,
            y=sum(enu[d][1] for d in self.drones) / n,
            z=sum(enu[d][2] for d in self.drones) / n,
        )
        # Horizontal separation is what matters for collision/formation logic,
        # matching SwarmPositions.distance(), which also ignores altitude.
        msg.distance_matrix = [
            math.hypot(enu[a][0] - enu[b][0], enu[a][1] - enu[b][1])
            for a in self.drones for b in self.drones
        ]
        self.pub_state.publish(msg)

    # -- formation geometry ------------------------------------------------

    def _publish_markers(self, stamp, enu) -> None:
        if len(self.drones) != 4:
            return          # the four-square construction is specific to a quad

        corners, center, midpoints = formation_geometry(
            [enu[d] for d in self.drones])

        arr = MarkerArray()
        arr.markers.append(self._line_marker(
            "perimeter", 0, stamp, PERIMETER_COLOR, 1.2,
            [corners[i] for i in (0, 1, 1, 2, 2, 3, 3, 0)]))

        # Centre-to-midpoint cuts: these are the lines that turn the quadrilateral
        # into four squares. The corner-to-corner diagonals stay undrawn on purpose.
        seg_pts: list[tuple[float, float, float]] = []
        for mp in midpoints:
            seg_pts.extend([center, mp])
        arr.markers.append(self._line_marker(
            "segments", 1, stamp, SEGMENT_COLOR, 0.8, seg_pts))

        for i, d in enumerate(self.drones):
            arr.markers.append(self._sphere_marker(
                "drones", 2 + i, stamp, DRONE_COLORS[i % len(DRONE_COLORS)], enu[d], d))
        self.pub_markers.publish(arr)

    def _base_marker(self, ns: str, mid: int, stamp) -> Marker:
        m = Marker()
        m.header.stamp = stamp
        m.header.frame_id = frames.FrameNames.MAP
        m.ns = ns
        m.id = mid
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        if self._lifetime > 0:
            m.lifetime.sec = int(self._lifetime)
            m.lifetime.nanosec = int((self._lifetime % 1) * 1e9)
        return m

    def _line_marker(self, ns, mid, stamp, color, width, pts) -> Marker:
        m = self._base_marker(ns, mid, stamp)
        # LINE_LIST, not LINE_STRIP: the points come in disjoint pairs.
        m.type = Marker.LINE_LIST
        m.scale.x = float(width)
        m.color = color
        m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in pts]
        return m

    def _sphere_marker(self, ns, mid, stamp, color, p, label) -> Marker:
        m = self._base_marker(ns, mid, stamp)
        m.type = Marker.SPHERE
        m.scale.x = m.scale.y = m.scale.z = 3.0
        m.color = color
        m.pose.position.x, m.pose.position.y, m.pose.position.z = (
            float(p[0]), float(p[1]), float(p[2]))
        m.text = label
        return m

    def destroy_node(self) -> bool:
        try:
            self.client.close()
        except Exception:                       # noqa: BLE001
            pass
        return super().destroy_node()


# --------------------------------------------------------------------------
# geometry (shared with swarm_edge_to_center_viz's target drawing)
# --------------------------------------------------------------------------

def formation_geometry(points):
    """4 drone positions -> (corners, diagonal-intersection, edge midpoints).

    Corners come back ordered by angle around the centroid. That ordering is the
    whole trick: ``DRONES`` lists the drones in spawn order, and for this swarm's
    layout tracing the perimeter in that order self-crosses into a bowtie. Ported
    from ``swarm/swarm_lines_viz.py``, which does the same thing for its Open3D
    ``LineSet``.
    """
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    corners = sorted(points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

    center = _segment_intersection(corners[0], corners[2], corners[1], corners[3])
    midpoints = [_midpoint(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    return corners, center, midpoints


def _segment_intersection(p1, p2, p3, p4):
    """Where segment p1-p2 crosses p3-p4; the 4-point average if near-parallel."""
    x1, y1, _ = p1
    x2, y2, _ = p2
    x3, y3, _ = p3
    x4, y4, _ = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-9:
        n = 4.0
        return ((x1 + x2 + x3 + x4) / n, (y1 + y2 + y3 + y4) / n,
                (p1[2] + p2[2] + p3[2] + p4[2]) / n)
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1), p1[2] + t * (p2[2] - p1[2]))


def _midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0, (p1[2] + p2[2]) / 2.0)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = SwarmStateNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, SystemExit):
        pass
    except RpcError as exc:
        print(f"swarm_state: {exc}")
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
