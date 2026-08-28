#!/usr/bin/env python3
"""Publishes one AirSim drone's sensors, odometry and TF onto ROS 2.

One node, one drone, one RPC connection. That split is not incidental:
msgpack-rpc has no thread safety (a rule this repo already follows in
``flight/``), and ``simGetImages`` is slow enough that four drones sharing a
socket would serialise into a stall. Running four processes lets the images
overlap and keeps a wedged camera on one drone from freezing the others.

Rates are independent timers rather than one loop, because the sensors want very
different periods -- IMU at 100 Hz, LiDAR at 10 Hz, stereo at 5 Hz (image RPC is
by far the most expensive call). ``settings.json`` is the authority on what the
rig *is*; :mod:`slam.config` mirrors it and is what this node reads.

Published under the drone's namespace (``Drone1`` -> ``/drone1``):

    /drone1/imu                      sensor_msgs/Imu
    /drone1/lidar/points             sensor_msgs/PointCloud2
    /drone1/gps                      sensor_msgs/NavSatFix
    /drone1/stereo/left/image_raw    sensor_msgs/Image      (+ camera_info)
    /drone1/stereo/right/image_raw   sensor_msgs/Image      (+ camera_info)
    /drone1/odom_gt                  nav_msgs/Odometry      ground truth
    /drone1/path_gt                  nav_msgs/Path

plus TF ``map -> <ns>/odom -> <ns>/base_link`` and the static sensor mounts.
"""
from __future__ import annotations

import math
import threading

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image, Imu, NavSatFix, PointCloud2, PointField
from std_msgs.msg import Header
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from . import frames
from .airsim_api import MultirotorClient
from .rpc import RpcError

# Best-effort sensor QoS: a dropped LiDAR sweep is better than a growing queue
# when the estimator falls behind. Matches the ROS convention for live sensors.
SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=5,
)
# Latched: a late-joining RViz still needs the static mounts and camera_info.
LATCHED_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    depth=1,
)


class AirSimBridge(Node):

    def __init__(self) -> None:
        super().__init__("airsim_bridge")

        self.declare_parameter("vehicle_name", "Drone1")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 41451)
        self.declare_parameter("imu_rate", 100.0)
        self.declare_parameter("lidar_rate", 10.0)
        self.declare_parameter("stereo_rate", 5.0)
        self.declare_parameter("odom_rate", 50.0)
        self.declare_parameter("enable_lidar", True)
        self.declare_parameter("enable_stereo", True)
        self.declare_parameter("enable_gps", True)
        # Who owns TF's odom -> base_link edge. True (the default) means this
        # bridge publishes ground truth there, which is what you want for a pure
        # sensor/visualisation run. Launch files that start rtabmap set it False:
        # icp_odometry publishes that edge itself, and two publishers on one TF
        # edge is a corrupt tree. Ground truth then goes to <ns>/base_link_gt so
        # it stays visible for comparison without fighting the estimator.
        self.declare_parameter("gt_owns_base_link", True)
        self.declare_parameter("path_max_poses", 5000)
        # Stamp everything from AirSim's clock rather than the node's. See _stamp:
        # sensor payloads carry a sim timestamp and TF does not, so mixing the two
        # is what silently breaks every downstream tf2_ros::MessageFilter. Set
        # False to put the whole node on wall time instead (self-consistent too,
        # but then IMU dt no longer reflects sim time, which matters whenever the
        # sim is not running at 1x).
        self.declare_parameter("use_airsim_time", True)
        # Spawn offset in world NED, from settings.json. Left at NaN it is looked
        # up from swarm_comms.SPAWNS, which is the same table settings.json uses.
        self.declare_parameter("spawn_x", float("nan"))
        self.declare_parameter("spawn_y", float("nan"))

        g = lambda n: self.get_parameter(n).value          # noqa: E731
        self.vehicle: str = g("vehicle_name")
        self.frames = frames.FrameNames(self.vehicle)
        self.enable_lidar: bool = g("enable_lidar")
        self.enable_stereo: bool = g("enable_stereo")
        self.enable_gps: bool = g("enable_gps")
        self.gt_owns_base_link: bool = g("gt_owns_base_link")
        self.path_max: int = int(g("path_max_poses"))
        self.use_airsim_time: bool = g("use_airsim_time")
        # sim_clock - node_clock, in ns. Refreshed by every sensor read; 0 until
        # the first one arrives, which degrades _stamp() to plain wall time.
        self._clock_offset_ns: int = 0

        self._cfg = self._load_rig_config()

        # --- connect ------------------------------------------------------
        self.client = MultirotorClient(g("host"), int(g("port")))
        self.get_logger().info(
            f"connecting to AirSim at {g('host')}:{g('port')} for {self.vehicle} "
            f"(first boot compiles shaders, this can take ~60 s)")
        self.client.confirmConnection(120.0)
        self._check_vehicle_exists()

        # --- world offset -------------------------------------------------
        self._offset = self._calibrate_offset(g("spawn_x"), g("spawn_y"))
        self.get_logger().info(
            f"{self.vehicle}: local->world NED offset "
            f"({self._offset[0]:+.2f}, {self._offset[1]:+.2f}) m")

        # --- publishers ---------------------------------------------------
        # Topic names are built from the vehicle name, not from `~` or a launch
        # namespace. `~` would fold in the NODE name (/airsim_bridge/imu), and a
        # launch namespace would not apply when the node is run bare. Deriving
        # both topics and frame ids from vehicle_name keeps /drone1/lidar/points
        # and drone1/lidar_link agreeing however the node was started.
        ns = f"/{self.frames.ns}"
        self.pub_imu = self.create_publisher(Imu, f"{ns}/imu", SENSOR_QOS)
        self.pub_odom = self.create_publisher(Odometry, f"{ns}/odom_gt", SENSOR_QOS)
        self.pub_path = self.create_publisher(Path, f"{ns}/path_gt", 1)
        if self.enable_lidar:
            self.pub_cloud = self.create_publisher(
                PointCloud2, f"{ns}/lidar/points", SENSOR_QOS)
        if self.enable_gps:
            self.pub_gps = self.create_publisher(NavSatFix, f"{ns}/gps", SENSOR_QOS)
        if self.enable_stereo:
            from sensor_msgs.msg import CameraInfo
            self.pub_left = self.create_publisher(
                Image, f"{ns}/stereo/left/image_raw", SENSOR_QOS)
            self.pub_right = self.create_publisher(
                Image, f"{ns}/stereo/right/image_raw", SENSOR_QOS)
            self.pub_left_info = self.create_publisher(
                CameraInfo, f"{ns}/stereo/left/camera_info", LATCHED_QOS)
            self.pub_right_info = self.create_publisher(
                CameraInfo, f"{ns}/stereo/right/camera_info", LATCHED_QOS)

        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_static = StaticTransformBroadcaster(self)
        self._publish_static_transforms()

        self._path = Path()
        self._path.header.frame_id = frames.FrameNames.MAP
        # One lock around every RPC use: the timers run on the executor and would
        # otherwise interleave two calls on a socket that allows exactly one.
        self._rpc_lock = threading.Lock()
        self._warned: set[str] = set()

        # --- timers -------------------------------------------------------
        self._cov_orient, self._cov_gyro, self._cov_accel = \
            self._imu_covariances(float(g("imu_rate")))
        self._timer(g("imu_rate"), self._tick_imu, "imu")
        self._timer(g("odom_rate"), self._tick_odom, "odom")
        if self.enable_lidar:
            self._timer(g("lidar_rate"), self._tick_lidar, "lidar")
        if self.enable_stereo:
            self._timer(g("stereo_rate"), self._tick_stereo, "stereo")
        if self.enable_gps:
            self._timer(min(10.0, float(g("odom_rate"))), self._tick_gps, "gps")

        self.get_logger().info(
            f"bridging {self.vehicle} -> /{self.frames.ns}  "
            f"(lidar={self.enable_lidar} stereo={self.enable_stereo} gps={self.enable_gps})")

    # -- setup helpers -----------------------------------------------------

    def _timer(self, rate_hz: float, cb, label: str) -> None:
        if rate_hz and rate_hz > 0:
            self.create_timer(1.0 / float(rate_hz), cb)
        else:
            self.get_logger().info(f"{label} publishing disabled (rate <= 0)")

    def _load_rig_config(self):
        """Read the rig from ``slam/config.py`` -- the repo's mirror of settings.json.

        Re-deriving intrinsics here would create a second place to update when
        the camera block changes, which ``tools/probe_setup.py`` is specifically
        there to prevent.
        """
        from .repo import add_repo_to_path

        add_repo_to_path()
        from slam import config as cfg          # noqa: PLC0415
        return cfg

    def _check_vehicle_exists(self) -> None:
        try:
            names = self.client.listVehicles()
        except RpcError:
            return                              # older builds lack listVehicles
        if names and self.vehicle not in names:
            raise SystemExit(
                f"{self.vehicle!r} is not in the running simulator (has: {names}). "
                f"settings.json is only read at startup -- restart scripts/run_swarm.sh "
                f"after editing it.")

    def _calibrate_offset(self, spawn_x: float, spawn_y: float) -> tuple[float, float]:
        """(dx, dy) added to a raw local NED reading to land in the world frame.

        AirSim reports each vehicle's position in its own local reference, and for
        non-Drone1 vehicles that reference is *not* its spawn point --
        ``swarm/swarm_comms.py`` documents this and works around it by
        snapshotting the raw reading at startup and treating it as this run's
        zero. The same formula is used here: world = spawn + (local - baseline).

        It only holds if the drone has not moved yet, which is why the bridge is
        started before takeoff.
        """
        if math.isnan(spawn_x) or math.isnan(spawn_y):
            try:
                from .repo import import_swarm_module
                spawns = import_swarm_module("swarm_comms").SPAWNS
                spawn_x, spawn_y = spawns[self.vehicle]
            except (ImportError, KeyError) as exc:
                self.get_logger().warning(
                    f"no spawn offset for {self.vehicle} ({exc}); assuming (0, 0). "
                    f"Multi-drone maps will not share a frame.")
                spawn_x = spawn_y = 0.0

        pos = self.client.getMultirotorState(self.vehicle).kinematics_estimated.position
        return (float(spawn_x) - pos.x_val, float(spawn_y) - pos.y_val)

    def _publish_static_transforms(self) -> None:
        """Sensor mounts (constant) and ``map -> odom`` (this run's calibration)."""
        f, cfg = self.frames, self._cfg
        # tf2 treats static transforms as valid at every time and ignores this
        # stamp, so it is not load-bearing -- but route it through _stamp() anyway
        # so there is exactly one place in this node that decides what time is.
        now = self._stamp()
        out = []

        # map -> odom: the spawn offset, converted NED -> ENU. Orientation is
        # identity because both frames are world-aligned; only the origin moves.
        ox, oy = self._offset
        # The offset is a delta between two world-NED points, so it converts with
        # the same swap as a position: (x_e, y_n) = (y_ned, x_ned).
        out.append(self._tf(f.MAP, f.odom, (oy, ox, 0.0), (0.0, 0.0, 0.0, 1.0), now))

        # LiDAR: settings.json puts it at NED (0, 0, -0.1) with no rotation, so in
        # the FLU base_link it is 10 cm UP and axis-aligned.
        lx, ly, lz = frames.frd_to_flu(*cfg.T_BODY_LIDAR[:3, 3])
        out.append(self._tf(f.base_link, f.lidar, (lx, ly, lz), (0.0, 0.0, 0.0, 1.0), now))
        out.append(self._tf(f.base_link, f.imu, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), now))

        # Stereo: slam.config stores body<-optical mounts. The link frame is the
        # FLU-ified mount point; the optical frame carries ROS's z-forward
        # convention as a fixed child rotation, which is what image pipelines expect.
        for T, link, optical in ((cfg.T_BODY_CAM_LEFT, f.cam_left, f.cam_left_optical),
                                 (cfg.T_BODY_CAM_RIGHT, f.cam_right, f.cam_right_optical)):
            p = frames.frd_to_flu(*T[:3, 3])
            out.append(self._tf(f.base_link, link, p, (0.0, 0.0, 0.0, 1.0), now))
            # FLU -> optical (x right, y down, z forward): -90 deg about z then
            # -90 about x. As a quaternion that is (-0.5, 0.5, -0.5, 0.5).
            out.append(self._tf(link, optical, (0.0, 0.0, 0.0),
                                (-0.5, 0.5, -0.5, 0.5), now))

        self.tf_static.sendTransform(out)

    @staticmethod
    def _tf(parent: str, child: str, t, q, stamp) -> TransformStamped:
        m = TransformStamped()
        m.header.stamp = stamp
        m.header.frame_id = parent
        m.child_frame_id = child
        m.transform.translation.x = float(t[0])
        m.transform.translation.y = float(t[1])
        m.transform.translation.z = float(t[2])
        m.transform.rotation.x = float(q[0])
        m.transform.rotation.y = float(q[1])
        m.transform.rotation.z = float(q[2])
        m.transform.rotation.w = float(q[3])
        return m

    # -- time --------------------------------------------------------------

    def _stamp(self, sim_time_ns=None):
        """One clock for everything this node publishes: AirSim's.

        Sensor payloads carry a sim timestamp; TF and odometry do not. Stamping
        the first from AirSim's clock and the second from the node's is a silent,
        total failure whenever the sim is not running at 1x wall speed -- and with
        four drones it routinely runs at ~0.5x. The sensor stamps then fall further
        and further behind the TF cache, and every consumer built on
        ``tf2_ros::MessageFilter`` (octomap_server, rtabmap, icp_odometry) drops
        100% of messages with only an INFO line to show for it. That is exactly
        what ``cooperative_mapping.launch.py`` was doing: 46 of 46 clouds dropped
        for 'the timestamp on the message is earlier than all the data in the
        transform cache', an empty octree, and no error anywhere.

        So: readings that come with a sim timestamp use it *and* record the
        sim-minus-node offset; readings that do not (odometry, TF) project the
        node clock through that offset. Projecting rather than reusing the last
        cached sim stamp is deliberate -- TF is published faster than the sensors
        that source the offset, and reusing a cached value would emit repeated
        identical stamps on a moving transform, which tf2 cannot interpolate.

        The projection over-runs the true sim clock slightly between refreshes
        (it advances at wall rate, the sim does not), so a refresh can step the
        stamp back by up to (1 - rate) x the sensor period -- ~5 ms at 100 Hz IMU.
        tf2 sorts out-of-order inserts, and this is far below the LiDAR period, so
        it is harmless; keeping TF marginally *ahead* of the sensor data is in any
        case what MessageFilter wants, since it interpolates but will not
        extrapolate.
        """
        from builtin_interfaces.msg import Time

        now_ns = self.get_clock().now().nanoseconds
        if not self.use_airsim_time:
            ns = now_ns
        elif sim_time_ns:
            ns = int(sim_time_ns)
            self._clock_offset_ns = ns - now_ns
        else:
            ns = now_ns + self._clock_offset_ns
        # builtin_interfaces/Time is unsigned; a sim clock based at 0 rather than
        # the epoch would otherwise produce a negative stamp here.
        ns = max(0, ns)
        return Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)

    def _guard(self, what: str, exc: Exception) -> None:
        """Log an RPC failure once per kind, then keep going.

        A camera that returns a short buffer for a few frames while a render
        target spins up is normal; tearing the node down for it is not.
        """
        if what not in self._warned:
            self._warned.add(what)
            self.get_logger().warning(f"{what} failed ({exc}); will keep retrying quietly")

    # -- timers ------------------------------------------------------------

    def _imu_covariances(self, imu_rate_hz: float):
        """Per-sample IMU variances, derived from the noise block in settings.json.

        ``robot_localization`` and rtabmap's IMU handling weight by these, and a
        zero covariance means "unknown" in ROS -- leaving them empty silently
        mis-tunes both, so they are computed rather than guessed.

        AirSim states its IMU noise as random walks:

            AngularRandomWalk    deg / sqrt(hour)     (0.30 in settings.json)
            VelocityRandomWalk   m/s / sqrt(hour)     (0.24 in settings.json)

        Converting a random walk to a noise density is a change of units only --
        divide by sqrt(3600 s) = 60 -- and the discrete per-sample variance of a
        white noise of density N sampled at f is then N^2 * f:

            gyro  N = 0.30 * (pi/180) / 60 = 8.7e-5 rad/s/sqrt(Hz)
                  var at 100 Hz            = 7.6e-7 (rad/s)^2
            accel N = 0.24 / 60            = 4.0e-3 m/s^2/sqrt(Hz)
                  var at 100 Hz            = 1.6e-3 (m/s^2)^2

        Orientation is not a measured quantity here -- AirSim reports the true
        attitude -- but publishing 0 would claim it is exact, so it gets a small
        nominal variance instead.
        """
        arw_deg_sqrt_hr, vrw_sqrt_hr = 0.30, 0.24          # settings.json defaults
        try:
            import json

            from .repo import settings_json_path

            sensors = (json.loads(settings_json_path().read_text())
                       .get("Vehicles", {}).get(self.vehicle, {}).get("Sensors", {}))
            imu = sensors.get(self._cfg.IMU_NAME, {})
            arw_deg_sqrt_hr = float(imu.get("AngularRandomWalk", arw_deg_sqrt_hr))
            vrw_sqrt_hr = float(imu.get("VelocityRandomWalk", vrw_sqrt_hr))
        except Exception as exc:                            # noqa: BLE001
            self.get_logger().warning(
                f"could not read the IMU noise block from settings.json ({exc}); "
                f"using AirSim defaults ARW={arw_deg_sqrt_hr}, VRW={vrw_sqrt_hr}")

        f = max(1.0, float(imu_rate_hz))
        gyro_var = (math.radians(arw_deg_sqrt_hr) / 60.0) ** 2 * f
        accel_var = (vrw_sqrt_hr / 60.0) ** 2 * f
        self.get_logger().info(
            f"IMU noise: ARW={arw_deg_sqrt_hr} deg/sqrt(hr), VRW={vrw_sqrt_hr} m/s/sqrt(hr)"
            f" -> var(gyro)={gyro_var:.3e}, var(accel)={accel_var:.3e} at {f:g} Hz")

        diag = lambda v: [v, 0.0, 0.0, 0.0, v, 0.0, 0.0, 0.0, v]   # noqa: E731
        return diag(1e-4), diag(gyro_var), diag(accel_var)

    def _tick_imu(self) -> None:
        try:
            with self._rpc_lock:
                d = self.client.getImuData(self._cfg.IMU_NAME, self.vehicle)
        except (RpcError, OSError) as exc:
            return self._guard("getImuData", exc)

        m = Imu()
        m.header.stamp = self._stamp(getattr(d, "time_stamp", None))
        m.header.frame_id = self.frames.imu
        # Angular velocity and acceleration are body-frame vectors: FRD -> FLU.
        wx, wy, wz = frames.frd_to_flu(d.angular_velocity.x_val,
                                       d.angular_velocity.y_val,
                                       d.angular_velocity.z_val)
        ax, ay, az = frames.frd_to_flu(d.linear_acceleration.x_val,
                                       d.linear_acceleration.y_val,
                                       d.linear_acceleration.z_val)
        m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = wx, wy, wz
        m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = ax, ay, az

        q = frames.ned_quat_to_enu(d.orientation.x_val, d.orientation.y_val,
                                   d.orientation.z_val, d.orientation.w_val)
        m.orientation.x, m.orientation.y, m.orientation.z, m.orientation.w = (
            float(q[0]), float(q[1]), float(q[2]), float(q[3]))

        m.orientation_covariance = self._cov_orient
        m.angular_velocity_covariance = self._cov_gyro
        m.linear_acceleration_covariance = self._cov_accel
        self.pub_imu.publish(m)

    def _tick_odom(self) -> None:
        try:
            with self._rpc_lock:
                k = self.client.simGetGroundTruthKinematics(self.vehicle)
        except (RpcError, OSError) as exc:
            return self._guard("simGetGroundTruthKinematics", exc)

        stamp = self._stamp()
        f = self.frames

        # Ground truth arrives in the vehicle's LOCAL NED frame; the calibrated
        # offset is carried by the static map->odom transform, so the pose here is
        # published relative to odom and nothing double-counts it.
        px, py, pz = frames.ned_point_to_enu(k.position.x_val, k.position.y_val,
                                             k.position.z_val)
        q = frames.ned_quat_to_enu(k.orientation.x_val, k.orientation.y_val,
                                   k.orientation.z_val, k.orientation.w_val)

        child = f.base_link if self.gt_owns_base_link else f.base_link_gt
        self.tf_broadcaster.sendTransform(
            self._tf(f.odom, child, (px, py, pz), q, stamp))

        o = Odometry()
        o.header.stamp = stamp
        o.header.frame_id = f.odom
        o.child_frame_id = child
        o.pose.pose.position.x, o.pose.pose.position.y, o.pose.pose.position.z = px, py, pz
        (o.pose.pose.orientation.x, o.pose.pose.orientation.y,
         o.pose.pose.orientation.z, o.pose.pose.orientation.w) = (
            float(q[0]), float(q[1]), float(q[2]), float(q[3]))

        # Twist is child-frame (body) per REP-103, so it takes the FRD->FLU flip,
        # not the world swap.
        vx, vy, vz = frames.frd_to_flu(k.linear_velocity.x_val, k.linear_velocity.y_val,
                                       k.linear_velocity.z_val)
        wx, wy, wz = frames.frd_to_flu(k.angular_velocity.x_val, k.angular_velocity.y_val,
                                       k.angular_velocity.z_val)
        o.twist.twist.linear.x, o.twist.twist.linear.y, o.twist.twist.linear.z = vx, vy, vz
        o.twist.twist.angular.x, o.twist.twist.angular.y, o.twist.twist.angular.z = wx, wy, wz
        # This is ground truth: assert near-zero covariance so anything fusing it
        # treats it as such.
        o.pose.covariance = [0.0] * 36
        o.twist.covariance = [0.0] * 36
        for i in range(6):
            o.pose.covariance[i * 7] = 1e-9
            o.twist.covariance[i * 7] = 1e-9
        self.pub_odom.publish(o)

        # Path is in map, so it needs the offset the odom pose deliberately omits.
        if self.pub_path.get_subscription_count():
            from geometry_msgs.msg import PoseStamped

            ps = PoseStamped()
            ps.header.stamp = stamp
            ps.header.frame_id = f.MAP
            ox, oy = self._offset
            ps.pose.position.x = px + oy      # offset is world-NED; ENU x comes from NED y
            ps.pose.position.y = py + ox
            ps.pose.position.z = pz
            ps.pose.orientation = o.pose.pose.orientation
            self._path.header.stamp = stamp
            self._path.poses.append(ps)
            if len(self._path.poses) > self.path_max:
                del self._path.poses[0]
            self.pub_path.publish(self._path)

    def _tick_lidar(self) -> None:
        try:
            with self._rpc_lock:
                d = self.client.getLidarData(self._cfg.LIDAR_NAME, self.vehicle)
        except (RpcError, OSError) as exc:
            return self._guard("getLidarData", exc)

        raw = getattr(d, "point_cloud", None)
        # AirSim returns a single 0.0 (not an empty list) when a sweep has no
        # returns -- e.g. while the drone is still inside spawn geometry.
        if raw is None or len(raw) < 3:
            return

        pts = frames.frd_points_to_flu(np.asarray(raw, dtype=np.float32))
        # Drop non-finite returns rather than letting them poison ICP's centroid.
        pts = pts[np.isfinite(pts).all(axis=1)]
        if not len(pts):
            return

        self.pub_cloud.publish(self._cloud_msg(
            pts, self.frames.lidar, self._stamp(getattr(d, "time_stamp", None))))

    @staticmethod
    def _cloud_msg(pts: np.ndarray, frame_id: str, stamp) -> PointCloud2:
        """(N,3) float32 -> PointCloud2, built directly from the array's buffer.

        ``sensor_msgs_py.create_cloud_xyz32`` walks the points in Python, which at
        100k points a sweep is far too slow for a 10 Hz timer; the memory layout
        of a contiguous float32 (N,3) array is already exactly the wire format.
        """
        msg = PointCloud2()
        msg.header = Header(stamp=stamp, frame_id=frame_id)
        msg.height = 1
        msg.width = int(pts.shape[0])
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12 * msg.width
        msg.is_dense = True
        msg.data = np.ascontiguousarray(pts, dtype=np.float32).tobytes()
        return msg

    def _tick_gps(self) -> None:
        try:
            with self._rpc_lock:
                d = self.client.getGpsData("", self.vehicle)
        except (RpcError, OSError) as exc:
            return self._guard("getGpsData", exc)

        g = d.gnss.geo_point
        m = NavSatFix()
        m.header.stamp = self._stamp(getattr(d, "time_stamp", None))
        m.header.frame_id = self.frames.base_link
        m.latitude = float(g.latitude)
        m.longitude = float(g.longitude)
        m.altitude = float(g.altitude)
        m.status.status = 0        # STATUS_FIX
        m.status.service = 1       # SERVICE_GPS
        m.position_covariance_type = 1     # APPROXIMATED
        eph = float(getattr(d.gnss, "eph", 1.0))
        epv = float(getattr(d.gnss, "epv", 1.0))
        m.position_covariance = [eph ** 2, 0.0, 0.0,
                                 0.0, eph ** 2, 0.0,
                                 0.0, 0.0, epv ** 2]
        self.pub_gps.publish(m)

    def _tick_stereo(self) -> None:
        from .airsim_api import ImageRequest, ImageType

        cfg = self._cfg
        try:
            with self._rpc_lock:
                resp = self.client.simGetImages([
                    ImageRequest(cfg.CAM_LEFT, ImageType.Scene, False, False),
                    ImageRequest(cfg.CAM_RIGHT, ImageType.Scene, False, False),
                ], self.vehicle)
        except (RpcError, OSError) as exc:
            return self._guard("simGetImages", exc)

        if len(resp) != 2:
            return
        stamp = self._stamp(getattr(resp[0], "time_stamp", None))
        pairs = ((resp[0], self.pub_left, self.pub_left_info, self.frames.cam_left_optical, False),
                 (resp[1], self.pub_right, self.pub_right_info, self.frames.cam_right_optical, True))
        for r, pub_img, pub_info, frame_id, is_right in pairs:
            img = self._image_msg(r, frame_id, stamp)
            if img is None:
                continue
            pub_img.publish(img)
            pub_info.publish(frames.camera_info_msg(
                img.width, img.height, cfg.FX, cfg.FY,
                # Intrinsics in slam.config are for the full-size camera; the
                # stereo pair renders smaller, so scale rather than mis-report.
                cfg.CX * img.width / cfg.IMAGE_WIDTH,
                cfg.CY * img.height / cfg.IMAGE_HEIGHT,
                frame_id, stamp,
                baseline=cfg.STEREO_BASELINE, is_right=is_right))

    def _image_msg(self, r, frame_id: str, stamp) -> Image | None:
        """AirSim image response -> ``sensor_msgs/Image`` (bgr8), or None if junk.

        AirSim serves 0x0 frames while a render target spins up and occasionally a
        short buffer; both have to be dropped rather than reshaped, exactly as
        ``slam/source.py:_decode_bgr`` does on the host side.
        """
        data = getattr(r, "image_data_uint8", b"")
        want = int(r.width) * int(r.height) * 3
        if not r.width or not r.height or len(data) != want:
            return None
        m = Image()
        m.header = Header(stamp=stamp, frame_id=frame_id)
        m.height = int(r.height)
        m.width = int(r.width)
        m.encoding = "bgr8"
        m.is_bigendian = 0
        m.step = 3 * int(r.width)
        m.data = bytes(data)
        return m

    def destroy_node(self) -> bool:
        try:
            self.client.close()
        except Exception:                     # noqa: BLE001 - shutdown must not raise
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = AirSimBridge()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, SystemExit) as exc:
        if isinstance(exc, SystemExit) and exc.code:
            print(f"airsim_bridge: {exc}")
    except RpcError as exc:
        print(f"airsim_bridge: {exc}")
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
