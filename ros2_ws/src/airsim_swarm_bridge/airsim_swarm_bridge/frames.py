"""Coordinate-frame conversions between AirSim and ROS, plus the rig geometry.

AirSim is **NED / FRD** (world x=north, y=east, z=**down**; body x=forward,
y=right, z=down). ROS is **ENU / FLU** (REP-103: world x=east, y=north, z=up;
body x=forward, y=left, z=up). Everything crossing the bridge is converted here,
once, so no node ever has to remember which convention it is holding.

Getting this wrong is quiet rather than loud -- a missed z-flip produces a map
that looks plausible but is mirrored, and ICP will happily converge on it. The
two rotations involved are both exact 180-degree turns:

* ``R_ENU_NED`` swaps x/y and negates z. As a rotation that is 180 degrees about
  the axis (1, 1, 0)/sqrt(2), i.e. the quaternion (x, y, z, w) = (r, r, 0, 0)
  with r = 1/sqrt(2). det = +1, so it is a rotation and not a reflection.
* ``R_FLU_FRD`` negates y and z: 180 degrees about x, quaternion (1, 0, 0, 0).

A world pose therefore converts as
``T_enu_flu = R_ENU_NED @ T_ned_frd @ R_FRD_FLU``, which in quaternion terms is
``q_out = q_ENU_NED * q_in * q_FRD_FLU``.

The rig itself (mount offsets, intrinsics) is *not* re-derived here -- it is read
from :mod:`slam.config`, which is the repo's single mirror of ``settings.json``.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "R_ENU_NED", "R_FLU_FRD", "Q_ENU_NED", "Q_FRD_FLU",
    "ned_point_to_enu", "ned_quat_to_enu", "ned_vector_to_enu",
    "frd_to_flu", "frd_points_to_flu",
    "enu_point_to_ned", "FrameNames", "camera_info_msg",
    "quat_multiply", "quat_normalize",
]

# world ENU <- world NED
R_ENU_NED = np.array([[0.0, 1.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [0.0, 0.0, -1.0]], dtype=np.float64)

# body FLU <- body FRD  (self-inverse, same matrix either direction)
R_FLU_FRD = np.array([[1.0, 0.0, 0.0],
                      [0.0, -1.0, 0.0],
                      [0.0, 0.0, -1.0]], dtype=np.float64)

_R = 1.0 / math.sqrt(2.0)
Q_ENU_NED = np.array([_R, _R, 0.0, 0.0])      # (x, y, z, w)
Q_FRD_FLU = np.array([1.0, 0.0, 0.0, 0.0])    # (x, y, z, w)


# -- small quaternion helpers (avoids depending on tf_transformations, which is
#    not part of ros-jazzy-desktop) --------------------------------------------

def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two (x, y, z, w) quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Unit-normalise, falling back to identity for the all-zero quaternion.

    AirSim hands back (0, 0, 0, 0) while the sim is still starting up; the host
    code guards against exactly this in ``slam/geometry.airsim_pose_to_matrix``.
    """
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if not np.isfinite(n) or n < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / n


# -- world frame --------------------------------------------------------------

def ned_point_to_enu(x: float, y: float, z: float) -> tuple[float, float, float]:
    """World NED position -> world ENU. ``(x_e, y_n, z_u) = (y, x, -z)``."""
    return (float(y), float(x), float(-z))


def ned_vector_to_enu(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Same mapping as :func:`ned_point_to_enu`; named for velocities/accelerations."""
    return (float(y), float(x), float(-z))


def ned_quat_to_enu(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Orientation of an FRD body in NED -> orientation of an FLU body in ENU.

    Returns ``(x, y, z, w)``, the order ROS uses. Note AirSim stores ``w`` first
    in its own struct, so read the fields by name rather than by position.
    """
    q = quat_normalize(np.array([x, y, z, w], dtype=np.float64))
    return quat_normalize(quat_multiply(quat_multiply(Q_ENU_NED, q), Q_FRD_FLU))


def enu_point_to_ned(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Inverse of :func:`ned_point_to_enu` -- the mapping is its own inverse."""
    return (float(y), float(x), float(-z))


# -- body frame ---------------------------------------------------------------

def frd_to_flu(x: float, y: float, z: float) -> tuple[float, float, float]:
    """A vector in a body-fixed FRD frame -> the matching FLU frame."""
    return (float(x), float(-y), float(-z))


def frd_points_to_flu(points: np.ndarray) -> np.ndarray:
    """(N,3) sensor-frame points, FRD -> FLU, vectorised.

    AirSim's LiDAR is configured ``"DataFrame": "SensorLocalFrame"`` with zero
    mount rotation, so the scan arrives in a body-aligned FRD frame and this flip
    is the whole conversion. The points are deliberately *not* placed in the
    world here -- doing that is the estimator's job, and pre-transforming them by
    the simulator's own pose is what makes ``flight/lidar_viz.py`` a
    perfect-localisation baseline rather than SLAM.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    out = pts.copy()
    out[:, 1] *= -1.0
    out[:, 2] *= -1.0
    return out


# -- naming -------------------------------------------------------------------

class FrameNames:
    """TF frame ids for one drone.

    ``Drone1`` -> namespace ``drone1`` -> frames ``drone1/base_link`` and so on.
    ``map`` is shared by the whole swarm; every drone hangs off it through its own
    ``odom``, whose offset comes from ``swarm_comms.SwarmPositions.offset()``.
    """

    MAP = "map"

    def __init__(self, vehicle_name: str):
        self.vehicle = vehicle_name
        self.ns = vehicle_name.lower()
        self.odom = f"{self.ns}/odom"
        self.base_link = f"{self.ns}/base_link"
        self.base_link_gt = f"{self.ns}/base_link_gt"
        self.lidar = f"{self.ns}/lidar_link"
        self.imu = f"{self.ns}/imu_link"
        self.cam_left = f"{self.ns}/stereo_left_link"
        self.cam_right = f"{self.ns}/stereo_right_link"
        self.cam_left_optical = f"{self.ns}/stereo_left_optical"
        self.cam_right_optical = f"{self.ns}/stereo_right_optical"

    def __repr__(self) -> str:
        return f"<FrameNames {self.vehicle} -> {self.ns}/*>"


# -- camera intrinsics --------------------------------------------------------

def camera_info_msg(width: int, height: int, fx: float, fy: float,
                    cx: float, cy: float, frame_id: str, stamp,
                    baseline: float = 0.0, is_right: bool = False):
    """A ``CameraInfo`` for one of AirSim's ideal pinholes.

    AirSim renders a perfect pinhole with square pixels and no distortion, so D
    is zero and rectification is the identity -- worth stating plainly in any
    writeup, because real stereo hardware never is.

    For the right camera, ``P[0,3] = -fx * baseline``: that is the ROS
    convention that lets a stereo pipeline recover disparity-to-depth without
    being told the baseline separately.
    """
    from sensor_msgs.msg import CameraInfo

    msg = CameraInfo()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.width = int(width)
    msg.height = int(height)
    msg.distortion_model = "plumb_bob"
    msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    msg.k = [fx, 0.0, cx,
             0.0, fy, cy,
             0.0, 0.0, 1.0]
    msg.r = [1.0, 0.0, 0.0,
             0.0, 1.0, 0.0,
             0.0, 0.0, 1.0]
    tx = -fx * baseline if is_right else 0.0
    msg.p = [fx, 0.0, cx, tx,
             0.0, fy, cy, 0.0,
             0.0, 0.0, 1.0, 0.0]
    return msg
