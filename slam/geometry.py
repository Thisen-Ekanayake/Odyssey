"""
SE(3) helpers, in AirSim's NED world frame throughout.

Deliberately free of any ``airsim`` import so the offline benchmark can run
without the simulator (and without dragging in the tornado<5 RPC stack). The
AirSim bridges below duck-type ``.x_val``-style attributes instead.

Quaternion order is ``[x, y, z, w]`` (scipy's) everywhere in this package.
AirSim's ``Quaternionr`` exposes ``w_val/x_val/y_val/z_val`` -- the bridges
here are the only place that ordering is untangled.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

__all__ = [
    "quat_to_R", "R_to_quat", "pose_to_matrix", "matrix_to_pose",
    "invert_se3", "relative_se3", "rotation_angle", "se3_log",
    "airsim_pose_to_matrix", "airsim_vec_to_np", "airsim_quat_to_np",
    "interpolate_pose", "interpolate_trajectory",
    "write_tum", "read_tum", "umeyama",
]


# --------------------------------------------------------------------------
# basic conversions
# --------------------------------------------------------------------------

def quat_to_R(q: Sequence[float]) -> np.ndarray:
    """[x, y, z, w] -> 3x3 rotation matrix."""
    return Rotation.from_quat(np.asarray(q, dtype=np.float64)).as_matrix()


def R_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> [x, y, z, w]."""
    return Rotation.from_matrix(np.asarray(R, dtype=np.float64)).as_quat()


def pose_to_matrix(position: Sequence[float], quat: Sequence[float]) -> np.ndarray:
    """(t, [x,y,z,w]) -> 4x4 SE(3)."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_to_R(quat)
    T[:3, 3] = np.asarray(position, dtype=np.float64)
    return T


def matrix_to_pose(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """4x4 SE(3) -> (translation, [x,y,z,w])."""
    return np.asarray(T[:3, 3], dtype=np.float64).copy(), R_to_quat(T[:3, :3])


def invert_se3(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid transform, without a general matrix inverse."""
    Ti = np.eye(4, dtype=np.float64)
    Rt = T[:3, :3].T
    Ti[:3, :3] = Rt
    Ti[:3, 3] = -Rt @ T[:3, 3]
    return Ti


def relative_se3(T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    """Transform taking frame ``a`` to frame ``b``:  ``inv(T_a) @ T_b``."""
    return invert_se3(T_a) @ T_b


def rotation_angle(R: np.ndarray) -> float:
    """Geodesic rotation magnitude in radians. Clipped against FP drift."""
    cos = (np.trace(R[:3, :3]) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def se3_log(T: np.ndarray) -> np.ndarray:
    """SE(3) -> 6-vector [translation(3), rotvec(3)]. Used for RPE reporting."""
    return np.concatenate([T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_rotvec()])


# --------------------------------------------------------------------------
# AirSim bridges (duck-typed -- no airsim import)
# --------------------------------------------------------------------------

def airsim_vec_to_np(v) -> np.ndarray:
    """``airsim.Vector3r`` -> (3,) array."""
    return np.array([v.x_val, v.y_val, v.z_val], dtype=np.float64)


def airsim_quat_to_np(q) -> np.ndarray:
    """``airsim.Quaternionr`` -> [x, y, z, w] (note AirSim stores w first)."""
    return np.array([q.x_val, q.y_val, q.z_val, q.w_val], dtype=np.float64)


def airsim_pose_to_matrix(pose) -> np.ndarray:
    """``airsim.Pose`` -> 4x4 SE(3) in NED.

    Works for anything exposing ``.position`` / ``.orientation``, which covers
    ``simGetVehiclePose``, ``LidarData.pose``, and ``KinematicsState`` after
    picking off ``.position``/``.orientation``.

    AirSim occasionally hands back an all-zero quaternion before the sim is
    fully up; that is not normalizable, so fall back to identity rather than
    letting scipy raise deep inside a capture loop.
    """
    q = airsim_quat_to_np(pose.orientation)
    if not np.isfinite(q).all() or np.linalg.norm(q) < 1e-9:
        q = np.array([0.0, 0.0, 0.0, 1.0])
    return pose_to_matrix(airsim_vec_to_np(pose.position), q / np.linalg.norm(q))


# --------------------------------------------------------------------------
# interpolation (deskew, ground-truth association)
# --------------------------------------------------------------------------

def interpolate_pose(T0: np.ndarray, T1: np.ndarray, alpha: float) -> np.ndarray:
    """Constant-velocity blend between two poses: slerp rotation, lerp translation.

    ``alpha`` is clamped to [0, 1] so callers can pass raw ratios.
    """
    a = float(np.clip(alpha, 0.0, 1.0))
    key = Rotation.from_matrix(np.stack([T0[:3, :3], T1[:3, :3]]))
    R = Slerp([0.0, 1.0], key)([a]).as_matrix()[0]
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = (1.0 - a) * T0[:3, 3] + a * T1[:3, 3]
    return T


def interpolate_trajectory(
    times: np.ndarray, poses: np.ndarray, query: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a pose trajectory at arbitrary times.

    ``times`` (N,) ascending, ``poses`` (N,4,4), ``query`` (M,). Queries outside
    the trajectory span are dropped rather than extrapolated -- extrapolated
    ground truth would silently corrupt the metrics.

    Returns ``(kept_query_times, (K,4,4) poses)``.
    """
    times = np.asarray(times, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    if len(times) < 2:
        raise ValueError("need at least two poses to interpolate")

    keep = (query >= times[0]) & (query <= times[-1])
    q = query[keep]
    if len(q) == 0:
        return q, np.empty((0, 4, 4))

    rots = Rotation.from_matrix(np.asarray(poses)[:, :3, :3])
    slerp = Slerp(times, rots)
    R = slerp(q).as_matrix()

    xyz = np.stack([
        np.interp(q, times, np.asarray(poses)[:, i, 3]) for i in range(3)
    ], axis=1)

    out = np.tile(np.eye(4), (len(q), 1, 1))
    out[:, :3, :3] = R
    out[:, :3, 3] = xyz
    return q, out


# --------------------------------------------------------------------------
# TUM trajectory I/O
# --------------------------------------------------------------------------

def write_tum(path, times: Iterable[float], poses: Iterable[np.ndarray], header: str = "") -> None:
    """Write ``timestamp tx ty tz qx qy qz qw`` (TUM RGB-D convention).

    Timestamps are seconds. Everything in this package records AirSim sim time
    in nanoseconds, so divide by 1e9 before calling.
    """
    lines = []
    if header:
        lines.extend(f"# {ln}" for ln in header.strip().splitlines())
    lines.append("# timestamp tx ty tz qx qy qz qw")
    for t, T in zip(times, poses):
        p, q = matrix_to_pose(np.asarray(T))
        lines.append(
            f"{t:.9f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
            f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}"
        )
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def read_tum(path) -> tuple[np.ndarray, np.ndarray]:
    """Read a TUM trajectory file -> ``(times (N,), poses (N,4,4))``."""
    raw = np.loadtxt(path, comments="#", ndmin=2)
    if raw.size == 0:
        return np.empty(0), np.empty((0, 4, 4))
    times = raw[:, 0]
    poses = np.tile(np.eye(4), (len(raw), 1, 1))
    poses[:, :3, :3] = Rotation.from_quat(raw[:, 4:8]).as_matrix()
    poses[:, :3, 3] = raw[:, 1:4]
    return times, poses


# --------------------------------------------------------------------------
# alignment
# --------------------------------------------------------------------------

def umeyama(
    src: np.ndarray, dst: np.ndarray, with_scale: bool = False
) -> tuple[np.ndarray, np.ndarray, float]:
    """Least-squares similarity transform mapping ``src`` onto ``dst``.

    Umeyama (1991). Returns ``(R, t, s)`` minimising ``||dst - (s*R*src + t)||``.
    scipy has no equivalent (``align_vectors`` solves rotation only, without the
    translation and scale terms), hence the local implementation.

    ``with_scale=False`` gives the rigid SE(3) alignment used for metric methods;
    the scale term is only meaningful for scale-ambiguous (monocular) estimates
    and is reported as a diagnostic.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"need matching (N,3) inputs, got {src.shape} and {dst.shape}")
    n = src.shape[0]
    if n < 3:
        raise ValueError(f"need at least 3 correspondences, got {n}")

    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    sc, dc = src - mu_src, dst - mu_dst

    cov = (dc.T @ sc) / n
    U, D, Vt = np.linalg.svd(cov)

    # Reflection guard: without it a degenerate (near-planar) trajectory can
    # produce a mirrored "alignment" with a flatteringly small error.
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0

    R = U @ S @ Vt
    var_src = (sc ** 2).sum() / n
    s = float(np.trace(np.diag(D) @ S) / var_src) if (with_scale and var_src > 1e-12) else 1.0
    t = mu_dst - s * (R @ mu_src)
    return R, t, s
