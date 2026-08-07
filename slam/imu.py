"""
IMU propagation: the motion prior that seeds registration, and the sweep
motion used to de-skew the LiDAR.

This is dead reckoning and it *will* drift -- that is expected and fine. Its
job is to hand the front end a good enough initial guess that ICP converges to
the right basin instead of a neighbouring one, and to tell the de-skew step how
far the drone moved during a single 100 ms revolution. Registration then
corrects the drift every frame; the IMU is never trusted on its own.

Frame conventions (NED, z DOWN):

    a_world = R_wb @ (f_body - bias_a) + g_ned,   g_ned = [0, 0, +9.80665]

so a level, stationary vehicle reads ``f_body = [0, 0, -9.80665]``: the
accelerometer measures specific force, which opposes gravity.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from . import config as cfg
from .geometry import interpolate_pose, invert_se3
from .source import ImuSample

__all__ = ["ImuState", "ImuPropagator", "deskew"]


@dataclass
class ImuState:
    """Navigation state in the world (NED) frame."""
    R: np.ndarray = field(default_factory=lambda: np.eye(3))    # world <- body
    v: np.ndarray = field(default_factory=lambda: np.zeros(3))  # m/s, world
    p: np.ndarray = field(default_factory=lambda: np.zeros(3))  # m, world

    def matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.p
        return T

    def copy(self) -> "ImuState":
        return ImuState(self.R.copy(), self.v.copy(), self.p.copy())

    @staticmethod
    def from_matrix(T: np.ndarray, v: np.ndarray | None = None) -> "ImuState":
        return ImuState(T[:3, :3].copy(), np.zeros(3) if v is None else v.copy(),
                        T[:3, 3].copy())


class ImuPropagator:
    """Strapdown integration with fixed biases.

    Biases are estimated once, from a short quasi-stationary window at the
    start of the run, and then held. Estimating them online would need a filter
    or a factor graph over IMU factors -- out of scope here, where the IMU only
    has to survive the ~100 ms between LiDAR sweeps before registration pins
    the pose down again.
    """

    def __init__(self, bias_gyro: np.ndarray | None = None,
                 bias_accel: np.ndarray | None = None) -> None:
        self.bias_gyro = np.zeros(3) if bias_gyro is None else np.asarray(bias_gyro, float)
        self.bias_accel = np.zeros(3) if bias_accel is None else np.asarray(bias_accel, float)
        self.initialized = bias_gyro is not None

    # -- initialisation ----------------------------------------------------

    def initialize(self, samples: list[ImuSample],
                   assume_level: bool = True) -> np.ndarray:
        """Estimate biases from a quasi-stationary window; return the initial attitude.

        Gravity alignment fixes roll and pitch from the measured specific-force
        direction. Yaw is unobservable from an accelerometer, so it is set to
        zero: the estimated trajectory lives in its own frame anyway, and the
        evaluation aligns it to ground truth with Umeyama. Nothing here reads
        the simulator's pose.

        ``assume_level`` treats the hover attitude as level for the purposes of
        splitting the accelerometer reading into gravity and bias. A multirotor
        holding station is level to within a degree or two, which is well inside
        the accuracy this prior needs.
        """
        if not samples:
            return np.eye(3)

        n = max(1, min(len(samples), int(cfg.IMU_BIAS_INIT_SECONDS * cfg.IMU_RATE_HZ)))
        gyro = np.array([s.gyro for s in samples[:n]])
        accel = np.array([s.accel for s in samples[:n]])

        # Stationary -> the true angular rate is zero, so the mean is the bias.
        self.bias_gyro = gyro.mean(axis=0)

        f_mean = accel.mean(axis=0)
        norm = np.linalg.norm(f_mean)
        if norm < 1.0:
            self.initialized = True
            return np.eye(3)

        if assume_level:
            R0 = np.eye(3)
        else:
            # Rotate measured specific force onto -g: gives roll/pitch, not yaw.
            R0 = _align(f_mean / norm, -cfg.GRAVITY_NED / np.linalg.norm(cfg.GRAVITY_NED))

        # Whatever is left after removing gravity is accelerometer bias.
        self.bias_accel = f_mean - R0.T @ (-cfg.GRAVITY_NED)
        self.initialized = True
        return R0

    # -- propagation -------------------------------------------------------

    def integrate(self, state: ImuState, samples: list[ImuSample],
                  t_end: float | None = None) -> ImuState:
        """Advance ``state`` through ``samples`` (midpoint rule on attitude).

        Returns a new state; the input is not mutated, so callers can keep the
        pre-integration pose for the de-skew step.
        """
        s = state.copy()
        if not samples:
            return s

        for i, sample in enumerate(samples):
            if i + 1 < len(samples):
                dt = samples[i + 1].t - sample.t
            elif t_end is not None:
                dt = t_end - sample.t
            else:
                dt = 1.0 / cfg.IMU_RATE_HZ
            if not (0.0 < dt < 1.0):        # guard against gaps and bad stamps
                dt = 1.0 / cfg.IMU_RATE_HZ

            w = sample.gyro - self.bias_gyro
            f = sample.accel - self.bias_accel

            # Specific force is rotated by the mid-interval attitude, which is
            # noticeably better than the start-of-interval one while turning.
            R_half = s.R @ Rotation.from_rotvec(w * (0.5 * dt)).as_matrix()
            a = R_half @ f + cfg.GRAVITY_NED

            s.p = s.p + s.v * dt + 0.5 * a * dt * dt
            s.v = s.v + a * dt
            s.R = s.R @ Rotation.from_rotvec(w * dt).as_matrix()

        # Renormalise: repeated matrix products drift off SO(3) over a long run.
        s.R = Rotation.from_matrix(s.R).as_matrix()
        return s

    def predict(self, state: ImuState, samples: list[ImuSample],
                t_end: float | None = None) -> tuple[np.ndarray, ImuState]:
        """Convenience wrapper: returns ``(predicted 4x4 pose, predicted state)``."""
        s = self.integrate(state, samples, t_end)
        return s.matrix(), s


def _align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector ``a`` onto unit vector ``b``."""
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


# --------------------------------------------------------------------------
# motion compensation
# --------------------------------------------------------------------------

def deskew(points: np.ndarray, T_start: np.ndarray, T_end: np.ndarray,
           n_buckets: int = 20) -> np.ndarray:
    """Undo the motion smear within one LiDAR revolution.

    At 10 rev/s and 4 m/s the sensor travels 40 cm during a single sweep, so
    points captured early in the revolution belong to a different sensor pose
    than points captured late. Registering the raw sweep as if it were an
    instantaneous snapshot bakes that 40 cm in as a systematic error.

    Points arrive in azimuth order, so their index is a proxy for capture time.
    They are re-expressed in the sensor frame at the *end* of the sweep, which
    is the pose ``LidarData.pose`` reports and the one the front end tracks.

    Buckets rather than per-point transforms: 20 rigid transforms on
    contiguous slices costs a fraction of 30 000 individual ones, and the
    residual within a bucket is ~2 cm of the total sweep motion -- far below
    the sensor's own noise.
    """
    n = len(points)
    if n == 0:
        return points

    # Sensor-frame motion across the sweep: maps start-of-sweep into end-of-sweep.
    T_delta = invert_se3(T_end) @ T_start
    if np.allclose(T_delta, np.eye(4), atol=1e-6):
        return points

    out = np.empty_like(points, dtype=np.float32)
    edges = np.linspace(0, n, n_buckets + 1).astype(int)
    identity = np.eye(4)

    for b in range(n_buckets):
        lo, hi = edges[b], edges[b + 1]
        if hi <= lo:
            continue
        alpha = (b + 0.5) / n_buckets          # bucket centre, 0 = start of sweep
        C = interpolate_pose(T_delta, identity, alpha)
        out[lo:hi] = points[lo:hi] @ C[:3, :3].T.astype(np.float32) + C[:3, 3].astype(np.float32)

    return out
