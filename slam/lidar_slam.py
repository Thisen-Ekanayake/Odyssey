"""
LiDAR-inertial SLAM.

Pipeline per sweep:

    IMU prior  ->  de-skew  ->  GICP scan-to-submap  ->  keyframe?
                                                            |
                                      loop closure + pose-graph optimisation

The IMU is a *prior*, never a measurement of position: it seeds registration so
ICP starts inside the right basin of convergence, and it supplies the intra-
sweep motion for de-skewing. Registration corrects its drift every frame.

Nothing here reads a simulator pose. ``Frame.gt_pose`` is carried through the
source for evaluation only, and this module never touches it -- ``run()``
records ground truth into the result purely so ``evaluate`` can score against
it afterwards.

Trajectory reconstruction deserves a note: when loop closure fires, every past
keyframe pose moves. Per-frame poses are therefore stored *relative to their
reference keyframe* and re-expanded at the end against the optimised keyframe
poses, so the returned trajectory reflects the final optimised graph rather
than a stale running estimate.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import open3d as o3d

from . import config as cfg
from .backend import PoseGraphBackend
from .geometry import invert_se3, rotation_angle
from .imu import ImuPropagator, ImuState, deskew
from .mapping import SubmapWindow, VoxelMap, preprocess
from .source import Frame, SensorSource

__all__ = ["SlamResult", "LidarInertialSLAM"]

_reg = o3d.pipelines.registration


@dataclass
class SlamResult:
    """Everything a benchmark run needs to score and visualise a method."""
    method: str = ""
    times: np.ndarray = field(default_factory=lambda: np.empty(0))
    poses: np.ndarray = field(default_factory=lambda: np.empty((0, 4, 4)))
    keyframe_times: np.ndarray = field(default_factory=lambda: np.empty(0))
    keyframe_poses: np.ndarray = field(default_factory=lambda: np.empty((0, 4, 4)))
    gt_times: np.ndarray = field(default_factory=lambda: np.empty(0))
    gt_poses: np.ndarray = field(default_factory=lambda: np.empty((0, 4, 4)))
    map_points: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    stats: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)

    def summary(self) -> str:
        s = self.stats
        return (
            f"{self.method}: {s.get('frames', 0)} frames, "
            f"{s.get('keyframes', 0)} keyframes, "
            f"{s.get('loop_closures', 0)} loop closures, "
            f"{s.get('tracking_failures', 0)} tracking failures "
            f"({100 * s.get('tracking_failure_rate', 0):.1f}%), "
            f"{len(self.map_points)} map points"
        )


class _Timer:
    """Per-stage wall-clock accumulator, reported as ms/frame."""

    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def add(self, stage: str, seconds: float) -> None:
        self.totals[stage] = self.totals.get(stage, 0.0) + seconds
        self.counts[stage] = self.counts.get(stage, 0) + 1

    def report(self) -> dict:
        return {f"{k}_ms": 1000.0 * v / max(1, self.counts[k])
                for k, v in self.totals.items()}


class LidarInertialSLAM:
    def __init__(
        self,
        use_imu: bool = True,
        use_loop_closure: bool = True,
        use_deskew: bool = True,
        verbose: bool = True,
    ) -> None:
        self.use_imu = use_imu
        self.use_loop_closure = use_loop_closure
        self.use_deskew = use_deskew
        self.verbose = verbose

        self.imu = ImuPropagator()
        self.backend = PoseGraphBackend(verbose=verbose)
        self.submap = SubmapWindow()
        self.map = VoxelMap()
        self.timer = _Timer()

        self.state = ImuState()
        self.T_sensor = np.eye(4)          # current world <- LiDAR estimate
        self._initialized = False
        self._init_samples: list = []

        # per-frame trajectory, anchored to keyframes so loop closure propagates
        self._frame_times: list[float] = []
        self._frame_anchor: list[int] = []
        self._frame_rel: list[np.ndarray] = []

        self._last_kf_pose = np.eye(4)
        self._kf_points: list[np.ndarray] = []
        self._prev_t: float | None = None

        self.n_frames = 0
        self.n_failures = 0
        self.fitness_log: list[float] = []

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _body_to_sensor(T_body: np.ndarray) -> np.ndarray:
        return T_body @ cfg.T_BODY_LIDAR

    @staticmethod
    def _sensor_to_body(T_sensor: np.ndarray) -> np.ndarray:
        return T_sensor @ invert_se3(cfg.T_BODY_LIDAR)

    def _initialize(self, frame: Frame) -> None:
        """Seed biases and the initial attitude from the first second of IMU."""
        self._init_samples.extend(frame.imu)
        need = int(cfg.IMU_BIAS_INIT_SECONDS * cfg.IMU_RATE_HZ)
        if self.use_imu and len(self._init_samples) < need and frame.lidar is None:
            return

        R0 = self.imu.initialize(self._init_samples) if self.use_imu else np.eye(3)
        self.state = ImuState(R=R0, v=np.zeros(3), p=np.zeros(3))
        self.T_sensor = self._body_to_sensor(self.state.matrix())
        self._initialized = True

        if self.verbose:
            print(f"  IMU initialised: gyro bias {np.round(self.imu.bias_gyro, 5)} rad/s, "
                  f"accel bias {np.round(self.imu.bias_accel, 4)} m/s^2")

    # -- main step ---------------------------------------------------------

    def process(self, frame: Frame) -> np.ndarray:
        """Consume one frame; returns the current world <- body pose estimate."""
        if not self._initialized:
            self._initialize(frame)
            if not self._initialized:
                return self.state.matrix()

        # ---- IMU prior ----
        t0 = time.perf_counter()
        T_body_prev = self.state.matrix()
        if self.use_imu and frame.imu:
            T_body_pred, state_pred = self.imu.predict(self.state, frame.imu, frame.t)
        else:
            T_body_pred, state_pred = T_body_prev, self.state.copy()
        T_sensor_pred = self._body_to_sensor(T_body_pred)
        self.timer.add("imu", time.perf_counter() - t0)

        if frame.lidar is None or len(frame.lidar.points) < 100:
            self.state = state_pred
            self.T_sensor = T_sensor_pred
            return self.state.matrix()

        # Counted here rather than on the success path, so n_frames means "frames
        # on which registration was attempted" -- the denominator that makes
        # tracking_failure_rate a real rate. Matches stereo_slam.process().
        self.n_frames += 1

        # ---- de-skew ----
        t0 = time.perf_counter()
        points = frame.lidar.points
        if self.use_deskew:
            points = deskew(points, self.T_sensor, T_sensor_pred)
        self.timer.add("deskew", time.perf_counter() - t0)

        # ---- register ----
        t0 = time.perf_counter()
        source = preprocess(points, cfg.VOXEL_SIZE, with_normals=True)
        target = self.submap.target()

        if target is None or len(target.points) < 100:
            T_sensor_est = T_sensor_pred            # first sweep defines the origin
            fitness, rmse, ok = 1.0, 0.0, True
        else:
            result = _reg.registration_generalized_icp(
                source, target, cfg.GICP_MAX_CORRESPONDENCE, T_sensor_pred,
                _reg.TransformationEstimationForGeneralizedICP(),
                _reg.ICPConvergenceCriteria(max_iteration=cfg.GICP_MAX_ITER))
            fitness, rmse = float(result.fitness), float(result.inlier_rmse)
            ok = fitness >= cfg.GICP_MIN_FITNESS and rmse <= cfg.GICP_MAX_RMSE
            # A rejected registration falls back to the IMU prior rather than
            # accepting a bad match. Counting these is the point: the failure
            # rate is how degradation shows up before the trajectory diverges.
            T_sensor_est = np.asarray(result.transformation) if ok else T_sensor_pred
            if not ok:
                self.n_failures += 1
        self.fitness_log.append(fitness)
        self.timer.add("register", time.perf_counter() - t0)

        # ---- update state ----
        T_body_est = self._sensor_to_body(T_sensor_est)
        dt = (frame.t - self._prev_t) if self._prev_t else 0.0
        # Velocity from the registered pose difference, not from integrating
        # acceleration: the latter drifts within a couple of seconds and would
        # poison the next frame's prior.
        v = ((T_body_est[:3, 3] - T_body_prev[:3, 3]) / dt) if dt > 1e-6 else state_pred.v
        self.state = ImuState(R=T_body_est[:3, :3].copy(), v=v, p=T_body_est[:3, 3].copy())
        self.T_sensor = T_sensor_est
        self._prev_t = frame.t

        # ---- keyframe? ----
        # The anchor for a NON-keyframe must be captured before the keyframe
        # step runs, because that step may optimise the graph and move every
        # pose. Anchoring afterwards would express this frame relative to a
        # corrected pose using an uncorrected estimate, exactly cancelling the
        # loop closure back out.
        prev_anchor = max(0, len(self.backend.poses) - 1)
        prev_anchor_rel = (invert_se3(self.backend.poses[prev_anchor]) @ T_sensor_est
                           if self.backend.poses else np.eye(4))

        new_kf = self._maybe_keyframe(frame, points, T_sensor_est)

        if new_kf is not None:
            # A keyframe IS its own anchor, so it follows the optimised pose.
            anchor, rel = new_kf, np.eye(4)
        else:
            anchor, rel = prev_anchor, prev_anchor_rel

        self._frame_times.append(frame.t)
        self._frame_anchor.append(anchor)
        self._frame_rel.append(rel)

        return self.state.matrix()

    def _maybe_keyframe(self, frame: Frame, points: np.ndarray,
                        T_sensor: np.ndarray) -> int | None:
        """Add a keyframe if the motion threshold is crossed. Returns its index."""
        first = not self.backend.poses
        if not first:
            delta = invert_se3(self._last_kf_pose) @ T_sensor
            moved = float(np.linalg.norm(delta[:3, 3]))
            turned = rotation_angle(delta[:3, :3])
            if moved < cfg.KEYFRAME_TRANS and turned < cfg.KEYFRAME_ROT:
                return None

        t0 = time.perf_counter()
        idx = self.backend.add_keyframe(points, T_sensor, frame.t)
        self._kf_points.append(np.asarray(points, dtype=np.float32))
        self.submap.add(points, T_sensor)
        self.map.insert(points @ T_sensor[:3, :3].T + T_sensor[:3, 3])
        self._last_kf_pose = T_sensor.copy()
        self.timer.add("keyframe", time.perf_counter() - t0)

        if self.use_loop_closure:
            t0 = time.perf_counter()
            closures = self.backend.try_close_loops(idx)
            self.timer.add("loop_search", time.perf_counter() - t0)

            if closures:
                t0 = time.perf_counter()
                self.backend.optimize()
                self.timer.add("optimize", time.perf_counter() - t0)

                # The front end must track against the corrected map, and the
                # current pose must follow the graph -- otherwise the next frame
                # registers against a map the back end has already moved.
                self.submap.update_poses(self.backend.poses)
                self._last_kf_pose = self.backend.poses[-1].copy()
                self.T_sensor = self.backend.poses[-1].copy()
                T_body = self._sensor_to_body(self.T_sensor)
                self.state = ImuState(R=T_body[:3, :3].copy(), v=self.state.v,
                                      p=T_body[:3, 3].copy())
        return idx

    # -- finish ------------------------------------------------------------

    def finalize(self) -> SlamResult:
        """Re-expand the trajectory and rebuild the map under the final poses."""
        kf_poses = self.backend.poses

        if kf_poses and self._frame_times:
            sensor_poses = np.stack([
                kf_poses[a] @ rel for a, rel in zip(self._frame_anchor, self._frame_rel)])
            body_poses = np.stack([self._sensor_to_body(T) for T in sensor_poses])
        else:
            body_poses = np.empty((0, 4, 4))

        # The map implied by the optimised trajectory is not the one that was
        # accumulated live -- every keyframe moved, so it is rebuilt.
        vm = (VoxelMap.rebuild(self._kf_points, kf_poses) if kf_poses else self.map)

        stats = {
            "frames": self.n_frames,
            "tracking_failures": self.n_failures,
            "tracking_failure_rate": self.n_failures / max(1, self.n_frames),
            "mean_fitness": float(np.mean(self.fitness_log)) if self.fitness_log else 0.0,
            "median_fitness": float(np.median(self.fitness_log)) if self.fitness_log else 0.0,
            "map_points": len(vm.points),
            **self.backend.stats(),
        }

        return SlamResult(
            method="lidar_inertial",
            times=np.array(self._frame_times),
            poses=body_poses,
            keyframe_times=np.array(self.backend.times),
            keyframe_poses=(np.stack([self._sensor_to_body(T) for T in kf_poses])
                            if kf_poses else np.empty((0, 4, 4))),
            map_points=vm.points,
            stats=stats,
            timings=self.timer.report(),
        )

    def run(self, source: SensorSource, progress_every: int = 50) -> SlamResult:
        """Drive the whole pipeline over a source and return the finished result."""
        gt_t, gt_T = [], []
        try:
            total = len(source)
        except TypeError:
            total = None

        wall0 = time.perf_counter()
        for i, frame in enumerate(source):
            self.process(frame)
            if frame.gt_pose is not None:
                gt_t.append(frame.t)
                gt_T.append(frame.gt_pose)

            if self.verbose and progress_every and i % progress_every == 0:
                pct = f"{100 * i / total:5.1f}%" if total else f"{i:5d}"
                print(f"\r  [lidar] {pct}  kf {len(self.backend.poses):4d}  "
                      f"loops {len(self.backend.loops):3d}  "
                      f"fail {self.n_failures:3d}", end="", flush=True)

        result = self.finalize()
        result.gt_times = np.array(gt_t)
        result.gt_poses = np.stack(gt_T) if gt_T else np.empty((0, 4, 4))
        result.stats["wall_seconds"] = time.perf_counter() - wall0

        if self.verbose:
            print(f"\r  {result.summary()}")
        return result
