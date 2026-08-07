"""
Sensor-source abstraction: the same SLAM code runs live against the simulator
and offline against a recorded dataset.

That symmetry is the point. The live driver is a demo; the benchmark needs
*identical input bytes* for every method and every re-run, which only a
recording gives. Both produce the same ``Frame`` stream, so ``lidar_slam`` and
``stereo_slam`` never know which they are attached to.

On-disk layout produced by ``recorder`` and consumed by ``DatasetSource``:

    datasets/<condition>/
        sensor.yaml         rig spec, weather params, git SHA, capture mode
        groundtruth.txt     TUM, body pose in world NED, at IMU rate
        imu.txt             t wx wy wz ax ay az
        lidar/<ts_ns>.npy   float32 (N,3), SENSOR-LOCAL frame
        lidar_poses.txt     TUM, ground-truth LiDAR sensor pose per scan
        cam0/<ts_ns>.png    StereoLeft
        cam1/<ts_ns>.png    StereoRight
        cam_poses.txt       TUM, ground-truth left-camera OPTICAL pose per frame
        depth_gt/<ts_ns>.npy  float32 (H,W) planar depth -- evaluation only

Timestamps are AirSim sim time. Filenames use integer nanoseconds (exact);
``Frame.t`` and the TUM files use seconds (convenient).
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

from . import config as cfg
from .geometry import read_tum

__all__ = ["ImuSample", "LidarScan", "StereoFrame", "Frame",
           "SensorSource", "DatasetSource", "LiveSource"]


# --------------------------------------------------------------------------
# messages
# --------------------------------------------------------------------------

@dataclass
class ImuSample:
    """One IMU reading in the vehicle body frame (NED: x fwd, y right, z down)."""
    t: float                       # sim seconds
    gyro: np.ndarray               # (3,) rad/s
    accel: np.ndarray              # (3,) m/s^2, specific force (gravity included)


@dataclass
class LidarScan:
    """One LiDAR revolution, in the SENSOR's own frame.

    ``points`` must never be pre-transformed by a simulator pose -- placing them
    in the world is the estimator's job, and the entire study is about how well
    it does it. ``gt_pose`` is carried alongside for evaluation only.
    """
    t: float
    points: np.ndarray             # (N,3) float32, sensor-local
    gt_pose: np.ndarray | None = None   # (4,4) world<-sensor, evaluation only


@dataclass
class StereoFrame:
    """A rectified stereo pair. AirSim's cameras are ideal parallel pinholes,
    so rectification is the identity and no undistortion step is needed."""
    t: float
    left: np.ndarray               # (H,W,3) uint8 BGR
    right: np.ndarray              # (H,W,3) uint8 BGR
    gt_pose: np.ndarray | None = None    # (4,4) world<-left-camera-OPTICAL
    depth_gt: np.ndarray | None = None   # (H,W) float32, evaluation only


@dataclass
class Frame:
    """One synchronised capture tick.

    ``imu`` holds every sample since the previous frame (not just the nearest
    one), because the front ends integrate across the interval to build a
    motion prior and to de-skew the LiDAR sweep.
    """
    t: float
    imu: list[ImuSample] = field(default_factory=list)
    lidar: LidarScan | None = None
    stereo: StereoFrame | None = None
    gt_pose: np.ndarray | None = None    # (4,4) world<-body, evaluation only


# --------------------------------------------------------------------------

class SensorSource(ABC):
    """Anything that can produce a stream of synchronised ``Frame``s."""

    @abstractmethod
    def __iter__(self) -> Iterator[Frame]:
        ...

    @property
    def has_lidar(self) -> bool:
        return True

    @property
    def has_stereo(self) -> bool:
        return True

    def __len__(self) -> int:
        raise TypeError(f"{type(self).__name__} has no known length")


# --------------------------------------------------------------------------
# offline
# --------------------------------------------------------------------------

class DatasetSource(SensorSource):
    """Replays a recorded dataset from disk.

    Images and clouds are loaded lazily, one frame at a time -- a full run is
    several GB and must never be held in memory at once.

    ``lidar_transform`` is the hook ``slam.degradation`` plugs into: adverse
    weather is applied as a post-process on the recorded scan, so a single
    recording per condition serves both the raw and the degraded evaluation
    without re-flying anything.
    """

    def __init__(
        self,
        path: str | Path,
        load_lidar: bool = True,
        load_stereo: bool = True,
        load_depth_gt: bool = False,
        lidar_transform: Callable[[np.ndarray], np.ndarray] | None = None,
        max_frames: int | None = None,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_dir():
            raise FileNotFoundError(f"no dataset at {self.path}")

        self.load_lidar = load_lidar
        self.load_stereo = load_stereo
        self.load_depth_gt = load_depth_gt
        self.lidar_transform = lidar_transform

        self.meta = self._read_meta()

        imu = np.loadtxt(self.path / "imu.txt", comments="#", ndmin=2)
        self._imu_t = imu[:, 0] if imu.size else np.empty(0)
        self._imu_gyro = imu[:, 1:4] if imu.size else np.empty((0, 3))
        self._imu_accel = imu[:, 4:7] if imu.size else np.empty((0, 3))

        self._gt_t, self._gt_poses = self._read_optional_tum("groundtruth.txt")
        self._lidar_pose_t, self._lidar_poses = self._read_optional_tum("lidar_poses.txt")
        self._cam_pose_t, self._cam_poses = self._read_optional_tum("cam_poses.txt")

        self._lidar_ns = self._stamps("lidar", ".npy") if load_lidar else []
        self._cam_ns = self._stamps("cam0", ".png") if load_stereo else []

        # The capture timeline is the union of the sensor stamps. In lockstep
        # recordings these coincide exactly; the union keeps the free-running
        # fallback working too.
        stamps = sorted(set(self._lidar_ns) | set(self._cam_ns))
        self._stamps_ns = stamps[:max_frames] if max_frames else stamps

        self._has_lidar = bool(self._lidar_ns)
        self._has_stereo = bool(self._cam_ns)

    # -- metadata ----------------------------------------------------------

    def _read_meta(self) -> dict:
        for name, loader in (("sensor.yaml", _load_yaml), ("sensor.json", json.load)):
            p = self.path / name
            if p.exists():
                with open(p) as fh:
                    return loader(fh) or {}
        return {}

    @property
    def condition(self) -> str:
        return str(self.meta.get("weather", {}).get("condition", self.path.name))

    def _read_optional_tum(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        p = self.path / name
        if not p.exists():
            return np.empty(0), np.empty((0, 4, 4))
        return read_tum(p)

    def _stamps(self, sub: str, suffix: str) -> list[int]:
        d = self.path / sub
        if not d.is_dir():
            return []
        return sorted(int(f.stem) for f in d.glob(f"*{suffix}"))

    # -- lookup helpers ----------------------------------------------------

    @staticmethod
    def _nearest(times: np.ndarray, poses: np.ndarray, t: float,
                 tol: float = 0.05) -> np.ndarray | None:
        """Nearest recorded pose within ``tol`` seconds, else None.

        Returning None rather than the nearest-at-any-distance keeps a gap in
        the recording from quietly becoming a bogus ground-truth association.
        """
        if len(times) == 0:
            return None
        i = int(np.searchsorted(times, t))
        best, bestd = None, np.inf
        for j in (i - 1, i, i + 1):
            if 0 <= j < len(times):
                d = abs(times[j] - t)
                if d < bestd:
                    best, bestd = j, d
        return poses[best] if best is not None and bestd <= tol else None

    # -- iteration ---------------------------------------------------------

    @property
    def has_lidar(self) -> bool:
        return self._has_lidar

    @property
    def has_stereo(self) -> bool:
        return self._has_stereo

    def __len__(self) -> int:
        return len(self._stamps_ns)

    def __iter__(self) -> Iterator[Frame]:
        lidar_set = set(self._lidar_ns)
        cam_set = set(self._cam_ns)
        prev_t = -np.inf

        for ns in self._stamps_ns:
            t = ns / 1e9

            lo = np.searchsorted(self._imu_t, prev_t, side="right")
            hi = np.searchsorted(self._imu_t, t, side="right")
            imu = [
                ImuSample(float(self._imu_t[k]), self._imu_gyro[k], self._imu_accel[k])
                for k in range(lo, hi)
            ]

            scan = None
            if ns in lidar_set:
                pts = np.load(self.path / "lidar" / f"{ns}.npy")
                if self.lidar_transform is not None:
                    pts = self.lidar_transform(pts)
                scan = LidarScan(
                    t=t,
                    points=np.ascontiguousarray(pts, dtype=np.float32),
                    gt_pose=self._nearest(self._lidar_pose_t, self._lidar_poses, t),
                )

            stereo = None
            if ns in cam_set:
                left = cv2.imread(str(self.path / "cam0" / f"{ns}.png"), cv2.IMREAD_COLOR)
                right = cv2.imread(str(self.path / "cam1" / f"{ns}.png"), cv2.IMREAD_COLOR)
                if left is not None and right is not None:
                    depth = None
                    if self.load_depth_gt:
                        dp = self.path / "depth_gt" / f"{ns}.npy"
                        if dp.exists():
                            depth = np.load(dp)
                    stereo = StereoFrame(
                        t=t, left=left, right=right, depth_gt=depth,
                        gt_pose=self._nearest(self._cam_pose_t, self._cam_poses, t),
                    )

            yield Frame(
                t=t, imu=imu, lidar=scan, stereo=stereo,
                gt_pose=self._nearest(self._gt_t, self._gt_poses, t),
            )
            prev_t = t

    def ground_truth(self) -> tuple[np.ndarray, np.ndarray]:
        """Full ground-truth body trajectory: ``(times (N,), poses (N,4,4))``."""
        return self._gt_t, self._gt_poses


def _load_yaml(fh):
    """PyYAML if available, else fall back to JSON (the writer emits a subset
    of YAML that is not valid JSON, so this only helps for .json siblings)."""
    try:
        import yaml
        return yaml.safe_load(fh)
    except ImportError:
        return json.load(fh)


# --------------------------------------------------------------------------
# live
# --------------------------------------------------------------------------

class LiveSource(SensorSource):
    """Streams straight from a running simulator, for ``flight/slam_live.py``.

    Free-running by design: the live demo wants wall-clock responsiveness, not
    the determinism the recorder goes to lockstep lengths for. Frames carry the
    simulator's own timestamps, so the front ends behave identically either way.

    The caller owns the ``airsim`` client. msgpack-rpc's Tornado IOLoop is not
    thread-safe, so if this is polled from a background thread that thread must
    own the connection it passes in -- the pattern used throughout ``flight/``.
    """

    def __init__(self, client, rate_hz: float = 10.0,
                 use_lidar: bool = True, use_stereo: bool = True,
                 with_ground_truth: bool = True) -> None:
        import airsim  # local: keeps the offline path free of the RPC stack

        self._airsim = airsim
        self.client = client
        self.dt = 1.0 / rate_hz
        self.use_lidar = use_lidar
        self.use_stereo = use_stereo
        self.with_ground_truth = with_ground_truth
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    @property
    def has_lidar(self) -> bool:
        return self.use_lidar

    @property
    def has_stereo(self) -> bool:
        return self.use_stereo

    def __iter__(self) -> Iterator[Frame]:
        import time

        from .geometry import airsim_pose_to_matrix, airsim_vec_to_np

        airsim = self._airsim
        prev_imu_t: float | None = None

        while not self._stop:
            tick = time.time()

            imu_d = self.client.getImuData(imu_name=cfg.IMU_NAME, vehicle_name=cfg.DRONE)
            t = int(imu_d.time_stamp) / 1e9

            # Live polling can only ever see the newest IMU sample; the
            # recorder is what captures the full 100 Hz stream.
            samples = []
            if prev_imu_t is None or t > prev_imu_t:
                samples.append(ImuSample(
                    t=t,
                    gyro=airsim_vec_to_np(imu_d.angular_velocity),
                    accel=airsim_vec_to_np(imu_d.linear_acceleration),
                ))
                prev_imu_t = t

            scan = None
            if self.use_lidar:
                d = self.client.getLidarData(lidar_name=cfg.LIDAR_NAME,
                                             vehicle_name=cfg.DRONE)
                if len(d.point_cloud) >= 3:
                    scan = LidarScan(
                        t=int(d.time_stamp) / 1e9 or t,
                        points=np.array(d.point_cloud, dtype=np.float32).reshape(-1, 3),
                        gt_pose=airsim_pose_to_matrix(d.pose),
                    )

            stereo = None
            if self.use_stereo:
                resp = self.client.simGetImages([
                    airsim.ImageRequest(cfg.CAM_LEFT, airsim.ImageType.Scene, False, False),
                    airsim.ImageRequest(cfg.CAM_RIGHT, airsim.ImageType.Scene, False, False),
                ], vehicle_name=cfg.DRONE)
                imgs = [_decode_bgr(r) for r in resp] if len(resp) == 2 else [None, None]
                if imgs[0] is not None and imgs[1] is not None:
                    stereo = StereoFrame(t=t, left=imgs[0], right=imgs[1])

            gt = None
            if self.with_ground_truth:
                k = self.client.simGetGroundTruthKinematics(vehicle_name=cfg.DRONE)
                gt = airsim_pose_to_matrix(k)

            yield Frame(t=t, imu=samples, lidar=scan, stereo=stereo, gt_pose=gt)

            time.sleep(max(0.0, self.dt - (time.time() - tick)))


def _decode_bgr(resp) -> np.ndarray | None:
    """Raw AirSim uint8 response -> (H,W,3) BGR, or None if the frame is junk.

    AirSim hands back 0x0 frames while the render target spins up, and
    occasionally a short buffer; both must be dropped rather than reshaped.
    """
    want = resp.width * resp.height * 3
    if resp.width == 0 or resp.height == 0 or len(resp.image_data_uint8) != want:
        return None
    return np.frombuffer(resp.image_data_uint8, np.uint8).reshape(
        resp.height, resp.width, 3).copy()
