"""
Records a synchronised multi-sensor dataset from a live AirSim run.

Why record at all, when the sim is right there? Because a fair comparison
needs *identical input bytes*. Running SLAM live means every method sees a
different trajectory, different timing jitter, and a different render, so any
difference in the results is partly the harness. Recording once per weather
condition and replaying it into every method removes that entirely -- and lets
the benchmark re-run without the simulator.

Two capture modes:

**lockstep** (default, preferred) -- ``simPause`` the sim, read every sensor at
one exact instant, ``simContinueForTime`` a fixed dt. Zero cross-sensor time
skew and a fully deterministic run. Slower than real time, which does not
matter for an offline dataset.

**free-running** (``--free-running``, or automatic fallback) -- an IMU thread
and a sensor loop poll on the wall clock and everything is associated after
the fact by each message's own ``time_stamp``. Use if ``tools/probe_setup.py``
reports that lockstep stepping misbehaves on this build.

Bandwidth note: colour frames are requested with ``compress=True`` so UE4 does
the PNG encode and we write its bytes straight to disk. The raw path would push
~55 MB/s over RPC and ~80 GB across the study; PNG lands near 3 GB per run.
"""
from __future__ import annotations

import json
import math
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import airsim

from . import config as cfg
from . import weather as weather_mod
from .config import WeatherCondition
from .geometry import (airsim_pose_to_matrix, airsim_vec_to_np, pose_to_matrix,
                       write_tum)

# Stop once the drone is inside this radius of the final waypoint.
FINISH_RADIUS = 6.0
# Hard cap, as a multiple of the analytically expected duration.
DURATION_MARGIN = 1.8
# Stall detector. The route has no obstacle avoidance, so a tree or a roof
# intersecting it pins the drone against the collision mesh while
# moveOnPathAsync keeps pushing. _route_finished never fires, so without this
# the run burns its full step cap -- 20+ minutes -- recording a dataset that
# goes nowhere. At ROUTE_SPEED the drone covers ~20 m per window; anything
# under STALL_DISTANCE means it is not flying.
STALL_WINDOW = 5.0        # sim seconds between progress checks
STALL_DISTANCE = 1.5      # m -- less travel than this over a window is a stall


# --------------------------------------------------------------------------

class _WriterPool:
    """Background disk writers.

    A run writes ~20k files. Doing that inline would stall the capture loop on
    every tick; in lockstep that inflates wall-clock time, and in free-running
    mode it actively drops samples. The payloads are already-encoded bytes or
    numpy arrays, so the writes release the GIL and threads genuinely help.
    """

    def __init__(self, n_workers: int = 3) -> None:
        self.q: queue.Queue = queue.Queue(maxsize=256)
        self.errors: list[str] = []
        self._threads = [
            threading.Thread(target=self._work, daemon=True) for _ in range(n_workers)
        ]
        for t in self._threads:
            t.start()

    def _work(self) -> None:
        while True:
            item = self.q.get()
            if item is None:
                self.q.task_done()
                return
            path, payload = item
            try:
                if isinstance(payload, (bytes, bytearray)):
                    with open(path, "wb") as fh:
                        fh.write(payload)
                else:
                    np.save(path, payload)
            except Exception as exc:                       # pragma: no cover
                self.errors.append(f"{path}: {exc}")
            finally:
                self.q.task_done()

    def submit(self, path: Path, payload) -> None:
        self.q.put((str(path), payload))

    def close(self) -> None:
        self.q.join()
        for _ in self._threads:
            self.q.put(None)
        for t in self._threads:
            t.join(timeout=5.0)


@dataclass
class RecordStats:
    """What actually made it to disk -- surfaced so a short run is obvious."""
    mode: str = "lockstep"
    imu_samples: int = 0
    lidar_scans: int = 0
    stereo_frames: int = 0
    dropped_frames: int = 0
    sim_seconds: float = 0.0
    wall_seconds: float = 0.0
    lidar_points_mean: float = 0.0
    repeated_lidar: int = 0
    measured_imu_hz: float = 0.0
    collisions: int = 0
    collided_with: list[str] = field(default_factory=list)
    stuck: bool = False
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        rt = (self.sim_seconds / self.wall_seconds) if self.wall_seconds else 0.0
        out = (
            f"mode={self.mode}  sim={self.sim_seconds:.1f}s  wall={self.wall_seconds:.1f}s "
            f"({rt:.2f}x real time)\n"
            f"  IMU {self.imu_samples}   LiDAR {self.lidar_scans} "
            f"(mean {self.lidar_points_mean:.0f} pts)   stereo {self.stereo_frames}"
            f"   dropped {self.dropped_frames}"
        )
        if self.measured_imu_hz:
            out += (f"\n  measured IMU {self.measured_imu_hz:.1f} Hz "
                    f"(declared {cfg.IMU_RATE_HZ:.0f})")
        if self.repeated_lidar:
            out += (f"   resampled LiDAR repeats {self.repeated_lidar} "
                    f"(skipped, not on disk)")
        if self.collisions:
            out += (f"\n  collisions {self.collisions} "
                    f"({', '.join(self.collided_with[:4])})")
        if self.stuck:
            out += "\n  ABORTED: drone stuck -- this recording is incomplete"
        return out


# --------------------------------------------------------------------------

class DatasetRecorder:
    def __init__(
        self,
        client: airsim.MultirotorClient,
        out_dir: Path,
        condition: WeatherCondition,
        lockstep: bool = True,
        record_depth_gt: bool = False,
        stereo: bool = True,
    ) -> None:
        self.client = client
        self.out = Path(out_dir)
        self.condition = condition
        self.lockstep = lockstep
        self.record_depth_gt = record_depth_gt
        self.stereo = stereo

        self.stats = RecordStats(mode="lockstep" if lockstep else "free-running")
        self._writer = _WriterPool()

        # accumulated trajectories, flushed to TUM files at the end
        self._gt_t: list[float] = []
        self._gt_T: list[np.ndarray] = []
        self._lidar_pose_t: list[float] = []
        self._lidar_pose_T: list[np.ndarray] = []
        self._cam_pose_t: list[float] = []
        self._cam_pose_T: list[np.ndarray] = []
        self._imu_rows: list[tuple] = []
        self._point_counts: list[int] = []
        self._cam_pose_checked = False

        # stall / collision tracking
        self._stall_ref_pos: np.ndarray | None = None
        self._stall_ref_t = 0.0
        self._last_collision_ts = 0
        self._last_lidar_ts = 0

    # -- setup -------------------------------------------------------------

    def _make_dirs(self) -> None:
        for sub in ("lidar", "cam0", "cam1") + (("depth_gt",) if self.record_depth_gt else ()):
            (self.out / sub).mkdir(parents=True, exist_ok=True)

    def _reported_fov(self) -> float | None:
        """Horizontal FOV as the live sim reports it, or None if unavailable."""
        try:
            return round(float(
                self.client.simGetCameraInfo(cfg.CAM_LEFT, vehicle_name=cfg.DRONE).fov), 4)
        except Exception:
            return None

    def _git_sha(self) -> str:
        try:
            return subprocess.run(
                ["git", "-C", str(cfg.REPO_ROOT), "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip() or "unknown"
        except Exception:
            return "unknown"

    def _write_meta(self) -> None:
        meta = {
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_sha": self._git_sha(),
            "capture_mode": self.stats.mode,
            "vehicle": cfg.DRONE,
            "weather": {
                "condition": self.condition.name,
                "airsim_params": dict(self.condition.airsim_params),
                "rain_rate_mmph": self.condition.rain_rate_mmph,
                "fog_visibility_m": (None if math.isinf(self.condition.fog_visibility_m)
                                     else self.condition.fog_visibility_m),
                "note": ("AirSim weather is a RENDERING effect only. The camera "
                         "streams in this dataset are genuinely degraded; the LiDAR "
                         "scans are NOT -- raycasts pass through rain and fog "
                         "particles. Apply slam.degradation to model LiDAR "
                         "degradation at a matching severity."),
            },
            "camera": {
                "width": cfg.IMAGE_WIDTH, "height": cfg.IMAGE_HEIGHT,
                "fov_degrees": cfg.FOV_DEGREES,
                "fx": cfg.FX, "fy": cfg.FY, "cx": cfg.CX, "cy": cfg.CY,
                "baseline_m": cfg.STEREO_BASELINE,
                # What the simulator actually reports, next to what we declared.
                # AirSim typically returns 89.9 deg for a declared 90 -- a float
                # artifact worth ~0.17% of fx, far below the stereo depth noise
                # floor, but recorded so the dataset is self-describing rather
                # than relying on the declaration being exact.
                "fov_degrees_reported_by_sim": self._reported_fov(),
                "model": "ideal pinhole, zero distortion",
                "T_body_cam_left": cfg.T_BODY_CAM_LEFT.tolist(),
                "T_body_cam_right": cfg.T_BODY_CAM_RIGHT.tolist(),
            },
            "lidar": {
                "channels": cfg.LIDAR_CHANNELS,
                "range_m": cfg.LIDAR_RANGE,
                "points_per_second": cfg.LIDAR_POINTS_PER_SECOND,
                "rotations_per_second": cfg.LIDAR_ROTATIONS_PER_SECOND,
                "vfov_upper_deg": cfg.LIDAR_VFOV_UPPER,
                "vfov_lower_deg": cfg.LIDAR_VFOV_LOWER,
                "data_frame": "SensorLocalFrame",
                "T_body_lidar": cfg.T_BODY_LIDAR.tolist(),
            },
            # Declared vs measured. simContinueForTime does not deliver exactly
            # the dt it is asked for -- this build advances ~9 ms for a
            # requested 10 -- so the nominal rates are a request, not a fact.
            # Nothing in slam/ computes on the declared rate (integration uses
            # the per-sample stamps), but a dataset that only states the
            # request invites someone downstream to trust it.
            "rates": {"imu_hz": cfg.IMU_RATE_HZ,
                      "sensor_hz": cfg.IMU_RATE_HZ / cfg.SENSOR_DECIMATION,
                      "imu_hz_measured": round(self.stats.measured_imu_hz, 3) or None,
                      "step_dt_requested": cfg.STEP_DT},
            # A collision puts a real disturbance into the IMU and the
            # trajectory, and "stuck" means the circuit was never completed --
            # so neither is safe to leave implicit in a dataset that gets
            # replayed into every method.
            "integrity": {"collisions": self.stats.collisions,
                          "collided_with": list(self.stats.collided_with),
                          "aborted_stuck": self.stats.stuck},
            "route": {"altitude_ned": cfg.ROUTE_ALTITUDE, "speed_mps": cfg.ROUTE_SPEED,
                      "size_m": [cfg.ROUTE_LENGTH_X, cfg.ROUTE_LENGTH_Y],
                      "laps": cfg.ROUTE_LAPS},
        }
        try:
            import yaml
            with open(self.out / "sensor.yaml", "w") as fh:
                yaml.safe_dump(meta, fh, sort_keys=False, default_flow_style=False)
        except ImportError:
            with open(self.out / "sensor.json", "w") as fh:
                json.dump(meta, fh, indent=2)

    # -- capture primitives ------------------------------------------------

    def _capture_imu(self) -> int:
        """Read one IMU sample and the body ground truth. Returns sim time (ns)."""
        d = self.client.getImuData(imu_name=cfg.IMU_NAME, vehicle_name=cfg.DRONE)
        ns = int(d.time_stamp)
        t = ns / 1e9
        g = d.angular_velocity
        a = d.linear_acceleration
        self._imu_rows.append((t, g.x_val, g.y_val, g.z_val, a.x_val, a.y_val, a.z_val))

        k = self.client.simGetGroundTruthKinematics(vehicle_name=cfg.DRONE)
        self._gt_t.append(t)
        self._gt_T.append(airsim_pose_to_matrix(k))
        self.stats.imu_samples += 1
        return ns

    def _capture_lidar(self, ns: int) -> None:
        d = self.client.getLidarData(lidar_name=cfg.LIDAR_NAME, vehicle_name=cfg.DRONE)
        if len(d.point_cloud) < 3:
            self.stats.dropped_frames += 1
            return

        pts = np.array(d.point_cloud, dtype=np.float32).reshape(-1, 3)
        stamp = int(d.time_stamp) or ns

        # The capture loop samples faster than the LiDAR spins (a lockstep tick
        # advances ~9 ms, not the requested 10, so sensors are polled at ~11 Hz
        # against a 10 Hz sweep), so roughly one poll in nine returns the sweep
        # we already have. Left alone it rewrites the same file and adds a
        # duplicate row to lidar_poses.txt, making lidar_scans overcount what is
        # actually on disk. Oversampling is the right side to err on -- it never
        # misses a sweep -- so drop the repeat rather than slowing the poll.
        if stamp == self._last_lidar_ts:
            self.stats.repeated_lidar += 1
            return
        self._last_lidar_ts = stamp

        self._writer.submit(self.out / "lidar" / f"{stamp}.npy", pts)

        self._lidar_pose_t.append(stamp / 1e9)
        self._lidar_pose_T.append(airsim_pose_to_matrix(d.pose))
        self._point_counts.append(len(pts))
        self.stats.lidar_scans += 1

    def _capture_stereo(self, ns: int) -> None:
        reqs = [
            # compress=True -> UE4 returns PNG bytes we write straight through.
            airsim.ImageRequest(cfg.CAM_LEFT, airsim.ImageType.Scene, False, True),
            airsim.ImageRequest(cfg.CAM_RIGHT, airsim.ImageType.Scene, False, True),
        ]
        if self.record_depth_gt:
            reqs.append(airsim.ImageRequest(cfg.CAM_LEFT, airsim.ImageType.DepthPlanar,
                                            True, False))

        resp = self.client.simGetImages(reqs, vehicle_name=cfg.DRONE)
        if len(resp) < 2 or not resp[0].image_data_uint8 or not resp[1].image_data_uint8:
            self.stats.dropped_frames += 1
            return

        self._writer.submit(self.out / "cam0" / f"{ns}.png", bytes(resp[0].image_data_uint8))
        self._writer.submit(self.out / "cam1" / f"{ns}.png", bytes(resp[1].image_data_uint8))

        if self.record_depth_gt and len(resp) > 2 and resp[2].image_data_float:
            depth = np.array(resp[2].image_data_float, dtype=np.float32).reshape(
                resp[2].height, resp[2].width)
            self._writer.submit(self.out / "depth_gt" / f"{ns}.npy", depth)

        self._record_camera_pose(ns, resp[0])
        self.stats.stereo_frames += 1

    def _record_camera_pose(self, ns: int, resp) -> None:
        """Ground-truth left-camera OPTICAL pose in world NED.

        AirSim reports the camera pose in its body-like convention (+x forward),
        so it is composed with ``R_BODY_FROM_OPTICAL`` to reach the optical frame
        (+z forward) that OpenCV's PnP and triangulation expect. Getting this
        wrong would silently rotate every stereo result by 90 degrees, so on the
        first frame it is cross-checked against the pose derived independently
        from the body ground truth and the configured extrinsic.
        """
        T_cam_ned = pose_to_matrix(
            airsim_vec_to_np(resp.camera_position),
            _quat_xyzw(resp.camera_orientation),
        )
        T_opt = np.eye(4)
        T_opt[:3, :3] = cfg.R_BODY_FROM_OPTICAL
        T_world_cam = T_cam_ned @ T_opt

        if np.linalg.norm(T_cam_ned[:3, 3]) < 1e-9 and self._gt_T:
            # AirSim did not populate the field; fall back to body o extrinsic.
            T_world_cam = self._gt_T[-1] @ cfg.T_BODY_CAM_LEFT

        if not self._cam_pose_checked and self._gt_T:
            derived = self._gt_T[-1] @ cfg.T_BODY_CAM_LEFT
            err = float(np.linalg.norm(derived[:3, 3] - T_world_cam[:3, 3]))
            if err > 0.5:
                self.stats.warnings.append(
                    f"camera GT pose disagrees with body-o-extrinsic by {err:.2f} m -- "
                    f"check the Cameras block in settings.json against "
                    f"config.T_BODY_CAM_LEFT")
            self._cam_pose_checked = True

        self._cam_pose_t.append(ns / 1e9)
        self._cam_pose_T.append(T_world_cam)

    # -- collision / stall watchdog ----------------------------------------

    def _baseline_collision(self) -> None:
        """Discard the collision AirSim is already reporting before we start.

        ``simGetCollisionInfo`` latches the most recent collision and keeps
        reporting it, and the drone has been resting on the ground since spawn
        -- so the very first poll always returns that ground contact ('Road_89'
        in AirSimNH) and the watchdog counts a collision that never happened in
        flight. Latch its timestamp here so only genuinely new events count.
        """
        try:
            c = self.client.simGetCollisionInfo(vehicle_name=cfg.DRONE)
            if c.has_collided:
                self._last_collision_ts = int(c.time_stamp)
        except Exception:
            pass

    def _poll_collision(self) -> None:
        """Note any new collision. Cheap enough to call at sensor rate.

        A glancing hit does not necessarily end the run -- the drone often
        bounces off and carries on -- but it does put a physical disturbance
        into the IMU and a jolt into the trajectory, so the dataset has to
        declare it rather than let it be discovered as an unexplained spike
        in the ATE later.
        """
        try:
            c = self.client.simGetCollisionInfo(vehicle_name=cfg.DRONE)
        except Exception:
            return
        if not c.has_collided:
            return
        ts = int(c.time_stamp)
        if ts == self._last_collision_ts:
            return                       # same event we already counted
        self._last_collision_ts = ts
        self.stats.collisions += 1
        name = c.object_name or "?"
        if name not in self.stats.collided_with:
            self.stats.collided_with.append(name)

    def _is_stuck(self, elapsed: float) -> bool:
        """True once the drone has stopped making progress along the route.

        Measured on the ground-truth position over a sliding window of sim
        time, so it is independent of how slowly the lockstep loop happens to
        be running in wall clock.
        """
        if not self._gt_T:
            return False
        pos = self._gt_T[-1][:3, 3]

        if self._stall_ref_pos is None:
            self._stall_ref_pos, self._stall_ref_t = pos.copy(), elapsed
            return False
        if elapsed - self._stall_ref_t < STALL_WINDOW:
            return False

        moved = float(np.linalg.norm(pos - self._stall_ref_pos))
        self._stall_ref_pos, self._stall_ref_t = pos.copy(), elapsed
        if moved >= STALL_DISTANCE:
            return False

        obj = self.stats.collided_with[-1] if self.stats.collided_with else "unknown object"
        self.stats.stuck = True
        self.stats.warnings.append(
            f"STUCK at NED ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) after "
            f"{elapsed:.1f} s of sim time -- moved {moved:.2f} m in {STALL_WINDOW:.0f} s, "
            f"last collision with '{obj}'. The route runs through geometry at this "
            f"altitude; raise slam.config.ROUTE_ALTITUDE (currently "
            f"{abs(cfg.ROUTE_ALTITUDE):.0f} m AGL) or move the circuit, then re-record.")
        return True

    # -- flight ------------------------------------------------------------

    def _takeoff_and_start_route(self) -> tuple[list, float]:
        wps = cfg.route_waypoints()
        length = sum(
            float(np.linalg.norm(np.array(wps[i + 1]) - np.array(wps[i])))
            for i in range(len(wps) - 1)
        )
        expected = length / cfg.ROUTE_SPEED

        print(f"  route: {len(wps)} waypoints, {length:.0f} m, "
              f"~{expected / 60:.1f} min of sim time")

        self.client.enableApiControl(True, cfg.DRONE)
        self.client.armDisarm(True, cfg.DRONE)
        self.client.takeoffAsync(vehicle_name=cfg.DRONE).join()
        self.client.moveToPositionAsync(wps[0][0], wps[0][1], wps[0][2], 5.0,
                                        vehicle_name=cfg.DRONE).join()
        self.client.hoverAsync(vehicle_name=cfg.DRONE).join()
        time.sleep(1.0)

        path = [airsim.Vector3r(*w) for w in wps]
        self.client.moveOnPathAsync(
            path, cfg.ROUTE_SPEED,
            timeout_sec=expected * DURATION_MARGIN,
            drivetrain=airsim.DrivetrainType.ForwardOnly,
            yaw_mode=airsim.YawMode(False, 0),
            lookahead=-1, adaptive_lookahead=1,
            vehicle_name=cfg.DRONE,
        )
        return wps, expected

    def _route_finished(self, wps, elapsed_sim: float, expected: float) -> bool:
        """Done when back at the final waypoint, having flown most of the route.

        The elapsed-time guard matters because the route starts *and* ends at
        the same point -- without it, the very first tick would look finished.
        """
        if elapsed_sim < 0.6 * expected:
            return False
        if not self._gt_T:
            return False
        pos = self._gt_T[-1][:3, 3]
        goal = np.array(wps[-1])
        return float(np.linalg.norm(pos[:2] - goal[:2])) < FINISH_RADIUS

    # -- capture loops -----------------------------------------------------

    def _run_lockstep(self, wps, expected: float) -> None:
        max_steps = int(expected * DURATION_MARGIN * cfg.IMU_RATE_HZ)
        t_start = None
        self.client.simPause(True)
        try:
            for step in range(max_steps):
                ns = self._capture_imu()
                if t_start is None:
                    t_start = ns
                elapsed = (ns - t_start) / 1e9

                if step % cfg.SENSOR_DECIMATION == 0:
                    self._capture_lidar(ns)
                    if self.stereo:
                        self._capture_stereo(ns)
                    self._poll_collision()
                    if self._is_stuck(elapsed):
                        print(f"\n  ABORT: {self.stats.warnings[-1]}")
                        self.stats.sim_seconds = elapsed
                        return
                    if step % (cfg.SENSOR_DECIMATION * 50) == 0:
                        self._progress(elapsed, expected)

                if self._route_finished(wps, elapsed, expected):
                    print(f"\n  route complete at {elapsed:.1f} s of sim time")
                    break

                self.client.simContinueForTime(cfg.STEP_DT)
            else:
                self.stats.warnings.append(
                    f"hit the {max_steps}-step cap before reaching the final "
                    f"waypoint; the recording may be truncated")
            self.stats.sim_seconds = elapsed
        finally:
            self.client.simPause(False)

    def _run_free(self, wps, expected: float) -> None:
        """Wall-clock fallback. IMU on its own thread, sensors in the main loop.

        Cross-sensor skew is real here (up to one poll interval), which is
        exactly why lockstep is preferred -- but every message carries its own
        simulator timestamp, so ``DatasetSource`` still associates correctly.
        """
        stop = threading.Event()
        t0 = [None]

        def imu_thread():
            imu_client = airsim.MultirotorClient()   # Tornado IOLoop is not thread-safe
            imu_client.confirmConnection()
            period = 1.0 / cfg.IMU_RATE_HZ
            while not stop.is_set():
                tick = time.time()
                d = imu_client.getImuData(imu_name=cfg.IMU_NAME, vehicle_name=cfg.DRONE)
                t = int(d.time_stamp) / 1e9
                g, a = d.angular_velocity, d.linear_acceleration
                self._imu_rows.append((t, g.x_val, g.y_val, g.z_val,
                                       a.x_val, a.y_val, a.z_val))
                k = imu_client.simGetGroundTruthKinematics(vehicle_name=cfg.DRONE)
                self._gt_t.append(t)
                self._gt_T.append(airsim_pose_to_matrix(k))
                self.stats.imu_samples += 1
                if t0[0] is None:
                    t0[0] = t
                time.sleep(max(0.0, period - (time.time() - tick)))

        th = threading.Thread(target=imu_thread, daemon=True)
        th.start()
        period = cfg.SENSOR_DECIMATION / cfg.IMU_RATE_HZ
        deadline = time.time() + expected * DURATION_MARGIN
        n = 0
        try:
            while time.time() < deadline:
                tick = time.time()
                ns = int(time.time() * 1e9) if t0[0] is None else None
                d = self.client.getImuData(imu_name=cfg.IMU_NAME, vehicle_name=cfg.DRONE)
                ns = int(d.time_stamp)

                self._capture_lidar(ns)
                if self.stereo:
                    self._capture_stereo(ns)

                elapsed = (ns / 1e9 - (t0[0] or ns / 1e9))
                if n % 50 == 0:
                    self._progress(elapsed, expected)
                self._poll_collision()
                if self._is_stuck(elapsed):
                    print(f"\n  ABORT: {self.stats.warnings[-1]}")
                    break
                if self._route_finished(wps, elapsed, expected):
                    print(f"\n  route complete at {elapsed:.1f} s of sim time")
                    break
                n += 1
                time.sleep(max(0.0, period - (time.time() - tick)))
            self.stats.sim_seconds = elapsed
        finally:
            stop.set()
            th.join(timeout=3.0)

    def _progress(self, elapsed: float, expected: float) -> None:
        pct = min(100.0, 100.0 * elapsed / expected) if expected else 0.0
        print(f"\r  [{self.condition.name}] {pct:5.1f}%  sim {elapsed:6.1f}s  "
              f"lidar {self.stats.lidar_scans:5d}  stereo {self.stats.stereo_frames:5d}",
              end="", flush=True)

    # -- entry point -------------------------------------------------------

    def record(self) -> RecordStats:
        self._make_dirs()
        self._write_meta()

        wall0 = time.time()
        try:
            print(f"\nApplying weather: {self.condition.name} "
                  f"({self.condition.description})")
            weather_mod.apply(self.client, self.condition)

            wps, expected = self._takeoff_and_start_route()
            self._baseline_collision()
            print(f"  capturing ({self.stats.mode})...")

            if self.lockstep:
                self._run_lockstep(wps, expected)
            else:
                self._run_free(wps, expected)
        finally:
            self.stats.wall_seconds = time.time() - wall0
            print("\n  finishing: flushing writers and landing...")
            try:
                self.client.simPause(False)
                self.client.cancelLastTask(vehicle_name=cfg.DRONE)
                weather_mod.reset(self.client)
                self.client.landAsync(vehicle_name=cfg.DRONE).join()
                self.client.armDisarm(False, cfg.DRONE)
                self.client.enableApiControl(False, cfg.DRONE)
            except Exception as exc:
                self.stats.warnings.append(f"cleanup: {exc}")

            self._flush()

        return self.stats

    def _flush(self) -> None:
        self._writer.close()
        self.stats.warnings.extend(self._writer.errors[:5])

        if len(self._imu_rows) > 1:
            span = self._imu_rows[-1][0] - self._imu_rows[0][0]
            if span > 0:
                self.stats.measured_imu_hz = (len(self._imu_rows) - 1) / span

        if self._imu_rows:
            np.savetxt(
                self.out / "imu.txt", np.array(self._imu_rows),
                fmt="%.9f", header="timestamp wx wy wz ax ay az  (body NED, rad/s, m/s^2)",
            )
        if self._gt_T:
            write_tum(self.out / "groundtruth.txt", self._gt_t, self._gt_T,
                      header="ground-truth BODY pose, world NED (simGetGroundTruthKinematics)")
        if self._lidar_pose_T:
            write_tum(self.out / "lidar_poses.txt", self._lidar_pose_t, self._lidar_pose_T,
                      header="ground-truth LIDAR SENSOR pose, world NED (LidarData.pose)")
        if self._cam_pose_T:
            write_tum(self.out / "cam_poses.txt", self._cam_pose_t, self._cam_pose_T,
                      header="ground-truth LEFT CAMERA OPTICAL pose, world NED")

        self.stats.lidar_points_mean = float(np.mean(self._point_counts)) if self._point_counts else 0.0

        # Update the metadata now that the real counts are known.
        self._write_meta()


def _quat_xyzw(q) -> np.ndarray:
    return np.array([q.x_val, q.y_val, q.z_val, q.w_val], dtype=np.float64)
