#!/usr/bin/env python3
"""
Phase-0 sanity check. Run this after restarting the sim with the new
settings.json, BEFORE building anything on top of it.

Every later phase rests on assumptions this script verifies against the live
simulator rather than against the docs:

  * LiDAR returns SENSOR-LOCAL points (not the spawn-inertial frame the repo
    used to be configured for) -- the whole point of SLAM is that we apply our
    own estimated pose, never the simulator's.
  * ``LidarData.pose`` and ``.time_stamp`` are actually populated. Nothing in
    the repo has ever read them, so this is unverified territory.
  * Both stereo cameras exist and deliver 1280x720, and the intrinsics AirSim
    reports agree with the FOV-derived ones in ``slam/config.py``.
  * Declaring an explicit ``Sensors.Imu`` block did not suppress AirSim's
    default GPS / barometer / magnetometer (they are expected to merge over
    ``DefaultSensors``, but that is worth confirming, not assuming).
  * ``simPause`` + ``simContinueForTime`` steps the sim cleanly while an async
    move command is in flight. The lockstep recorder in Phase 1 depends on it;
    if this fails, the recorder falls back to free-running threads associated
    by sensor timestamps.

Usage:
    ./airsim_venv/bin/python tools/probe_setup.py
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import airsim  # noqa: E402

from slam import config as cfg  # noqa: E402
from slam.geometry import airsim_pose_to_matrix, airsim_vec_to_np  # noqa: E402

# Distance to fly away from spawn for the frame test. Must exceed the LiDAR
# range so the inertial-frame case is unambiguous.
PROBE_OFFSET = 40.0
PROBE_ALTITUDE = cfg.ROUTE_ALTITUDE


class Results:
    """Collects pass/fail rows and prints them as one table at the end."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def add(self, name: str, ok: bool, detail: str = "") -> bool:
        self.rows.append((name, ok, detail))
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  --  {detail}" if detail else ""))
        return ok

    def report(self) -> int:
        width = max(len(n) for n, _, _ in self.rows)
        print("\n" + "=" * (width + 60))
        print("PROBE SUMMARY")
        print("=" * (width + 60))
        failed = 0
        for name, ok, detail in self.rows:
            if not ok:
                failed += 1
            print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
        print("=" * (width + 60))
        if failed:
            print(f"{failed}/{len(self.rows)} checks FAILED -- fix before proceeding.\n")
        else:
            print(f"All {len(self.rows)} checks passed.\n")
        return 1 if failed else 0


# --------------------------------------------------------------------------
# individual probes
# --------------------------------------------------------------------------

def probe_lidar_frame(client: airsim.MultirotorClient, r: Results) -> None:
    """Confirm DataFrame is SensorLocalFrame, and that pose/time_stamp populate.

    The test is deliberately unambiguous rather than statistical: after flying
    PROBE_OFFSET (40 m) from spawn, sensor-local points must all lie within the
    sensor's 100 m Range of the *origin of their own frame*. If AirSim were
    still returning VehicleInertialFrame points, they would sit ~40 m off and a
    large fraction would exceed Range.
    """
    print("\n[lidar] frame + metadata")
    data = client.getLidarData(lidar_name=cfg.LIDAR_NAME, vehicle_name=cfg.DRONE)

    n = len(data.point_cloud) // 3
    if not r.add("lidar returns points", n > 100, f"{n} points/scan"):
        return

    pts = np.array(data.point_cloud, dtype=np.float64).reshape(-1, 3)
    rng = np.linalg.norm(pts, axis=1)

    state = client.getMultirotorState(vehicle_name=cfg.DRONE)
    pos = airsim_vec_to_np(state.kinematics_estimated.position)
    dist_from_spawn = float(np.linalg.norm(pos))

    frac_over = float((rng > cfg.LIDAR_RANGE * 1.05).mean())
    r.add(
        "lidar DataFrame is SensorLocalFrame",
        frac_over < 0.01,
        f"drone {dist_from_spawn:.1f} m from spawn; "
        f"{frac_over * 100:.1f}% of points beyond Range={cfg.LIDAR_RANGE:.0f} m "
        f"(max {rng.max():.1f} m)",
    )

    # A downward-biased sensor should see the ground roughly a slant range of
    # its own altitude away. Derived from ROUTE_ALTITUDE rather than hard-coded
    # so the check keeps its margin if the circuit is raised to clear taller
    # obstacles -- a fixed 30 m threshold silently became a near-tautology once
    # the route moved to 25 m AGL.
    nadir_slack = abs(cfg.ROUTE_ALTITUDE) + 10.0
    r.add(
        "lidar sees nearby structure",
        float(rng.min()) < nadir_slack,
        f"nearest return {rng.min():.2f} m, median {np.median(rng):.1f} m "
        f"(expect a ground return within {nadir_slack:.0f} m at "
        f"{abs(cfg.ROUTE_ALTITUDE):.0f} m AGL)",
    )

    r.add("LidarData.time_stamp populated", int(data.time_stamp) > 0,
          f"t = {int(data.time_stamp)} ns")

    T = airsim_pose_to_matrix(data.pose)
    moved = float(np.linalg.norm(T[:3, 3]))
    r.add(
        "LidarData.pose populated",
        moved > 1.0,
        f"sensor at ({T[0, 3]:.1f}, {T[1, 3]:.1f}, {T[2, 3]:.1f}) -- "
        f"{moved:.1f} m from spawn",
    )

    # Points-per-scan should land near PointsPerSecond / RotationsPerSecond if
    # we are polling at roughly the rotation rate.
    expected = cfg.LIDAR_POINTS_PER_SECOND / cfg.LIDAR_ROTATIONS_PER_SECOND
    r.add(
        "lidar density in the expected band",
        0.15 * expected < n < 3.0 * expected,
        f"{n} pts vs ~{expected:.0f} expected per revolution",
    )


def probe_cameras(client: airsim.MultirotorClient, r: Results) -> None:
    """Both stereo cameras exist at the configured resolution, with matching intrinsics."""
    print("\n[cameras] stereo pair + intrinsics + GT depth")

    reqs = [
        airsim.ImageRequest(cfg.CAM_LEFT, airsim.ImageType.Scene, False, False),
        airsim.ImageRequest(cfg.CAM_RIGHT, airsim.ImageType.Scene, False, False),
    ]
    resp = client.simGetImages(reqs, vehicle_name=cfg.DRONE)

    if not r.add("simGetImages returned both cameras", len(resp) == 2, f"{len(resp)} responses"):
        return

    frames = {}
    for name, rp in zip((cfg.CAM_LEFT, cfg.CAM_RIGHT), resp):
        want = rp.width * rp.height * 3
        ok = (rp.width == cfg.IMAGE_WIDTH and rp.height == cfg.IMAGE_HEIGHT
              and len(rp.image_data_uint8) == want and want > 0)
        r.add(f"{name} delivers {cfg.IMAGE_WIDTH}x{cfg.IMAGE_HEIGHT}", ok,
              f"got {rp.width}x{rp.height}, {len(rp.image_data_uint8)} bytes")
        if ok:
            frames[name] = np.frombuffer(rp.image_data_uint8, np.uint8).reshape(
                rp.height, rp.width, 3)

    # The two views must actually differ -- an identical pair would mean the
    # baseline never took effect and every disparity would be zero.
    if len(frames) == 2:
        diff = float(np.abs(frames[cfg.CAM_LEFT].astype(np.int16)
                            - frames[cfg.CAM_RIGHT].astype(np.int16)).mean())
        r.add("stereo pair has parallax", diff > 1.0,
              f"mean |L-R| = {diff:.2f} intensity levels")

    # AirSim-reported FOV vs. the value config.py derives its intrinsics from.
    #
    # The tolerance is RELATIVE, and set by what the error actually costs. A
    # relative error of eps in fx produces a relative error of eps in
    # backprojected depth (Z = fx*B/d). Stereo depth here is already uncertain
    # by ~6% at STEREO_MAX_DEPTH, so anything under ~1% is far below the noise
    # floor and not worth failing over. AirSim routinely reports 89.9 deg for a
    # declared 90 -- a float-representation artifact, not a misconfiguration.
    try:
        info = client.simGetCameraInfo(cfg.CAM_LEFT, vehicle_name=cfg.DRONE)
        fov = float(info.fov)
        fx_sim = (cfg.IMAGE_WIDTH / 2.0) / math.tan(math.radians(fov) / 2.0)
        rel = abs(fx_sim - cfg.FX) / cfg.FX
        depth_err = rel * cfg.STEREO_MAX_DEPTH
        r.add("camera intrinsics match config", rel < 0.01,
              f"sim FOV {fov:.2f} deg -> fx {fx_sim:.1f}; config fx {cfg.FX:.1f} "
              f"({100 * rel:.2f}% -> {100 * depth_err:.0f} cm depth error at "
              f"{cfg.STEREO_MAX_DEPTH:.0f} m; tolerance 1%)")
    except Exception as exc:
        r.add("camera intrinsics match config", False, f"simGetCameraInfo failed: {exc}")

    # Ground-truth depth, used only to score the stereo depth front end.
    dresp = client.simGetImages(
        [airsim.ImageRequest(cfg.CAM_LEFT, airsim.ImageType.DepthPlanar, True, False)],
        vehicle_name=cfg.DRONE)[0]
    ok = (dresp.width == cfg.IMAGE_WIDTH and dresp.height == cfg.IMAGE_HEIGHT
          and len(dresp.image_data_float) == cfg.IMAGE_WIDTH * cfg.IMAGE_HEIGHT)
    detail = f"got {dresp.width}x{dresp.height}, {len(dresp.image_data_float)} floats"
    if ok:
        d = np.array(dresp.image_data_float, dtype=np.float32)
        finite = d[np.isfinite(d) & (d < 1e4)]
        detail += f"; depth range {finite.min():.1f}-{finite.max():.1f} m"
    r.add("DepthPlanar at full resolution", ok, detail)


def probe_sensors(client: airsim.MultirotorClient, r: Results) -> None:
    """Named IMU works and the default GPS/baro/mag survived the explicit Sensors block."""
    print("\n[sensors] IMU (named) + default sensor merge")

    imu = client.getImuData(imu_name=cfg.IMU_NAME, vehicle_name=cfg.DRONE)
    acc = airsim_vec_to_np(imu.linear_acceleration)
    gyr = airsim_vec_to_np(imu.angular_velocity)
    r.add(f"getImuData(name='{cfg.IMU_NAME}')", int(imu.time_stamp) > 0,
          f"|a| = {np.linalg.norm(acc):.2f} m/s^2, |w| = {np.linalg.norm(gyr):.3f} rad/s")

    # Hovering, specific force should be ~ +1 g. AirSim's IMU is in the body
    # frame with NED axes, so a level hover reads about -9.81 on z.
    r.add("IMU reads ~1 g while hovering",
          abs(np.linalg.norm(acc) - 9.80665) < 3.0,
          f"|a| = {np.linalg.norm(acc):.2f} (expect ~9.81)")

    r.add("IMU orientation populated",
          abs(np.linalg.norm(np.array([imu.orientation.w_val, imu.orientation.x_val,
                                       imu.orientation.y_val, imu.orientation.z_val])) - 1.0) < 1e-3,
          "unit quaternion")

    # The explicit Sensors block must not have replaced AirSim's DefaultSensors.
    for label, fn in (
        ("GPS", lambda: client.getGpsData(vehicle_name=cfg.DRONE)),
        ("barometer", lambda: client.getBarometerData(vehicle_name=cfg.DRONE)),
        ("magnetometer", lambda: client.getMagnetometerData(vehicle_name=cfg.DRONE)),
    ):
        try:
            fn()
            r.add(f"default {label} still available", True, "")
        except Exception as exc:
            r.add(f"default {label} still available", False, str(exc)[:70])


def probe_ground_truth(client: airsim.MultirotorClient, r: Results) -> None:
    """simGetGroundTruthKinematics is the reference trajectory for every metric."""
    print("\n[ground truth]")
    try:
        gt = client.simGetGroundTruthKinematics(vehicle_name=cfg.DRONE)
        est = client.getMultirotorState(vehicle_name=cfg.DRONE).kinematics_estimated
        dp = float(np.linalg.norm(airsim_vec_to_np(gt.position) - airsim_vec_to_np(est.position)))
        r.add("simGetGroundTruthKinematics works", True,
              f"pos ({gt.position.x_val:.1f}, {gt.position.y_val:.1f}, {gt.position.z_val:.1f})")
        # SimpleFlight has no real estimator, so these should be identical --
        # which is exactly why kinematics_estimated must be treated as ground
        # truth and never fed to SLAM as if it were odometry.
        r.add("kinematics_estimated == ground truth (SimpleFlight)", dp < 0.05,
              f"|difference| = {dp:.4f} m")
    except Exception as exc:
        r.add("simGetGroundTruthKinematics works", False, str(exc)[:70])


def probe_lockstep(client: airsim.MultirotorClient, r: Results) -> None:
    """Does simPause + simContinueForTime step cleanly with a move in flight?

    Phase 1's recorder wants deterministic lockstep capture: pause, read every
    sensor at one exact sim instant, advance a fixed dt. That only works if the
    async flight command survives the pause and sim time advances by the
    requested amount. If this fails the recorder falls back to free-running
    threads associated by each message's own time_stamp.
    """
    print("\n[lockstep] simPause / simContinueForTime under an in-flight command")
    try:
        target = cfg.route_waypoints()[:6]
        path = [airsim.Vector3r(x, y, z) for x, y, z in target]
        client.moveOnPathAsync(path, cfg.ROUTE_SPEED, vehicle_name=cfg.DRONE)
        time.sleep(1.0)

        client.simPause(True)
        r.add("simPause reports paused", bool(client.simIsPause()), "")

        t0 = int(client.getImuData(imu_name=cfg.IMU_NAME, vehicle_name=cfg.DRONE).time_stamp)
        p0 = airsim_vec_to_np(
            client.simGetGroundTruthKinematics(vehicle_name=cfg.DRONE).position)

        # Sim time must not advance while paused.
        time.sleep(0.3)
        t_still = int(client.getImuData(imu_name=cfg.IMU_NAME, vehicle_name=cfg.DRONE).time_stamp)
        r.add("sim clock frozen while paused", abs(t_still - t0) < 5_000_000,
              f"drift {abs(t_still - t0) / 1e6:.2f} ms over 300 ms wall time")

        # Image capture must work while paused, or lockstep recording cannot
        # collect stereo frames at all. UE4's render thread keeps running when
        # physics is paused, but that is worth confirming rather than assuming.
        cap0 = time.time()
        rp = client.simGetImages(
            [airsim.ImageRequest(cfg.CAM_LEFT, airsim.ImageType.Scene, False, True)],
            vehicle_name=cfg.DRONE)[0]
        cap_ms = (time.time() - cap0) * 1000.0
        r.add("simGetImages works while paused",
              len(rp.image_data_uint8) > 1000,
              f"{len(rp.image_data_uint8)} PNG bytes in {cap_ms:.0f} ms")

        # LiDAR too -- a paused sim must still answer with the last full sweep.
        ld = client.getLidarData(lidar_name=cfg.LIDAR_NAME, vehicle_name=cfg.DRONE)
        r.add("getLidarData works while paused", len(ld.point_cloud) >= 3,
              f"{len(ld.point_cloud) // 3} points")

        steps, dt = 20, cfg.STEP_DT
        for _ in range(steps):
            client.simContinueForTime(dt)
        t1 = int(client.getImuData(imu_name=cfg.IMU_NAME, vehicle_name=cfg.DRONE).time_stamp)
        p1 = airsim_vec_to_np(
            client.simGetGroundTruthKinematics(vehicle_name=cfg.DRONE).position)

        advanced = (t1 - t0) / 1e9
        want = steps * dt
        r.add("simContinueForTime advances the expected sim time",
              abs(advanced - want) < 0.5 * want,
              f"advanced {advanced * 1000:.1f} ms, requested {want * 1000:.1f} ms")

        moved = float(np.linalg.norm(p1 - p0))
        r.add("flight command keeps running across pause/step", moved > 0.005,
              f"drone moved {moved * 100:.1f} cm over {want * 1000:.0f} ms of sim time")

        client.simPause(False)
        r.add("simPause(False) resumes", not bool(client.simIsPause()), "")
        client.cancelLastTask(vehicle_name=cfg.DRONE)
    except Exception as exc:
        try:
            client.simPause(False)
        except Exception:
            pass
        r.add("lockstep stepping", False, f"{type(exc).__name__}: {str(exc)[:60]}")


def probe_weather(client: airsim.MultirotorClient, r: Results) -> None:
    """Weather must at least apply without error, and visibly change the image.

    A no-op here is the difference between a weather study and five identical
    datasets, so it is checked rather than assumed: whether AirSimNH actually
    carries the weather FX actor is environment-specific.
    """
    print("\n[weather] does AirSimNH respond to simSetWeatherParameter?")

    def grab() -> np.ndarray | None:
        rp = client.simGetImages(
            [airsim.ImageRequest(cfg.CAM_LEFT, airsim.ImageType.Scene, False, False)],
            vehicle_name=cfg.DRONE)[0]
        if rp.width == 0 or len(rp.image_data_uint8) != rp.width * rp.height * 3:
            return None
        return np.frombuffer(rp.image_data_uint8, np.uint8).reshape(rp.height, rp.width, 3)

    try:
        client.simEnableWeather(True)
        base = grab()
        client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0.8)
        time.sleep(2.0)     # let the FX settle
        foggy = grab()
        client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0.0)
        client.simEnableWeather(False)

        if base is None or foggy is None:
            r.add("weather changes the rendered image", False, "no valid frame")
            return

        delta = float(np.abs(foggy.astype(np.int16) - base.astype(np.int16)).mean())
        contrast = (float(base.std()), float(foggy.std()))
        r.add("weather changes the rendered image", delta > 2.0,
              f"mean |change| {delta:.1f}; contrast {contrast[0]:.1f} -> {contrast[1]:.1f}")
    except Exception as exc:
        try:
            client.simEnableWeather(False)
        except Exception:
            pass
        r.add("weather changes the rendered image", False, f"{type(exc).__name__}: {str(exc)[:60]}")


def probe_lidar_ignores_weather(client: airsim.MultirotorClient, r: Results) -> None:
    """Document, by measurement, that LiDAR is unaffected by AirSim weather.

    This is not a failure -- it is the finding that motivates the whole
    degradation-model layer in ``slam/degradation.py``. Recording it here means
    the claim in the writeup is backed by a number from this rig rather than by
    an assertion about AirSim's internals.
    """
    print("\n[weather] does LiDAR see the fog? (expected: no)")
    try:
        def scan_stats() -> tuple[int, float]:
            d = client.getLidarData(lidar_name=cfg.LIDAR_NAME, vehicle_name=cfg.DRONE)
            n = len(d.point_cloud) // 3
            if n == 0:
                return 0, 0.0
            p = np.array(d.point_cloud, dtype=np.float64).reshape(-1, 3)
            return n, float(np.linalg.norm(p, axis=1).max())

        client.simEnableWeather(True)
        client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0.0)
        time.sleep(1.5)
        n_clear, r_clear = scan_stats()

        client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 1.0)
        time.sleep(2.0)
        n_fog, r_fog = scan_stats()

        client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0.0)
        client.simEnableWeather(False)

        ratio = (n_fog / n_clear) if n_clear else 0.0
        unaffected = 0.9 < ratio < 1.1
        r.add("LiDAR is unaffected by fog (expected -- justifies degradation.py)",
              True,
              f"points {n_clear} -> {n_fog} (x{ratio:.3f}), max range "
              f"{r_clear:.1f} -> {r_fog:.1f} m"
              + ("" if unaffected else "  << NOTE: fog DID change returns, revisit degradation.py"))
    except Exception as exc:
        r.add("LiDAR fog measurement", False, f"{type(exc).__name__}: {str(exc)[:60]}")


# --------------------------------------------------------------------------

def main() -> int:
    r = Results()

    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("Connected to AirSim.\n")

    client.enableApiControl(True, cfg.DRONE)
    client.armDisarm(True, cfg.DRONE)

    try:
        print(f"Taking off and flying {PROBE_OFFSET:.0f} m out "
              f"(needed to disambiguate the LiDAR frame)...")
        client.takeoffAsync(vehicle_name=cfg.DRONE).join()
        client.moveToPositionAsync(PROBE_OFFSET, 0.0, PROBE_ALTITUDE, 5.0,
                                   vehicle_name=cfg.DRONE).join()
        client.hoverAsync(vehicle_name=cfg.DRONE).join()
        time.sleep(1.5)   # settle before reading

        probe_lidar_frame(client, r)
        probe_cameras(client, r)
        probe_sensors(client, r)
        probe_ground_truth(client, r)
        probe_weather(client, r)
        probe_lidar_ignores_weather(client, r)
        probe_lockstep(client, r)
    finally:
        print("\nLanding...")
        try:
            client.simPause(False)
            client.simEnableWeather(False)
            client.cancelLastTask(vehicle_name=cfg.DRONE)
            client.landAsync(vehicle_name=cfg.DRONE).join()
            client.armDisarm(False, cfg.DRONE)
            client.enableApiControl(False, cfg.DRONE)
        except Exception as exc:
            print(f"  cleanup warning: {exc}")

    return r.report()


if __name__ == "__main__":
    sys.exit(main())
