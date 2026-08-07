#!/usr/bin/env python3
"""
Generate a synthetic dataset in the recorder's on-disk format, without AirSim.

Purpose: make the entire offline half of this project verifiable before
committing to a simulator session. Recording the real benchmark needs Docker,
the GPU, a working Vulkan stack and roughly an hour of flying; this produces
the same file layout in a couple of minutes so ``tools/verify_dataset.py`` and
``tools/run_benchmark.py`` can be exercised end to end.

It is a *test fixture*, not a substitute for the real data. The world is a
blocky neighbourhood raycast with Open3D, the camera texture is procedural, and
the IMU is derived analytically from the trajectory. Numbers from it say the
pipeline works; they say nothing about AirSimNH.

One property is deliberately faithful: fog and rain attenuate the rendered
camera image (as UE4's post-process does) but leave the LiDAR untouched (as
AirSim's raycast does). That way the synthetic data reproduces the exact
asymmetry ``slam/degradation.py`` exists to address.

Usage:
    ./airsim_venv/bin/python tools/synthetic_dataset.py --out datasets_synth
    ./airsim_venv/bin/python tools/synthetic_dataset.py --out datasets_synth \\
        --conditions clear fog_heavy --frames 400
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slam import config as cfg  # noqa: E402
from slam import degradation as deg  # noqa: E402
from slam.geometry import write_tum  # noqa: E402


# --------------------------------------------------------------------------
# world
# --------------------------------------------------------------------------

def build_scene(seed: int = 0, extent=(-40, 170, -40, 130)):
    """Ground plane plus houses lining the flight corridor.

    Structure along the route matters: over a bare ground plane, in-plane
    translation is unobservable and scan matching simply refuses to move.
    """
    rng = np.random.default_rng(seed)
    x0, x1, y0, y1 = extent

    ground = o3d.geometry.TriangleMesh.create_box(x1 - x0, y1 - y0, 0.5)
    ground.translate((x0, y0, 0.0))
    meshes = [ground]

    def box(cx, cy, w, d, h, yaw=0.0):
        b = o3d.geometry.TriangleMesh.create_box(w, d, h)
        b.translate((-w / 2, -d / 2, -h))          # NED: -z is up
        if yaw:
            b.rotate(Rotation.from_euler("z", yaw).as_matrix(), center=(0, 0, 0))
        b.translate((cx, cy, 0))
        return b

    def row(alongs, make_xy):
        for a in alongs:
            for off in (-11.0, 11.0, -24.0, 24.0):
                for cx, cy in make_xy(a, off):
                    if x0 < cx < x1 and y0 < cy < y1:
                        meshes.append(box(cx + rng.uniform(-2, 2), cy + rng.uniform(-2, 2),
                                          rng.uniform(7, 11), rng.uniform(7, 11),
                                          rng.uniform(4, 12), rng.uniform(-0.3, 0.3)))

    row(np.arange(-20, 145, 14.0), lambda a, o: ((a, o), (a, cfg.ROUTE_LENGTH_Y - o)))
    row(np.arange(-6, 92, 14.0), lambda a, o: ((o, a), (cfg.ROUTE_LENGTH_X - o, a)))

    for _ in range(220):     # street furniture, breaks up the ground plane
        meshes.append(box(rng.uniform(x0, x1), rng.uniform(y0, y1),
                          rng.uniform(0.6, 1.6), rng.uniform(0.6, 1.6),
                          rng.uniform(2.5, 6.0)))

    combined = meshes[0]
    for m in meshes[1:]:
        combined += m
    combined.compute_vertex_normals()

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(combined))
    return scene


def full_route_frames() -> int:
    """Sensor frames the real route takes: length / speed * rate, at 10 Hz."""
    wps = np.array(cfg.route_waypoints(), dtype=float)
    length = float(np.linalg.norm(np.diff(wps, axis=0), axis=1).sum())
    rate = cfg.IMU_RATE_HZ / cfg.SENSOR_DECIMATION
    return int(round(length / cfg.ROUTE_SPEED * rate))


def trajectory(n: int) -> np.ndarray:
    """Ground-truth body poses along the configured route, starting from rest."""
    wps = np.array(cfg.route_waypoints(), dtype=float)
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(wps, axis=0), axis=1))])

    ramp = np.clip(np.linspace(0, 1, n) / 0.03, 0, 1)   # matches the recorder's hover start
    prof = np.cumsum(ramp)
    s = prof / prof[-1] * cum[-1]
    xyz = np.stack([np.interp(s, cum, wps[:, i]) for i in range(3)], axis=1)
    xyz[:, 2] += 0.4 * np.sin(np.linspace(0, 9 * np.pi, n))
    xyz[:, 0] += 0.3 * np.sin(np.linspace(0, 5 * np.pi, n))

    T = np.tile(np.eye(4), (n, 1, 1))
    for k in range(n):
        fwd = xyz[min(k + 1, n - 1)] - xyz[max(k - 1, 0)]
        yaw = np.arctan2(fwd[1], fwd[0]) if np.linalg.norm(fwd[:2]) > 1e-9 else 0.0
        T[k, :3, :3] = Rotation.from_euler("z", yaw).as_matrix()
        T[k, :3, 3] = xyz[k]
    return T


# --------------------------------------------------------------------------
# sensors
# --------------------------------------------------------------------------

def lidar_rays(n_az: int = 900) -> np.ndarray:
    el = np.radians(np.linspace(cfg.LIDAR_VFOV_UPPER, cfg.LIDAR_VFOV_LOWER,
                                cfg.LIDAR_CHANNELS))
    az = np.linspace(-np.pi, np.pi, n_az, endpoint=False)
    A, E = np.meshgrid(az, el, indexing="ij")        # azimuth-major = scan order
    d = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), -np.sin(E)], axis=-1)
    return d.reshape(-1, 3).astype(np.float32)


def cast(scene, T_world_sensor, dirs):
    R = T_world_sensor[:3, :3].astype(np.float32)
    t = T_world_sensor[:3, 3].astype(np.float32)
    rays = np.hstack([np.tile(t, (len(dirs), 1)), dirs @ R.T])
    hit = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))["t_hit"].numpy()
    return hit


def lidar_scan(scene, T, dirs, rng) -> np.ndarray:
    hit = cast(scene, T, dirs)
    ok = np.isfinite(hit) & (hit > 0.5) & (hit < cfg.LIDAR_RANGE)
    r = hit[ok] + rng.normal(0, 0.02, ok.sum()).astype(np.float32)
    return (dirs[ok] * r[:, None]).astype(np.float32)


def pixel_rays():
    u, v = np.meshgrid(np.arange(cfg.IMAGE_WIDTH), np.arange(cfg.IMAGE_HEIGHT))
    d = np.stack([(u - cfg.CX) / cfg.FX, (v - cfg.CY) / cfg.FY, np.ones_like(u, float)], -1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return d.reshape(-1, 3).astype(np.float32)


def _texture(p: np.ndarray) -> np.ndarray:
    """Blocky hash texture keyed on world position.

    ORB needs corners; a smooth gradient gives it nothing. Hashing at ~25 cm
    granularity yields dense, repeatable, view-consistent corners.
    """
    q = np.floor(p * 4.0).astype(np.int64)
    h = (q[:, 0] * 73856093) ^ (q[:, 1] * 19349663) ^ (q[:, 2] * 83492791)
    h = (h ^ (h >> 13)) * 1274126177
    return ((h ^ (h >> 16)) & 0xFF).astype(np.uint8)


def render(scene, T_world_cam_optical, rays, fog_alpha, rng):
    """One camera view. Returns (BGR image, planar depth)."""
    R = T_world_cam_optical[:3, :3].astype(np.float32)
    t = T_world_cam_optical[:3, 3].astype(np.float32)
    wd = rays @ R.T
    org = np.tile(t, (len(rays), 1))
    hit = scene.cast_rays(o3d.core.Tensor(np.hstack([org, wd]),
                                          dtype=o3d.core.Dtype.Float32))["t_hit"].numpy()
    ok = np.isfinite(hit) & (hit > 0.1)

    gray = np.full(len(rays), 200, np.uint8)          # sky
    depth = np.zeros(len(rays), np.float32)
    if ok.any():
        p = org[ok] + wd[ok] * hit[ok][:, None]
        gray[ok] = _texture(p)
        depth[ok] = (hit[ok] * rays[ok, 2]).astype(np.float32)

    g = gray.astype(np.float32)
    if fog_alpha > 0:
        # Beer-Lambert blend toward a bright sky, i.e. the contrast collapse
        # UE4's fog post-process produces. The LiDAR above is untouched.
        d = np.where(ok, hit, cfg.LIDAR_RANGE)
        tr = np.exp(-fog_alpha * d)
        g = g * tr + 220.0 * (1.0 - tr)
    g += rng.normal(0, 2.0, g.shape)

    shape = (cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH)
    img = np.clip(g, 0, 255).astype(np.uint8).reshape(shape)
    return np.repeat(img[:, :, None], 3, axis=2), depth.reshape(shape)


def imu_from_trajectory(T: np.ndarray, dt: float, rng) -> np.ndarray:
    """IMU samples consistent with the pose sequence (body NED, specific force)."""
    pos = T[:, :3, 3]
    acc_w = np.gradient(np.gradient(pos, dt, axis=0), dt, axis=0)
    bias_g = np.array([0.004, -0.003, 0.002])
    bias_a = np.array([0.05, -0.04, 0.03])

    rows = []
    for k in range(len(T)):
        if k < len(T) - 1:
            w = Rotation.from_matrix(T[k, :3, :3].T @ T[k + 1, :3, :3]).as_rotvec() / dt
        else:
            w = np.zeros(3)
        f = T[k, :3, :3].T @ (acc_w[k] - cfg.GRAVITY_NED)
        rows.append((k * dt, *(w + bias_g + rng.normal(0, 2e-4, 3)),
                     *(f + bias_a + rng.normal(0, 2e-3, 3))))
    return np.array(rows)


# --------------------------------------------------------------------------

def generate(out: Path, condition: str, n_frames: int, scene, seed: int = 0,
             stereo: bool = True) -> None:
    cond = cfg.WEATHER_CONDITIONS[condition]
    # Camera attenuation uses the same extinction coefficient the LiDAR model
    # derives, so the two arms of the study are driven from one number.
    fog_alpha = deg.for_condition(cond).alpha

    rng = np.random.default_rng(seed)
    for sub in ("lidar", "cam0", "cam1"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    dec = cfg.SENSOR_DECIMATION
    dt = 1.0 / cfg.IMU_RATE_HZ

    # The trajectory is ALWAYS generated at the real capture density -- the
    # full route at 10 Hz and ROUTE_SPEED -- and then truncated. Scaling the
    # sampling to n_frames instead would silently change the metres-per-frame,
    # and at anything below full density the frame-to-frame motion exceeds
    # GICP's correspondence distance, so scan matching degrades for reasons
    # that have nothing to do with the algorithm under test.
    full_frames = full_route_frames()
    T_full = trajectory(full_frames * dec)
    imu = imu_from_trajectory(T_full, dt, rng)

    n_frames = min(n_frames, full_frames)
    T = T_full[:n_frames * dec]
    imu = imu[:n_frames * dec]

    dirs = lidar_rays()
    prays = pixel_rays() if stereo else None

    gt_t, gt_T = list(imu[:, 0]), list(T)
    lp_t, lp_T, cp_t, cp_T = [], [], [], []

    for f in range(n_frames):
        k = f * dec
        ns = int(round(k * dt * 1e9))

        T_l = T[k] @ cfg.T_BODY_LIDAR
        np.save(out / "lidar" / f"{ns}.npy", lidar_scan(scene, T_l, dirs, rng))
        lp_t.append(ns / 1e9)
        lp_T.append(T_l)

        if stereo:
            TL = T[k] @ cfg.T_BODY_CAM_LEFT
            TR = T[k] @ cfg.T_BODY_CAM_RIGHT
            L, _ = render(scene, TL, prays, fog_alpha, rng)
            R, _ = render(scene, TR, prays, fog_alpha, rng)
            cv2.imwrite(str(out / "cam0" / f"{ns}.png"), L)
            cv2.imwrite(str(out / "cam1" / f"{ns}.png"), R)
            cp_t.append(ns / 1e9)
            cp_T.append(TL)

        if f % 100 == 0:
            print(f"\r  {condition}: {f}/{n_frames}", end="", flush=True)

    np.savetxt(out / "imu.txt", imu, fmt="%.9f",
               header="timestamp wx wy wz ax ay az  (body NED, rad/s, m/s^2)")
    write_tum(out / "groundtruth.txt", gt_t, gt_T, header="SYNTHETIC ground-truth body pose")
    write_tum(out / "lidar_poses.txt", lp_t, lp_T, header="SYNTHETIC lidar sensor pose")
    if cp_T:
        write_tum(out / "cam_poses.txt", cp_t, cp_T, header="SYNTHETIC left-camera optical pose")

    json.dump({
        "synthetic": True,
        "note": "Generated by tools/synthetic_dataset.py -- NOT recorded from AirSim.",
        "weather": {"condition": condition,
                    "airsim_params": dict(cond.airsim_params),
                    "camera_extinction_alpha": fog_alpha},
        "camera": {"width": cfg.IMAGE_WIDTH, "height": cfg.IMAGE_HEIGHT,
                   "fx": cfg.FX, "fy": cfg.FY, "cx": cfg.CX, "cy": cfg.CY,
                   "baseline_m": cfg.STEREO_BASELINE},
        "lidar": {"range_m": cfg.LIDAR_RANGE, "data_frame": "SensorLocalFrame"},
    }, open(out / "sensor.json", "w"), indent=2)
    print(f"\r  {condition}: {n_frames} frames -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=cfg.REPO_ROOT / "datasets_synth")
    ap.add_argument("--conditions", nargs="+", default=cfg.BENCHMARK_ORDER)
    ap.add_argument("--frames", type=int, default=0,
                   help="truncate to this many frames (0 = the whole route). "
                        "Density is always the real 10 Hz / ROUTE_SPEED rate, so "
                        "this shortens the flight rather than coarsening it.")
    ap.add_argument("--no-stereo", action="store_true")
    args = ap.parse_args()

    full = full_route_frames()
    frames = args.frames if args.frames > 0 else full
    per_frame = cfg.ROUTE_SPEED / (cfg.IMU_RATE_HZ / cfg.SENSOR_DECIMATION)
    print(f"full route = {full} frames at {per_frame:.2f} m/frame; "
          f"generating {min(frames, full)}")
    print("building synthetic scene...")
    scene = build_scene()
    for c in args.conditions:
        if c not in cfg.WEATHER_CONDITIONS:
            print(f"unknown condition {c!r}; valid: {cfg.BENCHMARK_ORDER}", file=sys.stderr)
            return 1
        generate(args.out / c, c, frames, scene, stereo=not args.no_stereo)

    print(f"\nDone. Verify with:\n"
          f"  ./airsim_venv/bin/python tools/verify_dataset.py {args.out}/clear\n"
          f"  ./airsim_venv/bin/python tools/run_benchmark.py --datasets {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
