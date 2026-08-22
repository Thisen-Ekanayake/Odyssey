#!/usr/bin/env python3
"""Converts a recorded ``datasets/<condition>/`` run into a rosbag2.

This is what makes the ROS SLAM stack comparable to the existing one rather than
merely adjacent. ``tools/run_benchmark.py`` gets its reproducibility from
replaying *recorded bytes* -- identical input for every method and every re-run.
A bag built from the same recording gives rtabmap exactly that input, so an ATE
from rtabmap and an ATE from ``slam/lidar_slam.py`` are measuring the same flight
and can be put in one table.

Topic names and frame ids match ``bridge_node`` exactly, so a SLAM launch file
does not care whether it is fed by a live simulator or by a bag.

    ros2 run airsim_swarm_bridge dataset_to_rosbag datasets/clear --out bags/clear
    ros2 run airsim_swarm_bridge dataset_to_rosbag datasets/fog_heavy --degraded --out bags/fog_heavy_degraded

**On weather:** AirSim's weather is a rendering effect only. The camera streams
in a ``rain_*``/``fog_*`` recording really are degraded, but the LiDAR is a
raycast against collision geometry and rain/fog particles have none -- the scans
are bit-identical to ``clear``. ``--degraded`` applies ``slam/degradation.py``,
the same hand-built extinction model ``run_benchmark.py`` uses, so the LiDAR half
is *modelled*. Never report a weather comparison without saying which half is
which; the bag's metadata records the choice for you.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    try:
        from .repo import repo_root
        return repo_root()
    except ImportError:                       # run as a plain script
        here = Path(__file__).resolve()
        for parent in here.parents:
            if (parent / "settings.json").exists() and (parent / "slam").is_dir():
                return parent
        raise


class BagWriter:
    """Thin wrapper over ``rosbag2_py`` that registers topics on first use."""

    def __init__(self, out: Path, storage_id: str = "sqlite3"):
        import rosbag2_py

        self._rosbag2_py = rosbag2_py
        self.writer = rosbag2_py.SequentialWriter()
        self.writer.open(
            rosbag2_py.StorageOptions(uri=str(out), storage_id=storage_id),
            rosbag2_py.ConverterOptions(input_serialization_format="cdr",
                                        output_serialization_format="cdr"),
        )
        self._known: set[str] = set()

    def write(self, topic: str, msg, stamp_ns: int) -> None:
        from rclpy.serialization import serialize_message

        if topic not in self._known:
            type_name = (f"{type(msg).__module__.split('.')[0]}/msg/"
                         f"{type(msg).__name__}")
            # Jazzy's TopicMetadata takes a mandatory positional `id`; older
            # distros did not have it. Numbering them in registration order is
            # what `ros2 bag record` itself does.
            self.writer.create_topic(self._rosbag2_py.TopicMetadata(
                id=len(self._known), name=topic, type=type_name,
                serialization_format="cdr",
                offered_qos_profiles=self._qos_for(topic)))
            self._known.add(topic)
        self.writer.write(topic, serialize_message(msg), int(stamp_ns))

    def _qos_for(self, topic: str) -> list:
        """QoS to record against a topic, so `ros2 bag play` re-offers it faithfully.

        This matters for exactly one topic and it is easy to miss: /tf_static is
        published once, at the start, and every tf2 listener subscribes to it with
        TRANSIENT_LOCAL durability so that joining late still gets the frames. If
        the bag records it with the default VOLATILE profile, playback offers
        VOLATILE, no listener ever receives it, and the symptom is a SLAM node
        insisting `drone1/base_link ... does not exist` while /tf flows normally.
        """
        from rosbag2_py._storage import QoS

        if topic != "/tf_static":
            return []
        return [QoS(1).transient_local().reliable().keep_last(1)]


def _stamp(ns: int):
    from builtin_interfaces.msg import Time
    ns = int(ns)
    return Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)


def _read_tum(path: Path):
    """``(times, positions, quaternions)`` from a TUM file; empty if absent.

    Deliberately not ``slam.geometry.read_tum``: that returns 4x4 matrices via
    scipy, and everything here needs the raw quaternion anyway.
    """
    if not path.exists():
        return np.empty(0), np.empty((0, 3)), np.empty((0, 4))
    raw = np.loadtxt(path, comments="#", ndmin=2)
    if raw.size == 0:
        return np.empty(0), np.empty((0, 3)), np.empty((0, 4))
    return raw[:, 0], raw[:, 1:4], raw[:, 4:8]


def _nearest(times: np.ndarray, t: float) -> int | None:
    if not len(times):
        return None
    return int(np.abs(times - t).argmin())


def convert(dataset: Path, out: Path, namespace: str, degraded: bool,
            storage_id: str, max_frames: int | None, seed: int,
            skip_stereo: bool = False, gt_decimate: int = 1) -> int:
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Image, Imu, PointCloud2
    from tf2_msgs.msg import TFMessage

    sys.path.insert(0, str(_repo_root()))
    from slam import config as cfg                       # noqa: PLC0415

    from . import frames                                 # noqa: PLC0415
    from .bridge_node import AirSimBridge                # noqa: PLC0415

    f = frames.FrameNames(namespace)

    # -- metadata ---------------------------------------------------------
    meta = {}
    for name in ("sensor.yaml", "sensor.json"):
        p = dataset / name
        if p.exists():
            import yaml
            meta = (yaml.safe_load(p.read_text()) if name.endswith("yaml")
                    else json.loads(p.read_text())) or {}
            break
    condition = str(meta.get("weather", {}).get("condition", dataset.name))

    # -- degradation model ------------------------------------------------
    degrade = None
    if degraded:
        from slam import degradation                     # noqa: PLC0415
        cond = cfg.WEATHER_CONDITIONS.get(condition)
        if cond is None:
            print(f"ERROR: no weather condition named {condition!r} in slam/config.py; "
                  f"have {sorted(cfg.WEATHER_CONDITIONS)}", file=sys.stderr)
            return 1
        # Seeded: the degradation draws per-point detections at random, and an
        # unseeded run would make two 'identical' bags differ.
        model = degradation.for_condition(cond, seed=seed)
        degrade = model
        print(f"  degradation: {model.label}  alpha={model.alpha:.4f} /m  "
              f"effective range {model.effective_range():.1f} m  (seed={seed})")

    # -- inputs -----------------------------------------------------------
    imu_raw = np.loadtxt(dataset / "imu.txt", comments="#", ndmin=2)
    gt_t, gt_p, gt_q = _read_tum(dataset / "groundtruth.txt")
    lidar_files = sorted((dataset / "lidar").glob("*.npy")) if (dataset / "lidar").is_dir() else []
    cam0 = []
    if not skip_stereo and (dataset / "cam0").is_dir():
        cam0 = sorted((dataset / "cam0").glob("*.png"))
    cam1_dir = dataset / "cam1"
    if max_frames:
        lidar_files = lidar_files[:max_frames]
        cam0 = cam0[:max_frames]

    print(f"  imu   {len(imu_raw):6d} samples")
    print(f"  lidar {len(lidar_files):6d} scans")
    print(f"  cam   {len(cam0):6d} stereo pairs")
    print(f"  gt    {len(gt_t):6d} poses")
    if not len(imu_raw) and not lidar_files:
        print("ERROR: nothing to convert", file=sys.stderr)
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"ERROR: {out} already exists; remove it or pass a different --out",
              file=sys.stderr)
        return 1
    bag = BagWriter(out, storage_id)

    # -- static TF (sensor mounts), written once at the very start ---------
    # Earliest stamp in the recording; /tf_static is written there so it is
    # already in place before the first sensor message replays.
    candidates = []
    if len(imu_raw):
        candidates.append(int(imu_raw[0, 0] * 1e9))
    if lidar_files:
        candidates.append(int(lidar_files[0].stem))
    if cam0:
        candidates.append(int(cam0[0].stem))
    t0_ns = min(candidates) if candidates else 0
    tf_static = TFMessage()
    st = _stamp(t0_ns)
    lx, ly, lz = frames.frd_to_flu(*cfg.T_BODY_LIDAR[:3, 3])
    tf_static.transforms.append(AirSimBridge._tf(f.MAP, f.odom, (0.0, 0.0, 0.0),
                                                 (0.0, 0.0, 0.0, 1.0), st))
    tf_static.transforms.append(AirSimBridge._tf(f.base_link, f.lidar, (lx, ly, lz),
                                                 (0.0, 0.0, 0.0, 1.0), st))
    tf_static.transforms.append(AirSimBridge._tf(f.base_link, f.imu, (0, 0, 0),
                                                 (0.0, 0.0, 0.0, 1.0), st))
    for T, link, optical in ((cfg.T_BODY_CAM_LEFT, f.cam_left, f.cam_left_optical),
                             (cfg.T_BODY_CAM_RIGHT, f.cam_right, f.cam_right_optical)):
        p = frames.frd_to_flu(*T[:3, 3])
        tf_static.transforms.append(AirSimBridge._tf(f.base_link, link, p,
                                                     (0.0, 0.0, 0.0, 1.0), st))
        tf_static.transforms.append(AirSimBridge._tf(link, optical, (0, 0, 0),
                                                     (-0.5, 0.5, -0.5, 0.5), st))
    bag.write("/tf_static", tf_static, t0_ns)

    ns = f"/{f.ns}"
    written = {"imu": 0, "lidar": 0, "stereo": 0, "gt": 0}

    # -- IMU ---------------------------------------------------------------
    cov_o, cov_g, cov_a = _imu_cov(dataset, cfg, meta)
    for row in imu_raw:
        t = float(row[0])
        m = Imu()
        m.header.stamp = _stamp(int(t * 1e9))
        m.header.frame_id = f.imu
        wx, wy, wz = frames.frd_to_flu(row[1], row[2], row[3])
        ax, ay, az = frames.frd_to_flu(row[4], row[5], row[6])
        m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = wx, wy, wz
        m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = ax, ay, az
        i = _nearest(gt_t, t)
        if i is not None:
            q = frames.ned_quat_to_enu(*gt_q[i])
            (m.orientation.x, m.orientation.y,
             m.orientation.z, m.orientation.w) = map(float, q)
        m.orientation_covariance = cov_o
        m.angular_velocity_covariance = cov_g
        m.linear_acceleration_covariance = cov_a
        bag.write(f"{ns}/imu", m, int(t * 1e9))
        written["imu"] += 1

    # -- ground truth: odom + its own TF branch ----------------------------
    # Ground truth goes to <ns>/base_link_gt, never to base_link: icp_odometry
    # publishes odom -> base_link itself, and two writers on one TF edge is a
    # corrupt tree.
    for i, (t, p, q) in enumerate(zip(gt_t, gt_p, gt_q)):
        if gt_decimate > 1 and i % gt_decimate:
            continue
        stamp = _stamp(int(t * 1e9))
        x, y, z = frames.ned_point_to_enu(*p)
        qe = frames.ned_quat_to_enu(*q)

        o = Odometry()
        o.header.stamp = stamp
        o.header.frame_id = f.MAP
        o.child_frame_id = f.base_link_gt
        o.pose.pose.position.x, o.pose.pose.position.y, o.pose.pose.position.z = x, y, z
        (o.pose.pose.orientation.x, o.pose.pose.orientation.y,
         o.pose.pose.orientation.z, o.pose.pose.orientation.w) = map(float, qe)
        bag.write(f"{ns}/odom_gt", o, int(t * 1e9))

        tfm = TFMessage()
        tfm.transforms.append(AirSimBridge._tf(f.MAP, f.base_link_gt, (x, y, z), qe, stamp))
        bag.write("/tf", tfm, int(t * 1e9))
        written["gt"] += 1

    # -- LiDAR --------------------------------------------------------------
    lidar_t, _, _ = _read_tum(dataset / "lidar_poses.txt")
    for path in lidar_files:
        ts_ns = int(path.stem)
        pts = np.load(path).astype(np.float32).reshape(-1, 3)
        if degrade is not None:
            # Applied in the SENSOR frame, before any axis flip -- the model is
            # written in terms of range from the sensor origin, which the flip
            # preserves but which is clearer to do first.
            pts = degrade(pts).astype(np.float32)
        if not len(pts):
            continue
        pts = frames.frd_points_to_flu(pts)
        bag.write(f"{ns}/lidar/points",
                  AirSimBridge._cloud_msg(pts, f.lidar, _stamp(ts_ns)), ts_ns)
        written["lidar"] += 1

    # -- stereo -------------------------------------------------------------
    if cam0:
        import cv2

        from .frames import camera_info_msg
        for path in cam0:
            ts_ns = int(path.stem)
            right = cam1_dir / path.name
            left_img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            right_img = cv2.imread(str(right), cv2.IMREAD_COLOR) if right.exists() else None
            if left_img is None:
                continue
            stamp = _stamp(ts_ns)
            h, w = left_img.shape[:2]
            sx, sy = w / cfg.IMAGE_WIDTH, h / cfg.IMAGE_HEIGHT
            for img, side, optical, is_right in (
                    (left_img, "left", f.cam_left_optical, False),
                    (right_img, "right", f.cam_right_optical, True)):
                if img is None:
                    continue
                m = Image()
                m.header.stamp = stamp
                m.header.frame_id = optical
                m.height, m.width = int(h), int(w)
                m.encoding = "bgr8"
                m.is_bigendian = 0
                m.step = 3 * int(w)
                m.data = np.ascontiguousarray(img).tobytes()
                bag.write(f"{ns}/stereo/{side}/image_raw", m, ts_ns)
                bag.write(f"{ns}/stereo/{side}/camera_info",
                          camera_info_msg(w, h, cfg.FX * sx, cfg.FY * sy,
                                          cfg.CX * sx, cfg.CY * sy, optical, stamp,
                                          baseline=cfg.STEREO_BASELINE,
                                          is_right=is_right), ts_ns)
            written["stereo"] += 1

    del bag        # close the writer before touching the directory it wrote

    # A sidecar note, so a bag can never be separated from what was done to it.
    (out / "airsim_swarm.json").write_text(json.dumps({
        "source_dataset": str(dataset),
        "condition": condition,
        "namespace": f.ns,
        "lidar_degraded": bool(degraded),
        "stereo_included": not skip_stereo,
        "gt_decimate": gt_decimate,
        "degradation_seed": seed if degraded else None,
        "messages": written,
        "note": ("AirSim weather is a RENDERING effect only: the camera streams are "
                 "genuinely degraded by the simulator, the LiDAR is not. "
                 + ("LiDAR degradation here is MODELLED by slam/degradation.py."
                    if degraded else
                    "LiDAR scans in this bag are RAW -- identical to the clear condition.")),
    }, indent=2) + "\n")

    print(f"\nwrote {out}")
    for k, v in written.items():
        print(f"  {k:7s} {v}")
    return 0


def _imu_cov(dataset: Path, cfg, meta: dict):
    """Same ARW/VRW -> variance derivation the live bridge uses (see bridge_node)."""
    imu_meta = (meta.get("imu") or {})
    arw = float(imu_meta.get("angular_random_walk", 0.30))
    vrw = float(imu_meta.get("velocity_random_walk", 0.24))
    f = float(getattr(cfg, "IMU_RATE_HZ", 100.0))
    g = (math.radians(arw) / 60.0) ** 2 * f
    a = (vrw / 60.0) ** 2 * f
    diag = lambda v: [v, 0.0, 0.0, 0.0, v, 0.0, 0.0, 0.0, v]      # noqa: E731
    return diag(1e-4), diag(g), diag(a)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Convert a recorded AirSim SLAM dataset into a rosbag2.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("dataset", type=Path, help="datasets/<condition>/")
    ap.add_argument("--out", type=Path, default=None, help="output bag directory")
    ap.add_argument("--namespace", default="Drone1",
                    help="vehicle name -> topic namespace (default: Drone1 -> /drone1)")
    ap.add_argument("--degraded", action="store_true",
                    help="apply slam/degradation.py to the LiDAR (see the note above)")
    ap.add_argument("--seed", type=int, default=0,
                    help="degradation RNG seed; keep fixed for comparable runs")
    ap.add_argument("--storage", default="sqlite3", choices=("sqlite3", "mcap"))
    ap.add_argument("--max-frames", type=int, default=None,
                    help="stop after N lidar/stereo frames (quick smoke tests)")
    ap.add_argument("--skip-stereo", action="store_true",
                    help="omit the camera streams. LiDAR SLAM does not use them "
                         "and raw Image messages dominate bag size (~3 GB per "
                         "condition), so drop them unless you need stereo")
    ap.add_argument("--gt-decimate", type=int, default=1, metavar="N",
                    help="keep every Nth ground-truth pose. Ground truth is "
                         "recorded at the 100 Hz IMU rate, far denser than any "
                         "evaluation needs -- evaluate.py interpolates anyway "
                         "(default: 1, keep all)")
    args = ap.parse_args(argv)

    dataset = args.dataset if args.dataset.is_absolute() else _repo_root() / args.dataset
    if not dataset.is_dir():
        print(f"ERROR: no dataset at {dataset}", file=sys.stderr)
        return 1

    out = args.out or Path("bags") / (dataset.name + ("_degraded" if args.degraded else ""))
    if not out.is_absolute():
        out = _repo_root() / out

    print(f"converting {dataset} -> {out}")
    return convert(dataset, out, args.namespace, args.degraded,
                   args.storage, args.max_frames, args.seed,
                   args.skip_stereo, max(1, args.gt_decimate))


if __name__ == "__main__":
    raise SystemExit(main())
