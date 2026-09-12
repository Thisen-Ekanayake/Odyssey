#!/usr/bin/env python3
"""
Step 1: fly one drone over the whole environment and record what a 3-D
panoptic map needs -- LiDAR sweeps + the nadir camera's Segmentation and
DepthPlanar images, all stamped with ground-truth sensor poses.

Needs the sim booted with the panoptic rig:
    PROFILE=panoptic ./scripts/run_swarm.sh AirSimNH
then:
    ./airsim_venv/bin/python panoptic/capture.py                   # discover bounds, fly, record
    ./airsim_venv/bin/python panoptic/capture.py --list-objects    # audit name -> class mapping
    ./airsim_venv/bin/python panoptic/capture.py --dry-run         # plan + write meta, no flight
    ./airsim_venv/bin/python panoptic/capture.py --bounds -200 200 -200 200 --altitude 30

What happens:
  1. every scene object is classified by name (config.CLASS_KEYWORDS) and
     given a segmentation id (labels.assign_ids); the id->colour palette is
     measured and cached
  2. the env extent is inferred from object positions (coverage.discover_bounds)
  3. the drone takes off, flies to the nearest corner, then a lawnmower that
     passes through all four corners at ALTITUDE, camera looking straight down
  4. capture is lockstep (simPause / simContinueForTime, same as
     slam/recorder.py) so LiDAR, segmentation and depth of one frame are
     from one instant

Output: datasets_panoptic/<timestamp>/
    meta.json           intrinsics, bounds, waypoints, id table, palette
    lidar/<ns>.npy      float32 (N,3) sensor-local sweep
    seg/<ns>.png        raw segmentation triples (decode with meta palette)
    depth/<ns>.npy      float16 (H,W) DepthPlanar, metres
    scene/<ns>.png      RGB preview (optional, --scene)
    index.json          per frame: exact ns stamp + camera / LiDAR 4x4 poses (what fuse.py reads)
    lidar_poses.txt     TUM, LiDAR sensor pose per frame (world NED)
    cam_poses.txt       TUM, nadir camera body pose per frame (world NED)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import airsim  # noqa: E402

import config as cfg  # noqa: E402
from camera import Intrinsics  # noqa: E402
from coverage import discover_bounds, lawnmower, path_length  # noqa: E402
from labels import _rgb, apply_ids, assign_ids, calibrate_palette, classify  # noqa: E402
from slam.geometry import airsim_pose_to_matrix, pose_to_matrix, write_tum  # noqa: E402

FINISH_RADIUS = 8.0
DURATION_MARGIN = 2.5


def _quat_xyzw(q):
    return np.array([q.x_val, q.y_val, q.z_val, q.w_val], dtype=np.float64)


class Capture:
    def __init__(self, client, out: Path, intr: Intrinsics, save_scene: bool):
        self.client, self.out, self.intr, self.save_scene = client, out, intr, save_scene
        for d in ("lidar", "seg", "depth") + (("scene",) if save_scene else ()):
            (out / d).mkdir(parents=True, exist_ok=True)
        self.lidar_t, self.lidar_T, self.cam_t, self.cam_T = [], [], [], []
        self.frames = 0
        self.dropped = 0
        self._last_lidar = 0
        self.index: list[dict] = []      # exact ns stamps + poses; TUM's float seconds lose ns

    def tick(self) -> np.ndarray | None:
        """Grab one frame of everything. Returns the drone position or None."""
        reqs = [airsim.ImageRequest(cfg.CAM, airsim.ImageType.Segmentation, False, False),
                airsim.ImageRequest(cfg.CAM, airsim.ImageType.DepthPlanar, True, False)]
        if self.save_scene:
            reqs.append(airsim.ImageRequest(cfg.CAM, airsim.ImageType.Scene, False, True))
        resp = self.client.simGetImages(reqs, vehicle_name=cfg.DRONE)
        d = self.client.getLidarData(lidar_name=cfg.LIDAR, vehicle_name=cfg.DRONE)

        if len(resp) < 2 or not resp[0].image_data_uint8 or not resp[1].image_data_float:
            self.dropped += 1
            return None
        ns = int(resp[0].time_stamp)
        seg = _rgb(resp[0])
        depth = np.array(resp[1].image_data_float, dtype=np.float32).reshape(
            resp[1].height, resp[1].width)
        if seg.shape[:2] != (self.intr.height, self.intr.width):
            raise RuntimeError(f"camera returned {seg.shape[1]}x{seg.shape[0]}, config says "
                               f"{self.intr.width}x{self.intr.height}: settings.panoptic.json "
                               f"and config.py disagree")
        cv2.imwrite(str(self.out / "seg" / f"{ns}.png"), seg)
        np.save(self.out / "depth" / f"{ns}.npy", depth.astype(np.float16))
        if self.save_scene and len(resp) > 2 and resp[2].image_data_uint8:
            (self.out / "scene" / f"{ns}.png").write_bytes(bytes(resp[2].image_data_uint8))
        self.cam_t.append(ns / 1e9)
        self.cam_T.append(pose_to_matrix(
            [resp[0].camera_position.x_val, resp[0].camera_position.y_val,
             resp[0].camera_position.z_val], _quat_xyzw(resp[0].camera_orientation)))

        entry = {"ns": ns, "T_cam": self.cam_T[-1].reshape(-1).tolist(), "T_lidar": None}
        if len(d.point_cloud) >= 3 and int(d.time_stamp) != self._last_lidar:
            self._last_lidar = int(d.time_stamp)
            pts = np.array(d.point_cloud, dtype=np.float32).reshape(-1, 3)
            np.save(self.out / "lidar" / f"{ns}.npy", pts)      # keyed by the FRAME stamp
            T_l = airsim_pose_to_matrix(d.pose)
            self.lidar_t.append(ns / 1e9)
            self.lidar_T.append(T_l)
            entry["T_lidar"] = T_l.reshape(-1).tolist()
        self.index.append(entry)
        self.frames += 1
        return self.cam_T[-1][:3, 3]

    def finish(self):
        (self.out / "index.json").write_text(json.dumps(self.index))
        write_tum(self.out / "lidar_poses.txt", self.lidar_t, self.lidar_T,
                  header="LiDAR sensor pose, world NED (LidarData.pose), keyed by frame stamp")
        write_tum(self.out / "cam_poses.txt", self.cam_t, self.cam_T,
                  header="nadir camera BODY pose (+x optical axis), world NED")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None, help="dataset dir (default: datasets_panoptic/<ts>)")
    ap.add_argument("--altitude", type=float, default=cfg.ALTITUDE)
    ap.add_argument("--speed", type=float, default=cfg.SPEED)
    ap.add_argument("--bounds", type=float, nargs=4, metavar=("XMIN", "XMAX", "YMIN", "YMAX"))
    ap.add_argument("--list-objects", action="store_true", help="print name -> class and exit")
    ap.add_argument("--dry-run", action="store_true", help="plan + write meta.json, no flight")
    ap.add_argument("--recalibrate", action="store_true", help="ignore palette_cache.json")
    ap.add_argument("--scene", action="store_true", help="also save RGB previews")
    ap.add_argument("--max-minutes", type=float, default=40.0)
    args = ap.parse_args()

    client = airsim.MultirotorClient()
    client.confirmConnection()
    intr = Intrinsics(cfg.CAM_W, cfg.CAM_H, cfg.CAM_FOV_DEG)

    names = client.simListSceneObjects()
    print(f"{len(names)} scene objects")
    if args.list_objects:
        for n in sorted(names):
            print(f"  {classify(n) or '-':10s} {n}")
        return
    by_name, table = assign_ids(names)
    per_class = {}
    for ent in table.values():
        per_class[ent["class"]] = per_class.get(ent["class"], 0) + len(ent["objects"])
    print("  objects per class:", ", ".join(f"{k}={v}" for k, v in sorted(per_class.items())))
    print(f"  {len(table)} segmentation ids in use")

    apply_ids(client, by_name)
    palette = calibrate_palette(client, sorted(table), force=args.recalibrate)
    apply_ids(client, by_name, verbose=False)      # calibration painted everything; restore

    bounds = tuple(args.bounds) if args.bounds else discover_bounds(client, names)
    start = client.simGetVehiclePose(vehicle_name=cfg.DRONE).position
    z_fly = start.z_val - args.altitude
    swath = intr.swath_at(args.altitude)
    wps = lawnmower(bounds, (start.x_val, start.y_val), z_fly, swath)
    length = path_length(wps)
    expected = length / args.speed
    print(f"  bounds x[{bounds[0]:.0f},{bounds[1]:.0f}] y[{bounds[2]:.0f},{bounds[3]:.0f}]  "
          f"swath {swath:.0f} m  {len(wps)} waypoints, {length/1000:.2f} km, "
          f"~{expected/60:.1f} min at {args.speed} m/s")

    out = args.out or cfg.DATASET_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps({
        "camera": {"name": cfg.CAM, "width": intr.width, "height": intr.height,
                   "fov_deg": intr.fov_deg, "fx": intr.fx, "cx": intr.cx, "cy": intr.cy},
        "lidar": cfg.LIDAR, "altitude": args.altitude, "speed": args.speed,
        "bounds": bounds, "waypoints": wps, "capture_dt": cfg.CAPTURE_DT,
        "ids": {str(k): v for k, v in table.items()},
        "palette": {str(k): list(v) for k, v in palette.items()},
    }, indent=1))
    print(f"  meta -> {out / 'meta.json'}")
    if args.dry_run:
        return

    cap = Capture(client, out, intr, args.scene)
    client.enableApiControl(True, cfg.DRONE)
    client.armDisarm(True, cfg.DRONE)
    print("take-off ...")
    client.takeoffAsync(vehicle_name=cfg.DRONE).join()
    client.moveToZAsync(z_fly, 3.0, vehicle_name=cfg.DRONE).join()
    print(f"to first corner ({wps[0][0]:.0f}, {wps[0][1]:.0f}) ...")
    client.moveToPositionAsync(wps[0][0], wps[0][1], z_fly, args.speed,
                               drivetrain=airsim.DrivetrainType.ForwardOnly,
                               yaw_mode=airsim.YawMode(False, 0), vehicle_name=cfg.DRONE).join()
    client.hoverAsync(vehicle_name=cfg.DRONE).join()
    time.sleep(1.0)

    client.moveOnPathAsync([airsim.Vector3r(*w) for w in wps], args.speed,
                           timeout_sec=expected * DURATION_MARGIN,
                           drivetrain=airsim.DrivetrainType.ForwardOnly,
                           yaw_mode=airsim.YawMode(False, 0),
                           lookahead=-1, adaptive_lookahead=1, vehicle_name=cfg.DRONE)
    goal = np.array(wps[-1][:2])
    max_steps = int(min(expected * DURATION_MARGIN, args.max_minutes * 60) / cfg.CAPTURE_DT)
    t0 = time.time()
    print("recording (lockstep) ...")
    client.simPause(True)
    try:
        for step in range(max_steps):
            pos = cap.tick()
            elapsed = step * cfg.CAPTURE_DT
            if step % 25 == 0:
                print(f"\r  {cap.frames} frames  sim {elapsed:6.0f}/{expected:.0f} s  "
                      f"wall {time.time()-t0:5.0f} s  dropped {cap.dropped}", end="", flush=True)
            if pos is not None and elapsed > 0.5 * expected and \
                    np.linalg.norm(pos[:2] - goal) < FINISH_RADIUS:
                print("\n  route complete")
                break
            client.simContinueForTime(cfg.CAPTURE_DT)
        else:
            print("\n  WARNING: step cap hit before the last waypoint; coverage may be partial")
    finally:
        client.simPause(False)
    cap.finish()

    print("landing ...")
    client.hoverAsync(vehicle_name=cfg.DRONE).join()
    client.landAsync(vehicle_name=cfg.DRONE).join()
    client.armDisarm(False, cfg.DRONE)
    client.enableApiControl(False, cfg.DRONE)
    print(f"done: {cap.frames} frames, {len(cap.lidar_t)} LiDAR sweeps -> {out}")
    print(f"next: ./airsim_venv/bin/python panoptic/fuse.py --dataset {out}")


if __name__ == "__main__":
    main()
