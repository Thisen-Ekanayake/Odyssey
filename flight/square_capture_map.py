#!/usr/bin/env python3
"""
Fly a closed square circuit -- take off to 25 m, then 50 m forward, turn
right, 50 m, turn right, 50 m, turn right, 50 m (back near the start) --
while recording LiDAR scans + ground-truth sensor poses. Mapping happens
AFTER landing, offline, by merging the recorded scans with their recorded
poses -- not a live SLAM/viewer run like flight/slam_live.py or
flight/lidar_viz.py.

Standalone script: does not import or modify any other flight/*.py file. It
does import the shared rig constants/helpers from slam/config.py,
slam/geometry.py and slam/mapping.py (read-only) so the LiDAR extrinsic and
voxel sizes can't drift from the rest of the project.

Flight stability (see CLAUDE.md "Conventions / gotchas for edits" -- these
are exactly the fixes that keep this drone's trajectory smooth, hence "do
this to make the drone stable"):
  - ONE moveOnPathAsync call with a velocity-derived lookahead, instead of
    chaining moveToPositionAsync calls or hand-rolling per-waypoint velocity
    control. Both of the latter cause visible tilt/lurch at waypoint
    transitions, and that lurch shows up as spurious IMU acceleration.
  - each 50 m leg is densified into WAYPOINT_SPACING-m waypoints (see
    slam/config.py's route_waypoints() for the same pattern) so the
    lookahead doesn't cut the 90-degree corners hard -- a bare 4-corner path
    lurches at every turn.
  - drivetrain=ForwardOnly with yaw following velocity, so the drone yaws
    into each turn instead of crabbing sideways through it.

Run after the sim is up:
    ./airsim_venv/bin/python flight/square_capture_map.py

Output goes to datasets_square/<timestamp>/ (gitignored, same convention as
datasets/ and datasets_synth/):
    lidar/*.npy         raw per-scan points, LiDAR-sensor-local frame
    lidar_poses.txt     TUM-format ground-truth LiDAR sensor pose per scan
    map.pcd             merged, voxel-downsampled point-cloud map (world NED)
    map_topdown.png     quick top-down height-colored preview (matplotlib,
                         works headless -- no Vulkan/GLFW window needed)
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import airsim  # noqa: E402

from slam.config import DRONE, LIDAR_NAME, REPO_ROOT  # noqa: E402
from slam.geometry import airsim_pose_to_matrix, write_tum  # noqa: E402
from slam.mapping import VoxelMap  # noqa: E402

# --------------------------------------------------------------------------
# route: closed square, 50 m legs, right turns, 25 m AGL
# --------------------------------------------------------------------------
ALTITUDE = -25.0          # m, NED (negative = up)
LEG_LENGTH = 50.0         # m per leg
WAYPOINT_SPACING = 5.0    # m between densified waypoints (see stability note above)
SPEED = 3.0               # m/s along the path
LIDAR_POLL_HZ = 10        # LiDAR poll rate while flying

OUT_ROOT = REPO_ROOT / "datasets_square"


def build_waypoints() -> list[tuple[float, float, float]]:
    """Densified square in the vehicle's local NED frame: forward, right, right, right."""
    corners = [
        (0.0, 0.0),
        (LEG_LENGTH, 0.0),
        (LEG_LENGTH, LEG_LENGTH),
        (0.0, LEG_LENGTH),
        (0.0, 0.0),
    ]
    pts: list[tuple[float, float, float]] = []
    for i in range(len(corners) - 1):
        a = np.array(corners[i])
        b = np.array(corners[i + 1])
        n = max(1, int(np.linalg.norm(b - a) / WAYPOINT_SPACING))
        for k in range(n):
            p = a + (b - a) * (k / n)
            pts.append((float(p[0]), float(p[1]), ALTITUDE))
    pts.append((float(corners[-1][0]), float(corners[-1][1]), ALTITUDE))
    return pts


def fly_and_capture(client: airsim.MultirotorClient,
                     waypoints: list[tuple[float, float, float]]) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Fly the path via moveOnPathAsync while polling LiDAR on the main thread.

    The path-follow runs on a background thread (its .join() blocks until the
    circuit is done); the main thread polls LiDAR concurrently and stops as
    soon as that thread signals completion.
    """
    length = sum(
        float(np.linalg.norm(np.array(waypoints[i + 1][:2]) - np.array(waypoints[i][:2])))
        for i in range(len(waypoints) - 1)
    )
    expected = length / SPEED
    print(f"Route: {len(waypoints)} waypoints, {length:.0f} m total, "
          f"~{expected:.0f} s at {SPEED} m/s")

    client.enableApiControl(True, DRONE)
    client.armDisarm(True, DRONE)
    print("Taking off...")
    client.takeoffAsync(vehicle_name=DRONE).join()
    client.moveToPositionAsync(0, 0, ALTITUDE, 3.0, vehicle_name=DRONE).join()
    client.hoverAsync(vehicle_name=DRONE).join()
    time.sleep(1.0)

    path = [airsim.Vector3r(*w) for w in waypoints]
    flight_done = threading.Event()
    flight_error: list[Exception] = []

    def _fly():
        try:
            client.moveOnPathAsync(
                path, SPEED,
                timeout_sec=expected * 3.0,
                drivetrain=airsim.DrivetrainType.ForwardOnly,
                yaw_mode=airsim.YawMode(False, 0),
                lookahead=-1, adaptive_lookahead=1,
                vehicle_name=DRONE,
            ).join()
        except Exception as exc:  # pragma: no cover
            flight_error.append(exc)
        finally:
            flight_done.set()

    threading.Thread(target=_fly, daemon=True).start()

    # Own RPC connection -- msgpackrpc's Tornado IOLoop is not thread-safe,
    # so the polling loop can't share the client driving the flight thread.
    poll_client = airsim.MultirotorClient()
    poll_client.confirmConnection()

    scans: list[tuple[int, np.ndarray, np.ndarray]] = []
    last_ts = 0
    print("Capturing LiDAR while flying the square...")
    while not flight_done.is_set():
        d = poll_client.getLidarData(lidar_name=LIDAR_NAME, vehicle_name=DRONE)
        if len(d.point_cloud) >= 3:
            ts = int(d.time_stamp)
            if ts != last_ts:
                last_ts = ts
                pts = np.array(d.point_cloud, dtype=np.float32).reshape(-1, 3)
                T = airsim_pose_to_matrix(d.pose)
                scans.append((ts, pts, T))
                print(f"\r  captured {len(scans)} scans...", end="", flush=True)
        time.sleep(1.0 / LIDAR_POLL_HZ)

    print(f"\nFlight done -- {len(scans)} LiDAR scans captured.")
    if flight_error:
        raise flight_error[0]
    return scans


def save_raw_capture(out_dir: Path, scans: list[tuple[int, np.ndarray, np.ndarray]]) -> None:
    (out_dir / "lidar").mkdir(parents=True, exist_ok=True)
    times, poses = [], []
    for ts, pts, T in scans:
        np.save(out_dir / "lidar" / f"{ts}.npy", pts)
        times.append(ts / 1e9)
        poses.append(T)
    write_tum(out_dir / "lidar_poses.txt", times, poses,
              header="ground-truth LIDAR SENSOR pose, world NED (LidarData.pose)")
    print(f"Raw capture written to {out_dir}")


def build_map(scans: list[tuple[int, np.ndarray, np.ndarray]]) -> VoxelMap:
    """Offline mapping: place every scan by its recorded ground-truth pose and merge.

    Perfect localization, no SLAM -- the same "no SLAM" baseline
    flight/lidar_viz.py uses live, just done as a batch after landing instead
    of accumulated frame-by-frame during flight.
    """
    vm = VoxelMap()
    for _, pts, T in scans:
        pts_world = pts.astype(np.float64) @ T[:3, :3].T + T[:3, 3]
        vm.insert(pts_world)
    vm.flush()
    return vm


def save_topdown_preview(out_path: Path, vm: VoxelMap, waypoints: list[tuple[float, float, float]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = vm.points
    north, east, up = pts[:, 0], pts[:, 1], -pts[:, 2]   # NED: x=north, y=east, z=down

    fig, ax = plt.subplots(figsize=(8, 8))
    sca = ax.scatter(east, north, c=up, cmap="turbo", s=0.5, linewidths=0)
    fig.colorbar(sca, ax=ax, label="height above origin (m)")

    wp = np.array(waypoints)
    ax.plot(wp[:, 1], wp[:, 0], color="white", linewidth=1.2, linestyle="--",
            label="commanded path")
    ax.legend(loc="upper right")

    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")
    ax.set_title(f"Square-circuit LiDAR map -- top-down ({len(pts)} points)")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Top-down preview written to {out_path}")


def main() -> int:
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("Connected to AirSim.")

    waypoints = build_waypoints()
    scans: list[tuple[int, np.ndarray, np.ndarray]] = []
    try:
        scans = fly_and_capture(client, waypoints)
    finally:
        print("Landing...")
        try:
            client.landAsync(vehicle_name=DRONE).join()
            client.armDisarm(False, DRONE)
            client.enableApiControl(False, DRONE)
        except Exception as exc:
            print(f"  (cleanup warning: {exc})")

    if not scans:
        print("No LiDAR scans captured -- nothing to map.", file=sys.stderr)
        return 1

    out_dir = OUT_ROOT / time.strftime("%Y%m%d_%H%M%S")
    save_raw_capture(out_dir, scans)

    print("Building offline point-cloud map from recorded scans + poses...")
    vm = build_map(scans)
    vm.save(out_dir / "map.pcd")
    print(f"Map: {len(vm.points)} points -> {out_dir / 'map.pcd'}")

    save_topdown_preview(out_dir / "map_topdown.png", vm, waypoints)

    print(f"\nDone. Outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
