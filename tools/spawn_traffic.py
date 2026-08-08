#!/usr/bin/env python3
"""
Scatter static "traffic" (parked cars, and other outdoor props if present)
around the flight area in the currently loaded environment, so YOLO has
something to detect (see flight/detect_objects.py).

AirSimNH ships with no moving traffic or pedestrians -- only Microsoft's
CityEnviron does, which isn't in our downloaded set (see fetch_envs.sh) -- so
this spawns *static* mesh assets already baked into the level via
simSpawnObject, not driving/walking AI. Assets are discovered at runtime with
simListAssets() rather than hardcoded, since exact asset paths vary by level.

Run after the sim is up:
    ./airsim_venv/bin/python tools/spawn_traffic.py

    # list matching assets only, without spawning anything:
    ./airsim_venv/bin/python tools/spawn_traffic.py --list

    # spawn a different count, or force a specific asset:
    ./airsim_venv/bin/python tools/spawn_traffic.py --count 12
    ./airsim_venv/bin/python tools/spawn_traffic.py --asset /Game/Foo/Car_02.Car_02
"""
import argparse
import math
import random

import airsim

# Case-insensitive substrings to look for in the level's asset registry.
ASSET_KEYWORDS = ["car", "vehicle", "truck", "bus", "bench", "prop"]

SPAWN_CENTER = (0.0, 8.0)   # matches the swarm's centroid used in swarm_circle.py
SPAWN_RADIUS = 25.0          # scatter radius (m)
MIN_RADIUS = 5.0             # keep clear of the drones' spawn line
GROUND_Z = -0.5               # NED z: slightly above ground; physics settles it down
COUNT = 8


def find_candidates(client: airsim.MultirotorClient) -> list:
    assets = client.simListAssets()
    return sorted({a for a in assets if any(k in a.lower() for k in ASSET_KEYWORDS)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="List matching assets and exit")
    parser.add_argument("--count", type=int, default=COUNT)
    parser.add_argument("--asset", help="Force a specific asset name instead of auto-picking")
    args = parser.parse_args()

    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("Connected to AirSim.")

    candidates = find_candidates(client)
    if not candidates:
        print("No car/vehicle/prop-like assets found in this level's asset registry.")
        print("This environment likely has nothing spawnable to use as traffic.")
        return

    print(f"Found {len(candidates)} candidate asset(s):")
    for a in candidates:
        print(f"  {a}")

    if args.list:
        return

    asset = args.asset or candidates[0]
    print(f"\nSpawning {args.count}x '{asset}' ...")

    cx, cy = SPAWN_CENTER
    for i in range(args.count):
        angle = random.uniform(0, 2 * math.pi)
        r = random.uniform(MIN_RADIUS, SPAWN_RADIUS)
        x, y = cx + r * math.cos(angle), cy + r * math.sin(angle)
        yaw = random.uniform(0, 2 * math.pi)
        pose = airsim.Pose(
            airsim.Vector3r(x, y, GROUND_Z),
            airsim.to_quaternion(0, 0, yaw),
        )
        try:
            name = client.simSpawnObject(
                f"traffic_{i}", asset, pose, airsim.Vector3r(1, 1, 1),
                physics_enabled=True,
            )
            print(f"  spawned {name} at ({x:.1f}, {y:.1f})")
        except Exception as e:
            print(f"  failed to spawn #{i}: {e}")

    print("\nDone. Run flight/detect_objects.py to see if YOLO picks them up.")
    print("(Objects persist until the sim restarts, or destroy with simDestroyObject.)")


if __name__ == "__main__":
    main()
