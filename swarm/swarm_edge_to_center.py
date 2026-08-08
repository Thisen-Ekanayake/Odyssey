#!/usr/bin/env python3
"""
Takes off the swarm (Drone1-4), then each drone flies its own leg of the
formation quadrilateral's "4 small squares" segmentation (see
swarm_lines_viz.py): forward along its perimeter edge toward the next
corner, until it reaches that edge's midpoint, then a slow left turn onto
the line heading toward the shared diagonal-intersection center -- stopping
STANDOFF_M short of it, since all 4 center-bound lines meet at the exact
same point and flying every drone all the way there would collide.

Direction of travel around the perimeter is picked so the midpoint-to-center
turn comes out as a genuine LEFT turn: for a convex quadrilateral the turn
sense (left vs right) is the same at every vertex for a given direction, so
testing it once at the first corner (via a 2D cross product between the
travel heading and the turn-toward-center direction) fixes the direction for
all 4 drones, whichever way SPAWNS happens to be laid out.

All the geometry (corners, midpoints, center) is computed in the shared
WORLD frame -- it has to be, since it depends on relationships between
different vehicles' positions. But moveOnPathAsync/moveToPositionAsync
target each vehicle's own LOCAL frame, which in this multi-vehicle settings
is offset from world by an undocumented per-vehicle amount (same gotcha
swarm_comms.SwarmPositions.offset() exists to invert when READING a
position -- see that module's docstring). Every waypoint handed to
moveOnPathAsync below is converted world -> local via that same offset,
subtracted this time instead of added, right before it's queued.

The "slow, no sudden turn" cornering is AirSim's own moveOnPathAsync
lookahead blending the waypoints along corner -> midpoint -> standoff point
(densified per CLAUDE.md's guidance so the lookahead can't cut the corner) --
the same built-in path-following the project prefers over hand-rolled
per-waypoint velocity control (see record_dataset.py / slam/recorder.py),
which visibly tilts/lurches at corners.

Run after the sim is up, with Drone1-4 armed at their settings.json corners:
    ./airsim_venv/bin/python swarm/swarm_edge_to_center.py
"""
from __future__ import annotations

import math
import time

import airsim

from swarm_comms import DRONES, SPAWNS, SwarmPositions

ALTITUDE = -50.0          # NED z; clears AirSimNH rooftops/canopy (matches swarm_converge.py)
CLIMB_SPEED = 3.0         # m/s for the initial climb
SPEED = 3.0               # m/s along each leg -- kept modest so the corner turn reads as gradual
STANDOFF_M = 5.0          # m short of the shared center each drone stops at
WAYPOINTS_PER_LEG = 4     # densify each leg so AirSim's lookahead can't cut the corner


def _lerp(p1, p2, n):
    """n points strictly between p1 and p2, evenly spaced, ending at p2."""
    return [
        (p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t, p1[2] + (p2[2] - p1[2]) * t)
        for t in (i / n for i in range(1, n + 1))
    ]


def _segment_intersection(p1, p2, p3, p4):
    """Point where segment p1-p2 crosses segment p3-p4 (2D, z taken off p1-p2's line)."""
    x1, y1, _ = p1
    x2, y2, _ = p2
    x3, y3, _ = p3
    x4, y4, _ = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1), p1[2] + t * (p2[2] - p1[2]))


def run(client: airsim.MultirotorClient) -> None:
    """Fly the full edge-to-center maneuver on an already-connected client."""
    print(f"Connected. Swarm: {', '.join(DRONES)}")

    for d in DRONES:
        client.enableApiControl(True, d)
        client.armDisarm(True, d)

    print("Taking off...")
    for f in [client.takeoffAsync(vehicle_name=d) for d in DRONES]:
        f.join()

    print(f"Climbing to {abs(ALTITUDE):.0f} m...")
    for f in [client.moveToZAsync(ALTITUDE, CLIMB_SPEED, vehicle_name=d) for d in DRONES]:
        f.join()

    swarm = SwarmPositions(client)
    positions = swarm.refresh()
    world = {d: (positions[d].world_x, positions[d].world_y, ALTITUDE) for d in DRONES}

    # Perimeter order by angle around the centroid, same technique as
    # swarm_lines_viz.py -- gives the correct cyclic adjacency (which drone
    # neighbors which) regardless of which direction it happens to wind.
    cx = sum(p[0] for p in world.values()) / len(world)
    cy = sum(p[1] for p in world.values()) / len(world)
    order = sorted(DRONES, key=lambda d: math.atan2(world[d][1] - cy, world[d][0] - cx))
    n = len(order)
    center = _segment_intersection(world[order[0]], world[order[2]], world[order[1]], world[order[3]])

    # Fix the winding direction so the midpoint-to-center cut is a LEFT turn
    # (checked once at order[0] -- consistent at every vertex for a convex
    # quadrilateral, see module docstring).
    corner0, next0 = world[order[0]], world[order[1]]
    mid0 = ((corner0[0] + next0[0]) / 2, (corner0[1] + next0[1]) / 2, corner0[2])
    heading0 = (next0[0] - corner0[0], next0[1] - corner0[1])
    turn0 = (center[0] - mid0[0], center[1] - mid0[1])
    cross0 = heading0[0] * turn0[1] - heading0[1] * turn0[0]
    step = 1 if cross0 < 0 else -1
    print(f"Perimeter order: {order} ({'forward' if step == 1 else 'reversed'} winding "
          f"confirms a left turn at each midpoint)")
    print(f"Diagonal intersection (shared center): ({center[0]:.1f}, {center[1]:.1f})")

    print(f"Flying each drone: corner -> edge midpoint -> {STANDOFF_M:.0f} m short of center...")
    futures = []
    for i, d in enumerate(order):
        corner = world[d]
        nxt = world[order[(i + step) % n]]
        midpoint = ((corner[0] + nxt[0]) / 2, (corner[1] + nxt[1]) / 2, corner[2])

        to_center = (center[0] - midpoint[0], center[1] - midpoint[1])
        dist_to_center = math.hypot(*to_center)
        ratio = max(0.0, (dist_to_center - STANDOFF_M) / dist_to_center) if dist_to_center > 1e-6 else 0.0
        standoff = (midpoint[0] + to_center[0] * ratio, midpoint[1] + to_center[1] * ratio, midpoint[2])

        leg = _lerp(corner, midpoint, WAYPOINTS_PER_LEG) + _lerp(midpoint, standoff, WAYPOINTS_PER_LEG)
        dx, dy = swarm.offset(d)
        path = [airsim.Vector3r(x - dx, y - dy, z) for x, y, z in leg]
        length = (math.hypot(midpoint[0] - corner[0], midpoint[1] - corner[1])
                  + math.hypot(standoff[0] - midpoint[0], standoff[1] - midpoint[1]))

        futures.append(client.moveOnPathAsync(
            path, SPEED,
            timeout_sec=(length / SPEED) * 2.0 + 20.0,
            drivetrain=airsim.DrivetrainType.ForwardOnly,
            yaw_mode=airsim.YawMode(False, 0),
            lookahead=-1, adaptive_lookahead=1,
            vehicle_name=d,
        ))
    for f in futures:
        f.join()

    print("Holding position...")
    for d in DRONES:
        client.hoverAsync(vehicle_name=d)
    time.sleep(2.0)

    positions = swarm.refresh()
    for d in DRONES:
        p = positions[d]
        print(f"[{d}] world=({p.world_x:.1f},{p.world_y:.1f})")
    time.sleep(3.0)

    print("Landing...")
    for f in [client.landAsync(vehicle_name=d) for d in DRONES]:
        f.join()
    for d in DRONES:
        client.armDisarm(False, d)
        client.enableApiControl(False, d)
    print("Done.")


def main() -> None:
    client = airsim.MultirotorClient()
    client.confirmConnection()
    run(client)


if __name__ == "__main__":
    main()
