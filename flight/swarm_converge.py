#!/usr/bin/env python3
"""
Converge the 4-drone swarm (Drone1..Drone4) onto a shared virtual center
without colliding.

The virtual center is Drone1's position, read LIVE each tick through
SwarmPositions (swarm_comms.py) -- so this is a real use of the shared-GPS
comms layer, not a hardcoded point: it starts at (150, 150) simply because
that's where Drone1 spawns. Every other drone's target is its own spawn
offset from that center, uniformly SCALED DOWN toward zero over time. A
uniform scale is a similarity transform, so every pair of drones keeps a
constant RATIO of distance to each other throughout the approach -- the
whole formation shrinks like a photograph, nobody moves independently.

The scale-down stops at a floor set by MIN_SEPARATION_M instead of reaching
zero, because zero is exactly where Drone1 is sitting -- that floor is the
collision-avoidance: Drone2..4 end up a few meters out from Drone1 (and from
each other) on a small shrunk copy of their original square, not stacked on
top of it.

Run after the sim is up, with Drone1..Drone4 armed at their settings.json
corners:
    ./airsim_venv/bin/python flight/swarm_converge.py
"""
from __future__ import annotations

import time

import airsim

from swarm_comms import DRONES, SPAWNS, SwarmPositions

ALTITUDE = -25.0          # NED z; clears AirSimNH rooftops/canopy (see slam/config.py)
MIN_SEPARATION_M = 5.0    # closest any two drones may end up to each other
# Farthest mover (Drone4) starts ~440m from center. Verified live at
# DURATION_S=100: it covered ~335m (438m -> 103m remaining) while still
# accelerating, averaging 3.35 m/s -- SimpleFlight's own acceleration ramp
# takes a while to reach cruise speed. 150s leaves real margin to close the
# rest at a similar or faster rate rather than just approaching the target.
DURATION_S = 150.0         # time to fully shrink the formation
DT = 0.2                   # control tick (s)
KP = 0.6                   # P-gain: velocity toward the shrinking target (1/s)
MAX_SPEED = 8.0             # m/s cap on commanded velocity
VEL_CMD_DURATION = 0.3      # > DT for continuity, matches swarm_circle.py's convention
CLIMB_SPEED = 3.0           # m/s used for the initial climb to ALTITUDE


def main() -> None:
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print(f"Connected. Swarm: {', '.join(DRONES)}")

    for d in DRONES:
        client.enableApiControl(True, d)
        client.armDisarm(True, d)

    print("Taking off...")
    for f in [client.takeoffAsync(vehicle_name=d) for d in DRONES]:
        f.join()

    swarm = SwarmPositions(client)
    positions = swarm.refresh()

    # Offsets from Drone1's INITIAL position -- fixed for the whole run, so
    # scaling them stays a clean similarity transform even though Drone1's
    # live position (re-read every tick below) can drift slightly.
    center0 = positions["Drone1"]
    offsets = {
        d: (SPAWNS[d][0] - center0.world_x, SPAWNS[d][1] - center0.world_y)
        for d in DRONES
    }
    print(f"Virtual center (Drone1's shared position): "
          f"({center0.world_x:.1f}, {center0.world_y:.1f})")

    # Drone1's own offset is (0, 0) -- it IS the center, so the tightest
    # constraint is always some other drone closing in on it. Floor the
    # scale so that drone never gets closer than MIN_SEPARATION_M to Drone1
    # (which, by construction of a uniform scale, also keeps every OTHER
    # pair at or above that separation -- see the module docstring).
    non_anchor_mags = [
        (ox ** 2 + oy ** 2) ** 0.5
        for d, (ox, oy) in offsets.items()
        if d != "Drone1"
    ]
    min_scale = MIN_SEPARATION_M / min(non_anchor_mags)
    print(f"Shrinking formation to scale={min_scale:.4f} "
          f"(keeps >= {MIN_SEPARATION_M:.1f} m between any two drones)")

    print(f"Climbing to {abs(ALTITUDE):.0f} m...")
    for f in [client.moveToZAsync(ALTITUDE, CLIMB_SPEED, vehicle_name=d) for d in DRONES]:
        f.join()
    client.hoverAsync(vehicle_name="Drone1")   # Drone1 is the anchor -- it never moves

    movers = [d for d in DRONES if d != "Drone1"]
    print(f"Converging {movers} onto Drone1 over {DURATION_S:.0f}s...")
    t = 0.0
    while t < DURATION_S:
        positions = swarm.refresh()
        center = positions["Drone1"]
        scale = 1.0 - (1.0 - min_scale) * min(t / DURATION_S, 1.0)
        for d in movers:
            ox, oy = offsets[d]
            target_x = center.world_x + scale * ox
            target_y = center.world_y + scale * oy
            p = positions[d]
            vx = KP * (target_x - p.world_x)
            vy = KP * (target_y - p.world_y)
            speed = (vx ** 2 + vy ** 2) ** 0.5
            if speed > MAX_SPEED:
                vx, vy = vx * MAX_SPEED / speed, vy * MAX_SPEED / speed
            client.moveByVelocityZAsync(
                vx, vy, ALTITUDE, VEL_CMD_DURATION,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(False, 0),
                vehicle_name=d)
        time.sleep(DT)
        t += DT

    print("Holding final formation...")
    for d in DRONES:
        client.hoverAsync(vehicle_name=d)
    time.sleep(2.0)

    positions = swarm.refresh()
    for d in DRONES:
        p = positions[d]
        others = ", ".join(f"{n}={swarm.distance(d, n):.1f}m" for n in swarm.neighbors(d))
        print(f"[{d}] world=({p.world_x:.1f},{p.world_y:.1f})  neighbors: {others}")
    time.sleep(3.0)

    print("Landing...")
    for f in [client.landAsync(vehicle_name=d) for d in DRONES]:
        f.join()
    for d in DRONES:
        client.armDisarm(False, d)
        client.enableApiControl(False, d)
    print("Done.")


if __name__ == "__main__":
    main()
