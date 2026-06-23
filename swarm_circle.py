#!/usr/bin/env python3
"""
5-drone ring formation that orbits as a whole about its center (AirSim/Blocks).

Run on the HOST after the Blocks sim window is up:
    ./airsim_venv/bin/python swarm_circle.py

Behaviour:
    arm -> take off together -> spread into a 5-point ring -> spin the whole ring
    about its center for N revolutions (each drone yaws to face inward) -> land.

Vehicle names + spawn offsets below MUST match settings.json (Drone1..Drone5).

Coordinate-frame note (important):
    Each vehicle moves in its OWN local NED frame, whose origin is that vehicle's
    spawn point (the X/Y in settings.json). To place a drone at an absolute WORLD
    point we subtract its spawn offset: local = world - spawn. Without this the
    ring would be distorted by the spawn spacing (up to 16 m here vs a 10 m radius).
    NED: x=North, y=East, z=Down (negative z is up).
"""
import math
import time
import airsim

# --- swarm layout: name -> spawn (X, Y) in world NED, must match settings.json ---
SPAWNS = {
    "Drone1": (0.0, 0.0),
    "Drone2": (0.0, 4.0),
    "Drone3": (0.0, 8.0),
    "Drone4": (0.0, 12.0),
    "Drone5": (0.0, 16.0),
}
DRONES = list(SPAWNS.keys())

# --- ring + orbit parameters ---
CENTER = (0.0, 8.0)     # world (X, Y) the ring is centered on (centroid of spawns)
RADIUS = 10.0           # ring radius (m)
ALTITUDE = -8.0         # NED z: -8 -> 8 m above ground
REV_PERIOD = 20.0       # seconds per full revolution of the whole ring
N_REVS = 2              # how many revolutions to fly
DIRECTION = +1          # +1 = counter-clockwise (seen from above), -1 = clockwise
DT = 0.1                # control tick (s); ~1/DT commands per drone per second
FACE_CENTER = True      # yaw each drone to look at the ring center while orbiting


def world_to_local(name, wx, wy):
    """World (X,Y) -> that vehicle's local-frame (x,y) by removing its spawn offset."""
    sx, sy = SPAWNS[name]
    return wx - sx, wy - sy


def ring_target(name, angle):
    """Local-frame (x, y) + inward-facing yaw (deg) for `name` at ring angle `angle`."""
    cx, cy = CENTER
    wx = cx + RADIUS * math.cos(angle)
    wy = cy + RADIUS * math.sin(angle)
    lx, ly = world_to_local(name, wx, wy)
    # yaw to face the center: direction (center - drone) in NED, measured from North(X) toward East(Y)
    yaw_deg = math.degrees(math.atan2(cy - wy, cx - wx))
    return lx, ly, yaw_deg


def main():
    client = airsim.MultirotorClient()      # defaults to 127.0.0.1:41451
    client.confirmConnection()
    print(f"Connected. Swarm: {', '.join(DRONES)}")

    n = len(DRONES)
    slot = 2.0 * math.pi / n                 # angular spacing between drones (72 deg for 5)
    omega = DIRECTION * 2.0 * math.pi / REV_PERIOD   # ring angular velocity (rad/s)
    # tangential speed each drone needs (v = omega*R), with margin to also correct drift
    speed = abs(omega) * RADIUS * 1.6

    # Arm + take control of every drone.
    for d in DRONES:
        client.enableApiControl(True, d)
        client.armDisarm(True, d)

    print("Taking off...")
    for f in [client.takeoffAsync(vehicle_name=d) for d in DRONES]:
        f.join()

    # Spread into the initial ring (phase 0). Drone i sits at angle i*slot.
    print("Forming ring...")
    futures = []
    for i, d in enumerate(DRONES):
        lx, ly, yaw = ring_target(d, i * slot)
        futures.append(client.moveToPositionAsync(
            lx, ly, ALTITUDE, speed,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(False, yaw),
            vehicle_name=d))
    for f in futures:
        f.join()
    print("Holding ring 2s...")
    time.sleep(2.0)

    # Orbit: advance the whole ring's phase over time and re-target every tick.
    # Re-targeting the exact circle point each tick self-corrects integration drift.
    print(f"Orbiting {N_REVS} rev(s), {REV_PERIOD:.0f}s each, "
          f"{'CCW' if DIRECTION > 0 else 'CW'}...")
    duration = N_REVS * REV_PERIOD
    t = 0.0
    while t < duration:
        phase = omega * t
        for i, d in enumerate(DRONES):
            lx, ly, yaw = ring_target(d, phase + i * slot)
            yaw_mode = airsim.YawMode(False, yaw) if FACE_CENTER else airsim.YawMode(False, 0)
            client.moveToPositionAsync(
                lx, ly, ALTITUDE, speed,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=yaw_mode,
                vehicle_name=d)               # fire-and-continue; next tick overrides
        time.sleep(DT)
        t += DT

    # Land + release.
    print("Landing...")
    for f in [client.landAsync(vehicle_name=d) for d in DRONES]:
        f.join()
    for d in DRONES:
        client.armDisarm(False, d)
        client.enableApiControl(False, d)
    print("Done.")


if __name__ == "__main__":
    main()
