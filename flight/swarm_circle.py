#!/usr/bin/env python3
"""
5-drone ring formation that orbits as a whole about its center (AirSim/Blocks).

Run on the HOST after the Blocks sim window is up:
    ./airsim_venv/bin/python flight/swarm_circle.py

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
KP_RADIAL = 1.0         # P-gain pulling each drone back onto the ring radius (1/s)
VEL_CMD_DURATION = 0.25 # how long each velocity command persists (s); > DT for continuity


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
    omega = 2.0 * math.pi / REV_PERIOD       # ring angular rate magnitude (rad/s)
    v_tan = DIRECTION * omega * RADIUS        # tangential speed for the orbit (signed, m/s)
    form_speed = omega * RADIUS * 1.5         # speed used only to fly into the initial ring

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
            lx, ly, ALTITUDE, form_speed,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(False, yaw),
            vehicle_name=d))
    for f in futures:
        f.join()
    print("Holding ring 2s...")
    time.sleep(2.0)

    # Orbit with closed-loop VELOCITY control (smooth + stable). Each tick we read
    # each drone's actual world position and command a velocity made of:
    #   - a tangential component (v_tan) that drives the rotation around the center, and
    #   - a radial P-correction that gently pulls it back onto the ring radius.
    # moveByVelocityZAsync holds altitude, so motion stays in-plane and continuous --
    # no stop-start re-targeting, which is what made the drones lurch and tilt before.
    print(f"Orbiting {N_REVS} rev(s), {REV_PERIOD:.0f}s each, "
          f"{'CCW' if DIRECTION > 0 else 'CW'}...")
    cx, cy = CENTER
    duration = N_REVS * REV_PERIOD
    t = 0.0
    while t < duration:
        for d in DRONES:
            p = client.getMultirotorState(vehicle_name=d).kinematics_estimated.position
            sx, sy = SPAWNS[d]
            wx, wy = p.x_val + sx, p.y_val + sy        # actual world position
            rx, ry = wx - cx, wy - cy
            rad = math.hypot(rx, ry) or 1e-3
            urx, ury = rx / rad, ry / rad              # radial unit (outward)
            utx, uty = -ury, urx                       # tangential unit (CCW)
            v_rad = -KP_RADIAL * (rad - RADIUS)        # pull back toward the ring radius
            vx = v_tan * utx + v_rad * urx             # world-frame NED velocity (x=N, y=E)
            vy = v_tan * uty + v_rad * ury
            yaw = math.degrees(math.atan2(cy - wy, cx - wx))   # face center
            yaw_mode = airsim.YawMode(False, yaw) if FACE_CENTER else airsim.YawMode(False, 0)
            client.moveByVelocityZAsync(
                vx, vy, ALTITUDE, VEL_CMD_DURATION,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=yaw_mode,
                vehicle_name=d)
        time.sleep(DT)
        t += DT

    # Arrest motion before landing so the drones don't drift off the ring.
    print("Stopping...")
    for d in DRONES:
        client.hoverAsync(vehicle_name=d)
    time.sleep(2.0)

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
