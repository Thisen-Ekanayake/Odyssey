#!/usr/bin/env python3
"""
Take off to 5 m and fly forward at 2 m/s. Drone1 only, no sensors.

Run after the sim is up:
    ./airsim_venv/bin/python flight/autonomous_navigate.py

Ctrl+C to land and exit.
"""
import time

import airsim

DRONE = "Drone1"
ALTITUDE = -5.0   # NED: negative = up
SPEED = 2.0       # m/s forward


def main():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("Connected to AirSim.")

    client.enableApiControl(True, DRONE)
    client.armDisarm(True, DRONE)

    print("Taking off...")
    client.takeoffAsync(vehicle_name=DRONE).join()
    client.moveToPositionAsync(0, 0, ALTITUDE, 3.0, vehicle_name=DRONE).join()
    print(f"Hovering at {abs(ALTITUDE):.0f} m. Flying forward at {SPEED} m/s — Ctrl+C to land and quit.")

    try:
        while True:
            client.moveByVelocityZBodyFrameAsync(
                SPEED, 0, ALTITUDE, 1.0,
                vehicle_name=DRONE,
            )
            time.sleep(1.0)

    finally:
        print("\nLanding...")
        client.landAsync(vehicle_name=DRONE).join()
        client.armDisarm(False, DRONE)
        client.enableApiControl(False, DRONE)
        print("Done.")


if __name__ == "__main__":
    main()
