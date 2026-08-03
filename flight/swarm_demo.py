#!/usr/bin/env python3
"""
Minimal 3-drone swarm demo for AirSim/Blocks.

Run this on the HOST (not in the container) after the Blocks sim window is up:
    ./airsim_venv/bin/python flight/swarm_demo.py

It connects to the sim over RPC (127.0.0.1:41451, exposed via --net=host),
takes all drones off together, flies a simple line formation, hovers, lands.
Vehicle names must match settings.json: Drone1, Drone2, Drone3.
"""
import time
import airsim

DRONES = ["Drone1", "Drone2", "Drone3"]
ALTITUDE = -8.0   # NED: negative is up -> 8 m altitude
SPEED = 4.0       # m/s


def main():
    client = airsim.MultirotorClient()   # defaults to 127.0.0.1:41451
    client.confirmConnection()
    print("Connected to AirSim.")

    # Arm + take control of every drone.
    for d in DRONES:
        client.enableApiControl(True, d)
        client.armDisarm(True, d)

    # Take off all drones simultaneously (async -> join).
    print("Taking off...")
    futures = [client.takeoffAsync(vehicle_name=d) for d in DRONES]
    for f in futures:
        f.join()

    # Climb to formation altitude, spread along X (each drone's own local frame).
    print("Forming up...")
    futures = []
    for i, d in enumerate(DRONES):
        futures.append(
            client.moveToPositionAsync(i * 5.0, 0.0, ALTITUDE, SPEED, vehicle_name=d)
        )
    for f in futures:
        f.join()

    print("Holding formation 5s...")
    time.sleep(5)

    # Land + release.
    print("Landing...")
    futures = [client.landAsync(vehicle_name=d) for d in DRONES]
    for f in futures:
        f.join()
    for d in DRONES:
        client.armDisarm(False, d)
        client.enableApiControl(False, d)
    print("Done.")


if __name__ == "__main__":
    main()
