#!/usr/bin/env python3
"""
Shared-position swarm comms (Drone1..Drone4, AirSimNH).

AirSim has no radio/RF simulation (see CLAUDE.md's "critical gotcha" section
on sensors) -- there is no channel to configure. Every drone in this swarm is
already controlled through the ONE AirSim client used by scripts like
swarm_circle.py, so "communicating" a drone's position to the rest of the
swarm is just reading its GPS through that same client: perfect, instant,
lossless sharing, not a simulated network. If a range limit, latency, or
packet loss ever needs modeling, do it explicitly here (the same way
slam/degradation.py hand-models LiDAR degradation AirSim doesn't simulate)
rather than pretending AirSim provides it.

Run standalone for a demo (arms + takes off all 4, prints shared positions
for a few seconds, lands):
    ./airsim_venv/bin/python flight/swarm_comms.py
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import airsim

# --- swarm layout: name -> spawn (X, Y) in world NED, must match settings.json ---
SPAWNS = {
    "Drone1": (150.0, 150.0),
    "Drone2": (150.0, -150.0),
    "Drone3": (-150.0, 150.0),
    "Drone4": (-150.0, -150.0),
}
DRONES = list(SPAWNS.keys())


@dataclass
class DronePosition:
    """One drone's last-known position, as shared with the rest of the swarm."""
    vehicle_name: str
    latitude: float
    longitude: float
    altitude: float   # GPS altitude, m (up positive)
    world_x: float     # world NED, m (spawn offset applied)
    world_y: float
    timestamp: float   # time.time() this reading was taken


class SwarmPositions:
    """Polls every drone's GPS through the single shared AirSim client.

    There's no separate radio here -- every drone is already reachable
    through this one client, so sharing position is reading everyone's GPS
    each tick, not sending anything over a simulated channel.
    """

    def __init__(self, client: airsim.MultirotorClient, drones: list[str] = DRONES):
        self.client = client
        self.drones = drones
        self._positions: dict[str, DronePosition] = {}

    def refresh(self) -> dict[str, DronePosition]:
        """Poll every drone's GPS + world position; returns the updated dict."""
        now = time.time()
        for name in self.drones:
            gps = self.client.getGpsData(gps_name="", vehicle_name=name).gnss.geo_point
            local = self.client.getMultirotorState(vehicle_name=name).kinematics_estimated.position
            sx, sy = SPAWNS[name]
            self._positions[name] = DronePosition(
                vehicle_name=name,
                latitude=gps.latitude,
                longitude=gps.longitude,
                altitude=gps.altitude,
                world_x=local.x_val + sx,
                world_y=local.y_val + sy,
                timestamp=now,
            )
        return self._positions

    def get(self, vehicle_name: str) -> DronePosition:
        return self._positions[vehicle_name]

    def neighbors(self, vehicle_name: str) -> dict[str, DronePosition]:
        """Every other drone's last-known position, for formation/collision logic."""
        return {n: p for n, p in self._positions.items() if n != vehicle_name}

    def distance(self, a: str, b: str) -> float:
        """Straight-line world-frame distance (m) between two drones' last readings."""
        pa, pb = self._positions[a], self._positions[b]
        return math.hypot(pa.world_x - pb.world_x, pa.world_y - pb.world_y)


def main():
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
    print("Sharing positions for 5s...")
    t_end = time.time() + 5.0
    while time.time() < t_end:
        positions = swarm.refresh()
        for name in DRONES:
            p = positions[name]
            others = ", ".join(
                f"{n}={swarm.distance(name, n):.1f}m" for n in swarm.neighbors(name)
            )
            print(f"[{name}] lat={p.latitude:.6f} lon={p.longitude:.6f} alt={p.altitude:.2f}m "
                  f"world=({p.world_x:.1f},{p.world_y:.1f})  neighbors: {others}")
        time.sleep(1.0)

    print("Landing...")
    for f in [client.landAsync(vehicle_name=d) for d in DRONES]:
        f.join()
    for d in DRONES:
        client.armDisarm(False, d)
        client.enableApiControl(False, d)
    print("Done.")


if __name__ == "__main__":
    main()
