#!/usr/bin/env python3
"""
Gymnasium environment wrapping AirSim for single-drone ring-orbit formation-keeping.

Observation (5,): [radial_error, tang_error, vx, vy, lidar_min_dist]
Action (2,):      [vx_cmd, vy_cmd] world-frame NED, clipped to ±VEL_LIMIT m/s

Train with train_drl.py, then swap the velocity-command block in swarm_circle.py.
"""
import math
import time

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import airsim

# Ring geometry — must match swarm_circle.py
SPAWN_XY   = (0.0, 0.0)          # Drone1 world spawn (X, Y) NED
CENTER_XY  = (0.0, 8.0)          # ring center world (X, Y) NED
RADIUS     = 10.0                 # ring radius (m)
ALTITUDE   = -8.0                 # NED z (negative = up)
SLOT_ANGLE = 0.0                  # Drone1's angle on the ring (radians)
OMEGA      = 2 * math.pi / 20.0  # ring rotation rate (rad/s), 20 s/rev

DT             = 0.1    # control tick (s)
MAX_STEPS      = 300    # episode length (~30 s)
MAX_RADIAL_ERR = 15.0   # early termination threshold (m from ring)
VEL_LIMIT      = 6.0    # action clamp (m/s per axis)
VEL_CMD_DUR    = 0.25   # velocity command hold duration (s)

_OBS_HIGH = np.array(
    [MAX_RADIAL_ERR, MAX_RADIAL_ERR, VEL_LIMIT, VEL_LIMIT, 50.0],
    dtype=np.float32,
)


class AirSimFormationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, drone_name: str = "Drone1"):
        super().__init__()
        self.drone  = drone_name
        self.client = None
        self._step  = 0
        self._t     = 0.0

        self.observation_space = spaces.Box(low=-_OBS_HIGH, high=_OBS_HIGH, dtype=np.float32)
        self.action_space      = spaces.Box(low=-VEL_LIMIT, high=VEL_LIMIT, shape=(2,), dtype=np.float32)

    # ------------------------------------------------------------------ helpers

    def _connect(self):
        if self.client is None:
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()

    def _state(self):
        s  = self.client.getMultirotorState(vehicle_name=self.drone)
        p  = s.kinematics_estimated.position
        v  = s.kinematics_estimated.linear_velocity
        sx, sy = SPAWN_XY
        return p.x_val + sx, p.y_val + sy, v.x_val, v.y_val

    def _lidar_min(self):
        """Distance from the drone to the nearest LiDAR return, in metres.

        Requires settings.json to declare DataFrame: "SensorLocalFrame" — the
        norm below is only a distance *from the sensor* if the points are in the
        sensor's own frame. Under the old "VehicleInertialFrame" setting this
        measured distance from the SPAWN ORIGIN instead, so the obstacle channel
        fed to PPO grew with how far the drone had flown rather than reporting
        anything about obstacles.

        NO_RETURN doubles as the observation-space bound, so a genuinely distant
        return and an empty scan both read as "clear" — intended, since this
        channel exists for collision avoidance, not for mapping.
        """
        NO_RETURN = 50.0
        data = self.client.getLidarData(lidar_name="LidarSensor1", vehicle_name=self.drone)
        if len(data.point_cloud) < 3:
            return NO_RETURN
        pts = np.array(data.point_cloud, dtype=np.float32).reshape(-1, 3)
        return min(float(np.linalg.norm(pts, axis=1).min()), NO_RETURN)

    def _ring_slot(self):
        angle = SLOT_ANGLE + OMEGA * self._t
        cx, cy = CENTER_XY
        return cx + RADIUS * math.cos(angle), cy + RADIUS * math.sin(angle)

    def _obs(self):
        wx, wy, vx, vy = self._state()
        tx, ty         = self._ring_slot()
        cx, cy         = CENTER_XY

        rx, ry  = wx - cx, wy - cy
        rad     = math.hypot(rx, ry) or 1e-3
        radial  = rad - RADIUS               # signed: positive = outside ring

        urx, ury = rx / rad, ry / rad
        utx, uty = -ury, urx                 # tangential unit (CCW)
        tang     = (wx - tx) * utx + (wy - ty) * uty  # signed tangential offset

        obs = np.array([radial, tang, vx, vy, self._lidar_min()], dtype=np.float32)
        return np.clip(obs, -_OBS_HIGH, _OBS_HIGH)

    # ------------------------------------------------------------------ gym API

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._connect()
        self._step = 0
        self._t    = 0.0

        self.client.reset()
        self.client.enableApiControl(True, self.drone)
        self.client.armDisarm(True, self.drone)
        self.client.takeoffAsync(vehicle_name=self.drone).join()

        tx, ty = self._ring_slot()
        sx, sy = SPAWN_XY
        self.client.moveToPositionAsync(
            tx - sx, ty - sy, ALTITUDE, 5.0, vehicle_name=self.drone
        ).join()

        return self._obs(), {}

    def step(self, action):
        vx_cmd, vy_cmd = float(action[0]), float(action[1])

        self.client.moveByVelocityZAsync(
            vx_cmd, vy_cmd, ALTITUDE, VEL_CMD_DUR,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(False, 0),
            vehicle_name=self.drone,
        )
        time.sleep(DT)
        self._t    += DT
        self._step += 1

        obs = self._obs()
        radial, tang = obs[0], obs[1]
        collided = self.client.simGetCollisionInfo(vehicle_name=self.drone).has_collided

        reward = (
            -abs(radial)
            - abs(tang)
            - 0.05 * math.hypot(vx_cmd, vy_cmd)
            - (100.0 if collided else 0.0)
        )

        terminated = collided or abs(radial) > MAX_RADIAL_ERR
        truncated  = self._step >= MAX_STEPS

        return obs, reward, terminated, truncated, {}

    def close(self):
        if self.client:
            try:
                self.client.armDisarm(False, self.drone)
                self.client.enableApiControl(False, self.drone)
            except Exception:
                pass
