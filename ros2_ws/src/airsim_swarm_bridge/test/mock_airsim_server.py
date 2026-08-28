#!/usr/bin/env python3
"""A stand-in AirSim RPC server, for testing the bridge with no simulator.

Run this with the HOST venv (``airsim_venv``, Python 3.10), because it is built
on the very ``msgpack-rpc-python``/``tornado`` stack the container cannot have.
That asymmetry is the point: the reference implementation serves, our
tornado-free client in :mod:`airsim_swarm_bridge.rpc` connects, and any
disagreement about packing (``use_bin_type``, ``raw``, ``strict_map_key``, the
``[type, msgid, ...]`` envelope) shows up immediately instead of at 41451 with a
simulator in the way.

    ./airsim_venv/bin/python ros2_ws/src/airsim_swarm_bridge/test/mock_airsim_server.py --port 41999

Responses mimic AirSim's shapes (nested maps, w-first quaternions, a flat
``point_cloud`` float list, binary image payloads) but not its physics -- the
drone hovers on a fixed circle so the numbers move.

Two knobs exist purely so the mock can be made to *fail* like the real thing:

``--clock-rate`` / ``--clock-skew``
    Sensor payloads are stamped from a simulated clock that need not track wall
    time. Real AirSim uses a SteppableClock for SimpleFlight multirotors and it
    routinely runs at ~0.5x with four drones loaded. Defaulting this to 1.0 was
    how the cooperative-mapping test came to pass against a bug that broke the
    real demo completely: with the mock's "sim clock" identical to the wall clock,
    a bridge that stamped sensors from one and TF from the other looked fine.
    Run the mapping test at ``--clock-rate 0.5`` and it does not.

``--points-per-sweep``
    Real AirSim returns ~16k points per sweep at the configured 300k points/s.
    The original 360 was three orders of magnitude off the load octomap_server
    actually has to absorb.
"""
from __future__ import annotations

import argparse
import math
import time

import msgpackrpc

T0 = time.time()

# Simulated-clock parameters; overridden from argv in main(). Defaults reproduce
# the old behaviour exactly (sim clock == wall clock).
CLOCK_RATE = 1.0
CLOCK_SKEW = 0.0


def _sim_ns() -> int:
    """The simulated clock, in nanoseconds -- what AirSim puts in ``time_stamp``.

    Advances at CLOCK_RATE relative to wall time, from an initial offset of
    CLOCK_SKEW seconds. At rate 1.0 / skew 0.0 this is just ``time.time()``.
    """
    return int((T0 + CLOCK_SKEW + (time.time() - T0) * CLOCK_RATE) * 1e9)


# Points per LiDAR sweep; overridden from argv in main().
POINTS_PER_SWEEP = 360
_SWEEP_CACHE: list[float] | None = None


def _sweep() -> list[float]:
    """One sensor-frame sweep, flat [x,y,z,...], built once and cached.

    16 channels over the rig's -45..+15 deg vertical FOV, azimuth filling the
    rest, on a 20 m cylinder. The shape matters to
    ``test_cooperative_mapping.py``: returns must stay well inside the 30 m radius
    that test uses to associate voxels with the drone that saw them.
    """
    global _SWEEP_CACHE
    if _SWEEP_CACHE is not None and len(_SWEEP_CACHE) == 3 * POINTS_PER_SWEEP:
        return _SWEEP_CACHE

    channels = 16
    per_channel = max(1, POINTS_PER_SWEEP // channels)
    pts: list[float] = []
    # Indexed so the sweep is EXACTLY POINTS_PER_SWEEP points: a flag that
    # silently rounds down is a flag that makes callers assert wrong numbers.
    for n in range(POINTS_PER_SWEEP):
        ch, i = n % channels, n // channels
        # SensorLocalFrame is FRD, so +z is DOWN: the downward-biased FOV of an
        # aerial rig gives mostly positive z here.
        elev = math.radians(15.0 - 60.0 * ch / (channels - 1))
        a = 2.0 * math.pi * (i % per_channel) / per_channel
        r = 20.0 * math.cos(elev)
        pts.extend([r * math.cos(a), r * math.sin(a), -20.0 * math.sin(elev)])
    _SWEEP_CACHE = pts
    return pts


def _vec(x=0.0, y=0.0, z=0.0):
    return {"x_val": float(x), "y_val": float(y), "z_val": float(z)}


def _quat(x=0.0, y=0.0, z=0.0, w=1.0):
    return {"w_val": float(w), "x_val": float(x), "y_val": float(y), "z_val": float(z)}


class MockAirSim:
    """Only the calls the bridge makes; anything else raises, as AirSim would."""

    VEHICLES = ["Drone1", "Drone2", "Drone3", "Drone4"]
    SPAWN = {"Drone1": (150.0, 150.0), "Drone2": (90.0, -210.0),
             "Drone3": (-150.0, 150.0), "Drone4": (-160.0, -160.0)}

    # -- housekeeping ------------------------------------------------------

    def ping(self):
        return True

    def getServerVersion(self):
        return 1

    def getMinRequiredClientVersion(self):
        return 1

    def listVehicles(self):
        return self.VEHICLES

    def enableApiControl(self, is_enabled, vehicle_name):
        return None

    def isApiControlEnabled(self, vehicle_name):
        return True

    def armDisarm(self, arm, vehicle_name):
        return True

    def simPause(self, is_paused):
        return None

    def simSetWeatherParameter(self, param, val):
        return None

    def simEnableWeather(self, enable):
        return None

    # -- state -------------------------------------------------------------

    def _pose(self, vehicle_name):
        """A slow circle at 5 m altitude, in the vehicle's own LOCAL frame.

        Deliberately LOCAL and not offset by spawn: real AirSim reports
        multi-vehicle positions in each vehicle's own reference, which is exactly
        the quirk swarm_comms.SwarmPositions calibrates away.
        """
        t = time.time() - T0
        r, w = 10.0, 0.2
        return (r * math.cos(w * t), r * math.sin(w * t), -5.0), w * t

    def simGetVehiclePose(self, vehicle_name):
        (x, y, z), yaw = self._pose(vehicle_name)
        return {"position": _vec(x, y, z),
                "orientation": _quat(z=math.sin(yaw / 2), w=math.cos(yaw / 2))}

    def simGetGroundTruthKinematics(self, vehicle_name):
        (x, y, z), yaw = self._pose(vehicle_name)
        return {
            "position": _vec(x, y, z),
            "orientation": _quat(z=math.sin(yaw / 2), w=math.cos(yaw / 2)),
            "linear_velocity": _vec(1.0, 0.0, 0.0),
            "angular_velocity": _vec(0.0, 0.0, 0.2),
            "linear_acceleration": _vec(0.0, 0.2, 0.0),
            "angular_acceleration": _vec(),
        }

    def getMultirotorState(self, vehicle_name):
        return {
            "collision": {"has_collided": False},
            "kinematics_estimated": self.simGetGroundTruthKinematics(vehicle_name),
            "timestamp": _sim_ns(),
            "landed_state": 1,
            "rc_data": {"timestamp": 0},
            "ready": True,
            "ready_message": "",
            "can_arm": True,
        }

    # -- sensors -----------------------------------------------------------

    def getImuData(self, imu_name, vehicle_name):
        (x, y, z), yaw = self._pose(vehicle_name)
        return {
            "time_stamp": _sim_ns(),
            "orientation": _quat(z=math.sin(yaw / 2), w=math.cos(yaw / 2)),
            "angular_velocity": _vec(0.0, 0.0, 0.2),
            "linear_acceleration": _vec(0.0, 0.2, 9.80665),
        }

    def getGpsData(self, gps_name, vehicle_name):
        return {
            "time_stamp": _sim_ns(),
            "gnss": {
                "time_utc": int(time.time()),
                "geo_point": {"latitude": 47.641468, "longitude": -122.140165,
                              "altitude": 122.0},
                "eph": 0.3, "epv": 0.4,
                "velocity": _vec(1.0, 0.0, 0.0),
                "fix_type": 3,
            },
        }

    def getLidarData(self, lidar_name, vehicle_name):
        """A ring of returns in the SENSOR frame, as a flat [x,y,z,x,y,z,...] list.

        Flat-and-float is how AirSim ships point clouds, and it is the one field
        the client must NOT walk element-by-element when wrapping responses.

        The sweep is built once and reused: it is static in the sensor frame, and
        at POINTS_PER_SWEEP=16k regenerating it per call would make the *mock* the
        bottleneck rather than whatever is under test.
        """
        (x, y, z), yaw = self._pose(vehicle_name)
        return {
            "time_stamp": _sim_ns(),
            "point_cloud": _sweep(),
            "pose": {"position": _vec(x, y, z - 0.1),
                     "orientation": _quat(z=math.sin(yaw / 2), w=math.cos(yaw / 2))},
            "segmentation": [],
        }

    def simGetImages(self, requests, vehicle_name, external):
        """One synthetic BGR frame per request, as raw bytes.

        AirSim sends image payloads as msgpack *bin*; this is the case that
        proves the client's ``raw=False`` unpacking hands back ``bytes`` rather
        than trying to utf-8 decode them.
        """
        out = []
        w, h = 64, 36
        for i, _req in enumerate(requests):
            payload = bytes(bytearray((j + i * 7) % 256 for j in range(w * h * 3)))
            out.append({
                "image_data_uint8": payload,
                "image_data_float": [],
                "camera_position": _vec(0.3, -0.125, 0.3),
                "camera_orientation": _quat(),
                "time_stamp": _sim_ns(),
                "message": "",
                "pixels_as_float": False,
                "compress": False,
                "width": w,
                "height": h,
                "image_type": 0,
            })
        return out

    # -- movement (return immediately; the bridge only needs them to succeed) --

    def takeoff(self, timeout_sec, vehicle_name):
        return True

    def land(self, timeout_sec, vehicle_name):
        return True

    def hover(self, vehicle_name):
        return True

    def moveOnPath(self, path, velocity, timeout_sec, drivetrain, yaw_mode,
                   lookahead, adaptive_lookahead, vehicle_name):
        # Echoing the length back lets the test assert that a list of Vector3r
        # survived the _unwrap() -> dict conversion intact.
        return len(path)

    def moveToPosition(self, x, y, z, velocity, timeout_sec, drivetrain, yaw_mode,
                       lookahead, adaptive_lookahead, vehicle_name):
        return True

    def moveToZ(self, z, velocity, timeout_sec, yaw_mode, lookahead,
                adaptive_lookahead, vehicle_name):
        return True

    def moveByVelocityZ(self, vx, vy, z, duration, drivetrain, yaw_mode, vehicle_name):
        return True

    def moveByVelocityZBodyFrame(self, vx, vy, z, duration, drivetrain, yaw_mode,
                                 vehicle_name):
        return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=41999,
                    help="deliberately NOT 41451, so a real sim can run alongside")
    ap.add_argument("--clock-rate", type=float, default=1.0,
                    help="sim clock speed relative to wall time. Real AirSim runs "
                         "well under 1.0 with four drones loaded; 1.0 (the default) "
                         "hides every clock-consistency bug in the bridge.")
    ap.add_argument("--clock-skew", type=float, default=0.0,
                    help="seconds to offset the sim clock from wall time at startup")
    ap.add_argument("--points-per-sweep", type=int, default=360,
                    help="LiDAR returns per sweep. Real AirSim gives ~16000.")
    args = ap.parse_args()

    global CLOCK_RATE, CLOCK_SKEW, POINTS_PER_SWEEP
    CLOCK_RATE = args.clock_rate
    CLOCK_SKEW = args.clock_skew
    POINTS_PER_SWEEP = args.points_per_sweep

    server = msgpackrpc.Server(MockAirSim(), pack_encoding="utf-8",
                               unpack_encoding="utf-8")
    server.listen(msgpackrpc.Address(args.host, args.port))
    print(f"mock AirSim RPC on {args.host}:{args.port} -- ctrl-c to stop\n"
          f"  clock: rate {CLOCK_RATE}x, skew {CLOCK_SKEW:+g}s | "
          f"lidar: {len(_sweep()) // 3} points/sweep", flush=True)
    server.start()


if __name__ == "__main__":
    main()
