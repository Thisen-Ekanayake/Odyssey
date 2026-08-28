#!/usr/bin/env python3
"""Wire-compatibility check for the tornado-free RPC client.

Run the reference server on the host venv (Python 3.10 + msgpack-rpc-python +
tornado), then run this in the ROS container (Python 3.12 + plain msgpack):

    ./airsim_venv/bin/python ros2_ws/src/airsim_swarm_bridge/test/mock_airsim_server.py &
    ./scripts/ros_enter.sh python3 ros2_ws/src/airsim_swarm_bridge/test/test_rpc_roundtrip.py

What it is actually asserting is that ``msgpack.Packer(use_bin_type=False)`` /
``Unpacker(raw=False, strict_map_key=False)`` reproduce msgpack 0.5's
``encoding='utf-8'`` behaviour on both directions of every shape AirSim uses:
nested maps, flat float arrays, binary blobs, and dict-ified parameter objects.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from airsim_swarm_bridge import airsim_api as airsim   # noqa: E402
from airsim_swarm_bridge.rpc import AirSimRpc          # noqa: E402

PORT = int(os.environ.get("MOCK_PORT", "41999"))

_passed = 0
_failed: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global _passed
    if cond:
        _passed += 1
        print(f"  ok   {name}")
    else:
        _failed.append(name)
        print(f"  FAIL {name}  {detail}")


def main() -> int:
    print(f"python {sys.version.split()[0]}  ->  mock server on 127.0.0.1:{PORT}\n")

    # --- raw transport ----------------------------------------------------
    print("transport")
    rpc = AirSimRpc("127.0.0.1", PORT, timeout=10.0)
    rpc.confirm_connection(15.0, quiet=True)
    check("ping", rpc.call("ping") is True)
    check("getServerVersion", rpc.call("getServerVersion") == 1)
    names = rpc.call("listVehicles")
    check("listVehicles decodes to str", names == ["Drone1", "Drone2", "Drone3", "Drone4"],
          f"got {names!r}")
    check("str came back as str not bytes", all(isinstance(n, str) for n in names))

    # Four requests in flight on one connection, joined out of order -- the
    # pattern the maneuver node uses to command the whole swarm at once.
    futures = [rpc.call_async("getImuData", "Imu", n) for n in names]
    joined = [f.join() for f in reversed(futures)]
    check("4 concurrent calls, joined in reverse", len(joined) == 4 and
          all(isinstance(r, dict) and "time_stamp" in r for r in joined))
    rpc.close()

    # --- the airsim_api shim ---------------------------------------------
    print("\nairsim_api shim")
    c = airsim.MultirotorClient("127.0.0.1", PORT)
    c.confirmConnection(15.0)

    kin = c.simGetGroundTruthKinematics("Drone1")
    check("nested attribute access", hasattr(kin.position, "x_val"),
          f"got {kin!r}")
    check("floats survive", isinstance(kin.position.x_val, float))
    check("quaternion is w-first on the wire",
          abs(kin.orientation.w_val) <= 1.0 and hasattr(kin.orientation, "x_val"))

    st = c.getMultirotorState("Drone1")
    check("two-level nesting (kinematics_estimated.position)",
          isinstance(st.kinematics_estimated.position.z_val, float))
    check("bool field", st.collision.has_collided is False)

    gps = c.getGpsData("", "Drone1")
    check("three-level nesting (gnss.geo_point.latitude)",
          abs(gps.gnss.geo_point.latitude - 47.641468) < 1e-6)

    lid = c.getLidarData("LidarSensor1", "Drone1")
    check("point_cloud stays a flat list", isinstance(lid.point_cloud, list))
    # The wire property is that a flat float list survives as triples -- not that
    # the mock happens to emit 360 returns. It is configurable now
    # (--points-per-sweep), so asserting the count would just couple this to the
    # mock's geometry.
    check("point_cloud length is a multiple of 3",
          len(lid.point_cloud) > 0 and len(lid.point_cloud) % 3 == 0,
          f"got {len(lid.point_cloud)}")
    check("point_cloud elements are numbers, NOT Structs",
          all(isinstance(v, float) for v in lid.point_cloud[:9]))
    check("lidar pose nested", isinstance(lid.pose.position.z_val, float))

    imgs = c.simGetImages([
        airsim.ImageRequest("StereoLeft", airsim.ImageType.Scene, False, False),
        airsim.ImageRequest("StereoRight", airsim.ImageType.Scene, False, False),
    ], "Drone1")
    check("simGetImages returns 2 responses", len(imgs) == 2, f"got {len(imgs)}")
    check("image payload is bytes, not str",
          isinstance(imgs[0].image_data_uint8, (bytes, bytearray)),
          f"got {type(imgs[0].image_data_uint8).__name__}")
    check("image payload length == w*h*3",
          len(imgs[0].image_data_uint8) == imgs[0].width * imgs[0].height * 3)
    check("the two eyes differ", bytes(imgs[0].image_data_uint8) != bytes(imgs[1].image_data_uint8))

    # Parameter objects must arrive as plain maps -- this is _unwrap()'s job.
    n = c.moveOnPathAsync(
        [airsim.Vector3r(0, 0, -5), airsim.Vector3r(10, 0, -5), airsim.Vector3r(10, 10, -5)],
        velocity=3.0, yaw_mode=airsim.YawMode(False, 0.0), vehicle_name="Drone1",
    ).join()
    check("list[Vector3r] param survives as 3 waypoints", n == 3, f"server saw {n}")
    check("takeoff", c.takeoffAsync(20, "Drone1").join() is True)
    check("enableApiControl (None-returning call)",
          c.enableApiControl(True, "Drone1") is None)

    # A missing field must raise AttributeError with a useful message rather
    # than silently returning None deep inside a publisher.
    try:
        _ = kin.no_such_field
        check("unknown field raises", False, "no exception")
    except AttributeError as exc:
        check("unknown field raises AttributeError", "no_such_field" in str(exc))

    c.close()

    print(f"\n{_passed} passed, {len(_failed)} failed")
    if _failed:
        print("failed: " + ", ".join(_failed))
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
