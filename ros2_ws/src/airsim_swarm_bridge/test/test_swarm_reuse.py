#!/usr/bin/env python3
"""The existing swarm/ modules must run unmodified against the RPC shim.

This is the load-bearing claim of the whole design: `maneuver_node` does not
reimplement any manoeuvre, it imports `swarm/swarm_edge_to_center.py` and calls
its own `run(client)`. That only works if `airsim_api` is a faithful enough
stand-in for `airsim.MultirotorClient` -- so it is checked directly, against the
reference-protocol mock, rather than assumed.

    ./scripts/ros_enter.sh python3 ros2_ws/src/airsim_swarm_bridge/test/test_swarm_reuse.py
"""
from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from airsim_swarm_bridge import airsim_api            # noqa: E402

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
    # This is exactly what maneuver_node does before importing anything from swarm/.
    airsim_api.install_as_airsim()
    check("shim registered as 'airsim'", sys.modules.get("airsim") is airsim_api)

    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    sys.path.insert(0, os.path.join(repo, "swarm"))

    import swarm_comms                                  # noqa: PLC0415
    import swarm_converge                               # noqa: PLC0415
    import swarm_edge_to_center                         # noqa: PLC0415

    check("swarm_comms imports", hasattr(swarm_comms, "SwarmPositions"))
    check("swarm_edge_to_center exposes run()", callable(swarm_edge_to_center.run))
    check("swarm_converge exposes run()", callable(swarm_converge.run))

    client = airsim_api.MultirotorClient("127.0.0.1", PORT)
    client.confirmConnection(15.0)

    swarm = swarm_comms.SwarmPositions(client)
    positions = swarm.refresh()
    check("SwarmPositions polls every drone", len(positions) == 4, f"got {len(positions)}")

    # The offset calibration must place each drone back on its declared spawn --
    # that is the whole point of it (AirSim reports positions in each vehicle's
    # own local reference, not the world).
    #
    # Tolerance, not equality: the mock drone orbits continuously, so it moves a
    # millimetre or two between SwarmPositions' baseline snapshot and this
    # refresh. A metre of slack still catches the failure that matters -- a
    # broken calibration puts the drone 150+ m out, at the raw local reading.
    worst = max(
        max(abs(positions[d].world_x - sx), abs(positions[d].world_y - sy))
        for d, (sx, sy) in swarm_comms.SPAWNS.items()
    )
    check("world positions land on the declared spawns", worst < 1.0,
          f"worst axis error {worst:.3e} m")

    # Drone1 (150,150) to Drone3 (-150,150) is exactly 300 m apart.
    check("pairwise distance is right",
          abs(swarm.distance("Drone1", "Drone3") - 300.0) < 1e-3,
          f"got {swarm.distance('Drone1', 'Drone3')}")

    # And the real manoeuvre runs end to end against the shim.
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            swarm_edge_to_center.run(client)
        ran = True
    except Exception as exc:                            # noqa: BLE001
        ran = False
        print(f"       run() raised: {exc!r}")
    out = buf.getvalue()
    check("swarm_edge_to_center.run() completes", ran)
    check("it computed a perimeter order", "Perimeter order:" in out)
    check("it computed the diagonal intersection", "Diagonal intersection" in out)

    client.close()
    print(f"\n{_passed} passed, {len(_failed)} failed")
    if _failed:
        print("failed: " + ", ".join(_failed))
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
