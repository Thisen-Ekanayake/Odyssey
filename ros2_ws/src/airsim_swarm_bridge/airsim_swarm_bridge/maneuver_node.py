#!/usr/bin/env python3
"""Exposes the existing ``swarm/`` manoeuvres as ROS 2 services.

    /swarm/takeoff          /swarm/land          /swarm/hover
    /swarm/edge_to_center   /swarm/converge

The flight code is **not** reimplemented. ``swarm/swarm_edge_to_center.py`` and
``swarm/swarm_converge.py`` each expose a ``run(client)`` entry point precisely so
another process can drive them, and this node calls those -- with the
tornado-free client shimmed in as ``airsim`` (see
:func:`airsim_api.install_as_airsim`). ROS and the standalone scripts therefore
always fly exactly the same manoeuvre, and a fix to either lands in both.

This node owns its own RPC connection, separate from every ``bridge_node``: the
manoeuvres block for minutes at a time inside ``moveOnPathAsync``, and sharing a
socket with a 100 Hz sensor timer would stall the sensor stream.

A manoeuvre runs on a background thread and the service returns immediately when
``background: true`` (the default for the long ones), because ``edge_to_center``
takes several minutes and most service clients will not wait that long. Progress
goes to the node's log either way.
"""
from __future__ import annotations

import io
import threading
from contextlib import redirect_stdout

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node

from airsim_swarm_msgs.srv import Maneuver

from .airsim_api import MultirotorClient
from .rpc import RpcError

# Manoeuvres that are inherently long-running; `background` defaults to true for
# these unless the caller explicitly asks to wait.
LONG_RUNNING = {"edge_to_center", "converge"}


class ManeuverNode(Node):

    def __init__(self) -> None:
        super().__init__("swarm_maneuver")

        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 41451)
        self.declare_parameter("drones", [""])
        # Refuse manoeuvres until every drone's bridge is publishing -- see
        # _bridges_ready. Set False when running this node without bridges: the
        # manoeuvres themselves do their own calibration through
        # swarm_comms.SwarmPositions and do not need one, it is the shared MAP
        # that does.
        self.declare_parameter("require_bridges", True)

        g = lambda n: self.get_parameter(n).value            # noqa: E731

        from .repo import import_swarm_module

        self._comms = import_swarm_module("swarm_comms")
        self._edge = import_swarm_module("swarm_edge_to_center")
        self._converge = import_swarm_module("swarm_converge")

        drones = [d for d in (g("drones") or []) if d]
        self.drones: list[str] = drones or list(self._comms.DRONES)
        self.require_bridges: bool = g("require_bridges")

        self.client = MultirotorClient(g("host"), int(g("port")))
        self.client.confirmConnection(120.0)
        self.get_logger().info(f"maneuver services ready for {', '.join(self.drones)}")

        # A single worker slot: two manoeuvres at once would fight over the same
        # vehicles, so a second request is refused rather than queued.
        self._busy = threading.Lock()
        self._current: str | None = None

        cb = ReentrantCallbackGroup()
        self._srv = {
            name: self.create_service(
                Maneuver, f"/swarm/{name}",
                lambda req, resp, n=name: self._handle(n, req, resp),
                callback_group=cb)
            for name in ("takeoff", "land", "hover", "edge_to_center", "converge")
        }

    # -- readiness ---------------------------------------------------------

    def _bridges_ready(self) -> tuple[bool, str]:
        """Refuse to move anything until every drone's bridge has come up.

        ``AirSimBridge._calibrate_offset`` snapshots each vehicle's raw local
        position at startup and treats it as that run's zero -- which is only
        correct if the drone has not moved yet. A manoeuvre that starts while one
        bridge is still inside ``confirmConnection`` therefore gives that drone a
        world offset measured from a position it has already left, and every cloud
        it contributes lands in the wrong place on the shared map. Nothing errors;
        the map is just quietly wrong for one of four drones.

        The race is real but narrow, which is what makes it worth a guard rather
        than a longer delay: ``cooperative_mapping.launch.py`` fires its
        ``auto_maneuver`` on a fixed 8 s timer while a cold UE4 start can leave a
        bridge connecting for far longer.

        A bridge publishing ``odom_gt`` has necessarily finished calibrating --
        the offset is computed in ``__init__``, before any publisher exists.

        Only the shared map needs this. The manoeuvres themselves calibrate via
        ``swarm_comms.SwarmPositions`` and are correct without any bridge running,
        so ``require_bridges:=false`` restores the standalone behaviour.
        """
        if not self.require_bridges:
            return True, ""
        missing = [d for d in self.drones
                   if self.count_publishers(f"/{d.lower()}/odom_gt") == 0]
        if not missing:
            return True, ""
        return False, (
            f"bridges not ready for {', '.join(missing)} (no odom_gt publisher). "
            f"They calibrate their world offset from a stationary pose at startup, "
            f"so moving now would misplace them on the shared map. Retry once "
            f"bridge.launch.py reports all drones connected.")

    # -- dispatch ----------------------------------------------------------

    def _handle(self, name: str, req: Maneuver.Request, resp: Maneuver.Response):
        # The service name selects the manoeuvre; req.name is an optional override
        # so a generic client can hit one endpoint and pass the rest as data.
        what = (req.name or name).strip()
        if what not in self._srv:
            resp.success = False
            resp.message = f"unknown maneuver {what!r}; have {sorted(self._srv)}"
            return resp

        if not self._busy.acquire(blocking=False):
            resp.success = False
            resp.message = f"busy running {self._current!r}"
            return resp

        ready, why = self._bridges_ready()
        if not ready:
            self._busy.release()
            resp.success = False
            resp.message = why
            return resp

        background = req.background or what in LONG_RUNNING
        self._current = what

        if background:
            threading.Thread(target=self._run_and_release, args=(what, req),
                             name=f"maneuver-{what}", daemon=True).start()
            resp.success = True
            resp.message = (f"{what} started in the background; "
                            f"watch this node's log or /swarm/state for progress")
            return resp

        ok, message = self._run_and_release(what, req)
        resp.success = ok
        resp.message = message
        return resp

    def _run_and_release(self, what: str, req: Maneuver.Request):
        try:
            self.get_logger().info(f"maneuver {what}: start")
            ok, message = self._run(what, req)
            self.get_logger().info(f"maneuver {what}: {message}")
            return ok, message
        except (RpcError, OSError) as exc:
            self.get_logger().error(f"maneuver {what} failed: {exc}")
            return False, str(exc)
        except Exception as exc:                            # noqa: BLE001
            self.get_logger().error(f"maneuver {what} raised: {exc!r}")
            return False, repr(exc)
        finally:
            self._current = None
            self._busy.release()

    def _run(self, what: str, req: Maneuver.Request) -> tuple[bool, str]:
        if what == "takeoff":
            return self._takeoff(req)
        if what == "land":
            return self._land()
        if what == "hover":
            for f in [self.client.hoverAsync(vehicle_name=d) for d in self.drones]:
                f.join()
            return True, "hovering"
        # The two imported manoeuvres print progress to stdout; capture it so it
        # lands in the ROS log instead of a terminal nobody is watching.
        module = self._edge if what == "edge_to_center" else self._converge
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                module.run(self.client)
        finally:
            for line in buf.getvalue().splitlines():
                if line.strip():
                    self.get_logger().info(f"[{what}] {line}")
        return True, f"{what} complete"

    def _takeoff(self, req: Maneuver.Request) -> tuple[bool, str]:
        for d in self.drones:
            self.client.enableApiControl(True, d)
            self.client.armDisarm(True, d)
        for f in [self.client.takeoffAsync(vehicle_name=d) for d in self.drones]:
            f.join()

        # Altitude is given as positive-up (the ROS/user convention); AirSim's z
        # is NED, so it has to be negated on the way in.
        alt = float(req.altitude)
        if alt > 0:
            speed = float(req.speed) if req.speed > 0 else 5.0
            for f in [self.client.moveToZAsync(-alt, speed, vehicle_name=d)
                      for d in self.drones]:
                f.join()
            return True, f"airborne at {alt:.1f} m"
        return True, "airborne"

    def _land(self) -> tuple[bool, str]:
        for f in [self.client.landAsync(vehicle_name=d) for d in self.drones]:
            f.join()
        for d in self.drones:
            self.client.armDisarm(False, d)
            self.client.enableApiControl(False, d)
        return True, "landed and disarmed"

    def destroy_node(self) -> bool:
        try:
            self.client.close()
        except Exception:                                   # noqa: BLE001
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = ManeuverNode()
        # Multi-threaded: a manoeuvre thread must not block the service callbacks,
        # otherwise /swarm/land could not interrupt a running edge_to_center.
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException, SystemExit):
        pass
    except RpcError as exc:
        print(f"swarm_maneuver: {exc}")
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
