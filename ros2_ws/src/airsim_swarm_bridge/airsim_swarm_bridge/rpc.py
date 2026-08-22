"""Minimal msgpack-rpc client for AirSim -- no tornado, no ``airsim`` pip package.

Why this file exists
--------------------
The ROS container is Ubuntu 24.04 / **Python 3.12**. ``pip install airsim`` pulls
``msgpack-rpc-python``, which hard-pins ``tornado<5``, and tornado 4.5.3 is dead on
3.12: ``tornado/httputil.py`` subclasses ``collections.MutableMapping`` (removed in
3.10) and ``tornado/util.py`` calls ``inspect.getargspec`` (removed in 3.11). The
host venv only survives because it is Python 3.10.

AirSim's server is plain `msgpack-rpc <https://github.com/msgpack-rpc/msgpack-rpc>`_
over TCP, so the whole dependency is replaceable by a socket and ``msgpack``:

    request   [0, msgid, method, params]
    response  [1, msgid, error, result]

The wire settings below are copied from the working host client rather than guessed:

* ``airsim/client.py`` builds ``msgpackrpc.Client(..., pack_encoding='utf-8',
  unpack_encoding='utf-8')``
* ``msgpackrpc/transport/tcp.py`` turns that into
  ``msgpack.Packer(encoding='utf-8', default=lambda x: x.to_msgpack())`` and
  ``msgpack.Unpacker(encoding='utf-8')``

Under msgpack 0.5.x, ``encoding=`` implied ``use_bin_type=False`` on the packer (str
goes out as the old ``raw`` type, which is what AirSim's rpclib server expects) and
utf-8 decoding of ``raw`` on the unpacker. The msgpack 1.x spellings of exactly that
are ``Packer(use_bin_type=False)`` and ``Unpacker(raw=False)``. ``strict_map_key=False``
is additionally required because msgpack 1.x rejects non-string map keys by default
and AirSim returns a few integer-keyed maps.

One wrinkle that only shows up on binary payloads: whether a ``bytes`` value
arrives as msgpack ``bin`` or as the older ``raw`` type depends on the *server*.
AirSim's C++ rpclib sends image buffers as ``bin`` (which is why the host code can
do ``np.frombuffer(resp.image_data_uint8, np.uint8)`` directly), but a
msgpack-0.5-based server packing with ``use_bin_type=False`` sends the same bytes
as ``raw`` -- and ``raw=False`` would then try to utf-8 decode a JPEG. Hence
``unicode_errors="surrogateescape"``: undecodable ``raw`` bytes round-trip
losslessly into a ``str`` instead of raising, and
:func:`airsim_api.to_uint8_array` normalises either shape back to ``bytes``.

``MsgpackMixin.to_msgpack`` in the reference client just returns ``self.__dict__``, so
every "type" AirSim accepts as a parameter is a plain dict on the wire. That is why
this module never needs the ``airsim`` type zoo -- see :mod:`airsim_api` for the thin
attribute-access shim built on top.
"""
from __future__ import annotations

import argparse
import socket
import threading
import time

import msgpack

__all__ = ["RpcError", "AirSimRpc", "Future"]

REQUEST = 0
RESPONSE = 1
NOTIFY = 2

DEFAULT_IP = "127.0.0.1"
DEFAULT_PORT = 41451


class RpcError(RuntimeError):
    """The server returned an error for a call, or the transport died."""


class Future:
    """Handle for a call that was sent but not yet waited on.

    AirSim's movement APIs (``takeoff``, ``moveOnPath``, ...) block server-side
    until the manoeuvre finishes, so commanding four drones in parallel means four
    outstanding requests on one connection. ``join()`` blocks until this specific
    ``msgid`` comes back; responses for other ids that arrive first are stashed by
    the connection, not dropped.
    """

    __slots__ = ("_conn", "_msgid", "_done", "_result")

    def __init__(self, conn: "AirSimRpc", msgid: int) -> None:
        self._conn = conn
        self._msgid = msgid
        self._done = False
        self._result = None

    def join(self, timeout: float | None = None):
        if not self._done:
            self._result = self._conn._await(self._msgid, timeout)
            self._done = True
        return self._result


class AirSimRpc:
    """One TCP connection to an AirSim RPC server.

    **One connection per thread.** The reference client inherits the same rule from
    msgpack-rpc's Tornado IOLoop (noted in this repo's CLAUDE.md), and the ROS bridge
    keeps to it by giving every drone its own node and its own connection. A lock is
    held across send/receive anyway so a stray shared use degrades to serialisation
    rather than a corrupted stream.
    """

    def __init__(self, ip: str = DEFAULT_IP, port: int = DEFAULT_PORT,
                 timeout: float = 30.0) -> None:
        self.ip = ip
        self.port = port
        self.timeout = timeout

        self._sock: socket.socket | None = None
        self._unpacker = msgpack.Unpacker(raw=False, strict_map_key=False,
                                          unicode_errors="surrogateescape")
        self._packer = msgpack.Packer(use_bin_type=False)
        self._msgid = 0
        self._pending: dict[int, tuple[object, object]] = {}   # msgid -> (error, result)
        self._lock = threading.RLock()

    # -- connection ---------------------------------------------------------

    def connect(self) -> "AirSimRpc":
        with self._lock:
            if self._sock is not None:
                return self
            s = socket.create_connection((self.ip, self.port), timeout=self.timeout)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._sock = s
            # A fresh stream needs a fresh unpacker: any half-message left over
            # from a dead connection would corrupt the first reply on the new one.
            self._unpacker = msgpack.Unpacker(raw=False, strict_map_key=False,
                                          unicode_errors="surrogateescape")
            self._pending.clear()
        return self

    def close(self) -> None:
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.close()
                finally:
                    self._sock = None

    def __enter__(self) -> "AirSimRpc":
        return self.connect()

    def __exit__(self, *_exc) -> None:
        self.close()

    def confirm_connection(self, retry_seconds: float = 60.0,
                           quiet: bool = False) -> None:
        """Block until the server answers ``ping``.

        First boot compiles UE4 shaders, so the process can be up for 30-60 s
        before the RPC server accepts anything -- retrying is the normal path
        here, not an error case.
        """
        deadline = time.time() + retry_seconds
        last: Exception | None = None
        while time.time() < deadline:
            try:
                self.connect()
                if self.call("ping"):
                    if not quiet:
                        print(f"connected to AirSim at {self.ip}:{self.port}")
                    return
            except (OSError, RpcError) as exc:
                last = exc
                self.close()
                time.sleep(1.0)
        raise RpcError(
            f"no AirSim RPC server at {self.ip}:{self.port} after {retry_seconds:.0f}s "
            f"(last error: {last}). Is ./scripts/run_swarm.sh running?"
        )

    # -- calls --------------------------------------------------------------

    def call(self, method: str, *params, timeout: float | None = None):
        """Send a request and block for its reply."""
        return self.call_async(method, *params).join(timeout)

    def call_async(self, method: str, *params) -> Future:
        """Send a request and return immediately; ``.join()`` collects the reply."""
        with self._lock:
            if self._sock is None:
                self.connect()
            self._msgid = (self._msgid + 1) & 0xFFFFFFFF
            msgid = self._msgid
            payload = self._packer.pack([REQUEST, msgid, method, list(params)])
            try:
                self._sock.sendall(payload)                       # type: ignore[union-attr]
            except OSError as exc:
                self.close()
                raise RpcError(f"send failed for {method!r}: {exc}") from exc
        return Future(self, msgid)

    def _await(self, msgid: int, timeout: float | None):
        """Read until ``msgid``'s response shows up, stashing any others."""
        deadline = time.time() + (self.timeout if timeout is None else timeout)
        with self._lock:
            while True:
                if msgid in self._pending:
                    error, result = self._pending.pop(msgid)
                    if error is not None:
                        raise RpcError(f"AirSim returned an error: {error!r}")
                    return result
                if time.time() > deadline:
                    raise RpcError(f"timed out waiting for RPC response #{msgid}")
                self._pump(deadline)

    def _pump(self, deadline: float) -> None:
        """Read one chunk off the socket and file away every complete message."""
        if self._sock is None:
            raise RpcError("not connected")
        self._sock.settimeout(max(0.05, deadline - time.time()))
        try:
            chunk = self._sock.recv(65536)
        except socket.timeout:
            return
        except OSError as exc:
            self.close()
            raise RpcError(f"receive failed: {exc}") from exc
        if not chunk:
            self.close()
            raise RpcError("AirSim closed the connection")

        self._unpacker.feed(chunk)
        for msg in self._unpacker:
            if not isinstance(msg, (list, tuple)) or not msg:
                continue
            if msg[0] == RESPONSE and len(msg) >= 4:
                _, mid, error, result = msg[0], msg[1], msg[2], msg[3]
                self._pending[mid] = (error, result)
            # NOTIFY messages are not part of AirSim's protocol; ignore anything else.


# --------------------------------------------------------------------------
# self-test: proves the transport works before any ROS code is involved
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Smoke-test the tornado-free AirSim RPC client.")
    ap.add_argument("--ip", default=DEFAULT_IP)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--ping", action="store_true", help="ping + server version")
    ap.add_argument("--list-vehicles", action="store_true", help="list vehicle names")
    ap.add_argument("--sensors", metavar="VEHICLE", help="one IMU/LiDAR/GPS readout")
    ap.add_argument("--timeout", type=float, default=60.0,
                    help="seconds to wait for the server to come up")
    args = ap.parse_args(argv)

    if not (args.ping or args.list_vehicles or args.sensors):
        args.ping = args.list_vehicles = True

    rpc = AirSimRpc(args.ip, args.port)
    try:
        rpc.confirm_connection(args.timeout)
    except RpcError as exc:
        print(f"FAIL: {exc}")
        return 1

    ok = True
    if args.ping:
        print(f"  ping            -> {rpc.call('ping')}")
        print(f"  serverVersion   -> {rpc.call('getServerVersion')}")
        print(f"  clientVersion   -> {rpc.call('getMinRequiredClientVersion')}")

    if args.list_vehicles:
        names = rpc.call("listVehicles")
        print(f"  listVehicles    -> {names}")
        ok = ok and bool(names)

    if args.sensors:
        v = args.sensors
        imu = rpc.call("getImuData", "Imu", v)
        gps = rpc.call("getGpsData", "", v)
        lidar = rpc.call("getLidarData", "LidarSensor1", v)
        kin = rpc.call("simGetGroundTruthKinematics", v)
        n_pts = len(lidar.get("point_cloud", [])) // 3
        print(f"  {v} imu t       -> {imu['time_stamp']}")
        print(f"  {v} imu gyro    -> {imu['angular_velocity']}")
        print(f"  {v} gps         -> {gps['gnss']['geo_point']}")
        print(f"  {v} lidar pts   -> {n_pts}")
        print(f"  {v} gt position -> {kin['position']}")
        # A LiDAR that returns nothing usually means the drone is still on the
        # ground inside geometry, or the sensor name does not match settings.json.
        if n_pts == 0:
            print("  NOTE: zero LiDAR points -- check LidarSensor1 exists in settings.json")

    rpc.close()
    print("OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
