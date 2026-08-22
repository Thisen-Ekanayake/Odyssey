"""A drop-in subset of ``airsim.MultirotorClient`` built on :mod:`rpc`.

The point is not to re-implement AirSim's client -- it is to be *interface
compatible* with the parts of it this repo actually uses, so that the existing
manoeuvre code runs unchanged inside the ROS container:

    swarm/swarm_edge_to_center.py :: run(client)
    swarm/swarm_converge.py       :: run(client)
    swarm/swarm_comms.py          :: SwarmPositions(client)

Those modules do ``import airsim`` at the top and then use ``airsim.Vector3r``,
``airsim.YawMode``, ``client.moveOnPathAsync(...)`` and so on. Rather than fork
them, :mod:`maneuver_node` installs this module *as* ``airsim`` in
``sys.modules`` (see :func:`install_as_airsim`) before importing them. That keeps
one copy of every manoeuvre, on the host and in ROS alike.

Two things make the shim small:

* ``MsgpackMixin.to_msgpack`` in the real client just returns ``self.__dict__``,
  so every parameter type is a plain dict on the wire -- :class:`_Msgpackable`
  reproduces that in a few lines.
* Responses are plain maps, so :class:`Struct` turns them into attribute objects
  recursively and ``.position.x_val`` keeps working.

Only the calls this repo makes are wrapped. Anything missing is a one-liner
against :meth:`MultirotorClient.call`; do not paper over a gap with ``__getattr__``,
because a typo'd method name should fail loudly here rather than at the server.
"""
from __future__ import annotations

import math
import sys
import types as _types

from .rpc import AirSimRpc, Future, RpcError

__all__ = [
    "MultirotorClient", "VehicleClient", "Struct",
    "Vector3r", "Quaternionr", "Pose", "YawMode", "DrivetrainType",
    "ImageType", "ImageRequest", "WeatherParameter", "LandedState",
    "to_quaternion", "to_eularian_angles", "to_uint8_array", "install_as_airsim",
    "RpcError", "Future",
]


# --------------------------------------------------------------------------
# parameter types  (packed as plain dicts, exactly like MsgpackMixin)
# --------------------------------------------------------------------------

class _Msgpackable:
    def to_msgpack(self, *_a, **_k) -> dict:
        return self.__dict__

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.__dict__}>"


class Vector3r(_Msgpackable):
    def __init__(self, x_val: float = 0.0, y_val: float = 0.0, z_val: float = 0.0):
        self.x_val = float(x_val)
        self.y_val = float(y_val)
        self.z_val = float(z_val)


class Quaternionr(_Msgpackable):
    # NOTE the argument order: x, y, z, w -- w LAST in the constructor but stored
    # w-first on the wire. Matches airsim.types.Quaternionr; getting this wrong
    # silently rotates everything.
    def __init__(self, x_val: float = 0.0, y_val: float = 0.0,
                 z_val: float = 0.0, w_val: float = 1.0):
        self.w_val = float(w_val)
        self.x_val = float(x_val)
        self.y_val = float(y_val)
        self.z_val = float(z_val)


class Pose(_Msgpackable):
    def __init__(self, position_val: Vector3r | None = None,
                 orientation_val: Quaternionr | None = None):
        self.position = position_val if position_val is not None else Vector3r()
        self.orientation = orientation_val if orientation_val is not None else Quaternionr()

    def to_msgpack(self, *_a, **_k) -> dict:
        return {"position": self.position.to_msgpack(),
                "orientation": self.orientation.to_msgpack()}


class YawMode(_Msgpackable):
    def __init__(self, is_rate: bool = True, yaw_or_rate: float = 0.0):
        self.is_rate = bool(is_rate)
        self.yaw_or_rate = float(yaw_or_rate)


class DrivetrainType:
    MaxDegreeOfFreedom = 0
    ForwardOnly = 1


class ImageType:
    Scene = 0
    DepthPlanar = 1
    DepthPerspective = 2
    DepthVis = 3
    DisparityNormalized = 4
    Segmentation = 5
    SurfaceNormals = 6
    Infrared = 7
    OpticalFlow = 8
    OpticalFlowVis = 9


class WeatherParameter:
    Rain = 0
    Roadwetness = 1
    Snow = 2
    RoadSnow = 3
    MapleLeaf = 4
    RoadLeaf = 5
    Dust = 6
    Fog = 7
    Enabled = 8


class LandedState:
    Landed = 0
    Flying = 1


class ImageRequest(_Msgpackable):
    def __init__(self, camera_name, image_type: int = ImageType.Scene,
                 pixels_as_float: bool = False, compress: bool = True):
        self.camera_name = str(camera_name)
        self.image_type = int(image_type)
        self.pixels_as_float = bool(pixels_as_float)
        self.compress = bool(compress)


def to_quaternion(pitch: float, roll: float, yaw: float) -> Quaternionr:
    """Euler (radians) -> quaternion, matching ``airsim.utils.to_quaternion``."""
    t0, t1 = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    t2, t3 = math.cos(roll * 0.5), math.sin(roll * 0.5)
    t4, t5 = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    return Quaternionr(
        x_val=t0 * t3 * t4 - t1 * t2 * t5,
        y_val=t0 * t2 * t5 + t1 * t3 * t4,
        z_val=t1 * t2 * t4 - t0 * t3 * t5,
        w_val=t0 * t2 * t4 + t1 * t3 * t5,
    )


# --------------------------------------------------------------------------
# binary payloads
# --------------------------------------------------------------------------

def to_uint8_array(value) -> bytes:
    """Normalise an AirSim image payload to ``bytes``, whichever shape it took.

    ``image_data_uint8`` can reach us three ways, and which one depends on the
    server, not on us:

    * ``bytes`` -- AirSim's C++ rpclib packs the buffer as msgpack ``bin``;
    * ``str`` -- a msgpack-0.5-style server packs the same bytes as ``raw``, and
      the unpacker's ``surrogateescape`` turns undecodable bytes into surrogate
      code points. Re-encoding with the same error handler is exactly lossless;
    * ``list[int]`` -- msgpack-c also has an adaptor that ships
      ``std::vector<uint8_t>`` as an array of integers.

    Callers get ``bytes`` and can hand it straight to ``np.frombuffer``.
    """
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8", "surrogateescape")
    if isinstance(value, (list, tuple)):
        return bytes(bytearray(value))
    raise TypeError(f"cannot read an image payload out of {type(value).__name__}")


def to_eularian_angles(q) -> tuple[float, float, float]:
    """Quaternion -> (pitch, roll, yaw) in radians, matching ``airsim.utils``."""
    z, y, x, w = q.z_val, q.y_val, q.x_val, q.w_val
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    roll = math.atan2(t0, t1)

    t2 = max(-1.0, min(1.0, +2.0 * (w * y - z * x)))
    pitch = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    yaw = math.atan2(t3, t4)
    return pitch, roll, yaw


# --------------------------------------------------------------------------
# response type: dict -> attribute access, recursively
# --------------------------------------------------------------------------

class Struct:
    """Attribute view over an AirSim response map.

    ``client.getMultirotorState(...).kinematics_estimated.position.x_val`` has to
    keep working for the existing scripts, and every level of that is just a
    nested msgpack map. Conversion is done once, eagerly, because these objects
    are small and callers index them repeatedly in tight loops.

    Lists of maps become lists of ``Struct`` (``simGetImages`` responses), but
    lists of numbers are left as-is -- ``point_cloud`` is hundreds of thousands
    of floats and must never be walked element by element.
    """

    __slots__ = ("_d",)

    def __init__(self, data: dict):
        object.__setattr__(self, "_d", {k: _wrap(v) for k, v in data.items()})

    def __getattr__(self, name):
        try:
            return object.__getattribute__(self, "_d")[name]
        except KeyError:
            raise AttributeError(
                f"AirSim response has no field {name!r}; got "
                f"{sorted(object.__getattribute__(self, '_d'))}"
            ) from None

    def __setattr__(self, name, value):
        object.__getattribute__(self, "_d")[name] = value

    def __contains__(self, name) -> bool:
        return name in object.__getattribute__(self, "_d")

    def __getitem__(self, name):
        return object.__getattribute__(self, "_d")[name]

    def as_dict(self) -> dict:
        return dict(object.__getattribute__(self, "_d"))

    def __repr__(self) -> str:
        return f"<Struct {sorted(object.__getattribute__(self, '_d'))}>"


def _wrap(v):
    if isinstance(v, dict):
        return Struct(v)
    # Only descend into lists whose first element is a map (image responses).
    # Numeric arrays -- point_cloud above all -- are returned untouched.
    if isinstance(v, list) and v and isinstance(v[0], dict):
        return [Struct(x) for x in v]
    return v


def _unwrap(v):
    """Parameter -> msgpack-ready value (mirrors the reference packer's ``default``)."""
    if isinstance(v, _Msgpackable):
        return v.to_msgpack()
    if isinstance(v, (list, tuple)):
        return [_unwrap(x) for x in v]
    return v


# --------------------------------------------------------------------------
# the client
# --------------------------------------------------------------------------

class VehicleClient:
    """Common (non-multirotor) half of the API."""

    def __init__(self, ip: str = "127.0.0.1", port: int = 41451,
                 timeout_value: float = 3600.0):
        self.client = AirSimRpc(ip, port, timeout=timeout_value)

    # -- plumbing ----------------------------------------------------------

    def call(self, method: str, *params):
        return self.client.call(method, *[_unwrap(p) for p in params])

    def call_async(self, method: str, *params) -> Future:
        return self.client.call_async(method, *[_unwrap(p) for p in params])

    def _struct(self, method: str, *params) -> Struct:
        return Struct(self.call(method, *params))

    def confirmConnection(self, retry_seconds: float = 60.0) -> None:
        self.client.confirm_connection(retry_seconds)

    def close(self) -> None:
        self.client.close()

    def ping(self) -> bool:
        return self.call("ping")

    def reset(self) -> None:
        self.call("reset")

    def getServerVersion(self) -> int:
        return self.call("getServerVersion")

    def listVehicles(self) -> list[str]:
        return self.call("listVehicles")

    # -- sim control -------------------------------------------------------

    def simPause(self, is_paused: bool) -> None:
        self.call("simPause", is_paused)

    def simIsPaused(self) -> bool:
        return self.call("simIsPaused")

    def simContinueForTime(self, seconds: float) -> None:
        self.call("simContinueForTime", seconds)

    def simEnableWeather(self, enable: bool) -> None:
        self.call("simEnableWeather", enable)

    def simSetWeatherParameter(self, param: int, val: float) -> None:
        self.call("simSetWeatherParameter", param, val)

    # -- pose / cameras ----------------------------------------------------

    def simGetVehiclePose(self, vehicle_name: str = "") -> Struct:
        return self._struct("simGetVehiclePose", vehicle_name)

    def simGetGroundTruthKinematics(self, vehicle_name: str = "") -> Struct:
        return self._struct("simGetGroundTruthKinematics", vehicle_name)

    def simSetCameraPose(self, camera_name, pose: Pose,
                         vehicle_name: str = "", external: bool = False) -> None:
        self.call("simSetCameraPose", str(camera_name), pose, vehicle_name, external)

    def simGetImages(self, requests, vehicle_name: str = "",
                     external: bool = False) -> list[Struct]:
        raw = self.call("simGetImages", list(requests), vehicle_name, external)
        out = []
        for r in (raw or []):
            st = Struct(r)
            # Normalise here, once, so no publisher ever has to care whether the
            # payload arrived as bin, raw-with-surrogates, or a list of ints.
            if "image_data_uint8" in st:
                st.image_data_uint8 = to_uint8_array(st.image_data_uint8)
            out.append(st)
        return out

    # -- sensors -----------------------------------------------------------

    def getImuData(self, imu_name: str = "", vehicle_name: str = "") -> Struct:
        return self._struct("getImuData", imu_name, vehicle_name)

    def getGpsData(self, gps_name: str = "", vehicle_name: str = "") -> Struct:
        return self._struct("getGpsData", gps_name, vehicle_name)

    def getBarometerData(self, barometer_name: str = "", vehicle_name: str = "") -> Struct:
        return self._struct("getBarometerData", barometer_name, vehicle_name)

    def getMagnetometerData(self, magnetometer_name: str = "", vehicle_name: str = "") -> Struct:
        return self._struct("getMagnetometerData", magnetometer_name, vehicle_name)

    def getLidarData(self, lidar_name: str = "", vehicle_name: str = "") -> Struct:
        return self._struct("getLidarData", lidar_name, vehicle_name)

    # -- control -----------------------------------------------------------

    def enableApiControl(self, is_enabled: bool, vehicle_name: str = "") -> None:
        self.call("enableApiControl", is_enabled, vehicle_name)

    def isApiControlEnabled(self, vehicle_name: str = "") -> bool:
        return self.call("isApiControlEnabled", vehicle_name)

    def armDisarm(self, arm: bool, vehicle_name: str = "") -> bool:
        return self.call("armDisarm", arm, vehicle_name)


class MultirotorClient(VehicleClient):
    """The multirotor movement API, as used by ``flight/`` and ``swarm/``."""

    def getMultirotorState(self, vehicle_name: str = "") -> Struct:
        return self._struct("getMultirotorState", vehicle_name)

    def takeoffAsync(self, timeout_sec: float = 20, vehicle_name: str = "") -> Future:
        return self.call_async("takeoff", timeout_sec, vehicle_name)

    def landAsync(self, timeout_sec: float = 60, vehicle_name: str = "") -> Future:
        return self.call_async("land", timeout_sec, vehicle_name)

    def hoverAsync(self, vehicle_name: str = "") -> Future:
        return self.call_async("hover", vehicle_name)

    def goHomeAsync(self, timeout_sec: float = 3e38, vehicle_name: str = "") -> Future:
        return self.call_async("goHome", timeout_sec, vehicle_name)

    def moveToZAsync(self, z, velocity, timeout_sec: float = 3e38,
                     yaw_mode: YawMode | None = None, lookahead: float = -1,
                     adaptive_lookahead: float = 1, vehicle_name: str = "") -> Future:
        return self.call_async("moveToZ", z, velocity, timeout_sec,
                               yaw_mode or YawMode(), lookahead,
                               adaptive_lookahead, vehicle_name)

    def moveToPositionAsync(self, x, y, z, velocity, timeout_sec: float = 3e38,
                            drivetrain: int = DrivetrainType.MaxDegreeOfFreedom,
                            yaw_mode: YawMode | None = None, lookahead: float = -1,
                            adaptive_lookahead: float = 1, vehicle_name: str = "") -> Future:
        return self.call_async("moveToPosition", x, y, z, velocity, timeout_sec,
                               drivetrain, yaw_mode or YawMode(), lookahead,
                               adaptive_lookahead, vehicle_name)

    def moveOnPathAsync(self, path, velocity, timeout_sec: float = 3e38,
                        drivetrain: int = DrivetrainType.MaxDegreeOfFreedom,
                        yaw_mode: YawMode | None = None, lookahead: float = -1,
                        adaptive_lookahead: float = 1, vehicle_name: str = "") -> Future:
        return self.call_async("moveOnPath", list(path), velocity, timeout_sec,
                               drivetrain, yaw_mode or YawMode(), lookahead,
                               adaptive_lookahead, vehicle_name)

    def moveByVelocityZAsync(self, vx, vy, z, duration,
                             drivetrain: int = DrivetrainType.MaxDegreeOfFreedom,
                             yaw_mode: YawMode | None = None,
                             vehicle_name: str = "") -> Future:
        return self.call_async("moveByVelocityZ", vx, vy, z, duration, drivetrain,
                               yaw_mode or YawMode(), vehicle_name)

    def moveByVelocityZBodyFrameAsync(self, vx, vy, z, duration,
                                      drivetrain: int = DrivetrainType.MaxDegreeOfFreedom,
                                      yaw_mode: YawMode | None = None,
                                      vehicle_name: str = "") -> Future:
        return self.call_async("moveByVelocityZBodyFrame", vx, vy, z, duration,
                               drivetrain, yaw_mode or YawMode(), vehicle_name)

    def rotateToYawAsync(self, yaw, timeout_sec: float = 3e38,
                         margin: float = 5, vehicle_name: str = "") -> Future:
        return self.call_async("rotateToYaw", yaw, timeout_sec, margin, vehicle_name)

    def rotateByYawRateAsync(self, yaw_rate, duration, vehicle_name: str = "") -> Future:
        return self.call_async("rotateByYawRate", yaw_rate, duration, vehicle_name)

    def cancelLastTask(self, vehicle_name: str = "") -> None:
        self.call("cancelLastTask", vehicle_name)


# --------------------------------------------------------------------------
# reuse hook for the existing swarm/ manoeuvres
# --------------------------------------------------------------------------

def install_as_airsim() -> _types.ModuleType:
    """Register this module under the name ``airsim`` in ``sys.modules``.

    ``swarm/swarm_edge_to_center.py`` and friends do ``import airsim``. Inside
    the ROS container that package is deliberately absent (it would drag tornado
    into Python 3.12 -- see :mod:`rpc`), so this aliases the shim into its place
    *before* those modules are imported. It is a no-op if a real ``airsim`` is
    already importable, which is what happens if this code is ever run on the
    host venv instead.

    Returns the module now registered as ``airsim``.
    """
    existing = sys.modules.get("airsim")
    if existing is not None:
        return existing
    try:
        import airsim as real_airsim       # noqa: F401  (host venv path)
        return real_airsim
    except ImportError:
        pass
    sys.modules["airsim"] = sys.modules[__name__]
    return sys.modules[__name__]
