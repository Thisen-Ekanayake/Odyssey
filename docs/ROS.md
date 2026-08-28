# ROS 2 layer

ROS 2 **Jazzy**, running in the `ros2` distrobox container, bridging the AirSim
swarm onto standard topics/TF and running `rtabmap_ros` as a second, independent
SLAM estimator alongside the repo's own `slam/` package.

Nothing in `flight/`, `swarm/`, `slam/` or `settings.json` changed to make this
work. The ROS layer is additive, and it *reuses* the existing code rather than
reimplementing it — the manoeuvre services call `swarm/swarm_edge_to_center.py`'s
own `run(client)`, and the comparison scores rtabmap with `slam/evaluate.py`.

---

## Quick start

```bash
# once, inside the container: apt deps, /ml symlink, colcon build
distrobox enter ros2 -- /ml/airsim_swarm/scripts/ros_setup.sh

# terminal 1 (host): the simulator
./scripts/run_swarm.sh AirSimNH

# terminal 2: the bridge, all four drones
./scripts/ros_enter.sh ros2 launch airsim_swarm_bridge bridge.launch.py

# terminal 3: look at it
./scripts/ros_enter.sh ros2 launch airsim_swarm_bridge viz.launch.py

# fly something
./scripts/ros_enter.sh ros2 service call /swarm/takeoff \
    airsim_swarm_msgs/srv/Maneuver "{name: takeoff, altitude: 50.0}"
./scripts/ros_enter.sh ros2 service call /swarm/edge_to_center \
    airsim_swarm_msgs/srv/Maneuver "{name: edge_to_center}"
```

`scripts/ros_enter.sh` takes an argv-style command (`ros2 topic list`) or drops
you in an interactive shell with no arguments.

---

## Two things about this container that will bite you

### 1. The `airsim` pip package cannot be installed here

The container is Ubuntu 24.04 / **Python 3.12**. `pip install airsim` pulls
`msgpack-rpc-python`, which hard-pins `tornado<5`, and tornado 4.5.3 is dead on
3.12 — `tornado/httputil.py` subclasses `collections.MutableMapping` (removed in
3.10) and `tornado/util.py` calls `inspect.getargspec` (removed in 3.11). The
host venv only survives because it is Python 3.10.

So the bridge does not use it. `airsim_swarm_bridge/rpc.py` is a ~250-line
msgpack-rpc client over a plain socket — AirSim's server is ordinary
msgpack-rpc, `request [0, msgid, method, params]` / `response [1, msgid, error,
result]`, and only ~15 of its 127 methods are needed. `airsim_api.py` layers an
`airsim.MultirotorClient`-compatible surface on top so the existing `swarm/`
scripts import and run unmodified.

The wire settings are copied from the working host client, not guessed:

| host client | msgpack 1.x equivalent used here |
|---|---|
| `pack_encoding='utf-8'` | `Packer(use_bin_type=False)` |
| `unpack_encoding='utf-8'` | `Unpacker(raw=False, unicode_errors="surrogateescape")` |
| — | `strict_map_key=False` (msgpack 1.x rejects non-str keys by default) |

`surrogateescape` is load-bearing: whether a binary image payload arrives as
msgpack `bin` or the older `raw` type depends on the server, and `raw=False`
alone would try to UTF-8 decode a camera frame. `airsim_api.to_uint8_array()`
normalises all three possible shapes back to `bytes`.

`test/test_rpc_roundtrip.py` verifies this across the version boundary: the
reference `msgpack-rpc-python` server runs on the host venv (3.10), the
tornado-free client connects from the container (3.12), 23 assertions.

### 2. `$HOME` is shared with the Arch host

distrobox mounts your home directory straight through, which has two consequences:

* **`python3` resolves to host interpreters.** The host's pyenv shims and
  `airsim_venv` are on `PATH` inside the container, and running an Arch-built
  interpreter under Ubuntu dies with `libcrypt.so.2: cannot open shared object
  file`. `scripts/ros_env.sh` strips those entries and puts `/usr/bin` first.
* **`~/.bashrc` is the same file the host reads.** The block `ros_setup.sh`
  appends therefore only sources `ros_env.sh`, which no-ops outside a container
  (it guards on `/run/.containerenv`).

The repo is not mounted at `/ml` either — only `$HOME` is. It *is* reachable at
`/run/host/ml/airsim_swarm`, so `ros_setup.sh` symlinks `/ml -> /run/host/ml`,
giving host and container identical absolute paths.

---

## Frames

AirSim is **NED/FRD**; ROS is **ENU/FLU** (REP-103). Everything is converted once,
in `frames.py`. A mistake here is silent — the map still looks like a map, ICP
still converges, and the world is just mirrored — so `test/test_frames.py` checks
the quaternion path against the matrix composition on 500 random rotations, plus
the cases with an obvious physical answer (5 m up must publish as `z = +5`).

```
map                            shared ENU world frame, all four drones
└─ droneN/odom                 static; the spawn offset (see below)
   └─ droneN/base_link         FLU
      ├─ droneN/lidar_link     (0, 0, +0.1) — settings.json has Z=-0.1 in NED
      ├─ droneN/imu_link
      ├─ droneN/stereo_left_link  → droneN/stereo_left_optical
      └─ droneN/stereo_right_link → droneN/stereo_right_optical
```

`map -> droneN/odom` comes from `swarm/swarm_comms.py::SwarmPositions.offset()`,
which is reused rather than reimplemented: AirSim reports each vehicle's position
in its own undocumented local reference, and that method already calibrates the
delta by snapshotting the raw reading at startup. **Start the bridge before
takeoff** — the calibration assumes the drone has not moved yet.

`gt_owns_base_link` decides who publishes TF's `odom -> base_link`. It defaults
to true (the bridge publishes ground truth there). The SLAM launch files set it
false, because `icp_odometry` publishes that edge itself and two writers on one
TF edge is a corrupt tree; ground truth then goes to `droneN/base_link_gt`.

---

## Topics

Per drone, namespaced by vehicle name (`Drone1` → `/drone1`):

| topic | type | rate |
|---|---|---|
| `/droneN/imu` | `sensor_msgs/Imu` | 100 Hz |
| `/droneN/lidar/points` | `sensor_msgs/PointCloud2` | 10 Hz |
| `/droneN/stereo/{left,right}/image_raw` (+ `camera_info`) | `sensor_msgs/Image` | 5 Hz |
| `/droneN/gps` | `sensor_msgs/NavSatFix` | 10 Hz |
| `/droneN/odom_gt`, `/droneN/path_gt` | `nav_msgs/Odometry`, `Path` | 50 Hz |

Shared: `/swarm/state` (`airsim_swarm_msgs/SwarmState`) and
`/swarm/formation_markers` (`visualization_msgs/MarkerArray`).

Stereo runs at 5 Hz on purpose. `simGetImages` is by far the most expensive RPC;
four drones × two cameras at 10 Hz does not keep up and the backlog degrades the
LiDAR stream. Pass `stereo:=false` for runs that do not need images.

One node and one RPC connection per drone — msgpack-rpc has no thread safety, and
one slow camera would otherwise stall the whole swarm.

---

## Cooperative mapping (demo)

```bash
PROFILE=swarm ./scripts/run_swarm.sh AirSimNH        # terminal 1 -- note the profile
./scripts/ros_enter.sh ros2 launch airsim_swarm_bridge cooperative_mapping.launch.py
# or, one command that also flies a maneuver a few seconds after startup:
./scripts/ros_enter.sh ros2 launch airsim_swarm_bridge cooperative_mapping.launch.py \
    auto_maneuver:=edge_to_center
```

Merges all four drones' LiDAR into one shared `octomap_server` instance and shows
it live in RViz alongside the four colour-coded raw clouds and the formation
markers -- the "4 drones building one map together" shot.

`PROFILE=swarm` matters: four drones on the full-resolution SLAM rig drag AirSim's
clock to roughly half real time, and while that reads as a framerate problem it is
not only one -- see "One clock, or no map" below.

`cloud_merger_node` is a pure **relay**, not a transform: it republishes each
drone's cloud unchanged, keeping its own `droneN/lidar_link` frame id.
`octomap_server` does its own per-message TF lookup and uses the resulting
transform's translation as the ray-tracing sensor origin (confirmed against the
installed binary -- it links `tf2_ros::MessageFilter<PointCloud2>` and
`pcl_ros::transformPointCloud`, the standard octomap_server pattern). Four
independently-posed frame ids arriving on one topic is no different to it than
one frame moving over time, so no cross-drone synchronisation is needed either.
`test_cooperative_mapping.py` checks this by geometry rather than by eye: occupied
voxels must cluster around each drone's true world position and not near the map
origin, using deliberately asymmetric spawns so an X/Y swap cannot hide.

### One clock, or no map

The single most important invariant here, and the one that broke it: **TF and
sensor data must be stamped from the same clock.**

AirSim runs a `SteppableClock` for SimpleFlight multirotors, and with four drones
loaded it advances at roughly 0.5x wall time. Sensor payloads carry that clock in
`time_stamp`; TF and odometry carry nothing, so an earlier `bridge_node` stamped
them from the node's wall clock instead. The two then drift apart without bound.
Every consumer built on `tf2_ros::MessageFilter` -- `octomap_server`,
`icp_odometry`, `rtabmap` -- silently discarded 100% of clouds:

```
[INFO] [octomap_server]: Message Filter dropping message: frame 'drone3/lidar_link'
  at time 1787427030.511 for reason 'the timestamp on the message is earlier than
  all the data in the transform cache'
```

That is an INFO line. There is no error, no crash, and RViz opens normally -- the
map is simply always empty. `bridge_node._stamp` now projects the node clock
through a continuously-updated sim-minus-wall offset, so odometry and TF share
AirSim's timeline with the sensors. `use_airsim_time:=false` puts the whole node
on wall time instead, which is also self-consistent, at the cost of IMU dt no
longer meaning sim time.

The reason this survived so long is worth recording: the test suite passed
throughout. `mock_airsim_server.py` stamped its "sim clock" with `time.time()`,
so under test the two clocks agreed by construction and the bug was invisible.
The mock now takes `--clock-rate` / `--clock-skew`, and `run_tests.sh` runs the
mapping suite at `--clock-rate 0.5 --clock-skew -30` with 16k-point sweeps. Put
the old `_stamp` back and that suite fails with "no message in 45s".

### RViz cannot display an octomap on this machine

`ros-jazzy-octomap-rviz-plugins` ships a `liboctomap_rviz_plugins.so` that does
not link `liboctomap`, so RViz fails to load it:

```
PluginlibFactory: The plugin for class 'octomap_rviz_plugins/OccupancyGrid' failed
to load. ... undefined symbol: _ZTIN7octomap13OcTreeStampedE
```

`swarm_live.rviz` therefore draws the shared map from
`/octomap_point_cloud_centers` -- the occupied-voxel centres octomap_server
publishes anyway -- with a stock `rviz_default_plugins/PointCloud2` in `Boxes`
style at the octomap resolution. It looks the same, drops a package from the
critical path, and shows the exact topic the test asserts on. If you want the real
plugin, `LD_PRELOAD` the system `liboctomap.so` into `rviz2`.

`test_rviz_config.py` now dlopens every display class named in `rviz/*.rviz`, so a
config referring to a plugin this machine cannot load fails offline in
milliseconds rather than silently showing nothing during a demo.

This is cooperative **mapping**, not cooperative SLAM: poses come from ground
truth (`gt_owns_base_link` stays true, since no SLAM node here needs the
`odom -> base_link` edge). For "how well would N independently-estimated poses
agree on one map", each drone needs its own SLAM front end -- a harder,
different exercise from placing four known-good poses into one map.

Recording it: `tools/window_recorder.py` captures the RViz window exactly as it
does AirSim's own; `tools/to_gif.sh` turns the capture into a shareable clip.

## SLAM

```bash
# live
./scripts/ros_enter.sh ros2 launch airsim_swarm_bridge slam_lidar.launch.py

# offline, from a recorded dataset — no simulator needed
./scripts/ros_enter.sh ros2 run airsim_swarm_bridge dataset_to_rosbag \
    datasets/clear --out bags/clear --skip-stereo
./scripts/ros_enter.sh ros2 launch airsim_swarm_bridge slam_replay.launch.py \
    bag:=bags/clear record_trajectory:=true \
    trajectory_out:=results/ros/clear_rtabmap.txt
```

`icp_odometry` → `rtabmap`, configured in `config/rtabmap_lidar.yaml` for the
aerial case: 6-DoF (`Reg/Force3DoF=false`, because a drone banks and that roll is
real motion), point-to-plane ICP (the scan looks mostly *down* — settings.json
gives it +15°/−45° vertical FOV — so ground-plane returns dominate and
point-to-point is under-constrained on a flat surface), and loop closure via
`RGBD/ProximityBySpace` since there is no visual front end.

rtabmap builds its own octomap when `Grid/3D` is on and publishes `/octomap_full`,
`/octomap_binary`, `/cloud_map` and `/map` itself, so the standalone
`octomap_server` is **off by default** (`octomap:=true` to add it — it re-voxelises
the same data and regenerates the whole cloud on every graph change).

`slam_2d.launch.py` (`pointcloud_to_laserscan` → `slam_toolbox`) exists but is a
sanity check, not a result: slicing a horizontal band out of a 3D cloud throws
away most of what the sensor measured, and any bank angle tilts the slice.

### Comparing against `slam/`

The two estimators run in different Python environments — `slam/` needs
scipy+open3d on the host, rtabmap lives in the container — so they meet at a TUM
trajectory file and are scored by **the same code**:

```bash
./airsim_venv/bin/python tools/compare_ros_slam.py --condition clear
```

`traj_recorder` writes `results/ros/<condition>_<method>.txt` in the format
`slam/geometry.py::write_tum` produces; `compare_ros_slam.py` reads it and calls
`slam/evaluate.py::evaluate_trajectory`, the same function `tools/run_benchmark.py`
uses. A second ATE implementation would differ in alignment and association and
the comparison would mean nothing.

Alignment is on by default: odometry starts at its own origin, not at the
world position.

Result on `clear`, both estimators over the full 798 m circuit, identical recorded
bytes, identical scoring code:

| estimator | ATE RMSE | RPE %/10 m | drift |
|---|---|---|---|
| `ros:rtabmap` (icp_odometry + loop closure) | 3.455 m | 3.67 | 0.64 % |
| `py:lidar` (`slam/`, odometry only) | **0.430 m** | 3.25 | 0.07 % |
| `py:stereo` (`slam/`, odometry only) | 44.948 m | 29.56 | 14.04 % |

Read the RPE column alongside the ATE. The two LiDAR estimators have almost the
same *local* accuracy (3.67 vs 3.25 %/10 m) — they disagree by 8× on ATE, which is
accumulated global drift, and rtabmap is carrying loop closure while `slam/` here
is not. This is a comparison of two configurations, not a verdict on two
libraries.

Do **not** compare against `results/metrics.json` rows produced with
`--max-frames`. ATE is an absolute distance and does not normalise by path length;
`compare_ros_slam.py` prints a loud warning when the two sides differ by more than
2× in trajectory length, which is exactly the state `results/` was in before the
full grid was run.

### Weather

**AirSim weather is a rendering effect only.** The camera streams in a `rain_*` /
`fog_*` recording genuinely are degraded. The LiDAR is a raycast against collision
geometry, rain and fog particles have none, and its scans are bit-identical to
`clear`. `dataset_to_rosbag --degraded` applies `slam/degradation.py` — the same
hand-built extinction model `run_benchmark.py` uses — so rtabmap sees the same
modelled points. Every bag carries an `airsim_swarm.json` sidecar recording which
half is which. Never report a weather comparison without saying so.

---

## Files

| path | purpose |
|---|---|
| `scripts/ros_setup.sh` | one-time container setup (idempotent) |
| `scripts/ros_env.sh` | sourced shell fragment: PATH repair + ROS sourcing; no-ops on the host |
| `scripts/ros_enter.sh` | host-side wrapper around `distrobox enter` |
| `ros2_ws/src/airsim_swarm_msgs/` | `DroneState`, `SwarmState`, `Maneuver.srv` |
| `…/airsim_swarm_bridge/rpc.py` | tornado-free msgpack-rpc client |
| `…/airsim_api.py` | `airsim.MultirotorClient`-compatible shim over it |
| `…/frames.py` | NED↔ENU / FRD↔FLU, frame ids, `CameraInfo` |
| `…/repo.py` | makes `swarm/` and `slam/` importable from ROS nodes |
| `…/bridge_node.py` | one drone → sensors, odometry, TF |
| `…/swarm_state_node.py` | `/swarm/state` + formation markers |
| `…/maneuver_node.py` | `/swarm/*` services wrapping the `swarm/` manoeuvres |
| `…/traj_recorder_node.py` | odometry topic → TUM file |
| `…/cloud_merger_node.py` | relays every drone's LiDAR onto one topic for a shared octomap |
| `…/dataset_to_rosbag.py` | `datasets/<condition>/` → rosbag2 |
| `…/launch/` | `bridge`, `cooperative_mapping`, `slam_lidar`, `slam_replay`, `slam_2d`, `viz` |
| `…/rviz/` | `swarm_live.rviz` (4-drone + shared map), `slam_single.rviz` (rtabmap) |
| `…/config/` | `drones.yaml`, `octomap.yaml`, `rtabmap_lidar.yaml`, `ekf.yaml` |
| `…/test/run_tests.sh` | all five offline suites, 107 assertions, no simulator |
| `…/test/mock_airsim_server.py` | reference msgpack-rpc server; **host venv only** (tornado). `--clock-rate`/`--clock-skew`/`--points-per-sweep` make it misbehave like the real sim |
| `…/test/test_frames.py` | NED/FRD ↔ ENU/FLU, incl. 500 random rotations |
| `…/test/test_rpc_roundtrip.py` | wire compatibility, host server ↔ container client |
| `…/test/test_swarm_reuse.py` | `swarm/` manoeuvres running unmodified through the shim |
| `…/test/test_rviz_config.py` | dlopens every display class named in `rviz/*.rviz` |
| `…/test/test_cooperative_mapping.py` | 4-drone octomap geometry, under a skewed sim clock |
| `tools/compare_ros_slam.py` | **host-side**; scores ROS estimates with `slam/evaluate.py` |

---

## Gotchas

* **`settings.json` is read only at simulator startup.** Adding a vehicle without
  restarting `run_swarm.sh` gets you a bridge that connects and then reports the
  vehicle does not exist (it checks `listVehicles` and says which are present).
* **`/tf_static` must be recorded with `TRANSIENT_LOCAL` durability.** tf2
  subscribes to it that way; a bag that records it with the default volatile QoS
  replays a publisher no listener ever hears, and the symptom is a SLAM node
  insisting `drone1/base_link ... does not exist` while `/tf` flows normally.
  `dataset_to_rosbag`'s `BagWriter._qos_for` handles this.
* **rtabmap's odometry is on `/odom`**, not `/rtabmap/odom` — the node name does
  not namespace its topics. Same for `/cloud_map`.
* **RViz renders on the Intel iGPU** (zink/Mesa). This container was not created
  with `distrobox create --nvidia`. That is fine for RViz; if it ever matters,
  `podman commit ros2 ros2-snap` preserves the installed packages and the
  container can be recreated with `--nvidia`.
* `ROS_DOMAIN_ID` is pinned to 42 and `ROS_LOCALHOST_ONLY=1` in `ros_env.sh`, so
  a stray ROS node elsewhere on the machine cannot join by accident.
