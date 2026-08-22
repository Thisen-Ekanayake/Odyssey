# CLAUDE.md — AirSim drone-swarm simulation

Workspace for drone-swarm research using **AirSim** (Blocks / AirSimNH / ZhangJiajie / Africa_Savannah)
in Docker on Arch Linux.

## Host facts
- GPU: NVIDIA RTX 4060 Laptop, driver 595, CUDA 13.2 (host)
- `nvidia-container-toolkit` installed; `nvidia` runtime registered; CDI spec at `/etc/cdi/nvidia.yaml`
- Display: **Wayland** (`XDG_SESSION_TYPE=wayland`), GUI apps go through XWayland on `DISPLAY=:1`
- Docker data-root: `/home/docker`; user is uid 1000 (matches container `airsim_user`)

## The critical gotcha
The base image `airsim_binary:10.0-devel-ubuntu18.04` is **only a runtime shell** (CUDA + X11 libs)
and was missing the **Vulkan loader** (`libvulkan1`). The nvidia runtime injects the NVIDIA Vulkan
ICD + driver libs but NOT the loader, so:
- the UE4 binary defaults to **OpenGL** → fatal "OpenGL is deprecated" dialog → instant exit
- forcing `-vulkan` without the loader → "Vulkan Driver is required"

**Fix:** use the derived image **`airsim_swarm:vk`** (built from `Dockerfile.vk` = base + `libvulkan1`
`vulkan-utils`, with the expired CUDA-10.0 apt repos removed), and **always pass `-vulkan`**.
Rebuild with: `docker build -f Dockerfile.vk -t airsim_swarm:vk .`

The image also ships **no simulator** — environments are downloaded separately (`fetch_envs.sh`) and
mounted in at runtime.

## Layout
| Path | Purpose |
|---|---|
| `Dockerfile.vk` | builds `airsim_swarm:vk` (base image + Vulkan loader) |
| `settings.json` | **four** SimpleFlight drones (`Drone1`-`Drone4`), each identically rigged for SLAM (mounted over `~/Documents/AirSim/settings.json`): a 16-channel/100 m LiDAR in **`SensorLocalFrame`**, a `StereoLeft`/`StereoRight` pair at 1280×720 / 90° FOV / 0.25 m baseline (left also exposes `DepthPlanar` as evaluation-only ground truth), an explicit `Imu` block with stated noise params, and the `ChaseCam` external camera. `slam/config.py` mirrors this file — **change both together**, and `tools/probe_setup.py` cross-checks them against the live sim. Note the two currently disagree on LiDAR rate: `settings.json` says `PointsPerSecond: 100000`, `slam/config.py` says `LIDAR_POINTS_PER_SECOND = 300_000`; `probe_setup.py` will flag it |
| `scripts/run_swarm.sh` | launches the sim (GPU + Vulkan + X11 + mounts); `./scripts/run_swarm.sh <Env>` picks the environment, `HEADLESS=1` for off-screen. Resolves the repo root as one level above its own location — if it's ever moved again, fix `WORKDIR` accordingly |
| `scripts/fetch_envs.sh` | downloads/extracts AirSim v1.8.1 environments from GitHub releases into `envs/` (same `WORKDIR`-is-one-level-up note as above) |
| `envs/` | downloaded environment binaries (gitignored) — `Blocks/`, `Africa_Savannah/`, `ZhangJiajie/`, `AirSimNH/`, each with a `LinuxNoEditor/<Env>.sh` launcher |
| `flight/` | single-drone AirSim client scripts: `sensor_demo.py` (single-drone sensor readout), `lidar_viz.py` (single-drone LiDAR mapping + chase-cam dual view; places scans by the SIMULATOR's `LidarData.pose`, so it is the perfect-localization "no SLAM" baseline — contrast `slam_live.py`), `detect_objects.py` (YOLO26 object detection on the drone's own FPV camera "0", while it yaw-scans in place), `segment_objects.py` (same setup, YOLO26 `-seg` instance segmentation instead of boxes), `segment_sam3.py` (open-vocabulary video segmentation + tracking on the FPV feed via Meta SAM 3's `Sam3VideoModel`/`Sam3VideoProcessor`, transformers>=5, text-prompted concepts instead of YOLO's fixed 80 classes, tracks object identity across frames via a streaming `Sam3VideoInferenceSession`; weights loaded locally from `models/sam_3`, gitignored — no auto-download), `follow_road.py` (steers by yaw-rate to keep a road centered under camera "3"/bottom_center, via HSV color thresholding, not ML — run with `--calibrate` first to tune `ROAD_HSV_LOW/HIGH` for the loaded environment), `autonomous_navigate.py` (take off to 5 m and fly straight ahead at constant body-frame speed, forever — no sensors, no ML; it is also the constants module `lidar_viz.py` imports `DRONE`/`ALTITUDE`/`SPEED` from), `record_dataset.py` (flies the benchmark circuit under one weather condition and records a synchronized LiDAR+stereo+IMU+ground-truth dataset), `slam_live.py` (live SLAM with a 3-panel Open3D viewer: camera feed, map built from ESTIMATED poses, estimated-vs-ground-truth trajectory), `square_capture_map.py` (flies a closed square circuit recording LiDAR + ground-truth poses; mapping happens AFTER landing, offline — not a live viewer like `slam_live.py`/`lidar_viz.py`; its output is read by `tools/view_square_map.py` and `tools/densify_map.py`) |
| `swarm/` | multi-drone (Drone1-4) AirSim client scripts: `swarm_comms.py` (`SwarmPositions` — shared-GPS swarm coordination: since AirSim has no radio/RF sim, "communicating" position is just polling every drone's GPS through the one client already used to control the swarm — perfect/instant, not a simulated channel; also calibrates each vehicle's LOCAL->WORLD position offset, since AirSim reports multi-vehicle positions in each vehicle's own local frame — every other script in this directory that needs world-frame coordinates imports `SwarmPositions`/`.offset()` for this; run standalone for a 4-drone takeoff + live position-share + land demo), `swarm_demo.py` (3-drone line formation), `swarm_circle.py` (5-drone ring orbit), `swarm_converge.py` (shrinks the 4-drone formation onto a virtual center — Drone1's live position, read through `swarm_comms.SwarmPositions` — via a uniform scale-down of each drone's spawn offset, so every pair keeps a constant distance RATIO throughout; the scale floors before reaching zero so the swarm settles a few meters apart around Drone1 instead of colliding with it; exposes `run(client)` for reuse), `swarm_converge_viz.py` (5-window live view of the same maneuver: one chase-cam feed per drone via `ChaseCam`/`ChaseCam2`/`ChaseCam3`/`ChaseCam4` in `settings.json`, plus a shared Open3D LiDAR map merging all 4 drones' scans, color-coded per drone and shaded dark-to-light by height (NED z, fixed range) so vertical structure is visible, with a top-right rotation panel — manual azimuth slider, "Auto-rotate 360°" checkbox, adjustable speed — orbiting a fixed elevation/radius around the formation center), `swarm_lines_viz.py` (takes off Drone1-4, then in its own Open3D window draws the formation quadrilateral segmented into 4 small squares — the 4 perimeter edges, ordered by angle around the centroid so the loop doesn't self-cross, plus a line from the diagonal intersection to each edge's midpoint; the corner-to-corner diagonals themselves are computed but not drawn, since drawing them would cut each square into triangles — no drone models or LiDAR, redrawn live each tick from `swarm_comms.SwarmPositions`), `swarm_edge_to_center.py` (takes off Drone1-4, then each drone flies its own perimeter edge to that edge's midpoint, slows into a left turn — via a densified `moveOnPathAsync` path, not hand-rolled per-waypoint velocity control — and heads toward the shared diagonal-intersection center, stopping 5 m short of it since all 4 center-bound lines meet at the same point; the perimeter winding direction is picked at runtime so the midpoint-to-center cut always comes out as a genuine left turn regardless of `SPAWNS` layout), `swarm_edge_to_center_viz.py` (live view of that maneuver across 6 windows: an Open3D window with the 4-square segmentation lines drawn once as the fixed target geometry, same construction as `swarm_lines_viz.py`, plus one colored dot per drone tracking its live position across it via `Open3DScene.set_geometry_transform`; one chase-cam window per drone via `ChaseCam`/`ChaseCam2`/`ChaseCam3`/`ChaseCam4` in `settings.json`; and a shared Open3D LiDAR map merging all 4 drones' scans, color-coded per drone and height-shaded — all three pieces the same techniques `swarm_converge_viz.py` established — while `swarm_edge_to_center.run()` drives the flight in a background thread). All cross-import each other by flat module name (e.g. `from swarm_comms import ...`) rather than as a package, so they must stay siblings in this one directory |
| `slam/` | the SLAM package (CPU-only, no new deps): `config.py` (mirrors `settings.json` + all tunables + the weather grid), `geometry.py` (SE(3)/NED helpers, Umeyama, TUM I/O), `source.py` (`DatasetSource`/`LiveSource` — same `Frame` stream offline and live), `recorder.py` (deterministic `simPause`+`simContinueForTime` lockstep capture), `weather.py`, `imu.py` (strapdown prior + LiDAR de-skew), `mapping.py` (sliding submap + batched voxel map), `backend.py` (**shared** pose graph + loop closure), `lidar_slam.py`, `stereo_slam.py`, `degradation.py` (adverse-weather LiDAR model), `evaluate.py` (ATE/RPE/map metrics). See `docs/SLAM.md` |
| `tools/` | offline/utility scripts, none of which fly a drone maneuver of their own: `probe_setup.py` (Phase-0 sanity check against the live sim — **run first** after any `settings.json` change), `verify_dataset.py`, `run_benchmark.py` (offline cross product of conditions × methods × raw/degraded → `results/`), `synthetic_dataset.py` (raycast fixture world producing the same on-disk format, so the whole offline pipeline can be tested with no simulator), `spawn_traffic.py` (scatters static car/prop assets via `simSpawnObject` so there's something for `flight/detect_objects.py`'s YOLO to find — AirSimNH has no built-in moving traffic), `view_square_map.py` (interactive Open3D viewer over a `flight/square_capture_map.py` recording, height-colored with the flight path overlaid), `densify_map.py` (K-nearest-neighbor gap-filling over a `flight/square_capture_map.py` map — not real surface reconstruction, just thickens sparse regions), `compare_ros_slam.py` (scores a ROS/rtabmap TUM trajectory against the SAME ground truth and with the SAME `slam/evaluate.py` code as `run_benchmark.py`, so the two estimators' ATE/RPE are comparable — runs on the HOST in `airsim_venv`, since the ROS container has no open3d) |
| `ros2_ws/` | the ROS 2 **Jazzy** layer, built and run inside the `ros2` distrobox (`scripts/ros_enter.sh`). `airsim_swarm_msgs/` (`SwarmState`, `DroneState`, `Maneuver.srv`) and `airsim_swarm_bridge/`: `rpc.py` (a tornado-free msgpack-rpc client — the `airsim` pip package CANNOT be installed in that container, see `docs/ROS.md`), `airsim_api.py` (an `airsim.MultirotorClient`-compatible shim over it, which `maneuver_node.py` aliases into `sys.modules` as `airsim` so `swarm/swarm_edge_to_center.py`'s and `swarm/swarm_converge.py`'s own `run(client)` execute unmodified as ROS services), `frames.py` (NED/FRD -> ENU/FLU, the one place that conversion happens), `bridge_node.py` (one process and one RPC connection PER DRONE), `cloud_merger_node.py` (relays every drone's LiDAR onto one topic, unchanged frame ids and all, so a single `octomap_server` builds ONE shared map via its own per-message TF/ray-origin lookup -- no per-point transform needed), `swarm_state_node.py` (reuses `swarm_lines_viz.py`'s corner-ordering for the formation markers), `dataset_to_rosbag.py` (`datasets/<condition>/` -> rosbag2, so rtabmap gets the same recorded bytes `run_benchmark.py` replays) and `traj_recorder_node.py` (odometry -> TUM). Launch files cover the bridge, `cooperative_mapping.launch.py` (live 4-drone merged-LiDAR octomap, geometry-verified against the mock server in `test/test_cooperative_mapping.py`), `rtabmap_ros` LiDAR SLAM live or from a bag, and the RViz/Foxglove/PlotJuggler layer |
| `rl/` | reinforcement learning: `airsim_gym_env.py` (Gymnasium env, ring-orbit formation-keeping), `train_drl.py` (PPO training entry point) |
| `docs/` | `SETUP_MANUAL.md` (host setup), `user_manual.md` (day-to-day usage), `SLAM.md` (the SLAM study: design, how to run, degradation model, verified numbers, known limitations), `ROS.md` (the ROS 2 layer: container setup, why the `airsim` pip package cannot be used there, frame conventions, every launch file, and the rtabmap-vs-`slam/` comparison workflow) |
| `scripts/` | `run_swarm.sh` + `fetch_envs.sh` (see above), the ROS helpers `ros_setup.sh` (one-time container setup), `ros_env.sh` (sourced shell fragment: repairs `PATH` so `python3` is the CONTAINER's, then sources ROS — no-ops outside a container, which matters because distrobox shares `$HOME`/`~/.bashrc` with the host) and `ros_enter.sh` (host-side `distrobox enter` wrapper), plus one-time host setup scripts (`setup_arch.sh`, `install_arch.sh`, `setup_ubuntu.sh`, `install_ubuntu.sh`) |
| `airsim_venv/` | Python client venv (airsim 1.8.1, numpy 2.2.6, torch+cu130, ultralytics, transformers>=5 for `segment_sam3.py`), gitignored |
| `models/` | gitignored artifact dir: PPO checkpoints (`train_drl.py`) and YOLO weights (`detect_objects.py`, auto-downloaded) |

`models/` and `logs/` (PPO checkpoints + tensorboard logs from `train_drl.py`) are created at the repo
root when training runs, and are gitignored. So are `datasets/`, `datasets_synth/` and `results/`
(recorded SLAM data is ~3 GB per weather condition).

Note `docs/SETUP_MANUAL.md` still describes the old 5-drone `settings.json` and an older
`lidar_viz.py` that used `moveOnPathAsync` over a waypoint pattern — both stale.

## Run
```bash
# download an environment once (Blocks/Africa_Savannah/ZhangJiajie/AirSimNH/...):
./scripts/fetch_envs.sh AirSimNH

# GUI window (default env) or pick one / go headless:
./scripts/run_swarm.sh
./scripts/run_swarm.sh AirSimNH
HEADLESS=1 ./scripts/run_swarm.sh AirSimNH

# in another terminal, command the drone (connects to 127.0.0.1:41451 via --net=host):
./airsim_venv/bin/python flight/lidar_viz.py

# SLAM (see docs/SLAM.md): probe first, then record, then benchmark offline
./airsim_venv/bin/python tools/probe_setup.py
./airsim_venv/bin/python flight/record_dataset.py --condition all
./airsim_venv/bin/python tools/run_benchmark.py
./airsim_venv/bin/python flight/slam_live.py --method lidar --weather fog_heavy

# --- ROS 2 (Jazzy, in the `ros2` distrobox) -- see docs/ROS.md ---
distrobox enter ros2 -- /ml/airsim_swarm/scripts/ros_setup.sh   # once
./scripts/ros_enter.sh ros2 launch airsim_swarm_bridge bridge.launch.py
./scripts/ros_enter.sh ros2 launch airsim_swarm_bridge viz.launch.py
# live 4-drone cooperative mapping demo (one shared octomap, all 4 LiDARs):
./scripts/ros_enter.sh ros2 launch airsim_swarm_bridge cooperative_mapping.launch.py \
    auto_maneuver:=edge_to_center
./scripts/ros_enter.sh ros2 service call /swarm/edge_to_center \
    airsim_swarm_msgs/srv/Maneuver "{name: edge_to_center}"
# offline SLAM from a recorded dataset, then score it against slam/:
./scripts/ros_enter.sh ros2 run airsim_swarm_bridge dataset_to_rosbag \
    datasets/clear --out bags/clear --skip-stereo
./scripts/ros_enter.sh ros2 launch airsim_swarm_bridge slam_replay.launch.py \
    bag:=bags/clear record_trajectory:=true trajectory_out:=results/ros/clear_rtabmap.txt
./airsim_venv/bin/python tools/compare_ros_slam.py --condition clear
```
Note `swarm/swarm_circle.py` (Drone1-5) will fail against the current four-drone `settings.json` —
add `Drone5` first if you want it. `swarm/swarm_demo.py` (Drone1-3) works as-is.
RPC server: `127.0.0.1:41451`. First boot is slow (UE4 shader compile) — the API server can take
~30-60s after the window/process appears.

## Conventions / gotchas for edits
- Container must run with `--runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=all --net=host`.
- Never drop `-vulkan` from the launch (OpenGL is broken in this build).
- `settings.json` is only read at sim **startup** — restart `run_swarm.sh` after editing it (e.g. after
  adding/moving an `ExternalCameras` entry).
- To resize the swarm: add vehicles in `settings.json` AND update the `DRONES`/`SPAWNS` list in the
  relevant `swarm/*.py` script (names must match). Space spawn positions apart (X/Y) so drones don't
  collide at start.
- Prefer AirSim's built-in path-following (`moveOnPathAsync` with a velocity-derived lookahead, see
  `flight/record_dataset.py` via `slam/recorder.py`) over chaining `moveToPositionAsync` calls or
  hand-rolled per-waypoint velocity control — both of the latter cause visible tilt/lurch or jerky
  cornering at waypoint transitions, and the resulting lurch shows up as spurious IMU acceleration.
  Densify the path (`slam/config.route_waypoints`) — with only corner waypoints the lookahead cuts
  corners badly.
- **AirSim weather is a RENDERING effect only.** `simSetWeatherParameter` changes what the cameras
  see; the LiDAR is a raycast against collision geometry and rain/fog particles have no collision, so
  LiDAR returns are identical in every condition. `slam/degradation.py` models the LiDAR side; never
  report a weather comparison without saying which half is simulated and which is modelled.
- Open3D's `global_optimization` **prunes edges from the `PoseGraph` you pass it**, so optimizing the
  same graph object repeatedly erodes it. `slam/backend.py` keeps an authoritative edge list and
  rebuilds a fresh graph per call — don't "simplify" that away.
- Seed Open3D's RANSAC (`o3d.utility.random.seed(...)`) before any run whose numbers you intend to
  compare; loop-closure verification is otherwise nondeterministic and results swing wildly.
- The `airsim` pip package fails under build isolation: install `numpy` first, then
  `pip install airsim --no-build-isolation`.
- `airsim`'s RPC dependency (`msgpack-rpc-python`) hard-pins `tornado<5`. Only install `airsim`
  into a venv dedicated to this project (`airsim_venv/`) — installing it into a general-purpose/shared
  venv (e.g. a Jupyter-based ML env) downgrades `tornado` and breaks jupyterlab/ipykernel/notebook there.
- Harmless log noise to ignore: ALSA "Couldn't open audio device" (no sound card),
  `LogStreaming` errors for editor-only assets.
- Verified end-to-end headless: all 3 drones armed, flew a formation, and landed.

- The ROS container (`ros2` distrobox, Ubuntu 24.04 / **Python 3.12**) **cannot** have the `airsim`
  pip package: it pins `tornado<5`, and tornado 4.5.3 uses `collections.MutableMapping` and
  `inspect.getargspec`, both removed by Python 3.10/3.11. `ros2_ws/.../rpc.py` talks raw msgpack-rpc
  instead. Never "fix" a ROS import error by pip-installing airsim in there.
- distrobox shares `$HOME` with the Arch host, so `~/.bashrc` is the SAME FILE and the host's pyenv
  shims + `airsim_venv` shadow the container's `python3` (which then dies on `libcrypt.so.2`).
  `scripts/ros_env.sh` fixes both and is guarded on `/run/.containerenv` — keep new shell wiring
  going through it rather than writing to `~/.bashrc` directly.
- The repo is NOT mounted at `/ml` inside the container; `ros_setup.sh` symlinks `/ml -> /run/host/ml`
  so both sides use identical absolute paths.
- A rosbag must record `/tf_static` with **TRANSIENT_LOCAL** durability or tf2 never receives it on
  replay, and SLAM nodes report `<frame> does not exist` while `/tf` looks fine.
- Anything scoring a trajectory must go through `slam/evaluate.py` (`tools/compare_ros_slam.py` does).
  A second ATE implementation would differ in alignment/association and the comparison would be void.

- Do not add co author to commits