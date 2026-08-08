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
| `settings.json` | **single** SimpleFlight drone (`Drone1`) rigged for SLAM (mounted over `~/Documents/AirSim/settings.json`): a 16-channel/100 m/300 k-pts-s LiDAR in **`SensorLocalFrame`**, a `StereoLeft`/`StereoRight` pair at 1280×720 / 90° FOV / 0.25 m baseline (left also exposes `DepthPlanar` as evaluation-only ground truth), an explicit `Imu` block with stated noise params, and the `ChaseCam` external camera. `slam/config.py` mirrors this file — **change both together**, and `tools/probe_setup.py` cross-checks them against the live sim |
| `scripts/run_swarm.sh` | launches the sim (GPU + Vulkan + X11 + mounts); `./scripts/run_swarm.sh <Env>` picks the environment, `HEADLESS=1` for off-screen. Resolves the repo root as one level above its own location — if it's ever moved again, fix `WORKDIR` accordingly |
| `scripts/fetch_envs.sh` | downloads/extracts AirSim v1.8.1 environments from GitHub releases into `envs/` (same `WORKDIR`-is-one-level-up note as above) |
| `envs/` | downloaded environment binaries (gitignored) — `Blocks/`, `Africa_Savannah/`, `ZhangJiajie/`, `AirSimNH/`, each with a `LinuxNoEditor/<Env>.sh` launcher |
| `flight/` | AirSim client scripts that fly the sim: `swarm_demo.py` (3-drone line formation), `swarm_circle.py` (5-drone ring orbit), `sensor_demo.py` (single-drone sensor readout), `lidar_viz.py` (single-drone LiDAR mapping + chase-cam dual view; places scans by the SIMULATOR's `LidarData.pose`, so it is the perfect-localization "no SLAM" baseline — contrast `slam_live.py`), `detect_objects.py` (YOLO26 object detection on the drone's own FPV camera "0", while it yaw-scans in place), `segment_objects.py` (same setup, YOLO26 `-seg` instance segmentation instead of boxes), `spawn_traffic.py` (scatters static car/prop assets via `simSpawnObject` so there's something for YOLO to find — AirSimNH has no built-in moving traffic), `segment_sam3.py` (open-vocabulary video segmentation + tracking on the FPV feed via Meta SAM 3's `Sam3VideoModel`/`Sam3VideoProcessor`, transformers>=5, text-prompted concepts instead of YOLO's fixed 80 classes, tracks object identity across frames via a streaming `Sam3VideoInferenceSession`; weights loaded locally from `models/sam_3`, gitignored — no auto-download), `follow_road.py` (steers by yaw-rate to keep a road centered under camera "3"/bottom_center, via HSV color thresholding, not ML — run with `--calibrate` first to tune `ROAD_HSV_LOW/HIGH` for the loaded environment), `autonomous_navigate.py` (take off to 5 m and fly straight ahead at constant body-frame speed, forever — no sensors, no ML; it is also the constants module `lidar_viz.py` imports `DRONE`/`ALTITUDE`/`SPEED` from), `record_dataset.py` (flies the benchmark circuit under one weather condition and records a synchronized LiDAR+stereo+IMU+ground-truth dataset), `slam_live.py` (live SLAM with a 3-panel Open3D viewer: camera feed, map built from ESTIMATED poses, estimated-vs-ground-truth trajectory), `swarm_comms.py` (`SwarmPositions` — shared-GPS swarm coordination for Drone1-4: since AirSim has no radio/RF sim, "communicating" position is just polling every drone's GPS through the one client already used to control the swarm — perfect/instant, not a simulated channel; run standalone for a 4-drone takeoff + live position-share + land demo), `swarm_converge.py` (shrinks the 4-drone formation onto a virtual center — Drone1's live position, read through `swarm_comms.SwarmPositions` — via a uniform scale-down of each drone's spawn offset, so every pair keeps a constant distance RATIO throughout; the scale floors before reaching zero so the swarm settles a few meters apart around Drone1 instead of colliding with it; exposes `run(client)` for reuse), `swarm_converge_viz.py` (5-window live view of the same maneuver: one chase-cam feed per drone via `ChaseCam`/`ChaseCam2`/`ChaseCam3`/`ChaseCam4` in `settings.json`, plus a shared Open3D LiDAR map merging all 4 drones' scans, color-coded per drone and shaded dark-to-light by height (NED z, fixed range) so vertical structure is visible, with a top-right rotation panel — manual azimuth slider, "Auto-rotate 360°" checkbox, adjustable speed — orbiting a fixed elevation/radius around the formation center) |
| `slam/` | the SLAM package (CPU-only, no new deps): `config.py` (mirrors `settings.json` + all tunables + the weather grid), `geometry.py` (SE(3)/NED helpers, Umeyama, TUM I/O), `source.py` (`DatasetSource`/`LiveSource` — same `Frame` stream offline and live), `recorder.py` (deterministic `simPause`+`simContinueForTime` lockstep capture), `weather.py`, `imu.py` (strapdown prior + LiDAR de-skew), `mapping.py` (sliding submap + batched voxel map), `backend.py` (**shared** pose graph + loop closure), `lidar_slam.py`, `stereo_slam.py`, `degradation.py` (adverse-weather LiDAR model), `evaluate.py` (ATE/RPE/map metrics). See `docs/SLAM.md` |
| `tools/` | `probe_setup.py` (Phase-0 sanity check against the live sim — **run first** after any `settings.json` change), `verify_dataset.py`, `run_benchmark.py` (offline cross product of conditions × methods × raw/degraded → `results/`), `synthetic_dataset.py` (raycast fixture world producing the same on-disk format, so the whole offline pipeline can be tested with no simulator) |
| `rl/` | reinforcement learning: `airsim_gym_env.py` (Gymnasium env, ring-orbit formation-keeping), `train_drl.py` (PPO training entry point) |
| `docs/` | `SETUP_MANUAL.md` (host setup), `user_manual.md` (day-to-day usage), `SLAM.md` (the SLAM study: design, how to run, degradation model, verified numbers, known limitations) |
| `scripts/` | `run_swarm.sh` + `fetch_envs.sh` (see above) plus one-time host setup scripts (`setup_arch.sh`, `install_arch.sh`, `setup_ubuntu.sh`, `install_ubuntu.sh`) |
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
```
Note `flight/swarm_demo.py` (Drone1-3) and `flight/swarm_circle.py` (Drone1-5) will fail against the
current single-drone `settings.json` — re-add those vehicles first if you want them.
RPC server: `127.0.0.1:41451`. First boot is slow (UE4 shader compile) — the API server can take
~30-60s after the window/process appears.

## Conventions / gotchas for edits
- Container must run with `--runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=all --net=host`.
- Never drop `-vulkan` from the launch (OpenGL is broken in this build).
- `settings.json` is only read at sim **startup** — restart `run_swarm.sh` after editing it (e.g. after
  adding/moving an `ExternalCameras` entry).
- To resize the swarm: add vehicles in `settings.json` AND update the `DRONES`/`SPAWNS` list in the
  relevant `flight/*.py` script (names must match). Space spawn positions apart (X/Y) so drones don't
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

- Do not add co author to commits