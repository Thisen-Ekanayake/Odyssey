# Odyssey

![Python](https://img.shields.io/badge/python-3.10-blue)
![AirSim](https://img.shields.io/badge/AirSim-1.8.1-orange)
![Docker](https://img.shields.io/badge/docker-vulkan--fixed-2496ED?logo=docker&logoColor=white)
![Platform](https://img.shields.io/badge/platform-linux-lightgrey)

A drone swarm research rig built on Microsoft AirSim: multi-drone formation flying, live LiDAR
mapping, a from-scratch SLAM study comparing LiDAR-inertial against stereo-inertial odometry, object
detection, road following, and reinforcement learning, all running in a single Docker image on top of
a UE4 simulator.

## Why this exists

Most AirSim setups run one drone through one demo script. This one asks harder questions instead:
what does a 4-drone swarm look like when it converges on a point, or splits into two-drone
subgroups, and does it need real communication to coordinate, or can shared GPS get you there? If
you drop LiDAR for a stereo camera pair to save cost, how much SLAM accuracy do you actually lose,
and does that answer change in fog?

That last question hides a trap. AirSim's weather is a **rendering** effect: `simSetWeatherParameter`
drives particle systems the cameras see, but the LiDAR is a raycast against collision geometry, and
rain and fog particles have no collision. So LiDAR returns come back identical in every weather
condition, no matter how bad it looks on screen. Believing the simulator here would quietly invalidate
half the study, so the LiDAR side of weather is modelled explicitly instead (`slam/degradation.py`),
and every result reports which half is simulated and which is modelled. See `docs/SLAM.md` for the
full writeup, including verified ATE/RPE numbers and the bugs this work found and fixed along the way.

## What's inside

- **Swarm flight** (`swarm/`): shared-GPS coordination across Drone1-4, ring orbits, formation
  convergence, and a maneuver where each drone flies its own edge of the formation toward a shared
  center, each with a live multi-window Open3D view of chase cams and merged LiDAR maps.
- **SLAM study** (`slam/`, `docs/SLAM.md`): LiDAR-inertial vs. stereo-inertial odometry sharing one
  pose-graph back end, benchmarked across weather conditions with a real degradation model.
- **Perception** (`flight/`): YOLO26 detection and segmentation, Meta SAM 3 open-vocabulary video
  segmentation, and HSV-based road following, all running against the drone's own FPV camera.
- **Reinforcement learning** (`rl/`): a Gymnasium environment and PPO training loop for ring-orbit
  formation-keeping.
- **The Vulkan fix** (`Dockerfile.vk`): the stock AirSim binary image ships CUDA and X11 but no Vulkan
  loader, so under the NVIDIA container runtime it silently falls back to OpenGL and crashes on
  launch. One extra layer (`libvulkan1`) fixes it. Small bug, total blocker until you find it.

## Quickstart

```bash
# one-time host setup (Arch or Ubuntu):
./scripts/install_arch.sh && ./scripts/setup_arch.sh

# download an environment and launch the sim:
./scripts/fetch_envs.sh AirSimNH
./scripts/run_swarm.sh AirSimNH

# in another terminal, fly something:
./airsim_venv/bin/python flight/lidar_viz.py
./airsim_venv/bin/python swarm/swarm_converge_viz.py
```

## Layout

| Directory | Contents |
|---|---|
| `flight/` | single-drone scripts: sensing, LiDAR mapping, detection, segmentation, road following |
| `swarm/` | multi-drone (Drone1-4) formations, convergence, and their live Open3D viewers |
| `slam/` | the SLAM library: LiDAR-inertial and stereo-inertial pipelines, shared pose graph, evaluation |
| `tools/` | offline utilities: dataset verification, benchmarking, map post-processing, scene setup |
| `rl/` | Gymnasium environment and PPO training for formation-keeping |
| `scripts/` | sim launcher, environment downloader, one-time host setup |
| `docs/` | setup manual, day-to-day usage guide, and the full SLAM study writeup |

Each of these has its own short README with a one-line description of every script. `CLAUDE.md`
has the full technical detail: exact sensor rig, coordinate frame gotchas, and conventions for edits.

## Verified working

Headless, end-to-end: multiple drones armed, flew a formation, and landed. LiDAR-inertial and
stereo-inertial SLAM both tracked a closed-loop circuit with sub-meter ATE before loop closure; see
`docs/SLAM.md` for the numbers and for the one known-unstable piece (loop closure itself, currently
stochastic across RANSAC seeds).

## ROS 2

A ROS 2 **Jazzy** layer lives in `ros2_ws/` and runs inside the `ros2` distrobox
container. It bridges all four drones onto standard topics/TF, exposes the
existing `swarm/` manoeuvres as services, and runs `rtabmap_ros` as an
independent LiDAR-SLAM estimator that can be scored against this repo's own
`slam/` package on identical recorded data.

```bash
distrobox enter ros2 -- /ml/airsim_swarm/scripts/ros_setup.sh   # one-time
./scripts/run_swarm.sh AirSimNH                                  # terminal 1
./scripts/ros_enter.sh ros2 launch airsim_swarm_bridge bridge.launch.py
./scripts/ros_enter.sh ros2 launch airsim_swarm_bridge viz.launch.py
# live 4-drone cooperative mapping (all 4 LiDARs merged into one shared octomap):
./scripts/ros_enter.sh ros2 launch airsim_swarm_bridge cooperative_mapping.launch.py
```

See [docs/ROS.md](docs/ROS.md) for the frame conventions, the SLAM comparison
workflow, and why the `airsim` pip package cannot be installed in that container.
