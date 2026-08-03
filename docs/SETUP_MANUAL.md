# AirSim Drone-Swarm Simulation — Setup Manual
**Platform:** Arch Linux · **Renderer:** AirSim v1.8 / Unreal Engine 4 in Docker · **Date:** 2026-06-26

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Host Dependencies](#2-host-dependencies)
3. [Docker Image Details](#3-docker-image-details)
4. [Project Layout](#4-project-layout)
5. [Step-by-Step Setup](#5-step-by-step-setup)
   - 5.1 [Install NVIDIA Drivers and CUDA](#51-install-nvidia-drivers-and-cuda)
   - 5.2 [Install Docker and NVIDIA Container Toolkit](#52-install-docker-and-nvidia-container-toolkit)
   - 5.3 [Configure Docker Runtime](#53-configure-docker-runtime)
   - 5.4 [Set Up XWayland for GUI Output](#54-set-up-xwayland-for-gui-output)
   - 5.5 [Clone / Place the Project](#55-clone--place-the-project)
   - 5.6 [Pull the Base Docker Image](#56-pull-the-base-docker-image)
   - 5.7 [Build the Vulkan-Fixed Image](#57-build-the-vulkan-fixed-image)
   - 5.8 [Download AirSim Environments](#58-download-airsim-environments)
   - 5.9 [Set Up the Python Client Virtualenv](#59-set-up-the-python-client-virtualenv)
6. [Configuration — settings.json](#6-configuration--settingsjson)
7. [Running the Simulation](#7-running-the-simulation)
   - 7.1 [GUI Window Mode](#71-gui-window-mode)
   - 7.2 [Headless Mode](#72-headless-mode)
8. [Demo Scripts](#8-demo-scripts)
   - 8.1 [swarm_demo.py — Line Formation](#81-swarm_demopy--line-formation)
   - 8.2 [swarm_circle.py — Ring Orbit](#82-swarm_circlepy--ring-orbit)
   - 8.3 [sensor_demo.py — Sensor Readout](#83-sensor_demopy--sensor-readout)
   - 8.4 [lidar_viz.py — Live LiDAR Visualizer](#84-lidar_vizpy--live-lidar-visualizer)
9. [Sensor Configuration (LiDAR)](#9-sensor-configuration-lidar)
10. [Expanding the Swarm](#10-expanding-the-swarm)
11. [Troubleshooting / Known Gotchas](#11-troubleshooting--known-gotchas)

---

## 1. System Requirements

| Component | Requirement |
|---|---|
| **OS** | Arch Linux (kernel 6.x; tested on 7.0.5-arch1-1) |
| **GPU** | NVIDIA GPU with Vulkan support (tested: RTX 4060 Laptop) |
| **NVIDIA Driver** | 595+ (host) |
| **CUDA (host)** | 13.2 (runtime only; the container uses CUDA 10.0 libs from the image) |
| **Display** | Wayland session; XWayland on `DISPLAY=:1` for container GUI |
| **RAM** | 16 GB recommended (UE4 + Docker overhead) |
| **Disk** | ~10 GB for the Docker image stack; ~0.14 GB for Blocks env alone, up to ~12 GB for all envs |
| **Internet** | Required for initial image pull and environment downloads |

---

## 2. Host Dependencies

Install the following packages on Arch Linux:

```bash
# Core tools
sudo pacman -S docker git curl unzip python python-pip

# NVIDIA driver (if not already installed)
sudo pacman -S nvidia nvidia-utils

# XWayland (for GUI in a Wayland session)
sudo pacman -S xorg-xwayland xorg-xhost

# NVIDIA Container Toolkit — from AUR
yay -S nvidia-container-toolkit
# or manually:
# https://github.com/NVIDIA/nvidia-container-toolkit
```

> **Python version:** CPython 3.10+ is recommended. The `airsim` pip package requires numpy and has a quirk with build isolation (see [Section 5.9](#59-set-up-the-python-client-virtualenv)).

---

## 3. Docker Image Details

### 3.1 Base Image — `airsim_binary:10.0-devel-ubuntu18.04`

| Field | Value |
|---|---|
| **Source** | `tiryoh/ros-desktop-vnc` derivative / Microsoft AirSim community base |
| **Base OS** | Ubuntu 18.04 (Bionic) |
| **CUDA** | 10.0-devel |
| **Contents** | CUDA runtime, cuDNN stubs, X11 client libs, AirSim runtime dependencies |
| **Missing** | Vulkan loader (`libvulkan1`) — this causes UE4 to fall back to OpenGL, which is broken |
| **User** | `airsim_user` (uid 1000) |

The NVIDIA container runtime injects the **NVIDIA Vulkan ICD** and driver libs at container start, but **not** the Vulkan loader itself. Without the loader, launching with `-vulkan` fails with `"Vulkan Driver is required"`; without `-vulkan`, UE4 tries OpenGL and immediately exits with `"OpenGL is deprecated"`.

### 3.2 Derived Image — `airsim_swarm:vk` *(use this one)*

Built from `Dockerfile.vk` in the project root:

```dockerfile
FROM airsim_binary:10.0-devel-ubuntu18.04

USER root

# Remove expired CUDA 10.0 / NVIDIA apt repos that break `apt-get update`
RUN rm -f /etc/apt/sources.list.d/cuda*.list \
          /etc/apt/sources.list.d/nvidia*.list \
          /etc/apt/sources.list.d/*ml*.list 2>/dev/null; \
    apt-get update && \
    apt-get install -y --no-install-recommends libvulkan1 vulkan-utils && \
    rm -rf /var/lib/apt/lists/*

USER airsim_user
```

| Field | Value |
|---|---|
| **Tag** | `airsim_swarm:vk` |
| **Adds** | `libvulkan1`, `vulkan-utils` |
| **Removes** | Expired NVIDIA/CUDA apt repo lists that prevented `apt-get update` |
| **Result** | Full Vulkan pipeline: loader (image) + ICD + driver libs (injected by runtime) |

> The image ships **no simulator binary**. The Blocks (or other) environment is downloaded separately and bind-mounted in at runtime.

---

## 4. Project Layout

```
airsim_swarm/
├── Dockerfile.vk           # Builds airsim_swarm:vk
├── settings.json           # AirSim vehicle/sensor config (bind-mounted into container)
├── flight/
│   ├── swarm_demo.py       # 3-drone line-formation demo
│   ├── swarm_circle.py     # 5-drone ring-orbit demo
│   ├── sensor_demo.py      # Single-drone sensor readout (camera, LiDAR, IMU, GPS, ...)
│   ├── lidar_viz.py        # Single-drone LiDAR mapping + chase-cam dual view (Open3D)
│   ├── detect_objects.py   # YOLO26 object detection on the drone's FPV camera
│   └── spawn_traffic.py    # Scatters static car/prop assets for detect_objects.py to find
├── rl/
│   ├── airsim_gym_env.py   # Gymnasium environment wrapper
│   └── train_drl.py        # DRL (PPO) training entry point
├── docs/                   # This manual + user_manual.md
├── scripts/                # run_swarm.sh, fetch_envs.sh, and one-time host setup scripts
│   ├── run_swarm.sh        # Main launcher script
│   ├── fetch_envs.sh       # Downloads AirSim environments from GitHub releases into envs/
│   └── setup_arch.sh, install_arch.sh, setup_ubuntu.sh, install_ubuntu.sh
├── airsim_venv/            # Python virtualenv (airsim 1.8.1, numpy 2.2.6, torch, ultralytics)
└── envs/                   # Downloaded environment binaries (gitignored)
    ├── Blocks/LinuxNoEditor/
    ├── Africa_Savannah/LinuxNoEditor/
    ├── ZhangJiajie/LinuxNoEditor/
    └── AirSimNH/LinuxNoEditor/
```

---

## 5. Step-by-Step Setup

### 5.1 Install NVIDIA Drivers and CUDA

```bash
# Check if the NVIDIA driver is already loaded
nvidia-smi

# If not installed:
sudo pacman -S nvidia nvidia-utils lib32-nvidia-utils

# Reboot to load the kernel module
sudo reboot

# Verify after reboot
nvidia-smi
```

Expected output of `nvidia-smi` should show your GPU (e.g., RTX 4060 Laptop), driver version (595+), and CUDA version.

### 5.2 Install Docker and NVIDIA Container Toolkit

```bash
# Install Docker
sudo pacman -S docker

# Enable and start Docker daemon
sudo systemctl enable --now docker

# Add your user to the docker group (log out/in after)
sudo usermod -aG docker $USER

# Install nvidia-container-toolkit from AUR
yay -S nvidia-container-toolkit
```

### 5.3 Configure Docker Runtime

Register the NVIDIA runtime with Docker and generate the CDI spec:

```bash
# Generate CDI device spec (creates /etc/cdi/nvidia.yaml)
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

# Register nvidia as a Docker runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker to pick up the new runtime
sudo systemctl restart docker

# Verify: this should print GPU info from inside a container
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all nvidia/cuda:12.0-base-ubuntu20.04 nvidia-smi
```

Optionally, change Docker's data root to avoid filling the system partition (the project uses `/home/docker`):

```bash
# /etc/docker/daemon.json
{
  "data-root": "/home/docker",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
```

```bash
sudo systemctl restart docker
```

### 5.4 Set Up XWayland for GUI Output

AirSim's UE4 binary is an X11 application. On a Wayland desktop it must go through XWayland.

```bash
# Confirm XWayland is running and which DISPLAY it's on
echo $DISPLAY        # should be :1 (or :0 on some setups)
xdpyinfo -display :1 | head -5   # confirm it's alive

# Allow local root (Docker) to connect to your X server
xhost +local:root
```

The `run_swarm.sh` script calls `xhost +local:root` automatically before launching the container, so you only need to do this manually if testing things outside the script.

### 5.5 Clone / Place the Project

```bash
# Place the project wherever you like, e.g.:
cd ~
git clone <your-repo-url> airsim_swarm
cd airsim_swarm
```

### 5.6 Pull the Base Docker Image

```bash
docker pull ghcr.io/microsoft/airsim/airsim_binary:10.0-devel-ubuntu18.04
# Tag it to the name the Dockerfile expects:
docker tag ghcr.io/microsoft/airsim/airsim_binary:10.0-devel-ubuntu18.04 \
           airsim_binary:10.0-devel-ubuntu18.04
```

> If the image is only available as a `.tar` archive, load it with:
> ```bash
> docker load -i airsim_binary.tar
> ```

### 5.7 Build the Vulkan-Fixed Image

This step adds the missing Vulkan loader on top of the base image.

```bash
cd ~/airsim_swarm
docker build -f Dockerfile.vk -t airsim_swarm:vk .
```

Expected output ends with `Successfully tagged airsim_swarm:vk`. Rebuild any time `Dockerfile.vk` changes.

Verify the image exists:

```bash
docker images | grep airsim_swarm
# airsim_swarm   vk   <id>   <size>
```

### 5.8 Download AirSim Environments

Environments are downloaded from the [AirSim v1.8.1 GitHub release](https://github.com/microsoft/AirSim/releases/tag/v1.8.1). The `fetch_envs.sh` helper handles this:

```bash
# Download the lightweight Blocks environment (~140 MB) only:
./scripts/fetch_envs.sh Blocks

# Download all available environments (~12 GB total):
./scripts/fetch_envs.sh

# Skip confirmation and keep zip archives:
YES=1 KEEP_ZIPS=1 ./scripts/fetch_envs.sh Africa_Savannah Blocks
```

Available environments:

| Name | Download Size |
|---|---|
| `Blocks` | 0.14 GB |
| `Africa_Savannah` | 1.11 GB |
| `MSBuild2018` | 0.79 GB |
| `ZhangJiajie` | 0.88 GB |
| `LandscapeMountains` | 1.15 GB |
| `AbandonedPark` | 1.66 GB |
| `AirSimNH` | 2.11 GB |
| `TrapCamera` | 3.85 GB (2-part archive) |

After extraction each environment lives at `<Env>/LinuxNoEditor/<Env>.sh`.

### 5.9 Set Up the Python Client Virtualenv

The `airsim` pip package has a build-isolation bug: it must find `numpy` before it builds its C extension. Install them in order:

```bash
cd ~/airsim_swarm

# Create the venv
python -m venv airsim_venv

# Install numpy first
airsim_venv/bin/pip install numpy

# Then install airsim without build isolation
airsim_venv/bin/pip install airsim --no-build-isolation

# Optional extras for the LiDAR visualizer
airsim_venv/bin/pip install open3d
```

Installed versions (verified):

| Package | Version |
|---|---|
| `airsim` | 1.8.1 |
| `numpy` | 2.2.6 |
| `open3d` | (latest compatible) |

---

## 6. Configuration — `settings.json`

`settings.json` is bind-mounted into the container at `/home/airsim_user/Documents/AirSim/settings.json`. AirSim reads it on startup.

Current configuration: **5-drone SimpleFlight swarm** with a 16-channel LiDAR on each drone, spawned in a line along the Y-axis (4 m spacing):

```json
{
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "ClockSpeed": 1.0,
  "ViewMode": "SpringArmChase",
  "Vehicles": {
    "Drone1": { "VehicleType": "SimpleFlight", "X": 0, "Y": 0,  "Z": 0, ... },
    "Drone2": { "VehicleType": "SimpleFlight", "X": 0, "Y": 4,  "Z": 0, ... },
    "Drone3": { "VehicleType": "SimpleFlight", "X": 0, "Y": 8,  "Z": 0, ... },
    "Drone4": { "VehicleType": "SimpleFlight", "X": 0, "Y": 12, "Z": 0, ... },
    "Drone5": { "VehicleType": "SimpleFlight", "X": 0, "Y": 16, "Z": 0, ... }
  }
}
```

LiDAR sensor on each drone (`SensorType: 6`):

| Parameter | Value |
|---|---|
| Channels | 16 |
| Rotations/s | 10 |
| Points/s | 100 000 |
| Offset (body frame) | Z = −0.1 m (just above CoM) |
| Data frame | `VehicleInertialFrame` |

> After **any** change to `settings.json`, restart the simulation container. AirSim only reads settings at startup.

---

## 7. Running the Simulation

### 7.1 GUI Window Mode

```bash
cd ~/airsim_swarm

# Blocks environment (default):
./scripts/run_swarm.sh Blocks

# Africa Savannah:
./scripts/run_swarm.sh Africa_Savannah

# Any env in a custom resolution:
./scripts/run_swarm.sh Blocks -ResX=1920 -ResY=1080
```

The script:
1. Calls `xhost +local:root` so the container can reach XWayland.
2. Runs `docker run --rm -it` with `--runtime=nvidia`, `--net=host`, bind-mounts for `settings.json` and the environment directory.
3. Passes `-vulkan -windowed -ResX=1280 -ResY=720` to the UE4 binary.

The UE4 window appears on `DISPLAY=:1` (XWayland). **First launch compiles shaders** — this takes 2–5 minutes and the window may appear frozen. The API server (`127.0.0.1:41451`) becomes ready 30–60 s after the window appears.

### 7.2 Headless Mode

No window is created; rendering goes off-screen. Useful for CI, data collection, or remote servers.

```bash
HEADLESS=1 ./scripts/run_swarm.sh Blocks
```

Passes `-vulkan -RenderOffscreen -ResX=640 -ResY=480` to UE4. The API is identical — connect with the Python client the same way.

---

The full `docker run` command issued by `run_swarm.sh` for reference:

```bash
docker run --rm -it \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e DISPLAY=:1 \
  -e QT_X11_NO_MITSHM=1 \
  -e SDL_VIDEODRIVER=x11 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --net=host \
  -v /path/to/settings.json:/home/airsim_user/Documents/AirSim/settings.json:ro \
  -v /path/to/Blocks/LinuxNoEditor:/home/airsim_user/Blocks:rw \
  airsim_swarm:vk \
  bash -lc "/home/airsim_user/Blocks/Blocks.sh -vulkan -windowed -ResX=1280 -ResY=720"
```

---

## 8. Demo Scripts

All scripts run **on the host** (not inside the container). They connect to the simulator over RPC via `127.0.0.1:41451` (exposed by `--net=host`).

### 8.1 `swarm_demo.py` — Line Formation

3-drone demo: arm → take off together → spread into a 5 m-spaced line along X → hover 5 s → land.

```bash
./airsim_venv/bin/python flight/swarm_demo.py
```

Drone names used: `Drone1`, `Drone2`, `Drone3` (must match `settings.json`). Formation altitude: −8 m NED (8 m above ground).

### 8.2 `swarm_circle.py` — Ring Orbit

5-drone ring orbit: arm → take off → spread into a 10 m-radius ring → orbit the center for 2 revolutions (20 s each) with velocity-controlled closed-loop tracking → land.

```bash
./airsim_venv/bin/python flight/swarm_circle.py
```

Key parameters (editable at top of file):

| Variable | Default | Description |
|---|---|---|
| `RADIUS` | 10.0 m | Ring radius |
| `ALTITUDE` | −8.0 m NED | Flying altitude |
| `REV_PERIOD` | 20.0 s | Time for one full revolution |
| `N_REVS` | 2 | Number of revolutions |
| `DIRECTION` | +1 | +1 = CCW, −1 = CW (viewed from above) |
| `FACE_CENTER` | True | Drones yaw to face ring center while orbiting |

Uses `moveByVelocityZAsync` with a P-gain radial correction to maintain the ring shape without stop-start jitter.

### 8.3 `sensor_demo.py` — Sensor Readout

Single-drone demo that takes off, hovers, and reads all available sensors once:

- RGB camera (scene)
- Depth camera (planar)
- LiDAR (`LidarSensor1`)
- IMU (angular velocity, linear acceleration)
- GPS (lat/lon/alt + velocity)
- Magnetometer
- Barometer / altimeter

```bash
./airsim_venv/bin/python flight/sensor_demo.py
```

Requires `LidarSensor1` configured on `Drone1` in `settings.json` (already done).

### 8.4 `lidar_viz.py` — LiDAR Mapping + Chase-Cam Viewer

Single-drone (`Drone1`) split-window viewer using **Open3D** (GPU renderer via Filament/Vulkan):
left panel is a live feed from the `ChaseCam` external camera (trails behind the drone's direction
of travel), right panel is a persistent, voxel-downsampled LiDAR point-cloud map that accumulates as
the drone flies, colored blue (high) → red (ground) by a fixed altitude range. The drone flies the
whole waypoint pattern in one `moveOnPathAsync` call (smooth cornering via lookahead, no stop-start
jerk) in a background thread while the viewer is open. Close the window to land.

```bash
./airsim_venv/bin/python flight/lidar_viz.py
```

Requires `open3d` installed in the venv:

```bash
airsim_venv/bin/pip install open3d
```

Requires the `ChaseCam` external camera and `CameraDefaults` block in `settings.json` (already
configured) — **restart the sim** after any `settings.json` change, since it's only read at startup.

> Open3D's GLFW backend is forced onto XWayland (`DISPLAY=:1`) at the top of `lidar_viz.py` to avoid Wayland compositor issues.

---

## 9. Sensor Configuration (LiDAR)

The LiDAR is configured per-vehicle in `settings.json` under the `Sensors` key. Current spec for all 5 drones:

```json
"LidarSensor1": {
  "SensorType": 6,
  "Enabled": true,
  "NumberOfChannels": 16,
  "RotationsPerSecond": 10,
  "PointsPerSecond": 100000,
  "X": 0, "Y": 0, "Z": -0.1,
  "DrawDebugPoints": false,
  "DataFrame": "VehicleInertialFrame"
}
```

`SensorType: 6` = LiDAR in AirSim's sensor enum. `VehicleInertialFrame` means points are returned relative to the vehicle's inertial frame origin (spawn position), not the body frame.

To enable debug point rendering in the sim window, set `"DrawDebugPoints": true` and restart.

---

## 10. Expanding the Swarm

To add more drones:

**1. Edit `settings.json`** — add a new vehicle block, spacing its Y position 4 m further than the last:

```json
"Drone6": {
  "VehicleType": "SimpleFlight",
  "DefaultVehicleState": "Armed",
  "EnableCollisions": true,
  "AllowAPIAlways": true,
  "AutoCreate": true,
  "X": 0, "Y": 20, "Z": 0, "Yaw": 0,
  "Sensors": {
    "LidarSensor1": { ... }
  }
}
```

**2. Update the Python script** — add `"Drone6"` to the `DRONES` list (or `SPAWNS` dict for scripts that track world positions):

```python
# swarm_circle.py
SPAWNS = {
    ...
    "Drone6": (0.0, 20.0),
}
```

**3. Restart the container** — `settings.json` is only read at startup.

> Spawn positions must be spaced apart (at least 2–3 m in X or Y) to prevent collision at startup.

---

## 11. Troubleshooting / Known Gotchas

### UE4 exits immediately — "OpenGL is deprecated"

You are launching with the base image (`airsim_binary:10.0-devel-ubuntu18.04`) instead of `airsim_swarm:vk`. The UE4 binary defaulted to OpenGL because the Vulkan loader was missing.

**Fix:** Build and use `airsim_swarm:vk` (`docker build -f Dockerfile.vk -t airsim_swarm:vk .`), and always pass `-vulkan` to the UE4 binary.

### "Vulkan Driver is required" on launch

The NVIDIA runtime did not inject the driver libs. Common causes:
- `--runtime=nvidia` missing from `docker run`
- `NVIDIA_DRIVER_CAPABILITIES=all` env var missing
- `nvidia-container-toolkit` not installed or the Docker daemon not restarted after configuration

**Fix:** Verify the runtime (`docker info | grep Runtimes`) and re-run `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`.

### Cannot connect to X server / black window

The container cannot reach XWayland.

```bash
# Allow Docker (root) to use your display
xhost +local:root

# Confirm DISPLAY is set correctly on the host
echo $DISPLAY   # should be :1

# Check XWayland is alive
xdpyinfo -display :1 | head -3
```

### Python RPC connection refused

The API server takes 30–60 s after the UE4 window appears (shader compilation). Wait and retry. In headless mode there is no visual cue — add a retry loop or `time.sleep(60)` before `confirmConnection()`.

```python
import time, airsim
client = airsim.MultirotorClient()
for _ in range(30):
    try:
        client.confirmConnection(); break
    except Exception:
        time.sleep(2)
```

### `pip install airsim` fails with build isolation error

```bash
# Install numpy first, then airsim without build isolation
airsim_venv/bin/pip install numpy
airsim_venv/bin/pip install airsim --no-build-isolation
```

### `apt-get update` fails inside the base image

The base image ships expired CUDA 10.0 apt repo lists with invalid GPG keys. The `Dockerfile.vk` removes these before calling `apt-get update`. If you need to add more packages, follow the same pattern:

```dockerfile
RUN rm -f /etc/apt/sources.list.d/cuda*.list \
          /etc/apt/sources.list.d/nvidia*.list && \
    apt-get update && apt-get install -y <package>
```

### Harmless log noise to ignore

| Message | Cause | Action |
|---|---|---|
| `ALSA lib: Couldn't open audio device` | No sound card in container | Ignore |
| `LogStreaming: Warning: ...` (editor assets) | UE4 editor assets not included in the binary build | Ignore |
| `LogInit: Warning: ... OpenGL` | Suppressed once `-vulkan` is passed | Ignore if UE4 starts normally |

---

*Generated from live project state — `/home/thisen-ekanayake/airsim_swarm/`*
