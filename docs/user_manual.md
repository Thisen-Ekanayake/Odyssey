# AirSim Drone-Swarm Simulation — User Manual

Step-by-step guide to run the complete 3-drone swarm simulation on this machine.

Everything lives in `~/airsim_swarm/`. You need **two terminals**:
- **Terminal A** runs the simulator (the Blocks world, inside Docker).
- **Terminal B** runs the Python script that commands the drones.

---

## 0. Prerequisites (one-time check)

These are already set up on this machine, but verify if something breaks later:

```bash
docker --version                      # Docker present
nvidia-smi                            # GPU visible on the host (RTX 4060)
docker images | grep airsim_swarm     # the fixed image 'airsim_swarm:vk' exists
groups | grep docker                  # your user is in the 'docker' group
```

If `airsim_swarm:vk` is **missing**, build it once (takes ~1 min):

```bash
cd ~/airsim_swarm
docker build -f Dockerfile.vk -t airsim_swarm:vk .
```

> Why: the original `airsim_binary` image lacks the Vulkan loader. `airsim_swarm:vk`
> adds it. Without it the simulator crashes on startup.

---

## 1. Start the simulator (Terminal A)

```bash
cd ~/airsim_swarm
./scripts/run_swarm.sh
```

What happens:
- A **Blocks** window opens on your desktop showing 3 drones on the ground.
- The script automatically handles the GPU, Vulkan, and X11 (display) setup.

**Be patient on first launch:** Unreal compiles shaders the first time, so the window
may sit on a loading screen and the control API may take **~30–60 seconds** to become ready.

The simulator is ready when you can connect to it on `127.0.0.1:41451` (the next step
will simply succeed instead of timing out).

### Prefer no window? (headless)
For batch/research runs without a GUI, start it like this instead:

```bash
HEADLESS=1 ./scripts/run_swarm.sh
```

The simulator renders off-screen; you still control the drones exactly the same way.

### Running the four-drone demos? Use the low-resolution rig

```bash
PROFILE=swarm ./scripts/run_swarm.sh AirSimNH
```

The default (`PROFILE=slam`, `settings.json`) is the full-resolution rig the SLAM study needs.
Four drones plus four chase cameras on it drag AirSim's clock to roughly **half real time**, which
is worse than it sounds — the ROS bridge stamps sensor data with AirSim's clock, so a slow sim
clock has broken the cooperative-mapping demo outright in the past. `PROFILE=swarm` uses
`settings.swarm.json` (640×360 stereo, 100k LiDAR points/sec) and keeps the sim responsive.

The rule of thumb: **`slam` for anything that records or measures, `swarm` for anything you watch.**
The script prints which profile it mounted.

---

## 2. Fly the swarm (Terminal B)

Open a **second** terminal (leave Terminal A running):

```bash
cd ~/airsim_swarm
./airsim_venv/bin/python swarm/swarm_demo.py
```

Expected output:

```
Connected!
Connected to AirSim.
Taking off...
Forming up...
Holding formation 5s...
Landing...
Done.
```

In the GUI window you'll see all three drones take off together, spread into a line,
hover, and land. That's the full pipeline working.

---

## 3. Stop the simulation

- **Terminal B**: the script exits on its own when the demo finishes.
- **Terminal A**: press **`Ctrl+C`**, or close the Blocks window. The container is
  started with `--rm`, so it cleans itself up automatically.

If a container ever gets stuck, force-remove it:

```bash
docker ps                       # find the container name/ID
docker rm -f <name-or-id>
```

---

## 4. Customize the swarm

### Change the number of drones

> There are **two** rig files and they must be kept in step: `settings.json` (the SLAM rig,
> 1280×720 stereo / 300k pts-per-sec, the default) and `settings.swarm.json` (640×360 / 100k, the
> low-resolution rig the four-drone demos use, selected with
> `PROFILE=swarm ./scripts/run_swarm.sh`). Add or move a vehicle in one and you must do the same in
> the other, or the two profiles will spawn different swarms.

1. Edit `settings.json` — add/remove vehicles under `"Vehicles"`. Give each a unique
   name and spawn position (`X`/`Y` in meters) so they don't overlap:
   ```json
   "Drone4": {
     "VehicleType": "SimpleFlight", "DefaultVehicleState": "Armed",
     "AutoCreate": true, "AllowAPIAlways": true,
     "X": 0, "Y": 12, "Z": 0, "Yaw": 0
   }
   ```
2. Make the same edit in `settings.swarm.json`.
3. Edit `swarm/swarm_demo.py` — add the same name to the `DRONES` list:
   ```python
   DRONES = ["Drone1", "Drone2", "Drone3", "Drone4"]
   ```
4. Restart the simulator (Terminal A) for `settings.json` changes to take effect.

### Write your own flight logic
Copy `swarm/swarm_demo.py` and use the AirSim Python API. Core calls:
```python
import airsim
c = airsim.MultirotorClient()        # connects to 127.0.0.1:41451
c.confirmConnection()
c.enableApiControl(True, "Drone1")
c.armDisarm(True, "Drone1")
c.takeoffAsync(vehicle_name="Drone1").join()
c.moveToPositionAsync(10, 0, -8, 5, vehicle_name="Drone1").join()  # x,y,z(NED),speed
c.landAsync(vehicle_name="Drone1").join()
```
> Note: positions are **NED** — `Z` is negative for altitude (e.g. `-8` = 8 m up).
> Each drone's coordinates are relative to its own spawn point.

---

## 5. Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| Sim window flashes and closes instantly | Missing `-vulkan` or wrong image. Use `./scripts/run_swarm.sh` (it sets both). |
| `airsim_swarm:vk` not found | Build it: `docker build -f Dockerfile.vk -t airsim_swarm:vk .` |
| Python client times out / "connection refused" | Sim not ready yet — wait ~60 s after launch, then retry. Confirm Terminal A is still running. |
| No GUI window appears | X11/Wayland permission. Re-run `xhost +local:root`, or just use `HEADLESS=1 ./scripts/run_swarm.sh`. |
| `ModuleNotFoundError: airsim` | Use the venv: `./airsim_venv/bin/python`, not the system `python`. |
| Drones collide at spawn | Spread their `X`/`Y` spawn positions in `settings.json`. |

**Harmless** log messages you can ignore: `ALSA: Couldn't open audio device` (no sound
card in the container) and `LogStreaming: Error` about editor-only assets.

---

## Quick reference

```bash
# Terminal A — start sim
cd ~/airsim_swarm && ./scripts/run_swarm.sh            # GUI
cd ~/airsim_swarm && HEADLESS=1 ./scripts/run_swarm.sh # headless

# Terminal B — control drones
cd ~/airsim_swarm && ./airsim_venv/bin/python swarm/swarm_demo.py

# Rebuild image (only if missing)
docker build -f Dockerfile.vk -t airsim_swarm:vk .
```
