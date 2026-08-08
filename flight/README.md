# flight/

Single-drone AirSim client scripts (Drone1 only).

| Script | What it does |
|---|---|
| `sensor_demo.py` | single-drone sensor readout: camera, depth, LiDAR, IMU, GPS, magnetometer, barometer |
| `lidar_viz.py` | single-drone LiDAR mapping + chase-cam dual view, using the simulator's own pose (no SLAM) |
| `detect_objects.py` | YOLO26 object detection on the FPV camera while yaw-scanning in place |
| `segment_objects.py` | same as `detect_objects.py` but YOLO26 instance segmentation instead of boxes |
| `segment_sam3.py` | open-vocabulary segmentation + tracking on the FPV feed via Meta SAM 3, text-prompted |
| `follow_road.py` | steers by yaw-rate to keep a road centered under the bottom camera, via HSV thresholding (not ML) |
| `autonomous_navigate.py` | takes off to 5 m and flies straight forever, no sensors/ML; also the constants module `lidar_viz.py` imports from |
| `record_dataset.py` | flies the benchmark circuit and records a synchronized LiDAR+stereo+IMU+ground-truth dataset |
| `slam_live.py` | live SLAM with a 3-panel Open3D viewer: camera feed, estimated map, estimated-vs-ground-truth trajectory |
| `square_capture_map.py` | flies a closed square circuit recording LiDAR + ground-truth poses; mapping happens offline after landing |

## Run

```bash
./scripts/run_swarm.sh AirSimNH          # sim must be up first
./airsim_venv/bin/python flight/<script>.py
```

## Common instructions

- Ctrl+C, `q` in the video window, or closing the Open3D window lands the drone and exits.
- `follow_road.py` needs a `--calibrate` pass first to tune HSV thresholds for the loaded environment.
- `slam_live.py` takes `--method lidar|stereo` and `--weather <condition>`.
- `record_dataset.py` takes `--condition clear|...|all`; run `tools/probe_setup.py` first.
