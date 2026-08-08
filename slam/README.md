# slam/

The SLAM package (CPU-only, no new deps). A library, not standalone scripts — imported by
`flight/record_dataset.py`, `flight/slam_live.py`, `tools/probe_setup.py`, `tools/run_benchmark.py`,
`tools/synthetic_dataset.py`. See `docs/SLAM.md` for the full design.

| Module | What it does |
|---|---|
| `config.py` | mirrors `settings.json` + all tunables + the weather grid |
| `geometry.py` | SE(3)/NED helpers, Umeyama, TUM I/O |
| `source.py` | `DatasetSource`/`LiveSource` — same `Frame` stream offline and live |
| `recorder.py` | deterministic `simPause`+`simContinueForTime` lockstep capture |
| `weather.py` | weather condition grid + `simSetWeatherParameter` helpers |
| `imu.py` | strapdown prior + LiDAR de-skew |
| `mapping.py` | sliding submap + batched voxel map |
| `backend.py` | shared pose graph + loop closure |
| `lidar_slam.py` | LiDAR-inertial pipeline: IMU prior → de-skew → GICP scan-to-submap → loop closure |
| `stereo_slam.py` | stereo-inertial pipeline: SGBM disparity → ORB + PnP → windowed BA → loop closure |
| `degradation.py` | adverse-weather LiDAR model |
| `evaluate.py` | ATE/RPE/map metrics |

## Common instructions

- No run commands — this is a library. Change `config.py` alongside `settings.json` together.
- `slam/backend.py`'s `global_optimization` prunes edges from the `PoseGraph` you pass it, so it
  rebuilds a fresh graph per call — don't collapse that into optimizing the same object repeatedly.
- Seed Open3D's RANSAC (`o3d.utility.random.seed(...)`) before any run whose numbers you compare.
