# tools/

Offline/utility scripts — none of these fly a drone maneuver of their own.

| Script | What it does |
|---|---|
| `probe_setup.py` | Phase-0 sanity check against the live sim — run first after any `settings.json` change |
| `verify_dataset.py` | validates a recorded dataset's structure/contents |
| `run_benchmark.py` | offline cross product of weather × method × raw/degraded → `results/` CSV + figures |
| `synthetic_dataset.py` | raycast fixture world producing the same on-disk format, no simulator needed |
| `spawn_traffic.py` | scatters static car/prop assets so `flight/detect_objects.py`'s YOLO has something to find |
| `view_square_map.py` | interactive Open3D viewer over a `flight/square_capture_map.py` recording |
| `densify_map.py` | K-nearest-neighbor gap-filling over a `flight/square_capture_map.py` map |

## Run

```bash
./airsim_venv/bin/python tools/<script>.py [args]
```

## Common instructions

- Needs the sim up: `probe_setup.py`, `spawn_traffic.py`.
- Fully offline (no sim needed): `run_benchmark.py`, `verify_dataset.py`, `synthetic_dataset.py`,
  `view_square_map.py`, `densify_map.py`.
- `view_square_map.py`/`densify_map.py` default to the latest run under `datasets_square/` — pass
  `--dir <path>` to target a specific one.
