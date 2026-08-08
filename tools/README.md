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
| `window_recorder.py` | not run directly — imported for its side effect. See below. |
| `scene_rotation.py` | not run directly — imported by LiDAR/SLAM map viewers. See below. |
| `render_frames.sh` | manual retry: renders `<run-dir>/<N>/frame_*.jpg` folders into `<N>.mp4` |
| `merge_clips.py` | merges 5 clips into one 1920x1440 grid (1 large + 4 small) via ffmpeg |
| `to_gif.sh` | converts an mp4 to a high-quality GIF via two-pass ffmpeg palette (palettegen/paletteuse) |

## Run

```bash
./airsim_venv/bin/python tools/<script>.py [args]
```

## Common instructions

- Needs the sim up: `probe_setup.py`, `spawn_traffic.py`.
- Fully offline (no sim needed): `run_benchmark.py`, `verify_dataset.py`, `synthetic_dataset.py`,
  `view_square_map.py`, `densify_map.py`, `render_frames.sh`, `merge_clips.py`, `to_gif.sh`.
- `view_square_map.py`/`densify_map.py` default to the latest run under `datasets_square/` — pass
  `--dir <path>` to target a specific one.

## `window_recorder.py` — automatic window capture

Any script that does `from tools import window_recorder` gets every GUI window it opens (cv2
`imshow` windows, Open3D viewers) recorded to video automatically — nothing else to call. On
import it starts a background thread that watches for X11 windows owned by the current process
(matched by `_NET_WM_PID`, not by title, so it needs no naming convention and picks up windows
opened at any point in the run), grabs each one's frames as JPEGs at 30 fps into
`recordings/<run-timestamp>/<N>/` (`N` = 1, 2, 3... in discovery order — the number carries no
meaning beyond "which window"), and registers an `atexit` hook that, once the process is exiting
(i.e. every window it owned is already closed), renders each `<N>/` folder into a 1920x1080
H.264 `<N>.mp4` next to it — letterboxed, since windows are captured at their own native size
(e.g. a 640x360 chase-cam tile), not forced to 1080p on screen — then deletes that `<N>/` frames
folder, since a run can leave thousands of JPEGs per window and they're pure disk cost once the
mp4 exists. Only deleted on a successful render; a folder whose encode failed keeps its frames
so it can be retried or debugged.

Already wired into every script that opens a window: `flight/{segment_objects,slam_live,
lidar_viz,segment_sam3,follow_road,detect_objects}.py`, `swarm/{swarm_converge_viz,
swarm_edge_to_center_viz,swarm_lines_viz}.py`, `tools/view_square_map.py`. Add the same
`from tools import window_recorder` line (after the usual `sys.path.insert(0, repo_root)`) to any
new script that opens one.

`recordings/` is gitignored, like `datasets/`/`results/`. Needs `python-xlib` (in `airsim_venv`)
plus `ffmpeg` and ImageMagick's `import` on `PATH`; missing any of them just disables recording
for that run rather than breaking the script. 30 fps is a deliberate choice, not a default that
happened to be there — see the comment above `FPS` in `window_recorder.py` before raising it,
since window capture reads pixels back off the same GPU AirSim's UE4 renderer is using.

## `scene_rotation.py` — shared map-rotation panel

`RotationPanel` is the top-right "Desmos-style" 3D-view control (manual azimuth slider,
"Auto-rotate 360°" checkbox, speed slider, orbiting a fixed elevation/radius around a scene's
center) that originally lived only in `swarm_converge_viz.py`. Construct it right after a
SceneWidget's own `setup_camera()` call, then call `.layout(rect)` from that window's
`set_on_layout` once the widget's own frame is set:

```python
self.rotation = scene_rotation.RotationPanel(
    self.win, self.widget, bounds, is_running=lambda: self._running)
...
self.rotation.layout(self.widget.frame)   # in set_on_layout, after widget.frame is set
```

Wired into every SceneWidget that shows sensor-derived data — a LiDAR point-cloud map or a SLAM
map: `flight/lidar_viz.py`, `flight/slam_live.py` (the map panel only, not the estimated-vs-GT
trajectory panel), `swarm/swarm_converge_viz.py`, `swarm/swarm_edge_to_center_viz.py` (the
merged LiDAR map window only, not the edge-to-center segmentation-lines window). Deliberately
**not** wired into chase-cam feeds (plain image widgets, no camera to orbit) or into
formation-geometry-only scenes with no sensor data (`swarm/swarm_lines_viz.py`, and the
segmentation-lines widget inside `swarm_edge_to_center_viz.py`). `tools/view_square_map.py` also
doesn't get it — it uses the older `o3d.visualization.Visualizer` API, not the `gui.Application`/
`SceneWidget` framework this panel attaches to, and already has its own mouse-drag orbit.
