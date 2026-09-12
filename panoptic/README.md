# panoptic/ — 3-D panoptic map of the environment, then a car that drives on it

Implements `project_idea.md`: one drone takes off, covers the whole environment
corner to corner, maps it with a segmentation camera + LiDAR, the recording is
fused into a **3-D panoptic map** (every voxel has a class *and* an instance id),
and a car spawned on a road drives through the env using only that map — never
leaving the road and never touching a building, tree, pole or parked car.

Everything lives in this folder; the only thing outside it is the
`PROFILE=panoptic` case in `scripts/run_swarm.sh` and a `.gitignore` line.

## Run

```bash
# 1. boot the sim with the panoptic rig (single drone, nadir seg/depth camera, CarCam)
PROFILE=panoptic ./scripts/run_swarm.sh AirSimNH

# 2. fly + record   (~10-15 min for AirSimNH at 35 m / 6 m/s; --dry-run to see the plan first)
./airsim_venv/bin/python panoptic/capture.py
./airsim_venv/bin/python panoptic/capture.py --list-objects     # audit the name -> class mapping
./airsim_venv/bin/python panoptic/capture.py --bounds -200 200 -200 200   # override the extent

# 3. fuse into the panoptic map (offline, ~1 min)
./airsim_venv/bin/python panoptic/fuse.py

# 4. look at it (C class / I instance / H height / D drivable)
./airsim_venv/bin/python panoptic/view_map.py

# 5. spawn a car and drive it through the map (sim still running)
./airsim_venv/bin/python panoptic/drive_car.py                 # auto: two far-apart road points
./airsim_venv/bin/python panoptic/drive_car.py --start 10 -40 --goal 120 90 --chase
./airsim_venv/bin/python panoptic/drive_car.py --dry-run       # no sim: route + animation on the map

# offline self-test of fuse + planner on a synthetic block world (no sim, ~40 s)
./airsim_venv/bin/python panoptic/test_offline.py
```

## How it works

| Step | File | What |
|---|---|---|
| labels | `labels.py`, `config.py` | Every `simListSceneObjects()` name is classified by keyword (`CLASS_KEYWORDS`) into 5 *stuff* classes (road, sidewalk, terrain, water, other) and 7 *thing* classes (building, vehicle, tree, bush, pole, fence, prop). Stuff classes share one segmentation id each; things get one id **per mesh** (`simSetSegmentationObjectID`). AirSim's id→colour palette is not exposed by the API, so it is *measured*: paint the whole scene with one id, grab a frame, read the colour — once per id, cached in `palette_cache.json`. |
| coverage | `coverage.py` | The env extent comes from the positions of the classified objects (percentile-trimmed + margin). The flight is a lawnmower whose first pass starts at the corner nearest take-off; passes run corner-to-corner, so all four corners are visited. Line spacing = camera swath × (1 − `OVERLAP`). |
| capture | `capture.py` | Lockstep recording (`simPause` / `simContinueForTime`, same as `slam/recorder.py`): per frame the nadir camera's `Segmentation` + `DepthPlanar` images, the LiDAR sweep, and ground-truth poses of both sensors. |
| fusion | `fuse.py` | Two point sources per frame vote into a `VOXEL`-sized grid: **LiDAR points** projected into the camera and given the label of the pixel they hit *if* their range agrees with `DepthPlanar` there (otherwise they are occluded and stay unlabeled), plus the **seg+depth pixels back-projected** on a stride to fill the ground densely. Majority vote per voxel. Thing ids are then split into instances by 26-connectivity — that also fixes ids capture had to share once the 255-id budget ran out. |
| planning | `planner.py` | 2-D grid from the map: road cells (class `road`), surface height (lowest ground-class voxel), obstacle cells (any non-ground voxel 0.35–4 m above the local surface — buildings, trees, poles, *parked cars*). Drivable = road eroded by half car width + clearance, minus obstacles dilated by the same. A* (no corner cutting, cost rising toward the lane edge) → string-pulling → 0.5 m polyline. |
| driving | `drive_car.py` | The sim is in Multirotor mode, so the car is a static mesh (`simSpawnObject`, physics off) moved with `simSetObjectPose` by a kinematic bicycle + pure pursuit. Every tick the car's full rectangle is re-checked against the map's road and obstacle grids; the run ends with the count of off-road / colliding ticks (expected 0). Open3D shows the map, route and car; `--chase` adds the sim's `CarCam` trailing it. |

Output layout (`datasets_panoptic/<timestamp>/`): `meta.json` (intrinsics, bounds,
waypoints, id table, palette), `index.json` (per-frame stamps + poses),
`seg/`, `depth/`, `lidar/`, and after fusion `panoptic_map.npz`
(`xyz`, `seg_id`, `class_id`, `instance_id`) plus `panoptic_class.ply` /
`panoptic_instance.ply`.

## Verified run (AirSimNH, 2026-09-12)

3948 frames / 3948 LiDAR sweeps over a 387 × 394 m extent (440 waypoints, 4.7 km, ~25 min
of sim time at 0.65 s per lockstep tick), fused to 4.69 M voxels at 0.4 m: 182 k road,
368 k building (2247 instances), 306 k tree, 18.6 k vehicle (87 instances), 2.5 k pole ...
Auto route (132,22) → (−13,124): 765 m, 3147 ticks, **0 off-road, 0 collision ticks**.

## Things to know (each one cost a run)

- **`msgpack` must be the C extension.** `msgpack-python==0.5.6` (pinned by the RPC lib)
  installs its pure-Python fallback on Python 3.10; each DepthPlanar frame then takes 6 s
  and a huge allocation. See the note in `CLAUDE.md` for the rebuild recipe.
- **The palette is measured through the external `CarCam` 30 m up**, never through the
  drone's ground-level nadir camera: that gave wrong colours for ids 1 and 2 (road and
  sidewalk!) with no error. `calibrate_palette` now requires ≥ 90 % of the frame to agree
  and retries. If `fuse.py` reports a class you expect as missing, delete
  `palette_cache.json` and re-run `capture.py --dry-run` (patch the recording's `meta.json`
  `palette` from the new cache; id → colour is fixed for the sim, so no re-flight).
- **`simGetSegmentationObjectID` returns −1 for actor names** (it compares component
  names) — it does not mean the id was not applied. `simSetSegmentationObjectID` with an
  exact actor name works; check by painting and looking, as the calibration does.
- **`simSpawnObject` with a name that already exists is a UE4 fatal** (segfault, sim
  gone). `drive_car.py` destroys every `PanopticCar*` and spawns a per-run name.
- **Planner margins are a real trade-off in AirSimNH**: hedges line both sides of every
  15 m road. Obstacles are grown by half width + clearance + `OBSTACLE_EXTRA` (0.3 m,
  the smallest value at which the car's front corner never sweeps a pole cell in turns —
  0.0 m gave 52 colliding ticks; the safe 2.5 m half-diagonal leaves 10 % of the roads).
  `OBSTACLE_BAND` is 2.2 m so the avenue's canopies (from 2.8 m up) don't block, and
  LiDAR-only "unlabeled" voxels don't count (`UNLABELED_BLOCKS=False`): they are
  collision-only geometry ~1 m over the road that the camera sees straight through, and
  counting them cut the road network in half.
- **Rig and config must agree.** `settings.panoptic.json` and `config.py` both state the
  camera size/FOV; `capture.py` refuses to record on a mismatch (`fx` derives from width).
- **Labels are simulator ground truth**, not perception: swapping the seg image for a
  SAM3/YOLO mask in `fuse.label_lidar` is the perception variant.
- **Taxonomy is keyword-based, tuned for AirSimNH** (`capture.py --list-objects` to audit;
  "Hedge_*" is also what the surrounding forest is made of, hence 2.7 M "bush" voxels).
  Unmatched meshes become `other` — stuff, never an obstacle — so check the D view.
- **255 ids**: thing ids are shared within a class past the budget (sqrt-proportional
  quota per class so 1700 house parts can't starve the 70 cars) and `fuse.py` separates
  them by connectivity — a hedge row is one instance.
- **Car mesh conventions vary**: `--yaw-offset` / `--z-offset` / `--asset` (default: the
  first "car"-like `simListAssets` entry; `009_SUV_ISM_NewMat` sits correctly as is).
- The car is checked against the *map*, not UE4 collision — that is the point, but a mesh
  the drone never saw is not avoided.
