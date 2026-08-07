# SLAM: LiDAR-inertial vs. stereo-inertial, across weather

A study answering three questions on this AirSim rig:

1. Can we do real SLAM with camera + IMU + LiDAR?
2. How much is lost by dropping the LiDAR — the expensive sensor — for a stereo pair?
3. How does each degrade in clear / rain / fog?

Everything is CPU-only numpy / Open3D / OpenCV / scipy. **No new dependencies** beyond what
`airsim_venv/` already ships — partly on principle, partly because `msgpack-rpc-python` pins
`tornado<5` and adding packages to that venv is how you break the AirSim client.

---

## The finding that shapes the whole design

**AirSim's weather is a rendering effect.** `simSetWeatherParameter` drives UE4 particle systems and
post-process volumes, so it changes what the cameras see. The LiDAR is a raycast against collision
geometry, and rain/fog particles have no collision — so **LiDAR returns are identical in every
weather condition**.

`tools/probe_setup.py` and `tools/verify_dataset.py` both *measure* this rather than asserting it.

Consequently:

| | source of degradation |
|---|---|
| cameras | real — rendered by UE4 into the recorded frames |
| LiDAR | **modelled** by `slam/degradation.py` at matching severity |

Every result is reported on both a `raw` and a `degraded` axis, so the simulator's limitation is part
of the finding rather than hidden inside it.

---

## Pipeline

```
                      ┌── flight/record_dataset.py ──┐
   AirSim ────────────┤  lockstep capture per        ├──► datasets/<condition>/
                      │  weather condition           │
                      └──────────────────────────────┘
                                     │
                      slam/source.py │ DatasetSource   (+ optional degradation)
                                     ▼
        ┌────────────────────────────┴────────────────────────────┐
        │                                                          │
  slam/lidar_slam.py                                     slam/stereo_slam.py
  IMU prior → de-skew → GICP                             SGBM → ORB → PnP/RANSAC
  scan-to-submap                                         → windowed BA
        │                                                          │
        └──────────────► slam/backend.py  (SHARED pose graph) ◄─────┘
                                     │
                          slam/evaluate.py → tools/run_benchmark.py
                                     ▼
                          results/metrics.csv + figures
```

The back end is shared **on purpose**. If the two methods optimised their trajectories differently,
any difference in the results would be partly the back end's. With one optimiser, the comparison
isolates the front end. (Loop *detection* is necessarily per-method — FPFH/GICP is meaningless on a
sparse triangulated stereo point set — so stereo verifies its own closures and submits them through
`backend.add_loop_edge`.)

---

## Running it

### 0. Configure and probe (once)

`settings.json` was rewritten for this study and **AirSim only reads it at startup**, so restart the
sim first:

```bash
./scripts/run_swarm.sh AirSimNH
./airsim_venv/bin/python tools/probe_setup.py
```

The probe must pass before anything else. It verifies, against the live sim:

- LiDAR returns **sensor-local** points (not the spawn-inertial frame the repo used to use)
- `LidarData.pose` / `.time_stamp` are populated
- both stereo cameras deliver 1280×720 with real parallax, and intrinsics match `slam/config.py`
- the explicit `Sensors.Imu` block did not suppress the default GPS/barometer/magnetometer
- `simPause` + `simContinueForTime` steps cleanly with an async move in flight, **and** images and
  LiDAR can still be read while paused (the lockstep recorder needs all of this)
- weather visibly changes the rendered image, and does **not** change LiDAR returns

### 1. Record

```bash
./airsim_venv/bin/python flight/record_dataset.py --condition all
./airsim_venv/bin/python tools/verify_dataset.py --all
```

~3 GB and a few minutes of wall clock per condition; five conditions. Add `--free-running` if the
probe reported that lockstep stepping misbehaves.

### 2. Benchmark (offline, no simulator)

```bash
./airsim_venv/bin/python tools/run_benchmark.py
```

Writes `results/metrics.csv`, `metrics.json`, and three figures: `degradation_curves.png`,
`robustness.png`, `trajectories.png`.

### 3. Live demo

```bash
./airsim_venv/bin/python flight/slam_live.py --method lidar --weather fog_heavy
```

Three panels: camera feed, the map built from **estimated** poses, and estimated vs. ground-truth
trajectory. Contrast with `flight/lidar_viz.py`, which places scans by the simulator's own pose —
that is the "perfect localisation" upper bound.

The map panel is thinned for display only (`--map-voxel`, default 0.9 m — the SLAM map itself stays
at `MAP_VOXEL_SIZE`); drawing all of it is an unreadable wall of points. Height colouring runs
blue (high) → cyan → green → yellow → red (ground) over a **2nd/98th-percentile** z range that only
ever widens, so a few stray returns can't flatten every point onto one colour. Use `--map-voxel 0`
`--point-size 1` if you want the raw density back.

### Testing without the simulator

`tools/synthetic_dataset.py` generates the same on-disk format from a raycast synthetic
neighbourhood, so `verify_dataset.py` and `run_benchmark.py` can be exercised without Docker, the
GPU, or an hour of flying:

```bash
./airsim_venv/bin/python tools/synthetic_dataset.py --out datasets_synth --frames 400
./airsim_venv/bin/python tools/run_benchmark.py --datasets datasets_synth
```

It is a **test fixture**. Numbers from it say the pipeline works; they say nothing about AirSimNH.

---

## Sensor configuration

`settings.json` and `slam/config.py` must stay in sync; the probe cross-checks them.

| | value | why |
|---|---|---|
| LiDAR | 16 ch, 100 m, 300 k pts/s, 10 Hz, VFOV +15°/−45° | The default `Range` is **10 m** — useless for mapping. VFOV is biased downward because at 25 m AGL a symmetric ±15° rig only returns ground in a ring ~45 m out. |
| `DataFrame` | `SensorLocalFrame` | Non-negotiable. The estimator must apply its *own* pose, never the simulator's. |
| Stereo | 2 × 1280×720, 90° FOV, 0.25 m baseline | `fx = 640`, so `fx·B = 160`. Depth error `Z²·δd/(fx·B)` ≈ 0.3 m at 10 m, 1.3 m at 20 m. |
| IMU | explicit block with stated noise params | So the noise model is reproducible rather than an undocumented default. |
| Route | 120×80 m rectangle, 25 m AGL, **2 laps**, 4 m/s | ~800 m. The second lap is what gives loop closure something to close. The altitude must clear everything in the environment — the circuit is a fixed script with no obstacle avoidance, and the recorder aborts if the drone wedges into geometry. |

Weather does not affect AirSim physics, so the same waypoints produce near-identical trajectories in
all five conditions — the sensor stream is the only variable.

---

## Degradation model

`slam/degradation.py`. Beer-Lambert attenuation inside the LiDAR range equation:

```
P_received ∝ ρ · exp(−2·α·r) / r²
p_detect(r) = min(1, (ρ/ρ₀) · (R_max/r)² · exp(−2·α·r))
```

- **Fog** — Koschmieder, `α = 3.912 / V`. Fog droplets are large relative to 905 nm, so the
  visible-band relation carries over.
- **Rain** — `α = k · R^0.67`, with `k` calibrated so 40 mm/h costs ~35 % of maximum range, matching
  the 20–40 % reduction reported for 905 nm automotive LiDAR.

Four effects: attenuation dropout, range noise, **backscatter clutter**, and rain speckle. Clutter
matters most — dropout merely thins the cloud, whereas clutter adds dense, plausible-looking
structure right where a scan matcher expects real geometry.

Measured behaviour of the model:

| condition | α (1/m) | effective range | points kept | clutter share |
|---|---|---|---|---|
| clear | 0 | 100 m | 100 % | 0 % |
| rain_light (15 mm/h) | 0.0025 | 100 m | 81 % | 0.2 % |
| rain_heavy (40 mm/h) | 0.0047 | 92 m | 73 % | 0.4 % |
| fog_light (510 m vis) | 0.0077 | 78 m | 73 % | 1.0 % |
| fog_heavy (60 m vis) | 0.0652 | 26 m | 24 % | 25 % |

**Rain barely touches LiDAR; fog cripples it.** That is the physically correct answer and one of the
more interesting things the study can show.

All coefficients live in one annotated `COEFFICIENTS` table. This is the study's main modelling
assumption and it is kept in one auditable place on purpose.

---

## Verified behaviour

All numbers below are from synthetic fixtures on the full 800 m two-lap route. **They validate the
pipeline, not AirSimNH.**

Two fixtures are used and their numbers are **not interchangeable**:

- **fixture A** — clean IMU (no added noise). Used for the LiDAR-vs-stereo comparison, so both
  methods see the same conditions.
- **fixture B** — `tools/synthetic_dataset.py`, which adds IMU noise (σ 2e-4 rad/s gyro,
  2e-3 m/s² accel). Harder, and the one the on-disk benchmark actually runs on.

### Odometry (loop closure disabled) — stable and reproducible

| fixture | method | ATE RMSE | drift | RPE | tracking failures |
|---|---|---|---|---|---|
| A | LiDAR-inertial | 3.82 m | 0.87 % | 6.7 %/10 m | 11 / 2000 |
| A | Stereo-inertial (VO only) | 16.18 m | 2.70 % | 7.4 %/10 m | 38 / 2000 |
| B | LiDAR-inertial | 23.53 m | 2.9 % | 8.5 %/10 m | 7 / 2000 |

**On the matched fixture, LiDAR odometry drifts ~3× less than stereo** (0.87 % vs 2.70 %) — the
headline answer to "how much do you lose dropping the LiDAR". The fixture-B row shows how sharply
LiDAR odometry degrades once the IMU prior is noisy, which is worth knowing before trusting the
real recording.

Odometry is **exactly reproducible**: seeds 42 and 43 produced byte-identical metrics
(23.531 m both). All nondeterminism in this system lives in loop-closure RANSAC.

Windowed BA was measured and **turned off by default**: 16.175 m → 16.201 m (no improvement) for
+55 % runtime (520 s → 806 s), with 45 of 399 solves discarded as diverged. Re-enable with
`--stereo-ba`.

Windowed BA was measured and **turned off by default**: 16.175 m → 16.201 m (no improvement) for
+55 % runtime (520 s → 806 s), with 45 of 399 solves discarded as diverged. Re-enable with
`--stereo-ba`.

### Loop closure — large gains, but **not yet reliable**

At its best, loop closure is transformative:

| fixture | without closure | with closure |
|---|---|---|
| A | 3.82 m (0.87 % drift) | **0.66 m** (0.12 % drift) |
| B | 23.53 m (2.9 % drift) | **0.77 m** |

All 971 edges are retained (they used to be silently pruned — see below), and closure edges scored
directly against ground truth in one run had median error 0.25 m, with 1 of 431 above 2 m.

But it is **stochastic and currently unstable**. On byte-identical fixture-B input, ATE has been
observed at 0.77 m, 5.10 m and 26.81 m depending only on the RANSAC seed. See "Known limitations" —
this is the one part of the system that should not be trusted from a single run.

### Metrics — validated against known inputs

Ground truth scored against itself gives ATE 0; a rigid offset or rotation gives ~0 (alignment
absorbs it, as ATE should); iid noise of σ=0.5 m/axis recovers 0.853 m against an expected 0.866; a
10 % scale error recovers `scale_estimate = 0.9091`; map completeness on a half-removed sparse cloud
recovers exactly 0.500.

### Weather — the asymmetry, demonstrated end to end

Full `run_benchmark.py` output on fixture B, odometry only (`--no-loop-closure --repeats 2`):

| condition | LiDAR degraded? | ATE RMSE | RPE %/10 m | keyframes |
|---|---|---|---|---|
| clear | no | 23.531 m | 8.50 | 388 |
| clear | yes | 23.531 m | 8.50 | 388 |
| fog_heavy | no | 23.531 m | 8.50 | 388 |
| fog_heavy | **yes** | **115.427 m** | **302.43** | 162 |

Read the first three rows carefully — they are **identical to the digit**, including the map point
count (394 921). That is the whole argument in one table:

- `clear` raw vs `clear` degraded are identical because the model correctly no-ops at α = 0.
- `clear` raw vs `fog_heavy` raw are identical because **the simulator's LiDAR stream is literally
  unchanged by fog**. Had the study stopped at "run SLAM under AirSim weather", its conclusion would
  have been "LiDAR is perfectly weather-proof" — an artifact of the simulator, not a fact about
  LiDAR.
- Only the modelled row moves, and it moves catastrophically: dense fog cuts the effective range to
  26 m and fills the near field with backscatter clutter, so scan matching collapses (keyframes drop
  from 388 to 162 because the estimator stops believing it is moving).

Both repeats of every cell agreed exactly, confirming per-cell seeding works.

---

## Bugs this work fixed in existing code

1. **`flight/lidar_viz.py`** was double-transforming every scan — `settings.json` had
   `VehicleInertialFrame` (points already in the spawn frame) and the code added the drone position
   again. With `SensorLocalFrame` it now applies the full `R·p + t` from `LidarData.pose`.
   Translation alone was never enough: it left scans un-rotated, so the map sheared apart on turns.
2. **`rl/airsim_gym_env.py`** used `norm(points)` as "distance to nearest obstacle". Under
   `VehicleInertialFrame` that measured distance from the **spawn origin**, so the obstacle channel
   fed to PPO grew with how far the drone had flown. Correct now that points are sensor-local.

Two bugs were also found and fixed inside this work, both worth knowing about:

3. **Open3D's `global_optimization` prunes edges from the `PoseGraph` you hand it.** An incremental
   system optimises repeatedly, so the graph erodes a little each call until even the odometry chain
   is gone. `slam/backend.py` keeps an authoritative edge list and rebuilds a fresh `PoseGraph` per
   call. Before the fix, loop closure made ATE *worse* (3.8 m → 21.7 m); after, better (3.8 m →
   0.66 m).
4. **Frame-to-keyframe anchoring must be captured before optimisation runs**, or the frame is
   expressed relative to a corrected pose using an uncorrected estimate — exactly cancelling the
   loop closure back out.

---

## Known limitations

### 1. Loop closure is stochastic and not yet reliable — the biggest open issue

On **byte-identical input** (the `clear` dataset, where the degradation model is a no-op), ATE has
been observed at:

| RANSAC seed | ATE RMSE | closures | max implied correction |
|---|---|---|---|
| 42 | 0.77 m | 563 | — |
| 1 | 5.10 m | 428 | 14.0 m |
| (a later cell, seed-shifted) | 26.81 m | — | — |

Verification uses RANSAC on FPFH features, which is stochastic; a repetitive scene (rows of similar
houses, and AirSimNH is similar) produces occasional false matches that survive both the fitness
gate and the drift-plausibility gate, and one bad edge can drag the graph.

What has been ruled out:

- **Not** the edge convention — unit-tested on a synthetic square loop, optimisation improves it.
- **Not** edge erosion — that was a real bug (see below) and is fixed; all 971 edges now survive.
- **Not** simply loose gating — tightening `LOOP_MAX_DRIFT_FRACTION` 0.08 → 0.03 and
  `LOOP_MIN_FITNESS` 0.45 → 0.55 made a sampled seed *worse* (5.1 m → 23.2 m), so it was reverted.

Mitigations in place, and what to do next:

- `tools/run_benchmark.py` seeds the RNG **per grid cell**, so results are reproducible and the grid
  can be re-ordered without changing numbers.
- `--repeats N` runs each cell across consecutive seeds so the spread is visible instead of implicit.
  **Use it. Do not report a single draw.**
- The most promising untried fix is to **decouple the front end from the back end**: currently an
  optimised graph is fed straight back into the odometry state and submap, so one bad optimisation
  corrupts all subsequent tracking. Running odometry unperturbed and applying the pose graph only as
  a correction layer would bound the damage to the final trajectory.
- Sequence consistency (accepting a closure only when neighbouring keyframes agree on a nearby
  target) is the standard place-recognition defence and is not implemented here.

**Until this is resolved, the trustworthy comparison is the odometry one** (loop closure disabled,
`--no-loop-closure`), which is stable and reproducible.

### 2. Other limitations

- **Windowed BA earns nothing measurable** (16.175 → 16.201 m) for +55 % runtime, so it is off by
  default. Diverged solves are detected and discarded (`BA_MAX_CORRECTION`). A useful implementation
  wants a real sparse solver (g2o/GTSAM) and landmarks tracked across keyframes rather than
  re-matched to the newest one — both ruled out by the no-new-dependencies constraint.
- **Stereo loop closure is implemented but unverified at full scale.** The full two-lap run with
  closures enabled was cut short; VO-only numbers above are complete and trustworthy.
- **IMU biases are estimated once** from a quasi-stationary window and then held. Fine for bridging
  the ~100 ms between sweeps; not a substitute for online bias estimation in a factor graph.
- **LiDAR odometry needs the recording to start from rest.** The recorder hovers before starting the
  route, so this holds — but starting mid-motion leaves the IMU prior with no velocity, and scan
  matching can then lock onto the initial submap and never move. This was observed and is why
  `tools/synthetic_dataset.py` ramps from rest too.
- **Scan matching needs vertical structure.** Over a near-flat ground plane, in-plane translation is
  unobservable and GICP simply refuses to move. AirSimNH's houses and fences supply this; open
  terrain (Africa_Savannah) likely would not.
- **`simContinueForTime` does not deliver the dt it is asked for.** On this build a requested 10 ms
  tick advances **9 ms**, so a "100 Hz" lockstep recording is really ~110 Hz and sensors are polled at
  ~11 Hz against a 10 Hz LiDAR. Consequences: about one poll in nine returns the sweep already
  captured (the recorder detects and skips these — `repeated_lidar` in the run summary), and the
  nominal rates in `slam/config.py` are a *request*, not a measurement. Nothing computes on the
  declared rate — integration uses per-sample timestamps throughout, and `sensor.yaml` now records
  `imu_hz_measured` alongside the declaration. Oversampling is the deliberate choice: it can duplicate
  a sweep, never miss one.
- **`simGetCollisionInfo` latches.** It keeps reporting the most recent collision, and the drone has
  been resting on the ground since spawn — so the first poll of any run returns that ground contact.
  The recorder baselines it after takeoff (`_baseline_collision`); a watchdog that skips this reports
  a phantom collision with the spawn surface in every single run.
- The synthetic test world is a **fixture**, not a validation of AirSimNH performance.
