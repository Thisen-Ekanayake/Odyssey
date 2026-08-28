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

A second, sharper version of the same trap turned up when the full five-condition grid was finally
run on the real recordings, and it is worth stating here rather than burying it: AirSim's LiDAR is
not merely weather-proof, it is **noiseless**. Every front-end constant in this pipeline was
therefore tuned against a sensor that cannot exist, and it turns out to tolerate essentially no
range noise at all — 1.5 cm is enough to break it, well under a real unit's 2–3 cm floor. That
makes the `raw` arm the trustworthy half and the `degraded` arm a confound rather than a weather
ranking. Full evidence under "Weather" and "Known limitations #2"; read those before quoting any
degraded number.

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
sim first. `PROFILE` defaults to `slam`, which is the rig this study needs — pass it explicitly if
you have been running the four-drone demos, which use `PROFILE=swarm`:

```bash
PROFILE=slam ./scripts/run_swarm.sh AirSimNH
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
./airsim_venv/bin/python tools/run_benchmark.py --no-loop-closure     # the trustworthy arm
./airsim_venv/bin/python tools/run_benchmark.py --repeats 3 --out results/loop_closure
```

15 cells (5 conditions × {lidar, stereo} × {raw, degraded}, minus the pointless stereo+degraded).
Roughly an hour per pass on a 32-core box.

Writes `results/metrics.csv`, `metrics.json`, three figures (`degradation_curves.png`,
`robustness.png`, `trajectories.png`), and `run_manifest.json` — which records **which conditions
actually ran**. That last one exists because it did not: `results/` sat for weeks holding a
2-condition, 60-frame synthetic smoke run while looking exactly like the finished 5-condition
study, and the tool silently dropped missing conditions rather than complaining. It now refuses
unless you pass `--allow-missing`.

Use `--repeats` for anything with loop closure on. See "Known limitations" — a single draw is not
a result there.

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

`synthetic_dataset.py` defaults to all five conditions, so a fixture generated as above is a
complete grid. If yours is not — the one on disk was generated `--no-stereo` and with only two
conditions — `run_benchmark.py` will now **refuse to run** rather than quietly produce a
2-condition `metrics.csv` that looks like the study. Regenerate it, or pass `--allow-missing`
to accept a partial grid deliberately; either way `results/run_manifest.json` records which
conditions actually ran.

It is a **test fixture**. Numbers from it say the pipeline works; they say nothing about AirSimNH.

---

## Sensor configuration

`settings.json` and `slam/config.py` must stay in sync; the probe cross-checks them against the
**live sim**, so run it after starting the sim, not before.

There are now two rigs. `settings.json` is this one — the SLAM rig, and the default.
`settings.swarm.json` (`PROFILE=swarm ./scripts/run_swarm.sh`) is a cut-down 640×360 / 100 k
version for the four-drone demos, which need the sim to keep up more than they need resolution.
**Never record a dataset on the swarm profile**: `cfg.FX` is *derived* from `IMAGE_WIDTH`, so
half the resolution silently halves the true focal length and every stereo depth comes out 2×
wrong, with nothing raising an error. `probe_setup.py` catches it in one line and now names the
profile to switch to.

| | value | why |
|---|---|---|
| LiDAR | 16 ch, 100 m, 300 k pts/s, 10 Hz, VFOV +15°/−45° | The default `Range` is **10 m** — useless for mapping. VFOV is biased downward because at 25 m AGL a symmetric ±15° rig only returns ground in a ring ~45 m out. |
| `DataFrame` | `SensorLocalFrame` | Non-negotiable. The estimator must apply its *own* pose, never the simulator's. |
| Stereo | 2 × 1280×720, 90° FOV, 0.25 m baseline | `fx = 640`, so `fx·B = 160`. Depth error `Z²·δd/(fx·B)` ≈ 0.3 m at 10 m, 1.3 m at 20 m. This baseline is the stereo arm's binding constraint at 25 m AGL — see "Stereo depth horizon" below. |
| IMU | explicit block with stated noise params | So the noise model is reproducible rather than an undocumented default. |
| Route | 120×80 m rectangle, 25 m AGL, **2 laps**, 4 m/s | ~800 m. The second lap is what gives loop closure something to close. The altitude must clear everything in the environment — the circuit is a fixed script with no obstacle avoidance, and the recorder aborts if the drone wedges into geometry. |

Weather does not affect AirSim physics, so the same waypoints produce near-identical trajectories in
all five conditions — the sensor stream is the only variable.

### Stereo depth horizon — the rig's binding constraint, and a bug it caused

The route flies at 25 m AGL with a **forward**-facing camera, so the scene it looks at sits at
about **71 m** median depth. A 0.25 m baseline at that range gives roughly **1.1 px** of disparity
in the half-resolution SGBM image (`fx·s·B/Z = 320·0.25/71`). That is the honest physical limit of
this rig, and no amount of tuning removes it.

`STEREO_MAX_DEPTH` was originally set to **25 m** from the depth-resolution argument in the table
above. The argument is correct; the value was badly wrong for this route, because it threw away
almost everything the camera could see. The cost was not subtle — measured over the full 798 m
circuit on `datasets/clear`, odometry only, seed 42:

| `STEREO_MAX_DEPTH` | tracking failures | ATE RMSE | drift | RPE %/10 m |
|---|---|---|---|---|
| 25 m (was) | 1217 / 2315 (52.6 %) | 193.18 m | 39.96 % | 196.65 |
| **60 m (now)** | **120 / 2315 (5.2 %)** | **44.95 m** | **14.04 %** | **29.56** |

60 m is a genuine optimum, not "more is better". Over the first 274 m a 120 m cap yields *more*
usable points per frame (740 vs 200 median) and tracks *worse* (ATE 21.2 m vs 15.0 m) — beyond
~60 m the extra points are noise wearing a depth. A 2 px-disparity floor would have argued for
40 m; that measured worse too (29.5 m over the same 274 m). The measurement decided it.

Two things worth taking from this. First, 14 % drift over 800 m is still a poor result — the
stereo arm is not competitive here, and the reason is the rig, not the algorithm. Second, this is
much the largest single lever found in the stereo pipeline, and it was a **constant chosen from a
sound argument that was never measured against the actual flight**. Windowed BA, by contrast, was
measured and earned nothing.

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

Two sources of numbers appear below and they are **not interchangeable**. Check which one you are
reading before quoting anything.

- **"Weather" (further down) is the real thing** — all five AirSimNH recordings, ~800 m each,
  through `run_benchmark.py`. That section carries the study's actual answers.
- **Everything between here and there is from synthetic fixtures.** They validate the pipeline, not
  AirSimNH, and they predate the real grid.

Two fixtures are used:

- **fixture A** — clean IMU (no added noise). Used for the LiDAR-vs-stereo comparison, so both
  methods see the same conditions.
- **fixture B** — `tools/synthetic_dataset.py`, which adds IMU noise (σ 2e-4 rad/s gyro,
  2e-3 m/s² accel). Harder.

Where the two disagree, the real recordings win. The most important disagreement: fixture A put
LiDAR ~3× ahead of stereo, and the real data puts it **~100×** ahead (0.43 m vs 44.9 m). The fixture
was too kind to stereo because its procedural texture gives ORB clean corners at any range, which
sidesteps the baseline limit that dominates on the real recordings.

### Odometry (loop closure disabled) — stable and reproducible

| fixture | method | ATE RMSE | drift | RPE | tracking failures |
|---|---|---|---|---|---|
| A | LiDAR-inertial | 3.82 m | 0.87 % | 6.7 %/10 m | 11 / 2000 |
| A | Stereo-inertial (VO only) | 16.18 m | 2.70 % | 7.4 %/10 m | 38 / 2000 |
| B | LiDAR-inertial | 23.53 m | 2.9 % | 8.5 %/10 m | 7 / 2000 |

On the matched fixture LiDAR odometry drifts ~3× less than stereo (0.87 % vs 2.70 %). **Treat that
ratio as a fixture artifact, not the answer** — on the real recordings the gap is ~100×, for the
reason given above. The fixture-B row shows how sharply LiDAR odometry degrades once the IMU prior
is noisy.

Odometry is **exactly reproducible**: seeds 42 and 43 produced byte-identical metrics
(23.531 m both).

That was once stated as "all nondeterminism in this system lives in loop-closure RANSAC", and
running the real grid twice falsified it. With loop closure **off** and the same seed, 14 of the 15
cells reproduced bit-for-bit across two runs — and `fog_light`+degraded did not (2410.1 m vs
2117.5 m, with a different failure count too). The distinguishing feature is that it is one of the
diverged cells: converged runs are reproducible, and a run that has already lost tracking is not.
The likely mechanism is non-associative floating-point reduction in Open3D's multi-threaded GICP,
where differences far below display precision get amplified once the trajectory is chaotic.

Practical consequence: **a diverged ATE is not a measurement.** Its magnitude carries no
information, it does not reproduce, and it should not be compared against another diverged ATE.
`TrajectoryMetrics.diverged` exists to mark exactly these rows.

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

### Weather — the asymmetry, on the real recordings

This is the study's central claim, and it is now measured on all five AirSimNH recordings rather
than on a fixture. `run_benchmark.py --no-loop-closure`, full 798 m circuit, seed 42, simulator
output exactly as recorded:

| condition | LiDAR-inertial ATE | drift | stereo-inertial ATE | drift | stereo tracking failures |
|---|---|---|---|---|---|
| clear | **0.430 m** | 0.07 % | 44.9 m | 14.0 % | 5.2 % |
| rain_light | **0.430 m** | 0.08 % | 28.1 m | 9.8 % | 4.7 % |
| rain_heavy | **0.505 m** | 0.11 % | 35.8 m | 7.2 % | 6.2 % |
| fog_light | **0.476 m** | 0.07 % | **663.0 m** | 125.5 % | 58.7 % |
| fog_heavy | **0.456 m** | 0.09 % | **16 409.8 m** | 3549 % | 94.4 % |

Two things fall out of it.

**The LiDAR does not care about the weather — because the simulator's LiDAR cannot.** The five
LiDAR numbers span 0.43–0.51 m, and that spread is trajectory variation between five separate
flights, not weather. This is the trap the whole study is built around, now demonstrated instead of
asserted: rain and fog are rendering effects, the LiDAR is a raycast against collision geometry, and
particles have no collision. A study that stopped here would conclude "LiDAR is weather-proof",
which is a fact about AirSim and not about LiDAR.

**The cameras do care, and the collapse is dramatic.** Stereo degrades from 44.9 m in clear to
663 m in light fog and 16.4 km in dense fog, with tracking failures rising 5 % → 59 % → 94 %. That
half of the comparison is genuinely simulated, and it is the half where AirSim earns its keep.

Rain is mildly *better* than clear for stereo (28.1 m and 35.8 m vs 44.9 m). Wet-road specular
highlights plausibly add ORB features, but with a marginal estimator at 5–6 % failure either way,
this is not a large enough gap to claim as a finding.

Note the LiDAR arm is ~100× more accurate than the stereo arm here (0.43 m vs 44.9 m in clear).
That is a far wider gap than the ~3× the synthetic fixture suggested, and the reason is the rig
rather than the algorithm — see "Stereo depth horizon" above.

#### The modelled-LiDAR arm does not currently rank weather — and here is why

`run_benchmark.py` also runs a `degraded` arm that applies `slam/degradation.py` to the LiDAR.
Those numbers are **not** a weather ranking and must not be read as one:

| condition | α (1/m) | points kept | degraded ATE |
|---|---|---|---|
| rain_light | 0.0025 | 82 % | 2130.9 m *(diverged)* |
| rain_heavy | 0.0047 | 75 % | 1142.7 m *(diverged)* |
| fog_light | 0.0077 | 78 % | 2117.5 m *(diverged)* |
| fog_heavy | 0.0652 | 11 % | 56.7 m |

The mildest condition diverges and the harshest survives. The degradation model itself is monotone
— measured on real scans it removes 18 % of points at `rain_light` and 89 % at `fog_heavy` — so the
inversion comes from the pipeline, not the model. Ablating the model's four effects one at a time
on `rain_light` (400 frames) isolates it to one:

| variant | GICP failures | mean fitness | ATE |
|---|---|---|---|
| all four effects | 14 | 0.707 | 143.4 m |
| no clutter | 16 | 0.701 | 37.1 m |
| no speckle | 18 | 0.747 | 39.0 m |
| no attenuation dropout | 16 | 0.721 | 145.7 m |
| **no range noise** | **0** | **0.956** | **0.247 m** |
| clutter only (no noise) | 0 | 0.958 | 0.266 m |

Range noise alone accounts for the whole effect. Removing it restores ATE from 143 m to 0.25 m and
GICP fitness from 0.71 to 0.96; removing any of the other three changes little.

Sweeping the noise magnitude on its own — every other effect disabled — shows it is not a
sensitivity but a **cliff**:

| σ multiplier | σ at 48 m | GICP failures | fitness | ATE |
|---|---|---|---|---|
| 0 | 0 | 0 | 0.960 | **0.244 m** |
| 0.25 | **1.5 cm** | 23 | 0.691 | 68.2 m |
| 0.5 | 3.1 cm | 96 | 0.648 | 558.3 m |
| 1.0 | 6.1 cm | 24 | 0.677 | 39.7 m |
| 2.0 | 12.2 cm | 28 | 0.739 | 39.2 m |

**1.5 cm of radial noise is enough to break it, and more noise is not meaningfully worse.** Past the
cliff the ATE is chaotic rather than graded — once tracking breaks, where it ends up is arbitrary,
which is also why the five degraded conditions rank the way they do.

The number that matters is the threshold. A real Velodyne or Ouster has a 2–3 cm range-noise floor,
so **as tuned, this pipeline would not work on data from real hardware at all** — it is calibrated
to a noiseless raycast. That is a more useful thing to know about it than any of the weather
numbers, and nothing in a simulator-only study would ever have surfaced it: AirSim's LiDAR is
perfect, so the entire tuning was done against a sensor that does not exist.

`NORMAL_RADIUS` is implicated but is **not** a fix. GICP is plane-to-plane, so it needs a local
neighbourhood with two real dimensions; at the 48 m median range a 16-channel rig over 60° puts
rings **3.35 m** apart, so the default 1.0 m radius sees a single ring — a 1-D arc, whose only
second dimension is the range noise itself. Widening it does help, but not consistently:

| `NORMAL_RADIUS` | rain_light | fog_light | fog_heavy |
|---|---|---|---|
| 1.0 m (default) | 143.4 m | 179.7 m | 38.4 m |
| 2.0 m | 172.0 m | 175.7 m | 39.9 m |
| 3.5 m | 268.4 m | 236.4 m | 40.1 m |
| 5.0 m | **0.32 m** | 248.6 m | 33.8 m |

5.0 m fully recovers `rain_light` (143 m → 0.32 m, 0 failures, fitness 0.96) and does nothing for
`fog_light`; 3.5 m is *worse* than the default. **These are single-seed draws inside a regime that
the magnitude sweep already showed to be chaotic, so this table locates the mechanism and does not
license a retune.** Changing the constant on this evidence would be the same mistake as reporting a
single loop-closure draw. It needs a multi-seed study — see "Known limitations".

The confound this exposes is in the **experiment design**, not the weather model. AirSim's LiDAR is
a raycast and therefore **perfectly noiseless** — the `raw` arm is not "clear weather", it is a
physically impossible sensor. The degradation model adds a baseline sensor noise term
(`SIGMA_BASE = 0.02 m`, a realistic 1σ for a real unit) on top of the weather term, so
`degraded − raw` measures *weather plus the arrival of any range noise at all*. The front end turns
out to be far more sensitive to the second than to the first, and the ordering that results is a
noise-robustness artifact.

This matters retroactively: the earlier fixture result in this document — "only the modelled row
moves, and it moves catastrophically" — was measuring the same thing, because
`tools/synthetic_dataset.py`'s LiDAR is a raycast too. The catastrophic fog row was never purely a
fog result.

The full 5-condition grid is what surfaced this. The fixture had only ever been run at severity 0
and severity 0.8 — the two points where the effect is invisible, because at α = 0 the model no-ops
entirely and at α = 0.065 the surviving cloud is so thin that the noise term is no longer what
dominates.

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

Running the full five-condition grid on the real recordings for the first time turned up four more,
every one of which had been silently producing numbers that looked fine:

5. **`map_rmse` was measuring the start-pose offset, not the map.** `evaluate_map` did a
   nearest-neighbour query between the estimated map and the ground-truth reference **with no
   alignment**, while ATE has always been Umeyama-aligned. A run with 0.221 m ATE scored a 24.917 m
   map RMSE. `TrajectoryMetrics` now carries the alignment it computed and `run_benchmark` applies
   it before scoring; the same run now scores **0.540 m**, and the real grid lands at 0.75–0.80 m
   across all five conditions.
6. **`tracking_failure_rate` could exceed 100 %.** `n_frames` was incremented only on the success
   path while `n_failures` was incremented on early returns, so the two had different denominators.
   A real stereo run reported **154 %**. Both front ends now count `n_frames` as "frames on which
   tracking was attempted", at the point where that becomes true.
7. **`STEREO_MAX_DEPTH` was 25 m against a scene at ~71 m** — see "Stereo depth horizon". 4.3× ATE.
8. **`run_benchmark.py` silently dropped conditions with no dataset directory.** A 5-condition
   invocation over a 2-condition tree produced a `metrics.csv` indistinguishable from the finished
   study, which is exactly what `results/` contained for three weeks. It now refuses unless
   `--allow-missing` is passed, and writes `run_manifest.json` recording what actually ran.

`settings.json` had also drifted from `slam/config.py` (640×360 / 100k vs the declared 1280×720 /
300k) after a performance tweak for the swarm demos. The offline benchmark was unaffected — the
recordings predate the drift — but any *new* recording would have had `fx` wrong by 2×, silently,
because `cfg.FX` is derived from `IMAGE_WIDTH`. The two rigs are now separate files and
`run_swarm.sh` selects between them with `PROFILE`.

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

### 2. The LiDAR front end has essentially no range-noise tolerance

Established above under "Weather": **1.5 cm of radial range noise takes LiDAR odometry from 0.24 m
ATE to 68 m**, and more noise is not meaningfully worse. AirSim's LiDAR is a noiseless raycast, so
the whole front end — `VOXEL_SIZE`, `NORMAL_RADIUS`, `GICP_MIN_FITNESS`, `GICP_MAX_RMSE` — was
tuned against a sensor that cannot exist. A real Velodyne or Ouster sits at 2–3 cm.

Two consequences, and they are different in kind:

- **For the study.** The `degraded` arm is not a weather ranking. Its ordering is dominated by the
  arrival of `SIGMA_BASE` (a baseline sensor-noise term the `raw` arm does not have) rather than by
  extinction, which is why the mildest condition diverges and the harshest does not. Report the
  `raw` arm; treat the `degraded` arm as a demonstration that the modelled-LiDAR axis exists, not
  as a measurement along it. Fixing the confound properly means either applying `SIGMA_BASE` to
  both arms — so the only difference is the weather — or reporting the degraded arm only against a
  noise-matched baseline.
- **For the pipeline.** This is the single biggest obstacle to this code ever touching real
  hardware, and it is invisible from inside a simulator. `NORMAL_RADIUS` is the identified lever
  (5.0 m recovers `rain_light` completely) but is not a general fix and behaves non-monotonically;
  the honest next step is a multi-seed sweep over `NORMAL_RADIUS × VOXEL_SIZE × GICP_MIN_FITNESS`
  against injected noise at a realistic 2–3 cm, scored on several seeds — not a constant changed on
  one draw.

Neither is a small job, and neither was visible until the full five-condition grid ran: the
synthetic fixture had only ever been exercised at severity 0 and 0.8, the two points where the
effect happens not to show.

### 3. Other limitations

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
