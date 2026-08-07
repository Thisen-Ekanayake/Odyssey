"""
Single source of truth for everything the SLAM package needs to know about
the rig, the flight route, the weather grid, and the algorithm tunables.

**This file mirrors ``settings.json`` and must be kept in sync with it.**
AirSim only reads ``settings.json`` at simulator startup, so any change to the
sensor block there needs a ``run_swarm.sh`` restart *and* a matching edit here.
``tools/probe_setup.py`` cross-checks the two against the live sim.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = REPO_ROOT / "datasets"
RESULTS_ROOT = REPO_ROOT / "results"

# --------------------------------------------------------------------------
# vehicle / sensor names (must match settings.json)
# --------------------------------------------------------------------------

DRONE = "Drone1"
LIDAR_NAME = "LidarSensor1"
IMU_NAME = "Imu"
CAM_LEFT = "StereoLeft"
CAM_RIGHT = "StereoRight"

# --------------------------------------------------------------------------
# cameras
# --------------------------------------------------------------------------

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
FOV_DEGREES = 90.0                 # AirSim's FOV_Degrees is the HORIZONTAL FOV

# AirSim renders an ideal pinhole with square pixels and zero distortion, so
# the full intrinsic matrix follows from the FOV alone. Worth stating plainly
# in any writeup: this rig has no calibration or distortion error, which real
# stereo hardware always does.
FX = (IMAGE_WIDTH / 2.0) / math.tan(math.radians(FOV_DEGREES) / 2.0)   # = 640.0
FY = FX
CX = IMAGE_WIDTH / 2.0
CY = IMAGE_HEIGHT / 2.0

K = np.array([[FX, 0.0, CX],
              [0.0, FY, CY],
              [0.0, 0.0, 1.0]], dtype=np.float64)

DIST_COEFFS = np.zeros(5, dtype=np.float64)   # ideal pinhole

STEREO_BASELINE = 0.25             # m, from the +/-0.125 Y offsets in settings.json

# Depth resolution follows dZ = Z^2 * dd / (fx * B). With fx*B = 160 and a
# half-pixel disparity error that is ~0.3 m at 10 m and ~1.3 m at 20 m, so the
# stereo front end is trusted only out to STEREO_MAX_DEPTH. The LiDAR's 100 m
# reach is a genuine hardware advantage, not something to engineer away.
STEREO_MAX_DEPTH = 25.0            # m
STEREO_MIN_DEPTH = 1.0             # m

# Camera *optical* frame (z forward, x right, y down) expressed in the vehicle
# body/NED frame (x forward, y right, z down).
R_BODY_FROM_OPTICAL = np.array([[0.0, 0.0, 1.0],
                                [1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0]], dtype=np.float64)


def _body_from_optical(x: float, y: float, z: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_BODY_FROM_OPTICAL
    T[:3, 3] = (x, y, z)
    return T


# Mount points, matching the Cameras block in settings.json.
T_BODY_CAM_LEFT = _body_from_optical(0.30, -0.125, 0.30)
T_BODY_CAM_RIGHT = _body_from_optical(0.30, +0.125, 0.30)

# LiDAR mount: 10 cm above the body origin, no rotation.
T_BODY_LIDAR = np.eye(4, dtype=np.float64)
T_BODY_LIDAR[:3, 3] = (0.0, 0.0, -0.1)

# --------------------------------------------------------------------------
# LiDAR spec (mirrors settings.json)
# --------------------------------------------------------------------------

LIDAR_CHANNELS = 16
LIDAR_RANGE = 100.0                # m
LIDAR_POINTS_PER_SECOND = 300_000
LIDAR_ROTATIONS_PER_SECOND = 10.0
LIDAR_VFOV_UPPER = 15.0            # deg
LIDAR_VFOV_LOWER = -45.0           # deg, biased downward for aerial mapping

# --------------------------------------------------------------------------
# recording
# --------------------------------------------------------------------------

IMU_RATE_HZ = 100.0                # IMU samples per second
SENSOR_DECIMATION = 10             # LiDAR + stereo every Nth IMU step -> 10 Hz
STEP_DT = 1.0 / IMU_RATE_HZ        # sim seconds advanced per lockstep tick

# --------------------------------------------------------------------------
# flight route
#
# A *closed* circuit flown twice: the second lap is what gives loop closure
# something to close, and it revisits every place at a different time, which
# is exactly the revisit structure the back end is built to exploit.
# Weather does not affect AirSim physics, so the same waypoints produce
# near-identical trajectories in all five conditions -- the sensor stream is
# the only variable under test.
# --------------------------------------------------------------------------

# NED: metres above the spawn point. There is no obstacle avoidance -- the
# circuit is a fixed script, on purpose, so every weather condition flies the
# identical trajectory. That makes clearance a *configuration* problem: the
# route must sit above everything in the environment it is flown in. 12 m was
# roof height in AirSimNH and below the tree canopy, so the drone wedged into
# geometry and the recorder ground on to its step cap. 25 m clears both.
# Raising this costs some LiDAR geometry -- fewer vertical structures in view,
# more of the 100 m budget spent on the gap to the ground -- so if you move it,
# re-record every condition rather than mixing altitudes across a benchmark.
ROUTE_ALTITUDE = -25.0
ROUTE_LENGTH_X = 120.0             # m
ROUTE_LENGTH_Y = 80.0              # m
ROUTE_LAPS = 2
ROUTE_SPEED = 4.0                  # m/s
ROUTE_SPACING = 10.0               # m between densified waypoints


def route_waypoints() -> list[tuple[float, float, float]]:
    """Densified rectangular circuit in the vehicle's local NED frame.

    Corners are sampled every ``ROUTE_SPACING`` metres so ``moveOnPathAsync``
    has a well-conditioned path -- with only four corner waypoints its
    lookahead cuts the corners badly, and the resulting lurch shows up as
    spurious IMU accelerations.
    """
    corners = [
        (0.0, 0.0),
        (ROUTE_LENGTH_X, 0.0),
        (ROUTE_LENGTH_X, ROUTE_LENGTH_Y),
        (0.0, ROUTE_LENGTH_Y),
    ]
    pts: list[tuple[float, float, float]] = []
    for _ in range(ROUTE_LAPS):
        for i in range(len(corners)):
            a = np.array(corners[i])
            b = np.array(corners[(i + 1) % len(corners)])
            n = max(1, int(np.linalg.norm(b - a) / ROUTE_SPACING))
            for k in range(n):
                p = a + (b - a) * (k / n)
                pts.append((float(p[0]), float(p[1]), ROUTE_ALTITUDE))
    pts.append((0.0, 0.0, ROUTE_ALTITUDE))   # close the circuit
    return pts


# --------------------------------------------------------------------------
# weather grid
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class WeatherCondition:
    """One row of the benchmark grid.

    ``airsim_params`` uses *string* keys naming ``airsim.WeatherParameter``
    attributes, so this module stays importable without the airsim package
    (and therefore without the tornado<5 RPC stack) for offline analysis.
    ``weather.py`` resolves the names against the enum.

    ``rain_rate_mmph`` and ``fog_visibility_m`` are the *physical* quantities
    the AirSim [0, 1] slider is taken to represent. AirSim itself defines no
    such mapping -- these are our modelling assumption, consumed by
    ``degradation.py`` to derive an extinction coefficient. They are stated
    here, in one place, so the assumption is auditable rather than buried.
    """
    name: str
    airsim_params: dict[str, float] = field(default_factory=dict)
    rain_rate_mmph: float = 0.0        # mm/h
    fog_visibility_m: float = math.inf  # meteorological visibility
    description: str = ""

    @property
    def severity(self) -> float:
        """Slider value driving this condition, for plotting degradation curves."""
        return max(self.airsim_params.values(), default=0.0)


# Slider -> physical mapping used below:
#   rain rate  R = 50 * val  mm/h        (0.3 -> 15 = heavy, 0.8 -> 40 = violent)
#   visibility V = 1000*(1-val)^2 + 20 m (0.3 -> 510 = mist, 0.8 -> 60 = dense fog)
# Both are monotonic and land the endpoints on recognised meteorological bands.
WEATHER_CONDITIONS: dict[str, WeatherCondition] = {
    "clear": WeatherCondition(
        name="clear",
        airsim_params={},
        description="Baseline. No weather effects enabled.",
    ),
    "rain_light": WeatherCondition(
        name="rain_light",
        airsim_params={"Rain": 0.3, "Roadwetness": 0.3},
        rain_rate_mmph=15.0,
        description="Heavy-ish rain, 15 mm/h. Roadwetness tracks Rain so the "
                    "specular road response matches the precipitation.",
    ),
    "rain_heavy": WeatherCondition(
        name="rain_heavy",
        airsim_params={"Rain": 0.8, "Roadwetness": 0.8},
        rain_rate_mmph=40.0,
        description="Violent rain, 40 mm/h.",
    ),
    "fog_light": WeatherCondition(
        name="fog_light",
        airsim_params={"Fog": 0.3},
        fog_visibility_m=510.0,
        description="Mist / light fog, 510 m visibility.",
    ),
    "fog_heavy": WeatherCondition(
        name="fog_heavy",
        airsim_params={"Fog": 0.8},
        fog_visibility_m=60.0,
        description="Dense fog, 60 m visibility.",
    ),
}

BENCHMARK_ORDER = ["clear", "rain_light", "rain_heavy", "fog_light", "fog_heavy"]

# --------------------------------------------------------------------------
# SLAM tunables
# --------------------------------------------------------------------------

# -- shared ----------------------------------------------------------------
KEYFRAME_TRANS = 2.0               # m   -- new keyframe after this much travel
KEYFRAME_ROT = math.radians(15.0)  # rad

# -- LiDAR front end -------------------------------------------------------
VOXEL_SIZE = 0.3                   # m, downsample before registration
NORMAL_RADIUS = 1.0                # m, neighbourhood for normal estimation
NORMAL_MAX_NN = 30
GICP_MAX_CORRESPONDENCE = 1.5      # m
GICP_MAX_ITER = 40
# Registration is rejected below this fitness / above this RMSE; the IMU prior
# is used instead and the frame is logged as a tracking failure. That failure
# count is a headline robustness metric, not just an error path.
GICP_MIN_FITNESS = 0.35
GICP_MAX_RMSE = 1.0                # m
SUBMAP_KEYFRAMES = 12              # sliding window registered against
MAP_VOXEL_SIZE = 0.4               # m, global map density

# -- loop closure ----------------------------------------------------------
LOOP_SEARCH_RADIUS = 15.0          # m between keyframe positions
LOOP_MIN_INDEX_GAP = 40            # keyframes; excludes trivially recent ones
LOOP_FPFH_RADIUS = 2.5             # m
LOOP_RANSAC_N = 4
LOOP_RANSAC_MAX_ITER = 100_000
LOOP_RANSAC_CONFIDENCE = 0.999
LOOP_MIN_FITNESS = 0.45            # gate on accepting a closure edge
LOOP_MAX_CANDIDATES = 3            # per new keyframe, best-first
# Plausibility gate on the correction a closure implies. Odometry drift scales
# with distance travelled, so a closure demanding a much larger correction than
# the loop could have accumulated is a false match. Without this, one bad
# closure in a repetitive scene drags the entire graph off.
#
# HONEST CAVEAT: this stops the grossest false closures but is NOT sufficient.
# Verification uses RANSAC, so it is stochastic, and on byte-identical input
# ATE has been observed anywhere from 0.77 m to 26.8 m depending only on the
# RNG seed. Tightening this gate to 0.03 (with LOOP_MIN_FITNESS 0.55) was tried
# and made a sampled seed *worse* -- 5.1 m -> 23.2 m -- so the residual
# instability is not simply loose gating. The values below are the ones that
# produced the best measured results; see docs/SLAM.md "Known limitations", and
# use `tools/run_benchmark.py --repeats N` to characterise the spread rather
# than trusting a single draw.
LOOP_MAX_DRIFT_FRACTION = 0.08     # of the path length around the loop
LOOP_MIN_DRIFT_BUDGET = 3.0        # m, floor for short loops

# -- pose graph ------------------------------------------------------------
PG_MAX_CORRESPONDENCE = 1.5        # m
PG_EDGE_PRUNE = 0.25
PG_PREFERENCE_LOOP = 0.1           # Open3D's "preference_loop_closure"

# -- stereo front end ------------------------------------------------------
# Disparity is computed at half resolution and upsampled: full-res SGBM at
# 1280x720 dominates the offline runtime, and features are still detected at
# full resolution, so only the depth *lookup* is coarsened.
STEREO_DEPTH_SCALE = 0.5
SGBM_MIN_DISPARITY = 0
SGBM_NUM_DISPARITIES = 96          # must be divisible by 16
SGBM_BLOCK_SIZE = 7
SGBM_UNIQUENESS_RATIO = 10
SGBM_SPECKLE_WINDOW = 100
SGBM_SPECKLE_RANGE = 2
SGBM_DISP12_MAX_DIFF = 1

ORB_N_FEATURES = 2000
ORB_SCALE_FACTOR = 1.2
ORB_N_LEVELS = 8
ORB_MATCH_RATIO = 0.75             # Lowe ratio test
PNP_REPROJ_ERROR = 3.0             # px, RANSAC inlier threshold
PNP_ITERATIONS = 200
PNP_CONFIDENCE = 0.999
PNP_MIN_INLIERS = 20               # below this the frame is a tracking failure

BA_WINDOW = 8                      # keyframes in the windowed bundle adjustment
BA_MAX_ITER = 30
BA_HUBER_DELTA = 2.0               # px
# Cap on landmarks per BA solve. The problem is 6*(W-1) + 3*L parameters, so
# the landmark term dominates; capping keeps each solve well under a second
# even though BA runs at every keyframe.
BA_MAX_LANDMARKS = 400

# Loop closure for stereo: no DBoW vocabulary ships with OpenCV's Python
# bindings, so candidates come from raw descriptor-match counts against
# spatially-near keyframes and are then verified geometrically by PnP.
STEREO_LOOP_MIN_MATCHES = 60
STEREO_LOOP_MIN_INLIERS = 30

# -- IMU -------------------------------------------------------------------
GRAVITY_NED = np.array([0.0, 0.0, 9.80665], dtype=np.float64)   # +z is DOWN
IMU_BIAS_INIT_SECONDS = 1.0        # stationary window used to seed biases

# -- evaluation ------------------------------------------------------------
RPE_DELTA_SECONDS = 1.0
RPE_DELTA_METERS = 10.0
MAP_EVAL_VOXEL = 0.5               # m, for cloud-to-cloud comparison
MAP_EVAL_INLIER_DIST = 1.0         # m, completeness threshold
# BA is a local refinement, so a metre-scale pose jump means the solve
# diverged (usually an ill-conditioned window). Such solves are discarded.
BA_MAX_CORRECTION = 1.0            # m
