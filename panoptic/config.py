"""
Tunables for the panoptic pipeline. Mirrors ``settings.panoptic.json`` --
change the camera / LiDAR blocks in both places together (AirSim only reads
settings at startup, so a change there also needs a ``run_swarm.sh`` restart).
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = REPO_ROOT / "datasets_panoptic"
PALETTE_CACHE = Path(__file__).resolve().parent / "palette_cache.json"

# -- rig (must match settings.panoptic.json) ---------------------------------
DRONE = "Drone1"
LIDAR = "LidarSensor1"
CAM = "Nadir"            # pitched -90: looks straight down
CAR_CAM = "CarCam"       # ExternalCamera repositioned to trail the car
CAM_W, CAM_H, CAM_FOV_DEG = 640, 480, 90.0   # FOV is HORIZONTAL, fx derives from W

# -- coverage flight ---------------------------------------------------------
ALTITUDE = 35.0          # m above the take-off point; AirSimNH's tallest trees are ~15 m
SPEED = 6.0              # m/s along the lawnmower
OVERLAP = 0.4            # fraction of the camera swath shared by neighbouring lines
WAYPOINT_SPACING = 10.0  # m; moveOnPathAsync cuts corners with sparse waypoints
CAPTURE_DT = 0.4         # sim seconds between lockstep captures (1.2 m at 6 m/s)
BOUNDS_MARGIN = 20.0     # m added around the discovered object extent
BOUNDS_PERCENTILE = 1.0  # discard the outermost 1 % of object positions per side

# -- labelling ---------------------------------------------------------------
MAX_DEPTH = 150.0        # DepthPlanar beyond this is sky / out of range
DEPTH_TOL_ABS = 0.5      # a LiDAR point takes a pixel's label only if its depth
DEPTH_TOL_REL = 0.02     #   agrees with DepthPlanar within abs + rel * depth
DENSE_STRIDE = 4         # back-project every Nth pixel of seg+depth as extra points

# -- fusion ------------------------------------------------------------------
VOXEL = 0.4              # m; panoptic map resolution

# -- car -------------------------------------------------------------------
CAR_LENGTH = 4.6
CAR_WIDTH = 2.0
CAR_WHEELBASE = 2.7
CAR_CLEARANCE = 0.4      # extra lateral margin to obstacles and the road edge
OBSTACLE_EXTRA = 0.3     # obstacles only: extra growth for the corner sweep in turns
CAR_SPEED = 5.0          # m/s
CAR_LOOKAHEAD = 5.0      # pure-pursuit lookahead
CAR_DT = 0.05            # s per kinematic step
CAR_MAX_STEER_DEG = 35.0
OBSTACLE_BAND = 2.2      # m above local ground that counts as blocking a car (car ~1.5 m:
                         #   AirSimNH's avenue canopies start at ~2.8 m and must not count)
GROUND_SLACK = 0.35      # voxels within this of the ground surface are not obstacles
UNLABELED_BLOCKS = False # LiDAR-only voxels (no camera label) as obstacles? In AirSimNH they
                         #   are collision-only geometry ~1 m over roads that the nadir camera
                         #   sees straight through; counting them cut the road network in half

# -- taxonomy ----------------------------------------------------------------
# "stuff" = amorphous, one segmentation id per class; "things" = countable,
# one id per mesh (falls back to sharing ids within a class past 255 meshes --
# fuse.py splits shared ids back into instances by 3-D connectivity).
UNLABELED = "unlabeled"
STUFF = ["road", "sidewalk", "terrain", "water", "other"]
THINGS = ["building", "vehicle", "tree", "bush", "pole", "fence", "prop"]
CLASSES = [UNLABELED] + STUFF + THINGS          # index = class_id
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}

# Case-insensitive substrings matched against simListSceneObjects() names,
# first hit wins, so order matters (e.g. "Streetlight" must reach "pole"
# before "street" reaches "road"). Unmatched names become "other". These are
# tuned for AirSimNH; run  capture.py --list-objects  to audit for another env.
CLASS_KEYWORDS = [
    ("pole",     ["streetlight", "lamp", "pole", "sign", "mailbox", "hydrant", "post", "power_line"]),
    ("bush",     ["bush", "hedge", "shrub", "plant", "flower", "foliage", "ivy", "leaves"]),
    ("vehicle",  ["car", "truck", "van", "bus", "vehicle", "suv", "sedan"]),
    # driveways are surface but NOT drivable for our car: they dead-end in garages
    ("sidewalk", ["sidewalk", "pavement", "curb", "kerb", "walkway", "driveway"]),
    ("road",     ["road", "street", "asphalt", "lane", "intersection", "crossing"]),
    ("water",    ["pool", "water", "pond"]),
    ("terrain",  ["landscape", "ground", "grass", "lawn", "terrain", "dirt", "soil", "field"]),
    ("building", ["house", "building", "garage", "roof", "shed", "home", "chimney",
                  "porch", "wall", "door", "window", "stair", "curtain", "shutter", "drain",
                  "cladding", "banister", "veranda", "floor", "kitchen", "beam", "pillar"]),
    ("tree",     ["tree", "oak", "pine", "birch", "palm", "maple", "spruce", "fir"]),
    ("fence",    ["fence", "gate", "railing"]),
    ("prop",     ["bench", "bin", "trash", "chair", "table", "cone", "barrel", "prop",
                  "basketball", "swing", "bbq", "grill", "boat", "rock", "tressel"]),
]

# Actors that are not meshes / never in the map; skipped from id assignment
# and from bounds discovery so a far-off sky sphere doesn't inflate the extent.
IGNORE_NAME_SUBSTRINGS = ["sky", "playerstart", "light", "fog", "camera", "drone",
                          "simpleflight", "spectator", "postprocess", "volume",
                          "brush", "defaultphysics", "gamestate", "hud",
                          "player", "controller", "airsim", "worldsettings", "netdriver"]

# Display colours per class (viewer + PLY export), RGB in [0,1].
CLASS_COLORS = {
    "unlabeled": (0.35, 0.35, 0.35),
    "road":      (0.25, 0.25, 0.28),
    "sidewalk":  (0.75, 0.70, 0.60),
    "terrain":   (0.30, 0.60, 0.25),
    "water":     (0.20, 0.45, 0.90),
    "other":     (0.55, 0.50, 0.55),
    "building":  (0.85, 0.35, 0.30),
    "vehicle":   (0.95, 0.80, 0.15),
    "tree":      (0.10, 0.40, 0.15),
    "bush":      (0.45, 0.75, 0.30),
    "pole":      (0.90, 0.55, 0.10),
    "fence":     (0.60, 0.40, 0.20),
    "prop":      (0.80, 0.30, 0.80),
}
DRIVABLE = ["road"]                        # the car may only be here
GROUND = ["road", "sidewalk", "terrain", "water"]   # surface classes (never obstacles)
