#!/usr/bin/env python3
"""
Offline self-test of fuse.py + planner.py -- no simulator.

A synthetic block world (flat terrain, a road, a house, two trees, a parked
car, a pole) is rendered into capture.py's exact on-disk format by a tiny
ray-marcher: nadir seg + depth images and LiDAR sweeps with poses along a
lawnmower. fuse() must recover the classes and split the two same-id trees
into two instances; RoadGrid must find a route down the road that passes the
parked car without touching it, and PurePursuit must keep the car on the
road for every tick.

    ./airsim_venv/bin/python panoptic/test_offline.py
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg  # noqa: E402
from camera import Intrinsics, to_world  # noqa: E402
from coverage import lawnmower  # noqa: E402
from fuse import fuse  # noqa: E402
from labels import STUFF_IDS, FIRST_THING_ID  # noqa: E402
from planner import PurePursuit, RoadGrid  # noqa: E402

T_ID, R_ID = STUFF_IDS["terrain"], STUFF_IDS["road"]
B_ID, TREE_ID, CAR_ID, POLE_ID = FIRST_THING_ID, FIRST_THING_ID + 1, FIRST_THING_ID + 2, FIRST_THING_ID + 3
IDS = {R_ID: "road", T_ID: "terrain", B_ID: "building", TREE_ID: "tree", CAR_ID: "vehicle", POLE_ID: "pole"}
PALETTE = {i: ((i * 37) % 256, (i * 91) % 256, (i * 53) % 256) for i in IDS}
CAR_XY = (10.0, 1.0)      # straddles the centre line: the route MUST swerve
# columns: (xmin, xmax, ymin, ymax, height, id)
BOXES = [(15, 25, 11, 19, 6.0, B_ID), (-22, -18, -17, -13, 8.0, TREE_ID), (38, 42, -18, -14, 8.0, TREE_ID),
         (CAR_XY[0] - 2.2, CAR_XY[0] + 2.2, CAR_XY[1] - 1, CAR_XY[1] + 1, 1.5, CAR_ID),
         (-0.3, 0.3, -4.9, -4.3, 5.0, POLE_ID)]


def surface(x, y):
    """(top surface NED z, seg id) of the block world at xy."""
    z = np.zeros_like(x)
    sid = np.full(x.shape, T_ID, dtype=np.uint8)
    road = (np.abs(y) < 5) & (np.abs(x) < 60)
    sid[road] = R_ID
    for x0, x1, y0, y1, h, i in BOXES:
        m = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        z[m] = -h
        sid[m] = i
    return z, sid


def raymarch(origin, dirs_world, t_max=90.0, step=0.25):
    """First point along each ray at/below the surface -> (hit xyz, seg id, t)."""
    n = len(dirs_world)
    t = np.zeros(n)
    hit = np.zeros(n, dtype=bool)
    out_id = np.zeros(n, dtype=np.uint8)
    for k in range(int(t_max / step)):
        tt = (k + 1) * step
        p = origin + dirs_world * tt
        s, sid = surface(p[:, 0], p[:, 1])
        new = (~hit) & (p[:, 2] >= s)
        t[new] = tt
        out_id[new] = sid[new]
        hit |= new
        if hit.all():
            break
    return origin + dirs_world * t[:, None], out_id, t, hit


R_NADIR = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=float)   # body x -> down, y -> east


def render_frame(intr: Intrinsics, pos, rng):
    T_cam = np.eye(4)
    T_cam[:3, :3] = R_NADIR
    T_cam[:3, 3] = pos
    vv, uu = np.mgrid[0:intr.height, 0:intr.width]
    d_cam = np.stack([np.ones(vv.size), (uu.ravel() - intr.cx) / intr.fx, (vv.ravel() - intr.cy) / intr.fy], 1)
    d_world = d_cam @ R_NADIR.T
    _, sid, t, hit = raymarch(pos, d_world)
    depth = np.where(hit, t * d_cam[:, 0], 1e5).reshape(intr.height, intr.width).astype(np.float32)
    seg = np.zeros((intr.height, intr.width, 3), np.uint8)
    for i, c in PALETTE.items():
        seg[sid.reshape(intr.height, intr.width) == i] = c
    # LiDAR: identity-oriented sensor at the same spot, random dirs 0..-80 deg elevation
    az = rng.uniform(-math.pi, math.pi, 6000)
    el = np.radians(rng.uniform(-80, -5, 6000))
    d_l = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), -np.sin(el)], 1)
    T_l = np.eye(4)
    T_l[:3, 3] = pos
    p_hit, _, _, hit_l = raymarch(pos, d_l)
    return T_cam, seg, depth, T_l, (p_hit[hit_l] - pos).astype(np.float32)


def build_dataset(ds: Path):
    intr = Intrinsics(160, 120, 90.0)
    bounds = (-70.0, 70.0, -30.0, 30.0)
    alt = 35.0
    wps = lawnmower(bounds, (-70, -30), -alt, intr.swath_at(alt), spacing=5.0)
    (ds / "seg").mkdir(parents=True)
    (ds / "depth").mkdir()
    (ds / "lidar").mkdir()
    rng = np.random.default_rng(0)
    index = []
    for k, (x, y, z) in enumerate(wps):
        T_cam, seg, depth, T_l, pts = render_frame(intr, np.array([x, y, z]), rng)
        ns = 1_000_000_000_000 + k * 400_000_000
        cv2.imwrite(str(ds / "seg" / f"{ns}.png"), seg)
        np.save(ds / "depth" / f"{ns}.npy", depth.astype(np.float16))
        np.save(ds / "lidar" / f"{ns}.npy", pts)
        index.append({"ns": ns, "T_cam": T_cam.reshape(-1).tolist(), "T_lidar": T_l.reshape(-1).tolist()})
    (ds / "index.json").write_text(json.dumps(index))
    (ds / "meta.json").write_text(json.dumps({
        "camera": {"width": intr.width, "height": intr.height, "fov_deg": intr.fov_deg},
        "bounds": bounds, "waypoints": wps,
        "ids": {str(i): {"class": c, "objects": [c]} for i, c in IDS.items()},
        "palette": {str(i): list(c) for i, c in PALETTE.items()},
    }))
    return len(wps)


def main():
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  PASS  " if cond else "  FAIL  ") + msg)
        ok &= bool(cond)

    with tempfile.TemporaryDirectory() as tmp:
        ds = Path(tmp) / "synth"
        n = build_dataset(ds)
        print(f"synthetic dataset: {n} frames")
        m = fuse(ds, voxel=0.5, use_lidar=True, use_dense=True, stride=1, crop_margin=5.0)
        cls = m["class_id"]
        name = lambda c: cfg.CLASS_ID[c]
        counts = {c: int((cls == name(c)).sum()) for c in cfg.CLASSES}
        print("  voxels:", {k: v for k, v in counts.items() if v})
        check(counts["road"] > 4000, "road voxels recovered")
        check(counts["terrain"] > counts["road"], "terrain dominates the flat ground")
        check(counts["unlabeled"] < 0.02 * len(cls), f"unlabeled fraction small ({counts['unlabeled']/len(cls):.3%})")
        for c in ("building", "tree", "vehicle", "pole"):
            check(counts[c] > 0, f"{c} present")
        n_inst = lambda c: len(np.unique(m["instance_id"][(cls == name(c)) & (m["instance_id"] > 0)]))
        check(n_inst("tree") == 2, f"two trees sharing one seg id split into {n_inst('tree')} instances")
        check(n_inst("building") == 1, f"house is {n_inst('building')} instance")
        check(n_inst("vehicle") == 1, f"parked car is {n_inst('vehicle')} instance")
        # house voxels really are where the house is
        hx = m["xyz"][cls == name("building")]
        check(hx[:, 0].min() > 14 and hx[:, 0].max() < 26 and hx[:, 2].min() < -5, "house geometry in place")

        grid = RoadGrid(m)
        check(grid.drivable.sum() > 0, f"{int(grid.drivable.sum())} drivable cells")
        ci, cj = grid.to_cell(*CAR_XY)
        check(grid.obstacle[ci, cj], "parked car is an obstacle cell")
        check(not grid.drivable[ci, cj], "parked car's cell is not drivable")
        pi, pj = grid.to_cell(0.0, -4.6)
        check(grid.obstacle[pi, pj], "pole is an obstacle cell")
        route = grid.route((-55, 0), (55, 0))
        d_car = np.linalg.norm(route - CAR_XY, axis=1).min()
        check(d_car > 1.0 + cfg.CAR_WIDTH / 2 + cfg.CAR_CLEARANCE - 0.05, f"route clears parked car ({d_car:.2f} m)")
        check(np.abs(route[:, 1]).max() < 5 - cfg.CAR_WIDTH / 2, f"route stays inside the road (|y| max {np.abs(route[:, 1]).max():.2f})")
        near = route[np.abs(route[:, 0] - CAR_XY[0]) < 1.0]
        check(len(near) and near[:, 1].max() < -1.4, f"route swerves left of the parked car (y={near[:, 1].max():.2f} beside it)")

        pp = PurePursuit(route)
        bad = 0
        steps = 0
        while not pp.done and steps < 20000:
            x, y, yaw = pp.step()
            on, free = grid.check_pose(x, y, yaw)
            bad += (not on) or (not free)
            steps += 1
        check(pp.done, f"car reached the goal in {steps} ticks")
        check(bad == 0, f"{bad} ticks off-road or in collision")
        check(abs(grid.z_at(0.0, 0.0)) < 0.6, f"road surface z at origin = {grid.z_at(0, 0):.2f}")
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
