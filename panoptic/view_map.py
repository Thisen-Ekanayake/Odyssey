#!/usr/bin/env python3
"""
Open3D viewer for a fuse.py panoptic map (no simulator needed).

    ./airsim_venv/bin/python panoptic/view_map.py                     # latest
    ./airsim_venv/bin/python panoptic/view_map.py --dataset datasets_panoptic/2026...
    ./airsim_venv/bin/python panoptic/view_map.py --class road --class building   # subset

Keys in the window:   C  colour by class     I  colour by instance
                      D  drivable cells only (what drive_car.py may use)
                      H  colour by height    plus Open3D's usual orbit/pan/zoom
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("DISPLAY", ":1")
os.environ["XDG_SESSION_TYPE"] = "x11"

import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg  # noqa: E402
from drive_car import ned_to_display  # noqa: E402
from fuse import class_colors, instance_colors, latest_dataset, summarize  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=None)
    ap.add_argument("--class", dest="only", action="append", help="show only these classes")
    args = ap.parse_args()
    ds = args.dataset or latest_dataset()
    m = dict(np.load(ds / "panoptic_map.npz"))
    summarize(m)
    if args.only:
        keep = np.isin(m["class_id"], [cfg.CLASS_ID[c] for c in args.only])
        m = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(keep) else v) for k, v in m.items()}

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(ned_to_display(m["xyz"]))
    palettes = {
        "class": class_colors(m["class_id"]),
        "instance": instance_colors(m["instance_id"]),
    }
    up = -m["xyz"][:, 2]
    t = np.clip((up - np.percentile(up, 2)) / max(np.ptp(up), 1e-6), 0, 1)
    palettes["height"] = np.stack([t, 0.3 * np.ones_like(t), 1 - t], axis=1)
    pc.colors = o3d.utility.Vector3dVector(palettes["class"])

    def set_mode(mode):
        def cb(vis):
            pc.colors = o3d.utility.Vector3dVector(palettes[mode])
            vis.update_geometry(pc)
            return False
        return cb

    def drivable(vis):
        from planner import RoadGrid
        g = RoadGrid(m)
        col = palettes["class"].copy() * 0.25
        i = ((m["xyz"][:, 0] - g.x0) / g.cell).astype(int)
        j = ((m["xyz"][:, 1] - g.y0) / g.cell).astype(int)
        ok = g.drivable[i, j] & (m["class_id"] == cfg.CLASS_ID["road"])
        col[ok] = (0.2, 0.9, 1.0)
        palettes["drivable"] = col
        pc.colors = o3d.utility.Vector3dVector(col)
        vis.update_geometry(pc)
        return False

    keys = {ord("C"): set_mode("class"), ord("I"): set_mode("instance"),
            ord("H"): set_mode("height"), ord("D"): drivable}
    print("keys: C class  I instance  H height  D drivable")
    o3d.visualization.draw_geometries_with_key_callbacks([pc], keys, window_name=f"Panoptic map - {ds.name}",
                                                          width=1400, height=900)


if __name__ == "__main__":
    main()
