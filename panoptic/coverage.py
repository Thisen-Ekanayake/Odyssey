"""
Where the environment is, and a lawnmower flight that covers all of it.

Bounds are discovered from the sim itself: the positions of every classified
scene object (roads, houses, trees ...), trimmed by a percentile so one stray
far-off mesh can't blow the extent up, plus a margin. Override with
``capture.py --bounds``.

The lawnmower runs corner-to-corner, so the drone visits all four corners of
the env: the first pass starts at the corner nearest to take-off, every pass
ends on the opposite side, and the last pass finishes at a far corner.
"""
from __future__ import annotations

import math

import numpy as np

import config as cfg
from labels import classify


def discover_bounds(client, names: list[str], percentile: float = cfg.BOUNDS_PERCENTILE,
                    margin: float = cfg.BOUNDS_MARGIN, verbose: bool = True):
    """(xmin, xmax, ymin, ymax) in world NED metres, from object positions."""
    pts = []
    keep = [n for n in names if classify(n) not in (None, "other")]
    for i, n in enumerate(keep):
        p = client.simGetObjectPose(n).position
        if all(math.isfinite(v) for v in (p.x_val, p.y_val)):
            pts.append((p.x_val, p.y_val))
        if verbose and i % 200 == 0:
            print(f"\r  bounds: polled {i}/{len(keep)} objects", end="", flush=True)
    if verbose:
        print(f"\r  bounds: polled {len(keep)} objects, {len(pts)} with a finite pose")
    if len(pts) < 10:
        raise RuntimeError("too few classified objects to infer the env extent; pass --bounds")
    a = np.asarray(pts)
    lo = np.percentile(a, percentile, axis=0) - margin
    hi = np.percentile(a, 100 - percentile, axis=0) + margin
    return float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])


def lawnmower(bounds, start_xy, altitude_z: float, swath: float,
              overlap: float = cfg.OVERLAP, spacing: float = cfg.WAYPOINT_SPACING):
    """Densified (x, y, z) waypoints in NED, first waypoint = nearest corner."""
    xmin, xmax, ymin, ymax = bounds
    step = swath * (1.0 - overlap)
    # sweep along the LONGER side so there are fewer turns
    along_x = (xmax - xmin) >= (ymax - ymin)
    if not along_x:
        xmin, xmax, ymin, ymax = ymin, ymax, xmin, xmax      # work in swapped coords
        start_xy = (start_xy[1], start_xy[0])
    n_lines = max(2, int(math.ceil((ymax - ymin) / step)) + 1)
    ys = np.linspace(ymin, ymax, n_lines)

    corners = {(xmin, ymin): (False, False), (xmax, ymin): (True, False),
               (xmin, ymax): (False, True), (xmax, ymax): (True, True)}
    c0 = min(corners, key=lambda c: (c[0] - start_xy[0]) ** 2 + (c[1] - start_xy[1]) ** 2)
    from_xmax, from_ymax = corners[c0]
    if from_ymax:
        ys = ys[::-1]
    x_ends = (xmax, xmin) if from_xmax else (xmin, xmax)

    wps = []
    for i, y in enumerate(ys):
        xa, xb = x_ends if i % 2 == 0 else x_ends[::-1]
        n = max(2, int(abs(xb - xa) / spacing) + 1)
        for x in np.linspace(xa, xb, n):
            wps.append((x, y))
    if not along_x:
        wps = [(y, x) for x, y in wps]
    return [(float(x), float(y), float(altitude_z)) for x, y in wps]


def path_length(wps) -> float:
    a = np.asarray(wps)[:, :2]
    return float(np.linalg.norm(np.diff(a, axis=0), axis=1).sum())
