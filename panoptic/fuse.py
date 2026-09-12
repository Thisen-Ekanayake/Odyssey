#!/usr/bin/env python3
"""
Step 2: turn a capture.py recording into a 3-D panoptic map -- offline, no
simulator needed.

Per frame, two sources of labelled points go into one voxel vote:
  * LiDAR sweep, placed by its ground-truth sensor pose, each point projected
    into the nadir camera and given the segmentation id of the pixel it lands
    on -- but only if its range agrees with DepthPlanar there (otherwise it is
    an occluded point behind whatever the camera saw, and stays unlabeled)
  * the seg + depth images themselves, back-projected on a pixel stride, which
    fills in the ground densely where the sparse LiDAR would leave gaps

Each voxel takes its most-voted id ("unlabeled" never outvotes a real id).
Stuff classes are done at that point; thing ids are split into instances by
26-connectivity, which also repairs ids that capture.py had to share between
meshes once the 255-id budget ran out.

    ./airsim_venv/bin/python panoptic/fuse.py                       # latest dataset
    ./airsim_venv/bin/python panoptic/fuse.py --dataset datasets_panoptic/2026...
    ./airsim_venv/bin/python panoptic/fuse.py --voxel 0.25 --no-dense

Writes into the dataset dir:
    panoptic_map.npz   xyz (N,3) voxel centres NED, seg_id, class_id, instance_id
    panoptic_class.ply / panoptic_instance.ply   coloured previews
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg  # noqa: E402
from camera import Intrinsics, to_local, to_world  # noqa: E402
from labels import FIRST_THING_ID, SegDecoder, class_of_ids  # noqa: E402

BITS = 18                       # per axis -> 2^18 voxels; at 0.4 m that is 105 km
OFF = 1 << (BITS - 1)
MASK = (1 << BITS) - 1


def pack(idx: np.ndarray) -> np.ndarray:
    i = idx.astype(np.int64) + OFF
    if (i < 0).any() or (i > MASK).any():
        raise ValueError("point outside the voxel key range")
    return (i[:, 0] << (2 * BITS)) | (i[:, 1] << BITS) | i[:, 2]


def unpack(key: np.ndarray) -> np.ndarray:
    return np.stack([(key >> (2 * BITS)) & MASK, (key >> BITS) & MASK, key & MASK], axis=1) - OFF


class VoxelVotes:
    """Sparse (voxel, id) -> count accumulator with periodic compaction."""

    def __init__(self, voxel: float):
        self.voxel = voxel
        self._keys: list[np.ndarray] = []
        self._counts: list[np.ndarray] = []
        self._pending = 0

    def add(self, pts_world: np.ndarray, ids: np.ndarray) -> None:
        if len(pts_world) == 0:
            return
        vox = pack(np.floor(pts_world / self.voxel))
        packed = (vox << 8) | ids.astype(np.int64)
        u, c = np.unique(packed, return_counts=True)
        self._keys.append(u)
        self._counts.append(c)
        self._pending += 1
        if self._pending >= 50:
            self.compact()

    def compact(self) -> None:
        if not self._keys:
            return
        k = np.concatenate(self._keys)
        c = np.concatenate(self._counts)
        u, inv = np.unique(k, return_inverse=True)
        self._keys, self._counts = [u], [np.bincount(inv, weights=c).astype(np.int64)]
        self._pending = 0

    def resolve(self):
        """-> (voxel keys, winning seg id, votes for it) one row per voxel."""
        self.compact()
        if not self._keys:
            return np.zeros(0, np.int64), np.zeros(0, np.uint8), np.zeros(0, np.int64)
        k, c = self._keys[0], self._counts[0]
        vox, sid = k >> 8, (k & 255).astype(np.uint8)
        eff = np.where(sid == 0, 0, c)             # unlabeled only wins by default
        order = np.lexsort((-eff, vox))            # by voxel, best vote first
        vox, sid, c = vox[order], sid[order], c[order]
        _, first = np.unique(vox, return_index=True)
        return vox[first], sid[first], c[first]


def label_lidar(pts_local, T_lidar, T_cam, seg_ids, depth, intr, tol_abs, tol_rel):
    pts_w = to_world(pts_local, T_lidar)
    p_c = to_local(pts_w, T_cam)
    u, v, x = intr.project(p_c)
    ok = (x > 0.1) & (u >= 0) & (u < intr.width - 1) & (v >= 0) & (v < intr.height - 1)
    ids = np.zeros(len(pts_w), dtype=np.uint8)
    ui, vi = u[ok].astype(int), v[ok].astype(int)
    d = depth[vi, ui]
    agree = np.abs(x[ok] - d) < (tol_abs + tol_rel * x[ok])
    sel = np.flatnonzero(ok)[agree]
    ids[sel] = seg_ids[vi[agree], ui[agree]]
    return pts_w, ids


def split_instances(keys: np.ndarray, seg_id: np.ndarray, thing: np.ndarray) -> np.ndarray:
    """Instance id per voxel (0 for stuff): 26-connected components of equal seg id."""
    inst = np.zeros(len(keys), dtype=np.int32)
    sel = np.flatnonzero(thing)
    if len(sel) == 0:
        return inst
    k = keys[sel]
    order = np.argsort(k)
    k, sel = k[order], sel[order]
    sid = seg_id[sel]
    idx = unpack(k)
    rows, cols = [], []
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
               if (dx, dy, dz) > (0, 0, 0)]        # half neighbourhood, edges are symmetric
    for off in offsets:
        nk = pack(idx + np.array(off))
        j = np.searchsorted(k, nk)
        j[j >= len(k)] = 0
        hit = (k[j] == nk) & (sid[j] == sid)
        rows.append(np.flatnonzero(hit))
        cols.append(j[hit])
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    g = coo_matrix((np.ones(len(r), dtype=np.int8), (r, c)), shape=(len(k), len(k)))
    _, comp = connected_components(g, directed=False)
    inst[sel] = comp + 1
    return inst


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pc.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
    o3d.io.write_point_cloud(str(path), pc)


def instance_colors(instance_id: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(0)
    lut = rng.uniform(0.15, 1.0, size=(int(instance_id.max()) + 1, 3))
    lut[0] = 0.3
    return lut[instance_id]


def class_colors(class_id: np.ndarray) -> np.ndarray:
    lut = np.array([cfg.CLASS_COLORS[c] for c in cfg.CLASSES])
    return lut[class_id]


def latest_dataset() -> Path:
    runs = sorted(p for p in cfg.DATASET_ROOT.glob("*") if (p / "index.json").exists())
    if not runs:
        sys.exit(f"no capture under {cfg.DATASET_ROOT}; run panoptic/capture.py first")
    return runs[-1]


def fuse(ds: Path, voxel: float, use_lidar: bool, use_dense: bool, stride: int,
         crop_margin: float | None) -> dict:
    meta = json.loads((ds / "meta.json").read_text())
    index = json.loads((ds / "index.json").read_text())
    cam = meta["camera"]
    intr = Intrinsics(cam["width"], cam["height"], cam["fov_deg"])
    decoder = SegDecoder({int(k): tuple(v) for k, v in meta["palette"].items()})
    table = {int(k): v for k, v in meta["ids"].items()}
    cls_of = class_of_ids(table)
    bounds = meta["bounds"]

    votes = VoxelVotes(voxel)
    n_lidar = n_dense = 0
    for i, fr in enumerate(index):
        ns = fr["ns"]
        T_cam = np.array(fr["T_cam"]).reshape(4, 4)
        seg_img = cv2.imread(str(ds / "seg" / f"{ns}.png"), cv2.IMREAD_UNCHANGED)
        depth = np.load(ds / "depth" / f"{ns}.npy").astype(np.float32)
        if seg_img is None:
            continue
        seg_ids = decoder(seg_img)
        depth[~np.isfinite(depth) | (depth > cfg.MAX_DEPTH)] = np.nan

        chunks = []
        if use_lidar and fr["T_lidar"] is not None and (ds / "lidar" / f"{ns}.npy").exists():
            pts = np.load(ds / "lidar" / f"{ns}.npy")
            d_chk = np.nan_to_num(depth, nan=-1.0)
            pw, ids = label_lidar(pts, np.array(fr["T_lidar"]).reshape(4, 4), T_cam,
                                  seg_ids, d_chk, intr, cfg.DEPTH_TOL_ABS, cfg.DEPTH_TOL_REL)
            chunks.append((pw, ids))
            n_lidar += len(pw)
        if use_dense:
            p_c, vv, uu = intr.backproject(depth, stride, cfg.MAX_DEPTH)
            chunks.append((to_world(p_c, T_cam), seg_ids[vv, uu]))
            n_dense += len(p_c)
        for pw, ids in chunks:
            if crop_margin is not None:
                keep = ((pw[:, 0] > bounds[0] - crop_margin) & (pw[:, 0] < bounds[1] + crop_margin) &
                        (pw[:, 1] > bounds[2] - crop_margin) & (pw[:, 1] < bounds[3] + crop_margin))
                pw, ids = pw[keep], ids[keep]
            votes.add(pw, ids)
        if i % 25 == 0:
            print(f"\r  frame {i}/{len(index)}  lidar pts {n_lidar/1e6:.1f} M  "
                  f"dense pts {n_dense/1e6:.1f} M", end="", flush=True)
    print()

    keys, seg_id, _ = votes.resolve()
    class_id = cls_of[seg_id]
    thing = seg_id >= FIRST_THING_ID
    instance_id = split_instances(keys, seg_id, thing)
    xyz = ((unpack(keys) + 0.5) * voxel).astype(np.float32)
    return {"xyz": xyz, "seg_id": seg_id, "class_id": class_id, "instance_id": instance_id,
            "voxel": voxel, "classes": np.array(cfg.CLASSES), "bounds": np.array(bounds)}


def summarize(m: dict) -> None:
    cls = m["class_id"]
    print(f"  {len(cls)} voxels at {m['voxel']} m")
    for i, name in enumerate(cfg.CLASSES):
        n = int((cls == i).sum())
        if n == 0:
            continue
        inst = np.unique(m["instance_id"][(cls == i) & (m["instance_id"] > 0)])
        print(f"    {name:10s} {n:9d} voxels" + (f"  {len(inst)} instances" if len(inst) else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=None)
    ap.add_argument("--voxel", type=float, default=cfg.VOXEL)
    ap.add_argument("--no-lidar", action="store_true")
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--stride", type=int, default=cfg.DENSE_STRIDE)
    ap.add_argument("--crop-margin", type=float, default=10.0,
                    help="drop points further than this outside the flight bounds; <0 to keep all")
    args = ap.parse_args()
    ds = args.dataset or latest_dataset()
    print(f"fusing {ds}")
    m = fuse(ds, args.voxel, not args.no_lidar, not args.no_dense, args.stride,
             None if args.crop_margin < 0 else args.crop_margin)
    np.savez_compressed(ds / "panoptic_map.npz", **m)
    write_ply(ds / "panoptic_class.ply", m["xyz"], class_colors(m["class_id"]))
    write_ply(ds / "panoptic_instance.ply", m["xyz"], instance_colors(m["instance_id"]))
    summarize(m)
    print(f"  -> {ds / 'panoptic_map.npz'}")
    print(f"next: ./airsim_venv/bin/python panoptic/view_map.py --dataset {ds}")


if __name__ == "__main__":
    main()
