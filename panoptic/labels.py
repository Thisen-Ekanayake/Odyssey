"""
Scene-object -> (class, segmentation id) bookkeeping, and the segmentation
camera's colour palette.

AirSim's ``Segmentation`` image paints every mesh with a colour that is a
fixed function of its 0-255 object id. By default the ids come from a hash of
the mesh name, i.e. they are meaningless -- so ``assign_ids`` first resets
EVERY mesh to the shared "other" id and then hands out ids by class (one per
stuff class, one per mesh for things). The id->colour table is not exposed by
the API, so ``calibrate_palette`` measures it: paint the whole scene with one
id, grab a frame, read the colour. Measured once, cached in
``palette_cache.json``.
"""
from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import config as cfg

MAX_ID = 255
OTHER_ID = 1          # every stuff class gets a fixed low id, in STUFF order
STUFF_IDS = {name: i + 1 for i, name in enumerate(cfg.STUFF)}   # road=1 ... other=5
FIRST_THING_ID = len(cfg.STUFF) + 1


def classify(name: str) -> str | None:
    """Class name for a scene object, or None if the actor should be ignored."""
    low = name.lower()
    for cls, kws in cfg.CLASS_KEYWORDS:
        if any(k in low for k in kws):
            return cls
    if any(s in low for s in cfg.IGNORE_NAME_SUBSTRINGS):
        return None
    return "other"


def assign_ids(names: list[str]) -> tuple[dict[str, int], dict[int, dict]]:
    """name -> seg id, and seg id -> {class, objects}.

    Things get unique ids from FIRST_THING_ID..255 in name order; once the
    pool is exhausted ids are recycled within the same class (fuse.py splits
    such shared ids into instances by connectivity, so this only costs
    accuracy where two same-class meshes touch).
    """
    by_name: dict[str, int] = {}
    table: dict[int, dict] = {i: {"class": c, "objects": []} for c, i in STUFF_IDS.items()}
    things_by_class: dict[str, list[str]] = defaultdict(list)
    for n in sorted(names):
        cls = classify(n)
        if cls is None:
            continue
        if cls in STUFF_IDS:
            by_name[n] = STUFF_IDS[cls]
            table[STUFF_IDS[cls]]["objects"].append(n)
        else:
            things_by_class[cls].append(n)

    # Share the 250 thing ids across classes in proportion to sqrt(count), so
    # 1700 house parts can't starve the 70 cars: every class keeps enough ids
    # that neighbouring meshes of one class rarely share one.
    pool_size = MAX_ID - FIRST_THING_ID + 1
    weights = {c: math.sqrt(len(o)) for c, o in things_by_class.items()}
    total_w = sum(weights.values()) or 1.0
    quota = {c: max(1, min(len(things_by_class[c]), int(pool_size * w / total_w)))
             for c, w in weights.items()}
    while sum(quota.values()) > pool_size:             # rounding safety
        big = max(quota, key=quota.get)
        quota[big] -= 1
    next_id = FIRST_THING_ID
    for cls, objs in things_by_class.items():
        ids = list(range(next_id, next_id + quota[cls]))
        next_id += quota[cls]
        for sid in ids:
            table[sid] = {"class": cls, "objects": []}
        for k, n in enumerate(objs):                  # round-robin over the class's ids
            sid = ids[k % len(ids)]
            table[sid]["objects"].append(n)
            by_name[n] = sid
    return by_name, table


def apply_ids(client, by_name: dict[str, int], verbose: bool = True) -> int:
    """Push the assignment into the sim. Returns how many meshes accepted it."""
    client.simSetSegmentationObjectID(".*", STUFF_IDS["other"], True)   # reset the hash ids
    ok = 0
    items = sorted(by_name.items(), key=lambda kv: kv[1])
    for i, (name, sid) in enumerate(items):
        if sid == STUFF_IDS["other"]:
            ok += 1                       # already covered by the regex reset
            continue
        if client.simSetSegmentationObjectID(name, sid, False):
            ok += 1
        if verbose and i % 200 == 0:
            print(f"\r  segmentation ids: {i}/{len(items)}", end="", flush=True)
    if verbose:
        print(f"\r  segmentation ids: {ok}/{len(items)} meshes accepted an id")
    return ok


def _rgb(resp) -> np.ndarray:
    """Uncompressed image bytes -> (H,W,3); tolerates a 4-channel (BGRA) build."""
    buf = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
    ch = buf.size // (resp.height * resp.width)
    return buf.reshape(resp.height, resp.width, ch)[:, :, :3]


def _grab_seg(client, cam: str, vehicle: str, external: bool) -> np.ndarray:
    import airsim
    resp = client.simGetImages([airsim.ImageRequest(cam, airsim.ImageType.Segmentation, False, False)],
                               vehicle_name=vehicle, external=external)[0]
    return _rgb(resp)


def calibrate_palette(client, ids: list[int], cam: str = cfg.CAR_CAM, vehicle: str = cfg.DRONE,
                      cache: Path = cfg.PALETTE_CACHE, force: bool = False,
                      min_fraction: float = 0.9) -> dict[int, tuple]:
    """id -> (c0,c1,c2) as the raw uint8 triple the API returns (channel order
    is irrelevant as long as decode uses the same one).

    Measured through the EXTERNAL CarCam parked 30 m above the drone looking
    straight down: a big, clean view of repainted meshes. (The drone's own
    nadir camera on the ground, 15 cm from the tarmac, gave wrong colours for
    several ids without any error -- hence the >= min_fraction agreement
    check and the retry.)
    """
    import airsim
    palette: dict[int, tuple] = {}
    if cache.exists() and not force:
        palette = {int(k): tuple(v) for k, v in json.loads(cache.read_text()).items()}
    missing = [i for i in ids if i not in palette]
    if not missing:
        return palette
    print(f"  calibrating segmentation palette for {len(missing)} id(s)...")
    p = client.simGetVehiclePose(vehicle_name=vehicle).position
    client.simSetCameraPose(cam, airsim.Pose(airsim.Vector3r(p.x_val, p.y_val, p.z_val - 30.0),
                                             airsim.to_quaternion(-np.pi / 2, 0, 0)), external=True)
    for k, sid in enumerate(missing):
        client.simSetSegmentationObjectID(".*", sid, True)
        for attempt in range(4):
            _grab_seg(client, cam, vehicle, True)          # first frame may predate the change
            time.sleep(0.05)
            flat = _grab_seg(client, cam, vehicle, True).reshape(-1, 3).astype(np.int64)
            key = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]
            vals, counts = np.unique(key, return_counts=True)
            j = int(np.argmax(counts))
            if counts[j] >= min_fraction * len(key):
                break
            time.sleep(0.2)
        else:
            raise RuntimeError(f"palette calibration for id {sid}: modal colour covers only "
                               f"{counts[j]/len(key):.0%} of the frame -- is the CarCam view clear?")
        best = int(vals[j])
        palette[sid] = ((best >> 16) & 255, (best >> 8) & 255, best & 255)
        if k % 20 == 0:
            print(f"\r    {k}/{len(missing)}", end="", flush=True)
    print(f"\r    {len(missing)}/{len(missing)} done")
    if len({palette[i] for i in ids}) != len(ids):
        raise RuntimeError("two ids calibrated to the same colour; calibration is unreliable")
    cache.write_text(json.dumps({str(k): list(v) for k, v in sorted(palette.items())}, indent=1))
    return palette


class SegDecoder:
    """Segmentation RGB image -> uint8 id image via a 16 M-entry lookup table."""

    def __init__(self, palette: dict[int, tuple]):
        self.lut = np.zeros(1 << 24, dtype=np.uint8)      # unknown colours -> 0 (unlabeled)
        for sid, (a, b, c) in palette.items():
            self.lut[(int(a) << 16) | (int(b) << 8) | int(c)] = sid

    def __call__(self, img: np.ndarray) -> np.ndarray:
        f = img.reshape(-1, 3).astype(np.int64)
        return self.lut[(f[:, 0] << 16) | (f[:, 1] << 8) | f[:, 2]].reshape(img.shape[:2])


def class_of_ids(table: dict[int, dict]) -> np.ndarray:
    """(256,) seg id -> class id."""
    out = np.zeros(256, dtype=np.uint8)
    for sid, ent in table.items():
        out[int(sid)] = cfg.CLASS_ID[ent["class"]]
    return out
