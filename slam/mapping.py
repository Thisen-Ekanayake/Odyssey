"""
Map representations: a sliding submap to register against, and a global
voxel map to accumulate and export.

Both exist to avoid the trap the old ``flight/lidar_viz.py`` fell into --
``np.vstack`` every scan onto one array and rebuild the whole Open3D cloud each
tick. That is O(total points) per frame and grinds to a halt after a minute of
flight. Here the registration target is a bounded window of recent keyframes,
and the global map is downsampled incrementally in batches.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import open3d as o3d

from . import config as cfg

__all__ = ["to_o3d", "preprocess", "SubmapWindow", "VoxelMap"]


def to_o3d(points: np.ndarray) -> o3d.geometry.PointCloud:
    """(N,3) array -> Open3D cloud. Open3D needs float64 internally."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return pcd


def preprocess(points: np.ndarray, voxel: float | None = None,
               with_normals: bool = True) -> o3d.geometry.PointCloud:
    """Downsample and (optionally) estimate normals.

    Generalized ICP needs normals on *both* clouds -- it fits a plane-to-plane
    covariance at every correspondence, which is what makes it markedly more
    robust than point-to-point on the large flat surfaces (roads, roofs, walls)
    that dominate a neighbourhood scan.
    """
    voxel = cfg.VOXEL_SIZE if voxel is None else voxel
    pcd = to_o3d(points)
    if voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
    if with_normals and len(pcd.points) >= 3:
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=cfg.NORMAL_RADIUS, max_nn=cfg.NORMAL_MAX_NN))
    return pcd


class SubmapWindow:
    """Sliding window of recent keyframe clouds, in world coordinates.

    Registering scan-to-submap rather than scan-to-scan is what keeps frame-to-
    frame drift from compounding: each new sweep is matched against the last
    ``SUBMAP_KEYFRAMES`` worth of accumulated structure, so a single bad match
    is outvoted instead of propagating.

    The target is rebuilt only when a keyframe is added or the window is
    explicitly refreshed after a pose-graph optimisation -- not per frame.
    """

    def __init__(self, capacity: int | None = None, voxel: float | None = None) -> None:
        self.capacity = cfg.SUBMAP_KEYFRAMES if capacity is None else capacity
        self.voxel = cfg.VOXEL_SIZE if voxel is None else voxel
        self._entries: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=self.capacity)
        self._target: o3d.geometry.PointCloud | None = None

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, points_local: np.ndarray, T_world: np.ndarray) -> None:
        """Add a keyframe: its sensor-frame points plus the pose that places them."""
        self._entries.append((np.asarray(points_local, dtype=np.float32), T_world.copy()))
        self._target = None

    def update_poses(self, poses: list[np.ndarray]) -> None:
        """Re-place the window's keyframes after the back end moved them.

        Called after loop closure: the submap must follow the optimised poses,
        or the front end keeps registering against a map the back end has
        already corrected.
        """
        tail = poses[-len(self._entries):] if self._entries else []
        for i, T in enumerate(tail):
            pts, _ = self._entries[i]
            self._entries[i] = (pts, np.asarray(T).copy())
        self._target = None

    def target(self) -> o3d.geometry.PointCloud | None:
        """The registration target, in world coordinates. Cached until invalidated."""
        if self._target is not None:
            return self._target
        if not self._entries:
            return None

        chunks = [pts @ T[:3, :3].T.astype(np.float32) + T[:3, 3].astype(np.float32)
                  for pts, T in self._entries]
        self._target = preprocess(np.vstack(chunks), self.voxel, with_normals=True)
        return self._target


class VoxelMap:
    """Global map, downsampled in batches.

    Points are buffered and flushed through ``voxel_down_sample`` every
    ``flush_every`` insertions rather than on every scan: downsampling is
    O(total points), so doing it per frame is what makes naive accumulators
    slow down as the map grows.
    """

    def __init__(self, voxel: float | None = None, flush_every: int = 20) -> None:
        self.voxel = cfg.MAP_VOXEL_SIZE if voxel is None else voxel
        self.flush_every = flush_every
        self._map = np.empty((0, 3), dtype=np.float32)
        self._pending: list[np.ndarray] = []

    def insert(self, points_world: np.ndarray) -> None:
        if len(points_world):
            self._pending.append(np.asarray(points_world, dtype=np.float32))
        if len(self._pending) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._pending:
            return
        merged = np.vstack([self._map] + self._pending)
        pcd = to_o3d(merged).voxel_down_sample(self.voxel)
        self._map = np.asarray(pcd.points, dtype=np.float32)
        self._pending = []

    @property
    def points(self) -> np.ndarray:
        self.flush()
        return self._map

    def __len__(self) -> int:
        return len(self._map) + sum(len(p) for p in self._pending)

    def as_pointcloud(self) -> o3d.geometry.PointCloud:
        return to_o3d(self.points)

    def save(self, path) -> None:
        o3d.io.write_point_cloud(str(path), self.as_pointcloud())

    @staticmethod
    def rebuild(keyframe_points: list[np.ndarray], poses: list[np.ndarray],
                voxel: float | None = None) -> "VoxelMap":
        """Rebuild a map from scratch under a new set of poses.

        Needed after loop closure: the optimised trajectory implies a different
        map, and incrementally patching the old one is not possible -- every
        keyframe moved.
        """
        vm = VoxelMap(voxel)
        for pts, T in zip(keyframe_points, poses):
            vm.insert(np.asarray(pts) @ T[:3, :3].T + T[:3, 3])
        vm.flush()
        return vm
