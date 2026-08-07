"""
Trajectory and map metrics.

Standard SLAM evaluation, implemented locally because ``evo`` is not installed
and the study is committed to adding no dependencies. Each metric is the usual
definition, stated explicitly so the numbers can be compared with published
ones:

* **ATE** -- absolute trajectory error. Estimated and ground-truth trajectories
  are associated by timestamp, aligned with a single rigid SE(3) transform
  (Umeyama), and the residual translation errors are summarised as RMSE. The
  alignment is necessary because a SLAM trajectory lives in its own frame,
  anchored to wherever the first keyframe happened to be.
* **RPE** -- relative pose error over fixed deltas, in both time (1 s) and
  distance (10 m). ATE is dominated by whether loop closure fired; RPE measures
  local drift and is the fairer read on odometry quality.
* **Map accuracy** -- cloud-to-cloud distance against a reference map, plus
  completeness (fraction of reference points explained). A trajectory can score
  well while the map is smeared, so both are reported.
* **Robustness / runtime** -- tracking-failure rate and per-stage ms/frame,
  carried through from the front end.

Scale is reported as a diagnostic but *not* corrected for: both methods here
are metric (LiDAR ranges, stereo baseline), so a scale drift away from 1.0 is a
real error, not a gauge freedom to be quotiented out.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from . import config as cfg
from .geometry import interpolate_trajectory, invert_se3, rotation_angle, umeyama

__all__ = ["TrajectoryMetrics", "MapMetrics", "evaluate_trajectory",
           "evaluate_map", "reference_map_from_ground_truth"]


@dataclass
class TrajectoryMetrics:
    n_pairs: int = 0
    ate_rmse: float = float("nan")
    ate_mean: float = float("nan")
    ate_median: float = float("nan")
    ate_max: float = float("nan")
    ate_std: float = float("nan")
    rpe_trans_rmse_1s: float = float("nan")     # m per 1 s
    rpe_rot_rmse_1s: float = float("nan")       # deg per 1 s
    rpe_trans_pct_10m: float = float("nan")     # % of 10 m travelled
    rpe_rot_deg_per_m: float = float("nan")
    final_drift: float = float("nan")           # m, endpoint error
    trajectory_length: float = float("nan")     # m, ground truth
    drift_pct: float = float("nan")             # final drift / length
    scale_estimate: float = float("nan")        # diagnostic; should be ~1.0
    diverged: bool = False

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        if self.diverged:
            return "DIVERGED (no usable trajectory)"
        return (f"ATE {self.ate_rmse:.3f} m rmse / {self.ate_max:.3f} m max  |  "
                f"RPE {self.rpe_trans_pct_10m:.2f}% per 10 m, "
                f"{self.rpe_rot_deg_per_m:.3f} deg/m  |  "
                f"drift {self.drift_pct:.2f}% of {self.trajectory_length:.0f} m")


@dataclass
class MapMetrics:
    n_points: int = 0
    rmse: float = float("nan")           # m, estimated -> reference
    mean_error: float = float("nan")
    median_error: float = float("nan")
    completeness: float = float("nan")   # fraction of reference within threshold
    accuracy: float = float("nan")       # fraction of estimated within threshold

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (f"map {self.n_points} pts, {self.rmse:.3f} m rmse, "
                f"{100 * self.accuracy:.1f}% accurate, "
                f"{100 * self.completeness:.1f}% complete")


# --------------------------------------------------------------------------
# association
# --------------------------------------------------------------------------

def associate(est_t: np.ndarray, est_T: np.ndarray,
              gt_t: np.ndarray, gt_T: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match estimates to ground truth in time by interpolating the reference.

    Ground truth is recorded at 100 Hz and estimates at 10 Hz, so interpolating
    the dense signal onto the sparse one is both cheap and more accurate than
    nearest-neighbour matching. Estimates outside the ground-truth span are
    dropped rather than extrapolated.
    """
    if len(est_t) == 0 or len(gt_t) < 2:
        return np.empty(0), np.empty((0, 4, 4)), np.empty((0, 4, 4))

    order = np.argsort(gt_t)
    gt_t, gt_T = gt_t[order], gt_T[order]

    kept_t, gt_interp = interpolate_trajectory(gt_t, gt_T, est_t)
    if len(kept_t) == 0:
        return np.empty(0), np.empty((0, 4, 4)), np.empty((0, 4, 4))

    mask = np.isin(est_t, kept_t)
    return kept_t, np.asarray(est_T)[mask], gt_interp


# --------------------------------------------------------------------------
# trajectory metrics
# --------------------------------------------------------------------------

def evaluate_trajectory(est_t, est_T, gt_t, gt_T,
                        align: bool = True) -> TrajectoryMetrics:
    """ATE + RPE for an estimated trajectory against ground truth."""
    m = TrajectoryMetrics()

    est_t = np.asarray(est_t)
    est_T = np.asarray(est_T)
    if len(est_t) < 3 or len(gt_t) < 3:
        m.diverged = True
        return m

    t, est, gt = associate(est_t, est_T, np.asarray(gt_t), np.asarray(gt_T))
    if len(t) < 3:
        m.diverged = True
        return m

    m.n_pairs = len(t)
    p_est, p_gt = est[:, :3, 3], gt[:, :3, 3]

    m.trajectory_length = float(np.linalg.norm(np.diff(p_gt, axis=0), axis=1).sum())

    # -- ATE ---------------------------------------------------------------
    if align:
        R, tr, _ = umeyama(p_est, p_gt, with_scale=False)
        aligned = p_est @ R.T + tr
        # Scale is measured separately as a diagnostic, never applied.
        _, _, m.scale_estimate = umeyama(p_est, p_gt, with_scale=True)
    else:
        aligned = p_est
        m.scale_estimate = 1.0

    err = np.linalg.norm(aligned - p_gt, axis=1)
    m.ate_rmse = float(np.sqrt((err ** 2).mean()))
    m.ate_mean = float(err.mean())
    m.ate_median = float(np.median(err))
    m.ate_max = float(err.max())
    m.ate_std = float(err.std())
    m.final_drift = float(err[-1])
    m.drift_pct = (100.0 * m.final_drift / m.trajectory_length
                   if m.trajectory_length > 1e-6 else float("nan"))

    # An estimate this far off is not a measurement of accuracy, it is a
    # failure -- flagged so the benchmark can report it as such rather than
    # letting a meaningless RMSE into a table.
    m.diverged = bool(m.ate_rmse > 0.25 * max(m.trajectory_length, 1.0))

    # -- RPE over a fixed time delta ---------------------------------------
    dt_med = float(np.median(np.diff(t))) if len(t) > 1 else 0.1
    step = max(1, int(round(cfg.RPE_DELTA_SECONDS / max(dt_med, 1e-6))))
    tr_err, rot_err = _relative_errors(est, gt, step)
    if len(tr_err):
        m.rpe_trans_rmse_1s = float(np.sqrt((tr_err ** 2).mean()))
        m.rpe_rot_rmse_1s = float(np.degrees(np.sqrt((rot_err ** 2).mean())))

    # -- RPE over a fixed distance delta -----------------------------------
    # Normalising by distance rather than time makes the number comparable
    # across runs that flew at different speeds.
    dist = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p_gt, axis=0), axis=1))])
    idx_pairs = _pairs_by_distance(dist, cfg.RPE_DELTA_METERS)
    if idx_pairs:
        i0 = np.array([a for a, _ in idx_pairs])
        i1 = np.array([b for _, b in idx_pairs])
        tr_e, rot_e, travelled = [], [], []
        for a, b in zip(i0, i1):
            d_est = invert_se3(est[a]) @ est[b]
            d_gt = invert_se3(gt[a]) @ gt[b]
            d = invert_se3(d_gt) @ d_est
            tr_e.append(np.linalg.norm(d[:3, 3]))
            rot_e.append(rotation_angle(d[:3, :3]))
            travelled.append(dist[b] - dist[a])
        tr_e = np.array(tr_e); rot_e = np.array(rot_e); travelled = np.array(travelled)
        m.rpe_trans_pct_10m = float(100.0 * np.sqrt((tr_e ** 2).mean()) / travelled.mean())
        m.rpe_rot_deg_per_m = float(np.degrees(rot_e.sum()) / max(travelled.sum(), 1e-6))

    return m


def _relative_errors(est: np.ndarray, gt: np.ndarray, step: int):
    """Relative pose error between poses ``step`` apart."""
    n = len(est) - step
    if n <= 0:
        return np.empty(0), np.empty(0)
    tr, rot = np.empty(n), np.empty(n)
    for i in range(n):
        d_est = invert_se3(est[i]) @ est[i + step]
        d_gt = invert_se3(gt[i]) @ gt[i + step]
        d = invert_se3(d_gt) @ d_est
        tr[i] = np.linalg.norm(d[:3, 3])
        rot[i] = rotation_angle(d[:3, :3])
    return tr, rot


def _pairs_by_distance(dist: np.ndarray, delta: float) -> list[tuple[int, int]]:
    """Non-overlapping index pairs spanning ``delta`` metres of travel."""
    pairs, i = [], 0
    n = len(dist)
    while i < n - 1:
        j = int(np.searchsorted(dist, dist[i] + delta))
        if j >= n:
            break
        pairs.append((i, j))
        i = j
    return pairs


# --------------------------------------------------------------------------
# map metrics
# --------------------------------------------------------------------------

def reference_map_from_ground_truth(source, voxel: float | None = None,
                                    max_scans: int | None = None) -> np.ndarray:
    """Build the reference map: recorded scans placed by ground-truth poses.

    This is the best map the sensor could produce with perfect localisation, so
    comparing an estimated map against it isolates the SLAM error from the
    sensor's own limitations. Built from the *clear* dataset -- a degraded map
    is not a sensible reference for anything.
    """
    import open3d as o3d

    voxel = cfg.MAP_EVAL_VOXEL if voxel is None else voxel
    chunks, n = [], 0
    for frame in source:
        if frame.lidar is None or frame.lidar.gt_pose is None:
            continue
        T = frame.lidar.gt_pose
        chunks.append(frame.lidar.points @ T[:3, :3].T + T[:3, 3])
        n += 1
        if max_scans and n >= max_scans:
            break
    if not chunks:
        return np.empty((0, 3))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack(chunks).astype(np.float64))
    return np.asarray(pcd.voxel_down_sample(voxel).points)


def evaluate_map(estimated: np.ndarray, reference: np.ndarray,
                 voxel: float | None = None,
                 inlier_dist: float | None = None) -> MapMetrics:
    """Cloud-to-cloud accuracy and completeness against a reference map.

    * accuracy -- fraction of ESTIMATED points close to the reference
      (low if the map has spurious structure, e.g. weather clutter)
    * completeness -- fraction of REFERENCE points close to the estimate
      (low if the map has holes, e.g. from attenuation dropout)

    Reporting both matters: attenuation and clutter fail in opposite
    directions, and a single number would hide which is happening.
    """
    from scipy.spatial import cKDTree

    voxel = cfg.MAP_EVAL_VOXEL if voxel is None else voxel
    inlier_dist = cfg.MAP_EVAL_INLIER_DIST if inlier_dist is None else inlier_dist

    m = MapMetrics()
    if len(estimated) == 0 or len(reference) == 0:
        return m

    est = _downsample(estimated, voxel)
    ref = _downsample(reference, voxel)
    m.n_points = len(est)

    d_est = cKDTree(ref).query(est)[0]
    m.rmse = float(np.sqrt((d_est ** 2).mean()))
    m.mean_error = float(d_est.mean())
    m.median_error = float(np.median(d_est))
    m.accuracy = float((d_est < inlier_dist).mean())

    d_ref = cKDTree(est).query(ref)[0]
    m.completeness = float((d_ref < inlier_dist).mean())
    return m


def _downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    """Voxel downsample via integer-grid uniquing (no Open3D round-trip)."""
    if voxel <= 0 or len(points) == 0:
        return np.asarray(points, dtype=np.float64)
    keys = np.floor(np.asarray(points, dtype=np.float64) / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return np.asarray(points, dtype=np.float64)[np.sort(idx)]
