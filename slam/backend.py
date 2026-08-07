"""
Pose-graph back end, shared by BOTH front ends.

Sharing it is deliberate and load-bearing for the study: if the LiDAR and
stereo pipelines optimised their trajectories differently, any difference in
the results would be partly attributable to the back end. With one back end,
the comparison isolates the front end -- which is the actual question.

Open3D's pose-graph conventions, which are easy to get backwards:

  * ``PoseGraphNode.pose`` is ``T_world_node`` -- it maps points from the
    node's own frame INTO the world.
  * ``PoseGraphEdge(source, target, transformation, ...)``'s transformation is
    ``T_target_source`` -- it maps the SOURCE frame into the TARGET frame, i.e.
    ``inv(T_world_target) @ T_world_source``.

Getting either backwards produces a graph that optimises to a plausible-looking
but completely wrong trajectory, so ``_relative`` below is the single place the
convention is applied.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from . import config as cfg
from .geometry import invert_se3, rotation_angle
from .mapping import preprocess

__all__ = ["LoopClosure", "PoseGraphBackend"]

_reg = o3d.pipelines.registration


@dataclass
class LoopClosure:
    """One accepted loop-closure edge, kept for reporting."""
    source: int
    target: int
    fitness: float
    rmse: float
    translation_correction: float   # m of drift the closure removed


class PoseGraphBackend:
    """Incremental pose graph with geometric loop closure.

    Keyframes are added with an odometry edge to their predecessor. Each new
    keyframe is then tested against spatially-near, temporally-distant earlier
    keyframes; surviving matches become loop edges and the graph is optimised.
    """

    def __init__(self, voxel: float | None = None, verbose: bool = False) -> None:
        self.voxel = cfg.VOXEL_SIZE if voxel is None else voxel
        self.verbose = verbose

        self.poses: list[np.ndarray] = []            # T_world_keyframe
        self.clouds: list[o3d.geometry.PointCloud] = []   # keyframe frame, downsampled
        self.times: list[float] = []
        self.loops: list[LoopClosure] = []

        # Authoritative edge list. Open3D's global_optimization PRUNES edges
        # from the PoseGraph it is handed (edge_prune_threshold), so optimising
        # the same graph object repeatedly -- as an incremental system must --
        # erodes it a little each time until even the odometry chain is gone,
        # leaving an under-constrained graph that optimises to nonsense.
        # Keeping our own list and rebuilding a fresh PoseGraph per call means
        # pruning only ever applies within a single optimisation.
        self._edges: list[tuple[int, int, np.ndarray, np.ndarray, bool]] = []

        self._features: list[o3d.pipelines.registration.Feature | None] = []
        self._kdtree: cKDTree | None = None
        self._kdtree_n = 0

    # -- graph construction ------------------------------------------------

    @staticmethod
    def _relative(T_world_source: np.ndarray, T_world_target: np.ndarray) -> np.ndarray:
        """Edge transformation for Open3D: maps SOURCE into TARGET."""
        return invert_se3(T_world_target) @ T_world_source

    def add_keyframe(self, points_local: np.ndarray, T_world: np.ndarray,
                     t: float = 0.0) -> int:
        """Append a keyframe and its odometry edge. Returns the new index."""
        pcd = preprocess(points_local, self.voxel, with_normals=True)
        idx = len(self.poses)

        self.poses.append(np.asarray(T_world, dtype=np.float64).copy())
        self.clouds.append(pcd)
        self.times.append(t)
        self._features.append(None)     # computed lazily, only if used as a candidate

        if idx > 0:
            src, tgt = idx - 1, idx
            trans = self._relative(self.poses[src], self.poses[tgt])
            info = self._information(self.clouds[src], self.clouds[tgt], trans)
            self._edges.append((src, tgt, trans, info, False))

        self._kdtree = None
        return idx

    def _information(self, source, target, transformation) -> np.ndarray:
        try:
            return _reg.get_information_matrix_from_point_clouds(
                source, target, cfg.PG_MAX_CORRESPONDENCE, transformation)
        except Exception:
            return np.eye(6)

    # -- loop closure ------------------------------------------------------

    def _positions(self) -> np.ndarray:
        return np.array([T[:3, 3] for T in self.poses])

    def _feature(self, idx: int):
        """FPFH descriptors, computed on demand and cached.

        Only a handful of keyframes are ever loop candidates, so computing
        features for all of them up front would dominate the runtime.
        """
        if self._features[idx] is None:
            pcd = self.clouds[idx]
            self._features[idx] = _reg.compute_fpfh_feature(
                pcd, o3d.geometry.KDTreeSearchParamHybrid(
                    radius=cfg.LOOP_FPFH_RADIUS, max_nn=100))
        return self._features[idx]

    def find_candidates(self, idx: int) -> list[int]:
        """Earlier keyframes near ``idx`` in space but far from it in sequence.

        The index gap is what makes this a *loop* closure rather than a
        restatement of odometry: neighbours from a few seconds ago are near by
        construction and carry no new information.
        """
        if idx < cfg.LOOP_MIN_INDEX_GAP:
            return []

        eligible = idx - cfg.LOOP_MIN_INDEX_GAP
        if self._kdtree is None or self._kdtree_n != eligible:
            if eligible <= 0:
                return []
            self._kdtree = cKDTree(self._positions()[:eligible])
            self._kdtree_n = eligible

        near = self._kdtree.query_ball_point(self.poses[idx][:3, 3], cfg.LOOP_SEARCH_RADIUS)
        if not near:
            return []

        # Nearest first -- the best-conditioned matches are tried before the
        # per-keyframe budget runs out.
        pos = self.poses[idx][:3, 3]
        near.sort(key=lambda j: float(np.linalg.norm(self.poses[j][:3, 3] - pos)))
        return near[:cfg.LOOP_MAX_CANDIDATES]

    def verify(self, source: int, target: int) -> tuple[bool, np.ndarray, float, float]:
        """Global RANSAC registration, refined by GICP, behind a fitness gate.

        Global first, because at loop-closure time the accumulated drift can be
        metres -- far outside ICP's basin of convergence, so a local method
        seeded with the current (drifted) estimate would confidently converge to
        the wrong answer.
        """
        src_pcd, tgt_pcd = self.clouds[source], self.clouds[target]
        if len(src_pcd.points) < 100 or len(tgt_pcd.points) < 100:
            return False, np.eye(4), 0.0, np.inf

        try:
            result = _reg.registration_ransac_based_on_feature_matching(
                src_pcd, tgt_pcd, self._feature(source), self._feature(target),
                mutual_filter=True,
                max_correspondence_distance=cfg.LOOP_FPFH_RADIUS,
                estimation_method=_reg.TransformationEstimationPointToPoint(False),
                ransac_n=cfg.LOOP_RANSAC_N,
                checkers=[
                    _reg.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    _reg.CorrespondenceCheckerBasedOnDistance(cfg.LOOP_FPFH_RADIUS),
                ],
                criteria=_reg.RANSACConvergenceCriteria(
                    cfg.LOOP_RANSAC_MAX_ITER, cfg.LOOP_RANSAC_CONFIDENCE),
            )
        except Exception:
            return False, np.eye(4), 0.0, np.inf

        if result.fitness < 0.1:
            return False, np.eye(4), float(result.fitness), float(result.inlier_rmse)

        refined = _reg.registration_generalized_icp(
            src_pcd, tgt_pcd, cfg.GICP_MAX_CORRESPONDENCE, result.transformation,
            _reg.TransformationEstimationForGeneralizedICP(),
            _reg.ICPConvergenceCriteria(max_iteration=cfg.GICP_MAX_ITER))

        ok = (refined.fitness >= cfg.LOOP_MIN_FITNESS
              and refined.inlier_rmse <= cfg.GICP_MAX_RMSE)
        return ok, np.asarray(refined.transformation), float(refined.fitness), \
            float(refined.inlier_rmse)

    def _path_length(self, a: int, b: int) -> float:
        """Distance travelled along the trajectory between two keyframes."""
        lo, hi = (a, b) if a < b else (b, a)
        return float(sum(
            np.linalg.norm(self.poses[i + 1][:3, 3] - self.poses[i][:3, 3])
            for i in range(lo, hi)))

    def try_close_loops(self, idx: int) -> list[LoopClosure]:
        """Test ``idx`` against its candidates; add an edge for each accepted match."""
        found = []
        for j in self.find_candidates(idx):
            ok, T_ji, fitness, rmse = self.verify(idx, j)
            if not ok:
                continue

            # How far the closure says the current estimate has drifted --
            # the headline number for "did loop closure actually do anything".
            implied = self.poses[j] @ T_ji
            drift = float(np.linalg.norm(implied[:3, 3] - self.poses[idx][:3, 3]))

            # Plausibility gate. Odometry drift grows roughly with distance
            # travelled, so a closure claiming a correction far larger than the
            # loop could plausibly have accumulated is a false match -- and in a
            # repetitive scene (rows of similar houses) FPFH+RANSAC produces
            # exactly that. Accepting one drags the whole graph with it, which
            # is worse than closing no loop at all.
            budget = max(cfg.LOOP_MIN_DRIFT_BUDGET,
                         cfg.LOOP_MAX_DRIFT_FRACTION * self._path_length(j, idx))
            if drift > budget:
                if self.verbose:
                    print(f"\n    rejected loop {idx} -> {j}: implies {drift:.1f} m "
                          f"correction, plausible budget {budget:.1f} m")
                continue

            info = self._information(self.clouds[idx], self.clouds[j], T_ji)
            # uncertain=True marks this as a loop edge, so Open3D's optimiser
            # applies its robust kernel and a bad closure cannot wreck the graph.
            self._edges.append((idx, j, T_ji, info, True))

            lc = LoopClosure(idx, j, fitness, rmse, drift)
            self.loops.append(lc)
            found.append(lc)
            if self.verbose:
                print(f"\n    loop closure {idx} -> {j}: fitness {fitness:.3f}, "
                      f"rmse {rmse:.3f} m, drift correction {drift:.2f} m")
        return found

    def add_loop_edge(self, source: int, target: int, T_target_source: np.ndarray,
                      information: np.ndarray | None = None,
                      fitness: float = 1.0, rmse: float = 0.0) -> LoopClosure | None:
        """Add an externally-verified loop edge, subject to the same drift gate.

        The stereo front end detects and verifies its own closures (descriptor
        matching plus PnP), because FPFH and GICP are meaningless on a sparse
        triangulated point set. It still shares this optimiser, which is what
        keeps the LiDAR-vs-stereo comparison a comparison of front ends.
        """
        implied = self.poses[target] @ T_target_source
        drift = float(np.linalg.norm(implied[:3, 3] - self.poses[source][:3, 3]))

        budget = max(cfg.LOOP_MIN_DRIFT_BUDGET,
                     cfg.LOOP_MAX_DRIFT_FRACTION * self._path_length(target, source))
        if drift > budget:
            if self.verbose:
                print(f"\n    rejected loop {source} -> {target}: implies "
                      f"{drift:.1f} m correction, budget {budget:.1f} m")
            return None

        info = np.eye(6) if information is None else information
        self._edges.append((source, target, np.asarray(T_target_source), info, True))
        lc = LoopClosure(source, target, fitness, rmse, drift)
        self.loops.append(lc)
        if self.verbose:
            print(f"\n    loop closure {source} -> {target}: "
                  f"drift correction {drift:.2f} m")
        return lc

    # -- optimisation ------------------------------------------------------

    def _build_graph(self) -> "o3d.pipelines.registration.PoseGraph":
        """Fresh PoseGraph from the current poses and the full edge list."""
        pg = _reg.PoseGraph()
        for T in self.poses:
            pg.nodes.append(_reg.PoseGraphNode(np.asarray(T, dtype=np.float64)))
        for src, tgt, trans, info, uncertain in self._edges:
            pg.edges.append(_reg.PoseGraphEdge(src, tgt, trans, info, uncertain=uncertain))
        return pg

    @property
    def graph(self):
        """A PoseGraph view of the current state (rebuilt on access)."""
        return self._build_graph()

    def optimize(self) -> None:
        """Levenberg-Marquardt over the whole graph; node 0 is held fixed."""
        if len(self.poses) < 2 or len(self._edges) < 1:
            return

        pg = self._build_graph()
        option = _reg.GlobalOptimizationOption(
            max_correspondence_distance=cfg.PG_MAX_CORRESPONDENCE,
            edge_prune_threshold=cfg.PG_EDGE_PRUNE,
            preference_loop_closure=cfg.PG_PREFERENCE_LOOP,
            reference_node=0,
        )
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
            _reg.global_optimization(
                pg,
                _reg.GlobalOptimizationLevenbergMarquardt(),
                _reg.GlobalOptimizationConvergenceCriteria(),
                option)

        optimized = [np.asarray(n.pose).copy() for n in pg.nodes]
        if any(not np.isfinite(T).all() for T in optimized):
            # A diverged solve must not be written back; keep the last good poses.
            return
        self.poses = optimized
        self._kdtree = None

    # -- accessors ---------------------------------------------------------

    def trajectory(self) -> tuple[np.ndarray, np.ndarray]:
        """``(times (N,), poses (N,4,4))`` over keyframes."""
        if not self.poses:
            return np.empty(0), np.empty((0, 4, 4))
        return np.array(self.times), np.stack(self.poses)

    def stats(self) -> dict:
        return {
            "keyframes": len(self.poses),
            "edges": len(self._edges),
            "loop_closures": len(self.loops),
            "loop_drift_corrected_m": (
                float(np.sum([l.translation_correction for l in self.loops]))
                if self.loops else 0.0),
            "max_loop_drift_m": (
                float(np.max([l.translation_correction for l in self.loops]))
                if self.loops else 0.0),
        }

    def total_rotation(self) -> float:
        """Cumulative heading change over the trajectory, in degrees (diagnostic)."""
        return float(sum(
            np.degrees(rotation_angle(invert_se3(self.poses[i])[:3, :3] @ self.poses[i + 1][:3, :3]))
            for i in range(len(self.poses) - 1)))
