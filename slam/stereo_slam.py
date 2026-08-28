"""
Stereo-inertial SLAM -- the "without the expensive sensor" arm of the study.

Pipeline per frame:

    SGBM disparity -> depth  ->  ORB features  ->  PnP + RANSAC vs. keyframe
                                                        |
                              keyframe? -> windowed BA -> loop closure -> pose graph

It shares ``backend.PoseGraphBackend`` with the LiDAR method, so the trajectory
optimiser is identical and the comparison isolates the front end. It does NOT
share loop *detection*: FPFH and GICP are meaningless on a sparse triangulated
point set, so candidates come from ORB descriptor matching and are verified by
PnP inlier count, then handed to the shared optimiser via ``add_loop_edge``.

Two honest caveats to carry into any writeup:

* AirSim's cameras are ideal pinholes with parallel optical axes and zero
  distortion, so rectification is the identity and there is no calibration
  error. Real stereo always has both.
* Depth precision falls off as ``Z^2``: with ``fx*B = 160`` a half-pixel
  disparity error is ~0.3 m at 10 m but ~1.3 m at 20 m. Features beyond
  ``STEREO_MAX_DEPTH`` are discarded rather than trusted, which is why this
  method's effective sensing horizon is a fraction of the LiDAR's 100 m.

Frames: tracking is done in the left camera's OPTICAL frame (z forward, x
right, y down) because that is what OpenCV's PnP and triangulation expect. Body
poses are recovered with ``config.T_BODY_CAM_LEFT`` on the way out.
"""
from __future__ import annotations

import time

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from . import config as cfg
from .backend import PoseGraphBackend
from .geometry import invert_se3, rotation_angle
from .imu import ImuPropagator, ImuState
from .lidar_slam import SlamResult, _Timer
from .source import Frame, SensorSource

__all__ = ["StereoInertialSLAM"]


# --------------------------------------------------------------------------
# stereo geometry
# --------------------------------------------------------------------------

def make_matcher() -> cv2.StereoSGBM:
    """SGBM tuned for the configured rig.

    P1/P2 follow OpenCV's recommended 8*C*B^2 and 32*C*B^2 smoothness terms.
    SGBM_3WAY is markedly faster than the 5-direction default at essentially
    the same quality, which matters across ~2000 frames x 5 conditions.
    """
    b = cfg.SGBM_BLOCK_SIZE
    return cv2.StereoSGBM_create(
        minDisparity=cfg.SGBM_MIN_DISPARITY,
        numDisparities=cfg.SGBM_NUM_DISPARITIES,
        blockSize=b,
        P1=8 * 3 * b * b,
        P2=32 * 3 * b * b,
        disp12MaxDiff=cfg.SGBM_DISP12_MAX_DIFF,
        uniquenessRatio=cfg.SGBM_UNIQUENESS_RATIO,
        speckleWindowSize=cfg.SGBM_SPECKLE_WINDOW,
        speckleRange=cfg.SGBM_SPECKLE_RANGE,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def disparity_to_depth(disp: np.ndarray, scale: float) -> np.ndarray:
    """Disparity at ``scale`` resolution -> metric depth.

    Z = fx_s * B / disp_s, where fx_s = fx_full * scale. Equivalent to using
    full-resolution values throughout, since disp_full = disp_s / scale.
    """
    fx_s = cfg.FX * scale
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = (fx_s * cfg.STEREO_BASELINE) / disp
    depth[~np.isfinite(depth)] = 0.0
    depth[(depth < cfg.STEREO_MIN_DEPTH) | (depth > cfg.STEREO_MAX_DEPTH)] = 0.0
    return depth


def backproject(uv: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Full-resolution pixels + depth -> 3D points in the camera optical frame."""
    x = (uv[:, 0] - cfg.CX) * z / cfg.FX
    y = (uv[:, 1] - cfg.CY) * z / cfg.FY
    return np.stack([x, y, z], axis=1)


# --------------------------------------------------------------------------

class _Keyframe:
    """Everything needed to track against, close a loop to, and bundle-adjust."""

    __slots__ = ("index", "t", "T_world_cam", "kp", "desc", "pts3d")

    def __init__(self, index, t, T_world_cam, kp, desc, pts3d):
        self.index = index
        self.t = t
        self.T_world_cam = T_world_cam
        self.kp = kp                 # (N,2) pixel coords with valid depth
        self.desc = desc             # (N,32) ORB descriptors
        self.pts3d = pts3d           # (N,3) camera-frame 3D points


class StereoInertialSLAM:
    def __init__(self, use_imu: bool = True, use_loop_closure: bool = True,
                 use_ba: bool = False, verbose: bool = True) -> None:
        """``use_ba`` defaults OFF, on measured evidence.

        Over the full 800 m route the windowed BA changed ATE from 16.175 m to
        16.201 m -- no improvement -- while adding 55% to the runtime (520 s ->
        806 s) and needing 45 of 399 solves discarded as diverged. A joint
        pose+structure BA really wants a proper sparse solver and landmarks
        tracked across keyframes rather than re-matched to the newest one; that
        is what g2o/GTSAM would provide, and the no-new-dependencies constraint
        rules them out. It is kept, working and guarded, behind this flag rather
        than deleted, because it is a real part of the design space -- but it is
        not on the default path pretending to earn its keep.
        """
        self.use_imu = use_imu
        self.use_loop_closure = use_loop_closure
        self.use_ba = use_ba
        self.verbose = verbose

        self.sgbm = make_matcher()
        self.orb = cv2.ORB_create(nfeatures=cfg.ORB_N_FEATURES,
                                  scaleFactor=cfg.ORB_SCALE_FACTOR,
                                  nlevels=cfg.ORB_N_LEVELS)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.imu = ImuPropagator()
        self.backend = PoseGraphBackend(verbose=verbose)
        self.timer = _Timer()

        self.state = ImuState()
        self.T_cam = np.eye(4)              # world <- left camera optical
        self._initialized = False
        self._init_samples: list = []

        self.keyframes: list[_Keyframe] = []
        self._ref: _Keyframe | None = None

        self._frame_times: list[float] = []
        self._frame_anchor: list[int] = []
        self._frame_rel: list[np.ndarray] = []

        self._prev_t: float | None = None
        self.n_frames = 0
        self.n_failures = 0
        self.n_ba_accepted = 0
        self.n_ba_rejected = 0
        self.inlier_log: list[int] = []

    # -- frames ------------------------------------------------------------

    @staticmethod
    def _cam_to_body(T_cam: np.ndarray) -> np.ndarray:
        return T_cam @ invert_se3(cfg.T_BODY_CAM_LEFT)

    @staticmethod
    def _body_to_cam(T_body: np.ndarray) -> np.ndarray:
        return T_body @ cfg.T_BODY_CAM_LEFT

    # -- perception --------------------------------------------------------

    def _features_with_depth(self, stereo) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ORB keypoints at full resolution, with SGBM depth attached.

        Features are detected at full resolution but depth is looked up from a
        half-resolution disparity map: full-res SGBM dominates the offline
        runtime and only the depth lookup is coarsened by it.
        """
        left, right = stereo.left, stereo.right
        s = cfg.STEREO_DEPTH_SCALE

        gl = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        if s != 1.0:
            sl = cv2.resize(gl, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            sr = cv2.resize(gr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        else:
            sl, sr = gl, gr

        disp = self.sgbm.compute(sl, sr).astype(np.float32) / 16.0
        depth = disparity_to_depth(disp, s)

        kp, desc = self.orb.detectAndCompute(gl, None)
        if desc is None or len(kp) < 10:
            return np.empty((0, 2)), np.empty((0, 32), np.uint8), np.empty((0, 3))

        uv = np.array([k.pt for k in kp], dtype=np.float32)
        du = np.clip((uv[:, 0] * s).astype(int), 0, depth.shape[1] - 1)
        dv = np.clip((uv[:, 1] * s).astype(int), 0, depth.shape[0] - 1)
        z = depth[dv, du]

        ok = z > 0
        if ok.sum() < 10:
            return np.empty((0, 2)), np.empty((0, 32), np.uint8), np.empty((0, 3))
        return uv[ok], desc[ok], backproject(uv[ok], z[ok])

    def _match(self, desc_a: np.ndarray, desc_b: np.ndarray) -> np.ndarray:
        """Lowe-ratio ORB matching. Returns (M,2) index pairs into (a, b)."""
        if len(desc_a) < 2 or len(desc_b) < 2:
            return np.empty((0, 2), dtype=int)
        raw = self.matcher.knnMatch(desc_a, desc_b, k=2)
        pairs = [(m.queryIdx, m.trainIdx) for m, n in
                 (r for r in raw if len(r) == 2)
                 if m.distance < cfg.ORB_MATCH_RATIO * n.distance]
        return np.array(pairs, dtype=int) if pairs else np.empty((0, 2), dtype=int)

    def _pnp(self, pts3d_ref: np.ndarray, uv_cur: np.ndarray):
        """PnP+RANSAC: pose of the current camera relative to the reference frame.

        Returns ``(T_cur_ref, inlier_indices)`` or ``(None, None)``.
        """
        if len(pts3d_ref) < cfg.PNP_MIN_INLIERS:
            return None, None
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d_ref.astype(np.float64), uv_cur.astype(np.float64),
            cfg.K, cfg.DIST_COEFFS,
            reprojectionError=cfg.PNP_REPROJ_ERROR,
            iterationsCount=cfg.PNP_ITERATIONS,
            confidence=cfg.PNP_CONFIDENCE,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok or inliers is None or len(inliers) < cfg.PNP_MIN_INLIERS:
            return None, None

        inl = inliers.ravel()
        # Non-linear refinement on the inlier set: RANSAC's hypothesis comes
        # from a minimal sample, so this is worth a few hundred microseconds.
        rvec, tvec = cv2.solvePnPRefineLM(
            pts3d_ref[inl].astype(np.float64), uv_cur[inl].astype(np.float64),
            cfg.K, cfg.DIST_COEFFS, rvec, tvec)

        T = np.eye(4)
        T[:3, :3] = cv2.Rodrigues(rvec)[0]
        T[:3, 3] = tvec.ravel()
        return T, inl

    # -- main step ---------------------------------------------------------

    def _initialize(self, frame: Frame) -> None:
        self._init_samples.extend(frame.imu)
        need = int(cfg.IMU_BIAS_INIT_SECONDS * cfg.IMU_RATE_HZ)
        if self.use_imu and len(self._init_samples) < need and frame.stereo is None:
            return
        R0 = self.imu.initialize(self._init_samples) if self.use_imu else np.eye(3)
        self.state = ImuState(R=R0, v=np.zeros(3), p=np.zeros(3))
        self.T_cam = self._body_to_cam(self.state.matrix())
        self._initialized = True

    def process(self, frame: Frame) -> np.ndarray:
        if not self._initialized:
            self._initialize(frame)
            if not self._initialized:
                return self.state.matrix()

        t0 = time.perf_counter()
        T_body_prev = self.state.matrix()
        if self.use_imu and frame.imu:
            T_body_pred, state_pred = self.imu.predict(self.state, frame.imu, frame.t)
        else:
            T_body_pred, state_pred = T_body_prev, self.state.copy()
        T_cam_pred = self._body_to_cam(T_body_pred)
        self.timer.add("imu", time.perf_counter() - t0)

        if frame.stereo is None:
            self.state = state_pred
            self.T_cam = T_cam_pred
            return self.state.matrix()

        # Counted here, not on the success path at the end: n_failures is
        # incremented on two early returns below, so counting only successes made
        # tracking_failure_rate a ratio of different denominators and it could
        # exceed 1. A real run reported "154.0%". n_frames now means "frames on
        # which tracking was attempted", which is the denominator the rate needs.
        self.n_frames += 1

        t0 = time.perf_counter()
        uv, desc, pts3d = self._features_with_depth(frame.stereo)
        self.timer.add("features", time.perf_counter() - t0)

        if len(uv) < cfg.PNP_MIN_INLIERS:
            self.n_failures += 1
            self.state, self.T_cam = state_pred, T_cam_pred
            self._record_frame(frame, T_cam_pred, None)
            return self.state.matrix()

        # ---- track against the reference keyframe ----
        t0 = time.perf_counter()
        T_cam_est, n_inliers = T_cam_pred, 0
        if self._ref is not None:
            pairs = self._match(desc, self._ref.desc)
            if len(pairs) >= cfg.PNP_MIN_INLIERS:
                T_cur_ref, inl = self._pnp(self._ref.pts3d[pairs[:, 1]], uv[pairs[:, 0]])
                if T_cur_ref is not None:
                    # T_cur_ref maps the reference camera frame into the current
                    # one, so the world pose composes with its inverse.
                    T_cam_est = self._ref.T_world_cam @ invert_se3(T_cur_ref)
                    n_inliers = len(inl)
        if n_inliers < cfg.PNP_MIN_INLIERS:
            self.n_failures += 1
            T_cam_est = T_cam_pred
        self.inlier_log.append(n_inliers)
        self.timer.add("track", time.perf_counter() - t0)

        # ---- update state ----
        T_body_est = self._cam_to_body(T_cam_est)
        dt = (frame.t - self._prev_t) if self._prev_t else 0.0
        v = ((T_body_est[:3, 3] - T_body_prev[:3, 3]) / dt) if dt > 1e-6 else state_pred.v
        self.state = ImuState(R=T_body_est[:3, :3].copy(), v=v, p=T_body_est[:3, 3].copy())
        self.T_cam = T_cam_est
        self._prev_t = frame.t

        new_kf = self._maybe_keyframe(frame, uv, desc, pts3d, T_cam_est, n_inliers)
        self._record_frame(frame, T_cam_est, new_kf)
        return self.state.matrix()

    def _record_frame(self, frame: Frame, T_cam: np.ndarray, new_kf: int | None) -> None:
        if new_kf is not None:
            anchor, rel = new_kf, np.eye(4)
        else:
            anchor = max(0, len(self.backend.poses) - 1)
            rel = (invert_se3(self.backend.poses[anchor]) @ T_cam
                   if self.backend.poses else np.eye(4))
        self._frame_times.append(frame.t)
        self._frame_anchor.append(anchor)
        self._frame_rel.append(rel)

    def _maybe_keyframe(self, frame, uv, desc, pts3d, T_cam, n_inliers) -> int | None:
        first = self._ref is None
        if not first:
            delta = invert_se3(self._ref.T_world_cam) @ T_cam
            moved = float(np.linalg.norm(delta[:3, 3]))
            turned = rotation_angle(delta[:3, :3])
            # A collapsing inlier count also forces a keyframe: it means the
            # view has changed enough that the reference is about to stop
            # working, which is the moment to re-anchor rather than after.
            weak = n_inliers < 2 * cfg.PNP_MIN_INLIERS
            if moved < cfg.KEYFRAME_TRANS and turned < cfg.KEYFRAME_ROT and not weak:
                return None

        t0 = time.perf_counter()
        idx = self.backend.add_keyframe(pts3d, T_cam, frame.t)
        kf = _Keyframe(idx, frame.t, T_cam.copy(), uv, desc, pts3d)
        self.keyframes.append(kf)
        self._ref = kf
        self.timer.add("keyframe", time.perf_counter() - t0)

        if self.use_ba and len(self.keyframes) >= 3:
            t0 = time.perf_counter()
            self._bundle_adjust()
            self.timer.add("ba", time.perf_counter() - t0)

        if self.use_loop_closure:
            t0 = time.perf_counter()
            closed = self._try_loops(idx)
            self.timer.add("loop_search", time.perf_counter() - t0)
            if closed:
                t0 = time.perf_counter()
                self.backend.optimize()
                self.timer.add("optimize", time.perf_counter() - t0)
                self._sync_from_backend()
        return idx

    def _sync_from_backend(self) -> None:
        """Adopt optimised poses into the front end after loop closure."""
        for kf, T in zip(self.keyframes, self.backend.poses):
            kf.T_world_cam = np.asarray(T).copy()
        if self.keyframes:
            self._ref = self.keyframes[-1]
            self.T_cam = self._ref.T_world_cam.copy()
            T_body = self._cam_to_body(self.T_cam)
            self.state = ImuState(R=T_body[:3, :3].copy(), v=self.state.v,
                                  p=T_body[:3, 3].copy())

    # -- loop closure ------------------------------------------------------

    def _try_loops(self, idx: int) -> bool:
        """Descriptor-matched, PnP-verified closures fed to the shared optimiser."""
        if idx < cfg.LOOP_MIN_INDEX_GAP or len(self.keyframes) <= cfg.LOOP_MIN_INDEX_GAP:
            return False

        eligible = idx - cfg.LOOP_MIN_INDEX_GAP
        pos = np.array([kf.T_world_cam[:3, 3] for kf in self.keyframes[:eligible]])
        if len(pos) == 0:
            return False

        cur = self.keyframes[idx]
        near = cKDTree(pos).query_ball_point(cur.T_world_cam[:3, 3], cfg.LOOP_SEARCH_RADIUS)
        if not near:
            return False
        near.sort(key=lambda j: float(np.linalg.norm(pos[j] - cur.T_world_cam[:3, 3])))

        closed = False
        for j in near[:cfg.LOOP_MAX_CANDIDATES]:
            cand = self.keyframes[j]
            pairs = self._match(cur.desc, cand.desc)
            if len(pairs) < cfg.STEREO_LOOP_MIN_MATCHES:
                continue
            T_cur_cand, inl = self._pnp(cand.pts3d[pairs[:, 1]], cur.kp[pairs[:, 0]])
            if T_cur_cand is None or len(inl) < cfg.STEREO_LOOP_MIN_INLIERS:
                continue

            # Edge convention: transformation maps SOURCE (cur) into TARGET
            # (cand), which is exactly T_cand_cur = inv(T_cur_cand).
            T_cand_cur = invert_se3(T_cur_cand)
            info = np.eye(6) * float(len(inl))
            lc = self.backend.add_loop_edge(idx, j, T_cand_cur, info,
                                            fitness=len(inl) / max(len(pairs), 1))
            closed = closed or (lc is not None)
        return closed

    # -- windowed bundle adjustment ---------------------------------------

    def _bundle_adjust(self) -> None:
        """Joint pose+structure BA over the last ``BA_WINDOW`` keyframes.

        Landmarks are the reference keyframe's 3D points, matched forward into
        the other window keyframes; only those seen at least twice are
        optimised. The oldest window pose is held fixed to remove the gauge
        freedom, and the Jacobian sparsity pattern is supplied explicitly so
        scipy's trust-region solver uses LSMR rather than forming a dense
        Jacobian (which at ~1500 parameters would dominate the runtime).
        """
        window = self.keyframes[-cfg.BA_WINDOW:]
        if len(window) < 3:
            return

        anchor = window[-1]
        obs_kf, obs_lm, obs_uv = [], [], []
        for wi, kf in enumerate(window):
            if kf is anchor:
                pairs = np.stack([np.arange(len(anchor.pts3d))] * 2, axis=1)
            else:
                pairs = self._match(kf.desc, anchor.desc)
            if len(pairs) == 0:
                continue
            obs_kf.append(np.full(len(pairs), wi))
            obs_lm.append(pairs[:, 1])
            obs_uv.append(kf.kp[pairs[:, 0]])

        if not obs_kf:
            return
        obs_kf = np.concatenate(obs_kf)
        obs_lm = np.concatenate(obs_lm)
        obs_uv = np.concatenate(obs_uv)

        # Keep only landmarks with multi-view support -- a single observation
        # constrains nothing and just inflates the problem.
        counts = np.bincount(obs_lm, minlength=len(anchor.pts3d))
        keep_lm = np.nonzero(counts >= 2)[0]
        if len(keep_lm) < 20:
            return
        if len(keep_lm) > cfg.BA_MAX_LANDMARKS:
            keep_lm = keep_lm[np.argsort(-counts[keep_lm])[:cfg.BA_MAX_LANDMARKS]]

        remap = -np.ones(len(anchor.pts3d), dtype=int)
        remap[keep_lm] = np.arange(len(keep_lm))
        sel = remap[obs_lm] >= 0
        obs_kf, obs_lm, obs_uv = obs_kf[sel], remap[obs_lm[sel]], obs_uv[sel]
        if len(obs_kf) < 50:
            return

        n_pose = len(window) - 1        # window[0] held fixed
        n_lm = len(keep_lm)

        # Landmarks in world coordinates, from the anchor keyframe.
        lm_world = (anchor.pts3d[keep_lm] @ anchor.T_world_cam[:3, :3].T
                    + anchor.T_world_cam[:3, 3])

        # Parameterise the world->camera transforms (what projection needs).
        T_cw = [invert_se3(kf.T_world_cam) for kf in window]
        x0 = np.concatenate([
            np.concatenate([np.concatenate([
                Rotation.from_matrix(T_cw[i][:3, :3]).as_rotvec(), T_cw[i][:3, 3]])
                for i in range(1, len(window))]) if n_pose else np.empty(0),
            lm_world.ravel(),
        ])
        fixed = T_cw[0]

        def residuals(x):
            poses = [fixed]
            for i in range(n_pose):
                p = x[6 * i:6 * i + 6]
                M = np.eye(4)
                M[:3, :3] = Rotation.from_rotvec(p[:3]).as_matrix()
                M[:3, 3] = p[3:]
                poses.append(M)
            lm = x[6 * n_pose:].reshape(-1, 3)

            P = np.stack([poses[k] for k in obs_kf])
            X = lm[obs_lm]
            cam = np.einsum("nij,nj->ni", P[:, :3, :3], X) + P[:, :3, 3]
            z = np.maximum(cam[:, 2], 1e-3)
            u = cfg.FX * cam[:, 0] / z + cfg.CX
            v = cfg.FY * cam[:, 1] / z + cfg.CY
            r = np.stack([u - obs_uv[:, 0], v - obs_uv[:, 1]], axis=1)
            # Points that fell behind the camera are meaningless; neutralise
            # them rather than letting a huge residual steer the solve.
            r[cam[:, 2] <= 1e-3] = 0.0
            return r.ravel()

        # Sparsity: each observation touches only its own pose block and its
        # own landmark block.
        from scipy.sparse import lil_matrix
        m = 2 * len(obs_kf)
        S = lil_matrix((m, len(x0)), dtype=int)
        rows = np.arange(len(obs_kf))
        for d in range(2):
            for c in range(6):
                mask = obs_kf > 0
                S[2 * rows[mask] + d, 6 * (obs_kf[mask] - 1) + c] = 1
            for c in range(3):
                S[2 * rows + d, 6 * n_pose + 3 * obs_lm + c] = 1

        try:
            res = least_squares(
                residuals, x0, jac_sparsity=S, method="trf",
                loss="huber", f_scale=cfg.BA_HUBER_DELTA,
                max_nfev=cfg.BA_MAX_ITER, xtol=1e-6, ftol=1e-6, verbose=0)
        except Exception:
            return

        if not np.isfinite(res.x).all():
            return

        # Convert back to camera->world, but do NOT trust the result blindly.
        refined = []
        for i in range(n_pose):
            p = res.x[6 * i:6 * i + 6]
            M = np.eye(4)
            M[:3, :3] = Rotation.from_rotvec(p[:3]).as_matrix()
            M[:3, 3] = p[3:]
            refined.append(invert_se3(M))

        # BA is a local refinement of an already-tracked window: corrections
        # are centimetres to tens of centimetres. A metre-scale jump means the
        # solve went somewhere else entirely -- usually a poorly-conditioned
        # window where the landmarks (anchored to the newest keyframe, whose
        # pose is itself being optimised) do not pin the gauge down. Writing
        # that back corrupts the trajectory and, worse, moves keyframes far
        # enough that loop-closure candidate search stops finding real
        # revisits. Discarding the solve costs nothing; accepting a bad one
        # loses the run.
        jump = max(float(np.linalg.norm(refined[i][:3, 3] - window[i + 1].T_world_cam[:3, 3]))
                   for i in range(n_pose))
        if jump > cfg.BA_MAX_CORRECTION:
            self.n_ba_rejected += 1
            return

        for i in range(n_pose):
            kf = window[i + 1]
            kf.T_world_cam = refined[i]
            self.backend.poses[kf.index] = refined[i].copy()
        self.n_ba_accepted += 1

        self.T_cam = self.keyframes[-1].T_world_cam.copy()
        T_body = self._cam_to_body(self.T_cam)
        self.state = ImuState(R=T_body[:3, :3].copy(), v=self.state.v,
                              p=T_body[:3, 3].copy())

    # -- finish ------------------------------------------------------------

    def finalize(self) -> SlamResult:
        kf_poses = self.backend.poses
        if kf_poses and self._frame_times:
            cam_poses = np.stack([kf_poses[a] @ rel
                                  for a, rel in zip(self._frame_anchor, self._frame_rel)])
            body_poses = np.stack([self._cam_to_body(T) for T in cam_poses])
        else:
            body_poses = np.empty((0, 4, 4))

        stats = {
            "frames": self.n_frames,
            "tracking_failures": self.n_failures,
            "tracking_failure_rate": self.n_failures / max(1, self.n_frames),
            "mean_inliers": float(np.mean(self.inlier_log)) if self.inlier_log else 0.0,
            "median_inliers": float(np.median(self.inlier_log)) if self.inlier_log else 0.0,
            "ba_accepted": self.n_ba_accepted,
            "ba_rejected": self.n_ba_rejected,
            **self.backend.stats(),
        }

        return SlamResult(
            method="stereo_inertial",
            times=np.array(self._frame_times),
            poses=body_poses,
            keyframe_times=np.array(self.backend.times),
            keyframe_poses=(np.stack([self._cam_to_body(T) for T in kf_poses])
                            if kf_poses else np.empty((0, 4, 4))),
            map_points=self._sparse_map(),
            stats=stats,
            timings=self.timer.report(),
        )

    def _sparse_map(self) -> np.ndarray:
        """Triangulated landmarks in world coordinates.

        Far sparser than the LiDAR map by construction -- a legitimate result
        to report, not a defect to hide.
        """
        chunks = [kf.pts3d @ T[:3, :3].T + T[:3, 3]
                  for kf, T in zip(self.keyframes, self.backend.poses)
                  if len(kf.pts3d)]
        return np.vstack(chunks) if chunks else np.empty((0, 3))

    def run(self, source: SensorSource, progress_every: int = 50) -> SlamResult:
        gt_t, gt_T = [], []
        try:
            total = len(source)
        except TypeError:
            total = None

        wall0 = time.perf_counter()
        for i, frame in enumerate(source):
            self.process(frame)
            if frame.gt_pose is not None:
                gt_t.append(frame.t)
                gt_T.append(frame.gt_pose)
            if self.verbose and progress_every and i % progress_every == 0:
                pct = f"{100 * i / total:5.1f}%" if total else f"{i:5d}"
                print(f"\r  [stereo] {pct}  kf {len(self.keyframes):4d}  "
                      f"loops {len(self.backend.loops):3d}  "
                      f"fail {self.n_failures:4d}", end="", flush=True)

        result = self.finalize()
        result.gt_times = np.array(gt_t)
        result.gt_poses = np.stack(gt_T) if gt_T else np.empty((0, 4, 4))
        result.stats["wall_seconds"] = time.perf_counter() - wall0
        if self.verbose:
            print(f"\r  {result.summary()}")
        return result
