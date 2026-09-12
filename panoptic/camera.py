"""
Pinhole maths for the nadir camera, in AirSim's camera BODY convention:
+x forward (optical axis), +y right, +z down -- the frame ``ImageResponse``'s
``camera_position`` / ``camera_orientation`` describe (verified against the
body-derived pose in ``slam/recorder.py``). No optical-frame swap is needed
here because we never hand anything to OpenCV.

DepthPlanar is the +x (optical-axis) distance, so a pixel back-projects as
    x = depth,  y = (u - cx) / fx * depth,  z = (v - cy) / fy * depth.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fov_deg: float

    @property
    def fx(self) -> float:
        return (self.width / 2.0) / math.tan(math.radians(self.fov_deg) / 2.0)

    @property
    def fy(self) -> float:
        return self.fx            # square pixels, no distortion in UE4's capture

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0

    def project(self, p_cam: np.ndarray):
        """(N,3) camera-body points -> (u, v, depth). Points behind the camera
        get depth <= 0; the caller masks them."""
        x = p_cam[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.cx + self.fx * p_cam[:, 1] / x
            v = self.cy + self.fy * p_cam[:, 2] / x
        return u, v, x

    def backproject(self, depth: np.ndarray, stride: int = 1, max_depth: float = np.inf):
        """Dense depth (H,W) -> (M,3) camera-body points + the (v,u) they came from."""
        vs = np.arange(0, self.height, stride)
        us = np.arange(0, self.width, stride)
        vv, uu = np.meshgrid(vs, us, indexing="ij")
        d = depth[vv, uu]
        ok = np.isfinite(d) & (d > 0.05) & (d < max_depth)
        d, vv, uu = d[ok], vv[ok], uu[ok]
        pts = np.stack([d, (uu - self.cx) / self.fx * d, (vv - self.cy) / self.fy * d], axis=1)
        return pts.astype(np.float32), vv, uu

    def swath_at(self, altitude: float) -> float:
        """Ground width seen by the horizontal FOV when looking straight down."""
        return 2.0 * altitude * math.tan(math.radians(self.fov_deg) / 2.0)


def to_world(p_local: np.ndarray, T: np.ndarray) -> np.ndarray:
    return p_local.astype(np.float64) @ T[:3, :3].T + T[:3, 3]


def to_local(p_world: np.ndarray, T: np.ndarray) -> np.ndarray:
    return (p_world.astype(np.float64) - T[:3, 3]) @ T[:3, :3]
