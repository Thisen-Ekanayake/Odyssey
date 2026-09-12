"""
From the panoptic map to a car route: a 2-D road grid, A*, and a footprint
check -- pure numpy/scipy, no simulator.

The map's classes do the work here. "road" voxels define where the car may
be; "sidewalk"/"terrain"/"water" are surface (never obstacles); everything
else (buildings, trees, parked cars, poles ...) and any unlabeled voxel that
sits above the local ground blocks the cells it stands on. Drivable cells are
road cells eroded by half the car width + clearance and cleared of dilated
obstacles, so a path over drivable cells keeps the whole car on the road and
away from everything.
"""
from __future__ import annotations

import heapq
import math

import numpy as np
from scipy import ndimage

import config as cfg


def _disk(r_cells: float) -> np.ndarray:
    r = int(math.ceil(r_cells))
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r_cells * r_cells


class RoadGrid:
    def __init__(self, m: dict, cell: float | None = None,
                 half_width: float = cfg.CAR_WIDTH / 2, clearance: float = cfg.CAR_CLEARANCE):
        xyz, cls = m["xyz"], m["class_id"]
        classes = [str(c) for c in m["classes"]]
        self.cell = float(cell or m["voxel"])
        # cell boundaries aligned to voxel boundaries, so a voxel centre never
        # sits on a cell edge and lands in a cell by rounding luck
        self.x0 = float(xyz[:, 0].min()) - self.cell / 2
        self.y0 = float(xyz[:, 1].min()) - self.cell / 2
        gi = ((xyz[:, 0] - self.x0) / self.cell).astype(int)
        gj = ((xyz[:, 1] - self.y0) / self.cell).astype(int)
        self.shape = (gi.max() + 1, gj.max() + 1)
        flat = gi * self.shape[1] + gj
        n = self.shape[0] * self.shape[1]
        z = xyz[:, 2].astype(np.float64)

        ground_cls = np.isin(cls, [classes.index(c) for c in cfg.GROUND])
        road_cls = np.isin(cls, [classes.index(c) for c in cfg.DRIVABLE])
        unl = cls == classes.index(cfg.UNLABELED)

        # surface height per cell = the LOWEST surface voxel there (NED: max z)
        ground_z = np.full(n, -np.inf)
        np.maximum.at(ground_z, flat[ground_cls], z[ground_cls])
        road_z = np.full(n, -np.inf)
        np.maximum.at(road_z, flat[road_cls], z[road_cls])
        self.road = (road_z > -np.inf).reshape(self.shape)
        known = ground_z > -np.inf
        # fill unknown cells with the nearest known ground so heights are comparable
        _, (ii, jj) = ndimage.distance_transform_edt(~known.reshape(self.shape), return_indices=True)
        ground_filled = ground_z.reshape(self.shape)[ii, jj]
        self.road_z = np.where(self.road, road_z.reshape(self.shape), np.nan)
        self.ground_z = ground_filled

        height = ground_filled.reshape(-1)[flat] - z          # metres above local surface
        blocking = (~ground_cls) & (~unl) & (height > cfg.GROUND_SLACK) & (height < cfg.OBSTACLE_BAND)
        if cfg.UNLABELED_BLOCKS:
            blocking |= unl & known[flat] & (height > cfg.GROUND_SLACK) & (height < cfg.OBSTACLE_BAND)
        obst = np.zeros(n, dtype=bool)
        obst[flat[blocking]] = True
        self.obstacle = obst.reshape(self.shape)

        # +0.5 cell: cells are checked at their centre. The road edge is eroded
        # by the half width + clearance; obstacles are grown by that plus
        # OBSTACLE_EXTRA, because in a turn the rectangle's front corner
        # sweeps wider than the path centre (a single pole cell fails the run).
        # The full half-diagonal (2.5 m) would be safe but leaves almost no
        # drivable road between AirSimNH's hedge rows -- see config.
        r_road = (half_width + clearance) / self.cell + 0.5
        r_obst = (half_width + clearance + cfg.OBSTACLE_EXTRA) / self.cell + 0.5
        self.margin_cells = r_road
        self.drivable = (ndimage.binary_erosion(self.road, _disk(r_road), border_value=0)
                         & ~ndimage.binary_dilation(self.obstacle, _disk(r_obst)))
        # keep only the largest connected drivable region
        lab, k = ndimage.label(self.drivable, structure=np.ones((3, 3)))
        if k > 1:
            sizes = ndimage.sum(self.drivable, lab, range(1, k + 1))
            self.drivable = lab == (int(np.argmax(sizes)) + 1)
        # where the car may START or STOP: its full length must fit, not just
        # its width (the footprint check tests the whole rectangle)
        self.endpoint_ok = ndimage.binary_erosion(
            self.drivable, _disk(cfg.CAR_LENGTH / 2 / self.cell + 0.5), border_value=0)
        if not self.endpoint_ok.any():
            self.endpoint_ok = self.drivable
        # steer toward the middle of the lane: cost grows near the drivable edge
        self.edge_dist = ndimage.distance_transform_edt(self.drivable)
        self.cost = 1.0 + 3.0 / np.maximum(self.edge_dist, 1.0)

    # -- coordinates -----------------------------------------------------
    def to_cell(self, x: float, y: float) -> tuple[int, int]:
        return int((x - self.x0) / self.cell), int((y - self.y0) / self.cell)

    def to_xy(self, i: int, j: int) -> tuple[float, float]:
        return self.x0 + (i + 0.5) * self.cell, self.y0 + (j + 0.5) * self.cell

    def inside(self, i: int, j: int) -> bool:
        return 0 <= i < self.shape[0] and 0 <= j < self.shape[1]

    def z_at(self, x: float, y: float) -> float:
        """Road surface (NED z) under a point: nearest road cell's lowest voxel."""
        i, j = self.to_cell(x, y)
        i, j = min(max(i, 0), self.shape[0] - 1), min(max(j, 0), self.shape[1] - 1)
        if not np.isnan(self.road_z[i, j]):
            return float(self.road_z[i, j])
        if not hasattr(self, "_road_idx"):
            _, self._road_idx = ndimage.distance_transform_edt(~self.road, return_indices=True)
        ii, jj = self._road_idx[0][i, j], self._road_idx[1][i, j]
        return float(self.road_z[ii, jj])

    def nearest_drivable(self, x: float, y: float) -> tuple[int, int]:
        i, j = self.to_cell(x, y)
        cells = np.argwhere(self.endpoint_ok)
        d = (cells[:, 0] - i) ** 2 + (cells[:, 1] - j) ** 2
        return tuple(int(v) for v in cells[int(np.argmin(d))])

    def farthest_pair(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """Two drivable cells far apart (double BFS sweep), for an auto route."""
        a = tuple(int(v) for v in np.argwhere(self.endpoint_ok)[0])
        b = self._bfs_farthest(a)
        c = self._bfs_farthest(b)
        return b, c

    def _bfs_farthest(self, s):
        dist = np.full(self.shape, -1, dtype=np.int32)
        dist[s] = 0
        frontier = [s]
        last = s
        while frontier:
            nxt = []
            for i, j in frontier:
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        a, b = i + di, j + dj
                        if self.inside(a, b) and self.drivable[a, b] and dist[a, b] < 0:
                            dist[a, b] = dist[i, j] + 1
                            nxt.append((a, b))
                            if self.endpoint_ok[a, b]:
                                last = (a, b)
            frontier = nxt
        return last

    # -- planning --------------------------------------------------------
    def astar(self, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
        if not (self.drivable[start] and self.drivable[goal]):
            raise ValueError("start/goal not on a drivable cell")
        h = lambda c: math.hypot(c[0] - goal[0], c[1] - goal[1])
        g = {start: 0.0}
        came = {}
        pq = [(h(start), start)]
        closed = set()
        while pq:
            _, cur = heapq.heappop(pq)
            if cur == goal:
                break
            if cur in closed:
                continue
            closed.add(cur)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    nb = (cur[0] + di, cur[1] + dj)
                    if not self.inside(*nb) or not self.drivable[nb]:
                        continue
                    if di and dj and not (self.drivable[cur[0] + di, cur[1]] and
                                          self.drivable[cur[0], cur[1] + dj]):
                        continue                    # no corner cutting
                    step = math.hypot(di, dj) * self.cost[nb]
                    ng = g[cur] + step
                    if ng < g.get(nb, np.inf):
                        g[nb] = ng
                        came[nb] = cur
                        heapq.heappush(pq, (ng + h(nb), nb))
        if goal not in came and goal != start:
            raise RuntimeError("no drivable route between start and goal")
        path = [goal]
        while path[-1] != start:
            path.append(came[path[-1]])
        return path[::-1]

    def line_free(self, a: tuple[int, int], b: tuple[int, int]) -> bool:
        n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) * 2 + 1
        for t in np.linspace(0, 1, n):
            i, j = int(round(a[0] + (b[0] - a[0]) * t)), int(round(a[1] + (b[1] - a[1]) * t))
            if not self.drivable[i, j]:
                return False
        return True

    def shortcut(self, cells: list[tuple[int, int]], max_span: float = 30.0) -> list[tuple[int, int]]:
        """Greedy string-pulling, capped so the route still follows the lane centre."""
        out = [cells[0]]
        i = 0
        cap = max_span / self.cell
        while i < len(cells) - 1:
            j = len(cells) - 1
            while j > i + 1 and (math.hypot(cells[j][0] - cells[i][0], cells[j][1] - cells[i][1]) > cap
                                 or not self.line_free(cells[i], cells[j])):
                j -= 1
            out.append(cells[j])
            i = j
        return out

    def route(self, start_xy, goal_xy, spacing: float = 0.5) -> np.ndarray:
        """(K,2) world xy polyline from start to goal, resampled every `spacing` m."""
        s = self.nearest_drivable(*start_xy)
        g = self.nearest_drivable(*goal_xy)
        cells = self.shortcut(self.astar(s, g))
        pts = np.array([self.to_xy(*c) for c in cells], dtype=np.float64)
        return resample(pts, spacing)

    # -- checking --------------------------------------------------------
    def footprint_cells(self, x, y, yaw, length=cfg.CAR_LENGTH, width=cfg.CAR_WIDTH):
        c, s = math.cos(yaw), math.sin(yaw)
        ls = np.linspace(-length / 2, length / 2, int(length / self.cell) + 2)
        ws = np.linspace(-width / 2, width / 2, int(width / self.cell) + 2)
        L, W = np.meshgrid(ls, ws, indexing="ij")
        px = x + L * c - W * s
        py = y + L * s + W * c
        return ((px - self.x0) / self.cell).astype(int).ravel(), ((py - self.y0) / self.cell).astype(int).ravel()

    def check_pose(self, x, y, yaw) -> tuple[bool, bool]:
        """(on_road, collision_free) for the car's full rectangle at this pose."""
        i, j = self.footprint_cells(x, y, yaw)
        ok = (i >= 0) & (i < self.shape[0]) & (j >= 0) & (j < self.shape[1])
        if not ok.all():
            return False, False
        return bool(self.road[i, j].all()), not bool(self.obstacle[i, j].any())


def resample(pts: np.ndarray, spacing: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] < spacing:
        return pts
    t = np.arange(0.0, cum[-1], spacing)
    out = np.stack([np.interp(t, cum, pts[:, 0]), np.interp(t, cum, pts[:, 1])], axis=1)
    return np.vstack([out, pts[-1:]])


class PurePursuit:
    """Kinematic bicycle following a polyline; returns successive (x, y, yaw)."""

    def __init__(self, path: np.ndarray, speed=cfg.CAR_SPEED, lookahead=cfg.CAR_LOOKAHEAD,
                 wheelbase=cfg.CAR_WHEELBASE, dt=cfg.CAR_DT, max_steer=math.radians(cfg.CAR_MAX_STEER_DEG)):
        self.path, self.v, self.ld, self.L, self.dt, self.max_steer = path, speed, lookahead, wheelbase, dt, max_steer
        self.x, self.y = path[0]
        d0 = path[min(3, len(path) - 1)] - path[0]
        self.yaw = math.atan2(d0[1], d0[0])
        self.i = 0
        self.done = False

    def step(self):
        p = self.path
        # nearest index, monotone so the car can't lock onto a later pass of the route
        window = p[self.i:self.i + 40]
        self.i += int(np.argmin(np.linalg.norm(window - (self.x, self.y), axis=1)))
        j = self.i
        while j < len(p) - 1 and math.hypot(p[j][0] - self.x, p[j][1] - self.y) < self.ld:
            j += 1
        tx, ty = p[j]
        alpha = math.atan2(ty - self.y, tx - self.x) - self.yaw
        alpha = (alpha + math.pi) % (2 * math.pi) - math.pi
        ld = max(math.hypot(tx - self.x, ty - self.y), 1e-3)
        steer = max(-self.max_steer, min(self.max_steer, math.atan2(2 * self.L * math.sin(alpha), ld)))
        v = self.v * (0.4 if abs(steer) > 0.5 * self.max_steer else 1.0)   # slow into tight turns
        self.yaw += v / self.L * math.tan(steer) * self.dt
        self.x += v * math.cos(self.yaw) * self.dt
        self.y += v * math.sin(self.yaw) * self.dt
        if math.hypot(p[-1][0] - self.x, p[-1][1] - self.y) < 1.0:
            self.done = True
        return self.x, self.y, self.yaw
