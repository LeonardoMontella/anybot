"""Navigation maps: an occupancy grid of the world (a 2D collision map) and a path planner on it.

The grid is built from the scene itself: every static collision geom (walls, tables, shelves,
conveyors, a scanned room's height field) that stands between the floor and robot height becomes
occupied cells. The planner finds the shortest path that keeps the robot's radius away from them
(Dijkstra on the grid, preferring corridor centres, then shortcut into a few straight legs), so
`navigate_to` can go through doors and around furniture in any world.

Robots and free objects are not in the static map: they move. A lidar (anybot.sensors) marks
what it sees on top of it.
"""

from __future__ import annotations

import mujoco
import numpy as np

CELL = 0.05
_SKIP = (mujoco.mjtGeom.mjGEOM_PLANE,)


def _clearance(free: np.ndarray, max_cells: int) -> np.ndarray:
    """Cells of free space between each cell and the nearest obstacle (octagonal distance, capped).

    Erosion alternates 4- and 8-neighbour steps: an octagon, within ~8% of the true (Euclidean) distance.
    Plain 4-neighbour erosion measures Manhattan distance, which overstates diagonal room by up to 41%:
    a table corner 0.49 m away would look 0.7 m away, and a long robot would clip it."""
    dist = np.zeros(free.shape, int)
    cur = free.copy()
    for k in range(1, max_cells + 1):
        dist[cur] = k
        nxt = cur.copy()
        nxt[1:, :] &= cur[:-1, :]
        nxt[:-1, :] &= cur[1:, :]
        nxt[:, 1:] &= cur[:, :-1]
        nxt[:, :-1] &= cur[:, 1:]
        if k % 2 == 0:  # diagonal neighbours too, every other step
            nxt[1:, 1:] &= cur[:-1, :-1]
            nxt[:-1, :-1] &= cur[1:, 1:]
            nxt[1:, :-1] &= cur[:-1, 1:]
            nxt[:-1, 1:] &= cur[1:, :-1]
        nxt[0, :] = nxt[-1, :] = nxt[:, 0] = nxt[:, -1] = False
        cur = nxt
        if not cur.any():
            break
    return dist


def _reachable(ok: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    """Cells connected to `start` through cells where `ok` holds (8-connected flood fill)."""
    seen = np.zeros_like(ok)
    if not ok[start]:
        return seen
    seen[start] = True
    frontier = seen.copy()
    while frontier.any():
        grow = np.zeros_like(ok)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                grow |= np.roll(np.roll(frontier, dr, 0), dc, 1)
        frontier = grow & ok & ~seen
        seen |= frontier
    return seen


class NoPath(Exception):
    """The goal cannot be reached from where the robot is."""


class Planner:
    """Shortest collision-free floor path on an occupancy grid (grid search + shortcutting).

    `static` is the map of the world (walls, tables, shelves); obstacles seen by a sensor can be added
    on top (`mark`) and fade away when they are not seen again (`clear_dynamic`)."""

    def __init__(self, occupied: np.ndarray, lo, cell: float):
        self.cell, self.lo = float(cell), np.asarray(lo, float)
        self.static = occupied.astype(bool)
        self.dynamic = np.zeros_like(self.static)
        self.last_route: list[list[float]] = []
        self._update()

    def _update(self) -> None:
        self.occupied = self.static | self.dynamic
        self.clear = _clearance(~self.occupied, 30)

    def mark(self, points_xy, robot_xy=None, keep_out: float = 0.0) -> None:
        """Obstacles seen by a sensor (e.g. lidar hits), except within `keep_out` of the robot itself."""
        dyn = np.zeros_like(self.static)
        for p in np.asarray(points_xy, float).reshape(-1, 2):
            if robot_xy is not None and np.hypot(*(p - np.asarray(robot_xy[:2]))) < keep_out:
                continue
            r, c = self._cell(p)
            dyn[r, c] = True
        self.dynamic = dyn & ~self.static
        self._update()

    def clear_dynamic(self) -> None:
        if self.dynamic.any():
            self.dynamic[:] = False
            self._update()

    def blocked(self, xy, radius: float) -> bool:
        return bool(self.clear[self._cell(xy)] * self.cell < radius * 0.75)

    def _cell(self, xy) -> tuple[int, int]:
        c = np.round((np.asarray(xy, float) - self.lo) / self.cell).astype(int)
        return int(np.clip(c[1], 0, self.clear.shape[0] - 1)), int(np.clip(c[0], 0, self.clear.shape[1] - 1))

    def _xy(self, rc) -> list[float]:
        return [float(self.lo[0] + rc[1] * self.cell), float(self.lo[1] + rc[0] * self.cell)]

    def _line_ok(self, a, b, need: int) -> bool:
        n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) + 1
        for t in np.linspace(0, 1, n + 1):
            r, c = int(round(a[0] + t * (b[0] - a[0]))), int(round(a[1] + t * (b[1] - a[1])))
            if self.clear[r, c] < need:
                return False
        return True

    def _search(self, s, g, need: int):
        import heapq

        ok = self.clear >= need
        ok[s] = ok[g] = True
        dist, prev, heap = {s: 0.0}, {}, [(0.0, s)]
        steps = [(dr, dc, float(np.hypot(dr, dc))) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
        H, W = ok.shape
        while heap:
            dcur, cur = heapq.heappop(heap)
            if cur == g:
                return prev
            if dcur > dist[cur]:
                continue
            for dr, dc, w in steps:
                nb = (cur[0] + dr, cur[1] + dc)
                if 0 <= nb[0] < H and 0 <= nb[1] < W and ok[nb]:
                    # prefer the middle of corridors: cells near obstacles cost a bit more
                    nd = dcur + w * (1.0 + 2.0 / max(self.clear[nb], 1))
                    if nd < dist.get(nb, np.inf):
                        dist[nb], prev[nb] = nd, cur
                        heapq.heappush(heap, (nd, nb))
        return None

    def route(self, start_xy, goal_xy, radius: float = 0.35, half_width: float | None = None) -> list[list[float]]:
        """Waypoints from start to goal (goal included), or [goal] if the straight line is clear or no
        path exists (the caller then just tries the straight line)."""
        s, g = self._cell(start_xy), self._cell(goal_xy)
        # Starting / ending close to furniture is fine: the margin applies to the way in between.
        need = int(min(np.ceil(radius / self.cell) + 1, max(self.clear[s], 2), max(self.clear[g], 2)))
        need = max(need, int(np.ceil(radius / self.cell)) - 2, 2)
        # Straight line if it is clear (start/end may be tight, but it never crosses an obstacle).
        if self._line_ok(s, g, max(1, min(need, self.clear[s], self.clear[g]))):
            self.last_route = [list(map(float, goal_xy))]
            return self.last_route
        # A narrow door: accept down to 60% of the radius (it is measured on the robot's diagonal,
        # its half-width is smaller) rather than no path.
        floor = int(np.ceil(0.6 * radius / self.cell))
        if half_width is not None:  # a long, narrow robot (Spot) goes through a door lengthwise
            floor = min(floor, int(np.ceil((half_width + 0.06) / self.cell)))
        floor = max(floor, 2)
        for margin in range(need, floor - 1, -1):  # a narrow door: accept a smaller margin rather than no path
            prev = self._search(s, g, margin)
            if prev is not None:
                need = margin
                break
        else:
            # The goal itself is unreachable (inside furniture, in a closed room): like Nav2's goal tolerance,
            # go to the nearest free spot that can be reached, if there is one close enough.
            reach = _reachable(self.clear >= floor, s)
            cand = np.argwhere(reach)
            if not len(cand):
                raise NoPath("the robot has no room to move from where it is")
            d = np.linalg.norm(cand - np.array(g), axis=1)
            if d.min() * self.cell > 1.5:
                raise NoPath("no way gets there (walls or furniture all around it)")
            g2 = tuple(int(v) for v in cand[int(np.argmin(d))])
            prev = self._search(s, g2, floor)
            if prev is None:
                raise NoPath("no way gets there")
            need, g, goal_xy = floor, g2, self._xy(g2)
        path = [g]
        while path[-1] != s:
            path.append(prev[path[-1]])
        path.reverse()
        # Shortcut: keep a waypoint only where the straight line from the last one is blocked.
        out, i = [], 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1 and not self._line_ok(path[i], path[j], need):
                j -= 1
            out.append(path[j])
            i = j
        pts = [self._xy(p) for p in out[:-1]]
        self.last_route = pts + [list(map(float, goal_xy))]
        return self.last_route


def _geom_box(m: mujoco.MjModel, d: mujoco.MjData, g: int) -> tuple[np.ndarray, np.ndarray]:
    """World-aligned bounding box of one geom (from its local AABB)."""
    c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
    R = d.geom_xmat[g].reshape(3, 3)
    centre = d.geom_xpos[g] + R @ c
    half = np.abs(R) @ h
    return centre - half, centre + half


def from_scene(m: mujoco.MjModel, d: mujoco.MjData, cell: float = CELL, z_min: float = 0.06, z_max: float = 1.8) -> Planner | None:
    """Occupancy grid of the static world (None if there is nothing to avoid)."""
    boxes, fields, extents = [], [], []
    for g in range(m.ngeom):
        if m.geom_type[g] in _SKIP or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        if m.body_weldid[m.geom_bodyid[g]] != 0:  # moves (robot links, free objects)
            continue
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_HFIELD:
            fields.append(g)
            continue
        lo, hi = _geom_box(m, d, g)
        if hi[2] < z_min or lo[2] > z_max or np.any(hi[:2] - lo[:2] > 40):
            continue
        boxes.append((lo, hi))
    pts = []
    for g in fields:  # sample the height field (assumed axis aligned)
        h = m.geom_dataid[g]
        rx, ry, top, _ = m.hfield_size[h]
        nr, nc = m.hfield_nrow[h], m.hfield_ncol[h]
        adr = m.hfield_adr[h]
        z = m.hfield_data[adr:adr + nr * nc].reshape(nr, nc) * top
        rr, cc = np.nonzero(z > z_min)
        pos = d.geom_xpos[g]
        pts.append(np.stack([pos[0] - rx + cc * 2 * rx / (nc - 1), pos[1] - ry + rr * 2 * ry / (nr - 1)], 1))
        extents.append((np.array([pos[0] - rx, pos[1] - ry]), np.array([pos[0] + rx, pos[1] + ry])))
    if not boxes and not pts:
        return None
    spans = [(b[0][:2], b[1][:2]) for b in boxes] + extents
    lo = np.min([a for a, _ in spans], axis=0) - 1.0
    hi = np.max([b for _, b in spans], axis=0) + 1.0
    lo, hi = np.maximum(lo, -25), np.minimum(hi, 25)
    shape = (int(np.ceil((hi[1] - lo[1]) / cell)) + 1, int(np.ceil((hi[0] - lo[0]) / cell)) + 1)
    occ = np.zeros(shape, bool)
    for blo, bhi in boxes:
        c0, r0 = np.floor((blo[:2] - lo) / cell).astype(int)
        c1, r1 = np.ceil((bhi[:2] - lo) / cell).astype(int)
        occ[max(r0, 0):max(r1, 0) + 1, max(c0, 0):max(c1, 0) + 1] = True
    for p in pts:
        ij = np.round((p - lo) / cell).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < shape[1]) & (ij[:, 1] >= 0) & (ij[:, 1] < shape[0])
        occ[ij[ok, 1], ij[ok, 0]] = True
    return Planner(occ, lo, cell)


class Path:
    """A polyline to follow, parametrised by distance along it (for pure-pursuit tracking)."""

    def __init__(self, points):
        self.points = np.asarray(points, float).reshape(-1, 2)
        seg = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        self.cum = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(self.cum[-1])

    def at(self, s: float) -> np.ndarray:
        s = float(np.clip(s, 0.0, self.length))
        i = int(np.clip(np.searchsorted(self.cum, s, side="right") - 1, 0, len(self.points) - 2)) if len(self.points) > 1 else 0
        if len(self.points) == 1:
            return self.points[0].copy()
        seg = self.cum[i + 1] - self.cum[i]
        u = 0.0 if seg < 1e-9 else (s - self.cum[i]) / seg
        return self.points[i] + u * (self.points[i + 1] - self.points[i])

    def project(self, p) -> float:
        """Distance along the path of the point nearest to p."""
        p = np.asarray(p, float)[:2]
        best, best_s = np.inf, 0.0
        for i in range(len(self.points) - 1):
            a, b = self.points[i], self.points[i + 1]
            ab = b - a
            L2 = float(ab @ ab)
            u = 0.0 if L2 < 1e-12 else float(np.clip((p - a) @ ab / L2, 0, 1))
            d = float(np.linalg.norm(a + u * ab - p))
            if d < best:
                best, best_s = d, float(self.cum[i] + u * np.sqrt(L2))
        return best_s

    def carrot(self, p, lookahead: float) -> np.ndarray:
        return self.at(self.project(p) + lookahead)
