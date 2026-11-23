"""Simulated sensors: a 2D lidar and an RGB-D camera carried by the robot.

Both are computed from the shared MuJoCo state on demand (no extra bodies in the model), so any robot
gets them without changes to its description:

* lidar: a planar 360° laser at a fixed height above the floor (like the 2D scanners on AMRs), from the
  robot's base; hits on the robot's own body are dropped. Output follows sensor_msgs/LaserScan.
* camera: a forward-looking pinhole camera on the robot (head height for humanoids, the top of the body
  for other mobile robots), slightly tilted down; RGB and metric depth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    from anybot.sim import Sim

LIDAR_HEIGHT = 0.3  # m above the floor
LIDAR_RAYS = 360
LIDAR_RANGE = 8.0
CAM_FOVY = 70.0
CAM_PITCH = -20.0  # degrees, negative looks down


@dataclass
class Scan:
    angle_min: float
    angle_increment: float
    range_max: float
    ranges: np.ndarray  # inf where nothing was hit
    origin: np.ndarray  # lidar position in the world
    yaw: float  # robot heading (angles are relative to it)

    def points(self) -> np.ndarray:
        """World xy of every hit."""
        ok = np.isfinite(self.ranges)
        ang = self.yaw + self.angle_min + self.angle_increment * np.nonzero(ok)[0]
        r = self.ranges[ok]
        return np.stack([self.origin[0] + r * np.cos(ang), self.origin[1] + r * np.sin(ang)], 1)


def _own_geoms(sim: "Sim") -> np.ndarray:
    if getattr(sim, "_own_geom_mask", None) is None or len(sim._own_geom_mask) != sim.m.ngeom:
        mask = np.zeros(sim.m.ngeom, bool)
        bodies = set(sim._subtree(sim._bid(sim.profile.root_body)))
        for g in range(sim.m.ngeom):
            mask[g] = int(sim.m.geom_bodyid[g]) in bodies
        for name in sim.held.values():  # what it carries is part of the robot for its sensors
            mask[sim.m.geom_bodyid == sim._bid(name)] = True
        sim._own_geom_mask = mask
    return sim._own_geom_mask


def lidar(sim: "Sim", d: mujoco.MjData | None = None, rays: int = LIDAR_RAYS, rmax: float = LIDAR_RANGE) -> Scan:
    m, d = sim.m, d if d is not None else sim.d
    pos, yaw = sim._base_pose()
    origin = np.array([pos[0], pos[1], LIDAR_HEIGHT])
    ang = -np.pi + 2 * np.pi * np.arange(rays) / rays
    vec = np.stack([np.cos(yaw + ang), np.sin(yaw + ang), np.zeros(rays)], 1).ravel()
    geomid = np.zeros(rays, np.int32)
    dist = np.zeros(rays)
    groups = np.array([1, 1, 1, 0, 0, 0], np.uint8)  # what is visible (a scan's hidden terrain is group 3)
    mujoco.mj_multiRay(m, d, origin, vec, groups, 1, sim._bid(sim.profile.root_body), geomid, dist, None, rays, rmax)
    own = _own_geoms(sim)
    sim._own_geom_mask = None  # held objects change
    hit = (geomid >= 0) & (dist >= 0)
    hit[hit] &= ~own[geomid[hit]]
    ranges = np.where(hit, dist, np.inf)
    return Scan(float(ang[0]), float(2 * np.pi / rays), rmax, ranges, origin, float(yaw))


def camera_pose(sim: "Sim", d: mujoco.MjData | None = None) -> tuple[np.ndarray, float]:
    """Camera position and heading: on the head of a humanoid, on top of other mobile robots,
    and above-behind a fixed arm, looking at its workspace."""
    d = d if d is not None else sim.d
    pos, yaw = sim._base_pose()
    fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    if not sim.profile.floating_base and not sim.profile.planar_base:  # fixed arm: overview camera
        return pos + np.array([0, 0, 0.9]) - 0.6 * fwd, yaw
    top = sim.profile.height_m if sim.profile.category != "humanoid" else 0.92 * sim.profile.height_m
    return np.array([pos[0], pos[1], 0.0]) + 0.12 * fwd + [0, 0, max(top, 0.25)], yaw


def _forearm_geoms(sim: "Sim") -> np.ndarray:
    """Geoms of the forearms and hands (elbow down): a head camera should see the hands at work."""
    if getattr(sim, "_forearm_mask", None) is None or len(sim._forearm_mask) != sim.m.ngeom:
        mask = np.zeros(sim.m.ngeom, bool)
        for a in sim.arms.values():
            if len(a.joints) >= 4:
                elbow = int(sim.m.jnt_bodyid[a.joints[3]])
                for b in sim._subtree(elbow):
                    mask[sim.m.geom_bodyid == b] = True
        sim._forearm_mask = mask
    return sim._forearm_mask


def wrist_arm(sim: "Sim"):
    """The arm whose wrist camera to show: the one holding something, else the busiest one."""
    if not sim.arms or sim.profile.category == "humanoid":
        return None
    for arm in sim.held:
        return sim.arms[arm]
    return next(iter(sim.arms.values()))


def wrist_view(sim: "Sim", d: mujoco.MjData, a) -> tuple[np.ndarray, np.ndarray]:
    """GoPro-style camera on the wrist: behind and beside the fingers, looking where the gripper points."""
    R = d.xmat[a.ee_body].reshape(3, 3)
    ap = R @ a.approach
    ap /= np.linalg.norm(ap) or 1.0
    side = R @ (a.grip_axis if a.grip_axis is not None else np.array([0.0, 1.0, 0.0]))
    side = side - (side @ ap) * ap
    side /= np.linalg.norm(side) or 1.0
    up = np.cross(ap, side)
    tcp = sim._tcp(a, d)
    # "Up" for the camera: towards the forearm (elbow side), whatever way this arm's wrist is built.
    elbow = d.xpos[int(sim.m.jnt_bodyid[a.joints[max(0, len(a.joints) - 4)]])]
    toward = elbow - tcp
    toward = toward - (toward @ ap) * ap
    if np.linalg.norm(toward) > 1e-3:
        toward /= np.linalg.norm(toward)
        lateral = np.cross(ap, toward)
        # On the side of the forearm, a little towards the elbow: a three-quarter view of the wrist and
        # fingers that does not hide what they reach for.
        up = 0.85 * lateral + 0.5 * toward
        up /= np.linalg.norm(up)
    # Strapped on the forearm, a hand's length back and above: the wrist, hand and fingers are in the
    # lower part of the picture, what they reach for in the middle.
    pos = tcp - 0.34 * ap + 0.13 * up
    look = (tcp + 0.12 * ap) - pos
    return pos, look / np.linalg.norm(look)


def _hand_geoms(sim: "Sim", a) -> np.ndarray:
    """Geoms of an arm's hand and fingers (after its last joint)."""
    mask = np.zeros(sim.m.ngeom, bool)
    # The last links of the arm (wrist), the hand and the fingers: enough arm to see what it is doing.
    # The last two joints; a third when the wrist is compact (UR-style, all three wrist joints within a
    # hand's length), so the hand never looks cut off from the arm.
    n = 2
    if len(a.joints) >= 3:
        third = sim.d.xpos[int(sim.m.jnt_bodyid[a.joints[-3]])]
        if np.linalg.norm(third - sim._tcp(a, sim.d)) < 0.2:
            n = 3
    first = int(sim.m.jnt_bodyid[a.joints[max(0, len(a.joints) - n)]])
    for b in sim._subtree(first):
        mask[sim.m.geom_bodyid == b] = True
    return mask


def look_target(sim: "Sim", d: mujoco.MjData) -> np.ndarray | None:
    """What a humanoid looks at while it works: the object in its hands, else its hands when they are
    out in front of it (reaching, placing). None while it just walks."""
    if sim.profile.category != "humanoid" or not sim.arms:
        return None
    pos, yaw = sim._base_pose()
    fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    held = list(dict.fromkeys(sim.held.values()))
    if held:
        return d.xpos[sim._bid(held[0])].copy()
    tcps = [sim._tcp(a, d) for a in sim.arms.values()]
    out = [t for t in tcps if (t - pos) @ fwd > 0.18]
    return np.mean(out, axis=0) if out else None


class Camera:
    """Renders what the robot sees. Create and use it from one thread (it owns a GL context)."""

    def __init__(self, model: mujoco.MjModel, width: int = 320, height: int = 240):
        self.model = model
        self.width, self.height = width, height
        self.rgb = mujoco.Renderer(model, height, width)
        self.depth = mujoco.Renderer(model, height, width)
        self.depth.enable_depth_rendering()
        self._cam = mujoco.MjvCamera()
        self._opt = mujoco.MjvOption()

    def close(self) -> None:
        self.rgb.close()
        self.depth.close()

    def render(self, sim: "Sim", d: mujoco.MjData) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """RGB (uint8 HxWx3), depth (float32 m), camera position, heading."""
        p, yaw = camera_pose(sim, d)
        c = self._cam
        c.type = mujoco.mjtCamera.mjCAMERA_FREE
        arm = wrist_arm(sim)
        fovy = CAM_FOVY
        if arm is not None:  # arms: the wrist camera
            p, look = wrist_view(sim, d, arm)
            fovy = 80.0
            c.distance = 0.05
            c.lookat[:] = p + 0.05 * look
            c.azimuth = float(np.degrees(np.arctan2(look[1], look[0])))
            c.elevation = float(np.clip(np.degrees(np.arcsin(np.clip(look[2], -1, 1))), -89.5, 89.5))
            return self._draw(sim, d, c, p, yaw, fovy, show=_hand_geoms(sim, arm))
        # Head camera: turn and tilt towards what the hands are doing, smoothly (a head does not snap).
        want_yaw, want_pitch = 0.0, CAM_PITCH
        target = look_target(sim, d)
        if target is not None:
            v = target - p
            rel = float((np.arctan2(v[1], v[0]) - yaw + np.pi) % (2 * np.pi) - np.pi)
            want_yaw = float(np.clip(np.degrees(rel), -45, 45))
            want_pitch = float(np.clip(np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1]))), -75, -10))
        self._head = getattr(self, "_head", np.array([0.0, CAM_PITCH]))
        self._head += 0.35 * (np.array([want_yaw, want_pitch]) - self._head)
        az, el = np.degrees(yaw) + self._head[0], self._head[1]
        look = np.array([np.cos(np.radians(az)) * np.cos(np.radians(el)), np.sin(np.radians(az)) * np.cos(np.radians(el)),
                         np.sin(np.radians(el))])
        c.distance = 0.05
        c.lookat[:] = p + 0.05 * look
        c.azimuth, c.elevation = float(az), float(el)
        show = _forearm_geoms(sim) if sim.profile.category == "humanoid" else None  # its own hands stay in view
        return self._draw(sim, d, c, p, yaw, fovy, show)

    def _draw(self, sim: "Sim", d: mujoco.MjData, c, p, yaw, fovy: float, show: np.ndarray | None = None):
        # Hide the robot's own body from its camera (it would fill the view), except `show` (hands, fingers).
        self._opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]  # what the studio shows (hidden helpers stay hidden)
        for r in (self.rgb, self.depth):
            r.update_scene(d, c, self._opt)
        own = _own_geoms(sim).copy()
        if show is not None:
            own &= ~show
        for name in set(sim.held.values()):  # and what it carries stays visible
            own[sim.m.geom_bodyid == sim._bid(name)] = False
        for r in (self.rgb, self.depth):
            scn = r.scene
            for gc in scn.camera:  # the sensor's field of view, without touching the model (the studio shares it)
                if r is self.rgb:  # see the fingers a few cm away (the default near plane scales with the scene);
                    gc.frustum_near = 0.012  # not for depth: MuJoCo converts depth to metres with the model's own near
                top = gc.frustum_near * np.tan(np.radians(fovy) / 2)
                gc.frustum_top, gc.frustum_bottom = top, -top
                gc.frustum_width = top * self.width / self.height
                gc.frustum_center = 0.0
            for i in range(scn.ngeom):
                g = scn.geoms[i]
                if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and 0 <= g.objid < len(own) and own[g.objid]:
                    g.rgba[3] = 0.0
                    g.pos[2] -= 1000.0  # out of sight: the depth buffer ignores transparency
        return self.rgb.render(), self.depth.render().astype(np.float32), p, yaw
