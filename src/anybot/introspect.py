"""Turn an arbitrary MjModel into a RobotProfile: what it can move, where its limbs end, what kind of robot it is.

Everything here is derived from the kinematic tree and the model's home pose, with no per-robot config.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import mujoco
import numpy as np

J = mujoco.mjtJoint
_DOF_JOINTS = {int(J.mjJNT_HINGE), int(J.mjJNT_SLIDE), int(J.mjJNT_BALL)}
_ROUND_GEOMS = {int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_SPHERE), int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)}
_GRIPPER_WORDS = ("finger", "grip", "claw", "jaw", "pinch")
_JOINT_TYPE = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}


def _name(m: mujoco.MjModel, obj: mujoco.mjtObj, i: int) -> str:
    return mujoco.mj_id2name(m, obj, i) or f"{obj.name.removeprefix('mjOBJ_').lower()}{i}"


@dataclass
class JointInfo:
    name: str
    type: str
    range: list[float] | None
    actuator: str | None
    actuator_type: str | None  # position | velocity | motor | other


@dataclass
class Limb:
    kind: str  # leg | arm | wheel | other
    tip: str  # body at the end of the limb (the "hand" body for grippers)
    joints: list[str]
    gripper: bool = False
    gripper_actuators: list[str] = field(default_factory=list)


@dataclass
class RobotProfile:
    name: str
    category: str
    floating_base: bool
    root_body: str
    mass_kg: float
    height_m: float
    joints: list[JointInfo]
    limbs: list[Limb]
    wheels: list[str]
    actuators: list[str]
    cameras: list[str]
    keyframes: list[str]
    injected_actuators: list[str] = field(default_factory=list)
    planar_base: list[str] = field(default_factory=list)  # x, y, yaw joints of a holonomic base

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        counts: dict[str, int] = {}
        for limb in self.limbs:
            if limb.kind != "wheel":  # wheels are reported from the joint list below
                counts[limb.kind] = counts.get(limb.kind, 0) + 1
        parts = ", ".join(f"{n} {k}{'s' if n > 1 else ''}" for k, n in counts.items()) or "no limbs"
        grip = sum(limb.gripper for limb in self.limbs)
        return (
            f"{self.name}: {self.category} | {'floating' if self.floating_base else 'fixed'} base | "
            f"{len(self.joints)} joints, {len(self.actuators)} actuators | {parts}"
            + (f", {grip} gripper(s)" if grip else "")
            + (f", {len(self.wheels)} wheels" if self.wheels else "")
            + f" | {self.mass_kg:.1f} kg, {self.height_m:.2f} m"
        )


def _actuator_kind(m: mujoco.MjModel, a: int) -> str:
    if m.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_NONE:
        return "motor"
    if m.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE:
        if m.actuator_biasprm[a, 1] != 0:
            return "position"
        if m.actuator_biasprm[a, 2] != 0:
            return "velocity"
    return "other"


class _Tree:
    """Kinematic-tree helpers for one robot inside a model."""

    def __init__(self, m: mujoco.MjModel, root: int | None = None):
        self.m = m
        self.children: dict[int, list[int]] = {b: [] for b in range(m.nbody)}
        for b in range(1, m.nbody):
            self.children[int(m.body_parentid[b])].append(b)
        if root is not None:  # several robots share the model: this one is given
            self.root = root
            return
        # The robot is the world-child subtree with the most joints (scenes may hold props).
        # Several jointed fixed-base trees (e.g. a bimanual rig) form one robot rooted at the world.
        jointed = [r for r in self.children[0] if self.subtree_joints(r)]
        fixed = [r for r in jointed if m.body_jntnum[r] == 0 or int(m.jnt_type[m.body_jntadr[r]]) != int(J.mjJNT_FREE)]
        if len(fixed) > 1 and len(fixed) == len(jointed):
            self.root = 0
        elif jointed:
            self.root = max(jointed, key=lambda r: len(self.subtree_joints(r)))
        else:
            # No articulation at all (drones, rigid bodies): take the largest world child.
            self.root = max(self.children[0], key=lambda r: len(self.subtree(r)), default=0)

    def own_joints(self, b: int) -> list[int]:
        adr, num = self.m.body_jntadr[b], self.m.body_jntnum[b]
        return [j for j in range(adr, adr + num) if int(self.m.jnt_type[j]) in _DOF_JOINTS]

    def subtree(self, b: int) -> list[int]:
        out, stack = [], [b]
        while stack:
            x = stack.pop()
            out.append(x)
            stack.extend(self.children[x])
        return out

    def subtree_joints(self, b: int) -> list[int]:
        return [j for x in self.subtree(b) for j in self.own_joints(x)]

    def path(self, b: int) -> list[int]:
        """Bodies from root (exclusive) down to b (inclusive)."""
        out = []
        while b != self.root and b != 0:
            out.append(b)
            b = int(self.m.body_parentid[b])
        return out[::-1]

    def path_joints(self, b: int) -> list[int]:
        return [j for x in self.path(b) for j in self.own_joints(x)]


def _find_tips(t: _Tree) -> dict[int, bool]:
    """Limb tips → whether the tip is a collapsed multi-finger hand/gripper."""
    leaves = [b for b in t.subtree(t.root) if not t.children[b] and b != t.root]
    # Jointless leaves (sensor mounts, feet geoms) belong to their nearest jointed ancestor.
    tips: set[int] = set()
    for b in leaves:
        while b != t.root and not t.own_joints(b):
            b = int(t.m.body_parentid[b])
        if b != t.root:
            tips.add(b)

    # A body deep in a chain whose branches are all short is a hand: collapse its fingers.
    def is_hand(b: int) -> bool:
        branches = [c for c in t.children[b] if t.subtree_joints(c)]
        if len(branches) < 2 or len(t.path_joints(b)) < 3:
            return False
        return all(len(t.subtree_joints(c)) <= 4 for c in branches)

    result: dict[int, bool] = {}
    for tip in tips:
        hand = next((b for b in t.path(tip) if is_hand(b)), None)
        if hand is not None:
            result[hand] = True
        else:
            result.setdefault(tip, False)
    return result


def actuator_joints(m: mujoco.MjModel, a: int) -> list[int]:
    """Joints an actuator moves, through a joint or a (fixed) tendon transmission."""
    if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
        return [int(m.actuator_trnid[a, 0])]
    if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_TENDON:
        t = int(m.actuator_trnid[a, 0])
        adr, num = m.tendon_adr[t], m.tendon_num[t]
        return [int(m.wrap_objid[k]) for k in range(adr, adr + num) if m.wrap_type[k] == mujoco.mjtWrap.mjWRAP_JOINT]
    return []


def body_geoms(m: mujoco.MjModel, bodies: list[int]) -> list[int]:
    """Non-plane geoms attached to the given bodies."""
    return [
        g
        for b in bodies
        for g in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b])
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE
    ]


def world_aabb(m: mujoco.MjModel, d: mujoco.MjData, geoms: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned world bounding box (min, max) of the given geoms, from their local AABBs."""
    if not geoms:
        return np.zeros(3), np.zeros(3)
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    pts = []
    for g in geoms:
        center, half = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
        local = center + signs * half
        pts.append(d.geom_xpos[g] + local @ d.geom_xmat[g].reshape(3, 3).T)
    allpts = np.concatenate(pts)
    return allpts.min(axis=0), allpts.max(axis=0)


def _lowest_z(t: _Tree, d: mujoco.MjData, b: int) -> float:
    geoms = [g for x in t.subtree(b) for g in range(t.m.body_geomadr[x], t.m.body_geomadr[x] + t.m.body_geomnum[x])]
    return float(min(d.geom_xpos[g, 2] for g in geoms)) if geoms else float(d.xpos[b, 2])


def _is_wheel_joint(t: _Tree, j: int) -> bool:
    m = t.m
    if m.jnt_type[j] != J.mjJNT_HINGE:
        return False
    name = _name(m, mujoco.mjtObj.mjOBJ_JOINT, j).lower()
    if "wheel" in name or "caster" in name:
        return True
    b = int(m.jnt_bodyid[j])
    round_geom = any(
        int(m.geom_type[g]) in _ROUND_GEOMS for g in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b])
    )
    # A continuous (unlimited) joint on a childless round body spins like a wheel.
    return not m.jnt_limited[j] and round_geom and not t.subtree_joints(b)[1:] and not t.children[b]


def _planar_base(m: mujoco.MjModel, t: "_Tree") -> list[int]:
    """Joints of a holonomic base modelled as slide-x, slide-y and yaw hinge on the root body."""
    own = t.own_joints(t.root)
    slides = [j for j in own if int(m.jnt_type[j]) == int(J.mjJNT_SLIDE) and abs(m.jnt_axis[j][2]) < 0.1]
    yaws = [j for j in own if int(m.jnt_type[j]) == int(J.mjJNT_HINGE) and abs(m.jnt_axis[j][2]) > 0.9]
    if len(slides) == 2 and len(yaws) == 1 and abs(float(np.dot(m.jnt_axis[slides[0]], m.jnt_axis[slides[1]]))) < 0.1:
        x = max(slides, key=lambda j: abs(m.jnt_axis[j][0]))
        y = slides[0] if x == slides[1] else slides[1]
        return [x, y, yaws[0]]
    return []


def _classify(
    floating: bool, limbs: list[Limb], wheels: list[str], n_dof: int, nu: int, actuators: list[str],
    planar: bool = False,
) -> str:
    legs = sum(limb.kind == "leg" for limb in limbs)
    arms = sum(limb.kind == "arm" for limb in limbs)
    if planar:
        return "mobile_manipulator" if arms else "mobile"
    if floating and n_dof == 0 and nu > 0:
        return "aerial"
    if legs >= 2 and wheels:
        return "wheel_legged"
    if legs == 2:
        return "humanoid" if arms else "biped"
    if legs == 4:
        return "quadruped_manipulator" if arms else "quadruped"
    if legs >= 6:
        return "hexapod"
    if len(wheels) >= 2 or (floating and legs == 0 and wheels):
        return "mobile_manipulator" if arms else "mobile"
    if not floating:
        if arms >= 3:
            return "hand"
        if arms == 2:
            return "dual_arm"
        if arms == 1:
            return "arm"
        if nu and any(w in a.lower() for a in actuators for w in _GRIPPER_WORDS):
            return "gripper"
    return "unknown"


def introspect(model: mujoco.MjModel, name: str = "robot", injected: list[str] | None = None,
               root: int | None = None, qpos: np.ndarray | None = None) -> RobotProfile:
    """Profile one robot. In a multi-robot scene pass its `root` body and the pose to analyse (`qpos`)."""
    m = model
    t = _Tree(m, root)
    d = mujoco.MjData(m)
    if qpos is not None:
        d.qpos[:] = qpos
    elif m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    own_joints = set(t.subtree_joints(t.root)) | set(range(m.body_jntadr[t.root], m.body_jntadr[t.root] + m.body_jntnum[t.root]))
    own_acts = [a for a in range(m.nu) if set(actuator_joints(m, a)) & own_joints
                or (root is None and not actuator_joints(m, a))]  # site/body drives (drones) when alone

    root_joints = [j for j in range(m.body_jntadr[t.root], m.body_jntadr[t.root] + m.body_jntnum[t.root])]
    floating = any(m.jnt_type[j] == J.mjJNT_FREE for j in root_joints)

    # Joints and their actuators.
    joint_act: dict[int, int] = {}
    for a in own_acts:
        if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
            joint_act.setdefault(int(m.actuator_trnid[a, 0]), a)
    dof_joints = [j for b in t.subtree(t.root) for j in t.own_joints(b)]
    jname = lambda j: _name(m, mujoco.mjtObj.mjOBJ_JOINT, j)  # noqa: E731
    aname = lambda a: _name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)  # noqa: E731
    joints = [
        JointInfo(
            name=jname(j),
            type=_JOINT_TYPE[int(m.jnt_type[j])],
            range=[round(float(x), 4) for x in m.jnt_range[j]] if m.jnt_limited[j] else None,
            actuator=aname(joint_act[j]) if j in joint_act else None,
            actuator_type=_actuator_kind(m, joint_act[j]) if j in joint_act else None,
        )
        for j in dof_joints
    ]

    # Limbs.
    tips = _find_tips(t)
    paths = {tip: set(t.path_joints(tip)) for tip in tips}
    wheel_joints = {j for j in dof_joints if _is_wheel_joint(t, j)}
    lowest = {tip: _lowest_z(t, d, tip) for tip in tips}
    all_geoms_z = [_lowest_z(t, d, t.root)] + list(lowest.values())
    ground = min(all_geoms_z)
    lo, hi = world_aabb(m, d, body_geoms(m, t.subtree(t.root)))
    height = float(hi[2] - lo[2])

    limbs: list[Limb] = []
    for tip, is_hand in sorted(tips.items()):
        others = set().union(*(p for o, p in paths.items() if o != tip))
        own = [j for j in t.path_joints(tip) if j not in others]
        # Hands own their finger joints too.
        if is_hand:
            own += [j for j in t.subtree_joints(tip) if j not in own]
        names = [jname(j) for j in own]
        non_wheel = [j for j in own if j not in wheel_joints]
        tip_name = _name(m, mujoco.mjtObj.mjOBJ_BODY, tip)

        # Gripper actuators must drive a joint of *this* limb (bimanual robots have one per arm).
        limb_joints = set(own) | set(t.subtree_joints(tip))
        gripper_acts = [
            aname(a)
            for a in own_acts
            if set(actuator_joints(m, a)) & limb_joints
            and (
                any(w in aname(a).lower() for w in _GRIPPER_WORDS)
                or any(w in jname(j).lower() for j in actuator_joints(m, a) for w in _GRIPPER_WORDS)
                or (is_hand and not set(actuator_joints(m, a)) & set(t.path_joints(tip)))
            )
        ]
        gripper = is_hand or any(w in n.lower() for n in names + [tip_name] for w in _GRIPPER_WORDS)

        if own and not non_wheel:
            kind = "wheel"
        elif floating and len(non_wheel) >= 2 and lowest[tip] <= ground + max(0.05, 0.15 * height):
            kind = "leg"
        elif len(non_wheel) >= 3:
            kind = "arm"
        else:
            kind = "other"
        limbs.append(Limb(kind=kind, tip=tip_name, joints=names, gripper=gripper and kind == "arm", gripper_actuators=gripper_acts if kind == "arm" else []))

    actuators = [aname(a) for a in own_acts]
    planar = _planar_base(m, t)
    category = _classify(floating, limbs, [jname(j) for j in wheel_joints], len(dof_joints), len(own_acts), actuators, bool(planar))
    if category == "hand":
        for limb in limbs:
            if limb.kind == "arm":
                limb.kind, limb.gripper, limb.gripper_actuators = "finger", False, []
    return RobotProfile(
        name=name,
        category=category,
        floating_base=floating,
        root_body=_name(m, mujoco.mjtObj.mjOBJ_BODY, t.root),
        mass_kg=round(float(m.body_subtreemass[t.root]), 3),
        height_m=round(height, 3),
        joints=joints,
        limbs=limbs,
        wheels=sorted(jname(j) for j in wheel_joints),
        actuators=actuators,
        cameras=[_name(m, mujoco.mjtObj.mjOBJ_CAMERA, c) for c in range(m.ncam)],
        keyframes=[_name(m, mujoco.mjtObj.mjOBJ_KEY, k) for k in range(m.nkey)],
        injected_actuators=list(injected or []),
        planar_base=[jname(j) for j in planar],
    )
