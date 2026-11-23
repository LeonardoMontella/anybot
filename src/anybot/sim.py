"""A live, thread-safe MuJoCo simulation of one robot with robot-agnostic skills.

Skills are built from the RobotProfile, not from per-robot code:
  * joint-space moves with min-jerk interpolation on position actuators,
  * end-effector moves via damped-least-squares IK on any arm limb,
  * grippers whose open/close direction is calibrated in simulation at load time,
  * keyframe poses (stand, home, ...), differential drive for wheeled bases,
  * spawning objects and rendering snapshots for the LLM to look at.
"""

from __future__ import annotations

import copy
import difflib
import io
import sys
import threading
import time
from dataclasses import dataclass, field

import mujoco
import numpy as np

from anybot.introspect import RobotProfile, body_geoms, introspect, world_aabb
from anybot.loader import load_robot

_GRIPPER_WORDS = ("finger", "grip", "claw", "jaw", "pinch")
_SHAPES = {
    "box": mujoco.mjtGeom.mjGEOM_BOX,
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
    "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
}
_COLORS = {
    "red": [0.85, 0.15, 0.15, 1], "green": [0.2, 0.7, 0.25, 1], "blue": [0.15, 0.35, 0.85, 1],
    "yellow": [0.95, 0.8, 0.1, 1], "orange": [1.0, 0.31, 0.0, 1], "white": [0.95, 0.95, 0.95, 1],
    "black": [0.1, 0.1, 0.1, 1], "gray": [0.5, 0.5, 0.5, 1], "purple": [0.55, 0.25, 0.7, 1],
}


J_HINGE = mujoco.mjtJoint.mjJNT_HINGE


class SkillError(ValueError):
    """A request the robot cannot fulfil; the message is meant for the LLM."""


def _suggest(name: str, options: list[str]) -> str:
    close = difflib.get_close_matches(name, options, n=3, cutoff=0.3)
    return f" Did you mean: {', '.join(close)}?" if close else f" Options: {', '.join(options[:30])}"


def _min_jerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s**3 * (10 - 15 * s + 6 * s * s)


@dataclass
class Arm:
    name: str  # tip body name, used as the limb's id
    ee_body: int
    tcp_offset: np.ndarray  # tool centre point in ee_body frame
    approach: np.ndarray  # approach axis in ee_body frame (unit)
    joints: list[int]  # hinge/slide joint ids driven by IK
    gripper_actuators: list[int] = field(default_factory=list)
    gripper_open: list[float] = field(default_factory=list)
    gripper_closed: list[float] = field(default_factory=list)
    finger_bodies: list[int] = field(default_factory=list)
    suction: bool = False  # no fingers: objects are held by an attach (suction-style) constraint
    grip_axis: np.ndarray | None = None  # direction the fingers close along, in ee_body frame
    assisted: bool = False  # multi-finger hand: fingers close on the object, a constraint secures it


class Sim:
    """One robot. Standalone (Sim("panda")) it owns its Scene; in a team it shares one (see team.py)."""

    # Shared physics state lives in the Scene: one model and one data for every robot in it.
    m = property(lambda self: self.scene.m, lambda self, v: setattr(self.scene, "m", v))
    d = property(lambda self: self.scene.d, lambda self, v: setattr(self.scene, "d", v))
    spec = property(lambda self: self.scene.spec)
    _lock = property(lambda self: self.scene.lock)
    cmd = property(lambda self: self.scene.cmd, lambda self, v: self.scene.cmd.__setitem__(slice(None), v))
    objects = property(lambda self: self.scene.objects)
    object_meta = property(lambda self: self.scene.object_meta)
    zones = property(lambda self: self.scene.zones, lambda self, v: setattr(self.scene, "zones", v))
    stations = property(lambda self: self.scene.stations, lambda self, v: setattr(self.scene, "stations", v))
    pad_half = property(lambda self: self.scene.pad_half, lambda self, v: setattr(self.scene, "pad_half", v))
    world = property(lambda self: self.scene.world)
    _running = property(lambda self: self.scene.running)

    def __init__(
        self,
        source: str,
        *,
        floating: bool | None = None,
        pin_base: bool = False,
        realtime: bool = True,
        viewer: bool = False,
        world: str | None = None,
    ):
        from anybot.scene import Scene

        robot = load_robot(source, floating=floating, world=world)
        self.scene = Scene(robot.spec, robot.model, world)
        self.prefix = ""
        self._home: dict | None = None  # standalone: the model's own keyframe is the home pose
        self._setup(robot.name, robot.injected_actuators, robot.notes, root=None, pin_base=pin_base)
        if self.world:
            from anybot import worlds

            worlds.populate(self)
        self.scene.start(realtime=realtime, viewer=viewer)

    @classmethod
    def in_scene(cls, scene, name: str, prefix: str, root: int, injected: list[str], notes: list[str],
                 home: dict) -> "Sim":
        """A robot that is part of a composed multi-robot scene (built by anybot.team)."""
        self = cls.__new__(cls)
        self.scene = scene
        self.prefix = prefix
        self._home = home  # {"qpos": {qpos address: value}, "ctrl": {actuator: value}}
        self._setup(name, injected, notes, root=root, pin_base=False)
        return self

    def _setup(self, name: str, injected: list[str], notes: list[str], root: int | None, pin_base: bool) -> None:
        self.name = name
        self.import_notes = notes
        home_q = self.home_qpos() if root is not None else None
        self.profile: RobotProfile = introspect(self.m, name, injected, root=root, qpos=home_q)
        root_id = self._bid(self.profile.root_body)
        self._own_joints = {j for b in self._subtree(root_id) for j in range(self.m.body_jntadr[b], self.m.body_jntadr[b] + self.m.body_jntnum[b])}
        from anybot.introspect import actuator_joints

        self._own_acts = [a for a in range(self.m.nu)
                          if set(actuator_joints(self.m, a)) & self._own_joints or (root is None and not actuator_joints(self.m, a))]
        self.pinned = False
        if pin_base and self.profile.floating_base:
            self._pin_base()

        self._init_servos()
        self.scene.ensure_data()
        self._traj: dict[int, tuple[float, float]] = {}
        self._traj_t0 = 0.0
        self._traj_dur = 0.0
        self.held: dict[str, str] = {}  # arm name → object it is holding
        self._welds: dict[str, tuple] = {}  # arm name → (equality name, rel pos, rel quat) while attached
        self.target: list[float] | None = None  # current navigation goal (drawn by the UI)
        self.navigating = False
        self._base_map: dict | None = None  # learned base actuation (tendon/virtual drives), lazily
        self.gait = None
        self._gravcomp: np.ndarray | None = None
        self.arms: dict[str, Arm] = {}
        if self not in self.scene.robots:
            self.scene.robots.append(self)
        self._reset_state()
        self._home_cmd = self.cmd.copy()
        self._rest_q = None  # found on first use (the arms are not known yet)
        self._folded: set[str] = set()  # arms folded in their travel pose

        self._build_skills()
        self._limit_base_force()
        arm_dofs = sorted({int(self.m.jnt_dofadr[j]) for a in self.arms.values() for j in a.joints})
        self._gravcomp = np.array(arm_dofs, dtype=int) if arm_dofs else None
        self.gait = self._make_gait()
        if getattr(self.gait, "keeps_balance", False) and not getattr(self.scene, "building", False):
            self.settle()

    def settle(self) -> None:
        """Let a balancing policy settle onto its feet, then stand still until asked to walk."""
        if getattr(self.gait, "keeps_balance", False):
            self.wait(1.2)
            self._freeze()

    def home_qpos(self) -> np.ndarray:
        """Full qpos vector with this robot in its home pose (other robots and objects as they are)."""
        if self._home is None:
            return self.m.key_qpos[0].copy() if self.m.nkey else self.m.qpos0.copy()
        q = self.d.qpos.copy() if self.d is not None else self.m.qpos0.copy()
        for adr, v in self._home["qpos"].items():
            q[adr] = v
        return q

    # ------------------------------------------------------------------ setup

    def _pin_base(self) -> None:
        """Remove the root free joint so a legged robot hangs in place (handy for arm demos)."""
        root = self.spec.worldbody.first_body()
        free = [j for j in root.joints if j.type == mujoco.mjtJoint.mjJNT_FREE]
        if not free:
            return
        q = self.m.key_qpos[0] if self.m.nkey else self.m.qpos0
        root.pos, root.quat = q[:3].copy(), q[3:7].copy()
        self.spec.delete(free[0])
        for key in self.spec.keys:
            if len(key.qpos):
                key.qpos = list(key.qpos)[7:]
            if len(key.qvel):
                key.qvel = list(key.qvel)[6:]
        self.m = self.spec.compile()
        self.pinned = True

    def _init_servos(self) -> None:
        """Map every hinge/slide joint to an actuator that can hold an angle.

        Native position actuators are used as is. Torque (`motor`) actuators get a software
        PD loop whose stiffness comes from the actuator's torque limit, so torque-controlled
        robots (Go2, H1, ...) accept the same angle commands.
        """
        m = self.m
        hinge_slide = {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}
        self.pos_act: dict[int, int] = {}
        self.pd: dict[int, tuple[int, float, float]] = {}  # actuator → (joint, kp, kd)
        for a in self._own_acts:
            if m.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
                continue
            j = int(m.actuator_trnid[a, 0])
            if int(m.jnt_type[j]) not in hinge_slide or j in self.pos_act:
                continue
            if m.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE and m.actuator_biasprm[a, 1] != 0:
                self.pos_act[j] = a
            elif m.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_NONE and m.actuator_dyntype[a] == mujoco.mjtDyn.mjDYN_NONE:
                gear = float(m.actuator_gear[a, 0]) or 1.0
                if m.actuator_ctrllimited[a]:
                    tmax = float(np.max(np.abs(m.actuator_ctrlrange[a]))) * gear
                else:
                    tmax = 50.0 * float(m.body_subtreemass[m.jnt_bodyid[j]])
                kp = tmax / 0.12
                self.pos_act[j] = a
                self.pd[a] = (j, kp, 0.03 * kp)

    def _write_ctrl(self) -> None:
        self.scene.write_ctrl()

    def _post_ctrl(self) -> None:
        """Called by the Scene after it copied the command vector into ctrl: this robot's extras."""
        m, d = self.m, self.d
        if getattr(self, "_gravcomp", None) is not None:
            # Gravity compensation on arm joints, as real arm controllers do: servos then track the
            # commanded path instead of sagging under the arm's own weight.
            d.qfrc_applied[self._gravcomp] = d.qfrc_bias[self._gravcomp]
        for a, (j, kp, kd) in self.pd.items():
            gear = float(m.actuator_gear[a, 0]) or 1.0
            q, qd = d.qpos[m.jnt_qposadr[j]], d.qvel[m.jnt_dofadr[j]]
            d.ctrl[a] = (kp * (self.cmd[a] - q) - kd * qd) / gear

    def _cmd_for_key(self, k: int | None) -> np.ndarray:
        """Command vector holding keyframe k (None → current pose). In a team, k=0 is the home pose."""
        if self._home is not None and k is not None:
            cmd = self.cmd.copy()
            for a, v in self._home["ctrl"].items():
                cmd[a] = v
            qpos = self.home_qpos()
        else:
            qpos = self.d.qpos if k is None else self.m.key_qpos[k]
            cmd = self.m.key_ctrl[k].copy() if k is not None else self.cmd.copy()
        for j, a in self.pos_act.items():
            cmd[a] = qpos[self.m.jnt_qposadr[j]]
        return cmd

    def _reset_state(self) -> None:
        if self._home is not None:  # team member: put only this robot back home
            q = self.home_qpos()
            cmd = self._cmd_for_key(0)
            for adr in self._home["qpos"]:
                self.d.qpos[adr] = q[adr]
            for j in self._own_joints:
                n = {0: 6, 1: 3}.get(int(self.m.jnt_type[j]), 1)
                self.d.qvel[self.m.jnt_dofadr[j]:self.m.jnt_dofadr[j] + n] = 0
            for a in self._own_acts:
                self.cmd[a] = cmd[a]
            self._write_ctrl()
            mujoco.mj_forward(self.m, self.d)
            return
        if self.m.nkey:
            mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
            self.cmd = self._cmd_for_key(0)
        else:
            mujoco.mj_resetData(self.m, self.d)
            self.cmd = self._cmd_for_key(None)
        self._write_ctrl()
        mujoco.mj_forward(self.m, self.d)

    def _jid(self, name: str) -> int:
        return mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, name)

    def _bid(self, name: str) -> int:
        return mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, name)

    def _clip_cmd(self, cmd: np.ndarray) -> np.ndarray:
        """Clip commands: PD servos to their joint range, everything else to its ctrl range."""
        m = self.m
        for a in range(m.nu):
            if a in self.pd:
                j = self.pd[a][0]
                if m.jnt_limited[j]:
                    cmd[a] = np.clip(cmd[a], *m.jnt_range[j])
            elif m.actuator_ctrllimited[a]:
                cmd[a] = np.clip(cmd[a], *m.actuator_ctrlrange[a])
        return cmd

    def _is_ancestor(self, anc: int, b: int) -> bool:
        while b > 0:
            if b == anc:
                return True
            b = int(self.m.body_parentid[b])
        return anc == 0

    def _subtree(self, b: int) -> list[int]:
        return [x for x in range(self.m.nbody) if self._is_ancestor(b, x)]

    def _build_skills(self) -> None:
        self.arms: dict[str, Arm] = {}
        for limb in self.profile.limbs:
            if limb.kind != "arm":
                continue
            tip = self._bid(limb.tip)
            jids = [self._jid(n) for n in limb.joints]
            hinge_slide = {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}
            is_grip = lambda j: any(w in self.m.joint(j).name.lower() for w in _GRIPPER_WORDS)  # noqa: E731
            arm_j = [
                j for j in jids
                if self._is_ancestor(int(self.m.jnt_bodyid[j]), tip) and not is_grip(j)
                and int(self.m.jnt_type[j]) in hinge_slide
            ]
            grip_j = [j for j in jids if j not in arm_j]
            ee = tip
            finger_bodies: list[int] = []
            if grip_j:
                # The palm carrying the fingers is the IK body; the fingers define the TCP.
                ee = int(self.m.body_parentid[self.m.jnt_bodyid[grip_j[0]]])
                finger_bodies = sorted({b for j in grip_j for b in self._subtree(int(self.m.jnt_bodyid[j]))})
            arm = Arm(
                name=limb.tip, ee_body=ee, tcp_offset=np.zeros(3), approach=np.array([0, 0, 1.0]), joints=arm_j
            )
            arm.finger_bodies = finger_bodies
            self._set_tcp(arm, finger_bodies)
            self._calibrate_gripper(arm, limb.gripper_actuators, grip_j, finger_bodies)
            if not arm.gripper_actuators:
                arm.suction = True
                self._set_suction_tcp(arm)
            else:
                arm.grip_axis = self._grip_axis(arm)
                self._hand_geometry(arm)
            self.arms[limb.tip] = arm

    def _set_tcp(self, arm: Arm, finger_bodies: list[int]) -> None:
        d = self.d
        R = d.xmat[arm.ee_body].reshape(3, 3)
        geoms = body_geoms(self.m, finger_bodies)
        if geoms:
            # TCP: centroid of the finger geoms, i.e. between the fingers.
            center = d.geom_xpos[geoms].mean(axis=0)
            arm.tcp_offset = R.T @ (center - d.xpos[arm.ee_body])
        if np.linalg.norm(arm.tcp_offset) > 1e-4:
            arm.approach = arm.tcp_offset / np.linalg.norm(arm.tcp_offset)

    def _hand_geometry(self, arm: Arm) -> None:
        """Multi-finger hands: grasp point between the thumb tip and the other fingertips (hand open),
        closing direction along the line thumb → fingers. Two-finger grippers keep their simpler model."""
        m = self.m
        if len(arm.gripper_actuators) < 3:
            # One or two drives: a parallel gripper (however many linkage parts it has). Grasp
            # point: centred between the fingers, 2 cm inside the fingertips (measured open).
            geoms = body_geoms(m, arm.finger_bodies)
            if not geoms or np.linalg.norm(arm.approach) < 0.5:
                return
            d = copy.copy(self.d)
            for _ in range(300):
                d.ctrl[arm.gripper_actuators] = arm.gripper_open
                mujoco.mj_step(m, d)
            R, ee = d.xmat[arm.ee_body].reshape(3, 3), d.xpos[arm.ee_body]
            ax = R @ arm.approach
            depth = max(float((d.geom_xpos[g] - ee) @ ax) for g in geoms)
            centre = d.geom_xpos[geoms].mean(axis=0)
            lateral = (centre - ee) - ((centre - ee) @ ax) * ax
            arm.tcp_offset = R.T @ (lateral + (depth - 0.02) * ax)
            return
        fingers = set(arm.finger_bodies)
        roots = [b for b in fingers if int(m.body_parentid[b]) not in fingers]
        if len(roots) < 3:
            return
        d = copy.copy(self.d)
        for _ in range(300):  # open the hand in a scratch copy
            d.ctrl[arm.gripper_actuators] = arm.gripper_open
            mujoco.mj_step(m, d)
        ee = d.xpos[arm.ee_body]
        tips = []
        for r in roots:
            geoms = body_geoms(m, self._subtree(r))
            if geoms:
                tips.append(max((d.geom_xpos[g] for g in geoms), key=lambda p: float(np.linalg.norm(p - ee))))
        # Linkage grippers (e.g. Robotiq) have several parts per side but only two fingertips:
        # merge tips that end up within 3 cm of each other before counting fingers.
        merged: list[np.ndarray] = []
        for t in tips:
            if all(np.linalg.norm(t - u) > 0.03 for u in merged):
                merged.append(t)
        if len(merged) < 3:
            return
        arm.assisted = True
        tips = np.array(merged)
        k = int(np.argmax([np.linalg.norm(t - np.delete(tips, i, axis=0).mean(axis=0)) for i, t in enumerate(tips)]))
        thumb, others = tips[k], np.delete(tips, k, axis=0).mean(axis=0)
        R = d.xmat[arm.ee_body].reshape(3, 3)
        arm.tcp_offset = R.T @ ((thumb + others) / 2 - ee)
        arm.approach = arm.tcp_offset / np.linalg.norm(arm.tcp_offset)
        axis = R.T @ (others - thumb)
        axis -= axis @ arm.approach * arm.approach
        if np.linalg.norm(axis) > 1e-3:
            arm.grip_axis = axis / np.linalg.norm(axis)

    def _grip_axis(self, arm: Arm) -> np.ndarray | None:
        """Closing direction of a two-sided gripper: the line between its two farthest-apart finger geoms."""
        geoms = body_geoms(self.m, arm.finger_bodies)
        if len(geoms) < 2:
            return None
        pts = self.d.geom_xpos[geoms]
        i, j = max(((i, j) for i in range(len(pts)) for j in range(i + 1, len(pts))),
                   key=lambda ij: float(np.linalg.norm(pts[ij[0]] - pts[ij[1]])))
        v = pts[i] - pts[j]
        if np.linalg.norm(v) < 0.02:
            return None
        R = self.d.xmat[arm.ee_body].reshape(3, 3)
        axis = R.T @ (v / np.linalg.norm(v))
        axis -= axis @ arm.approach * arm.approach  # keep it perpendicular to the approach direction
        n = np.linalg.norm(axis)
        return axis / n if n > 0.3 else None

    def _set_suction_tcp(self, arm: Arm) -> None:
        """Tool point of a fingerless arm: the far end of its last link (palm, flange)."""
        d, tip = self.d, arm.ee_body
        geoms = body_geoms(self.m, self._subtree(tip))
        if not geoms:
            return
        lo, hi = world_aabb(self.m, d, geoms)
        v = (lo + hi) / 2 - d.xpos[tip]
        n = float(np.linalg.norm(v))
        if n < 1e-3:
            return
        dirn = v / n
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        far = float(np.max((corners - d.xpos[tip]) @ dirn))
        R = d.xmat[tip].reshape(3, 3)
        arm.tcp_offset = R.T @ (dirn * min(far, 2.2 * n))
        arm.approach = R.T @ dirn

    def _finger_spread(self, d: mujoco.MjData, arm: Arm, finger_bodies: list[int]) -> float:
        per_body = [d.geom_xpos[body_geoms(self.m, [b])].mean(axis=0) for b in finger_bodies if body_geoms(self.m, [b])]
        if len(per_body) >= 2:
            return max(float(np.linalg.norm(a - b)) for i, a in enumerate(per_body) for b in per_body[i + 1:])
        if per_body:
            base = body_geoms(self.m, [arm.ee_body])
            ref = d.geom_xpos[base].mean(axis=0) if base else d.xpos[arm.ee_body]
            return float(np.linalg.norm(per_body[0] - ref))
        return 0.0

    def _calibrate_gripper(self, arm: Arm, act_names: list[str], grip_joints: list[int], fingers: list[int]) -> None:
        """Find which end of each gripper actuator's range opens the gripper by trying both in a scratch copy."""
        acts = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in act_names]
        acts += [self.pos_act[j] for j in grip_joints if j in self.pos_act]
        acts = sorted({a for a in acts if a >= 0})
        if not acts or not fingers:
            return
        lo = []
        hi = []
        for a in acts:
            if self.m.actuator_ctrllimited[a]:
                lo.append(float(self.m.actuator_ctrlrange[a, 0])); hi.append(float(self.m.actuator_ctrlrange[a, 1]))
            else:
                j = int(self.m.actuator_trnid[a, 0])
                lo.append(float(self.m.jnt_range[j, 0])); hi.append(float(self.m.jnt_range[j, 1]))

        def spread(values: list[float]) -> float:
            d = copy.copy(self.d)
            for _ in range(400):
                d.ctrl[acts] = values
                mujoco.mj_step(self.m, d)
            return self._finger_spread(d, arm, fingers)

        s_lo, s_hi = spread(lo), spread(hi)
        arm.gripper_actuators = acts
        arm.gripper_open, arm.gripper_closed = (hi, lo) if s_hi >= s_lo else (lo, hi)

    # ------------------------------------------------------------------ physics loop

    def _apply_traj(self) -> None:
        if not self._traj:
            return
        s = _min_jerk((self.d.time - self._traj_t0) / max(self._traj_dur, 1e-6))
        for a, (start, end) in self._traj.items():
            self.cmd[a] = start + s * (end - start)
        if s >= 1.0:
            self._traj = {}

    def _ramp_to(self, targets: dict[int, float], duration: float) -> None:
        """Move commands linearly to `targets` over `duration`, a little every physics step: a controller
        that updates at a few tens of Hz must not hand a stiff servo a step (it would jerk the robot)."""
        with self._lock:
            self._ramp = ({a: (float(self.cmd[a]), float(v)) for a, v in targets.items()}, float(self.d.time), max(duration, 1e-6))

    def _apply_ramp(self) -> None:
        ramp = getattr(self, "_ramp", None)
        if not ramp:
            return
        goals, t0, dur = ramp
        u = min(1.0, (self.d.time - t0) / dur)
        for a, (start, end) in goals.items():
            self.cmd[a] = start + u * (end - start)
        if u >= 1.0:
            self._ramp = None

    def _pre_step(self, tick: int, every: int) -> None:
        """Called by the Scene before each physics step: trajectories and gait control."""
        self._apply_traj()
        self._apply_ramp()
        if self.gait is not None and self.gait.active and tick % every == 0:
            self.gait.update(every * self.m.opt.timestep)

    def _step(self, n: int = 1) -> None:
        self.scene.step(n)

    def _open_viewer(self) -> None:
        self.scene.open_viewer()

    def close(self) -> None:
        if self.scene.robots and self.scene.robots[0] is self and self._home is None:
            self.scene.close()  # a standalone robot owns its scene

    def wait(self, seconds: float) -> None:
        """Let simulated time advance (real time when the physics thread runs)."""
        self.scene.wait(seconds)

    def _move_ctrl(self, targets: dict[int, float], duration: float, settle: float = 0.3) -> None:
        with self._lock:
            cmd = self.cmd.copy()
            for a, v in targets.items():
                cmd[a] = v
            cmd = self._clip_cmd(cmd)
            self._traj = {a: (float(self.cmd[a]), float(cmd[a])) for a in targets}
            self._traj_t0, self._traj_dur = self.d.time, max(duration, 0.05)
        self.wait(duration + settle)

    # ------------------------------------------------------------------ skills

    def state(self) -> dict:
        with self._lock:
            m, d = self.m, self.d
            out: dict = {"time_s": round(float(d.time), 2)}
            out["joints"] = {
                j.name: round(float(d.qpos[m.jnt_qposadr[self._jid(j.name)]]), 3)
                for j in self.profile.joints
                if j.type in ("hinge", "slide")
            }
            root = self._bid(self.profile.root_body)
            if self.profile.floating_base and not self.pinned:
                rpy = _quat_to_rpy(d.xquat[root])
                out["base"] = {"pos": _r(d.xpos[root]), "rpy_deg": _r(np.degrees(rpy), 1)}
            out["end_effectors"] = {n: _r(self._tcp(arm, d)) for n, arm in self.arms.items()}
            out["objects"] = {o: _r(d.xpos[self._bid(o)]) for o in self.objects}
            if self.held:
                out["holding"] = dict(self.held)
            return out

    def _tcp(self, arm: Arm, d: mujoco.MjData) -> np.ndarray:
        return d.xpos[arm.ee_body] + d.xmat[arm.ee_body].reshape(3, 3) @ arm.tcp_offset

    def set_joints(self, targets: dict[str, float], duration: float = 1.5) -> dict:
        names = [j.name for j in self.profile.joints]
        acts = {}
        for name, value in targets.items():
            j = self._jid(name)
            if j < 0:
                raise SkillError(f"Unknown joint '{name}'." + _suggest(name, names))
            if j not in self.pos_act:
                raise SkillError(f"Joint '{name}' has no position actuator, so it cannot be commanded by angle.")
            acts[self.pos_act[j]] = float(value)
        self._move_ctrl(acts, duration)
        return self.state()["joints"]

    def _arm(self, limb: str | None) -> Arm:
        if not self.arms:
            raise SkillError(f"{self.name} ({self.profile.category}) has no arm.")
        if limb is None:
            return next(iter(self.arms.values()))
        for n, arm in self.arms.items():
            if limb.lower() in n.lower():
                return arm
        raise SkillError(f"No arm matching '{limb}'. Arms: {', '.join(self.arms)}")

    def solve_ik(
        self, arm: Arm, target: np.ndarray, point_down: bool = False, iters: int = 400,
        seed: np.ndarray | None = None, align: np.ndarray | None = None, base_d: mujoco.MjData | None = None,
    ) -> tuple[np.ndarray, float]:
        """Damped least squares from the current pose; if that gets stuck (local minimum, wrist flipped
        the wrong way), retry from the home pose and a few random seeds and keep the best solution."""
        m = self.m
        with self._lock:
            d = copy.copy(self.d if base_d is None else base_d)  # base_d: the robot somewhere else (planning)
        qadr = np.array([m.jnt_qposadr[j] for j in arm.joints])
        dadr = np.array([m.jnt_dofadr[j] for j in arm.joints])
        lim = np.array([m.jnt_limited[j] for j in arm.joints], dtype=bool)
        rng = np.array([m.jnt_range[j] for j in arm.joints])
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        down = getattr(self, "_down_dir", None)  # "down" can be tilted for an angled grasp
        down = np.array([0, 0, -1.0]) if down is None else np.asarray(down, float)

        def run(q0: np.ndarray) -> tuple[np.ndarray, float, float]:
            d.qpos[qadr] = q0
            err_p, tilt = np.inf, 0.0
            for _ in range(iters):
                mujoco.mj_kinematics(m, d)
                mujoco.mj_comPos(m, d)
                R = d.xmat[arm.ee_body].reshape(3, 3)
                p = d.xpos[arm.ee_body] + R @ arm.tcp_offset
                e = target - p
                err_p = float(np.linalg.norm(e))
                mujoco.mj_jac(m, d, jacp, jacr, p, arm.ee_body)
                J = jacp[:, dadr]
                tilt = 0.0
                if point_down:
                    a = R @ arm.approach
                    tilt = 1.0 - float(a @ down)  # 0 when pointing straight down, 2 when straight up
                    er = np.cross(a, down)
                    if a @ down < 0:  # pointing up: the cross product vanishes, push sideways to flip
                        er += np.cross(a, [1.0, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1.0, 0])
                    if align is not None and arm.grip_axis is not None:
                        g = R @ arm.grip_axis
                        want = align if g @ align >= 0 else -align  # either way round is fine
                        er = er + 0.6 * np.cross(g, want)
                        tilt += 0.5 * (1.0 - abs(float(g @ align)))
                    J = np.vstack([J, 0.3 * jacr[:, dadr]])
                    e = np.concatenate([e, 0.3 * er])
                if err_p < 1e-3 and tilt < 2e-4:
                    break
                dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(len(e)), e)
                n = np.linalg.norm(dq)
                if n > 0.2:
                    dq *= 0.2 / n
                q = d.qpos[qadr] + dq
                q[lim] = np.clip(q[lim], rng[lim, 0], rng[lim, 1])
                d.qpos[qadr] = q
            return d.qpos[qadr].copy(), err_p, tilt

        good = lambda e, t: e < 2e-3 and t < 2e-3  # noqa: E731
        q_now = d.qpos[qadr].copy()
        spin = np.array([not lim[i] and int(m.jnt_type[j]) == int(J_HINGE) for i, j in enumerate(arm.joints)])

        def unwind(q: np.ndarray, ref: np.ndarray) -> np.ndarray:
            """Continuous joints: the same pose is reachable without turning a full circle."""
            q = q.copy()
            q[spin] = ref[spin] + (q[spin] - ref[spin] + np.pi) % (2 * np.pi) - np.pi
            return q
        best = run(q_now.copy() if seed is None else np.asarray(seed, dtype=float))
        if not good(best[1], best[2]) and seed is None:
            seeds = [np.array([self._home_cmd[self.pos_act[j]] if j in self.pos_act else d.qpos[m.jnt_qposadr[j]]
                               for j in arm.joints])]
            gen = np.random.default_rng(7)
            lo = np.where(lim, rng[:, 0], -np.pi)
            hi = np.where(lim, rng[:, 1], np.pi)
            seeds += [gen.uniform(lo, hi) for _ in range(6)]
            found = [best] + [run(q0) for q0 in seeds]
            ok = [c for c in found if good(c[1], c[2])]
            # Among solutions that work, the one closest to the current pose: no needless arm flips.
            found = [(unwind(c[0], q_now), c[1], c[2]) for c in found]
            ok = [c for c in found if good(c[1], c[2])]
            best = (min(ok, key=lambda c: float(np.max(np.abs(c[0] - q_now)))) if ok
                    else min(found, key=lambda c: c[1] + 0.05 * c[2]))
        return unwind(best[0], q_now if seed is None else np.asarray(seed, dtype=float)), best[1]

    def move_end_effector(
        self, xyz: list[float], limb: str | None = None, point_down: bool = False, duration: float = 2.0,
        align: list[float] | None = None,
    ) -> dict:
        """Move the tool point to xyz. `point_down` aims the gripper down; `align` (a horizontal world
        direction) also turns it so the fingers close along that direction."""
        arm = self._arm(limb)
        missing = [self.m.joint(j).name for j in arm.joints if j not in self.pos_act]
        if missing:
            raise SkillError(f"Arm joints without position actuators (torque-controlled): {missing}")
        target = np.asarray(xyz, dtype=float)
        align = None if align is None else np.asarray(align, dtype=float) / np.linalg.norm(align)
        q, err = self.solve_ik(arm, target, point_down, align=align)
        if err > 0.02:
            raise SkillError(
                f"Target {list(target)} is out of reach for '{arm.name}' (closest IK error {err:.3f} m). "
                "Try a point closer to the robot base."
            )
        # Straight line in space, not in joint space: a joint-space blend can dip the hand into
        # whatever stands between start and goal. Waypoints every ~6 cm, each seeded by the previous.
        path = [q]
        with self._lock:
            start = self._tcp(arm, self.d).copy()
            q_now = np.array([self.d.qpos[self.m.jnt_qposadr[j]] for j in arm.joints])
        n = int(np.ceil(np.linalg.norm(target - start) / 0.06))
        if n > 1:
            seed, waypoints = q_now, []
            for k in range(1, n + 1):
                qk, ek = self.solve_ik(arm, start + (target - start) * k / n, point_down, iters=150, seed=seed, align=align)
                if ek > 0.03 or np.max(np.abs(qk - seed)) > 0.6:
                    waypoints = []  # leaves the workspace or flips the arm: plain joint-space move
                    break
                waypoints.append(qk)
                seed = qk
            if waypoints and ek <= 0.02:
                path = waypoints
        # Collision check: if the arm would sweep through an object or a table on the way, go over it.
        if not getattr(self, "_avoiding", False):
            hit = self._path_hit(arm, [q_now, *path])
            if hit is not None:
                with self._lock:
                    start = self._tcp(arm, self.d).copy()
                safe = max(start[2], target[2], self._clear_height(start, target)) + 0.06
                self._avoiding = True
                try:
                    for via in ([start[0], start[1], safe], [target[0], target[1], safe]):
                        try:
                            self.move_end_effector(via, arm.name, point_down=point_down, duration=max(0.8, duration / 2), align=align)
                        except SkillError:
                            break  # cannot rise that high: go as planned
                    return self.move_end_effector(list(target), arm.name, point_down=point_down,
                                                  duration=max(0.8, duration / 2), align=align)
                finally:
                    self._avoiding = False
        # Joint speed limit (~1 rad/s): a big reconfiguration takes its time instead of flinging things.
        swing = float(np.max(np.abs(np.asarray(path[-1]) - q_now)))
        duration = max(duration, swing / 1.0)
        for i, qk in enumerate(path):
            last = i == len(path) - 1
            self._move_ctrl({self.pos_act[j]: float(v) for j, v in zip(arm.joints, qk)},
                            duration / len(path), settle=0.5 if last else 0.0)
        # Position servos sag under gravity: re-aim at target + residual until the TCP lands.
        aim = target.copy()
        for _ in range(3):
            with self._lock:
                reached = self._tcp(arm, self.d)
            residual = target - reached
            if np.linalg.norm(residual) < 3e-3:
                break
            aim = aim + residual
            with self._lock:
                q_cur = np.array([self.d.qpos[self.m.jnt_qposadr[j]] for j in arm.joints])
            # A small correction from where the arm is: never a jump to a different arm configuration.
            q, e_aim = self.solve_ik(arm, aim, point_down, iters=150, seed=q_cur, align=align)
            if e_aim > 0.01 or np.max(np.abs(q - q_cur)) > 0.3:
                break
            self._move_ctrl({self.pos_act[j]: float(v) for j, v in zip(arm.joints, q)}, 0.4, settle=0.4)
        with self._lock:
            reached = self._tcp(arm, self.d)
        return {"arm": arm.name, "target": _r(target), "reached": _r(reached),
                "error_m": round(float(np.linalg.norm(reached - target)), 4)}

    def _arm_bodies(self, arm: Arm) -> set[int]:
        return set(self._subtree(int(self.m.jnt_bodyid[arm.joints[0]])))

    def _path_hit(self, arm: Arm, qs: list[np.ndarray], steps: int = 6) -> str | None:
        """Kinematic dry run of a joint path: the first scene body the arm would touch, if any.

        The robot itself, the floor, held objects and whatever `self._touch_ok` lists (the object
        being grasped) do not count."""
        m = self.m
        with self._lock:
            d = copy.copy(self.d)
        qadr = [m.jnt_qposadr[j] for j in arm.joints]
        mine = self._arm_bodies(arm)
        root = self._bid(self.profile.root_body)
        held = {self._bid(o) for o in self.held.values()}
        allowed = held | set(getattr(self, "_touch_ok", ()))
        for a, b in zip(qs[:-1], qs[1:]):
            for k in range(1, steps + 1):
                d.qpos[qadr] = a + (np.asarray(b) - a) * k / steps
                mujoco.mj_kinematics(m, d)
                mujoco.mj_collision(m, d)
                for c in d.contact[:d.ncon]:
                    if c.dist > -0.002:
                        continue
                    b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
                    for here, other in ((b1, b2), (b2, b1)):
                        if here in mine and other not in allowed and other != 0 and not self._is_ancestor(root, other):
                            return m.body(other).name
        return None

    def _clear_height(self, a: np.ndarray, b: np.ndarray) -> float:
        """Height that clears everything standing along the straight line from a to b (plus the
        object in hand, which hangs below the tool)."""
        top = 0.0
        held = {self._bid(o) for o in self.held.values()}
        for t in np.linspace(0, 1, 9):
            p = a + (b - a) * t
            top = max(top, self.support_height(float(p[0]), float(p[1]), ignore=tuple(held)))
        hang = 0.0
        if held:
            lo, hi = world_aabb(self.m, self.d, body_geoms(self.m, list(held)))
            hang = float(hi[2] - lo[2]) + 0.03
        return top + hang + 0.08

    # ------------------------------------------------------------------ pick and place

    def _object_top(self, name: str) -> float:
        with self._lock:
            lo, hi = world_aabb(self.m, self.d, body_geoms(self.m, [self._bid(name)]))
        return float(hi[2])

    def _nearest_arm(self, point: np.ndarray, holding: bool | None = None) -> Arm:
        """The arm closest to a point (bimanual robots use the arm on that side)."""
        arms = list(self.arms.values())
        if holding is not None:
            arms = [a for a in arms if (a.name in self.held) == holding]
        if not arms:
            raise SkillError(f"{self.name} has no {'free ' if holding is False else ''}arm for this.")
        with self._lock:
            return min(arms, key=lambda a: float(np.linalg.norm(self._tcp(a, self.d)[:2] - point[:2])))

    def _fingertip_depth(self, a: Arm) -> float:
        """How far the fingertips reach below the TCP (with the gripper pointing down)."""
        geoms = body_geoms(self.m, a.finger_bodies)
        if not geoms:
            return 0.0
        with self._lock:
            lo, _ = world_aabb(self.m, self.d, geoms)
            return max(0.0, float(self._tcp(a, self.d)[2] - lo[2]))

    def _reach(self, a: Arm, xyz, duration: float, align=None) -> dict:
        """Move the tool point, palm/gripper facing down when the arm can, any orientation otherwise."""
        if align is not None:
            try:
                r = self.move_end_effector(list(xyz), a.name, point_down=True, duration=duration, align=align)
                if r["error_m"] < 0.02:
                    return r
                # The IK ignores self-collision: an aligned wrist can drive the arm into the body.
            except SkillError:
                pass
        try:
            return self.move_end_effector(list(xyz), a.name, point_down=True, duration=duration)
        except SkillError:
            if not (a.suction or a.assisted):
                raise
            return self.move_end_effector(list(xyz), a.name, duration=duration)

    def _lift(self, a: Arm, x: float, y: float, z: float, heights=(0.13, 0.1, 0.07, 0.05), duration=1.0) -> None:
        """Rise above (x, y, z) as high as the arm allows (at least a few cm)."""
        for h in heights:
            try:
                self._reach(a, [x, y, z + h], duration)
                return
            except SkillError:
                continue
        raise SkillError(f"'{a.name}' cannot lift from ({x:.2f}, {y:.2f}).")

    def _approach(self, a: Arm, x: float, y: float, z: float, clearances=(0.10, 0.06, 0.03), duration=1.5,
                  align=None) -> float:
        """Move above (x, y, z) at the highest clearance the arm can reach; returns the clearance used."""
        self._unfold(a, (x, y, z))
        if align is not None and not a.suction:
            # Prefer a lower hover with the right orientation over a higher one with a tilted or
            # turned gripper (which would then rotate on the way down and knock the object).
            for c in clearances:
                _, err = self.solve_ik(a, np.array([x, y, z + c]), point_down=True, align=align)
                if err < 0.004:
                    clearances = (c,) + tuple(cc for cc in clearances if cc < c)
                    break
        for c in clearances:
            try:
                self._reach(a, [x, y, z + c], duration, align=align)
                return c
            except SkillError:
                continue
        raise SkillError(f"({x:.2f}, {y:.2f}) is out of reach for '{a.name}' from above.")

    # -- world queries

    def support_height(self, x: float, y: float, ignore: tuple[int, ...] = (), from_z: float = 3.0) -> float:
        """Height of whatever one would put something on at (x, y): floor, table, another object.

        The robot itself and the bodies in `ignore` are looked through.
        """
        m, d = self.m, self.d
        root = self._bid(self.profile.root_body)
        origin, down = np.array([x, y, from_z]), np.array([0.0, 0.0, -1.0])
        geomid = np.zeros(1, dtype=np.int32)
        with self._lock:
            for _ in range(30):
                dist = mujoco.mj_ray(m, d, origin, down, None, 1, -1, geomid)
                if dist < 0:
                    return 0.0
                body = int(m.geom_bodyid[geomid[0]])
                if body in ignore or (body != 0 and self._is_ancestor(root, body)):
                    origin = origin + down * (dist + 1e-3)
                    continue
                return float(origin[2] - dist)
        return 0.0

    def _object_box(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            return world_aabb(self.m, self.d, body_geoms(self.m, [self._bid(name)]))

    # -- attach (suction-style) grasp for arms without fingers

    def _attach(self, a: Arm, name: str) -> None:
        m, d = self.m, self.d
        eq_name = f"hold::{a.name}::{name}"
        with self._lock:
            b1, b2 = a.ee_body, self._bid(name)
            R1 = d.xmat[b1].reshape(3, 3)
            rel_pos = R1.T @ (d.xpos[b2] - d.xpos[b1])
            q1_inv, rel_quat = np.zeros(4), np.zeros(4)
            mujoco.mju_negQuat(q1_inv, d.xquat[b1])
            mujoco.mju_mulQuat(rel_quat, q1_inv, d.xquat[b2])
            self._welds[a.name] = (eq_name, rel_pos, rel_quat)
            self._apply_welds()

    def _apply_welds(self) -> None:
        m, d = self.m, self.d
        for eq_name, rel_pos, rel_quat in self._welds.values():
            eq = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, eq_name)
            m.eq_data[eq, 0:3] = 0.0
            m.eq_data[eq, 3:6] = rel_pos
            m.eq_data[eq, 6:10] = rel_quat
            m.eq_data[eq, 10] = 1.0
            d.eq_active[eq] = 1

    def _detach(self, a: Arm) -> None:
        with self._lock:
            weld = self._welds.pop(a.name, None)
            if weld:
                eq = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_EQUALITY, weld[0])
                self.d.eq_active[eq] = 0

    # -- mobile manipulation: bring the base to where the arm can work

    def _can_walk(self) -> bool:
        return (self.gait is not None or bool(self.profile.planar_base)
                or bool(self.profile.wheels and self.profile.floating_base and not self.pinned))

    # -- standing still: a walking policy cannot stop stepping, so the base is held while idle

    def _anchor_relpose(self) -> tuple[np.ndarray, np.ndarray]:
        root = self._bid(self.profile.root_body)
        p, q = self.d.xpos[root], self.d.xquat[root]
        q_inv, rel_q = np.zeros(4), np.zeros(4)
        mujoco.mju_negQuat(q_inv, q)
        mujoco.mju_mulQuat(rel_q, q_inv, np.array([1.0, 0, 0, 0]))
        return self.d.xmat[root].reshape(3, 3).T @ (-p), rel_q

    @property
    def frozen(self) -> bool:
        return "__anchor__" in self._welds

    def _freeze(self) -> None:
        """Pause the walking policy and hold the base where it is, legs in their standing pose."""
        if self.frozen or mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_EQUALITY, self.prefix + "anybot_anchor") < 0:
            return
        with self._lock:
            rel_pos, rel_quat = self._anchor_relpose()
            self._welds["__anchor__"] = (self.prefix + "anybot_anchor", rel_pos, rel_quat)
            self._apply_welds()
            self.gait.active = False
            stand = {a: float(q) for a, q in zip(self.gait.acts, self.gait.DEFAULT)}
        self._move_ctrl(stand, 0.4, settle=0.1)

    def _unfreeze(self) -> None:
        if not self.frozen:
            return
        with self._lock:
            eq = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_EQUALITY, self.prefix + "anybot_anchor")
            self.d.eq_active[eq] = 0
            del self._welds["__anchor__"]
            self.gait.restart()

    # -- holonomic bases (slide x / slide y / yaw joints)

    def _planar_move(self, x: float, y: float, yaw: float | None, speed: float = 0.45) -> float:
        """Glide a holonomic base to (x, y, yaw) with a smooth profile; returns the time it took."""
        jx, jy, jt = (self._jid(n) for n in self.profile.planar_base)
        pos, now_yaw = self._base_pose()
        if yaw is None:
            yaw = float(np.arctan2(y - pos[1], x - pos[0])) if np.hypot(x - pos[0], y - pos[1]) > 0.05 else now_yaw
        dyaw = float((yaw - now_yaw + np.pi) % (2 * np.pi) - np.pi)
        with self._lock:
            q = {j: float(self.d.qpos[self.m.jnt_qposadr[j]]) for j in (jx, jy, jt)}
            # The slide axes in the world (the robot may have been placed rotated in a scene).
            A = np.array([self.d.xaxis[jx][:2], self.d.xaxis[jy][:2]]).T
            spin = float(np.sign(self.d.xaxis[jt][2])) or 1.0
        dq = np.linalg.solve(A, np.array([x - pos[0], y - pos[1]]))
        dist = float(np.hypot(x - pos[0], y - pos[1]))
        duration = max(0.6, dist / speed, abs(dyaw) / 0.9)
        self.target = [float(x), float(y), 0.0]
        self.navigating = True
        try:
            self._move_ctrl({self.pos_act[jx]: q[jx] + float(dq[0]), self.pos_act[jy]: q[jy] + float(dq[1]),
                             self.pos_act[jt]: q[jt] + spin * dyaw}, duration, settle=0.4)
        finally:
            self.navigating, self.target = False, None
        return duration

    def _approach_base(self, target: np.ndarray, ignore: tuple[int, ...] = (), standoff: float | None = None, box=None) -> None:
        """Walk to a free spot next to `target`, facing it. No-op for fixed robots or when already there."""
        if not self._can_walk():
            return
        if standoff is None:  # how far in front of the base the arm works best
            with self._lock:
                root = self._bid(self.profile.root_body)
                arm = next(iter(self.arms.values()))
                shoulder = self.d.xpos[self.m.jnt_bodyid[arm.joints[0]]]
                Rb = self.d.xmat[root].reshape(3, 3)
                ahead = float((Rb.T @ (shoulder - self.d.xpos[root]))[0])
                # Front edge of the base itself (wheels, chassis), so a wide base does not hit the table.
                arm_bodies = {b for a in self.arms.values() for b in self._subtree(int(self.m.jnt_bodyid[a.joints[0]]))}
                base_geoms = [g for g in body_geoms(self.m, self._subtree(root)) if int(self.m.geom_bodyid[g]) not in arm_bodies
                              and self.d.geom_xpos[g][2] < 0.6]
                front = 0.0
                if base_geoms:
                    lo, hi = world_aabb(self.m, self.d, base_geoms)
                    corners = np.array([[x, y, 0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])])
                    front = float(np.max((corners - [self.d.xpos[root][0], self.d.xpos[root][1], 0]) @ Rb[:, 0]))
            legged = any(limb.kind == "leg" for limb in self.profile.limbs)
            # A low wheeled base can roll under a table top: only what stands at its own height
            # (legs, crates) blocks it, so it can park close and the arm works within easy reach.
            standoff = float(np.clip(ahead + 0.33 if legged else ahead + 0.42, 0.3, 0.9))
        probe_z = 3.0
        if not any(limb.kind == "leg" for limb in self.profile.limbs):
            with self._lock:
                root = self._bid(self.profile.root_body)
                arm_bodies = {b for a in self.arms.values() for b in self._subtree(int(self.m.jnt_bodyid[a.joints[0]]))}
                base_geoms = [g for g in body_geoms(self.m, self._subtree(root)) if int(self.m.geom_bodyid[g]) not in arm_bodies]
                if base_geoms:
                    probe_z = float(world_aabb(self.m, self.d, base_geoms)[1][2]) + 0.05
        pos, yaw = self._base_pose()
        delta = np.asarray(target[:2]) - pos[:2]
        facing = abs((np.arctan2(delta[1], delta[0]) - yaw + np.pi) % (2 * np.pi) - np.pi)
        if standoff - 0.09 < np.linalg.norm(delta) < standoff + 0.07 and facing < 0.25:
            return
        # A table is worked from one of its sides, not its corner: prefer headings square to its edges.
        edge_yaw = long_yaw = None
        with self._lock:
            gid = np.zeros(1, np.int32)
            z = 2.5
            for _ in range(5):  # look through objects lying on it, down to the static surface
                hit = mujoco.mj_ray(self.m, self.d, np.array([target[0], target[1], z]), np.array([0, 0, -1.0]),
                                    None, 1, -1, gid)
                if hit < 0 or gid[0] < 0:
                    break
                if self.m.body_weldid[self.m.geom_bodyid[gid[0]]] != 0:
                    z -= hit + 0.002
                    continue
                if z - hit > 0.1:  # a raised surface (table, shelf, conveyor), not the floor
                    Rg = self.d.geom_xmat[gid[0]].reshape(3, 3)
                    edge_yaw = float(np.arctan2(Rg[1, 0], Rg[0, 0]))
                    sz = self.m.geom_size[gid[0]]
                    if self.m.geom_type[gid[0]] == mujoco.mjtGeom.mjGEOM_BOX and abs(sz[0] - sz[1]) > 0.05:
                        long_yaw = edge_yaw if sz[0] > sz[1] else edge_yaw + np.pi / 2  # the table's long axis
                break
        self._planned_lean = self._planned_tilt = 0.0
        best = self._plan_stand_spot(target, standoff, edge_yaw, long_yaw, pos, box, ignore, probe_z)
        if best is not None:
            best = (0.0, best[0], best[1], 0.0)
        # No map (an empty world): probe around the target with rays.
        # The best spot is at the arm's comfortable distance; if something big is right there (a robot
        # carrying the tote it should reach into, a crate on the floor), step back and reach further.
        for extra, k in (() if best is not None else ((e, k) for e in (0.0, 0.12, 0.24) for k in range(16))):
            if best is not None and extra > best[3]:
                break
            ang = 2 * np.pi * k / 16  # heading the robot would have, looking at the target
            fwd, side = np.array([np.cos(ang), np.sin(ang)]), np.array([-np.sin(ang), np.cos(ang)])
            p = np.asarray(target[:2]) - (standoff + extra) * fwd
            probes = [p, p + 0.16 * side, p - 0.16 * side, p - 0.18 * fwd, p + 0.1 * fwd]
            if probe_z < 3.0:  # wheeled base: probe its whole footprint, at its own height
                with self._lock:
                    half = max(0.2, float(np.max(np.abs(world_aabb(self.m, self.d, base_geoms)[1][:2] - self.d.xpos[root][:2]))))
                probes = [p + u * fwd * half + v * side * half for u in (-1, 0, 1) for v in (-1, 0, 1)]
            if any(self.support_height(q[0], q[1], ignore, from_z=probe_z) > 0.05 for q in probes):
                continue  # something stands there (table, shelf, box)
            cost = float(np.linalg.norm(p - pos[:2]))
            if edge_yaw is not None:
                off = abs((ang - edge_yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4)  # 0 when square to an edge
                cost += 1.5 * off
            if long_yaw is not None and abs(np.cos(ang - long_yaw)) > 0.7:
                cost += 1.2  # from the short end the legs keep the base away: work from the long side
            if best is None or cost < best[0]:
                best = (cost, p, ang, extra)
        if best is None:
            raise SkillError("No free place to stand next to that spot.")
        _, p, ang, _ = best
        self._back_off()
        if hasattr(self.gait, "hold_here"):
            self._walk_pose()  # clear of the table now: arms down by the sides for walking
        else:
            self._tuck_arms()
        # Go (around walls, through doors) to a point in front of the spot first, then dock straight in:
        # the last steps must face the table, and a learned gait overshoots if it walks straight at it.
        pre = p - 0.35 * np.array([np.cos(ang), np.sin(ang)])
        if np.linalg.norm(pre - pos[:2]) > 0.3:  # arrive already facing the table: the same last steps every time
            self.navigate_to(float(pre[0]), float(pre[1]), tolerance=0.15, yaw=float(ang))
        if getattr(self, "_folded", None):  # unfold before going in: a folded hand hangs below table tops
            t = np.asarray(target, float)
            t3 = t if len(t) == 3 else np.array([t[0], t[1], self.support_height(t[0], t[1]) + 0.05])
            for a in self.arms.values():
                self._unfold(a, t3)
        if hasattr(self.gait, "hold_here"):
            if not self.held:  # going to pick: hands up for the last steps, or they end up under the table top
                self._tuck_arms()  # (carrying: the free hand stays down; raising it would knock the load)
        self._docking = True
        try:
            out = self.navigate_to(float(p[0]), float(p[1]), tolerance=0.06, yaw=float(ang))
        finally:
            self._docking = False
        if not out["reached"] and out["distance_left_m"] > 0.12:
            raise SkillError("Could not get close enough to work there.")  # (a few cm short: the arm makes up for it)
        self._docked = True
        self.wait(0.3)

    def _base_radius(self) -> float:
        """Half the size of the robot's base (legs, wheels, chassis: not the arms) — what must fit at a spot."""
        if getattr(self, "_base_r", None) is None:
            with self._lock:
                root = self._bid(self.profile.root_body)
                arm_bodies = {b for a in self.arms.values() for b in self._subtree(int(self.m.jnt_bodyid[a.joints[0]]))}
                geoms = [g for g in body_geoms(self.m, self._subtree(root)) if int(self.m.geom_bodyid[g]) not in arm_bodies
                         and self.d.geom_xpos[g][2] < 0.6 and (self.m.geom_contype[g] or self.m.geom_conaffinity[g])]
                r = 0.25
                if geoms:  # the base's half width (it faces what it works at, so its width is what must fit)
                    R = self.d.xmat[root].reshape(3, 3)
                    pts = []
                    for g in geoms:
                        c, h = self.m.geom_aabb[g, :3], self.m.geom_aabb[g, 3:]
                        Rg = self.d.geom_xmat[g].reshape(3, 3)
                        for sx in (-1, 1):
                            for sy in (-1, 1):
                                for sz in (-1, 1):
                                    pts.append(R.T @ (self.d.geom_xpos[g] + Rg @ (c + h * [sx, sy, sz]) - self.d.xpos[root]))
                    pts = np.array(pts)
                    boxy = float(min(np.max(np.abs(pts[:, 0])), np.max(np.abs(pts[:, 1]))))
                    # Round bases (discs): a bounding circle per geom is tighter than rotated boxes' corners.
                    round_ = max(float(np.linalg.norm((R.T @ (self.d.geom_xpos[g] - self.d.xpos[root]))[:2])
                                       + np.max(self.m.geom_aabb[g, 3:])) for g in geoms)
                    r = min(boxy, round_)
            self._base_r = float(np.clip(r, 0.15, 0.6))
        return self._base_r

    def _angled_pick(self, a: Arm, name: str, body: int, pos, lo, hi) -> dict:
        """Reach in at an angle (gripper tilted away from the robot), for objects too far for a top-down grasp:
        back along the approach, in, close (fingers across the approach), lift."""
        base, _ = self._base_pose()
        self._down_dir = self._tilted_down(base[:2], pos, self._planned_tilt)
        h = self._down_dir[:2] / (np.linalg.norm(self._down_dir[:2]) or 1.0)
        lateral = np.array([-h[1], h[0], 0.0])  # the fingers close sideways, never into the approach...
        with self._lock:
            Rb = self.d.xmat[body].reshape(3, 3)
        faces = [np.array([Rb[0, i], Rb[1, i], 0.0]) for i in range(2)]
        faces = [f / np.linalg.norm(f) for f in faces if np.linalg.norm(f) > 0.3]
        # ...and on two opposite faces of the object, not its corners: the face pair nearest to sideways.
        align = max(faces, key=lambda f: abs(float(f @ lateral))) if faces else lateral
        if align @ lateral < 0:
            align = -align
        top = float(hi[2])
        # A little lower than for a top-down grasp: tilted fingers must still reach past the middle of its sides.
        grasp = np.array([pos[0], pos[1], max(float(pos[2]) - 0.01, float(lo[2]) + self._fingertip_depth(a) + 0.006)])
        try:
            self.gripper("close", a.name)
            for c in (0.10, 0.06, 0.03):
                try:
                    self._reach(a, list(self._hover_point(grasp, top, c)), 1.3, align=align)
                    break
                except SkillError:
                    continue
            self.gripper("open", a.name)
            self._reach(a, list(grasp), 1.0, align=align)
            self.gripper("close", a.name)
            self.wait(0.3)
            self._reach(a, list(grasp + [0, 0, 0.12]), 1.0, align=align)
        finally:
            self._down_dir = None
        with self._lock:
            now = self.d.xpos[body].copy()
        if now[2] - pos[2] < 0.04:
            self.gripper("open", a.name)
            raise SkillError(f"Grasp of '{name}' failed: it did not lift.")
        self.held[a.name] = name
        if self._can_walk():
            self._carry_pose(a)
        self._touch_ok = ()
        return {"arm": a.name, "picked": name, "grasp": "gripper (angled)", "object_at": _r(now)}

    def _tilted_down(self, base_xy, point, tilt: float) -> np.ndarray:
        """Approach direction for an angled grasp: down and away from the robot, `tilt` from vertical."""
        h = np.asarray(point[:2], float) - np.asarray(base_xy[:2], float)
        h = h / (np.linalg.norm(h) or 1.0)
        v = np.array([np.sin(tilt) * h[0], np.sin(tilt) * h[1], -np.cos(tilt)])
        return v / np.linalg.norm(v)

    def _hover_point(self, point, top: float, clearance: float) -> np.ndarray:
        """Where the tool waits before going in: above the object, or back along an angled approach."""
        down = getattr(self, "_down_dir", None)
        if down is None:
            return np.array([point[0], point[1], top + clearance])
        p = np.array([point[0], point[1], top])
        return p - (clearance + 0.03) / max(-float(down[2]), 0.3) * np.asarray(down)

    def _moving_obstacle(self, xy, ignore=()) -> bool:
        """Is there something the static map does not know at (x, y): another robot, an object on the floor?"""
        if getattr(self, "_own_bodies", None) is None:
            self._own_bodies = set(self._subtree(self._bid(self.profile.root_body)))
        m = self.m
        z = 2.5
        gid = np.zeros(1, np.int32)
        with self._lock:
            for _ in range(6):
                hit = mujoco.mj_ray(m, self.d, np.array([xy[0], xy[1], z]), np.array([0, 0, -1.0]), None, 1, -1, gid)
                if hit < 0 or gid[0] < 0:
                    return False
                b = int(m.geom_bodyid[gid[0]])
                if m.body_weldid[b] != 0 and b not in self._own_bodies and b not in ignore and z - hit > 0.05:
                    return True  # a moving body (robot part, free object) standing there
                z -= hit + 0.002
        return False

    def _waist_pitch(self):
        """(joint, actuator) of a humanoid's waist pitch, if it has one it can drive."""
        for j in range(self.m.njnt):
            if self.m.joint(j).name.endswith("waist_pitch_joint") and j in self.pos_act:
                return j, self.pos_act[j]
        return None

    def _virtual_base(self, xy: np.ndarray, yaw: float) -> mujoco.MjData:
        """A copy of the state with the base moved to (xy, yaw): for planning where to stand."""
        m = self.m
        with self._lock:
            d = copy.copy(self.d)
            pos, now_yaw = self._base_pose()
            dyaw = float(yaw - now_yaw)
            if self.profile.planar_base:
                jx, jy, jt = (self._jid(n) for n in self.profile.planar_base)
                A = np.array([self.d.xaxis[jx][:2], self.d.xaxis[jy][:2]]).T
                dq = np.linalg.solve(A, np.asarray(xy) - pos[:2])
                d.qpos[m.jnt_qposadr[jx]] += dq[0]
                d.qpos[m.jnt_qposadr[jy]] += dq[1]
                d.qpos[m.jnt_qposadr[jt]] += (float(np.sign(self.d.xaxis[jt][2])) or 1.0) * dyaw
            elif self.profile.floating_base:
                root = self._bid(self.profile.root_body)
                adr = m.jnt_qposadr[m.body_jntadr[root]]
                d.qpos[adr:adr + 2] = xy
                qz, q = np.zeros(4), np.zeros(4)
                mujoco.mju_axisAngle2Quat(qz, np.array([0.0, 0.0, 1.0]), dyaw)
                mujoco.mju_mulQuat(q, qz, self.d.qpos[adr + 3:adr + 7])
                d.qpos[adr + 3:adr + 7] = q
        for name in getattr(self, "_folded", ()):  # plan from the working pose, not the folded travel pose
            for j in self.arms[name].joints:
                if j in self.pos_act:
                    d.qpos[m.jnt_qposadr[j]] = float(self._home_cmd[self.pos_act[j]])
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        return d

    def _reach_error(self, xy: np.ndarray, yaw: float, point: np.ndarray, box=None) -> float:
        """How far the arm(s) would miss the grasp at `point` if the base stood at (xy, yaw): 0 = it reaches.
        The very poses of the grasp are checked: for two hands, open beside the object and closed on it."""
        d = self._virtual_base(xy, yaw)
        left = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        lo, hi = box if box is not None else (np.asarray(point) - 0.035, np.asarray(point) + 0.035)
        if self._two_hands_for(lo, hi) and float(lo[2]) >= 0.4:
            l, r = sorted(self.arms.values(), key=lambda a: -float((d.xpos[self.m.jnt_bodyid[a.joints[0]]][:2] - xy) @ left[:2]))
            half = (np.asarray(hi) - np.asarray(lo)) / 2
            hl = float(abs(left[0]) * half[0] + abs(left[1]) * half[1])
            c = (np.asarray(lo) + np.asarray(hi)) / 2
            c[2] = max(c[2], lo[2] + 0.035)
            waist = self._waist_pitch()
            best = np.inf
            for lean in ((0.0, 0.3) if waist is not None else (0.0,)):  # a little forward lean reaches further
                if lean:
                    d.qpos[self.m.jnt_qposadr[waist[0]]] = lean
                    mujoco.mj_kinematics(self.m, d)
                    mujoco.mj_comPos(self.m, d)
                err = 0.0
                for gap, dz in ((0.10, 0.12), (0.005, 0.0)):  # open above the sides, then closed on it
                    el = self.solve_ik(l, c + (hl + gap) * left + [0, 0, dz], iters=150, base_d=d)[1]
                    er = self.solve_ik(r, c - (hl + gap) * left + [0, 0, dz], iters=150, base_d=d)[1]
                    err = max(err, el, er)
                    if err > 0.03:
                        break
                if err < best:
                    best, self._last_lean = err, lean
                if err < 0.02:
                    break
            return best
        free = [a for a in self.arms.values() if a.name not in self.held] or list(self.arms.values())
        down = not self.held and any(a.gripper_actuators for a in free)
        top = float(hi[2])
        best = np.inf
        self._last_tilt = 0.0
        tilts = (0.0, np.radians(20), np.radians(28)) if down and getattr(self, "_allow_tilt", True) else (0.0,)
        for tilt in tilts:  # straight down first, else angled in
            self._down_dir = None if not tilt else self._tilted_down(xy, point, tilt)
            try:
                for a in free:  # the poses of the grasp: a hover (as the grasp tries them), then at the object
                    hover = np.inf
                    for c in (0.10, 0.06, 0.03):
                        hover = min(hover, self.solve_ik(a, self._hover_point(point, top, c), point_down=down,
                                                         iters=150, base_d=d)[1])
                        if hover < 0.02:
                            break
                    if hover > 0.04:
                        err = hover
                    else:
                        grasp = self.solve_ik(a, np.array([point[0], point[1], (top + float(np.asarray(point)[2])) / 2]),
                                              point_down=down, iters=150, base_d=d)[1]
                        err = max(hover, grasp)
                    if err < best:
                        best, self._last_tilt = err, tilt
            finally:
                self._down_dir = None
            if best < 0.02 or best > 0.15:  # reached, or hopelessly far: angling the gripper will not help
                break
        return best

    def _plan_stand_spot(self, target, standoff: float, edge_yaw, long_yaw, pos, box=None, ignore=(),
                         probe_z: float = 3.0) -> tuple[np.ndarray, float] | None:
        """Where to stand to work at `target`, chosen on the world's collision map: the base fits there and
        on the way in, and — checked with the arm's own IK as if the robot stood there — it reaches.
        None when the world has no map (then the caller probes with rays)."""
        planner = self.scene.navmap()
        if planner is None:
            return None
        target = np.asarray(target, float)
        if len(target) == 2:
            target = np.array([target[0], target[1], self.support_height(target[0], target[1]) + 0.08])
        rb = self._base_radius()
        legged = any(limb.kind == "leg" for limb in self.profile.limbs)
        need = int(np.ceil((rb + (0.01 if legged else 0.05)) / planner.cell))  # wheels cannot shuffle in
        avoid = getattr(self, "_avoid_spots", [])
        cands = []
        for dist in np.arange(max(rb + 0.05, standoff - 0.2), standoff + 0.45, 0.05):
            for k in range(32):
                ang = 2 * np.pi * k / 32
                fwd = np.array([np.cos(ang), np.sin(ang)])
                p = target[:2] - dist * fwd
                pre = p - 0.35 * fwd
                c = planner._cell(p)
                if planner.clear[c] < need or planner.clear[planner._cell(pre)] < need:
                    continue
                if not planner._line_ok(planner._cell(pre), c, need):  # the last straight steps in are free too
                    continue
                if any(np.linalg.norm(p - a) < 0.25 for a in avoid):
                    continue
                # Things the static map does not know: other robots, objects on the floor. Probe the footprint.
                side = np.array([-fwd[1], fwd[0]])
                probes = [p + u * fwd * rb + v * side * rb for u in (-1, 0, 1) for v in (-1, 0, 1)]
                if any(self._moving_obstacle(q, ignore) for q in probes):
                    continue
                cost = float(np.linalg.norm(p - pos[:2])) + 1.5 * abs(dist - standoff)
                if edge_yaw is not None:
                    cost += 1.5 * abs((ang - edge_yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4)
                if long_yaw is not None and abs(np.cos(ang - long_yaw)) > 0.7:
                    cost += 1.2
                cands.append((cost, p, ang))
        if not cands:
            raise SkillError("No free place to stand next to that spot.")
        cands.sort(key=lambda c: c[0])
        lo_box = box[0] if box is not None else target - 0.035
        if hasattr(self.gait, "hold_here") and float(lo_box[2]) < 0.4:
            # Picking up or putting down low with a crouch: the IK check does not model crouching; the proven distance does.
            near = [c for c in cands if abs(np.linalg.norm(c[1] - target[:2]) - standoff) < 0.06] or cands
            return near[0][1], near[0][2]
        for allow_tilt in (False, True):  # a straight top-down grasp from any spot beats an angled one
            self._allow_tilt = allow_tilt
            try:
                found = self._first_reachable(cands, target, box)
            finally:
                self._allow_tilt = True
            if found is not None:
                return found
        # Nowhere it can both stand and reach: say so now instead of trying and fumbling.
        raise SkillError("Cannot reach it from anywhere around: furniture (or other robots) keep the robot too far away.")

    def _first_reachable(self, cands, target, box):
        tried = 0
        for cost, p, ang in cands:  # the cheapest spot from which the arm really reaches
            if tried >= 10:
                break
            if any(np.linalg.norm(p - q) < 0.08 and abs(ang - b) < 0.3 for q, b in getattr(self, "_seen_spots", [])):
                continue
            tried += 1
            self._last_lean = self._last_tilt = 0.0
            if self._reach_error(p, ang, target, box) < 0.02:
                lean, tilt = self._last_lean, self._last_tilt
                # Robust to docking a few cm short (the last steps meet the furniture's edge): it must still reach.
                short = p - 0.06 * np.array([np.cos(ang), np.sin(ang)])
                if self._reach_error(short, ang, target, box) < 0.025:
                    self._planned_lean, self._planned_tilt = max(lean, self._last_lean), max(tilt, self._last_tilt)
                    return p, ang
        return None

    def _back_off(self, distance: float = 0.4) -> None:
        """Leaving a table: step straight back first, so turning around does not sweep what is on it."""
        if not getattr(self, "_docked", False):
            return
        self._docked = False
        pos, yaw = self._base_pose()
        if self.profile.planar_base:
            self._planar_move(pos[0] - distance * np.cos(yaw), pos[1] - distance * np.sin(yaw), yaw, speed=0.3)
            return
        plan = self._check_mobile()
        self._start_locomotion()
        if hasattr(self.gait, "hold_here"):  # the policy side-steps / backs up best through its hold mode
            with self._lock:
                self.gait.hold_here(float(pos[0] - distance * np.cos(yaw)), float(pos[1] - distance * np.sin(yaw)), yaw)
            self.wait(3.0)
        else:
            self._set_base_velocity(-0.15, 0.0, plan)
            self.wait(distance / 0.15)
        self._stop_locomotion(plan)

    def _unfold(self, a: Arm, toward=None) -> None:
        """Out of the travel pose before working: up and over to high above the work spot, in the arm
        configuration grasps are planned in (IK seeded from the ready pose), touching nothing on the way."""
        if a.name not in getattr(self, "_folded", set()):
            return
        self._folded.discard(a.name)
        ready = np.array([float(self._home_cmd[self.pos_act[j]]) if j in self.pos_act else 0.0 for j in a.joints])
        joints = {self.pos_act[j]: None for j in a.joints if j in self.pos_act}
        if toward is not None:
            pos, yaw = self._base_pose()
            fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            # A joint move (no Cartesian path from the folded pose) to high above the robot's own base: the hand
            # rises over the base, never towards the table; the grasp then comes down from there.
            for z in (float(toward[2]) + 0.25, float(toward[2]) + 0.15):
                for f in (0.2, 0.3, 0.1):
                    goal = np.array([pos[0], pos[1], z]) + f * fwd
                    q, err = self.solve_ik(a, goal, point_down=True, seed=ready)
                    if err < 0.03:
                        self._move_ctrl({self.pos_act[j]: float(v) for j, v in zip(a.joints, q) if j in self.pos_act}, 1.4, settle=0.2)
                        return
        self._move_ctrl({u: float(self._home_cmd[u]) for u in joints}, 1.4, settle=0.2)

    def _find_rest_pose(self) -> dict[int, float]:
        """Joint → angle of the model's folded travel pose ("retract", "rest", "stow"...), if it has one."""
        from anybot.team import REST_KEYS

        if self._home is not None:
            return dict(self._home.get("rest", {}))
        m = self.m
        arm_joints = {j for a in self.arms.values() for j in a.joints}
        for k in range(m.nkey):
            if m.key(k).name.lower() in REST_KEYS:
                return {j: float(m.key_qpos[k][m.jnt_qposadr[j]]) for j in arm_joints}
        return {}

    def _tuck_arms(self) -> None:
        """Before the base moves, pull free arms in and up so they cannot sweep across a table top."""
        pos, yaw = self._base_pose()
        fwd, left = np.array([np.cos(yaw), np.sin(yaw), 0]), np.array([-np.sin(yaw), np.cos(yaw), 0])
        root = self._bid(self.profile.root_body)
        humanoid = hasattr(self.gait, "hold_here")
        for a in self.arms.values():
            if a.name in self.held and humanoid:
                continue
            if self._rest_q is None:
                self._rest_q = self._find_rest_pose()
            rest = {self.pos_act[j]: self._rest_q[j] for j in a.joints if j in self._rest_q and j in self.pos_act}
            if not humanoid and a.name not in self.held and len(rest) == len(a.joints):
                # Empty-handed: fold into the arm's own travel pose. Straight up first, so the hand does not
                # sweep across whatever it was working on.
                with self._lock:
                    tcp = self._tcp(a, self.d).copy()
                try:
                    self.move_end_effector(list(tcp + [0, 0, 0.12]), a.name, duration=0.7)
                except SkillError:
                    pass
                self._move_ctrl(rest, 1.6, settle=0.2)
                self._folded.add(a.name)
                continue
            with self._lock:
                shoulder = self.d.xpos[self.m.jnt_bodyid[a.joints[0]]].copy()
                side = np.sign((self.d.xmat[root].reshape(3, 3).T @ (self._tcp(a, self.d) - pos))[1]) or 1.0
            if humanoid:  # hands up by the chest
                goals = [pos + 0.1 * fwd + side * 0.22 * left + [0, 0, 0.3]]  # close to the chest: clear of a table edge
            else:  # gripper high and close above the base
                goals = [np.array([shoulder[0], shoulder[1], 0]) + 0.25 * fwd + [0, 0, shoulder[2] + h] for h in (0.45, 0.35, 0.25)]
            for g in goals:
                try:
                    if a.name in self.held:  # keep the gripper pointing down: a wrist flip would drop it
                        self._reach(a, list(g), 1.2)
                    else:
                        self.move_end_effector(list(g), a.name, duration=1.0)
                    break
                except SkillError:
                    continue

    def _walk_pose(self) -> None:
        """Humanoids: free arms back to the home pose (down by the sides) before walking. Arms left
        where the last manipulation put them unbalance the gait and look wrong."""
        if not hasattr(self.gait, "hold_here"):
            return
        targets = {}
        with self._lock:
            # Joints shared with an arm that carries something (e.g. the waist) stay as they are.
            busy = {j for a in self.arms.values() if a.name in self.held for j in a.joints}
            for a in self.arms.values():
                if a.name in self.held:
                    continue  # carrying: _carry_pose keeps it in front of the chest
                for j in a.joints:
                    if j in busy:
                        continue
                    act = self.pos_act.get(j)
                    if act is not None and abs(float(self.d.qpos[self.m.jnt_qposadr[j]]) - float(self._home_cmd[act])) > 0.08:
                        targets[act] = float(self._home_cmd[act])
                for act in a.gripper_actuators:
                    targets[act] = float(self._home_cmd[act])
        if targets:
            self._move_ctrl(targets, 1.0)

    def _carry_pose(self, a: Arm) -> None:
        """Hold the object up and in front of the chest for walking: lift straight up first, then pull in."""
        if not hasattr(self.gait, "hold_here"):  # wheeled bases: carry it high above the base
            self._tuck_arms()
            return
        pos, yaw = self._base_pose()
        fwd = np.array([np.cos(yaw), np.sin(yaw), 0])
        with self._lock:
            tcp = self._tcp(a, self.d).copy()
        try:
            self._reach(a, [tcp[0], tcp[1], max(tcp[2] + 0.1, pos[2] + 0.4)], 0.8)
            self._reach(a, list(pos + 0.22 * fwd + [0, 0, 0.42]), 1.0)  # high and close: clears table tops
        except SkillError:
            pass  # carrying as it is

    def _rest_arm(self, a: Arm) -> None:
        if self._can_walk():  # mobile robots: hands up and in; dropping them would sweep the table
            self._tuck_arms()
            return
        self._move_ctrl({self.pos_act[j]: float(self._home_cmd[self.pos_act[j]]) for j in a.joints if j in self.pos_act}, 1.2)

    # -- humanoids: hold things the way a person without fingers would, between both palms

    def _hand_sides(self) -> tuple[Arm, Arm, np.ndarray, np.ndarray, np.ndarray]:
        """(left arm, right arm, base position, forward, left) for a robot with two arms."""
        pos, yaw = self._base_pose()
        fwd, left = np.array([np.cos(yaw), np.sin(yaw), 0.0]), np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        with self._lock:
            side = {a.name: float((self.d.xpos[self.m.jnt_bodyid[a.joints[0]]] - pos) @ left) for a in self.arms.values()}
        l, r = sorted(self.arms.values(), key=lambda a: -side[a.name])
        return l, r, pos, fwd, left

    def _two_hands_for(self, lo: np.ndarray, hi: np.ndarray) -> bool:
        """A walking humanoid with two free, fingerless (or hand) arms holds boxes and mugs with both hands."""
        if not hasattr(self.gait, "hold_here") or len(self.arms) != 2 or self.held:
            return False
        if not all(a.suction or a.assisted for a in self.arms.values()):
            return False
        width = float(np.max(hi[:2] - lo[:2]))
        return 0.04 <= width <= 0.45

    def _dual_move(self, goals: dict, duration: float, seeds: dict | None = None) -> dict:
        """Move both hands together (one smooth joint-space motion) to their goals; returns the joint solutions."""
        targets, sol = {}, {}
        for an, p in goals.items():
            a = self.arms[an]
            q, err = self.solve_ik(a, np.asarray(p, float), seed=None if seeds is None else seeds.get(a.name))
            if err > 0.035:
                raise SkillError(f"The hands cannot reach there together (off by {err * 100:.0f} cm).")
            sol[a.name] = q
            targets.update({self.pos_act[j]: float(v) for j, v in zip(a.joints, q) if j in self.pos_act})
        self._move_ctrl(targets, duration, settle=0.15)
        return sol

    # -- humanoids: crouching to reach low (the floor, a low shelf)

    def _can_squat(self) -> bool:
        return self.frozen and all(hasattr(self.gait, k) for k in ("LEG_JOINTS", "DEFAULT", "acts", "joints", "qadr", "base_qadr"))

    def _pelvis_target(self, drop: float, pitch: float) -> tuple[np.ndarray, np.ndarray]:
        """Pelvis pose `drop` lower and pitched forward by `pitch` (hips back a little, as people do)."""
        p0, q0 = self._stand_pose
        yaw = float(np.arctan2(2 * (q0[0] * q0[3] + q0[1] * q0[2]), 1 - 2 * (q0[2] ** 2 + q0[3] ** 2)))
        fwd, left = np.array([np.cos(yaw), np.sin(yaw), 0.0]), np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        pos = p0 - np.array([0.0, 0.0, drop]) - 0.35 * drop * fwd
        dq = np.zeros(4)
        mujoco.mju_axisAngle2Quat(dq, left, pitch)
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, dq, q0)
        return pos, q

    def _leg_ik(self, pos: np.ndarray, quat: np.ndarray, seed: np.ndarray, iters: int = 120) -> tuple[np.ndarray, float]:
        """Leg joints that put the pelvis at (pos, quat) while both feet stay exactly where they stand now."""
        g, m = self.gait, self.m
        d = mujoco.MjData(m)
        with self._lock:
            d.qpos[:] = self.d.qpos
        feet = [int(m.jnt_bodyid[g.joints[5]]), int(m.jnt_bodyid[g.joints[11]])]
        mujoco.mj_kinematics(m, d)
        goal_p = {f: self._feet_pose[f][0] for f in feet}
        goal_R = {f: self._feet_pose[f][1] for f in feet}
        d.qpos[g.base_qadr:g.base_qadr + 3] = pos
        d.qpos[g.base_qadr + 3:g.base_qadr + 7] = quat
        lo = np.array([m.jnt_range[j, 0] for j in g.joints]); hi = np.array([m.jnt_range[j, 1] for j in g.joints])
        dadr = np.array([m.jnt_dofadr[j] for j in g.joints])
        q = np.clip(seed.copy(), lo, hi)
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        err = np.inf
        for _ in range(iters):
            d.qpos[g.qadr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            dq = np.zeros(12)
            err = 0.0
            for leg, f in enumerate(feet):
                cols = slice(6 * leg, 6 * leg + 6)
                e_p = goal_p[f] - d.xpos[f]
                R = d.xmat[f].reshape(3, 3)
                e_r = 0.5 * sum(np.cross(R[:, i], goal_R[f][:, i]) for i in range(3))
                err = max(err, float(np.linalg.norm(e_p)), 0.1 * float(np.linalg.norm(e_r)))
                mujoco.mj_jacBody(m, d, jp, jr, f)
                J = np.vstack([jp[:, dadr[cols]], jr[:, dadr[cols]]])
                e = np.concatenate([e_p, e_r])
                dq[cols] = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
            if err < 1e-4:
                break
            q = np.clip(q + np.clip(dq, -0.2, 0.2), lo, hi)
        return q, err

    def _crouch(self, drop: float, lean: float = 0.0, pitch: float = 0.0, duration: float = 1.8) -> float:
        """Go down by up to `drop` with the pelvis pitched forward by `pitch` and the waist by `lean`, feet
        planted (leg IK keeps them where they are); drop=0, pitch=0, lean=0 stands back up.
        Returns how far down it actually went (as far as the joints allow)."""
        if not self._can_squat():
            return 0.0
        g, m = self.gait, self.m
        root = self._bid(self.profile.root_body)
        if not getattr(self, "_crouched", 0.0) and not getattr(self, "_pitched", 0.0):
            with self._lock:  # remember standing: where the pelvis and the feet are
                self._stand_pose = (self.d.xpos[root].copy(), self.d.xquat[root].copy())
                feet = [int(m.jnt_bodyid[g.joints[5]]), int(m.jnt_bodyid[g.joints[11]])]
                self._feet_pose = {f: (self.d.xpos[f].copy(), self.d.xmat[f].reshape(3, 3).copy()) for f in feet}
        with self._lock:
            q_now = np.array([self.d.qpos[a] for a in g.qadr])
            start_legs = np.array([self.cmd[a] for a in g.acts])
        # The deepest pose the legs can take (target first, then shallower).
        got, legs_goal = 0.0, g.DEFAULT.copy()
        if drop > 0 or pitch > 0:
            for k in range(8):
                dz = drop * (1 - k / 8)
                legs, err = self._leg_ik(*self._pelvis_target(dz, pitch), seed=q_now)
                if err < 0.004:
                    got, legs_goal = dz, legs
                    break
        target_pitch = pitch if (got > 0 or drop == 0) else 0.0
        now_drop, now_pitch = float(getattr(self, "_crouched", 0.0)), float(getattr(self, "_pitched", 0.0))
        waist = next((j for j in range(m.njnt) if m.joint(j).name.endswith("waist_pitch_joint")), None)
        wa = self.pos_act.get(waist) if waist is not None else None
        start_lean = float(self.cmd[wa]) if wa is not None else 0.0
        name, _, _ = self._welds["__anchor__"]
        steps = max(1, int(duration / 0.02))
        # Leg poses along the way (IK at a few points), so the feet stay put the whole time.
        keys = [(0.0, start_legs)]
        for f in (0.25, 0.5, 0.75):
            u = _min_jerk(f)
            pq = self._pelvis_target(now_drop + u * (got - now_drop), now_pitch + u * (target_pitch - now_pitch))
            keys.append((f, self._leg_ik(*pq, seed=keys[-1][1])[0]))
        keys.append((1.0, legs_goal))
        for k in range(1, steps + 1):
            f = k / steps
            u = _min_jerk(f)
            i = min(int(f * 4), 3)
            fa, qa = keys[i]
            fb, qb = keys[i + 1]
            legs = qa + (f - fa) / max(fb - fa, 1e-6) * (qb - qa)
            pos, quat = self._pelvis_target(now_drop + u * (got - now_drop), now_pitch + u * (target_pitch - now_pitch))
            with self._lock:
                R = np.zeros(9)
                mujoco.mju_quat2Mat(R, quat)
                rel_pos = R.reshape(3, 3).T @ (-pos)
                q_inv, rel_q = np.zeros(4), np.zeros(4)
                mujoco.mju_negQuat(q_inv, quat)
                mujoco.mju_mulQuat(rel_q, q_inv, np.array([1.0, 0, 0, 0]))
                self._welds["__anchor__"] = (name, rel_pos, rel_q)
                self._apply_welds()
                for a, v in zip(g.acts, legs):
                    self.cmd[a] = float(v)
                if wa is not None:
                    self.cmd[wa] = start_lean + u * (lean - start_lean)
            self.wait(0.02)
        self._crouched, self._pitched = got, target_pitch
        self.wait(0.25)
        return got

    def _two_hand_pick(self, name: str, body: int, lo: np.ndarray, hi: np.ndarray) -> dict:
        """Hands open on both sides of the object, close in on it, lift together, carry it before the chest."""
        left, right, base, fwd, lvec = self._hand_sides()
        c, half = (lo + hi) / 2, (hi - lo) / 2
        hl = float(abs(lvec[0]) * half[0] + abs(lvec[1]) * half[1])  # half width across the hands
        low = float(lo[2]) < 0.4  # on the floor or a low shelf
        zc = float(max(c[2], lo[2] + 0.035, 0.045 if low else 0.0))
        sign = {left.name: 1.0, right.name: -1.0}

        def beside(gap: float, dz: float, centre=None) -> dict:
            cc = np.array([c[0], c[1], zc]) if centre is None else np.asarray(centre, float)
            return {a.name: cc + sign[a.name] * (hl + gap) * lvec + [0, 0, dz] for a in (left, right)}

        with self._lock:
            start = self.d.xpos[body].copy()
        if low:  # crouch: knees and hips bent, pelvis tilted forward, feet planted
            self._crouch(0.45, lean=0.52, pitch=0.45)
            left, right, base, fwd, lvec = self._hand_sides()
        elif getattr(self, "_planned_lean", 0.0) and self._waist_pitch():  # far across a table: lean in
            self._move_ctrl({self._waist_pitch()[1]: self._planned_lean}, 0.6, settle=0.1)
        # Adapt on the spot: a few cm short (the walk never stops exactly), lean in a little more and retry.
        waist = self._waist_pitch()
        leans = [float(self.cmd[waist[1]])] + ([0.3, 0.45] if waist is not None and not low else [])
        for i, lean in enumerate(leans):
            if i and lean <= leans[0]:
                continue
            if i:
                self._move_ctrl({waist[1]: lean}, 0.5, settle=0.1)
            try:
                s1 = self._dual_move(beside(0.10, 0.12), 1.2)  # open, above the sides
                s2 = self._dual_move(beside(0.10, 0.0), 0.7, s1)  # down beside it
                s3 = self._dual_move(beside(0.005, 0.0), 0.7, s2)  # palms close on it
                break
            except SkillError:
                if i == len(leans) - 1:
                    if waist is not None:
                        self._move_ctrl({waist[1]: 0.0}, 0.5, settle=0.1)
                    raise
        self._attach(left, name)  # the squeeze holds it (friction between bare palms is not simulated well)
        self.wait(0.15)
        try:
            s4 = self._dual_move(beside(-0.004, 0.12), 1.0, s3)  # lift together
        except SkillError:
            self._detach(left)
            raise
        with self._lock:
            now = self.d.xpos[body].copy()
        if now[2] - start[2] < 0.05:
            self._detach(left)
            raise SkillError(f"Grasp of '{name}' failed: it did not lift.")
        self.held[left.name] = self.held[right.name] = name
        self._two_hand = (left.name, right.name, name, hl)
        if not low and self._waist_pitch() and abs(float(self.cmd[self._waist_pitch()[1]])) > 1e-3:
            self._move_ctrl({self._waist_pitch()[1]: 0.0}, 0.6, settle=0.1)  # straighten up with it
        if low:  # stand up with it: the arms keep their pose, the object rises with the body
            self._crouch(0.0, lean=0.0, pitch=0.0)
            left, right, base, fwd, lvec = self._hand_sides()
            with self._lock:
                zc = float(self.d.xpos[body][2])
        # Carry it close in front of the chest, forearms level: the way people carry a box.
        chest = base + 0.27 * fwd
        chest[2] = float(base[2]) + 0.2 if low else max(zc + 0.12, float(base[2]) + 0.2)  # forearms level
        try:
            self._dual_move(beside(-0.004, 0.0, chest), 1.2, s4)
        except SkillError:
            pass  # carry it where it is
        self._touch_ok = ()
        return {"arm": f"{left.name} + {right.name}", "picked": name, "grasp": "two hands", "object_at": _r(now)}

    def _two_hand_place(self, x: float, y: float, on: str | None, name: str) -> dict:
        lname, rname, _, hl = self._two_hand
        body = self._bid(name)
        if on is not None:
            if on not in self.objects or on == name:
                raise SkillError(f"Cannot place on '{on}'.")
            with self._lock:
                b = self.d.xpos[self._bid(on)].copy()
            x, y, surface = float(b[0]), float(b[1]), self._object_top(on)
        else:
            surface = self.support_height(x, y, ignore=(body,))
        low = surface < 0.45
        # On the floor nothing is in the way: stand close, the hands only reach so far down.
        self._approach_base(np.array([x, y]), ignore=(body,), standoff=0.27 if low else None)
        left, right, base, fwd, lvec = self._hand_sides()
        lo, hi = self._object_box(name)
        with self._lock:
            hold_z = float(self.d.xpos[body][2])
        down = hold_z - float(lo[2])  # object centre above its bottom
        sign = {left.name: 1.0, right.name: -1.0}

        def at(gap: float, dz: float) -> dict:
            cc = np.array([x, y, surface + down + 0.006 + dz])
            return {a.name: cc + sign[a.name] * (hl + gap) * lvec for a in (left, right)}

        if low:  # the floor or a low shelf: crouch and lean forward, as a person would
            self._crouch(0.45, lean=0.52, pitch=0.45)
        release = 0.0
        if not low:
            s1 = self._dual_move(at(-0.004, 0.12), 1.3)  # over the spot
            s2 = self._dual_move(at(-0.004, 0.0), 0.9, s1)  # set it down
        else:
            # Lower it step by step, as far down as the arms reach from the crouch, then let go.
            s2, seeds = None, None
            for lift in (0.40, 0.30, 0.22, 0.15, 0.10, 0.06, 0.03, 0.0):
                try:
                    seeds = self._dual_move(at(-0.004, lift), 0.6, seeds)
                    s2, release = seeds, lift
                except SkillError:
                    if s2 is not None:
                        break
            if s2 is None:
                self._crouch(0.0, lean=0.0)
                raise SkillError("Cannot reach down there, even crouching.")
        self._detach(self.arms[lname])
        self.wait(0.2)
        try:
            s3 = self._dual_move(at(0.09, release), 0.7, s2)  # open the hands
            self._dual_move(at(0.09, release + 0.14), 0.7, s3)  # and lift them away
        except SkillError:
            pass
        for n in (lname, rname):
            self.held.pop(n, None)
        self._two_hand = None
        if low:
            self._walk_pose()  # arms in, then stand up
            self._crouch(0.0, lean=0.0)
        with self._lock:
            final = self.d.xpos[body].copy()
        return {"arm": f"{lname} + {rname}", "placed": name, "on": on, "object_at": _r(final), "grasp": "two hands",
                "error_m": round(float(np.hypot(final[0] - x, final[1] - y)), 3)}

    def _grasp_axis(self, a: Arm, body: int) -> np.ndarray | None:
        """Horizontal face normal of the object that the fingers should close along (the one nearest
        to how the gripper is turned now, so the wrist turns as little as possible)."""
        if a.grip_axis is None:
            return None
        with self._lock:
            Ro = self.d.xmat[body].reshape(3, 3)
            g = self.d.xmat[a.ee_body].reshape(3, 3) @ a.grip_axis
        cands = []
        for k in range(3):
            v = Ro[:, k].copy()
            v[2] = 0.0
            if np.linalg.norm(v) > 0.5:
                cands.append(v / np.linalg.norm(v))
        if not cands:
            return None
        g[2] = 0.0

        def room(v: np.ndarray) -> float:
            """Free space on both sides of the object along v (a finger must fit on each side)."""
            m, d = self.m, self.d
            lo, hi = self._object_box(m.body(body).name)
            centre = (lo + hi) / 2
            gid = np.zeros(1, dtype=np.int32)
            free = []
            for sgn in (1.0, -1.0):
                ray = np.array([v[0] * sgn, v[1] * sgn, 0.0])
                start = centre.copy()
                dist, hit = 0.0, -1.0
                for _ in range(6):  # step past the object itself
                    hit = mujoco.mj_ray(m, d, start + ray * dist, ray, None, 1, body, gid)
                    break
                free.append(hit if hit >= 0 else 1.0)
            return min(free)

        with self._lock:
            ranked = sorted(cands, key=lambda v: (-min(room(v), 0.12), -abs(float(v @ g))))
        return ranked[0]

    # -- pick and place

    def pick(self, name: str, arm: str | None = None) -> dict:
        """Pick an object up and lift it; verified by checking that the object really rises.

        Arms with a gripper grasp from above; arms without fingers hold it suction-style.
        A walking or driving robot first goes to stand next to the object. With two free arms and
        none requested, a failed reach is retried once with the other arm (as a person would).
        """
        self._avoid_spots = []
        first = None
        try:
            for attempt in range(3):
                try:
                    return self._pick_once(name, arm)
                except SkillError as e:
                    first = first or e
                    if name in self.held.values() or name not in self.objects or "No object" in str(e):
                        raise
                    # Recover like a person would: put the hands back, then try differently.
                    if attempt == 0 and arm is None and not self._can_walk():
                        free = [a for a in self.arms.values() if a.name not in self.held]
                        other = next((a for a in free if a.name != getattr(self, "_last_pick_arm", None)), None)
                        if other is not None and len(free) >= 2:
                            try:
                                return self._pick_once(name, other.name)
                            except SkillError:
                                raise first from None
                    if not self._can_walk():
                        raise first from None
                    pos, _ = self._base_pose()
                    self._avoid_spots.append(pos[:2].copy())  # this spot did not work: stand elsewhere
                    self._rest_after_miss()
            raise first
        finally:
            self._avoid_spots = []

    def _rest_after_miss(self) -> None:
        """After a failed grasp: let go of anything half-held, hands up and out, step back."""
        for a in self.arms.values():
            if a.name not in self.held:
                self._detach(a)
                if a.gripper_actuators:
                    self.gripper("open", a.name)
        self._tuck_arms()
        self._back_off()

    def _pick_once(self, name: str, arm: str | None = None) -> dict:
        if name not in self.objects:
            raise SkillError(f"No object '{name}'." + _suggest(name, self.objects))
        if self.object_meta[name]["fixed"]:
            raise SkillError(f"'{name}' is fixed to the world and cannot be picked.")
        if name in self.held.values():
            raise SkillError(f"Already holding '{name}'.")
        if not self.arms:
            raise SkillError(f"{self.name} ({self.profile.category}) has no arm to pick with.")
        body = self._bid(name)
        self._touch_ok = (body,)  # the object being picked may be touched; nothing else
        with self._lock:
            pos = self.d.xpos[body].copy()
        lo, hi = self._object_box(name)
        low_two_hands = arm is None and self._two_hands_for(lo, hi) and float(lo[2]) < 0.4
        # Picking up from the floor: stand close, the hands only reach so far down even crouching.
        self._approach_base(pos, ignore=(body,), standoff=0.27 if low_two_hands else None, box=(lo, hi))
        if arm is None and self._two_hands_for(lo, hi):
            return self._two_hand_pick(name, body, lo, hi)
        a = self._arm(arm) if arm else self._nearest_arm(pos, holding=False)
        self._last_pick_arm = a.name
        centre, top = (lo + hi) / 2, float(hi[2])
        if a.suction or a.assisted:
            if a.assisted:
                self.gripper("open", a.name)
            # Come in from well above so the hand does not sweep through neighbouring objects.
            clearance = self._approach(a, centre[0], centre[1], top, clearances=(0.16, 0.08, 0.04), duration=1.5)
            self._reach(a, [centre[0], centre[1], top + 0.015], 1.0)  # hover just above: no knocking it over
            with self._lock:
                gap = float(np.linalg.norm(self._tcp(a, self.d) - [centre[0], centre[1], top]))
            if gap > 0.06:
                raise SkillError(f"Could not touch '{name}' (missed by {gap * 100:.0f} cm).")
            self._attach(a, name)  # secure first: closing fingers would otherwise flick a small object away
            if a.assisted:
                self.gripper("close", a.name)  # then the fingers wrap onto it
            self.wait(0.2)
            self._lift(a, centre[0], centre[1], top, heights=(clearance + 0.04, clearance, 0.05), duration=1.2)
        else:
            if getattr(self, "_planned_tilt", 0.0) and not self.held:
                return self._angled_pick(a, name, body, pos, lo, hi)
            self.gripper("close", a.name)  # narrow while travelling, open only once above the object
            align = self._grasp_axis(a, body)
            clearance = self._approach(a, pos[0], pos[1], top, align=align)
            self.gripper("open", a.name)
            # Grasp at the object's centre, but never drive the fingertips into what it stands on.
            grasp_z = max(float(pos[2]), float(lo[2]) + self._fingertip_depth(a) + 0.006)
            self._reach(a, [pos[0], pos[1], grasp_z], 1.0, align=align)
            self.gripper("close", a.name)
            self.wait(0.3)
            self._lift(a, pos[0], pos[1], top, heights=(clearance + 0.03, clearance, 0.05, 0.03), duration=1.2)
        with self._lock:
            now = self.d.xpos[body].copy()
        if now[2] - pos[2] < 0.04:
            self._detach(a) if (a.suction or a.assisted) else self.gripper("open", a.name)
            raise SkillError(f"Grasp of '{name}' failed: it did not lift (it may be too big, too heavy or out of reach).")
        self.held[a.name] = name
        if self._can_walk():
            self._carry_pose(a)
        self._touch_ok = ()
        kind = "suction" if a.suction else "hand" if a.assisted else "gripper"
        return {"arm": a.name, "picked": name, "grasp": kind, "object_at": _r(now)}

    def place(self, x: float, y: float, on: str | None = None, arm: str | None = None) -> dict:
        """Put the held object down at (x, y) on whatever is there (floor, table), or on top of object `on`."""
        if not self.held:
            raise SkillError("Not holding anything. Pick an object first.")
        a = self._arm(arm) if arm else self._nearest_arm(np.array([x, y, 0.0]), holding=True)
        if a.name not in self.held:
            raise SkillError(f"Arm '{a.name}' is not holding anything.")
        name = self.held[a.name]
        if getattr(self, "_two_hand", None) and self._two_hand[2] == name:
            return self._two_hand_place(x, y, on, name)
        body = self._bid(name)
        if on is not None:
            if on not in self.objects or on == name:
                raise SkillError(f"Cannot place on '{on}'." + _suggest(on or "", [o for o in self.objects if o != name]))
            with self._lock:
                base = self.d.xpos[self._bid(on)].copy()
            x, y, surface = float(base[0]), float(base[1]), self._object_top(on)
        else:
            surface = self.support_height(x, y, ignore=(body,))
        self._approach_base(np.array([x, y]), ignore=(body,))
        lo, hi = self._object_box(name)
        with self._lock:
            tcp = self._tcp(a, self.d).copy()
            centre_xy = self.d.xpos[body][:2] - tcp[:2]  # where the object hangs relative to the tool
        tcp_above_bottom = float(tcp[2] - lo[2])
        z = surface + 0.004 + tcp_above_bottom
        held_by_weld = a.suction or a.assisted
        tx, ty = (x - centre_xy[0], y - centre_xy[1]) if held_by_weld else (x, y)
        # Lift first, then travel: the object under the hand must clear what is already on the table.
        try:
            safe = max(float(tcp[2]), z + 0.16 + float(hi[2] - lo[2]))
            self._reach(a, [tcp[0], tcp[1], safe], 0.8)
            self._reach(a, [tx, ty, safe], 1.2)
        except SkillError:
            pass
        clearance = self._approach(a, tx, ty, z, duration=1.2)
        if held_by_weld:
            # The hand may turn on the way and the object swings with it: re-aim from where it really hangs.
            last_miss = np.inf
            for _ in range(3):
                with self._lock:
                    miss = np.array([x, y]) - self.d.xpos[body][:2]
                if np.linalg.norm(miss) < 0.01 or np.linalg.norm(miss) > 0.8 * last_miss:
                    break  # close enough, or not getting closer (the hand is blocked)
                last_miss = float(np.linalg.norm(miss))
                tx, ty = tx + float(miss[0]), ty + float(miss[1])
                try:
                    self._reach(a, [tx, ty, z + clearance], 0.6)
                except SkillError:
                    break
        self._reach(a, [tx, ty, z], 1.0)
        if a.assisted:
            self.gripper("open", a.name)  # open while still secured, then let go: no flick
        if held_by_weld:
            self._detach(a)
        if not (a.suction or a.assisted):
            self.gripper("open", a.name)
        del self.held[a.name]
        self._lift(a, tx, ty, z, heights=(clearance, 0.05, 0.03))
        if self._can_walk():
            self._rest_arm(a)
        self.wait(0.3)
        with self._lock:
            final = self.d.xpos[body].copy()
        return {"arm": a.name, "placed": name, "on": on, "object_at": _r(final),
                "error_m": round(float(np.linalg.norm(final[:2] - [x, y])), 3)}

    def gripper(self, action: str, limb: str | None = None) -> dict:
        arm = self._arm(limb)
        if not arm.gripper_actuators:
            raise SkillError(f"Arm '{arm.name}' has no gripper.")
        if action not in ("open", "close"):
            raise SkillError("action must be 'open' or 'close'")
        values = arm.gripper_open if action == "open" else arm.gripper_closed
        self._move_ctrl(dict(zip(arm.gripper_actuators, values)), 0.8)
        return {"arm": arm.name, "gripper": action}

    def keyframes(self) -> list[str]:
        if self._home is not None:
            return ["home"]
        return [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_KEY, k) or f"key{k}" for k in range(self.m.nkey)]

    def skills(self) -> list[str]:
        """Skill tools that make sense for this robot."""
        out = ["get_state", "snapshot", "spawn_object", "reset", "wait"]
        if self.pos_act:
            out.append("set_joint_positions")
        if self.keyframes() or not self.profile.floating_base or self.pinned:
            out.append("go_to_pose")
        if self.arms:
            out.append("move_end_effector")
        if any(a.gripper_actuators for a in self.arms.values()):
            out.append("gripper")
        if self.arms and all(j in self.pos_act for a in self.arms.values() for j in a.joints):
            out += ["pick_object", "place_object"]
        if self._can_walk():
            out += ["drive", "navigate_to"]
        return out

    def _make_gait(self):
        """A trot controller for free-standing quadrupeds whose leg joints all accept angle commands."""
        legs = [limb for limb in self.profile.limbs if limb.kind == "leg"]
        if not self.profile.floating_base or self.pinned:
            return None
        if len(legs) == 2:  # bipeds need a learned balance+walking policy
            from anybot.policies import make_policy

            return make_policy(self)
        if len(legs) != 4:
            return None
        if any(self._jid(n) not in self.pos_act for leg in legs for n in leg.joints):
            return None
        from anybot.gait import TrotGait

        return TrotGait(self)

    def go_to_pose(self, name: str, duration: float = 2.0) -> dict:
        """Move actuated joints to a keyframe ('home', 'stand', ...) or 'zero'."""
        if name == "zero":
            if self.profile.floating_base and not self.pinned:
                raise SkillError("'zero' (all joints straight) would topple a free-standing robot; use a keyframe.")
            cmd = self.cmd.copy()
            for j, a in self.pos_act.items():
                cmd[a] = self.m.qpos0[self.m.jnt_qposadr[j]]
        else:
            keys = self.keyframes()
            if name not in keys:
                raise SkillError(f"Unknown pose '{name}'. Available: {', '.join(keys + ['zero'])}")
            cmd = self._cmd_for_key(keys.index(name))
        acts = {a: float(cmd[a]) for a in self._own_acts}
        self._move_ctrl(acts, duration)
        return self.state()

    def _wheel_plan(self) -> dict[int, tuple[float, float, float]]:
        """Velocity actuator → (lateral offset y, wheel radius, spin sign) for each drive wheel."""
        m, d = self.m, self.d
        root = self._bid(self.profile.root_body)
        Rb = d.xmat[root].reshape(3, 3)
        plan = {}
        for wname in self.profile.wheels:
            j = self._jid(wname)
            acts = [a for a in range(m.nu) if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT and m.actuator_trnid[a, 0] == j]
            if not acts:
                continue
            b = int(m.jnt_bodyid[j])
            axis_b = Rb.T @ (d.xmat[b].reshape(3, 3) @ m.jnt_axis[j])
            y = float((Rb.T @ (d.xpos[b] - d.xpos[root]))[1])
            geoms = body_geoms(m, [b])
            radius = float(max(m.geom_size[g, 0] for g in geoms)) if geoms else 0.05
            sign = float(np.cross(axis_b, [0, 0, 1])[0])
            if abs(sign) >= 0.3:  # skip casters / steering joints
                plan[acts[0]] = (y, radius, sign)
        return plan

    def _calibrate_base(self) -> dict | None:
        """Learn how non-arm actuators move a floating base, by trying each one in a scratch copy.

        Covers bases driven through tendons or virtual 'forward/turn' actuators (e.g. Stretch),
        where no per-wheel velocity actuator exists. Returns {"acts", "B"} with B mapping
        actuator commands to (forward m/s, yaw rad/s), or None.
        """
        if not self.profile.floating_base or self.pinned:
            return None
        m = self.m
        arm_acts = {a for arm in self.arms.values() for a in arm.gripper_actuators}
        arm_acts |= {self.pos_act[j] for arm in self.arms.values() for j in arm.joints if j in self.pos_act}
        root = self._bid(self.profile.root_body)
        cols, acts = [], []
        for a in self._own_acts:
            if a in arm_acts or a in self.pos_act.values():
                continue
            u = 0.5 * (float(m.actuator_ctrlrange[a, 1]) if m.actuator_ctrllimited[a] else 1.0)
            d = copy.copy(self.d)
            R0, p0 = d.xmat[root].reshape(3, 3).copy(), d.xpos[root].copy()
            yaw0 = _quat_to_rpy(d.xquat[root])[2]
            ctrl = self.cmd.copy()
            ctrl[a] = u
            for _ in range(int(1.0 / m.opt.timestep)):
                d.ctrl[:] = ctrl
                mujoco.mj_step(m, d)
            fwd = float((R0.T @ (d.xpos[root] - p0))[0])
            dyaw = float((_quat_to_rpy(d.xquat[root])[2] - yaw0 + np.pi) % (2 * np.pi) - np.pi)
            if abs(fwd) > 0.02 or abs(dyaw) > 0.05:
                cols.append([fwd / u, dyaw / u])
                acts.append(a)
        if not acts:
            return None
        return {"acts": acts, "B": np.array(cols).T}  # 2 x n: per unit command, motion over 1 s

    def _set_base_velocity(self, v: float, w: float, plan: dict | None = None) -> None:
        if self.gait is not None:
            self.gait.set_velocity(v, 0.0, w)
            return
        with self._lock:
            if plan and "B" in plan:
                u = np.linalg.pinv(plan["B"]) @ np.array([v, w])
                for a, val in zip(plan["acts"], u):
                    lo, hi = self.m.actuator_ctrlrange[a] if self.m.actuator_ctrllimited[a] else (-np.inf, np.inf)
                    self.cmd[a] = float(np.clip(val, lo, hi))
                return
            for a, (y, r, sign) in (plan or self._wheel_plan()).items():
                self.cmd[a] = (v - w * y) / (r * sign)

    @staticmethod
    def _lookahead(planner, xy, longest: float) -> float:
        """Pure-pursuit look-ahead: long in open space (smooth), short where it is tight (doors, corners)."""
        if planner is None:
            return longest
        room = float(planner.clear[planner._cell(xy)]) * planner.cell
        return float(np.clip(0.8 * room, 0.2, longest))

    def _replan(self, planner, goal, current=None, at=None) -> "navmap.Path":
        """A fresh plan, but keep following the current path unless the new one is clearly better (40 cm
        shorter) or the current one got blocked: re-deciding every second between two equally good ways
        around an obstacle makes a robot zigzag and turn back."""
        new = self._plan_path(planner, goal)
        if current is None or planner is None or len(current.points) < 2:
            return new
        here = np.asarray(at if at is not None else self._base_pose()[0][:2], float)
        s0 = current.project(here)
        remaining = current.length - s0
        if new.length < remaining - 0.4:
            return new
        need = max(2, int(np.ceil(0.6 * self._footprint_radius() / planner.cell)))
        pts = [current.at(s0)] + [p for p, c in zip(current.points, current.cum) if c > s0]
        for a, b in zip(pts[:-1], pts[1:]):
            if not planner._line_ok(planner._cell(a), planner._cell(b), need):
                return new  # something new is in the way: take the new route
        self.nav_path = [list(here)] + [list(p) for p in pts[1:]]
        return current

    def _plan_path(self, planner, goal) -> "navmap.Path":
        """Path from here to goal around the world's obstacles and what the lidar has seen on the way."""
        from anybot import navmap, sensors

        here, _ = self._base_pose()
        goal = [float(goal[0]), float(goal[1])]
        if planner is None:
            self.nav_path = [list(here[:2]), goal]
            return navmap.Path([here[:2], goal])
        radius = self._footprint_radius()
        with self._lock:
            self._nav_seen.append(sensors.lidar(self).points())
        del self._nav_seen[:-30]
        planner.mark(np.concatenate(self._nav_seen), here[:2], keep_out=radius + 0.15)
        try:
            route = planner.route(here[:2], goal, radius=radius, half_width=self._base_radius())
        except navmap.NoPath as e:
            raise SkillError(f"Cannot get to ({goal[0]:.2f}, {goal[1]:.2f}): {e}.") from None
        if np.linalg.norm(np.array(route[-1]) - goal) > 1e-6 and self.target is not None:
            self.target = [float(route[-1][0]), float(route[-1][1]), 0.0]  # that spot is occupied: nearest free one
        path = navmap.Path([list(here[:2])] + route)
        self.nav_path = path.points.tolist()  # for the sensor view
        return path

    def _planar_follow(self, x: float, y: float, yaw: float | None, planner, timeout: float = 40.0,
                       vmax: float = 0.45, amax: float = 0.5, wmax: float = 0.9) -> float:
        """Holonomic base: glide along the planned path with smooth acceleration, turning to face the way it
        goes (and to `yaw` at the end). The reference never stops at corners: pure pursuit rounds them."""
        jx, jy, jt = (self._jid(n) for n in self.profile.planar_base)
        ax, ay, at = self.pos_act[jx], self.pos_act[jy], self.pos_act[jt]
        with self._lock:
            pos0, yaw0 = self._base_pose()
            q0 = {j: float(self.d.qpos[self.m.jnt_qposadr[j]]) for j in (jx, jy, jt)}
            A = np.array([self.d.xaxis[jx][:2], self.d.xaxis[jy][:2]]).T
            spin = float(np.sign(self.d.xaxis[jt][2])) or 1.0
        goal = np.array([float(x), float(y)])
        ref, ref_yaw, v = pos0[:2].copy(), float(yaw0), 0.0
        prev_yaw = ref_yaw
        lookahead = max(0.4, self._footprint_radius())
        dt, t = 0.02, 0.0
        self.target, self.navigating = [float(x), float(y), 0.0], True
        try:
            path, planned_t = self._plan_path(planner, goal), 0.0
        except SkillError:
            self.target, self.navigating = None, False
            raise
        goal = np.array(self.target[:2])  # may have moved to the nearest free spot
        timeout = max(timeout, 3.0 * path.length / vmax + 10.0)
        try:
            while t < timeout:
                if self.target is not None and np.linalg.norm(np.array(self.target[:2]) - goal) > 1e-6:
                    goal = np.array(self.target[:2])  # retargeted (a new click)
                    planned_t = -1e9
                if t - planned_t >= 1.0:
                    path, planned_t = self._replan(planner, goal, path, at=ref), t
                left = float(np.linalg.norm(goal - ref))
                look = self._lookahead(planner, ref, lookahead)
                aim = goal if left < look else path.carrot(ref, look)
                to_aim = aim - ref
                v = min(v + amax * dt, vmax, np.sqrt(2 * amax * left))  # accelerate, cruise, brake to stop
                if np.linalg.norm(to_aim) > 1e-6:
                    ref = ref + to_aim / np.linalg.norm(to_aim) * min(v * dt, float(np.linalg.norm(to_aim)))
                travel = np.arctan2(to_aim[1], to_aim[0]) if left > 0.3 else None
                want = travel if travel is not None else (yaw if yaw is not None else ref_yaw)
                ref_yaw += float(np.clip((want - ref_yaw + np.pi) % (2 * np.pi) - np.pi, -wmax * dt, wmax * dt))
                dq = np.linalg.solve(A, ref - pos0[:2])
                # Velocity feed-forward: the servo's damping would otherwise make the base lag behind.
                vel = (to_aim / np.linalg.norm(to_aim) * v) if np.linalg.norm(to_aim) > 1e-6 else np.zeros(2)
                dq = dq + (2.0 / self.BASE_OMEGA) * np.linalg.solve(A, vel)
                yaw_rate = (ref_yaw - prev_yaw) / dt
                prev_yaw = ref_yaw
                self._ramp_to({ax: q0[jx] + float(dq[0]), ay: q0[jy] + float(dq[1]),
                               at: q0[jt] + spin * (ref_yaw - yaw0 + (2.0 / self.YAW_OMEGA) * yaw_rate)}, dt)
                self.wait(dt)
                t += dt
                yaw_done = yaw is None or abs((yaw - ref_yaw + np.pi) % (2 * np.pi) - np.pi) < 0.01
                if left < 1e-3 and yaw_done:
                    break
            self.wait(0.4)  # let the base settle on the reference
            self._nav_goal = (float(goal[0]), float(goal[1]))
        finally:
            self.navigating, self.target = False, None
        return t + 0.4

    def _footprint_radius(self) -> float:
        """Half the robot's widest horizontal extent (m): how far walls must stay from its base."""
        if getattr(self, "_radius", None) is None:
            from anybot.introspect import body_geoms, world_aabb

            root = self._bid(self.profile.root_body)
            lo, hi = world_aabb(self.m, self.d, body_geoms(self.m, self._subtree(root)))
            base = self.d.xpos[root]
            r = float(np.max(np.abs(np.concatenate([lo[:2] - base[:2], hi[:2] - base[:2]]))))
            if self.profile.category == "humanoid":
                r = max(r, 0.38)  # arms swing while walking
            self._radius = float(np.clip(r, 0.2, 0.6))
        return self._radius

    def _base_pose(self) -> tuple[np.ndarray, float]:
        with self._lock:
            root = self._bid(self.profile.root_body)
            pos = self.d.xpos[root].copy()
            yaw = float(_quat_to_rpy(self.d.xquat[root])[2])
        return pos, yaw

    def _check_mobile(self) -> dict | None:
        if self.gait is not None:
            return None
        if self.profile.planar_base:
            return {"planar": True}
        if not self.profile.wheels:
            raise SkillError(f"{self.name} ({self.profile.category}) cannot move its base: no wheels and no gait.")
        with self._lock:
            plan = self._wheel_plan()
        if not plan:
            if self._base_map is None:
                with self._lock:
                    self._base_map = self._calibrate_base() or {}
            plan = self._base_map or None
        if not plan:
            raise SkillError("Could not find any actuator that moves the base.")
        return plan

    def _start_locomotion(self) -> None:
        if getattr(self.gait, "keeps_balance", False):
            self._unfreeze()  # resume walking from a standstill
            return
        if self.gait is not None and not self.gait.active:
            with self._lock:
                self._traj = {}
                self.gait.phase, self.gait.vel[:] = 0.0, 0.0
                self.gait.set_velocity(0, 0, 0)
                for leg in getattr(self.gait, "legs", []):  # feet start from their standing spots
                    leg.__dict__.pop("foot", None)
                self.gait.active = True

    def _stop_locomotion(self, plan: dict | None) -> None:
        if getattr(self.gait, "keeps_balance", False):
            with self._lock:
                if self.gait.hold is None:
                    self.gait.hold_here()  # keep the spot it reached while it settles, then stand still
            self.wait(0.8)
            self._freeze()
            return
        self._set_base_velocity(0.0, 0.0, plan)
        if plan and "acts" in plan:
            with self._lock:
                for a in plan["acts"]:
                    self.cmd[a] = 0.0
        if getattr(self.gait, "keeps_balance", False):
            self.wait(0.8)  # decelerate, then stop stepping
            self._freeze()
            return
        if self.gait is not None:
            self.wait(self.gait.period * 1.5)  # let the velocity ramp down while still stepping
            with self._lock:
                self.gait.active = False
                stance = self.gait.stance_command()
            self._move_ctrl(stance, 0.3, settle=0.2)

    def drive(self, v: float, w: float = 0.0, seconds: float = 2.0) -> dict:
        """Move the base at forward speed v (m/s) and yaw rate w (rad/s): wheels or trot gait."""
        plan = self._check_mobile()
        start, yaw0 = self._base_pose()
        if plan and plan.get("planar"):
            yaw1 = yaw0 + w * seconds
            mid = yaw0 + w * seconds / 2  # straight-line approximation of the arc
            self._planar_move(start[0] + v * seconds * np.cos(mid), start[1] + v * seconds * np.sin(mid), yaw1,
                              speed=max(abs(v), 0.05))
            end, yaw = self._base_pose()
            return {"start": _r(start), "end": _r(end), "yaw_deg": round(float(np.degrees(yaw)), 1)}
        self._start_locomotion()
        self._set_base_velocity(v, w, plan)
        self.wait(seconds)
        self._stop_locomotion(plan)
        end, yaw = self._base_pose()
        return {"start": _r(start), "end": _r(end), "yaw_deg": round(float(np.degrees(yaw)), 1)}

    def navigate_to(self, x: float, y: float, tolerance: float = 0.12, timeout: float = 40.0,
                    yaw: float | None = None) -> dict:
        """Go to (x, y) on the floor: turn towards the goal and move forward, closed loop on the base pose.

        While it runs, `self.target` can be changed from another thread (e.g. a new click) to retarget.
        """
        plan = self._check_mobile()
        planner = None
        if hasattr(self.gait, "hold_here") and not getattr(self, "_routing", False) and not getattr(self, "_docking", False):
            self._back_off()  # still at a table: step back before turning
            self._walk_pose()
        if not getattr(self, "_docking", False):  # docking is the last straight half metre: no planning
            planner = self.scene.navmap()
        self._nav_seen = []
        if plan and plan.get("planar"):
            took = self._planar_follow(x, y, yaw, planner, timeout)
            end, _ = self._base_pose()
            x, y = self._nav_goal  # the nearest free spot if (x, y) was occupied
            left = float(np.hypot(x - end[0], y - end[1]))
            return {"goal": _r([x, y]), "reached": left < max(tolerance, 0.05), "end": _r(end),
                    "distance_left_m": round(left, 3), "time_s": round(took, 1)}
        vmax = self.gait.max_speed * 0.7 if self.gait is not None else 0.5
        start, _ = self._base_pose()
        self.target = [float(x), float(y), 0.0]
        self.navigating = True
        self._start_locomotion()
        t, dt, reached = 0.0, 0.05, False
        progress_ref, progress_t = start[:2].copy(), 0.0
        fine = hasattr(self.gait, "hold_here")  # learned policies can side-step: finish with a precise docking
        coarse = max(tolerance, 0.3) if fine else tolerance
        lookahead = max(0.5, 1.5 * self._footprint_radius())
        path, path_goal, planned_t = None, None, -1e9
        recoveries = 0
        turn_sign, anchor = 0.0, None  # turning on the spot: keep the direction, and stay where it began
        try:
            if fine and np.hypot(x - start[0], y - start[1]) < coarse:
                reached = True  # already close: only the fine docking below
            while t < timeout and not reached:
                goal = np.array(self.target[:2])
                # One continuous path to the goal, refreshed every second with what the lidar sees
                # (no stopping at intermediate points: the robot steers towards a point ahead on the path).
                if path is None or t - planned_t >= 1.0 or np.linalg.norm(goal - path_goal) > 1e-6:
                    first = path is None and planned_t < 0
                    same_goal = path is not None and np.linalg.norm(goal - path_goal) < 1e-6
                    path = self._replan(planner, goal, path if same_goal else None)
                    path_goal, planned_t = goal.copy(), t
                    goal = np.array(self.target[:2])  # may have moved to the nearest free spot
                    if first:  # a long way round takes longer than the default timeout
                        timeout = max(timeout, 3.0 * path.length / max(0.6 * vmax, 0.1) + 10.0)
                pos, heading = self._base_pose()  # not `yaw`: that is the requested final heading
                dist = float(np.linalg.norm(goal - pos[:2]))
                if dist < coarse:
                    reached = True
                    break
                look = self._lookahead(planner, pos[:2], lookahead)  # short in doors: no corner cutting
                aim = goal if dist < look else path.carrot(pos[:2], look)
                delta = aim - pos[:2]
                err = float(np.arctan2(delta[1], delta[0]) - heading)
                err = (err + np.pi) % (2 * np.pi) - np.pi
                if abs(err) > 2.6 and turn_sign:  # nearly behind: do not flip-flop between turning left and right
                    err = turn_sign * abs(err)
                turn_sign = float(np.sign(err)) if abs(err) > 0.5 else 0.0
                w = float(np.clip(2.0 * err, -1.0, 1.0))
                vmin = getattr(self.gait, "min_speed", 0.0)
                v = float(np.clip(1.2 * dist, vmin, vmax) * max(0.0, np.cos(err)) ** 2)
                if fine and np.cos(err) < 0.3:
                    v = -0.08  # the walking policy creeps forward while turning on the spot: counter it
                if not fine and self.gait is not None and np.cos(err) < 0.3:
                    # A scripted gait slips sideways when it pivots: hold the spot where the turn began (in a
                    # corridor a long robot would otherwise drift into the wall while turning around).
                    anchor = pos[:2].copy() if anchor is None else anchor
                    c, s_ = np.cos(heading), np.sin(heading)
                    off = anchor - pos[:2]
                    self.gait.set_velocity(1.5 * (c * off[0] + s_ * off[1]), 1.5 * (-s_ * off[0] + c * off[1]), w)
                else:
                    anchor = None
                    self._set_base_velocity(v, w, plan)
                self.wait(dt)
                t += dt
                if self.profile.floating_base and abs(_quat_to_rpy(self.d.xquat[self._bid(self.profile.root_body)])[0]) > 1.0:
                    raise SkillError("The robot fell over while walking. Call reset and try a shorter or straighter path.")
                if t - progress_t > 5.0:  # stall check: the base must move or turn towards the goal
                    if np.linalg.norm(pos[:2] - progress_ref) < 0.05 and abs(err) < 0.5:
                        recoveries += 1
                        if recoveries > 2:
                            raise SkillError(f"{self.name} is stuck: it could not get past an obstacle on the way.")
                        # Recovery (as Nav2 does): back up, look again, replan around what blocked it.
                        self._set_base_velocity(-0.15, 0.0, plan)
                        self.wait(1.5)
                        t += 1.5
                        path = None
                    progress_ref, progress_t = self._base_pose()[0][:2].copy(), t
            if reached and fine:  # dock: side-step and turn until position and heading are right
                goal = np.array(self.target[:2])
                with self._lock:
                    self.gait.hold_here(float(goal[0]), float(goal[1]), yaw)
                t_fine = 0.0
                while t_fine < 8.0:
                    pos, heading = self._base_pose()
                    target_yaw = self.gait.hold[2]
                    yaw_err = abs((target_yaw - heading + np.pi) % (2 * np.pi) - np.pi)
                    if np.linalg.norm(goal - pos[:2]) < tolerance and yaw_err < 0.08:
                        break
                    self.wait(dt)
                    t_fine += dt
                reached = bool(np.linalg.norm(goal - self._base_pose()[0][:2]) < max(tolerance, 0.1))
                t += t_fine
            elif reached and yaw is not None:  # turn on the spot to the requested heading
                t_turn = 0.0
                while t_turn < 10.0:
                    _, now = self._base_pose()
                    err = float((yaw - now + np.pi) % (2 * np.pi) - np.pi)
                    if abs(err) < 0.07:
                        break
                    self._set_base_velocity(0.0, float(np.clip(1.5 * err, -0.6, 0.6)), plan)
                    self.wait(dt)
                    t_turn += dt
                t += t_turn
        finally:
            self._stop_locomotion(plan)
            self.navigating = False
            final_goal = list(self.target or [x, y, 0.0])
            self.target = None
        end, yaw = self._base_pose()
        return {"goal": _r(final_goal[:2]), "reached": reached, "end": _r(end),
                "distance_left_m": round(float(np.linalg.norm(np.array(final_goal[:2]) - end[:2])), 3), "time_s": round(t, 1)}

    def spawn_scanned(self, kind: str, xy: list[float], surface: float, name: str | None = None,
                      mass: float = 0.15, recompile: bool = True) -> dict:
        """Add a 3D-scanned real object (see anybot.assets) standing on a surface at height `surface`."""
        from anybot import assets

        if kind not in assets.SCANNED_OBJECTS:
            raise SkillError(f"Unknown scanned object '{kind}'. Options: {', '.join(assets.SCANNED_OBJECTS)}")
        name = name or kind
        if self._bid(name) >= 0 or name in self.objects:
            raise SkillError(f"An object named '{name}' already exists.")
        folder = assets.scanned_object(kind)
        lo, hi = assets.obj_bounds(folder / "model.obj")
        half, centre = (hi - lo) / 2, (hi + lo) / 2
        pos = [float(xy[0]), float(xy[1]), float(surface - lo[2] + 0.002)]
        with self._lock:
            sp = self.spec
            if not any(m.name == f"scan_{kind}" for m in sp.meshes):
                sp.add_mesh(name=f"scan_{kind}", file=str(folder / "model.obj"))
                sp.add_texture(name=f"scan_{kind}_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(folder / "texture.png"))
                mat = sp.add_material(name=f"scan_{kind}_mat")
                tex = list(mat.textures)
                tex[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = f"scan_{kind}_tex"
                mat.textures = tex
            body = sp.worldbody.add_body(name=name, pos=pos)
            body.add_freejoint()
            for key in sp.keys:
                key.qpos = list(key.qpos) + pos + [1, 0, 0, 0]
                key.qvel = list(key.qvel) + [0] * 6
            # The scan is what you see; a box of the same size is what collides (fast and stable).
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"scan_{kind}", material=f"scan_{kind}_mat",
                          contype=0, conaffinity=0, density=0, rgba=[1, 1, 1, 1])
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=list(centre), size=list(half), mass=mass, group=3,
                          rgba=[0, 0, 0, 0], friction=[1.2, 0.05, 0.001], contype=1, conaffinity=1, condim=4)
            self._register_object(name, {"shape": "scan", "kind": kind, "size": float(half[2]), "fixed": False, "color": None})
            if recompile:
                self.recompile()
        return {"name": name, "pos": _r(pos)}

    def move_object(self, name: str, x: float, y: float) -> dict:
        """Put an object somewhere else by hand (not a robot skill): it rests on whatever is at (x, y)."""
        if name not in self.objects:
            raise SkillError(f"No object '{name}'." + _suggest(name, self.objects))
        if self.object_meta[name].get("fixed"):
            raise SkillError(f"'{name}' is fixed to the world.")
        for r in (self.scene.robots or [self]):  # whoever holds it lets go
            for arm, held in list(r.held.items()):
                if held == name:
                    r._detach(r.arms[arm])
                    del r.held[arm]
                    if getattr(r, "_two_hand", None) and r._two_hand[2] == name:
                        r._two_hand = None
        body = self._bid(name)
        lo, _ = self._object_box(name)
        with self._lock:
            m, d = self.m, self.d
            jnt = int(m.body_jntadr[body])
            above = float(d.xpos[body][2] - lo[2])  # body origin above the object's bottom
            surface = self.support_height(x, y, ignore=(body,))
            qa, da = int(m.jnt_qposadr[jnt]), int(m.jnt_dofadr[jnt])
            d.qpos[qa:qa + 3] = [x, y, surface + above + 0.004]
            d.qvel[da:da + 6] = 0.0
            mujoco.mj_forward(m, d)
        self.wait(0.3)
        with self._lock:
            now = self.d.xpos[body].copy()
        return {"moved": name, "object_at": _r(now)}

    def remove_object(self, name: str) -> dict:
        """Take an object out of the scene (released first if a hand holds it)."""
        if name not in self.objects:
            raise SkillError(f"No object '{name}'." + _suggest(name, self.objects))
        for r in (self.scene.robots or [self]):  # whoever holds it lets go
            for arm, held in list(r.held.items()):
                if held == name:
                    r._detach(r.arms[arm])
                    del r.held[arm]
                    if getattr(r, "_two_hand", None) and r._two_hand[2] == name:
                        r._two_hand = None
            for key in [k for k, w in r._welds.items() if w[0].endswith("::" + name)]:
                del r._welds[key]
        with self._lock:
            sp = self.spec  # (MuJoCo drops the body's entries from the keyframes itself)
            for eq in list(sp.equalities):  # the (inactive) grip welds that point at it
                if eq.name2 == name or eq.name1 == name:
                    sp.delete(eq)
            sp.delete(sp.body(name))
        self.scene.objects.remove(name)
        self.scene.object_meta.pop(name, None)
        self.recompile()
        return {"removed": name, "objects": list(self.objects)}

    def _register_object(self, name: str, meta: dict) -> None:
        self.scene.register_object(name, meta)

    def recompile(self) -> None:
        """Rebuild the model after the spec changed (new objects), keeping the simulation state."""
        self.scene.recompile()

    BASE_OMEGA = 15.0  # rad/s: bandwidth of the holonomic base servo (translation)
    YAW_OMEGA = 25.0  # rad/s: heading (stiffer: an arm swinging on the base must not turn it)

    def _limit_base_force(self) -> None:
        """Holonomic bases often come with ideal servos (kp 1e6, no force limit): they push through walls,
        and clamping such a stiff servo turns it into a shaking bang-bang controller. Give x/y realistic,
        critically damped gains from the robot's mass, and the force of real wheel motors (~4 m/s²)."""
        if not self.profile.planar_base:
            return
        m = self.m
        M = np.zeros((m.nv, m.nv))
        with self._lock:
            mujoco.mj_forward(m, self.d)  # the mass matrix must be up to date
            mujoco.mj_fullM(m, self.d, M)
        for i, n in enumerate(self.profile.planar_base):
            j = self._jid(n)
            a = self.pos_act.get(j)
            if a is None or m.actuator_biastype[a] != mujoco.mjtBias.mjBIAS_AFFINE:
                continue
            inertia = float(M[m.jnt_dofadr[j], m.jnt_dofadr[j]])  # kg for x/y, kg m² for the heading
            w = self.BASE_OMEGA if i < 2 else self.YAW_OMEGA
            kp, kv = inertia * w * w, 2.0 * inertia * w
            m.actuator_gainprm[a, 0], m.actuator_biasprm[a, 1], m.actuator_biasprm[a, 2] = kp, -kp, -kv
            if i < 2:  # wheel motors push at most ~4 m/s² worth: a wall stops the robot
                m.actuator_forcelimited[a] = 1
                m.actuator_forcerange[a] = [-4.0 * inertia, 4.0 * inertia]

    def _after_recompile(self) -> None:
        """Model indices are unchanged (new bodies are appended), but welds and gaits hold model data."""
        self._limit_base_force()
        self._apply_welds()
        if self.gait is not None:
            active, vel = self.gait.active, self.gait.target_vel
            self.gait = self._make_gait()
            self.gait.active, self.gait.target_vel = active, vel

    def spawn_object(
        self, shape: str = "box", size: float = 0.03, pos: list[float] | None = None,
        color: str = "red", name: str | None = None, fixed: bool = False, mass: float = 0.1,
        recompile: bool = True,
    ) -> dict:
        if shape not in _SHAPES:
            raise SkillError(f"shape must be one of {list(_SHAPES)}")
        rgba = _COLORS.get(color)
        if rgba is None:
            raise SkillError(f"color must be one of {list(_COLORS)}")
        pos = list(pos or [0.5, 0, size])
        name = name or f"{color}_{shape}_{len(self.objects)}"
        if self._bid(name) >= 0 or name in self.objects:
            raise SkillError(f"An object named '{name}' already exists.")
        with self._lock:
            body = self.spec.worldbody.add_body(name=name, pos=pos)
            if not fixed:
                body.add_freejoint()
                for key in self.spec.keys:
                    key.qpos = list(key.qpos) + pos + [1, 0, 0, 0]
                    key.qvel = list(key.qvel) + [0] * 6
            sz = {"box": [size] * 3, "sphere": [size, 0, 0], "cylinder": [size, size, 0], "capsule": [size, size, 0]}[shape]
            body.add_geom(type=_SHAPES[shape], size=sz, rgba=rgba, mass=mass, friction=[1.5, 0.05, 0.001],
                          contype=1, conaffinity=1, condim=4)
            self._register_object(name, {"shape": shape, "size": float(size), "fixed": fixed, "color": color})
            if recompile:
                self.recompile()
        return {"name": name, "pos": _r(pos)}

    def reset(self) -> dict:
        for a in self.arms.values():
            self._detach(a)
        self.held.clear()
        self._two_hand = None
        with self._lock:
            self._traj = {}
            self._reset_state()
            for a in self.arms.values():  # back to the rest pose (may differ from the model's keyframe)
                for j in a.joints:
                    if j in self.pos_act:
                        v = float(self._home_cmd[self.pos_act[j]])
                        self.cmd[self.pos_act[j]] = v
                        self.d.qpos[self.m.jnt_qposadr[j]] = v
            mujoco.mj_forward(self.m, self.d)
        return self.state()

    def framing(self, d: mujoco.MjData | None = None) -> tuple[np.ndarray, float]:
        """Camera look-at point and distance that frame the robot and the spawned objects."""
        m = self.m
        if d is None:
            with self._lock:
                d = copy.copy(self.d)
        root = self._bid(self.profile.root_body)
        objs = {self._bid(o) for o in self.objects}
        bodies = [b for b in range(m.nbody) if self._is_ancestor(root, b) or b in objs]
        lo, hi = world_aabb(m, d, body_geoms(m, bodies))
        return (lo + hi) / 2, max(1.0, 1.8 * float(np.max(hi - lo)))

    def snapshot(
        self, width: int = 640, height: int = 480, azimuth: float = 135, elevation: float = -25,
        distance: float | None = None,
    ) -> bytes:
        from PIL import Image as PILImage

        with self._lock:
            m, d = self.m, copy.copy(self.d)
        lookat, auto_distance = self.framing(d)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = lookat
        cam.distance = distance or auto_distance
        cam.azimuth, cam.elevation = azimuth, elevation
        with mujoco.Renderer(m, height, width) as r:
            r.update_scene(d, cam)
            pixels = r.render()
        buf = io.BytesIO()
        PILImage.fromarray(pixels).save(buf, format="PNG")
        return buf.getvalue()


def _r(v, nd: int = 3) -> list[float]:
    return [round(float(x), nd) for x in v]


def _quat_to_rpy(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw])
