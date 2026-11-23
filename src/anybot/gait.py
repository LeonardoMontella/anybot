"""A scripted trot for any quadruped, derived from its RobotProfile.

No learned policy and no robot-specific numbers: the four legs come from the profile, the feet
are the lowest geoms of each leg, nominal stance is the robot's home pose, and every control
tick solves a small numerical IK per leg so the feet follow a trot pattern
(diagonal pairs alternate; stance feet slide back under the body, swing feet arc forward).

It is an open-loop gait meant for flat ground: good enough to walk to a clicked point, not a
substitute for an RL locomotion policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from anybot.introspect import body_geoms

if TYPE_CHECKING:
    from anybot.sim import Sim


# Per-robot styles where the generic gait does not suit the machine. height: stance height as a fraction of
# the leg's thigh+shank length; duty: fraction of the cycle each foot is down; period/step_height override.
STYLES = {
    # Spot walks tall with its legs well extended and a calm, overlapping trot, not crouched and scurrying.
    "spot": {"height": 0.78, "duty": 0.6, "period": 0.4, "step_height": 0.09},
}


@dataclass
class _Leg:
    name: str
    tip: int
    joints: list[int]
    qadr: np.ndarray
    dadr: np.ndarray
    acts: list[int]
    lo: np.ndarray
    hi: np.ndarray
    foot_offset: np.ndarray  # foot point in the tip body frame
    nominal: np.ndarray  # foot position in the base frame at the home pose
    phase_offset: float
    q: np.ndarray  # last IK solution (warm start)


class TrotGait:
    def __init__(self, sim: "Sim", period: float = 0.36, max_speed: float = 0.5, max_turn: float = 1.2,
                 duty: float | None = None):
        m = sim.m
        self.sim, self.m = sim, m
        self.period, self.max_speed, self.max_turn = period, max_speed, max_turn
        legs = [limb for limb in sim.profile.limbs if limb.kind == "leg"]
        if len(legs) != 4:
            raise ValueError("trot needs exactly four legs")

        # Kinematics scratch data: home pose with the base at the origin, so world frame = base frame.
        self.kd = mujoco.MjData(m)
        self.kd.qpos[:] = sim.home_qpos()
        self.root = sim._bid(sim.profile.root_body)
        free = m.body_jntadr[self.root]
        self.base_qadr = int(m.jnt_qposadr[free])
        self.kd.qpos[self.base_qadr:self.base_qadr + 7] = [0, 0, 0, 1, 0, 0, 0]
        mujoco.mj_kinematics(m, self.kd)

        self.legs: list[_Leg] = []
        for limb in legs:
            tip = sim._bid(limb.tip)
            joints = [sim._jid(n) for n in limb.joints if sim._jid(n) in sim.pos_act]
            geoms = body_geoms(m, sim._subtree(tip))
            foot_g = min(geoms, key=lambda g: self.kd.geom_xpos[g, 2])
            R = self.kd.xmat[tip].reshape(3, 3)
            offset = R.T @ (self.kd.geom_xpos[foot_g] - self.kd.xpos[tip])
            qadr = np.array([m.jnt_qposadr[j] for j in joints])
            self.legs.append(_Leg(
                name=limb.tip, tip=tip, joints=joints, qadr=qadr,
                dadr=np.array([m.jnt_dofadr[j] for j in joints]),
                acts=[sim.pos_act[j] for j in joints],
                lo=np.array([m.jnt_range[j, 0] if m.jnt_limited[j] else -np.inf for j in joints]),
                hi=np.array([m.jnt_range[j, 1] if m.jnt_limited[j] else np.inf for j in joints]),
                foot_offset=offset, nominal=self._foot(tip, offset).copy(), phase_offset=0.0,
                q=self.kd.qpos[qadr].copy(),
            ))
        if not m.nkey and sim._home is None:
            self._synthesize_stance()
        style = STYLES.get(getattr(sim, "name", ""), {})
        if "height" in style:
            self._raise_stance(style["height"])
        # Trot: front-left with rear-right, front-right with rear-left.
        cx = np.mean([leg.nominal[0] for leg in self.legs])
        for leg in self.legs:
            front, left = leg.nominal[0] > cx, leg.nominal[1] > 0
            leg.phase_offset = 0.0 if front == left else 0.5
        leg_len = float(np.mean([-leg.nominal[2] for leg in self.legs]))
        # Small robots need a relatively high step to clear the ground; big ones look (and are) better
        # with a low, flat step: about 6-7 cm whatever the size.
        self.step_height = float(np.clip(0.25 * leg_len, 0.04, 0.065))
        # Bigger robots step slower (dynamic similarity: period ∝ sqrt(leg length)).
        self.period = float(np.clip(period * np.sqrt(leg_len / 0.27), 0.3, 0.6))
        # Fraction of the cycle a foot is on the ground. 0.5 is a pure trot (two feet down at a time); a style
        # can ask for overlap (a "walking trot": all four feet share the load around every swap, so a heavy
        # body is never dropped onto the next pair).
        self.duty = float(duty if duty is not None else style.get("duty", 0.5))
        self.smooth = bool(style)  # matched-speed swing + body levelling (tuned per style; the plain trot otherwise)
        # Raibert foot-placement gain: about the time a body of this height takes to tip (sqrt(h/g)).
        self.raibert = float(style.get("raibert", np.sqrt(leg_len / 9.81)))
        self.period = float(style.get("period", self.period))
        self.step_height = float(style.get("step_height", self.step_height))
        self.mass = float(m.body_subtreemass[self.root])
        self._stiffen_native_servos()

        self.active = False
        self.phase = 0.0
        self.vel = np.zeros(3)  # current (smoothed) vx, vy, wz in the base frame
        self.target_vel = np.zeros(3)

    def _synthesize_stance(self) -> None:
        """No home pose in the model: legs are straight (a singular IK pose). Crouch to 80% leg length.

        The knee bends towards the middle of its joint range when the range says which way it bends,
        otherwise front knees bend backward and hind knees forward (the common "X" stance).
        """
        cx = np.mean([leg.nominal[0] for leg in self.legs])
        for leg in self.legs:
            knee = len(leg.joints) - 1
            mid = 0.5 * (leg.lo[knee] + leg.hi[knee]) if np.isfinite(leg.lo[knee]) else 0.0
            sign = np.sign(mid) if abs(mid) > 0.2 else (-1.0 if leg.nominal[0] > cx else 1.0)
            q = leg.q.copy()
            q[knee] += sign * 1.2
            if knee >= 1:
                q[knee - 1] -= sign * 0.6
            leg.q = np.clip(q, leg.lo, leg.hi)
            target = leg.nominal.copy()
            target[2] *= 0.8
            self._leg_ik(leg, target, iters=60)
            self.kd.qpos[leg.qadr] = leg.q
            mujoco.mj_kinematics(self.m, self.kd)
            leg.nominal = self._foot(leg.tip, leg.foot_offset).copy()
        # Hold the crouch right away so the robot does not start from (and fall out of) straight legs.
        for leg in self.legs:
            for a, v in zip(leg.acts, leg.q):
                self.sim.cmd[a] = v

    def _raise_stance(self, frac: float) -> None:
        """Stand taller: feet straight below where they are, at `frac` of the thigh+shank length under the hip."""
        m, kd = self.m, self.kd
        dz = []
        for leg in self.legs:
            pts = [kd.xpos[m.jnt_bodyid[j]].copy() for j in leg.joints] + [leg.nominal]
            length = sum(np.linalg.norm(pts[i + 1] - pts[i]) for i in range(1, len(pts) - 1))
            target = leg.nominal.copy()
            target[2] = pts[0][2] - frac * length
            for _ in range(10):
                self._leg_ik(leg, target, iters=10)
            kd.qpos[leg.qadr] = leg.q
            mujoco.mj_kinematics(m, kd)
            old_z = float(leg.nominal[2])
            leg.nominal = self._foot(leg.tip, leg.foot_offset).copy()
            dz.append(old_z - float(leg.nominal[2]))
        lift = float(np.mean(dz))  # how much higher the body now stands
        # Start (and reset) already standing tall, instead of popping up from the model's crouch.
        sim = self.sim
        qs = [sim.d.qpos] + [m.key_qpos[k] for k in range(m.nkey)]
        for q in qs:
            q[self.base_qadr + 2] += lift
            for leg in self.legs:
                q[leg.qadr] = leg.q
        for leg in self.legs:
            for a, v in zip(leg.acts, leg.q):
                sim.cmd[a] = v
        if sim._home is None:
            sim._home_cmd = sim.cmd.copy()

    def _stiffen_native_servos(self) -> None:
        """Walking needs tight joint tracking: raise native position-servo gains to reach full force at ~0.12 rad.

        Only model gains change (no recompile); software-PD joints are already tuned this way by Sim.
        """
        m = self.m
        for leg in self.legs:
            for j, a in zip(leg.joints, leg.acts):
                if a in self.sim.pd or m.actuator_biastype[a] != mujoco.mjtBias.mjBIAS_AFFINE:
                    continue
                if m.actuator_forcelimited[a]:
                    fmax = float(np.max(np.abs(m.actuator_forcerange[a])))
                elif m.jnt_actfrclimited[j]:
                    fmax = float(np.max(np.abs(m.jnt_actfrcrange[j])))
                else:
                    fmax = 3.0 * float(m.actuator_gainprm[a, 0]) * 0.12
                kp0 = float(m.actuator_gainprm[a, 0])
                kp = min(max(kp0, fmax / 0.12), 4.0 * kp0)  # declared limits can be unrealistically high
                kv = max(-float(m.actuator_biasprm[a, 2]), 0.03 * kp)
                m.actuator_gainprm[a, 0], m.actuator_biasprm[a, 1], m.actuator_biasprm[a, 2] = kp, -kp, -kv

    def _foot(self, tip: int, offset: np.ndarray) -> np.ndarray:
        return self.kd.xpos[tip] + self.kd.xmat[tip].reshape(3, 3) @ offset

    def set_velocity(self, vx: float, vy: float = 0.0, wz: float = 0.0) -> None:
        self.target_vel = np.array([
            np.clip(vx, -self.max_speed, self.max_speed),
            np.clip(vy, -0.5 * self.max_speed, 0.5 * self.max_speed),
            np.clip(wz, -self.max_turn, self.max_turn),
        ])

    def _leg_ik(self, leg: _Leg, target: np.ndarray, iters: int = 4) -> np.ndarray:
        m, kd = self.m, self.kd
        jacp = np.zeros((3, m.nv))
        q = leg.q.copy()
        for _ in range(iters):
            kd.qpos[leg.qadr] = q
            mujoco.mj_kinematics(m, kd)
            mujoco.mj_comPos(m, kd)
            p = self._foot(leg.tip, leg.foot_offset)
            e = target - p
            if e @ e < 1e-8:
                break
            mujoco.mj_jac(m, kd, jacp, None, p, leg.tip)
            J = jacp[:, leg.dadr]
            q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), e), leg.lo, leg.hi)
        leg.q = q
        return q

    def update(self, dt: float) -> None:
        """One control tick (called from the physics loop): advance the gait and write joint commands."""
        if not self.active:
            return
        # Rate-limit velocity changes so the body is not jerked around.
        step = np.array([1.0, 1.0, 3.0]) * dt
        self.vel += np.clip(self.target_vel - self.vel, -step, step)
        self.phase = (self.phase + dt / self.period) % 1.0
        vx, vy, wz = self.vel
        duty = self.duty
        stance_time = self.period * duty
        att, v_err = (self._attitude_correction(), self._velocity_error()) if self.smooth else (np.zeros(2), np.zeros(2))
        for leg in self.legs:
            phi = (self.phase + leg.phase_offset) % 1.0
            nx, ny, nz = leg.nominal
            # Foot velocity in the base frame for a body moving at (vx, vy) and turning at wz.
            vf = np.array([vx - wz * ny, vy + wz * nx])
            d = vf * stance_time
            if self.smooth:
                xy, z = self._smooth_foot(leg, phi, vf, d, dt, att, v_err)
            elif phi < duty:  # stance: foot slides back from +d/2 to -d/2 at the body's speed
                s = phi / duty
                xy, z = d * (0.5 - s), nz
            else:  # swing: lift and bring the foot forward, smooth start/end
                s = (phi - duty) / (1 - duty)
                s_smooth = s * s * (3 - 2 * s)
                xy, z = d * (s_smooth - 0.5), nz + self.step_height * np.sin(np.pi * s)
            q = self._leg_ik(leg, np.array([nx + xy[0], ny + xy[1], z]))
            for a, v in zip(leg.acts, q):
                self.sim.cmd[a] = v

    def _smooth_foot(self, leg: _Leg, phi: float, vf: np.ndarray, d: np.ndarray, dt: float, att: np.ndarray,
                     v_err: np.ndarray) -> tuple[np.ndarray, float]:
        """Foot target (xy offset from its nominal point, z) for the smooth walk. Stateful: a stance foot stays
        where it landed and slides back at the body's speed; a swing foot flies from where it lifted off to
        where it should land, with the ground speed matched at both ends. Landing spot: half a stride ahead,
        plus a step in the direction the body is drifting (Raibert's rule: catch the fall with the foot)."""
        nx, ny, nz = leg.nominal
        duty = self.duty
        if not hasattr(leg, "foot"):
            leg.foot, leg.lift, leg.swinging = np.zeros(2), np.zeros(2), False
        if phi < duty:
            if leg.swinging:  # touch-down
                leg.swinging = False
            leg.foot = leg.foot - vf * dt
            return leg.foot.copy(), nz + float(att @ np.array([ny, -nx]))
        if not leg.swinging:  # lift-off
            leg.swinging, leg.lift = True, leg.foot.copy()
        s = (phi - duty) / (1 - duty)
        T = self.period * (1 - duty)
        land = d / 2 + np.clip(self.raibert * v_err, -0.15, 0.15)
        h00, h10, h01, h11 = 2*s**3 - 3*s**2 + 1, s**3 - 2*s**2 + s, -2*s**3 + 3*s**2, s**3 - s**2
        leg.foot = leg.lift * h00 + land * h01 + (-vf * T) * (h10 + h11)
        lift = (np.sin(np.pi * s) ** 2) if s < 0.5 else np.sin(np.pi * s) ** 1.5  # soft lift, gentle landing
        return leg.foot.copy(), nz + self.step_height * lift

    def _velocity_error(self) -> np.ndarray:
        """Body velocity minus the commanded one, in the heading frame (xy)."""
        sim = self.sim
        dof = sim.m.jnt_dofadr[sim.m.body_jntadr[self.root]]
        v = sim.d.qvel[dof:dof + 2]
        R = sim.d.xmat[self.root].reshape(3, 3)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        c, s_ = np.cos(yaw), np.sin(yaw)
        vb = np.array([c * v[0] + s_ * v[1], -s_ * v[0] + c * v[1]])
        return vb - self.vel[:2]

    def _attitude_correction(self) -> np.ndarray:
        """Small (roll, pitch) feedback, as foot height per metre of lever: keeps a heavy body level."""
        sim = self.sim
        R = sim.d.xmat[self.root].reshape(3, 3)
        roll, pitch = np.arctan2(R[2, 1], R[2, 2]), -np.arcsin(np.clip(R[2, 0], -1, 1))
        w = sim.d.qvel[sim.m.jnt_dofadr[sim.m.body_jntadr[self.root]] + 3:][:2]
        k, kd = 0.6, 0.04
        return np.clip(np.array([k * roll + kd * w[0], k * pitch + kd * w[1]]), -0.15, 0.15)

    def stance_command(self) -> dict[int, float]:
        """Joint commands for standing still on the nominal footholds."""
        out = {}
        for leg in self.legs:
            q = self._leg_ik(leg, leg.nominal, iters=8)
            out.update({a: float(v) for a, v in zip(leg.acts, q)})
        return out
