"""Learned locomotion policies, plugged in behind the same interface as the scripted trot.

A policy controller has `active`, `set_velocity(vx, vy, wz)`, `update(dt)` and `max_speed`, so
`Sim.navigate_to` / `Sim.drive` work unchanged whether the legs are driven by a script or a network.

Registered today:
  * Unitree G1 walking policy from unitreerobotics/unitree_rl_gym (BSD-3-Clause), the one Unitree
    ships for sim-to-sim in MuJoCo. Weights are downloaded once to ~/.anybot/policies.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    from anybot.sim import Sim

CACHE = Path.home() / ".anybot" / "policies"
_UNITREE = "https://raw.githubusercontent.com/unitreerobotics/unitree_rl_gym/main/deploy/pre_train"


def _fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
    return dest


def _gravity_in_base(q: np.ndarray) -> np.ndarray:
    """Gravity direction expressed in the base frame (same formula as Unitree's deploy script)."""
    qw, qx, qy, qz = q
    return np.array([2 * (-qz * qx + qw * qy), -2 * (qz * qy + qw * qx), 1 - 2 * (qw * qw + qz * qz)])


class UnitreeG1Walk:
    """Velocity-commanded walking for the Unitree G1 (legs only; waist and arms are held still)."""

    LEG_JOINTS = [f"{side}_{j}_joint" for side in ("left", "right")
                  for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
    KPS = np.array([100, 100, 100, 150, 40, 40] * 2, dtype=float)
    KDS = np.array([2, 2, 2, 4, 2, 2] * 2, dtype=float)
    DEFAULT = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0] * 2)
    CMD_SCALE = np.array([2.0, 2.0, 0.25])
    POLICY_DT, PERIOD, ACTION_SCALE = 0.02, 0.8, 0.25

    max_speed, max_turn, period = 0.5, 0.8, 0.8
    keeps_balance = True  # must keep running while standing still
    min_speed = 0.22  # the policy ignores smaller forward commands

    @classmethod
    def matches(cls, sim: "Sim") -> bool:
        return sim.profile.category == "humanoid" and all(sim._jid(sim.prefix + n) >= 0 for n in cls.LEG_JOINTS)

    def __init__(self, sim: "Sim"):
        import warnings

        import torch  # heavy dependency, only needed for learned policies

        self.sim, self.m = sim, sim.m
        path = _fetch(f"{_UNITREE}/g1/motion.pt", CACHE / "unitree_g1" / "motion.pt")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)  # torch.jit is how Unitree ships the weights
            self.policy = torch.jit.load(str(path))
        self.policy.eval()
        self._torch = torch
        m = sim.m
        self.joints = [sim._jid(sim.prefix + n) for n in self.LEG_JOINTS]
        self.qadr = np.array([m.jnt_qposadr[j] for j in self.joints])
        self.dadr = np.array([m.jnt_dofadr[j] for j in self.joints])
        self.acts = [sim.pos_act[j] for j in self.joints]
        root = sim._bid(sim.profile.root_body)
        free = m.body_jntadr[root]
        self.base_qadr, self.base_dadr = int(m.jnt_qposadr[free]), int(m.jnt_dofadr[free])
        # Use the policy's PD gains on the leg servos (the network was trained with them).
        for a, kp, kd in zip(self.acts, self.KPS, self.KDS):
            m.actuator_gainprm[a, 0], m.actuator_biasprm[a, 1], m.actuator_biasprm[a, 2] = kp, -kp, -kd
            if a in sim.pd:
                j = sim.pd[a][0]
                sim.pd[a] = (j, kp, kd)
        self.action = np.zeros(12, dtype=np.float32)
        self.cmd = np.zeros(3)
        self.target_vel = np.zeros(3)
        self.vel = np.zeros(3)
        self.t = 0.0
        self._acc = 0.0
        self.hold: tuple[float, float, float] | None = None  # (x, y, yaw) to stay at while standing
        self.active = True  # balance from the first step
        for a, q in zip(self.acts, self.DEFAULT):
            sim.cmd[a] = q

    def hold_here(self, x: float | None = None, y: float | None = None, yaw: float | None = None) -> None:
        """Stand still at a pose (default: where the robot is now) instead of slowly drifting away."""
        d, q = self.sim.d, self.base_qadr
        qw, qx, qy, qz = d.qpos[q + 3:q + 7]
        now_yaw = float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
        self.hold = (float(d.qpos[q]) if x is None else x, float(d.qpos[q + 1]) if y is None else y,
                     now_yaw if yaw is None else yaw)

    def _hold_velocity(self) -> np.ndarray:
        d, q = self.sim.d, self.base_qadr
        qw, qx, qy, qz = d.qpos[q + 3:q + 7]
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        ex, ey = self.hold[0] - d.qpos[q], self.hold[1] - d.qpos[q + 1]
        c, s_ = np.cos(yaw), np.sin(yaw)
        eyaw = (self.hold[2] - yaw + np.pi) % (2 * np.pi) - np.pi
        return np.array([np.clip(2.0 * (c * ex + s_ * ey), -0.3, 0.3), np.clip(2.0 * (-s_ * ex + c * ey), -0.3, 0.3),
                         np.clip(2.0 * eyaw, -0.6, 0.6)])

    def restart(self) -> None:
        """Resume after a standstill: start from the standing pose with no motion memory."""
        self.action[:] = 0.0
        self.vel[:] = 0.0
        self.target_vel[:] = 0.0
        self.hold = None
        self.t = self._acc = 0.0
        self.active = True

    def set_velocity(self, vx: float, vy: float = 0.0, wz: float = 0.0) -> None:
        self.hold = None
        self.target_vel = np.array([np.clip(vx, -0.3, self.max_speed), np.clip(vy, -0.3, 0.3),
                                    np.clip(wz, -self.max_turn, self.max_turn)])

    def update(self, dt: float) -> None:
        self.t += dt
        self._acc += dt
        if self._acc + 1e-9 < self.POLICY_DT:
            return
        self._acc = 0.0
        step = np.array([0.6, 0.6, 1.5]) * self.POLICY_DT
        if self.hold is not None:
            self.target_vel = self._hold_velocity()
        self.vel += np.clip(self.target_vel - self.vel, -step, step)
        d = self.sim.d
        quat = d.qpos[self.base_qadr + 3:self.base_qadr + 7]
        omega = d.qvel[self.base_dadr + 3:self.base_dadr + 6]
        phase = (self.t % self.PERIOD) / self.PERIOD
        obs = np.concatenate([
            omega * 0.25,
            _gravity_in_base(quat),
            self.vel * self.CMD_SCALE,
            d.qpos[self.qadr] - self.DEFAULT,
            d.qvel[self.dadr] * 0.05,
            self.action,
            [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)],
        ]).astype(np.float32)
        with self._torch.no_grad():
            self.action = self.policy(self._torch.from_numpy(obs).unsqueeze(0)).numpy().squeeze().astype(np.float32)
        target = self.action * self.ACTION_SCALE + self.DEFAULT
        for a, q in zip(self.acts, target):
            self.sim.cmd[a] = float(q)


POLICIES = [UnitreeG1Walk]


def make_policy(sim: "Sim"):
    """A learned locomotion controller for this robot, or None if none is registered / available."""
    for cls in POLICIES:
        if cls.matches(sim):
            try:
                return cls(sim)
            except Exception as e:  # missing torch, offline, incompatible weights...
                print(f"[anybot] {cls.__name__} unavailable: {e}")
    return None
