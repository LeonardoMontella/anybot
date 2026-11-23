"""Scene: the physics shared by every robot in a simulation.

A single-robot Sim owns a Scene of its own. A team (several robots composed into one MuJoCo model)
shares one Scene: one model, one data, one physics loop, one set of objects, zones and conveyors.
Each robot (a Sim) only commands its own actuators, through the shared command vector.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    from anybot.sim import Sim


class TaskStopped(Exception):
    """The user stopped the running task (raised once, from inside a wait, in every running job)."""


@dataclass
class Conveyor:
    """A belt modelled kinematically: free objects resting on its top move at `speed` along `direction`.

    `end` is the stopper position: objects pile up there until something picks them up."""

    name: str
    centre: np.ndarray  # top surface centre (x, y, z)
    direction: np.ndarray  # unit vector in the ground plane
    half_length: float
    half_width: float
    speed: float = 0.15

    @property
    def start(self) -> np.ndarray:
        return self.centre[:2] - self.direction[:2] * (self.half_length - 0.1)

    @property
    def end(self) -> np.ndarray:
        """Where an object comes to rest against the stopper (for a 7 cm crate)."""
        return self.centre[:2] + self.direction[:2] * (self.half_length - 0.035)


class Scene:
    def __init__(self, spec: mujoco.MjSpec, model: mujoco.MjModel, world: str | None = None):
        self.spec, self.m = spec, model
        self.d: mujoco.MjData | None = None
        self.cmd: np.ndarray | None = None
        self.lock = threading.RLock()
        self.robots: list["Sim"] = []
        self.world = world if world and world != "empty" else None
        self.objects: list[str] = []
        self.object_meta: dict[str, dict] = {}  # name → {"shape", "size", "fixed", "color"}
        self.zones: dict[str, list[float]] = {}  # named places: floor zones, drop pads, belt ends
        self.stations: dict[str, dict] = {}  # tables: {"slots", "top", "xy"}
        self.pad_half = 0.0
        self.conveyors: list[Conveyor] = []
        self.ticks = 0
        self.building = False
        self._navmap = None  # anybot.navmap.Planner, built on first use
        self._navmap_built = False
        self.running = False
        self._thread: threading.Thread | None = None
        self.viewer = None
        # Without a physics thread, robots working in parallel share the clock: time only moves on
        # when every participant is waiting (so thinking time costs no simulated time).
        self._clock = threading.Condition(self.lock)
        self._participants: set[int] = set()
        self._waiting: dict[int, float] = {}
        # Stopping: every job remembers the stop generation it started in; a wait in a job from an
        # older generation raises TaskStopped once (cleanup code after it can still wait normally).
        self.stop_gen = 0
        self._job = threading.local()

    def ensure_data(self) -> None:
        if self.d is None or self.d.qpos.shape[0] != self.m.nq:
            self.d = mujoco.MjData(self.m)
            self.cmd = np.zeros(self.m.nu)

    # ------------------------------------------------------------------ physics

    def write_ctrl(self) -> None:
        m, d = self.m, self.d
        d.ctrl[:] = self.cmd
        for r in self.robots:
            r._post_ctrl()
        lim = m.actuator_ctrllimited.astype(bool)
        d.ctrl[lim] = np.clip(d.ctrl[lim], m.actuator_ctrlrange[lim, 0], m.actuator_ctrlrange[lim, 1])

    def _run_conveyors(self) -> None:
        """Move objects resting on a belt. Like a real line with singulation, an object waits while
        the one ahead is closer than `gap`, so a gripper always has room around the front one."""
        if not self.conveyors:
            return
        m, d = self.m, self.d
        gap = 0.06
        for c in self.conveyors:
            side = np.array([-c.direction[1], c.direction[0]])
            riding = []
            for name in self.objects:
                meta = self.object_meta[name]
                if meta.get("fixed"):
                    continue
                b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
                jnt = m.body_jntadr[b]
                if jnt < 0 or m.jnt_type[jnt] != mujoco.mjtJoint.mjJNT_FREE:
                    continue
                p = d.xpos[b]
                size = meta.get("size", 0.03)
                rel = p[:2] - c.centre[:2]
                along, across = float(rel @ c.direction[:2]), float(rel @ side)
                if abs(along) <= c.half_length and abs(across) <= c.half_width + 0.02 and abs(p[2] - size - c.centre[2]) < 0.02:
                    riding.append((along, size, m.jnt_dofadr[jnt]))
            riding.sort(reverse=True)  # front first
            ahead = None
            for along, size, dof in riding:
                at_end = along >= c.half_length - size - 0.004
                blocked = ahead is not None and ahead[0] - along < ahead[1] + size + gap
                v = np.zeros(2) if (at_end or blocked) else c.direction[:2] * c.speed
                d.qvel[dof:dof + 2] = v
                d.qvel[dof + 3:dof + 6] *= 0.5
                ahead = (along, size)

    def step(self, n: int = 1) -> None:
        every = max(1, round(0.01 / self.m.opt.timestep))  # 100 Hz control for gaits and policies
        for _ in range(n):
            for r in self.robots:
                r._pre_step(self.ticks, every)
            self.ticks += 1
            self.write_ctrl()
            self._run_conveyors()
            mujoco.mj_step(self.m, self.d)

    def _loop(self) -> None:
        debt, last = 0.0, time.perf_counter()
        while self.running:
            now = time.perf_counter()
            debt = min(debt + now - last, 0.1)
            last = now
            with self.lock:
                n = int(debt / self.m.opt.timestep)
                if n:
                    self.step(n)
                    debt -= n * self.m.opt.timestep
                if self.viewer is not None and self.viewer.is_running():
                    self.viewer.sync()
            time.sleep(0.004)

    def start(self, realtime: bool = True, viewer: bool = False) -> None:
        if viewer:
            self.open_viewer()
        if realtime and not self.running:
            self.running = True
            self._thread = threading.Thread(target=self._loop, name="anybot-physics", daemon=True)
            self._thread.start()

    def wait(self, seconds: float) -> None:
        """Let simulated time advance: real time with the physics thread, as fast as possible without.

        Without a thread several callers (robots working in parallel) can wait at once: each one
        steps the shared physics in small slices until the scene clock reaches its own deadline."""
        self._check_stop()
        if self.running:
            end = time.perf_counter() + seconds
            while time.perf_counter() < end:
                time.sleep(0.01)
                self._check_stop()
            return
        target = self.d.time + seconds - 1e-9
        me = threading.get_ident()
        if me not in self._participants:  # a single caller: just run the physics
            with self.lock:
                n = int(np.ceil((target - self.d.time) / self.m.opt.timestep))
                if n > 0:
                    self.step(n)
            return
        with self._clock:
            self._waiting[me] = target
            self._clock.notify_all()
            while self.d.time < target:
                if self.stop_gen > getattr(self._job, "gen", self.stop_gen):
                    del self._waiting[me]
                    self._clock.notify_all()
                    self._check_stop()
                if len(self._waiting) >= len(self._participants):  # everyone waits: advance to the first deadline
                    first = min(self._waiting.values())
                    n = int(np.ceil((first - self.d.time) / self.m.opt.timestep))
                    if n > 0:
                        self.step(n)
                        self._clock.notify_all()
                    else:  # someone's deadline has passed: let that thread wake up and go on
                        self._clock.notify_all()
                        self._clock.wait(0.01)
                else:
                    self._clock.wait(0.05)
            del self._waiting[me]
            self._clock.notify_all()

    def begin_job(self) -> None:
        """The calling thread starts a job that the user may stop."""
        self._job.gen = self.stop_gen

    def stop_jobs(self) -> None:
        self.stop_gen += 1
        with self._clock:
            self._clock.notify_all()

    def _check_stop(self) -> None:
        gen = getattr(self._job, "gen", None)
        if gen is not None and self.stop_gen > gen:
            self._job.gen = self.stop_gen  # once: cleanup after this may wait again
            raise TaskStopped()

    def join(self) -> None:
        """The calling thread takes part in a parallel run (see wait)."""
        with self._clock:
            self._participants.add(threading.get_ident())

    def leave(self) -> None:
        with self._clock:
            self._participants.discard(threading.get_ident())
            self._waiting.pop(threading.get_ident(), None)
            self._clock.notify_all()

    def open_viewer(self) -> None:
        try:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(self.m, self.d, show_left_ui=False, show_right_ui=False)
        except Exception as e:  # macOS needs `mjpython`
            print(f"[anybot] viewer unavailable ({e}); run the server with mjpython to see it", file=sys.stderr)
            self.viewer = None

    def close(self) -> None:
        self.running = False
        if self._thread:
            self._thread.join(timeout=1)
        if self.viewer is not None:
            self.viewer.close()

    # ------------------------------------------------------------------ model changes

    def recompile(self) -> None:
        """Rebuild the model after the spec changed (new objects), keeping the simulation state."""
        with self.lock:
            self.m, self.d = self.spec.recompile(self.m, self.d)
            for r in self.robots:
                r._after_recompile()
            mujoco.mj_forward(self.m, self.d)
            self._navmap_built = False  # a fixed object (table, wall) may have been added
            if self.viewer is not None:
                self.viewer.close()
                self.open_viewer()

    def register_object(self, name: str, meta: dict) -> None:
        self.objects.append(name)
        self.object_meta[name] = meta
        if meta.get("fixed"):
            return
        for r in self.robots:  # fingerless arms and hands hold objects through a (normally inactive) weld
            for a in r.arms.values():
                if a.suction or a.assisted:
                    self.spec.add_equality(
                        name=f"hold::{a.name}::{name}", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                        name1=self.m.body(a.ee_body).name, name2=name, active=False,
                    )

    def navmap(self):
        """The world's collision map and planner (None in an empty world)."""
        if not self._navmap_built:
            from anybot import navmap

            with self.lock:
                mujoco.mj_forward(self.m, self.d)
                self._navmap = navmap.from_scene(self.m, self.d) if self.world else None
            self._navmap_built = True
        return self._navmap

    def robot(self, name: str) -> "Sim":
        for r in self.robots:
            if r.name == name:
                return r
        raise KeyError(name)
