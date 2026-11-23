"""ToolHost: the one place that owns the live simulation and runs tools on it.

Every front end (MCP adapter, web UI buttons, a future chat agent) calls `host.call(name, args)`,
so they all drive the same robot and show up in the same activity log.
"""

from __future__ import annotations

import itertools
import threading
import time
from typing import Any, Callable

import numpy as np

from anybot.loader import available_descriptions
from anybot.scene import TaskStopped
from anybot.sim import Sim, SkillError


class ToolHost:
    def __init__(self, viewer: bool = False):
        self.sim: Sim | None = None
        self.viewer = viewer
        self.events: list[dict] = []
        self.version = 0  # bumps on every event change; the UI watches it
        self._ids = itertools.count(1)
        self._run_lock = threading.RLock()  # one tool at a time per robot (re-entrant: tasks call tools)
        self._robot_locks: dict[str, threading.RLock] = {}
        self._tls = threading.local()  # the robot a tool call is addressed to (teams run robots in parallel)
        self.scene = None  # set when a team is loaded
        self.busy = 0  # tools currently running
        self._last_load: dict = {}
        self._ev_lock = threading.Lock()
        self.tools: dict[str, Callable[..., Any]] = {
            "list_robots": self._list_robots,
            "load_robot": self._load_robot,
            "describe_robot": self._describe,
            "get_state": lambda: self._need().state(),
            "set_joint_positions": lambda positions, duration=1.5: self._need().set_joints(positions, duration),
            "move_end_effector": lambda x, y, z, arm=None, point_down=False, duration=2.0: self._need().move_end_effector(
                [x, y, z], arm, point_down, duration
            ),
            "gripper": lambda action, arm=None: self._need().gripper(action, arm),
            "go_to_pose": lambda name, duration=2.0: self._need().go_to_pose(name, duration),
            "drive": lambda forward_speed, turn_rate=0.0, seconds=2.0: self._need().drive(forward_speed, turn_rate, seconds),
            "navigate_to": lambda x, y: self._need().navigate_to(x, y),
            "go_to_point": self._go_to_point,
            "pick_object": lambda name, arm=None: self._need().pick(name, arm),
            "place_object": lambda x=0.0, y=0.0, on=None, arm=None: self._need().place(x, y, on, arm),
            "go_to_zone": self._go_to_zone,
            "retarget": self._retarget,
            "set_world": self._set_world,
            "load_team": self._load_team,
            "select_robot": self._select_robot,
            "list_tasks": lambda: self.tasks(),
            "run_task": self._run_task,
            "spawn_object": self._spawn,
            "snapshot": lambda azimuth=135, elevation=-25, distance=None: self._need().snapshot(
                640, 480, azimuth, elevation, distance
            ),
            "stop": self._stop,
            "remove_object": lambda name: self._need().remove_object(name),
            "move_object": lambda name, x, y: self._need().move_object(name, x, y),
            "spawn_asset": self._spawn_asset,
            "import_scene": self._import_scene,
            "get_scan": self._get_scan,
            "camera_image": self._camera_image,
            "wait": self._wait,
            "reset": lambda: self._need().reset(),
        }

    # ------------------------------------------------------------------ tools

    def _need(self) -> Sim:
        robot = getattr(self._tls, "robot", None)
        if robot is not None:
            return robot
        if self.sim is None:
            raise SkillError("No robot loaded. Call load_robot first (list_robots shows the options).")
        return self.sim

    @staticmethod
    def _list_robots(query: str = "") -> list[str]:
        return [r for r in available_descriptions() if query.lower() in r.lower()]

    def _close_current(self) -> None:
        if self.scene is not None:
            self.scene.close()
            self.scene = None
        elif self.sim is not None:
            self.sim.close()
        self.sim = None

    def _stop(self) -> dict:
        """Stop whatever is running: every robot stops where it is (walking robots stand still)."""
        sim = self.sim
        if sim is None:
            return {"stopped": False}
        scene = sim.scene
        running = [e["tool"] for e in self.events[-40:] if e["status"] == "running" and e["tool"] != "stop"]
        scene.stop_jobs()
        with sim._lock:
            for r in (scene.robots or [sim]):
                r._traj, r._ramp = {}, None  # arms freeze where they are
                r.target = None
        return {"stopped": bool(running), "was_running": running}

    def _get_scan(self) -> dict:
        """The robot's 2D lidar, summarised: nearest obstacle per direction sector, plus the raw ranges."""
        from anybot import sensors

        sim = self._need()
        if not (sim.profile.floating_base or sim.profile.planar_base):
            raise SkillError("Only mobile robots carry the lidar.")
        with sim._lock:
            scan = sensors.lidar(sim)
        r = np.where(np.isfinite(scan.ranges), scan.ranges, scan.range_max)
        n = len(r)
        sectors = {name: round(float(np.min(r[int((a + 180 - 22.5) % 360 / 360 * n):int((a + 180 + 22.5) % 360 / 360 * n) or n])), 2)
                   for name, a in (("front", 0), ("front_left", 45), ("left", 90), ("back_left", 135),
                                   ("front_right", -45), ("right", -90), ("back_right", -135))}
        back = np.concatenate([r[: int(22.5 / 360 * n)], r[n - int(22.5 / 360 * n):]])
        sectors["back"] = round(float(np.min(back)), 2)
        return {"frame": "robot (0° = straight ahead, counter-clockwise)", "height_m": sensors.LIDAR_HEIGHT,
                "nearest_m": sectors, "range_max_m": scan.range_max,
                "ranges_every_5deg": [round(float(v), 2) for v in r[::max(1, n // 72)]]}

    def _camera_image(self, depth: bool = False) -> bytes:
        """What the robot's own camera sees (PNG); depth=true gives a depth image (near = bright)."""
        import copy
        import io

        from PIL import Image

        from anybot import sensors

        sim = self._need()
        with sim._lock:
            d = copy.copy(sim.d)
        cam = sensors.Camera(sim.m, 640, 480)
        try:
            rgb, dep, _, _ = cam.render(sim, d)
        finally:
            cam.close()
        img = rgb if not depth else (255 * (1 - np.clip(dep / 6.0, 0, 1))).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="PNG")
        return buf.getvalue()

    def _load_team(self, team: str, realtime: bool = True) -> dict:
        """Several robots in one warehouse scene, with a conveyor line to run (see anybot.team)."""
        from anybot.team import build_team

        try:
            scene = build_team(team, realtime=realtime)
        except ValueError as e:
            raise SkillError(str(e)) from e
        self._close_current()
        self.scene, self.sim = scene, scene.robots[0]
        self._last_load = {"team": team}
        return self.robot_info()

    def _select_robot(self, name: str) -> dict:
        if self.scene is None:
            raise SkillError("No team loaded: there is only one robot.")
        try:
            self.sim = self.scene.robot(name)
        except KeyError:
            raise SkillError(f"No robot '{name}'. Robots: {', '.join(r.name for r in self.scene.robots)}") from None
        return self.robot_info()

    def _load_robot(self, robot: str, pin_base: bool = False, floating: bool = False, world: str = "empty") -> dict:
        if pin_base:
            world = "bench"
        try:
            new = Sim(robot, floating=floating or None, pin_base=pin_base, viewer=self.viewer, world=world)
            if new.profile.category == "humanoid" and not new.pinned and new.gait is None:
                # A humanoid with no walking controller would just fall: it works at the bench instead.
                new.close()
                pin_base, world = True, "bench"
                new = Sim(robot, floating=floating or None, pin_base=True, viewer=self.viewer, world="bench")
        except ValueError as e:
            raise SkillError(str(e)) from e
        self._last_load = {"robot": robot, "pin_base": pin_base, "floating": floating, "world": world}
        if self.scene is not None:  # leaving a team
            self.scene.close()
            self.scene, self.sim = None, None
        old, self.sim = self.sim, new
        if old is not None:
            old.close()
        return self.robot_info()

    def _set_world(self, world: str) -> dict:
        """Reload the current robot in another world."""
        self._need()
        if self.scene is not None:
            raise SkillError("A team has its own warehouse; load a single robot to change world.")
        return self._load_robot(**{"robot": self.sim.name, **self._last_load, "world": world})

    def _describe(self) -> dict:
        sim = self._need()
        d = sim.profile.to_dict()
        d["poses"] = sim.keyframes()
        d["skills"] = sim.skills()
        return d

    def _spawn(self, shape="box", size=0.025, x=0.5, y=0.0, z=None, color="red", name=None, fixed=False, mass=0.1):
        sim = self._need()
        if z is None:  # resting on whatever is there: the floor, a table, a shelf
            z = sim.support_height(x, y) + size + 0.003
        return sim.spawn_object(shape, size, [x, y, z], color, name, fixed, mass)

    def _import_scene(self, source: str, name: str, height: float = 2.6) -> dict:
        """A world from a video, photos or a scan of a real place (see anybot.capture)."""
        from anybot.capture import CaptureError, import_scene

        log = []
        try:
            out = import_scene(source, name, height=height, progress=lambda m: (log.append(m), print(f"[anybot] {m}")))
        except CaptureError as e:
            raise SkillError(str(e)) from None
        out["next"] = f"load_robot(robot, world='{out['world']}')"
        return out

    def _spawn_asset(self, kind: str = "mug", x: float = 0.5, y: float = 0.0) -> dict:
        """A 3D-scanned real object (mug, tea box, ...) standing on whatever is at (x, y)."""
        sim = self._need()
        name, i = kind, 2
        while name in sim.objects:
            name, i = f"{kind}_{i}", i + 1
        return sim.spawn_scanned(kind, [x, y], sim.support_height(x, y), name=name)

    def _go_to_point(self, x: float, y: float, z: float = 0.0) -> dict:
        """Whatever "go there" means for this robot: walk/drive to (x, y), or hover the gripper above (x, y, z)."""
        sim = self._need()
        skills = sim.skills()
        if "navigate_to" in skills:
            return {"mode": "navigate", **sim.navigate_to(x, y)}
        if "move_end_effector" in skills:
            return {"mode": "reach", **sim.move_end_effector([x, y, z + 0.08], point_down=True)}
        raise SkillError(f"{sim.name} ({sim.profile.category}) can neither move its base nor reach with an arm.")

    # ------------------------------------------------------------------ zones and tasks

    def _zone(self, zone: str) -> list[float]:
        sim = self._need()
        key = zone.lower().replace(" ", "_")
        if key not in sim.zones:
            raise SkillError(f"No zone '{zone}' for this robot. Zones: {', '.join(sim.zones) or 'none (load with world=warehouse)'}")
        return sim.zones[key]

    def _go_to_zone(self, zone: str) -> dict:
        """Walk/drive to a floor zone; an arm puts the held object on a pad (or hovers above it)."""
        sim = self._need()
        x, y = self._zone(zone)
        if "navigate_to" in sim.skills():
            return {"zone": zone, **sim.navigate_to(x, y)}
        if sim.held:
            return {"zone": zone, **sim.place(x, y)}
        return {"zone": zone, **self._go_to_point(x, y)}

    def _retarget(self, x: float, y: float) -> dict:
        sim = self._need()
        if not sim.navigating:
            raise SkillError("The robot is not navigating; use navigate_to.")
        sim.target = [float(x), float(y), 0.0]
        return {"goal": [round(x, 3), round(y, 3)]}

    def tasks(self) -> list[dict]:
        """One-click jobs that make sense for the loaded robot in the current world."""
        sim = self.sim
        if self.scene is not None and getattr(self.scene, "team", None) == "amr_fleet":
            return [{"id": "relay", "label": "Run the AMR relay", "group": "Team",
                     "description": "Pickers load and unload, the two carriers swap places on every trip."},
                    {"id": "reset_scene", "label": "Reset scene", "group": "Scene", "description": "Rebuild the fleet."}]
        if self.scene is not None:
            return [{"id": "production", "label": "Run the production line", "group": "Team",
                     "description": "Loader → conveyor → sorter → tote on the carrier → dock, all at once."},
                    {"id": "reset_scene", "label": "Reset scene", "group": "Scene", "description": "Rebuild the line."}]
        if sim is None or not sim.zones:
            return []
        if "navigate_to" in sim.skills():
            out = []
            if sim.stations:
                out += [
                    {"id": "carry_to_b", "label": "Carry to table B", "group": "Manipulation",
                     "description": "Pick every object on table A and set it down on table B."},
                    {"id": "carry_to_a", "label": "Bring back to table A", "group": "Manipulation",
                     "description": "Move everything on table B back to table A."},
                ]
            out += [{"id": f"go_{z}", "label": f"Go to {z.replace('_', ' ')}", "group": "Navigation",
                     "description": f"Walk or drive to {z.replace('_', ' ')}."} for z in sim.zones]
            stops = [z for z in sim.zones if not z.startswith(("table", "belt", "tote"))]
            return out + [{"id": "patrol", "label": "Patrol", "group": "Navigation",
                           "description": "Visit " + ", ".join(z.replace("_", " ") for z in stops) + " in turn."}]
        if "pick_object" in sim.skills():
            hands = any(a.assisted for a in sim.arms.values())
            out = [{"id": "sort", "label": "Sort by color", "group": "Manipulation",
                    "description": "Orange to the orange tray, blue to the blue tray."}]
            if not hands:  # hand placement (±3-5 cm) is not yet precise enough to stack 4 cm cubes
                out.append({"id": "stack", "label": "Stack a tower", "group": "Manipulation",
                            "description": "Build a tower of three boxes on pad A."})
            return out + [{"id": "reset_scene", "label": "Reset scene", "group": "Scene",
                           "description": "Put robot and objects back to the start."}]
        return []

    def _arm_for(self, xy) -> str | None:
        """Two-armed robots: the arm on the side of a destination, so it never reaches across the body."""
        sim = self.sim
        if len(sim.arms) < 2:
            return None
        m, d = sim.m, sim.d
        return min(sim.arms, key=lambda n: float(np.linalg.norm(
            d.xpos[m.jnt_bodyid[sim.arms[n].joints[0]]][:2] - np.asarray(xy[:2]))))

    def _side(self, arm: str, xy) -> float:
        """Signed lateral offset of a point from the body centre line, positive on that arm's side."""
        sim = self.sim
        m, d = sim.m, sim.d
        root = sim._bid(sim.profile.root_body)
        R = d.xmat[root].reshape(3, 3)
        sh = R.T @ (d.xpos[m.jnt_bodyid[sim.arms[arm].joints[0]]] - d.xpos[root])
        p = R.T @ (np.array([xy[0], xy[1], 0.0]) - d.xpos[root])
        return float(p[1] * np.sign(sh[1]))

    def _move_object(self, step, obj: str, dest: dict) -> None:
        """Pick `obj` (nearest free arm) and place it as `dest` says (x/y or on)."""
        step("pick_object", name=obj)
        step("place_object", **dest)

    def _run_task(self, task: str, source: str = "api") -> dict:
        sim = self._need()
        if self.scene is not None and task == "reset_scene":
            return self._load_team(self._last_load["team"])
        if self.scene is not None and task == "production":
            return self._run_production(source)
        if self.scene is not None and task == "relay":
            return self._run_relay(source)
        ids = [t["id"] for t in self.tasks()]
        if task not in ids:
            raise SkillError(f"No task '{task}' for {sim.name} here. Tasks: {', '.join(ids) or 'none (load with world=warehouse)'}")
        step = lambda tool, **args: self.call(tool, args, source)  # noqa: E731 - each step shows in the activity log
        if task.startswith("go_"):
            return step("go_to_zone", zone=task[3:])
        if task.startswith("carry_to_"):
            dst = "table_" + task[-1]
            src = "table_a" if dst == "table_b" else "table_b"

            def on_table(name: str, pos) -> bool:
                tx, ty = sim.stations[name]["xy"]
                return abs(pos[0] - tx) < 0.5 and abs(pos[1] - ty) < 0.3 and pos[2] > 0.4

            objects = sim.state()["objects"]
            todo = [o for o, p in objects.items() if on_table(src, p)]
            taken = [p for p in objects.values() if on_table(dst, p)]
            free = [s for s in sim.stations[dst]["slots"] if all(abs(s[0] - p[0]) > 0.12 for p in taken)]
            moved = []
            for obj, (x, y) in zip(todo, free):
                step("pick_object", name=obj)
                step("place_object", x=x, y=y)
                moved.append(obj)
            return {"task": task, "moved": moved, "left_behind": todo[len(free):], "objects": sim.state()["objects"]}
        if task == "patrol":
            stops = [z for z in sim.zones if not z.startswith(("table", "belt", "tote"))]
            reached = [step("go_to_zone", zone=z)["reached"] for z in stops]
            return {"task": task, "visited": sum(reached), "of": len(stops)}
        if task == "reset_scene":
            step("load_robot", **self._last_load)
            return {"task": task}
        movable = [o for o in sim.objects if not sim.object_meta[o]["fixed"]]
        if task == "sort":
            slots = {"pad_a": 0, "pad_b": 0}
            s = 1.6 * sim.object_meta[movable[0]]["size"] if movable else 0.0  # a gap between neighbours
            offsets = [(-s, -s), (s, s), (s, -s), (-s, s)]
            moved = 0
            for obj in movable:
                pad = "pad_a" if sim.object_meta[obj]["color"] == "orange" else "pad_b"
                px, py = sim.zones[pad]
                ox, oy = offsets[slots[pad] % 4]
                slots[pad] += 1
                pos = sim.state()["objects"][obj]
                if abs(pos[0] - px) < sim.pad_half and abs(pos[1] - py) < sim.pad_half:
                    continue  # already on its pad
                self._move_object(step, obj, {"x": px + ox, "y": py + oy})
                moved += 1
            sim.wait(0.5)
            final = sim.state()["objects"]
            wrong = [o for o in movable
                     if not (abs(final[o][0] - sim.zones["pad_a" if sim.object_meta[o]["color"] == "orange" else "pad_b"][0]) < sim.pad_half
                             and abs(final[o][1] - sim.zones["pad_a" if sim.object_meta[o]["color"] == "orange" else "pad_b"][1]) < sim.pad_half)]
            return {"task": task, "success": not wrong, "moved": moved, "misplaced": wrong, "objects": final}
        # stack: one tower per arm on a two-armed robot (each hand on its own side), else one tower of three
        pads = ["pad_a", "pad_b"] if len(sim.arms) > 1 else ["pad_a"]
        towers = {}
        pool = list(movable)
        for pad in pads:
            px, py = sim.zones[pad]
            pos = sim.state()["objects"]
            mine = sorted(pool, key=lambda o: np.hypot(pos[o][0] - px, pos[o][1] - py))[: 3 if len(pads) == 1 else 2]
            pool = [o for o in pool if o not in mine]
            below = None
            for obj in mine:
                self._move_object(step, obj, {"x": px, "y": py} if below is None else {"on": below})
                below = obj
            towers[pad] = mine
        sim.wait(0.5)
        final = sim.state()["objects"]
        standing = {pad: all(final[t[i]][2] > final[t[i - 1]][2] + sim.object_meta[t[i]]["size"] for i in range(1, len(t)))
                    for pad, t in towers.items()}
        return {"task": task, "success": all(standing.values()), "towers_standing": standing, "towers": towers,
                "tower": next(iter(towers.values())), "objects": final}

    def _run_relay(self, source: str = "api", timeout: float = 600.0) -> dict:
        """AMR relay: picker A loads the carrier at station L, the carriers swap stations (the loaded one
        south, the empty one north, on separate lanes), picker B unloads at station U onto table B."""
        from anybot.team import RELAY_L, RELAY_NORTH_X, RELAY_SOUTH_X, RELAY_U

        scene = self.scene
        bots = {r.name: r for r in scene.robots}
        pa, pb = bots["picker_a"], bots["picker_b"]
        carriers = [bots["carrier_1"], bots["carrier_2"]]
        at = {"carrier_1": "L", "carrier_2": "U"}  # where each carrier is parked
        load: dict[str, str | None] = {"carrier_1": None, "carrier_2": None}  # crate in its tote
        crates = [o for o in scene.objects if o.startswith("crate")]
        delivered: list[str] = []
        errors: list[str] = []
        lock = threading.Lock()
        pb_busy = threading.Event()  # picker B is carrying a crate to table B: keep the station clear

        def step(robot, tool, **args):
            return self.call(tool, {**args, "robot": robot.name}, source)

        def base(r):
            return r.d.xpos[r._bid(r.profile.root_body)][:2].copy()

        def parked(where):
            return next((c for c in carriers if at[c.name] == where), None)

        def until(cond, who, limit=timeout):
            t0 = scene.d.time
            while not cond() and not errors and scene.d.time - t0 < limit:
                who.wait(0.5)
            return cond()

        def picker_a_job():
            for c in crates:
                step(pa, "pick_object", name=c)
                if not until(lambda: (k := parked("L")) is not None and load[k.name] is None, pa):
                    return
                k = parked("L")
                tote = base(k)
                step(pa, "place_object", x=float(tote[0]), y=float(tote[1]))
                with lock:
                    load[k.name] = c

        def picker_b_job():
            slots = scene.stations["table_b"]["slots"]
            for i in range(len(crates)):
                if not until(lambda: (k := parked("U")) is not None and load[k.name] is not None, pb):
                    return
                k = parked("U")
                c = load[k.name]
                pb_busy.set()
                try:
                    step(pb, "pick_object", name=c)
                    with lock:
                        load[k.name] = None
                    x, y = float(slots[i % len(slots)][0]), float(slots[i % len(slots)][1])
                    try:
                        step(pb, "place_object", x=x, y=y)
                    except SkillError:  # nudged on the way: try once more
                        step(pb, "place_object", x=x, y=y)
                finally:
                    pb_busy.clear()
                delivered.append(c)

        def drive(r, lane_x, a, b, where):
            """One carrier's trip: into its lane, along it, out to the other station."""
            def job():
                scene.begin_job()
                scene.join()
                try:
                    step(r, "navigate_to", x=lane_x, y=float(np.sign(a[1]) * 1.0))
                    step(r, "navigate_to", x=lane_x, y=float(np.sign(b[1]) * 1.0))
                    step(r, "navigate_to", x=float(b[0]), y=float(b[1]))
                    at[r.name] = where
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{r.name}: {e}")
                finally:
                    scene.leave()
            return threading.Thread(target=job, daemon=True)

        def dispatcher_job():
            for _ in range(len(crates)):
                # A full carrier at L and an empty one at U (or none there yet): swap them.
                ok = until(lambda: (k := parked("L")) is not None and load[k.name] is not None
                           and ((u := parked("U")) is None or load[u.name] is None) and not pb_busy.is_set(), carriers[0])
                if not ok:
                    return
                full, empty = parked("L"), parked("U")
                at[full.name] = "→U"
                trips = [drive(full, RELAY_SOUTH_X, RELAY_L, RELAY_U, "U")]
                if empty is not None:
                    at[empty.name] = "→L"
                    trips.append(drive(empty, RELAY_NORTH_X, RELAY_U, RELAY_L, "L"))
                scene.leave()  # the dispatcher only waits for the trips: it must not hold the clock
                for t in trips:
                    t.start()
                for t in trips:
                    t.join()
                scene.join()
                if errors:
                    return

        jobs = [(picker_a_job, "picker_a"), (picker_b_job, "picker_b"), (dispatcher_job, "dispatcher")]
        started = threading.Barrier(len(jobs) + 1)

        def guard(job, who):
            def run():
                scene.begin_job()
                scene.join()
                started.wait()
                try:
                    job()
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{who}: {e}")
                finally:
                    scene.leave()
            return run

        threads = [threading.Thread(target=guard(j, w), daemon=True) for j, w in jobs]
        for t in threads:
            t.start()
        started.wait()
        for t in threads:
            t.join()
        on_b = [c for c in crates if scene.d.xpos[pa._bid(c)][1] < -2.3 and scene.d.xpos[pa._bid(c)][2] > 0.6]
        return {"task": "relay", "success": not errors and len(on_b) == len(crates), "delivered": delivered,
                "on_table_b": on_b, "errors": errors, "sim_time_s": round(float(scene.d.time), 1)}

    def _run_production(self, source: str = "api", timeout: float = 400.0) -> dict:
        """Run the line with every robot working at the same time; each step shows up in the log."""
        scene = self.scene
        roles = {r.name: r for r in scene.robots}
        loader, sorter, carrier = roles.get("loader"), roles.get("sorter"), roles.get("carrier")
        crates = [o for o in scene.objects if o.startswith("crate")]
        belt = scene.conveyors[0]
        belt_in, belt_out = np.array(scene.zones["belt_in"]), np.array(scene.zones["belt_out"])
        sorted_into: list[str] = []
        errors: list[str] = []
        done_loading = threading.Event()

        def step(robot, tool, **args):
            return self.call(tool, {**args, "robot": robot.name}, source)

        def pos(o):
            return np.array(scene.robots[0].state()["objects"][o])

        def loader_job():
            for c in crates:
                step(loader, "pick_object", name=c)
                t0 = scene.d.time
                while any(np.linalg.norm(pos(o)[:2] - belt_in) < 0.14 for o in crates if o != c) and scene.d.time - t0 < 60:
                    loader.wait(0.5)  # the start of the belt is still occupied
                step(loader, "place_object", x=float(belt_in[0]), y=float(belt_in[1]))
            done_loading.set()

        def sorter_job():
            t0 = scene.d.time
            while len(sorted_into) < len(crates) and scene.d.time - t0 < timeout:
                ready = [o for o in crates if o not in sorted_into and abs((pos(o)[:2] - belt_out) @ belt.direction[:2]) < 0.03 and np.linalg.norm(pos(o)[:2] - belt_out) < 0.07
                         and abs(pos(o)[2] - (belt.centre[2] + 0.035)) < 0.03
                         and np.linalg.norm(scene.d.cvel[sorter._bid(o)][3:5]) < 0.01]  # stopped at the stopper
                if not ready:
                    sorter.wait(0.5)
                    continue
                box = ready[0]
                sorter.wait(0.6)  # let it settle against the stopper
                step(sorter, "pick_object", name=box)
                if carrier is not None:
                    tote = carrier.d.xpos[carrier._bid(carrier.profile.root_body)][:2]
                else:
                    tote = np.array(scene.zones["tote"])
                slot = [(-0.06, -0.06), (0.06, 0.06), (0.06, -0.06), (-0.06, 0.06)][len(sorted_into) % 4]
                step(sorter, "place_object", x=float(tote[0] + slot[0]), y=float(tote[1] + slot[1]))
                sorted_into.append(box)

        def carrier_job():
            t0 = scene.d.time
            while len(sorted_into) < len(crates) and scene.d.time - t0 < timeout and not errors:
                carrier.wait(0.5)
            if len(sorted_into) == len(crates):
                step(carrier, "go_to_zone", zone="dock")

        jobs = [(loader_job, "loader"), (sorter_job, "sorter")] + ([(carrier_job, "carrier")] if carrier else [])
        started = threading.Barrier(len(jobs) + 1)

        def guard(job, who):
            def run():
                scene.begin_job()
                scene.join()  # shared clock: time moves on only while every robot waits
                started.wait()
                try:
                    job()
                except Exception as e:  # noqa: BLE001 - one robot failing must not hang the others
                    errors.append(f"{who}: {e}")
                finally:
                    scene.leave()
            return run

        threads = [threading.Thread(target=guard(j, w), name=f"anybot-{w}", daemon=True) for j, w in jobs]
        for t in threads:
            t.start()
        started.wait()
        for t in threads:
            t.join(timeout)
        final = scene.robots[0].state()["objects"]
        delivered = None
        if carrier is not None:
            dock = np.array(scene.zones["dock"])
            cpos = carrier.d.xpos[carrier._bid(carrier.profile.root_body)][:2]
            in_tote = [o for o in crates if np.linalg.norm(np.array(final[o][:2]) - cpos) < 0.2 and final[o][2] > 0.15]
            delivered = {"at_dock": bool(np.linalg.norm(cpos - dock) < 0.4), "in_tote": in_tote}
        ok = not errors and len(sorted_into) == len(crates) and (delivered is None or (delivered["at_dock"] and len(delivered["in_tote"]) == len(crates)))
        return {"task": "production", "success": ok, "sorted": sorted_into, "delivered": delivered, "errors": errors,
                "sim_time_s": round(float(scene.d.time), 1), "objects": final}

    # ------------------------------------------------------------------ pointing in the 3D view

    def propose(self, x: float, y: float, z: float, body: str | None = None) -> dict:
        """What a click on this point would do, for the user to confirm: {tool, args, label}."""
        sim = self._need()
        skills = sim.skills()
        x, y, z = round(float(x), 3), round(float(y), 3), round(float(z), 3)
        zone = next((n for n, (zx, zy) in sim.zones.items()
                     if abs(x - zx) < max(sim.pad_half, 0.6 if n in ("zone_a", "zone_b", "dock") else 0)
                     and abs(y - zy) < max(sim.pad_half, 0.6 if n in ("zone_a", "zone_b", "dock") else 0)), None)
        where = zone.replace("_", " ").upper() if zone else "HERE"
        if sim.navigating:
            return {"tool": "retarget", "args": {"x": x, "y": y}, "label": f"CHANGE GOAL TO {where}"}
        is_object = body in sim.objects
        if sim.stations and not zone:  # a click on a table top
            zone = next((n for n, st in sim.stations.items()
                         if abs(x - st["xy"][0]) < 0.5 and abs(y - st["xy"][1]) < 0.3 and z > 0.4), None)
            where = zone.replace("_", " ").upper() if zone else where
        if "pick_object" in skills and sim.held:
            held = next(iter(sim.held.values()))
            if is_object and body not in sim.held.values():
                return {"tool": "place_object", "args": {"on": body}, "label": f"STACK {held} ON {body}".upper()}
            return {"tool": "place_object", "args": {"x": x, "y": y}, "label": f"PLACE {held.upper()} {'ON ' + where if zone else 'HERE'}"}
        if "pick_object" in skills and is_object and not sim.object_meta[body]["fixed"]:
            return {"tool": "pick_object", "args": {"name": body}, "label": f"PICK {body.upper()}"}
        if "navigate_to" in skills:
            verb = "WALK" if sim.gait is not None else "DRIVE"
            return {"tool": "navigate_to", "args": {"x": x, "y": y}, "label": f"{verb} {'TO ' + where if zone else 'HERE'}"}
        if "move_end_effector" in skills:
            return {"tool": "go_to_point", "args": {"x": x, "y": y, "z": z}, "label": "REACH HERE"}
        raise SkillError(f"{sim.name} ({sim.profile.category}) has nothing to do with a point in space.")

    def click(self, x: float, y: float, z: float, body: str | None = None, source: str = "ui") -> dict:
        """Propose and run in one go (the UI asks for confirmation in between)."""
        p = self.propose(x, y, z, body)
        return self.call(p["tool"], p["args"], source)

    def _wait(self, seconds: float = 1.0) -> dict:
        sim = self._need()
        sim.wait(seconds)
        return sim.state()

    def robot_info(self) -> dict:
        sim = self._need()
        p = sim.profile
        return {
            "summary": p.summary(),
            "name": p.name,
            "category": p.category,
            "base_pinned": sim.pinned,
            "arms": {n: {"joints": len(a.joints), "gripper": bool(a.gripper_actuators)} for n, a in sim.arms.items()},
            "wheels": p.wheels,
            "poses": sim.keyframes(),
            "skills": sim.skills(),
            "end_effectors_now": sim.state()["end_effectors"],
            "world": sim.world or "empty",
            "mass_kg": p.mass_kg, "height_m": p.height_m, "n_joints": len(p.joints), "n_actuators": len(p.actuators),
            "floating_base": p.floating_base,
            "notes": list(sim.import_notes),
            "zones": {n: [round(v, 3) for v in xy] for n, xy in sim.zones.items()},
            "tasks": self.tasks(),
            "objects": {o: {"color": sim.object_meta[o].get("color"), "pos": p} for o, p in sim.state()["objects"].items()},
            "team": None if self.scene is None else {
                "name": getattr(self.scene, "team", None),
                "robots": [{"name": r.name, "model": getattr(r, "model_name", r.name), "category": r.profile.category,
                            "busy": any(e.get("robot") == r.name and e["status"] == "running" for e in self.events[-40:])}
                           for r in self.scene.robots],
            },
        }

    # ------------------------------------------------------------------ dispatch + activity log

    def call(self, name: str, args: dict | None = None, source: str = "api") -> Any:
        """Run a tool by name. Raises SkillError with an LLM-readable message on failure."""
        fn = self.tools.get(name)
        if fn is None:
            raise SkillError(f"Unknown tool '{name}'. Tools: {', '.join(self.tools)}")
        if name == "stop":  # never waits for the job it stops
            ev = self._event(source=source, tool=name, args={})
            result = fn()
            self._finish(ev, "ok", "stopped " + ", ".join(result.get("was_running", [])) if result.get("stopped") else "nothing was running", time.perf_counter())
            return result
        args = dict(args or {})
        target = None
        if "robot" in args and name not in ("load_robot", "load_team"):
            who = args.pop("robot")
            if who is not None:
                if self.scene is None:
                    raise SkillError("'robot' only applies when a team is loaded (load_team).")
                try:
                    target = self.scene.robot(who)
                except KeyError:
                    raise SkillError(f"No robot '{who}' in the team. Robots: {', '.join(r.name for r in self.scene.robots)}") from None
        quiet = name in ("get_state", "describe_robot", "list_robots", "list_tasks", "get_scan")  # reads: don't spam the log
        if name == "run_task":
            args = {**args, "source": source}
        if name == "retarget":  # must not wait for the navigation it redirects
            ev = self._event(source=source, tool=name, args=args)
            try:
                result = fn(**args)
            except (SkillError, TypeError) as e:
                self._finish(ev, "error", str(e), time.perf_counter())
                raise SkillError(str(e)) from e
            self._finish(ev, "ok", "new goal", time.perf_counter())
            return result
        shown = {k: v for k, v in args.items() if k != "source"}
        prev = getattr(self._tls, "robot", None)
        acting = target or prev or self.sim
        team_level = name in ("load_team", "load_robot", "select_robot") or (name == "run_task" and self.scene is not None)
        extra = {"robot": acting.name} if self.scene is not None and acting is not None and not team_level else {}
        ev = None if quiet else self._event(source=source, tool=name, args=shown, **extra)
        lock = (self._robot_locks.setdefault(acting.name, threading.RLock()) if self.scene is not None and acting is not None
                and not team_level else self._run_lock if self.scene is None else threading.RLock())
        t0 = time.perf_counter()
        try:
            self._tls.robot = target or prev
            if (target or self.sim) is not None:
                (target or self.sim).scene.begin_job()  # the user may stop it
            with lock:
                self.busy += 1
                try:
                    result = fn(**args)
                finally:
                    self.busy -= 1
                    self._tls.robot = prev
        except TaskStopped:
            self._finish(ev, "stopped", "stopped by the user", t0)
            raise SkillError("Stopped by the user.") from None
        except TypeError as e:  # wrong/missing arguments
            self._finish(ev, "error", str(e), t0)
            raise SkillError(f"Bad arguments for {name}: {e}") from e
        except SkillError as e:
            self._finish(ev, "error", str(e), t0)
            raise
        except Exception as e:  # noqa: BLE001 - surface anything else to the caller, but keep serving
            self._finish(ev, "error", f"{type(e).__name__}: {e}", t0)
            raise SkillError(f"{name} failed: {type(e).__name__}: {e}") from e
        self._finish(ev, "ok", _summarize(name, result), t0)
        return result

    def _event(self, **fields) -> dict:
        ev = {"id": next(self._ids), "time": time.time(), "status": "running", "summary": "", "duration_s": None, **fields}
        with self._ev_lock:
            self.events.append(ev)
            del self.events[:-200]
            self.version += 1
        return ev

    def _finish(self, ev: dict | None, status: str, summary: str, t0: float) -> None:
        if ev is None:
            return
        with self._ev_lock:
            ev.update(status=status, summary=summary, duration_s=round(time.perf_counter() - t0, 2))
            self.version += 1

    def close(self) -> None:
        if self.sim is not None:
            self.sim.close()


def _summarize(name: str, result: Any) -> str:
    if isinstance(result, bytes):
        return "image"
    if name in ("load_robot", "set_world"):
        return result["summary"]
    if name == "move_end_effector":
        return f"reached {result['reached']} (error {result['error_m'] * 1000:.1f} mm)"
    if name == "spawn_object":
        return f"{result['name']} at {result['pos']}"
    if name == "drive":
        return f"{result['start']} → {result['end']}"
    if name in ("navigate_to", "go_to_point") and "reached" in result and "goal" in result:
        status = "reached" if result["reached"] else f"stopped {result['distance_left_m']} m short"
        return f"{status} {result['goal']} in {result['time_s']} s"
    if name == "go_to_point" and result.get("mode") == "reach":
        return f"gripper above {result['reached']}"
    if name == "gripper":
        return f"gripper {result['gripper']}"
    if name == "load_team":
        return "team: " + ", ".join(f"{r['name']} ({r['model']})" for r in result["team"]["robots"])
    if name == "run_task" and result.get("task") == "production":
        d = result.get("delivered") or {}
        return (f"{len(result['sorted'])} boxes sorted" + (f", {len(d.get('in_tote', []))} delivered to the dock" if d else "")
                + (f"; problems: {'; '.join(result['errors'])}" if result["errors"] else ""))
    if name == "run_task":
        if result.get("success") is False:
            what = result.get("misplaced") or [p for p, ok in result.get("towers_standing", {}).items() if not ok]
            return f"finished, but not all went right: {', '.join(map(str, what))}"
        return "task complete"
    if name == "go_to_zone" and "reached" in result and "goal" in result:
        return f"{'reached' if result['reached'] else 'did not reach'} {result['zone']} in {result['time_s']} s"
    if name == "pick_object":
        return f"holding {result['picked']} with {result['arm']}"
    if name == "place_object":
        where = f"on {result['on']}" if result["on"] else f"at {result['object_at'][:2]}"
        return f"placed {result['placed']} {where}"
    return "done"
