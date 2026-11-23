"""Teams: several robots composed into one MuJoCo scene, plus a small production line to run.

The line, inside the warehouse:

    table A (crates) ──loader──▶ conveyor ═══════▶ stopper ──sorter──▶ tote on the carrier ──▶ dock

* loader  — a mobile robot with an arm (AMR + arm, humanoid): table A → conveyor start;
* sorter  — a fixed arm on a pedestal at the conveyor end: conveyor → tote;
* carrier — a mobile robot without an arm (AMR, quadruped) with a tote on its back: delivers to the dock.

Every robot is a normal `Sim` sharing one `Scene`, so all skills (pick, place, navigate...) work as
they do for a single robot, and the robots can act at the same time.
"""

from __future__ import annotations

import warnings

import mujoco
import numpy as np

from anybot import worlds
from anybot.loader import load_robot
from anybot.scene import Conveyor, Scene
from anybot.sim import Sim

# Production line layout (warehouse coordinates).
BELT_Y, BELT_X0, BELT_X1, BELT_TOP = 0.75, -0.7, 1.1, 0.7
SORTER_XY = (1.62, 0.75)
TOTE_SPOT = (1.62, 0.13)  # within easy reach of the sorter, clear of its pedestal

# AMR relay: two stations by the tables, two lanes between them (one each way, so carriers pass).
RELAY_L, RELAY_U = (1.1, 1.75), (1.1, -1.75)
RELAY_SOUTH_X, RELAY_NORTH_X = 1.6, 0.6

TEAMS = {
    "amr_line": {
        "label": "AMR line",
        "description": "Tidybot loads the conveyor, a Panda sorts into the tote on a carrier AMR.",
        "members": [
            {"role": "loader", "robot": "tidybot", "xy": (0.0, 1.75), "yaw": 90},
            {"role": "sorter", "robot": "panda", "xy": SORTER_XY, "yaw": 180, "pedestal": True},
            {"role": "carrier", "robot": "amr", "xy": TOTE_SPOT, "yaw": 180},
        ],
    },
    "amr_fleet": {
        "label": "AMR relay",
        "description": "Two Tidybots load and unload, two AMRs carry the crates and swap places on every trip.",
        "line": False,
        "members": [
            {"role": "picker_a", "robot": "tidybot", "xy": (-1.0, 1.6), "yaw": 90},
            {"role": "picker_b", "robot": "tidybot", "xy": (-1.0, -1.6), "yaw": -90},
            {"role": "carrier_1", "robot": "amr", "xy": RELAY_L, "yaw": 90, "tote": True},
            {"role": "carrier_2", "robot": "amr", "xy": RELAY_U, "yaw": -90, "tote": True},
        ],
    },
    "mixed_line": {
        "label": "Humanoid + dog line",
        "description": "A G1 humanoid loads the conveyor, a UR5e sorts into the tote on a Go2.",
        "members": [
            {"role": "loader", "robot": "g1", "xy": (0.0, 1.75), "yaw": 90},
            {"role": "sorter", "robot": "ur5e", "xy": SORTER_XY, "yaw": 180, "pedestal": True},
            {"role": "carrier", "robot": "go2", "xy": (TOTE_SPOT[0], 0.36), "yaw": 180},
        ],
    },
}

_BOX = mujoco.mjtGeom.mjGEOM_BOX
REST_KEYS = ("retract", "rest", "stow", "tuck", "transport")


def _yaw_quat(deg: float) -> np.ndarray:
    a = np.radians(deg) / 2
    return np.array([np.cos(a), 0.0, 0.0, np.sin(a)])


def _line(spec: mujoco.MjSpec, with_bin: bool) -> None:
    """Conveyor with side rails and an end stopper, a pedestal for the sorter, a static bin if no carrier."""
    cx, half = (BELT_X0 + BELT_X1) / 2, (BELT_X1 - BELT_X0) / 2
    # Narrow belt: crates travel in single file. Low rails: grippers reach over them.
    worlds._solid(spec, "line_belt", [cx, BELT_Y, BELT_TOP - 0.02], [half, 0.06, 0.02], [0.17, 0.18, 0.2, 1])
    worlds._solid(spec, "line_frame", [cx, BELT_Y, (BELT_TOP - 0.04) / 2], [half, 0.05, (BELT_TOP - 0.04) / 2], worlds.STEEL)
    for side in (-1, 1):
        worlds._solid(spec, f"line_rail{side}", [cx, BELT_Y + side * 0.068, BELT_TOP + 0.006], [half, 0.006, 0.014], worlds.SLATE)
    worlds._solid(spec, "line_stopper", [BELT_X1 + 0.012, BELT_Y, BELT_TOP + 0.02], [0.012, 0.074, 0.04], worlds.ORANGE)
    worlds._solid(spec, "line_pedestal", [SORTER_XY[0], SORTER_XY[1], 0.2], [0.17, 0.17, 0.2], worlds.STEEL)
    worlds._pad(spec, "line_tote_spot", TOTE_SPOT, 0.5, worlds.BLUE)
    if with_bin:
        _tote(spec.worldbody, [TOTE_SPOT[0], TOTE_SPOT[1], 0.0], "line_bin", stand=0.35)


def _tote(body, base: list[float], name: str, stand: float = 0.0, mass: float = 0.3) -> None:
    """An open box (floor + 4 walls) whose floor sits at base z + stand."""
    x, y, z = base
    floor = z + stand
    kw = dict(contype=1, conaffinity=1, rgba=[0.15, 0.35, 0.85, 1])
    if stand > 0:
        body.add_geom(name=f"{name}_stand", type=_BOX, pos=[x, y, z + stand / 2], size=[0.14, 0.14, stand / 2],
                      rgba=list(worlds.STEEL), contype=1, conaffinity=1)
    body.add_geom(name=f"{name}_floor", type=_BOX, pos=[x, y, floor + 0.005], size=[0.16, 0.16, 0.005], mass=mass / 5, **kw)
    for i, (dx, dy, hx, hy) in enumerate(((0.155, 0, 0.005, 0.16), (-0.155, 0, 0.005, 0.16), (0, 0.155, 0.16, 0.005), (0, -0.155, 0.16, 0.005))):
        body.add_geom(name=f"{name}_wall{i}", type=_BOX, pos=[x + dx, y + dy, floor + 0.05], size=[hx, hy, 0.045],
                      mass=mass / 5, **kw)


def build_team(team: str, realtime: bool = True) -> Scene:
    if team not in TEAMS:
        raise ValueError(f"Unknown team '{team}'. Teams: {', '.join(TEAMS)}")
    members = TEAMS[team]["members"]
    has_line = TEAMS[team].get("line", True)
    has_carrier = any(m["role"] == "carrier" for m in members)

    spec = mujoco.MjSpec()
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.worldbody.add_geom(name="anybot_floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
                            contype=0xFFFF, conaffinity=0xFFFF)
    spec.worldbody.add_light(pos=[0, 0, 3], dir=[0.2, 0.1, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                             diffuse=[0.45, 0.45, 0.45], specular=[0.1, 0.1, 0.1])
    worlds.style_scene(spec)
    worlds.build_warehouse(spec)
    if has_line:
        _line(spec, with_bin=not has_carrier)
    else:  # the relay's two stations
        for i, xy in enumerate((RELAY_L, RELAY_U)):
            worlds._pad(spec, f"relay_station_{i}", xy, 0.45, worlds.BLUE)
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 720

    loaded = []
    for mem in members:
        robot = load_robot(mem["robot"])
        child = robot.spec
        for g in list(child.worldbody.geoms):
            if g.name == "anybot_floor":
                child.delete(g)
        for light in list(child.worldbody.lights):
            child.delete(light)
        # Home pose from the robot's own keyframe, by joint / actuator name (scene keys are not used).
        cm = robot.model
        cd = mujoco.MjData(cm)
        if cm.nkey:
            mujoco.mj_resetDataKeyframe(cm, cd, 0)
        home_q = {cm.joint(j).name: cd.qpos[cm.jnt_qposadr[j]:cm.jnt_qposadr[j] + {0: 7, 1: 4}.get(int(cm.jnt_type[j]), 1)].copy()
                  for j in range(cm.njnt)}
        home_u = {cm.actuator(a).name: float(cd.ctrl[a]) for a in range(cm.nu)}
        # A folded "rest" pose for moving around, if the model has one (kept by joint name: keys are dropped).
        rest_q = {}
        for k in range(cm.nkey):
            if cm.key(k).name.lower() in REST_KEYS:
                rest_q = {cm.joint(j).name: float(cm.key_qpos[k][cm.jnt_qposadr[j]]) for j in range(cm.njnt)
                          if int(cm.jnt_type[j]) in (2, 3)}
                break
        root_name = child.worldbody.first_body().name
        if mem["role"] == "carrier" or mem.get("tote"):  # the tote rides on the carrier's back
            cd2 = mujoco.MjData(cm)
            cd2.qpos[:] = cd.qpos
            mujoco.mj_forward(cm, cd2)
            from anybot.introspect import body_geoms, world_aabb

            rb = mujoco.mj_name2id(cm, mujoco.mjtObj.mjOBJ_BODY, root_name)
            top = float(world_aabb(cm, cd2, body_geoms(cm, [rb]))[1][2] - cd2.xpos[rb][2])
            _tote(child.worldbody.first_body(), [0.0, 0.0, top + 0.002], "tote")
        for k in list(child.keys):
            child.delete(k)
        lift = 0.4 if mem.get("pedestal") else 0.0
        frame = spec.worldbody.add_frame(pos=[mem["xy"][0], mem["xy"][1], lift], quat=list(_yaw_quat(mem["yaw"])))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spec.attach(child, prefix=f"{mem['role']}/", frame=frame)
        loaded.append((mem, robot, root_name, home_q, home_u, rest_q))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # children's solver options differ: the scene's settings win
        model = spec.compile()
    scene = Scene(spec, model, world="warehouse")
    scene.ensure_data()
    m, d = scene.m, scene.d
    homes = []
    for mem, robot, root_name, home_q, home_u, rest_q in loaded:
        pre = f"{mem['role']}/"
        qmap: dict[int, float] = {}
        R = np.array([[np.cos(np.radians(mem["yaw"])), -np.sin(np.radians(mem["yaw"])), 0],
                      [np.sin(np.radians(mem["yaw"])), np.cos(np.radians(mem["yaw"])), 0], [0, 0, 1]])
        offset = np.array([mem["xy"][0], mem["xy"][1], 0.4 if mem.get("pedestal") else 0.0])
        for jname, val in home_q.items():
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, pre + jname)
            if j < 0:
                continue
            adr = m.jnt_qposadr[j]
            if int(m.jnt_type[j]) == 0:  # free joint: the keyframe pose, moved to where the robot stands
                pos = offset + R @ val[:3]
                quat = np.zeros(4)
                mujoco.mju_mulQuat(quat, _yaw_quat(mem["yaw"]), val[3:7])
                for i, v in enumerate([*pos, *quat]):
                    qmap[adr + i] = float(v)
            else:
                for i, v in enumerate(val):
                    qmap[adr + i] = float(v)
        umap = {}
        for aname, val in home_u.items():
            a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, pre + aname)
            if a >= 0:
                umap[a] = val
        for adr, v in qmap.items():
            d.qpos[adr] = v
        rest = {}
        for jname, val in rest_q.items():
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, pre + jname)
            if j >= 0:
                rest[j] = val
        homes.append({"qpos": qmap, "ctrl": umap, "rest": rest})
    mujoco.mj_forward(m, d)

    scene.building = True  # robots settle once all of them exist (an unconfigured robot would collapse meanwhile)
    for (mem, robot, root_name, _, _, _), home in zip(loaded, homes):
        root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{mem['role']}/{root_name}")
        sim = Sim.in_scene(scene, mem["role"], f"{mem['role']}/", root, robot.injected_actuators, robot.notes, home)
        sim.role, sim.model_name = mem["role"], mem["robot"]
    scene.building = False

    # Shared places and the conveyor.
    if has_line:
        scene.conveyors.append(Conveyor("belt", np.array([(BELT_X0 + BELT_X1) / 2, BELT_Y, BELT_TOP]),
                                        np.array([1.0, 0.0]), (BELT_X1 - BELT_X0) / 2, 0.06))
        belt = scene.conveyors[0]
        scene.zones.update({
            "belt_in": [float(belt.start[0]), float(belt.start[1])], "belt_out": [float(belt.end[0]), float(belt.end[1])],
            "tote": [float(v) for v in next((m["xy"] for m in members if m["role"] == "carrier"), TOTE_SPOT)],
        })
    else:
        scene.zones.update({"station_l": list(RELAY_L), "station_u": list(RELAY_U)})
    scene.zones.update({n: list(z["xy"]) for n, z in worlds.FLOOR_ZONES.items()})
    scene.stations = {n: {"slots": worlds.table_slots(n), "top": worlds.TABLE_TOP, "xy": list(worlds.TABLES[n])} for n in worlds.TABLES}
    first = scene.robots[0]
    for i, (xy, color) in enumerate(zip(worlds.table_slots("table_a"), ("orange", "blue", "yellow"))):
        first.spawn_object("box", 0.035, [xy[0], xy[1], worlds.TABLE_TOP + 0.037], color, name=f"crate_{i + 1}", recompile=False)
    scene.recompile()
    for r in scene.robots:
        r.settle()
    scene.team = team
    scene.start(realtime=realtime)
    return scene
