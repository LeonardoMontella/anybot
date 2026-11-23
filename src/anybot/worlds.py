"""Worlds: a scene built around whatever robot was loaded.

`warehouse` is a small logistics hall: racks with crates along the sides, pallets, floor zones
(ZONE A, ZONE B, DOCK) and a marked robot cell at the origin. What goes *inside the cell* depends
on the robot, so every robot gets something useful to do:

  * fixed arm with a gripper → a workstation: boxes to handle and two drop pads within its reach;
  * walking / driving robot  → an open floor with the three zones as navigation goals.
"""

from __future__ import annotations

import zlib
from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    from anybot.sim import Sim

WORLDS = ["warehouse", "lab", "apartment", "bench", "empty"]
SCAN_SPOTS: dict[str, list[float]] = {}  # named floor spots of the last scanned world built

ORANGE = [1.0, 0.31, 0.0, 1]
SLATE = [0.36, 0.47, 0.6, 1]
WOOD = [0.78, 0.66, 0.5, 1]
WHITE = [0.96, 0.96, 0.95, 1]
BLUE = [0.1, 0.35, 0.9, 1]
YELLOW = [1.0, 0.78, 0.0, 1]
INK = [0.13, 0.14, 0.16, 1]
STEEL = [0.34, 0.37, 0.41, 1]
CARDBOARD = [0.74, 0.58, 0.4, 1]
PAPER = [0.93, 0.92, 0.89, 1]

# Navigation goals for mobile robots (the lines between them and the origin are kept free).
FLOOR_ZONES = {
    "zone_a": {"xy": [3.0, 2.0], "rgba": ORANGE},
    "zone_b": {"xy": [3.0, -2.0], "rgba": BLUE},
    "dock": {"xy": [-2.5, 0.0], "rgba": YELLOW},
}
_BOX, _PLANE = mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_PLANE

# Two work tables, reachable in a straight line from the robot cell. Objects start on A.
TABLE_TOP = 0.72
TABLES = {"table_a": [0.0, 2.7], "table_b": [0.0, -2.7]}
_SLOTS_X = (-0.33, 0.0, 0.33)  # wide apart: a hand needs room next to the object


def table_slots(name: str) -> list[list[float]]:
    """Places for objects on a table, along its front edge (the side facing the robot cell)."""
    x, y = TABLES[name]
    front = y - 0.17 * np.sign(y)
    return [[x + dx, float(front)] for dx in _SLOTS_X]


def _table(spec, name, xy, top_rgba, leg_rgba):
    x, y = xy
    _solid(spec, f"{name}_top", [x, y, TABLE_TOP - 0.02], [0.5, 0.3, 0.02], top_rgba)
    for i, (sx, sy) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1))):
        _solid(spec, f"{name}_leg{i}", [x + sx * 0.45, y + sy * 0.25, (TABLE_TOP - 0.04) / 2], [0.025, 0.025, (TABLE_TOP - 0.04) / 2], leg_rgba)



def style_scene(spec: mujoco.MjSpec) -> None:
    """Light studio look for every scene: gridded floor (1 m tiles) and a soft backdrop."""
    names = {t.name for t in spec.textures}
    if "anybot_grid" in names:
        return
    spec.add_texture(
        name="anybot_grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        width=512, height=512, rgb1=[0.86, 0.86, 0.84], rgb2=[0.86, 0.86, 0.84],
        mark=mujoco.mjtMark.mjMARK_EDGE, markrgb=[0.7, 0.7, 0.68],
    )
    mat = spec.add_material(name="anybot_floor_mat", texrepeat=[1, 1], texuniform=True, reflectance=0.03)
    tex = list(mat.textures)
    tex[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "anybot_grid"
    mat.textures = tex
    if not any(t.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX for t in spec.textures):
        spec.add_texture(
            name="anybot_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
            width=256, height=256, rgb1=[0.95, 0.94, 0.92], rgb2=[0.8, 0.8, 0.78],
        )
    for g in spec.worldbody.geoms:
        if g.type == _PLANE:
            g.material, g.rgba = "anybot_floor_mat", [0.8, 0.8, 0.8, 1]
    spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
    spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
    spec.visual.headlight.specular = [0.05, 0.05, 0.05]


def _solid(spec, name, pos, half, rgba, collide=True):
    bits = 1 if collide else 0
    spec.worldbody.add_geom(name=name, type=_BOX, pos=pos, size=half, rgba=rgba, contype=bits, conaffinity=bits)


def _pad(spec, name, xy, half, rgba):
    """A floor marking: coloured square with a dark frame, no collision."""
    _solid(spec, f"{name}_frame", [xy[0], xy[1], 0.0010], [half * 1.08, half * 1.08, 0.0010], INK, collide=False)
    _solid(spec, name, [xy[0], xy[1], 0.0022], [half, half, 0.0012], rgba, collide=False)


def _rack(spec, tag, x, y):
    """A 2 m wide pallet rack with three levels of crates."""
    w, dpt, h = 1.0, 0.3, 2.0
    for i, (sx, sy) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1))):
        _solid(spec, f"{tag}_post{i}", [x + sx * w, y + sy * dpt, h / 2], [0.03, 0.03, h / 2], STEEL)
    rng = np.random.default_rng(zlib.crc32(tag.encode()))  # same layout on every run
    for lvl, z in enumerate((0.12, 0.8, 1.48)):
        _solid(spec, f"{tag}_beam{lvl}", [x, y, z], [w, dpt, 0.025], SLATE)
        cx = x - w + 0.1
        k = 0
        while cx < x + w - 0.3:
            half = [float(rng.uniform(0.14, 0.24)), float(rng.uniform(0.16, 0.24)), float(rng.uniform(0.1, 0.22))]
            color = [CARDBOARD, CARDBOARD, PAPER, STEEL][int(rng.integers(0, 4))]
            _solid(spec, f"{tag}_crate{lvl}_{k}", [cx + half[0], y, z + 0.025 + half[2]], half, color)
            cx += 2 * half[0] + float(rng.uniform(0.04, 0.14))
            k += 1


def _pallet(spec, tag, x, y, layers):
    _solid(spec, f"{tag}_base", [x, y, 0.06], [0.6, 0.4, 0.06], [0.55, 0.42, 0.28, 1])
    z = 0.12
    for i in range(layers):
        _solid(spec, f"{tag}_load{i}", [x, y, z + 0.2], [0.52 - 0.04 * i, 0.34, 0.2], [CARDBOARD, PAPER][i % 2])
        z += 0.4


def build_warehouse(spec: mujoco.MjSpec) -> None:
    """Static shell of the warehouse (everything except what depends on the robot)."""
    for i, x in enumerate((-3.3, -1.1, 1.1, 3.3, 5.5)):
        _rack(spec, f"wh_rack_l{i}", x, 4.3)
        _rack(spec, f"wh_rack_r{i}", x, -4.3)
    _solid(spec, "wh_wall_left", [1.0, 5.0, 1.5], [7.0, 0.05, 1.5], PAPER)
    _solid(spec, "wh_wall_back", [-6.0, 0.0, 1.5], [0.05, 5.0, 1.5], PAPER)
    _solid(spec, "wh_wall_stripe", [-5.94, 0.0, 1.1], [0.01, 5.0, 0.08], SLATE, collide=False)
    _table(spec, "wh_table_a", TABLES["table_a"], STEEL, INK)
    _table(spec, "wh_table_b", TABLES["table_b"], STEEL, INK)
    _pallet(spec, "wh_pallet0", 5.6, 1.4, 2)
    _pallet(spec, "wh_pallet1", 5.6, -1.4, 3)
    _pallet(spec, "wh_pallet2", -4.6, 2.6, 1)
    _pallet(spec, "wh_pallet3", -4.6, -2.6, 2)
    for name, zone in FLOOR_ZONES.items():
        _pad(spec, f"wh_{name}", zone["xy"], 0.6, zone["rgba"])
    # Robot cell: safety line around the origin.
    for i, (px, py, hx, hy) in enumerate(((0, 1.2, 1.2, 0.02), (0, -1.2, 1.2, 0.02), (1.2, 0, 0.02, 1.2), (-1.2, 0, 0.02, 1.2))):
        _solid(spec, f"wh_cell{i}", [px, py, 0.0012], [hx, hy, 0.0012], INK, collide=False)
    spec.worldbody.add_light(pos=[2, 0, 5], dir=[0, 0, -1], diffuse=[0.08, 0.08, 0.08], specular=[0, 0, 0])


def build_lab(spec: mujoco.MjSpec) -> None:
    """A bright robotics lab: two white tables, cabinets, a sofa corner and the same three floor zones."""
    for g in spec.worldbody.geoms:
        if g.type == _PLANE:
            g.rgba = [0.82, 0.77, 0.7, 1]  # warm tint over the grid
    _solid(spec, "lab_wall_left", [0.5, 4.0, 1.4], [5.5, 0.05, 1.4], WHITE)
    _solid(spec, "lab_wall_back", [-5.0, 0.0, 1.4], [0.05, 4.0, 1.4], WHITE)
    _solid(spec, "lab_skirting", [-4.94, 0.0, 0.06], [0.01, 4.0, 0.06], SLATE, collide=False)
    _solid(spec, "lab_board", [-4.93, -1.2, 1.5], [0.01, 1.1, 0.6], [0.99, 0.99, 0.99, 1], collide=False)
    _solid(spec, "lab_board_frame", [-4.94, -1.2, 1.5], [0.01, 1.14, 0.64], STEEL, collide=False)
    _table(spec, "lab_table_a", TABLES["table_a"], WHITE, WOOD)
    _table(spec, "lab_table_b", TABLES["table_b"], WHITE, WOOD)
    for i, x in enumerate((-3.2, -1.9, 2.2, 3.5)):  # cabinets along the left wall
        _solid(spec, f"lab_cabinet{i}", [x, 3.62, 0.5 + 0.35 * (i % 2)], [0.6, 0.3, 0.5 + 0.35 * (i % 2)], [WOOD, WHITE][i % 2])
        _solid(spec, f"lab_cabinet{i}_handle", [x, 3.31, 0.7], [0.12, 0.008, 0.012], STEEL, collide=False)
    _solid(spec, "lab_sofa_seat", [-4.2, 2.2, 0.22], [0.45, 0.9, 0.22], SLATE)
    _solid(spec, "lab_sofa_back", [-4.55, 2.2, 0.55], [0.12, 0.9, 0.33], SLATE)
    _solid(spec, "lab_rug", [-2.8, 2.2, 0.003], [0.9, 1.1, 0.003], [0.87, 0.85, 0.8, 1], collide=False)
    _solid(spec, "lab_plant_pot", [-4.4, -3.3, 0.2], [0.18, 0.18, 0.2], [0.82, 0.5, 0.36, 1])
    _solid(spec, "lab_plant", [-4.4, -3.3, 0.75], [0.12, 0.12, 0.35], [0.3, 0.55, 0.36, 1], collide=False)
    _solid(spec, "lab_desk", [4.6, -2.6, 0.36], [0.4, 0.8, 0.36], WOOD)
    for name, zone in FLOOR_ZONES.items():
        _pad(spec, f"lab_{name}", zone["xy"], 0.6, zone["rgba"])
    # Rooms: a central corridor (robot start, dock), the workshop with table A to the north, the office with
    # table B to the south, the storage room with zones A and B to the east. One door into each room,
    # never in line with the goal, so every trip needs a planned path.
    for name, y in (("north", LAB_WALL_Y), ("south", -LAB_WALL_Y)):
        _wall(spec, f"lab_wall_{name}", (-5.0, y), (LAB_WALL_X, y), doors=[LAB_DOORS[name]])
    _wall(spec, "lab_wall_east", (LAB_WALL_X, -4.0), (LAB_WALL_X, 4.0), doors=[LAB_DOORS["east"]])
    _solid(spec, "lab_wall_right", [6.0, 0.0, LAB_WALL_H / 2], [0.05, 4.0, LAB_WALL_H / 2], WHITE)
    _solid(spec, "lab_wall_front", [0.5, -4.0, LAB_WALL_H / 2], [5.5, 0.05, LAB_WALL_H / 2], WHITE)
    spec.worldbody.add_light(pos=[1, 0, 4.5], dir=[0, 0, -1], diffuse=[0.08, 0.08, 0.08], specular=[0, 0, 0])


LAB_WALL_X, LAB_WALL_Y, LAB_WALL_H = 1.6, 1.2, 1.1
LAB_DOORS = {"north": (-2.2, -1.2), "south": (-2.2, -1.2), "east": (-0.5, 0.5)}  # openings along each wall


def _wall(spec, name, a, b, doors=(), thick=0.05):
    """An interior wall from a to b (axis aligned) with door openings [(start, end), ...] along it."""
    horizontal = a[1] == b[1]
    lo, hi = (a[0], b[0]) if horizontal else (a[1], b[1])
    cuts = [lo] + [v for d in sorted(doors) for v in d] + [hi]
    for i in range(0, len(cuts), 2):
        s0, s1 = cuts[i], cuts[i + 1]
        if s1 - s0 < 0.05:
            continue
        mid, half = (s0 + s1) / 2, (s1 - s0) / 2
        pos = [mid, a[1], LAB_WALL_H / 2] if horizontal else [a[0], mid, LAB_WALL_H / 2]
        size = [half, thick, LAB_WALL_H / 2] if horizontal else [thick, half, LAB_WALL_H / 2]
        _solid(spec, f"{name}{i // 2}", pos, size, WHITE)
        _solid(spec, f"{name}{i // 2}_trim", [pos[0], pos[1], LAB_WALL_H + 0.005],
               [size[0] + 0.002, size[1] + 0.002, 0.005], SLATE, collide=False)


def build_bench(spec: mujoco.MjSpec) -> None:
    """A quiet manipulation cell: plain walls, soft light. The work table is added per robot (its height
    depends on the robot), see _bench."""
    _solid(spec, "bench_wall_back", [1.6, 0.0, 1.2], [0.05, 2.0, 1.2], WHITE)
    _solid(spec, "bench_wall_left", [0.0, 2.0, 1.2], [1.6, 0.05, 1.2], WHITE)
    _solid(spec, "bench_stripe", [1.54, 0.0, 0.9], [0.01, 2.0, 0.03], SLATE, collide=False)
    spec.worldbody.add_light(pos=[0.6, 0, 2.6], dir=[0, 0, -1], diffuse=[0.1, 0.1, 0.1], specular=[0, 0, 0])


def all_worlds() -> list[str]:
    """Built-in worlds plus the places imported with `anybot import-scene`."""
    from anybot import scan

    return WORLDS + [w for w in scan.captured() if w not in WORLDS]


def is_scan(world: str | None) -> bool:
    """A world made from a 3D scan of a real place (rooms and walls; robots navigate it with the map)."""
    from anybot import scan

    return world == "apartment" or (world is not None and world not in WORLDS and world in scan.captured())


def build_scan_world(spec: mujoco.MjSpec, world: str) -> None:
    """A digital twin: a real scanned place (see anybot.scan), its own walls and furniture as obstacles."""
    from anybot import scan

    folder = scan.folder_for(world)
    meta = scan.build_scan(spec, folder)
    SCAN_SPOTS.clear()
    SCAN_SPOTS.update(meta["spots"])
    for g in spec.worldbody.geoms:
        if g.type == _PLANE:
            g.group = 3  # hidden: the scan has its own floor (the plane still collides)
    spec.worldbody.add_light(pos=[0, 0, 2.4], dir=[0, 0, -1], diffuse=[0.25, 0.25, 0.25], specular=[0, 0, 0])


def populate(sim: "Sim") -> None:
    """Fill the robot cell for this robot and register the zones it can use."""
    can_pick = list(sim.arms.values()) if "pick_object" in sim.skills() else []
    fixed = not sim.profile.floating_base or sim.pinned
    if sim.world == "bench":
        if can_pick and fixed:  # the bench is a fixed manipulation cell
            _bench(sim)
        return
    if is_scan(sim.world):
        if "navigate_to" in sim.skills():
            sim.zones = dict(SCAN_SPOTS)
            _fit_start(sim)
        return
    if fixed and can_pick and sim.profile.category in ("arm", "dual_arm"):
        _workstation(sim, can_pick[0])
    elif "navigate_to" in sim.skills():
        sim.zones = {name: list(z["xy"]) for name, z in FLOOR_ZONES.items()}
        if "pick_object" in sim.skills() and sim.profile.height_m > 1.0:  # tall enough to work at a table
            _tables(sim)


def _fit_start(sim: "Sim") -> None:
    """The scan's start spot is sized for an average robot: a big one (Spot is 1.1 m long) would appear with
    its legs inside a wall and be thrown over. Move it, and its keyframes (reset), to the nearest spot where
    its whole footprint (a rectangle around the base, plus a margin) is clear of the map."""
    from anybot.introspect import body_geoms, world_aabb

    planner = sim.scene.navmap()
    if planner is None or not sim.profile.floating_base or sim.pinned:
        return
    m, d = sim.m, sim.d
    root = sim._bid(sim.profile.root_body)
    pos, yaw = sim._base_pose()
    lo, hi = world_aabb(m, d, body_geoms(m, sim._subtree(root)))
    c, s_ = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, s_], [-s_, c]])  # world → base (xy)
    corners = np.array([[lo[0], lo[1]], [lo[0], hi[1]], [hi[0], lo[1]], [hi[0], hi[1]]]) - pos[:2]
    loc = corners @ R.T
    half = np.max(np.abs(loc), axis=0) + 0.1  # half length, half width (+ margin)
    occ = np.argwhere(planner.static)
    pts = np.array([planner._xy(rc) for rc in occ]) if len(occ) else np.zeros((0, 2))

    def fits(xy):
        if not len(pts):
            return True
        q = (pts - xy) @ R.T
        return not np.any((np.abs(q[:, 0]) < half[0]) & (np.abs(q[:, 1]) < half[1]))

    if fits(pos[:2]):
        return
    for r in np.arange(0.05, 2.0, 0.05):  # nearest first: rings of candidates around the start
        ring = [pos[:2] + r * np.array([np.cos(a), np.sin(a)]) for a in np.linspace(0, 2 * np.pi, int(12 + 40 * r), endpoint=False)]
        good = [xy for xy in ring if fits(xy)]
        if good:
            new = good[0]
            break
    else:
        return
    adr = int(m.jnt_qposadr[m.body_jntadr[root]])
    shift = new - pos[:2]
    d.qpos[adr:adr + 2] += shift
    for k in range(m.nkey):
        m.key_qpos[k][adr:adr + 2] += shift
    mujoco.mj_forward(m, d)


def _bench(sim: "Sim") -> None:
    """A work table in front of the robot at hand height, two trays, small cubes: dexterous practice."""
    m, d = sim.m, sim.d
    arms = list(sim.arms.values())
    root = sim._bid(sim.profile.root_body)
    base = d.xpos[root].copy()
    yaw = float(np.arctan2(d.xmat[root].reshape(3, 3)[1, 0], d.xmat[root].reshape(3, 3)[0, 0]))
    fwd, left = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
    shoulders = np.array([d.xpos[m.jnt_bodyid[a.joints[0]]] for a in arms])
    reach = float(np.mean([np.linalg.norm(sim._tcp(a, d) - d.xpos[m.jnt_bodyid[a.joints[0]]]) for a in arms]))
    top = float(np.clip(np.mean(shoulders[:, 2]) - 0.45 * reach - 0.12, 0.3, 1.1))  # below the shoulders
    near = float(np.mean((shoulders[:, :2] - base[:2]) @ fwd)) + 0.18
    depth, width = 0.22, max(0.35, 0.9 * float(np.ptp(shoulders[:, :2] @ left)) + 0.3)
    centre = base[:2] + fwd * (near + depth)
    for i, (part, pos, half) in enumerate((
        ("top", [centre[0], centre[1], top - 0.02], [depth, width, 0.02]),
    )):
        sim.spec.worldbody.add_geom(name=f"bench_table_{part}", type=_BOX, pos=pos, size=half, rgba=WHITE,
                                    euler=[0, 0, float(np.degrees(yaw))], contype=1, conaffinity=1)
    for i, (sx, sy) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1))):
        xy = centre + fwd * sx * (depth - 0.03) + left * sy * (width - 0.03)
        _solid(sim.spec, f"bench_leg{i}", [xy[0], xy[1], (top - 0.04) / 2], [0.02, 0.02, (top - 0.04) / 2], WOOD)
    # Hands up over the table before it exists in the physics: a hanging hand would be trapped under it.
    for a in arms:
        side = float(np.sign((d.xpos[m.jnt_bodyid[a.joints[0]]][:2] - base[:2]) @ left)) or 1.0
        ready = base[:2] + fwd * (near + 0.02) + left * side * 0.26  # out to the side: the middle stays free
        try:
            sim.move_end_effector([float(ready[0]), float(ready[1]), top + 0.14], a.name, duration=1.0)
        except Exception:  # noqa: BLE001 - keep whatever pose it reached
            pass
    for a in arms:  # this is now the rest pose
        for j in a.joints:
            if j in sim.pos_act:
                sim._home_cmd[sim.pos_act[j]] = float(sim.cmd[sim.pos_act[j]])
    sim.pad_half = 0.1
    for name, side, rgba in (("pad_a", 1, ORANGE), ("pad_b", -1, BLUE)):
        xy = base[:2] + fwd * (near + 0.08) + left * side * min(0.22, width - 0.1)  # within easy reach
        sim.spec.worldbody.add_geom(name=f"bench_{name}", type=_BOX, pos=[xy[0], xy[1], top + 0.002],
                                    size=[sim.pad_half, sim.pad_half, 0.002], rgba=rgba, contype=0, conaffinity=0)
        sim.zones[name] = [float(xy[0]), float(xy[1])]
    size = 0.02
    count: dict[str, int] = {}
    # A loose 2 x 2 grid, 11 cm apart: room for a hand between neighbours.
    # Each hand has its own half: orange cubes and the orange tray on the left, blue on the right.
    for along, across, color in ((0.04, 0.08, "orange"), (0.12, 0.08, "orange"), (0.04, -0.08, "blue"), (0.12, -0.08, "blue")):
        xy = base[:2] + fwd * (near + along) + left * across
        count[color] = count.get(color, 0) + 1
        sim.spawn_object("box", size, [float(xy[0]), float(xy[1]), top + size + 0.001], color,
                         name=f"cube_{color}_{count[color]}", recompile=False, mass=0.05)
    sim.recompile()


def _tables(sim: "Sim") -> None:
    """Things to carry on table A; table B starts empty. Scanned real objects in the lab, boxes elsewhere."""
    sim.stations = {name: {"slots": table_slots(name), "top": TABLE_TOP, "xy": list(TABLES[name])} for name in TABLES}
    for name, (x, y) in TABLES.items():  # where to stand to work at the table
        sim.zones[name] = [x, float(y - 0.83 * np.sign(y))]
    slots = sim.stations["table_a"]["slots"]
    # Scanned household objects (up to ~25 cm) are for suction-style holders; a parallel gripper
    # (typically < 9 cm opening) gets crates it can actually close on.
    grippers = any(a.gripper_actuators and not a.assisted for a in sim.arms.values())
    scanned = ["mug", "tea_box", "blue_mug"] if sim.world == "lab" and not grippers else []
    try:
        for kind, xy in zip(scanned, slots):
            sim.spawn_scanned(kind, xy, TABLE_TOP, recompile=False)
    except OSError as e:  # offline: fall back to plain boxes
        print(f"[anybot] scanned objects unavailable ({e}); using boxes")
        scanned = []
    if not scanned:
        for i, (xy, color) in enumerate(zip(slots, ("orange", "blue", "yellow"))):
            # 7 cm: fits common parallel grippers (Robotiq 2F-85 opens to 8.5 cm)
            sim.spawn_object("box", 0.035, [xy[0], xy[1], TABLE_TOP + 0.037], color, name=f"crate_{i + 1}", recompile=False)
    sim.recompile()


def _workstation(sim: "Sim", arm) -> None:
    """Four boxes and two drop pads, laid out in front of the arm and scaled to its reach."""
    m, d = sim.m, sim.d
    base = d.xpos[m.jnt_bodyid[arm.joints[0]]][:2].copy()
    reach = sim._tcp(arm, d)[:2] - base
    dist = float(np.linalg.norm(reach))
    u = reach / dist if dist > 0.1 else np.array([1.0, 0.0])
    v = np.array([-u[1], u[0]])
    r = float(np.clip(0.9 * dist, 0.15, 0.6))
    size = float(np.clip(0.05 * r, 0.012, 0.025))
    sim.pad_half = 4 * size
    for name, side, rgba in (("pad_a", 1, ORANGE), ("pad_b", -1, BLUE)):
        xy = base + u * r * 0.7 + side * v * r * 0.7
        _pad(sim.spec, f"wh_{name}", xy, sim.pad_half, rgba)
        sim.zones[name] = [float(xy[0]), float(xy[1])]
    spots = [(0.85, 0.2, "orange"), (0.85, -0.2, "blue"), (1.12, 0.2, "blue"), (1.12, -0.2, "orange")]
    count: dict[str, int] = {}
    for along, across, color in spots:
        xy = base + u * r * along + v * r * across
        count[color] = count.get(color, 0) + 1
        sim.spawn_object("box", size, [float(xy[0]), float(xy[1]), size], color,
                         name=f"box_{color}_{count[color]}", recompile=False)
    sim.recompile()
