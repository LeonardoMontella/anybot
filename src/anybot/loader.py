"""Load any robot (MJCF, URDF, or a `robot_descriptions` name) into a ready-to-simulate MjSpec.

The loader guarantees three things regardless of the source:
  * every hinge/slide joint is driven by an actuator (position actuators are injected if missing),
  * the scene has a floor and a light,
  * URDF `package://` mesh paths are resolved to absolute paths.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

_HINGE_SLIDE = {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}


@dataclass
class LoadedRobot:
    name: str
    source: str
    spec: mujoco.MjSpec
    model: mujoco.MjModel
    injected_actuators: list[str]
    notes: list[str] = field(default_factory=list)  # what the importer had to adapt


# Menagerie models that robot_descriptions has no shortcut for (paths inside the Menagerie checkout).
EXTRA_MODELS = {
    "tidybot": "stanford_tidybot/tidybot.xml",  # holonomic AMR + Kinova Gen3 arm + Robotiq 2F-85
    "amr": "stanford_tidybot/base.xml",  # the same AMR base on its own
}


def _menagerie_root() -> Path:
    from robot_descriptions import panda_mj_description

    return Path(panda_mj_description.MJCF_PATH).parents[1]


def available_descriptions() -> list[str]:
    """Names of MuJoCo robots: robot_descriptions shortcuts (e.g. 'panda', 'go2') plus EXTRA_MODELS."""
    import robot_descriptions

    bundled = [
        m.name.removesuffix("_mj_description")
        for m in pkgutil.iter_modules(robot_descriptions.__path__)
        if m.name.endswith("_mj_description")
    ]
    return sorted(set(EXTRA_MODELS) | set(bundled))


def resolve_source(source: str) -> tuple[str, Path]:
    """Map a user string to (robot name, model file path)."""
    path = Path(source).expanduser()
    if path.suffix.lower() in (".xml", ".urdf", ".mjcf") and path.exists():
        return path.stem, path.resolve()

    if source in EXTRA_MODELS:
        return source, _menagerie_root() / EXTRA_MODELS[source]
    module = source if source.endswith("_description") else f"{source}_mj_description"
    try:
        desc = importlib.import_module(f"robot_descriptions.{module}")
    except ModuleNotFoundError as e:
        raise ValueError(
            f"Unknown robot '{source}'. Pass a .xml/.urdf path or one of: "
            + ", ".join(available_descriptions())
        ) from e
    file = getattr(desc, "MJCF_PATH", None) or getattr(desc, "URDF_PATH")
    return source.removesuffix("_mj_description").removesuffix("_description"), Path(file)


_MESH_OK = (".stl", ".obj", ".msh")


def _prepare_urdf(path: Path) -> tuple[Path, list[str]]:
    """Make a URDF loadable by MuJoCo. Returns (patched copy, notes about what had to be changed).

    * `package://` and relative mesh paths become absolute;
    * meshes MuJoCo cannot read (.dae) are swapped for an .stl/.obj of the same name if one exists
      next to them, otherwise that visual/collision element is dropped;
    * zero-sized primitives get a tiny size instead of failing the whole model.
    """
    import xml.etree.ElementTree as ET

    notes: list[str] = []
    tree = ET.parse(path)
    root = tree.getroot()

    def locate(uri: str) -> Path | None:
        if uri.startswith("package://"):
            pkg, _, rest = uri[len("package://"):].partition("/")
            for parent in [path.parent, *path.parents]:
                for cand in (parent / pkg / rest, parent / rest):
                    if cand.exists():
                        return cand.resolve()
            return None
        cand = Path(uri[len("file://"):] if uri.startswith("file://") else uri)
        if not cand.is_absolute():
            for base in [path.parent, *list(path.parents)[:3]]:
                if (base / cand).exists():
                    return (base / cand).resolve()
            return None
        return cand if cand.exists() else None

    def readable(file: Path) -> Path | None:
        if file.suffix.lower() in _MESH_OK:
            return file
        for folder in (file.parent, file.parent.parent / "collision", file.parent.parent / "stl", file.parent.parent):
            for ext in _MESH_OK:
                alt = folder / (file.stem + ext)
                if alt.exists():
                    return alt
        return None

    dropped = swapped = 0
    for link in root.iter("link"):
        for kind in ("visual", "collision"):
            for elem in list(link.findall(kind)):
                mesh = elem.find("geometry/mesh")
                if mesh is None:
                    continue
                found = locate(mesh.get("filename", ""))
                usable = readable(found) if found else None
                if usable is None:
                    link.remove(elem)
                    dropped += 1
                    continue
                swapped += usable != found
                mesh.set("filename", str(usable))
    if swapped:
        notes.append(f"{swapped} unsupported meshes (.dae) replaced by .stl/.obj versions found next to them")
    if dropped:
        notes.append(f"{dropped} visual/collision meshes dropped (missing or in a format MuJoCo cannot read)")
    fixed = 0
    for tag, attrs in (("box", ("size",)), ("cylinder", ("radius", "length")), ("sphere", ("radius",))):
        for g in root.iter(tag):
            for attr in attrs:
                vals = [float(v) for v in g.get(attr, "0").split()]
                if any(v <= 0 for v in vals):
                    g.set(attr, " ".join(str(v if v > 0 else 1e-3) for v in vals))
                    fixed += 1
    if fixed:
        notes.append(f"{fixed} zero-sized primitives given a minimal size")
    if root.find("mujoco") is None:
        mj = ET.SubElement(root, "mujoco")
        ET.SubElement(mj, "compiler", discardvisual="false", strippath="false", fusestatic="false", balanceinertia="true")
    tmp = Path(tempfile.mkdtemp(prefix="anybot_")) / path.name
    tree.write(tmp)
    return tmp, notes


def _has_floor(spec: mujoco.MjSpec) -> bool:
    return any(g.type == mujoco.mjtGeom.mjGEOM_PLANE for g in spec.worldbody.geoms)


def _inject_actuators(spec: mujoco.MjSpec, model: mujoco.MjModel) -> list[str]:
    """Add a position servo to every hinge/slide joint that has none.

    Gains come from the inertia each joint actually moves (a critically damped ~8 Hz servo), so a
    200 kg base and a 5 g finger both get a servo that is stiff enough and still stable.
    """
    actuated = {
        model.actuator_trnid[i, 0]
        for i in range(model.nu)
        if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT
    }
    omega = 2 * np.pi * 8.0
    added = []
    for j in range(model.njnt):
        if int(model.jnt_type[j]) not in _HINGE_SLIDE or j in actuated:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if not name:
            continue
        inertia = max(float(model.dof_M0[model.jnt_dofadr[j]]), 1e-5)
        kp = inertia * omega**2
        act = spec.add_actuator(name=f"anybot_{name}", target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.set_to_position(kp=kp, kv=2 * inertia * omega)
        if model.jnt_limited[j]:
            act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
            act.ctrlrange = model.jnt_range[j].copy()
        added.append(name)
    if added:
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST  # stable with stiff servos
    return added


def _rest_on_floor(spec: mujoco.MjSpec, model: mujoco.MjModel) -> bool:
    """If a free-floating robot starts inside the floor (models whose origin is not at the feet),
    lift it so its lowest geom touches z=0. Returns True when the spec was changed."""
    root = spec.worldbody.first_body()
    if root is None or not any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in root.joints):
        return False
    d = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, d, 0)
    mujoco.mj_forward(model, d)
    from anybot.introspect import body_geoms, world_aabb  # local import: introspect imports nothing from here

    lo, _ = world_aabb(model, d, body_geoms(model, list(range(1, model.nbody))))
    if lo[2] > -0.005:
        return False
    dz = float(-lo[2]) + 0.002
    root.pos = [root.pos[0], root.pos[1], root.pos[2] + dz]
    for key in spec.keys:
        if len(key.qpos) >= 3:
            q = list(key.qpos)
            q[2] += dz
            key.qpos = q
    return True


def load_robot(source: str, *, floating: bool | None = None, world: str | None = None) -> LoadedRobot:
    """Load a robot by robot_descriptions name or file path.

    `floating` adds a free joint to the root body (for URDFs of legged/mobile robots,
    which have no notion of a floating base). Default: leave the model as is.
    """
    name, path = resolve_source(source)
    is_urdf = path.suffix.lower() == ".urdf"
    notes: list[str] = []
    if is_urdf:
        patched, notes = _prepare_urdf(path)
    spec = mujoco.MjSpec.from_file(str(patched if is_urdf else path))
    if is_urdf:
        # Mesh paths are now absolute; MjSpec otherwise resolves them against the temp dir.
        spec.meshdir = ""

    root = spec.worldbody.first_body()
    if floating and root is not None and not root.joints:
        root.add_freejoint()

    if not _has_floor(spec):
        spec.worldbody.add_geom(
            name="anybot_floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], rgba=[0.8, 0.8, 0.8, 1],
            # Explicit bits: the model's default class may disable collisions (e.g. contype=0).
            contype=0xFFFF, conaffinity=0xFFFF,
        )
    from anybot import worlds  # local import: worlds only needs mujoco

    if world not in (None, *worlds.all_worlds()):
        raise ValueError(f"Unknown world '{world}'. Worlds: {', '.join(worlds.all_worlds())}")
    worlds.style_scene(spec)
    if world == "warehouse":
        worlds.build_warehouse(spec)
    elif world == "lab":
        worlds.build_lab(spec)
    elif world == "bench":
        worlds.build_bench(spec)
    elif worlds.is_scan(world):
        worlds.build_scan_world(spec, world)
    # Room for HD offscreen rendering (live stream / snapshots).
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, 1280)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, 720)
    if not spec.worldbody.lights:
        spec.worldbody.add_light(pos=[0, 0, 3], dir=[0.2, 0.1, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                                 diffuse=[0.45, 0.45, 0.45], specular=[0.1, 0.1, 0.1])

    root = spec.worldbody.first_body()
    if root is not None and any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in root.joints):
        # Inactive until a walking robot stops to manipulate: then it holds the base still.
        spec.add_equality(name="anybot_anchor", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                          name1=root.name, active=False)
    model = spec.compile()
    if _rest_on_floor(spec, model):
        model = spec.compile()
    # Only fully unactuated models (typical URDFs) get servos: in an MJCF that already has
    # actuators, unactuated joints are deliberate (passive joints, tendon-coupled fingers).
    injected = _inject_actuators(spec, model) if model.nu == 0 else []
    if injected:
        model = spec.compile()
    if injected:
        notes.append(f"{len(injected)} position servos added (the model had no actuators)")
    return LoadedRobot(name=name, source=str(path), spec=spec, model=model, injected_actuators=injected, notes=notes)
