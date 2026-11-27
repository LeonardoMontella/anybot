"""Render anybot's promo video (or a storyboard of it) straight from the simulation.

    uv run python scripts/promo/promo.py --storyboard out.png     # one still per scene, for review
    uv run python scripts/promo/promo.py --video out.mp4           # the whole thing (~80 s, 1280x720)

Simulations are recorded in simulated time (positions saved every frame, rendered afterwards), so heavy
scenes play back smoothly whatever the laptop's speed. Captions are drawn with PIL; video is encoded with
the ffmpeg bundled by imageio-ffmpeg.
"""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS = 1280, 720, 30
ACCENT, ACCENT_LIGHT, INK, PAPER = (74, 104, 136), (127, 157, 189), (20, 24, 30), (246, 247, 249)
AVENIR = "/System/Library/Fonts/Avenir Next.ttc"
MENLO = "/System/Library/Fonts/Menlo.ttc"


def font(size: int, weight: str = "regular") -> ImageFont.FreeTypeFont:
    index = {"regular": 7, "medium": 5, "demi": 2, "bold": 0}[weight]
    try:
        return ImageFont.truetype(AVENIR, size, index=index)
    except OSError:
        return ImageFont.load_default(size)


def mono(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(MENLO, size)
    except OSError:
        return ImageFont.load_default(size)


# ---------------------------------------------------------------------- recording and rendering

class Recorder:
    """Saves qpos (and an optional per-frame extra) every 1/fps of simulated time while a scene steps."""

    def __init__(self, scene, fps: float = FPS, extra=None):
        self.scene, self.dt, self.extra = scene, 1.0 / fps, extra
        self.frames: list[tuple[np.ndarray, object]] = []
        self._next = scene.d.time
        self._orig = scene.step
        scene.step = self._step

    def _step(self, n: int = 1) -> None:
        for _ in range(n):
            self._orig(1)
            d = self.scene.d
            if d.time >= self._next:
                self.frames.append((d.qpos.copy(), self.extra() if self.extra else None))
                self._next += self.dt

    def close(self) -> None:
        self.scene.step = self._orig


class Shot:
    def __init__(self, m: mujoco.MjModel):
        self.m = m
        self.r = mujoco.Renderer(m, H, W)
        self.d = mujoco.MjData(m)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.opt = mujoco.MjvOption()

    def frame(self, qpos, lookat, distance, azimuth, elevation) -> Image.Image:
        self.d.qpos[:] = qpos
        mujoco.mj_kinematics(self.m, self.d)
        self.cam.lookat[:] = lookat
        self.cam.distance, self.cam.azimuth, self.cam.elevation = distance, azimuth, elevation
        self.r.update_scene(self.d, self.cam, self.opt)
        return Image.fromarray(self.r.render())

    def close(self) -> None:
        self.r.close()


def speed_label(n_frames: int, rec_fps: float, seconds: float) -> str:
    """How much faster than real time a recording squeezed into `seconds` plays."""
    x = (n_frames / rec_fps) / max(seconds, 1e-6)
    return f"{x:.0f}×" if x >= 1.5 else ("real time" if x > 0.8 else f"{x:.1f}×")


def ease(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def run_bg(fn) -> threading.Thread:
    th = threading.Thread(target=fn, daemon=True)
    th.start()
    return th


# ---------------------------------------------------------------------- overlays

def caption(img: Image.Image, title: str, sub: str = "", alpha: float = 1.0, x: int = 56) -> Image.Image:
    """Lower-left caption on a soft dark plate, accent bar on the left."""
    if alpha <= 0:
        return img
    img = img.convert("RGBA")
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(layer)
    ft, fs = font(40, "demi"), font(23)
    tw = max(dr.textlength(title, font=ft), dr.textlength(sub, font=fs) if sub else 0)
    y, hgt = H - (150 if sub else 112), 104 if sub else 66
    dr.rounded_rectangle([x - 22, y - 18, x + tw + 30, y + hgt], 14, fill=(14, 18, 24, int(170 * alpha)))
    dr.rectangle([x - 22, y - 18, x - 16, y + hgt], fill=ACCENT_LIGHT + (int(255 * alpha),))
    dr.text((x, y - 4), title, font=ft, fill=(255, 255, 255, int(255 * alpha)))
    if sub:
        dr.text((x, y + 50), sub, font=fs, fill=(208, 216, 226, int(255 * alpha)))
    return Image.alpha_composite(img, layer).convert("RGB")


def pill(img: Image.Image, text: str, xy, fill=(14, 18, 24, 180)) -> Image.Image:
    img = img.convert("RGBA")
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(layer)
    f = font(22, "medium")
    w = dr.textlength(text, font=f)
    x, y = xy
    dr.rounded_rectangle([x, y, x + w + 28, y + 40], 20, fill=fill)
    dr.text((x + 14, y + 6), text, font=f, fill=(255, 255, 255, 255))
    return Image.alpha_composite(img, layer).convert("RGB")


def chat_panel(img: Image.Image, prompt: str, typed: float, calls: list[str], shown: float) -> Image.Image:
    """An MCP client on the left: the user's prompt being typed, then the tool calls the agent makes."""
    img = img.convert("RGBA")
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(layer)
    x0, y0, x1, y1 = 40, 40, 500, H - 40
    dr.rounded_rectangle([x0, y0, x1, y1], 22, fill=(250, 251, 252, 236))
    dr.text((x0 + 26, y0 + 22), "Claude  ·  anybot MCP", font=font(20, "demi"), fill=ACCENT + (255,))
    # prompt bubble, typed out
    text = prompt[: int(len(prompt) * min(typed, 1.0))]
    f = font(24)
    lines, line = [], ""
    for word in text.split(" "):
        test = (line + " " + word).strip()
        if dr.textlength(test, font=f) > 370:
            lines.append(line)
            line = word
        else:
            line = test
    lines.append(line)
    by = y0 + 80
    bh = 22 + 34 * len(lines)
    dr.rounded_rectangle([x0 + 40, by, x1 - 24, by + bh], 18, fill=ACCENT + (255,))
    for i, ln in enumerate(lines):
        dr.text((x0 + 60, by + 12 + 34 * i), ln, font=f, fill=(255, 255, 255, 255))
    cy = by + bh + 30
    fm = mono(19)
    for i, c in enumerate(calls):
        a = min(max(shown * len(calls) - i, 0.0), 1.0)
        if a <= 0:
            break
        dr.rounded_rectangle([x0 + 24, cy, x1 - 40, cy + 44], 12, fill=(236, 240, 245, int(255 * a)))
        dr.text((x0 + 40, cy + 11), "⚙ " + c, font=fm, fill=INK + (int(255 * a),))
        cy += 56
    return Image.alpha_composite(img, layer).convert("RGB")


def minimap(img: Image.Image, planner, pose, path, scan_pts, box=(W - 380, 30, W - 30, 330)) -> Image.Image:
    """Inset sensor view: collision map, lidar hits, planned path and the robot (like the studio panel)."""
    x0, y0, x1, y1 = box
    sz = (x1 - x0, y1 - y0)
    span = 7.0  # metres shown, centred on the robot
    img = img.convert("RGBA")
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(layer)
    dr.rounded_rectangle(box, 18, fill=(14, 18, 24, 215))
    cx, cy = pose[0], pose[1]
    k = sz[0] / span

    def px(p):
        return x0 + sz[0] / 2 + (p[0] - cx) * k, y0 + sz[1] / 2 - (p[1] - cy) * k

    occ = np.argwhere(planner.static)
    if len(occ):
        xy = np.stack([planner.lo[0] + occ[:, 1] * planner.cell, planner.lo[1] + occ[:, 0] * planner.cell], 1)
        near = np.all(np.abs(xy - [cx, cy]) < span / 2 * np.array([1, sz[1] / sz[0]]), axis=1)
        for p in xy[near][::1]:
            u, v = px(p)
            dr.rectangle([u, v, u + k * planner.cell, v + k * planner.cell], fill=(120, 134, 152, 255))
    for p in scan_pts:
        u, v = px(p)
        if x0 < u < x1 and y0 < v < y1:
            dr.ellipse([u - 2, v - 2, u + 2, v + 2], fill=(255, 120, 40, 255))
    if path is not None and len(path) > 1:
        dr.line([px(p) for p in path], fill=ACCENT_LIGHT + (255,), width=4)
    u, v = px(pose)
    dr.ellipse([u - 9, v - 9, u + 9, v + 9], fill=(255, 255, 255, 255))
    dr.line([u, v, u + 18 * np.cos(pose[2]), v - 18 * np.sin(pose[2])], fill=(255, 255, 255, 255), width=4)
    dr.text((x0 + 16, y1 - 32), "map · lidar · path", font=font(18, "medium"), fill=(220, 226, 234, 255))
    return Image.alpha_composite(img, layer).convert("RGB")


def card(title: str, sub: str = "", bg: Image.Image | None = None, small: str = "") -> Image.Image:
    img = (bg.filter(ImageFilter.GaussianBlur(6)) if bg is not None else Image.new("RGB", (W, H), INK)).convert("RGBA")
    shade = Image.new("RGBA", img.size, (12, 15, 20, 190 if bg is not None else 0))
    img = Image.alpha_composite(img, shade)
    dr = ImageDraw.Draw(img)
    ft = font(96, "bold")
    tw = dr.textlength(title, font=ft)
    dr.text(((W - tw) / 2, H / 2 - 100), title, font=ft, fill=(255, 255, 255))
    if sub:
        fs = font(32)
        dr.text(((W - dr.textlength(sub, font=fs)) / 2, H / 2 + 44), sub, font=fs, fill=(214, 222, 232))
    if small:
        fsm = font(18)
        dr.text(((W - dr.textlength(small, font=fsm)) / 2, H - 60), small, font=fsm, fill=(160, 170, 184))
    return img.convert("RGB")


# ---------------------------------------------------------------------- scenes

def _host(robot: str, world: str):
    from anybot.host import ToolHost
    from anybot.sim import Sim

    host = ToolHost()
    host.sim = Sim(robot, realtime=False, world=world)
    host._last_load = {"robot": robot, "world": world}
    return host


def scene_title(seconds: float, still: bool):
    host = _host("g1", "warehouse")
    sim = host.sim
    sh = Shot(sim.m)
    bg = sh.frame(sim.d.qpos, sim.d.xpos[sim._bid(sim.profile.root_body)] + [0, 0, 0.1], 3.2, 150, -12)
    sh.close()
    img = card("anybot", "Any robot.  Any place.  Plain text.", bg)
    return [img] * (1 if still else int(seconds * FPS))


def scene_prompt(seconds: float, still: bool):
    host = _host("panda", "warehouse")
    sim = host.sim
    rec = Recorder(sim.scene)
    tasks = [t["id"] for t in host.tasks()]
    run = "stack" if "stack" in tasks else tasks[0]
    host.call("run_task", {"task": run})
    rec.close()
    sh = Shot(sim.m)
    base = sim.d.xpos[sim._bid(sim.profile.root_body)]
    frames = rec.frames
    n = int(seconds * FPS)
    idx = [len(frames) // 2] if still else list(np.linspace(0, len(frames) - 1, n - 2 * FPS).astype(int))
    lead = 0 if still else 2 * FPS  # the prompt is typed before the robot moves
    calls = ["load_robot(\"panda\")", f"run_task(\"{run}\")", "snapshot()"]
    prompt = "Load the Panda and stack the cubes into a tower, then show me a snapshot."
    for k in range(1 if still else n):
        i = idx[min(max(k - lead, 0), len(idx) - 1)] if not still else idx[0]
        t = 0.6 if still else k / n
        img = sh.frame(frames[i][0], base + [0.45, 0.0, 0.25], 1.45, 200 - 25 * t, -24)
        img = chat_panel(img, prompt, 1.0 if still else ease(k / (1.6 * FPS)), calls, 1.0 if still else ease((k - 1.8 * FPS) / (2 * FPS)))
        img = caption(img, "Plain text in, robot actions out", "Any MCP client drives the simulation", x=548)
        yield img
    sh.close()


MONTAGE = [("panda", "Franka Panda  ·  arm"), ("ur5e", "UR5e  ·  arm"), ("spot", "Spot  ·  quadruped"),
           ("go2", "Unitree Go2  ·  quadruped"), ("g1", "Unitree G1  ·  humanoid"), ("tidybot", "Tidybot  ·  mobile manipulator")]


def scene_montage(seconds: float, still: bool):
    per = seconds / len(MONTAGE)
    for name, label in (MONTAGE[2:3] if still else MONTAGE):
        host = _host(name, "warehouse")
        sim = host.sim
        rec = Recorder(sim.scene)
        skills = sim.skills()
        try:
            if "drive" in skills:
                sim.drive(0.4, 0.3, per + 0.5)
            elif "move_end_effector" in skills:
                arm = next(iter(sim.arms.values()))
                p = sim._tcp(arm, sim.d)
                sim.move_end_effector(list(p + [0.0, 0.15, 0.1]), arm.name, duration=per)
            else:
                sim.wait(per)
        except Exception:  # noqa: BLE001 - a montage shot never stops the render
            sim.wait(per)
        rec.close()
        sh = Shot(sim.m)
        root = sim._bid(sim.profile.root_body)
        frames = rec.frames or [(sim.d.qpos.copy(), None)]
        n = 1 if still else int(per * FPS)
        for k in range(n):
            q = frames[int(len(frames) * (0.6 if still else k / n)) % len(frames)][0]
            sh.d.qpos[:] = q
            mujoco.mj_kinematics(sim.m, sh.d)
            look = sh.d.xpos[root].copy()
            look[2] = max(look[2], 0.35)
            img = sh.frame(q, look, 1.9 + 0.6 * (sim.profile.height_m > 1.0), 135 + 30 * (k / max(n, 1)), -14)  # robot-fixed framing is fine: it walks in place of the shot
            img = pill(img, label, (56, 56))
            img = caption(img, "Any robot, zero configuration", "Kinematics alone tell anybot what it is and which skills apply")
            yield img
        sh.close()


def scene_relay(seconds: float, still: bool):
    from anybot.host import ToolHost

    host = ToolHost()
    host._load_team("amr_fleet", realtime=False)
    scene = host.scene
    rec = Recorder(scene, fps=10)  # 3x time-lapse at 30 fps
    try:
        host.call("run_task", {"task": "relay"})
    finally:
        rec.close()
    sh = Shot(scene.m)
    frames = rec.frames
    n = 1 if still else int(seconds * FPS)
    for k in range(n):
        f = 0.45 if still else k / n
        # 8x real time from the start of the relay: fast, but every move still reads.
        i = min(int(len(frames) * 0.1) if still else int(k * 8 * 10 / FPS), len(frames) - 1)
        img = sh.frame(frames[i][0], [0.0, 0.0, 0.3], 7.5, 120 + 40 * f, -42)
        img = pill(img, "8×", (W - 110, 40))
        img = caption(img, "Teams: four robots, one physics scene", "Pickers load totes, carrier AMRs swap stations and deliver")
        yield img
    sh.close()


def scene_spot(seconds: float, still: bool):
    from anybot import sensors

    host = _host("go2", "apartment")
    sim = host.sim
    planner = sim.scene.navmap()

    def extra():
        pos, yaw = sim._base_pose()
        pts = sensors.lidar(sim).points() if sim.navigating else np.zeros((0, 2))
        path = [p[:2] for p in (getattr(sim, "nav_path", None) or [])] if sim.navigating else None
        return (float(pos[0]), float(pos[1]), float(yaw)), pts, path

    rec = Recorder(sim.scene, fps=30, extra=extra)
    try:
        for zone in ("spot_1", "spot_3"):
            host.call("go_to_zone", {"zone": zone})
            if sim.d.time > 2 * seconds + 2:  # enough walking for the shot
                break
    finally:
        rec.close()
    sh = Shot(sim.m)
    frames = rec.frames
    n = 1 if still else int(seconds * FPS)
    for k in range(n):
        f = 0.35 if still else k / n
        # Played at exactly 2x real time: legs keep their real rhythm (a squeezed walk looks like sliding).
        i = int(len(frames) * 0.35) if still else 2 * k
        q, (pose, pts, path) = frames[min(i, len(frames) - 1)]
        # The camera follows the robot smoothly but never turns with it: the robot turns, the room stays put.
        img = sh.frame(q, [pose[0], pose[1], 0.2], 3.6, 110 + 15 * f, -78)
        img = minimap(img, planner, pose, path, pts)
        img = pill(img, "2×", (W - 110, 350))
        img = caption(img, "Navigates a real scanned apartment", "Collision map + lidar: through doors, around furniture, replanning on the way")
        yield img
    sh.close()


def scene_g1(seconds: float, still: bool):
    host = _host("g1", "warehouse")
    sim = host.sim
    host.call("spawn_object", {"shape": "box", "size": 0.035, "color": "red", "x": 1.0, "y": 1.0, "name": "floor_box"})
    rec = Recorder(sim.scene, fps=20)  # 1.5x
    try:
        host.call("pick_object", {"name": "floor_box"})
        host.call("place_object", {"x": 2.0, "y": 0.5})
    finally:
        rec.close()
    sh = Shot(sim.m)
    root = sim._bid(sim.profile.root_body)
    frames = rec.frames
    n = 1 if still else int(seconds * FPS)
    for k in range(n):
        f = 0.3 if still else k / n
        q = frames[min(int(len(frames) * f), len(frames) - 1)][0]
        sh.d.qpos[:] = q
        mujoco.mj_kinematics(sim.m, sh.d)
        look = sh.d.xpos[root].copy()
        look[2] = 0.7
        img = sh.frame(q, look, 2.8, 110 + 50 * f, -12)
        img = caption(img, "Learned walking, whole-body grasping", "The G1 squats to pick a box off the floor and carries it with both hands")
        yield img
    sh.close()


def scene_capture(seconds: float, still: bool, video_frames: Path | None = None, world: str | None = None):
    """Phone video → 3D → world. Until a real capture exists this uses the apartment as a stand-in."""
    host = _host("tidybot", world or "apartment")
    sim = host.sim
    rec = Recorder(sim.scene, fps=20)  # 1.5x
    try:
        zones = list(sim.zones)
        for zone in zones[:2] or []:
            host.call("go_to_zone", {"zone": zone})
    finally:
        rec.close()
    frames = rec.frames
    sh = Shot(sim.m)
    root = sim._bid(sim.profile.root_body)
    n = 1 if still else int(seconds * FPS)
    thumbs = sorted(video_frames.glob("*.jpg"))[::4][:6] if video_frames else []
    for k in range(n):
        f = 0.5 if still else k / n
        q = frames[min(int(len(frames) * f), len(frames) - 1)][0]
        sh.d.qpos[:] = q
        mujoco.mj_kinematics(sim.m, sh.d)
        p = sh.d.xpos[root]
        img = sh.frame(q, [p[0], p[1], 0.3], 4.2, 60 + 20 * f, -76)
        if thumbs:
            for j, t in enumerate(thumbs):
                th = Image.open(t)
                th.thumbnail((180, 120))
                img.paste(th, (56 + j * 196, 40))
        else:
            img = pill(img, "example world: a scanned apartment", (56, 40))
        img = pill(img, "phone video  ›  3D reconstruction  ›  world", (56, 96 if not thumbs else 176))
        img = pill(img, speed_label(len(frames), 20, seconds), (W - 120, 40))
        img = caption(img, "Video to world", "anybot import-scene room.mp4: a real place, ready for robots in minutes")
        yield img
    sh.close()


def _depth_rgb(depth: np.ndarray, far: float = 3.0) -> np.ndarray:
    """Near = warm, far = cool (the studio's palette), nothing = black."""
    x = np.clip(depth / far, 0, 1)
    img = np.stack([np.clip(1.6 - 2 * x, 0, 1), np.clip(1.2 - 1.6 * abs(x - 0.35), 0, 1),
                    np.clip(2 * x - 0.4, 0, 1) * (x < 0.999)], -1)
    return (255 * img).astype(np.uint8)


def _eyes_take(host_robot: str, action, seconds: float, still: bool, look, label: str, far: float = 3.0):
    """A robot working, with what its own camera sees (RGB and depth) beside the 3D view."""
    from anybot import sensors

    host = _host(host_robot, "warehouse")
    sim = host.sim
    cam = sensors.Camera(sim.m, 480, 360)

    def extra():
        rgb, depth, _, _ = cam.render(sim, sim.d)
        return rgb.copy(), _depth_rgb(depth, far)

    setup = action(host)
    rec = Recorder(sim.scene, fps=30, extra=extra)
    try:
        setup()
    finally:
        rec.close()
    cam.close()
    frames = rec.frames
    sh = Shot(sim.m)
    root = sim._bid(sim.profile.root_body)
    n = 1 if still else int(seconds * FPS)
    for k in range(n):
        f = 0.55 if still else k / n
        q, (rgb, dep) = frames[min(int(len(frames) * (0.15 + 0.8 * f)), len(frames) - 1)]
        sh.d.qpos[:] = q
        mujoco.mj_kinematics(sim.m, sh.d)
        img = sh.frame(q, look(sh.d, root), *([1.6, 150 + 20 * f, -20] if host_robot != "g1" else [2.4, 130 + 25 * f, -10]))
        img = img.convert("RGBA")
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        dr = ImageDraw.Draw(layer)
        x0, w, h = W - 36 - 360, 360, 270
        for i, (pic, name) in enumerate(((rgb, label + "  ·  RGB"), (dep, "depth  ·  near → far"))):
            y0 = 36 + i * (h + 64)
            dr.rounded_rectangle([x0 - 8, y0 - 8, x0 + w + 8, y0 + h + 40], 16, fill=(14, 18, 24, 215))
            layer.paste(Image.fromarray(pic).resize((w, h)), (x0, y0))
            dr.text((x0 + 4, y0 + h + 8), name.replace("→", ">"), font=font(19, "medium"), fill=(220, 226, 234, 255))
        img = Image.alpha_composite(img, layer).convert("RGB")
        yield caption(img, "What the robot sees", "RGB-D cameras: on the wrist of an arm, on the head of a humanoid")
    sh.close()


def scene_eyes(seconds: float, still: bool):
    def panda(host):
        sim = host.sim
        name = sim.objects[0]
        return lambda: host.call("pick_object", {"name": name})

    def g1(host):
        host.call("spawn_object", {"shape": "box", "size": 0.035, "color": "red", "x": 1.0, "y": 1.0, "name": "floor_box"})
        return lambda: host.call("pick_object", {"name": "floor_box"})

    def look_arm(d, root):
        return d.xpos[root] + [0.4, 0.0, 0.3]

    def look_g1(d, root):
        p = d.xpos[root].copy()
        p[2] = 0.6
        return p

    if still:
        yield from _eyes_take("panda", panda, seconds, True, look_arm, "wrist camera", far=0.9)
        return
    yield from _eyes_take("panda", panda, seconds / 2, False, look_arm, "wrist camera", far=0.9)
    yield from _eyes_take("g1", g1, seconds / 2, False, look_g1, "head camera", far=4.0)


def scene_studio(seconds: float, still: bool, shots: list[Path] | None = None):
    n = 1 if still else int(seconds * FPS)
    imgs = [Image.open(p).convert("RGB").resize((W, H)) for p in (shots or [])]
    for k in range(n):
        if imgs:
            base = imgs[min(int(len(imgs) * k / n), len(imgs) - 1)]
        else:
            base = card("studio", "placeholder: screenshots of the web studio go here")
        yield (caption(base, "anybot studio", "Live 3D view, task timeline, sensors: map, lidar, RGB-D, wrist camera"))


def scene_end(seconds: float, still: bool):
    img = card("anybot", "github.com/LeonardoMontella/anybot",
               small="Built with MuJoCo and MCP  ·  Robot models: MuJoCo Menagerie  ·  "
                     "Apartment scan: Replica (Meta), CC BY-NC 4.0  ·  G1 policy: Unitree")
    return [img] * (1 if still else int(seconds * FPS))


SCENES = [("title", scene_title, 4), ("prompt", scene_prompt, 12), ("montage", scene_montage, 10),
          ("relay", scene_relay, 12), ("spot", scene_spot, 12), ("g1", scene_g1, 9), ("eyes", scene_eyes, 12),
          ("capture", scene_capture, 10), ("studio", scene_studio, 6), ("end", scene_end, 5)]


STUDIO_SHOTS: list[Path] = []


def storyboard(path: Path, only: list[str] | None = None) -> None:
    stills = []
    for name, fn, secs in SCENES:
        if only and name not in only:
            continue
        print(f"[promo] {name}...", flush=True)
        img = next(iter(fn(secs, True, STUDIO_SHOTS) if name == "studio" else fn(secs, True)))
        stills.append((name, secs, img))
        img.save(path.with_name(f"{path.stem}_{name}.png"))
    cols = 3
    rows = int(np.ceil(len(stills) / cols))
    tw, th = 640, 360
    sheet = Image.new("RGB", (cols * tw + (cols + 1) * 16, rows * (th + 44) + 16), (236, 238, 242))
    dr = ImageDraw.Draw(sheet)
    for i, (name, secs, img) in enumerate(stills):
        x, y = 16 + (i % cols) * (tw + 16), 16 + (i // cols) * (th + 44)
        sheet.paste(img.resize((tw, th)), (x, y))
        dr.text((x, y + th + 8), f"{i + 1}. {name}  ({secs} s)", font=font(22, "demi"), fill=INK)
    sheet.save(path)
    print(f"[promo] storyboard: {path}")


def video(path: Path, studio_shots: list[Path] | None = None, capture_frames: Path | None = None,
          capture_world: str | None = None) -> None:
    import imageio_ffmpeg

    writer = imageio_ffmpeg.write_frames(str(path), (W, H), fps=FPS, codec="libx264", quality=None,
                                         bitrate="2600k", macro_block_size=8,
                                         output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    writer.send(None)
    prev = None
    for name, fn, secs in SCENES:
        print(f"[promo] {name}...", flush=True)
        if name == "studio":
            frames = fn(secs, False, studio_shots)
        elif name == "capture":
            frames = fn(secs, False, capture_frames, capture_world)
        else:
            frames = fn(secs, False)
        last = None
        for k, img in enumerate(frames):
            img = img.convert("RGB")
            last = img
            if prev is not None and k < 10:  # short cross-fade between scenes
                img = Image.blend(prev, img, (k + 1) / 11)
            writer.send(np.asarray(img))
        prev = last
    writer.close()
    print(f"[promo] video: {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--storyboard", type=Path)
    ap.add_argument("--video", type=Path)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--studio-shots", nargs="*", type=Path)
    ap.add_argument("--capture-frames", type=Path)
    ap.add_argument("--capture-world")
    a = ap.parse_args()
    STUDIO_SHOTS[:] = a.studio_shots or []
    if a.storyboard:
        storyboard(a.storyboard, a.only)
    if a.video:
        video(a.video, a.studio_shots, a.capture_frames, a.capture_world)


if __name__ == "__main__":
    main()
