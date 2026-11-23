"""anybot studio: the live simulation as a local web app.

  * /            web UI (live robot, activity feed, robot panel, quick actions)
  * /stream      MJPEG video of the simulation, rendered offscreen by MuJoCo
  * /api/...     tools, state, events and camera, used by the UI and by the MCP adapter

Run:  anybot studio            (opens http://127.0.0.1:8765)
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import io
import json
import os
import threading
import time
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
import mujoco
import numpy as np
from PIL import Image
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from anybot import worlds
from anybot.foxglove_bridge import FoxgloveBridge
from anybot.sensorview import SensorFeed
from anybot.host import ToolHost
from anybot.sim import SkillError

PORT = int(os.environ.get("ANYBOT_PORT", "8765"))
WEB = Path(__file__).parent / "web"


class Streamer:
    """Renders the current robot ~30 times a second into a JPEG the browser streams."""

    def __init__(self, host: ToolHost, width: int = 960, height: int = 540, fps: int = 30):
        self.host, self.width, self.height, self.fps = host, width, height, fps
        self.frame: bytes | None = None
        self.seq = 0
        self.fps_measured = 0.0
        self.viewers = 0
        self.last_poll = 0.0
        self.cam = {"azimuth": 135.0, "elevation": -22.0, "distance": 2.0, "lookat": [0.0, 0.0, 0.3], "follow": True}
        self.preview: list[float] | None = None  # picked point waiting for the user's confirmation
        self._model = None
        self._renderer: mujoco.Renderer | None = None
        self._running = False

    def start(self) -> None:
        self._running = True
        threading.Thread(target=self._loop, name="anybot-render", daemon=True).start()

    def stop(self) -> None:
        self._running = False

    def reframe(self) -> None:
        sim = self.host.sim
        if sim is None:
            return
        if getattr(self.host.scene, "team", None):  # a team: frame the whole production line
            self.cam.update(lookat=[0.2, 0.7, 0.4], distance=5.2, azimuth=150.0, elevation=-32.0, follow=False)
            return
        lookat, dist = sim.framing()
        if sim.zones and "navigate_to" in sim.skills():  # show the floor zones around a mobile robot
            dist = max(dist, 6.5)
            self.cam["elevation"] = min(self.cam["elevation"], -38.0)
        if worlds.is_scan(sim.world):  # rooms have walls all around: look in from above
            dist, self.cam["elevation"] = 5.5, -62.0
        self.cam.update(lookat=[float(x) for x in lookat], distance=dist, follow=True)

    def _draw_markers(self, sim) -> None:
        """Overlay the pending (unconfirmed) target and the active navigation goal. Render-only."""
        scn = self._renderer.scene
        scale = 1.0 if "navigate_to" in sim.skills() else 0.4  # arms work at a smaller scale
        if sim.target is not None:
            _marker(scn, sim.target, 1.0, scale)
        if self.preview is not None:
            _marker(scn, self.preview, 0.45 + 0.4 * abs(np.sin(time.time() * 4)), scale)

    def _loop(self) -> None:
        period = 1.0 / self.fps
        count, t_fps = 0, time.perf_counter()
        while self._running:
            t0 = time.perf_counter()
            sim = self.host.sim
            watched = self.viewers > 0 or time.time() - self.last_poll < 1.0
            if sim is None or not watched:  # nobody watching the 3D view: leave the CPU to the physics
                time.sleep(0.1)
                continue
            try:
                if sim.m is not self._model:  # new robot, or objects spawned
                    first = self._model is None or self._renderer is None
                    if self._renderer is not None:
                        self._renderer.close()
                    self._model = sim.m
                    self._renderer = mujoco.Renderer(sim.m, self.height, self.width)
                    scene_id = id(self.host.scene) if self.host.scene is not None else id(sim)
                    if first or self.cam["follow"] or scene_id != getattr(self, "_scene_id", None):
                        self.reframe()
                    self._scene_id = scene_id
                with sim._lock:
                    d = copy.copy(sim.d)
                if self.cam["follow"] and sim.profile.floating_base and not sim.pinned:
                    # Keep a walking/driving robot in view: ease the look-at point towards its base.
                    base = d.xpos[sim._bid(sim.profile.root_body)]
                    look = np.asarray(self.cam["lookat"], dtype=float)
                    look[:2] += 0.08 * (base[:2] - look[:2])
                    self.cam["lookat"] = [float(x) for x in look]
                cam = mujoco.MjvCamera()
                cam.lookat[:] = self.cam["lookat"]
                cam.azimuth, cam.elevation, cam.distance = self.cam["azimuth"], self.cam["elevation"], self.cam["distance"]
                self._renderer.update_scene(d, cam)
                self._draw_markers(sim)
                buf = io.BytesIO()
                Image.fromarray(self._renderer.render()).save(buf, format="JPEG", quality=82)
                self.frame, self.seq = buf.getvalue(), self.seq + 1
            except Exception as e:  # keep streaming even if one frame fails (e.g. during a model swap)
                print(f"[anybot] render error: {e}")
                self._model = None
                time.sleep(0.2)
            count += 1
            if time.perf_counter() - t_fps > 1:
                self.fps_measured, count, t_fps = count / (time.perf_counter() - t_fps), 0, time.perf_counter()
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))


def _marker(scn: mujoco.MjvScene, pos, alpha: float, scale: float = 1.0) -> None:
    """A target pin: dark disc and pole (readable on any floor colour) with an orange ball on top."""
    p = np.asarray(pos, dtype=float)
    ink, orange = [0.1, 0.13, 0.17, alpha], [0.36, 0.48, 0.62, min(1.0, alpha + 0.3)]
    parts = (
        (mujoco.mjtGeom.mjGEOM_CYLINDER, [0.12 * scale, 0.004, 0.0], 0.004, ink),
        (mujoco.mjtGeom.mjGEOM_CYLINDER, [0.008 * scale + 0.002, 0.17 * scale, 0.0], 0.17 * scale, ink),
        (mujoco.mjtGeom.mjGEOM_SPHERE, [0.045 * scale, 0.0, 0.0], 0.36 * scale, orange),
    )
    for kind, size, dz, rgba in parts:
        if scn.ngeom >= scn.maxgeom:
            return
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], kind, np.array(size), p + [0, 0, dz], np.eye(3).flatten(),
                            np.array(rgba, dtype=np.float32))
        scn.ngeom += 1


def pick(u: float, v: float) -> dict | None:
    """Cast a ray from the stream camera through image point (u, v) ∈ [0,1]² and return what it hits."""
    sim = host.sim
    if sim is None:
        return None
    c = streamer.cam
    az, el = np.radians(c["azimuth"]), np.radians(c["elevation"])
    fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    origin = np.asarray(c["lookat"]) - c["distance"] * fwd
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    t = np.tan(np.radians(sim.m.vis.global_.fovy) / 2)
    ray = fwd + (2 * u - 1) * t * (streamer.width / streamer.height) * right + (1 - 2 * v) * t * up
    ray /= np.linalg.norm(ray)
    with sim._lock:
        m, d = sim.m, copy.copy(sim.d)
    geomid = np.zeros(1, dtype=np.int32)
    dist = mujoco.mj_ray(m, d, origin, ray, None, 1, -1, geomid)
    if dist < 0:  # nothing hit: use the ground plane
        if ray[2] >= 0:
            return None
        dist, geomid[0] = -origin[2] / ray[2], -1
    point = origin + dist * ray
    gid = int(geomid[0])
    body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[gid])) if gid >= 0 else None
    return {"point": [round(float(x), 3) for x in point], "body": body}


host = ToolHost()
streamer = Streamer(host)
foxglove = FoxgloveBridge(host, port=int(os.environ.get("ANYBOT_FOXGLOVE_PORT", "8767")))
FOXGLOVE_ON = os.environ.get("ANYBOT_FOXGLOVE", "0") == "1"  # optional: the studio has its own sensor view
sensor_feed = SensorFeed(host)


# ---------------------------------------------------------------------- routes

# ---------------------------------------------------------------------- robot library + previews

# A small, verified starter set (all from MuJoCo Menagerie, one download). Any other name or file
# path can still be previewed and spawned from the search box.
CATALOG = {
    "Arms": ["panda", "xarm7", "ur5e", "so_arm100"],
    "Mobile (AMR)": ["amr", "tidybot", "stretch_3"],
    "Quadrupeds": ["go2", "spot", "anymal_c"],
    "Humanoids": ["g1"],
}
PREVIEWS = Path.home() / ".anybot" / "previews"
_preview_lock = threading.Lock()


def build_preview(robot: str) -> dict:
    """Load a robot off-screen (not into the live scene) and describe it, with a rendered picture."""
    from anybot.sim import Sim

    key = "".join(c if c.isalnum() or c in "-_+" else "_" for c in robot)
    meta, png = PREVIEWS / f"{key}.json", PREVIEWS / f"{key}.png"
    is_file = "/" in robot or robot.endswith((".urdf", ".xml"))
    if meta.exists() and png.exists() and not is_file:
        return json.loads(meta.read_text())
    with _preview_lock:
        sim = Sim(robot, realtime=False)
        bench_only = sim.profile.category == "humanoid" and sim.gait is None
        try:
            p = sim.profile
            counts: dict[str, int] = {}
            for limb in p.limbs:
                counts[limb.kind] = counts.get(limb.kind, 0) + 1
            info = {
                "robot": robot, "name": p.name, "category": p.category, "floating_base": p.floating_base,
                "mass_kg": p.mass_kg, "height_m": p.height_m, "n_joints": len(p.joints), "n_actuators": len(p.actuators),
                "limbs": counts, "wheels": len(p.wheels), "skills": sim.skills(), "notes": list(sim.import_notes),
                "grasp": (("hand" if any(a.assisted for a in sim.arms.values())
                           else "gripper" if any(a.gripper_actuators for a in sim.arms.values()) else "suction")
                          if sim.arms else None),
                "key": key, "bench_only": bench_only,
            }
            PREVIEWS.mkdir(parents=True, exist_ok=True)
            png.write_bytes(sim.snapshot(480, 360, azimuth=140, elevation=-18))
            meta.write_text(json.dumps(info))
        finally:
            sim.close()
    return info


async def catalog(_: Request) -> Response:
    from anybot import worlds

    from anybot.team import TEAMS

    teams = [{"id": k, "label": t["label"], "description": t["description"],
              "members": [{"role": m["role"], "robot": m["robot"]} for m in t["members"]]} for k, t in TEAMS.items()]
    from anybot.assets import SCANNED_OBJECTS

    return JSONResponse({"groups": CATALOG, "worlds": worlds.all_worlds(), "teams": teams, "assets": list(SCANNED_OBJECTS)})


async def preview(request: Request) -> Response:
    robot = request.query_params.get("robot", "").strip()
    if not robot:
        return JSONResponse({"ok": False, "error": "Which robot?"}, status_code=400)
    try:
        info = await anyio.to_thread.run_sync(lambda: build_preview(robot))
    except Exception as e:  # noqa: BLE001 - any import failure is the answer the user needs
        return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}, status_code=400)
    return JSONResponse({"ok": True, **info})


async def preview_image(request: Request) -> Response:
    robot = request.query_params.get("robot", "").strip()
    try:
        info = await anyio.to_thread.run_sync(lambda: build_preview(robot))
    except Exception:  # noqa: BLE001
        return Response(status_code=404)
    return FileResponse(PREVIEWS / f"{info['key']}.png", headers={"Cache-Control": "max-age=3600"})


async def index(_: Request) -> Response:
    return FileResponse(WEB / "index.html")


async def health(_: Request) -> Response:
    return JSONResponse({"ok": True, "robot": host.sim.name if host.sim else None})


async def stream(_: Request) -> Response:
    async def frames():
        last = -1
        streamer.viewers += 1
        try:
            while True:
                if streamer.seq != last and streamer.frame is not None:
                    last, jpg = streamer.seq, streamer.frame
                    yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n"
                await asyncio.sleep(1 / 60)
        finally:
            streamer.viewers -= 1

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


async def events(_: Request) -> Response:
    async def feed():
        last = -1
        while True:
            if host.version != last:
                last = host.version
                yield f"data: {json.dumps({'version': last, 'events': host.events[-60:]})}\n\n"
            await asyncio.sleep(0.2)

    return StreamingResponse(feed(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


async def events_json(request: Request) -> Response:
    """Polled version of /api/events (no long-lived connection)."""
    since = int(request.query_params.get("since", "-1"))
    if host.version == since:
        return JSONResponse({"version": since})
    return JSONResponse({"version": host.version, "events": host.events[-60:]})


async def frame(request: Request) -> Response:
    """The latest 3D frame (204 if there is none newer than `after`): polled by the UI one at a time."""
    streamer.last_poll = time.time()
    after = int(request.query_params.get("after", "-1"))
    jpg, seq = streamer.frame, streamer.seq
    if jpg is None or seq == after:
        return Response(status_code=204, headers={"x-seq": str(seq)})
    return Response(jpg, media_type="image/jpeg", headers={"x-seq": str(seq), "Cache-Control": "no-store"})


async def state(_: Request) -> Response:
    sim = host.sim
    if sim is None:
        return JSONResponse({"robot": None, "fps": round(streamer.fps_measured, 1)})
    info, st = await anyio.to_thread.run_sync(lambda: (host.robot_info(), sim.state()))
    joints = [
        {"name": j.name, "range": j.range, "value": st["joints"].get(j.name), "actuator": j.actuator_type}
        for j in sim.profile.joints
        if j.type in ("hinge", "slide")
    ]
    return JSONResponse({
        "robot": info, "state": st, "joints": joints, "limbs": [l.__dict__ for l in sim.profile.limbs],
        "busy": host.busy > 0, "fps": round(streamer.fps_measured, 1),
    })


async def call_tool(request: Request) -> Response:
    name = request.path_params["name"]
    try:
        args = await request.json()
    except json.JSONDecodeError:
        args = {}
    source = request.headers.get("x-anybot-source", "ui")
    streamer.preview = None  # whatever was pending is now either confirmed or superseded
    if request.headers.get("x-anybot-async") == "1":
        # The UI does not wait on long jobs (a walk can take a minute): it would hold one of the browser's
        # few connections per site and stall everything else. Outcome and errors arrive with the events.
        def job():
            try:
                host.call(name, args, source)
            except Exception:  # noqa: BLE001 - already recorded as an error event by host.call
                pass
            if name in ("load_robot", "set_world") or args.get("task") == "reset_scene":
                streamer.reframe()

        threading.Thread(target=job, name=f"anybot-{name}", daemon=True).start()
        return JSONResponse({"ok": True, "started": True})
    try:
        result = await anyio.to_thread.run_sync(lambda: host.call(name, args, source))
    except SkillError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    if name in ("load_robot", "set_world") or args.get("task") == "reset_scene":
        streamer.reframe()
    if isinstance(result, bytes):
        return Response(result, media_type="image/png")
    return JSONResponse({"ok": True, "result": _jsonable(result)})


async def pick_point(request: Request) -> Response:
    """Step 1 of pointing: resolve the click to a 3D point and say what confirming it would do."""
    body = await request.json()
    hit = pick(float(body["u"]), float(body["v"]))
    if hit is None:
        return JSONResponse({"ok": False, "error": "Point at the floor or at an object."}, status_code=400)
    try:
        proposal = host.propose(*hit["point"], hit["body"])
    except SkillError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    streamer.preview = hit["point"]
    return JSONResponse({"ok": True, "hit": hit, **proposal})


async def point(request: Request) -> Response:
    """The 3D point under a click (for placing things), shown as the pending marker."""
    body = await request.json()
    hit = await anyio.to_thread.run_sync(lambda: pick(float(body["u"]), float(body["v"])))
    if hit is None:
        return JSONResponse({"ok": False, "error": "Nothing there: click the floor, a table or a shelf."}, status_code=400)
    streamer.preview = hit["point"]
    return JSONResponse({"ok": True, **hit})


async def clear_preview(_: Request) -> Response:
    streamer.preview = None
    return JSONResponse({"ok": True})


async def camera(request: Request) -> Response:
    body = await request.json()
    c = streamer.cam
    if body.get("reset"):
        c.update(azimuth=135.0, elevation=-22.0)
        streamer.reframe()
    c["azimuth"] = (c["azimuth"] + float(body.get("d_azimuth", 0))) % 360
    c["elevation"] = float(np.clip(c["elevation"] + float(body.get("d_elevation", 0)), -89, 10))
    c["distance"] = float(np.clip(c["distance"] * float(body.get("zoom", 1.0)), 0.2, 20))
    if "pan" in body:  # screen-space pan → move look-at in the ground plane
        dx, dy = body["pan"]
        az = np.radians(c["azimuth"])
        right = np.array([np.sin(az), -np.cos(az), 0.0])
        fwd = np.array([np.cos(az), np.sin(az), 0.0])
        s = c["distance"] * 0.0015
        c["lookat"] = list(np.asarray(c["lookat"]) - right * dx * s + fwd * dy * s)
        c["follow"] = False
    return JSONResponse({k: (list(map(float, v)) if k == "lookat" else v) for k, v in c.items()})


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


@asynccontextmanager
async def lifespan(_app):
    streamer.start()
    sensor_feed.start()
    if FOXGLOVE_ON:
        try:
            foxglove.start()
        except Exception as e:  # noqa: BLE001 - optional (port busy, sdk missing)
            print(f"[anybot] Foxglove bridge not started: {e}")
    yield
    sensor_feed.stop()
    if FOXGLOVE_ON:
        foxglove.stop()
    streamer.stop()
    host.close()


async def sensors_live(_: Request) -> Response:
    try:
        data = await anyio.to_thread.run_sync(sensor_feed.live)
    except Exception as e:  # noqa: BLE001 - during a model swap
        return JSONResponse({"robot": None, "error": str(e)})
    return JSONResponse(data)


async def sensors_map(_: Request) -> Response:
    png = await anyio.to_thread.run_sync(sensor_feed.map_png)
    if png is None:
        return Response(status_code=404)
    return Response(png, media_type="image/png", headers={"Cache-Control": "no-store"})


async def sensors_camera(request: Request) -> Response:
    jpg = sensor_feed.camera(request.query_params.get("mode", "rgb"))
    if jpg is None:
        return Response(status_code=204)
    return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


async def foxglove_info(_: Request) -> Response:
    from anybot.foxglove_bridge import LAYOUT

    return JSONResponse({"url": foxglove.url, "running": FOXGLOVE_ON and foxglove.server is not None, "layout": str(LAYOUT),
                         "open": f"https://app.foxglove.dev/view?ds=foxglove-websocket&ds.url={foxglove.url}"})


app = Starlette(
    routes=[
        Route("/", index),
        Route("/stream", stream),
        Route("/api/health", health),
        Route("/api/foxglove", foxglove_info),
        Route("/api/sensors/live", sensors_live),
        Route("/api/sensors/map.png", sensors_map),
        Route("/api/sensors/camera.jpg", sensors_camera),
        Route("/api/catalog", catalog),
        Route("/api/preview", preview),
        Route("/api/preview.png", preview_image),
        Route("/api/events", events),
        Route("/api/events.json", events_json),
        Route("/api/frame.jpg", frame),
        Route("/api/state", state),
        Route("/api/tools/{name}", call_tool, methods=["POST"]),
        Route("/api/camera", camera, methods=["POST"]),
        Route("/api/pick", pick_point, methods=["POST"]),
        Route("/api/preview/clear", clear_preview, methods=["POST"]),
        Route("/api/point", point, methods=["POST"]),
    ],
    lifespan=lifespan,
)


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    parser = argparse.ArgumentParser(prog="anybot studio")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--robot", help="robot to load at startup, e.g. panda")
    parser.add_argument("--world", default="warehouse", help="warehouse, lab, apartment, bench, empty or an imported scene")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    if args.robot:
        host.call("load_robot", {"robot": args.robot, "world": args.world}, source="studio")
    url = f"http://127.0.0.1:{args.port}"
    print(f"[anybot] studio running at {url}")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    # Browsers keep the video stream open forever: don't wait for it when shutting down.
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning", timeout_graceful_shutdown=1)


if __name__ == "__main__":
    main()
