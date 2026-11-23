"""Data for the studio's built-in sensor view (an rviz-like panel, no external app needed).

* map: the collision map as a PNG (walls/furniture dark, obstacles the lidar found orange), rebuilt at most
  once a second and only when it changed;
* live: robot pose, lidar hits, planned path and goal (and the other robots of a team) as JSON;
* camera: the robot's camera (RGB or depth) as JPEG, rendered by a worker thread only while the view is
  open (nothing is computed when nobody looks).
"""

from __future__ import annotations

import copy
import io
import threading
import time
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from anybot.host import ToolHost


class SensorFeed:
    def __init__(self, host: "ToolHost", fps: float = 8.0):
        self.host, self.fps = host, fps
        self._wanted = {"rgb": 0.0, "depth": 0.0}  # last time each image was asked for
        self._jpeg: dict[str, bytes] = {}
        self._map_png: bytes | None = None
        self._map_key = None
        self._map_t = 0.0
        self._running = False

    def start(self) -> None:
        self._running = True
        threading.Thread(target=self._camera_loop, name="anybot-camera", daemon=True).start()

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------ camera (worker thread owns the GL context)

    def camera(self, mode: str) -> bytes | None:
        mode = "depth" if mode == "depth" else "rgb"
        self._wanted[mode] = time.time()
        return self._jpeg.get(mode)

    def _camera_loop(self) -> None:
        from anybot import sensors

        cam, model = None, None
        while self._running:
            t0 = time.perf_counter()
            sim = self.host.sim
            active = [m for m, t in self._wanted.items() if time.time() - t < 2.0]
            if sim is None or not active:
                time.sleep(0.1)
                continue
            try:
                if model is not sim.m:
                    if cam is not None:
                        cam.close()
                    cam, model = sensors.Camera(sim.m, 640, 480), sim.m
                with sim._lock:
                    d = copy.copy(sim.d)
                rgb, depth, _, _ = cam.render(sim, d)
                for mode in active:
                    if mode == "rgb":
                        img = rgb
                    else:  # near = warm, far = dark (turbo-like ramp), nothing = black
                        x = np.clip(depth / 6.0, 0, 1)
                        img = (255 * np.stack([np.clip(1.6 - 2 * x, 0, 1), np.clip(1.2 - 1.6 * abs(x - 0.35), 0, 1),
                                               np.clip(2 * x - 0.4, 0, 1) * (x < 0.999)], -1)).astype(np.uint8)
                    buf = io.BytesIO()
                    Image.fromarray(img).save(buf, format="JPEG", quality=78)
                    self._jpeg[mode] = buf.getvalue()
            except Exception as e:  # noqa: BLE001 - model swaps mid-render: try again with a new renderer
                print(f"[anybot] camera view: {e}")
                cam, model = None, None
                time.sleep(0.3)
            time.sleep(max(0.0, 1.0 / self.fps - (time.perf_counter() - t0)))

    # ------------------------------------------------------------------ map + live data

    def map_png(self) -> bytes | None:
        sim = self.host.sim
        planner = sim.scene.navmap() if sim is not None else None
        if planner is None:
            return None
        key = (id(planner), int(planner.dynamic.sum()))
        if self._map_png is None or (key != self._map_key and time.time() - self._map_t > 1.0):
            img = np.full(planner.static.shape + (4,), 0, np.uint8)
            img[planner.static] = (96, 110, 128, 235)  # mid grey: readable on light and dark themes
            img[planner.dynamic] = (255, 110, 30, 220)
            buf = io.BytesIO()
            Image.fromarray(img[::-1]).save(buf, format="PNG")  # row 0 = top of the picture = max y
            self._map_png, self._map_key, self._map_t = buf.getvalue(), key, time.time()
        return self._map_png

    def live(self) -> dict:
        from anybot import sensors

        sim = self.host.sim
        if sim is None:
            return {"robot": None}
        planner = sim.scene.navmap()
        mobile = bool(sim.profile.floating_base or sim.profile.planar_base)
        with sim._lock:
            pos, yaw = sim._base_pose()
            scan = sensors.lidar(sim) if mobile else None
        out = {
            "robot": sim.name, "pose": [round(float(pos[0]), 3), round(float(pos[1]), 3), round(float(yaw), 3)],
            "radius": round(sim._footprint_radius(), 3) if mobile else 0.2,
            "scan": np.round(scan.points(), 3).tolist() if scan is not None else [],
            "origin": [round(float(v), 3) for v in scan.origin[:2]] if scan is not None else None,
            "goal": [round(float(v), 3) for v in sim.target[:2]] if sim.target is not None else None,
            "path": [[round(float(a), 3), round(float(b), 3)] for a, b in (getattr(sim, "nav_path", None) or [])[1:]] if sim.navigating else [],
            "map": None,
            "others": [],
        }
        if planner is not None:
            out["map"] = {"lo": [float(planner.lo[0]), float(planner.lo[1])], "cell": planner.cell,
                          "rows": int(planner.static.shape[0]), "cols": int(planner.static.shape[1]),
                          "key": f"{id(planner)}-{int(planner.dynamic.sum())}"}
        scene = getattr(self.host, "scene", None)
        if scene is not None:
            for r in scene.robots:
                if r is sim:
                    continue
                p, y = r._base_pose()
                out["others"].append({"name": r.name, "pose": [round(float(p[0]), 3), round(float(p[1]), 3), round(float(y), 3)],
                                      "radius": round(r._footprint_radius(), 3) if (r.profile.floating_base or r.profile.planar_base) else 0.25})
        return out
