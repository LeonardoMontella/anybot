"""Live sensor and map view in Foxglove (https://foxglove.dev), without ROS.

The studio starts a Foxglove WebSocket server (default ws://localhost:8767). Open Foxglove (web or
desktop app) → "Open connection" → Foxglove WebSocket → that URL, and import the layout written to
~/.anybot/foxglove_layout.json (3D view + camera). Messages use the standard Foxglove schemas, which map
one-to-one to ROS 2 messages (LaserScan, CompressedImage, TF, OccupancyGrid-like Grid), so the same
topics can later be bridged to ROS 2 / Nav2.

Topics:
  /tf               world → base_link (selected robot), base_link → laser, base_link → camera
  /scan             2D lidar (LaserScan)
  /camera/image     robot camera (JPEG)          /camera/depth   metric depth (32FC1)
  /map              collision map of the world + obstacles seen by the lidar (Grid)
  /plan             the planned path and goal (SceneUpdate lines)
  /scene            every body as simple shapes, robots included (SceneUpdate)
"""

from __future__ import annotations

import copy
import io
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from anybot.host import ToolHost

PORT = 8767
LAYOUT = Path.home() / ".anybot" / "foxglove_layout.json"


def _ts(t: float):
    from foxglove.messages import Timestamp

    return Timestamp(sec=int(t), nsec=int((t % 1) * 1e9))


def _quat_yaw(yaw: float):
    from foxglove.messages import Quaternion

    return Quaternion(x=0.0, y=0.0, z=float(np.sin(yaw / 2)), w=float(np.cos(yaw / 2)))


def _pose(p, q=None):
    from foxglove.messages import Pose, Quaternion, Vector3

    q = q if q is not None else Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    return Pose(position=Vector3(x=float(p[0]), y=float(p[1]), z=float(p[2])), orientation=q)


def write_layout(path: Path = LAYOUT) -> Path:
    """A ready Foxglove layout: 3D view (scene, map, lidar, path) beside the robot camera and depth."""
    import foxglove.layouts as fl

    topics = {
        "/scene": fl.BaseRendererSceneUpdateTopicSettings(visible=True),
        "/plan": fl.BaseRendererSceneUpdateTopicSettings(visible=True),
        "/map": fl.BaseRendererGridTopicSettings(visible=True, color_mode="gradient", gradient=("#ffffff00", "#1a1d22ff"),
                                                 color_field="occupancy", draw_behind=True),
        "/scan": fl.BaseRendererLaserScanTopicSettings(visible=True),
    }
    three = fl.ThreeDeePanel(title="anybot 3D", config=fl.ThreeDeeConfig(
        follow_tf="base_link", topics=topics))
    cam = fl.ImagePanel(title="camera", config=fl.ImageConfig(image_mode=fl.ImageModeConfig(image_topic="/camera/image")))
    depth = fl.ImagePanel(title="depth", config=fl.ImageConfig(image_mode=fl.ImageModeConfig(image_topic="/camera/depth")))
    layout = fl.Layout(content=fl.SplitContainer(direction="row", items=[
        fl.SplitItem(proportion=2, content=three),
        fl.SplitItem(proportion=1, content=fl.SplitContainer(direction="column", items=[
            fl.SplitItem(content=cam), fl.SplitItem(content=depth)])),
    ]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(layout.to_json())
    return path


class FoxgloveBridge:
    def __init__(self, host: "ToolHost", port: int = PORT, rate: float = 10.0):
        self.host, self.port, self.rate = host, port, rate
        self.server = None
        self._running = False
        self._camera = None
        self._model = None
        self._map_sent = (None, 0.0)

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}"  # not "localhost": browsers may try IPv6 (::1) first

    def start(self) -> None:
        import foxglove

        self.server = foxglove.start_server(name="anybot", host="127.0.0.1", port=self.port)
        try:
            write_layout()
        except Exception as e:  # noqa: BLE001 - the layout is a convenience
            print(f"[anybot] foxglove layout not written: {e}")
        self._running = True
        threading.Thread(target=self._loop, name="anybot-foxglove", daemon=True).start()

    def stop(self) -> None:
        self._running = False
        if self.server is not None:
            self.server.stop()

    # ------------------------------------------------------------------ loop

    def _loop(self) -> None:
        period, k = 1.0 / self.rate, 0
        while self._running:
            t0 = time.perf_counter()
            sim = self.host.sim
            if sim is not None and sim.profile is not None:
                try:
                    self._publish(sim, k)
                except Exception as e:  # noqa: BLE001 - keep publishing through model swaps
                    print(f"[anybot] foxglove: {type(e).__name__}: {e}")
                    self._camera, self._model = None, None
                    time.sleep(0.5)
            k += 1
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))

    def _publish(self, sim, k: int) -> None:
        import foxglove
        from foxglove.messages import (
            CompressedImage, FrameTransform, FrameTransforms, LaserScan, RawImage, Vector3,
        )

        from anybot import sensors

        now = time.time()
        stamp = _ts(now)
        with sim._lock:
            d = copy.copy(sim.d)
            pos, yaw = sim._base_pose()
            scan = sensors.lidar(sim) if sim.profile.floating_base or sim.profile.planar_base else None
        # TF
        tfs = [FrameTransform(timestamp=stamp, parent_frame_id="world", child_frame_id="base_link",
                              translation=Vector3(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])), rotation=_quat_yaw(yaw))]
        if scan is not None:
            tfs.append(FrameTransform(timestamp=stamp, parent_frame_id="world", child_frame_id="laser",
                                      translation=Vector3(x=float(scan.origin[0]), y=float(scan.origin[1]), z=float(scan.origin[2])),
                                      rotation=_quat_yaw(scan.yaw)))
        foxglove.log("/tf", FrameTransforms(transforms=tfs))
        if scan is not None:
            ranges = [float(r) if np.isfinite(r) else float("nan") for r in scan.ranges]
            foxglove.log("/scan", LaserScan(timestamp=stamp, frame_id="laser", pose=_pose([0, 0, 0]),
                                            start_angle=scan.angle_min,
                                            end_angle=scan.angle_min + scan.angle_increment * (len(ranges) - 1),
                                            ranges=ranges))
        if k % 2 == 0:  # 5 Hz: camera and scene
            if self._model is not sim.m:
                if self._camera is not None:
                    self._camera.close()
                self._camera, self._model = sensors.Camera(sim.m), sim.m
            rgb, depth, cpos, cyaw = self._camera.render(sim, d)
            buf = io.BytesIO()
            Image.fromarray(rgb).save(buf, format="JPEG", quality=80)
            foxglove.log("/camera/image", CompressedImage(timestamp=stamp, frame_id="camera", data=buf.getvalue(), format="jpeg"))
            depth = np.where(depth > 20, np.nan, depth).astype(np.float32)
            foxglove.log("/camera/depth", RawImage(timestamp=stamp, frame_id="camera", width=depth.shape[1], height=depth.shape[0],
                                                   encoding="32FC1", step=depth.shape[1] * 4, data=depth.tobytes()))
            self._scene(sim, d, stamp)
        if k % 10 == 0:  # 1 Hz: map and plan
            self._map(sim, stamp)

    def _scene(self, sim, d: mujoco.MjData, stamp) -> None:
        """Every collision-relevant geom as a simple shape (meshes as their bounding box)."""
        import foxglove
        from foxglove.messages import (
            Color, CubePrimitive, CylinderPrimitive, SceneEntity, SceneUpdate, SpherePrimitive, Vector3,
        )

        m = sim.m
        cubes, spheres, cyls = [], [], []
        G = mujoco.mjtGeom
        for g in range(m.ngeom):
            t = m.geom_type[g]
            if t in (G.mjGEOM_PLANE, G.mjGEOM_HFIELD) or m.geom_rgba[g, 3] < 0.05:
                continue
            if m.geom_group[g] > 2:
                continue
            size = m.geom_size[g]
            rgba = m.geom_rgba[g]
            if m.geom_matid[g] >= 0:
                rgba = m.mat_rgba[m.geom_matid[g]]
            color = Color(r=float(rgba[0]), g=float(rgba[1]), b=float(rgba[2]), a=float(rgba[3]))
            R = d.geom_xmat[g].reshape(3, 3)
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, d.geom_xmat[g])
            from foxglove.messages import Quaternion

            quat = Quaternion(x=float(q[1]), y=float(q[2]), z=float(q[3]), w=float(q[0]))
            p = d.geom_xpos[g]
            if t == G.mjGEOM_BOX:
                cubes.append(CubePrimitive(pose=_pose(p, quat), size=Vector3(x=2 * size[0], y=2 * size[1], z=2 * size[2]), color=color))
            elif t == G.mjGEOM_SPHERE:
                spheres.append(SpherePrimitive(pose=_pose(p, quat), size=Vector3(x=2 * size[0], y=2 * size[0], z=2 * size[0]), color=color))
            elif t in (G.mjGEOM_CYLINDER, G.mjGEOM_CAPSULE):
                h = 2 * size[1] + (2 * size[0] if t == G.mjGEOM_CAPSULE else 0)
                cyls.append(CylinderPrimitive(pose=_pose(p, quat), size=Vector3(x=2 * size[0], y=2 * size[0], z=h),
                                              bottom_scale=1.0, top_scale=1.0, color=color))
            else:  # mesh, ellipsoid...: its bounding box
                c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
                if np.any(h > 6):  # a whole scanned room: shown by /map instead
                    continue
                cubes.append(CubePrimitive(pose=_pose(p + R @ c, quat), size=Vector3(x=2 * h[0], y=2 * h[1], z=2 * h[2]), color=color))
        ent = SceneEntity(timestamp=stamp, frame_id="world", id="world", cubes=cubes, spheres=spheres, cylinders=cyls)
        foxglove.log("/scene", SceneUpdate(entities=[ent]))

    def _map(self, sim, stamp) -> None:
        import foxglove
        from foxglove.messages import (
            Color, Grid, LinePrimitive, PackedElementField, PackedElementFieldNumericType, Point3, SceneEntity,
            SceneUpdate, Vector2,
        )

        planner = sim.scene.navmap()
        if planner is None:
            return
        cells = np.where(planner.static, 100, np.where(planner.dynamic, 60, 0)).astype(np.uint8)
        rows, cols = cells.shape
        foxglove.log("/map", Grid(
            timestamp=stamp, frame_id="world", pose=_pose([planner.lo[0], planner.lo[1], 0.005]), column_count=cols,
            cell_size=Vector2(x=planner.cell, y=planner.cell), row_stride=cols, cell_stride=1,
            fields=[PackedElementField(name="occupancy", offset=0, type=PackedElementFieldNumericType.Uint8)],
            data=cells.tobytes()))
        route = planner.last_route
        pos, _ = sim._base_pose()
        if route:
            pts = [Point3(x=float(pos[0]), y=float(pos[1]), z=0.05)] + [Point3(x=float(x), y=float(y), z=0.05) for x, y in route]
            line = LinePrimitive(thickness=0.05, points=pts, color=Color(r=1.0, g=0.31, b=0.0, a=1.0))
            foxglove.log("/plan", SceneUpdate(entities=[SceneEntity(timestamp=stamp, frame_id="world", id="plan", lines=[line])]))
