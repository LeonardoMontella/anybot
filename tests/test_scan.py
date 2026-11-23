"""The scanned apartment (digital twin): mesh to look at, height field to collide with, paths around walls."""

import json

import mujoco
import numpy as np
import pytest

from anybot import scan, worlds
from anybot.sim import Sim

FOLDER = scan.SCANS / "apartment_1"
pytestmark = pytest.mark.skipif(not (scan.SCANS / "apartment_1.glb").exists() and not (FOLDER / "scan.json").exists(),
                                reason="apartment scan not downloaded (50 MB)")


def test_scan_world_has_visual_mesh_and_collision_terrain():
    sim = Sim("go2", realtime=False, world="apartment")
    m = sim.m
    vis, col = m.geom("scan_visual"), m.geom("scan_collision")
    assert m.geom_contype[vis.id] == 0 and m.geom_contype[col.id] != 0
    assert m.mesh_vertnum[m.geom_dataid[vis.id]] > 100_000  # the real scan, not a placeholder
    assert len(sim.zones) >= 3
    # The height field matches the scan: where the mesh shows furniture, robots bump into it.
    meta = json.loads((FOLDER / "scan.json").read_text())
    h = np.load(FOLDER / "terrain.npy")
    lo = np.array(meta["lo"])
    agree = []
    for r, c in np.random.default_rng(0).choice(np.argwhere((h > 0.3) & (h < 1.5)), 40):
        x, y = lo[0] + c * meta["cell"], lo[1] + r * meta["cell"]
        hit = mujoco.mj_rayHfield(m, sim.d, col.id, np.array([x, y, 3.0]), np.array([0, 0, -1.0]))
        agree.append(abs((3.0 - hit) - h[r, c]) < 0.05)
    assert np.mean(agree) > 0.85


def test_planner_goes_around_walls():
    planner = Sim("amr", realtime=False, world="apartment").scene.navmap()
    spots = worlds.SCAN_SPOTS
    far = max(spots.values(), key=lambda p: p[1])  # the living room, behind a wall from the start
    route = planner.route([0.0, 0.0], far, radius=0.4)
    assert len(route) >= 3 and route[-1] == far


@pytest.mark.slow
def test_quadruped_walks_to_another_room():
    sim = Sim("go2", realtime=False, world="apartment")
    far = max(sim.zones.values(), key=lambda p: p[1])
    sim.navigate_to(*far)
    base = sim.d.xpos[sim._bid(sim.profile.root_body)]
    assert np.linalg.norm(base[:2] - np.array(far)) < 0.2 and base[2] > 0.2  # there, and standing


def test_robots_cannot_pass_through_scanned_walls():
    """Regression: thin scanned walls used to be removed as noise, and the AMR's ideal servos pushed through."""
    h = np.load(FOLDER / "terrain.npy")
    meta = json.loads((FOLDER / "scan.json").read_text())
    lo = np.array(meta["lo"])
    for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        wall = next(t for t in np.arange(0.2, 6, 0.02)
                    if h[int(round((t * d[1] - lo[1]) / meta["cell"])), int(round((t * d[0] - lo[0]) / meta["cell"]))] > 1.0)
        sim = Sim("amr", realtime=False, world="apartment")
        sim._planar_move(d[0] * (wall + 0.6), d[1] * (wall + 0.6), None)  # straight at it, no planner
        base = sim.d.xpos[sim._bid(sim.profile.root_body)][:2]
        assert float(np.dot(base, d)) < wall, d
