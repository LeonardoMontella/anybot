"""Generic trot gait, navigation and click-to-go picking."""

import mujoco
import numpy as np
import pytest

from anybot import studio
from anybot.host import ToolHost
from anybot.sim import Sim, SkillError


@pytest.fixture(scope="module")
def go2():
    sim = Sim("go2", realtime=False)
    sim.wait(0.5)
    return sim


def test_quadrupeds_get_a_trot_gait(go2):
    assert go2.gait is not None
    assert {"drive", "navigate_to"} <= set(go2.skills())
    # Diagonal pairs move together.
    phases = {leg.name: leg.phase_offset for leg in go2.gait.legs}
    assert phases["FL_calf"] == phases["RR_calf"] != phases["FR_calf"] == phases["RL_calf"]
    assert Sim("panda", realtime=False).gait is None


def test_go2_trots_forward_and_stays_upright(go2):
    out = go2.drive(0.3, 0.0, 3.0)
    assert out["end"][0] - out["start"][0] > 0.3
    base = go2.state()["base"]
    assert base["pos"][2] > 0.22 and abs(base["rpy_deg"][0]) < 10 and abs(base["rpy_deg"][1]) < 10


def test_go2_navigates_to_a_point_and_back(go2):
    assert go2.navigate_to(1.0, 0.8)["reached"]
    assert go2.navigate_to(0.0, 0.0)["reached"]
    assert go2.target is None and not go2.navigating


def test_arm_cannot_navigate():
    with pytest.raises(SkillError, match="cannot move its base"):
        Sim("panda", realtime=False).navigate_to(1, 1)


def test_go_to_point_means_reach_for_an_arm():
    host = ToolHost()
    host.call("load_robot", {"robot": "panda"})
    host.sim.close()
    host.sim = Sim("panda", realtime=False)
    out = host.call("go_to_point", {"x": 0.45, "y": 0.1, "z": 0.0})
    assert out["mode"] == "reach" and out["error_m"] < 0.01


def test_pick_hits_the_object_under_the_cursor():
    sim = Sim("go2", realtime=False)
    sim.spawn_object("sphere", 0.05, [0.8, -0.5, 0.05], "red", name="probe", fixed=True)
    mujoco.mj_forward(sim.m, sim.d)
    studio.host.sim = sim
    try:
        cam = studio.streamer.cam
        cam.update(azimuth=120.0, elevation=-30.0, distance=3.0, lookat=[0.2, 0.0, 0.2])
        W, H = studio.streamer.width, studio.streamer.height
        with mujoco.Renderer(sim.m, H, W) as r:
            mc = mujoco.MjvCamera()
            mc.lookat[:], mc.azimuth, mc.elevation, mc.distance = cam["lookat"], 120, -30, 3.0
            r.update_scene(sim.d, mc)
            img = r.render().astype(int)
        ys, xs = np.nonzero((img[:, :, 0] - img[:, :, 1] > 60) & (img[:, :, 0] - img[:, :, 2] > 60))
        hit = studio.pick(xs.mean() / W, ys.mean() / H)
        assert hit["body"] == "probe"
        assert np.linalg.norm(np.array(hit["point"][:2]) - [0.8, -0.5]) < 0.06
        # Off to the side the ray lands on the ground plane.
        assert abs(studio.pick(0.2, 0.9)["point"][2]) < 1e-6
    finally:
        studio.host.sim = None


@pytest.mark.parametrize("robot", ["spot", "a1", "anymal_c"])
def test_other_quadrupeds_navigate(robot):
    sim = Sim(robot, realtime=False)
    sim.wait(0.5)
    assert sim.navigate_to(1.0, 0.8)["reached"]
    assert abs(sim.state()["base"]["rpy_deg"][0]) < 15


@pytest.mark.parametrize("robot", ["stretch", "stretch_3"])
def test_wheeled_robots_navigate(robot):
    sim = Sim(robot, realtime=False)
    sim.wait(0.5)
    assert sim.navigate_to(1.0, 0.5)["reached"]


def test_g1_walks_with_the_unitree_policy():
    g1 = Sim("g1", realtime=False)
    assert type(g1.gait).__name__ == "UnitreeG1Walk"
    g1.wait(2.0)  # balancing in place
    assert g1.state()["base"]["pos"][2] > 0.7
    out = g1.navigate_to(1.5, 1.0)
    assert out["reached"] and g1.state()["base"]["pos"][2] > 0.7


def test_pinned_g1_has_no_locomotion():
    g1 = Sim("g1", pin_base=True, realtime=False)
    assert g1.gait is None and "navigate_to" not in g1.skills()
