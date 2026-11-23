"""Pick and place, and the context-dependent meaning of a click in the 3D view."""

import pytest

from anybot.host import ToolHost
from anybot.sim import Sim, SkillError

STACK_CASES = {  # robot: (red xy, blue xy, cube half-size)
    "panda": ([0.45, 0.0], [0.4, 0.2], 0.025),
    "xarm7": ([0.4, 0.0], [0.35, 0.2], 0.025),
    "aloha": ([-0.1, 0.05], [0.1, 0.05], 0.015),
}


@pytest.mark.parametrize("robot", STACK_CASES)
def test_pick_and_stack(robot):
    red, blue, size = STACK_CASES[robot]
    sim = Sim(robot, realtime=False)
    sim.spawn_object("box", size, [*red, size], "red", name="red")
    sim.spawn_object("box", size, [*blue, size], "blue", name="blue")
    assert sim.pick("red")["picked"] == "red"
    assert sim.state()["holding"]
    sim.place(on="blue", x=0, y=0)
    sim.wait(0.5)
    objs = sim.state()["objects"]
    assert abs(objs["red"][2] - objs["blue"][2] - 2 * size) < 0.01  # resting on top
    assert "holding" not in sim.state()


def test_bimanual_uses_the_arm_on_the_objects_side():
    sim = Sim("aloha", realtime=False)
    sim.spawn_object("box", 0.015, [-0.1, 0.05, 0.015], "red", name="left_cube")
    assert sim.pick("left_cube")["arm"].startswith("left")


def test_place_on_floor_and_errors():
    sim = Sim("panda", realtime=False)
    sim.spawn_object("box", 0.025, [0.45, 0.0, 0.025], "red", name="cube")
    with pytest.raises(SkillError, match="Not holding"):
        sim.place(0.4, 0.2)
    with pytest.raises(SkillError, match="Did you mean"):
        sim.pick("cub")
    sim.pick("cube")
    out = sim.place(0.35, -0.2)
    assert out["error_m"] < 0.03 and abs(out["object_at"][2] - 0.025) < 0.01


def test_arm_without_gripper_picks_suction_style():
    sim = Sim("ur5e", realtime=False)
    sim.spawn_object("box", 0.03, [0.4, 0.3, 0.03], "red", name="cube")
    sim.spawn_object("box", 0.03, [0.3, 0.5, 0.03], "blue", name="support")
    assert "pick_object" in sim.skills() and "gripper" not in sim.skills()
    assert sim.pick("cube")["grasp"] == "suction"
    sim.place(0, 0, on="support")
    sim.wait(0.5)
    objs = sim.state()["objects"]
    assert abs(objs["cube"][2] - objs["support"][2] - 0.06) < 0.01


def test_click_picks_then_places():
    host = ToolHost()
    host.sim = Sim("panda", realtime=False)
    host.sim.spawn_object("box", 0.025, [0.45, 0.0, 0.025], "red", name="red")
    host.sim.spawn_object("box", 0.025, [0.4, 0.2, 0.025], "blue", name="blue")
    host.click(0.45, 0.0, 0.05, body="red")  # click on the red cube → pick
    assert host.sim.held
    host.click(0.4, 0.2, 0.05, body="blue")  # click on the blue cube → stack
    tools = [e["tool"] for e in host.events]
    assert tools == ["pick_object", "place_object"]
    objs = host.sim.state()["objects"]
    assert objs["red"][2] > 0.06


def test_click_on_floor_with_empty_gripper_reaches():
    host = ToolHost()
    host.sim = Sim("panda", realtime=False)
    out = host.click(0.45, 0.1, 0.0, body=None)
    assert out["mode"] == "reach"
