"""Skill tests run with realtime=False: simulated time advances as fast as the CPU allows."""

import numpy as np
import pytest

from anybot.sim import Sim, SkillError
from test_profile import DIFF_DRIVE


@pytest.fixture(scope="module")
def panda():
    return Sim("panda", realtime=False)


def test_gripper_direction_is_calibrated():
    # Panda opens at 255, xArm7 opens at 0: anybot must find both on its own.
    assert Sim("panda", realtime=False).arms["hand"].gripper_open == [255.0]
    xarm = Sim("xarm7", realtime=False)
    assert next(iter(xarm.arms.values())).gripper_open == [0.0]


@pytest.mark.parametrize("robot", ["panda", "ur5e", "xarm7"])
def test_end_effector_reaches_target(robot):
    sim = Sim(robot, realtime=False)
    start = np.array(next(iter(sim.state()["end_effectors"].values())))
    result = sim.move_end_effector(list(start + [0.05, 0.05, -0.05]))
    assert result["error_m"] < 0.005


def test_unreachable_target_is_explained(panda):
    with pytest.raises(SkillError, match="out of reach"):
        panda.move_end_effector([3.0, 0, 0.5])


def test_pick_and_place():
    sim = Sim("panda", realtime=False)
    sim.spawn_object("box", 0.02, [0.5, 0.0, 0.02], "red", name="cube")
    sim.gripper("open")
    sim.move_end_effector([0.5, 0, 0.15], point_down=True)
    sim.move_end_effector([0.5, 0, 0.02], point_down=True)
    sim.gripper("close")
    sim.move_end_effector([0.5, 0.2, 0.25], point_down=True)
    cube = sim.state()["objects"]["cube"]
    assert cube[2] > 0.15 and cube[1] > 0.15


def test_bimanual_grippers_are_per_arm():
    sim = Sim("aloha", realtime=False)
    acts = [sim.m.actuator(a.gripper_actuators[0]).name for a in sim.arms.values()]
    assert sorted(acts) == ["left/gripper", "right/gripper"]


def test_torque_controlled_quadruped_stands_with_pd():
    go2 = Sim("go2", realtime=False)  # Menagerie Go2 has motor (torque) actuators
    assert go2.pd, "expected software PD servos"
    go2.wait(1.5)
    base = go2.state()["base"]
    assert base["pos"][2] > 0.22 and abs(base["rpy_deg"][1]) < 10


def test_pinned_humanoid_arm(tmp_path):
    g1 = Sim("g1", pin_base=True, realtime=False)
    assert g1.pinned and len(g1.arms) == 2
    assert g1.move_end_effector([0.3, 0.25, 1.1], arm := "left")["error_m"] < 0.005


def test_diff_drive(tmp_path):
    xml = tmp_path / "rover.xml"
    xml.write_text(DIFF_DRIVE)
    rover = Sim(str(xml), realtime=False)
    out = rover.drive(0.3, 0.0, 2.0)
    assert out["end"][0] - out["start"][0] > 0.2  # moved forward along x
    assert "drive" in rover.skills()


def test_snapshot_is_png(panda):
    png = panda.snapshot(160, 120)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_unknown_joint_suggests_names(panda):
    with pytest.raises(SkillError, match="Did you mean"):
        panda.set_joints({"joint_1": 0.3})
