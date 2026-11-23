import json

import pytest

from anybot import load_robot, profile
from anybot.introspect import introspect

# One or two robots per category, all from MuJoCo Menagerie via robot_descriptions.
EXPECTED = {
    "panda": "arm",
    "ur5e": "arm",
    "aloha": "dual_arm",
    "allegro_hand": "hand",
    "robotiq_2f85": "gripper",
    "go2": "quadruped",
    "spot": "quadruped",
    "g1": "humanoid",
    "h1": "humanoid",
    "cassie": "biped",
    "stretch": "mobile_manipulator",
    "skydio_x2": "aerial",
}


@pytest.mark.parametrize("name,category", EXPECTED.items())
def test_menagerie_classification(name, category):
    p = profile(name)
    assert p.category == category, p.summary()
    json.dumps(p.to_dict())  # profiles must be JSON-serialisable for MCP


def test_panda_details():
    p = profile("panda")
    arm = next(limb for limb in p.limbs if limb.kind == "arm")
    assert len(arm.joints) == 9  # 7 arm + 2 fingers
    assert arm.gripper and arm.tip == "hand"
    assert not p.floating_base


def test_go2_legs_have_three_joints():
    legs = [limb for limb in profile("go2").limbs if limb.kind == "leg"]
    assert len(legs) == 4 and all(len(leg.joints) == 3 for leg in legs)


TWO_LINK_URDF = """<?xml version="1.0"?>
<robot name="twolink">
  <link name="base"><inertial><mass value="2"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><geometry><box size="0.2 0.2 0.1"/></geometry></visual></link>
  {links}
</robot>"""

LINK = """
  <link name="l{i}"><inertial><mass value="1"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
    <collision><geometry><cylinder radius="0.03" length="0.3"/></geometry></collision></link>
  <joint name="j{i}" type="revolute"><parent link="{parent}"/><child link="l{i}"/>
    <origin xyz="0 0 0.2"/><axis xyz="0 1 0"/><limit lower="-2" upper="2" effort="50" velocity="2"/></joint>"""


def test_urdf_without_actuators_gets_servos(tmp_path):
    links = "".join(LINK.format(i=i, parent="base" if i == 0 else f"l{i - 1}") for i in range(4))
    urdf = tmp_path / "twolink.urdf"
    urdf.write_text(TWO_LINK_URDF.format(links=links))

    robot = load_robot(str(urdf))
    assert robot.injected_actuators == ["j0", "j1", "j2", "j3"]
    assert robot.model.nu == 4
    p = introspect(robot.model, robot.name, robot.injected_actuators)
    assert p.category == "arm"
    assert all(j.actuator_type == "position" for j in p.joints)
    assert p.joints[0].range == [-2.0, 2.0]


DIFF_DRIVE = """
<mujoco>
  <option integrator="implicitfast"/>
  <worldbody>
    <body name="chassis" pos="0 0 0.1">
      <freejoint/>
      <geom type="box" size="0.2 0.15 0.05" mass="5"/>
      <geom name="caster_f" type="sphere" size="0.05" pos="0.15 0 -0.05" condim="1" priority="1" mass="0.1"/>
      <geom name="caster_b" type="sphere" size="0.05" pos="-0.15 0 -0.05" condim="1" priority="1" mass="0.1"/>
      <body name="left" pos="0 0.18 0" euler="90 0 0">
        <joint name="left_axle" axis="0 0 1"/><geom type="cylinder" size="0.1 0.02" mass="0.5"/>
      </body>
      <body name="right" pos="0 -0.18 0" euler="90 0 0">
        <joint name="right_axle" axis="0 0 1"/><geom type="cylinder" size="0.1 0.02" mass="0.5"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <velocity joint="left_axle" kv="1"/><velocity joint="right_axle" kv="1"/>
  </actuator>
</mujoco>"""


def test_diff_drive_detected_without_wheel_in_names(tmp_path):
    xml = tmp_path / "rover.xml"
    xml.write_text(DIFF_DRIVE)
    p = profile(str(xml))
    assert p.category == "mobile"
    assert p.wheels == ["left_axle", "right_axle"]
    assert {j.actuator_type for j in p.joints} == {"velocity"}


def test_unknown_robot_lists_alternatives():
    with pytest.raises(ValueError, match="panda"):
        profile("definitely_not_a_robot")
