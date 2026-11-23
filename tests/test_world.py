"""Warehouse world, one-press tasks and the point → confirm flow."""

import httpx
import pytest

from anybot import studio
from anybot.host import ToolHost
from anybot.sim import Sim, SkillError


def _host(robot: str) -> ToolHost:
    host = ToolHost()
    host.sim = Sim(robot, realtime=False, world="warehouse")
    host._last_load = {"robot": robot, "world": "warehouse"}
    return host


def test_arm_gets_a_workstation_scaled_to_its_reach():
    big, small = Sim("panda", realtime=False, world="warehouse"), Sim("so_arm100", realtime=False, world="warehouse")
    assert set(big.zones) == {"pad_a", "pad_b"} and len(big.objects) == 4
    assert small.object_meta["box_orange_1"]["size"] < big.object_meta["box_orange_1"]["size"]


def test_mobile_robot_gets_floor_zones_and_no_boxes():
    go2 = Sim("go2", realtime=False, world="warehouse")
    assert set(go2.zones) == {"zone_a", "zone_b", "dock"} and not go2.objects


def test_humanoid_carries_a_scanned_object_between_tables():
    host = _host("g1")
    host.sim.close()
    host.sim = Sim("g1", realtime=False, world="lab")
    sim = host.sim
    assert {"table_a", "table_b"} <= set(sim.zones) and len(sim.objects) == 3
    assert [t["id"] for t in host.tasks()][:2] == ["carry_to_b", "carry_to_a"]
    sim.wait(1.0)
    obj = sim.objects[0]
    assert sim.pick(obj)["grasp"] == "two hands"  # a fingerless humanoid holds it between its palms
    x, y = sim.stations["table_b"]["slots"][1]
    out = sim.place(x, y)
    assert out["error_m"] < 0.08 and out["object_at"][2] > 0.6  # on table B, not on the floor
    assert sim.state()["base"]["pos"][2] > 0.7  # still standing


def test_urdf_importer_adapts_what_mujoco_cannot_read(tmp_path):
    from anybot.loader import load_robot

    urdf = tmp_path / "odd.urdf"
    urdf.write_text("""<robot name="odd">
      <link name="base"><inertial><mass value="1"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
        <visual><geometry><mesh filename="package://odd/meshes/base.dae"/></geometry></visual>
        <collision><geometry><cylinder radius="0" length="0.1"/></geometry></collision></link>
      <link name="tip"><inertial><mass value="0.01"/><inertia ixx="1e-6" iyy="1e-6" izz="1e-6" ixy="0" ixz="0" iyz="0"/></inertial></link>
      <joint name="j" type="revolute"><parent link="base"/><child link="tip"/><axis xyz="0 1 0"/>
        <limit lower="-1" upper="1" effort="1" velocity="1"/></joint></robot>""")
    robot = load_robot(str(urdf))
    notes = " ".join(robot.notes)
    assert "dropped" in notes and "zero-sized" in notes and "servos added" in notes
    sim = Sim(str(urdf), realtime=False)
    sim.set_joints({"j": 0.5})  # a 10 g link with an auto-tuned servo: stable and on target
    assert abs(sim.state()["joints"]["j"] - 0.5) < 0.05


def test_empty_world_has_no_zones_or_tasks():
    host = ToolHost()
    host.sim = Sim("panda", realtime=False)
    assert host.sim.zones == {} and host.tasks() == []


def test_unknown_world_is_rejected():
    with pytest.raises(SkillError, match="Unknown world"):
        ToolHost().call("load_robot", {"robot": "panda", "world": "moon"})


def test_sort_task_puts_each_colour_on_its_pad():
    host = _host("panda")
    assert [t["id"] for t in host.tasks()] == ["sort", "stack", "reset_scene"]
    out = host.call("run_task", {"task": "sort"}, "ui")
    assert out["moved"] == 4
    sim = host.sim
    for name, pos in sim.state()["objects"].items():
        pad = sim.zones["pad_a" if "orange" in name else "pad_b"]
        assert abs(pos[0] - pad[0]) < sim.pad_half and abs(pos[1] - pad[1]) < sim.pad_half, name
    # Every step is in the log, attributed to whoever started the task.
    assert [e["tool"] for e in host.events].count("pick_object") == 4
    assert {e["source"] for e in host.events} == {"ui"}


def test_stack_task_builds_a_tower():
    host = _host("panda")
    out = host.call("run_task", {"task": "stack"})
    heights = [out["objects"][o][2] for o in out["tower"]]
    assert heights[0] < heights[1] < heights[2] and heights[2] > 0.11


def test_go_to_zone_for_a_walking_robot():
    host = _host("go2")
    host.sim.wait(0.5)
    out = host.call("run_task", {"task": "go_zone_a"})
    assert out["reached"]
    with pytest.raises(SkillError, match="No zone"):
        host.call("go_to_zone", {"zone": "pad_a"})


def test_proposals_depend_on_context():
    host = _host("panda")
    sim = host.sim
    box = sim.state()["objects"]["box_orange_1"]
    assert host.propose(*box, "box_orange_1") == {"tool": "pick_object", "args": {"name": "box_orange_1"}, "label": "PICK BOX_ORANGE_1"}
    assert host.propose(0.3, 0.0, 0.0, None)["tool"] == "go_to_point"
    sim.pick("box_orange_1")
    pad = sim.zones["pad_b"]
    p = host.propose(pad[0], pad[1], 0.0, None)
    assert p["tool"] == "place_object" and p["label"].endswith("ON PAD B")
    assert host.propose(0, 0, 0, "box_blue_1")["args"] == {"on": "box_blue_1"}
    go2 = _host("go2")
    assert go2.propose(3.1, 2.1, 0.0, None)["label"] == "WALK TO ZONE A"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_pick_previews_and_only_confirm_acts():
    studio.host.sim = Sim("go2", realtime=False, world="warehouse")
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=studio.app), base_url="http://studio") as c:
            await c.post("/api/camera", json={"reset": True})
            n_events = len(studio.host.events)
            r = (await c.post("/api/pick", json={"u": 0.5, "v": 0.8})).json()
            assert r["ok"] and r["tool"] == "navigate_to" and r["label"].startswith("WALK")
            assert studio.streamer.preview == r["hit"]["point"]
            assert len(studio.host.events) == n_events and not studio.host.sim.navigating  # nothing ran
            await c.post("/api/preview/clear")
            assert studio.streamer.preview is None
            s = (await c.get("/api/state")).json()
            assert s["robot"]["world"] == "warehouse" and len(s["robot"]["tasks"]) == 4
    finally:
        studio.host.sim = None
