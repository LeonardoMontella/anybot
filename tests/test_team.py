"""Teams of robots in one scene, and the production line they run together."""

import numpy as np
import pytest

from anybot.host import SkillError, ToolHost


def _team(name: str) -> ToolHost:
    host = ToolHost()
    host._load_team(name, realtime=False)
    return host


def test_team_robots_share_one_scene_and_stand_still():
    host = _team("amr_line")
    scene = host.scene
    assert [r.name for r in scene.robots] == ["loader", "sorter", "carrier"]
    assert all(r.m is scene.m for r in scene.robots)
    roots = {r.name: scene.d.xpos[r._bid(r.profile.root_body)].copy() for r in scene.robots}
    scene.wait(2.0)
    for r in scene.robots:  # nobody drifts or falls while idle
        assert np.linalg.norm(scene.d.xpos[r._bid(r.profile.root_body)] - roots[r.name]) < 0.05, r.name
    info = host.robot_info()
    assert info["team"]["name"] == "amr_line" and {"belt_in", "belt_out", "tote"} <= set(info["zones"])
    assert [t["id"] for t in info["tasks"]][0] == "production"


def test_commands_target_one_team_member():
    host = _team("amr_line")
    carrier = host.scene.robot("carrier")
    start = carrier.d.xpos[carrier._bid(carrier.profile.root_body)][:2].copy()
    host.call("navigate_to", {"x": float(start[0]), "y": float(start[1] - 0.4), "robot": "carrier"})
    end = carrier.d.xpos[carrier._bid(carrier.profile.root_body)][:2]
    assert abs(end[1] - (start[1] - 0.4)) < 0.05
    with pytest.raises(SkillError):
        host.call("get_state", {"robot": "nobody"})
    host.call("select_robot", {"name": "sorter"})
    assert host.sim.name == "sorter"


def test_conveyor_carries_a_crate_to_the_stopper():
    host = _team("amr_line")
    scene = host.scene
    belt = scene.conveyors[0]
    loader = scene.robot("loader")
    loader.spawn_object("box", 0.035, [float(belt.start[0]), float(belt.start[1]), belt.centre[2] + 0.036], "red", name="probe")
    scene.wait(14.0)
    p = scene.d.xpos[loader._bid("probe")]
    assert abs(p[0] - belt.end[0]) < 0.03 and abs(p[2] - (belt.centre[2] + 0.035)) < 0.02


@pytest.mark.slow
@pytest.mark.parametrize("team", ["amr_line", "mixed_line"])
def test_production_line_delivers_every_crate(team):
    host = _team(team)
    out = host.call("run_task", {"task": "production"})
    assert out["errors"] == []
    assert out["sorted"] == ["crate_1", "crate_2", "crate_3"]
    assert out["delivered"]["at_dock"] and sorted(out["delivered"]["in_tote"]) == ["crate_1", "crate_2", "crate_3"]
