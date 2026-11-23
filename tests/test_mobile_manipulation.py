"""AMRs, mobile manipulation, humanoids that stand still, and the dexterous bench."""

import numpy as np

from anybot.host import ToolHost
from anybot.sim import Sim


def _host(robot: str, world: str, **kw) -> ToolHost:
    host = ToolHost()
    host.sim = Sim(robot, realtime=False, world=world, **kw)
    host._last_load = {"robot": robot, "world": world, **kw}
    return host


def test_holonomic_amr_is_mobile_and_navigates():
    amr = Sim("amr", realtime=False)
    assert amr.profile.category == "mobile" and len(amr.profile.planar_base) == 3
    out = amr.navigate_to(1.0, 0.5)
    assert out["reached"] and out["distance_left_m"] < 0.02


def test_amr_with_arm_carries_crates_between_tables():
    host = _host("tidybot", "warehouse")
    assert host.sim.profile.category == "mobile_manipulator"
    out = host.call("run_task", {"task": "carry_to_b"})
    assert out["moved"] == ["crate_1", "crate_2", "crate_3"]
    for name, pos in out["objects"].items():
        assert pos[1] < -2.3 and pos[2] > 0.7, name  # standing on table B


def test_walking_humanoid_stands_still_when_idle():
    g1 = Sim("g1", realtime=False)
    assert g1.frozen
    g1.wait(0.5)  # let the legs finish settling into the standing pose
    legs = g1.d.qpos[g1.gait.qadr].copy()
    g1.wait(1.0)
    assert np.max(np.abs(g1.d.qpos[g1.gait.qadr] - legs)) < 0.01  # no stepping in place
    assert g1.navigate_to(1.0, 0.4)["reached"]
    assert g1.frozen  # stops again on arrival


def test_humanoid_carries_objects_to_the_other_table():
    host = _host("g1", "lab")
    out = host.call("run_task", {"task": "carry_to_b"})
    assert out["moved"] == ["mug", "tea_box", "blue_mug"]
    for name, pos in out["objects"].items():
        assert pos[1] < -2.3 and pos[2] > 0.6, name


def test_pinned_humanoid_works_at_the_bench():
    host = ToolHost()
    info = host.call("load_robot", {"robot": "g1", "pin_base": True, "world": "lab"})
    assert info["name"] == "g1" and info["world"] == "bench" and info["base_pinned"]
    assert [t["id"] for t in host.tasks()][0] == "sort"
    host.sim.close()
    host.sim = Sim("g1", realtime=False, world="bench", pin_base=True)
    out = host.call("run_task", {"task": "sort"})
    assert out["success"], out["misplaced"]


def test_imported_humanoid_that_cannot_walk_goes_to_the_bench():
    """Any humanoid imported by name or file (the library only offers the G1) and without a gait."""
    host = ToolHost()
    info = host.call("load_robot", {"robot": "h1", "world": "warehouse"})
    host.sim.close()
    assert info["world"] == "bench" and info["base_pinned"]
