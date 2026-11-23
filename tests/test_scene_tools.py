"""Adding and removing objects, the wrist camera, the AMR relay, and the humanoid working at floor level."""

import copy

import numpy as np
import pytest

from anybot import sensors
from anybot.host import ToolHost
from anybot.sim import Sim


def _host(robot: str, world: str = "warehouse") -> ToolHost:
    host = ToolHost()
    host.sim = Sim(robot, realtime=False, world=world)
    host._last_load = {"robot": robot, "world": world}
    return host


def test_objects_can_be_removed_even_from_a_hand():
    host = _host("panda")
    sim = host.sim
    name = sim.objects[1]
    host.call("pick_object", {"name": name})
    host.call("remove_object", {"name": name})
    assert name not in sim.objects and not sim.held
    host.call("reset")  # the keyframes still match the model
    sim.wait(0.5)


def test_scanned_assets_can_be_spawned_on_a_surface():
    host = _host("panda")
    out = host.call("spawn_asset", {"kind": "tea_box", "x": 0.45, "y": 0.1})
    host.sim.wait(1.0)
    assert out["name"] in host.sim.objects
    assert abs(host.sim.d.xpos[host.sim._bid(out["name"])][0] - 0.45) < 0.05


def test_arms_see_through_a_wrist_camera():
    sim = Sim("tidybot", realtime=False, world="warehouse")
    arm = next(iter(sim.arms.values()))
    assert sensors.wrist_arm(sim) is arm
    rgb, depth, pos, _ = sensors.Camera(sim.m, 160, 120).render(sim, copy.copy(sim.d))
    assert rgb.shape == (120, 160, 3) and np.linalg.norm(pos - sim._tcp(arm, sim.d)) < 0.45  # strapped on the forearm


@pytest.mark.slow
def test_amr_relay_moves_every_crate_with_carriers_swapping():
    host = ToolHost()
    host._load_team("amr_fleet", realtime=False)
    out = host.call("run_task", {"task": "relay"})
    assert out["errors"] == [] and sorted(out["on_table_b"]) == ["crate_1", "crate_2", "crate_3"]


@pytest.mark.slow
def test_humanoid_picks_up_from_the_floor_and_puts_it_back_down():
    host = _host("g1")
    sim = host.sim
    host.call("spawn_object", {"shape": "box", "size": 0.035, "color": "red", "x": 1.0, "y": 1.0, "name": "floor_box"})
    assert host.call("pick_object", {"name": "floor_box"})["grasp"] == "two hands"
    assert sim.d.xpos[sim._bid("floor_box")][2] > 0.7  # carried up at the chest
    out = host.call("place_object", {"x": 2.0, "y": 0.5})
    assert out["object_at"][2] < 0.05  # on the floor
    assert sim.d.xpos[sim._bid(sim.profile.root_body)][2] > 0.7  # standing again


def test_objects_can_be_moved_onto_another_surface():
    host = _host("tidybot")
    sim = host.sim
    name = sim.objects[0]
    out = host.call("move_object", {"name": name, "x": 0.0, "y": -2.6})  # onto table B
    sim.wait(0.5)
    p = sim.d.xpos[sim._bid(name)]
    assert abs(p[0]) < 0.02 and abs(p[1] + 2.6) < 0.02 and p[2] > 0.7, out
    host.call("move_object", {"name": name, "x": 1.5, "y": 0.0})  # and onto the floor
    sim.wait(0.5)
    assert sim.d.xpos[sim._bid(name)][2] < 0.06
