"""Collision maps, planned navigation through doors, sensors, and the humanoid's walking posture."""

import numpy as np
import pytest

from anybot import sensors, worlds
from anybot.host import ToolHost
from anybot.sim import Sim


def test_lab_map_has_walls_with_doors():
    sim = Sim("amr", realtime=False, world="lab")
    planner = sim.scene.navmap()
    assert planner.static[planner._cell([-3.5, worlds.LAB_WALL_Y])]  # wall
    door = np.mean(worlds.LAB_DOORS["north"])
    assert not planner.static[planner._cell([door, worlds.LAB_WALL_Y])]  # door
    route = planner.route([0.0, 0.0], [0.0, 2.0], radius=0.4)  # behind the north wall
    assert len(route) >= 2
    assert any(abs(p[0] - door) < 0.6 and abs(p[1] - worlds.LAB_WALL_Y) < 0.6 for p in route[:-1])


def test_lidar_sees_the_walls():
    sim = Sim("go2", realtime=False, world="lab")
    scan = sensors.lidar(sim)
    n = len(scan.ranges)
    left = scan.ranges[int(0.75 * n)]  # +90°: the north wall at y = 1.2
    assert abs(left - (worlds.LAB_WALL_Y - 0.05)) < 0.08
    assert np.isfinite(scan.ranges).sum() > n // 2


def test_robot_goes_around_a_box_it_only_sees_with_the_lidar():
    sim = Sim("amr", realtime=False, world="lab")
    sim.spawn_object("box", 0.25, [-1.6, 0.0, 0.25], "orange", name="obstacle", mass=5.0)
    out = sim.navigate_to(-3.6, 0.0)
    assert out["reached"]
    assert np.linalg.norm(sim.d.xpos[sim._bid("obstacle")][:2] - [-1.6, 0.0]) < 0.02  # not pushed


def test_amr_base_cannot_push_through_a_wall():
    sim = Sim("amr", realtime=False, world="lab")
    sim._planar_move(0.0, 2.2, None)  # straight at the north wall, no planner
    assert sim.d.xpos[sim._bid(sim.profile.root_body)][1] < worlds.LAB_WALL_Y


def test_sensor_tools():
    host = ToolHost()
    host.sim = Sim("go2", realtime=False, world="lab")
    scan = host.call("get_scan")
    assert abs(scan["nearest_m"]["left"] - 1.15) < 0.1 and len(scan["ranges_every_5deg"]) == 72
    png = host.call("camera_image")
    assert png[:4] == b"\x89PNG"


@pytest.mark.slow
def test_humanoid_walks_with_arms_down_after_manipulating():
    host = ToolHost()
    host.sim = sim = Sim("g1", realtime=False, world="warehouse")
    host._last_load = {"robot": "g1", "world": "warehouse"}
    host.call("pick_object", {"name": "crate_1"})
    host.call("place_object", {"x": 0.0, "y": 2.45})
    host.call("go_to_zone", {"zone": "zone_a"})
    arm_joints = [j for a in sim.arms.values() for j in a.joints]
    dev = max(abs(float(sim.d.qpos[sim.m.jnt_qposadr[j]]) - float(sim._home_cmd[sim.pos_act[j]])) for j in arm_joints)
    assert dev < 0.15


def test_stop_halts_a_running_task():
    import threading
    import time

    from anybot.host import SkillError

    host = ToolHost()
    host.sim = Sim("go2", realtime=False, world="lab")
    out = {}

    def job():
        try:
            host.call("navigate_to", {"x": 3.0, "y": 2.0}, "ui")  # through a door: takes a while
            out["result"] = "finished"
        except SkillError as e:
            out["result"] = str(e)

    t = threading.Thread(target=job)
    t.start()
    time.sleep(0.4)
    assert host.call("stop", {}, "ui")["stopped"]
    t.join(timeout=20)
    assert not t.is_alive() and "Stopped" in out["result"]
    assert host.events[-2]["status"] == "stopped" or any(e["status"] == "stopped" for e in host.events)
    base = host.sim.d.xpos[host.sim._bid(host.sim.profile.root_body)]
    assert base[2] > 0.2  # it stopped standing, it did not collapse
    host.call("navigate_to", {"x": 0.5, "y": 0.0}, "ui")  # and it takes new commands afterwards
