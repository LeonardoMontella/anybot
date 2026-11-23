"""Spot's own walk: tall stance, smooth body, catches pushes with its feet, fits where it spawns."""

import numpy as np

from anybot.sim import Sim


def _rp(sim, root):
    R = sim.d.xmat[root].reshape(3, 3)
    return np.degrees([np.arctan2(R[2, 1], R[2, 2]), -np.arcsin(np.clip(R[2, 0], -1, 1))])


def test_spot_walks_tall_and_level():
    sim = Sim("spot", realtime=False, world="empty")
    root = sim._bid(sim.profile.root_body)
    sim.wait(0.5)
    sim._start_locomotion()
    sim.gait.set_velocity(0.5, 0.0, 0.0)
    sim.wait(1.0)
    att, z = [], []
    for _ in range(150):
        sim.wait(0.02)
        att.append(_rp(sim, root))
        z.append(sim.d.xpos[root][2])
    att = np.array(att)
    assert np.mean(z) > 0.5  # standing tall, not crouched
    assert att[:, 1].std() < 0.5 and att[:, 0].std() < 1.0 and np.std(z) < 0.004  # no rocking or bobbing


def test_spot_catches_a_shove_with_its_feet():
    rng = np.random.default_rng(1)
    for _ in range(3):
        sim = Sim("spot", realtime=False, world="empty")
        root = sim._bid(sim.profile.root_body)
        sim._start_locomotion()
        sim.gait.set_velocity(0.3, 0.0, rng.uniform(-0.8, 0.8))
        sim.wait(1.0 + rng.uniform(0, 0.4))
        a = rng.uniform(0, 2 * np.pi)
        sim.d.xfrc_applied[root, :3] = [300 * np.cos(a), 300 * np.sin(a), 0]
        sim.wait(0.15)
        sim.d.xfrc_applied[root] = 0
        sim.wait(2.0)
        assert sim.d.xmat[root][8] > 0.9


def test_big_robots_spawn_clear_of_the_scan_walls():
    sim = Sim("spot", realtime=False, world="apartment")
    root = sim._bid(sim.profile.root_body)
    p0 = sim.d.xpos[root][:2].copy()
    sim.wait(1.0)
    assert sim.d.xmat[root][8] > 0.99 and np.linalg.norm(sim.d.xpos[root][:2] - p0) < 0.05  # stands still
