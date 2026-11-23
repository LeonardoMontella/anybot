"""Video → world: frame picking, and a rebuilt mesh (arbitrary size and orientation) turned into a world."""

import numpy as np
import pytest

from anybot import capture, scan, worlds


def _box(lo, hi):
    """Quads of an axis-aligned box (the faces a photogrammetry mesh of furniture would have)."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    q = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return v, q


def _room_obj(path, scale=0.31, seed=0):
    """A 5 x 4 m room, 2.6 m high, with a table and a tall cupboard; then shrunk and turned arbitrarily, as
    a mesh rebuilt from plain photos comes out (no real size, no gravity direction)."""
    parts = [_box((0, 0, -0.02), (5, 4, 0)), _box((0, 0, 2.6), (5, 4, 2.62)),  # floor, ceiling
             _box((-0.1, 0, 0), (0, 4, 2.6)), _box((5, 0, 0), (5.1, 4, 2.6)),  # walls
             _box((0, -0.1, 0), (5, 0, 2.6)), _box((0, 4, 0), (5, 4.1, 2.6)),
             _box((1.0, 2.6, 0), (2.2, 3.4, 0.75)),  # table
             _box((4.4, 0.2, 0), (5.0, 1.6, 2.0))]  # cupboard
    rng = np.random.default_rng(seed)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    lines, base = [], 1
    for v, quads in parts:
        for p in v:
            p = scale * (R @ np.asarray(p, float)) + [3.0, -1.0, 7.0]
            lines.append("v %.6f %.6f %.6f" % tuple(p))
        lines += ["f " + " ".join(str(base + i) for i in quad) for quad in quads]
        base += len(v)
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def scans(tmp_path, monkeypatch):
    monkeypatch.setattr(scan, "SCANS", tmp_path / "scans")
    return tmp_path


def test_a_rebuilt_room_becomes_a_level_metric_world(scans):
    obj = scans / "room.obj"
    _room_obj(obj)
    out = capture.import_scene(obj, "testroom", height=2.6, progress=lambda m: None)
    assert out["world"] == "testroom" and "testroom" in scan.captured() and worlds.is_scan("testroom")
    w, h = sorted(out["size_m"])
    assert 3.9 < w < 4.6 and 4.9 < h < 5.6  # back to metres, walls along x / y
    terrain = np.load(scans / "scans" / "testroom" / "terrain.npy")
    tops = np.round(terrain[terrain > 0], 1)
    assert np.any(np.abs(tops - 0.75) < 0.1)  # the table stands at table height
    assert np.mean(terrain == 0) > 0.5  # most of the room is free floor


@pytest.mark.slow
def test_robots_navigate_an_imported_room(scans):
    obj = scans / "room.obj"
    _room_obj(obj, scale=2.7, seed=3)
    capture.import_scene(obj, "testroom2", height=2.6, progress=lambda m: None)
    from anybot.sim import Sim

    sim = Sim("go2", realtime=False, world="testroom2")
    pos, _ = sim._base_pose()
    goal = [-pos[0], -pos[1]]  # across the room (the start is near the middle)
    out = sim.navigate_to(goal[0] if abs(goal[0]) > 0.8 else 1.2, goal[1])
    assert out["reached"]


def test_the_sharpest_frames_of_a_video_are_kept(tmp_path):
    import imageio_ffmpeg

    video = tmp_path / "walk.mp4"
    w, h, fps = 320, 240, 30
    rng = np.random.default_rng(0)
    pattern = (rng.random((h, w + 400, 3)) * 255).astype(np.uint8)
    writer = imageio_ffmpeg.write_frames(str(video), (w, h), fps=fps, quality=9)
    writer.send(None)
    for i in range(fps * 12):  # a 12 s pan across a textured wall, every other frame motion-blurred
        frame = pattern[:, i:i + w].astype(float)
        if i % 2:
            frame = (frame + np.roll(frame, 6, 1) + np.roll(frame, 12, 1)) / 3
        writer.send(np.ascontiguousarray(frame.astype(np.uint8)))
    writer.close()
    frames = capture.extract_frames(video, tmp_path / "frames", progress=lambda m: None)
    assert 22 <= len(frames) <= 26  # ~2 per second


def test_bad_inputs_are_explained(scans):
    with pytest.raises(capture.CaptureError, match="built-in"):
        capture.import_scene(scans, "warehouse")
    with pytest.raises(capture.CaptureError, match="at least 20"):
        capture.import_scene(scans, "emptydir")
