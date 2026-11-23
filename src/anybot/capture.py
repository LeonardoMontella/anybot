"""Video → world: turn a phone video (or photos, or a phone scan) of a real place into an anybot world.

    anybot import-scene kitchen.mp4 --name kitchen          # a plain video: rebuilt in 3D on this Mac
    anybot import-scene ./photos --name lab2                # a folder of photos: same, without frame picking
    anybot import-scene room.glb --name office              # a LiDAR scan app's export (Polycam, Scaniverse)

Pipeline:
1. frames: about 2 per second of video, the sharpest of each window (motion blur ruins reconstruction),
   at most ~200, long side 1600 px (ffmpeg, bundled by imageio-ffmpeg);
2. 3D reconstruction: Apple Object Capture (RealityKit PhotogrammetrySession, on the Mac's GPU) through
   a small Swift helper compiled on first use → textured mesh (OBJ);
3. world: the same converter as the apartment scan (anybot.scan.convert): levelled so the floor is flat,
   scaled to the room height (photos alone carry no size), height field for collisions, free spots.

The world then shows up next to the built-in ones: load_robot(..., world="kitchen").
Temporary frames and meshes are deleted at the end: only the compact world cache (~10-20 MB) stays.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from anybot import scan

VIDEO = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
PHOTOS = {".jpg", ".jpeg", ".png", ".heic"}
NATIVE = Path(__file__).parent / "native" / "reconstruct.swift"
BIN = Path.home() / ".anybot" / "bin" / "anybot-reconstruct"
FILMING_TIPS = ("Film slowly, walking around the room with the camera at chest height, panning so each part "
                "of the room appears in several frames; avoid fast turns, mirrors and pointing at bare walls.")


class CaptureError(RuntimeError):
    pass


def _say(progress: Callable[[str], None] | None, msg: str) -> None:
    (progress or print)(msg)


def _sharpness(img: Image.Image) -> float:
    """Variance of the Laplacian of a small grey copy: low = blurred."""
    g = np.asarray(img.convert("L").resize((320, max(1, int(320 * img.height / img.width)))), float)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def extract_frames(video: Path, out: Path, per_second: float = 2.0, max_frames: int = 200, size: int = 1600,
                   progress: Callable[[str], None] | None = None) -> list[Path]:
    """Save the sharpest frame of every 1/per_second window of the video as JPEG; returns the files."""
    import imageio_ffmpeg

    out.mkdir(parents=True, exist_ok=True)
    reader = imageio_ffmpeg.read_frames(str(video))
    meta = next(reader)
    w, h = meta["size"]
    fps = float(meta.get("fps") or 30.0)
    total = int(float(meta.get("duration") or 0) * fps)
    window = max(1, int(round(fps / per_second)))
    if total and total / window > max_frames:  # a long video: wider windows, same number of frames
        window = int(np.ceil(total / max_frames))
    saved, best, best_s = [], None, -1.0
    for i, raw in enumerate(reader):
        img = Image.frombuffer("RGB", (w, h), raw, "raw", "RGB", 0, 1)
        s = _sharpness(img)
        if s > best_s:
            best, best_s = img.copy(), s
        if (i + 1) % window == 0:
            best.thumbnail((size, size))
            path = out / f"frame_{len(saved):04d}.jpg"
            best.save(path, quality=92)
            saved.append(path)
            best, best_s = None, -1.0
            if len(saved) >= max_frames:
                break
    if best is not None and len(saved) < max_frames:
        best.thumbnail((size, size))
        path = out / f"frame_{len(saved):04d}.jpg"
        best.save(path, quality=92)
        saved.append(path)
    reader.close()
    _say(progress, f"{len(saved)} frames picked from {video.name}")
    if len(saved) < 20:
        raise CaptureError(f"Only {len(saved)} usable frames: the video is too short. {FILMING_TIPS}")
    return saved


def _swift_ok() -> bool:
    try:
        out = subprocess.run(["swiftc", "--version"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return False
    m = re.search(r"Swift version (\d+)\.(\d+)", out)
    return bool(m) and (int(m.group(1)), int(m.group(2))) >= (5, 5)


def _helper(progress=None) -> Path:
    """The Object Capture helper, compiled once from native/reconstruct.swift."""
    if BIN.exists() and BIN.stat().st_mtime >= NATIVE.stat().st_mtime:
        return BIN
    if not _swift_ok():
        raise CaptureError(
            "3D reconstruction needs Apple's Swift 5.5+ (Object Capture, macOS 12+). Update the Command Line Tools "
            "(`softwareupdate --list`, then install 'Command Line Tools for Xcode'), or import a .glb exported "
            "by a phone scanning app (Polycam, Scaniverse) instead of a video.")
    BIN.parent.mkdir(parents=True, exist_ok=True)
    _say(progress, "compiling the Object Capture helper (once)...")
    r = subprocess.run(["swiftc", "-O", "-target", "arm64-apple-macos12", str(NATIVE), "-o", str(BIN)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise CaptureError(f"Could not build the reconstruction helper:\n{r.stderr[-1500:]}")
    return BIN


def reconstruct(images: Path, out: Path, detail: str = "reduced", progress=None) -> Path:
    """Photos → textured OBJ with Apple Object Capture; returns the .obj file."""
    exe = _helper(progress)
    out.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([str(exe), str(images), str(out), detail], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    last, error = -1, None
    for line in proc.stdout:
        line = line.strip()
        if line.startswith("progress "):
            pct = int(100 * float(line.split()[1]))
            if pct >= last + 5:
                _say(progress, f"reconstructing: {pct}%")
                last = pct
        elif line.startswith("error"):
            error = line[6:]
    proc.wait()
    objs = sorted(out.rglob("*.obj"))
    if proc.returncode != 0 or not objs:
        raise CaptureError(f"3D reconstruction failed: {error or 'no model was produced'}. {FILMING_TIPS}")
    return objs[0]


def _check_name(name: str) -> str:
    from anybot import worlds

    name = name.strip().lower()
    if not re.fullmatch(r"[a-z][a-z0-9_]{1,30}", name):
        raise CaptureError("The world name must be 2-31 characters: lowercase letters, digits, underscores.")
    if name in worlds.WORLDS or name == "apartment_1":
        raise CaptureError(f"'{name}' is a built-in world: choose another name.")
    return name


def import_scene(source: str | Path, name: str, height: float = 2.6, scale: float | None = None,
                 detail: str = "reduced", progress: Callable[[str], None] | None = None) -> dict:
    """Make a world from a video, a folder of photos, a .glb scan or an .obj mesh of a real place."""
    src = Path(source).expanduser()
    name = _check_name(name)
    if not src.exists():
        raise CaptureError(f"{src} does not exist")
    out = scan.SCANS / name
    tmp = Path(tempfile.mkdtemp(prefix=f"anybot-{name}-"))
    try:
        ext = src.suffix.lower()
        if src.is_file() and ext == ".glb":  # a phone scan: already a metric mesh
            _say(progress, f"converting {src.name}...")
            meta = scan.convert(src, tmp / "world")
        elif src.is_file() and ext == ".obj":
            meta = scan.convert(src, tmp / "world", height=height, scale=scale)
        else:
            if src.is_dir():
                images = sorted(p for p in src.iterdir() if p.suffix.lower() in PHOTOS)
                if len(images) < 20:
                    raise CaptureError(f"{src} has {len(images)} photos: at least 20 are needed. {FILMING_TIPS}")
                frames = src
            elif ext in VIDEO:
                frames = tmp / "frames"
                extract_frames(src, frames, progress=progress)
            else:
                raise CaptureError(f"Unsupported input {src.name}: use a video, a folder of photos, a .glb or an .obj.")
            obj = reconstruct(frames, tmp / "model", detail=detail, progress=progress)
            _say(progress, "building the world (collision map, free spots)...")
            meta = scan.convert(obj, tmp / "world", height=height, scale=scale)
        if out.exists():
            shutil.rmtree(out)
        shutil.move(str(tmp / "world"), out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    size = meta["size"]
    _say(progress, f"world '{name}' ready: {size[0]:.1f} x {size[1]:.1f} m, spots: {', '.join(meta['spots']) or 'none'}")
    return {"world": name, "size_m": [round(size[0], 2), round(size[1], 2)], "spots": list(meta["spots"]),
            "folder": str(out)}


def remove_scene(name: str) -> None:
    folder = scan.SCANS / _check_name(name)
    if folder.exists():
        shutil.rmtree(folder)
