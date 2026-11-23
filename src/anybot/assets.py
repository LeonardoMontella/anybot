"""Real-world assets fetched on demand and cached in ~/.anybot/assets (a few MB, never the whole dataset).

Scanned objects come from Google Scanned Objects (3D scans of real household items, CC-BY 4.0),
through the MuJoCo port at github.com/kevinzakka/mujoco_scanned_objects (MIT).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

CACHE = Path.home() / ".anybot" / "assets"
_SCANNED = "https://raw.githubusercontent.com/kevinzakka/mujoco_scanned_objects/main/models"

# Short name → dataset folder. Small, light things a robot can carry.
SCANNED_OBJECTS = {
    "mug": "ACE_Coffee_Mug_Kristen_16_oz_cup",
    "tea_box": "The_Coffee_Bean_Tea_Leaf_KCup_Packs_Jasmine_Green_Tea_16_count",
    "lunch_box": "Olive_Kids_Dinosaur_Land_Lunch_Box",
    "blue_mug": "Cole_Hardware_Mug_Classic_Blue",
}


def scanned_object(name: str) -> Path:
    """Folder with model.obj and texture.png for a scanned object, downloading it the first time."""
    folder = CACHE / "scanned" / name
    for file in ("model.obj", "texture.png"):
        dest = folder / file
        if not dest.exists():
            folder.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            urllib.request.urlretrieve(f"{_SCANNED}/{SCANNED_OBJECTS[name]}/{file}", tmp)
            tmp.rename(dest)
    return folder


def obj_bounds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounds of an OBJ mesh (min, max)."""
    verts = np.array([[float(x) for x in line.split()[1:4]] for line in path.read_text().splitlines() if line.startswith("v ")])
    return verts.min(axis=0), verts.max(axis=0)
