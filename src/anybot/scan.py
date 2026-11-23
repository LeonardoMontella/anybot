"""Scanned environments ("digital twins") as MuJoCo worlds.

Isaac Sim imports a scanned scene (USD, glTF) as one asset. MuJoCo has no scene importer, so a scan is
turned into the two things MuJoCo needs:

* what you see: the scan's textured triangle mesh, as a visual-only mesh (no collisions);
* what robots touch: a height field (MuJoCo's native terrain geom) rasterised from the same triangles,
  so walls, sofas, tables and steps are where they look to be. A height field cannot represent
  overhangs (the space under a table is solid), which is the safe side for robots that drive or walk.

The scan used by the "apartment" world is `apartment_1` from the Habitat test scenes
(huggingface.co/datasets/ai-habitat/habitat_test_scenes), a scan of a real apartment from the Replica
dataset (Meta, CC BY-NC 4.0: non-commercial use). It is downloaded once (50 MB) to ~/.anybot/assets/scans,
converted to a compact cache there (~20 MB) and the download is deleted. Any other glTF binary (.glb) scan of a room works the same way.
"""

from __future__ import annotations

import io
import json
import struct
import urllib.request
from pathlib import Path

import mujoco
import numpy as np

from anybot.navmap import _clearance, _reachable

SCANS = Path.home() / ".anybot" / "assets" / "scans"
APARTMENT_URL = "https://huggingface.co/datasets/ai-habitat/habitat_test_scenes/resolve/main/apartment_1.glb"
CELL = 0.04  # height field resolution (m)
TEXTURE_MAX = 2048  # px: the scan's 8k texture would take ~200 MB of RAM


def _quat_mat(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def read_glb(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, bytes | None]:
    """Vertices (world frame, z up), triangles, texture coordinates and the base-colour image of a .glb."""
    raw = path.read_bytes()
    if raw[:4] != b"glTF":
        raise ValueError(f"{path} is not a binary glTF (.glb) file")
    jlen = struct.unpack_from("<I", raw, 12)[0]
    gltf = json.loads(raw[20:20 + jlen])
    blob = raw[20 + jlen + 8:]

    def accessor(i: int) -> np.ndarray:
        a = gltf["accessors"][i]
        view = gltf["bufferViews"][a["bufferView"]]
        dtype = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}[a["componentType"]]
        width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a["type"]]
        x = np.frombuffer(blob, dtype, a["count"] * width, view.get("byteOffset", 0) + a.get("byteOffset", 0))
        return x.reshape(-1, width) if width > 1 else x

    verts, faces, uvs, image = [], [], [], None
    base = 0
    for node in gltf["nodes"]:
        if "mesh" not in node:
            continue
        R = _quat_mat(node.get("rotation", [0, 0, 0, 1]))
        S = np.asarray(node.get("scale", [1, 1, 1]), float)
        T = np.asarray(node.get("translation", [0, 0, 0]), float)
        for prim in gltf["meshes"][node["mesh"]]["primitives"]:
            if prim.get("mode", 4) != 4:
                continue
            p = accessor(prim["attributes"]["POSITION"]).astype(float)
            verts.append((p * S) @ R.T + T)
            faces.append(accessor(prim["indices"]).astype(np.int64).reshape(-1, 3) + base)
            if "TEXCOORD_0" in prim["attributes"]:
                uvs.append(accessor(prim["attributes"]["TEXCOORD_0"]).astype(float))
            base += len(p)
            mat = gltf["materials"][prim["material"]] if "material" in prim else {}
            tex = mat.get("pbrMetallicRoughness", {}).get("baseColorTexture")
            if tex is not None and image is None:
                src = gltf["images"][gltf["textures"][tex["index"]]["source"]]
                view = gltf["bufferViews"][src["bufferView"]]
                image = blob[view.get("byteOffset", 0):view.get("byteOffset", 0) + view["byteLength"]]
    v = np.concatenate(verts)
    uv = np.concatenate(uvs) if uvs and len(uvs) == len(verts) else None
    return v, np.concatenate(faces), uv, image


def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, bytes | None]:
    """Same as read_glb, for a Wavefront .obj (what photogrammetry tools write) and its first diffuse texture."""
    pos, tex, faces, image = [], [], [], None
    mtl_files = []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("v "):
            pos.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("vt "):
            tex.append([float(x) for x in line.split()[1:3]])
        elif line.startswith("f "):
            corners = []
            for c in line.split()[1:]:
                parts = c.split("/")
                vi = int(parts[0])
                ti = int(parts[1]) if len(parts) > 1 and parts[1] else 0
                corners.append((vi - 1 if vi > 0 else len(pos) + vi, ti - 1 if ti > 0 else (len(tex) + ti if ti else -1)))
            for k in range(1, len(corners) - 1):  # polygons → triangle fan
                faces.append((corners[0], corners[k], corners[k + 1]))
        elif line.startswith("mtllib "):
            mtl_files.append(line.split(None, 1)[1].strip())
    if not faces:
        raise ValueError(f"{path} has no faces")
    for mtl in mtl_files:  # the first map_Kd found is the texture
        mp = path.parent / mtl
        if image is None and mp.exists():
            for line in mp.read_text(errors="replace").splitlines():
                if line.strip().startswith("map_Kd"):
                    img = path.parent / line.split()[-1]
                    if img.exists():
                        image = img.read_bytes()
                        break
    pos = np.asarray(pos, float)
    tex = np.asarray(tex, float) if tex else np.zeros((0, 2))
    # One vertex per (position, uv) pair: a mesh vertex carries exactly one texture coordinate.
    pairs = np.array([c for tri in faces for c in tri], np.int64)
    uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
    v = pos[uniq[:, 0]]
    uv = tex[np.clip(uniq[:, 1], 0, None)] if len(tex) and (uniq[:, 1] >= 0).all() else None
    return v, inv.reshape(-1, 3), uv, image


def _level(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Rotate a scan with no known orientation (photogrammetry) so the floor is horizontal and the walls run
    along x and y. The room's three main surface directions come from the area-weighted face normals; "up"
    is the one with a big flat surface at an end of the room (the floor), on the side where the furniture is."""
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    n = np.cross(b - a, c - a)
    area = np.linalg.norm(n, axis=1)
    ok = area > 0
    n, area, cen = n[ok] / area[ok, None], area[ok], ((a + b + c) / 3)[ok]
    _, vecs = np.linalg.eigh((n * area[:, None]).T @ n)
    best, up = -1.0, vecs[:, 2]
    for k in range(3):
        e = vecs[:, k]
        h = cen @ e
        lo, hi = np.percentile(h, 1), np.percentile(h, 99)
        flat = np.abs(n @ e) > 0.9
        band = 0.06 * (hi - lo)
        score = max(area[flat & (h < lo + band)].sum(), area[flat & (h > hi - band)].sum())
        if score > best:
            best, up = score, e
    others = [vecs[:, k] for k in range(3) if not np.allclose(np.abs(vecs[:, k] @ up), 1.0)]
    x = others[0] - (others[0] @ up) * up
    x /= np.linalg.norm(x)
    R = np.stack([x, np.cross(up, x), up])  # rows: new x, y, z in the scan's frame
    out = v @ R.T
    z = out[:, 2]
    mid = 0.5 * (z.min() + z.max())
    if np.sum(z < mid) < np.sum(z > mid):  # floor at the bottom: the furnished half is denser
        out = out * np.array([1, -1, -1])
    return out


def _up_axis(v: np.ndarray) -> np.ndarray:
    """Rotate so that z is up: the vertical is the axis whose extent is room height (2-4 m)."""
    ext = np.ptp(v, axis=0)
    up = int(np.argmin(np.abs(ext - 2.8)))
    if up == 2:
        out = v
    elif up == 1:  # glTF convention: y up
        out = np.stack([v[:, 0], -v[:, 2], v[:, 1]], 1)
    else:
        out = np.stack([v[:, 1], v[:, 2], v[:, 0]], 1)
    # Floor at the bottom: furniture makes the lower half of the room denser than the upper half.
    z = out[:, 2]
    mid = 0.5 * (z.min() + z.max())
    if np.sum(z < mid) < np.sum(z > mid):
        out = out * np.array([1, -1, -1])
    return out


def _surface_samples(v: np.ndarray, f: np.ndarray, spacing: float) -> np.ndarray:
    """Points covering every triangle at about `spacing` (vertices alone leave holes in big triangles)."""
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    n = np.ceil(area / (0.5 * spacing * spacing)).astype(int)
    # Big flat triangles (a decimated mesh's floor, a wall) need many points; cap the total, not each one.
    total = int(n.sum())
    if total > 8_000_000:
        n = np.ceil(n * (8_000_000 / total)).astype(int)
    big = n > 1
    rep = np.repeat(np.nonzero(big)[0], n[big] - 1)
    rng = np.random.default_rng(0)
    r1, r2 = rng.random(len(rep)), rng.random(len(rep))
    flip = r1 + r2 > 1
    r1[flip], r2[flip] = 1 - r1[flip], 1 - r2[flip]
    pts = a[rep] + r1[:, None] * (b[rep] - a[rep]) + r2[:, None] * (c[rep] - a[rep])
    return np.concatenate([v, pts])


def _small_islands(mask: np.ndarray, min_cells: int) -> np.ndarray:
    """Cells of 8-connected components of `mask` with fewer than `min_cells` cells."""
    seen = np.zeros_like(mask)
    small = np.zeros_like(mask)
    H, W = mask.shape
    for r0, c0 in zip(*np.nonzero(mask)):
        if seen[r0, c0]:
            continue
        stack, comp = [(r0, c0)], []
        seen[r0, c0] = True
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            if len(comp) > min_cells:  # big enough: no need to walk all of it now (its cells stay unmarked)
                break
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
        if len(comp) <= min_cells and not stack:
            for r, c in comp:
                small[r, c] = True
        else:  # finish marking the big component as seen so its cells are not restarted
            while stack:
                r, c = stack.pop()
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < H and 0 <= cc < W and mask[rr, cc] and not seen[rr, cc]:
                            seen[rr, cc] = True
                            stack.append((rr, cc))
    return small


def convert(glb: Path, out: Path, height: float | None = None, scale: float | None = None) -> dict:
    """Convert a scan to the cached files the world loads: mesh.npz, texture.png, terrain.npy, scan.json.

    A .glb from a phone scanning app (LiDAR) is already in metres and z or y up. A mesh rebuilt from plain
    photos (.obj) has no known size or orientation: it is levelled and scaled so that the room is `height`
    metres from floor to ceiling (or multiplied by `scale`)."""
    from PIL import Image

    glb = Path(glb)
    v, f, uv, image = read_obj(glb) if glb.suffix.lower() == ".obj" else read_glb(glb)
    if height is not None or scale is not None or glb.suffix.lower() == ".obj":
        v = _level(v, f)
        if scale is not None:
            v = v * float(scale)
        else:
            z = v[:, 2]
            v = v * (float(height or 2.6) / max(float(np.percentile(z, 99.5) - np.percentile(z, 0.5)), 1e-6))
    else:
        v = _up_axis(v)
    # Floor height: the densest horizontal layer near the bottom of the scan.
    z = v[:, 2]
    hist, edges = np.histogram(z[z < z.min() + 0.5], 50)
    v[:, 2] -= edges[np.argmax(hist)] + 0.5 * (edges[1] - edges[0]) - 0.004  # just above the ground plane
    ceiling = float(np.percentile(v[:, 2], 99.5))

    # Drop the ceiling (and anything hanging from it) so the rooms can be seen from above.
    keep = v[f][:, :, 2].min(axis=1) < ceiling - 0.35
    f = f[keep]

    # Height field from the triangles below the ceiling.
    pts = _surface_samples(v, f, CELL * 0.7)
    # Only what a robot can bump into: below ~1.85 m. Above that are door lintels: counting them would
    # close every doorway (the height field takes the highest point of each column).
    pts = pts[pts[:, 2] < min(ceiling - 0.35, 1.85)]
    lo, hi = pts[:, :2].min(0) - 0.1, pts[:, :2].max(0) + 0.1
    ncol, nrow = (np.ceil((hi - lo) / CELL).astype(int) + 1)
    ix = ((pts[:, 0] - lo[0]) / CELL).astype(int)
    iy = ((pts[:, 1] - lo[1]) / CELL).astype(int)
    h = np.zeros((nrow, ncol))
    np.maximum.at(h, (iy, ix), np.clip(pts[:, 2], 0, None))
    seen = np.zeros((nrow, ncol), bool)
    seen[iy, ix] = True
    h[h < 0.06] = 0.0  # scan noise on the floor would make every step a bump
    # Remove specks (single noisy cells) among LOW obstacles only: keep them where a 3x3 block is all
    # obstacle (opening). Tall ones (walls, cupboards) stay, however thin the scan made them.
    tall = h >= 0.3
    obst = (h > 0.0) & ~tall
    core = obst.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            core &= np.roll(np.roll(obst, dr, 0), dc, 1)
    kept = core.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            kept |= np.roll(np.roll(core, dr, 0), dc, 1)
    h[obst & ~kept] = 0.0
    # Floating scan fragments make thin "poles" that no robot can see in the mesh: drop every obstacle island
    # smaller than ~1 dm² (walls are long, and table/chair legs merge with their top into one block).
    h[_small_islands(h > 0.0, 6)] = 0.0
    # Not scanned: outside the rooms it is a wall, but a small gap in the middle of a floor is floor
    # (a cell no sampled point happened to land in), not a 2 m post.
    holes = ~seen
    tall_near = np.zeros_like(holes)  # a gap touching a wall is the inside of that wall: solid
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            tall_near |= np.roll(np.roll(h >= 0.3, dr, 0), dc, 1)
    floor_gap = _small_islands(holes, 20) & ~tall_near
    h[holes & ~floor_gap] = 2.0
    # Free space and a start point: the most open floor spot near the middle of the scan.
    free = h == 0.0
    clear = _clearance(free, 25)
    yy, xx = np.mgrid[0:nrow, 0:ncol]
    centre = np.array([nrow, ncol]) / 2
    score = np.minimum(clear, 18) - 0.02 * np.hypot(yy - centre[0], xx - centre[1])
    sy, sx = np.unravel_index(np.argmax(score), score.shape)
    origin = np.array([lo[0] + sx * CELL, lo[1] + sy * CELL])

    # A few named spots to send mobile robots to: open floor, spread over the rooms, all connected to the
    # start by paths wide enough for a humanoid.
    start = (int(sy), int(sx))
    reach = _reachable(clear >= 9, start)
    cand = np.argwhere(reach & (clear >= 17))
    spots: list[np.ndarray] = []
    pts_far = [np.array(start)]
    for _ in range(4):
        if not len(cand):
            break
        dmin = np.min([np.linalg.norm(cand - p, axis=1) for p in pts_far], axis=0)
        best = int(np.argmax(dmin))
        if dmin[best] * CELL < 1.2:
            break
        spots.append(cand[best])
        pts_far.append(cand[best])

    out.mkdir(parents=True, exist_ok=True)
    shift = np.array([origin[0], origin[1], 0.0])
    used = np.unique(f)
    remap = np.full(len(v), -1)
    remap[used] = np.arange(len(used))
    np.savez_compressed(out / "mesh.npz", verts=(v[used] - shift).astype(np.float32), faces=remap[f].astype(np.int32),
                        uv=(uv[used].astype(np.float32) if uv is not None else np.zeros((0, 2), np.float32)))
    if image is not None:
        img = Image.open(io.BytesIO(image)).convert("RGB")
        img.thumbnail((TEXTURE_MAX, TEXTURE_MAX))
        img.save(out / "texture.png", optimize=True)
    np.save(out / "terrain.npy", h.astype(np.float32))
    meta = {
        "source": glb.name, "cell": CELL, "lo": (lo - origin).tolist(), "size": [float(ncol * CELL), float(nrow * CELL)],
        "ceiling": ceiling, "triangles": int(len(f)),
        "spots": {f"spot_{i + 1}": [float(lo[0] + c[1] * CELL - origin[0]), float(lo[1] + c[0] * CELL - origin[1])]
                  for i, c in enumerate(spots)},
    }
    (out / "scan.json").write_text(json.dumps(meta))
    return meta


def captured() -> list[str]:
    """Names of the worlds imported with `anybot import-scene` (converted scans other than the apartment)."""
    if not SCANS.exists():
        return []
    return sorted(p.name for p in SCANS.iterdir() if (p / "scan.json").exists() and p.name != "apartment_1")


def folder_for(world: str) -> Path:
    """The converted scan behind a scan world ('apartment' or an imported one)."""
    return ensure_apartment() if world == "apartment" else SCANS / world


def ensure_apartment() -> Path:
    """Download (once, 50 MB) and convert (once) the apartment scan; returns its cache folder."""
    out = SCANS / "apartment_1"
    if (out / "scan.json").exists():
        return out
    glb = SCANS / "apartment_1.glb"
    if not glb.exists():
        SCANS.mkdir(parents=True, exist_ok=True)
        tmp = glb.with_suffix(".part")
        urllib.request.urlretrieve(APARTMENT_URL, tmp)
        tmp.rename(glb)
    convert(glb, out)
    glb.unlink()  # the converted cache (~20 MB) is all the world needs: keep the disk light
    return out


def build_scan(spec: mujoco.MjSpec, folder: Path) -> dict:
    """Add a converted scan to a scene: the textured mesh to look at, the height field to collide with."""
    meta = json.loads((folder / "scan.json").read_text())
    mesh = np.load(folder / "mesh.npz")
    h = np.load(folder / "terrain.npy").astype(float)
    # The floor itself is the world's plane. Bare-floor cells of the height field sit a little below it: a
    # height field level with the plane makes mesh feet (Spot's) collide with both and get kicked off the
    # ground. Obstacles keep their real height.
    sink = 0.04
    h = np.where(h > 0, h + sink, 0.0)
    nrow, ncol = h.shape
    top = float(max(h.max(), 0.05))
    # ncol samples span (ncol - 1) cells: sample (r, c) sits exactly at lo + (c, r) * cell.
    sx, sy = (ncol - 1) * meta["cell"], (nrow - 1) * meta["cell"]
    spec.add_hfield(name="scan_terrain", size=[sx / 2, sy / 2, top, 0.05], nrow=nrow, ncol=ncol,
                    userdata=(h / top).ravel().tolist())
    lo = np.array(meta["lo"])
    spec.worldbody.add_geom(name="scan_collision", type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="scan_terrain",
                            pos=[lo[0] + sx / 2, lo[1] + sy / 2, -sink],
                            rgba=[0, 0, 0, 0], contype=0xFFFF, conaffinity=0xFFFF, group=3)  # like the floor
    kw = {}
    if (folder / "texture.png").exists() and len(mesh["uv"]):
        spec.add_texture(name="scan_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(folder / "texture.png"))
        # The scan's texture already holds the real lighting (glTF "unlit"): show it as is.
        mat = spec.add_material(name="scan_mat", rgba=[1, 1, 1, 1], specular=0.0, shininess=0.0, emission=1.0)
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "scan_tex"
        kw = dict(usertexcoord=mesh["uv"].ravel().tolist())
    m = spec.add_mesh(name="scan_mesh", uservert=mesh["verts"].ravel().tolist(), userface=mesh["faces"].ravel().tolist(), **kw)
    m.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
    spec.worldbody.add_geom(name="scan_visual", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="scan_mesh",
                            material="scan_mat" if kw else "", rgba=[1, 1, 1, 1], contype=0, conaffinity=0, group=1)
    return meta
