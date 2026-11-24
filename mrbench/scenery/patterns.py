"""Table tops drawn from a pattern rather than a photograph: optical breadboards, aluminium extrusion and a
workbench's worn steel.

Each is a square tile ``TILE_M`` on a side, made here as a height map and drawn as
the three maps a photographed finish has (diffuse, OpenGL normal, roughness),
then repeated ``1 / TILE_M`` times a metre, in world space, like the photographs:

* **breadboard** -- M6 holes on a 25 mm grid (8 x 8 to a tile), each a dark hole
  6 mm across in a bright 8 mm chamfer, in black anodised aluminium or brushed
  stainless steel;
* **extrusion** -- 40 mm aluminium profiles side by side along x (5 to a tile),
  each with an 8 mm T-slot down its middle, a hairline seam between them and the
  surface brushed along their length;
* **workbench** -- a steel top brushed along x and worn: faint smudges a few
  centimetres to a hand's breadth across and fine scratches, mostly along the
  brushing, on a larger tile (``BENCH_M``) so that the wear does not repeat across
  the table.  No stains: on Rea-04's table they would pass for its spills.

The holes and slots are drawn, not modelled: the table's collider stays flat.
The maps are made once, written as PNGs into ``third_party/looks/patterns/``
(``texture``); nothing is downloaded.  Making them needs numpy; writing them, Pillow.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

#: the tile's side (m) and its resolution (pixels)
TILE_M, PX = 0.2, 1024
#: breadboard: hole pitch, hole and chamfer diameters (m); the chamfer, bare metal, is what shows from afar
PITCH, HOLE_D, CHAMFER_D = 0.025, 0.006, 0.009
#: extrusion: profile width, slot opening, seam width (m)
PROFILE, SLOT, SEAM = 0.04, 0.008, 0.0006
#: workbench: its tile's side (m) and how many scratches a tile has
BENCH_M, SCRATCHES = 0.8, 70
#: pattern -> (surface sRGB 0-1, hole/slot sRGB, chamfer/edge sRGB, surface roughness, brushed along x).
#: Rough enough not to mirror the room: at 0.33-0.36 the steel and the extrusion render dark grey
PATTERNS = {
    "breadboard_black": ((0.055, 0.055, 0.06), (0.0, 0.0, 0.0), (0.78, 0.78, 0.80), 0.42, False),
    "breadboard_steel": ((0.70, 0.70, 0.71), (0.04, 0.04, 0.04), (0.84, 0.84, 0.85), 0.5, True),
    "extrusion": ((0.74, 0.75, 0.77), (0.03, 0.03, 0.035), (0.86, 0.86, 0.87), 0.45, True),
    # (no holes; the scratches).  Rougher than the others: at 0.5, smudged, it renders dark and blotched in Isaac
    "workbench_steel": ((0.74, 0.75, 0.76), (0.0, 0.0, 0.0), (0.80, 0.80, 0.81), 0.62, True),
}


def tile_m(name: str) -> float:
    """A pattern's tile side (m): the workbench's larger, the others ``TILE_M``."""
    return BENCH_M if name.startswith("workbench") else TILE_M


def _grid(px: int = PX):
    import numpy as np

    m = (np.arange(px) + 0.5) * TILE_M / px  # metres across the tile
    return np.meshgrid(m, m)  # x along the columns, y along the rows


def height(name: str, px: int = PX):
    """The tile's height map (0 the surface, -1 the bottom of a hole or slot) and its chamfer/edge mask."""
    import numpy as np

    x, y = _grid(px)
    if name.startswith("breadboard"):
        dx, dy = (x % PITCH) - PITCH / 2, (y % PITCH) - PITCH / 2
        r = np.hypot(dx, dy)
        hole = r < HOLE_D / 2
        chamfer = (r >= HOLE_D / 2) & (r < CHAMFER_D / 2)
        h = np.where(hole, -1.0, np.where(chamfer, -(CHAMFER_D / 2 - r) / ((CHAMFER_D - HOLE_D) / 2), 0.0))
        return h, chamfer
    if name == "extrusion":
        d = (y % PROFILE) - PROFILE / 2  # from the middle of a profile, across it
        slot = np.abs(d) < SLOT / 2
        seam = np.abs(np.abs(d) - PROFILE / 2) < SEAM / 2
        lip = (np.abs(d) >= SLOT / 2) & (np.abs(d) < SLOT / 2 + 0.0008)  # the slot's rounded lips catch the light
        return np.where(slot | seam, -1.0, 0.0), lip
    if name.startswith("workbench"):  # scratches: shallow grooves, bright metal, wrapping round the tile
        rng = np.random.default_rng(1)
        scratch = np.zeros((px, px), bool)
        for _ in range(SCRATCHES):
            x0, y0 = rng.uniform(0, px, 2)
            a = rng.normal(0.0, 0.3) if rng.random() < 0.8 else rng.uniform(0, np.pi)  # mostly along the brushing
            n = max(2, int(rng.uniform(0.01, 0.06) / BENCH_M * px * 2))  # 1-6 cm, two points a pixel
            t = np.linspace(0, n / 2, n)
            scratch[((y0 + t * np.sin(a)) % px).astype(int), ((x0 + t * np.cos(a)) % px).astype(int)] = True
        return np.where(scratch, -0.15, 0.0), scratch
    raise KeyError(name)


def _smooth(rng, px: int, sigma_px: float):
    """Noise blurred over ``sigma_px`` pixels, wrapping round the tile, scaled to at most 1 either way."""
    import numpy as np

    k = np.fft.fftfreq(px)
    kx, ky = np.meshgrid(k, k)
    blur = np.exp(-2 * (np.pi * sigma_px) ** 2 * (kx ** 2 + ky ** 2))
    v = np.real(np.fft.ifft2(np.fft.fft2(rng.normal(size=(px, px))) * blur))
    return v / np.abs(v).max()


def maps(name: str, px: int = PX):
    """The diffuse, normal (OpenGL) and roughness maps of a pattern, as uint8 arrays."""
    import numpy as np

    surface, dark, bright, rough, brushed = PATTERNS[name]
    h, edge = height(name, px)
    rng = np.random.default_rng(0)
    if brushed:  # fine streaks along x
        streak = rng.normal(0.0, 1.0, (px, 1)) * 0.03 + rng.normal(0.0, 1.0, (px, px)) * 0.008
    else:
        streak = rng.normal(0.0, 1.0, (px, px)) * 0.006
    if name.startswith("workbench"):  # smudges a few cm and a hand's breadth across
        streak = streak + _smooth(rng, px, px * 0.03 / BENCH_M) * 0.025 + _smooth(rng, px, px * 0.10 / BENCH_M) * 0.02
    base = np.empty((px, px, 3))
    for c in range(3):
        v = np.full((px, px), surface[c]) * (1.0 + streak)
        v = np.where(edge, bright[c], v)
        v = np.where(h <= -0.999, dark[c], v)
        base[..., c] = v
    diffuse = (np.clip(base, 0, 1) * 255).round().astype(np.uint8)
    # the normal from the height map's slope: a hole's wall, a chamfer, a slot's sides
    depth = 0.002  # m, what -1 stands for
    gy, gx = np.gradient(h * depth, tile_m(name) / px)
    n = np.stack([-gx, gy, np.ones_like(h)], axis=-1)  # OpenGL: +y (green) up the picture
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    normal = ((n * 0.5 + 0.5) * 255).round().astype(np.uint8)
    r = np.where(h <= -0.999, 0.9, np.where(edge, rough * 0.7, rough + streak * 0.5))
    roughness = (np.clip(r, 0, 1) * 255).round().astype(np.uint8)
    return diffuse, normal, roughness


def pattern_dir() -> Path:
    from .downloads import looks_dir

    return looks_dir() / "patterns"


def texture(name: str, part: str) -> Path:
    """One map of a pattern (``part``: "diff", "nor_gl" or "rough"), made the first time."""
    if name not in PATTERNS:
        raise KeyError(f"no pattern {name!r}; have {sorted(PATTERNS)}")
    # named by what the maps are made from: a changed pattern is made again, never read stale
    key = hashlib.sha1(repr((PATTERNS[name], TILE_M, PX, PITCH, HOLE_D, CHAMFER_D, PROFILE, SLOT, SEAM)
                            + ((BENCH_M, SCRATCHES) if name.startswith("workbench") else ())).encode())
    stem = f"{name}-{key.hexdigest()[:8]}"
    out = pattern_dir() / f"{stem}_{part}.png"
    if not out.exists():
        from PIL import Image

        out.parent.mkdir(parents=True, exist_ok=True)
        for p, a in zip(("diff", "nor_gl", "rough"), maps(name)):
            path = pattern_dir() / f"{stem}_{p}.png"
            tmp = path.with_name(f".{path.name}.{os.getpid()}.part")  # processes starting together
            Image.fromarray(a).save(tmp, format="PNG")
            tmp.replace(path)
    return out
