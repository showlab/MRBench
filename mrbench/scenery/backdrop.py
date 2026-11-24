"""A furnished corner of the room behind the table (``Look.backdrop``), so that the table does not stand in an empty
room: walls close by and, along them,

* **kitchen** -- a run of kitchen cabinets with a counter, shelves with things on them, a fridge, a print;
* **workshop** -- a workbench under a pegboard of hand tools, a rolling tool cabinet, steel shelving with bins;
* **home** -- a sideboard under three pictures, a window with curtains, a bookcase, an armchair and a floor lamp.

Each is built from boxes, cylinders and balls, and from the YCB scans the props use (``props.PROPS``).  Everything is drawn only -- no collider, no body -- and stands
``REACH_GAP`` beyond every arm's reach (``tests/test_scenery``): it cannot touch the task, and nothing the team is told
about names it.  The corner is laid out from the table's size: the cabinets ``AISLE_N`` north of the table, the west
wall ``AISLE_W`` west of it, which leaves every camera of ``render`` and ``agent.snapshot`` in the open.
Engine-independent: the Isaac engine builds it (``engines/isaac/environment.py``).
"""

from __future__ import annotations

import functools
import random
from dataclasses import dataclass
from typing import Optional

from ..engines.base import Vec3
from .patterns import TILE_M

#: the backdrops a look may name
BACKDROPS = ("none", "kitchen", "workshop", "home")

#: from the table's north edge to the cabinets' fronts, from its west edge to the west wall (m)
AISLE_N, AISLE_W = 1.1, 1.25
#: the walls; the base cabinets: depth, the plinth under them, their height, the counter's thickness (m)
WALL_T, WALL_H = 0.1, 3.2
DEPTH, PLINTH, BASE_H, TOP_T = 0.6, 0.1, 0.78, 0.04
COUNTER_Z = PLINTH + BASE_H + TOP_T
#: the ceiling light over the counter (Isaac's units; the dome is 1000 times the look's light), in a cone that keeps
#: it off the table: the task is lit as without the backdrop
LAMP_INTENSITY, LAMP_CONE = 6000.0, 25.0
#: the room's inside (``environment.ROOM_INSIDE``): the north wall runs to its east wall, the west wall to its south
ROOM_EAST, ROOM_SOUTH = 4.4, -3.3
#: everything stands at least this far beyond every arm's reach (``RobotModel.reach_m``; a Franka's links sweep
#: about 15 cm past it low over the table), and this far from every camera (m)
REACH_GAP, CAMERA_GAP = 0.5, 0.1


@dataclass(frozen=True)
class Piece:
    """A drawn shape: "box" (x, y, z), "cylinder" (radius, height; along z) or "sphere" (radius), its centre at
    ``pos``, turned ``yaw_deg`` about z, in the finish ``finish`` names (``finishes``)."""

    name: str
    shape: str
    size: tuple[float, ...]
    pos: Vec3
    finish: str
    yaw_deg: float = 0.0

    def half(self) -> Vec3:
        """Half its extent along x, y, z (unturned)."""
        if self.shape == "box":
            return self.size[0] / 2, self.size[1] / 2, self.size[2] / 2
        if self.shape == "cylinder":
            return self.size[0], self.size[0], self.size[1] / 2
        return (self.size[0],) * 3


@dataclass(frozen=True)
class Lamp:
    """A rectangle of light ``size`` (x, y) at ``pos``, facing down, ``intensity`` in Isaac's units, its light in a
    cone ``cone_deg`` from straight down."""

    name: str
    size: tuple[float, float]
    pos: Vec3
    intensity: float
    cone_deg: float = 90.0


@dataclass(frozen=True)
class Ornament:
    """A YCB scan (``props.PROPS[kind]``) standing upright, drawn only: the centre of its upright bounding box at
    ``pos``."""

    name: str
    kind: str
    pos: Vec3
    yaw_deg: float = 0.0


@functools.lru_cache(maxsize=None)
def finishes() -> dict:
    """The finishes the pieces name (``scenery.Finish``)."""
    from . import Finish

    return {
        "plaster": Finish(colour=(0.86, 0.84, 0.79), roughness=0.92),
        "skirting": Finish(colour=(0.92, 0.91, 0.88), roughness=0.5),
        # the base cabinets sage, the wall cabinets off-white: two whites (the table's top and these) side by side
        # washed out
        "sage": Finish(colour=(0.47, 0.55, 0.48), roughness=0.45),
        "cream": Finish(colour=(0.90, 0.89, 0.85), roughness=0.4),
        "oak": Finish("oak_veneer_01", 1.5),
        "tiles": Finish("large_grey_tiles", 3.0, tint=(1.15, 1.15, 1.12)),
        # brushed: polished, it mirrors the dim room and renders black (as the table's steel does)
        "steel": Finish(colour=(0.74, 0.75, 0.76), roughness=0.6, metallic=True),
        "black": Finish(colour=(0.05, 0.05, 0.055), roughness=0.5),
        "paper": Finish(colour=(0.88, 0.86, 0.80), roughness=0.9),
        "navy": Finish(colour=(0.10, 0.16, 0.24), roughness=0.7),
        "terracotta": Finish(colour=(0.71, 0.36, 0.24), roughness=0.8),
        "leaf": Finish(colour=(0.24, 0.45, 0.20), roughness=0.7),
        "leaf_dark": Finish(colour=(0.17, 0.35, 0.15), roughness=0.7),
        # the workshop: painted block, grey over slate; powder-coated steel; a hardboard pegboard (the breadboard's
        # 25 mm holes, browned)
        "paint": Finish(colour=(0.76, 0.77, 0.75), roughness=0.9),
        "paint_dark": Finish(colour=(0.33, 0.38, 0.41), roughness=0.8),
        "gunmetal": Finish(colour=(0.22, 0.23, 0.25), roughness=0.5),
        "plywood": Finish("plywood", 2.0),
        "pegboard": Finish("pattern:breadboard_steel", 1 / TILE_M, tint=(0.78, 0.55, 0.36)),
        "toolred": Finish(colour=(0.70, 0.07, 0.05), roughness=0.35),
        "yellow": Finish(colour=(0.93, 0.72, 0.08), roughness=0.55),
        "blue": Finish(colour=(0.11, 0.30, 0.62), roughness=0.55),
        "aid_green": Finish(colour=(0.0, 0.52, 0.27), roughness=0.6),
        # the home: warm clay plaster (a cream cloth on the table stands out against it), walnut, brass, linen,
        # a pale sky through the window, and the books' and the fabrics' colours
        "clay": Finish(colour=(0.79, 0.62, 0.50), roughness=0.92),
        "walnut": Finish("oak_veneer_01", 3.0, tint=(0.42, 0.30, 0.24)),
        "brass": Finish(colour=(0.78, 0.62, 0.32), roughness=0.4, metallic=True),
        "linen": Finish(colour=(0.94, 0.90, 0.80), roughness=0.9),
        "sky": Finish(colour=(0.70, 0.85, 0.97), roughness=0.3),
        "teal": Finish(colour=(0.12, 0.38, 0.40), roughness=0.85),
        "mustard": Finish(colour=(0.80, 0.58, 0.16), roughness=0.8),
        "olive": Finish(colour=(0.45, 0.48, 0.30), roughness=0.9),
        "wine": Finish(colour=(0.48, 0.10, 0.13), roughness=0.7),
        "slate": Finish(colour=(0.32, 0.34, 0.37), roughness=0.7),
    }


def _box(name, x0, x1, y0, y1, z0, z1, finish) -> Piece:
    """A box from its extents."""
    return Piece(name, "box", (x1 - x0, y1 - y0, z1 - z0), ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), finish)


def _on(name: str, kind: str, x: float, y: float, z: float, yaw: float = 0.0) -> Ornament:
    """A scan standing on a surface at height ``z``."""
    from .props import PROPS

    return Ornament(name, kind, (x, y, z + PROPS[kind].size[2] / 2), yaw)


def _flat(name, wall, face, u0, u1, z0, z1, d0, d1, finish) -> Piece:
    """A box against a wall -- "n", its face at y = ``face``, or "w", at x = ``face`` -- from ``u0`` to ``u1`` along
    it, ``d0`` to ``d1`` out from it."""
    if wall == "n":
        return _box(name, u0, u1, face - d1, face - d0, z0, z1, finish)
    return _box(name, face + d0, face + d1, u0, u1, z0, z1, finish)


def _picture(name, wall, face, u, z, w, h, frame, art) -> list[Piece]:
    """A framed picture ``w`` x ``h`` on a wall (``_flat``), its middle at ``u`` along it and ``z`` up: a frame, a
    paper mount and ``art`` -- (u0, u1, z0, z1, finish) from the middle -- each a hair proud of the last."""
    out = [_flat(f"{name}_frame", wall, face, u - w / 2, u + w / 2, z - h / 2, z + h / 2, 0.0, 0.025, frame),
           _flat(name, wall, face, u - w / 2 + 0.035, u + w / 2 - 0.035, z - h / 2 + 0.035, z + h / 2 - 0.035,
                 0.025, 0.029, "paper")]
    for i, (u0, u1, z0, z1, finish) in enumerate(art):
        out.append(_flat(f"{name}_art{i}", wall, face, u + u0, u + u1, z + z0, z + z1, 0.029, 0.030 + 0.001 * i,
                         finish))
    return out


def kitchen(table_size) -> tuple[tuple[Piece, ...], tuple[Ornament, ...]]:
    """The kitchen corner north and west of a table of ``table_size`` (x, y, top z)."""
    tx, ty = table_size[0] / 2, table_size[1] / 2
    yw = ty + AISLE_N + DEPTH  # the north wall's inside face
    xw = -(tx + AISLE_W)  # the west wall's
    xe = tx + 1.2  # the east end of the cabinets
    yf = yw - DEPTH  # the base cabinets' fronts
    p: list[Piece] = []
    o: list[Ornament] = []
    # the walls and their skirting
    p.append(_box("wall_n", xw - WALL_T, ROOM_EAST, yw, yw + WALL_T, 0.0, WALL_H, "plaster"))
    p.append(_box("wall_w", xw - WALL_T, xw, ROOM_SOUTH, yw, 0.0, WALL_H, "plaster"))
    p.append(_box("skirting_w", xw, xw + 0.015, ROOM_SOUTH, yw, 0.0, 0.09, "skirting"))
    p.append(_box("skirting_n", xe, ROOM_EAST, yw - 0.015, yw, 0.0, 0.09, "skirting"))

    # the fridge, in the corner
    fx0, fx1, fy0 = xw + 0.03, xw + 0.80, yw - 0.70
    p.append(_box("fridge", fx0, fx1, fy0, yw - 0.02, 0.0, 1.95, "cream"))
    p.append(_box("fridge_seam", fx0, fx1, fy0 - 0.004, fy0, 1.24, 1.255, "black"))
    for i, (z0, z1) in enumerate(((0.55, 1.10), (1.40, 1.85))):
        p.append(_box(f"fridge_handle{i}", fx1 - 0.09, fx1 - 0.07, fy0 - 0.05, fy0 - 0.01, z0, z1, "steel"))

    # the base cabinets: a drawer over two doors to a unit, under one counter
    x0 = fx1 + 0.02
    n = max(1, round((xe - x0) / 0.8))
    w = (xe - x0) / n
    zc = PLINTH + BASE_H
    p.append(_box("plinth", x0, xe, yf + 0.06, yw, 0.0, PLINTH, "black"))
    for i in range(n):
        a, b = x0 + i * w, x0 + (i + 1) * w
        p.append(_box(f"cabinet{i}", a + 0.002, b - 0.002, yf + 0.02, yw - 0.01, PLINTH, zc, "sage"))
        p.append(_box(f"drawer{i}", a + 0.004, b - 0.004, yf, yf + 0.02, zc - 0.17, zc - 0.006, "sage"))
        p.append(_box(f"drawer{i}_handle", (a + b) / 2 - 0.08, (a + b) / 2 + 0.08, yf - 0.025, yf,
                      zc - 0.095, zc - 0.083, "steel"))
        for j, (c, d) in enumerate(((a + 0.004, (a + b) / 2 - 0.002), ((a + b) / 2 + 0.002, b - 0.004))):
            p.append(_box(f"door{i}_{j}", c, d, yf, yf + 0.02, PLINTH + 0.006, zc - 0.176, "sage"))
            hx = d - 0.04 if j == 0 else c + 0.03
            p.append(_box(f"door{i}_{j}_handle", hx, hx + 0.012, yf - 0.025, yf, zc - 0.36, zc - 0.24, "steel"))
    p.append(_box("counter", x0 - 0.01, xe + 0.01, yf - 0.03, yw, zc, COUNTER_Z, "oak"))
    p.append(_box("splashback", x0, xe, yw - 0.012, yw, COUNTER_Z, 1.45, "tiles"))

    # wall cabinets over the east half, open shelves over the west
    xu = 0.3
    zu0, zu1, du = 1.55, 2.25, 0.35
    m = max(1, round((xe - xu) / 0.6))
    wu = (xe - xu) / m
    for i in range(m):
        a, b = xu + i * wu, xu + (i + 1) * wu
        p.append(_box(f"wall_cabinet{i}", a + 0.002, b - 0.002, yw - du, yw, zu0, zu1, "cream"))
        p.append(_box(f"wall_cabinet{i}_handle", (a + b) / 2 - 0.006, (a + b) / 2 + 0.006, yw - du - 0.025,
                      yw - du, zu0 + 0.04, zu0 + 0.16, "steel"))
    s0, s1, ds = x0 + 0.15, xu - 0.3, 0.26
    for k, z in enumerate((1.55, 1.95)):
        p.append(_box(f"shelf{k}", s0, s1, yw - ds, yw, z - 0.035, z, "oak"))
        for j, x in enumerate((s0 + 0.15, s1 - 0.15)):
            p.append(_box(f"shelf{k}_bracket{j}", x - 0.012, x + 0.012, yw - ds + 0.04, yw, z - 0.165, z - 0.035,
                          "black"))
    yo = yw - ds / 2
    span = s1 - s0
    for name, kind, f, z, yaw in (
            ("mug", "ycb:025_mug", 0.08, 1.55, 200.0), ("bowl", "ycb:024_bowl", 0.22, 1.55, 0.0),
            ("coffee", "ycb:002_master_chef_can", 0.42, 1.55, 30.0), ("sugar", "ycb:004_sugar_box", 0.58, 1.55, 0.0),
            ("mustard", "ycb:006_mustard_bottle", 0.78, 1.55, 160.0),
            ("crackers", "ycb:003_cracker_box", 0.15, 1.95, 90.0), ("pitcher", "ycb:019_pitcher_base", 0.45, 1.95, 220.0),
            ("soup", "ycb:005_tomato_soup_can", 0.66, 1.95, 0.0), ("soup_2", "ycb:005_tomato_soup_can", 0.73, 1.95, 40.0),
            ("meat", "ycb:010_potted_meat_can", 0.88, 1.95, 0.0)):
        o.append(_on(f"shelf_{name}", kind, s0 + f * span, yo, z, yaw))

    # on the counter: a board with a bowl of bananas, a kettle, a plant, the cleanser by the sink end
    yc = yf + DEPTH / 2
    xb = x0 + 0.55
    p.append(_box("board", xb - 0.2, xb + 0.2, yc - 0.12, yc + 0.16, COUNTER_Z, COUNTER_Z + 0.02, "oak"))
    o.append(_on("counter_bowl", "ycb:024_bowl", xb + 0.06, yc + 0.02, COUNTER_Z + 0.02))
    o.append(_on("counter_banana", "ycb:011_banana", xb - 0.1, yc - 0.03, COUNTER_Z + 0.02, 25.0))
    xk = 0.0
    p.append(Piece("kettle", "cylinder", (0.085, 0.2), (xk, yc, COUNTER_Z + 0.1), "steel"))
    p.append(Piece("kettle_lid", "cylinder", (0.04, 0.02), (xk, yc, COUNTER_Z + 0.21), "black"))
    p.append(_box("kettle_handle", xk - 0.01, xk + 0.01, yc - 0.06, yc + 0.06, COUNTER_Z + 0.2, COUNTER_Z + 0.23,
                  "black"))
    xp = xe - 0.35
    p.append(Piece("plant_pot", "cylinder", (0.09, 0.18), (xp, yc + 0.05, COUNTER_Z + 0.09), "terracotta"))
    for j, (dx, dy, dz, r, f) in enumerate(((0.0, 0.0, 0.29, 0.13, "leaf"), (0.08, -0.04, 0.24, 0.09, "leaf_dark"),
                                            (-0.07, 0.03, 0.25, 0.09, "leaf_dark"))):
        p.append(Piece(f"plant_leaves{j}", "sphere", (r,), (xp + dx, yc + 0.05 + dy, COUNTER_Z + dz), f))
    o.append(_on("counter_cleanser", "ycb:021_bleach_cleanser", xe - 0.75, yc + 0.12, COUNTER_Z, 200.0))

    # a framed print on the west wall
    yp, zp = 0.2, 1.6
    p.append(_box("print_frame", xw, xw + 0.025, yp - 0.42, yp + 0.42, zp - 0.3, zp + 0.3, "black"))
    p.append(_box("print", xw + 0.025, xw + 0.03, yp - 0.38, yp + 0.38, zp - 0.26, zp + 0.26, "paper"))
    for j in range(5):
        p.append(_box(f"print_line{j}", xw + 0.03, xw + 0.034, yp - 0.25 + j * 0.03, yp + 0.2 - j * 0.04,
                      zp - 0.16 + j * 0.07, zp - 0.15 + j * 0.07, "navy"))
    return tuple(p), tuple(o)


def workshop(table_size) -> tuple[tuple[Piece, ...], tuple[Ornament, ...]]:
    """The workshop corner north and west of a table of ``table_size`` (x, y, top z)."""
    tx, ty = table_size[0] / 2, table_size[1] / 2
    yw = ty + AISLE_N + DEPTH  # the north wall's inside face
    xw = -(tx + AISLE_W)  # the west wall's
    xe = tx + 1.2  # the east end of the shelving
    yf = yw - DEPTH  # the bench's front
    p: list[Piece] = []
    o: list[Ornament] = []
    # the walls, painted block, grey over slate; a yellow and black stripe on the floor along the bench
    p.append(_box("wall_n", xw - WALL_T, ROOM_EAST, yw, yw + WALL_T, 0.0, WALL_H, "paint"))
    p.append(_box("wall_w", xw - WALL_T, xw, ROOM_SOUTH, yw, 0.0, WALL_H, "paint"))
    p.append(_box("band_n", xw, ROOM_EAST, yw - 0.01, yw, 0.0, 1.0, "paint_dark"))
    p.append(_box("band_w", xw, xw + 0.01, ROOM_SOUTH, yw - 0.01, 0.0, 1.0, "paint_dark"))
    n = round((xe - xw) / 0.15)
    for i in range(n):
        a = xw + i * (xe - xw) / n
        p.append(_box(f"stripe{i}", a, a + (xe - xw) / n, yf - 0.13, yf - 0.05, 0.0, 0.006,
                      "yellow" if i % 2 == 0 else "black"))

    # the rolling tool cabinet, in the corner: drawers on casters, a chest of shallow ones on top
    cx0, cx1, cy0 = xw + 0.03, xw + 0.78, yw - 0.5
    for i, (x, y) in enumerate(((cx0 + 0.07, cy0 + 0.07), (cx1 - 0.07, cy0 + 0.07), (cx0 + 0.07, yw - 0.09),
                                (cx1 - 0.07, yw - 0.09))):
        p.append(Piece(f"caster{i}", "cylinder", (0.035, 0.08), (x, y, 0.04), "black"))
    p.append(_box("chest_lid", cx0 + 0.03, cx1 - 0.03, cy0 + 0.08, yw - 0.02, 1.4, 1.47, "toolred"))
    for tag, a, b, y, z, heights in (("cabinet", cx0, cx1, cy0, 0.98, (0.08, 0.08, 0.12, 0.12, 0.2, 0.3)),
                                     ("chest", cx0 + 0.03, cx1 - 0.03, cy0 + 0.08, 1.4,
                                      (0.06, 0.06, 0.06, 0.07, 0.08, 0.09))):
        # dark inside, red sides and drawer fronts: the gaps between the drawers show
        z0 = z - sum(heights)
        p.append(_box(tag, a + 0.012, b - 0.012, y, yw - 0.02, z0, z, "black"))
        p.append(_box(f"{tag}_side0", a, a + 0.012, y, yw - 0.02, z0, z, "toolred"))
        p.append(_box(f"{tag}_side1", b - 0.012, b, y, yw - 0.02, z0, z, "toolred"))
        for i, h in enumerate(heights):
            p.append(_box(f"{tag}_drawer{i}", a + 0.014, b - 0.014, y - 0.015, y, z - h + 0.004, z - 0.004, "toolred"))
            p.append(_box(f"{tag}_drawer{i}_handle", a + 0.06, b - 0.06, y - 0.03, y - 0.015, z - 0.03, z - 0.016,
                          "steel"))
            z -= h

    # the workbench, from the cabinet to the shelving: a steel frame, a shelf low in it, a thick plywood top
    x0, x1, zt = cx1 + 0.05, xe - 1.05, COUNTER_Z
    p.append(_box("bench_top", x0, x1, yf - 0.03, yw - 0.005, zt - 0.06, zt, "plywood"))
    p.append(_box("bench_apron", x0 + 0.02, x1 - 0.02, yf, yf + 0.025, zt - 0.15, zt - 0.06, "gunmetal"))
    p.append(_box("bench_rail", x0 + 0.02, x1 - 0.02, yf + 0.005, yf + 0.03, 0.1, 0.14, "gunmetal"))
    p.append(_box("bench_shelf", x0 + 0.02, x1 - 0.02, yf + 0.005, yw - 0.035, 0.14, 0.16, "plywood"))
    n = max(1, round((x1 - x0) / 1.3))
    for i in range(n + 1):
        x = x0 + 0.045 + i * (x1 - x0 - 0.09) / n
        for j, y in enumerate((yf + 0.03, yw - 0.06)):
            p.append(_box(f"bench_leg{i}_{j}", x - 0.025, x + 0.025, y - 0.025, y + 0.025, 0.0, zt - 0.06, "gunmetal"))
    # under it, bins and a toolbox; on it, a vice at the west end, a drill, a clamp, wood, a can of screws
    yc = yf + DEPTH / 2
    for i, (f, w, h, finish) in enumerate(((0.04, 0.4, 0.2, "blue"), (0.16, 0.4, 0.2, "blue"),
                                           (0.42, 0.5, 0.22, "toolred"), (0.72, 0.4, 0.2, "yellow"),
                                           (0.84, 0.4, 0.2, "yellow"))):
        a = x0 + f * (x1 - x0)
        p.append(_box(f"bin{i}", a, a + w, yc - 0.17, yc + 0.17, 0.16, 0.16 + h, finish))
    a = x0 + 0.42 * (x1 - x0)
    p.append(_box("toolbox_handle", a + 0.15, a + 0.35, yc - 0.015, yc + 0.015, 0.38, 0.41, "black"))
    xv = x0 + 0.3
    p.append(_box("vice_base", xv - 0.08, xv + 0.08, yf, yf + 0.2, zt, zt + 0.03, "gunmetal"))
    p.append(_box("vice_body", xv - 0.06, xv + 0.06, yf - 0.05, yf + 0.18, zt + 0.03, zt + 0.11, "gunmetal"))
    p.append(_box("vice_jaws", xv - 0.09, xv + 0.09, yf - 0.06, yf - 0.02, zt + 0.06, zt + 0.15, "steel"))
    p.append(_box("vice_bar", xv - 0.13, xv + 0.13, yf - 0.1, yf - 0.085, zt + 0.07, zt + 0.085, "steel"))
    for name, kind, f, dy, yaw in (("drill", "ycb:035_power_drill", 0.22, 0.02, 200.0),
                                   ("clamp", "ycb:052_extra_large_clamp", 0.4, 0.05, 30.0),
                                   ("block", "ycb:036_wood_block", 0.6, 0.0, 90.0),
                                   ("block_2", "ycb:036_wood_block", 0.64, 0.05, 75.0),
                                   ("can", "ycb:002_master_chef_can", 0.82, 0.12, 0.0),
                                   ("marker", "ycb:040_large_marker", 0.86, -0.08, 60.0)):
        o.append(_on(f"bench_{name}", kind, x0 + f * (x1 - x0), yc + dy, zt, yaw))

    # a pegboard over the bench, its tools hung low on it, where the main camera sees their ends -- a hammer,
    # screwdrivers, spanners, a saw, a square -- and a shelf at its east end
    pb0, pb1, yp, zh = x0 + 0.35, x1 - 0.35, yw - 0.03, 1.35  # the board's ends and front; the hooks' height
    yt = yp - 0.012  # a tool's front
    p.append(_box("pegboard", pb0, pb1, yp, yw - 0.015, zt + 0.02, 1.8, "pegboard"))
    x = pb0 + 0.15
    p.append(_box("hammer_handle", x - 0.014, x + 0.014, yt, yp, zh - 0.35, zh - 0.005, "yellow"))
    p.append(_box("hammer_head", x - 0.07, x + 0.07, yt - 0.015, yp, zh - 0.005, zh + 0.04, "steel"))
    for i, finish in enumerate(("toolred", "yellow", "blue", "black", "toolred")):
        x, shaft = pb0 + 0.35 + i * 0.07, 0.16 - 0.015 * i
        p.append(Piece(f"screwdriver{i}", "cylinder", (0.015, 0.11), (x, yp - 0.016, zh - 0.02), finish))
        p.append(Piece(f"screwdriver{i}_shaft", "cylinder", (0.004, shaft), (x, yp - 0.016, zh - 0.075 - shaft / 2),
                       "steel"))
    for i in range(6):
        x, length = pb0 + 0.78 + i * 0.065, 0.3 - 0.03 * i
        p.append(_box(f"spanner{i}", x - 0.01, x + 0.01, yt, yp, zh - length, zh, "steel"))
        p.append(_box(f"spanner{i}_head", x - 0.024, x + 0.024, yt, yp, zh - 0.04, zh + 0.005, "steel"))
        p.append(_box(f"spanner{i}_ring", x - 0.017, x + 0.017, yt, yp, zh - length - 0.02, zh - length + 0.015,
                      "steel"))
    x = pb0 + 1.25
    p.append(_box("saw_blade", x, x + 0.45, yt, yp, zh - 0.25, zh - 0.14, "steel"))
    p.append(_box("saw_handle", x + 0.45, x + 0.58, yt - 0.012, yp, zh - 0.27, zh - 0.12, "toolred"))
    x = pb0 + 1.95
    p.append(_box("square_blade", x, x + 0.025, yt, yp, zh - 0.33, zh - 0.03, "steel"))
    p.append(_box("square_stock", x, x + 0.15, yt - 0.01, yp, zh - 0.33, zh - 0.29, "black"))
    p.append(_box("pegboard_shelf", pb1 - 0.5, pb1 - 0.05, yp - 0.15, yp, 1.48, 1.5, "plywood"))
    for name, kind, f, yaw in (("coffee", "ycb:002_master_chef_can", 0.12, 0.0),
                               ("soup", "ycb:005_tomato_soup_can", 0.4, 30.0),
                               ("glue", "ycb:006_mustard_bottle", 0.7, 0.0)):
        o.append(_on(f"pegboard_{name}", kind, pb1 - 0.5 + f * 0.45, yp - 0.075, 1.5, yaw))

    # steel shelving at the east end: bins, and things put away
    s0, s1, sy = xe - 1.0, xe, yw - 0.45
    for i, (x, y) in enumerate(((s0, sy), (s1 - 0.035, sy), (s0, yw - 0.055), (s1 - 0.035, yw - 0.055))):
        p.append(_box(f"rack_post{i}", x, x + 0.035, y, y + 0.035, 0.0, 2.0, "steel"))
    for k, z in enumerate((0.12, 0.5, 0.88, 1.26, 1.64, 2.0)):
        p.append(_box(f"rack_shelf{k}", s0, s1, sy, yw - 0.02, z - 0.02, z, "steel"))
    for k, z, finish, m, h in ((0, 0.12, "yellow", 2, 0.26), (1, 0.5, "blue", 3, 0.2), (4, 1.64, "yellow", 3, 0.2)):
        w = (s1 - s0 - 0.08) / m
        for i in range(m):
            a = s0 + 0.04 + i * w
            p.append(_box(f"rack_bin{k}_{i}", a + 0.012, a + w - 0.012, sy + 0.02, yw - 0.04, z, z + h, finish))
    for name, kind, f, z, yaw in (("bleach", "ycb:021_bleach_cleanser", 0.14, 0.88, 160.0),
                                  ("brick", "ycb:061_foam_brick", 0.38, 0.88, 0.0),
                                  ("clamp", "ycb:051_large_clamp", 0.64, 0.88, 90.0),
                                  ("tuna", "ycb:007_tuna_fish_can", 0.88, 0.88, 0.0),
                                  ("drill", "ycb:035_power_drill", 0.25, 1.26, 160.0),
                                  ("block", "ycb:036_wood_block", 0.62, 1.26, 0.0),
                                  ("block_2", "ycb:036_wood_block", 0.78, 1.26, 0.0)):
        o.append(_on(f"rack_{name}", kind, s0 + f * (s1 - s0), sy + 0.22, z, yaw))

    # by the cabinet a fire extinguisher, further along the west wall a first-aid sign; strip lights overhead
    ye = cy0 - 0.3
    p.append(Piece("extinguisher", "cylinder", (0.075, 0.48), (xw + 0.1, ye, 0.78), "toolred"))
    p.append(Piece("extinguisher_valve", "cylinder", (0.025, 0.08), (xw + 0.1, ye, 1.06), "black"))
    p.append(_box("extinguisher_handle", xw + 0.07, xw + 0.2, ye - 0.012, ye + 0.012, 1.09, 1.105, "black"))
    p.append(_box("extinguisher_hose", xw + 0.17, xw + 0.185, ye - 0.008, ye + 0.008, 0.7, 1.06, "black"))
    ys, zs = 0.2, 1.6
    p.append(_flat("sign", "w", xw, ys - 0.2, ys + 0.2, zs - 0.2, zs + 0.2, 0.0, 0.02, "aid_green"))
    p.append(_flat("sign_cross0", "w", xw, ys - 0.04, ys + 0.04, zs - 0.13, zs + 0.13, 0.02, 0.024, "skirting"))
    p.append(_flat("sign_cross1", "w", xw, ys - 0.13, ys + 0.13, zs - 0.04, zs + 0.04, 0.02, 0.024, "skirting"))
    for j, y in enumerate((yw - 0.6, yw - 0.3)):
        p.append(_box(f"strip_light{j}", xw + 0.4, xe - 0.4, y - 0.08, y + 0.08, WALL_H - 0.29, WALL_H - 0.23,
                      "skirting"))
    return tuple(p), tuple(o)


def home(table_size) -> tuple[tuple[Piece, ...], tuple[Ornament, ...]]:
    """The living-room corner north and west of a table of ``table_size`` (x, y, top z)."""
    tx, ty = table_size[0] / 2, table_size[1] / 2
    yw = ty + AISLE_N + DEPTH  # the north wall's inside face
    xw = -(tx + AISLE_W)  # the west wall's
    xe = tx + 1.2  # 0.35 m east of the window
    p: list[Piece] = []
    o: list[Ornament] = []
    # the walls, warm plaster, and their skirting
    p.append(_box("wall_n", xw - WALL_T, ROOM_EAST, yw, yw + WALL_T, 0.0, WALL_H, "clay"))
    p.append(_box("wall_w", xw - WALL_T, xw, ROOM_SOUTH, yw, 0.0, WALL_H, "clay"))
    p.append(_box("skirting_w", xw, xw + 0.015, ROOM_SOUTH, yw, 0.0, 0.09, "skirting"))
    p.append(_box("skirting_n", xw, ROOM_EAST, yw - 0.015, yw, 0.0, 0.09, "skirting"))

    # the bookcase, in the corner: walnut, rows of books in many colours, a jug in one
    bx0, bx1, by = xw + 0.02, xw + 0.94, yw - 0.36
    p.append(_box("bookcase_back", bx0, bx1, yw - 0.02, yw, 0.0, 2.0, "walnut"))
    p.append(_box("bookcase_side0", bx0, bx0 + 0.025, by, yw - 0.02, 0.0, 2.0, "walnut"))
    p.append(_box("bookcase_side1", bx1 - 0.025, bx1, by, yw - 0.02, 0.0, 2.0, "walnut"))
    rng = random.Random("home/books")
    colours = ("wine", "navy", "olive", "mustard", "paper", "teal", "terracotta", "slate")
    for k, z in enumerate((0.1, 0.48, 0.86, 1.24, 1.62, 2.0)):
        p.append(_box(f"bookcase_shelf{k}", bx0 + 0.025, bx1 - 0.025, by, yw - 0.02, z - 0.025, z, "walnut"))
        x, end, i = bx0 + 0.035, bx1 - (0.3 if k == 2 else rng.uniform(0.04, 0.2)), 0
        while k < 5:
            w, h, d = rng.uniform(0.022, 0.05), rng.uniform(0.2, 0.3), rng.uniform(0.17, 0.24)
            if x + w > end:
                break
            p.append(_box(f"book{k}_{i}", x, x + w, by + 0.02, by + 0.02 + d, z, z + h, rng.choice(colours)))
            x, i = x + w + 0.002, i + 1
    o.append(_on("bookcase_jug", "ycb:019_pitcher_base", bx1 - 0.16, by + 0.17, 0.86, 30.0))

    # the sideboard, walnut on short legs: a lamp, books, a bowl and a plant on it
    s0, s1, sy, zs = bx1 + 0.45, xe - 2.3, yw - 0.46, 0.75  # its ends, front and top
    for i, (x, y) in enumerate(((s0 + 0.06, sy + 0.06), (s1 - 0.06, sy + 0.06), (s0 + 0.06, yw - 0.07),
                                (s1 - 0.06, yw - 0.07))):
        p.append(Piece(f"sideboard_leg{i}", "cylinder", (0.018, 0.14), (x, y, 0.07), "black"))
    p.append(_box("sideboard", s0, s1, sy + 0.012, yw - 0.01, 0.14, zs - 0.03, "walnut"))
    p.append(_box("sideboard_top", s0 - 0.01, s1 + 0.01, sy - 0.005, yw - 0.01, zs - 0.03, zs, "walnut"))
    n = max(2, round((s1 - s0) / 0.5))
    for i in range(n):
        a, b = s0 + i * (s1 - s0) / n, s0 + (i + 1) * (s1 - s0) / n
        p.append(_box(f"sideboard_door{i}", a + 0.004, b - 0.004, sy, sy + 0.012, 0.16, zs - 0.05, "walnut"))
        p.append(_box(f"sideboard_door{i}_handle", (a + b) / 2 - 0.05, (a + b) / 2 + 0.05, sy - 0.015, sy,
                      zs - 0.1, zs - 0.088, "brass"))
    yc = (sy + yw) / 2
    x = s0 + 0.28
    p.append(Piece("lamp_base", "cylinder", (0.075, 0.24), (x, yc, zs + 0.12), "teal"))
    p.append(Piece("lamp_stem", "cylinder", (0.01, 0.12), (x, yc, zs + 0.3), "brass"))
    p.append(Piece("lamp_shade", "cylinder", (0.16, 0.22), (x, yc, zs + 0.45), "linen"))
    x, z = s0 + 0.38 * (s1 - s0), zs
    for j, (w, d, h, f) in enumerate(((0.26, 0.19, 0.04, "navy"), (0.24, 0.17, 0.035, "mustard"),
                                      (0.21, 0.15, 0.03, "wine"))):
        p.append(_box(f"sideboard_book{j}", x - w / 2, x + w / 2, yc - d / 2, yc + d / 2, z, z + h, f))
        z += h
    o.append(_on("sideboard_bowl", "ycb:024_bowl", s0 + 0.6 * (s1 - s0), yc, zs))
    # a snake plant: upright blades in a pot
    x = s1 - 0.2
    p.append(Piece("plant_pot", "cylinder", (0.1, 0.2), (x, yc, zs + 0.1), "terracotta"))
    for j, (dx, dy, h, yaw) in enumerate(((0.0, 0.0, 0.55, 0.0), (0.05, 0.03, 0.42, 60.0), (-0.05, 0.02, 0.47, 120.0),
                                          (0.02, -0.05, 0.38, 30.0), (-0.03, -0.03, 0.5, 150.0),
                                          (0.05, -0.02, 0.33, 95.0))):
        p.append(Piece(f"plant_leaf{j}", "box", (0.045, 0.012, h), (x + dx, yc + dy, zs + 0.18 + h / 2),
                       "leaf_dark" if j % 2 else "leaf", yaw))

    # three pictures over it: a landscape between two prints
    xc = (s0 + s1) / 2
    p += _picture("picture", "n", yw, xc, 1.4, 0.8, 0.6, "black", (
        (-0.33, 0.33, -0.03, 0.23, "sky"), (-0.33, 0.1, -0.23, 0.05, "olive"), (-0.33, 0.33, -0.23, -0.08, "mustard"),
        (0.15, 0.23, 0.1, 0.18, "terracotta")))
    p += _picture("print_w", "n", yw, xc - 0.68, 1.38, 0.34, 0.44, "walnut", (
        (-0.1, -0.03, -0.14, 0.12, "teal"), (-0.03, 0.04, -0.14, 0.02, "terracotta"), (0.04, 0.1, -0.14, 0.07, "navy")))
    p += _picture("print_e", "n", yw, xc + 0.68, 1.38, 0.34, 0.44, "walnut", tuple(
        (-0.1 + 0.02 * j, 0.1 - 0.02 * j, -0.12 + 0.06 * j, -0.1 + 0.06 * j, "wine") for j in range(5)))

    # a window east of it: a pale sky, white frame and sill, curtains drawn back
    wx0, wx1, wz0, wz1 = xe - 1.75, xe - 0.35, 0.9, 2.3
    xm = (wx0 + wx1) / 2
    p.append(_box("window", wx0, wx1, yw - 0.006, yw, wz0, wz1, "sky"))
    for name, a, b, c, d in (("window_top", wx0 - 0.06, wx1 + 0.06, wz1, wz1 + 0.06),
                             ("window_bottom", wx0 - 0.06, wx1 + 0.06, wz0 - 0.03, wz0 + 0.03),
                             ("window_w", wx0 - 0.06, wx0, wz0, wz1), ("window_e", wx1, wx1 + 0.06, wz0, wz1),
                             ("window_mullion", xm - 0.025, xm + 0.025, wz0, wz1),
                             ("window_transom", wx0, wx1, 1.83, 1.87)):
        p.append(_box(name, a, b, yw - 0.05, yw, c, d, "skirting"))
    p.append(_box("window_sill", wx0 - 0.05, wx1 + 0.05, yw - 0.1, yw, wz0 - 0.07, wz0 - 0.03, "skirting"))
    for j, x in enumerate((wx0 - 0.42, wx1 + 0.12)):
        for i in range(5):
            p.append(Piece(f"curtain{j}_{i}", "cylinder", (0.045, 2.45), (x + 0.045 + i * 0.06, yw - 0.12, 1.255),
                           "olive"))
    p.append(_box("curtain_rod", wx0 - 0.5, wx1 + 0.5, yw - 0.13, yw - 0.11, 2.5, 2.52, "brass"))

    # by the west wall an armchair facing the room, a floor lamp beside it, a picture over it
    xa, ay0, ay1 = xw + 0.04, yw - 1.8, yw - 0.95  # its back; its south and north sides
    for i, (x, y) in enumerate(((xa + 0.06, ay0 + 0.06), (xa + 0.74, ay0 + 0.06), (xa + 0.06, ay1 - 0.06),
                                (xa + 0.74, ay1 - 0.06))):
        p.append(Piece(f"chair_leg{i}", "cylinder", (0.02, 0.1), (x, y, 0.05), "walnut"))
    p.append(_box("chair_seat", xa, xa + 0.8, ay0, ay1, 0.1, 0.4, "teal"))
    p.append(_box("chair_back", xa, xa + 0.2, ay0, ay1, 0.4, 0.95, "teal"))
    p.append(_box("chair_arm0", xa, xa + 0.8, ay0, ay0 + 0.14, 0.4, 0.64, "teal"))
    p.append(_box("chair_arm1", xa, xa + 0.8, ay1 - 0.14, ay1, 0.4, 0.64, "teal"))
    p.append(_box("chair_cushion", xa + 0.2, xa + 0.79, ay0 + 0.14, ay1 - 0.14, 0.4, 0.49, "teal"))
    ya = (ay0 + ay1) / 2
    p.append(_box("chair_pillow", xa + 0.2, xa + 0.32, ya - 0.2, ya + 0.2, 0.49, 0.82, "mustard"))
    x, y = xw + 0.32, yw - 0.66
    p.append(Piece("floor_lamp_base", "cylinder", (0.14, 0.03), (x, y, 0.015), "black"))
    p.append(Piece("floor_lamp_pole", "cylinder", (0.012, 1.42), (x, y, 0.74), "black"))
    p.append(Piece("floor_lamp_shade", "cylinder", (0.2, 0.3), (x, y, 1.58), "linen"))
    p += _picture("chair_picture", "w", xw, ya, 1.5, 0.6, 0.45, "walnut", (
        (-0.2, 0.2, -0.15, 0.15, "navy"), (-0.12, 0.06, -0.08, 0.1, "mustard")))
    return tuple(p), tuple(o)


def lamps(name: str, table_size) -> tuple[Lamp, ...]:
    """The lights of backdrop ``name``: the dome light, far off, leaves a corner by its walls dim.  Every corner has
    the same one, over its north run, in the look's colour (the workshop's cool, the home's warm)."""
    if name == "none":
        return ()
    tx, ty = table_size[0] / 2, table_size[1] / 2
    yw, xw = ty + AISLE_N + DEPTH, -(tx + AISLE_W)
    xe = tx + 1.2
    return (Lamp("ceiling_light", (xe - xw, 0.6), ((xw + xe) / 2, yw - 0.45, WALL_H - 0.3), LAMP_INTENSITY, LAMP_CONE),)


def build(name: str, table_size) -> tuple[tuple[Piece, ...], tuple[Ornament, ...]]:
    """The pieces and ornaments of backdrop ``name`` round a table of ``table_size``."""
    if name == "none":
        return (), ()
    if name == "kitchen":
        return kitchen(table_size)
    if name == "workshop":
        return workshop(table_size)
    if name == "home":
        return home(table_size)
    raise ValueError(f"no backdrop {name!r}; have {', '.join(BACKDROPS)}")


def footprint(item, ornament_size: Optional[Vec3] = None) -> tuple[float, float, float, float]:
    """(x0, x1, y0, y1) of a piece or an ornament, its yaw ignored but its half-diagonal allowed for when turned."""
    import math

    if isinstance(item, Piece):
        hx, hy, _ = item.half()
        if item.yaw_deg:
            hx = hy = math.hypot(hx, hy)
    else:
        sx, sy, _ = ornament_size
        hx = hy = math.hypot(sx, sy) / 2
    return item.pos[0] - hx, item.pos[0] + hx, item.pos[1] - hy, item.pos[1] + hy
