"""Props: things on the table that the task does not use, and where they may stand.

A prop is a scanned YCB object from Isaac's asset server (a mug, a cracker box, a
power drill) or a thing built from boxes, cylinders and balls (a stack of books, a
potted plant, a desk lamp).  Either is one rigid body that can be knocked over.

Props are set out in groups (``GROUPS``) of one or two things, as a person leaves
them -- a stack of books and a mug, a row of books between bookends, a plant --
square to the table, the group's back to its north or south edge.
``place_props`` stands a look's groups where no arm works: each prop's footprint
stays ``REACH_GAP`` beyond every arm's reach (``RobotModel.reach_m``, where its hand
can work on the table), ``TASK_GAP`` clear of every body, region and cloth of the
task, and on the table, and a group ``GROUP_GAP`` clear of the other groups.  A tall
prop stands in the table's north half, behind the task as the main camera sees it.
On three Frankas' table that leaves its four corners; with Pipers, more.  An arm stretched flat to the
table can still sweep past its reach to a prop (a Franka's links reach 1.0 m low over
the table): a prop that moves is a disturbance the run reports, not a collision the
task is built around.  Engine-independent.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

from ..core.errors import ConfigError
from ..engines.base import Look, Part, PropSpec, RGB, SceneSpec, Vec3, quat_rotate
from ..robots import get_model

#: beyond every arm's reach, from the edge of the table, from the task's things, and between groups (m)
REACH_GAP, EDGE, TASK_GAP, GROUP_GAP = 0.05, 0.03, 0.08, 0.12
#: a group stands this far in from the edge at its back, and is tried every STEP along it
BACK, STEP = 0.05, 0.05
#: a prop taller than this stands in the north half of the table (y >= 0), behind the task from the main camera
TALL_M = 0.15


@dataclass(frozen=True)
class Prop:
    kind: str
    #: its upright bounding box (m)
    size: Vec3
    mass_kg: float
    #: a scan: its file under the asset server's ``Isaac`` directory, the axis of the file's frame that points up
    #: ("-y": most of Isaac's YCB files stand on their +y side -- upright, their labels read the right way up;
    #: "+z": it lies as it is) and the centre of its bounding box in that frame
    usd: str = ""
    up: str = "+z"
    centre: Vec3 = (0.0, 0.0, 0.0)
    #: built: its shapes, in a frame at the centre of its bounding box
    parts: tuple[Part, ...] = ()
    colour: RGB = (0.6, 0.6, 0.6)

    @property
    def radius(self) -> float:
        """Half its footprint's diagonal: its footprint, however it is turned, is inside this circle."""
        return math.hypot(self.size[0], self.size[1]) / 2


#: a file stood upright: as it is, or a quarter turn about x taking its -y to +z
UP_QUAT = {"+z": (1.0, 0.0, 0.0, 0.0), "-y": (math.sqrt(0.5), -math.sqrt(0.5), 0.0, 0.0)}


def _scan(ident: str, lo: Vec3, hi: Vec3, up: str, mass: float) -> Prop:
    """A YCB scan, from its bounding box as Isaac loads the file (``lo``, ``hi``, in the file's frame)."""
    size = [h - l for l, h in zip(lo, hi)]
    centre = tuple((l + h) / 2 for l, h in zip(lo, hi))
    upright = quat_rotate(UP_QUAT[up], size)
    return Prop(f"ycb:{ident}", tuple(round(abs(v), 4) for v in upright), mass,  # type: ignore[arg-type]
                usd=f"Props/YCB/Axis_Aligned/{ident}.usd", up=up, centre=centre)  # type: ignore[arg-type]


def _sym(ident: str, half: Vec3, up: str, mass: float) -> Prop:
    return _scan(ident, tuple(-h for h in half), half, up, mass)  # type: ignore[arg-type]


def _rgb(hex_: str) -> RGB:
    return tuple(int(hex_[i:i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


def _p(name, shape, size, pos, yaw=0.0, colour: Optional[str] = None) -> Part:
    return Part(name, shape, tuple(size), tuple(pos), yaw, _rgb(colour) if colour else None)  # type: ignore[arg-type]


def _built(kind: str, mass: float, colour: str, parts: list[Part]) -> Prop:
    """A prop made of ``parts`` given from the middle of its foot (z up from the table): re-centred on its
    bounding box."""
    lo, hi = [math.inf] * 3, [-math.inf] * 3
    for p in parts:
        hx, hy, hz = (p.size[0], p.size[0], p.size[1] / 2) if p.shape == "cylinder" else \
            (p.size[0],) * 3 if p.shape == "sphere" else (p.size[0] / 2, p.size[1] / 2, p.size[2] / 2)
        c, s = abs(math.cos(math.radians(p.yaw_deg))), abs(math.sin(math.radians(p.yaw_deg)))
        ext = (c * hx + s * hy, s * hx + c * hy, hz)
        lo = [min(a, q - e) for a, q, e in zip(lo, p.pos, ext)]
        hi = [max(a, q + e) for a, q, e in zip(hi, p.pos, ext)]
    mid = [(a + b) / 2 for a, b in zip(lo, hi)]
    moved = tuple(Part(p.name, p.shape, p.size, tuple(round(v - m, 5) for v, m in zip(p.pos, mid)),  # type: ignore
                       p.yaw_deg, p.color) for p in parts)
    return Prop(kind, tuple(round(b - a, 4) for a, b in zip(lo, hi)), mass, parts=moved,  # type: ignore[arg-type]
                colour=_rgb(colour))


_SCANS = [  # bounding boxes as Isaac 4.5 loads Props/YCB/Axis_Aligned (measured); masses from the YCB set
    _sym("002_master_chef_can", (0.0512, 0.0701, 0.0512), "-y", 0.414),
    # lying on its face: stood on its end, it leans back and slides as it settles
    _sym("003_cracker_box", (0.082, 0.1067, 0.0359), "+z", 0.411),
    _sym("004_sugar_box", (0.0463, 0.0881, 0.0226), "-y", 0.514),
    _sym("005_tomato_soup_can", (0.0338, 0.0509, 0.0339), "-y", 0.349),
    _sym("006_mustard_bottle", (0.048, 0.0957, 0.0291), "-y", 0.603),
    _sym("007_tuna_fish_can", (0.0428, 0.0168, 0.0428), "-y", 0.171),
    _sym("008_pudding_box", (0.0568, 0.0449, 0.0192), "+z", 0.187),
    _sym("009_gelatin_box", (0.0446, 0.0366, 0.015), "+z", 0.097),
    _sym("010_potted_meat_can", (0.0508, 0.0418, 0.0288), "-y", 0.370),
    _sym("011_banana", (0.0986, 0.0193, 0.037), "-y", 0.066),
    _sym("019_pitcher_base", (0.0943, 0.1212, 0.0666), "-y", 0.178),
    _sym("021_bleach_cleanser", (0.0512, 0.1253, 0.0338), "-y", 1.131),
    _sym("024_bowl", (0.0806, 0.0275, 0.0807), "-y", 0.147),
    _sym("025_mug", (0.0585, 0.0407, 0.0465), "-y", 0.118),
    _sym("035_power_drill", (0.0922, 0.0934, 0.0286), "+z", 0.895),
    _sym("036_wood_block", (0.0448, 0.103, 0.0454), "+z", 0.729),
    _scan("037_scissors", (-0.0631, -0.0988, -0.0093), (0.0279, 0.0985, 0.0079), "+z", 0.082),
    _sym("040_large_marker", (0.0094, 0.0604, 0.0097), "+z", 0.016),
    _sym("051_large_clamp", (0.0825, 0.0608, 0.0182), "+z", 0.125),
    _sym("052_extra_large_clamp", (0.1013, 0.0826, 0.0182), "+z", 0.202),
    _sym("061_foam_brick", (0.0389, 0.0256, 0.0263), "-y", 0.028),
]

_BUILT = [
    # squared up: three books, the smaller on the larger, their backs flush
    _built("books", 1.4, "7a2e2e", [
        _p("bottom", "box", (0.24, 0.17, 0.035), (0, 0, 0.0175), 0, "7a2e2e"),
        _p("middle", "box", (0.23, 0.16, 0.03), (-0.005, 0.005, 0.05), 0, "2e4a7a"),
        _p("top", "box", (0.21, 0.15, 0.028), (-0.015, 0.01, 0.079), 0, "d8cfb2")]),
    # six books standing between two bookends
    _built("book_row", 1.8, "4a4a52", [
        _p("end_w", "box", (0.01, 0.13, 0.15), (-0.112, 0, 0.075), 0, "4a4a52"),
        _p("b1", "box", (0.035, 0.15, 0.235), (-0.0885, 0, 0.1175), 0, "7a2e2e"),
        _p("b2", "box", (0.028, 0.15, 0.215), (-0.057, 0, 0.1075), 0, "2e4a7a"),
        _p("b3", "box", (0.04, 0.15, 0.24), (-0.022, 0, 0.12), 0, "c9b27a"),
        _p("b4", "box", (0.03, 0.15, 0.22), (0.014, 0, 0.11), 0, "3d6b4f"),
        _p("b5", "box", (0.038, 0.15, 0.23), (0.049, 0, 0.115), 0, "8a6a3a"),
        _p("b6", "box", (0.033, 0.15, 0.205), (0.0845, 0, 0.1025), 0, "5a3d6b"),
        _p("end_e", "box", (0.01, 0.13, 0.15), (0.106, 0, 0.075), 0, "4a4a52")]),
    _built("pen_cup", 0.3, "3a3a3d", [
        _p("cup", "cylinder", (0.04, 0.10), (0, 0, 0.05), 0, "3a3a3d"),
        _p("pen_blue", "cylinder", (0.005, 0.14), (0.015, 0.01, 0.07), 0, "2450b0"),
        _p("pen_red", "cylinder", (0.005, 0.15), (-0.016, 0.006, 0.075), 0, "c0302a"),
        _p("pencil", "cylinder", (0.0045, 0.16), (0.0, -0.017, 0.08), 0, "e8b830")]),
    _built("plant", 1.2, "b5573a", [
        _p("pot", "cylinder", (0.07, 0.12), (0, 0, 0.06), 0, "b5573a"),
        _p("soil", "cylinder", (0.063, 0.012), (0, 0, 0.12), 0, "3d2a1e"),
        _p("leaves", "sphere", (0.075,), (0, 0, 0.19), 0, "3f7a35"),
        _p("leaves_e", "sphere", (0.055,), (0.05, 0.02, 0.17), 0, "4b8a3c"),
        _p("leaves_w", "sphere", (0.05,), (-0.045, -0.025, 0.165), 0, "356a2e")]),
    # its weight over its base: a shade further out tipped it over
    _built("desk_lamp", 1.5, "2a2a2c", [
        _p("base", "cylinder", (0.09, 0.03), (0, 0, 0.015), 0, "2a2a2c"),
        _p("stem", "cylinder", (0.01, 0.28), (-0.03, 0, 0.17), 0, "b8b8bc"),
        _p("arm", "box", (0.15, 0.018, 0.018), (0.035, 0, 0.30), 0, "b8b8bc"),
        _p("shade", "cylinder", (0.045, 0.07), (0.10, 0, 0.26), 0, "e8e4d8")]),
    _built("tissue_box", 0.2, "8fb8d8", [
        _p("box", "box", (0.22, 0.12, 0.09), (0, 0, 0.045), 0, "8fb8d8"),
        _p("tissue", "sphere", (0.028,), (0, 0, 0.098), 0, "f4f4f2")]),
    _built("laptop", 1.4, "a8abb0", [
        _p("base", "box", (0.32, 0.22, 0.012), (0, 0, 0.006), 0, "a8abb0"),
        _p("lid", "box", (0.32, 0.22, 0.008), (0, 0, 0.016), 0, "8d9096")]),
    _built("notebook", 0.3, "2e6b5a", [
        _p("cover", "box", (0.21, 0.15, 0.014), (0, 0, 0.007), 0, "2e6b5a")]),
    _built("coffee_cup", 0.25, "f2efe8", [
        _p("cup", "cylinder", (0.042, 0.11), (0, 0, 0.055), 0, "f2efe8"),
        _p("sleeve", "cylinder", (0.044, 0.04), (0, 0, 0.055), 0, "9a6a3e"),
        _p("lid", "cylinder", (0.045, 0.012), (0, 0, 0.116), 0, "2b2b2b")]),
    _built("jar", 0.5, "c98a2c", [
        _p("jar", "cylinder", (0.045, 0.12), (0, 0, 0.06), 0, "c98a2c"),
        _p("lid", "cylinder", (0.047, 0.02), (0, 0, 0.13), 0, "b03028")]),
    _built("water_bottle", 0.6, "3d8f6a", [
        _p("bottle", "cylinder", (0.035, 0.19), (0, 0, 0.095), 0, "3d8f6a"),
        _p("neck", "cylinder", (0.017, 0.04), (0, 0, 0.21), 0, "3d8f6a"),
        _p("cap", "cylinder", (0.019, 0.02), (0, 0, 0.24), 0, "e8e8e8")]),
    # a lab bench's
    _built("multimeter", 0.35, "f2c230", [
        _p("body", "box", (0.08, 0.15, 0.04), (0, 0, 0.02), 0, "f2c230"),
        _p("display", "box", (0.055, 0.045, 0.004), (0, 0.035, 0.041), 0, "20262a"),
        _p("dial", "cylinder", (0.016, 0.006), (0, -0.02, 0.043), 0, "2a2a2c"),
        _p("jack_red", "cylinder", (0.004, 0.004), (0.02, -0.06, 0.041), 0, "c0302a"),
        _p("jack_black", "cylinder", (0.004, 0.004), (-0.02, -0.06, 0.041), 0, "1e1e1e")]),
    _built("oscilloscope", 3.5, "c8c9cc", [
        _p("case", "box", (0.32, 0.14, 0.16), (0, 0, 0.08), 0, "c8c9cc"),
        _p("screen", "box", (0.17, 0.004, 0.11), (-0.06, -0.071, 0.09), 0, "1b2a33"),
        _p("panel", "box", (0.11, 0.004, 0.12), (0.09, -0.071, 0.085), 0, "3a3d42"),
        _p("knob_a", "cylinder", (0.01, 0.012), (0.07, -0.06, 0.03), 0, "1e1e1e"),
        _p("knob_b", "cylinder", (0.01, 0.012), (0.11, -0.06, 0.03), 0, "1e1e1e")]),
    _built("cable_reel", 0.8, "1e1e1e", [
        _p("reel", "cylinder", (0.07, 0.06), (0, 0, 0.03), 0, "1e1e1e"),
        _p("cable", "cylinder", (0.06, 0.04), (0, 0, 0.03), 0, "e86a1a"),
        _p("hub", "cylinder", (0.02, 0.064), (0, 0, 0.032), 0, "8d9096")]),
    _built("parts_box", 0.3, "2f6fb0", [
        _p("box", "box", (0.16, 0.10, 0.05), (0, 0, 0.025), 0, "2f6fb0"),
        _p("lid", "box", (0.164, 0.104, 0.008), (0, 0, 0.054), 0, "9ec4e8")]),
    _built("screwdriver", 0.1, "d8402a", [
        _p("handle", "box", (0.10, 0.026, 0.026), (-0.045, 0, 0.013), 0, "d8402a"),
        _p("grip", "box", (0.04, 0.028, 0.028), (-0.055, 0, 0.014), 0, "f2c230"),
        _p("shaft", "box", (0.09, 0.006, 0.006), (0.05, 0, 0.013), 0, "b8b8bc")]),
    _built("solder_spool", 0.25, "c9b27a", [
        _p("spool", "cylinder", (0.035, 0.03), (0, 0, 0.015), 0, "c9b27a"),
        _p("flange", "cylinder", (0.038, 0.004), (0, 0, 0.032), 0, "1e1e1e"),
        _p("hub", "cylinder", (0.012, 0.036), (0, 0, 0.018), 0, "1e1e1e")]),
    _built("toolcase", 2.0, "b83228", [
        _p("case", "box", (0.32, 0.17, 0.12), (0, 0, 0.06), 0, "b83228"),
        _p("handle", "box", (0.12, 0.025, 0.025), (0, 0, 0.14), 0, "1e1e1e"),
        _p("post_e", "box", (0.02, 0.02, 0.03), (0.05, 0, 0.13), 0, "1e1e1e"),
        _p("post_w", "box", (0.02, 0.02, 0.03), (-0.05, 0, 0.13), 0, "1e1e1e")]),
]

PROPS: dict[str, Prop] = {p.kind: p for p in _SCANS + _BUILT}

@dataclass(frozen=True)
class Group:
    """One or two props set out together: (kind, x, y, yaw) in the group's frame -- x along the edge it stands
    against, y toward that edge, yaw in degrees."""

    name: str
    items: tuple[tuple[str, float, float, float], ...]


def _group(name, *items) -> Group:
    return Group(name, tuple(items))


GROUPS: dict[str, Group] = {g.name: g for g in (
    # at home
    _group("books_and_mug", ("books", 0.0, 0.0, 0.0), ("ycb:025_mug", 0.215, -0.03, 200.0)),
    _group("book_row", ("book_row", 0.0, 0.0, 0.0)),
    _group("plant_and_tissues", ("plant", 0.0, 0.0, 0.0), ("tissue_box", 0.235, -0.01, 0.0)),
    _group("lamp_and_cup", ("desk_lamp", 0.0, 0.02, 0.0), ("coffee_cup", 0.21, -0.04, 0.0)),
    # in a kitchen
    _group("cans", ("ycb:002_master_chef_can", -0.06, 0.0, 0.0), ("ycb:005_tomato_soup_can", 0.06, -0.01, 0.0)),
    _group("bowl_and_banana", ("ycb:024_bowl", 0.0, 0.02, 0.0), ("ycb:011_banana", 0.0, -0.12, 0.0)),
    _group("pitcher_and_mug", ("ycb:019_pitcher_base", 0.0, 0.0, 0.0), ("ycb:025_mug", 0.18, -0.03, 0.0)),
    _group("jar_and_mustard", ("jar", 0.0, 0.0, 0.0), ("ycb:006_mustard_bottle", 0.12, 0.0, 0.0)),
    # in a lab
    _group("oscilloscope", ("oscilloscope", 0.0, 0.0, 0.0)),
    _group("meter_and_screwdriver", ("multimeter", 0.0, 0.0, 0.0), ("screwdriver", 0.0, -0.13, 0.0)),
    _group("parts_and_solder", ("parts_box", 0.0, 0.0, 0.0), ("solder_spool", 0.14, -0.02, 0.0)),
    _group("reel_and_cup", ("cable_reel", 0.0, 0.0, 0.0), ("coffee_cup", 0.13, -0.02, 0.0)),
    # on a desk
    _group("laptop_and_mug", ("laptop", 0.0, 0.0, 0.0), ("ycb:025_mug", 0.255, -0.03, 200.0)),
    _group("books_and_pens", ("books", 0.0, 0.0, 0.0), ("pen_cup", 0.18, 0.02, 0.0)),
    _group("notebook_and_marker", ("notebook", 0.0, 0.0, 0.0), ("ycb:040_large_marker", 0.14, 0.0, 0.0)),
    # on a workbench
    _group("toolcase", ("toolcase", 0.0, 0.0, 0.0)),
    _group("drill_and_block", ("ycb:035_power_drill", 0.0, 0.0, 0.0), ("ycb:036_wood_block", 0.17, 0.0, 0.0)),
    _group("clamps", ("ycb:051_large_clamp", 0.0, 0.02, 0.0), ("ycb:052_extra_large_clamp", 0.0, -0.145, 0.0)),
    _group("bleach_and_brick", ("ycb:021_bleach_cleanser", 0.0, 0.0, 0.0), ("ycb:061_foam_brick", 0.11, -0.02, 0.0)),
)}

#: prop sets: the groups that stand about at home, in a kitchen, in a lab, on a desk, on a workbench
CLUTTER: dict[str, tuple[str, ...]] = {
    "home": ("books_and_mug", "book_row", "plant_and_tissues", "lamp_and_cup"),
    "kitchen": ("cans", "bowl_and_banana", "pitcher_and_mug", "jar_and_mustard"),
    "lab": ("oscilloscope", "meter_and_screwdriver", "parts_and_solder", "reel_and_cup"),
    "desk": ("laptop_and_mug", "books_and_pens", "lamp_and_cup", "notebook_and_marker"),
    "workshop": ("toolcase", "drill_and_block", "clamps", "bleach_and_brick"),
}
CLUTTER["mixed"] = tuple(GROUPS)


def footprint(kind: str, x: float, y: float, yaw_deg: float) -> tuple[float, float, float, float]:
    """A prop's footprint on the table, standing at (x, y) turned ``yaw_deg``: the box (x0, y0, x1, y1) round it."""
    sx, sy, _ = PROPS[kind].size
    c, s = abs(math.cos(math.radians(yaw_deg))), abs(math.sin(math.radians(yaw_deg)))
    hx, hy = (c * sx + s * sy) / 2, (s * sx + c * sy) / 2
    return (x - hx, y - hy, x + hx, y + hy)


def _union(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def overlap(a, b, gap: float = 0.0) -> bool:
    """Do two boxes on the table come closer than ``gap``?"""
    return a[0] < b[2] + gap and b[0] < a[2] + gap and a[1] < b[3] + gap and b[1] < a[3] + gap


def _footprints(scene: SceneSpec) -> list[tuple[float, float, float, float]]:
    """The task's bodies, regions and cloths as boxes on the table: (x0, y0, x1, y1)."""
    out = []
    for o in scene.objects:
        corners = [(sx * o.size[0] / 2, sy * o.size[1] / 2, 0.0) for sx in (-1, 1) for sy in (-1, 1)] \
            if o.shape in ("box", "compound") else [(sx * o.size[0], sy * o.size[0], 0.0) for sx in (-1, 1)
                                                    for sy in (-1, 1)]
        if o.parts:  # a compound's own frame need not be the middle of its box: take its parts, a box by its corners
            corners = []
            for p in o.parts:
                if p.shape == "box":
                    c, s_ = math.cos(math.radians(p.yaw_deg)), math.sin(math.radians(p.yaw_deg))
                    hx, hy = p.size[0] / 2, p.size[1] / 2
                    corners += [(p.pos[0] + c * sx * hx - s_ * sy * hy, p.pos[1] + s_ * sx * hx + c * sy * hy, p.pos[2])
                                for sx in (-1, 1) for sy in (-1, 1)]
                else:
                    r = p.size[0]
                    corners += [(p.pos[0] + sx * r, p.pos[1] + sy * r, p.pos[2]) for sx in (-1, 1) for sy in (-1, 1)]
        pts = [quat_rotate(o.pose[3:], c) for c in corners]
        xs, ys = [o.pos[0] + p[0] for p in pts], [o.pos[1] + p[1] for p in pts]
        out.append((min(xs), min(ys), max(xs), max(ys)))
    for r in scene.regions:
        out.append((r.center[0] - r.half[0], r.center[1] - r.half[1], r.center[0] + r.half[0], r.center[1] + r.half[1]))
    for c in scene.cloths:
        half = math.hypot(*c.size) / 2
        out.append((c.pos[0] - half, c.pos[1] - half, c.pos[0] + half, c.pos[1] + half))
    return out


def _point_box(pt, box) -> float:
    """From a point to a box on the table (0 inside it)."""
    (x, y), (x0, y0, x1, y1) = pt, box
    return math.hypot(max(x0 - x, 0.0, x - x1), max(y0 - y, 0.0, y - y1))


def spot_ok(scene: SceneSpec, kind: str, x: float, y: float, yaw_deg: float = 0.0) -> bool:
    """May a ``kind`` prop stand at (x, y) turned ``yaw_deg``: its footprint on the table, beyond every arm's
    reach, clear of the task's things, and -- if it is tall -- in the table's north half?"""
    box = footprint(kind, x, y, yaw_deg)
    tx, ty, _ = scene.table_size
    if box[0] < -tx / 2 + EDGE or box[2] > tx / 2 - EDGE or box[1] < -ty / 2 + EDGE or box[3] > ty / 2 - EDGE:
        return False
    if PROPS[kind].size[2] > TALL_M and y < 0.0:
        return False
    if any(_point_box(a.pos[:2], box) < get_model(a.model).reach_m + REACH_GAP for a in scene.arms):
        return False
    return not any(overlap(box, b, TASK_GAP) for b in _footprints(scene))


def _set_out(group: Group, x: float, y: float, back: int) -> list[tuple[str, float, float, float]]:
    """A group's props in the world, the middle of its box at (x, y), its back to the north (``back`` 1) or the
    south (-1) edge: (kind, x, y, yaw)."""
    box = _union([footprint(k, ix, iy, yaw) for k, ix, iy, yaw in group.items])
    mx, my = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return [(k, round(x + back * (ix - mx), 4), round(y + back * (iy - my), 4),
             round((yaw + (0.0 if back > 0 else 180.0)) % 360.0, 1)) for k, ix, iy, yaw in group.items]


def place_props(scene: SceneSpec, look: Look, rng: random.Random) -> tuple[PropSpec, ...]:
    """Up to ``look.clutter_n`` of the look's groups, drawn by ``rng``, each against the north or south edge of
    the table at a spot drawn from those (every ``STEP`` along it) where all its props may stand and it keeps
    ``GROUP_GAP`` from the groups already there.  A group with no such spot is left out."""
    if look.clutter == "none" or look.clutter_n <= 0:
        return ()
    if look.clutter not in CLUTTER:
        raise ConfigError(f"no prop set {look.clutter!r}; have {sorted(CLUTTER)} or none")
    names = list(CLUTTER[look.clutter])
    rng.shuffle(names)
    tx, ty, _ = scene.table_size
    taken, out = [], []
    for name in names:
        if len(taken) == look.clutter_n:
            break
        group = GROUPS[name]
        box = _union([footprint(k, ix, iy, yaw) for k, ix, iy, yaw in group.items])
        w, d = box[2] - box[0], box[3] - box[1]
        n = int((tx - 2 * EDGE - w) / STEP) + 1
        spots = [(round(-tx / 2 + EDGE + w / 2 + i * STEP, 4), back) for i in range(n) for back in (1, -1)]
        rng.shuffle(spots)
        for x, back in spots:
            y = round(back * (ty / 2 - BACK - d / 2), 4)
            props = _set_out(group, x, y, back)
            outline = _union([footprint(*p) for p in props])
            if all(spot_ok(scene, k, px, py, yaw) for k, px, py, yaw in props) and \
                    not any(overlap(outline, t, GROUP_GAP) for t in taken):
                taken.append(outline)
                z = {k: round(scene.table_top_z + PROPS[k].size[2] / 2, 4) for k, *_ in props}
                out += [PropSpec(f"prop_{len(out) + i}", k, (px, py, z[k]), yaw, PROPS[k].size, PROPS[k].mass_kg)
                        for i, (k, px, py, yaw) in enumerate(props)]
                break
    return tuple(out)
