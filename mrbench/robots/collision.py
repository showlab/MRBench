"""Collision geometry for planning and checking: each arm as capsules, the scene as boxes.

An arm is a chain of capsules between its joint frames (``kinematics.frames``),
a little fatter than its links, plus its hand and fingers and its wrist camera; the
fingers move with the gripper's opening.  The table, static bodies and every part of a moving body
are oriented boxes.  Distances are the gap between surfaces (negative: they
overlap).  Engine-independent (needs numpy).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

from . import kinematics as kin
from .models import get_model


@dataclass(frozen=True)
class Box:
    name: str
    center: "kin.np.ndarray"
    rot: "kin.np.ndarray"  # columns: the box's axes in the world
    half: "kin.np.ndarray"


@dataclass(frozen=True)
class Capsule:
    name: str
    a: "kin.np.ndarray"
    b: "kin.np.ndarray"
    r: float
    #: the base column stands on the table: it does not count against it
    on_table: bool = False


#: per model: (name, from, to, radius) between chain points -- index i is joint i's frame, -1 the
#: ee link's frame, "base" the base origin -- and the hand: (half width across the fingers' travel,
#: how far out along the tool axis its middle is, its radius); fingers: (from, to along the tool axis, radius)
_LINKS = {
    "panda": ([("base", "base", 0, 0.10), ("shoulder", 1, 1, 0.09), ("upper_arm", 1, 2, 0.065),
               ("elbow", 2, 3, 0.075), ("forearm", 3, 4, 0.065), ("wrist", 5, 6, 0.06), ("flange", 6, -1, 0.055)],
              (0.09, 0.055, 0.035), (0.06, 0.112, 0.012)),
    "piper": ([("base", "base", 0, 0.065), ("upper_arm", 1, 2, 0.05), ("forearm", 2, 3, 0.045),
               ("wrist", 3, 5, 0.045), ("flange", 5, -1, 0.04)],
              (0.06, 0.03, 0.035), (0.06, 0.135, 0.012)),
    "so101": ([("base", "base", 0, 0.05), ("upper_arm", 1, 2, 0.035), ("forearm", 2, 3, 0.035),
               ("wrist", 3, -1, 0.035)],
              (0.03, -0.03, 0.03), (-0.04, 0.0, 0.012)),
}


#: a single revolute jaw's length, its pivot to the tool point (the SO-101's: 7.5 cm, from its URDF)
JAW_M = 0.075


def grip_span_m(m) -> float:
    """How far apart a hand's fingers open (m): the model's ``grip_open_m``, or for parallel fingers their joints'
    travel (metres); for a single revolute jaw, whose travel is an angle, how far its tip swings out."""
    if m.grip_open_m is not None:
        return m.grip_open_m
    travel = sum(abs(o - c) for o, c in zip(m.gripper_open, m.gripper_closed))
    return travel if len(m.gripper_joints) > 1 else JAW_M * math.sin(min(travel, math.pi / 2))


def arm_capsules(model: str, q: Sequence[float], opening: float, base: "kin.np.ndarray") -> list[Capsule]:
    """The arm's capsules in the world, its fingers ``opening`` (0..1) apart."""
    np = kin.np
    links, (hand_w, hand_z, hand_r), (f0, f1, fr) = _LINKS[model]
    fr_ = kin.frames(model, q)
    pts = {"base": np.zeros(3), -1: fr_[-1][:3, 3]}
    pts.update({i: f[:3, 3] for i, f in enumerate(fr_[:-1])})
    world = lambda p: base[:3, :3] @ p + base[:3, 3]  # noqa: E731
    out = [Capsule(n, world(pts[i]), world(pts[j]), r, on_table=(i == "base")) for n, i, j, r in links]
    ee = base @ fr_[-1]
    m = get_model(model)
    at = lambda x, y, z: (ee @ np.array([x, y, z, 1.0]))[:3]  # noqa: E731
    out.append(Capsule("hand", at(0, -hand_w, hand_z), at(0, hand_w, hand_z), hand_r))
    w = opening * grip_span_m(m) / 2 + fr
    for s in (-1, 1):
        out.append(Capsule("finger", at(0, s * w, f0), at(0, s * w, f1), fr))
    cam = m.wrist_camera
    if cam is not None:  # its housing as a ball round it, its bracket as a rod out from the hand
        (cx, cy, cz), half = cam.housing
        cz += m.tcp_offset_m
        out.append(Capsule("camera", at(cx, cy, cz), at(cx, cy, cz), max(half)))
        if cam.bracket is not None:
            (bx, by, bz), (hx, hy, hz) = cam.bracket
            bz += m.tcp_offset_m
            out.append(Capsule("camera", at(bx - hx, by, bz), at(bx + hx, by, bz), max(hy, hz)))
    return out


def boxes_of(obj, pose: Sequence[float]) -> list[Box]:
    """A body's shapes as boxes in the world, at ``pose`` (x y z qw qx qy qz)."""
    np = kin.np
    t = kin.pose_matrix(pose)
    shapes = obj.parts or ()
    out = []
    if not shapes:
        out.append(Box(obj.name, t[:3, 3], t[:3, :3], _half(obj.shape, obj.size)))
    for p in shapes:
        local = kin.transform(p.pos, kin.rpy_matrix(0.0, 0.0, math.radians(p.yaw_deg)))
        w = t @ local
        out.append(Box(f"{obj.name}.{p.name}", w[:3, 3], w[:3, :3], _half(p.shape, p.size)))
    return out


def grip_span(obj, pose: Sequence[float], tool: Sequence[float], across: Sequence[float], reach: float):
    """How wide the part of a body that closing fingers would take is, across them: of the body's shapes within
    ``reach`` of the tool point, the narrowest along ``across`` (the fingers' travel) -- a sphere's diameter, a box's
    or cylinder's extent along it.  None if no shape is that near (the fingers close on nothing)."""
    np = kin.np
    a = np.asarray(across, dtype=float)
    a = a / np.linalg.norm(a)
    p = np.asarray(tool, dtype=float)[None, :]
    shapes = [(obj.shape, obj.size)] if not obj.parts else [(q.shape, q.size) for q in obj.parts]
    spans = []
    for b, (shape, size) in zip(boxes_of(obj, pose), shapes):
        if float(point_box(p, b)[0]) > reach:
            continue
        if shape == "sphere":
            spans.append(2.0 * float(size[0]))
        elif shape == "cylinder":
            c = abs(float(b.rot[:, 2] @ a))
            spans.append(2.0 * float(size[0]) * math.sqrt(max(0.0, 1.0 - c * c)) + float(size[1]) * c)
        else:
            spans.append(float(2.0 * np.sum(np.abs(b.rot.T @ a) * b.half)))
    return min(spans) if spans else None


def clip_jaws(obj, pose: Sequence[float]):
    """A tool's clip (``script.Clip``), its body at ``pose``: the point midway between its parts ``jaw_1`` and
    ``jaw_2``, the unit vector from the first to the second, and how far apart their inner faces stand.  None if the
    body has no such parts."""
    np = kin.np
    jaws = {b.name.rsplit(".", 1)[-1]: b for b in boxes_of(obj, pose)}
    if "jaw_1" not in jaws or "jaw_2" not in jaws:
        return None
    a, b = jaws["jaw_1"], jaws["jaw_2"]
    d = b.center - a.center
    n = float(np.linalg.norm(d))
    if not math.isfinite(n) or n <= 1e-12:
        return None
    u = d / n
    reach = lambda box: float(np.sum(np.abs(box.rot.T @ u) * box.half))  # noqa: E731 -- half its extent along u
    opening = n - reach(a) - reach(b)
    if not math.isfinite(opening) or opening <= 0.0:
        return None
    return (a.center + b.center) / 2, u, opening


#: a clip's jaws are rigid: they hold a body only if it fills the opening to within this (m) -- the body no more than
#: this narrower across the jaws than they stand apart.  Rea-04's prop clip (5.8 cm) on its sponge lying flat leaves at
#: most 8 mm (5 cm across, more if turned); 10 mm keeps 2 mm over that, refuses the sponge stood on its side (4.5 cm
#: across, 13 mm loose: 3 mm over) and refuses an 11 cm clip on the sponge turned any way (9.4 cm corner to corner,
#: 15.7 mm loose).
CLIP_LOOSE = 0.010


def clip_takes(tool, tool_pose, target, target_pose, reach: float, tol: float = 0.0,
               loose: float = CLIP_LOOSE) -> tuple[bool, str]:
    """Would ``tool``'s clip take ``target`` (both ObjectSpecs, at their poses): within ``reach`` of the point between
    the jaws, no wider across them than they stand apart (with ``tol``), and no narrower than that by more than
    ``loose`` (rigid jaws that touch neither side hold nothing)?  (ok, why not)."""
    if tool.name == target.name:
        return False, "a tool cannot clip itself"
    jaws = clip_jaws(tool, tool_pose)
    if jaws is None:
        return False, f"{tool.name} has no clip"
    mid, u, opening = jaws
    gap = min(float(point_box(mid[None, :], b)[0]) for b in boxes_of(target, target_pose))
    if gap > reach:
        return False, f"{target.name} is {gap * 100:.1f} cm from {tool.name}'s clip"
    w = grip_span(target, target_pose, mid, u, reach)
    if w is not None and w > opening + tol:
        return False, f"{target.name} is {w * 100:.1f} cm across {tool.name}'s clip, which opens {opening * 100:.1f} cm"
    if w is not None and opening - w > loose:
        return False, (f"{target.name} is {w * 100:.1f} cm across {tool.name}'s clip, which opens {opening * 100:.1f} cm:"
                       f" the jaws touch neither side")
    return True, ""


def squeezed(centre: Sequence[float], tool: Sequence[float], others: Sequence[Sequence[float]],
             max_cos: float = -0.5) -> bool:
    """Is a body taken between two hands -- is another hand's tool point (``others``, already known to be on the
    body) on its opposite side from ``tool``, as seen from its ``centre`` (the angle between the two at least 120
    degrees)?  A body too wide for one hand's fingers is taken only so (``clearance.too_wide``)."""
    np = kin.np
    c = np.asarray(centre, dtype=float)
    a = np.asarray(tool, dtype=float) - c
    if np.linalg.norm(a) < 1e-9:
        return False
    a = a / np.linalg.norm(a)
    for o in others:
        b = np.asarray(o, dtype=float) - c
        if np.linalg.norm(b) > 1e-9 and float(a @ (b / np.linalg.norm(b))) <= max_cos:
            return True
    return False


def prop_box(p, pose: Sequence[float] | None = None) -> Box:
    """A prop (``PropSpec``) as its upright bounding box, standing as it was put down."""
    np = kin.np
    t = kin.pose_matrix(p.pose if pose is None else pose)
    return Box(p.name, t[:3, 3], t[:3, :3], np.array(p.size) / 2)


def _half(shape: str, size) -> "kin.np.ndarray":
    np = kin.np
    if shape in ("box", "compound"):
        return np.array(size) / 2
    if shape == "cylinder":
        return np.array([size[0], size[0], size[1] / 2])
    return np.array([size[0]] * 3)


def table_box(scene) -> Box:
    np = kin.np
    tx, ty, tz = scene.table_size
    return Box("table", np.array([0.0, 0.0, scene.table_top_z - tz / 2]), np.eye(3), np.array([tx / 2, ty / 2, tz / 2]))


# ---------------------------------------------------------------------------------- distances
def point_box(p: "kin.np.ndarray", b: Box) -> "kin.np.ndarray":
    """Signed distance from points (n, 3) to a box: negative inside."""
    np = kin.np
    local = (p - b.center) @ b.rot
    q = np.abs(local) - b.half
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.max(q, axis=-1), 0.0)
    return outside + inside


@lru_cache(maxsize=256)
def _unit_line(n):
    points = kin.np.linspace(0.0, 1.0, n)[:, None]
    points.setflags(write=False)
    return points


def _samples(a, b, step: float = 0.01):
    np = kin.np
    n = max(2, int(math.ceil(np.linalg.norm(b - a) / step)) + 1)
    return a + (b - a) * _unit_line(n)


def capsule_box(c: Capsule, b: Box) -> float:
    return float(point_box(_samples(c.a, c.b), b).min() - c.r)


def capsule_capsule(c: Capsule, d: Capsule) -> float:
    """Gap between two capsules (segment-to-segment distance minus both radii)."""
    np = kin.np
    u, v, w = c.b - c.a, d.b - d.a, c.a - d.a
    a, b_, cc, dd, e = u @ u, u @ v, v @ v, u @ w, v @ w
    den = a * cc - b_ * b_
    s = 0.0 if den < 1e-12 else min(max((b_ * e - cc * dd) / den, 0.0), 1.0)
    t = (b_ * s + e) / cc if cc > 1e-12 else 0.0
    if t < 0.0 or t > 1.0:
        t = min(max(t, 0.0), 1.0)
        s = min(max((b_ * t - dd) / a, 0.0), 1.0) if a > 1e-12 else 0.0
    return float(np.linalg.norm((c.a + s * u) - (d.a + t * v)) - c.r - d.r)


@lru_cache(maxsize=16)
def _unit_surface(n):
    np = kin.np
    k = np.linspace(-1.0, 1.0, n)
    g = np.array([(x, y, z) for x in k for y in k for z in k if max(abs(x), abs(y), abs(z)) == 1.0])
    g.setflags(write=False)
    return g


def box_points(b: Box, n: int = 5) -> "kin.np.ndarray":
    """Points over a box's surface, ``n`` a side, in the world (a thin plate crossing an edge is caught)."""
    # Cache only the immutable unit grid, never a moving body's world points.
    return b.center + (_unit_surface(n) * b.half) @ b.rot.T


def box_box(a: Box, b: Box) -> float:
    """Gap between two boxes, approximately: from each one's sample points to the other."""
    return float(min(point_box(box_points(a), b).min(), point_box(box_points(b), a).min()))
