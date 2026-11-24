"""Helpers shared by task implementations: primitives on the table, regions, containers."""

from __future__ import annotations

import math
import random
from typing import Iterable, Mapping, Optional

from ...core.errors import ConfigError
from ...engines.base import (ArmPlacement, ClothSpec, FitSpec, HingeSpec, ObjectSpec, Part, Pose7, Quat, RegionSpec,
                             SceneSpec, pose7, quat_rotate)
from ...scoring.predicates import Temporal
from ..instance import Milestone, TaskInstance
from ..layouts import TABLE_SIZE, TABLE_TOP_Z, mount
from ..spec import TaskSpec

Z = TABLE_TOP_Z

COLORS = {
    "red": (0.85, 0.20, 0.20),
    "green": (0.20, 0.70, 0.30),
    "blue": (0.20, 0.40, 0.85),
    "yellow": (0.90, 0.80, 0.20),
    "orange": (0.95, 0.55, 0.15),
    "purple": (0.55, 0.30, 0.75),
    "white": (0.92, 0.92, 0.92),
    "grey": (0.50, 0.50, 0.50),
    "wood": (0.62, 0.46, 0.30),
    "rubber": (0.12, 0.12, 0.12),
    # fruit
    "apple": (0.70, 0.09, 0.07),
    "peel": (0.96, 0.52, 0.06),
    "leaf": (0.27, 0.52, 0.16),
    "stem": (0.36, 0.24, 0.13),
    "coffee": (0.32, 0.19, 0.09),
    # factory materials, for the borrow configurations' props and distractors
    "steel": (0.62, 0.64, 0.67),
    "iron": (0.24, 0.25, 0.27),
    "aluminium": (0.80, 0.82, 0.85),
    "pvc": (0.56, 0.58, 0.61),
    "safety_yellow": (0.96, 0.76, 0.04),
    "safety_orange": (0.96, 0.42, 0.06),
    "nylon": (0.93, 0.91, 0.84),
    # the hand-over places of Rea-01 and Rea-04, flat cork coasters on the counter
    "cork": (0.76, 0.62, 0.44),
    # the head of Rea-04's long-handled sponge brush, a kitchen brush's green plastic
    "plastic_green": (0.16, 0.56, 0.36),
    # Con-04's pressure test: its bleed valves' brass bodies and its gauge's stem
    "brass": (0.78, 0.60, 0.26),
}
PALETTE = ("red", "green", "blue", "yellow", "orange", "purple")


def rng_for(spec: TaskSpec, level: str, seed: int) -> random.Random:
    return random.Random(f"{spec.seed_id}/{level}/{seed}")


def jitter(rng: random.Random, xy: tuple[float, float], r: float = 0.015) -> tuple[float, float]:
    return (xy[0] + rng.uniform(-r, r), xy[1] + rng.uniform(-r, r))


def arms(spec: TaskSpec) -> tuple[ArmPlacement, ...]:
    out = []
    for a in spec.arms:
        m = mount(a.mount)
        out.append(ArmPlacement(slot=a.slot, model=a.model, pos=(m.x, m.y, Z), yaw_deg=m.yaw_deg))
    return tuple(out)


def box(name, xy, size, color="grey", mass=0.05, z=None, yaw=0.0, dynamic=True, glass=False) -> ObjectSpec:
    z = Z + size[2] / 2 if z is None else z
    return ObjectSpec(name, "box", tuple(size), (xy[0], xy[1], z), yaw, COLORS[color], mass, dynamic, glass=glass)


def cylinder(name, xy, radius, height, color="grey", mass=0.05, z=None, dynamic=True) -> ObjectSpec:
    z = Z + height / 2 if z is None else z
    return ObjectSpec(name, "cylinder", (radius, height), (xy[0], xy[1], z), 0.0, COLORS[color], mass, dynamic)


def sphere(name, xy, radius, color="grey", mass=0.02, z=None) -> ObjectSpec:
    z = Z + radius if z is None else z
    return ObjectSpec(name, "sphere", (radius,), (xy[0], xy[1], z), 0.0, COLORS[color], mass, True)


def pad(name, xy, half_xy, color="yellow", hz=0.06) -> RegionSpec:
    """A region resting on the table top: objects up to ``2*hz`` tall count as inside."""
    return RegionSpec(name, (xy[0], xy[1], Z + hz), (half_xy[0], half_xy[1], hz), COLORS[color])


def zone(name, xyz, half, color="purple", visible=False) -> RegionSpec:
    """A region floating above the table (e.g. a handover zone)."""
    return RegionSpec(name, tuple(xyz), tuple(half), COLORS[color], visible)


def container(name, xy, inner, wall_h=0.05, wall_t=0.01, color="wood") -> tuple[list[ObjectSpec], RegionSpec]:
    """An open-top box built from static slabs, plus a region covering its interior."""
    (x, y), (lx, ly) = xy, inner
    base_t = 0.01
    parts = [box(f"{name}_base", (x, y), (lx + 2 * wall_t, ly + 2 * wall_t, base_t), color, dynamic=False)]
    hz = Z + base_t + wall_h / 2
    for side, (dx, dy, sx, sy) in {
        "n": (0, (ly + wall_t) / 2, lx + 2 * wall_t, wall_t), "s": (0, -(ly + wall_t) / 2, lx + 2 * wall_t, wall_t),
        "e": ((lx + wall_t) / 2, 0, wall_t, ly), "w": (-(lx + wall_t) / 2, 0, wall_t, ly),
    }.items():
        parts.append(box(f"{name}_wall_{side}", (x + dx, y + dy), (sx, sy, wall_h), color, z=hz, dynamic=False))
    interior = RegionSpec(f"{name}_in", (x, y, Z + base_t + wall_h / 2), (lx / 2, ly / 2, wall_h / 2), COLORS[color], False)
    return parts, interior


def part(name, shape, size, pos=(0.0, 0.0, 0.0), yaw=0.0, color: Optional[str] = None, friction: Optional[float] = None,
         friction_combine="average") -> Part:
    return Part(name, shape, tuple(size), tuple(pos), yaw, COLORS[color] if color else None, friction, friction_combine)


def beam(name, a, b, thick, z=0.0, color: Optional[str] = None) -> Part:
    """A square bar ``thick`` across from (x, y) ``a`` to ``b`` in the body's x-y plane."""
    (x0, y0), (x1, y1) = a, b
    length = math.hypot(x1 - x0, y1 - y0)
    return part(name, "box", (length, thick, thick), ((x0 + x1) / 2, (y0 + y1) / 2, z),
                math.degrees(math.atan2(y1 - y0, x1 - x0)), color)


def compound(name, pos, parts: Iterable[Part], color="grey", mass=0.1, yaw=0.0, quat: Optional[Quat] = None,
             dynamic=True) -> ObjectSpec:
    """One rigid body made of ``parts``; ``pos`` is where its frame's origin is."""
    parts = tuple(parts)
    lo, hi = [math.inf] * 3, [-math.inf] * 3
    for p in parts:  # the box round the parts, in the body's frame
        hx, hy, hz = (p.size[0], p.size[0], p.size[1] / 2) if p.shape == "cylinder" else \
            (p.size[0],) * 3 if p.shape == "sphere" else (p.size[0] / 2, p.size[1] / 2, p.size[2] / 2)
        c, s = abs(math.cos(math.radians(p.yaw_deg))), abs(math.sin(math.radians(p.yaw_deg)))
        ext = (c * hx + s * hy, s * hx + c * hy, hz)
        lo = [min(l, q - e) for l, q, e in zip(lo, p.pos, ext)]
        hi = [max(h, q + e) for h, q, e in zip(hi, p.pos, ext)]
    size = tuple(h - l for l, h in zip(lo, hi))
    return ObjectSpec(name, "compound", size, tuple(pos), yaw, COLORS[color], mass, dynamic, parts, quat)


def hinge(name, body: ObjectSpec, pivot, sign, **kw) -> HingeSpec:
    """A hinge turning ``body`` about a line through ``pivot`` (``axis=``, vertical by default)."""
    if "axis" in kw:
        kw["axis"] = tuple(kw["axis"])
    return HingeSpec(name, body.name, tuple(pivot), sign, **kw)


def rotate_about(pose: Pose7, pivot, quat: Quat) -> Pose7:
    """A body's pose after turning by ``quat`` about the point ``pivot``."""
    from ...engines.base import quat_mul

    d = quat_rotate(quat, tuple(p - q for p, q in zip(pose[:3], pivot)))
    return (*(q + e for q, e in zip(pivot, d)), *quat_mul(quat, pose[3:]))  # type: ignore[return-value]


def scene(spec: TaskSpec, objects: Iterable[ObjectSpec], regions: Iterable[RegionSpec] = (),
          hinges: Iterable[HingeSpec] = (), cloths: Iterable[ClothSpec] = (),
          fits: Iterable[FitSpec] = ()) -> SceneSpec:
    return SceneSpec(arms=arms(spec), objects=tuple(objects), regions=tuple(regions), table_size=TABLE_SIZE,
                     table_top_z=TABLE_TOP_Z, hinges=tuple(hinges), cloths=tuple(cloths), fits=tuple(fits))


def at(xyz, yaw=0.0, quat: Optional[Quat] = None) -> Pose7:
    return pose7(tuple(xyz), yaw, quat)


def make_instance(spec: TaskSpec, level: str, seed: int, scene: SceneSpec, instruction: str,
                  success: Temporal, milestones: Mapping[str, Temporal], reach=(), out_of_reach=(), apart=(),
                  **metadata) -> TaskInstance:
    if tuple(milestones) != spec.milestones:
        raise ConfigError(f"{spec.id}: milestones {list(milestones)} != card {list(spec.milestones)}")
    return TaskInstance(spec=spec, level=level, seed=seed, scene=scene, instruction=instruction,
                        success=success, milestones=tuple(Milestone(k, v) for k, v in milestones.items()),
                        reach=tuple(reach), out_of_reach=tuple(out_of_reach), apart=tuple(apart),
                        metadata={"knob": spec.knob.field, "knob_value": spec.knob.levels[level], **metadata})
