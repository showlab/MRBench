"""Which robots could stand on each mount: a task's reach rules, robot by robot.

The mounts are fixed (``layouts.py``); which robot stands on each is the task's choice,
and a task is meant to be tried with other combinations too.  Some will not do: a robot
that cannot reach what its mount's role needs cannot do the task from there, and one
that reaches further than the role allows may let the task be done without the arm it
was meant to need.  Both follow from the rules a task's impl declares:

* **can do** -- every place the role must reach is within the robot's usable reach
  (``validate.REACH_MARGIN`` of nominal), at both levels and every checked seed;
* **still needs every arm** -- nothing the role must not reach is within the robot's
  nominal reach, and no two arms that must stay apart can meet.

Reach is the nominal horizontal reach from the base axis, as in ``validate``: it says
nothing of payload, force or the height of the work.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product

from ..robots import MODELS, get_model
from .instance import TaskInstance, build
from .loader import get
from .spec import LEVELS
from .validate import CHECK_SEEDS, REACH_MARGIN


@dataclass(frozen=True)
class Fit:
    can_do: bool
    needs_others: bool
    #: the farthest place this mount's role must reach (m), over every checked instance
    farthest: float


def _dist(inst: TaskInstance, slot: str, target: str) -> float:
    return math.dist(inst.scene.arm(slot).pos[:2], inst.scene.point(target)[:2])


def _instances(task: str) -> list[TaskInstance]:
    return [build(task, lv, s) for lv in LEVELS for s in CHECK_SEEDS]


def mount_fits(task: str, insts=None) -> dict[str, dict[str, Fit]]:
    """mount -> model -> how that robot would do there."""
    insts = insts or _instances(task)
    out: dict[str, dict[str, Fit]] = {}
    for arm in get(task).arms:
        slot = arm.slot
        far = max((_dist(i, s, t) for i in insts for s, t in i.reach if s == slot), default=0.0)
        near = min((_dist(i, s, t) for i in insts for s, t in i.out_of_reach if s == slot), default=math.inf)
        out[slot] = {m: Fit(far <= REACH_MARGIN * get_model(m).reach_m, near > get_model(m).reach_m, round(far, 3))
                     for m in MODELS}
    return out


def combinations(task: str) -> list[tuple[dict[str, str], bool, bool]]:
    """Every assignment of robots to the mounts: (mount -> model, can do it, still needs every arm)."""
    insts = _instances(task)
    fits = mount_fits(task, insts)
    slots = [a.slot for a in get(task).arms]
    apart = {pair for i in insts for pair in i.apart}
    out = []
    for models in product(MODELS, repeat=len(slots)):
        combo = dict(zip(slots, models))
        can = all(fits[s][m].can_do for s, m in combo.items())
        needs = all(fits[s][m].needs_others for s, m in combo.items())
        for a, b in apart:
            d = math.dist(insts[0].scene.arm(a).pos[:2], insts[0].scene.arm(b).pos[:2])
            needs &= d > get_model(combo[a]).reach_m + get_model(combo[b]).reach_m
        out.append((combo, can, needs))
    return out
