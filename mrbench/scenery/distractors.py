"""Distractors: everyday things within an arm's reach that the task does not need (``Look.distract``).

They are bodies of the task's scene (``ObjectSpec``): the team sees them, is told what each is and about what it
weighs, and may move them -- unlike a look's clutter, which stands where no arm reaches.  Each stands within the
nominal reach (``REACH`` of it) of at least one arm, ``SWEEP_GAP`` outside the cells the task's scripted reference
sweeps (``sweep``: planned for each built version and level), ``TASK_GAP`` clear of the task's bodies, regions and
cloths, and ``APART`` from the others; turned any way.

A distractor must not do an arm's work, or the task's answer would change.  What a prop does instead of an arm is
each task's borrow design (the card's ``borrow``), and the kinds here are small and light against them: none is long
(the longest, a book, 22 cm, against a 0.6 m stay, a 1.3 m channel, a 30 cm plank, a 45 cm mop handle) or heavy
(at most 0.45 kg, against a 3 kg weight).  Where a small thing could still do it, ``ALLOWED`` keeps only the kinds
that cannot, and says why.  Engine-independent.
"""

from __future__ import annotations

import dataclasses
import math
import random
from dataclasses import dataclass
from typing import Optional

from ..core.errors import ConfigError
from ..engines.base import ObjectSpec, Part, RGB

#: a distractor stands within this share of an arm's nominal reach from its base; this far outside the cells the
#: reference sweeps, clear of the task's things, from the table's edge and from the other distractors (m)
REACH, SWEEP_GAP, TASK_GAP, EDGE, APART = 0.9, 0.04, 0.08, 0.05, 0.06
#: candidate places are tried on a grid this fine (m)
GRID = 0.02


@dataclass(frozen=True)
class Kind:
    #: what the team is told it is
    what: str
    #: "box" (x, y, z) or "cylinder" (radius, height), or parts
    shape: str
    size: tuple[float, ...]
    mass_kg: float
    colours: tuple[RGB, ...]
    parts: tuple[Part, ...] = ()

    @property
    def height(self) -> float:
        return self.size[1] if self.shape == "cylinder" else self.size[2]

    @property
    def radius(self) -> float:
        """Its footprint, however it is turned, lies inside this circle."""
        if self.shape == "cylinder":
            return self.size[0] + (0.025 if self.parts else 0.0)  # (a mug's handle)
        return math.hypot(self.size[0], self.size[1]) / 2


_PAPER = ((0.85, 0.82, 0.74), (0.62, 0.15, 0.12), (0.18, 0.32, 0.55), (0.25, 0.45, 0.30))
_GLAZE = ((0.93, 0.93, 0.90), (0.75, 0.20, 0.18), (0.20, 0.35, 0.65), (0.95, 0.75, 0.25))
KINDS: dict[str, Kind] = {
    "book": Kind("a paperback book, lying flat", "box", (0.15, 0.22, 0.03), 0.45, _PAPER),
    "pen": Kind("a pen", "box", (0.14, 0.012, 0.012), 0.015, ((0.10, 0.10, 0.12), (0.15, 0.25, 0.60))),
    "notepad": Kind("a notepad, lying flat", "box", (0.10, 0.15, 0.012), 0.12, _PAPER),
    "cup": Kind("a paper cup, empty", "cylinder", (0.035, 0.09), 0.01, ((0.95, 0.95, 0.95), (0.80, 0.70, 0.55))),
    "mug": Kind("a mug, empty", "cylinder", (0.04, 0.095), 0.30, _GLAZE,
                # (no part may share a body's name: Isaac finds a body by its name anywhere under the scene, so a
                # part called "cup" would stand in for a paper cup on the same table)
                parts=(Part("shell", "cylinder", (0.04, 0.095)),
                       Part("handle", "box", (0.025, 0.012, 0.06), (0.052, 0.0, 0.0)))),
    "can": Kind("a can of soup", "cylinder", (0.033, 0.12), 0.40, ((0.75, 0.15, 0.12), (0.85, 0.80, 0.70))),
    "tea": Kind("a small box of tea, standing", "box", (0.07, 0.12, 0.15), 0.12, ((0.30, 0.50, 0.30), (0.70, 0.55, 0.25))),
}

#: per task, the kinds it may have (the rest: every kind), and why
ALLOWED: dict[str, tuple[tuple[str, ...], str]] = {
    "Con-01": (("book", "pen", "notepad"),
               "a thing in a door lever's way back holds the door (the borrow version's weight); the lever's bar is "
               "5 cm up and passes over these (the borrow distractor, 4 cm, it passes over)"),
    "Con-02": (("book", "pen", "notepad"),
               "a thing under the blue leaf's bar holds it level 20 cm up (the borrow version's wedge); these are "
               "flat, and all three stacked are under 6 cm"),
}


def allowed(task: str) -> tuple[str, ...]:
    return ALLOWED.get(task, (tuple(KINDS), ""))[0]


def _body(name: str, kind: Kind, xy, yaw: float, colour: RGB, z0: float) -> ObjectSpec:
    if kind.parts:
        parts = tuple(dataclasses.replace(p, pos=(p.pos[0], p.pos[1], p.pos[2] + kind.height / 2)) for p in kind.parts)
        return ObjectSpec(name, "compound", (2 * kind.radius, 2 * kind.radius, kind.height),
                          (xy[0], xy[1], z0), yaw, colour, kind.mass_kg, parts=parts)
    return ObjectSpec(name, kind.shape, kind.size, (xy[0], xy[1], z0 + kind.height / 2), yaw, colour, kind.mass_kg)


def _clear_of_task(scene, x: float, y: float, r: float) -> bool:
    """Whether a footprint of radius ``r`` at (x, y) is TASK_GAP clear of every body, region and cloth."""
    for o in scene.objects:
        half = max(o.size[0], o.size[1]) / 2 if o.shape != "sphere" else o.size[0]
        if o.shape == "cylinder":
            half = o.size[0]
        if math.hypot(x - o.pos[0], y - o.pos[1]) < r + half + TASK_GAP:
            return False
    for g in scene.regions:
        if abs(x - g.center[0]) < g.half[0] + r + TASK_GAP and abs(y - g.center[1]) < g.half[1] + r + TASK_GAP:
            return False
    for c in scene.cloths:
        for px, py, _ in c.flat_points():
            if math.hypot(x - px, y - py) < r + TASK_GAP:
                return False
    for p in scene.props:
        if math.hypot(x - p.pos[0], y - p.pos[1]) < r + 0.1 + TASK_GAP:
            return False
    return True


def place(inst, n: int, rng: random.Random):
    """``inst`` with ``n`` distractors added (fewer if the table has no more room); ``metadata["distractors"]`` names
    each and says what it is."""
    from ..robots import get_model
    from . import sweep

    if n <= 0:
        return inst
    task = inst.spec.id
    swept = sweep.cells(task)
    if not swept:
        raise ConfigError(f"{task} has no sweeps (python -m mrbench.scenery.sweep {task}): distractors need them")
    near = int(math.ceil(SWEEP_GAP / sweep.CELL))
    sc = inst.scene
    tx, ty = sc.table_size[0] / 2, sc.table_size[1] / 2
    reach = [(a.pos[0], a.pos[1], REACH * get_model(a.model).reach_m) for a in sc.arms]
    kinds = list(allowed(task))
    added, names, shown = [], {}, {}
    for _ in range(n):
        kind_name = rng.choice(kinds)
        kind = KINDS[kind_name]
        r = kind.radius
        spots = []
        for i in range(int(2 * tx / GRID) + 1):
            for j in range(int(2 * ty / GRID) + 1):
                x, y = -tx + i * GRID, -ty + j * GRID
                if abs(x) > tx - EDGE - r or abs(y) > ty - EDGE - r:
                    continue
                if not any(math.hypot(x - bx, y - by) <= rr for bx, by, rr in reach):
                    continue
                ci, cj = sweep.cell_of(x, y)
                k = near + int(math.ceil(r / sweep.CELL))
                if any((ci + di, cj + dj) in swept for di in range(-k, k + 1) for dj in range(-k, k + 1)):
                    continue
                if any(math.hypot(x - o.pos[0], y - o.pos[1]) < r + KINDS[w].radius + APART for o, w in added):
                    continue
                if not _clear_of_task(sc, x, y, r):
                    continue
                spots.append((x, y))
        if not spots:
            break
        x, y = rng.choice(spots)
        names[kind_name] = names.get(kind_name, 0) + 1
        name = kind_name if names[kind_name] == 1 else f"{kind_name}_{names[kind_name]}"
        body = _body(name, kind, (x, y), rng.uniform(0.0, 360.0), rng.choice(kind.colours), sc.table_top_z)
        added.append((body, kind_name))
        shown[name] = kind.what
    scene = dataclasses.replace(sc, objects=sc.objects + tuple(b for b, _ in added))
    return dataclasses.replace(inst, scene=scene, metadata={**inst.metadata, "distractors": shown})
