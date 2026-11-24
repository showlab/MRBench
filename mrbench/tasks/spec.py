"""Frozen task specifications, parsed from the YAML cards in ``tasks/cards``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Union

LEVELS = ("easy", "hard")
#: planned   -- card only
#: scaffold  -- scene + success formula + state-level oracle, checked on the mock engine
#: implemented -- also validated in Isaac Sim with real assets and a reference policy
STATUSES = ("planned", "scaffold", "implemented")
#: Every task must put at least this many arms in the scene.
MIN_ARMS_PER_TASK = 3

#: The suites, by why one arm is not enough:
#:
#: * ``concurrency`` (Concurrency-bound) -- the task needs several hands at once: something has to be held still
#:   (a door against its spring, a strip that would lift) while another hand works;
#: * ``reach`` (Reach-bound) -- the task spreads over more of the table than any one arm reaches: things go from
#:   one arm to the next, or each arm works where only it reaches.
#:
#: ``extension`` and ``composition`` are accepted for task cards outside the benchmark (``BENCHMARK_SUITES``), such
#: as a plugin's.
SUITES = ("concurrency", "reach", "extension", "composition")
BENCHMARK_SUITES = ("concurrency", "reach")

#: The versions of a task: the same goal, another team on the table.  ``exact`` is the standard one -- the card's
#: arms, exactly a minimal support set.  The others perturb the team, and each changes the right answer:
#:
#: * ``borrow`` -- one arm fewer, plus a prop that can do that arm's job and a distractor that looks as if it
#:   could but cannot: use the prop, with one arm fewer;
#: * ``surplus`` -- one arm more (on a fourth place): leave one arm idle;
#: * ``shortfall`` -- a key arm taken away: say the task cannot be done, and which job has no doer;
#: * ``swapped`` -- one arm of the exact team replaced by another robot on the same place, chosen so that the answer
#:   changes: it cannot be done though every place is staffed (the new robot cannot do its job), or fewer arms do it.
#:
#: The right answer of each version is the solver's over the task's declared jobs (``tasks/jobs.py``, docs/TASKS.md).
VARIANTS = ("exact", "borrow", "surplus", "shortfall", "swapped")

#: How much the agent is told: ``given`` -- the arms' capabilities and the scene's facts; ``query`` -- the scene's
#: facts, and the arms only by name and place: it asks for their capabilities.
INFO_LEVELS = ("given", "query")

Number = Union[int, float]


@dataclass(frozen=True)
class Knob:
    """The single quantity that separates the easy and hard level of a task."""

    field: str
    description: str
    levels: Mapping[str, Number]
    unit: str


@dataclass(frozen=True)
class ArmSlot:
    mount: str  # one of tasks.layouts.MOUNTS; also the arm's name in the scene
    model: str  # key of mrbench.robots.MODELS: the robot this task puts there
    #: what this arm does in the reference solution: a hidden label for the docs and the scripted reference,
    #: never shown to an agent (its instruction gives only the goal)
    role: str

    @property
    def slot(self) -> str:
        """The arm's name in the scene: its mount, whichever robot stands there."""
        return self.mount


@dataclass(frozen=True)
class Borrow:
    """How the task's ``borrow`` version takes an arm away: the job a prop on the table does instead of it, the
    prop, and the distractor that looks as if it could do the job but cannot; ``remove``, the mount of the arm taken
    away (needed once the version is built)."""

    replaces: str
    prop: str
    distractor: str
    remove: Optional[str] = None


@dataclass(frozen=True)
class TaskSpec:
    id: str
    name: str
    suite: str
    summary: str
    arms: tuple[ArmSlot, ...]
    knob: Knob
    horizon_s: float
    success: str  # prose; the formula lives in the impl module
    milestones: tuple[str, ...]  # ids, must match the impl's milestone formulas
    status: str
    impl: str  # dotted module path exposing build(level, seed) and keyframes(inst)
    todo: tuple[str, ...] = ()
    #: how the task's scene looks unless told otherwise: text for ``scenery.parse_look`` (every card's: a workshop)
    look: str = "default"
    #: what an agent is told about the scene that it cannot see (a door's spring, a plug's tight fit)
    facts: tuple[str, ...] = ()
    #: the borrow version's design (every benchmark task has one)
    borrow: Optional[Borrow] = None
    #: the configurations the card builds (``VARIANTS``)
    variants: tuple[str, ...] = ("exact",)
    #: the id the task's random streams are seeded with, where it differs from ``id`` (``seed_id``)
    was: Optional[str] = None
    #: a version's own time budget where it differs from ``horizon_s`` (version, seconds): a borrow version, whose
    #: team also fetches and sets its prop, may get longer
    horizons: tuple[tuple[str, float], ...] = ()

    def horizon(self, variant: str = "exact") -> float:
        """The time budget (s) of the task's ``variant`` version: its own (``horizons``), else ``horizon_s``."""
        return dict(self.horizons).get(variant, self.horizon_s)

    @property
    def seed_id(self) -> str:
        """What the task's random streams (jitter, props, an agent's rough masses) are seeded with: ``was`` if the
        card gives one, else the task's id."""
        return self.was or self.id

    @property
    def models(self) -> tuple[str, ...]:
        return tuple(a.model for a in self.arms)
