"""Each job's own failure signal (docs/PROTOCOL.md): what a run shows when one of a
task's jobs was tried and could not be carried through -- a door that, once opened, swung shut with the box still
inside -- as against the program's own faults (a grip that closed on nothing, an arm behind its plan, a clash), which
say nothing about the team (``agent/failure.py``, ``RULE``).

A signal is a temporal formula over a run's frames, named after the jobs of the task's ``jobs(inst)`` it belongs to.
``scoring.scorer.score`` evaluates an instance's signals as it does its milestones; a run's report lists those that
showed (``signals_hit``), and the failure report passes them on.  A hold gives a signal only once it was *tried* (the
door opened first): a program that never tried a job shows nothing about the team.

They are kept here, not in the tasks' impls, so that declaring them changes no impl (whose source keys its sweeps),
and they read only the instance (its scene, level, metadata) and the impls' constants, never ``jobs(inst)``, so that
scoring a run stays cheap; ``tests/test_signals.py`` checks every name against ``jobs(inst)``.

The eight benchmark tasks have them.  The reference shows none, nor does the borrow configuration's prop; its
distractor shows its job's, and a two-arm attempt with the shortfall team the missing job's.

Where a job fails because no arm left reaches it -- a piece or a sponge carried on from arm to arm -- the thing shows
it where it is left on its way: on a hand-over pad of the missing arm, or in a lane short of the next arm's reach, and
never taken on (``LeftAt``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from ..scoring.predicates import (Eventually, FallenAt, HoldsFor, InBox, InRegion, LeftAt, Level, Lifted, Opened,
                                  Temporal, Then, ThroughSlots, Unless, any_of)

if TYPE_CHECKING:
    from .instance import TaskInstance

#: a hinged body back within this of shut, once it had been opened past it (as the distractor runs' safe stops)
SHUT_DEG = 10.0
#: how long a failure must last to count: a frame or two of contact does not
HOLD_S = 0.5
#: a body this far below where it started is off the table
FELL_M = 0.10
#: a domino run that has not reached the far side of a gap this long after the near side fell has stopped there
STOPPED_S = 2.0
#: a thing set down on its way and left there to the run's end, at least this long, was never taken on
LEFT_S = 3.0


@dataclass(frozen=True)
class Signal:
    #: the jobs whose failure it shows (names from the task's ``jobs(inst)``)
    jobs: tuple[str, ...]
    #: what the run showed
    what: str
    formula: Temporal
    #: a short name for several jobs (the name is otherwise the jobs')
    label: str = ""

    @property
    def name(self) -> str:
        return f"{self.label or ' / '.join(self.jobs)}: {self.what}"


def _con01(inst: "TaskInstance") -> tuple[Signal, ...]:
    inside = InRegion("box", "cabinet_in")
    return tuple(Signal((f"hold the {side} door open by its knob",), "it swung shut with the box still inside",
                        Then(Opened(door, SHUT_DEG), ~Opened(door, SHUT_DEG) & inside, HOLD_S))
                 for side, door in (("west", "door_w"), ("east", "door_e")))


def _con02(inst: "TaskInstance") -> tuple[Signal, ...]:
    from .impl import con02_hinged_leaves as M

    slots = M.slots(inst.metadata["slot_clearance_m"])
    through = ThroughSlots("pin", M.KEY[1], (M.KEY[0], M.KEY[2]), slots)
    # the pin in a slot of either leaf: it was being pushed home (``pin_in_one`` is the orange leaf's first)
    in_one = any_of(*(ThroughSlots("pin", M.KEY[1], (M.KEY[0], M.KEY[2]), (s,)) for s in slots))
    out = []
    for colour, leaf in (("blue", "leaf_w"), ("orange", "leaf_e")):
        job = (f"hold the {colour} leaf level by its bar",)
        out.append(Signal(job, "once level, it dropped before the pin was through",
                          Then(Level(leaf, M.LEVEL_DEG), ~Level(leaf, 2 * M.LEVEL_DEG) & ~through, HOLD_S)))
        # never held level (nothing held it, or a prop lifted it short): the pin pushed in stops at its slot
        out.append(Signal(job, "the pin was pushed in while it hung out of line",
                          Then(in_one, ~Level(leaf, M.LEVEL_DEG) & ~through, HOLD_S)))
    return tuple(out)


def _con03(inst: "TaskInstance") -> tuple[Signal, ...]:
    from .impl import con03_lid_ball as M

    in_cradle = InRegion("ball", "cradle_in")
    out = Lifted("ball", M.WALL_H) | in_cradle
    sides = ("lift the ball by its west side into the cradle", "lift the ball by its east side into the cradle")
    return (Signal(("hold the lid open by its tab",), "it fell shut with the ball still in the box",
                   Then(Opened("lid", SHUT_DEG), ~Opened("lid", SHUT_DEG) & ~out, HOLD_S)),
            Signal(sides, "once lifted over the wall, the ball fell back outside the cradle",
                   Then(Lifted("ball", M.WALL_H), ~Lifted("ball", 0.05) & ~in_cradle, HOLD_S),
                   label="lifting the ball (both sides)"))


def _con04(inst: "TaskInstance") -> tuple[Signal, ...]:
    """The pipe pressure test: a bleed valve, once shut, let open while air came in, or the pedal, once
    down, let up while both valves were shut -- before the pipe held its pressure for its test.  Borrow version: a
    weight set in the pocket left the pedal up while both valves were shut (the distractor)."""
    from .impl import con04_pressure_test as M

    shut = {v: Opened(v, M.VALVE_SHUT) for v in ("valve_w", "valve_e")}
    both, air = shut["valve_w"] & shut["valve_e"], Opened("pedal", M.AIR_DEG)
    held = HoldsFor(both & air, 1.0)  # the task's success
    pedal = "let air in: hold the foot valve's pedal down by its toe bar"
    out = [Signal((f"shut the {side} bleed valve: hold its lever down by its grip",),
                  "once shut, it opened again while air came in, before the pipe held its pressure",
                  Unless(Then(shut[v], ~shut[v] & air, HOLD_S), held))
           for side, v in (("west", "valve_w"), ("east", "valve_e"))]
    out.append(Signal((pedal,), "once down, it rose and shut while both bleed valves were shut, before the pipe held "
                                "its pressure", Unless(Then(air, ~air & both, HOLD_S), held)))
    b = inst.metadata.get("borrow")
    if b:
        mid = M.pocket(inst, 0.0, 0.03)  # a weight's middle set in the pocket, the pedal up
        half = (M.INNER / 2, M.INNER / 2, 0.04)
        in_pocket = any_of(*(InBox(n, mid, half) for n in (*b.get("props", ()), *b.get("decoys", ()))))
        out.append(Signal((pedal,), "a weight set in its pocket left it up while both bleed valves were shut",
                          Unless(Then(in_pocket, in_pocket & ~air & both, HOLD_S), held)))
    return tuple(out)


def _rea01(inst: "TaskInstance") -> tuple[Signal, ...]:
    from .impl import rea01_relay as M

    # the lane through the partition's covered channel: as wide as the channel inside and no higher than its roof, from
    # over one hand-over pad to over the other -- where the board carries a piece from the west arm's reach to the
    # east one's (a piece left short of the channel's mouth is in it too: in Isaac the short board's stopped 5 cm short)
    far = max(abs(inst.scene.region(p).center[0]) + inst.scene.region(p).half[0] for p in ("pad_a", "pad_b"))
    lane = ((0.0, M.LANE_Y, M.Z + M.TUNNEL_IN[1] / 2), (far, M.TUNNEL_IN[0] / 2, M.TUNNEL_IN[1] / 2))
    out = []
    for piece in inst.metadata["fruit"]:
        job = (f"carry {piece} from the start area into the crate",)
        out.append(Signal(job, "it fell off the table", Eventually(~Lifted(piece, -FELL_M) & ~InRegion(piece, "bin_in"))))
        on_way = InRegion(piece, "pad_a") | InRegion(piece, "pad_b") | InBox(piece, *lane)
        out.append(Signal(job, "it was left on its way, on a hand-over pad or in the lane through the covered channel",
                          LeftAt(piece, on_way, LEFT_S)))
    return tuple(out)


def _rea02(inst: "TaskInstance") -> tuple[Signal, ...]:
    from .impl import rea02_tomato_flume as M

    # the negation of the success's own rule: the tomato always in the pot, over it, or no lower than the lowest
    # channel
    on_track = (InRegion("tomato", "pot_in") | InRegion("tomato", "pot_over")
                | Lifted("tomato", M.LOW - M.SHELF_TOP - M.TOMATO_R))
    lays = ("lay the lowest channel", "lay the middle channel", "lay the upper channel")
    return (Signal(lays, "the tomato fell off the track", Eventually(~on_track), label="laying the track"),)


def _rea03(inst: "TaskInstance") -> tuple[Signal, ...]:
    from .impl import rea03_domino_run as M

    spacing = inst.metadata["spacing_m"]
    names = tuple(o.name for o in inst.scene.objects if o.name.startswith(("domino_", "plank_")))

    def fell(i: int):
        return FallenAt(names, (M.place_x(i, spacing), M.LINE_Y, M.Z), (spacing / 2, M.FIRST_ACROSS, M.FIRST_UP),
                        M.DOWN_DEG)

    # the middle stretch: places 12-14 (from 1) are empty; in the borrow version 11-14, a plank stood in 11
    first_gap = 11 if inst.variant == "borrow" else 12
    jobs = tuple(f"fill place {k}" for k in range(first_gap, 15))
    near, far = first_gap - 2, 14  # (from 0) the domino before the stretch and the one after it
    return (Signal(jobs, "the run reached the middle stretch and stopped there",
                   Then(fell(near), ~fell(far), STOPPED_S), label="filling the middle stretch"),)


def _rea04(inst: "TaskInstance") -> tuple[Signal, ...]:
    carries = ("wipe the west spill with the sponge from its dish",
               "bring the sponge on from the west spill and wipe the middle one",
               "bring the sponge on from the middle spill and wipe the east one",
               "bring the sponge on from the east spill into the sink")
    on_pad = InRegion("sponge", "pad_a") | InRegion("sponge", "pad_b")
    return (Signal(carries, "the sponge fell off the counter",
                   Eventually(~Lifted("sponge", -FELL_M) & ~InRegion("sponge", "sink_in")),
                   label="passing the sponge on"),
            Signal(carries, "it was left on a hand-over pad, never taken on", LeftAt("sponge", on_pad, LEFT_S),
                   label="passing the sponge on"))


#: task id -> its signals, for each of the eight benchmark tasks
REGISTRY: dict[str, Callable[["TaskInstance"], tuple[Signal, ...]]] = {
    "Con-01": _con01, "Con-02": _con02, "Con-03": _con03, "Con-04": _con04,
    "Rea-01": _rea01, "Rea-02": _rea02, "Rea-03": _rea03, "Rea-04": _rea04,
}


def declared(inst: "TaskInstance") -> bool:
    """Whether the instance's task declares failure signals."""
    return inst.spec.id in REGISTRY


def of(inst: "TaskInstance") -> tuple[Signal, ...]:
    """The instance's failure signals (none for a task that declares none)."""
    f = REGISTRY.get(inst.spec.id)
    return f(inst) if f else ()
