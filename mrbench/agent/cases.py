"""The agent's test cases: one task at one level and seed, its team changed and heavy blocks put down.

* ``exact`` -- the task as it is;
* ``surplus`` -- a fourth arm, a Franka, on the north side;
* ``shortfall`` -- one arm taken away;
* ``borrow`` -- one arm taken away and, for the eight, the task's own borrow version (its prop and its distractor;
  ``tasks.build(..., variant="borrow")``); for a task without one, two heavy steel blocks on the table
  (``TaskSetup.blocks``);
* ``spare`` -- every arm there, and the two blocks too;
* ``swapped`` -- the eight: one arm of the exact team replaced by another robot on the same place (``TaskSetup.swap``).

``Expect`` is what a right answer says; it is what ``grade`` compares a submission with.  For the benchmark's
eight it comes from the solver over the tasks' declared jobs (``tasks/jobs.py``, written to ``gold.json`` by
``python -m mrbench.tasks.jobs gold``), with the ``why`` written here; a plugin's tasks give their own.
"""

from __future__ import annotations

import dataclasses
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Union

from ..engines.base import ArmPlacement, ObjectSpec
from ..scenery.paint import repaint_text
from ..tasks import TaskInstance, build, get
from ..tasks.layouts import TABLE_TOP_Z

VARIANTS = ("exact", "surplus", "shortfall", "borrow", "spare", "swapped")
#: the versions whose gold the solver gives (the benchmark's eight), and where it keeps it
GOLD_VARIANTS = ("exact", "surplus", "shortfall", "swapped", "borrow")
GOLD = Path(__file__).with_name("gold.json")
#: the surplus arm: on the north side, facing south, 35 cm in from the table's edge -- nearer the middle than the
#: south arm, so that it reaches into the middle of the table
EXTRA_ARM = ArmPlacement("north", "panda", (0.0, 0.55, TABLE_TOP_Z), -90.0)


#: a heavy block: steel, 10 x 6 x 5 cm (2.4 kg: 23 N)
BLOCK, BLOCK_MASS, STEEL = (0.10, 0.06, 0.05), 2.4, (0.36, 0.37, 0.40)
BLOCK_TEXT = "steel block, 10 x 6 x 5 cm"


@dataclass(frozen=True)
class Expect:
    feasible: bool
    #: every acceptable set of arms that does it with the fewest arms (empty when it cannot be done)
    minimal: tuple[frozenset, ...] = ()
    #: whether the fewest arms need the blocks (None: no blocks in the scene)
    blocks: Optional[bool] = None
    #: what an answer that says it cannot be done should name as missing
    missing: str = ""
    why: str = ""
    #: answers that work but use more than the fewest (graded apart: right, not minimal)
    workable: tuple[frozenset, ...] = ()
    #: the things on the table the fewest arms use, and those they must leave alone (the eight's borrow versions:
    #: the prop and the distractor, by name)
    things: tuple[str, ...] = ()
    rejected: tuple[str, ...] = ()
    #: the solver's: what is missing at each level ((level, missing), ...), when a level has jobs of its own
    missing_by_level: tuple = field(default=(), compare=False)


@dataclass(frozen=True)
class TaskSetup:
    #: the arm the shortfall and borrow versions take away
    remove: str
    #: where the blocks start: (x, y, yaw deg) -- for the versions with the two steel blocks (borrow, spare)
    blocks: tuple = ()
    #: what each body shown is, as the team would recognise it; bodies not named are not shown.  A body may be a
    #: cloth.  Or a function of the task's instance giving them (many like bodies: the dominoes)
    bodies: Union[dict, Callable] = field(default_factory=dict)
    #: static parts shown as one thing: name -> (what it is, its parts)
    groups: dict[str, tuple[str, tuple[str, ...]]] = field(default_factory=dict)
    #: marked places on the table (painted, visible): name -> what it is
    places: dict[str, str] = field(default_factory=dict)
    #: what a person would tell the team about what does not show (springs, a tight fit); None: the card's facts
    notes: Optional[tuple[str, ...]] = ()
    expect: dict[str, Expect] = field(default_factory=dict)
    #: the parts of a body the cameras pick out (``locate``'s "thing:part"): shown name -> the body's parts it covers
    parts: dict = field(default_factory=dict)
    #: a hinged body whose 0 deg stop is not "shut" in the task's words: body -> (what it is at 0 deg, what it is at its
    #: far stop).  The physics tools count a hinge's turn from 0 deg and call that way "shut"; Con-04's bleed valve is
    #: open there (its lever up), so without this "its spring turns it shut" would say the opposite of the briefing
    hinge_ends: dict = field(default_factory=dict)
    #: where the surplus version's fourth arm stands, if not at ``EXTRA_ARM``'s place (where it would hit the scene)
    extra: Optional[ArmPlacement] = None
    #: the swapped version: (the place, the robot put there instead of the card's)
    swap: Optional[tuple[str, str]] = None
    #: what the borrow version's own things are (the eight's: its prop and its distractor, described alike)
    borrow_bodies: dict[str, str] = field(default_factory=dict)


def _sets(*names: str) -> tuple[frozenset, ...]:
    return tuple(frozenset(n.split("+")) for n in names)


def _gold(task: str, why: dict[str, str]) -> dict[str, Expect]:
    """``task``'s answers from the solver (``GOLD``), each with its ``why``: version -> Expect.  A thing is kept by its
    part in the borrow design ("prop", "decoy"), and what is missing by level; ``Case.expect`` names them for its
    scene."""
    kept = json.loads(GOLD.read_text()).get(task, {}) if GOLD.exists() else {}
    out = {}
    for v, g in kept.items():
        by_level = tuple((lv, "; ".join(m)) for lv, m in sorted(g["missing"].items()))
        out[v] = Expect(g["feasible"], tuple(frozenset(s) for s in g["minimal"]), None, dict(by_level).get("easy", ""),
                        why.get(v, ""), tuple(frozenset(s) for s in g["workable"]), tuple(g["things"]),
                        tuple(g["rejected"]), by_level)
    return out


SETUPS: dict[str, TaskSetup] = {
    "Con-01": TaskSetup(
        remove="east",
        swap=("south", "piper"),
        extra=ArmPlacement("north", "panda", (0.0, 0.7, TABLE_TOP_Z), -90.0),  # clear of the scene
        bodies={"door_w": "west door of the cabinet, opening inward; worked by a low steel lever outside the cabinet's "
                          "west side: a bar with a stub and a T-handle (its stem the knob)",
                "door_e": "east door of the cabinet, opening inward; worked by a low steel lever outside the cabinet's "
                          "east side: a bar with a stub and a T-handle (its stem the knob)",
                "box": "orange box inside the cabinet, behind the doors, with a T-handle on top whose post runs in a "
                       "slot in the ceiling"},
        groups={"cabinet": ("cabinet fixed to the table, its doorway facing south: in front a low door zone under a "
                            "fixed ceiling, behind it a taller closed zone with a slot over the ceiling",
                            ("plinth", "cab_back", "cab_roof", "side_low_w", "side_w", "ceiling_w", "front_w",
                             "side_low_e", "side_e", "ceiling_e", "front_e"))},
        places={"mat": "green mat"},
        parts={"door_w": {"knob": ("knob", "knob_grip"), "lever": ("lever", "stub"), "slab": ("slab",)},
               "door_e": {"knob": ("knob", "knob_grip"), "lever": ("lever", "stub"), "slab": ("slab",)},
               "box": {"handle": ("handle", "post"), "body": ("body",)}},
        notes=None,
        borrow_bodies={n: "cast-iron counterweight, 9 x 9 cm, on a rubber pad, a yellow lifting handle on top" for n in ("block_1",
                                                                                                     "block_2")},
        expect=_gold("Con-01", why={
            "exact": "both doors held open at once while a third hand draws the box out; neither end arm reaches the "
                     "other's knob",
            "surplus": "the north arm, behind the cabinet (moved back to y 0.70, clear of it), reaches neither knob nor "
                       "the handle (IK): it stays idle",
            "shortfall": "two doors held and the box drawn out: three hands at once, two arms -- the south arm reaches "
                         "the east knob, but then no hand draws the box (a run with west and south: the east door "
                         "stayed shut)",
            "swapped": "a PiPER on the south place reaches neither the box's handle in the box zone nor any other way to "
                       "draw it out, and no other arm does: every place staffed, it cannot be done",
            "borrow": "the 9 cm weight, set in the east lever's way back, holds the east door open: the west arm holds "
                      "its own door, the south arm sets the weight and draws the box out; the 4 cm one lets the "
                      "lever's bar pass over it",
        })),
    "Con-02": TaskSetup(
        remove="west",
        # the north arm clear of the guard (at y 0.55 it would stand in it); from here it reaches the orange leaf's bar
        # (IK) and, with the east arm's moves, holds the leaf level while the pin goes in, the east arm idle at home.
        # Facing east: facing south its hand would hang at home 15 cm over that bar (facing east its job plans as facing
        # south)
        extra=ArmPlacement("north", "panda", (0.35, 0.65, TABLE_TOP_Z), 0.0),
        swap=("west", "so101"),
        bodies={"leaf_w": "blue leaf, hinged on its post at its west end, its east end a slotted tongue; a bar to hold "
                          "it by, outside the guard",
                "leaf_e": "orange leaf, hinged on its post at its east end, its west end a slotted fork of two "
                          "cheeks; a bar to hold it by, outside the guard",
                "pin": "grey pin: a flat key with a head, lying on the table"},
        groups={"guard": ("glass guard round the joint, fixed: a slot in each side wall for a leaf, a small opening "
                          "in its south wall", ("guard_w_0", "guard_w_1", "guard_w_2", "guard_w_3", "guard_e_0",
                                                "guard_e_1", "guard_e_2", "guard_e_3", "guard_s_0", "guard_s_1",
                                                "guard_s_2", "guard_s_3", "guard_n", "guard_top")),
                "posts": ("the leaves' two posts, fixed to the table", ("bracket_w", "bracket_e"))},
        parts={"leaf_w": {"bar": ("bar",), "tongue": ("tongue_top", "tongue_bottom", "tongue_a", "tongue_b")},
               "leaf_e": {"bar": ("bar",), "fork": ("cheek_s_top", "cheek_s_bottom", "cheek_s_a", "cheek_s_b",
                                                    "cheek_n_top", "cheek_n_bottom", "cheek_n_a", "cheek_n_b",
                                                    "yoke")},
               "pin": {"key": ("key",), "head": ("head",)}},
        notes=None,
        borrow_bodies={n: "yellow polymer lifting wedge, like a wheel chock, lying on the table south of the blue leaf's bar: a block with a "
                          "ramp toward the north and a black rubber tread on its flat top; a yellow handle rises from its back toward the "
                          "south to a short upright with a rubber T-grip across its top" for n in ("wedge_1", "wedge_2")},
        # surplus: the solver's third team (the east arm sliding the pin, the north arm holding the orange leaf).  The
        # east Franka reaches the pin's head on the table (its hand 30-60 degrees down) and where the pin ends (IK, no
        # margin), and the reference's pin moves given to it plan by IK; the clearance walk finds its hand 0.6-0.9 cm
        # from the guard as the pin goes home (1 cm is the rule) and 2-3 cm from the idle south arm's hand at home
        expect=_gold("Con-02", why={
            "exact": "both leaves held level at once, each by its bar, which only the arm at its end reaches, while a "
                     "third hand slides the pin through",
            "surplus": "the north arm, east of the guard, can hold the orange leaf level in the east arm's place, and "
                       "the east arm can then slide the pin in the south arm's: any three arms with the west one (only "
                       "it reaches the blue leaf's bar); one arm idle",
            "shortfall": "two leaves held and the pin slid in: three hands, two arms; no arm but the west one reaches "
                         "the blue leaf's bar outside the guard",
            "swapped": "an SO-101 on the west place reaches no part of the blue leaf's bar outside the guard, and no "
                       "other arm does: every place staffed, it cannot be done",
            "borrow": "the wedge whose top is 0.8 mm under the level bar, pushed in under it, holds the blue leaf "
                      "level: the east arm holds the orange leaf, the south arm pushes the wedge and slides the pin; "
                      "on the wedge 2 cm lower the leaf rests 4 degrees low",
        })),
    "Con-03": TaskSetup(
        remove="south",
        # EXTRA_ARM's place, facing east: facing south its hand would hang at home where the ball is carried to the
        # cradle
        extra=ArmPlacement("north", "panda", (0.0, 0.55, TABLE_TOP_Z), 0.0),
        swap=("west", "piper"),
        borrow_bodies={"stay_1": "a black steel prop rod with a hook (its tip yellow) and a pivoting foot",
                       "stay_2": "a black steel prop rod with a hook (its tip yellow) and a pivoting foot",
                       "stay_1_socket_w": "a fixed cheek of the first stay's pivot socket",
                       "stay_1_socket_e": "a fixed cheek of the first stay's pivot socket",
                       "stay_2_socket_w": "a fixed cheek of the second stay's pivot socket",
                       "stay_2_socket_e": "a fixed cheek of the second stay's pivot socket"},
        bodies={"lid": "the box's lid, hinged along the box's south edge, with a handle (its bar the tab) near "
                       "its free edge",
                "ball": "orange ball, 22 cm across, inside the box"},
        groups={"box": ("long box fixed to the table (1.0 x 0.44 m), its inner walls lipped at the top",
                        ("box_floor", "wall_w", "wall_e", "wall_s", "wall_n", "lip_s", "lip_n", "lip_w", "lip_e")),
                "cradle": ("cradle on a block north of the box: a block with rails round its top",
                           ("cradle", "rail_w", "rail_e", "rail_s", "rail_n"))},
        parts={"lid": {"tab": ("tab",), "plate": ("plate",)}},
        notes=None,
        # surplus: from EXTRA_ARM's place the north arm reaches the lid's tab only while the
        # lid is open less than about 30 degrees, and with the reference's moves given to it, none of the three jobs
        # plans: it cannot follow the lid's opening once it has the tab, and cannot come over the box beside the ball
        # from either side (the reference planned with each arm's moves given to the north arm).  The solver agrees
        expect=_gold("Con-03", why={
            "exact": "one hand holds the lid open the whole time the ball comes out, and the ball, wider than a hand "
                     "opens, is taken between two hands",
            "surplus": "the north arm can hold the lid only while it is nearly shut, and reaches neither side of the "
                       "ball over the box: it stays idle",
            "shortfall": "the lid held and the ball taken between two hands: three hands at once, two arms (the end "
                         "arms reach the open lid's tab too, but then one hand is left for the ball)",
            "swapped": "a PiPER on the west place reaches neither side of the ball in the box: only the east arm "
                       "lifts it, and two hands must -- every place staffed, it cannot be done",
            "borrow": "the shorter stay, swung over the lid's free edge, holds the lid open: one end arm opens the lid "
                      "by its end, the other swings the stay on its side onto the edge, then both take the ball "
                      "between them; the stay 12 cm longer passes over the edge and catches nothing",
        })),
    "Con-04": TaskSetup(
        remove="south",
        # back from the pipe, as Con-01's: at EXTRA_ARM's y 0.55 it would reach the pedal at full stretch over the
        # pipe, its hand level (IK, 0.85 m); here it reaches either lever's grip, not the pedal
        extra=ArmPlacement("north", "panda", (0.0, 0.70, TABLE_TOP_Z), -90.0),
        swap=("west", "so101"),
        bodies={"valve_w": "the west bleed valve's lever: a steel lever on the pipe's top, hinged on the brass valve "
                           "body, pointing in along the pipe, a red grip on a stem at its end",
                "valve_e": "the east bleed valve's lever: a steel lever on the pipe's top, hinged on the brass valve "
                           "body, pointing in along the pipe, a red grip on a stem at its end",
                "pedal": "the foot valve's yellow pedal, hinged along its north edge, a steel toe bar across its south "
                         "end and a pocket with low steel rails on it"},
        groups={"pipe": ("steel test pipe with blind flanges on two stands, its bleed valves' bodies and its pressure "
                         "gauge, the blue hose from the foot valve and the foot valve's yellow housing, fixed",
                         ("pipe", "pipe_fittings", "gauge", "hose", "pedal_base"))},
        parts={"valve_w": {"grip": ("grip", "grip_stem")}, "valve_e": {"grip": ("grip", "grip_stem")},
               "pedal": {"toe bar": ("toe_bar",), "pocket": ("rail_n", "rail_s", "rail_e", "rail_w")}},
        hinge_ends={**{v: ("its lever is up and the valve is open, bleeding the pipe",
                           "its lever is pressed down and the valve is shut") for v in ("valve_w", "valve_e")},
                    "pedal": ("it is up and the foot valve is shut, letting no air in",
                              "it is pressed down and the foot valve lets air in")},
        notes=None,
        borrow_bodies={n: "steel test weight, 9 cm round, a knob on a neck on top" for n in ("weight_1", "weight_2")},
        expect=_gold("Con-04", why={
            "exact": "both bleed valves held shut and the foot valve's pedal held down at once: three hands; each "
                     "lever's grip is reached only by the arm at its end of the table",
            "surplus": "any three hands, one on each lever and one on the pedal: the north arm reaches either lever's "
                       "grip (not the pedal), and an end arm whose lever it holds reaches the pedal",
            "shortfall": "the two levers and the pedal held down at once: three hands, two arms (the end arms reach "
                         "the pedal too, but their hands are on the levers)",
            "swapped": "an SO-101 on the west place does not reach the west lever's grip, and no other arm does: every "
                       "place staffed, it cannot be done",
            "borrow": "the 2.4 kg weight, set in the pocket on the pedal, holds it down: an end arm sets it, then the "
                      "two hold the levers; under the 1.2 kg one the pedal does not move",
        })),
    "Rea-01": TaskSetup(
        remove="south",
        extra=ArmPlacement("north", "panda", (0.3, 0.55, TABLE_TOP_Z), -90.0),  # clear of the scene
        # (an SO-101 in the middle stands, at home, 3-4 cm into the partition's low part: at the west end instead)
        swap=("west", "so101"),
        bodies={"apple_0": "apple, 5.8 cm across, standing on a small flat foot"},
        borrow_bodies={n: "long wooden paddle, 7 cm wide, lying east-west by the west arm: a square black knob as a "
                          "handle at its west end, a low rim round where a piece stands near its east end"
                       for n in ("board_1", "board_2")},
        groups={"crate": ("open crate fixed to the table", ("bin_base", "bin_wall_n", "bin_wall_s", "bin_wall_e",
                                                             "bin_wall_w")),
                "partition": ("white partition across the middle of the table, fixed: its tall northern part",
                              ("partition",)),
                "partition_low": ("the same partition's low southern part, fixed", ("partition_low",)),
                "tunnel": ("a low pass-through (a guarded channel) through the partition south of its low part, 14 cm wide and 11 cm high "
                           "inside: walls 39 cm long, roofed over its middle 30 cm, fixed",
                           ("tunnel_n", "tunnel_s", "tunnel_roof")),
                "curb": ("an 11 cm curb continuing the partition from the channel to the table's south edge, round "
                         "the west side of the south arm's base, fixed", ("curb_s", "curb_jog", "curb_w"))},
        places={"start": "marked start area", "pad_a": "cork-coloured square A", "pad_b": "cork-coloured square B"},
        notes=None,
        # The end arms share no place: the partition, in two parts 50 and 34 cm high, leaves them no hand-over at the
        # table's middle nor above it.  surplus: checked by IK: the north Franka reaches over the tall partition -- the
        # west arm hands it the apple at (-0.15, 0.25, 1.15) -- and into the crate.  The solver finds it too
        expect=_gold("Rea-01", why={
            "exact": "neither end arm reaches the other's end nor any place in common: the apple passes through the "
                     "middle arm",
            "surplus": "the north arm reaches over the partition: the west arm hands it the apple above the "
                       "partition's west side and it puts it in the crate; south and east idle",
            "shortfall": "the end arms share no place: no arm in the middle for them to hand to",
            "swapped": "an SO-101 at the west end does not reach the start area behind it, and no other arm does: "
                       "every place staffed, it cannot be done",
            "borrow": "the west arm stands each piece on the long board's east end and pushes the board through the "
                      "channel; the east arm takes it off where it comes out; the short board leaves it in the "
                      "channel's middle, out of both end arms' reach",
        })),
    "Rea-02": TaskSetup(
        remove="south",
        swap=("east", "panda"),
        bodies={"channel_1": "wooden channel (a trough) with a handle", "channel_2": "wooden channel (a trough) with a "
                "handle", "channel_3": "wooden channel (a trough) with a handle",
                "stop": "stop at the edge of the shelf's tray, holding the tomato back",
                "tomato": "cherry tomato, 4 cm across, on the shelf's tray"},
        groups={"shelf": ("shelf at the west end, fixed, its top a tray sloping toward the east", ("shelf", "tray")),
                "stand_1": ("stand for a channel, fixed", ("stand_1",)),
                "stand_2": ("stand for a channel, fixed", ("stand_2",)),
                "pot": ("pot at the east end, fixed", ("pot",)),
                "cradle": ("cradle the middle channel lies in, fixed: lips higher than its walls along its north "
                           "side (but by its east end) and across its ends; it comes out only lifted",
                           ("middle_cradle",))},
        parts={f"channel_{k}": {"handle": ("handle",), "floor": ("floor",)} for k in (1, 2, 3)},
        notes=None,
        borrow_bodies={n: "grey moulded HDPE chute (a trough) as wide as the upper channel, a handle near its middle, lying north of "
                          "the west arm" for n in ("channel_4", "channel_5")},
        # surplus: the north arm, at its default place, reaches the lowest channel's handle where it starts but none of
        # the places the channels are laid, nor the stop (IK); it lifts the channel 21 cm, carries it west and then
        # south past the idle east arm's hand, turns it half round in the table's middle and sets it down there, and the
        # south arm takes it and lays it before its own.  The solver agrees
        expect=_gold("Rea-02", why={
            "exact": "each channel starts where only one arm reaches it, and without the middle one the track has a "
                     "gap",
            "surplus": "the north arm can bring the lowest channel to the table's middle, where the south arm takes it "
                       "and lays it: the east or the north arm, with the west and the south arm; one arm idle",
            "shortfall": "the middle channel lies out of the end arms' reach",
            "swapped": "a Franka on the east place reaches the middle channel where it lies and where it is laid, as "
                       "well as the lowest: the west and the east arm lay the track, the south arm idle",
            "borrow": "the 1.30 m channel, laid by the west arm from the shelf into the lowest, does the upper and the "
                      "middle channel's work: the west and the east arm; the 0.45 m one bridges nothing",
        })),
    "Rea-03": TaskSetup(
        remove="south",
        swap=("south", "so101"),
        bodies=lambda inst: {o.name: "wooden domino, 12 cm tall" for o in inst.scene.objects
                             if o.name.startswith("domino_")},
        places={f"place_{i:02d}": f"white mark for a domino in the line, place {i + 1} of 25 from the west end"
                for i in range(25)},
        notes=None,
        borrow_bodies={n: "light wooden slat, like a long domino, 4 x 2 cm, standing on end near the west arm" for n in ("plank_1", "plank_2")},
        expect=_gold("Rea-03", why={
            "exact": "each gap is within reach of one arm only: the line is too long for fewer arms",
            "surplus": "the north arm reaches none of the gaps nor the waiting dominoes (IK): it stays idle",
            "shortfall": "the middle gaps lie out of the end arms' reach",
            "swapped": "an SO-101 on the south place cannot fill the middle gaps from the dominoes waiting for them, "
                       "and no other arm reaches those gaps: every place staffed, it cannot be done",
            "borrow": "the 30 cm plank, stood by the west arm in the empty place 11, falls across the empty middle onto "
                      "domino_14: the west and the east arm (the end places are still theirs alone); the other, by the "
                      "seed 24 cm (it falls into the gap) or 36 cm (its side comes down on domino_14's top edge and "
                      "stays leaning on it), sets nothing off",
        })),
    "Rea-04": TaskSetup(
        remove="south",
        swap=("east", "panda"),
        bodies={"sponge": "yellow sponge, on its dish"},
        # (the two mops alike in words: which clip fits the sponge is for the agent to see or measure)
        borrow_bodies={**{n: "long-handled sponge brush lying by the west arm: a 42 cm wooden handle, its end on a "
                             "rest, and a green plastic head with white bristles along its sides over a clip of two nylon jaws, "
                             "which can be closed onto a thing between them"
                          for n in ("mop_1", "mop_2")},
                       **{n: "small steel rest under a brush handle's end, fixed" for n in ("rest_1", "rest_2")}},
        groups={"riser": ("white riser, 6 cm high and 9 cm square, the middle spill on its top, fixed", ("riser",)),
                "dish": ("white dish at the west end, fixed", ("dish",)),
                "sink": ("sink at the east end, fixed", ("sink_base", "sink_wall_n", "sink_wall_s", "sink_wall_e",
                                                         "sink_wall_w")),
                "spill_1": ("coffee spill on the counter", ("spill_west",)),
                "spill_2": ("coffee spill on the counter", ("spill_south",)),
                "spill_3": ("coffee spill on the counter", ("spill_east",))},
        places={"pad_a": "pad A", "pad_b": "pad B"},
        notes=None,
        expect=_gold("Rea-04", why={
            "exact": "each spill is within reach of one arm only: the counter is too long for fewer arms",
            "surplus": "the north arm reaches the sponge but no spill, pad or the sink (IK): it stays idle",
            "shortfall": "the middle spill lies out of the end arms' reach: the sponge can be brought neither onto it "
                         "nor on from it",
            "swapped": "a Franka on the east place, holding the sponge by its end, reaches the middle spill too (IK, "
                       "the reach audit's 8.5 cm): the west arm wipes its spill and hands the sponge on, the east arm "
                       "wipes the other two; the south arm idle",
            "borrow": "the west arm wipes its spill, leaves the sponge on pad A, clips it with the mop whose clip fits "
                      "it, wipes the middle spill on the riser with it and leaves it where the east arm takes it; the "
                      "other mop's clip, too narrow or too wide, takes the sponge no way",
        })),
}


@dataclass(frozen=True)
class Case:
    task: str
    variant: str
    level: str
    seed: int
    #: the instance with its scene changed (arms, blocks); its success formula is the task's
    inst: TaskInstance
    setup: TaskSetup
    removed: Optional[str]
    added: Optional[str]
    blocks: tuple[str, ...]
    #: the swapped version's (place, robot put there)
    swapped: Optional[tuple[str, str]] = None

    @property
    def id(self) -> str:
        return f"{self.task}-{self.variant}"

    @property
    def expect(self) -> Expect:
        """The case's answer, the borrow design's things ("prop", "decoy") named as in this scene (which is which is
        drawn from the seed)."""
        e = self.setup.expect[self.variant]
        if e.missing_by_level:
            e = dataclasses.replace(e, missing=dict(e.missing_by_level).get(self.level, e.missing))
        if not (e.things or e.rejected):
            return e
        b = self.inst.metadata.get("borrow") or {}
        named = {"prop": tuple(b.get("props", ())), "decoy": tuple(b.get("decoys", ()))}
        return dataclasses.replace(e, things=tuple(n for t in e.things for n in named.get(t, (t,))),
                                   rejected=tuple(n for t in e.rejected for n in named.get(t, (t,))))

    def rng(self) -> random.Random:
        # seeded by the task's seed id, not its name, so that a task's cases do not depend on what it is called
        return random.Random(f"agent/{self.inst.spec.seed_id}-{self.variant}/{self.level}/{self.seed}")


def add_setups(more: dict[str, TaskSetup]) -> None:
    """The cases of a plugin's tasks (``core/plugins.py``): their team and expected answers by task id."""
    clash = set(more) & set(SETUPS)
    if clash:
        raise ValueError(f"case setups for {', '.join(sorted(clash))} exist already")
    SETUPS.update(more)


def setups() -> dict[str, TaskSetup]:
    """Every task's case setup, the plugins' too."""
    from ..core import plugins

    plugins.load()
    return SETUPS


def cases(tasks=None) -> list[tuple[str, str]]:
    """Every (task, variant) with an expected answer."""
    return [(t, v) for t, s in setups().items() if tasks is None or t in tasks for v in VARIANTS if v in s.expect]


#: the scenes a case may be played in (a case id's ``@level.sSEED.SETTING``): the look each setting dresses the task in
#: (None: its card's).  "varied": the room by the seed (``SEED_LOOKS``), the task's things in colours drawn from the
#: seed, and three everyday things within the arms' reach that the task does not need (``scenery.distractors``)
SETTINGS = {"card": None, "varied": "{room},colors=random,distract=3"}
#: a "varied" scene's room by its seed (seed % 3): the kitchen, a workshop (the cards' look), a home -- each with its
#: furnished corner behind the table (``scenery.backdrop``); at home a cream cloth on the table, against which brown
#: things stand out (Rea-02's channels, Rea-03's dominoes, Rea-04's spills)
SEED_LOOKS = ("kitchen", "workshop", "home,table=cloth:cream")


def setting_look(setting: str, seed: int):
    """The look a setting dresses a scene of ``seed`` in (None: the card's)."""
    look = SETTINGS[setting]
    return look and look.format(room=SEED_LOOKS[seed % len(SEED_LOOKS)])


def split_case(case_id: str) -> tuple[str, str]:
    """A case id's task and variant: "Con-02-borrow" -> ("Con-02", "borrow") (task ids have a dash of their own);
    a scene after an "@" ("Con-01-exact@hard.s2.varied") is not part of them."""
    task, variant = case_id.partition("@")[0].rsplit("-", 1)
    return task, variant


def parse_case(case_id: str) -> tuple[str, str, str, int, str]:
    """(task, variant, level, seed, setting) of a case id: "Con-01-exact" is ("Con-01", "exact", "easy", 0, "card"),
    "Con-01-exact@hard.s2.varied" ("Con-01", "exact", "hard", 2, "varied")."""
    task, variant = split_case(case_id)
    scene = case_id.partition("@")[2]
    if not scene:
        return task, variant, "easy", 0, "card"
    try:
        level, seed, setting = scene.split(".")
        if not seed.startswith("s") or setting not in SETTINGS:
            raise ValueError
        return task, variant, level, int(seed[1:]), setting
    except ValueError:
        raise ValueError(f"cannot read the scene in {case_id!r}: give @LEVEL.sSEED.SETTING, a setting one of "
                         f"{', '.join(SETTINGS)}") from None


def scene_ids(case_ids, levels=("easy", "hard"), seeds=(0, 1, 2, 3, 4, 5), setting: str = "varied") -> list[str]:
    """Each case id of the benchmark's eight in each scene: ``case@level.sSEED.setting``.  (Distractors stand outside
    the cells a task's reference sweeps, which only the eight have: ``scenery.sweep.BENCHMARK``.)"""
    from ..scenery.sweep import BENCHMARK

    return [f"{c}@{lv}.s{s}.{setting}" for c in case_ids if split_case(c)[0] in BENCHMARK
            for lv in levels for s in seeds]


def case_from_id(case_id: str) -> "Case":
    task, variant, level, seed, setting = parse_case(case_id)
    return make_case(task, variant, level, seed, setting_look(setting, seed))


def _repainted(setup: TaskSetup, words) -> TaskSetup:
    """``setup``'s descriptions with the colour words the look's painting changed (``scenery.paint``)."""
    if not words:
        return setup

    def what(v):
        return repaint_text(v, words) if isinstance(v, str) else (repaint_text(v[0], words), *v[1:])

    return dataclasses.replace(setup, bodies={k: what(v) for k, v in setup.bodies.items()},
                               groups={k: what(v) for k, v in setup.groups.items()},
                               places={k: what(v) for k, v in setup.places.items()},
                               notes=tuple(repaint_text(n, words) for n in setup.notes))


def own_borrow(task: str) -> bool:
    """Whether ``task``'s borrow case is its own borrow version (built on its card), not the steel blocks."""
    return "borrow" in get(task).variants


def has_variant(task: str, variant: str) -> bool:
    """Whether ``task``'s case can be set up in ``variant`` (whether or not it has an answer yet)."""
    s = setups().get(task)
    if s is None:
        return False
    if variant == "swapped":
        return s.swap is not None
    if variant in ("borrow", "spare"):
        return bool(s.blocks) or (variant == "borrow" and own_borrow(task))
    return variant in ("exact", "surplus", "shortfall")


def make_case(task: str, variant: str, level: str = "easy", seed: int = 0, look=None, check_gold: bool = True) -> Case:
    """The case of ``task``'s ``variant`` at ``level`` and ``seed``, in ``look`` (``tasks.build``'s; None: the card's).
    ``check_gold=False``: also a version that has no answer in ``gold.json``."""
    setup = setups()[task]
    if variant not in setup.expect and (check_gold or not has_variant(task, variant)):
        raise ValueError(f"{task} has no {variant} case (has {', '.join(setup.expect)})")
    own = variant == "borrow" and own_borrow(task)
    inst = build(task, level, seed, look, variant="borrow" if own else "exact")
    bodies = setup.bodies(inst) if callable(setup.bodies) else setup.bodies
    setup = dataclasses.replace(setup, bodies={**bodies, **setup.borrow_bodies} if own else bodies,
                                notes=inst.spec.facts if setup.notes is None else setup.notes)
    setup = _repainted(setup, inst.metadata.get("repaint"))
    if inst.metadata.get("distractors"):  # shown with the task's own things
        setup = dataclasses.replace(setup, bodies={**setup.bodies, **inst.metadata["distractors"]})
    arms, objects = list(inst.scene.arms), list(inst.scene.objects)
    removed = added = swapped = None
    if variant == "surplus":
        extra = setup.extra or EXTRA_ARM
        arms.append(extra)
        added = extra.slot
    if variant in ("shortfall", "borrow"):  # (the task's own borrow version has left it out already)
        arms = [a for a in arms if a.slot != setup.remove]
        removed = setup.remove
    if variant == "swapped":
        swapped = setup.swap
        arms = [dataclasses.replace(a, model=swapped[1]) if a.slot == swapped[0] else a for a in arms]
    blocks: tuple[str, ...] = ()
    if variant in ("borrow", "spare") and not own:
        for k, (x, y, yaw) in enumerate(setup.blocks, start=1):
            objects.append(ObjectSpec(f"block_{k}", "box", BLOCK, (x, y, TABLE_TOP_Z + BLOCK[2] / 2), yaw, STEEL,
                                      BLOCK_MASS))
        blocks = ("block_1", "block_2")
    scene = dataclasses.replace(inst.scene, arms=tuple(arms), objects=tuple(objects))
    return Case(task, variant, level, seed, dataclasses.replace(inst, scene=scene), setup, removed, added, blocks,
                swapped)
