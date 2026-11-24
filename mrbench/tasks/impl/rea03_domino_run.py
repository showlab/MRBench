"""Rea-03 domino run: a line of wooden dominoes 2 m long down the table has three gaps -- the first place and the last,
and a stretch of three places in the middle -- each in one arm's part of it; the gaps must be filled from the dominoes
standing by, and the line set off from its west end so that it falls all the way to its east end.

Each gap is within reach of one arm only, and only the west arm reaches the first domino: the line is too long for
fewer arms.  (The middle stretch lies well clear of the end arms' reach -- checked by IK with margins, ``validate
--ik --slack`` -- with the line 40 cm in front of the south arm, an end arm would reach to within 2 cm of the middle
gap.)  A domino stood in its gap stays: no arm holds anything while another works.  Success asks that the line be
whole when it is set off (``ChainStands``): set off in two halves, with the middle left empty, it is not."""

from __future__ import annotations

import math
from dataclasses import replace

from ...engines.base import RegionSpec, axis_quat, yaw_quat
from ...policies.script import Grip, Move, Open, Release, back_off, carry, offset, phase, radial_tool, tool_quat
from ...scoring.predicates import (AtOnset, ChainStands, DownOnLine, Eventually, FallenAt, HoldsFor, InRegion, Upright,
                                   all_of, all_temporal)
from ..layouts import mount
from ..loader import get
from ._common import COLORS, Z, at, box, jitter, make_instance, rng_for, scene

#: the line: where across the table (20 cm in front of the south arm's base: there the end arms reach the middle no
#: nearer than x = -0.14 m, at a domino's top), and how many places it has (west to east)
LINE_Y, N = -0.50, 25
#: a domino: thickness (along the line), width and height
DOMINO = (0.02, 0.06, 0.12)
#: the places left empty, each arm's in its part of the line: the first and the last (beyond the south arm's reach,
#: 0.92 m from its base), and the middle three
GAPS = {"west": (0,), "south": (11, 12, 13), "east": (24,)}
#: where the missing dominoes wait, standing, in the same order (each turned so that the hand that stands it in its
#: gap takes it with the same hold; no more than about 130 degrees round from where a Piper faces: its base turns 150
#: degrees each way).  The south arm's three wait 17 cm north of the line, 10 cm apart: at 32 cm the south Piper,
#: pointing 60 degrees down, cannot lift one
WAITS = {"west": ((-0.88, 0.20),), "south": ((-0.10, -0.33), (0.0, -0.33), (0.10, -0.33)), "east": ((0.88, 0.20),)}
#: every place of the line is painted on the table (MARK: half its size along the line and across it, a little more
#: than a domino's foot): the empty ones at the ends show, as the success asks the line whole from place 0 to the
#: last (unmarked, domino_01 to domino_23 may read as the whole line).
MARK = (0.016, 0.036)
#: a domino counts as down past this tilt (a fallen one rests on the next at 27-33 degrees), and as standing within this
DOWN_DEG, STANDING_DEG = 15.0, 10.0
#: the line is set off when the domino on its first place falls: one whose foot is within half a spacing of the place
#: along the line, this far across it and this far above the table (a tipping domino's foot rises no more than half
#: its thickness; one carried over the place is far higher)
FIRST_ACROSS, FIRST_UP = 0.03, 0.02
#: the borrow version (no south arm): a light plank stood in place 10, the last place the west arm reaches (left
#: empty: EMPTY), falls east across the empty middle stretch and its top end strikes domino_14's west face, below
#: its top, which tips it on.  A plank whose side comes down on domino_14's top corner first (longer than the corner
#: is from its foot), or whose end meets the face within a centimetre or two of the top, presses the domino down and
#: stays leaning on it.  In Isaac a 30 cm plank carries the run at both levels from 28.3 to 29.5 cm off domino_14 (its
#: end meeting the face 10 to 6 cm up) and sticks at 27.5; a 24 cm one falls into the gap.  So the plank stands
#: PLANK_GAP from domino_14 whatever the spacing (0.9 cm east of place 10 at hard, 0.7 cm west of it at easy).  Its
#: size (thickness, width, height), the distractor's height and their weight (about a domino's).  It is a light wooden
#: slat in the dominoes' wood, like a long domino
PLANK, SHORT_PLANK, PLANK_KG, PLANK_GAP = (0.02, 0.04, 0.30), 0.24, 0.05, 0.287
#: the distractor errs either way, which drawn from the seed (so that the prop is not always the larger): SHORT_PLANK, shorter than PLANK_GAP, falls into the gap; LONG_PLANK, longer than the 31.1 cm from its
#: foot to domino_14's top west edge (PLANK_GAP along, a domino's height up), comes down with its side on that edge,
#: 22.7 degrees above level, so that it pushes the domino more down than east -- to tip, the edge would have to slide
#: up along the plank, and friction holds it -- and stays leaning on it: the run stops at domino_14.  In Isaac (the
#: plank 28.0-29.5 cm off domino_14, both levels) 33 to 36 cm stick; 38 and 40 cm, striking the edge harder, can carry
#: the run on
LONG_PLANK = 0.36
#: the plank's end strikes domino_14's face at least this far below its top for the middle stretch to count as bridged
#: (a 30 cm plank sticks at 27.5 cm from domino_14, its end meeting the face at the top; it carries the run from 28.3,
#: 10 cm up): a taller plank, whose side comes down on domino_14's top corner, does not count
STRIKE_BELOW_TOP = 0.01
#: a standing thing falling reaches this much less than its height (ChainStands): for dominoes 5-6 cm apart it does
#: not matter; for the plank, 1 cm (the predicate's default) leaves only gaps where its end meets domino_14 near its top
REACH_SLACK = 0.005
#: the west arm reaches the plank's place only with its hand level (IK: 10-22 cm up), so it holds a plank by the
#: side, this high over the table, and lifts it this far to carry it (the arm, loaded, sags 2-3 cm): its hand, 9.5 cm
#: behind the fingers, then passes over domino_09 (held 11-16 cm up, the hand came within 1 cm of it; 18-19 cm up,
#: the hard level's place, 0.9 cm further, is out of reach).  Level, it reaches nothing nearer its base than 0.55 m
PLANK_HOLD, PLANK_LIFT = 0.17, 0.03
#: and goes this far up the plank once it has let go, before it backs away
PLANK_UP = 0.04
#: where the two planks wait, standing, 0.66 m from the west arm's base (its hand comes in from 8 cm nearer), south
#: to north (which is which drawn from the seed); a plank is carried round to the line SWING from the base, inside
#: the middle dominoes waiting for the south arm
PLANK_WAITS, SWING = ((-0.043, -0.058), (-0.054, 0.137)), 0.58
#: the borrow version leaves the plank's place, place 10, empty (domino_10 is not on the table): the west arm cannot
#: take a domino out of it -- holding one there by the side, its hand, 9.5 cm behind the fingers and as low as the
#: dominoes' tops, runs 1.3-2.5 cm into domino_09 at any pitch and turn it reaches with (IK, the clearance check's
#: capsules)
EMPTY = 10


def place_x(i: int, spacing: float) -> float:
    return (i - (N - 1) / 2) * spacing


def standing(x: float, yaw: float = 0.0):
    return (x, LINE_Y, Z + DOMINO[2] / 2, *yaw_quat(yaw))


def _turn(slot: str, frm, to) -> float:
    """How far (deg, about the vertical) a hand pointing out from ``slot``'s base turns between reaching ``to`` and
    reaching ``frm``."""
    m = mount(slot)
    return math.degrees(math.atan2(frm[1] - m.y, frm[0] - m.x) - math.atan2(to[1] - m.y, to[0] - m.x))


def gap_regions() -> list:
    """(region name, place, arm) of every gap's place: gap_west, gap_middle_a..c, gap_east."""
    out = []
    for slot, places in GAPS.items():
        names = [f"gap_{'middle' if slot == 'south' else slot}" + (f"_{'abc'[k]}" if len(places) > 1 else "")
                 for k in range(len(places))]
        out += [(nm, i, slot) for nm, i in zip(names, places)]
    return out


def waiting() -> list:
    """(place, arm, where it waits) of every missing domino."""
    return [(i, slot, xy) for slot, places in GAPS.items() for i, xy in zip(places, WAITS[slot])]


def plank_x(spacing: float) -> float:
    """Where the borrow version's plank stands: PLANK_GAP from domino_14's west face."""
    return place_x(14, spacing) - DOMINO[0] / 2 - PLANK_GAP - PLANK[0] / 2


def build(level: str, seed: int, variant: str = "exact"):
    spec = get("Rea-03")
    rng = rng_for(spec, level, seed)
    spacing = spec.knob.levels[level] / 1000
    waits = {i: (slot, xy) for i, slot, xy in waiting()}
    dominoes = []
    for i in range(N):
        name = f"domino_{i:02d}"
        if variant == "borrow" and i == EMPTY:
            continue
        if i in waits:  # the missing one waits by its arm, standing
            slot, xy = waits[i]
            xy = jitter(rng, xy, 0.008)
            yaw = _turn(slot, xy, (place_x(i, spacing), LINE_Y))
            dominoes.append(box(name, xy, DOMINO, "wood", mass=0.04, yaw=yaw))
        else:
            dominoes.append(box(name, (place_x(i, spacing), LINE_Y), DOMINO, "wood", mass=0.04))
    regions = [RegionSpec(nm, standing(place_x(i, spacing))[:3], (0.015, 0.015, 0.01), visible=False)
               for nm, i, _ in gap_regions()]
    regions += [RegionSpec(f"wait_{i:02d}", (*xy, Z + DOMINO[2] / 2), (0.01,) * 3, visible=False)
                for i, _, xy in waiting()]
    regions.append(RegionSpec("first", standing(place_x(0, spacing))[:3], (0.01,) * 3, visible=False))
    regions += [RegionSpec(f"place_{i:02d}", (place_x(i, spacing), LINE_Y, Z + 0.001), (*MARK, 0.001), COLORS["white"])
                for i in range(N)]
    planks, extra = [], {}
    if variant == "borrow":  # no south arm: a plank stood in place 10 falls across the middle stretch
        heights = [PLANK[2], SHORT_PLANK]  # south to north
        if rng.random() < 0.5:
            heights.reverse()
        if rng.random() < 0.5:  # the distractor too tall, not too short (drawn after the places)
            heights = [LONG_PLANK if h == SHORT_PLANK else h for h in heights]
        px = plank_x(spacing)
        for k, (xy, h) in enumerate(zip(PLANK_WAITS, heights)):
            planks.append(box(f"plank_{k + 1}", xy, (*PLANK[:2], h), "wood", mass=PLANK_KG,
                              yaw=_turn("west", xy, (px, LINE_Y))))
        prop = planks[heights.index(PLANK[2])].name
        extra = {"borrow": {"props": [prop], "decoys": [p.name for p in planks if p.name != prop]}}
    sc = scene(spec, dominoes + planks, regions)
    if variant == "borrow":
        sc = replace(sc, arms=tuple(a for a in sc.arms if a.slot != "south"))

    # whatever stands on the line is part of it: a plank too
    names = tuple(d.name for d in dominoes + planks)
    x0, x1 = place_x(0, spacing), place_x(N - 1, spacing)
    # (the first one leaning as it is pushed)
    whole = ChainStands(names, LINE_Y, x0, x1, max_tilt=DOWN_DEG, margin=REACH_SLACK)
    down = DownOnLine(names, LINE_Y, x0, x1, DOWN_DEG)
    # the line is set off when the domino on its first place falls -- whichever domino stands there; domino_00
    # knocked where it waits, off the line, is not
    set_off = FallenAt(names, (x0, LINE_Y, Z), (spacing / 2, FIRST_ACROSS, FIRST_UP), DOWN_DEG)
    filled = {nm: InRegion(f"domino_{i:02d}", nm) & Upright(f"domino_{i:02d}", STANDING_DEG)
              for nm, i, _ in gap_regions()}
    middle = [nm for nm, _, slot in gap_regions() if slot == "south"]
    gap = {slot: [nm for nm, _, s in gap_regions() if s == slot] for slot in GAPS}
    middle_done = all_of(*[filled[nm] for nm in middle])
    reach = [(slot, f"wait_{i:02d}") for i, slot, _ in waiting()] + [(slot, nm) for nm, _, slot in gap_regions()]
    reach += [("west", "first")]
    out_of_reach = [(s, nm) for nm, _, slot in gap_regions() for s in GAPS if s != slot]
    out_of_reach += [("south", f"wait_{i:02d}") for i, slot, _ in waiting() if slot != "south"]
    out_of_reach += [("south", "first"), ("east", "first")]
    if variant == "borrow":
        # the middle stretch is bridged: domino_09 and domino_14 standing, a plank between them that reaches across
        bridge = ("domino_09", *(p.name for p in planks), "domino_14")
        # -- one that strikes domino_14's face, not one whose side comes down on its top corner (a taller plank's)
        middle_done = middle_done | ChainStands(bridge, LINE_Y, place_x(9, spacing), place_x(14, spacing),
                                                max_tilt=STANDING_DEG, margin=REACH_SLACK, below_top=STRIKE_BELOW_TOP)
        # (no claims for what the west arm does with the planks: its level hand works 0.56 to 0.74 m out, beyond
        # the design rules' nominal reach of 0.56 m -- IK's alone, which the scripted reference's plan checks)
        reach = [r for r in reach if r[0] != "south"]
        out_of_reach = [r for r in out_of_reach if r[0] != "south"]
    return make_instance(
        spec, level, seed, sc,
        # the goal only: that the gaps must be filled, and from which dominoes, is for the agent to see
        instruction="Set the line of dominoes off from its west end so that it falls all the way to its east end.",
        success=all_temporal(AtOnset(whole, set_off), HoldsFor(down, 1.0)),
        milestones={
            "west_gap_filled": Eventually(all_of(*[filled[nm] for nm in gap["west"]])),
            "middle_gap_filled": Eventually(middle_done),
            "east_gap_filled": Eventually(all_of(*[filled[nm] for nm in gap["east"]])),
            "first_down": Eventually(set_off),
            "all_down": Eventually(down),
        },
        reach=reach,
        # the end PiPERs do reach the middle dominoes where they wait, but not their gap, which only the south arm
        # reaches: three arms are still needed
        out_of_reach=out_of_reach,
        spacing_m=spacing, **extra,
    )


#: the keyframe `mrbench render` shows between start and goal: every gap filled, the line about to be set off
MIDDLE = 2


def fallen(x: float, deg: float, size=DOMINO):
    """A domino (or a plank, ``size``) that stood at ``x`` tipped east by ``deg`` about its bottom east edge."""
    t, (hx, _, hz) = math.radians(deg), (size[0] / 2, 0, size[2] / 2)
    dx, dz = -hx * math.cos(t) + hz * math.sin(t), hx * math.sin(t) + hz * math.cos(t)
    return (x + hx + dx, LINE_Y, Z + dz, *axis_quat((0.0, 1.0, 0.0), deg))


def keyframes(inst):
    spacing = inst.metadata["spacing_m"]
    lean = math.degrees(math.asin(min(1.0, (spacing - DOMINO[0]) / DOMINO[2])))  # resting on the next one
    if inst.variant == "borrow":  # the end gaps filled; the plank in the empty place (MIDDLE); all down
        prop = inst.metadata["borrow"]["props"][0]
        size, px = inst.scene.obj(prop).size, plank_x(spacing)
        tip = math.degrees(math.asin(min(1.0, PLANK_GAP / size[2])))  # its top end against domino_14
        line = [i for i in range(N) if i != EMPTY and i not in GAPS["south"]]
        return [
            *({f"domino_{i:02d}": standing(place_x(i, spacing)) for i in GAPS[slot]} for slot in ("west", "east")),
            {prop: (px, LINE_Y, Z + size[2] / 2, *yaw_quat(0.0))},
            {**{f"domino_{i:02d}": fallen(place_x(i, spacing), lean if i < N - 1 else 90.0) for i in line},
             prop: fallen(px, tip, size)},
        ]
    frames = [{f"domino_{i:02d}": standing(place_x(i, spacing)) for i in places} for places in GAPS.values()]
    frames.append({f"domino_{i:02d}": fallen(place_x(i, spacing), lean if i < N - 1 else 90.0) for i in range(N)})
    return frames


#: the scripted reference: how far below level a hand points, by arm (the south Piper 60, 20 cm from its base; a
#: Franka there finds hardly a configuration that stands a domino at the line's middle, at any pitch); how high a
#: domino is carried (its bottom over the line's tops) and how far above the table it is let go; how far the push
#: that sets the line off goes
PITCH = {"west": 30.0, "south": 60.0, "east": 30.0}
CLEAR, LET_GO, PUSH = 0.03, 0.003, 0.03
#: a domino is taken by its top this far down (lower, the hand comes within a centimetre of its neighbours' tops),
#: and the hand that pushes the first one points this far down (at 30 degrees it passes 8 mm over the second)
TAKE_DOWN, PUSH_PITCH = 0.005, 45.0
#: the south arm brings a domino to its place from this far north of it, this far up (its bottom clear of the table)
SOUTH_IN = (0.08, 0.015)


def _broad(inst, slot) -> bool:
    """A Franka's hand, broad along its fingers' travel, closes across a domino's width (its broad side across the
    line); a Piper's across its thickness."""
    return inst.scene.arm(slot).model == "panda"


def _domino_moves(inst, slot, i):
    """The tool poses that take the domino missing from place ``i`` by its top where it waits and stand it there."""
    name = f"domino_{i:02d}"
    obj = inst.scene.obj(name)
    target = standing(place_x(i, inst.metadata["spacing_m"]))
    arm = inst.scene.arm(slot)
    top = (target[0], target[1], target[2] + DOMINO[2] / 2 - TAKE_DOWN)
    place = (*top, *radial_tool(arm.pos, top, PITCH[slot], (0.0, 1.0, 0.0) if _broad(inst, slot) else None))
    return name, carry(place, target, obj.pose), place


def _stand(inst, slot, i, first_note=""):
    """``slot`` takes the domino missing from place ``i`` where it waits and stands it there: (phases, as lists of
    actions per phase, to be merged with other arms').  An end arm brings it over the line, high enough for its
    bottom to clear the line's tops, and down; the south arm, 20 cm from its base, low from the north side (nothing
    stands there): over the line it reaches hardly 20 cm up."""
    name, take, place = _domino_moves(inst, slot, i)
    # how wide the domino is across the fingers, which close square to the reach, not to the line: at an end place
    # 22 (easy) to 26 (hard) degrees round from its thickness, 4.1 to 4.5 cm.  (Opened 2.5 cm wider than a domino's
    # thickness, 4.3 cm, the end hands come down on the top of the domino they take and tip it where it waits;
    # gripped so, it is stood leaning.)
    width = _span(inst.scene.obj(name), take) - 0.002
    up = DOMINO[2] + CLEAR  # a carried domino's bottom clears the line's tops
    over = offset(place, dy=SOUTH_IN[0], dz=SOUTH_IN[1]) if slot == "south" else offset(place, dz=up)
    lift = offset(take, dz=SOUTH_IN[1] + 0.015) if slot == "south" else offset(take, dz=up)
    return [
        (3.0, first_note, [Move(slot, back_off(take, 0.08), joints=True), Open(slot, width + 0.025)]),
        (1.2, "", [Move(slot, take)]),
        (0.4, "and takes it by its top", [Grip(slot, name, width)]),
        (1.5, "", [Move(slot, lift)]),
        (3.0, "carries it over its gap", [Move(slot, over, joints=True)]),
        (2.0, "and stands it in the gap", [Move(slot, offset(place, dz=LET_GO))]),
        # let go with the fingers kept nearly shut: a hand that lets go opening them wide comes onto the neighbours
        (0.4, "", [Release(slot), Open(slot, width + 0.008)]),
        (1.5, "", [Move(slot, offset(place, dz=0.08))]),  # straight up, clear of the line
    ]


def _push(inst):
    """The west arm, its fingers shut, pushes the first domino's top east."""
    first = standing(place_x(0, inst.metadata["spacing_m"]))  # where the west arm stood the first domino
    w = inst.scene.arm("west")
    tip = (first[0] - DOMINO[0] / 2 - 0.014, first[1], first[2] + DOMINO[2] / 2 - 0.015)
    push = (*tip, *radial_tool(w.pos, tip, PUSH_PITCH))
    return [
        # the fingers shut as the hand sets off; at the push, a grip that takes nothing (the hand works on the first
        # domino).  (A grip in the same phase as this move makes the planner bring the hand straight in from home.)
        [Move("west", offset(back_off(push, 0.06), dz=0.03), joints=True), Open("west", 0.0)],
        [Move("west", push)],
        [Grip("west", "domino_00", 0.0, attach=False), Move("west", offset(push, dx=PUSH))],
        [Release("west"), Open("west", 0.0), Move("west", back_off(offset(push, dx=PUSH), 0.08))],
    ]


def script(inst):
    """Privileged: a grip fixes the domino to the hand.  The three arms stand their first dominoes in their gaps at
    once, each lowered from above the line between its neighbours; the south arm then stands the other two of the
    middle stretch; then the west arm, its fingers shut, pushes the first domino's top east."""
    from ...policies.script import home_pose

    if inst.variant == "borrow":
        return _borrow_script(inst)
    legs = {slot: _stand(inst, slot, places[0], "each arm goes to the domino by it") for slot, places in GAPS.items()}
    out = [phase(sec, note, *[a for slot in GAPS for a in legs[slot][k][2]])
           for k, (sec, note, _) in enumerate(legs["south"])]
    out += [phase(2.0, "", Move("west", home_pose(inst, "west"), joints=True),
                  Move("east", home_pose(inst, "east"), joints=True))]
    for i in GAPS["south"][1:]:
        out += [phase(sec, note if k else "the south arm stands the next one", *acts)
                for k, (sec, note, acts) in enumerate(_stand(inst, "south", i))]
    go, at_, tip, off = _push(inst)
    return out + [
        phase(2.5, "the west arm goes to the first domino", *go, Move("south", home_pose(inst, "south"), joints=True)),
        phase(1.0, "", *at_),
        phase(0.8, "and tips it", *tip),
        phase(1.0, "the run goes down the line", *off),
        phase(5.0, ""),
    ]


#: the borrow version: the west hand, level, points this far round from straight out from its base (clockwise seen
#: from above, toward the line: its hand further from domino_09; at -12 the hard level's place is out of reach)
SIDE_TURN = -6.0


def _level_hand(inst, slot, point):
    """The tool point at ``point``, the hand level, pointing out from the base turned SIDE_TURN, its fingers level."""
    base = inst.scene.arm(slot).pos
    b = math.atan2(point[1] - base[1], point[0] - base[0]) + math.radians(SIDE_TURN)
    return (*point, *tool_quat((math.cos(b), math.sin(b), 0.0), (-math.sin(b), math.cos(b), 0.0)))


def _at_distance(pose, base, r):
    """``pose`` moved along the line from ``base`` to ``r`` from it (horizontally)."""
    d = math.hypot(pose[0] - base[0], pose[1] - base[1])
    return offset(pose, dx=(pose[0] - base[0]) * (r / d - 1), dy=(pose[1] - base[1]) * (r / d - 1))


def _span(obj, hand) -> float:
    """How wide a standing box is across the fingers of a level hand at ``hand`` (its pose)."""
    from ...engines.base import quat_rotate, quat_yaw_deg

    f = quat_rotate(hand[3:], (0.0, 1.0, 0.0))  # the fingers' travel (tool_quat's ``across``)
    y = math.radians(quat_yaw_deg(obj.pose[3:]))
    ex, ey = (math.cos(y), math.sin(y)), (-math.sin(y), math.cos(y))
    return obj.size[0] * abs(f[0] * ex[0] + f[1] * ex[1]) + obj.size[1] * abs(f[0] * ey[0] + f[1] * ey[1])


def _borrow_script(inst):
    """The borrow configuration (no south arm): the end arms stand the first and the last domino as in the exact one;
    the west arm then stands the plank, held by its side, in the empty place 10, PLANK_GAP from domino_14, and tips
    the first domino."""
    from ...policies.script import home_pose

    w, s = "west", inst.metadata["spacing_m"]
    base = inst.scene.arm(w).pos
    legs = {slot: _stand(inst, slot, GAPS[slot][0], "the end arms go to the dominoes by them") for slot in (w, "east")}
    out = [phase(sec, note, *legs[w][k][2], *legs["east"][k][2]) for k, (sec, note, _) in enumerate(legs[w])]
    out += [phase(2.0, "", Move(w, home_pose(inst, w), joints=True),
                  Move("east", home_pose(inst, "east"), joints=True))]

    prop = inst.scene.obj(inst.metadata["borrow"]["props"][0])
    stood = (plank_x(s), LINE_Y, Z + prop.size[2] / 2, *yaw_quat(0.0))
    stand = _level_hand(inst, w, (stood[0], LINE_Y, Z + PLANK_HOLD))
    take = carry(stand, stood, prop.pose)
    span = _span(prop, take)
    over = offset(stand, dy=0.08, dz=PLANK_LIFT)  # in from the north, between domino_09 and the empty places
    out += [
        phase(3.0, "goes to the plank", Move(w, back_off(take, 0.08), joints=True), Open(w, span + 0.012)),
        phase(1.5, "", Move(w, take)),
        phase(0.4, "takes it by its side", Grip(w, prop.name, span - 0.002)),
        phase(1.0, "", Move(w, offset(take, dz=PLANK_LIFT))),
        phase(1.5, "", Move(w, _at_distance(offset(take, dz=PLANK_LIFT), base, SWING))),
        phase(3.0, "carries it round to the empty place", Move(w, _at_distance(over, base, SWING), joints=True)),
        phase(1.5, "", Move(w, over)),
        phase(1.5, "", Move(w, offset(stand, dz=PLANK_LIFT))),
        phase(1.5, "and stands it there", Move(w, offset(stand, dz=LET_GO))),
        phase(0.4, "", Release(w), Open(w, span + 0.008)),
        # up the plank before it backs away: the arm, which holds the plank 2-3 cm under its plan, can sweep domino_09
        # over backing straight away, and the line falls west
        # (out of the fingers' reach of it within 1.5 s of letting go, as the clearance check allows)
        phase(0.5, "", Move(w, offset(stand, dz=PLANK_UP))),
        phase(0.9, "", Move(w, offset(back_off(stand, 0.08), dz=PLANK_UP))),
        # home first: from the plank straight to the first domino the hand goes low along the line
        phase(2.0, "", Move(w, home_pose(inst, w), joints=True)),
    ]
    go, at_, tip, off = _push(inst)
    return out + [
        phase(2.5, "the west arm goes to the first domino", *go),
        phase(1.0, "", *at_),
        phase(0.8, "and tips it", *tip),
        phase(1.0, "the run goes down the line: the plank falls across the empty middle onto domino_14", *off),
        phase(5.0, ""),
    ]


def jobs(inst):
    """What the task asks (``tasks/jobs.py``): each gap filled from a waiting domino -- carried by one arm, or passed
    between two where they meet -- and the line set off at its first place.  In the borrow version place 11 (place_10)
    starts empty with no domino waiting for it, and the plank, stood there, does the whole middle stretch's work; the
    other one (too short or too tall, by the seed) does nothing."""
    from ..jobs import Job, Jobs, Thing

    margin = 0.03  # the reach audit's (`validate --ik --slack 0.03 Rea-03`): a domino's half-thickness and a grip
    fill = [Job(f"fill place {i + 1}", "carry", sites=(f"domino_{i:02d}", nm), slack=margin, body=f"domino_{i:02d}")
            for nm, i, _ in gap_regions()]
    things = ()
    if inst.variant == "borrow":
        # nothing waits for the empty place: no arm can fill it (None: nothing to take)
        fill.insert(1, Job(f"fill place {EMPTY + 1}", "carry", sites=(None, f"place_{EMPTY:02d}"), slack=margin))
        b = inst.metadata["borrow"]
        prop, decoy = b["props"][0], b["decoys"][0]
        middle = [j.name for j in fill if j.name in {f"fill place {i + 1}" for i in (EMPTY, *GAPS["south"])}]
        stood = (plank_x(inst.metadata["spacing_m"]), LINE_Y, Z + PLANK_HOLD)
        tall = inst.scene.obj(decoy).size[2]
        edge = math.hypot(PLANK_GAP, DOMINO[2])  # from the plank's foot to domino_14's top west edge
        things = (Thing(prop, does=tuple(middle),
                        placing=(Job(f"stand {prop} in place {EMPTY + 1}", "carry", sites=(prop, stood), slack=margin,
                                     body=prop),),
                        why=f"{PLANK[2] * 100:.0f} cm tall, stood {PLANK_GAP * 100:.1f} cm from domino_14: it falls "
                            "across the empty middle onto domino_14"),
                  Thing(decoy, why=f"{tall * 100:.0f} cm tall, shorter than the {PLANK_GAP * 100:.1f} cm to "
                                   "domino_14: it falls into the gap" if tall < PLANK_GAP else
                        f"{tall * 100:.0f} cm tall, longer than the {edge * 100:.1f} cm from its foot to domino_14's "
                        "top edge: its side comes down on that edge, presses the domino down and stays leaning on it"))
    return Jobs((*fill, Job("set the line off at its first place", sites=("first",), slack=margin)), things)
