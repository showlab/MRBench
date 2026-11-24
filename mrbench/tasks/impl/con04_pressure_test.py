"""Con-04 pressure test: a steel test pipe holds its pressure only with both of its bleed valves shut while the foot
valve lets air in.  Each bleed valve's lever, and the foot valve's pedal, has a spring and comes back up when let go:
two hands hold the levers down while a third holds the pedal down, for a second (like playing a flute: two arms each
stop a hole while one blows -- only together is it the right note).

The pipe is a round steel pipe with blind flanges at its ends, lying east-west on two stands in the table's north half,
a pressure gauge on its top in the middle; its air comes into its underside through a blue hose from a pneumatic foot
valve on the table in front of the south arm.  The foot valve: a yellow housing and a pedal hinged along its north
edge, held up at REST_DEG by its spring; held down level it lets air in.  A bleed valve is a brass valve body on the
pipe's top with a steel lever hinged on it, pointing in along the pipe, a red grip on a stem at its end; pressed down
it shuts the valve; let go, the spring lifts it and the valve bleeds the pipe.

Why three: three bodies held at once and one hand holds one.  Each bleed valve's grip is reached only by the arm at
its end of the table (the south arm is 1.04 m from either, the far end arm 1.02 m: with the pipe 8 cm further south and
the valves 4 cm further in, a Franka's hand held level would come within 2 cm of them, IK); the end arms reach the pedal's
toe bar too, but their hands are on the levers.  Borrow version (no south arm): a steel test weight set in the pocket
on the pedal holds it down, and the end arm on its side sets it there first; the distractor is the same weight half as
tall, half as heavy, under which the pedal does not move.

The pedal's spring leaves a margin both ways: the heaviest thing the varied scenes put within an arm's reach (a 0.45 kg
book) at the pedal's far end does not move it either (PEDAL_*)."""

from __future__ import annotations

import dataclasses
import math
import random

from ...engines.base import axis_quat, transform
from ...policies.script import Grip, Move, Open, Release, back_off, home_pose, offset, phase, tool_quat
from ...scoring.predicates import Eventually, HoldsFor, Opened
from ..loader import get
from ._common import Z, compound, hinge, jitter, make_instance, part, rng_for, rotate_about, scene, zone

#: the pipe: PIPE_R round, its top PIPE_TOP over the table, PIPE_L long, its middle at (x0, PIPE_Y) (x0 drawn from the
#: seed); blind flanges FLANGE_R round at its ends; on two stands STAND_X out from its middle
PIPE_Y, PIPE_R, PIPE_L, PIPE_TOP, STAND_X = 0.30, 0.035, 1.10, Z + 0.20, 0.46
PIPE_AXIS = PIPE_TOP - PIPE_R
FLANGE_R, FLANGE_T = 0.060, 0.018
#: the bleed valves, VALVE_X out from the pipe's middle (so the grips are 0.52 m from their end arm's base, 1.02 m from
#: the other end arm's and 1.04 m from the south arm's)
VALVE_X = 0.34
#: a valve's lever: hinged PIVOT_OUT outside VALVE_X, PIVOT_UP over the pipe's top (axis along y) on the valve body's
#: head; its grip GRIP_IN inside VALVE_X (0.12 m from the hinge), GRIP_UP over the lever on a stem (pressed down, the
#: hand's fingers stay 2 cm and more over the pipe)
PIVOT_OUT, PIVOT_UP, GRIP_IN, GRIP_UP = 0.05, 0.030, 0.07, 0.031
LEVER_T, LEVER_W = 0.008, 0.014
GRIP_L, GRIP_W = 0.060, 0.020
#: the valve body under the hinge, BODY_R round, its top BODY_GAP under the lever
BODY_R, BODY_GAP = 0.022, 0.008
#: a lever's stop VALVE_MAX; a valve counts as shut past VALVE_SHUT degrees; the reference holds it at VALVE_HOLD
VALVE_MAX, VALVE_SHUT, VALVE_HOLD = 9.0, 7.0, 8.5
#: each lever's spring is wound this far at rest (degrees): with the card's stiffness (2.0 / 3.0 N*m/rad) a hand holds
#: the grip down with 17.0-19.0 / 25.7-28.8 N from rest to the stop (the lever's own weight, 0.06 N*m, helps it).  A
#: weight set on the grip must weigh 1.9 / 2.9 kg on its middle (1.6 / 2.3 at its end) to hold the valve shut: the
#: borrow version's distractor (1.2 kg) cannot, nor anything the varied scenes put within reach (at 0.5 / 1.0 N*m/rad
#: the distractor on a grip would hold its valve shut and free that end arm for the pedal)
VALVE_PRELOAD_DEG = 60.0
LEVER_MASS, LEVER_DAMPING = 0.06, 0.04
#: the gauge on the pipe's top in its middle, its face toward the south arm: GAUGE_R round, GAUGE_T deep, its middle
#: GAUGE_UP over the pipe's top
GAUGE_R, GAUGE_T, GAUGE_UP = 0.050, 0.030, 0.080
#: the foot valve: its middle (x) drawn from the seed, the pedal's hinge (its north edge) at y PEDAL_N; the pedal
#: PEDAL_W x PEDAL_L x PEDAL_T, at rest REST_DEG up at its south end; pressed down level (its stop) it lies 2 mm over
#: the housing.  It lets air in past AIR_DEG; the reference holds it at PEDAL_HOLD
PEDAL_N, PEDAL_W, PEDAL_L, PEDAL_T = -0.03, 0.24, 0.30, 0.015
BASE_T, HOUSING_TOP = 0.02, 0.088
HINGE_Z = Z + HOUSING_TOP + 0.002 + PEDAL_T / 2
REST_DEG, AIR_DEG, PEDAL_HOLD = 14.0, 12.0, 13.5
#: the hose from the foot valve to the pipe's underside, HOSE_R round
HOSE_R = 0.012
#: its toe bar: a bar along x on two posts, TOE_IN in from the pedal's south edge, TOE_UP over the pedal's top
TOE_IN, TOE_UP, TOE_L, BAR = 0.015, 0.037, 0.10, 0.014
#: the pocket the weight goes in: its middle POCKET from the hinge, INNER across inside its rails (RAIL_T thick,
#: RAIL_H tall)
POCKET, INNER, RAIL_T, RAIL_H = 0.20, 0.12, 0.008, 0.010
#: the pedal's spring: PEDAL_TORQUE (N*m) at rest, PEDAL_K (N*m/rad) more as it goes down, so wound
#: PEDAL_TORQUE / PEDAL_K (148 degrees) at rest and 162 at its stop.  It must stay under 180 there: Isaac drives a
#: hinge toward its spring's target the shorter way round, so a spring wound past 180 turns the body the other way --
#: at 1.5 N*m/rad (wound 168 degrees) the pedal, pressed past 11.9 degrees, would stay down when let go, and one end
#: arm could press it, let go and hold its own valve (``tasks/validate.py`` checks every hinge).  The
#: pedal itself (PEDAL_MASS at its middle) adds 0.74 N*m; the prop in the pocket 4.6-4.7, together 5.3-5.5 against
#: 4.40-4.82 (it goes down to its stop); the distractor 2.35, together 3.0-3.1 against 4.40 (it does not move); the
#: heaviest thing the varied scenes put within an arm's reach, a 0.45 kg book, at the pedal's far end 1.3, together
#: 2.0 (it does not move)
PEDAL_TORQUE, PEDAL_K, PEDAL_MASS, PEDAL_DAMPING = 4.4, 1.7, 0.50, 0.30
#: the borrow version's weights: steel (7.85 g/cm3) WEIGHT_R round, the prop WEIGHT_H tall (2.4 kg), the distractor
#: half as tall (1.2 kg); a knob on a neck on top that the hand takes; they stand by the end arms, which is which
#: drawn from the seed
WEIGHT_R, WEIGHT_H, STEEL = 0.045, 0.0481, 7850.0
NECK_R, NECK_H, KNOB_R, KNOB_H = 0.009, 0.018, 0.016, 0.012
WEIGHT_XY = ((-0.34, -0.30), (0.34, -0.30))
#: the pipe's and the foot valve's places (x) move this much with the seed
JITTER = 0.008

#: a side's letter in a part's name (a USD prim's name takes no sign)
_EW, _SN = {-1: "w", 1: "e"}, {-1: "s", 1: "n"}
#: a body's z turned along x (the pipe), along -y (the gauge's face toward the south arm), along y (the hose)
ALONG_X, FACE_S, ALONG_Y = axis_quat((0, 1, 0), 90.0), axis_quat((1, 0, 0), 90.0), axis_quat((1, 0, 0), -90.0)


def _mass(h):
    return round(STEEL * math.pi * WEIGHT_R ** 2 * h, 3)


def _lever(side, x0, spring):
    """The ``side`` bleed valve's lever (-1 west, 1 east) on a pipe whose middle is at ``x0``: hinged on the valve body
    outside VALVE_X, pointing in along the pipe, its grip on a stem at its end."""
    name = "valve_w" if side < 0 else "valve_e"
    pivot = (x0 + side * (VALVE_X + PIVOT_OUT), PIPE_Y, PIPE_TOP + PIVOT_UP)
    inward = -side
    reach = PIVOT_OUT + GRIP_IN  # hinge to grip
    parts = [part("lever", "box", (reach + 0.012, LEVER_W, LEVER_T), (inward * (reach / 2), 0, 0)),
             part("grip_stem", "box", (0.010, 0.010, GRIP_UP - GRIP_W / 2 - LEVER_T / 2),
                  (inward * reach, 0, (GRIP_UP - GRIP_W / 2 + LEVER_T / 2) / 2)),
             part("grip", "box", (GRIP_L, GRIP_W, GRIP_W), (inward * reach, 0, GRIP_UP), color="red")]
    obj = compound(name, pivot, parts, "steel", mass=LEVER_MASS)
    # the grip goes down as the lever turns about +y by +deg on the west (its grip east of the hinge), -deg on the east
    joint = hinge(name + "_hinge", obj, pivot, sign=-side, axis=(0, 1, 0), open_max_deg=VALVE_MAX, stiffness=spring,
                  damping=LEVER_DAMPING, preload_deg=VALVE_PRELOAD_DEG)
    return obj, joint


def _pipe(x0):
    """The pipe and its blind flanges (one body, its z along x)."""
    parts = [part("pipe", "cylinder", (PIPE_R, PIPE_L), (0, 0, 0))]
    for side in (-1, 1):  # the body's z is world x
        parts += [part(f"flange_{_EW[side]}", "cylinder", (FLANGE_R, FLANGE_T),
                       (0, 0, side * (PIPE_L / 2 + FLANGE_T / 2))),
                  part(f"cap_{_EW[side]}", "cylinder", (FLANGE_R * 0.55, 0.006),
                       (0, 0, side * (PIPE_L / 2 + FLANGE_T + 0.003)), color="iron")]
    return compound("pipe", (x0, PIPE_Y, PIPE_AXIS), parts, "steel", dynamic=False, quat=ALONG_X)


def _fittings(x0, px):
    """What stands upright on and under the pipe: its stands, the valve bodies and the cheeks their levers' hinges run
    between, the gauge's stem, and the hose's riser from the foot valve (``px``) into the pipe's underside; the body's
    frame on the table under the pipe's middle."""
    top, under = PIPE_TOP - Z, PIPE_AXIS - PIPE_R - Z
    parts = []
    for side in (-1, 1):
        parts.append(part(f"stand_{_EW[side]}", "box", (0.06, 0.10, under + 0.012),
                          (side * STAND_X, 0, (under + 0.012) / 2), color="grey"))
        hx = side * (VALVE_X + PIVOT_OUT)
        lo, hi = top - 0.015, top + PIVOT_UP - LEVER_T / 2 - BODY_GAP
        parts.append(part(f"valve_body_{_EW[side]}", "cylinder", (BODY_R, hi - lo), (hx, 0, (lo + hi) / 2),
                          color="brass"))
        for dy in (-1, 1):  # the cheeks the lever's hinge runs between, 5 mm off the lever
            parts.append(part(f"cheek_{_EW[side]}_{_SN[dy]}", "box", (0.016, 0.004, PIVOT_UP + 0.006),
                              (hx, dy * (LEVER_W / 2 + 0.007), top + (PIVOT_UP + 0.006) / 2), color="steel"))
    stem = GAUGE_UP - GAUGE_R + 0.005
    parts += [part("gauge_stem", "cylinder", (0.008, stem + 0.01), (0, 0, top - 0.01 + (stem + 0.01) / 2),
                   color="brass"),
              part("riser", "cylinder", (HOSE_R, under + 0.01), (px - x0, 0, (under + 0.01) / 2), color="blue")]
    return compound("pipe_fittings", (x0, PIPE_Y, Z), parts, "steel", dynamic=False)


def _gauge(x0):
    """The pressure gauge, its face toward the south arm (its z along -y)."""
    parts = [part("case", "cylinder", (GAUGE_R, GAUGE_T), (0, 0, 0)),
             part("face", "cylinder", (GAUGE_R - 0.006, 0.002), (0, 0, GAUGE_T / 2 + 0.001), color="white"),
             # the needle at the test pressure, up and to the right (the body's y is up)
             part("needle", "box", (0.003, 0.034, 0.002), (0.012, 0.012, GAUGE_T / 2 + 0.003), -45.0, color="red")]
    return compound("gauge", (x0, PIPE_Y, PIPE_TOP + GAUGE_UP), parts, "steel", dynamic=False, quat=FACE_S)


def _hose(px):
    """The hose from the foot valve's housing along the table to under the pipe (its z along y)."""
    y0, y1 = PEDAL_N + 0.01, PIPE_Y
    return compound("hose", (px, (y0 + y1) / 2, Z + HOSE_R + 0.002), [part("hose", "cylinder", (HOSE_R, y1 - y0))],
                    "blue", dynamic=False, quat=ALONG_Y)


def _housing(px):
    """The foot valve's base plate and housing (fixed)."""
    yc, length = PEDAL_N - PEDAL_L / 2, PEDAL_L + 0.02
    parts = [part("base", "box", (PEDAL_W + 0.02, length, BASE_T), (0, 0, BASE_T / 2), color="grey"),
             part("housing", "box", (PEDAL_W - 0.02, PEDAL_L - 0.02, HOUSING_TOP - BASE_T),
                  (0, 0, BASE_T + (HOUSING_TOP - BASE_T) / 2))]
    return compound("pedal_base", (px, yc, Z), parts, "safety_yellow", dynamic=False)


def _pedal(px):
    """The foot valve's pedal, hinged along its north edge: built pressed level and turned REST_DEG up."""
    pivot = (px, PEDAL_N, HINGE_Z)
    top = PEDAL_T / 2
    ty = -(PEDAL_L - TOE_IN)
    parts = [part("pedal", "box", (PEDAL_W, PEDAL_L, PEDAL_T), (0, -PEDAL_L / 2, 0)),
             part("toe_bar", "box", (TOE_L, BAR, BAR), (0, ty, top + TOE_UP - BAR / 2), color="steel")]
    parts += [part(f"post_{_EW[s]}", "box", (0.012, 0.012, TOE_UP - BAR), (s * 0.04, ty, top + (TOE_UP - BAR) / 2),
                   color="steel") for s in (-1, 1)]
    a = INNER / 2 + RAIL_T / 2
    for n, (dx, dy, sx, sy) in {"n": (0, a, INNER + 2 * RAIL_T, RAIL_T), "s": (0, -a, INNER + 2 * RAIL_T, RAIL_T),
                                "e": (a, 0, RAIL_T, INNER), "w": (-a, 0, RAIL_T, INNER)}.items():
        parts.append(part(f"rail_{n}", "box", (sx, sy, RAIL_H), (dx, -POCKET + dy, top + RAIL_H / 2), color="steel"))
    body = compound("pedal", pivot, parts, "safety_yellow", mass=PEDAL_MASS, quat=axis_quat((1, 0, 0), -REST_DEG))
    # pressed, the south end goes down: about +x by +deg
    joint = hinge("pedal_hinge", body, pivot, sign=1, axis=(1, 0, 0), open_max_deg=REST_DEG,
                  stiffness=PEDAL_K, damping=PEDAL_DAMPING, preload_deg=math.degrees(PEDAL_TORQUE / PEDAL_K))
    return body, joint


def _weight(name, xy, h):
    """A steel test weight ``h`` tall, a knob on a neck on top."""
    parts = [part("body", "cylinder", (WEIGHT_R, h), (0, 0, 0)),
             part("neck", "cylinder", (NECK_R, NECK_H), (0, 0, (h + NECK_H) / 2)),
             part("knob", "cylinder", (KNOB_R, KNOB_H), (0, 0, h / 2 + NECK_H + KNOB_H / 2))]
    return compound(name, (*xy, Z + h / 2), parts, "iron", mass=_mass(h))


def _turned(inst, name, deg):
    """Body ``name``'s pose with its hinge ``deg`` from rest."""
    sc = inst.scene
    obj, h = sc.obj(name), next(h for h in sc.hinges if h.body == name)
    return rotate_about(obj.pose, h.pivot, axis_quat(h.axis, h.sign * deg))


def grip(inst, name, deg=0.0):
    """Where a valve lever's grip is (its middle) with the lever ``deg`` down."""
    return transform(_turned(inst, name, deg), inst.scene.obj(name).part("grip").pos)


def toe_bar(inst, deg=0.0):
    """Where the pedal's toe bar is with the pedal ``deg`` down."""
    return transform(_turned(inst, "pedal", deg), inst.scene.obj("pedal").part("toe_bar").pos)


def pocket(inst, deg=0.0, up=0.0):
    """The pocket's middle on the pedal's top, ``up`` over it along the pedal's normal, with the pedal ``deg`` down."""
    return transform(_turned(inst, "pedal", deg), (0, -POCKET, PEDAL_T / 2 + up))


def build(level: str, seed: int, variant: str = "exact"):
    spec = get("Con-04")
    rng = rng_for(spec, level, seed)
    spring = spec.knob.levels[level]
    x0 = jitter(rng, (0.0, 0.0), JITTER)[0]
    px = x0 + jitter(rng, (0.0, 0.0), JITTER)[0]
    lw, hw = _lever(-1, x0, spring)
    le, he = _lever(1, x0, spring)
    pedal, hp = _pedal(px)
    objects = [_pipe(x0), _fittings(x0, px), _gauge(x0), _hose(px), _housing(px), lw, le, pedal]
    regions = [zone("grip_w", transform(lw.pose, lw.part("grip").pos), (0.01,) * 3),
               zone("grip_e", transform(le.pose, le.part("grip").pos), (0.01,) * 3),
               zone("toe_bar", transform(pedal.pose, pedal.part("toe_bar").pos), (0.01,) * 3)]
    reach = [("west", "grip_w"), ("east", "grip_e"), ("south", "toe_bar")]
    out_of_reach = [("south", "grip_w"), ("south", "grip_e"), ("west", "grip_e"), ("east", "grip_w")]
    extra = {}
    if variant == "borrow":  # no south arm: a weight holds the pedal down; the other is half as heavy
        heights = [WEIGHT_H, WEIGHT_H / 2]
        random.Random(f"borrow/{spec.seed_id}/{level}/{seed}").shuffle(heights)
        names = ("weight_1", "weight_2")
        objects += [_weight(n, xy, h) for n, xy, h in zip(names, WEIGHT_XY, heights)]
        prop = names[heights.index(WEIGHT_H)]
        hp = dataclasses.replace(hp, propped=True)  # the clearance check: the pedal stays where the weight holds it
        for n, xy, h in zip(names, WEIGHT_XY, heights):
            regions.append(zone(f"{n}_top", (*xy, Z + h + NECK_H / 2), (0.01,) * 3))
        set_at = transform(pedal.pose, (0, -POCKET, PEDAL_T / 2 + WEIGHT_H + NECK_H / 2 + 0.004))
        regions.append(zone("weight_set", set_at, (0.01,) * 3))
        reach = [r for r in reach if r[0] != "south"] + \
            [(s, f"{n}_top") for s, n in zip(("west", "east"), names)] + \
            [("west", "weight_set"), ("east", "weight_set")]
        out_of_reach = [r for r in out_of_reach if r[0] != "south"]
        extra = {"borrow": {"props": [prop], "decoys": [n for n in names if n != prop]},
                 "weight_masses_kg": {n: _mass(h) for n, h in zip(names, heights)}}
    sc = scene(spec, objects, regions, hinges=(hw, he, hp))
    if variant == "borrow":
        sc = dataclasses.replace(sc, arms=tuple(a for a in sc.arms if a.slot != "south"))
    shut = Opened("valve_w", VALVE_SHUT) & Opened("valve_e", VALVE_SHUT)
    air = Opened("pedal", AIR_DEG)
    return make_instance(
        spec, level, seed, sc,
        instruction=("Hold the steel pipe at its test pressure for a second: both bleed valves shut while the foot "
                     "valve lets air in."),
        success=HoldsFor(shut & air, 1.0),
        milestones={"air_in": Eventually(air), "bleeds_shut": Eventually(shut),
                    "pressure_held": HoldsFor(shut & air, 1.0)},
        reach=reach, out_of_reach=out_of_reach, **extra)


#: the keyframe `mrbench render` shows between start and goal: the bleed valves held shut (the borrow version: the
#: weight in the pocket, the pedal down under it); the goal is the pressure held, all three down
MIDDLE = 0


def keyframes(inst):
    """The bleed valves held shut, then the pedal down too: the pressure held.  Borrow version: the weight set in the
    pocket holds the pedal down first, then the valves."""
    levers = {n: _turned(inst, n, VALVE_HOLD) for n in ("valve_w", "valve_e")}
    if inst.variant != "borrow":
        return [levers, {"pedal": _turned(inst, "pedal", PEDAL_HOLD)}]
    prop = inst.metadata["borrow"]["props"][0]
    h = inst.scene.obj(prop).part("body").size[1]
    pedal = _turned(inst, "pedal", REST_DEG)  # on its stop under the weight
    return [{"pedal": pedal, prop: (*pocket(inst, REST_DEG, h / 2), *pedal[3:])}, levers]


#: a hand pointing down, its fingers closing along y: across a lever's grip, across the pedal's toe bar; along x:
#: across a weight's neck (opening along y over the pocket, a finger comes within 0.4 cm of the toe bar)
DOWN_Y = tool_quat((0, 0, -1), (0, 1, 0))
DOWN_X = tool_quat((0, 0, -1), (1, 0, 0))


def _grip(inst, name, deg=0.0, q=DOWN_Y):
    """The hand (``q``) on ``name``'s grip (a lever) or toe bar (the pedal), turned with it as it goes ``deg`` down."""
    p = toe_bar(inst) if name == "pedal" else grip(inst, name)
    h = next(h for h in inst.scene.hinges if h.body == name)
    return rotate_about((*p, *q), h.pivot, axis_quat(h.axis, h.sign * deg))


#: who holds what in the reference (the surplus routes give the north arm one of the levers)
REFERENCE = {"valve_w": "west", "valve_e": "east", "pedal": "south"}


def _hold(inst, who, settle=0.0, hands=None):
    """Every body in ``who`` (body -> arm) taken and held down together for the test, then let go; ``hands`` (body ->
    tool orientation) where a hand does not point straight down."""
    hold = {"valve_w": VALVE_HOLD, "valve_e": VALVE_HOLD, "pedal": PEDAL_HOLD}
    width = {"valve_w": GRIP_W, "valve_e": GRIP_W, "pedal": BAR}
    q = {n: (hands or {}).get(n, DOWN_Y) for n in who}
    above = [Move(a, back_off(_grip(inst, n, q=q[n]), 0.10)) for n, a in who.items()]
    at = [Move(a, _grip(inst, n, q=q[n])) for n, a in who.items()]
    press = [Move(a, (lambda s, n=n: _grip(inst, n, hold[n] * s, q[n]))) for n, a in who.items()]
    up = [Move(a, back_off(_grip(inst, n, q=q[n]), 0.12)) for n, a in who.items()]
    return [
        phase(2.5, "the hands come over the valves' grips and the pedal's toe bar", *above,
              *[Open(a, 0.06) for a in who.values()]),
        phase(1.5, "", *at),
        phase(0.4, "and take them", *[Grip(a, n, width[n]) for n, a in who.items()]),
        phase(2.0, "they press them down: the bleed valves shut, the foot valve lets air in", *press),
        phase(2.0 + settle, "and hold them: the pipe holds its test pressure"),
        phase(0.4, "they let go: the springs lift the levers and the pedal", *[Release(a) for a in who.values()]),
        phase(1.5, "", *up),
        phase(1.0, "done"),
    ]


#: where the hand goes once it has let the weight go: this far back toward its base and up (m), clear at both levels,
#: prop and distractor (farther back the fingers come within 1 cm of the knob or the toe bar)
RETREAT = (0.04, 0.08)


def _weight_phases(inst, prop):
    """The end arm on the prop's side takes it by its knob, carries it over the pedal and lets it down into the
    pocket; the pedal goes down under it."""
    b = inst.scene.obj(prop)
    a = "west" if b.pos[0] < 0 else "east"
    h = b.part("body").size[1]
    take = (b.pos[0], b.pos[1], Z + h + NECK_H / 2, *DOWN_X)
    # let go level with its bottom 4 mm over the pedal's top under its uphill (south) edge (the pedal at rest)
    over = pocket(inst)
    drop = (over[0], over[1], over[2] + WEIGHT_R * math.tan(math.radians(REST_DEG)) + 0.004 + h + NECK_H / 2,
            *DOWN_X)
    high = Z + 0.29  # (the tool point at 0.35 over the table, over the pocket, is out of an end arm's reach)
    return [
        phase(2.0, f"the {a} arm reaches for a weight", Move(a, offset(take, dz=0.12)), Open(a, 0.05)),
        phase(1.5, "", Move(a, take)),
        phase(0.4, "takes it by its knob", Grip(a, prop, 2 * NECK_R)),
        phase(1.5, "lifts it", Move(a, (*take[:2], high, *DOWN_X))),
        phase(3.0, "carries it over the foot valve", Move(a, (*drop[:2], high, *DOWN_X))),
        phase(2.0, "lets it down into the pocket on the pedal", Move(a, drop)),
        phase(0.4, "and lets go: the pedal goes down under it", Release(a)),
        # up and back toward its own base: straight up 12 cm, an end arm's hand is out of reach where the pedal
        # stands 7-13 mm toward the other end
        phase(1.5, "", Move(a, offset(drop, dx=RETREAT[0] * (-1 if a == "west" else 1), dz=RETREAT[1]))),
        phase(2.0, "", Move(a, offset(home_pose(inst, a), dz=0.05))),
    ]


def script(inst, who=None, hands=None):
    """The scripted reference (privileged grips): the end arms each take their valve lever's grip and the south arm
    the pedal's toe bar; together they press them down and hold them while the pipe holds its pressure, then let go.
    Borrow version: an end arm first sets the weight in the pedal's pocket, then the end arms hold the levers.  ``who``
    (body -> arm) gives the holds to other arms and ``hands`` (body -> tool orientation) turns their hands."""
    if inst.variant != "borrow":
        return _hold(inst, who or REFERENCE, hands=hands)
    prop = inst.metadata["borrow"]["props"][0]
    return _weight_phases(inst, prop) + _hold(inst, who or {"valve_w": "west", "valve_e": "east"}, settle=1.0)


def jobs(inst):
    """What the task asks (``tasks/jobs.py``): both bleed valves' levers held down and the foot valve's pedal held
    down, all at once.  Borrow version: the right weight, set in the pocket, holds the pedal down by itself."""
    from ..jobs import Job, Jobs, Thing

    slack = 0.02
    test = [Job("shut the west bleed valve: hold its lever down by its grip",
                sites=(grip(inst, "valve_w"), grip(inst, "valve_w", VALVE_HOLD)), group="test", slack=slack),
            Job("shut the east bleed valve: hold its lever down by its grip",
                sites=(grip(inst, "valve_e"), grip(inst, "valve_e", VALVE_HOLD)), group="test", slack=slack),
            Job("let air in: hold the foot valve's pedal down by its toe bar",
                sites=(toe_bar(inst), toe_bar(inst, PEDAL_HOLD)), group="test", slack=slack)]
    things = ()
    if inst.variant == "borrow":
        b = inst.metadata["borrow"]
        prop, decoy = b["props"][0], b["decoys"][0]
        m = inst.metadata["weight_masses_kg"]
        things = (Thing(prop, does=(test[2].name,),
                        placing=(Job(f"set {prop} in the pocket on the foot valve's pedal",
                                     sites=(f"{prop}_top", "weight_set"), slack=slack),),
                        why=f"{m[prop]:.1f} kg in the pocket: its weight holds the pedal down against its spring"),
                  Thing(decoy, why=f"{m[decoy]:.1f} kg: in the pocket the pedal does not go down (its spring holds "
                                   f"{PEDAL_TORQUE:.1f} N*m at rest, the weight gives "
                                   f"{m[decoy] * 9.81 * POCKET:.1f})"))
    return Jobs(tuple(test), things)
