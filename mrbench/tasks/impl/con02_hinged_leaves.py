"""Con-02 hinged leaves: two leaves hinged on posts must both be held level in the air while a third
hand pins them together; pinned, they hold each other level.

Each leaf is hinged at its outer end on a fixed post, so its slots come into line with the other's only in the air, at
the joint, and only while both are held level: let go, the west leaf hangs down and the east leaf springs up.  (Two
loose rods could be laid on the table with their eyelets in line, pinned there and lifted together by two arms.)

Why three arms, and not two taking turns: a key in one leaf's slot, its pin held by a hand, holds that leaf level
by itself -- so if one arm could hold both leaves, it could hold the first while the pin went into it, then let go
and bring the second (two arms).  Here no arm reaches both: the joint is inside a guard, and
outside it each leaf's bar is within its own end arm's reach only.
"""

from __future__ import annotations

import dataclasses
import math
import random

from ...engines.base import axis_quat, quat_mul, transform
from ...policies.script import Grip, Move, Open, Release, back_off, home_pose, offset, phase, tool_quat
from ...scoring.predicates import Eventually, HoldsFor, Level, ThroughSlots
from ..loader import get
from ._common import Z, at, box, compound, hinge, jitter, make_instance, part, rng_for, rotate_about, scene, zone

#: the joint: where the slots meet, in the air over the table's north half, and how far each leaf reaches from its
#: hinge to its slots' middle
JOINT, LEAF_L = (0.0, 0.29, Z + 0.22), 0.50
#: a clevis.  The west leaf ends in a tongue, the east one in a fork whose two cheeks straddle it, all three with a slot
#: through them along y.  The pin pins all three: the tongue, weighed down, pushes the key down in the middle; the
#: cheeks, sprung up, push it up evenly either side of it -- no couple on the key, so the pinned leaves stay level
#: (with two slots side by side, one each, the two pushes 12 cm apart tilted the key in its slots and the east leaf
#: rose 7 degrees when let go)
TONGUE_T, CHEEK_T, CHEEK_GAP = 0.085, 0.085, 0.012
CHEEK_Y = TONGUE_T / 2 + CHEEK_GAP + CHEEK_T / 2
#: a slot's block, along the leaf and up; its hole is the pin's key and the knob's clearance each side
RING = 0.10
#: along the leaf: the tongue's neck (between its bar and its slot block), the fork's yoke (joining its cheeks to its
#: bar), and how far the cheeks reach past their slot blocks toward the yoke
NECK, YOKE, CHEEK_REACH = 0.09, 0.09, 0.02
#: a leaf's bar: square, from just off its hinge to its neck or yoke
BAR, BAR_FROM = 0.04, 0.03
#: how far each leaf turns from level when let go (its stop): the west one down (its tip 3 cm over the table), the
#: east one up (its tip 5 cm under the guard's top)
REST_DEG = 15.0
#: the east leaf's spring: N*m/rad and how far past its stop it pulls.  (Leaves twice as heavy, on a spring three
#: times as stiff, put the hands holding them level 7-10 mm off their plan, and the slots with them: the key, 6 mm
#: clear, met a slot's edge.)
SPRING, PRELOAD_DEG, DAMPING = 0.6, 60.0, 0.2
LEAF_MASS = 0.15
#: the guard round the joint: its inside (half along x, half along y, height from the table), its walls' thickness.
#: The bars come out through slots in its west and east walls, 5 mm clear of them (the slots as tall as a bar sweeps
#: between its stop and level); the pin goes in through an opening in its south wall, 8 cm square round the joint's
#: line, 1 cm in front of the south cheek's face -- which is all a finger poked through it meets: never a top or an
#: underside to hold a leaf by (with the wall 3 cm off, a fingertip tilted up could have reached the cheek's top edge)
GUARD, GUARD_H, WALL = (0.20, CHEEK_Y + CHEEK_T / 2 + 0.01), 0.46, 0.01
#: a bar's slot leaves it this much room either side, and this much above and below its sweep (a bar's corner, tilted
#: at its stop, came 1.5 mm from the slot's edge with 5 mm)
SLOT_ROOM, SLOT_ROOM_Z, OPENING = 0.005, 0.012, 0.08
#: the pin: a flat key (across, along, thick) with a head at its south end; it lies on the table in front of the
#: south arm, south of the guard, along y, its head toward that arm
KEY, HEAD, PIN_MASS = (0.034, 0.37, 0.014), (0.044, 0.06, 0.04), 0.06
PIN_XY = (0.0, -0.17)
#: a leaf counts as level within this many degrees
LEVEL_DEG = 3.0
#: where a leaf is held: on its bar outside the guard, this far from its hinge, a hand's width and more off the guard
#: (at 24 cm, a hand's camera came within 6-9 mm of the guard's wall coming in)
HOLD_AT = 0.21
#: the posts' brackets: thin plates beside each leaf's hinge end, standing on the table
PLATE_T, PLATE_W = 0.012, 0.07
#: the borrow version (no west arm): a wedge the south arm pushes north along the corridor between the west post's
#: plate (x <= -0.465) and the guard (x >= -0.21), under the west leaf's bar outside the guard.  A block whose flat top
#: is WEDGE_TOP over the table -- 0.8 mm under the bar's underside when the leaf is level: the leaf rests 0.2-0.3
#: degrees low on it, its tongue's slot 1.5-2.5 mm low, inside the slots' room (level with the bar's underside, a
#: wedge pushed in a little askew -- 2 cm short, turned 4.6 degrees -- lifts the leaf past its stop, the tongue's slot
#: 3.8 mm high, and the key meets it at the hard level) -- with a ramp to the north, a plate
#: RAMP_T thick on a foot, RAMP_DEG up from its front edge RAMP_FRONT over the table (under the hanging bar, 14-16 cm
#: up where the wedges go) to the top, so that, pushed north, it lifts the bar onto the top: on a 25-degree slope the
#: push is about 1.3 times the lift (mu 0.5), and the hand pushing comes out a few millimetres off its way (on a
#: rounded nose, which the bar meets 26-48 degrees up, the push is 2.6-10 times the lift); and a handle out south at
#: HANDLE_Z, long enough that the south arm still reaches it with the top under the bar (within the design rule's
#: 0.77 m: with a 24 cm handle it is 0.79-0.83 m away).  (The engine turns a part only about
#: the vertical, so the wedge is built on its side -- its frame turned a quarter about y -- for the ramp to rise along
#: its way.)  The distractor is the same with its top WEDGE_LOW: the leaf then rests 4 degrees low or more, its slot
#: 3.5 cm or more out of line (the slots leave 3-6 mm).  The two stand side by side in the corridor (named wedge_1 and
#: wedge_2, west to east), which is which drawn from the seed (IK: the south arm reaches either handle all the way),
#: 16 and 24 cm from the hinge, where a top 1 mm off moves the tongue's slot 3.1 and 2.1 mm (5 mm at 10 cm, too much
#: for the hard level)
WEDGE_TOP, WEDGE_LOW, WEDGE_W, WEDGE_D, WEDGE_MASS = 0.1992, 0.180, 0.06, 0.10, 0.4
HANDLE_L, HANDLE_T, HANDLE_Z = 0.33, 0.02, 0.08
#: the handle rises HANDLE_RISE_DEG from the block's back, as a dock wheel chock's placing handle does, to a short
#: upright with a T-grip across its top.  It reaches HANDLE_L south, the grip HANDLE_RISE + POST_H high -- nearer the
#: south arm's shoulder with the wedge pushed in.  Handle and upright are yellow, as the block; the handle leaves the block through a steel plate
#: (BRACKET: thickness, height) on its south face.  The T-grip is round rubber HANDLE_T across (the hand takes it as
#: it would a crossbar: ``Grip(..., HANDLE_T - 0.003)``), GRIP_W long, centred on the upright; the upright, POST_H
#: from the handle's top end to the grip's middle, keeps the fingers (closing along y round the grip) 2 cm over the
#: handle.  (Steeper, 45 degrees, the grip would be 0.46 m over the table: the south arm, pointing down, reaches
#: neither over the wide distractor's grip at WIDE_X nor 22 cm over the grip pushed in.)  A black rubber TREAD covers
#: the flat top, within WEDGE_TOP
HANDLE_RISE_DEG, POST_H, TREAD = 30.0, 0.04, 0.003
HANDLE_RISE = HANDLE_L * math.tan(math.radians(HANDLE_RISE_DEG))
BRACKET = (0.006, 0.06)
RAMP_DEG, RAMP_FRONT, RAMP_T = 25.0, 0.12, 0.01
WEDGE_XS, WEDGE_Y0 = (-0.34, -0.26), -0.03
#: how far the wedge is pushed: till the north end of its flat top (where the ramp meets it) is under the bar's
#: middle, and no further.  The west leaf's hinge stops it at level, so a bar lifted level is held between the top and
#: that stop: pushed on 7 cm under it, to the block's middle, the wedge jams and the arm pushing it gives way
#: sideways
WEDGE_PUSH = JOINT[1] - WEDGE_D / 2 - WEDGE_Y0
#: the hand takes the T-grip (along x, GRIP_W long) from above, its fingers closing along y: closing across the handle
#: itself, the hand's broad side (20 cm along its fingers' travel) comes within a centimetre of the other wedge beside
#: it.  GRIP_W leaves 2 cm between the two wedges' grips, 8 cm apart (on a T-grip straight on the handle a finger
#: comes down on the handle -- hence the upright)
GRIP_W = 0.06
#: the borrow version's other distractor: the prop's wedge WEDGE_WIDE wide instead of WEDGE_W, its top as
#: high and its mass the same -- the larger of the two, so that "take the larger one" is wrong as often as right.  Under
#: the west leaf's bar the corridor between the west post's plate (its east face at x -0.465, its y 0.254-0.266) and
#: the guard (its west face at x -0.21, its y 0.13-0.45) is CORRIDOR wide, and the wedge, whose top must come under the
#: bar (x -0.47 to -0.21, y 0.27-0.31), crosses both their bands on its way north: a wedge wider than CORRIDOR meets the
#: plate or the guard before its ramp reaches the bar.  Turned, it is no narrower: its block and ramp are
#: WEDGE_D + their run (26.9 cm) long, and with the handle 61 cm, so at any turn it is more than 26.9 cm across x.
#: Which distractor a scene has: the wide one where seed + level (easy 0, hard 1) is even -- half the scenes of
#: either level -- else the low one.  With the wide one the prop stands at WEDGE_XS[0], in the corridor's line, and the
#: wide wedge at WIDE_X, west of the post's plate (the corridor holds only one of them), both WIDE_Y0 from the guard's
#: south face to their blocks' middles: either place then holds either wedge, the wide ramp's tip 2 cm short
#: of the guard (at WEDGE_Y0, the wide one in the corridor's line would stand in the guard's south wall)
CORRIDOR, WEDGE_WIDE, WIDE_X, WIDE_Y0 = -0.21 - (-0.465), 0.32, -0.66, -0.11


#: with the wide distractor, the wedge standing in the corridor's line when the other one is to be pushed is first moved
#: out of its way: taken by its crossbar, carried east over the pin and set down with its block's
#: middle at PARK_XY, east of the pin (the wide one's west side 19 cm from the pin's line: the hand on the pin's head is
#: 20 cm across) and as far south as the corridor's place (its ramp's tip 2 cm short of the guard); the wedge to be
#: pushed is then slid along the table from WIDE_X into the corridor's line.  The moved one is lifted LIFT_OVER at first
#: -- the hand pointing down then passes 0.2 m in front of the south arm's base, where it has no IK solution at 0.89-0.98
#: m up; at 1.08 it has (IK) -- so it is carried up there past the base (till its crossbar is CARRY_PAST east of the
#: middle), then let down to LIFT_PIN (over the pin's head, 4 cm) and on to its place, while the east arm waits 27 cm
#: over its home (EAST_AWAY: at home its hand is over the wedge's place; let down at x 0.15 the hand has no IK solution)
PARK_XY, LIFT_OVER, LIFT_PIN, CARRY_PAST, EAST_AWAY = (0.35, -0.11), 0.25, 0.06, 0.20, (0.0, 0.0, 0.27)


def decoy_kind(level: str, seed: int) -> str:
    """Which distractor the borrow version's scene has: "wide" (the larger) or "low" (the smaller)."""
    return "wide" if (seed + (level == "hard")) % 2 == 0 else "low"


def hole_size(clearance: float) -> tuple[float, float]:
    """A slot's hole (along the leaf, up): the key's cross-section and ``clearance`` each side."""
    return KEY[0] + 2 * clearance, KEY[2] + 2 * clearance


def _block(name, x0, x1, y, thick, hole=None, slot_x=0.0):
    """A block from x0 to x1 along the leaf, ``thick`` along y at ``y``, RING tall; with ``hole`` (hx, hz), a slot
    through it along y at ``slot_x``, the block built round it."""
    lo, hi = min(x0, x1), max(x0, x1)
    if hole is None:
        return [part(name, "box", (hi - lo, thick, RING), ((lo + hi) / 2, y, 0.0), color="grey")]
    hx, hz = hole
    cap = (RING - hz) / 2
    parts = [part(f"{name}_top", "box", (hx, thick, cap), (slot_x, y, hz / 2 + cap / 2), color="grey"),
             part(f"{name}_bottom", "box", (hx, thick, cap), (slot_x, y, -hz / 2 - cap / 2), color="grey")]
    for k, (a, b) in (("a", (lo, slot_x - hx / 2)), ("b", (slot_x + hx / 2, hi))):
        parts.append(part(f"{name}_{k}", "box", (b - a, thick, RING), ((a + b) / 2, y, 0.0), color="grey"))
    return parts


def _leaf(name: str, side: int, clearance: float):
    """side -1: the west leaf (the tongue), +1: the east one (the fork).  The body's frame is at its hinge, its x
    along the leaf, y and z the world's when level; built level, then turned to its stop."""
    hole = hole_size(clearance)
    s = -side  # from the hinge toward the joint: +x for the west leaf, -x for the east
    slot, tip = s * LEAF_L, s * (LEAF_L + RING / 2)
    if side < 0:  # the tongue: its slot block, the neck, then the bar
        near = LEAF_L - RING / 2
        parts = _block("tongue", s * near, tip, 0.0, TONGUE_T, hole, slot)
        parts += _block("neck", s * (near - NECK), s * near, 0.0, TONGUE_T)
        bar_end = near - NECK
    else:  # the fork: two cheeks reaching CHEEK_REACH past their slot blocks, the yoke across them, then the bar
        near = LEAF_L - RING / 2 - CHEEK_REACH
        parts = []
        for k, cy in (("s", -CHEEK_Y), ("n", CHEEK_Y)):
            parts += _block(f"cheek_{k}", s * near, tip, cy, CHEEK_T, hole, slot)
        parts += [part("yoke", "box", (YOKE, 2 * CHEEK_Y + CHEEK_T, RING), (s * (near - YOKE / 2), 0.0, 0.0),
                       color="grey")]
        bar_end = near - YOKE
    bar_len = bar_end - BAR_FROM
    parts.append(part("bar", "box", (bar_len, BAR, BAR), (s * (BAR_FROM + bar_len / 2), 0.0, 0.0)))
    pivot = (JOINT[0] + side * LEAF_L, JOINT[1], JOINT[2])
    color = "blue" if side < 0 else "orange"
    level = compound(name, pivot, parts, color, mass=LEAF_MASS)
    # both stops are the level leaf turned REST_DEG the same way about +y: the west tip down, the east tip up
    rest = rotate_about(level.pose, pivot, axis_quat((0.0, 1.0, 0.0), REST_DEG))
    leaf = compound(name, rest[:3], parts, color, mass=LEAF_MASS, quat=rest[3:])
    # opening = turning back toward level (-y); the west leaf has no spring (its weight keeps it on its stop), the
    # east one a spring that holds it up
    kw = dict(stiffness=SPRING, damping=DAMPING, preload_deg=PRELOAD_DEG) if side > 0 else \
        dict(stiffness=0.0, damping=DAMPING, preload_deg=0.0)
    return leaf, hinge(f"{name}_hinge", leaf, pivot, sign=-1, open_max_deg=REST_DEG, axis=(0.0, 1.0, 0.0), **kw)


def _bracket(name: str, side: int):
    """A post's plate, beside the leaf's hinge end (south of it), up past the hinge."""
    px, py, pz = JOINT[0] + side * LEAF_L, JOINT[1], JOINT[2]
    y = py - (BAR / 2 + 0.004 + PLATE_T / 2)
    h = pz + 0.03 - Z
    return box(name, (px, y), (PLATE_W, PLATE_T, h), "grey", z=Z + h / 2, dynamic=False)


def _wall_with_hole(name, axis, at_, span, hole_lo, hole_hi, height):
    """A wall standing on the table, square to ``axis`` ("x": a west or east wall, spanning y; "y": the south wall,
    spanning x) at ``at_``, from span[0] to span[1] along it, ``height`` tall, with a hole between hole_lo and
    hole_hi -- each (along the wall, height above the table) -- built as four slabs round it."""
    (a0, a1), (h0, z0), (h1, z1) = span, hole_lo, hole_hi
    pieces = [((a0, a1), (0.0, z0)), ((a0, a1), (z1, height)), ((a0, h0), (z0, z1)), ((h1, a1), (z0, z1))]
    out = []
    for k, ((u0, u1), (v0, v1)) in enumerate(pieces):
        if u1 - u0 < 1e-6 or v1 - v0 < 1e-6:
            continue
        mid, size = (u0 + u1) / 2, u1 - u0
        xy, dims = ((at_, mid), (WALL, size, v1 - v0)) if axis == "x" else ((mid, at_), (size, WALL, v1 - v0))
        out.append(box(f"{name}_{k}", xy, dims, "white", z=Z + (v0 + v1) / 2, dynamic=False, glass=True))
    return out


def _guard():
    """The guard round the joint: west and east walls with a slot for each bar, a south wall with the pin's opening,
    a north wall and a top.  Inside it every leaf's neck, yoke and slot blocks move clear of it, stop to level."""
    gx, gy = GUARD
    x, y = JOINT[0], JOINT[1]
    jz = JOINT[2] - Z
    sweep = (LEAF_L - gx) * math.sin(math.radians(REST_DEG))  # how far a bar moves up or down at a wall
    walls = []
    span_y = (y - gy - WALL, y + gy + WALL)
    for side, name in ((-1, "guard_w"), (1, "guard_e")):
        lo, hi = (jz - sweep, jz) if side < 0 else (jz, jz + sweep)
        hole_lo = (y - BAR / 2 - SLOT_ROOM, lo - BAR / 2 - SLOT_ROOM_Z)
        hole_hi = (y + BAR / 2 + SLOT_ROOM, hi + BAR / 2 + SLOT_ROOM_Z)
        walls += _wall_with_hole(name, "x", x + side * (gx + WALL / 2), span_y, hole_lo, hole_hi, GUARD_H)
    walls += _wall_with_hole("guard_s", "y", y - gy - WALL / 2, (x - gx, x + gx), (x - OPENING / 2, jz - OPENING / 2),
                             (x + OPENING / 2, jz + OPENING / 2), GUARD_H)
    walls.append(box("guard_n", (x, y + gy + WALL / 2), (2 * gx, WALL, GUARD_H), "white", z=Z + GUARD_H / 2,
                     dynamic=False, glass=True))
    walls.append(box("guard_top", (x, y), (2 * (gx + WALL), 2 * (gy + WALL), WALL), "white", z=Z + GUARD_H + WALL / 2,
                     dynamic=False, glass=True))
    return walls


def _wedge(name: str, x: float, top: float, width: float = WEDGE_W, y0: float = WEDGE_Y0):
    """A borrow version's wedge (``top`` high, ``width`` across), its frame on the table under the middle of its block at
    (x, y0), turned a quarter about y so that its ramp, turned about the frame's vertical, rises toward the block along
    the way it is pushed; its handle rises south from the block's back to a T-grip, centred."""
    def w(dx, dy, dz):  # a world offset from the frame, in the turned frame
        return (-dz, dy, dx)

    a = math.radians(RAMP_DEG)
    run = (top - RAMP_FRONT) / math.tan(a)  # the ramp's length along the way, north of the block
    # its top face from the block's top edge down to RAMP_FRONT; the plate RAMP_T thick under it, on a foot
    cy, cz = WEDGE_D / 2 + run / 2, (top + RAMP_FRONT) / 2
    foot = RAMP_FRONT - RAMP_T
    end_y, end_z = -WEDGE_D / 2 - HANDLE_L, HANDLE_Z + HANDLE_RISE  # the handle's top end, under the grip
    parts = [part("block", "box", (top - TREAD, WEDGE_D, width), w(0.0, 0.0, (top - TREAD) / 2)),
             part("tread", "box", (TREAD, WEDGE_D, width), w(0.0, 0.0, top - TREAD / 2), color="rubber"),
             part("ramp", "box", (RAMP_T, run / math.cos(a), width),
                  w(0.0, cy - math.sin(a) * RAMP_T / 2, cz - math.cos(a) * RAMP_T / 2), yaw=-RAMP_DEG),
             part("foot", "box", (foot, run, width), w(0.0, WEDGE_D / 2 + run / 2, foot / 2)),
             part("bracket", "box", (BRACKET[1], BRACKET[0], width), w(0.0, -WEDGE_D / 2 - BRACKET[0] / 2, HANDLE_Z),
                  color="steel"),
             # (a turn about the frame's vertical, here the world's x: the handle rises south, as the ramp rises south)
             part("handle", "box", (HANDLE_T, HANDLE_L / math.cos(math.radians(HANDLE_RISE_DEG)), HANDLE_T),
                  w(0.0, -WEDGE_D / 2 - HANDLE_L / 2, HANDLE_Z + HANDLE_RISE / 2), yaw=-HANDLE_RISE_DEG),
             # the upright (along the frame's x, the world's vertical) and the round grip across its top (a cylinder's
             # axis is the frame's z, the world's x).  (Crossbars 10 cm long and centred, 8 cm apart, would overlap 2 cm,
             # and pushing one wedge would take the other along.)
             part("post", "box", (POST_H, HANDLE_T, HANDLE_T), w(0.0, end_y, end_z + POST_H / 2)),
             part("grip", "cylinder", (HANDLE_T / 2, GRIP_W), w(0.0, end_y, end_z + POST_H), color="rubber")]
    return compound(name, (x, y0, Z), parts, "safety_yellow", mass=WEDGE_MASS, quat=axis_quat((0.0, 1.0, 0.0), 90.0))


def wedge_push(y0: float = WEDGE_Y0) -> float:
    """How far a wedge standing with its block's middle at ``y0`` is pushed: till the north end of its flat top is under
    the bar's middle (WEDGE_PUSH from WEDGE_Y0)."""
    return JOINT[1] - WEDGE_D / 2 - y0


def wedge_grip(x: float, pushed: bool = False, y0: float = WEDGE_Y0):
    """Where the hand takes a wedge standing at (x, y0) by its handle's T-grip, before (or after) the push."""
    y = y0 - WEDGE_D / 2 - HANDLE_L + (wedge_push(y0) if pushed else 0.0)
    return (x, y, Z + HANDLE_Z + HANDLE_RISE + POST_H)


def wedge_places(level: str, seed: int, prop_first: bool):
    """The borrow version's two wedges, (x, y0, top, width) each, named wedge_1 and wedge_2 west to east, and
    which of them is the prop (0 or 1).  With the low distractor (``decoy_kind``) they stand side by side in the
    corridor's line (WEDGE_XS), the prop in the first place if ``prop_first``; with the wide one, one west of the post's
    plate and one in the corridor's line (WIDE_X, WEDGE_XS[0], at WIDE_Y0), the prop the western one if
    ``prop_first`` (then the wide one stands in its way and is moved first: ``script``)."""
    if decoy_kind(level, seed) == "low":
        tops = (WEDGE_TOP, WEDGE_LOW) if prop_first else (WEDGE_LOW, WEDGE_TOP)
        return [(x, WEDGE_Y0, t, WEDGE_W) for x, t in zip(WEDGE_XS, tops)], (0 if prop_first else 1)
    widths = (WEDGE_W, WEDGE_WIDE) if prop_first else (WEDGE_WIDE, WEDGE_W)
    return [(x, WIDE_Y0, WEDGE_TOP, wd) for x, wd in zip((WIDE_X, WEDGE_XS[0]), widths)], (0 if prop_first else 1)


def _pin(xy):
    parts = [part("key", "box", KEY), part("head", "box", HEAD, (0.0, -(KEY[1] + HEAD[1]) / 2, HEAD[2] / 2 - KEY[2] / 2),
                                           color="grey")]
    return compound("pin", (xy[0], xy[1], Z + KEY[2] / 2), parts, "grey", mass=PIN_MASS)


def slots(clearance: float):
    """The three slots the pin must go through (body, middle in its frame, hole): the tongue's and both cheeks'."""
    hx, hz = hole_size(clearance)
    return (("leaf_w", (LEAF_L, 0.0, 0.0), hx, hz), ("leaf_e", (-LEAF_L, -CHEEK_Y, 0.0), hx, hz),
            ("leaf_e", (-LEAF_L, CHEEK_Y, 0.0), hx, hz))


def level_pose(inst, name):
    """A leaf's pose held level (its hinge's angle at its stop's opposite end)."""
    h = next(h for h in inst.scene.hinges if h.body == name)
    return rotate_about(inst.scene.obj(name).pose, h.pivot, axis_quat(h.axis, h.sign * h.open_max_deg))


def build(level: str, seed: int, variant: str = "exact"):
    spec = get("Con-02")
    rng = rng_for(spec, level, seed)
    clearance = spec.knob.levels[level] / 1000
    leaf_w, hinge_w = _leaf("leaf_w", -1, clearance)
    leaf_e, hinge_e = _leaf("leaf_e", +1, clearance)
    pin = _pin(jitter(rng, PIN_XY, 0.01))
    sl = slots(clearance)
    x, y, z = JOINT
    gx = GUARD[0] + WALL
    regions = [zone("joint", JOINT, (RING / 2, CHEEK_Y + CHEEK_T / 2, RING / 2)),
               # where the hand sliding the pin ends: on its head, the key home (it never reaches the joint itself)
               zone("head_home", (x, PIN_HOME_Y - KEY[1] / 2 - HEAD[1] / 2, z), (0.01,) * 3)]
    for name, side in (("leaf_w", -1), ("leaf_e", 1)):
        # where the leaf is held level, and its bar's end at the guard: the nearest of it to any other arm
        regions.append(zone(f"{name}_grip", (x + side * (LEAF_L - HOLD_AT), y, z + BAR / 2), (0.01,) * 3))
        regions.append(zone(f"{name}_bar_in", (x + side * (gx + 0.01), y, z + BAR / 2), (0.01,) * 3))
    objects = [leaf_w, leaf_e, _bracket("bracket_w", -1), _bracket("bracket_e", 1), *_guard(), pin]
    reach = [("west", "leaf_w_grip"), ("east", "leaf_e_grip"), ("south", "pin"), ("south", "head_home")]
    # each leaf is held only by its own end arm: no other arm reaches its bar where it leaves the guard
    out_of_reach = [("east", "leaf_w_bar_in"), ("south", "leaf_w_bar_in"), ("west", "leaf_e_bar_in"),
                    ("south", "leaf_e_bar_in")]
    extra = {}
    if variant == "borrow":  # no west arm: a wedge holds the west leaf level, and a lower one that cannot
        # named for where they stand, west to east (a name must not tell which one does it); which is the prop is
        # drawn from the seed, on its own stream (the exact configuration's scene is unchanged)
        tops = [WEDGE_TOP, WEDGE_LOW]
        random.Random(f"borrow/{spec.seed_id}/{level}/{seed}").shuffle(tops)  # the low distractor's side
        places, k = wedge_places(level, seed, tops[0] == WEDGE_TOP)
        names = ("wedge_1", "wedge_2")
        objects += [_wedge(n, x, top, width, y0) for n, (x, y0, top, width) in zip(names, places)]
        prop = names[k]
        regions += [zone(f"{n}_grip", wedge_grip(x, y0=y0), (0.01,) * 3) for n, (x, y0, _, _) in zip(names, places)]
        _, py0, _, _ = places[k]
        px = WEDGE_XS[0] if decoy_kind(level, seed) == "wide" else places[k][0]  # pushed from the corridor's line
        regions.append(zone("wedge_pushed", wedge_grip(px, pushed=True, y0=py0), (0.01,) * 3))
        reach = [r for r in reach if r[0] != "west"] + [("south", f"{n}_grip") for n in names] + \
            [("south", "wedge_pushed")]
        if decoy_kind(level, seed) == "wide":  # where a wedge in the corridor's way is set down, and where it is slid to
            regions += [zone("wedge_park", wedge_grip(PARK_XY[0], y0=PARK_XY[1]), (0.01,) * 3),
                        zone("wedge_in_line", wedge_grip(WEDGE_XS[0], y0=WIDE_Y0), (0.01,) * 3)]
            reach += [("south", "wedge_park"), ("south", "wedge_in_line")]
        out_of_reach = [r for r in out_of_reach if r[0] != "west"]
        extra = {"borrow": {"props": [prop], "decoys": [n for n in names if n != prop]},
                 "decoy_kind": decoy_kind(level, seed)}
    sc = scene(spec, objects, regions, hinges=[hinge_w, hinge_e])
    if variant == "borrow":
        sc = dataclasses.replace(sc, arms=tuple(a for a in sc.arms if a.slot != "west"))

    through = ThroughSlots("pin", KEY[1], (KEY[0], KEY[2]), sl)
    level_w, level_e = Level("leaf_w", LEVEL_DEG), Level("leaf_e", LEVEL_DEG)
    return make_instance(
        spec, level, seed, sc,
        instruction=("Pin the blue and the orange leaf together through all their slots with the grey pin, so "
                     "that they hold each other level between their posts."),
        success=HoldsFor(through & level_w & level_e, 1.0),
        milestones={
            "both_level": Eventually(level_w & level_e),
            "pin_in_one": Eventually(ThroughSlots("pin", KEY[1], (KEY[0], KEY[2]), sl[1:2])),
            "pinned": Eventually(through & level_w & level_e),
        },
        reach=reach, out_of_reach=out_of_reach, slot_clearance_m=clearance,
        paint={"leaf_w": "blue", "leaf_e": "orange"}, **extra,
    )


#: the keyframe `mrbench render` shows between start and goal: both leaves held level, the pin at the slots
MIDDLE = 1


def _pinned_at(inst, y):
    """The pin along y through the joint, its key's middle at ``y``."""
    return at((JOINT[0], y, JOINT[2]))


def _slide(inst, w) -> float:
    """How far (x) the wedge ``w`` is slid before it is pushed: from WIDE_X into the corridor's line in a wide-distractor
    scene, else 0."""
    return WEDGE_XS[0] - w.pos[0] if inst.metadata.get("decoy_kind") == "wide" and w.pos[0] < WEDGE_XS[0] - 0.1 \
        else 0.0


def keyframes(inst):
    lw, le = level_pose(inst, "leaf_w"), level_pose(inst, "leaf_e")
    first = {"leaf_w": lw, "leaf_e": le}
    if inst.variant == "borrow":  # the prop pushed under the west leaf's bar
        prop = inst.metadata["borrow"]["props"][0]
        w = inst.scene.obj(prop)
        first[prop] = offset(w.pose, dx=_slide(inst, w), dy=wedge_push(w.pos[1]))
        if _slide(inst, w):  # the other one, which stood in its way, moved east of the pin
            o = inst.scene.obj(inst.metadata["borrow"]["decoys"][0])
            first[o.name] = offset(o.pose, dx=PARK_XY[0] - o.pos[0], dy=PARK_XY[1] - o.pos[1])
    return [
        first,
        {"pin": _pinned_at(inst, JOINT[1] - GUARD[1] - WALL - KEY[1] / 2)},
        {"pin": _pinned_at(inst, PIN_HOME_Y)},
    ]


#: how far the fingers close on a leaf's bar (40 mm square) and on the pin's head, across it (44 mm)
BAR_GRIP, HEAD_GRIP = BAR - 0.003, HEAD[0] - 0.003
#: how far the hands holding the leaves lean out from straight down, their bodies toward their hinges; and the hand on
#: the pin (leaning 8 degrees or more, it could not come straight in to the head and keep it on the slide)
LEAN_DEG, PIN_LEAN_DEG = 30.0, 0.0
#: the pin's key goes in from this far short of the guard's south wall, along +y
PIN_START = 0.03
#: where the key's middle stops: short of the joint, its tip 6.5 cm into the north cheek's 8.5 cm slot -- 2.25 cm past
#: the slot's middle, where the success asks for 1 cm (``ThroughSlots.reach``): in Isaac the key ends a few millimetres
#: short of its plan and, let go, slides back 2 mm more.  The hand on the head may be turned either way round (the
#: planner turns a parallel grip half round as it suits), and turned with its wrist camera north the camera is 19 cm up
#: and 7.5 cm ahead of the hand: 5.5 cm further in, it meets the guard's south wall over the pin's opening
PIN_HOME_Y = JOINT[1] - 0.065


def script(inst):
    """The scripted reference (privileged: a grip fixes what is gripped to the hand)."""
    sc = inst.scene
    down = (0.0, 0.0, -1.0)

    def leaf_grip(name, pose):
        """The tool point on a leaf's bar, HOLD_AT from its hinge, the fingers across it, the hand pointing down, its
        body leaning LEAN_DEG out toward its hinge, and its wrist camera on its outer side (on the inner one, the east
        hand's camera came within 9 mm of the guard)."""
        h = next(h for h in sc.hinges if h.body == name)
        s = 1 if name == "leaf_w" else -1
        point = transform(pose, (s * HOLD_AT, 0.0, BAR / 2 - 0.012))
        lean = math.radians(LEAN_DEG)
        way = (s * math.sin(lean), 0.0, -math.cos(lean))  # fingers toward the joint: the hand's body leans out
        return (*point, *quat_mul(pose[3:], tool_quat(way, (0.0, s, 0.0)))), h

    def arc(name, a0, a1):
        """The held leaf turning from a0 to a1 degrees open (0 = at its stop, REST_DEG = level)."""
        g, h = leaf_grip(name, sc.obj(name).pose)
        return lambda u: rotate_about(g, h.pivot, axis_quat(h.axis, h.sign * (a0 + (a1 - a0) * u)))

    gw, _ = leaf_grip("leaf_w", sc.obj("leaf_w").pose)
    ge, _ = leaf_grip("leaf_e", sc.obj("leaf_e").pose)
    pin = sc.obj("pin")
    head = transform(pin.pose, pin.part("head").pos)
    # the head taken from above, the fingers closing across the pin: the hand's long side across the way the pin goes
    # in, so that it comes up to the guard's wall, its wrist camera on its south side (on its north one it came within
    # 1 cm of the wall).  1.5 cm down the head, the fingertips came within 8 mm of the table
    lean = math.radians(PIN_LEAN_DEG)
    take = (head[0], head[1], head[2] + HEAD[2] / 2 - 0.010,
            *tool_quat((0.0, math.sin(lean), -math.cos(lean)), (-1.0, 0.0, 0.0)))  # fingers north: the wrist leans south
    to_tool = [t - c for t, c in zip(take[:3], pin.pos)]  # from the pin's middle to the tool point
    # the key in line with the slots, its tip PIN_START short of the guard's south wall; then home along +y
    wall = JOINT[1] - GUARD[1] - WALL
    start_y = wall - PIN_START - KEY[1] / 2
    lined = (JOINT[0] + to_tool[0], start_y + to_tool[1], JOINT[2] + to_tool[2], *take[3:])
    home_ = offset(lined, dy=PIN_HOME_Y - start_y)
    w, e, s = "west", "east", "south"
    pin_steps = [
        phase(2.5, "the middle arm comes over the pin", Move(s, back_off(take, 0.10), joints=True), Open(s, 0.07)),
        phase(1.2, "", Move(s, take)),
        phase(0.8, ""),
        phase(0.4, "takes it by its head", Grip(s, "pin", HEAD_GRIP)),
        phase(1.5, "lifts it", Move(s, offset(take, dz=0.10))),
        phase(3.0, "brings it in line with the opening", Move(s, offset(lined, dy=-0.04))),
        phase(1.0, "", Move(s, lined)),
        phase(1.0, "lines it up"),
        phase(5.0, "and slides it through all three slots", Move(s, home_)),
        phase(0.4, "lets go of it", Release(s)),
        phase(1.5, "and backs off", Move(s, offset(home_, dy=-0.06, dz=0.06))),
    ]
    if inst.variant == "borrow":
        # no west arm: the middle arm takes the wedge by its handle's crossbar (from above) and pushes it
        # north under the blue leaf's bar, whose ramp lifts the leaf onto its top, level; then it lets go of the wedge
        # and pins the leaves while the east arm holds its own, as in the exact version
        prop = inst.metadata["borrow"]["props"][0]
        w0 = sc.obj(prop)  # the hand on its crossbar, whichever side it is (read from the body: a distractor swapped in

        def on_bar(body):  # for the prop has its own)
            b = transform(body.pose, body.part("grip").pos)
            return (b[0], b[1], b[2], *tool_quat(down, (0.0, 1.0, 0.0)))  # fingers across the crossbar

        g0 = on_bar(w0)
        slide = _slide(inst, w0)
        g0s = offset(g0, dx=slide)  # where the push starts: in the corridor's line
        g1 = offset(g0s, dy=wedge_push(w0.pos[1]))
        first = []
        if slide:  # the other wedge stands in the corridor's line, in this one's way: moved east of the pin first
            o = sc.obj(inst.metadata["borrow"]["decoys"][0])
            h0 = on_bar(o)
            h1 = offset(h0, dx=PARK_XY[0] - o.pos[0], dy=PARK_XY[1] - o.pos[1])
            past = offset(h0, dx=CARRY_PAST - h0[0], dy=h1[1] - h0[1])  # its crossbar east of the arm's front
            first = [
                phase(1.5, "the middle arm comes over the handle of the wedge standing in the corridor's line; the "
                           "east arm makes room", Move(s, offset(h0, dz=0.10), joints=True), Open(s, 0.04),
                      Move(e, offset(home_pose(inst, e), *EAST_AWAY))),
                phase(1.2, "", Move(s, h0)),
                phase(0.8, ""),
                phase(0.4, "takes it", Grip(s, o.name, HANDLE_T - 0.003)),
                phase(1.5, "lifts it", Move(s, offset(h0, dz=LIFT_OVER))),
                phase(2.5, "and carries it out of the way, over the pin", Move(s, offset(past, dz=LIFT_OVER))),
                phase(1.0, "", Move(s, offset(past, dz=LIFT_PIN))),
                phase(1.5, "to east of it", Move(s, offset(h1, dz=LIFT_PIN))),
                phase(1.5, "", Move(s, offset(h1, dz=0.003))),
                phase(0.4, "sets it down", Release(s)),
                # up well clear of it before swinging west (10 cm up, the hand passed 3 mm from its block)
                phase(1.5, "", Move(s, offset(h1, dz=0.20))),
            ]
        slid = [phase(3.0, "slides the wedge along the table into the corridor's line", Move(s, g0s))] if slide else []
        # which way the hand steps aside once it lets go (as the crossbar went out, when it went out to one side): east
        # with the wide distractor, else away from the other wedge's place
        aside = 0.03 if inst.metadata.get("decoy_kind") == "wide" or w0.pos[0] >= sum(WEDGE_XS) / 2 else -0.03
        return [
            *first,
            phase(1.5, "the east arm reaches out; the middle arm comes over the wedge's handle",
                  Move(e, offset(back_off(ge, 0.10), dx=0.10)), Open(e, 0.07),
                  Move(s, offset(g0, dz=0.10), joints=True), Open(s, 0.04)),
            phase(1.0, "", Move(e, back_off(ge, 0.10))),
            # down onto the crossbar in the move before the grip (the clearance walk counts a hand as working on what it
            # takes from then; a phase earlier, its fingers were measured against the crossbar they came down on)
            phase(1.2, "", Move(e, ge), Move(s, g0)),
            phase(0.8, "they settle"),
            phase(0.4, "and take the orange leaf and the wedge", Grip(e, "leaf_e", BAR_GRIP),
                  Grip(s, prop, HANDLE_T - 0.003)),
            *slid,
            phase(4.0, "the wedge goes in under the blue leaf's bar and lifts it level; the east arm turns its leaf "
                       "level", Move(s, g1), Move(e, arc("leaf_e", 0.0, REST_DEG))),
            phase(0.4, "the middle arm lets go of the wedge", Release(s)),
            # up over the wedges' tops (20 cm), and away from the other wedge, before going round to the pin: lower, it
            # swept over the other wedge, and straight up its wrist camera came within 7 mm of it
            phase(1.5, "", Move(s, offset(g1, dx=aside, dy=-0.04, dz=0.22))),
            *pin_steps,
            phase(3.0, "the leaves are pinned"),
        ]
    return [
        # out past their bars first: straight from home the hands ran along the guard's walls
        phase(1.5, "the end arms reach out", Move(w, offset(back_off(gw, 0.10), dx=-0.10)),
              Move(e, offset(back_off(ge, 0.10), dx=0.10)), Open(w, 0.07), Open(e, 0.07)),
        phase(1.0, "and come over their leaves", Move(w, back_off(gw, 0.10)), Move(e, back_off(ge, 0.10))),
        phase(1.2, "", Move(w, gw), Move(e, ge)),
        # a grip fixes the body where the hand is: a hand still settling (5 mm off) holds it that far off
        phase(0.8, "they settle"),
        phase(0.4, "and take them", Grip(w, "leaf_w", BAR_GRIP), Grip(e, "leaf_e", BAR_GRIP)),
        phase(3.0, "they turn them level: the west one up, the east one down against its spring",
              Move(w, arc("leaf_w", 0.0, REST_DEG)), Move(e, arc("leaf_e", 0.0, REST_DEG))),
        *pin_steps,
        # the end arms keep their leaves (that the pinned leaves hold themselves is played separately, letting go)
        phase(3.0, "the leaves are pinned"),
    ]


#: where along a leaf's bar outside the guard (from its hinge) a hand may hold it level: the jobs' sites
BAR_HOLDS = (0.08, 0.15, HOLD_AT, LEAF_L - GUARD[0] - WALL - 0.01)


def jobs(inst):
    """What the task asks (``tasks/jobs.py``): both leaves held level at once, each by its bar anywhere outside the
    guard, while the pin is taken by its head and slid home through the three slots.  The borrow version's wedge holds
    the blue leaf level once it is pushed in under its bar; the low one does not."""
    from ..jobs import AnyOf, Job, Jobs, Thing

    x, y, z = JOINT
    pin = inst.scene.obj("pin")
    grip = 0.02
    held = [Job(f"hold the {colour} leaf level by its bar", group="pinned", slack=grip,
                sites=(AnyOf(tuple((x + side * (LEAF_L - d), y, z + BAR / 2) for d in BAR_HOLDS)),))
            for colour, side in (("blue", -1), ("orange", 1))]
    out = [*held, Job("slide the pin home through the three slots", group="pinned", slack=grip,
                      sites=(transform(pin.pose, pin.part("head").pos), "head_home"))]
    things = ()
    if inst.variant == "borrow":
        b = inst.metadata["borrow"]
        prop, decoy = b["props"][0], b["decoys"][0]
        things = (Thing(prop, does=(held[0].name,),
                        placing=(Job(f"push {prop} in under the blue leaf's bar", sites=(f"{prop}_grip", "wedge_pushed"),
                                     slack=grip),),
                        why=f"its top {WEDGE_TOP * 100:.2f} cm up, 0.8 mm under the bar held level: it holds the leaf "
                            "level"),
                  Thing(decoy, why=(f"its top {WEDGE_LOW * 100:.0f} cm up, 2 cm lower: the leaf rests 4 degrees low or "
                                    "more on it, its slot 3 cm or more out of line") if inst.metadata["decoy_kind"] == "low"
                        else (f"{WEDGE_WIDE * 100:.0f} cm wide: wider than the {CORRIDOR * 100:.1f} cm between the west "
                              "post's plate and the guard, it meets one of them before it gets under the bar")))
    return Jobs(tuple(out), things)
