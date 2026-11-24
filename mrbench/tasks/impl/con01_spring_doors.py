"""Con-01 spring doors: two spring doors that open inward must both be held open, each by its lever outside the
cabinet, while a third hand takes a box out through them; the box then goes on a mat on the table, the doors
closed."""

from __future__ import annotations

import dataclasses
import math
import random

from ...engines.base import RegionSpec, quat_mul, transform, yaw_quat
from ...policies.script import Grip, Move, Open, Release, home_pose, offset, phase, tool_quat
from ...scoring.predicates import Before, Eventually, HoldsAtEnd, HoldsFor, InRegion, Opened
from ..loader import get
from ._common import Z, at, box, compound, hinge, jitter, make_instance, pad, part, rng_for, rotate_about, scene

#: The cabinet stands across the middle of the table, its doorway facing the south arm.  It has two parts.  In front,
#: the *door zone*: as deep as the doors swing and only as high as the box, under a fixed ceiling -- low enough to stay
#: under the arms' hands at home (the south hand is 0.24 m from the table's middle).  Behind it, the *box zone*, taller,
#: closed but for a slot in its front wall over the ceiling: the box starts there.  The box is held by a T-handle whose
#: post runs in a narrow slot in the door zone's ceiling, so a hand holding it, and every link behind that hand, stays
#: over the ceiling: nothing that holds the box gets to an open door (they lie under the ceiling, flat by the side
#: walls) to keep it open.  (With a tall door zone the south arm's wrist could hold a door flat against its wall while
#: its hand drew the box out -- IK: two arms.)  An arm's elbow is within its upper arm's 0.316 m of its shoulder, so
#: the south arm's never gets in under the ceiling either (its front is 0.38 m from that shoulder).
FRONT_Y = -0.32
#: inside width; the walls' and plates' thickness; the plinth the box stands on (the doors' levers run below its top)
INNER_W, WALL, PLATE, PLINTH = 0.34, 0.02, 0.015, 0.10
SHELF_Z = Z + PLINTH  # the plinth's top: the box's floor
#: the plinth's front is this far behind the doorway, so that a door and the post under its hinge clear it
PLINTH_BACK = 0.03
#: each door: thickness, how much higher than the box, how far in from the side wall its hinge is (the open door lies
#: flat by the wall, 5 mm off it; every part of it swings 4.5 mm or more clear of the walls), and how far behind the
#: front plane its middle is
DOOR_T, DOOR_UP, HINGE_IN, DOOR_Y = 0.016, 0.02, 0.013, 0.010
#: a door's gap over the plinth (and under the ceiling)
DOOR_GAP = 0.006
DOOR_L = INNER_W / 2 - HINGE_IN - 0.002  # to the seam
#: the box: depth and height -- both more than one door's way (15 cm) and than the box zone's slot (19 cm), so no turn
#: of it gets it out with a door shut, or through the slot.  Its T-handle: a post up through the ceiling's slot and a bar
#: along x on top (a hand on it shows the slots its thin side)
BOX_D, BOX_H, POST_H, HANDLE_L, BAR = 0.20, 0.20, 0.10, 0.10, 0.012
#: the handle stands this far west of the box's middle (the hand's wrist camera, on its east side, then clears the
#: slot's east edge), and the ceiling's slot with it
HANDLE_X = -0.015
#: the door zone's ceiling: just over the doors; the slot in it for the handle's post
CEIL_Z = SHELF_Z + DOOR_GAP + BOX_H + DOOR_UP + DOOR_GAP  # its underside
POST_SLOT = 0.03
#: the door zone's depth: a closed door's free end sweeps 0.16 m in
ZONE_D = 0.19
ZONE_Y = FRONT_Y + ZONE_D  # the box zone's front wall
#: the box zone: its depth (the box and 2 cm), its roof's underside, and the slot in its front wall over the ceiling
BOX_ZONE_D, ROOF_Z, SLOT_W = BOX_D + 0.02, Z + 0.70, 0.19
#: a door's lever, below the plinth's top: from a post on the hinge's line, a bar LEVER_L out beside the side wall in
#: front of the cabinet, and a stub STUB_L back along the wall's outside with a knob at its end.  Opened, the bar points
#: south under the box's way out and the stub west (east).  A door is held open by its knob, low in front of the
#: cabinet's corner: no link of an arm whose hand holds the box gets down there (IK).
LEVER_Z, LEVER_FRONT, LEVER_L, STUB_L, KNOB_H = Z + 0.05, 0.010, 0.17, 0.14, 0.06
#: the knob is a T-handle's stem (steel linkage, and the stem topped by a rubber grip GRIP_L long along y and GRIP_H
#: high, in the top of its height: the fingers close across the stem, along x, and the handle is no taller)
GRIP_L, GRIP_H = 0.06, 0.016
#: the doors' springs (N*m/rad): at 90 degrees a door pulls back about 1.05 N*m, 7 N at the knob; damped past critical
SPRING, SPRING_DAMPING, PRELOAD_DEG, OPEN_MAX = 0.6, 0.2, 10.0, 90.0
#: a door counts as closed within this many degrees of its stop
CLOSED_DEG = 5.0
#: the box starts in the box zone, 1 cm behind its front wall
BOX_Y = ZONE_Y + 0.01 + BOX_D / 2
#: how far the side arms hold the doors open (the stop is at 90, the door then flat against its wall)
HOLD_DEG = 88.0
#: how far a bar is between the fingers when they close on it (the knobs and the handle are 12 mm)
BAR_GRIP = 0.014
#: where the middle arm waits while the doors are opened: in front of the cabinet, above the levers
PARK = (0.0, -0.52, Z + 0.45, *tool_quat((0.0, 0.0, -1.0), (1.0, 0.0, 0.0)))
#: the middle arm's hand on the handle in the box zone points north and this far down
HANDLE_PITCH_DEG = 55.0
X0 = 0.0  # the cabinet's middle (x)
#: where the box goes: a mat on the table in front of the cabinet, east of the doorway, clear of the levers once the
#: doors are shut (with them open, the east lever's stub lies across it)
MAT = (0.25, FRONT_Y - 0.18)
#: the box's bottom this much higher than the plinth while it is held in front of the doorway as the doors close
#: under it (the levers' tops are 4 cm below the plinth's)
HOLD_UP = 0.004
#: the borrow version (no east arm): the south arm keeps the east door open with a block set in its lever's way back.
#: No hand is free to hold the knob while another sets the block, so the south arm, holding the block, pushes the
#: lever round with it and lets go of it there, between the bar and the stub; springing back, the lever holds it by
#: friction (1.03 N*m at 89 degrees on the bar 13 cm out: 8 N, against 9.8 N at mu 0.5; nearer the hinge it would
#: slide).  The distractor is the same block BLOCK_LOW tall: the bar (5 cm up)
#: passes over it.  Both weigh BLOCK_MASS and stand east of the mat (named block_1 and block_2, north to south), which
#: is which drawn from the seed.  With the doors open the east lever's stub lies across the mat, so the box waits west
#: of the doorway (WAIT) while the block is lifted away and the door closes, and then goes on the mat.
BLOCK, BLOCK_LOW, BLOCK_MASS = (0.09, 0.09, 0.09), 0.04, 3.0
#: each is a broad, low weight with a handle on top (a 6 cm block on a rubber foot does not slide but tips over against
#: the lever, pushed 5 cm up): a body BODY_H tall (the distractor's
#: BODY_LOW, under the lever's bar) and a HANDLE on it the hand takes
BODY_H, BODY_LOW, HANDLE = 0.06, 0.02, (0.04, 0.04)
BLOCK_XY = ((0.46, -0.45), (0.46, -0.60))
WAIT = (-0.25, FRONT_Y - 0.18)
#: the push: the block let down into the corner between the shut lever's bar and stub, BLOCK_GAP off each, then
#: turned with the lever round the hinge till the door is PUSH_DEG open (just short of its 90-degree stop); the hand
#: takes it BLOCK_TAKE under its top, the fingers over the bar.  Let down 5 mm off them, the block comes to rest on the
#: bar's top in Isaac: the arm holding it runs up to 13 mm off its plan
BLOCK_GAP, PUSH_DEG, BLOCK_TAKE = 0.02, 89.0, 0.012
#: off the stub
STUB_GAP = 0.02
#: both blocks stand on a rubber foot FOOT thick (friction BLOCK_MU, taken over the table's: "max"), their sides as
#: the default material.  On the default material a block let go in the lever's way does not hold it: the lever pushes
#: 2-6 kg blocks back along the table, as if their friction were about 0.06 (Isaac).  A block at friction 1-3 all
#: over holds the door 38-73 degrees open -- the box needs 81 (easy) and 87 (hard) -- and at 3.0 the lever, turning
#: back against the block's side with that friction, lifts it onto an edge.  Pushing the lever the block is carried
#: PUSH_UP over the table (1 cm up, the loaded arm sags and it drags); the push dwells at its end (the arm lags 7
#: degrees behind it), then sets the block down SET_UP over the table, and the hand opens as it lets go (closed,
#: lifted away, it would take the block up with it)
BLOCK_MU, FOOT, PUSH_UP, SET_UP = 3.0, 0.01, 0.025, 0.002
#: the hand on the block leans this far east, its wrist away from the cabinet (straight down or leaning west, its
#: wrist and forearm come within a centimetre of the side wall and the ceiling; leaning south, its fingers' travel
#: tilted across the block measures it 8.4 cm, wider than they open); the box goes from WAIT to the mat CARRY_UP over
#: the table, over the shut levers
BLOCK_TILT_DEG, CARRY_UP = 30.0, 0.15
#: how much of the lever's turn the hand on the weight's square handle turns with it (none: at the push's end its broad
#: side, along the push, comes within 0.6 cm of the side wall; all of it: across the push, it meets the knob)
HAND_TURN = 0.5
#: how wide the hand opens to let go of the handle (4 cm), and to take it again: all the way (opened 4.8 cm, the
#: fingers rising off it can catch the handle and tip the weight, and the door falls)
RELEASE_OPEN = 0.08
#: which way round the south hand reaches in for the box (``Move.turn``): as the exact plan does
REACH_TURN = True
#: and which way round it takes the weight, holds it, rises off it and takes it again: its wrist camera on the side
#: away from the east knob (the other way, the camera passes 8 mm from the knob at the push's end; left free as it
#: rises, the planner turns the hand half round about its pointing, and its fingers and hand sweep the handle and the
#: knob)
HOLD_TURN = False
#: then, still that way round, how far east and up from where it let go it moves (its fingertips 12 cm over the
#: knob's top), and from there it comes back for the weight
RISE_AWAY = (0.10, 0.15)
#: how high it lifts the weight away at the end, its bottom over the east knob's top (11 cm up)
LIFT_AWAY = 0.15


def pivot(side: int) -> tuple[float, float, float]:
    """A door's hinge axis (vertical), at its outer end: side -1 west, +1 east."""
    return (X0 + side * (INNER_W / 2 - HINGE_IN), FRONT_Y + DOOR_Y, SHELF_Z)


def doors_needed_deg(width: float) -> float:
    """How far both doors must be open for a box ``width`` wide (1 cm to spare each side) to pass between their
    inner faces: the free end's inner corner stays 1 cm outside the box's side."""
    half = width / 2 + 0.01
    for d10 in range(0, 901):
        a = math.radians(d10 / 10)
        inner = DOOR_L * math.cos(a) + DOOR_T / 2 * math.sin(a)  # the free end's inner corner, in from the hinge
        if INNER_W / 2 - HINGE_IN - inner >= half:
            return d10 / 10
    return 90.0


def _door(name: str, side: int):
    """A door and its lever, as one body; side -1 west (opens anticlockwise from above), +1 east."""
    px, py, _ = pivot(side)
    door_h = BOX_H + DOOR_UP
    zc = SHELF_Z + DOOR_GAP + door_h / 2
    cx = px - side * DOOR_L / 2  # the panel's middle, toward the seam
    rel = lambda x, y, z: (x - cx, y - py, z - zc)  # noqa: E731 -- world to the body's frame
    lever_y = FRONT_Y - LEVER_FRONT
    out_x = px + side * LEVER_L  # the lever bar's outer end
    bottom = zc - door_h / 2
    # the panel ends on the hinge's line: its outer corners then swing 2 mm clear of the wall (a panel running 8 mm
    # past it cuts 1.3 mm into the wall between 25 and 73 degrees, and a door let go at 88 sticks at 70)
    parts = [part("slab", "box", (DOOR_L, DOOR_T, door_h)),
             # a post on the hinge's line from under the door down to the lever, and a knuckle out to the bar
             part("post", "box", (BAR, BAR, bottom - LEVER_Z + BAR / 2), rel(px, py, (bottom + LEVER_Z - BAR / 2) / 2),
                  color="steel"),
             part("knuckle", "box", (BAR, py - lever_y + BAR, BAR), rel(px, (py + lever_y) / 2, LEVER_Z), color="steel"),
             part("lever", "box", (LEVER_L + BAR, BAR, BAR), rel(px + side * LEVER_L / 2, lever_y, LEVER_Z), color="steel"),
             part("stub", "box", (BAR, STUB_L + BAR, BAR), rel(out_x, lever_y + STUB_L / 2, LEVER_Z), color="steel"),
             part("knob", "box", (BAR, BAR, KNOB_H), rel(out_x, lever_y + STUB_L, LEVER_Z + KNOB_H / 2), color="steel"),
             part("knob_grip", "box", (BAR, GRIP_L, GRIP_H), rel(out_x, lever_y + STUB_L, LEVER_Z + KNOB_H - GRIP_H / 2),
                  color="rubber")]
    door = compound(name, (cx, py, zc), parts, "wood", mass=0.8)
    return door, hinge(f"{name}_hinge", door, (px, py, zc), sign=-side, open_max_deg=OPEN_MAX, stiffness=SPRING,
                       damping=SPRING_DAMPING, preload_deg=PRELOAD_DEG)


def _cabinet():
    """The carcass, as static slabs, and the region inside it."""
    x0, w = X0, INNER_W
    back_y = ZONE_Y + BOX_ZONE_D
    outer = w + 2 * WALL
    ceil_top = CEIL_Z + PLATE
    slabs = [box("plinth", (x0, (FRONT_Y + PLINTH_BACK + back_y) / 2), (w, back_y - FRONT_Y - PLINTH_BACK, PLINTH),
                 "wood", z=Z + PLINTH / 2, dynamic=False),
             box("cab_back", (x0, back_y + WALL / 2), (outer, WALL, ROOF_Z - Z), "wood", z=(Z + ROOF_Z) / 2,
                 dynamic=False),
             box("cab_roof", (x0, (ZONE_Y + back_y + WALL) / 2), (outer, back_y + WALL - ZONE_Y, PLATE), "wood",
                 z=ROOF_Z + PLATE / 2, dynamic=False)]
    for name, s in (("w", -1), ("e", 1)):
        xs = x0 + s * (w / 2 + WALL / 2)
        # the side walls: as high as the ceiling along the door zone, as the roof along the box zone
        slabs.append(box(f"side_low_{name}", (xs, (FRONT_Y + ZONE_Y) / 2), (WALL, ZONE_D, ceil_top - Z), "wood",
                         z=(Z + ceil_top) / 2, dynamic=False))
        slabs.append(box(f"side_{name}", (xs, (ZONE_Y + back_y) / 2), (WALL, BOX_ZONE_D, ROOF_Z - Z), "wood",
                         z=(Z + ROOF_Z) / 2, dynamic=False))
        # the door zone's ceiling either side of the post's slot
        sx = x0 + HANDLE_X + s * POST_SLOT / 2  # the slot's edge
        ex = x0 + s * (w / 2 + WALL)  # the cabinet's side
        slabs.append(box(f"ceiling_{name}", ((sx + ex) / 2, (FRONT_Y + ZONE_Y) / 2), (abs(ex - sx), ZONE_D, PLATE),
                         "wood", z=CEIL_Z + PLATE / 2, dynamic=False))
        # the box zone's front wall over the ceiling, either side of its slot
        fw = w / 2 + WALL - SLOT_W / 2
        slabs.append(box(f"front_{name}", (x0 + s * (SLOT_W / 2 + fw / 2), ZONE_Y - PLATE / 2), (fw, PLATE, ROOF_Z - CEIL_Z),
                         "wood", z=(CEIL_Z + ROOF_Z) / 2, dynamic=False))
    inside = RegionSpec("cabinet_in", (x0, (FRONT_Y + back_y) / 2, (SHELF_Z + ROOF_Z) / 2),
                        (w / 2, (back_y - FRONT_Y) / 2, (ROOF_Z - SHELF_Z) / 2), visible=False)
    return slabs, inside


def _box(width: float, xy):
    parts = [part("body", "box", (width, BOX_D, BOX_H)),
             part("post", "box", (BAR, BAR, POST_H), (HANDLE_X, 0.0, BOX_H / 2 + POST_H / 2), color="grey"),
             part("handle", "box", (HANDLE_L, BAR, BAR), (HANDLE_X, 0.0, BOX_H / 2 + POST_H + BAR / 2), color="grey")]
    return compound("box", (xy[0], xy[1], SHELF_Z + BOX_H / 2 + 0.001), parts, "orange", mass=0.6)


def _block(name, xy, height):
    """A weight BLOCK[0] x BLOCK[1] on a rubber foot, its handle on top, ``height`` tall in all (BLOCK[2]: the prop's;
    BLOCK_LOW: the distractor's); its frame at the middle of the box round it."""
    body = BODY_H if height == BLOCK[2] else BODY_LOW
    grip = height - body  # the handle's height
    bottom = -height / 2
    return compound(name, (*xy, Z + height / 2),
                    [part("foot", "box", (BLOCK[0], BLOCK[1], FOOT), (0.0, 0.0, bottom + FOOT / 2), color="rubber",
                          friction=BLOCK_MU, friction_combine="max"),
                     part("body", "box", (BLOCK[0], BLOCK[1], body - FOOT), (0.0, 0.0, bottom + (FOOT + body) / 2)),
                     part("handle", "box", (*HANDLE, grip), (0.0, 0.0, height / 2 - grip / 2), color="safety_yellow")],
                    "iron", mass=BLOCK_MASS)


def block_corner():
    """Where the block's middle is let down for the push: in the corner between the shut east lever's bar and stub,
    BLOCK_GAP off the bar and STUB_GAP off the stub (x, y)."""
    out_x = pivot(1)[0] + LEVER_L
    return (out_x - BAR / 2 - STUB_GAP - BLOCK[0] / 2, FRONT_Y - LEVER_FRONT + BAR / 2 + BLOCK_GAP + BLOCK[1] / 2)


def block_turn(xy, deg: float):
    """A point ``xy`` turned round the east door's hinge as the door opens ``deg``."""
    px, py, _ = pivot(1)
    a = math.radians(-deg)  # the east door opens clockwise from above
    dx, dy = xy[0] - px, xy[1] - py
    return (px + dx * math.cos(a) - dy * math.sin(a), py + dx * math.sin(a) + dy * math.cos(a))


def _gap_deg() -> float:
    """How far the block turns round the hinge before it meets the shut bar (BLOCK_GAP away)."""
    cx, cy = block_corner()
    px, py, _ = pivot(1)
    return math.degrees(math.asin(BLOCK_GAP / math.hypot(cx - px, cy - py)))



def box_set_down(at_end: bool = True):
    """Con-01's success: the box out of the cabinet, then on the mat with both doors closed for 1 s -- and still so
    when the run ends, through its last second at least (``HoldsAtEnd``: a box set down for a second and carried off
    again, or still in the hand on the mat, does not count).  ``at_end=False``: set down for a second, at any time."""
    closed = ~Opened("door_w", CLOSED_DEG) & ~Opened("door_e", CLOSED_DEG)
    placed = InRegion("box", "mat") & closed
    return Before(Eventually(~InRegion("box", "cabinet_in")), (HoldsAtEnd if at_end else HoldsFor)(placed, 1.0))


def build(level: str, seed: int, variant: str = "exact"):
    spec = get("Con-01")
    rng = rng_for(spec, level, seed)
    width = spec.knob.levels[level] / 100
    slabs, inside = _cabinet()
    door_w, hinge_w = _door("door_w", -1)
    door_e, hinge_e = _door("door_e", +1)
    the_box = _box(width, jitter(rng, (X0, BOX_Y + 0.003), 0.003))
    knobs = [RegionSpec(f"knob_{d.name[-1]}", transform(d.pose, d.part("knob").pos), (0.01,) * 3, visible=False)
             for d in (door_w, door_e)]
    regions = [inside, pad("mat", MAT, (0.06, 0.05), "green", hz=0.12), *knobs,
               RegionSpec("handle", transform(the_box.pose, the_box.part("handle").pos), (0.01,) * 3, visible=False)]
    objects = slabs + [door_w, door_e, the_box]
    reach = [("west", "knob_w"), ("east", "knob_e"), ("south", "handle"), ("south", "mat")]
    out_of_reach = [("west", "knob_e"), ("east", "knob_w")]
    extra = {}
    if variant == "borrow":  # no east arm: a block keeps the east door open, and a lower one that cannot
        # which is the prop is drawn from the seed, on its own stream (the exact configuration's scene is unchanged)
        heights = [BLOCK[2], BLOCK_LOW]
        random.Random(f"borrow/{spec.seed_id}/{level}/{seed}").shuffle(heights)
        names = ("block_1", "block_2")
        objects += [_block(n, xy, hz) for n, xy, hz in zip(names, BLOCK_XY, heights)]
        prop = names[heights.index(BLOCK[2])]
        hinge_e = dataclasses.replace(hinge_e, propped=True)  # the clearance check: it stays where the block holds it
        set_xy = block_turn(block_corner(), PUSH_DEG + _gap_deg())
        regions += [RegionSpec(f"{n}_top", (*xy, Z + hz - BLOCK_TAKE), (0.01,) * 3, visible=False)
                    for n, xy, hz in zip(names, BLOCK_XY, heights)]
        regions += [RegionSpec("block_set", (*set_xy, Z + BLOCK[2] - BLOCK_TAKE), (0.01,) * 3, visible=False),
                    RegionSpec("block_in", (*block_corner(), Z + BLOCK[2] - BLOCK_TAKE), (0.01,) * 3, visible=False),
                    RegionSpec("wait", (*WAIT, Z + BOX_H + POST_H + BAR / 2), (0.01,) * 3, visible=False)]
        reach = [r for r in reach if r[0] != "east"] + [("south", f"{n}_top") for n in names] + \
            [("south", "block_in"), ("south", "block_set"), ("south", "wait")]
        out_of_reach = [r for r in out_of_reach if r[0] != "east"]
        extra = {"borrow": {"props": [prop], "decoys": [n for n in names if n != prop]}}
    sc = scene(spec, objects, regions, hinges=[hinge_w, hinge_e])
    if variant == "borrow":
        sc = dataclasses.replace(sc, arms=tuple(a for a in sc.arms if a.slot != "east"))

    need = doors_needed_deg(width)
    closed = ~Opened("door_w", CLOSED_DEG) & ~Opened("door_e", CLOSED_DEG)
    out = ~InRegion("box", "cabinet_in")
    on_mat = InRegion("box", "mat")
    return make_instance(
        spec, level, seed, sc,
        instruction="Take the orange box out of the cabinet and set it on the green mat, with both doors closed.",
        success=box_set_down(),
        milestones={
            "doors_open": Eventually(Opened("door_w", need) & Opened("door_e", need)),
            "box_out": Eventually(out),
            "box_on_mat": Eventually(on_mat),
            "doors_closed": Eventually(on_mat & closed),
        },
        reach=reach,
        # a knob only its own side's arm holds (or the south arm's hand).  The three jobs -- each door held, the box
        # drawn out -- need three arms because no arm does two at once, by IK (an end arm's hand can reach the handle
        # through the slot, but then no part of that arm comes within 3.4 cm of either door; the south arm's, holding
        # the handle, within 1.4 cm, the ceiling between: docs/TASKS.md)
        out_of_reach=out_of_reach,
        box_width_m=width, doors_needed_deg=round(need, 1), paint={"box": "orange", "mat": "green"}, **extra,
    )


#: the keyframe `mrbench render` shows between start and goal: both doors held open, the box drawn out between them
MIDDLE = 1


def _opened(sc, deg: float) -> dict:
    return {h.body: rotate_about(sc.obj(h.body).pose, h.pivot, yaw_quat(h.sign * deg)) for h in sc.hinges}


def keyframes(inst):
    sc = inst.scene
    z = sc.obj("box").pos[2]
    first = _opened(sc, HOLD_DEG)
    if inst.variant == "borrow":  # the east door held open by the block, set in its lever's way back
        prop = inst.metadata["borrow"]["props"][0]
        first["door_e"] = _opened(sc, PUSH_DEG)["door_e"]
        turn = PUSH_DEG + _gap_deg()
        pose = _block_pose(sc.obj(prop), turn)
        first[prop] = (*pose[:3], *quat_mul(yaw_quat(-HAND_TURN * turn), pose[3:]))  # turned as the hand left it
    return [
        first,
        {"box": at((X0, FRONT_Y - BOX_D / 2 - 0.03, z))},
        {"box": at((*MAT, Z + BOX_H / 2 + 0.001))},
        {h.body: sc.obj(h.body).pose for h in sc.hinges},
    ]


def _block_pose(blk, deg: float):
    """The block carried ``deg`` round the east lever's hinge with the bearing it started with (the hand on it,
    ``on_block``, turns it as it turns): its middle where it would be if it stayed square to the lever.  Turned against
    the lever, its corner then reaches 1.7 cm past the bar's face at the push's end (and 0.6 cm past the stub's): the
    loaded arm lags about that much -- placed with its corner just on the bar, the weight pushes the door 77 degrees
    open only, and 1.7 cm past it, onto its stop (Isaac)."""
    xy = block_turn(block_corner(), deg)
    return (*xy, blk.pos[2], *blk.pose[3:])


def script(inst):
    """The scripted reference (privileged: a grip fixes what is gripped to the hand)."""
    if inst.variant == "borrow":
        return _borrow_script(inst)
    sc = inst.scene
    grips = {}
    for h in sc.hinges:  # from above, the fingers closing across the knob
        door = sc.obj(h.body)
        knob = transform(door.pose, door.part("knob").pos)
        knob = (knob[0], knob[1], LEVER_Z + KNOB_H - 0.015)
        # the fingers' travel along x: turned with the door, the hand's long side lies along y, its thin side toward
        # the box going out between the two hands
        grips[h.body] = ((*knob, *tool_quat((0.0, 0.0, -1.0), (1.0, 0.0, 0.0))), h)

    def arc(body, a0, a1):
        pose, h = grips[body]
        return lambda s: rotate_about(pose, h.pivot, yaw_quat(h.sign * (a0 + (a1 - a0) * s)))

    (gw, _), (ge, _) = grips["door_w"], grips["door_e"]
    b = sc.obj("box")

    def on_handle(y, pitch_deg):
        """The tool point on the box's handle with the box's middle at ``y`` (where it starts, in x and z), the hand
        pointing north and ``pitch_deg`` down; the fingers close across the bar (along x) in the plane square to the
        hand: its thin side across the slots."""
        p = math.radians(pitch_deg)
        h = transform(b.pose, b.part("handle").pos)
        if pitch_deg >= 90.0:
            return (h[0], h[1] + y - b.pos[1], h[2], *tool_quat((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)))
        return (h[0], h[1] + y - b.pos[1], h[2], *tool_quat((0.0, math.cos(p), -math.sin(p)),
                                                              (0.0, math.sin(p), math.cos(p))))

    inside = on_handle(b.pos[1], HANDLE_PITCH_DEG)
    # drawn out of the box zone pointing in, then taken again from straight above over the door zone's ceiling, and
    # drawn out through the doors
    out1 = ZONE_Y - 0.10  # its handle 10 cm out of the box zone: the hand straight above it clears the roof's front
    takes = [(b.pos[1], out1, HANDLE_PITCH_DEG), (out1, FRONT_Y - BOX_D / 2 - 0.03, 90.0)]
    on_mat = offset(on_handle(MAT[1], 90.0), dx=MAT[0] - b.pos[0], dz=(Z + BOX_H / 2 + 0.002) - b.pos[2])
    w, e, s = "west", "east", "south"
    held_w, held_e = arc("door_w", HOLD_DEG, HOLD_DEG)(0), arc("door_e", HOLD_DEG, HOLD_DEG)(0)
    drawn = []
    for k, (y0, y1, pd) in enumerate(takes):
        if k:  # let go and take it again, pointing further down
            drawn += [phase(0.4, "lets go", Release(s)),
                      phase(1.2, "", Move(s, offset(on_handle(y0, takes[k - 1][2]), dy=-0.03, dz=0.05))),
                      phase(2.0, "and takes it again from straight above", Move(s, offset(on_handle(y0, pd), dz=0.06),
                                                                                joints=True)),
                      phase(1.0, "", Move(s, offset(on_handle(y0, pd), dz=0.003))),
                      phase(0.4, "", Grip(s, "box", BAR_GRIP))]
        # lifted a centimetre at the end of the first take (2.7 cm under the ceiling): its fingers, opening to let go,
        # then clear the ceiling
        up = HOLD_UP if k == len(takes) - 1 else 0.01
        note = ("draws it out of the box zone, under the ceiling", "and out through the open doors")[k]
        drawn.append(phase(3.5, note, Move(s, offset(on_handle(y1, pd), dz=0.003 + up))))
    return [
        # the side arms' hands at home are 4 cm from the cabinet's sides: they back away before they go round
        phase(1.5, "the side arms back away from the cabinet; the middle arm makes room",
              Move(w, offset(home_pose(inst, w), dx=-0.10, dz=0.06)), Move(e, offset(home_pose(inst, e), dx=0.10, dz=0.06)),
              Move(s, PARK)),
        phase(2.5, "the side arms come over the knobs", Move(w, offset(gw, dz=0.10)), Move(e, offset(ge, dz=0.10)),
              Open(w, 0.05), Open(e, 0.05)),
        phase(1.0, "", Move(w, gw), Move(e, ge)),
        phase(0.4, "and grip them", Grip(w, "door_w", BAR_GRIP), Grip(e, "door_e", BAR_GRIP)),
        phase(4.0, "they swing the levers round: the doors open inward against their springs",
              Move(w, arc("door_w", 0, HOLD_DEG)), Move(e, arc("door_e", 0, HOLD_DEG))),
        phase(3.0, "and hold them; the middle arm comes up to the box zone's slot",
              Move(s, offset(inside, dy=-0.20, dz=0.08), joints=True)),
        phase(2.0, "reaches in over the box", Move(s, offset(inside, dz=0.04))),
        phase(1.0, "", Move(s, inside)),
        phase(0.4, "takes the box by its handle", Grip(s, "box", BAR_GRIP)),
        *drawn,
        # held up in front of the doorway, its bottom over the levers: the doors close under it, the levers swinging
        # back out beside it
        phase(0.3, "the side arms let go: the springs close the doors", Release(w), Release(e)),
        phase(1.5, "", Move(w, offset(held_w, dz=0.12)), Move(e, offset(held_e, dz=0.12))),
        phase(2.0, "and go home", Move(w, home_pose(inst, w)), Move(e, home_pose(inst, e))),
        phase(2.5, "the middle arm carries the box over the mat", Move(s, offset(on_mat, dz=0.06))),
        phase(1.5, "and sets it down", Move(s, on_mat)),
        phase(0.4, "", Release(s)),
        phase(1.5, "", Move(s, offset(on_mat, dz=0.12))),
        phase(2.0, "", Move(s, PARK)),
        phase(3.0, "done"),
    ]


def _borrow_script(inst):
    """The borrow configuration (no east arm): the west arm holds its door open as in the exact one; the south arm,
    holding the block, pushes the east lever round with it and leaves it in the lever's way back, takes the box out as
    in the exact configuration and sets it down west of the doorway, lifts the block away (the east door closes), and puts
    the box on the mat."""
    sc = inst.scene
    w, s = "west", "south"
    prop = inst.metadata["borrow"]["props"][0]
    blk = sc.obj(prop)
    h_w = next(h for h in sc.hinges if h.body == "door_w")
    door = sc.obj("door_w")
    knob = transform(door.pose, door.part("knob").pos)
    gw = (knob[0], knob[1], LEVER_Z + KNOB_H - 0.015, *tool_quat((0.0, 0.0, -1.0), (1.0, 0.0, 0.0)))

    def arc_w(a0, a1):
        return lambda u: rotate_about(gw, h_w.pivot, yaw_quat(h_w.sign * (a0 + (a1 - a0) * u)))

    held_w = arc_w(HOLD_DEG, HOLD_DEG)(0)
    # the block taken from above, the fingers closing along y on its top (over the bar, clear of it); turned round
    # the hinge with the lever, the hand turns with it
    up = blk.size[2] / 2 - BLOCK_TAKE  # the tool point over the block's middle

    turn = PUSH_DEG + _gap_deg()

    def on_block(pose, turned=0.0):
        """The hand on the handle, turned ``turned`` degrees round with the lever (the handle is square: the weight
        keeps its bearing, the hand turns about the vertical, so that at the end its broad side lies across the push,
        clear of the cabinet's corner -- along it, it comes within 0.6 cm of the side wall), leaning BLOCK_TILT_DEG
        all the while (a grip fixes the weight to the hand: turned upright as it pushes, the weight rides onto the
        lever's bar in Isaac)"""
        t = math.radians(BLOCK_TILT_DEG)
        q = tool_quat((-math.sin(t), 0.0, -math.cos(t)), (0.0, 1.0, 0.0))  # the wrist east of the fingers
        return (pose[0], pose[1], pose[2] + up, *_turn_quat(q, quat_mul(yaw_quat(-HAND_TURN * turned), pose[3:])))

    start = on_block(blk.pose)
    lifted = PUSH_UP  # its bottom over the table while it is pushed
    over = 0.14  # carried over the lever, its bottom clear of the knob's top (11 cm up)

    def push(u):  # let down into the corner, turned round the hinge, held there, then set down
        if u < 0.15:
            return offset(on_block(_block_pose(blk, 0.0)), dz=lifted + over * (1.0 - u / 0.15))
        if u < 0.55:
            a = turn * (u - 0.15) / 0.4
            return offset(on_block(_block_pose(blk, a), a), dz=lifted)
        if u < 0.9:
            return offset(on_block(_block_pose(blk, turn), turn), dz=lifted)
        return offset(on_block(_block_pose(blk, turn), turn), dz=lifted + (SET_UP - lifted) * (u - 0.9) / 0.1)

    set_ = offset(on_block(_block_pose(blk, turn), turn), dz=SET_UP)
    away = offset(set_, dx=RISE_AWAY[0], dz=RISE_AWAY[1])
    b = sc.obj("box")

    def on_handle(y, pitch_deg):
        p = math.radians(pitch_deg)
        h = transform(b.pose, b.part("handle").pos)
        if pitch_deg >= 90.0:
            return (h[0], h[1] + y - b.pos[1], h[2], *tool_quat((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)))
        return (h[0], h[1] + y - b.pos[1], h[2], *tool_quat((0.0, math.cos(p), -math.sin(p)),
                                                              (0.0, math.sin(p), math.cos(p))))

    inside = on_handle(b.pos[1], HANDLE_PITCH_DEG)
    out1 = ZONE_Y - 0.10
    takes = [(b.pos[1], out1, HANDLE_PITCH_DEG), (out1, FRONT_Y - BOX_D / 2 - 0.03, 90.0)]
    out = on_handle(FRONT_Y - BOX_D / 2 - 0.03, 90.0)

    def box_on(xy):  # the hand on the handle with the box standing on the table at xy
        return offset(on_handle(xy[1], 90.0), dx=xy[0] - b.pos[0], dz=(Z + BOX_H / 2 + 0.002) - b.pos[2])

    wait, on_mat = box_on(WAIT), box_on(MAT)
    drawn = []
    for k, (y0, y1, pd) in enumerate(takes):
        if k:
            drawn += [phase(0.4, "lets go", Release(s)),
                      phase(1.2, "", Move(s, offset(on_handle(y0, takes[k - 1][2]), dy=-0.03, dz=0.05))),
                      phase(2.0, "and takes it again from straight above", Move(s, offset(on_handle(y0, pd), dz=0.06),
                                                                                joints=True)),
                      phase(1.0, "", Move(s, offset(on_handle(y0, pd), dz=0.003))),
                      phase(0.4, "", Grip(s, "box", BAR_GRIP))]
        lift = HOLD_UP if k == len(takes) - 1 else 0.01
        note = ("draws it out of the box zone, under the ceiling", "and out through the open doors")[k]
        drawn.append(phase(3.5, note, Move(s, offset(on_handle(y1, pd), dz=0.003 + lift))))
    return [
        phase(1.5, "the west arm backs away from the cabinet; the middle arm makes room",
              Move(w, offset(home_pose(inst, w), dx=-0.10, dz=0.06)), Move(s, PARK)),
        phase(2.5, "the west arm comes over its knob; the middle arm over the block", Move(w, offset(gw, dz=0.10)),
              Open(w, 0.05), Move(s, offset(start, dz=0.10), turn=HOLD_TURN), Open(s, 0.08)),
        phase(1.0, "", Move(w, gw), Move(s, start, turn=HOLD_TURN)),
        phase(0.4, "they take the knob and the block", Grip(w, "door_w", BAR_GRIP), Grip(s, prop, HANDLE[1] - 0.003)),
        phase(4.0, "the west arm swings its lever round and holds the door open; the middle arm lifts the block",
              Move(w, arc_w(0, HOLD_DEG)), Move(s, offset(start, dz=over))),
        phase(3.0, "and carries it over the east lever's corner",
              Move(s, offset(on_block(_block_pose(blk, 0.0)), dz=lifted + over), joints=True)),
        phase(9.0, "lets it down there and pushes the east lever round with it: the east door opens; sets it down",
              Move(s, push)),
        phase(0.4, "and lets go of it in the lever's way back", Release(s), Open(s, RELEASE_OPEN)),
        # the fingers open wide before the hand rises (still closing round the handle as it rises, they tip the
        # weight and the lever pushes it back); turned, their pads lie along the rise.  Drawn back along itself
        # instead, the turned hand meets the knob
        # then straight up, as tilted as it held the weight: turning upright 4 cm up, its fingertips swing into the
        # handle and tip the weight, and the door falls back
        phase(0.5, ""),
        phase(1.0, "", Move(s, offset(set_, dz=0.06), turn=HOLD_TURN)),
        phase(1.0, "", Move(s, offset(set_, dz=0.10), turn=HOLD_TURN)),
        # east, away from the cabinet's corner, before it turns for what comes next (turned by it, the arm swings
        # into the side wall; straight up from there, the elbow runs out of bend)
        phase(1.5, "", Move(s, away, turn=HOLD_TURN)),
        # to where it waits in the exact configuration, home and back, from which it reaches in as there
        phase(1.5, "", Move(s, PARK)),
        phase(2.0, "", Move(s, home_pose(inst, s), joints=True)),
        phase(1.5, "", Move(s, PARK)),
        # the way round the exact plan reaches in, its wrist camera on the hand's east side (the handle stands west
        # of the box's middle for it to clear the slot's east edge): back from the block, the planner would keep the
        # other way and the camera come within 1 mm of the slot's west edge
        phase(3.0, "the middle arm comes up to the box zone's slot", Move(s, offset(inside, dy=-0.20, dz=0.08),
                                                                          joints=True, turn=REACH_TURN)),
        phase(2.0, "reaches in over the box", Move(s, offset(inside, dz=0.04), turn=REACH_TURN)),
        phase(1.0, "", Move(s, inside, turn=REACH_TURN)),
        phase(0.4, "takes the box by its handle", Grip(s, "box", BAR_GRIP)),
        *drawn,
        phase(0.3, "the west arm lets go: its door closes under the box", Release(w)),
        phase(1.5, "", Move(w, offset(held_w, dz=0.12))),
        phase(2.0, "and goes home", Move(w, home_pose(inst, w))),
        phase(3.0, "the middle arm sets the box down west of the doorway", Move(s, offset(wait, dz=0.06))),
        phase(1.5, "", Move(s, wait)),
        phase(0.4, "", Release(s)),
        phase(1.0, "", Move(s, offset(wait, dz=0.10))),
        phase(2.0, "", Move(s, PARK, joints=True)),
        # back the way it came: to where it turned, east of the block, and down to it the way round it let go (in
        # joint space to 20 cm over it, the hand swings under the ceiling's front edge; straight there, its wrist
        # turns 160-210 degrees on the way, the arm lags and swings into the cabinet's corner)
        phase(3.0, "takes the block again", Move(s, away, turn=HOLD_TURN), Open(s, RELEASE_OPEN)),
        phase(1.5, "", Move(s, offset(set_, dz=0.06), turn=HOLD_TURN)),
        phase(1.0, "", Move(s, set_, turn=HOLD_TURN)),
        phase(0.4, "", Grip(s, prop, HANDLE[1] - 0.003)),
        # straight up (moved east too as it rises, the weight is still at the lever's height when the door swings
        # shut, and the knob knocks it over), over the knob's top, and held there while the loaded arm catches up and
        # the door shuts (taken on at once, the arm behind, the weight goes east at the lever's height and the closing
        # door's stub drags it over)
        phase(1.5, "and lifts it out of the lever's way: the east door closes", Move(s, offset(set_, dz=LIFT_AWAY))),
        phase(1.0, ""),
        phase(3.0, "puts it back where it was", Move(s, offset(start, dz=LIFT_AWAY), joints=True)),
        phase(1.5, "", Move(s, offset(start, dz=0.002))),
        phase(0.4, "", Release(s)),
        phase(1.0, "", Move(s, offset(start, dz=0.10))),
        phase(2.0, "", Move(s, PARK, joints=True)),
        phase(3.0, "the middle arm takes the box again", Move(s, offset(wait, dz=0.06), joints=True)),
        phase(1.0, "", Move(s, wait)),
        phase(0.4, "", Grip(s, "box", BAR_GRIP)),
        phase(1.5, "", Move(s, offset(wait, dz=CARRY_UP))),
        # over the shut levers in front of the doorway: straight across, the box passes 8 mm from the middle arm's base
        phase(2.0, "carries it over the mat", Move(s, offset(box_on((X0, FRONT_Y - 0.13)), dz=CARRY_UP))),
        phase(2.0, "", Move(s, offset(on_mat, dz=CARRY_UP))),
        phase(2.0, "and sets it down", Move(s, on_mat)),
        phase(0.4, "", Release(s)),
        phase(1.5, "", Move(s, offset(on_mat, dz=0.12))),
        phase(2.0, "", Move(s, PARK)),
        phase(3.0, "done"),
    ]


def _turn_quat(q, body_q):
    """Tool orientation ``q`` turned as a body turned from no yaw to ``body_q`` (about the vertical)."""
    from ...engines.base import quat_mul

    return quat_mul(body_q, q)


def jobs(inst):
    """What the task asks (``tasks/jobs.py``): both doors held open at once, each by its knob, while the box is drawn
    out between them by its handle; then the box goes on the mat.  The borrow configuration's weight holds the east door
    open once it is set in its lever's way back; the low one does not."""
    from ..jobs import Job, Jobs, Thing

    sc = inst.scene

    def knob(body, deg):  # the knob's middle with the door ``deg`` open
        h = next(h for h in sc.hinges if h.body == body)
        door = sc.obj(body)
        return transform(rotate_about(door.pose, h.pivot, yaw_quat(h.sign * deg)), door.part("knob").pos)

    b = sc.obj("box")
    handle = transform(b.pose, b.part("handle").pos)
    up = handle[2] - b.pos[2]
    drawn = (handle[0], FRONT_Y - BOX_D / 2 - 0.03 + handle[1] - b.pos[1], handle[2] + HOLD_UP)  # out of the doorway
    on_mat = (MAT[0] + handle[0] - b.pos[0], MAT[1] + handle[1] - b.pos[1], Z + BOX_H / 2 + 0.002 + up)
    grip = 0.02  # a hand on a knob or a handle, within a grip's reach
    held = [Job(f"hold the {side} door open by its knob", sites=(knob(f"door_{side[0]}", 0.0),
                                                                  knob(f"door_{side[0]}", HOLD_DEG)),
                group="box out", slack=grip) for side in ("west", "east")]
    out = [*held, Job("draw the box out between the open doors by its handle", sites=(handle, drawn), group="box out",
                      slack=grip),
           Job("set the box on the mat", "carry", sites=(drawn, on_mat), slack=grip, body="box")]
    things = ()
    if inst.variant == "borrow":
        b = inst.metadata["borrow"]
        prop, decoy = b["props"][0], b["decoys"][0]
        # (taken by its handle, and set where it holds the door open)
        things = (Thing(prop, does=(held[1].name,),
                        placing=(Job(f"set {prop} in the east lever's way back", sites=(f"{prop}_top", "block_set"),
                                     slack=grip),),
                        why=f"{BLOCK[2] * 100:.0f} cm tall: the lever's bar ({(LEVER_Z - Z) * 100:.0f} cm up) cannot "
                            "swing back past it"),
                  Thing(decoy, why=f"{BLOCK_LOW * 100:.0f} cm tall: the lever's bar ({(LEVER_Z - Z) * 100:.0f} cm up) "
                                   "passes over it and the door closes"))
    return Jobs(tuple(out), things)


def decoy_stop(inst):
    """A distractor run's safe stop (``mrbench demo --decoy``): the east door, once opened, back within 10 degrees of
    shut -- the distractor has not kept it open."""
    from ...scoring.predicates import Opened

    return ~Opened("door_e", 10.0)
