"""Con-03 spring lid and basketball: one arm holds a spring lid open, the other two lift a ball too big for a hand
out of the box between them and set it in a cradle."""

from __future__ import annotations

import dataclasses
import math
import random

from ...engines.base import axis_quat, quat_mul
from ...policies.script import (Grip, Move, Open, Release, back_off, blend, home_pose, offset, phase, smooth,
                                tool_quat)
from ...scoring.predicates import Eventually, HoldsAtEnd, HoldsFor, InRegion, Lifted, Opened
from ..loader import get
from ._common import (Z, at, beam, box, compound, hinge, jitter, make_instance, part, rng_for, rotate_about, scene,
                      sphere, zone)

#: the box: its middle (x, y), outer width (along x) and depth, wall thickness and height (top at Z + WALL_H), floor
#: thickness.  Long along x, the lifters' arms come down into it from the west and east over its end walls, which
#: are lower (END_H: at 27 cm the west forearm came within 9 mm of its wall); the lid rests on the long walls, and
#: the 7 cm gaps under it at the ends are far too low for the ball.  42 cm deep inside, 10 cm round the ball (at
#: 38, the hands and forearms came within 1 cm of the upright lid or the north wall).
BOX_XY, BOX_W, BOX_D, WALL_T, WALL_H, END_H, FLOOR_T = (0.0, -0.26), 1.00, 0.44, 0.012, 0.27, 0.20, 0.024
#: the walls are slick (their friction against anything at most this): a ball squeezed between one hand and a wall
#: needs the hand to push 0.6 kg x 9.81 / (2 x 0.05) = 59 N to climb it (it rolls, so each contact carries half).
#: And each wall has a lip along its top, LIP wide inward (thickness LIP_T): a ball rolled up any wall stops under
#: it.  The ball comes out only lifted straight up through the middle of the opening, held from both its sides:
#: by IK, each end arm can put a hand on almost any side of the ball, so without the lips one hand with the lid
#: held might roll it up a wall and out (59 N)
WALL_FRICTION, LIP, LIP_T = 0.05, 0.03, 0.012
#: the ball: radius (22 cm across; a hand opens 8), mass, friction (rubber; with the fingers' 0.5, 1.0)
BALL_R, BALL_MASS, BALL_FRICTION = 0.11, 0.60, 1.5
#: the lid: a plate over the whole box, hinged along its south edge (it opens toward the south arm), its mass, the
#: tab on its top near the free (north) edge that a hand takes, how far the lid is held open (degrees: upright, at
#: the box's south edge, clear of the ball's way), its spring (the knob, N*m/rad, preloaded 10 degrees past shut) and
#: damping.  Gravity shuts it below upright and opens it past; the spring shuts it with 0.3 x 1.75 = 0.52 N*m held
#: upright, and at its 110 degree stop with 0.63 against gravity's 0.35 x 9.81 x 0.22 x sin 20 = 0.26: no angle
#: holds it open
LID_T, LID_MASS, OPEN_DEG, PRELOAD_DEG, LID_DAMPING = 0.018, 0.35, 90.0, 10.0, 0.08
TAB, TAB_Y = (0.02, 0.10, 0.04), 0.11
#: the tab is a handle: a grip bar TAB[0] across (the fingers close across it), TAB[1] long and
#: HANDLE_BAR thick on two posts HANDLE_POST long, its middle where the hand takes the tab (the tab's upper half).  The
#: box is a tote's, grey plastic, and its lid's top is ribbed: RIBS (lid-local y) across it, RIB_X each way from its
#: middle, joined at their ends, RIB wide and high -- clear of the lid's ends (an end arm takes the lid there), of its
#: free edge (a stay's hook goes over it) and of the handle
HANDLE_BAR, HANDLE_POST = 0.012, 0.02
RIBS, RIB_X, RIB = (-0.14, -0.075, -0.01), 0.38, (0.012, 0.008)
#: the hand on the tab is fixed to the lid and turns with it: it takes the tab pointing this far below level, 30
#: degrees past straight down toward the south: from 75 to 105 no IK solution followed the lid round to upright
#: (taken pointing straight down, the hand ends level near the shoulder's height); from 120 and 135 most do
TAB_GRIP_DEG = 120.0
#: the cradle: a block north of the box, its top this high, four rails on it the ball sits in.  54 cm north of the
#: ball (at y = 0.28): nearer, the south arm came within 5 cm of it (IK), and it should be the end arms' only
CRADLE_DY, CRADLE_TOP, CRADLE_BLOCK = 0.54, 0.30, (0.30, 0.22)
RAIL_H, RAIL_T, RAIL_X, RAIL_Y = 0.06, 0.012, 0.09, 0.08
#: how far the fingertips (closed, 1 cm past the tool point) press into the ball's side, and how high it is lifted.
#: The hands first touch it together (TOUCH: a hand that reached it first pushed it 3.5 cm along the floor and the
#: other then pressed alone), then press in together
PRESS, LIFT, TOUCH = 0.01, 0.37, 0.003
#: and how much higher still it is lifted and carried to the cradle, to be lowered by as much over it before it is set
#: down.  In Isaac the ball runs 2.6-3.9 cm under its plan over the cradle's south rail (lower in the hands than
#: planned, and the loaded hands sag as they reach north): carried at the lift's height, planned 3.4 cm over the rail,
#: it scrapes over it and can catch on it.  Carried this much higher, it is planned 8.4 cm over the rail
CARRY_UP = 0.05
FINGERTIPS = 0.01
#: how far off the ball's side the fingertips stop, coming down into the box beside it: 3 cm off, the west forearm
#: can come within 1 cm of the box's west lip (the clearance rule); 2 cm off, it does not
BESIDE = 0.02
#: the lifting hands point this far below level: held level, a Franka could not carry the ball north at the lift's
#: height (no IK solution follows the line: a level hand straight out near the shoulder's height).  Their
#: fingertips touch the ball just below its middle (CONTACT_DZ), where pressing in also lifts it a little: 3 cm
#: above it, they rode up over the ball as they pressed and lifted without it.  The hands are rolled so that their
#: fingers are one over the other: the wrist camera, on the hand's front, is then beside the hand, not over the
#: ball (with the fingers side by side it came within 1 cm of the ball at its middle)
SIDE_PITCH, CONTACT_DZ = 45.0, -0.01

#: the borrow version (no south arm): a hooked stay holds the lid open.  Each of two stays stands in a socket on the
#: table south of the box, its foot on a pivot along x (STAY_X either side of the middle, STAY_Y, PIVOT_H up), leaning
#: STAY_REST degrees south on its stop: it swings north (to STAY_MAX past upright) and, the lid open past it, its hook
#: -- a bight over the lid's free edge, a tip down its underside -- catches the edge and holds the lid EQ_DEG open
#: (upright and 2 degrees past: the lifters' route is clear only with the lid upright).  The stay pulls the edge
#: toward its foot: it is in tension, so its foot is on a pivot (a foot in an open socket would be pulled out).  The
#: distractor is the same stay DECOY_DL longer or DECOY_DL shorter (one attribute, the bar's length; which way is drawn
#: from the seed, so the prop is the bigger of the two in half the scenes).  The lid's free edge comes 0.662-0.772 m
#: from the pivot over its whole 0-110 degree travel (its corners; nearest at the 110 degree stop, furthest at 48
#: degrees).  The longer stay's hook tip ends 0.822 m out, beyond the edge at every opening: its hook passes over the
#: edge and catches nothing.  The shorter stay's bight is 0.602 m out at most, short of the edge at every opening: the
#: edge can never get under it.  (A wider or narrower hook mouth would not make a distractor: a narrow one would not go
#: over the lid's 1.8 cm edge, but a wide one, pulled taut, still catches it.)  Neither stay can prop
#: the lid from below: its pivot is 0.22 m south of the lid's hinge and it swings north at most STAY_MAX, so it only
#: ever meets the lid's top face, which a stay can push only toward shut.  Its north stop (STAY_MAX) keeps either stay
#: south of the lid's sweep: further over, a stay on its stop would stand in the closing lid's way and prop it.  Which
#: stands west and which east is drawn from the seed.  A rod from a socket south of the box to the lid's tab would not
#: do: the tab is 0.49 m from such a socket even with the lid shut, 0.61 m at 60-90 degrees, and the end arms reach the
#: tab only while the lid is open 60 degrees or less, where the lifters' hands meet it (IK).  Each end arm reaches its
#: own stay and the lid's own end (IK): one opens the lid by its end, the other swings the stay on its side over the
#: edge.  STAY_BAR: the bar's side; HOOK_SIDE: how far the bar is from the lid's top face, hooked on; HOOK_TIP: how far
#: the hook's tip reaches down the lid's underside.
STAY_X, STAY_Y, PIVOT_H, STAY_BAR, HOOK_SIDE, HOOK_TIP = 0.46, -0.70, 0.03, 0.016, 0.01, 0.02
EQ_DEG, DECOY_DL, STAY_REST, STAY_MAX, STAY_KG = 92.0, 0.12, 4.0, 15.5, 0.3
#: the reference: the lid's holder takes it by its end (LID_TAKE along it from the hinge, LID_INSET in from the end,
#: the hand pointing in turned LID_YAW toward the lid's free edge and LID_PITCH down: so an end arm reaches the end,
#: and 6 cm back from it, at every opening to the stop -- pointing straight in, it could not draw back from the shut
#: lid, IK -- its hand clear of the end wall under the lid) and opens it until the stay's tip clears the edge by
#: HOOK_CLEAR as it swings over; the other arm takes its stay STAY_GRIP up from the pivot, its hand pointing in turned
#: STAY_YAW north (its palm, wide along the bar's swing, south of it and clear of the lid)
LID_TAKE, LID_INSET, LID_YAW, LID_PITCH, HOOK_CLEAR = 0.30, -0.006, 30.0, 0.0, 0.012
STAY_GRIP, STAY_YAW = 0.30, 0.0
#: then both hands go north round the lid's ends, to (x out from the middle, y, up) before the lifting begins
AROUND = (0.62, -0.30, 0.50)


def ball_rest_z() -> float:
    """The ball's middle when it sits in the cradle: on the inner top edges of the two rails along x (closer in)."""
    d = RAIL_Y - RAIL_T / 2
    return Z + CRADLE_TOP + RAIL_H + math.sqrt(BALL_R ** 2 - d ** 2)


def _box(x, y):
    hx, hy = BOX_W / 2, BOX_D / 2
    wz = Z + WALL_H / 2
    walls = [box("box_floor", (x, y), (BOX_W, BOX_D, FLOOR_T), "pvc", dynamic=False),
             box("wall_w", (x - hx + WALL_T / 2, y), (WALL_T, BOX_D, END_H), "pvc", dynamic=False),
             box("wall_e", (x + hx - WALL_T / 2, y), (WALL_T, BOX_D, END_H), "pvc", dynamic=False),
             box("wall_s", (x, y - hy + WALL_T / 2), (BOX_W - 2 * WALL_T, WALL_T, WALL_H), "pvc", z=wz, dynamic=False),
             box("wall_n", (x, y + hy - WALL_T / 2), (BOX_W - 2 * WALL_T, WALL_T, WALL_H), "pvc", z=wz, dynamic=False)]
    inner_w, inner_d = BOX_W - 2 * WALL_T, BOX_D - 2 * WALL_T
    lz, ez = Z + WALL_H - LIP_T / 2, Z + END_H - LIP_T / 2
    walls += [box("lip_s", (x, y - inner_d / 2 + LIP / 2), (inner_w, LIP, LIP_T), "pvc", z=lz, dynamic=False),
              box("lip_n", (x, y + inner_d / 2 - LIP / 2), (inner_w, LIP, LIP_T), "pvc", z=lz, dynamic=False),
              box("lip_w", (x - inner_w / 2 + LIP / 2, y), (LIP, inner_d - 2 * LIP, LIP_T), "pvc", z=ez,
                  dynamic=False),
              box("lip_e", (x + inner_w / 2 - LIP / 2, y), (LIP, inner_d - 2 * LIP, LIP_T), "pvc", z=ez,
                  dynamic=False)]
    return [dataclasses.replace(w, friction=WALL_FRICTION, friction_combine="min") if w.name != "box_floor" else w
            for w in walls]


def _lid(x, y, spring):
    grip = LID_T / 2 + TAB[2] * 3 / 4  # the handle's bar: where the hand takes the tab
    post = grip - HANDLE_BAR / 2 - LID_T / 2
    parts = [part("plate", "box", (BOX_W, BOX_D, LID_T)),
             *(part(f"rib_{i}", "box", (2 * RIB_X + RIB[0], RIB[0], RIB[1]), (0.0, y, LID_T / 2 + RIB[1] / 2))
               for i, y in enumerate(RIBS)),
             *(part(f"rib_end_{k}", "box", (RIB[0], RIBS[-1] - RIBS[0], RIB[1]),
                    (s * RIB_X, (RIBS[0] + RIBS[-1]) / 2, LID_T / 2 + RIB[1] / 2)) for k, s in (("w", -1), ("e", 1))),
             part("tab", "box", (TAB[0], TAB[1], HANDLE_BAR), (0.0, TAB_Y, grip), color="iron"),
             *(part(f"tab_post_{k}", "box", (TAB[0], HANDLE_POST, post),
                    (0.0, TAB_Y + s * (TAB[1] - HANDLE_POST) / 2, LID_T / 2 + post / 2), color="iron")
               for k, s in (("s", -1), ("n", 1)))]
    lid = compound("lid", (x, y, Z + WALL_H + LID_T / 2), parts, "blue", mass=LID_MASS)
    pivot = (x, y - BOX_D / 2, Z + WALL_H + LID_T / 2)  # its south edge
    return lid, hinge("lid_hinge", lid, pivot, sign=1, axis=(1.0, 0.0, 0.0), open_max_deg=110.0, stiffness=spring,
                      damping=LID_DAMPING, preload_deg=PRELOAD_DEG)


def lid_corners(deg: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """The (y, z) of the lid's free edge's two corners with the lid ``deg`` open: its underside's (north, when upright)
    and its top face's."""
    t, r, h = math.radians(deg), BOX_D, LID_T / 2
    y, z = BOX_XY[1] - BOX_D / 2 + r * math.cos(t), Z + WALL_H + LID_T / 2 + r * math.sin(t)
    return (y + h * math.sin(t), z - h * math.cos(t)), (y - h * math.sin(t), z + h * math.cos(t))


def stay_pivot() -> tuple[float, float]:
    """(y, z) of the stays' pivots."""
    return STAY_Y, Z + PIVOT_H


#: the stays' frame: its x north and its y up (the plane it swings in), its z along the pivot (west to east) -- so
#: that every part is a bar in its x-y plane (the engine turns a part only about the frame's z)
SWING_FRAME = (0.5, 0.5, 0.5, 0.5)


def _hook() -> dict:
    """The prop's shape in the plane it swings in, (north, up) from its pivot, leaning as it does on the lid's edge
    with the lid EQ_DEG open: its bar's top ``top``; its bight, level, its underside ``under`` up, on the edge; its
    tip, upright, its inner face ``tip`` north, on the lid's underside; the bar HOOK_SIDE south of the lid's top
    face."""
    (ny, nz), (sy, sz) = lid_corners(EQ_DEG)
    fy, fz = stay_pivot()
    under = max(nz, sz) - fz
    return {"top": (sy - fy - HOOK_SIDE - STAY_BAR / 2, under + STAY_BAR / 2), "under": under, "tip": ny - fy}


def _turn(pt, deg: float):
    """A point (north, up) of the swing plane turned ``deg`` north about the pivot."""
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return (pt[0] * c + pt[1] * s, -pt[0] * s + pt[1] * c)


def stay_lean() -> float:
    """How far (degrees) the prop's bar leans north with its hook on the lid's edge, the lid EQ_DEG open."""
    u, v = _hook()["top"]
    return math.degrees(math.atan2(u, v))


def tip_end(lean: float) -> tuple[float, float]:
    """(y, z) of the prop's tip's inner lower corner, the prop leaning ``lean``: of its hook, what passes nearest the
    pivot as it swings."""
    h = _hook()
    u, v = _turn((h["tip"], h["under"] - HOOK_TIP), lean - stay_lean())
    fy, fz = stay_pivot()
    return fy + u, fz + v


def hold_deg() -> float:
    """How far the reference opens the lid to put the prop's hook over its edge: till the tip's inner lower corner,
    swung round the pivot, passes HOOK_CLEAR beyond both of the edge's corners."""
    fy, fz = stay_pivot()
    ty, tz = tip_end(0.0)
    tip = math.hypot(ty - fy, tz - fz)
    lo, hi = EQ_DEG, 110.0
    for _ in range(40):
        mid = (lo + hi) / 2
        far = max(math.hypot(y - fy, z - fz) for y, z in lid_corners(mid))
        lo, hi = (lo, mid) if tip - far >= HOOK_CLEAR else (mid, hi)
    return hi


def over_lean(deg: float) -> float:
    """How far the reference swings the prop north to put its hook over the edge of the lid ``deg`` open: till its
    tip's inner face is HOOK_CLEAR north of the lid's underside corner."""
    ny = lid_corners(deg)[0][0]
    lo, hi = -STAY_REST, STAY_MAX
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if tip_end(mid)[0] < ny + HOOK_CLEAR else (lo, mid)
    return hi


def decoy_dl(level: str, seed: int) -> float:
    """How much longer the distractor is than the prop: +DECOY_DL or -DECOY_DL, alternating with the seed (and the
    other way round at the hard level), so that over any run of seeds the prop is the bigger stay half the time."""
    return DECOY_DL if (seed + (level == "hard")) % 2 == 0 else -DECOY_DL


def _stay(name: str, x: float, longer: float = 0.0):
    """A hooked stay on its pivot, leaning STAY_REST south on its stop; its frame (SWING_FRAME) at the pivot;
    ``longer`` longer than the prop (negative: shorter), its hook as the prop's."""
    h, s = _hook(), STAY_BAR
    lean = math.radians(stay_lean())
    du, dv = longer * math.sin(lean), longer * math.cos(lean)  # the hook moved up along the bar
    (tu, tv), u_tip, under = h["top"], h["tip"] + s / 2, h["under"]
    tu, tv, u_tip, under = tu + du, tv + dv, u_tip + du, under + dv
    parts = [beam("bar", (-s / 2 * math.sin(lean), -s / 2 * math.cos(lean)), (tu, tv), s),
             beam("bight", (tu - s / 2, tv), (u_tip + s / 2, tv), s),
             beam("tip", (u_tip, tv + s / 2), (u_tip, under - HOOK_TIP), s, color="safety_yellow")]
    fy, fz = stay_pivot()
    rest = quat_mul(axis_quat((1.0, 0.0, 0.0), stay_lean() + STAY_REST), SWING_FRAME)
    return compound(name, (x, fy, fz), parts, "iron", mass=STAY_KG, quat=rest)


def _socket(name: str, x: float) -> list:
    """The socket the stay's foot pivots in: two cheeks either side of it."""
    off = STAY_BAR / 2 + 0.004 + 0.003
    return [box(f"{name}_{k}", (x + sgn * off, STAY_Y), (0.006, 0.04, PIVOT_H + 0.02), "steel", dynamic=False)
            for k, sgn in (("w", -1), ("e", 1))]


def stay_at(obj, lean: float):
    """The pose of stay ``obj`` leaning ``lean`` degrees north (negative: south)."""
    return rotate_about(obj.pose, obj.pos, axis_quat((1.0, 0.0, 0.0), -(lean + STAY_REST)))


def _cradle(x, y):
    top = Z + CRADLE_TOP
    rz = top + RAIL_H / 2
    return [box("cradle", (x, y), (*CRADLE_BLOCK, CRADLE_TOP), "wood", dynamic=False),
            box("rail_w", (x - RAIL_X, y), (RAIL_T, 2 * RAIL_Y + RAIL_T, RAIL_H), "grey", z=rz, dynamic=False),
            box("rail_e", (x + RAIL_X, y), (RAIL_T, 2 * RAIL_Y + RAIL_T, RAIL_H), "grey", z=rz, dynamic=False),
            box("rail_s", (x, y - RAIL_Y), (2 * RAIL_X - RAIL_T, RAIL_T, RAIL_H), "grey", z=rz, dynamic=False),
            box("rail_n", (x, y + RAIL_Y), (2 * RAIL_X - RAIL_T, RAIL_T, RAIL_H), "grey", z=rz, dynamic=False)]


def build(level: str, seed: int, variant: str = "exact"):
    spec = get("Con-03")
    rng = rng_for(spec, level, seed)
    spring = spec.knob.levels[level]
    bx, by = BOX_XY
    walls = _box(bx, by)
    lid, lid_hinge = _lid(bx, by, spring)
    cx, cy = bx, by + CRADLE_DY
    bxy = jitter(rng, (bx, by), 0.01)  # the ball, give or take a centimetre (8 cm of room each side)
    ball = dataclasses.replace(sphere("ball", bxy, BALL_R, "orange", mass=BALL_MASS, z=Z + FLOOR_T + BALL_R),
                               friction=BALL_FRICTION)
    tab = (lid.pos[0], lid.pos[1] + TAB_Y, lid.pos[2] + LID_T / 2 + TAB[2] / 2)
    regions = [zone("cradle_in", (cx, cy, ball_rest_z()), (0.04, 0.04, 0.03)),
               zone("lid_tab", tab, (0.01, 0.01, 0.01))]
    objects, hinges = walls + [lid, ball] + _cradle(cx, cy), [lid_hinge]
    reach = [("south", "lid_tab"), ("west", "ball"), ("east", "ball"), ("west", "cradle_in"), ("east", "cradle_in")]
    out_of_reach, extra = [("south", "cradle_in")], {}
    if variant == "borrow":  # no south arm: a hooked stay holds the lid open, and a longer one that cannot
        # which side the prop stands is drawn from the seed, on its own stream (the exact configuration's scene is
        # unchanged)
        longer = [0.0, decoy_dl(level, seed)]
        random.Random(f"borrow/{spec.seed_id}/{level}/{seed}").shuffle(longer)
        names = ("stay_1", "stay_2")  # west, east
        stays = [_stay(n, sgn * STAY_X, d) for n, sgn, d in zip(names, (-1, 1), longer)]
        objects += stays + [c for n, sgn in zip(names, (-1, 1)) for c in _socket(f"{n}_socket", sgn * STAY_X)]
        hinges = [dataclasses.replace(lid_hinge, propped=True)]  # the clearance check: it rests where a stay holds it
        hinges += [hinge(f"{st.name}_hinge", st, st.pos, sign=-1, axis=(1.0, 0.0, 0.0),
                         open_max_deg=STAY_REST + STAY_MAX, stiffness=0.0, damping=0.05, preload_deg=0.0,
                         propped=True) for st in stays]
        prop = names[longer.index(0.0)]
        # where each end arm takes its stay, and the lid by its own end, shut and held open
        rest = math.radians(STAY_REST)
        for n, sgn, st in zip(names, (-1, 1), stays):
            regions.append(zone(f"{n}_grip", (st.pos[0], STAY_Y - STAY_GRIP * math.sin(rest),
                                              st.pos[2] + STAY_GRIP * math.cos(rest)), (0.01,) * 3))
            for deg, k in ((0.0, "shut"), (hold_deg(), "open")):
                t = math.radians(deg)
                regions.append(zone(f"lid_end_{'we'[sgn > 0]}_{k}",
                                    (sgn * (BOX_W / 2 - LID_INSET), lid_hinge.pivot[1] + LID_TAKE * math.cos(t),
                                     lid_hinge.pivot[2] + LID_TAKE * math.sin(t)), (0.01,) * 3))
        reach = [r for r in reach if r[0] != "south"]
        reach += [(slot, f"{n}_grip") for slot, n in zip(("west", "east"), names)]
        reach += [(slot, f"lid_end_{slot[0]}_{k}") for slot in ("west", "east") for k in ("shut", "open")]
        out_of_reach = [r for r in out_of_reach if r[0] != "south"]
        extra = {"borrow": {"props": [prop], "decoys": [n for n in names if n != prop]}}
    sc = scene(spec, objects, regions, hinges=hinges)
    if variant == "borrow":
        sc = dataclasses.replace(sc, arms=tuple(a for a in sc.arms if a.slot != "south"))
    in_cradle = InRegion("ball", "cradle_in")
    return make_instance(
        spec, level, seed, sc,
        instruction="Take the orange ball out of the box with the spring lid and set it in the cradle north of the box.",
        # the ball is in the cradle when the run ends, its last second at least: held there a second and
        # carried off, or let go and rolled out, it is not.  This asks more than the ``ball_settled`` milestone; it is
        # met at the same moment where the ball, once in the zone, does not leave it again
        success=HoldsAtEnd(in_cradle, 1.0),
        milestones={
            "lid_open": Eventually(Opened("lid", 80.0)),
            "ball_out": Eventually(Lifted("ball", WALL_H)),
            "ball_in_cradle": Eventually(in_cradle),
            "ball_settled": HoldsFor(in_cradle, 1.0),
        },
        # the lid's tab: the south arm (the end arms reach it too: whichever holds the lid, two others lift); the ball
        # in the box and in the cradle: each end arm reaches its side; the cradle is beyond the south arm
        reach=reach, out_of_reach=out_of_reach,
        lid_spring_nm_rad=spring, paint={"ball": "orange", "lid": "blue"}, **extra,
    )


#: the keyframe `mrbench render` shows between start and goal: the lid held open, the ball lifted over the box
MIDDLE = 1


def _opened(sc, deg):
    lid = sc.obj("lid")
    h = sc.hinges[0]
    return rotate_about(lid.pose, h.pivot, axis_quat(h.axis, h.sign * deg))


def keyframes(inst):
    sc = inst.scene
    ball = sc.obj("ball")
    cradle = sc.region("cradle_in").center
    if inst.variant == "borrow":  # the lid held open by the stay, hooked over its edge; it stays open
        prop = inst.metadata["borrow"]["props"][0]
        held = {"lid": _opened(sc, EQ_DEG), prop: stay_at(sc.obj(prop), stay_lean())}
        return [held, {"ball": at((ball.pos[0], ball.pos[1], ball.pos[2] + LIFT))}, {"ball": at(cradle)}]
    return [
        {"lid": _opened(sc, OPEN_DEG)},
        {"ball": at((ball.pos[0], ball.pos[1], ball.pos[2] + LIFT))},
        {"ball": at(cradle), "lid": sc.obj("lid").pose},
    ]


def _lift_phases(inst) -> list:
    """The end arms' part of the reference: over the box, down beside the ball, squeeze it between their fingertips,
    lift it out, carry it to the cradle, set it down and part."""
    sc = inst.scene
    ball = sc.obj("ball")
    bx, by, bz = ball.pos
    cradle = sc.region("cradle_in").center
    # each end hand points in at the ball's side, pitched SIDE_PITCH down, its fingers closed, across along y
    p = math.radians(SIDE_PITCH)
    side = {"west": -1.0, "east": 1.0}
    pointing = {a: (-side[a] * math.cos(p), 0.0, -math.sin(p)) for a in side}
    # the fingers travel in the vertical plane through the hand (square to the pointing): one finger over the other
    stacked = {a: (-side[a] * math.sin(p), 0.0, math.cos(p)) for a in side}
    point = {a: tool_quat(pointing[a], stacked[a]) for a in side}

    def at_side(slot, x, y, z, gap):  # fingertips `gap` off the ball's surface (negative: pressed in)
        contact = (x + side[slot] * (BALL_R + gap), y, z)
        return (*(c - FINGERTIPS * v for c, v in zip(contact, pointing[slot])), *point[slot])

    w, e = "west", "east"
    dz = CONTACT_DZ

    def held_after(a, b, k):
        """From ``a`` to ``b`` as ``Move(b)`` goes in the first ``k`` of the phase, then held at ``b``.  The planner
        hands a path its progress already eased (``smooth`` of the phase's time): the time is found again first."""
        def at(s):
            lo, hi = 0.0, 1.0
            for _ in range(40):
                lo, hi = (0.5 * (lo + hi), hi) if smooth(0.5 * (lo + hi)) < s else (lo, 0.5 * (lo + hi))
            return blend(a, b, smooth(min(1.0, 0.5 * (lo + hi) / k)))
        return at

    over = {a: back_off(at_side(a, bx, by, bz + dz, BESIDE), 0.15) for a in (w, e)}  # back along the hand
    high ={a: (*over[a][:2], Z + WALL_H + 0.15, *over[a][3:]) for a in (w, e)}  # straight above it, over the walls
    beside = {a: at_side(a, bx, by, bz + dz, BESIDE) for a in (w, e)}
    touch = {a: at_side(a, bx, by, bz + dz, TOUCH) for a in (w, e)}
    press = {a: at_side(a, bx, by, bz + dz, -PRESS) for a in (w, e)}
    lifted = {a: offset(press[a], dz=LIFT + CARRY_UP) for a in (w, e)}
    carried = {a: at_side(a, cradle[0], cradle[1], bz + LIFT + CARRY_UP + dz, -PRESS) for a in (w, e)}
    above = {a: offset(carried[a], dz=-CARRY_UP) for a in (w, e)}  # over the cradle at the lift's height
    lowered = {a: at_side(a, cradle[0], cradle[1], cradle[2] + 0.005 + dz, -PRESS) for a in (w, e)}
    eased = at_side(e, cradle[0], cradle[1], cradle[2] + 0.005 + dz, TOUCH)
    apart = {a: offset(at_side(a, cradle[0], cradle[1], cradle[2], 0.08), dz=0.08) for a in (w, e)}  # out and up
    return [
        # over the box, then straight down into it beside the ball (swung into the box in joint space, they passed
        # within 1 cm of its walls)
        phase(3.0, "the end arms come over the box", Move(w, high[w], joints=True), Move(e, high[e], joints=True),
              Open(w, 0.0), Open(e, 0.0)),
        phase(2.0, "and down into it beside the ball", Move(w, over[w]), Move(e, over[e])),
        phase(2.0, "and down beside it", Move(w, beside[w]), Move(e, beside[e])),
        phase(0.2, "", Grip(w, "ball", 0.0, False), Grip(e, "ball", 0.0, False)),
        phase(1.5, "they touch it, together", Move(w, touch[w]), Move(e, touch[e])),
        phase(1.5, "and squeeze it between their fingertips", Move(w, press[w]), Move(e, press[e])),
        # held between the two hands (privileged: the grip attaches it to the west hand only because the east one
        # is on its far side; either letting go drops it)
        phase(0.4, "", Grip(w, "ball", 0.0)),
        phase(3.0, "lift it out of the box", Move(w, lifted[w]), Move(e, lifted[e])),
        phase(3.5, "carry it to the cradle", Move(w, carried[w]), Move(e, carried[e])),
        # the loaded arms lag: they catch up over the cradle before setting it down.  2.5 s, not 1: the east hand,
        # slid 5-6 cm south along the ball while squeezing, pushes into it as they carry it and can shove the ball,
        # held by the west hand, back south; after 1 s the west hand can still be several centimetres off and let it
        # go off the cradle.  Then down CARRY_UP over the cradle, where the set-down begins
        phase(2.5, ""),
        phase(1.0, "and lower it over it", Move(w, above[w]), Move(e, above[e])),
        # setting it down, the east hand stops pressing; then both let go before parting (held, the ball went with
        # the west hand), and wait for it to settle (pressed in as they let go, they flicked it over 4 cm rails).
        # Letting go keeps the fingers closed: opened, the lower finger pried the ball up off the rails.
        # Set down (1.5 s), then held still 1 s before they let go.  The west hand, which holds the ball, ends the move
        # 3-5 cm north of its plan and 2-3 cm high; let go the moment the move ends (the arm still moving, the ball
        # sliding along the north rail's inner edge toward the middle) it can swing north-west and take the ball up
        # over that rail.  Held still first, the hand moves off slowly as it lets go, and the ball, free on the rail's
        # edge, rolls into the middle.  One move, so that the move that puts the ball down is the one before the release (``clearance``)
        phase(2.5, "and set it down in it", Move(w, held_after(above[w], lowered[w], 0.6)),
              Move(e, held_after(above[e], eased, 0.6))),
        phase(0.3, "let go", Release(w), Open(w, 0.0), Release(e), Open(e, 0.0)),
        phase(0.3, ""),
        phase(1.0, "", Move(w, apart[w]), Move(e, apart[e])),
    ]


def _borrow_script(inst):
    """The borrow configuration (no south arm): the end arm on the side away from the prop takes the lid by its own end and
    opens it; the other swings its stay north over the lid's edge; the two then let the lid back together, the hook
    coming down onto the edge, and let go of the stay, then of the lid, which the stay holds open.  Then the end
    arms lift the ball out as in the exact version.  Privileged: a grip fixes the lid, or the stay, to the hand."""
    sc = inst.scene
    lid_h = next(h for h in sc.hinges if h.body == "lid")
    stay = sc.obj(inst.metadata["borrow"]["props"][0])
    side = 1 if stay.pos[0] > 0 else -1  # the stay's side; the arm on the other holds the lid
    st, ho = ("east", "west") if side > 0 else ("west", "east")
    # the holder's hand points in at the lid's own end, LID_TAKE from the hinge, turned LID_YAW toward the free edge,
    # its fingers across the lid
    yaw, pitch = math.radians(LID_YAW), math.radians(LID_PITCH)
    end = (-side * (BOX_W / 2 - LID_INSET), lid_h.pivot[1] + LID_TAKE, lid_h.pivot[2],
           *tool_quat((side * math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), -math.sin(pitch)),
                      (0.0, 0.0, 1.0)))

    def lid_at(deg):
        return rotate_about(end, lid_h.pivot, axis_quat(lid_h.axis, lid_h.sign * deg))

    # the stay arm's hand points in at its stay, STAY_GRIP up it, turned STAY_YAW toward the north, its fingers
    # across the bar
    rest, a = math.radians(STAY_REST), math.radians(STAY_YAW)
    grip = (stay.pos[0], stay.pos[1] - STAY_GRIP * math.sin(rest), stay.pos[2] + STAY_GRIP * math.cos(rest),
            *tool_quat((-side * math.cos(a), math.sin(a) * math.cos(rest), math.sin(a) * math.sin(rest)),
                       (0.0, -math.cos(rest), -math.sin(rest))))

    def stay_hand(lean):
        return rotate_about(grip, stay.pos, axis_quat((1.0, 0.0, 0.0), -(lean + STAY_REST)))

    hold, eq = hold_deg(), stay_lean()
    over = over_lean(hold)  # the hook over the edge, its tip clear of the lid's underside
    let_go = EQ_DEG + 0.5  # the lid let go just short of the stay: its spring takes it the last half degree

    def together(s):  # the lid back and the stay down onto its edge, together
        return lid_at(hold + (let_go - hold) * s)

    def swing(a0, a1):
        return lambda s: stay_hand(a0 + (a1 - a0) * s)

    out = [
        phase(2.0, "clear the box ends before approaching the lid and stay",
              Move(ho, (-side * 0.70, -0.21, Z + 0.45, *lid_at(0.0)[3:]), joints=True),
              Move(st, lambda s: rotate_about(home_pose(inst, st), sc.arm(st).pos,
                                              axis_quat((0., 0., 1.), side * 90. * s)), allow_flip=False)),
        phase(2.5, "one end arm reaches for the lid's end, the other for its stay",
              Release(st),  # establish an empty hand before choosing its grip approach
              Move(ho, back_off(lid_at(0.0), 0.06)), Open(ho, 0.04),
              Move(st, back_off(stay_hand(-STAY_REST), 0.06), joints=True), Open(st, 0.04)),
        phase(1.2, "", Move(ho, lid_at(0.0)), Move(st, stay_hand(-STAY_REST))),
        phase(0.4, "they take them", Grip(ho, "lid", LID_T - 0.002), Grip(st, stay.name, STAY_BAR - 0.002)),
        phase(4.0, "the lid opens past upright", Move(ho, lambda s: lid_at(hold * s))),
        phase(3.0, "the stay swings over its edge", Move(st, swing(-STAY_REST, over))),
        phase(3.0, "the lid comes back as the hook comes down onto its edge", Move(ho, together),
              Move(st, swing(over, eq))),
        phase(0.4, "the stay is let go", Release(st), Open(st, 0.05)),
        phase(1.0, "", Move(st, back_off(stay_hand(eq), 0.08))),
        phase(0.4, "then the lid: the stay holds it open", Release(ho), Open(ho, 0.04)),
        phase(1.5, "", Move(ho, back_off(lid_at(let_go), 0.08))),
        # north round the lid's ends before coming over the box (swung there in joint space, the holder runs into the
        # lid and knocks it open off the stay's hook)
        phase(2.0, "", Move(ho, (-side * AROUND[0], AROUND[1], Z + AROUND[2], *back_off(lid_at(let_go), 0.08)[3:])),
              Move(st, (side * AROUND[0], AROUND[1], Z + AROUND[2], *back_off(stay_hand(eq), 0.08)[3:]))),
        *_lift_phases(inst),
        phase(3.0, "done"),
    ]

    return out


def script(inst):
    """The scripted reference.  The lid's tab is gripped (privileged: fixed to the hand); the ball is not -- it is
    22 cm across, the fingers open 8 -- but squeezed between the two end hands' closed fingertips (``Grip`` with
    ``attach=False``: they only press), then taken by the west hand, which the east one on its far side lets it do
    (pressed by position alone, the ball slipped out sideways)."""
    if inst.variant == "borrow":
        return _borrow_script(inst)
    sc = inst.scene
    h = sc.hinges[0]
    tab = sc.region("lid_tab").center
    g = math.radians(TAB_GRIP_DEG)
    down = tool_quat((0.0, math.cos(g), -math.sin(g)), (1.0, 0.0, 0.0))  # fingers closing across the tab (x)
    take = (tab[0], tab[1], tab[2] + TAB[2] / 4, *down)  # its upper half: the fingers clear the lid

    def arc(a0, a1):
        return lambda s: rotate_about(take, h.pivot, axis_quat(h.axis, h.sign * (a0 + (a1 - a0) * s)))

    w, e, s = "west", "east", "south"
    # the end arms' home hands are where the lid's free edge swings up past: they draw back north first
    north = {a: offset(home_pose(inst, a), dy=0.20, dz=0.05) for a in (w, e)}
    return [
        phase(2.5, "the middle arm reaches for the lid's tab; the end arms draw back", Move(s, back_off(take, 0.06)),
              Open(s, 0.05), Move(w, north[w]), Move(e, north[e])),
        phase(1.2, "", Move(s, take)),
        phase(0.4, "takes it", Grip(s, "lid", TAB[0] - 0.004)),
        phase(4.0, "opens the lid against its spring and holds it", Move(s, arc(0.0, OPEN_DEG))),
        *_lift_phases(inst),
        phase(4.0, "the middle arm lets the lid down", Move(s, arc(OPEN_DEG, 2.0))),
        phase(0.4, "", Release(s)),
        phase(1.5, "", Move(s, back_off(take, 0.06))),
        phase(3.0, "done"),
    ]


def _base_jobs(inst):
    """What the task asks (``tasks/jobs.py``): the lid held open by its tab the whole time the ball comes out, and the
    ball, wider than a hand opens, lifted between two hands on its west and east sides and set in the cradle -- three
    jobs at once."""
    from ..jobs import Job, Jobs

    sc = inst.scene
    tab = sc.region("lid_tab").center
    h = sc.hinges[0]
    opened = rotate_about((*tab, 1.0, 0.0, 0.0, 0.0), h.pivot, axis_quat(h.axis, h.sign * OPEN_DEG))[:3]
    bx, by, bz = sc.obj("ball").pos
    cx, cy, cz = sc.region("cradle_in").center
    grip = 0.02

    def side(s):  # where the ball's side is touched, in the box, lifted out of it, and in the cradle
        return ((bx + s * BALL_R, by, bz + CONTACT_DZ), (bx + s * BALL_R, by, bz + CONTACT_DZ + LIFT),
                (cx + s * BALL_R, cy, cz + CONTACT_DZ))

    return Jobs((Job("hold the lid open by its tab", sites=(tab, opened), group="ball out", slack=grip),
                 Job("lift the ball by its west side into the cradle", sites=side(-1), group="ball out", slack=grip),
                 Job("lift the ball by its east side into the cradle", sites=side(1), group="ball out", slack=grip)))


def jobs(inst):
    """The jobs (``_base_jobs``); in the borrow configuration the stay holds the lid open once two hands have set it."""
    decl = _base_jobs(inst)
    if inst.variant != "borrow":
        return decl
    from ..jobs import Job, Jobs, Thing

    sc = inst.scene
    b = inst.metadata["borrow"]
    prop, decoy = b["props"][0], b["decoys"][0]
    stay = sc.obj(prop)
    holder = "west" if stay.pos[0] > 0 else "east"
    letter = holder[0]
    initial = sc.region(f"{prop}_grip").center
    seated = rotate_about((*initial, 1., 0., 0., 0.), stay.pos,
                          axis_quat((1., 0., 0.), -(stay_lean() + STAY_REST)))[:3]
    placing = (
        Job("open the lid by its opposite end for the stay", group="fit stay", slack=.02,
            sites=(f"lid_end_{letter}_shut", f"lid_end_{letter}_open")),
        Job("swing the hooked stay onto the lid edge", group="fit stay", slack=.02,
            sites=(f"{prop}_grip", seated)),
    )
    things = (
        Thing(prop, does=(decl.jobs[0].name,), placing=placing,
              why="the shorter hooked stay catches the free edge and holds the spring lid open"),
        Thing(decoy, why=(f"its hook is {DECOY_DL * 100:.0f} cm "
                          + ("further from" if decoy_dl(inst.level, inst.seed) > 0 else "nearer to")
                          + " the pivot and cannot catch the free edge at any opening")),
    )
    return Jobs(decl.jobs, things)


def decoy_stop(inst):
    """A distractor run's safe stop (``mrbench demo --decoy``): the lid, once opened,
    back within 10 degrees of shut -- the distractor stay has not caught it."""
    from ...scoring.predicates import Opened

    return ~Opened("lid", 10.0)
