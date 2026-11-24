"""Rea-02 tomato flume: three wooden channels, laid end to end on stands from a shelf at the west end of the table to
a pot at the east end, make a track; a cherry tomato pushed off the shelf rolls down it into the pot.

Each channel starts where only one arm reaches it -- the first and the last north of the west and east arms, the
middle (short) one behind the south arm, out of both end arms' reach -- and only the west arm reaches the stop
that holds the tomato.  Without the middle channel the track has a gap, so it takes three arms.  Laid on their stands, the channels stay put by themselves: no
arm holds anything while the tomato rolls.  Each channel is wider than the one before it, so a channel's lower
end lies inside the next one's upper end and the tomato drops only a floor's thickness at a joint; the track is
laid from the pot up."""

from __future__ import annotations

import math
from dataclasses import replace

from ...engines.base import RegionSpec, axis_quat, quat_mul, quat_rotate, transform, yaw_quat
from ...policies.script import Grip, Move, Open, Release, back_off, home_pose, offset, phase, tool_quat
from ...scoring.predicates import (Always, EndIn, Eventually, HoldsFor, InRegion, Lifted, SoonAfter, all_temporal,
                                   any_of)
from ..layouts import TABLE_SIZE, mount
from ..loader import get
from ._common import Z, at, box, compound, jitter, make_instance, part, rng_for, scene, sphere

#: the track's line on the table (south of the middle, in front of the south arm) and its fall
Y, SLOPE_DEG = -0.30, 4.0
#: a channel: length, floor and wall thickness, how high its walls stand above its floor, and the inside width of
#: the upper one.  Each one down is wider: its inside leaves the one above twice the room a stand's slot gives a
#: channel (the knob), and 2 mm, each side -- however far the slots let two channels stand off the line, the upper
#: one's lower end still falls inside.  (With 15 mm in the slots and 6 mm inside, the middle channel can lie 17 mm off
#: the line, and the upper one's end come down on its wall.)
#: The middle channel is the short one: it starts behind the south arm (STORES), where a longer one would reach
#: into the east arm's reach; the other two make up the length (as long as three of 0.72 m).
LENGTHS, FLOOR_T, WALL_T, WALL_H = (0.80, 0.55, 0.80), 0.006, 0.005, 0.025
INNER_UPPER, NEST_EXTRA = 0.055, 0.002
#: how far a channel's lower end reaches into the next one (along the track), and how far above the next one's
#: floor it is laid (it settles onto it)
OVERLAP, SETTLE = 0.05, 0.001
#: the shelf: centre and footprint, and how high its top is above the table.  On it lies a tray falling toward the
#: track, its east edge just over the shelf's top, where the first channel's upper end butts, that channel's floor
#: DROP below the shelf's top.  The tomato (4 cm across) leans on a stop standing STOP_BACK in from the tray's east
#: edge; lifted away, the tomato rolls down the tray and off into the channel.  (To push it off a level top a Piper,
#: reaching that far, points its hand only away from its base, and coming down behind the tomato its hand sweeps over
#: it.  With the stop at the edge, the tomato falls off it almost straight down, and a channel laid 2 cm short of the
#: shelf leaves it sitting in the gap.)
SHELF_XY, SHELF_W, SHELF_TOP = (-1.07, Y), 0.12, 0.25
TRAY_T, TRAY_TILT_DEG, TRAY_UP, STOP_BACK = 0.01, 4.0, 0.004, 0.05
TOMATO_R, DROP = 0.02, 0.03
#: the stop: thickness (along the track), width and height -- twice the tomato's height, so a hand takes it well
#: above it (taken 2 cm over the tomato, a finger comes within a centimetre of it)
STOP = (0.015, 0.025, 0.08)
#: a stand's top: a plate this long under a channel, with a lip each side of it this thick and high
PLATE_L, PLATE_T, LIP_T, LIP_H = 0.05, 0.012, 0.006, 0.012
#: the pot: radius (inside), wall thickness, floor thickness, and how far the last channel's end reaches in over
#: its rim
POT_R, POT_T, POT_FLOOR, POT_IN = 0.10, 0.006, 0.008, 0.04
#: each channel's handle: a bar along it, out from its south wall when laid, 2 cm over the walls' top (taken lower,
#: off the table, a Piper's fingers come within a centimetre of it): where along the channel its middle is (body x), its
#: length, its bar's thickness, how far out from the wall it stands and how far its bottom is above the walls' top.
#: The middle channel's is toward its lower end, 12 cm short of where the lowest one's wall comes over it
HANDLE_X = (0.0, 0.10, 0.0)
HANDLE_L, HANDLE_BAR, HANDLE_OUT, HANDLE_UP = 0.10, 0.012, 0.025, 0.02
#: the borrow version's channels (the long one and the short one) hold their handles this far over their walls: the
#: west hand takes the long one by its handle on its far side, its body over the channel -- with the others' 2 cm, its
#: body comes down 1.4 cm into the wall under the handle and pushes the channel away before the grip; at 6 cm it
#: clears that wall by 2.6 cm
LONG_HANDLE_UP = 0.06
#: and this far from their middles toward their lower ends (along them), so that each lies that much nearer the table's
#: middle than its handle's place (SPARES): with the handle at its middle the long one, lying with its handle at
#: (-0.95, 0.15), reaches the table's west edge, and lifted, swings its end down past the edge, where it catches on
#: the side of a table built down to the floor (the workshop's; on a table top on legs it passes under the top).
#: Moving the place instead, the hand finds no straight way in to the handle (-0.92 and -0.90, plan checks)
SPARE_HANDLE_X = 0.05
#: where each channel starts, lying on the table: its middle (x, y) and turn.  The first and last are turned half
#: round, north of their arms: an arm that swings one round to the track turns it the right way round.  The middle
#: one lies behind the south arm, east of its base, its handle toward it: there no part of it is within 0.75 m of
#: the east arm (a PiPER's kinematic reach near the table), which on the table's middle line would reach its handle --
#: and then the two end arms could lay the whole track between them.  At (0.42, -0.82) its north-east corner would be
#: 0.78 m from the east arm's base, within the audit's 5 cm (`validate --ik --slack 0.05`); 3 cm further south it is
#: 0.81 m
STORES = (((-0.70, 0.30), 180.0), ((0.40, -0.85), 180.0), ((0.70, 0.30), 180.0))
#: a tomato's middle this low over the table, outside the pot, has fallen off the track
LOW = 0.045
#: how far (x, y, z) the upper end of the borrow version's long channel may be from where it is laid and still count;
#: its lower end counts anywhere across inside the lowest channel (where that lies), as far along and as high as this
END_TOL = (0.03, 0.02, 0.02)
#: the borrow version (no south arm): two more channels, as wide inside as the upper one, lie north of the west arm,
#: their handles at their middles: a long one, the upper and the middle channel in one, that the west arm lays from the
#: shelf into the lowest channel; and a short one that does not reach it from the shelf (nor, laid from the first
#: stand, the lowest channel: 0.50 m lie between them).  A 1.05 m one would not do: laid from the first stand, the west
#: arm holding its upper end (IK, 5 cm to spare), its lower end lies well inside the lowest channel and the upper
#: channel can be laid onto it -- two arms, the distractor used; hence SHORT.  The distractor is always the shorter: a longer channel laid from the shelf only reaches further down inside the
#: lowest one (their floors are parallel, 4 mm apart), and a wider one -- 7.5, 13 and 17.1 cm inside, too wide for the
#: ledge's or the first stand's lips, the widest two too wide to nest in the lowest channel -- rides on the lips and
#: the lowest channel's walls and still takes the tomato into the pot (Isaac).  They lie turned half round like the upper one, in two places, which is which drawn from the seed:
#: the arm holds a channel the same way taking and laying it, and far out a Piper's hand points about away from its
#: base, so the long one lies only where the hand that lays it, pointing about 134 degrees round from east, reaches its
#: handle -- north of the upper channel, or between it and the west arm's base, west of the base.  From (-0.78, 0.17)
#: the long one, lifted and swung over, passes 0.2-0.3 cm into the other one's handle in any north place within
#: reach, and nearer the base the arm cannot lift it higher; from (-0.95, 0.15) the arm swings it round the other way
#: (plan checks, both levels; at (-0.66, 0.17) and (-1.00, 0.16) out of reach, at (-0.76, 0.16), (-0.82, 0.16) and
#: (-0.88, 0.14) no straight approach).  Named channel_4 and channel_5, south to north.
LONG, SHORT = 1.30, 0.45
SPARES = ((-0.95, 0.15), (-0.66, 0.45))
#: the middle channel starts in a cradle fixed to the table (``middle_cradle``): a lip along its north side and one
#: across each open end, CRADLE_GAP off it, CRADLE_T thick and CRADLE_H high -- higher than its walls (3.1 cm over
#: the table), so that pulled north or along it stops against them: it comes out only lifted.  Under its handle's
#: brackets (3.1 cm over the table) the north lip is CRADLE_LOW high, and it leaves the last CRADLE_OPEN of the north
#: wall by the east end free: a Franka on the east place takes the channel there (the swapped configuration).  12 mm
#: off, as the south hand, lifting the channel out, runs up to 6 mm off its plan (GPU).  Without the cradle, the east
#: Piper can take the lowest channel by its handle, put its far end down inside the middle one and draw it north in a
#: few strokes, then take it and, with the west Piper, lay the whole track: two arms (Isaac).  With it, no pose of
#: that rake the east arm reaches puts its far end in past the lips (IK), and forced, its strokes do not draw the
#: middle channel north.  Rea-02 built as itself has it (``own_scene``).
CRADLE_GAP, CRADLE_T, CRADLE_H, CRADLE_LOW, CRADLE_OPEN = 0.012, 0.008, 0.045, 0.027, 0.065


def _tilt() -> tuple:
    return axis_quat((0.0, 1.0, 0.0), SLOPE_DEG)


def track() -> list[tuple[float, float]]:
    """Each channel's upper end, laid: (x, height of its floor's top above the table), upper to lower."""
    th = math.radians(SLOPE_DEG)
    ends = [(SHELF_XY[0] + SHELF_W / 2, SHELF_TOP - DROP)]
    for k in range(2):
        x, z = ends[-1]
        xl, zl = x + LENGTHS[k] * math.cos(th), z - LENGTHS[k] * math.sin(th)
        # the next one's floor, OVERLAP back from this one's lower end, just under this one's floor there
        ends.append((xl - OVERLAP, zl - FLOOR_T / math.cos(th) + OVERLAP * math.tan(th) + SETTLE))
    return ends


def laid(k: int):
    """Channel ``k`` laid in the track: its frame's origin is the middle of its floor's top."""
    return laid_at(track()[k], LENGTHS[k])


def laid_at(upper, length: float):
    """A channel ``length`` long laid down the track from ``upper`` (x, the height of its floor's top over the table)."""
    th = math.radians(SLOPE_DEG)
    x, z = upper
    return (x + length * math.cos(th) / 2, Y, Z + z - length * math.sin(th) / 2, *_tilt())


def floor_bottom(k: int, x: float) -> float:
    """The height of channel ``k``'s underside, laid, at ``x``."""
    th = math.radians(SLOPE_DEG)
    x0, z0 = track()[k]
    return Z + z0 - (x - x0) * math.tan(th) - FLOOR_T / math.cos(th)


def inner(k: int, room: float) -> float:
    """The inside width of channel ``k`` when the stands' slots give a channel ``room`` each side."""
    return INNER_UPPER + k * 2 * (WALL_T + 2 * room + NEST_EXTRA)


def handle(k: int, room: float, up: float = HANDLE_UP, width=None) -> tuple[float, float, float]:
    """Where the middle of channel ``k``'s handle bar is, in its frame (``up`` over its wall); ``width``: its inside
    width, if not channel ``k``'s."""
    w = inner(k, room) if width is None else width
    return (HANDLE_X[k], -(w / 2 + WALL_T + HANDLE_OUT + HANDLE_BAR / 2), WALL_H + up + HANDLE_BAR / 2)


def channel(k: int, pose, room: float, name=None, length=None, width=None):
    """Channel ``k`` (its width, length and handle); or, with ``name`` and ``length``, one as wide as it (or
    ``width`` inside), that long, its handle SPARE_HANDLE_X from its middle (the borrow version's)."""
    up = HANDLE_UP if length is None else LONG_HANDLE_UP
    w, hx, hy, hz = (inner(k, room) if width is None else width), *handle(k, room, up, width)
    if length is not None:
        hx = SPARE_HANDLE_X
    wy = w / 2 + WALL_T / 2
    n = LENGTHS[k] if length is None else length
    parts = [part("floor", "box", (n, w + 2 * WALL_T, FLOOR_T), (0.0, 0.0, -FLOOR_T / 2)),
             part("wall_n", "box", (n, WALL_T, WALL_H + FLOOR_T), (0.0, wy, (WALL_H - FLOOR_T) / 2)),
             part("wall_s", "box", (n, WALL_T, WALL_H + FLOOR_T), (0.0, -wy, (WALL_H - FLOOR_T) / 2)),
             part("handle", "box", (HANDLE_L, HANDLE_BAR, HANDLE_BAR), (hx, hy, hz), color="grey")]
    # two brackets, sitting on the wall's top, hold the bar out from it
    parts += [part(f"bracket_{i}", "box", (HANDLE_BAR, WALL_T + HANDLE_OUT, up + HANDLE_BAR),
                   (hx + dx, -(w / 2 + (WALL_T + HANDLE_OUT) / 2), WALL_H + (up + HANDLE_BAR) / 2),
                   color="grey")
              for i, dx in enumerate((-(HANDLE_L - HANDLE_BAR) / 2, (HANDLE_L - HANDLE_BAR) / 2))]
    # the borrow version's spare channels (``length``) are moulded HDPE chutes, as light as the wooden ones
    return compound(name or f"channel_{k + 1}", pose[:3], parts, "wood" if length is None else "pvc", mass=0.15,
                    quat=tuple(pose[3:]))


def _in_lowest(long_laid, room: float):
    """Where the long channel's lower end lies, laid, in the lowest channel's frame, and how far it may be from there
    (x, y, z): along and up END_TOL, across as far as its walls stay inside the lowest one's (and 2 mm)."""
    from ...engines.base import quat_rotate as rot

    end, low = transform(long_laid, (LONG / 2, 0.0, 0.0)), laid(2)
    at = rot((low[3], -low[4], -low[5], -low[6]), tuple(end[i] - low[i] for i in range(3)))
    across = inner(2, room) / 2 - (inner(0, room) / 2 + WALL_T) + 0.002
    return tuple(at), (END_TOL[0], across, END_TOL[2])


def dims(obj) -> tuple[float, float]:
    """A channel's length and inside width, read off its floor."""
    floor = next(p for p in obj.parts if p.name == "floor")
    return floor.size[0], floor.size[1] - 2 * WALL_T


def _plate(name: str, k: int, x0: float, room: float, origin) -> list:
    """A stand's top under channel ``k`` from ``x0`` east, at the height the channel's underside comes down to at
    the plate's east edge (its lowest point over the plate), with a lip each side; in the frame at ``origin``."""
    out = inner(k, room) + 2 * WALL_T
    ox, oy, oz = origin
    top = floor_bottom(k, x0 + PLATE_L) - SETTLE
    cx = x0 + PLATE_L / 2 - ox
    parts = [part(f"{name}_plate", "box", (PLATE_L, out + 2 * (room + LIP_T), PLATE_T),
                  (cx, Y - oy, top - PLATE_T / 2 - oz))]
    parts += [part(f"{name}_lip_{i}", "box", (PLATE_L, LIP_T, LIP_H),
                   (cx, Y - oy + s * (out / 2 + room + LIP_T / 2), top + LIP_H / 2 - oz))
              for i, s in enumerate((-1, 1))]
    return parts


def _shelf(room: float):
    sx, sy = SHELF_XY
    parts = [part("tower", "box", (SHELF_W, SHELF_W, SHELF_TOP), (0.0, 0.0, SHELF_TOP / 2))]
    parts += _plate("ledge", 0, sx + SHELF_W / 2, room, (sx, sy, Z))
    return compound("shelf", (sx, sy, Z), parts, "wood", dynamic=False)


def _stand(k: int, room: float):
    """The stand under channel ``k``'s upper end, just in from it (``k`` 1 or 2)."""
    x0 = track()[k][0] + 0.005
    top = floor_bottom(k, x0 + PLATE_L) - SETTLE - PLATE_T
    sx = x0 + PLATE_L / 2
    parts = [part("post", "box", (0.03, 0.03, top - Z), (0.0, 0.0, (top - Z) / 2))]
    parts += _plate("top", k, x0, room, (sx, Y, Z))
    return compound(f"stand_{k}", (sx, Y, Z), parts, "wood", dynamic=False)


def _cradle(start, room: float):
    """The cradle round the middle channel lying at ``start``: its north lip (low under the handle's brackets, short of
    the east end) and a lip across each open end, as far south as the table's edge.  Its frame is the channel's,
    dropped to the table."""
    hw = inner(1, room) / 2 + WALL_T  # the channel's half width, outside
    xe = LENGTHS[1] / 2 + CRADLE_GAP + CRADLE_T / 2  # each end lip's middle
    north = -(hw + CRADLE_GAP + CRADLE_T)  # the north lip's outer face (the handle is on the channel's -y side)
    bx = [HANDLE_X[1] + s * (HANDLE_L - HANDLE_BAR) / 2 for s in (-1, 1)]
    lo, hi = bx[0] - HANDLE_BAR / 2 - CRADLE_GAP, bx[1] + HANDLE_BAR / 2 + CRADLE_GAP  # under the brackets
    x0, x1 = -xe - CRADLE_T / 2, xe + CRADLE_T / 2
    south = min(hw + CRADLE_GAP, TABLE_SIZE[1] / 2 + start[1] - 0.001)  # the table's edge, in the channel's frame

    def lip(name, a, b, y0, y1, h):
        return part(name, "box", (b - a, y1 - y0, h), ((a + b) / 2, (y0 + y1) / 2, h / 2))

    xo = -LENGTHS[1] / 2 + CRADLE_OPEN  # the north lip starts here, CRADLE_OPEN in from the east end (-x as it lies)
    parts = [lip("north_e", xo, lo, north, north + CRADLE_T, CRADLE_H),
             lip("north_low", lo, hi, north, north + CRADLE_T, CRADLE_LOW),
             lip("north_w", hi, x1, north, north + CRADLE_T, CRADLE_H),
             lip("end_e", x0, x0 + CRADLE_T, north, south, CRADLE_H),
             lip("end_w", x1 - CRADLE_T, x1, north, south, CRADLE_H)]
    return compound("middle_cradle", (start[0], start[1], Z), parts, "wood", dynamic=False, quat=tuple(start[3:]))


def own_scene(inst):
    """Rea-02 built as itself (``tasks.build``): the middle channel lies in its cradle."""
    start = inst.scene.obj("channel_2").pose
    sc = replace(inst.scene, objects=inst.scene.objects + (_cradle(start, inst.metadata["fit_room_m"]),))
    return replace(inst, scene=sc)


def pot_xy() -> tuple[float, float]:
    th = math.radians(SLOPE_DEG)
    x, _ = track()[2]
    return (x + LENGTHS[2] * math.cos(th) - POT_IN + POT_R, Y)


def _pot():
    px, py = pot_xy()
    rim = floor_bottom(2, px - POT_R) - SETTLE - Z  # the last channel rests on the rim's inner edge, west
    n, r = 16, POT_R + POT_T / 2
    chord = 2 * (POT_R + POT_T) * math.sin(math.pi / n) * 1.05
    parts = [part("bottom", "cylinder", (POT_R + POT_T, POT_FLOOR), (0.0, 0.0, POT_FLOOR / 2))]
    for i in range(n):
        a = 2 * math.pi * i / n
        parts.append(part(f"wall_{i}", "box", (chord, POT_T, rim), (r * math.cos(a), r * math.sin(a), rim / 2),
                          math.degrees(a) + 90.0))
    return compound("pot", (px, py, Z), parts, "grey", dynamic=False), rim


def tray_top(x: float) -> float:
    """The height of the tray's top at ``x`` (it falls toward the east)."""
    east = SHELF_XY[0] + SHELF_W / 2
    return Z + SHELF_TOP + TRAY_UP + (east - x) * math.tan(math.radians(TRAY_TILT_DEG))


def _tray_stop_tomato():
    """The tray on the shelf, the stop at its east edge and the tomato leaning on the stop, all tilted with it."""
    tilt = axis_quat((0.0, 1.0, 0.0), TRAY_TILT_DEG)
    th = math.radians(TRAY_TILT_DEG)
    sx = SHELF_XY[0]
    tray = box("tray", SHELF_XY, (SHELF_W, SHELF_W, TRAY_T), "wood", z=tray_top(sx) - TRAY_T / (2 * math.cos(th)),
               dynamic=False)
    tray = replace(tray, quat=tilt)
    xs = sx + SHELF_W / 2 - STOP_BACK
    stop = replace(box("stop", (xs, Y), STOP, "wood", mass=0.02, z=tray_top(xs) + STOP[2] / 2), quat=tilt)
    xt = xs - STOP[0] / 2 - TOMATO_R
    tomato = sphere("tomato", (xt, Y), TOMATO_R, "red", mass=0.012, z=tray_top(xt) + TOMATO_R)
    return tray, stop, tomato


def store_pose(k: int, xy=None):
    (x, y), yaw = STORES[k]
    x, y = xy or (x, y)
    return (x, y, Z + FLOOR_T, *yaw_quat(yaw))


def build(level: str, seed: int, variant: str = "exact"):
    spec = get("Rea-02")
    rng = rng_for(spec, level, seed)
    room = spec.knob.levels[level] / 1000
    starts = [jitter(rng, STORES[k][0], 0.008) for k in range(3)]  # where each channel lies, give or take 8 mm
    channels = [channel(k, store_pose(k, starts[k]), room) for k in range(3)]
    pot, rim = _pot()
    tray, stop, tomato = _tray_stop_tomato()
    px, py = pot_xy()
    regions = [RegionSpec("shelf_top", (*SHELF_XY, Z + SHELF_TOP + TOMATO_R), (SHELF_W / 2, SHELF_W / 2, 0.02),
                          visible=False),
               # all of the inside: a tomato that rolls in comes to rest against the far wall
               RegionSpec("pot_in", (px, py, Z + POT_FLOOR + 0.04), (POT_R - TOMATO_R + 0.005,) * 2 + (0.04,),
                          visible=False),
               # over the pot, its rim included, up to the rim's top: a tomato there is going into the pot, not lying
               # on the table (a tomato dropping from the last channel's end over the rim can be below the lowest
               # channel and not yet in ``pot_in`` for a frame, and the success's ``Always`` would count it off the
               # track)
               RegionSpec("pot_over", (px, py, Z + (rim + TOMATO_R) / 2),
                          (POT_R + POT_T + TOMATO_R,) * 2 + ((rim + TOMATO_R) / 2 + 0.005,), visible=False)]
    for k in range(3):  # where a channel is laid (its frame's origin), and where its handle is, there and at the start
        regions.append(RegionSpec(f"laid_{k + 1}", laid(k)[:3], (0.02, 0.015, 0.015), visible=False))
        regions.append(RegionSpec(f"grip_{k + 1}", transform(store_pose(k, starts[k]), handle(k, room)), (0.01,) * 3,
                                  visible=False))
        regions.append(RegionSpec(f"hold_{k + 1}", transform(laid(k), handle(k, room)), (0.01,) * 3, visible=False))
    # the corners of the middle channel's walls where it lies, for the claims that the end arms reach none of it
    half = (LENGTHS[1] / 2, inner(1, room) / 2 + WALL_T)
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        regions.append(RegionSpec(f"store_2_{i}", transform(store_pose(1, starts[1]), (sx * half[0], sy * half[1], WALL_H)),
                                  (0.01,) * 3, visible=False))
    reach = [("west", "stop"), ("west", "grip_1"), ("west", "hold_1"), ("south", "grip_2"),
             ("south", "hold_2"), ("east", "grip_3"), ("east", "hold_3")]
    # the middle channel, where it starts, is the south arm's alone: the end arms reach none of it
    out_of_reach = [("south", "grip_1"), ("east", "grip_1"), ("west", "grip_2"), ("east", "grip_2"),
                    ("west", "grip_3"), ("south", "grip_3"), ("south", "stop"), ("east", "stop"),
                    *[(a, f"store_2_{i}") for a in ("west", "east") for i in range(4)]]
    laid_any = {"middle": InRegion("channel_2", "laid_2"), "upper": InRegion("channel_1", "laid_1")}
    extra = {}
    if variant == "borrow":  # no south arm: a long channel does the upper and the middle one's work, a short one cannot
        lengths = [SHORT, LONG]  # south to north (SPARES), or the other way round
        if rng.random() < 0.5:
            lengths.reverse()
        names = ("channel_4", "channel_5")
        prop = names[lengths.index(LONG)]
        for name, xy, length in zip(names, SPARES, lengths):
            # the handle where SPARES says, the channel SPARE_HANDLE_X from it along its length
            pose = store_pose(0, xy)
            off = quat_rotate(pose[3:], (SPARE_HANDLE_X, 0.0, 0.0))
            pose = (pose[0] - off[0], pose[1] - off[1], *pose[2:])
            channels.append(channel(0, pose, room, name=name, length=length))
            regions.append(RegionSpec(f"grip_{name[-1]}", transform(pose, (SPARE_HANDLE_X,
                                                                             *handle(0, room, LONG_HANDLE_UP)[1:])),
                                      (0.01,) * 3, visible=False))
        long_laid = laid_at(track()[0], LONG)
        # where the long channel's upper end lies, laid: on the shelf's ledge (END_TOL each way)
        regions.append(RegionSpec("long_top", transform(long_laid, (-LONG / 2, 0.0, 0.0))[:3], END_TOL, visible=False))
        regions += [RegionSpec("laid_long", long_laid[:3], (0.02, 0.015, 0.015), visible=False),
                    RegionSpec("hold_long", transform(long_laid, (SPARE_HANDLE_X, *handle(0, room, LONG_HANDLE_UP)[1:])), (0.01,) * 3,
                               visible=False)]
        # laid, the long channel does the middle and the upper channel's work: either spare channel with both its ends
        # in place (by its own length: a short one let go where the long one's middle goes is not laid)
        low = _in_lowest(long_laid, room)
        long_down = any_of(*(EndIn(n, -1.0, region="long_top") & EndIn(n, 1.0, body="channel_3", at=low[0], half=low[1])
                             for n in names))
        laid_any = {k: f | long_down for k, f in laid_any.items()}
        reach = [r for r in reach if r[0] != "south"] + [("west", "grip_4"), ("west", "grip_5"), ("west", "hold_long")]
        out_of_reach = [r for r in out_of_reach if r[0] != "south"]
        extra = {"borrow": {"props": [prop], "decoys": [n for n in names if n != prop]}}
    sc = scene(spec, channels + [_shelf(room), tray, _stand(1, room), _stand(2, room), pot, stop,
                                 tomato], regions)
    if variant == "borrow":
        sc = replace(sc, arms=tuple(a for a in sc.arms if a.slot != "south"))

    in_pot = InRegion("tomato", "pot_in")
    return make_instance(
        spec, level, seed, sc,
        # the goal only: how the track is made (three channels, or one long one where a version has it) is not said
        instruction=("Roll the cherry tomato from the top of the shelf at the west end of the table into the pot at "
                     "the east end, along a track laid on the stands. Do not carry it."),
        success=all_temporal(SoonAfter(InRegion("tomato", "shelf_top"), in_pot, 5.0), HoldsFor(in_pot, 1.0),
                             Always(in_pot | InRegion("tomato", "pot_over")
                                    | Lifted("tomato", LOW - SHELF_TOP - TOMATO_R))),
        milestones={
            "lower_laid": Eventually(InRegion("channel_3", "laid_3")),
            "middle_laid": Eventually(laid_any["middle"]),
            "upper_laid": Eventually(laid_any["upper"]),
            "tomato_off_shelf": Eventually(~InRegion("tomato", "shelf_top")),
            "tomato_in_pot": Eventually(in_pot),
        },
        reach=reach, out_of_reach=out_of_reach, fit_room_m=room, pot_rim_m=round(rim, 3), **extra,
    )


#: the keyframe `mrbench render` shows between start and goal: the track laid, the tomato in the first channel
MIDDLE = 3


def _on(k: int, s: float):
    """A tomato resting on channel ``k``'s floor, ``s`` of the way down it, laid."""
    return at(transform(laid(k), ((s - 0.5) * LENGTHS[k], 0.0, TOMATO_R)))


def keyframes(inst):
    px, py = pot_xy()
    if inst.variant == "borrow":  # the long channel laid from the shelf into the lowest one
        long_laid = laid_at(track()[0], LONG)
        return [
            {"channel_3": laid(2)},
            {inst.metadata["borrow"]["props"][0]: long_laid},
            {"tomato": at(transform(long_laid, (-0.2 * LONG, 0.0, TOMATO_R)))},
            {"tomato": _on(2, 0.5)},
            {"tomato": at((px, py, Z + POT_FLOOR + TOMATO_R))},
        ]
    return [
        {"channel_3": laid(2)},
        {"channel_2": laid(1)},
        {"channel_1": laid(0)},
        {"tomato": _on(0, 0.3)},
        {"tomato": _on(2, 0.5)},
        {"tomato": at((px, py, Z + POT_FLOOR + TOMATO_R))},
    ]


#: the scripted reference: how far below level a hand points (degrees), by robot -- a Piper pointing straight down
#: reaches no higher than about 10 cm above its base, and at 45 degrees no higher than 30 cm, so it comes at a
#: handle 30 degrees down, away from its base -- how far apart the fingers close on a handle and are as they come
#: in to it, how high a channel is
#: carried over where it goes, and how far above its stands it is let go.  The upper channel waits lifted where it
#: started until the middle one is down: over its place it is where the middle one comes down, and 10 cm west of
#: there out of the west Piper's reach either way round
PITCH = {"piper": 30.0}
#: how high the south arm lifts the middle channel where it lay before bringing it round over the track
HIGH = 0.25
BAR_GRIP, HANDLE_SHAPE, STOP_GRIP, STOP_SHAPE, CARRY, LET_GO = 0.014, 0.03, 0.02, 0.045, 0.10, 0.003
#: the hands are still this long before they close on the channels' handles, and the south hand over the middle
#: channel's place before it lets it down the last 10 cm (s): the south hand, reaching down behind its base, can still
#: be 6 mm off its plan as it closes (GPU), and the middle channel it then holds comes off its line, its upper end on
#: the first stand's lip
PAUSE = 0.8


def _grasp(inst, slot: str, k: int, name=None, pose=None):
    """The tool pose that holds channel ``k`` by its handle, laid and at its start; or channel ``name``, as wide as
    channel ``k``, its handle SPARE_HANDLE_X from its middle, laid at ``pose`` (the borrow version's long channel)."""
    arm = inst.scene.arm(slot)
    pose = laid(k) if pose is None else pose
    room = inst.metadata["fit_room_m"]
    hpos = handle(k, room) if name is None else \
        (SPARE_HANDLE_X, *handle(k, room, LONG_HANDLE_UP, dims(inst.scene.obj(name))[1])[1:])
    h = transform(pose, hpos)
    p = math.radians(PITCH.get(arm.model, 90.0))
    rx, ry = h[0] - arm.pos[0], h[1] - arm.pos[1]
    n = math.hypot(rx, ry)
    approach = (rx / n * math.cos(p), ry / n * math.cos(p), -math.sin(p))
    bar = quat_rotate(pose[3:], (1.0, 0.0, 0.0))
    across = (approach[1] * bar[2] - approach[2] * bar[1], approach[2] * bar[0] - approach[0] * bar[2],
              approach[0] * bar[1] - approach[1] * bar[0])
    tool = tool_quat(approach, across)
    rel = quat_mul((pose[3], -pose[4], -pose[5], -pose[6]), tool)  # the hand in the channel's frame
    start = inst.scene.obj(name or f"channel_{k + 1}").pose
    return (*h, *tool), (*transform(start, hpos), *quat_mul(start[3:], rel))


def script(inst):
    """Privileged: a grip fixes the channel to the hand.  All three arms take their channels at once; the east
    arm lays the lowest, then the south arm the middle one into it, then the west arm the upper one; the west arm
    then lifts the stop away, and the tomato rolls into the pot on its own."""
    w, s, e = "west", "south", "east"
    if inst.variant == "borrow":
        return _borrow_script(inst)
    (lay_w, get_w), (lay_s, get_s), (lay_e, get_e) = (_grasp(inst, w, 0), _grasp(inst, s, 1), _grasp(inst, e, 2))
    # the stop: taken from above, the hand pointing down and away from its base, and lifted straight up
    stop = inst.scene.obj("stop").pos
    pw = math.radians(PITCH.get(inst.scene.arm(w).model, 90.0))
    rx, ry = stop[0] - mount(w).x, stop[1] - mount(w).y
    n = math.hypot(rx, ry)
    approach = (rx / n * math.cos(pw), ry / n * math.cos(pw), -math.sin(pw))
    take = (stop[0], stop[1], stop[2] + STOP[2] / 2 - 0.01, *tool_quat(approach, (-approach[1], approach[0], 0.0)))
    return [
        # the Pipers' fingers half shut: tilted, a wide-open hand's lower finger comes within a centimetre of the table
        phase(3.0, "all three arms go to their channels", Move(w, back_off(get_w, 0.08), joints=True),
              Move(s, back_off(get_s, 0.08), joints=True), Move(e, back_off(get_e, 0.08), joints=True),
              Open(w, HANDLE_SHAPE), Open(e, HANDLE_SHAPE)),
        phase(1.5, "", Move(w, get_w), Move(s, get_s), Move(e, get_e)),
        phase(PAUSE, ""),
        phase(0.4, "and take them by their handles", Grip(w, "channel_1", BAR_GRIP), Grip(s, "channel_2", BAR_GRIP),
              Grip(e, "channel_3", BAR_GRIP)),
        # the east arm's swing is the longest, past the second stand: lifted higher, it keeps clear of its lips.  The
        # middle channel waits lifted behind the south arm until the lowest one is down: carried over at the same
        # time, the two meet over the track's east half
        phase(1.5, "lift them", Move(w, offset(get_w, dz=CARRY)), Move(s, offset(get_s, dz=CARRY)),
              Move(e, offset(get_e, dz=1.25 * CARRY))),
        phase(4.0, "the east arm carries the lowest channel over the track",
              Move(e, offset(lay_e, dz=1.75 * CARRY), joints=True)),
        phase(2.0, "and lays it on its stand and the pot's rim", Move(e, offset(lay_e, dz=LET_GO))),
        phase(0.4, "", Release(e)),
        phase(1.5, "", Move(e, offset(back_off(lay_e, 0.10), dz=0.06))),
        phase(2.0, "", Move(e, home_pose(inst, e), joints=True)),
        # lifted high, brought round and let down over its place: swung straight to just over it, it runs into the
        # lowest channel's handle, and swung round from just above where it lay, it passes 7 mm from that handle
        phase(1.5, "the south arm lifts the middle one high", Move(s, offset(get_s, dz=HIGH))),
        phase(4.0, "and brings it round from behind it", Move(s, offset(lay_s, dz=2.6 * CARRY), joints=True)),
        phase(2.0, "and lays it, its lower end inside the lowest", Move(s, offset(lay_s, dz=CARRY))),
        phase(PAUSE, ""),
        phase(1.5, "", Move(s, offset(lay_s, dz=LET_GO))),
        phase(0.4, "", Release(s)),
        phase(3.0, "the west arm swings the upper one over",
              Move(s, offset(back_off(lay_s, 0.10), dz=0.08)), Move(w, offset(lay_w, dz=CARRY), joints=True)),
        phase(0.5, "and lays it from the shelf into the middle one"),
        phase(1.5, "", Move(w, offset(lay_w, dz=LET_GO))),
        phase(0.4, "", Release(w)),
        phase(1.5, "", Move(w, offset(back_off(lay_w, 0.10), dz=0.06)), Move(s, home_pose(inst, s), joints=True)),
        phase(2.5, "the west arm goes to the stop in front of the tomato",
              Move(w, offset(back_off(take, 0.06), dz=0.04), joints=True), Open(w, STOP_SHAPE)),
        phase(1.0, "", Move(w, take)),
        phase(0.4, "", Grip(w, "stop", STOP_GRIP)),
        # up 5 cm (the tomato is 4 cm tall), then back toward the base at that height: a Piper's hand 30 degrees
        # down reaches no higher than 40 cm
        phase(1.0, "and lifts it away: the tomato rolls off the shelf", Move(w, offset(take, dz=0.05))),
        phase(1.0, "down the track", Move(w, offset(take, dx=-rx / n * 0.08, dy=-ry / n * 0.08, dz=0.05))),
        phase(5.0, "into the pot"),
    ]


def _stop_take(inst):
    """The tool pose that takes the stop from above, the hand pointing down and away from the west arm's base."""
    stop = inst.scene.obj("stop").pos
    pw = math.radians(PITCH.get(inst.scene.arm("west").model, 90.0))
    rx, ry = stop[0] - mount("west").x, stop[1] - mount("west").y
    n = math.hypot(rx, ry)
    approach = (rx / n * math.cos(pw), ry / n * math.cos(pw), -math.sin(pw))
    return (stop[0], stop[1], stop[2] + STOP[2] / 2 - 0.01, *tool_quat(approach, (-approach[1], approach[0], 0.0))), \
        (rx / n, ry / n)


#: how far above the long channel's handle the west hand comes before it comes down onto it
APPROACH_UP = 0.05


def _borrow_script(inst):
    """The borrow configuration (no south arm): the east arm lays the lowest channel as in the exact one; the west arm
    takes the long channel by its handle, swings it over and lays it from the shelf into the lowest one, then lifts
    the stop away."""
    w, e = "west", "east"
    prop = inst.metadata["borrow"]["props"][0]  # (with --decoy, the distractor: held as the long one is, it is let go
    # where the long one would be)
    lay_l, get_l = _grasp(inst, w, 0, name=prop, pose=laid_at(track()[0], LONG))
    lay_e, get_e = _grasp(inst, e, 2)
    take, (ux, uy) = _stop_take(inst)
    return [
        # the west hand comes straight down onto the long channel's handle, which is on its far side: coming in along
        # its pointing (4 cm; 8 cm back is too near its own base to reach), its fingers sweep over the near wall and
        # push the channel away before the grip
        phase(3.0, "the end arms go to their channels", Move(w, offset(get_l, dz=APPROACH_UP), joints=True),
              Move(e, back_off(get_e, 0.08), joints=True), Open(w, HANDLE_SHAPE), Open(e, HANDLE_SHAPE)),
        phase(1.5, "", Move(w, get_l), Move(e, get_e)),
        phase(0.4, "and take them by their handles: the west arm the long one", Grip(w, prop, BAR_GRIP),
              Grip(e, "channel_3", BAR_GRIP)),
        phase(1.5, "lift them", Move(w, offset(get_l, dz=CARRY)), Move(e, offset(get_e, dz=1.25 * CARRY))),
        phase(4.0, "the east arm carries the lowest channel over the track",
              Move(e, offset(lay_e, dz=1.75 * CARRY), joints=True)),
        phase(2.0, "and lays it on its stand and the pot's rim", Move(e, offset(lay_e, dz=LET_GO))),
        phase(0.4, "", Release(e)),
        phase(1.5, "", Move(e, offset(back_off(lay_e, 0.10), dz=0.06))),
        phase(2.0, "", Move(e, home_pose(inst, e), joints=True)),
        phase(4.0, "the west arm swings the long channel over", Move(w, offset(lay_l, dz=CARRY), joints=True)),
        phase(0.5, "and lays it from the shelf into the lowest one"),
        phase(1.5, "", Move(w, offset(lay_l, dz=LET_GO))),
        phase(0.4, "", Release(w)),
        phase(1.5, "", Move(w, offset(back_off(lay_l, 0.10), dz=0.06))),
        phase(2.5, "the west arm goes to the stop in front of the tomato",
              Move(w, offset(back_off(take, 0.06), dz=0.04), joints=True), Open(w, STOP_SHAPE)),
        phase(1.0, "", Move(w, take)),
        phase(0.4, "", Grip(w, "stop", STOP_GRIP)),
        phase(1.0, "and lifts it away: the tomato rolls off the shelf", Move(w, offset(take, dz=0.05))),
        phase(1.0, "down the track", Move(w, offset(take, dx=-ux * 0.08, dy=-uy * 0.08, dz=0.05))),
        phase(5.0, "into the pot"),
    ]


def jobs(inst):
    """What the task asks (``tasks/jobs.py``): each channel taken where it lies (any part of it) and laid by its handle
    -- carried by one arm, or passed between two where they meet -- and the stop lifted away.  In the borrow version
    the long channel does the upper and the middle channel's work once it is laid; the short one does nothing."""
    from ..jobs import Job, Jobs, Thing

    margin = 0.03  # the reach audit's (`validate --ik --slack 0.03 Rea-02`)
    lay = [Job(f"lay the {what} channel", "carry", sites=(f"channel_{k}", f"hold_{k}"), slack=margin,
               body=f"channel_{k}") for k, what in ((3, "lowest"), (2, "middle"), (1, "upper"))]
    out = (*lay, Job("lift the stop away", sites=("stop",), slack=margin))
    things = ()
    if inst.variant == "borrow":
        b = inst.metadata["borrow"]
        prop, decoy = b["props"][0], b["decoys"][0]
        things = (Thing(prop, does=(lay[1].name, lay[2].name),
                        placing=(Job(f"lay {prop} from the shelf into the lowest channel", "carry",
                                     sites=(prop, "hold_long"), slack=margin, body=prop),),
                        why=f"{LONG:.2f} m: laid from the shelf, past the first stand, its lower end falls inside the "
                            "lowest channel"),
                  Thing(decoy, why=f"{SHORT:.2f} m: laid from the shelf it ends far short of the lowest channel, and "
                                   "from the first stand the lowest channel does not reach it"))
    return Jobs(out, things)
