"""Rea-01 cross-zone relay: fruit from the west end, west -> pad A -> south -> pad B -> east -> a crate at the east end."""

from __future__ import annotations

import math
import random
from dataclasses import replace

from ...policies.script import Grip, Move, Open, Release, back_off, offset, phase, tool_quat
from ...scoring.predicates import Eventually, HoldsAtEnd, InRegion, all_of, any_of
from ..layouts import TABLE_SIZE, mount
from ..loader import get
from ...engines.base import RegionSpec
from ._common import Z, at, box, compound, container, jitter, make_instance, pad, part, rng_for, scene

#: the start area and the bin are far out at the two ends of the table, behind the west and east arms: this far
#: from each base, this far round from straight ahead (a Piper's base turns 150 degrees each way) -- 2.1 m apart
BEHIND_R, BEHIND_DEG = 0.48, 140.0


def _behind(arm: str) -> tuple[float, float]:
    m = mount(arm)
    a = math.radians(m.yaw_deg + (BEHIND_DEG if arm == "west" else -BEHIND_DEG))
    return (m.x + BEHIND_R * math.cos(a), m.y + BEHIND_R * math.sin(a))


#: the pads between the west arm and the south one, and the south one and the east one
START, PAD_A, PAD_B, BIN = _behind("west"), (-0.40, -0.40), (0.40, -0.40), _behind("east")
#: a partition across the middle of the counter, in two parts (x thickness, y from .. to, height): the end PiPERs
#: reach 0.75 m, so without it they meet at the table's middle (0.70 m from each) and two arms do the task.  Two hands
#: 5 cm apart meet up to
#: 0.38 m over the table at y 0, 0.34 at y -0.21, 0.30 at y -0.24 and nowhere south of y -0.26 (IK): the tall part
#: stands where they meet high, the low one south of it -- under the south arm's hand at rest (y -0.24, 0.39 m up),
#: and where it carries each piece over, at y -0.40
PARTITION = (0.02, (-0.20, 0.70), 0.50)
PARTITION_LOW = (0.02, (-0.41, -0.20), 0.34)
#: south of the low part the partition goes on to the table's south edge, low (no two hands meet anywhere south of
#: y -0.26, at any height: it only has to stop a piece slid along the table), and the only way through is a covered
#: channel (TUNNEL).  From the channel's south wall a curb on the partition's line to just short of the south arm's base
#: (its capsule, radius 0.10 round (0, -0.70), reaches y -0.60), a jog west along CURB_JOG_Y to CURB_X and a curb
#: south from there to the table's edge, clear of that base by 2 cm -- the same in every version, so taking the south
#: arm away (shortfall, borrow) opens no gap where its base stood.  CURB_H tall.
CURB_H, CURB_T, CURB_JOG_Y, CURB_X = 0.11, 0.02, -0.575, -0.13
#: the channel: a straight passage along the lane LANE_Y (below), through the partition, its walls TUNNEL_L long (x
#: -0.195 .. +0.195), its roof ROOF_L (x -0.15 .. +0.15: where neither end arm's tool point gets, so a longer roof
#: would take no reach away; shorter than the walls, it stays 7 cm clear of a piece set on the board at rest) and TUNNEL_IN (width, height) inside, walls and roof TUNNEL_T thick.  It spans the middle where
#: neither end arm reaches: at y -0.48 a Piper's tool point, 3.5 cm over the table and at its best pitch, gets no
#: nearer the middle than x -0.157 / +0.157, and between the walls only its
#: fingers fit (its palm is 19 cm across the fingers' travel and 7 cm through, its back 6 cm behind the tool point;
#: robots/collision.py), so it gets in no further than its reach and no further than about 6 cm past a mouth.  A piece
#: on the long board (its top 7.8 cm up, an apple's stalk 8.8) passes 2.2 cm under the roof; the board's handle (and
#: the hand on it) never comes within 14 cm of a mouth.  The walls stand 3.5 cm from the board each side (its east end,
#: 63 cm from the hand, swings 2.2 cm for the 2 degrees the hand's turn is off in Isaac; see also REENTER).  Even 14 cm
#: wide the channel takes no palm under its roof: 19 cm long and 7 cm through, it does not fit the 14 x 11 cm section
#: any way round (diagonal 17.8 cm)
TUNNEL_L, ROOF_L, TUNNEL_IN, TUNNEL_T = 0.39, 0.30, (0.14, 0.11), 0.01
#: the fruit: apples and oranges 5.8 cm across -- a Piper's fingers open 7 cm -- each standing on a flat foot
#: (a ball let go over a pad rolls off it), its middle FRUIT_Z over what it stands on: where a hand holds it
FRUITS = ("apple", "orange")
FRUIT_R, FOOT_R, FOOT_H = 0.029, 0.015, 0.005
FRUIT_Z = FOOT_H - 0.0005 + FRUIT_R
#: fruit stands this far apart, in a row along the line from the arm that takes it (at the start, from the
#: west arm; in the crate, from the east arm): a hand's fingers close across that line and miss the others
ROW = 0.075
#: the crate's inside (roomy: the fruit stands in a row turned to the east arm, and its tilted hand comes in from
#: its side), its walls' height (low, for the same hand) and its floor's thickness
BIN_INNER, BIN_WALL, BIN_FLOOR = (0.25, 0.25), 0.03, 0.01
#: the borrow version (no south arm): a long board slides the fruit across through the channel.  A turntable would
#: not do: south of the partition the end arms reach no nearer the middle than x -0.15 / +0.15, a turntable's rim
#: comes within reach of both only 0.85 m across or more, 30-40 cm over the table's edge, and the fruit set on its rim
#: is still at the edge of reach (IK).  A board slid east carries a piece in a straight line: the west arm stands it on the board's east end (0.66 m from its
#: base) and pushes the board by the handle at its west end along the lane LANE_Y, past the partition's end, until
#: the piece is as far from the east arm (DELIVER); the east arm takes it off, the west arm pulls the board back for
#: the next.  The distractor is the same board SHORT_L long.  Where a piece passes the partition's end (y -0.455 or
#: further south) the west arm's tool point gets no further east than x -0.14 and the east arm's no further west than
#: +0.14 (IK, any pitch, 2-5 cm over the table; a grip takes a piece whose surface is within 2 cm), so the short
#: board, pushed by its knob, leaves its piece out of the east arm's reach.  (At 22 cm, used as a rake, it would bring
#: a piece 2 cm within the east arm's grip: two arms.  What matters is the length, so here the distractor is the
#: smaller.)  The board, as long as the prop BOARD[0], its width and thickness; how far in from its west end its
#: handle (a square POST) stands and from its east end the piece goes (SPOT_IN), in a low square COLLAR (inside
#: half-width, thickness, height) that keeps its foot from sliding (without it, the piece slides off the board's end
#: as the push ends); their weight, alike
LANE_Y, BOARD, SHORT_L, BOARD_KG = -0.485, (0.67, 0.07, 0.015), 0.16, 0.2
#: (the collar 5 cm long inside along the board, 6 cm across: set down by its middle (the hand's hold on it turned with
#: the hand, PLACE_DROP), a piece lands within about 1.3 cm of its place across the board and well within it along
#: (Isaac); 9 cm long inside, a piece would slide 3-5 cm in it as the board starts and stops, and 9 cm across, the
#: collar would come within 0.3 cm of the parked board's as the board slides into the lane; 1.2 cm high: 5 mm high,
#: it lets an orange tip over its side as the board starts)
POST, POST_IN, SPOT_IN, COLLAR = (0.03, 0.03, 0.05), 0.025, 0.04, (0.025, 0.03, 0.006, 0.012)
#: between pieces the board rests with its east end REENTER past where it starts, 1.5 cm between the channel's walls,
#: so that it is never pushed into the channel from outside again: let go out of it after each pull back, the board
#: takes up the hand's 2-degree turn error each time, and its end, off the lane, can run into the west end of the
#: channel's south wall
#: where the boards' handles stand at the start: both at HANDLE_X, one board each side of the lane, PARK_Y (which is
#: which drawn from the seed; the lane itself starts empty -- only in it, next to the partition's end, does the long
#: board bring a piece within both end arms' reach); how far the handle is pushed (the piece goes from 0.66 m from
#: the west arm's base to 0.66 m from the east arm's), and no farther than PUSH_R from the west arm's base
HANDLE_X, PARK_Y, PUSH, PUSH_R = -0.86, (-0.38, -0.59), 0.52, 0.61
#: the other distractor, the larger way round: the long board, as long and as heavy as the prop, whose collar's east
#: side (the one toward the channel) stands TALL_T over the table instead of 2 cm, 1 cm more than the channel's height
#: under its roof, so that pushed into the channel it runs into the roof's west edge (x -0.15) and the board goes no
#: further -- its piece stays 4 cm short of the roof, in the west arm's area.  (A board 12 cm thick all along, or its
#: last 10 cm raised to 12 cm, would be in the west arm's way as it works the prop beside it; parks further from the
#: lane are out of reach.)  Which of the two a scene has is drawn from the seed: the tall one
#: when seed + (1 at hard) is odd, so that over seeds 0-4 of both levels the prop is the larger of the pair in 5 of 10
TALL_T = 0.12


def tall_decoy(level: str, seed: int) -> bool:
    """Whether the borrow version's distractor at ``level``, ``seed`` is the tall board (else the short one)."""
    return (seed + (level == "hard")) % 2 == 1


#: the bodies partition_south() adds
CHANNEL_BODIES = ("tunnel_n", "tunnel_s", "tunnel_roof", "curb_s", "curb_jog", "curb_w")


def partition_south() -> list:
    """The partition south of its low part: the covered channel through it on the lane, and the curb from the
    channel to the table's south edge round the west side of the south arm's base (static, white)."""
    (w, h), t, half = TUNNEL_IN, TUNNEL_T, TUNNEL_L / 2
    s_out = LANE_Y - w / 2 - t  # the channel's south wall's outer face
    edge = -TABLE_SIZE[1] / 2
    return [
        box("tunnel_n", (0.0, LANE_Y + w / 2 + t / 2), (TUNNEL_L, t, h), "white", dynamic=False),
        box("tunnel_s", (0.0, LANE_Y - w / 2 - t / 2), (TUNNEL_L, t, h), "white", dynamic=False),
        box("tunnel_roof", (0.0, LANE_Y), (ROOF_L, w + 2 * t, t), "white", dynamic=False, z=Z + h + t / 2),
        box("curb_s", (0.0, (s_out + CURB_JOG_Y) / 2), (CURB_T, s_out - CURB_JOG_Y, CURB_H), "white", dynamic=False),
        box("curb_jog", ((CURB_X + CURB_T / 2) / 2, CURB_JOG_Y + CURB_T / 2), (CURB_T / 2 - CURB_X, CURB_T, CURB_H),
            "white", dynamic=False),
        box("curb_w", (CURB_X, (CURB_JOG_Y + edge) / 2), (CURB_T, CURB_JOG_Y - edge, CURB_H), "white", dynamic=False),
    ]


def board(name: str, length: float, handle_xy, rim_top: float = BOARD[2] + COLLAR[3]):
    """A board (a long wooden paddle, its handle a black knob) ``length`` long lying east-west, its handle (a post at
    its west end) at ``handle_xy``, the east side of its collar (the rim round a piece's place) ``rim_top`` over the
    table; its frame is the middle of the slab.  (The west hand may set a piece down on the rim's corner, and it drops
    into the rim only as the board slides under it, 0.3 cm; a lighter, slimmer spatula slides further there, and the
    piece stays standing on the rim, where the channel's roof knocks it off.)"""
    cx = handle_xy[0] - POST_IN + length / 2
    sx, (rx, ry, t, h) = length / 2 - SPOT_IN, COLLAR  # the piece's place, from the board's middle; its collar
    he = rim_top - BOARD[2]  # the collar's east side's height
    z = BOARD[2] / 2 + h / 2
    parts = (part("slab", "box", (length, BOARD[1], BOARD[2])),
             part("handle", "box", POST, (-length / 2 + POST_IN, 0.0, BOARD[2] / 2 + POST[2] / 2), color="rubber"),
             *(part(f"collar_{k}", "box", size, (sx + dx, dy, zz)) for k, (dx, dy, size, zz) in {
                 "n": (0.0, ry + t / 2, (2 * (rx + t), t, h), z), "s": (0.0, -ry - t / 2, (2 * (rx + t), t, h), z),
                 "e": (rx + t / 2, 0.0, (t, 2 * ry, he), BOARD[2] / 2 + he / 2),
                 "w": (-rx - t / 2, 0.0, (t, 2 * ry, h), z)}.items()))
    return compound(name, (cx, handle_xy[1], Z + BOARD[2] / 2), parts, "wood", mass=BOARD_KG)


def spot(b) -> tuple[float, float]:
    """Where a piece of fruit stands on board ``b`` (an ObjectSpec, where it lies): SPOT_IN from its east end."""
    return (b.pos[0] + b.part("slab").size[0] / 2 - SPOT_IN, b.pos[1])


#: where the long board, pushed, brings a piece: the east arm takes it there
DELIVER = (HANDLE_X - POST_IN + BOARD[0] - SPOT_IN + PUSH, LANE_Y)
REENTER = 0.03


def _radial(arm: str, xy) -> tuple[float, float]:
    """The unit vector, on the table, from an arm's base to ``xy``."""
    m = mount(arm)
    dx, dy = xy[0] - m.x, xy[1] - m.y
    n = math.hypot(dx, dy)
    return dx / n, dy / n


def _row(arm: str, centre, n: int, step: float = ROW) -> list[tuple[float, float]]:
    """``n`` places ``step`` apart on the line from ``arm`` through ``centre``, centred on it."""
    rx, ry = _radial(arm, centre)
    return [(centre[0] + (i - (n - 1) / 2) * step * rx, centre[1] + (i - (n - 1) / 2) * step * ry) for i in range(n)]


def _square_to(arm: str, xy) -> float:
    """The yaw (deg) that turns a body square to the line from ``arm`` to ``xy``."""
    rx, ry = _radial(arm, xy)
    return math.degrees(math.atan2(ry, rx))


def fruit(name: str, kind: str, xy, yaw: float = 0.0):
    """An apple (a stalk and a leaf) or an orange (a green calyx), standing on its foot at ``xy``; its frame is its
    middle."""
    tip = (part("stalk", "cylinder", (0.0025, 0.012), (0.0, 0.0, FRUIT_R + 0.004), color="stem"),
           part("leaf", "box", (0.02, 0.009, 0.002), (0.009, 0.0, FRUIT_R + 0.006), 30.0, color="leaf")) \
        if kind == "apple" else (part("calyx", "cylinder", (0.006, 0.003), (0.0, 0.0, FRUIT_R), color="leaf"),)
    parts = (part("flesh", "sphere", (FRUIT_R,)), part("foot", "cylinder", (FOOT_R, FOOT_H),
                                                       (0.0, 0.0, -FRUIT_Z + FOOT_H / 2)), *tip)
    return compound(name, (xy[0], xy[1], Z + FRUIT_Z), parts, "apple" if kind == "apple" else "peel", mass=0.12,
                    yaw=yaw)


def build(level: str, seed: int, variant: str = "exact"):
    spec = get("Rea-01")
    rng = rng_for(spec, level, seed)
    n = int(spec.knob.levels[level])
    kinds = [FRUITS[i % len(FRUITS)] for i in range(n)]
    fruit_ = [fruit(f"{k}_{i}", k, jitter(rng, xy, 0.005), yaw=_square_to("west", START))
              for i, (k, xy) in enumerate(zip(kinds, _row("west", START, n)))]
    bin_parts, bin_in = container("bin", BIN, BIN_INNER, wall_h=BIN_WALL)
    # inside the crate: up to a fruit's height above its floor (its walls are lower than the fruit)
    bin_in = RegionSpec("bin_in", (*BIN, Z + BIN_FLOOR + FRUIT_Z), (BIN_INNER[0] / 2, BIN_INNER[1] / 2, FRUIT_Z),
                        bin_in.color, False)
    regions = [pad("start", START, (0.12, 0.12), "wood"), pad("pad_a", PAD_A, (0.05, 0.05), "cork"),
               pad("pad_b", PAD_B, (0.05, 0.05), "cork"), bin_in]
    walls = []
    for name, (t, (y0, y1), h) in (("partition", PARTITION), ("partition_low", PARTITION_LOW)):
        walls.append(box(name, (0.0, (y0 + y1) / 2), (t, y1 - y0, h), "white", dynamic=False))
    walls += partition_south()
    boards, extra = [], {}
    if variant == "borrow":  # no south arm: a long board slides the fruit through the channel; a short one does not
        # reach, a tall one does not pass (which of the two: tall_decoy)
        kind = "tall" if tall_decoy(level, seed) else "short"
        rim = BOARD[2] + COLLAR[3]
        dims = [(BOARD[0], rim), (BOARD[0], TALL_T) if kind == "tall" else (SHORT_L, rim)]
        random.Random(f"borrow/{spec.seed_id}/{level}/{seed}").shuffle(dims)  # north to south: drawn from the seed
        boards = [board(f"board_{k + 1}", length, (HANDLE_X, y), top)
                  for k, ((length, top), y) in enumerate(zip(dims, PARK_Y))]
        prop = boards[dims.index((BOARD[0], rim))]
        # where the west arm sets a piece on the long board slid into the lane, and where the board brings it (the
        # milestones' pads here: they stay painted, the hand-off places of the south arm, which is not on the table)
        top = Z + BOARD[2] + FRUIT_Z
        regions += [RegionSpec(nm, (*xy, top), (0.04, 0.04, 0.02), visible=False)
                    for nm, xy in (("board_load", (spot(prop)[0] + REENTER, LANE_Y)), ("board_far", DELIVER))]
        extra = {"borrow": {"props": [prop.name], "decoys": [b.name for b in boards if b is not prop]},
                 "decoy_kind": kind}
    sc = scene(spec, fruit_ + bin_parts + walls + boards, regions)
    if variant == "borrow":
        sc = replace(sc, arms=tuple(a for a in sc.arms if a.slot != "south"))

    names = [c.name for c in fruit_]
    in_bin = [InRegion(c, "bin_in") for c in names]
    on_a = [InRegion(c, "pad_a") for c in names]
    on_b = [InRegion(c, "pad_b") for c in names]
    reach = [("west", "start"), ("west", "pad_a"), ("south", "pad_a"), ("south", "pad_b"),
             ("east", "pad_b"), ("east", "bin_in")]
    out_of_reach = [("west", "pad_b"), ("south", "start"), ("south", "bin_in"), ("east", "pad_a")]
    if variant == "borrow":  # the board's ends do what the pads did: the first hand-off and the second
        on_a += [InRegion(c, "board_load") for c in names]
        on_b += [InRegion(c, "board_far") for c in names]
        # (no claims for the board's ends: 0.66 m from the end arms' bases, beyond the design rules' nominal reach
        # of 0.56 m -- IK's alone, which the scripted reference's plan checks)
        reach = [r for r in reach if r[0] != "south"]
        out_of_reach = [r for r in out_of_reach if r[0] != "south"]
    what = {1: "the apple", 2: "the apple and the orange"}.get(n, f"the {n} pieces of fruit")
    return make_instance(
        spec, level, seed, sc,
        instruction=(f"Move {what} from the start area at the far west end of the table into the crate at the far "
                     "east end."),
        # every piece in the crate when the run ends, through its last second at least: a piece held in the crate
        # for a second and taken out again does not count
        success=HoldsAtEnd(all_of(*in_bin), 1.0),
        milestones={
            "fruit_on_pad_a": Eventually(any_of(*on_a)),
            "fruit_on_pad_b": Eventually(any_of(*on_b)),
            "fruit_in_bin": Eventually(any_of(*in_bin)),
            "all_in_bin": Eventually(all_of(*in_bin)),
        },
        reach=reach, out_of_reach=out_of_reach,
        apart=[("west", "east")], fruit=names, **extra,
    )


#: in the borrow version the fruit stands further apart in the crate: set on the board by a hand turned otherwise than
#: the one that took it, a piece comes to the crate turned (an apple's leaf 1.3 cm from the next piece, the east
#: fingers 0.8 cm from it, at ROW; pieces 1.2 cm apart at 8.5 cm)
BORROW_BIN_ROW = 0.095


def bin_slots(n: int, step: float = ROW) -> list[tuple[float, float]]:
    """Where the fruit goes in the crate, in the order it goes in: in a row along the line from the east arm, the
    farthest first, so that the hand never reaches in over a piece already there."""
    return _row("east", BIN, n, step)[::-1]


def keyframes(inst):
    z = Z + FRUIT_Z
    fruit_ = [inst.scene.obj(n) for n in inst.metadata["fruit"]]  # (not every body that moves: a distractor)
    frames = []
    if inst.variant == "borrow":  # the long board slid into the lane; each piece set on it, the board pushed east, the
        b = inst.scene.obj(inst.metadata["borrow"]["props"][0])  # piece into the crate
        lane = replace(b, pos=(b.pos[0] + REENTER, LANE_Y, b.pos[2]))  # where it rests between pieces
        pushed = offset(lane.pose, dx=PUSH - REENTER)
        frames.append({b.name: lane.pose})
        for o, slot in zip(fruit_, bin_slots(len(fruit_), BORROW_BIN_ROW)):
            frames += [{o.name: at((*spot(lane), z + BOARD[2]))},
                       {b.name: pushed, o.name: at((*DELIVER, z + BOARD[2]))},
                       {b.name: lane.pose, o.name: at((*slot, z + BIN_FLOOR))}]
        return frames
    for o, slot in zip(fruit_, bin_slots(len(fruit_))):
        frames += [{o.name: at((*PAD_A, z))}, {o.name: at((*PAD_B, z))}, {o.name: at((*slot, z + BIN_FLOOR))}]
    return frames


#: the scripted reference: how high a hand carries a piece of fruit (its tool point over the table), how far apart
#: the fingers close on one (taken by its middle), and how far above where it goes it is let go
HOVER, FRUIT_GRIP, DROP = 0.12, 2 * FRUIT_R - 0.002, 0.004
#: one leg of the relay, in phases: over the fruit, down, take it, up, across, down, let go, up, back to wait (a
#: hand that goes straight from letting go to where it waits can drag a piece off the pad with it)
LEG = (2.0, 1.0, 0.3, 1.0, 2.0, 1.0, 0.3, 1.0, 1.5)
#: how far below level a hand points (degrees), by robot: a Piper pointing straight down reaches no higher than
#: about 10 cm above its base, so it comes at the fruit from its own side, its fingers closing square to its reach
PITCH = {"piper": 45.0}


#: how far over the partition's top the south arm carries a piece (its hand's lowest point, the fruit, below the tool point)
OVER_PARTITION = 0.10


#: the borrow version's push: the west hand takes the board's handle this far under the post's top, coming in along
#: its pointing from this far back, and holds one turn the whole way -- pointing PUSH_HEADING (deg, from east) and
#: PUSH_PITCH below level -- so that the board goes straight (pointing out from its base instead, the hand would turn
#: it 55 degrees on the way).  Held so, the hand follows the push from the board's start to its end by IK; pointing
#: -80 to -110 degrees at 30-45 down, it reaches every point of it but loses the way between them
POST_TAKE, POST_BACK, PUSH_HEADING, PUSH_PITCH = 0.012, 0.06, -65.0, 45.0
#: where the east hand draws a piece back to, toward its base, before it swings round to the crate (its base turns
#: 188 degrees the only way its joint allows, through straight west: from where it takes the piece, the piece passes
#: over the channel's roof 0.6 cm from it; lifted higher there first, the hand has no IK solution)
EAST_BACK = (0.42, -0.30)
#: how far below level the west hand points setting a piece on the board (at rest, the board puts it 3 cm further east
#: than where it starts: pointing 45 degrees down the hand does not reach the place above it, IK)
BOARD_PITCH = 30.0
#: and which way it points there (degrees from east): IK holds at -15 to -45 (30 down), and at -15 its palm is furthest
#: from the lane's north side
PLACE_HEADING = -15.0
#: how far over its place a piece is let go on the long board (its middle above where it stands)
PLACE_DROP = 0.005
#: how far south of DELIVER the east hand takes a piece off the long board: pushed, the board turns about 3.5 degrees
#: clockwise -- the west hand's turn is that much further off at the push's end than where it takes the handle, the
#: same at every push, the lane being the same in every scene -- and the piece arrives 2-2.4 cm (the board's middle)
#: and 2.8-4.7 cm (the piece) south of DELIVER (Isaac).  (Rails in the channel to hold the board square would not do:
#: the stiffly held board, pressed sideways at its foot, is lifted 2.4-4 cm passing them and throws its piece off.)
TAKE_DY = -0.035


def _rot(q) -> list:
    """The rotation matrix (rows) of a unit quaternion (w, x, y, z)."""
    w, x, y, z = q
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]
#: how long the push and the pull back take
PUSH_S, PULL_S = 3.5, 2.5
#: how much higher than it carries a piece the west hand goes between the board's ends (lower, its fingers come within
#: a centimetre of a piece on the short board and of the handle as they sweep past)
OVER = 0.05


def _borrow_script(inst):
    """The borrow version (no south arm), privileged as the exact one.  For each piece: the west arm takes it from the
    start area and stands it on the long board's east end, takes the board by its handle and pushes it east along its
    lane, past the partition's end, and lets go; the east arm takes the piece off (where the long board brings it,
    DELIVER) and puts it in the crate while the west arm pulls the board back for the next.  First the west arm slides
    the board sideways into the lane (it starts beside it).  With the distractor (``--decoy``) the west arm slides the
    short board into the lane and pushes it as far as it reaches: the piece stops far short of the east arm, which
    closes on nothing."""
    sc = inst.scene
    pieces = list(inst.metadata["fruit"])
    b0 = sc.obj(inst.metadata["borrow"]["props"][0])
    b = replace(b0, pos=(b0.pos[0], LANE_Y, b0.pos[2]))  # slid into the lane
    length = b.part("slab").size[0]
    thick = BOARD[2]
    tall = BOARD[2] + b.part("collar_e").size[2] > TUNNEL_IN[1]  # the tall distractor (--decoy)
    grasp_up = .020 if length < .20 else 0.
    z = Z + FRUIT_Z + grasp_up
    on_board = z + thick
    hx, hy = b.pos[0] - length / 2 + POST_IN, b.pos[1]  # the handle, in the lane
    w = mount("west")
    stroke = min(PUSH, w.x + math.sqrt(PUSH_R ** 2 - (hy - w.y) ** 2) - hx)
    load = (spot(b)[0] + REENTER, spot(b)[1])  # (the board at rest, its end in the channel's mouth)
    short = math.dist((spot(b)[0] + stroke, b.pos[1]), DELIVER) > 0.02 or tall
    if short:  # a distractor: the first piece never reaches the
        pieces = pieces[:1]  # east arm and stays on the board, where the next would go -- one shows it

    def across(arm, xy):  # the fingers' line: square to the arm's reach there, on the table
        rx, ry = _radial(arm, xy)
        return (-ry, rx, 0.0)

    def hand(slot, xy, zz, pitch=PITCH["piper"], fingers=None):
        """The tool point at (xy, zz), pointing ``pitch`` below level away from the arm's base, fingers square to it
        (or as near ``fingers`` as square to the pointing lets them)."""
        p = math.radians(pitch)
        rx, ry = _radial(slot, xy)
        return (*xy, zz, *tool_quat((rx * math.cos(p), ry * math.cos(p), -math.sin(p)), fingers or across(slot, xy)))

    h, p = math.radians(PUSH_HEADING), math.radians(PUSH_PITCH)
    on_handle = (hx - (.010 if length < .20 else 0.), hy, Z + BOARD[2] + POST[2] - POST_TAKE + grasp_up,
                 *tool_quat((math.cos(h) * math.cos(p), math.sin(h) * math.cos(p), -math.sin(p)),
                            (-math.sin(h), math.cos(h), 0.0)))
    pushed = offset(on_handle, dx=stroke)
    parked = offset(on_handle, dy=b0.pos[1] - LANE_Y)
    rest = offset(on_handle, dx=REENTER)  # the handle with the board at rest
    if tall:  # the tall distractor: pushed until its collar's east side meets the roof's west edge, and 2 cm on, then
        # held and let go (a safe stop)
        meets = -ROOF_L / 2 - (spot(b)[0] + REENTER + COLLAR[0] + COLLAR[2])  # its east side's travel from rest
        pushed = offset(rest, dx=meets + 0.02)
    up = HOVER - FRUIT_Z  # from a piece's middle to the carrying height
    slots = bin_slots(len(pieces), BORROW_BIN_ROW)
    out = [
        phase(2.0, "west: slides the board into the lane",
              Move("west", offset(back_off(parked, POST_BACK), dz=OVER), joints=True), Open("west", 0.05)),
        phase(1.0, "", Move("west", back_off(parked, POST_BACK))),
        phase(1.0, "", Move("west", parked)),
        phase(0.25, "", Grip("west", b.name, POST[1])),
        phase(2.0, "", Move("west", on_handle)),
        phase(1.0, "west: sets the board's end into the channel's mouth", Move("west", rest)),
        phase(0.5, ""),
        phase(0.25, "", Release("west")),
        phase(1.0, "", Move("west", offset(back_off(rest, POST_BACK), dz=OVER))),
    ]
    for i, name in enumerate(pieces):
        fruit_raise = .012 if length > .20 else 0.
        handle_now, load_now = rest, load  # (the channel keeps the board on its lane: no re-centring)
        pick = hand("west", sc.obj(name).pos[:2], z + fruit_raise)
        if length > .20:
            # the long board: the hand points PLACE_HEADING, BOARD_PITCH down, fingers closing north-south (pointing out
            # from its base, its palm comes within 0.8 cm of a tall distractor's collar parked north of the lane; 45
            # down, it does not reach the place above; fingers square to the reach open to within 0.8 cm of the
            # roof from x -0.18).  The piece hangs in the hand as it was taken (its middle fruit_raise under
            # the tool point, in the turn it was taken with): the tool point goes where that puts the piece's middle
            # PLACE_DROP over its place (placed by the tool point, it stands 3 cm east, on its collar's rim)
            hd, pt = math.radians(PLACE_HEADING), math.radians(BOARD_PITCH)
            q = tool_quat((math.cos(hd) * math.cos(pt), math.sin(hd) * math.cos(pt), -math.sin(pt)), (0.0, 1.0, 0.0))
            rq, rp = _rot(q), _rot(pick[3:])
            m = [[sum(rq[r][k] * rp[c][k] for k in range(3)) for c in range(3)] for r in range(3)]  # rq @ rp.T
            v = [m[r][2] * fruit_raise for r in range(3)]  # the tool point from the piece's middle, turned
            mid = (*load_now, Z + BOARD[2] + FRUIT_Z + PLACE_DROP)
            place = (mid[0] + v[0], mid[1] + v[1], mid[2] + v[2], *q)
        else:  # the short board's place, behind the west arm's base: 45 degrees down, fingers square to the reach
            place = hand("west", load_now, on_board + DROP + fruit_raise)
        take = (hand("east", (DELIVER[0], DELIVER[1] + TAKE_DY), on_board + .020 - grasp_up, BOARD_PITCH)
                if length > .20 else hand("east", DELIVER, on_board + .020 - grasp_up))  # (30 down: 45 does not reach)
        # over the place on the long board the hand stands 6 cm west of it (straight over it, the hand comes within
        # 0.8 cm of a tall distractor's collar parked north of the lane)
        aside = -0.06 if length > .20 else 0.0
        drop = hand("east", slots[i], z + BIN_FLOOR + DROP + .020 - grasp_up)
        # (5 cm higher than it carries: opening lower over the crate, its fingers come within 0.8 cm of the piece in
        # the crate's middle)
        first = [Move("east", hand("east", BIN, Z + HOVER + 0.05), joints=True)] if i else []
        out += [
            phase(2.0, f"west: piece {i + 1} onto the board", Move("west", offset(pick, dz=up), joints=True),
                  Open("west", 0.07), *first),
            phase(1.0, "", Move("west", pick)),
            phase(0.25, "", Grip("west", name, FRUIT_GRIP)),
            phase(1.0, "", Move("west", offset(pick, dz=up))),
            phase(1., "carry inside the west workspace before extending",
                  Move("west", hand("west", (-.45, -.40), Z + HOVER + fruit_raise), joints=True)),
            phase(1.5, "", Move("west", offset(place, dx=aside, dz=up))),
            phase(1.0, "", Move("west", place)),
            phase(0.25, "", Release("west")),
            phase(1.0, "", Move("west", offset(place, dx=aside, dz=up))),
            phase(2.0, "west: pushes the board east",
                  Move("west", offset(back_off(handle_now, POST_BACK), dz=OVER), joints=True), Open("west", 0.05)),
            phase(1.0, "", Move("west", back_off(handle_now, POST_BACK))),
            phase(1.0, "", Move("west", handle_now)),
            phase(0.25, "", Grip("west", b.name, POST[1])),
            phase(PUSH_S + 0.5, "", Move("west", pushed), Move("east", offset(take, dz=up), joints=True),
                  Open("east", 0.07)),
            phase(1.0, f"east: takes piece {i + 1} off the held board", Move("east", take)),
            phase(0.25, "transfer the fruit while releasing the board", Release("west"), Grip("east", name, FRUIT_GRIP)),
            phase(1.0, "", Move("east", offset(take, dz=up))),
            phase(1.0, "east: draws it back, clear of the channel",
                  Move("east", hand("east", EAST_BACK, on_board + .020 - grasp_up + up))),
            phase(0.25, "west: takes the board back", Grip("west", b.name, POST[1])),
            phase(PULL_S, "west: pulls the board back",
                  Move("west", rest), Move("east", offset(drop, dz=up - BIN_FLOOR - DROP),
                                                             joints=True)),
            # (held there a moment before it is let go: let go at the pull's end, the hand still 1.2-1.5 cm short of
            # where it rests, the board rests that much further east each time, and the hand misses its handle)
            phase(0.5, "west: holds the board at rest"),
            phase(1.0, "", Release("west"), Move("east", drop)),
            phase(0.25, "", Release("east")),
            phase(1.0, "", Move("west", offset(back_off(handle_now, POST_BACK), dz=OVER)),
                  Move("east", offset(drop, dz=up - BIN_FLOOR - DROP))),
        ]
    out += [phase(1.5, "", Move("east", hand("east", BIN, Z + HOVER + 0.05), joints=True)), phase(0.5, "done")]
    if short:
        stop = next(k for k, ph in enumerate(out) if ph.note == "transfer the fruit while releasing the board")
        return out[:stop + 1] + [
            phase(1., "withdraw from the released board", Move("west", offset(pushed, dz=.10))),
            phase(2., "observe the fruit left short of the east hand"),
        ]
    return out


def script(inst):
    """Privileged: a grip fixes the fruit to the hand.  The arms work as a pipeline: in each stage the west arm
    takes a piece to pad A while the south arm takes the one before from pad A to pad B and the east arm the one
    before that from pad B into the crate.  A pad is emptied before the next piece is put on it.

    A Piper takes and puts a piece square to its own reach; the south Franka, pointing straight down, takes it
    however the west arm left it and turns it on its way, so that it leaves it square to the east arm."""
    if inst.variant == "borrow":
        return _borrow_script(inst)
    sc = inst.scene
    pieces = list(inst.metadata["fruit"])
    z = Z + FRUIT_Z  # the tool point on a piece of fruit standing on the table: its middle

    def across(arm, xy):  # the fingers' line: square to the arm's reach there, on the table
        rx, ry = _radial(arm, xy)
        return (-ry, rx, 0.0)

    def hand(slot, xy, zz, fingers):
        """The tool point at (xy, zz), pointing down -- tilted, for a Piper, away from its base -- fingers along
        ``fingers``."""
        arm = sc.arm(slot)
        p = math.radians(PITCH.get(arm.model, 90.0))
        rx, ry = _radial(slot, xy)
        approach = (rx * math.cos(p), ry * math.cos(p), -math.sin(p))
        return (*xy, zz, *tool_quat(approach, fingers))

    def turned_like(ref, f):
        """``f`` or ``f`` turned a quarter turn -- round fruit is held either way -- whichever is nearer ``ref``."""
        g = (-f[1], f[0], 0.0)
        return max((f, g, (-f[0], -f[1], 0.0), (-g[0], -g[1], 0.0)), key=lambda v: v[0] * ref[0] + v[1] * ref[1])

    on_a, on_b = across("west", PAD_A), across("east", PAD_B)
    slots = bin_slots(len(pieces))
    legs = {  # (where from, the fingers there, where to, the fingers there)
        "west": lambda i: (sc.obj(pieces[i]).pos[:2], across("west", sc.obj(pieces[i]).pos), PAD_A, on_a),
        "south": lambda i: (PAD_A, on_a, PAD_B, turned_like(on_a, on_b)),
        "east": lambda i: (PAD_B, on_b, slots[i], across("east", slots[i])),
    }
    rise = {"west": 0.0, "south": 0.0, "east": BIN_FLOOR}  # what each arm puts its piece down on
    # where each arm waits after letting go, over its own zone: one that waits over the pad it has just
    # filled is in the way of the arm coming to empty it
    # (the south arm waits south-east of the partition's end: right in front of it, its wrist camera comes within 6 mm)
    rest = {"west": START, "south": (0.20, -0.50), "east": BIN}
    lag = {"west": 0, "south": 1, "east": 2}
    phases = []
    for k in range(len(pieces) + 2):
        jobs = {slot: k - d for slot, d in lag.items() if 0 <= k - d < len(pieces)}
        note = "  ".join(f"{slot}: piece {i + 1}" for slot, i in jobs.items())
        steps = [[] for _ in LEG]
        for slot, i in jobs.items():
            src, f_src, dst, f_dst = legs[slot](i)
            if sc.arm(slot).model != "piper":  # a hand pointing straight down keeps its turn between the two
                f_src = turned_like(f_dst, f_src)
            pick, place = hand(slot, src, z, f_src), hand(slot, dst, z + rise[slot] + DROP, f_dst)
            up = Z + HOVER - z
            piper = sc.arm(slot).model == "piper"
            # the south arm crosses the partition's low part both ways, so it goes between the pads over it (it
            # starts right above it, too): straight up or down only over a pad
            over = (Z + PARTITION_LOW[2] + OVER_PARTITION - z) if slot == "south" else up
            steps[0] += [Move(slot, offset(pick, dz=over), joints=piper), Open(slot, 0.07)]
            steps[1].append(Move(slot, pick))
            steps[2].append(Grip(slot, pieces[i], FRUIT_GRIP))
            steps[3].append(Move(slot, offset(pick, dz=over)))
            # a Piper goes between behind it and in front of it turning about its base, in joint space; the south
            # arm carries over the partition (straight across at the hover height, its wrist camera comes within 6 mm
            # of the partition's end; a path round its end leaves the arm where it cannot come straight in next)
            steps[4].append(Move(slot, offset(place, dz=over - rise[slot] - DROP), joints=piper))
            steps[5].append(Move(slot, place))  # (down from over the partition, for the south arm)
            steps[6].append(Release(slot))
            steps[7].append(Move(slot, offset(place, dz=over - rise[slot] - DROP)))
            # (the south arm rests as high as it goes over the partition: from there to pad A is a straight line
            # south of both parts; from lower down its path swings north into the tall one)
            rest_z = z + over if slot == "south" else Z + HOVER
            steps[8].append(Move(slot, hand(slot, rest[slot], rest_z, across(slot, rest[slot]) if piper
                                            else (1.0, 0.0, 0.0)), joints=piper))
        phases += [phase(sec, note if j == 0 else "", *acts) for j, (sec, acts) in enumerate(zip(LEG, steps))]
    phases.append(phase(2.0, "done"))
    return phases


def jobs(inst):
    """What the task asks (``tasks/jobs.py``): each piece of fruit carried from where it stands at the start into the
    crate -- by one arm, or passed from arm to arm where two meet (the partition and its channel between).  In the
    borrow version the long board does the crossing: put in place, for each piece, by an arm that stands the piece in
    its collar on the board's lane and pushes it by its handle through the channel, and one that takes the piece off
    where the board brings it (DELIVER) into the crate; the short board brings nothing within reach."""
    from ..jobs import Job, Jobs, Thing

    carry = tuple(Job(f"carry {name} from the start area into the crate", "carry", sites=(name, "bin_in"),
                      slack=0.05, body=name) for name in inst.metadata["fruit"])
    if inst.variant != "borrow":
        return Jobs(carry)
    b = inst.metadata["borrow"]
    prop, decoy = b["props"][0], b["decoys"][0]
    board_ = inst.scene.obj(prop)
    lane = replace(board_, pos=(board_.pos[0] + REENTER, LANE_Y, board_.pos[2]))  # at rest, its end in the channel
    on = Z + BOARD[2] + FRUIT_Z  # a piece's middle, standing on the board
    load = (*spot(lane), on)
    handle = (lane.pos[0] - BOARD[0] / 2 + POST_IN, LANE_Y, Z + BOARD[2] + POST[2])
    placing = (Job(f"push {prop} by its handle through the channel", sites=(handle,
                                                                            (handle[0] + PUSH - REENTER, *handle[1:])),
                   slack=0.05),)
    for name in inst.metadata["fruit"]:
        placing += (Job(f"stand {name} on {prop} at its east end", "carry", sites=(name, load), slack=0.05, body=name),
                    Job(f"take {name} off {prop} into the crate", "carry", sites=((*DELIVER, on), "bin_in"),
                        slack=0.05, body=name))
    things = (Thing(prop, does=tuple(j.name for j in carry), placing=placing,
                    why=f"{BOARD[0] * 100:.0f} cm long: pushed {PUSH * 100:.0f} cm by its handle, it brings a piece in "
                        "its collar out of the channel's east end within the east arm's reach"),
              Thing(decoy, why=(f"its collar's east side {TALL_T * 100:.0f} cm high, taller than the channel's "
                                f"{TUNNEL_IN[1] * 100:.0f} cm under its roof: it does not go in, and a piece on it stays "
                                "in the west arm's area")
                    if inst.metadata["decoy_kind"] == "tall" else
                    (f"{SHORT_L * 100:.0f} cm long: pushed by its handle, which no hand takes into the channel, it "
                     "leaves a piece short of the channel's middle, out of the east arm's reach")))
    return Jobs(carry, things)
