"""Rea-04 wipe the spills: three coffee spills lie along the counter, one in each arm's part of it, and there is one
sponge, on its dish at the west end; every spill must be wiped, and the sponge end in the sink at the east end.

Each spill is within reach of one arm only, the sponge's dish of the west arm only and the sink of the east arm
only: the counter is too long for fewer arms.  The middle spill lies well clear of the end arms' reach (checked by IK
with margins, ``validate --ik --slack``): an end arm reaching within 5 cm of it, holding the sponge, could wipe it,
and pass the sponge on over the table's middle, which both end arms reach.  The sponge goes from arm to arm, put down
where both can reach; put down, it stays, so no arm holds anything while another works -- a tool that travels to the work."""

from __future__ import annotations

import math
import random
from dataclasses import replace

from ...engines.base import RegionSpec, yaw_quat
from ...policies.script import (Grip, Move, Open, Release, back_off, carry, offset, phase, radial_tool,
                                tool_quat)
from ...scoring.predicates import HoldsAtEnd, HoldsFor, InRegion, all_temporal
from ..layouts import mount
from ..loader import get
from ._common import Z, box, compound, container, cylinder, jitter, make_instance, pad, part, rng_for, scene

#: the spills: where, how wide (radius) and thick.  The middle one 19 cm in front of the south arm's base, 0.87 m
#: from each end arm's: a PiPER reaches 0.75 m near the table, and a hand holding the sponge by its end (within a
#: grip's 2 cm of it) puts its middle no more than 6 cm off, so an end arm wipes it only from within 8.5 cm (with
#: WIPE_R), and comes no nearer than 12.  At 40 cm in front of the south arm (y = -0.30) an end arm would reach to
#: within 3 cm of it; at 25 cm, within 8 cm.  So near its base the south arm is a Piper (a Franka there hardly reaches 8 cm up)
SPILLS = {"west": (-0.95, -0.25), "south": (0.0, -0.51), "east": (0.95, -0.25)}
SPILL_R, SPILL_T = 0.055, 0.002
#: a wipe counts with the sponge's middle this near the spill's middle (each way)
WIPE_R = 0.025
#: the sponge (length, width, height), its dish (where, size), where it is put down between arms, and the sink (the
#: dish and the sink no more than about 130 degrees round from where their arm faces: a Piper's base turns 150 each way)
SPONGE = (0.08, 0.05, 0.045)
DISH_XY, DISH = (-0.86, 0.22), (0.13, 0.10, 0.008)
#: (0.50 m from the end Piper beside it, which reaches it pointing 75 degrees down, and 0.51 m from the south one)
PADS = {"a": (-0.42, -0.41), "b": (0.42, -0.41)}
SINK_XY, SINK_IN, SINK_WALL = (0.86, 0.24), (0.14, 0.12), 0.05
#: a sponge counts as on a spill when its middle is over it, down on it (within this of resting on it: a Piper
#: taking it 0.5 m out sags a centimetre, and holds it that much higher than planned)
PRESSED = 0.010
#: a riser under the middle spill (in every configuration), fixed to the counter: a block RISER_HALF each way from the spill's middle and RISER_H high, the spill on its top (as
#: wide as the top: the middle spill is 9 cm across, the end ones 11).  Why: a sponge gets onto it only lifted --
#: pushed, it meets a 6 cm step -- so no pushing route reaches the middle spill (a sponge stood on its 8 cm end could
#: be levered over a rim round the spill; a riser has nothing to lever over).  Why so small: an end
#: arm's tool point comes no nearer the spill's middle than 12 cm at the height it holds a sponge lying on the riser
#: (14 cm at 4 cm over the counter; IK), and a hand holding the sponge by its end (within a grip's 2 cm of
#: it) puts the sponge's middle no nearer than 6 cm: 1.5 cm beyond the riser's 4.5 cm edge, so an end arm cannot set
#: the sponge down on it (it would tip off).  The sponge's middle stays over the top as it scrubs (WIPE_R + the
#: scrub's 1.2 cm, 3.7 cm, inside 4.5); turned, its corners may stand over the edge
RISER_HALF, RISER_H = 0.045, 0.06
#: the borrow version (no south arm): a sponge mop.  Holding its handle, the west arm presses its clip round the
#: sponge (left on pad A), lifts the sponge onto the riser's top over the middle spill and wipes it with the sponge
#: itself, then leaves it at HANDOFF, where the east arm takes it, and lays the mop back.  Its clip: two jaws along the
#: sponge's long sides (length, thickness, height: down to 2 cm over the sponge's bottom) under a plate; a neck; the
#: handle, MOP_HANDLE long from the neck, held MOP_D from the clip's middle (the hand pointing MOP_PITCH down along
#: it), lying with its end on a rest, MOP_REST across.  The prop's jaws stand MOP_OPEN apart (the sponge is 5 cm
#: across); the distractor's either MOP_NARROW, narrower than the sponge is any way round (4.5 cm), or MOP_WIDE,
#: wider than it is any way round by more than a clip's slack (its widest, corner to corner, 9.4 cm, plus
#: ``collision.CLIP_LOOSE``): its rigid jaws touch neither side, so it takes the sponge no way either.  Which way the
#: distractor errs is drawn from the seed (``decoy_kind``), so "take the bigger one" is right only for half the
#: scenes.  Either distractor can only push the sponge, and a pushed sponge does not get up onto the riser.  The
#: handles are alike, the mops weigh alike; only the clip opening differs, including the plate width that spans it.
MOP_OPEN, MOP_NARROW, MOP_WIDE = 0.058, 0.036, 0.110
JAW, PLATE_T, NECK = (0.07, 0.005, 0.025), 0.006, (0.02, 0.02, 0.025)
MOP_HANDLE, HANDLE_T, MOP_D, MOP_PITCH, MOP_KG, MOP_REST = 0.42, 0.022, 0.33, 60.0, 0.15, (0.04, 0.04)
#: how it looks (a long-handled sponge brush): a beech
#: handle with a green plastic end cap HANDLE_END long (the hand takes the handle MOP_D from the clip, on the wood), a
#: green plastic head -- the plate over the jaws and the neck -- and a row of white bristles along the outside of each
#: jaw, BRISTLE (thickness, height) down from the plate's underside and up beside the plate to its top, so that they show
#: from above (under a plate as wide as they are, the head would read as a white block: a mallet).  Their ends are 2.9 cm
#: over the sponge's bottom, higher than the jaws' (2 cm), so nothing new comes nearer the counter, the riser or a pad;
#: across, they widen the head by BRISTLE[0] on each side (the prop's 6.8 cm across the jaws' outsides to 8.4 cm)
HANDLE_END, BRISTLE = 0.05, (0.008, 0.016)
#: where the mops lie: the clip's middle MOP_R from the west arm's base, on these bearings (which is which drawn from
#: the seed), the handle toward the base; where the sponge is left for the east arm (0.90 m from the west arm's base,
#: 0.50 m from the east arm's, as pad B is); how far over the riser's top the sponge is carried
MOP_R, MOP_BEARINGS, HANDOFF, OVER_RIM = 0.84, (-149.0, 149.0), (0.20, -0.05), 0.03


def radial_deg(slot: str, xy) -> float:
    m = mount(slot)
    return math.degrees(math.atan2(xy[1] - m.y, xy[0] - m.x))


def sponge_at(xy, z_bottom: float, yaw: float):
    return (xy[0], xy[1], z_bottom + SPONGE[2] / 2, *yaw_quat(yaw))


def riser(xy):
    """The riser under the middle spill, centred at ``xy``: a static block, its top RISER_H over the counter."""
    return box("riser", xy, (2 * RISER_HALF, 2 * RISER_HALF, RISER_H), "white", dynamic=False)


def floor_z(slot: str) -> float:
    """The height of what ``slot``'s spill lies on: the counter, or the riser's top for the middle one."""
    return Z + (RISER_H if slot == "south" else 0.0)


def decoy_kind(level: str, seed: int) -> str:
    """Which way the borrow version's distractor mop errs: its clip too narrow or too wide for the sponge -- by the
    seed's parity, shifted at the hard level, so each level's seeds 0-4 have both and seed 0 differs between levels."""
    return ("narrow", "wide")[(seed + (level == "hard")) % 2]


def mop(name: str, opening: float, bearing: float, jaw_len: float = JAW[0]):
    """A sponge mop (a long-handled sponge brush: nylon jaws under a green head with white bristles along its sides, a beech
    handle) lying by the west arm, its clip's jaws ``opening`` apart and ``jaw_len`` long, its handle toward the arm's base on
    ``bearing``.  Its frame: the middle of the clip, where a sponge it holds has its middle; x along
    the handle, from its end to the clip; z up.  Lying, it stands on its jaws and its handle's end rests on a rest
    (``mop_rest``)."""
    top = SPONGE[2] / 2  # the jaws' top and the plate's underside: the top of a sponge held
    jaw = (jaw_len, *JAW[1:])
    head = opening + 2 * JAW[1]  # across the jaws' outsides
    parts = (part("jaw_1", "box", jaw, (0.0, -(opening + JAW[1]) / 2, top - JAW[2] / 2), color="nylon"),
             part("jaw_2", "box", jaw, (0.0, (opening + JAW[1]) / 2, top - JAW[2] / 2), color="nylon"),
             *(part(f"bristles_{k}", "box", (jaw_len, BRISTLE[0], BRISTLE[1] + PLATE_T),
                    (0.0, s * (head + BRISTLE[0]) / 2, top + (PLATE_T - BRISTLE[1]) / 2), color="white")
               for k, s in ((1, -1), (2, 1))),
             part("plate", "box", (jaw_len, head, PLATE_T), (0.0, 0.0, top + PLATE_T / 2), color="plastic_green"),
             part("neck", "box", NECK, (0.0, 0.0, top + PLATE_T + NECK[2] / 2), color="plastic_green"),
             part("handle", "box", (MOP_HANDLE - HANDLE_END, HANDLE_T, HANDLE_T),
                  (NECK[0] / 2 - (MOP_HANDLE - HANDLE_END) / 2, 0.0, _handle_z()), color="wood"),
             part("handle_end", "box", (HANDLE_END, HANDLE_T, HANDLE_T),
                  (NECK[0] / 2 - MOP_HANDLE + HANDLE_END / 2, 0.0, _handle_z()), color="plastic_green"))
    m = mount("west")
    b = math.radians(bearing)
    xy = (m.x + MOP_R * math.cos(b), m.y + MOP_R * math.sin(b))
    return compound(name, (*xy, Z + JAW[2] - top), parts, "aluminium", mass=MOP_KG, yaw=bearing)


def _mop_at(xy, bottom: float):
    """The mop's pose -- and the sponge's, in its clip -- with the sponge's bottom ``bottom`` high over ``xy``, the
    handle toward the west arm's base."""
    return sponge_at(xy, bottom, radial_deg("west", xy))


def _handle_z() -> float:
    """The handle's middle over the clip's middle (in the mop's frame)."""
    return SPONGE[2] / 2 + PLATE_T + NECK[2] + HANDLE_T / 2


def mop_rest(name: str, m):
    """The rest under mop ``m``'s handle's end, fixed to the counter."""
    h = m.pos[2] + _handle_z() - HANDLE_T / 2 - Z
    x = NECK[0] / 2 - MOP_HANDLE + MOP_REST[0] / 2
    c, s = math.cos(math.radians(m.yaw_deg)), math.sin(math.radians(m.yaw_deg))
    return box(name, (m.pos[0] + c * x, m.pos[1] + s * x), (*MOP_REST, h), "steel", yaw=m.yaw_deg, dynamic=False)


def build(level: str, seed: int, variant: str = "exact"):
    spec = get("Rea-04")
    rng = rng_for(spec, level, seed)
    wipe_s = float(spec.knob.levels[level])
    spills = []
    for slot, xy in SPILLS.items():
        at_xy = jitter(rng, xy, 0.01)
        spills.append(cylinder(f"spill_{slot}", at_xy, RISER_HALF if slot == "south" else SPILL_R, SPILL_T,
                               "coffee", dynamic=False, z=floor_z(slot) + SPILL_T / 2))
    stand = riser(spills[1].pos[:2])
    dish = box("dish", DISH_XY, DISH, "white", dynamic=False)
    # lengthwise along the west arm's reach: its fingers close across the sponge's width
    sponge = box("sponge", DISH_XY, SPONGE, "yellow", mass=0.02, z=Z + DISH[2] + SPONGE[2] / 2,
                 yaw=radial_deg("west", DISH_XY))
    sink, sink_in = container("sink", SINK_XY, SINK_IN, wall_h=SINK_WALL, color="grey")
    regions = [pad(f"pad_{k}", xy, (0.06, 0.05), "cork") for k, xy in PADS.items()]
    regions.append(sink_in)
    for s in spills:  # the sponge down on the spill, its middle over it
        regions.append(RegionSpec(f"wipe_{s.name[6:]}", (s.pos[0], s.pos[1], floor_z(s.name[6:]) + SPILL_T + SPONGE[2] / 2),
                                  (WIPE_R, WIPE_R, PRESSED), visible=False))
    mops, extra = [], {}
    if variant == "borrow":  # no south arm: a mop whose clip holds the sponge, and one whose clip is too narrow or too wide
        kind = decoy_kind(level, seed)
        openings = [MOP_OPEN, MOP_NARROW if kind == "narrow" else MOP_WIDE]
        # which lies where is drawn from the seed, on its own stream (the exact configuration's scene is unchanged)
        random.Random(f"borrow/{spec.seed_id}/{level}/{seed}").shuffle(openings)
        mops = [mop(f"mop_{k + 1}", w, b)
                for k, (w, b) in enumerate(zip(openings, MOP_BEARINGS))]
        mops += [mop_rest(f"rest_{k + 1}", m) for k, m in enumerate(mops)]
        prop = f"mop_{openings.index(MOP_OPEN) + 1}"
        extra = {"borrow": {"props": [prop], "decoys": [n for n in ("mop_1", "mop_2") if n != prop],
                            "decoy_kind": kind}}
    sc = scene(spec, [*spills, stand, dish, sponge, *sink, *mops], regions)
    if variant == "borrow":
        sc = replace(sc, arms=tuple(a for a in sc.arms if a.slot != "south"))
    reach = [("west", "sponge"), ("west", "spill_west"), ("west", "pad_a"), ("south", "pad_a"),
             ("south", "spill_south"), ("south", "pad_b"), ("east", "pad_b"), ("east", "spill_east"),
             ("east", "sink_in")]
    # (the middle wipe's place: with --slack 0.085, the reach of a hand holding the sponge by its end)
    out_of_reach = [("south", "sponge"), ("east", "sponge"), ("south", "spill_west"), ("east", "spill_west"),
                    ("west", "spill_south"), ("east", "spill_south"), ("west", "wipe_south"), ("east", "wipe_south"),
                    ("west", "spill_east"), ("south", "spill_east"), ("south", "sink_in"), ("west", "sink_in")]
    if variant == "borrow":  # (no claims for the mops: what the west arm does with one, its plan checks)
        reach = [r for r in reach if r[0] != "south"]
        out_of_reach = [r for r in out_of_reach if r[0] != "south"]

    wiped = {slot: HoldsFor(InRegion("sponge", f"wipe_{slot}"), wipe_s) for slot in SPILLS}
    in_sink = HoldsFor(InRegion("sponge", "sink_in"), 1.0)
    return make_instance(
        spec, level, seed, sc,
        instruction=(f"Wipe each of the three coffee spills on the counter with the sponge, keeping it down on each "
                     f"spill for {wipe_s:g} s, then drop the sponge in the sink at the east end."),
        # the sponge in the sink when the run ends, through its last second at least; the milestone keeps the second
        # in the sink at any time
        success=all_temporal(*wiped.values(), HoldsAtEnd(InRegion("sponge", "sink_in"), 1.0)),
        milestones={"west_spill_wiped": wiped["west"], "middle_spill_wiped": wiped["south"],
                    "east_spill_wiped": wiped["east"], "sponge_in_sink": in_sink},
        reach=reach, out_of_reach=out_of_reach,
        # (no "apart": the end PiPERs both reach the table's middle; the middle work, which only the south arm
        # reaches, is what needs a third arm)
        wipe_s=wipe_s, paint={"sponge": "yellow"}, **extra,
    )


def _wiping(inst, slot):
    """The sponge down on ``slot``'s spill, lengthwise along the reach of the arm that holds it."""
    s = inst.scene.obj(f"spill_{slot}").pos
    holder = {"west": "west", "south": "south", "east": "east"}[slot]
    yaw = radial_deg(holder, s[:2]) if inst.scene.arm(holder).model == "piper" else 0.0
    return sponge_at(s[:2], floor_z(slot) + SPILL_T, yaw)


def _on_pad(k: str, yaw: float):
    return sponge_at(PADS[k], Z, yaw)


#: the keyframe `mrbench render` shows between start and goal: the middle spill being wiped
MIDDLE = 4


def keyframes(inst):
    nudge = lambda p, d: (p[0] + d, *p[1:])  # noqa: E731 -- still on the spill
    sink = (*SINK_XY, Z + 0.01 + SPONGE[2] / 2, *yaw_quat(0.0))
    if inst.variant == "borrow":  # the middle spill wiped with the sponge in the mop's clip, left at HANDOFF
        w, e = _wiping(inst, "west"), _wiping(inst, "east")
        mp = inst.scene.obj(inst.metadata["borrow"]["props"][0])
        m = _mop_at(inst.scene.obj("spill_south").pos[:2], floor_z("south") + SPILL_T)
        off = _mop_at(HANDOFF, Z)
        return [{"sponge": w}, {"sponge": nudge(w, 0.01)}, {"sponge": _on_pad("a", 0.0)}, {"sponge": m, mp.name: m},
                {"sponge": nudge(m, 0.01), mp.name: nudge(m, 0.01)}, {"sponge": off, mp.name: mp.pose},
                {"sponge": e}, {"sponge": nudge(e, 0.01)}, {"sponge": sink}]
    w, m, e = (_wiping(inst, s) for s in ("west", "south", "east"))
    return [{"sponge": w}, {"sponge": nudge(w, 0.01)}, {"sponge": _on_pad("a", 0.0)}, {"sponge": m},
            {"sponge": nudge(m, 0.01)}, {"sponge": _on_pad("b", 0.0)}, {"sponge": e}, {"sponge": nudge(e, 0.01)},
            {"sponge": sink}]


#: the scripted reference: how far below level a hand points, by arm (the south Piper 60: it takes the sponge 0.51 m
#: out and wipes 0.19 m out with the same hold); how far apart the fingers close on the
#: sponge (across its width) and are as they come; how far below its top the hand takes it (1.2 cm down, a tilted
#: hand's lower finger comes within 8 mm of the table); how high it is carried;
#: the scrub (a figure of eight on the spill: how far each way)
PITCH = {"west": 75.0, "south": 60.0, "east": 75.0}
GRIP_W, SHAPE_W, TAKE_DOWN, CARRY_UP, SCRUB = SPONGE[1] - 0.002, SPONGE[1] + 0.008, 0.006, 0.08, (0.012, 0.008)
#: how far below resting on the middle spill the south arm holds the sponge as it scrubs: taking it off pad A, 0.51 m
#: out, its hand sags 1.3 cm and holds it that much higher than planned (the end arms take it nearer: 5 mm over)
PRESS = -0.002
#: where the south arm's hand is before it comes down onto the middle spill (dx, dy, dz from wiping it): straight over
#: it, 5 cm up (the riser's top already 6 cm over the counter); and how high it carries the sponge between the pads and the riser (from where it
#: takes or leaves it: its bottom 4 cm over the riser's top)
SOUTH_OVER, SOUTH_CARRY = (0.0, 0.0, 0.05), RISER_H + 0.04


def _hold(inst, slot, sponge):
    """The tool pose that holds the sponge at ``sponge``: from above, by its top, fingers across its width."""
    arm = inst.scene.arm(slot)
    yaw = 2 * math.atan2(sponge[6], sponge[3])  # it only ever turns about the vertical
    across = (-math.sin(yaw), math.cos(yaw), 0.0)
    take_down = TAKE_DOWN - .010 if inst.variant == "borrow" else TAKE_DOWN
    top = (sponge[0], sponge[1], sponge[2] + SPONGE[2] / 2 - take_down)
    pitch = 65. if inst.variant == "borrow" and slot == "east" else PITCH[slot]
    return (*top, *radial_tool(arm.pos, top, pitch, across))


def _scrub(pose, seconds_per_loop=1.0):
    def path(s):
        return (pose[0] + SCRUB[0] * math.sin(2 * math.pi * 2 * s), pose[1] + SCRUB[1] * math.sin(2 * math.pi * 4 * s),
                *pose[2:])
    return path


def _leg(inst, slot, start, scrub_at, end, end_note, release_above=0.0, over=(0.0, 0.0, CARRY_UP), press=0.0005,
         lift=CARRY_UP):
    """``slot`` takes the sponge at ``start``, wipes its spill at ``scrub_at`` and leaves it at ``end``; it comes down
    onto the spill from ``over`` it (dx, dy, dz), and carries it ``lift`` over where it takes and leaves it."""
    take = _hold(inst, slot, start)
    wipe = carry(take, start, scrub_at)
    put = carry(take, start, end)
    above = offset(wipe, *over)
    loops = max(2, math.ceil(inst.metadata["wipe_s"] + 0.6))
    return [
        phase(2.5, f"the {slot} arm takes the sponge", Move(slot, back_off(take, 0.08), joints=True),
              Open(slot, SHAPE_W)),
        phase(1.2, "", Move(slot, take)),
        phase(0.4, "", Grip(slot, "sponge", GRIP_W)),
        phase(1.0, "", Move(slot, offset(take, dz=lift))),
        phase(2.5, "and wipes the spill it reaches", Move(slot, above, joints=True)),
        phase(1.2, "", Move(slot, offset(wipe, dz=press))),
        phase(float(loops), "", Move(slot, _scrub(offset(wipe, dz=press)))),
        phase(1.0, "", Move(slot, above)),
        phase(2.5, end_note, Move(slot, offset(put, dz=lift), joints=True)),
        phase(1.2, "", Move(slot, offset(put, dz=0.004 + release_above))),
        phase(0.4, "", Release(slot)),
        phase(1.0, "", Move(slot, offset(put, dz=lift))),
    ]


def script(inst):
    """Privileged: a grip fixes the sponge to the hand.  The west arm takes it off its dish, wipes the west spill and
    puts it down on pad A; the south arm takes it from there, wipes the middle spill and puts it on pad B; the east
    arm wipes the east spill and drops it in the sink.  A Piper holds the sponge lengthwise along its reach."""
    from ...policies.script import home_pose

    if inst.variant == "borrow":
        return _borrow_script(inst)
    start = inst.scene.obj("sponge").pose
    w, m, e = (_wiping(inst, s) for s in ("west", "south", "east"))
    # put down where the next arm takes it: for a Piper, lengthwise along its reach
    on_a = _on_pad("a", math.degrees(2 * math.atan2(w[6], w[3])) + radial_deg("west", PADS["a"]) - radial_deg("west", w[:2]))
    # the south Piper turns the sponge with its reach (its hand pointing out from its base wherever it holds it: held
    # along its reach at the spill, as taken off pad A, the hand points 21 degrees off and cannot get there); the
    # east arm takes it as it lies on pad B
    m = turned_with("south", m, on_a)
    on_b = turned_with("south", _on_pad("b", 0.0), on_a)
    sink = carry_yaw((*SINK_XY, Z + 0.01 + SPONGE[2] / 2 + 0.02, *on_b[3:]), on_b, inst=inst)
    out = _leg(inst, "west", start, w, on_a, "and puts it down where the south arm reaches")
    out += [phase(2.0, "", Move("west", home_pose(inst, "west"), joints=True))]
    out += _leg(inst, "south", on_a, m, on_b, "and puts it down where the east arm reaches", over=SOUTH_OVER,
                press=PRESS, lift=SOUTH_CARRY)
    out += [phase(2.0, "", Move("south", home_pose(inst, "south"), joints=True))]
    out += _leg(inst, "east", on_b, carry_yaw(e, on_b, inst=inst), sink, "and drops it in the sink", 0.02)
    out += [phase(2.0, "done")]
    return out


def _mop_grasp(mp):
    """The west hand on mop ``mp``'s handle (an ObjectSpec, where it lies): MOP_D from the clip's middle, pointing
    MOP_PITCH down along the handle toward the clip, its fingers across the handle."""
    p = math.radians(MOP_PITCH)
    local = (-MOP_D, 0.0, _handle_z(), *tool_quat((math.cos(p), 0.0, -math.sin(p)), (0.0, 1.0, 0.0)))
    return carry(local, (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0), mp.pose)


def _borrow_script(inst):
    """The borrow configuration (no south arm), privileged as the exact one -- and the mop's clip as a hand's grip: pressed
    round the sponge, it fixes it to the mop, if the sponge fits between its jaws.  The west arm wipes the west spill
    and puts the sponge on pad A as in the exact version; it then takes the mop by its handle, presses the clip round
    the sponge, lifts it onto the riser's top over the middle spill, scrubs it, leaves the sponge at HANDOFF and
    lays the mop back on its rest; the east arm takes the sponge from there, wipes the east spill and drops it in the
    sink.  With the distractor (``--decoy``) the clip takes nothing (its jaws too narrow to go round the sponge, or too
    wide to touch it): the sponge stays on pad A."""
    from ...policies.script import Clip, Unclip, home_pose

    sc = inst.scene
    mp = sc.obj(inst.metadata["borrow"]["props"][0])
    start = sc.obj("sponge").pose
    w, e = _wiping(inst, "west"), _wiping(inst, "east")
    on_a = _on_pad("a", math.degrees(2 * math.atan2(w[6], w[3])) + radial_deg("west", PADS["a"]) -
                   radial_deg("west", w[:2]))
    grasp = _mop_grasp(mp)
    hand = lambda at: carry(grasp, mp.pose, at)  # noqa: E731 -- the hand holding the mop, the mop at ``at``
    clip_at = offset(_mop_at(PADS["a"], Z), dz=0.002)  # the plate just over the sponge's top
    wipe_at = _mop_at(sc.obj("spill_south").pos[:2], floor_z("south") + SPILL_T)
    off = _mop_at(HANDOFF, Z)
    lift = RISER_H + OVER_RIM  # the sponge's bottom this high over the counter: over the riser's top
    loops = max(2, math.ceil(inst.metadata["wipe_s"] + 0.6))
    out = _leg(inst, "west", start, w, on_a, "and puts it down on pad A")
    out += [
        phase(2.0, "the west arm takes the mop", Move("west", back_off(grasp, 0.06), joints=True), Open("west", 0.05)),
        phase(1.0, "", Move("west", grasp)),
        phase(0.4, "", Grip("west", mp.name, HANDLE_T - 0.002)),
        phase(1.0, "", Move("west", hand(offset(mp.pose, dz=0.05)))),
        phase(2.5, "presses its clip round the sponge", Move("west", hand(offset(clip_at, dz=0.06)), joints=True)),
        phase(1.2, "", Move("west", hand(clip_at))),
        phase(0.4, "", Clip(mp.name, "sponge")),
        phase(1.0, "", Move("west", hand(offset(clip_at, dz=lift)))),
        phase(2.5, "and wipes the middle spill with the sponge",
              Move("west", hand(offset(wipe_at, dz=OVER_RIM)), joints=True)),
        phase(1.2, "", Move("west", hand(wipe_at))),
        phase(float(loops), "", Move("west", _scrub(hand(wipe_at)))),
        phase(1.0, "", Move("west", hand(offset(wipe_at, dz=OVER_RIM)))),
        phase(2.5, "and leaves the sponge where the east arm reaches",
              Move("west", hand(offset(off, dz=lift)), joints=True)),
        phase(1.2, "", Move("west", hand(offset(off, dz=0.004)))),
        phase(0.4, "", Unclip(mp.name)),
        phase(1.0, "", Move("west", hand(offset(off, dz=lift)))),
        phase(2.5, "and lays the mop back", Move("west", hand(offset(mp.pose, dz=0.05)), joints=True)),
        phase(1.0, "", Move("west", hand(offset(mp.pose, dz=0.002)))),
        phase(0.4, "", Release("west")),
        phase(1.0, "", Move("west", back_off(grasp, 0.06))),
        phase(2.0, "", Move("west", home_pose(inst, "west"), joints=True)),
    ]
    sink = carry_yaw((*SINK_XY, Z + 0.01 + SPONGE[2] / 2 + 0.02, *off[3:]), off, inst=inst)
    out += _leg(inst, "east", off, carry_yaw(e, off, inst=inst), sink, "and drops it in the sink", 0.02)
    out += [phase(2.0, "done")]
    return out


def turned_with(slot, pose, held_from):
    """``pose`` with the sponge turned as a Piper at ``slot`` holding it from ``held_from`` turns it: its hand pointing
    out from its base both times."""
    yaw = math.degrees(2 * math.atan2(held_from[6], held_from[3])) + radial_deg(slot, pose[:2]) - \
        radial_deg(slot, held_from[:2])
    return (*pose[:3], *yaw_quat(yaw))


def carry_yaw(pose, held_from, inst=None):
    """``pose`` with the sponge turned as it is held from ``held_from`` (a Franka keeps its hand's turn; a Piper's hand,
    pointing out from its base, turns with its reach -- ``inst`` given: the east Piper)."""
    if inst is None:
        return (*pose[:3], *held_from[3:])
    yaw = math.degrees(2 * math.atan2(held_from[6], held_from[3])) + radial_deg("east", pose[:2]) - \
        radial_deg("east", held_from[:2])
    return (*pose[:3], *yaw_quat(yaw))


def jobs(inst):
    """What the task asks (``tasks/jobs.py``): the one sponge brought from its dish down onto each spill in turn (a
    wipe) and then into the sink -- each leg by one arm, or passed between two where they meet.  A leg starts where
    the last one ended: no one reaching a spill, neither the leg onto it nor the leg from it has a doer.

    In the borrow version the prop mop does the two legs through the middle spill (onto it, and on from it toward the
    east spill): put to use by an arm that sets the sponge down on pad A, one hand that holds the mop by its handle
    throughout -- taking it off its rest, pressing the clip round the sponge on pad A, wiping the middle spill with the
    sponge in the clip, leaving the sponge at HANDOFF (and laying the mop back) -- and an arm that takes the sponge from
    HANDOFF onto the east spill.  The distractor's clip takes the sponge no way."""
    from ..jobs import Job, Jobs, Thing

    margin = 0.085  # the reach audit's (`validate --ik --slack 0.085 Rea-04`): a hand holding the sponge by its end
    legs = (("wipe the west spill with the sponge from its dish", "sponge", "wipe_west"),
            ("bring the sponge on from the west spill and wipe the middle one", "wipe_west", "wipe_south"),
            ("bring the sponge on from the middle spill and wipe the east one", "wipe_south", "wipe_east"),
            ("bring the sponge on from the east spill into the sink", "wipe_east", "sink_in"))
    carry_legs = tuple(Job(name, "carry", sites=(a, b), slack=margin, body="sponge") for name, a, b in legs)
    if inst.variant != "borrow":
        return Jobs(carry_legs)
    b = inst.metadata["borrow"]
    prop, decoy = b["props"][0], b["decoys"][0]
    mp = inst.scene.obj(prop)
    grasp = _mop_grasp(mp)
    hand = lambda at: tuple(carry(grasp, mp.pose, at)[:3])  # noqa: E731 -- the hand on the mop, the mop at ``at``
    clip_at = offset(_mop_at(PADS["a"], Z), dz=0.002)
    wipe_at = _mop_at(inst.scene.obj("spill_south").pos[:2], floor_z("south") + SPILL_T)
    off = _mop_at(HANDOFF, Z)
    placing = (
        Job("bring the sponge on from the west spill to pad A", "carry", sites=("wipe_west", "pad_a"), slack=margin,
            body="sponge"),
        # taken off its rest by any part of it a hand reaches (the reference holds it 60 degrees down along the
        # handle, and plays at both levels)
        Job(f"take {prop} off its rest", sites=(prop,), slack=0.05),
        Job(f"hold {prop} by its handle: its clip round the sponge on pad A, the sponge on the middle spill, left "
            "where the east arm reaches", sites=(hand(clip_at), hand(wipe_at), hand(off))),
        Job("bring the sponge on from where the mop left it and wipe the east spill", "carry",
            sites=((*HANDOFF, Z + SPONGE[2] / 2), "wipe_east"), slack=margin, body="sponge"),
    )
    kind = b.get("decoy_kind", "narrow")
    things = (
        Thing(prop, does=(legs[1][0], legs[2][0]), placing=placing,
              why=f"its clip opens {MOP_OPEN * 100:.1f} cm round the {SPONGE[1] * 100:.0f} cm sponge and holds it: "
                  f"held by the handle {MOP_D * 100:.0f} cm back, it sets the sponge down on the middle spill, which "
                  "no end arm's hand reaches"),
        Thing(decoy, why=(f"its clip opens {MOP_NARROW * 100:.1f} cm, narrower than the sponge any way round: it "
                          "takes it no way, and a pushed sponge does not get up onto the riser")
              if kind == "narrow" else
              (f"its clip opens {MOP_WIDE * 100:.0f} cm, wider than the sponge any way round by more than a clip's "
               "slack: its jaws touch neither side, and a pushed sponge does not get up onto the riser")),
    )
    return Jobs(carry_legs, things)
