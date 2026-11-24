"""Scripted routes for gold teams other than a task's reference team, played in Isaac (docs/TASKS.md):

* ``Con-02-surplus`` -- west + north + east: the north arm holds the orange leaf, the east arm slides the pin, the
  south arm idle;
* ``Con-02-surplus-nsw`` -- north + south + west: the north arm holds the orange leaf, the east arm idle;
* ``Rea-02-swapped`` -- west + east (the east arm a Franka): the east arm lays the lowest and the middle channel;
* ``Rea-02-surplus-nsw`` -- north + south + west: the north arm brings the lowest channel to the table's middle, the
  south arm lays it and then the middle one, the east arm idle;
* ``Rea-01-surplus-nw`` -- north + west: the west arm hands each piece to the north arm in the air west of the tall
  partition, the north arm carries it over the partition into the crate, the south and the east arm idle;
* ``Rea-04-swapped`` -- west + east (the east arm a Franka): the west arm wipes the west spill and puts the sponge down
  on the table's middle, the east arm takes it by its end and wipes the middle and the east spill and drops it in the
  sink, the south arm idle;
* ``Con-04-surplus-nsw`` / ``Con-04-surplus-nes`` -- north + south + west / north + east + south: the north arm holds
  the east / the west bleed valve shut in place of that end arm, which stays idle at home; ``Con-04-surplus-enw`` --
  east + north + west: the north arm holds the west valve and the west arm the foot valve's pedal, the south arm idle;
* ``Con-04-borrow-decoyvalve`` -- not a gold team but a route that must fail: in the borrow scene (west + east), the
  end arm on the 1.2 kg distractor's side sets it on its own bleed valve's grip, too light to hold the valve shut, and
  then holds the pedal, the other end arm its own valve;
* ``Con-04-shortfall-attempt`` -- not a gold team but a route that must fail: west + east, the west hand on the pedal
  first and then on its own valve.

Each scene is the agent case's (``agent.cases.make_case``: the north arm, the swapped robot where the case puts
them); the task's scene, success formula and exact reference are not touched -- a route here is played in place of
the reference by handing it to ``demo.play`` for this process only.

    python -m mrbench.agent.gold_routes check Con-02-surplus [--levels easy hard] [--seeds 0 1 2]   # no Isaac
    python -m mrbench.agent.gold_routes run Con-02-surplus -o runs/gold [--levels ...] [--seeds ...]  # Isaac
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from mrbench.agent.cases import make_case
from mrbench.engines.base import quat_mul, transform
from mrbench.policies.script import (Grip, Move, Open, Release, back_off, home_pose, offset, phase, tool_quat)

ROUTES: dict = {}


def route(case_id: str):
    def put(fn):
        ROUTES[case_id] = fn
        return fn
    return put


# ---------------------------------------------------------------------------------------------------- Con-02 surplus
#: the route's choices (each found by planning and walking it for clearance, ``check``):
#: * the east hand on the pin's head points 75 degrees down toward the west (``approach``), its fingers closing nearly
#:   along x (``across``), across the head: so the hand is narrow across the way the pin goes in (fingers closing along
#:   y, as a hand pointing away from its base has them, brought the forearm and wrist 0.6-0.9 cm from the guard as the
#:   pin went home); it comes down onto the head from 8 cm over it, its tool point 2 mm over the head's top
#:   (``take_dz``: lower, its lower finger came within 1 cm of the table), its fingers 5.8 cm open;
#: * the idle south arm first moves its hand down and west, to ``south_away``: at home it was 1-3 cm from the east hand
#:   coming over the pin and carrying it in line;
#: * the north hand comes down onto the orange leaf's bar from over it, leaning 20 degrees out toward its hinge
#:   (``north_lean``: with the reference's 30, its wrist camera came 8-9 mm from the bar)
C02 = {"approach": (-0.259, 0.0, -0.966), "across": (-0.966, 0.0, 0.259), "turn": None, "north_turn": None,
       "north_in": "above", "south_away": (-0.30, -0.45, 1.05), "lift": 0.10, "east_via": None, "take_dz": 0.012,
       "north_lean": 20.0, "north_hold": None, "east_open": 0.058, "east_in": "above", "upto": None, "east_grip": 0.046}


def _norm(v):
    n = math.sqrt(sum(x * x for x in v))
    return tuple(x / n for x in v)


@route("Con-02-surplus")
def con02_surplus(inst, p=None):
    """West + north + east, the south arm idle (it only moves out of the way): the west arm holds the blue leaf as in
    the reference, the north arm the orange leaf (the reference's east moves), the east arm slides the pin (the
    reference's south moves, its hand narrow across the way the pin goes in).  ``p``: changes to ``C02``."""
    from mrbench.engines.base import axis_quat
    from mrbench.tasks.impl import con02_hinged_leaves as m
    from mrbench.tasks.impl._common import rotate_about

    p = {**C02, **(p or {})}
    sc = inst.scene

    def leaf_grip(name, pose):
        h = next(h for h in sc.hinges if h.body == name)
        s = 1 if name == "leaf_w" else -1
        hold = p["north_hold"] if s < 0 and p["north_hold"] is not None else m.HOLD_AT
        point = transform(pose, (s * hold, 0.0, m.BAR / 2 - 0.012))
        lean = math.radians(p["north_lean"] if s < 0 and p["north_lean"] is not None else m.LEAN_DEG)
        way = (s * math.sin(lean), 0.0, -math.cos(lean))
        return (*point, *quat_mul(pose[3:], tool_quat(way, (0.0, s, 0.0)))), h

    def arc(name, a0, a1):
        g, h = leaf_grip(name, sc.obj(name).pose)
        return lambda u: rotate_about(g, h.pivot, axis_quat(h.axis, h.sign * (a0 + (a1 - a0) * u)))

    gw, _ = leaf_grip("leaf_w", sc.obj("leaf_w").pose)
    ge, _ = leaf_grip("leaf_e", sc.obj("leaf_e").pose)
    pin = sc.obj("pin")
    head = transform(pin.pose, pin.part("head").pos)
    q = tool_quat(_norm(p["approach"]), _norm(p["across"]))
    take = (head[0], head[1], head[2] + m.HEAD[2] / 2 - 0.010 + p["take_dz"], *q)
    to_tool = [t - c for t, c in zip(take[:3], pin.pos)]
    wall = m.JOINT[1] - m.GUARD[1] - m.WALL
    start_y = wall - m.PIN_START - m.KEY[1] / 2
    lined = (m.JOINT[0] + to_tool[0], start_y + to_tool[1], m.JOINT[2] + to_tool[2], *take[3:])
    home_ = offset(lined, dy=m.PIN_HOME_Y - start_y)
    w, n, e, s = "west", "north", "east", "south"
    t = p["turn"]
    out = []
    if p["south_away"] is not None:  # the idle south arm first out of the east hand's way
        away = (*p["south_away"], *tool_quat((0.0, 0.0, -1.0), (1.0, 0.0, 0.0)))
        out.append(phase(2.0, "the idle south arm moves out of the way", Move(s, away, joints=True)))
    if p["north_in"] == "reference":
        n_in = [Move(n, offset(back_off(ge, 0.10), dx=0.10), turn=p["north_turn"]),
                Move(n, back_off(ge, 0.10), turn=p["north_turn"])]
    elif p["north_in"] == "above":  # down onto the bar from over it
        n_in = [Move(n, offset(ge, dx=0.05, dz=0.16), turn=p["north_turn"]),
                Move(n, offset(ge, dz=0.08), turn=p["north_turn"])]
    else:  # back along its pointing, d1 and then d2
        d1, d2 = p["north_in"]
        n_in = [Move(n, back_off(ge, d1), turn=p["north_turn"]), Move(n, back_off(ge, d2), turn=p["north_turn"])]
    out += [
        phase(1.5, "the west and the north arm reach out", Move(w, offset(back_off(gw, 0.10), dx=-0.10)), n_in[0],
              Open(w, 0.07), Open(n, 0.07)),
        phase(1.0, "and come over their leaves", Move(w, back_off(gw, 0.10)), n_in[1]),
        phase(1.2, "", Move(w, gw), Move(n, ge, turn=p["north_turn"])),
        phase(0.8, "they settle"),
        phase(0.4, "and take them", Grip(w, "leaf_w", m.BAR_GRIP), Grip(n, "leaf_e", m.BAR_GRIP)),
        phase(3.0, "they turn them level: the west one up, the orange one down against its spring",
              Move(w, arc("leaf_w", 0.0, m.REST_DEG)), Move(n, arc("leaf_e", 0.0, m.REST_DEG))),
    ]
    if p["east_via"] is not None:
        out.append(phase(2.0, "the east arm comes round", Move(e, p["east_via"], joints=True)))
    out += [
        phase(2.5, "the east arm comes over the pin",
              Move(e, offset(take, dz=0.08) if p["east_in"] == "above" else back_off(take, 0.10), joints=True, turn=t),
              Open(e, p["east_open"])),
        phase(1.2, "", Move(e, take, turn=t)),
        phase(0.8, ""),
        phase(0.4, "takes it by its head", Grip(e, "pin", p["east_grip"] or m.HEAD_GRIP)),
        phase(1.5, "lifts it", Move(e, offset(take, dz=p["lift"]), turn=t)),
        phase(3.0, "brings it in line with the opening", Move(e, offset(lined, dy=-0.04), turn=t)),
        phase(1.0, "", Move(e, lined, turn=t)),
        phase(1.0, "lines it up"),
        phase(5.0, "and slides it through all three slots", Move(e, home_, turn=t)),
        phase(0.4, "lets go of it", Release(e)),
        phase(1.5, "and backs off", Move(e, offset(home_, dy=-0.06, dz=0.06), turn=t)),
        phase(3.0, "the leaves are pinned"),
    ]
    return out[:p["upto"]] if p["upto"] else out


@route("Con-02-surplus-nsw")
def con02_surplus_nsw(inst, p=None):
    """North + south + west, the east arm idle at home: the west arm holds the blue leaf and the south arm slides the
    pin as in the reference; the north arm holds the orange leaf, coming down onto its bar from over it as in
    ``con02_surplus``."""
    from mrbench.engines.base import axis_quat
    from mrbench.tasks.impl import con02_hinged_leaves as m
    from mrbench.tasks.impl._common import rotate_about

    p = {**C02, **(p or {})}
    sc = inst.scene

    def leaf_grip(name, pose):
        h = next(h for h in sc.hinges if h.body == name)
        s = 1 if name == "leaf_w" else -1
        point = transform(pose, (s * m.HOLD_AT, 0.0, m.BAR / 2 - 0.012))
        lean = math.radians(p["north_lean"] if s < 0 else m.LEAN_DEG)
        way = (s * math.sin(lean), 0.0, -math.cos(lean))
        return (*point, *quat_mul(pose[3:], tool_quat(way, (0.0, s, 0.0)))), h

    def arc(name, a0, a1):
        g, h = leaf_grip(name, sc.obj(name).pose)
        return lambda u: rotate_about(g, h.pivot, axis_quat(h.axis, h.sign * (a0 + (a1 - a0) * u)))

    gw, _ = leaf_grip("leaf_w", sc.obj("leaf_w").pose)
    ge, _ = leaf_grip("leaf_e", sc.obj("leaf_e").pose)
    pin = sc.obj("pin")
    head = transform(pin.pose, pin.part("head").pos)
    lean = math.radians(m.PIN_LEAN_DEG)  # the reference's south hand on the pin's head
    take = (head[0], head[1], head[2] + m.HEAD[2] / 2 - 0.010,
            *tool_quat((0.0, math.sin(lean), -math.cos(lean)), (-1.0, 0.0, 0.0)))
    to_tool = [t - c for t, c in zip(take[:3], pin.pos)]
    wall = m.JOINT[1] - m.GUARD[1] - m.WALL
    start_y = wall - m.PIN_START - m.KEY[1] / 2
    lined = (m.JOINT[0] + to_tool[0], start_y + to_tool[1], m.JOINT[2] + to_tool[2], *take[3:])
    home_ = offset(lined, dy=m.PIN_HOME_Y - start_y)
    w, n, s = "west", "north", "south"
    out = [
        phase(1.5, "the west and the north arm reach out", Move(w, offset(back_off(gw, 0.10), dx=-0.10)),
              Move(n, offset(ge, dx=0.05, dz=0.16)), Open(w, 0.07), Open(n, 0.07)),
        phase(1.0, "and come over their leaves", Move(w, back_off(gw, 0.10)), Move(n, offset(ge, dz=0.08))),
        phase(1.2, "", Move(w, gw), Move(n, ge)),
        phase(0.8, "they settle"),
        phase(0.4, "and take them", Grip(w, "leaf_w", m.BAR_GRIP), Grip(n, "leaf_e", m.BAR_GRIP)),
        phase(3.0, "they turn them level: the west one up, the orange one down against its spring",
              Move(w, arc("leaf_w", 0.0, m.REST_DEG)), Move(n, arc("leaf_e", 0.0, m.REST_DEG))),
        phase(2.5, "the south arm comes over the pin", Move(s, back_off(take, 0.10), joints=True), Open(s, 0.07)),
        phase(1.2, "", Move(s, take)),
        phase(0.8, ""),
        phase(0.4, "takes it by its head", Grip(s, "pin", m.HEAD_GRIP)),
        phase(1.5, "lifts it", Move(s, offset(take, dz=0.10))),
        phase(3.0, "brings it in line with the opening", Move(s, offset(lined, dy=-0.04))),
        phase(1.0, "", Move(s, lined)),
        phase(1.0, "lines it up"),
        phase(5.0, "and slides it through all three slots", Move(s, home_)),
        phase(0.4, "lets go of it", Release(s)),
        phase(1.5, "and backs off", Move(s, offset(home_, dy=-0.06, dz=0.06))),
        phase(3.0, "the leaves are pinned"),
    ]
    return out[:p["upto"]] if p["upto"] else out


# ---------------------------------------------------------------------------------------------------- Rea-02 swapped
#: the route's choices (each found by planning and walking it for clearance, ``check``):
#: * the east Franka reaches no part of the middle channel where it lies but the top of its north wall by its east end
#:   (IK): its hand pointing south, ``pitch`` below level, rolled so that its fingers close nearly along y across the
#:   wall, ``at`` along the channel from its middle (its upper end, east as it lies), the tool point ``up`` over the
#:   wall's top -- within the 2 cm a privileged grip takes (pointing down or west, or with the tool point lower, no IK
#:   solution, or the hand within 1 cm of the table).  It lifts the channel ``lift`` -- out of the cradle it lies in
#:   (its lips stand 4.5 cm over the table; the cradle leaves this stretch of the north wall free), so 6.5 cm --
#:   draws it ``pull`` north and ``pull_dx`` east and
#:   sets it down (drawn 28 cm north, the hand coming down onto its handle ran within 1 cm of the lowest channel,
#:   already laid), then takes it by its handle from above and lays it as the south arm does in the reference;
#: * it draws the middle channel out first, then lays the lowest (drawn after, the hand came 1 cm from the lowest
#:   channel's handle);
#: * lifted ``high``, the middle channel is turned a ``quarter`` round about its handle, its long end over the track,
#:   and then on round to where it is laid, ``round`` over it: the half turn the other way, as the planner chose it,
#:   swept its long end past the south arm's base (0.3-1 cm), and 26 cm over its place was out of the Franka's reach;
#: * the idle south arm first moves its hand up and west, to ``south_away``: at home, the east wrist came 2.8 cm from
#:   it bringing the channel round;
#: * having set the middle channel down, the east hand rises ``swing_up`` before it swings to the lowest channel, and
#:   comes straight down from as high to where it reached for it before (the channels' places move with the seed:
#:   swung from 8 cm over the middle channel, the hand can pass into stand_2's top plate, and 15 cm clears seeds 0-9;
#:   going on from that height straight to the handle in the reach's 1.5 s, the hand takes it 1.4 cm high and lays it
#:   3 cm off, and the tomato can fall off the track)
R02 = {"pitch": 80.0, "at": -0.255, "up": 0.008, "pull": 0.22, "pull_dx": 0.08, "lift": 0.065, "wall_grip": 0.003,
       "upto": None, "east_first": "middle", "high": 0.25, "round": 0.15, "regrip_turn": None, "quarter": 90.0,
       "round_joints": False, "south_away": (-0.20, -0.62, 1.30), "swing_up": 0.15}


def _handle_grasp(inst, r, slot, k, start, laid_pose):
    """``rea02._grasp`` with the channel starting at ``start``: the hand (pointing down, a Franka's) that holds
    channel ``k`` by its handle, laid and at ``start``."""
    from mrbench.engines.base import quat_rotate

    arm = inst.scene.arm(slot)
    room = inst.metadata["fit_room_m"]
    hpos = r.handle(k, room)
    h = transform(laid_pose, hpos)
    p = math.radians(r.PITCH.get(arm.model, 90.0))
    rx, ry = h[0] - arm.pos[0], h[1] - arm.pos[1]
    n = math.hypot(rx, ry)
    approach = (rx / n * math.cos(p), ry / n * math.cos(p), -math.sin(p))
    bar = quat_rotate(laid_pose[3:], (1.0, 0.0, 0.0))
    across = (approach[1] * bar[2] - approach[2] * bar[1], approach[2] * bar[0] - approach[0] * bar[2],
              approach[0] * bar[1] - approach[1] * bar[0])
    tool = tool_quat(approach, across)
    rel = quat_mul((laid_pose[3], -laid_pose[4], -laid_pose[5], -laid_pose[6]), tool)
    return (*h, *tool), (*transform(start, hpos), *quat_mul(start[3:], rel))


@route("Rea-02-swapped")
def rea02_swapped(inst, p=None):
    """West + east, the east arm a Franka, the south arm idle: the east arm lays the lowest channel (the reference's
    moves, its hand pointing down), then draws the middle channel out from behind the south arm by its north wall,
    sets it down nearer, takes it by its handle and lays it into the lowest one (the reference's south moves); the
    west arm lays the upper channel and lifts the stop as in the reference."""
    from mrbench.tasks.impl import rea02_tomato_flume as r

    p = {**R02, **(p or {})}
    sc = inst.scene
    w, e = "west", "east"
    (lay_w, get_w), (lay_e, get_e) = r._grasp(inst, w, 0), r._grasp(inst, e, 2)
    # the middle channel's north wall (its wall_s, the handle's side: the channel lies turned half round)
    ch = sc.obj("channel_2")
    room = inst.metadata["fit_room_m"]
    wall_f = (p["at"], -(r.inner(1, room) / 2 + r.WALL_T / 2), r.WALL_H + p["up"])
    gp = transform(ch.pose, wall_f)
    a = math.radians(p["pitch"])
    g0 = (*gp, *tool_quat((0.0, -math.cos(a), -math.sin(a)), (0.0, -math.sin(a), math.cos(a))))
    moved = offset(ch.pose, dx=p["pull_dx"], dy=p["pull"])  # where it is set down, drawn north (and east)
    g1 = offset(g0, dx=p["pull_dx"], dy=p["pull"])
    lay_s, get_s = _handle_grasp(inst, r, e, 1, moved, r.laid(1))
    take, (ux, uy) = r._stop_take(inst)
    C = r.CARRY
    rt = p["regrip_turn"]
    # lifted, then turned a quarter round about its handle (its long end over the track, away from the south arm's
    # base), then on round to where it is laid: the half turn it needs, the long way past the south arm avoided
    high = offset(get_s, dz=p["high"])
    quarter = None
    if p["quarter"]:
        from mrbench.engines.base import axis_quat
        from mrbench.policies.script import carry
        from mrbench.tasks.impl._common import rotate_about

        lifted = offset(moved, dz=p["high"])
        quarter = carry(high, lifted, rotate_about(lifted, high[:3], axis_quat((0.0, 0.0, 1.0), p["quarter"])))
    draw = [  # the middle channel, behind the south arm, by its north wall's top
        phase(3.0, "the east arm reaches for the middle channel's north wall",
              Move(e, offset(g0, dz=0.06), joints=True), Open(e, 0.02)),
        phase(1.5, "", Move(e, g0)),
        phase(0.4, "and pinches it", Grip(e, "channel_2", p["wall_grip"])),
        phase(1.0, "lifts it a little", Move(e, offset(g0, dz=p["lift"]))),
        phase(3.0, "draws it out north", Move(e, offset(g1, dz=p["lift"]))),
        phase(1.5, "and sets it down", Move(e, offset(g1, dz=0.003))),
        phase(0.4, "", Release(e)),
        phase(1.0, "", Move(e, offset(g1, dz=0.08))),
    ]
    su = p["swing_up"] if p["east_first"] == "middle" else None  # rises from where it set the middle channel down
    lowest = [
        *([phase(1.0, "the east hand rises", Move(e, offset(g1, dz=su)))] if su else []),
        phase(3.0, "the east arm goes to the lowest channel",
              Move(e, offset(back_off(get_e, 0.08), dz=su or 0.0), joints=True), Open(e, r.HANDLE_SHAPE)),
        *([phase(1.0, "", Move(e, back_off(get_e, 0.08)))] if su else []),
        phase(1.5, "", Move(e, get_e)),
        phase(0.4, "and takes it by its handle", Grip(e, "channel_3", r.BAR_GRIP)),
        phase(1.5, "lifts it", Move(e, offset(get_e, dz=1.25 * C))),
        phase(4.0, "carries it over the track", Move(e, offset(lay_e, dz=1.75 * C), joints=True)),
        phase(2.0, "and lays it on its stand and the pot's rim", Move(e, offset(lay_e, dz=r.LET_GO))),
        phase(0.4, "", Release(e)),
        phase(1.5, "", Move(e, offset(back_off(lay_e, 0.10), dz=0.06))),
    ]
    west = [  # the west arm takes the upper channel and waits with it lifted, as in the reference
        phase(3.0, "the west arm goes to the upper channel", Move(w, back_off(get_w, 0.08), joints=True),
              Open(w, r.HANDLE_SHAPE)),
        phase(1.5, "", Move(w, get_w)),
        phase(0.4, "and takes it by its handle", Grip(w, "channel_1", r.BAR_GRIP)),
        phase(1.5, "lifts it", Move(w, offset(get_w, dz=C))),
    ]
    # the west arm's first moves beside the east arm's first ones
    first = draw + lowest if p["east_first"] == "middle" else lowest + draw
    out = [dataclasses.replace(ph, actions=ph.actions + west[k].actions,
                               note="; ".join(x for x in (ph.note, west[k].note) if x)) if k < len(west) else ph
           for k, ph in enumerate(first)]
    if p["south_away"] is not None:  # the idle south arm, first, up out of the middle channel's way
        away = (*p["south_away"], *tool_quat((0.0, 0.0, -1.0), (1.0, 0.0, 0.0)))
        out[0] = dataclasses.replace(out[0], actions=out[0].actions + (Move("south", away, joints=True),))
    out += [
        phase(2.5, "the east arm takes it by its handle", Move(e, back_off(get_s, 0.08), joints=True, turn=rt),
              Open(e, r.HANDLE_SHAPE)),
        phase(1.5, "", Move(e, get_s, turn=rt)),
        phase(0.4, "", Grip(e, "channel_2", r.BAR_GRIP)),
        phase(1.5, "lifts it high", Move(e, high)),
        *([phase(3.0, "turns it a quarter round over the track", Move(e, quarter))] if quarter else []),
        phase(4.0, "and brings it round", Move(e, offset(lay_s, dz=p["round"]), joints=p["round_joints"])),
        phase(2.0, "and lays it, its lower end inside the lowest", Move(e, offset(lay_s, dz=C))),
        phase(1.5, "", Move(e, offset(lay_s, dz=r.LET_GO))),
        phase(0.4, "", Release(e)),
        phase(3.0, "the west arm swings the upper one over",
              Move(e, offset(back_off(lay_s, 0.10), dz=0.08)), Move(w, offset(lay_w, dz=C), joints=True)),
        phase(0.5, "and lays it from the shelf into the middle one"),
        phase(1.5, "", Move(w, offset(lay_w, dz=r.LET_GO))),
        phase(0.4, "", Release(w)),
        phase(1.5, "", Move(w, offset(back_off(lay_w, 0.10), dz=0.06)), Move(e, home_pose(inst, e), joints=True)),
        phase(2.5, "the west arm goes to the stop in front of the tomato",
              Move(w, offset(back_off(take, 0.06), dz=0.04), joints=True), Open(w, r.STOP_SHAPE)),
        phase(1.0, "", Move(w, take)),
        phase(0.4, "", Grip(w, "stop", r.STOP_GRIP)),
        phase(1.0, "and lifts it away: the tomato rolls off the shelf", Move(w, offset(take, dz=0.05))),
        phase(1.0, "down the track", Move(w, offset(take, dx=-ux * 0.08, dy=-uy * 0.08, dz=0.05))),
        phase(5.0, "into the pot"),
    ]
    return out[:p["upto"]] if p["upto"] else out


# ----------------------------------------------------------------------------------------------- Rea-02 surplus nsw
#: the route's choices (each found by planning and walking it for clearance, ``check``):
#: * the north hand comes to the lowest channel's handle in a straight line (swung round in joint space, it passed
#:   1.7-2.7 cm from the idle east arm's hand), pointing down, and takes it where it lies;
#: * it lifts the channel ``n_lift`` and carries it through ``n_way`` -- the channel's middle (x, y), how high its floor
#:   is over where it lay, and its turn (degrees; 180 as it lies, its handle north) -- then sets it down at ``mid``,
#:   turned half round, its handle south: first west and up to 22 cm, north of the idle east arm, then south-west to
#:   the table's middle, and only there turned, clockwise about its own middle (sweeping 0.41 m round it, clear of the
#:   north arm's base, over the stands and under the end arms' hands).  Turned while carried (about (0.20, 0.12)), its
#:   east end swept into the east arm's base and its west end into the north arm's; turned at y 0, the north hand came
#:   2.8 cm from the south hand at home.  The north wrist turns 167 degrees (joint 7) with it;
#: * the north arm goes home, and the south arm takes the channel there by its handle, lifts it ``s_lift``, carries it
#:   south over the track (``s_dx`` east of ``mid``, ``s_over`` over where it is laid), then east over its place and
#:   lays it as the east arm does in the reference;
#: * the south arm goes home (``s_home``: from where it let the lowest channel go, its swing with the middle one came
#:   within 1 cm of its own base and upper arm) and lays the middle channel, and the west arm the upper one (it took
#:   it and waits with it lifted, as in the reference) and lifts the stop, all with the reference's moves
R02N = {"n_lift": 0.10, "n_way": ((0.40, 0.25, 0.22, 180.0), (0.05, 0.02, 0.22, 180.0), (0.05, 0.02, 0.22, 0.0)),
        "mid": (0.05, 0.02), "n_seconds": 3.0, "n_in_joints": False, "s_lift": 0.20, "s_dx": 0.0, "s_over": 0.175,
        "s_home": True, "south_away": None, "upto": None}


@route("Rea-02-surplus-nsw")
def rea02_surplus_nsw(inst, p=None):
    """North + south + west, the east arm idle at home: the north arm brings the lowest channel to the table's middle,
    turning it half round, and sets it down; the south arm takes it there and lays it, then lays the middle channel
    (the reference's moves); the west arm lays the upper channel and lifts the stop as in the reference."""
    from mrbench.engines.base import yaw_quat
    from mrbench.policies.script import carry
    from mrbench.tasks.impl import rea02_tomato_flume as r

    p = {**R02N, **(p or {})}
    sc = inst.scene
    w, s, n = "west", "south", "north"
    (lay_w, get_w), (lay_s, get_s) = r._grasp(inst, w, 0), r._grasp(inst, s, 1)
    _, get_n = r._grasp(inst, n, 2)  # the north hand on the lowest channel's handle where it lies
    start = sc.obj("channel_3").pose
    zf = start[2]  # its floor's top over the table, lying

    def body(x, y, dz, yaw):
        return (x, y, zf + dz, *yaw_quat(yaw))

    mid = body(*p["mid"], 0.0, 0.0)
    lay_e, get_m = _handle_grasp(inst, r, s, 2, mid, r.laid(2))  # the south hand on it there, and laid

    def leg(a, b):  # the channel's middle, height and turn, each straight from a to b
        return lambda u: carry(get_n, start, body(*(x + u * (y - x) for x, y in zip(a, b))))

    way = [(start[0], start[1], 0.0, 180.0), (start[0], start[1], p["n_lift"], 180.0), *p["n_way"]]
    down = (*p["mid"], r.LET_GO, 0.0)
    take, (ux, uy) = r._stop_take(inst)
    C, P = r.CARRY, r.PAUSE
    up = leg(way[-1], down)(1.0)
    over = offset(lay_e, dx=mid[0] + p["s_dx"] - r.laid(2)[0], dz=p["s_over"])  # over the track, south of ``mid``
    out = []
    if p["south_away"] is not None:
        away = (*p["south_away"], *tool_quat((0.0, 0.0, -1.0), (1.0, 0.0, 0.0)))
        out.append(phase(2.0, "the south arm moves out of the way", Move(s, away, joints=True)))
    out += [
        phase(3.0, "the north and the west arm go to their channels",
              Move(n, back_off(get_n, 0.08), joints=p["n_in_joints"]),
              Move(w, back_off(get_w, 0.08), joints=True), Open(w, r.HANDLE_SHAPE)),
        phase(1.5, "", Move(n, get_n), Move(w, get_w)),
        phase(P, ""),
        phase(0.4, "and take them by their handles", Grip(n, "channel_3", r.BAR_GRIP),
              Grip(w, "channel_1", r.BAR_GRIP)),
        phase(1.5, "lift them", Move(n, leg(way[0], way[1])), Move(w, offset(get_w, dz=C))),
        *[phase(p["n_seconds"], "the north arm brings the lowest channel round to the table's middle" if k == 1
                else "", Move(n, leg(a, b))) for k, (a, b) in enumerate(zip(way[1:], way[2:]), start=1)],
        phase(1.5, "and sets it down there, turned half round", Move(n, leg(way[-1], down))),
        phase(0.4, "", Release(n)),
        phase(1.0, "", Move(n, offset(up, dz=0.08))),
        phase(2.0, "", Move(n, home_pose(inst, n), joints=True)),
        phase(2.5, "the south arm takes it by its handle", Move(s, back_off(get_m, 0.08), joints=True)),
        phase(1.5, "", Move(s, get_m)),
        phase(P, ""),
        phase(0.4, "", Grip(s, "channel_3", r.BAR_GRIP)),
        phase(1.5, "lifts it", Move(s, offset(get_m, dz=p["s_lift"]))),
        phase(2.5, "carries it south over the track", Move(s, over)),
        phase(2.5, "and east over its place", Move(s, offset(lay_e, dz=p["s_over"]))),
        phase(2.0, "and lays it on its stand and the pot's rim", Move(s, offset(lay_e, dz=r.LET_GO))),
        phase(0.4, "", Release(s)),
        phase(1.5, "", Move(s, offset(back_off(lay_e, 0.10), dz=0.06))),
        # the middle channel, the west arm's upper one and the stop: the reference's moves
        *([phase(2.0, "", Move(s, home_pose(inst, s), joints=True))] if p["s_home"] else []),
        phase(3.0, "the south arm goes to the middle channel", Move(s, back_off(get_s, 0.08), joints=True)),
        phase(1.5, "", Move(s, get_s)),
        phase(P, ""),
        phase(0.4, "and takes it by its handle", Grip(s, "channel_2", r.BAR_GRIP)),
        phase(1.5, "lifts it out of its cradle", Move(s, offset(get_s, dz=C))),
        phase(1.5, "and high", Move(s, offset(get_s, dz=r.HIGH))),
        phase(4.0, "and brings it round from behind it", Move(s, offset(lay_s, dz=2.6 * C), joints=True)),
        phase(2.0, "and lays it, its lower end inside the lowest", Move(s, offset(lay_s, dz=C))),
        phase(P, ""),
        phase(1.5, "", Move(s, offset(lay_s, dz=r.LET_GO))),
        phase(0.4, "", Release(s)),
        phase(3.0, "the west arm swings the upper one over",
              Move(s, offset(back_off(lay_s, 0.10), dz=0.08)), Move(w, offset(lay_w, dz=C), joints=True)),
        phase(0.5, "and lays it from the shelf into the middle one"),
        phase(1.5, "", Move(w, offset(lay_w, dz=r.LET_GO))),
        phase(0.4, "", Release(w)),
        phase(1.5, "", Move(w, offset(back_off(lay_w, 0.10), dz=0.06)), Move(s, home_pose(inst, s), joints=True)),
        phase(2.5, "the west arm goes to the stop in front of the tomato",
              Move(w, offset(back_off(take, 0.06), dz=0.04), joints=True), Open(w, r.STOP_SHAPE)),
        phase(1.0, "", Move(w, take)),
        phase(0.4, "", Grip(w, "stop", r.STOP_GRIP)),
        phase(1.0, "and lifts it away: the tomato rolls off the shelf", Move(w, offset(take, dz=0.05))),
        phase(1.0, "down the track", Move(w, offset(take, dx=-ux * 0.08, dy=-uy * 0.08, dz=0.05))),
        phase(5.0, "into the pot"),
    ]
    return out[:p["upto"]] if p["upto"] else out


# ---------------------------------------------------------------------------------------------------- Rea-01 surplus
#: the route's choices (each found by planning and walking it for clearance, ``check``, and playing it):
#: * the west hand takes each piece as the reference does (45 degrees down, square to its reach) but with its tool point
#:   ``west_back`` behind the piece's middle along its pointing (within a grip's 2 cm of it), so that the piece is held
#:   that much ahead of its fingers; at the hand-over it points level away from its base, the piece's middle at
#:   ``meet`` (x, y, height over the table), 47 cm from its base (at 60 cm, (-0.15, 0.25) 40 cm up, the hand sagged
#:   2.3 cm under its plan holding the piece out, and the north hand coming down from over it closed 3.9 cm off it);
#: * the north hand takes the piece from ahead of it -- from the side away from the west hand, so that the two hands
#:   stay 3 cm apart (from straight over it, its fingers came 2.8-3.0 cm from the west fingers under them) and a sag
#:   of the west hand moves the piece across the north fingers rather than away from them: pointing back toward the
#:   west arm's base, ``north_tilt`` below level (level, it ran into the partition), its fingers closing in the
#:   upright plane of its pointing (square to the west fingers), its tool point ``north_gap`` from the piece's nearest
#:   box and ``north_sag`` lower for the west hand's sag (a grip takes within 2 cm); it comes in from ``back`` behind
#:   (from 8 cm, or with the piece 42 cm up, its forearm came 0.3-0.9 cm from the partition);
#: * the north hand lifts the piece straight up to ``high`` over the table and swings it in joint space over the
#:   partition to over ``north_way`` (x, y), then to ``north_pre`` (x, y, height), north of the idle east arm (swung
#:   straight to the crate, the piece and the north hand passed 1.2-3 cm from the east hand at home), and to
#:   ``crate_high`` over its slot, its hand turned to point straight down (tilted as it took the piece, it came 0.8 cm
#:   from the crate's west wall), the piece hanging under it (carried straight across with the hand held as it took the
#:   piece, the hand's last joint turned 244 degrees, and no solution followed);
#: * the crate: each piece let go with its lowest point ``drop`` over the floor at ``slots``, 8 cm apart along y (7 cm
#:   apart, the piece carried in came 1.4 cm from the one there), the southernmost first, so that the hand never
#:   carries a piece over one already there
R01 = {"west_back": 0.04, "meet": (-0.34, 0.30, 0.46), "north_tilt": 30.0, "north_gap": 0.010, "north_sag": 0.0,
       "high": 0.62, "north_way": (0.40, 0.15), "north_pre": (0.80, 0.45, 0.35), "crate_high": 0.25,
       "slots": {1: [(1.01, 0.40)], 3: [(1.01, 0.24), (1.01, 0.32), (1.01, 0.40)]},
       "drop": 0.05, "back": 0.05, "upto": None}


@route("Rea-01-surplus-nw")
def rea01_surplus_nw(inst, p=None):
    """North + west, the south and the east arm idle at home: for each piece the west arm takes it from the start area
    and holds it out west of the tall partition; the north arm reaches over the partition and takes it from ahead of
    it, the west hand lets go; the north arm carries it up over the partition, round north of the east arm and into
    the crate, and comes back for the next."""
    from mrbench.robots import collision as col
    from mrbench.robots import kinematics as kin
    from mrbench.tasks.impl import rea01_relay as r
    from mrbench.tasks.impl._common import Z

    np = kin.np
    p = {**R01, **(p or {})}
    sc = inst.scene
    w, n = "west", "north"
    pieces = list(inst.metadata["fruit"])
    slots = p["slots"][len(pieces)]
    zc = Z + r.FRUIT_Z  # a piece's middle, standing on the table
    up = r.HOVER - r.FRUIT_Z

    def radial(xy):
        a = sc.arm(w)
        dx, dy = xy[0] - a.pos[0], xy[1] - a.pos[1]
        k = math.hypot(dx, dy)
        return dx / k, dy / k

    def west_hand(xy, zz, pitch):
        rx, ry = radial(xy)
        pr = math.radians(pitch)
        return (*xy, zz, *tool_quat((rx * math.cos(pr), ry * math.cos(pr), -math.sin(pr)), (-ry, rx, 0.0)))

    nb = sc.arm(n).pos

    def turn_z(m, frm, to):  # pose matrix ``m`` turned about z as the north base turns from pointing at ``frm`` to ``to``
        t = math.atan2(to[1] - nb[1], to[0] - nb[0]) - math.atan2(frm[1] - nb[1], frm[0] - nb[0])
        rz = np.eye(4)
        rz[:2, :2] = [[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]]
        out = m.copy()
        out[:3, :3] = rz[:3, :3] @ m[:3, :3]
        return out

    def at(m, xyz):
        out = m.copy()
        out[:3, 3] = xyz
        return kin.matrix_pose(out)

    mx, my, mh = p["meet"]
    rx, ry = radial((mx, my))
    # the west hand at the hand-over, level, the piece's middle at ``meet``
    give = west_hand((mx - rx * p["west_back"], my - ry * p["west_back"]), Z + mh, 0.0)
    t = math.radians(p["north_tilt"])
    approach = (-rx * math.cos(t), -ry * math.cos(t), -math.sin(t))  # back toward the west base, tilted down
    hand_q = tool_quat(approach, (0.0, 0.0, 1.0))
    out = []
    for i, name in enumerate(pieces):
        obj = sc.obj(name)
        pick = back_off(west_hand(obj.pos[:2], zc, r.PITCH["piper"]), p["west_back"])
        # where the piece is at the hand-over: held as the west hand took it
        held = kin.pose_matrix(give) @ np.linalg.inv(kin.pose_matrix(pick)) @ kin.pose_matrix(obj.pose)
        boxes = col.boxes_of(obj, kin.matrix_pose(held))
        c = held[:3, 3]
        a = np.asarray(approach)
        d = next(d for d in np.arange(0.02, 0.10, 0.0005)
                 if min(float(col.point_box((c - d * a)[None], b)[0]) for b in boxes) >= p["north_gap"])
        tool = c - d * a + np.array([0.0, 0.0, -p["north_sag"]])
        take = (*tool, *hand_q)
        m_take = kin.pose_matrix(take)
        rel = np.linalg.inv(m_take) @ held  # the piece in the north hand
        wait = offset(back_off(take, p["back"]), dz=Z + p["high"] - take[2])
        way = at(turn_z(m_take, (mx, my), p["north_way"]), (*p["north_way"], Z + p["high"]))
        pre = at(turn_z(m_take, (mx, my), p["north_pre"]), (p["north_pre"][0], p["north_pre"][1], Z + p["north_pre"][2]))
        # in the crate the hand points straight down, its fingers closing along its pointing there turned with the base
        # (tilted as it took the piece, its hand came 0.8 cm from the crate's west wall), the piece under it, its middle
        # over the slot, its lowest point ``drop`` over the floor
        sx, sy = slots[i]
        ax = turn_z(m_take, (mx, my), (sx, sy))[:3, 2]
        k = math.hypot(ax[0], ax[1])
        m_hand = kin.pose_matrix((0.0, 0.0, 0.0, *tool_quat((0.0, 0.0, -1.0), (ax[0] / k, ax[1] / k, 0.0))))
        m_piece = m_hand @ rel
        low = min(float(col.box_points(b, 3)[:, 2].min()) for b in col.boxes_of(obj, kin.matrix_pose(m_piece)))
        shift = np.array([sx, sy, Z + r.BIN_FLOOR + p["drop"]]) - np.array([m_piece[0, 3], m_piece[1, 3], low])
        put = at(m_hand, m_hand[:3, 3] + shift)
        over = offset(put, dz=Z + p["crate_high"] - put[2])
        first = [Move(n, offset(way, dz=0.0), joints=True), Open(n, 0.08)] if i == 0 else []
        out += [
            phase(2.0, f"west: takes piece {i + 1}", Move(w, offset(pick, dz=up), joints=True), Open(w, 0.07), *first),
            phase(1.0, "", Move(w, pick), *([Move(n, wait)] if i == 0 else [])),
            phase(0.25, "", Grip(w, name, r.FRUIT_GRIP)),
            phase(1.0, "", Move(w, offset(pick, dz=up))),
            phase(2.5, "and holds it out west of the partition", Move(w, give, joints=True)),
            phase(1.5, "north: reaches over the partition to it", Move(n, back_off(take, p["back"]))),
            phase(1.0, "and comes in from ahead of it", Move(n, take)),
            phase(0.25, "and takes it", Grip(n, name, r.FRUIT_GRIP)),
            phase(0.25, "west: lets go", Release(w)),
            phase(1.0, "", Move(w, back_off(give, p["back"])), Move(n, offset(take, dz=0.04))),
            phase(1.0, "north: lifts it", Move(n, offset(take, dz=Z + p["high"] - take[2]))),
            phase(2.0, "and carries it over the partition", Move(n, way, joints=True)),
            phase(2.0, "round north of the east arm", Move(n, pre, joints=True)),
            phase(1.5, "into the crate", Move(n, over, joints=True)),
            phase(1.0, "", Move(n, put)),
            phase(0.25, "", Release(n)),
            phase(1.0, "", Move(n, over)),
        ]
        if i + 1 < len(pieces):
            out += [phase(1.5, "north: back over the partition", Move(n, pre)),
                    phase(2.0, "", Move(n, way)),
                    phase(2.0, "", Move(n, wait))]
    out.append(phase(1.5, "done", Move(w, home_pose(inst, w), joints=True)))
    return out[:p["upto"]] if p["upto"] else out


# ---------------------------------------------------------------------------------------------------- Rea-04 swapped
#: the route's choices (each found by planning and walking it for clearance, ``check``):
#: * the idle south arm (a Piper, its hand at home 25 cm over the middle spill, where the east wrist came 2.6-3.0 cm
#:   from it) first draws its hand back ``south_away`` (dx, dy, dz) from home, turned as at home (moved west, or
#:   pointing down, no IK);
#: * the west arm wipes its spill and puts the sponge down on pad A with the reference's moves, then takes it again
#:   there, its hand pointing ``w_pitch`` down, and puts it down at ``meet``, on the line between the end arms' bases,
#:   lengthwise along it, 0.60 m from the west base (held as the reference holds it, 75 degrees down, the west hand
#:   reaches 0.55 m out at the table and no further, IK; and put down straight from the west spill even there, the
#:   plan found no way in to the sponge on its dish);
#: * the east Franka takes it there by its east end: its hand pointing ``e_pitch`` below level toward the west (the
#:   way it points out from its base), its fingers across the sponge's width, the tool point ``e_at`` from the
#:   sponge's middle toward its east end (4 cm), 6 mm under its top as the reference takes it -- held so, the sponge's
#:   middle reaches the middle spill on its riser, 0.86 m from the east base: the hand alone comes no nearer than 3 cm
#:   to the spill's middle at that height, and only pointing 60 or 45 degrees down (IK);
#: * it sets the sponge down on the middle spill, its middle ``m_toward`` from the spill's toward the east base and
#:   ``press`` below resting on it, scrubs it with ``scrub`` (each way; the reference's 1.2 x 0.8 cm), lifts it
#:   ``lift`` and puts it down on pad B (the reference's south arm leaves it there too), the turn it holds it with
#:   going round with its reach (``carry_yaw``);
#: * it takes it again there from above, pointing ``regrip`` down, wipes the east spill with it turned ``b_turn``
#:   degrees from how it lay on pad B (the hand then crosswise to its reach, as on pad B), lifts it ``high`` (in a
#:   straight line, or with ``up_joints`` in joint space), swings it over the sink, turned to ``sink_yaw`` degrees
#:   (across its reach there: the hand turns 9 degrees from the east spill), and lets it fall in from there, ``drop``
#:   over where it would lie 2 cm over the sink's floor (``down_joints``: brought down to there in joint space; with
#:   ``drop`` as ``high``, not brought down at all): its bottom 22 cm over the floor, where the reference's east arm
#:   lets go 4.4 cm over it, and it falls flat into the sink.  Held by its end from the middle spill on, at 60 degrees
#:   down, the hand did not reach the sink (0.25 m from its base)
#:   unless the sponge turned round, and every way round one of its swings took the sponge into the table or the hand
#:   into the sink's walls; held as on pad B through the east spill, or let down into the sink in a straight line from
#:   20 cm over it, there was no way in to the sponge on pad B from which the rest could be followed without changing
#:   the arm's joint solution (easy seed 1; hard seeds 0 and 2); let down in joint space, the hand went into the sink's
#:   walls (hard seed 0); turned east-west in the sink, the arm swung its base round 150 degrees and the sponge past it
#:   (0.8 cm, hard seed 2); lying along its reach in the sink, the sponge came 0.2-1.4 cm from the sink's walls (a held
#:   body keeps 1.5); swung over at 8 cm, it went into the south wall
R04 = {"south_away": (0.0, -0.10, 0.05), "w_pitch": 60.0, "meet": (-0.10, 0.0), "e_pitch": 60.0, "e_at": 0.035,
       "m_toward": 0.0, "press": -0.002, "scrub": (0.008, 0.005), "lift": 0.08, "regrip": 90.0, "b_turn": 81.0,
       "high": 0.20, "up_joints": False, "sink_yaw": 145.7, "down_joints": True, "drop": 0.20, "upto": None}


def _r04_hold(inst, r, slot, sponge, pitch):
    """``rea04._hold`` with the hand ``pitch`` below level: from above, by the sponge's top, its fingers across its
    width, pointing out from the arm's base."""
    from mrbench.policies.script import radial_tool

    arm = inst.scene.arm(slot)
    yaw = 2 * math.atan2(sponge[6], sponge[3])
    across = (-math.sin(yaw), math.cos(yaw), 0.0)
    top = (sponge[0], sponge[1], sponge[2] + r.SPONGE[2] / 2 - r.TAKE_DOWN)
    return (*top, *radial_tool(arm.pos, top, pitch, across))


@route("Rea-04-swapped")
def rea04_swapped(inst, p=None):
    """West + east, the east arm a Franka, the south arm idle (its hand drawn back): the west arm wipes the west spill
    and puts the sponge down on pad A (the reference's moves), then on the table's middle; the east arm takes it there
    by its east end, wipes the middle spill on its riser, puts it down on pad B, takes it again from above, wipes the
    east spill and drops it in the sink."""
    from mrbench.engines.base import yaw_quat
    from mrbench.policies.script import carry
    from mrbench.tasks.impl import rea04_wipe_spills as r

    p = {**R04, **(p or {})}
    sc = inst.scene
    w, e = "west", "east"
    start = sc.obj("sponge").pose
    wiped_w = r._wiping(inst, "west")
    yaw_w = math.degrees(2 * math.atan2(wiped_w[6], wiped_w[3]))
    turned = lambda xy: yaw_w + r.radial_deg(w, xy) - r.radial_deg(w, wiped_w[:2])  # noqa: E731 -- as the west hand
    on_m = r.sponge_at(p["meet"], r.Z, turned(p["meet"]))
    L = p["lift"]
    out = []
    if p["south_away"] is not None:  # the idle south arm, first, out of the east hand's way
        away = offset(home_pose(inst, "south"), *p["south_away"])  # its hand turned as at home
        out.append(phase(2.0, "the idle south arm moves out of the way", Move("south", away, joints=True)))
    # the reference's leg to pad A, then taken again there and put down at ``meet``
    on_a = r._on_pad("a", turned(r.PADS["a"]))
    out += r._leg(inst, w, start, wiped_w, on_a, "and puts it down on pad A")
    take_a = _r04_hold(inst, r, w, on_a, p["w_pitch"])
    put = carry(take_a, on_a, on_m)
    out += [
        phase(2.0, "takes it again there", Move(w, back_off(take_a, 0.08), joints=True), Open(w, r.SHAPE_W)),
        phase(1.2, "", Move(w, take_a)),
        phase(0.4, "", Grip(w, "sponge", r.GRIP_W)),
        phase(1.0, "", Move(w, offset(take_a, dz=L))),
        phase(2.0, "and puts it down on the table's middle", Move(w, offset(put, dz=L), joints=True)),
        phase(1.2, "", Move(w, offset(put, dz=0.004))),
        phase(0.4, "", Release(w)),
        phase(1.0, "", Move(w, offset(put, dz=L))),
    ]
    # the east hand on the sponge's east end: pointing out from its base toward the sponge, ``e_pitch`` down
    ex, ey = sc.arm(e).pos[:2]
    ux, uy = ex - on_m[0], ey - on_m[1]
    n = math.hypot(ux, uy)
    ux, uy = ux / n, uy / n  # from the sponge's middle toward the east base
    a = math.radians(p["e_pitch"])
    take = (on_m[0] + p["e_at"] * ux, on_m[1] + p["e_at"] * uy, on_m[2] + r.SPONGE[2] / 2 - r.TAKE_DOWN,
            *tool_quat((-ux * math.cos(a), -uy * math.cos(a), -math.sin(a)), (-uy, ux, 0.0)))

    def at_spill(slot, toward=0.0):  # the sponge down on ``slot``'s spill, turned as the east hand holds it
        s = sc.obj(f"spill_{slot}").pos
        k = toward / math.hypot(ex - s[0], ey - s[1])
        xy = (s[0] + k * (ex - s[0]), s[1] + k * (ey - s[1]))
        return r.carry_yaw(r.sponge_at(xy, r.floor_z(slot) + r.SPILL_T, 0.0), on_m, inst=inst)

    def scrub(pose):
        return lambda u: (pose[0] + p["scrub"][0] * math.sin(2 * math.pi * 2 * u),
                          pose[1] + p["scrub"][1] * math.sin(2 * math.pi * 4 * u), *pose[2:])

    hold = lambda body: carry(take, on_m, body)  # noqa: E731 -- the east hand, the sponge at ``body``
    mid = hold(at_spill("south", p["m_toward"]))
    # put down on pad B, and taken again there from above (``regrip``: pointing so far down) for the rest
    on_b = r.carry_yaw(r.sponge_at(r.PADS["b"], r.Z, 0.0), on_m, inst=inst)
    put_b = hold(on_b)
    take_b = _r04_hold(inst, r, e, on_b, p["regrip"])
    turn_e = yaw_quat(p["b_turn"])  # the sponge on the east spill turned so from how it lay on pad B
    east_b = carry(take_b, on_b, (*r.sponge_at(sc.obj("spill_east").pos[:2], r.Z + r.SPILL_T, 0.0)[:3],
                                  *quat_mul(turn_e, on_b[3:])))
    sink_b = carry(take_b, on_b, (*r.SINK_XY, r.Z + 0.01 + r.SPONGE[2] / 2 + 0.02, *yaw_quat(p["sink_yaw"])))
    loops = max(2, math.ceil(inst.metadata["wipe_s"] + 0.6))
    out += [
        phase(2.5, "the west arm goes home; the east arm comes to the sponge's east end",
              Move(w, home_pose(inst, w), joints=True), Move(e, back_off(take, 0.08), joints=True),
              Open(e, r.SHAPE_W)),
        phase(1.2, "", Move(e, take)),
        phase(0.4, "and takes it by its end", Grip(e, "sponge", r.GRIP_W)),
        phase(1.0, "", Move(e, offset(take, dz=L))),
        phase(2.5, "wipes the middle spill on its riser", Move(e, offset(mid, dz=L), joints=True)),
        phase(1.2, "", Move(e, offset(mid, dz=p["press"]))),
        phase(float(loops), "", Move(e, scrub(offset(mid, dz=p["press"])))),
        phase(1.0, "", Move(e, offset(mid, dz=L))),
        phase(2.5, "puts it down on pad B", Move(e, offset(put_b, dz=L), joints=True)),
        phase(1.2, "", Move(e, offset(put_b, dz=0.004))),
        phase(0.4, "", Release(e)),
        phase(1.0, "", Move(e, offset(put_b, dz=L))),
        # (an empty hand made sure of first, as Con-03's borrow reference does: the lift off pad B then goes straight
        # up, and the arm changes its joint solution for the hold from above on the way over, not with its fingers by
        # the table, where they came 0.7 cm from it at hard seed 0)
        phase(2.0, "takes it again from above", Release(e), Move(e, back_off(take_b, 0.08), joints=True),
              Open(e, r.SHAPE_W)),
        phase(1.2, "", Move(e, take_b)),
        phase(0.4, "", Grip(e, "sponge", r.GRIP_W)),
        phase(1.0, "", Move(e, offset(take_b, dz=L))),
        phase(2.5, "wipes the east spill", Move(e, offset(east_b, dz=L), joints=True)),
        phase(1.2, "", Move(e, offset(east_b, dz=0.0005))),
        phase(float(loops), "", Move(e, scrub(offset(east_b, dz=0.0005)))),
        phase(1.0, "", Move(e, offset(east_b, dz=L))),
        phase(1.0, "", Move(e, offset(east_b, dz=p["high"]), joints=p["up_joints"])),
        phase(2.5, "and drops it in the sink", Move(e, offset(sink_b, dz=p["high"]), joints=True)),
        phase(1.5, "", Move(e, offset(sink_b, dz=p["drop"]), joints=p["down_joints"])),
        phase(0.4, "", Release(e)),
        phase(1.0, "", Move(e, offset(sink_b, dz=max(L, p["drop"] + 0.04)))),
        phase(2.0, "done"),
    ]
    return out[:p["upto"]] if p["upto"] else out


# ---------------------------------------------------------------------------------------------------- Con-04 surplus
@route("Con-04-surplus-nsw")
def con04_surplus_nsw(inst, p=None):
    """North + south + west, the east arm idle at home: the north arm holds the east bleed valve's lever down by its
    grip, as the east arm does in the reference."""
    from mrbench.tasks.impl import con04_pressure_test as m

    return m._hold(inst, {"valve_w": "west", "valve_e": "north", "pedal": "south"})  # (``play`` replaces m.script)


@route("Con-04-surplus-nes")
def con04_surplus_nes(inst, p=None):
    """North + east + south, the west arm idle at home: the north arm holds the west valve's lever down."""
    from mrbench.tasks.impl import con04_pressure_test as m

    return m._hold(inst, {"valve_w": "north", "valve_e": "east", "pedal": "south"})


@route("Con-04-surplus-enw")
def con04_surplus_enw(inst, p=None):
    """East + north + west, the south arm idle: the north arm holds the west valve's lever down and the west arm, its
    hand free, the foot valve's pedal by its toe bar, pointing 30 degrees down toward the east (straight down it does
    not reach 10 cm over the toe bar, IK; at 45 and 60 its hand came within 3 cm of the south hand).  The south arm
    first draws its hand back east and toward its base, out of the west hand's way (at home its hand is 2.7 cm from
    it)."""
    from mrbench.tasks.impl import con04_pressure_test as m

    a = math.radians(30.0)
    lean = tool_quat((math.cos(a), 0.0, -math.sin(a)), (0.0, 1.0, 0.0))
    away = phase(1.5, "the south arm, idle, draws its hand back", Move("south", offset(home_pose(inst, "south"),
                                                                                         dx=0.15, dy=-0.10)))
    return [away] + m._hold(inst, {"valve_w": "north", "valve_e": "east", "pedal": "west"}, hands={"pedal": lean})


# ---------------------------------------------------------------- Con-04 borrow: the distractor on a bleed valve
@route("Con-04-borrow-decoyvalve")
def con04_borrow_decoyvalve(inst, p=None):
    """West + east in the borrow scene, the prop left where it stands: the end arm on the
    distractor's side (the 1.2 kg weight, half as tall as the prop) takes it by its knob and lets it down onto its own
    bleed valve's grip, its bottom 4 mm over the grip's top, and lets go -- too light to hold the valve shut
    (``con04_pressure_test.VALVE_PRELOAD_DEG``); that arm then holds the foot valve's pedal by its toe bar and the
    other end arm its own valve.  The hand on the pedal points down toward the middle: the west one 30 degrees, as ``Con-04-surplus-enw``'s; the east one 50, its fingers along -y
    (at 30 or 40 its wrist camera came 0.9-1.0 cm from the pedal)."""
    from mrbench.tasks.impl import con04_pressure_test as m

    b = inst.metadata["borrow"]
    decoy = b["decoys"][0]
    w = inst.scene.obj(decoy)
    a = "west" if w.pos[0] < 0 else "east"
    s = 1.0 if a == "west" else -1.0
    other = "east" if a == "west" else "west"
    valve, other_valve = ("valve_w", "valve_e") if a == "west" else ("valve_e", "valve_w")
    h = w.part("body").size[1]
    take = (w.pos[0], w.pos[1], m.Z + h + m.NECK_H / 2, *m.DOWN_X)
    g = m.grip(inst, valve)
    drop = (g[0], g[1], g[2] + m.GRIP_W / 2 + 0.004 + h + m.NECK_H / 2, *m.DOWN_X)
    high = max(m.Z + 0.29, drop[2] + 0.08)
    t = math.radians(30.0 if a == "west" else 50.0)
    lean = tool_quat((s * math.cos(t), 0.0, -math.sin(t)), (0.0, s, 0.0))
    return [
        phase(2.0, f"the {a} arm reaches for {decoy}", Move(a, offset(take, dz=0.12)), Open(a, 0.05)),
        phase(1.5, "", Move(a, take)),
        phase(0.4, "takes it by its knob", Grip(a, decoy, 2 * m.NECK_R)),
        phase(1.5, "lifts it", Move(a, (*take[:2], high, *m.DOWN_X))),
        phase(3.0, f"carries it over the {valve} grip", Move(a, (*drop[:2], high, *m.DOWN_X))),
        phase(2.0, "lets it down onto the grip", Move(a, drop)),
        phase(0.4, "and lets go: the lever goes down under it", Release(a)),
        phase(1.5, "", Move(a, offset(drop, dz=0.10))),
        phase(2.0, "", Move(a, offset(home_pose(inst, a), dz=0.05))),
        phase(1.0, "the weight on the valve's grip"),
    ] + m._hold(inst, {other_valve: other, "pedal": a}, settle=1.0, hands={"pedal": lean})


# ------------------------------------------------------------ Con-04 shortfall: the two-arm attempt (must fail)
@route("Con-04-shortfall-attempt")
def con04_shortfall_attempt(inst, p=None):
    """West + east, the south arm away: the west hand holds the pedal down first, leaning 30 degrees east as
    ``Con-04-surplus-enw``'s, while the east arm shuts its own valve; then the west hand lets the pedal go for its own
    valve, and both hold their valves before letting go.  It must fail: let go, the pedal rises on its spring."""
    from mrbench.tasks.impl import con04_pressure_test as m

    a = math.radians(30.0)
    lean = tool_quat((math.cos(a), 0.0, -math.sin(a)), (0.0, 1.0, 0.0))
    first = m._hold(inst, {"pedal": "west", "valve_e": "east"}, hands={"pedal": lean})
    then = m._hold(inst, {"valve_w": "west"})
    up = [phase(0.4, "the west hand lets the pedal go for its own valve", Release("west")),
          phase(1.5, "", Move("west", offset(m._grip(inst, "pedal", q=lean), dz=0.12)))]
    let_go, lift = then[5], then[6]
    let_go = dataclasses.replace(let_go, actions=let_go.actions + (Release("east"),))
    lift = dataclasses.replace(lift, actions=lift.actions + (Move("east", back_off(m._grip(inst, "valve_e"), 0.12)),))
    return first[:5] + up + then[:5] + [let_go, lift] + then[7:]


# ---------------------------------------------------------------------------------------------------------- driver
#: the setting the scenes are built in (``--setting``): the card's look, or the agent's "varied" scene of the seed
SETTING = "card"


def scene(case_id: str, level: str, seed: int):
    # "Con-02-surplus" or "Con-02-surplus-nsw" (a route's own team after the version)
    from mrbench.agent.cases import setting_look

    task, variant = "-".join(case_id.split("-")[:2]), case_id.split("-")[2]
    return make_case(task, variant, level, seed, look=setting_look(SETTING, seed)).inst


def check(case_id: str, level: str, seed: int) -> dict:
    from mrbench.agent import program

    program.MAX_CLASHES = 1000
    inst = scene(case_id, level, seed)
    r = program.check(inst, ROUTES[case_id](inst))
    return {"ok": r["ok"], "problems": r["problems"], "warnings": r.get("warnings") or []}


def play(case_id: str, level: str, seed: int, out: Path, video: bool = True) -> dict:
    """In this (Isaac) process: the case's scene, the route played and scored as ``mrbench demo`` plays a reference."""
    from mrbench import demo

    inst = scene(case_id, level, seed)
    mod = importlib.import_module(inst.spec.impl)
    mod.script = ROUTES[case_id]  # this process only: demo.play takes the route for the reference
    name = f"{case_id}-{level}-s{seed}"
    rep = demo.play(inst, out, name, video=video)
    rep = {"case": case_id, "team": sorted({a.slot for a in inst.scene.arms}),
           "robots": {a.slot: a.model for a in inst.scene.arms}, **rep}
    (out / f"{name}.json").write_text(json.dumps(rep, indent=1))
    return rep


#: what an Isaac process that found no room on the GPU says (Kit's renderer, torch, PhysX)
OUT_OF_MEMORY = ("out of memory", "out_of_device_memory", "failed to allocate memory", "not enough gpu memory")


def _wait_for_gpu(mib: int) -> None:
    """Until the GPU this process is given (CUDA_VISIBLE_DEVICES) has ``mib`` MiB free, for 20 s running."""
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    ok = 0
    while ok < 2:
        q = subprocess.run(["nvidia-smi", "-i", gpu, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True)
        ok = ok + 1 if q.returncode == 0 and int(q.stdout.split()[0]) >= mib else 0
        time.sleep(10)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("check", "run", "child"))
    p.add_argument("case", choices=sorted(ROUTES))
    p.add_argument("--levels", nargs="+", default=["easy", "hard"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("-o", "--out", default="runs/gold")
    p.add_argument("--timeout", type=float, default=1500.0)
    p.add_argument("--no-video", action="store_true", help="no cameras: the score alone (less GPU memory)")
    p.add_argument("--min-free-mib", type=int, default=3000, help="start a scene only with this much GPU memory free")
    p.add_argument("--setting", default="card", choices=("card", "varied"),
                   help="varied: the agent's scene of the seed (cases.SETTINGS: its room, colours, three everyday things)")
    a = p.parse_args(argv)
    global SETTING
    SETTING = a.setting
    if a.cmd == "check":
        bad = 0
        for lv in a.levels:
            for s in a.seeds:
                r = check(a.case, lv, s)
                bad += not r["ok"]
                print(f"{a.case} {lv} s{s}: {'clear' if r['ok'] else 'NOT clear'}", flush=True)
                for x in r["problems"]:
                    print("  -", x)
                for x in r["warnings"]:
                    print("  (warning)", x)
        return 1 if bad else 0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "child":  # one scene, then leave without Kit's teardown
        code = 0
        try:
            r = play(a.case, a.levels[0], a.seeds[0], out, video=not a.no_video)
            print(json.dumps({k: r.get(k) for k in ("success", "milestones_hit", "missed_grips")}), flush=True)
        except Exception:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            code = 1
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    # run: one Isaac process per scene, one at a time, each when the GPU has room for it (it is shared: a process
    # that started without room ran out of memory loading the scene) and once more if it ran out all the same
    (out / "logs").mkdir(exist_ok=True)
    failed = 0
    for lv in a.levels:
        for s in a.seeds:
            name = f"{a.case}-{lv}-s{s}"
            if (out / f"{name}.json").exists():
                continue
            t0 = time.time()
            for _ in range(4):
                _wait_for_gpu(a.min_free_mib)
                with open(out / "logs" / f"{name}.log", "w") as log:
                    proc = subprocess.Popen([sys.executable, __file__, "child", a.case, "--levels", lv, "--seeds",
                                             str(s), "-o", str(out), "--setting", a.setting]
                                            + (["--no-video"] if a.no_video else []),
                                            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                    try:
                        proc.wait(timeout=a.timeout)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                text = (out / "logs" / f"{name}.log").read_text(errors="replace").lower()
                if proc.returncode == 0 or not any(m in text for m in OUT_OF_MEMORY):
                    break
                print(f"{name}: out of GPU memory, once more", flush=True)
            f = out / f"{name}.json"
            r = json.loads(f.read_text()) if f.exists() else {}
            failed += proc.returncode != 0
            print(f"{name}: rc={proc.returncode} success={r.get('success')} milestones={r.get('milestones_hit')} "
                  f"({time.time() - t0:.0f} s)", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
