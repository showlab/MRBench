"""Does a planned script keep every arm clear of every other arm and of the scene?

The plan is walked on a time grid.  At each instant every arm's joints and finger
opening come from the plan, every body that is held follows the hand that holds
it (as the privileged grip fixes it), a hinged body let go is back at its stop (a propped one -- ``HingeSpec.propped``
-- swings back only as far as the bodies in its way let it, and a held body coming into it pushes it open, onto its
stop at most; neither is measured against what pushes or holds it), and every pair is measured:

* arm with arm: ``ARM_GAP`` -- the arms track their plans to a centimetre or two;
* arm with the table, the static bodies and the bodies it is not working on: ``ENV_GAP``
  (a hand and its fingers are not measured against what they are about to take or hold);
* a held body with the static bodies, the other arms and the other bodies: ``HELD_GAP`` (a body resting on a
  held body -- a dish on a tray that is slid -- goes with it)
  (not against what it rests on, nor -- for the two moves after it is taken and the move
  that puts it down -- against what it touched when taken or touches when let go: a
  hanger lifted off its peg and hooked onto a rail, a plug pulled out of its strip.  Only
  the last move is excused before a release: a hanger carried to its rail may run
  into the rail's end).

What a held tool's clip takes (``script.Clip``, if it fits it: ``collision.clip_takes``, as the engine's) goes with
the tool and is measured as a held body is, but not against the tool; nor is the tool against it from the move before
the clip closes until a second and a half after it lets go.

A grip that attaches a rigid body must close on a part of it no wider across the fingers than they open
(``too_wide``): a privileged grip would otherwise fix a 24 cm ball to a hand whose fingers open 8 cm.  A body too
wide for one hand is taken only between two: when another hand is on it, on its far side (``collision.squeezed``),
the grip attaches it (to the hand that grips), as the engine's does; either hand letting go drops it there.

A grip on a cloth only closes the fingers here: cloth is soft, and is not measured; while a hand
works on a cloth its fingers may come down to the table under it (``PINCH_INTO_TABLE``).  So does a
grip that attaches nothing (a hand pressing a body down): the body stays where it is.

Engine-independent (needs numpy).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..robots import collision as col
from ..robots import get_model
from ..robots import kinematics as kin
from .script import Clip, Grip, Open, Release, Unclip

ARM_GAP, ENV_GAP, HELD_GAP = 0.03, 0.01, 0.015
#: a hand that pinches cloth lying on the table brings its fingertips down onto it: its finger capsules (1.2 cm
#: round the fingers) may then reach this far into the table -- the fingers themselves touch the cloth, not it
PINCH_INTO_TABLE = 0.012
STEP = 0.05
#: a body counts as resting on a box when its bottom is this close above the box's top
REST = 0.02
HAND = ("hand", "finger", "flange")
#: how near the tool point a body's shape must be for closing fingers to take it (as the privileged grip), and how
#: much wider than the fingers open a part may be and still be taken (a soft grip, rounding)
GRIP_REACH, FIT_TOL = 0.02, 0.003
#: a propped hinged body turns in steps of this many degrees, and is in a body's way within this gap of it
PROP_STEP, PROP_GAP = 1.0, 0.0


@dataclass(frozen=True)
class Clash:
    t: float
    phase: str
    a: str
    b: str
    gap: float

    def __str__(self) -> str:
        return f"t={self.t:.2f}s ({self.phase or 'no note'}): {self.a} and {self.b} {self.gap * 100:+.1f} cm apart"


def _bottom(boxes) -> float:
    return min(float((col.box_points(b, 3)[:, 2]).min()) for b in boxes)


def _rests_on(held, box: col.Box) -> bool:
    """Is a held body sitting on (or just above) ``box``?"""
    np = kin.np
    top = float(col.box_points(box, 3)[:, 2].max())
    if not -0.003 <= _bottom(held) - top <= REST:
        return False
    pts = np.vstack([col.box_points(b, 3) for b in held])
    lo, hi = col.box_points(box, 3).min(axis=0), col.box_points(box, 3).max(axis=0)
    return bool(((pts[:, 0] > lo[0]) & (pts[:, 0] < hi[0]) & (pts[:, 1] > lo[1]) & (pts[:, 1] < hi[1])).any())


def turned(h, closed, deg: float) -> tuple:
    """The pose of hinge ``h``'s body ``deg`` open, ``closed`` its pose at its stop."""
    np = kin.np
    axis = np.asarray(h.axis, float) / np.linalg.norm(h.axis)
    r = kin._rot(axis, math.radians(h.sign * deg))
    p = np.asarray(h.pivot, float)
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = r, p - r @ p
    return kin.matrix_pose(m @ kin.pose_matrix(closed))


def opened(h, closed, pose) -> float:
    """How far open (degrees) hinge ``h``'s body is at ``pose``, by its turn about the hinge's axis from ``closed``."""
    np = kin.np
    r = kin.pose_matrix(pose)[:3, :3] @ kin.pose_matrix(closed)[:3, :3].T
    axis = np.asarray(h.axis, float) / np.linalg.norm(h.axis)
    sin = 0.5 * float(axis @ np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]))
    cos = 0.5 * (float(np.trace(r)) - 1.0)
    return min(max(math.degrees(math.atan2(sin, cos)) * h.sign, 0.0), h.open_max_deg)


def settle(h, obj, deg: float, others: dict) -> tuple[float, set]:
    """Where propped hinge ``h``'s body (``obj``), let go ``deg`` open, is now, with the other bodies' boxes
    ``others`` (name -> boxes): turned on, if one of them is in it, till it is clear (pushed open); else back toward
    its stop as far as they let it.  Returns the angle and the bodies pushing or holding it there."""
    def in_way(d):
        boxes = col.boxes_of(obj, turned(h, obj.pose, d))
        return {n for n, obs in others.items() if any(col.box_box(b, ob) < PROP_GAP for b in boxes for ob in obs)}

    hit = in_way(deg)
    if hit:
        d = deg
        while d < h.open_max_deg:
            d = min(h.open_max_deg, d + PROP_STEP)
            if not in_way(d):
                return d, hit
        # it cannot get clear: pressed past its stop -- a hand pushes what it holds a little further than the stop
        # lets the body turn, as a loaded arm lags its plan -- it is on its stop, held there by what presses it
        return h.open_max_deg, hit
    while deg > 0.0:
        d = max(0.0, deg - PROP_STEP)
        hit = in_way(d)
        if hit:
            return deg, hit
        deg = d
    return 0.0, set()


def _reach(h, door) -> float:
    """How far from hinge ``h``'s pivot any of its body (``door``) reaches, at any angle (by bounding spheres)."""
    np = kin.np
    pivot = np.asarray(h.pivot, float)
    return max(float(np.linalg.norm(b.center - pivot) + np.linalg.norm(b.half)) for b in col.boxes_of(door, door.pose))


def _within(o, pose, h, reach: float) -> bool:
    """Can body ``o`` at ``pose`` be in the way of hinge ``h``'s body, ``reach`` round its pivot?"""
    np = kin.np
    pivot = np.asarray(h.pivot, float)
    return any(float(np.linalg.norm(b.center - pivot)) <= reach + float(np.linalg.norm(b.half))
               for b in col.boxes_of(o, pose))


def check(inst, plan, step: float = STEP) -> list[Clash]:
    """Every pair that comes closer than its margin, at the first instant it does."""
    touched = _walk(inst, plan, step, None)  # first pass: what each held body touches when taken and let go
    return _walk(inst, plan, step, touched)


def too_wide(inst, plan) -> list[str]:
    """The grips that would close on a part of a rigid body wider across the fingers than they open."""
    return [n for k, n in _walk(inst, plan, STEP, None).items() if k[0] == "too_wide"]


def _walk(inst, plan, step: float, touched):
    """Walk the plan.  With ``touched`` None, only note, for each grip and release, the static boxes
    the body is in contact with then (a dict); otherwise measure every pair (a list of Clash)."""
    np = kin.np
    scene = inst.scene
    arms = {a.slot: a for a in scene.arms}
    bases = {s: kin.transform(a.pos, kin.rpy_matrix(0.0, 0.0, np.radians(a.yaw_deg))) for s, a in arms.items()}
    statics = [col.table_box(scene)] + [b for o in scene.objects if not o.dynamic for b in col.boxes_of(o, o.pose)]
    statics += [col.prop_box(p) for p in scene.props]  # standing where no arm works: a plan never comes near
    hinged = {h.body for h in scene.hinges}
    propped = {h.body: h for h in scene.hinges if h.propped}
    reach = {b: _reach(h, scene.obj(b)) for b, h in propped.items()}
    ajar = {b: 0.0 for b in propped}  # how far open each propped body is while no hand holds it
    holders: dict[str, set] = {}  # a propped body -> the bodies pushing or holding it now
    events = sorted(plan.events, key=lambda e: e[0])
    spans = {}  # (slot, body) -> [from, to): when the hand works on that body
    approach = {}
    for t, act in events:
        if isinstance(act, Grip):
            starts = [p[0] for p in plan.phases if p[0] < t - 1e-9]
            approach[act.slot] = (starts[-1] if starts else 0.0, act.target)
            spans[(act.slot, act.target, t)] = None
    clip_windows = []  # (tool, target, from, to): a tool's clip working on a body -- not measured against it
    for t0, act in events:
        if isinstance(act, Clip):
            if (act.tool == act.target or act.tool not in {o.name for o in scene.dynamic_objects}
                    or col.clip_jaws(scene.obj(act.tool), scene.obj(act.tool).pose) is None):
                continue
            t1 = next((t for t, a in events if t > t0 and isinstance(a, Unclip) and a.tool == act.tool), plan.duration)
            before = max((p[0] for p in plan.phases if p[0] < t0 - 1e-9), default=0.0)
            before = max((p[0] for p in plan.phases if p[0] < before - 1e-9), default=0.0)
            clip_windows.append((act.tool, act.target, before, t1 + 1.5))

    def clipping(a, b, t) -> bool:
        return any({a, b} == {tool_, target} and lo <= t <= hi for tool_, target, lo, hi in clip_windows)

    windows = []
    for (slot, body, t0) in spans:
        t1 = next((t for t, a in events if t > t0 and isinstance(a, Release) and a.slot == slot), plan.duration)
        # from the start of the move before the grip, to a second and a half after letting go
        before = max((p[0] for p in plan.phases if p[0] < t0 - 1e-9), default=0.0)
        before = max((p[0] for p in plan.phases if p[0] < before - 1e-9), default=0.0)
        windows.append((slot, body, before, t1 + 1.5))

    def working_on(slot, body, t) -> bool:
        return any(s == slot and b == body and lo <= t <= hi for s, b, lo, hi in windows)

    poses = {o.name: o.pose for o in scene.dynamic_objects}
    opening = {s: 1.0 for s in arms}
    span = {s: sum(abs(o - c) for o, c in zip(m.gripper_open, m.gripper_closed))
            for s, m in ((s, __import__("mrbench.robots", fromlist=["get_model"]).get_model(a.model))
                         for s, a in arms.items())}
    held: dict[str, tuple[str, "np.ndarray"]] = {}  # slot -> (body, body in the tool frame)
    riders: dict[str, list] = {}  # slot -> [(body resting on what it holds, that body in the held body's frame)]
    clips: dict[str, tuple[str, "np.ndarray"]] = {}  # a tool -> (what its clip holds, that body in the tool's frame)
    clashes: dict[tuple[str, str], Clash] = {}
    cloths = {c.name for c in scene.cloths}
    notes: dict = {}  # (grip or release time, body) -> static boxes in contact then
    starts = [p[0] for p in plan.phases] + [plan.duration]
    ei = 0

    def phases_from(t0, n):  # the end of the n-th phase after the one starting at t0
        later = [x for x in starts if x > t0 + 1e-9]
        return later[min(n, len(later) - 1)] if later else plan.duration

    def phases_before(t1, n):
        earlier = [x for x in starts if x < t1 - 1e-9]
        return earlier[-n] if len(earlier) >= n else 0.0

    def in_contact(body):
        """The static boxes, and the boxes of other bodies, that ``body`` touches now."""
        boxes = col.boxes_of(scene.obj(body), poses[body])
        near = {st.name for st in statics if min(col.box_box(b, st) for b in boxes) < 0.02}
        for o in scene.dynamic_objects:
            if o.name != body:
                near |= {ob.name for ob in col.boxes_of(o, poses[o.name])
                         if min(col.box_box(b, ob) for b in boxes) < 0.02}
        return near

    def excused(body, st_name, t) -> bool:
        """Is ``body`` allowed to touch box ``st_name`` now: just taken off it (out of it), or about to be put on it?"""
        for (t_ev, b, kind), names in (touched or {}).items():
            if b != body or st_name not in names:
                continue
            if kind == "grip" and t_ev <= t <= phases_from(t_ev, 2):
                return True
            if kind == "release" and phases_before(t_ev, 1) <= t <= t_ev:
                return True
        return False

    def tool(slot, q):
        return bases[slot] @ kin.tcp(arms[slot].model, q)

    def through(slot, name, boxes, t) -> bool:
        """Does ``slot``'s hand work on propped body ``name`` through a body: one it holds that pushes or holds it, or
        may touch it now (the move that lets it go there), or one it is about to take that holds it?"""
        if name not in propped:
            return False
        hb = held.get(slot, (None,))[0]
        if hb is not None and (hb in holders.get(name, ()) or any(excused(hb, ob.name, t) for ob in boxes)):
            return True
        return any(working_on(slot, b, t) for b in holders.get(name, ()))

    def note(t):
        return next((n for t0, _, n in reversed(plan.phases) if t0 <= t + 1e-9), "")

    def clash(t, a, b, gap, need):
        if gap < need and (a, b) not in clashes:
            clashes[(a, b)] = Clash(round(t, 2), note(t), a, b, gap)

    for k in range(int(round(plan.duration / step)) + 1):
        t = k * step
        q = {s: plan.arms[s].at(t) for s in arms}
        while ei < len(events) and events[ei][0] <= t + 1e-9:
            et, act = events[ei]
            qs = plan.arms[act.slot].at(et) if isinstance(act, (Grip, Open, Release)) else None
            if isinstance(act, Grip):
                opening[act.slot] = min(1.0, act.width / span[act.slot]) if span[act.slot] else 0.0
                if act.target not in cloths and act.attach:
                    tm = tool(act.slot, qs)
                    obj = scene.obj(act.target)
                    w = col.grip_span(obj, poses[act.target], tm[:3, 3], tm[:3, 1], GRIP_REACH)
                    widest = get_model(arms[act.slot].model).grip_open_m
                    boxes = col.boxes_of(obj, poses[act.target])
                    on_it = [p for p in (tool(o, plan.arms[o].at(et))[:3, 3] for o in arms if o != act.slot)
                             if min(float(col.point_box(p[None], b)[0]) for b in boxes) <= GRIP_REACH]
                    if w is not None and widest and w > widest + FIT_TOL and not col.squeezed(
                            poses[act.target][:3], tm[:3, 3], on_it):
                        notes[("too_wide", et, act.target)] = (
                            f"t={et:.2f}s ({note(et)}): {act.slot} cannot take {act.target}: {w * 100:.1f} cm across "
                            f"its fingers, which open {widest * 100:.0f} cm")
                    rel = np.linalg.inv(tm) @ kin.pose_matrix(poses[act.target])
                    held[act.slot] = (act.target, rel)
                    notes[(et, act.target, "grip")] = in_contact(act.target)
                    # what stands on it goes with it (dishes on a tray that is slid)
                    under = col.boxes_of(scene.obj(act.target), poses[act.target])
                    t_inv = np.linalg.inv(kin.pose_matrix(poses[act.target]))
                    riders[act.slot] = [(o.name, t_inv @ kin.pose_matrix(poses[o.name])) for o in scene.dynamic_objects
                                        if o.name != act.target and o.name not in cloths
                                        and any(_rests_on(col.boxes_of(o, poses[o.name]), b) for b in under)]
            elif isinstance(act, Open):
                opening[act.slot] = min(1.0, act.width / span[act.slot]) if span[act.slot] else 1.0
            elif isinstance(act, (Clip, Unclip)):
                holder = next((s for s, (b, _) in held.items() if b == act.tool), None)
                tm = kin.pose_matrix(poses[act.tool])
                if holder is not None:  # where the tool is now, in the hand that holds it
                    tm = tool(holder, plan.arms[holder].at(et)) @ held[holder][1]
                    poses[act.tool] = kin.matrix_pose(tm)
                if isinstance(act, Clip):
                    # a static body (no pose here: it never moves) is never taken: the agent's check refuses it before
                    # planning; a task's own keyframes or `mrbench check` reach here with it
                    ok = act.target in poses and col.clip_takes(scene.obj(act.tool), poses[act.tool],
                                                                scene.obj(act.target), poses[act.target],
                                                                GRIP_REACH, FIT_TOL)[0]
                    if ok and holder is not None:
                        clips[act.tool] = (act.target, np.linalg.inv(tm) @ kin.pose_matrix(poses[act.target]))
                        notes[(et, act.target, "grip")] = in_contact(act.target)
                elif act.tool in clips and holder is not None:
                    body, rel = clips.pop(act.tool)
                    poses[body] = kin.matrix_pose(tm @ rel)
                    notes[(et, body, "release")] = in_contact(body)
            elif isinstance(act, Release):
                opening[act.slot] = 1.0
                riders.pop(act.slot, None)
                if act.slot in held:
                    body, rel = held.pop(act.slot)
                    poses[body] = kin.matrix_pose(tool(act.slot, qs) @ rel)
                    notes[(et, body, "release")] = in_contact(body)
                    if body in propped:  # its spring takes it back as far as what is in its way lets it
                        ajar[body] = opened(propped[body], scene.obj(body).pose, poses[body])
                    elif body in hinged:
                        poses[body] = scene.obj(body).pose  # its spring takes it back to its stop
            ei += 1
        for slot, (body, rel) in held.items():
            poses[body] = kin.matrix_pose(tool(slot, q[slot]) @ rel)
            for name, r in riders.get(slot, ()):
                poses[name] = kin.matrix_pose(kin.pose_matrix(poses[body]) @ r)
        for tool_, (body, rel) in clips.items():
            poses[body] = kin.matrix_pose(kin.pose_matrix(poses[tool_]) @ rel)
        for body, h in propped.items():
            if any(b == body for b, _ in held.values()):
                holders.pop(body, None)
                continue
            obj = scene.obj(body)
            near = {o.name: col.boxes_of(o, poses[o.name]) for o in scene.dynamic_objects
                    if o.name != body and o.name not in cloths and _within(o, poses[o.name], h, reach[body])}
            ajar[body], holders[body] = settle(h, obj, ajar[body], near)
            poses[body] = turned(h, obj.pose, ajar[body])
        if touched is None:
            continue
        caps = {s: col.arm_capsules(arms[s].model, q[s], opening[s], bases[s]) for s in arms}
        bodies = {o.name: col.boxes_of(o, poses[o.name]) for o in scene.dynamic_objects}
        held_bodies = {b for b, _ in held.values()}
        # what is held: by a hand, or by the clip of a tool a hand holds (with that tool: not measured against it)
        carried = [(slot, body, None) for slot, (body, _) in held.items()]
        carried += [(s, body, tool_) for tool_, (body, _) in clips.items() for s, (b, _) in held.items() if b == tool_]

        slots = list(arms)
        for i, s1 in enumerate(slots):  # arm with arm
            for s2 in slots[i + 1:]:
                for c in caps[s1]:
                    for d in caps[s2]:
                        if np.linalg.norm((c.a + c.b) / 2 - (d.a + d.b) / 2) > 1.0:
                            continue
                        clash(t, f"{s1}.{c.name}", f"{s2}.{d.name}", col.capsule_capsule(c, d), ARM_GAP)
        for s in slots:  # arm with the scene
            pinching = any(working_on(s, cl, t) for cl in cloths)
            for c in caps[s]:
                for b in statics:
                    if c.on_table and b.name == "table":
                        continue
                    if pinching and c.name in HAND and b.name == "table":  # fingertips down on cloth on the table
                        clash(t, f"{s}.{c.name}", b.name, col.capsule_box(c, b), -PINCH_INTO_TABLE)
                        continue
                    clash(t, f"{s}.{c.name}", b.name, col.capsule_box(c, b), ENV_GAP)
                for name, boxes in bodies.items():
                    if c.name in HAND and working_on(s, name, t):
                        continue
                    if name in held_bodies and held.get(s, (None,))[0] == name and c.name in HAND:
                        continue
                    if c.name in HAND and through(s, name, boxes, t):
                        continue  # it works on that propped body through what it holds or takes
                    for b in boxes:
                        clash(t, f"{s}.{c.name}", b.name, col.capsule_box(c, b), ENV_GAP)
        for slot, body, carrier in carried:  # what is held, with everything else
            if body in hinged:
                continue  # a door turns on its own hinge, next to its cabinet
            for b in bodies[body]:
                for st in statics:
                    if not _rests_on(bodies[body], st) and not excused(body, st.name, t):
                        clash(t, b.name, st.name, col.box_box(b, st), HELD_GAP)
                for other, boxes in bodies.items():
                    if other == carrier or clipping(body, other, t):
                        continue  # a clipped body works with its carrier
                    if other != body and body not in holders.get(other, ()):
                        for ob in boxes:
                            if not excused(body, ob.name, t):
                                clash(t, b.name, ob.name, col.box_box(b, ob), HELD_GAP)
                for s in slots:
                    if s != slot:
                        for c in caps[s]:
                            if c.name in HAND and working_on(s, body, t):
                                continue  # the other hand of two that hold it between them
                            clash(t, b.name, f"{s}.{c.name}", col.capsule_box(c, b), HELD_GAP)
    if touched is None:
        return notes
    return sorted(clashes.values(), key=lambda c: c.t)


def closest(inst, frames) -> dict:
    """What actually happened: from each frame's joint readings, the smallest gap between every two
    arms, between each arm and the static scene (the table, the cabinet), and between each arm and the
    props, over the whole run."""
    np = kin.np
    scene = inst.scene
    arms = {a.slot: a for a in scene.arms}
    bases = {s: kin.transform(a.pos, kin.rpy_matrix(0.0, 0.0, np.radians(a.yaw_deg))) for s, a in arms.items()}
    statics = [col.table_box(scene)] + [b for o in scene.objects if not o.dynamic for b in col.boxes_of(o, o.pose)]
    best: dict[str, tuple[float, float]] = {}

    def note(key, gap, t):
        if key not in best or gap < best[key][0]:
            best[key] = (round(gap, 4), round(t, 2))

    for f in frames:
        caps = {s: col.arm_capsules(arms[s].model, f.arms[s]["q"], max(0.0, f.arms[s]["gripper"]), bases[s])
                for s in arms}
        slots = list(arms)
        for i, s1 in enumerate(slots):
            for s2 in slots[i + 1:]:
                note(f"{s1}~{s2}", min(col.capsule_capsule(c, d) for c in caps[s1] for d in caps[s2]), f.t)
            note(f"{s1}~scene", min(col.capsule_box(c, b) for c in caps[s1] for b in statics
                                    if not (c.on_table and b.name == "table")), f.t)
            if scene.props:
                note(f"{s1}~props", min(col.capsule_box(c, col.prop_box(p, f.props.get(p.name)))
                                        for c in caps[s1] for p in scene.props), f.t)
    return {k: {"gap_m": g, "at_s": t} for k, (g, t) in sorted(best.items())}
