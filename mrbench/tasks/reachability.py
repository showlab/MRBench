"""Reach by inverse kinematics, not by the nominal reach radius: whether an arm's tool point gets to a point, its links
kept 1 cm clear of the table and out of the fixed bodies (its hand may touch them: the point may be on them).

``validate`` checks the design rules with the nominal reach (``RobotModel.reach_m``, fast); this checks a task's reach
claims the way a robot would meet them.  The two differ: a PiPER's nominal reach is 0.626 m, and its kinematics put
its tool point 0.75 m out near the table.

    mrbench validate --ik [TASK ...]
"""

from __future__ import annotations

import math
from typing import Optional

#: an arm's links (not its hand) keep this far from the table
TABLE_GAP = 0.01
#: the ways a hand may point: straight down, or pointing away from the base this far below level
PITCHES = (90.0, 75.0, 60.0, 45.0, 30.0, 0.0)
SEEDS = 10


def touches(scene, arm, q, base, point, hand_free: bool = False) -> Optional[str]:
    """What the arm at ``q`` runs into: its links within TABLE_GAP of the table, or any part of it inside a fixed body
    -- its hand only into one the point is on or in (a hand may not pass through a wall to a point beyond it)."""
    from ..robots import collision as col
    from ..robots import kinematics as kin

    p = kin.np.array([point])
    fixed = [(o.name, b, float(col.point_box(p, b).min()) < 0.03)
             for o in scene.objects if not o.dynamic for b in col.boxes_of(o, o.pose)]
    table = col.table_box(scene)
    for c in col.arm_capsules(arm.model, q, 1.0, base):
        hand = c.name in ("hand", "finger", "camera")
        if c.on_table:
            continue
        if not hand and col.capsule_box(c, table) < TABLE_GAP:
            return "the table"
        for name, b, at_target in fixed:
            if hand and (at_target or hand_free):
                continue
            if col.capsule_box(c, b) < 0.0:
                return name
    return None


def reaches(scene, slot: str, point, pitches=PITCHES, hand_free: bool = False) -> bool:
    """Whether ``slot``'s tool point gets to ``point`` with the hand pointing one of ``pitches`` ways.  With
    ``hand_free`` the hand may brush any fixed body (a point inside a crate, its walls round the hand): the lenient
    check, for claims that an arm *does* reach somewhere."""
    from ..policies.script import radial_tool, tool_quat
    from ..robots import get_model
    from ..robots import kinematics as kin

    np = kin.np
    arm = scene.arm(slot)
    m = get_model(arm.model)
    base = kin.transform(arm.pos, kin.rpy_matrix(0.0, 0.0, math.radians(arm.yaw_deg)))
    inv = np.linalg.inv(base)
    lo = np.array([a for a, _ in m.arm_limits])
    hi = np.array([b for _, b in m.arm_limits])
    rng = np.random.default_rng(0)
    seeds = [np.array(m.home)] + [rng.uniform(lo, hi) for _ in range(SEEDS - 1)]
    dx, dy = point[0] - arm.pos[0], point[1] - arm.pos[1]
    for pitch in pitches:
        if pitch >= 90.0:
            a = math.atan2(dx, -dy)
            quats = [tool_quat((0.0, 0.0, -1.0), (math.cos(a + k * math.pi / 2), math.sin(a + k * math.pi / 2), 0.0))
                     for k in range(2)]
        else:
            quats = [radial_tool(arm.pos, point, pitch)]
        for q in quats:
            target = inv @ kin.pose_matrix((*point, *q))
            for s in seeds:
                sol = kin.ik(arm.model, target, s, iters=250)
                if sol is not None and touches(scene, arm, sol, base, point, hand_free) is None:
                    return True
    return False


#: a privileged grip takes a body whose surface is this close to the tool point (``engines/isaac/puppeteer.py``)
GRIP_REACH = 0.02


def near_points(sc, slot: str, target, slack: float, n: int = 3) -> list:
    """The points from which ``slot`` would do something to ``target`` without reaching its middle: on a body, the
    ``n`` points nearest the arm's base on its surface pushed out by GRIP_REACH; at a place (a region, or a point
    (x, y, z) given as such), its middle moved ``slack`` toward the arm (a thing held by its near end, or a hand a
    grip's reach off it)."""
    from ..robots import collision as col
    from ..robots import kinematics as kin

    np = kin.np
    base = np.array(sc.arm(slot).pos[:2])
    low = sc.table_top_z + 0.03
    names = {o.name: o for o in sc.objects}
    if not isinstance(target, str):
        c = np.array(target, dtype=float)
    elif target in names:
        o = names[target]
        pts = []
        for b in col.boxes_of(o, o.pose):
            grown = col.Box(b.name, b.center, b.rot, b.half + GRIP_REACH)
            pts += [p for p in col.box_points(grown, 3)]
        # nearest first; distances rounded so that points the same distance off by symmetry, which differ in the last
        # bit by platform (macOS and Linux), keep the order they were made in
        pts = sorted(pts, key=lambda p: round(float(np.linalg.norm(p[:2] - base)), 9))[:n]
        return [(float(p[0]), float(p[1]), max(float(p[2]), low)) for p in pts]
    else:
        c = np.array(sc.point(target), dtype=float)
    d = base - c[:2]
    d = d / max(float(np.linalg.norm(d)), 1e-9)
    return [(float(c[0] + slack * d[0]), float(c[1] + slack * d[1]), max(float(c[2]), low + 0.01))]


def check(inst, slack: float = 0.0) -> list[str]:
    """The instance's reach claims that its arms' kinematics break -- the ones that make arms needed: an arm said not
    to reach a place that reaches it, two arms said to share no place that share a free one.  With ``slack``, an arm
    also breaks a claim if it reaches a body's surface within a grip's reach, or a place ``slack`` short of its
    middle (``near_points``): the margins a hand holding a thing by its end, or a privileged grip, has."""
    from ..robots import collision as col
    from ..robots import kinematics as kin

    sc = inst.scene
    out = []

    def at(name):
        p = list(sc.point(name))
        p[2] = max(p[2], sc.table_top_z + 0.04)
        return p

    for slot, target in inst.out_of_reach:
        if reaches(sc, slot, at(target)):
            out.append(f"{inst.spec.id}: {slot} reaches {target} (said not to)")
        elif slack > 0.0:
            for p in near_points(sc, slot, target, slack):
                if reaches(sc, slot, p, hand_free=True):
                    out.append(f"{inst.spec.id}: {slot} comes within reach of {target} at "
                               f"{[round(v, 2) for v in p]} (said not to; slack {slack:.2f} m)")
                    break
    # (the claims that an arm does reach somewhere are left to the scripted references, which go there: a body's
    # centre -- a door's, a toolbox's -- is not where a hand takes it)
    fixed = [b for o in sc.objects if not o.dynamic for b in col.boxes_of(o, o.pose)]
    for a, b in inst.apart:
        hit = shared_point(sc, a, b, fixed, slack)
        if hit is not None:
            pa_, pb_ = hit
            out.append(f"{inst.spec.id}: {a} and {b} both reach the middle -- {a} at {[round(v, 2) for v in pa_]}, "
                       f"{b} at {[round(v, 2) for v in pb_]} (said to share no place; slack {slack:.2f} m)")
    return out


#: where two arms said to share no place are looked for a place they share: heights over the table (m) -- up to
#: over a partition, where a thing can be handed across in the air -- and offsets from the middle between their bases
APART_Z = (0.04, 0.12, 0.20, 0.28, 0.36, 0.44, 0.52)
APART_ALONG = (0.0, -0.06, 0.06)
APART_ACROSS = (0.0, -0.15, 0.15, -0.3, 0.3, -0.45, 0.45)


def shared_point(sc, a: str, b: str, fixed, slack: float = 0.0):
    """A free place both arms reach -- with ``slack``, each from its own side, their tool points ``slack`` apart
    across a thing they pass between them -- as (a's point, b's point), or None."""
    from ..robots import collision as col
    from ..robots import kinematics as kin

    np = kin.np
    pa, pb = np.array(sc.arm(a).pos[:2]), np.array(sc.arm(b).pos[:2])
    u = (pb - pa) / max(float(np.linalg.norm(pb - pa)), 1e-9)  # along the line from a to b
    v = np.array([-u[1], u[0]])  # across it
    mid = (pa + pb) / 2
    half = slack / 2
    for z in APART_Z:
        for oa in APART_ALONG:
            for oc in APART_ACROSS:
                c = mid + oa * u + oc * v
                if abs(c[0]) > 1.6 or abs(c[1]) > 0.9:
                    continue
                p = (float(c[0]), float(c[1]), sc.table_top_z + z)
                if fixed and min(float(col.point_box(np.array([p]), f).min()) for f in fixed) < 0.03 + half:
                    continue
                qa = (p[0] - half * u[0], p[1] - half * u[1], p[2])
                qb = (p[0] + half * u[0], p[1] + half * u[1], p[2])
                if reaches(sc, a, qa) and reaches(sc, b, qb):
                    return qa, qb
    return None
