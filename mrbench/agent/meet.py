"""``meet``: where two arms can pass a thing from hand to hand -- a reasoning tool every row is given (docs/BASELINES.md).

A relay arm is needed only if the two arms on either side of it share no place, and that place may be in the air, not
only at the table (Rea-01's surplus version: west to north, above the partition).  This searches a grid between the
two bases, over the table and up to ``HEIGHTS``' top, for a place each arm reaches from its own side with their tool
points ``gap`` apart across the thing, the thing clear of fixed bodies.
"""

from __future__ import annotations

import math

GAP = 0.06
#: heights over the table top searched, lowest first; offsets along and across the line between the bases (m)
HEIGHTS = (0.05, 0.15, 0.25, 0.35, 0.45, 0.55)
ALONG = (0.0, -0.15, 0.15, -0.3, 0.3)
ACROSS = (0.0, -0.2, 0.2, -0.4, 0.4)
#: a point farther than this from an arm's base (horizontally) is not tried
FAR_M = 0.95
#: how the hands may point at a hand-over, tried in this order
POINTINGS = ("level", "tilted", "steep", "down")


def meeting_places(obs, a: str, b: str, gap: float = GAP, most: int = 3) -> dict:
    from ..robots import collision as col
    from ..robots import kinematics as kin

    np = kin.np
    sc = obs.case.inst.scene
    arms = {x.slot: x for x in sc.arms}
    for x in (a, b):
        if x not in arms:
            raise KeyError(f"no arm {x!r}; arms: {', '.join(arms)}")
    if a == b:
        raise ValueError("name two different arms")
    pa, pb = np.array(arms[a].pos[:2]), np.array(arms[b].pos[:2])
    u = (pb - pa) / max(float(np.linalg.norm(pb - pa)), 1e-9)
    v = np.array([-u[1], u[0]])
    mid = (pa + pb) / 2
    fixed = [bx for o in sc.objects if not o.dynamic for bx in col.boxes_of(o, o.pose)]
    half = gap / 2
    found, tried = [], 0
    for h in HEIGHTS:
        for oa in ALONG:
            for oc in ACROSS:
                c = mid + oa * u + oc * v
                if abs(c[0]) > 1.55 or abs(c[1]) > 0.85:
                    continue
                if min(float(np.linalg.norm(c - pa)), float(np.linalg.norm(c - pb))) > FAR_M or \
                        max(float(np.linalg.norm(c - pa)), float(np.linalg.norm(c - pb))) > FAR_M:
                    continue
                p = np.array([c[0], c[1], sc.table_top_z + h])
                if fixed and min(float(col.point_box(np.array([p]), f).min()) for f in fixed) < half + 0.03:
                    continue
                tried += 1
                qa, qb = p - half * np.append(u, 0.0), p + half * np.append(u, 0.0)
                wa = next((w for w in POINTINGS if obs.reach_once(a, tuple(qa), w, seeds=4)[0]), None)
                if wa is None:
                    continue
                wb = next((w for w in POINTINGS if obs.reach_once(b, tuple(qb), w, seeds=4)[0]), None)
                if wb is None:
                    continue
                found.append({"thing_at": [round(float(x), 3) for x in p],
                              "height_over_table_m": round(h, 2),
                              a: {"tool_point": [round(float(x), 3) for x in qa], "pointing": wa},
                              b: {"tool_point": [round(float(x), 3) for x in qb], "pointing": wb}})
                if len(found) >= most:
                    return {"arms": [a, b], "gap_m": gap, "places": found, "searched": tried}
    out = {"arms": [a, b], "gap_m": gap, "places": found, "searched": tried}
    if not found:
        out["note"] = (f"no place found among {tried} tried (heights {HEIGHTS[0]}-{HEIGHTS[-1]} m over the table, "
                       f"between the bases): a thing goes from {a} to {b} through a third arm, a place both reach "
                       f"at different times, or a thing that carries it")
    return out
