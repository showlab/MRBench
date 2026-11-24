"""Physical information for the agent, and two physical queries.

* ``facts`` -- what a team would know or could look up about the physics: each arm's payload, how hard it can press
  and grip; each thing's rough mass, whether it is fixed, what it rests on; springs (stiffness, preload) and tight fits
  (the force that pulls them out); a friction estimate.
* ``check_hold`` -- a quasi-static check (as Holladay et al.'s ``Fixtured`` asks): does a body stay put under a load,
  held by presses, grips, weights or stops?  A body resting on the table may lift off or tip about any edge of its
  footprint, or slide; a hinged body (a door) turns about its hinge until something stops it.  Each answer is given
  twice: with the rough masses, and with masses 30 % lower and the load 50 % higher (the margin the prompt asks for).
* ``heft`` -- an arm lifts a thing a little and reads its joint torques: the mass to a few per cent, or "too heavy".

Approximations, said so in the replies: rigid bodies, point loads, one friction coefficient, arm capacities from the
makers' figures.
"""

from __future__ import annotations

import math
import random
from typing import Optional

from ..robots import get_model
from ..robots import kinematics as kin

G = 9.81
MU = 0.5
#: what each arm can do with its hand: rated payload (kg), how hard it can press down (N), grip force (N), and the
#: twist a grip resists about the line between the fingers (N m: friction over the finger pads)
CAPACITY = {"panda": {"payload_kg": 3.0, "press_n": 40.0, "grip_n": 70.0, "grip_twist_nm": 0.7,
                      "hand_stiffness_n_per_m": 1600.0},
            "piper": {"payload_kg": 1.5, "press_n": 15.0, "grip_n": 20.0, "grip_twist_nm": 0.2,
                      "hand_stiffness_n_per_m": 600.0},
            # no maker's figures (robots/models.py: the payload is not published; ~0.2 kg is an estimate): the
            # PiPER's, scaled by that estimate
            "so101": {"payload_kg": 0.2, "press_n": 2.0, "grip_n": 2.7, "grip_twist_nm": 0.03,
                      "hand_stiffness_n_per_m": 80.0}}
#: a hand is not rigid: pulled on, it gives (its joints are position-controlled springs).  Within a tolerance it
#: resists only stiffness x tolerance -- plus, for a press, the push it was already giving.  In Isaac a grip holding
#: one side of a pulled body lets it rise a few cm; a press near the pull holds it to a few mm.
TOLERANCE_M = 0.01
#: room a hand needs round the point it presses (m): its fingertips, and its palm just above them
HAND_HALF = 0.025
#: the margin: masses this much of their estimate, loads this much more
LIGHT, HEAVY = 0.7, 1.5
#: an arm holding a handle against a spring: the force it holds (N), as hard as it presses
HOLD_N = {m: c["press_n"] for m, c in CAPACITY.items()}
HEFT_SD = 0.03


def arm_capacity(model: str) -> dict:
    return dict(CAPACITY[model])


def _rests_on(case, obs, name: str) -> str:
    """What a body stands on: the table, or the shown thing whose top is just under its bottom."""
    box = obs.locate(name)
    bottom = box["box_min"][2]
    if bottom - case.inst.scene.table_top_z < 0.012:
        return "the table"
    best = None
    for t in obs.things():
        other = t["name"]
        if other == name or t["mass"] == "-":
            continue
        b = obs.locate(other)
        over = b["box_min"][0] < box["centre"][0] < b["box_max"][0] and b["box_min"][1] < box["centre"][1] < b["box_max"][1]
        if over and -0.03 < bottom - b["box_max"][2] < 0.03 or (over and b["box_min"][2] < bottom < b["box_max"][2]):
            best = other
    return best or "something under it"


def facts(case, obs) -> dict:
    """The physical side of the scene graph."""
    sc = case.inst.scene
    arms = [{"arm": a.slot, "robot": get_model(a.model).display_name, **arm_capacity(a.model)} for a in sc.arms]
    things = []
    for t in obs.things():
        if t["mass"] == "-":
            continue
        d = {"name": t["name"], "mass": t["mass"]}
        if t["mass"] != "fixed":
            d["rests_on"] = _rests_on(case, obs, t["name"])
        things.append(d)
    joints = []
    for h in sc.hinges:
        joints.append(_hinge_facts(case, obs, h))
    for f in sc.fits:
        joints.append({"thing": f.body, "kind": f"tight fit in {f.base}", "pulls_out_with_n": f.break_force_n,
                       "direction": "straight up (+z)"})
    return {"friction_estimate": MU, "arms": arms, "things": things, "joints": joints,
            "note": "rigid bodies, approximate figures: makers' ratings for the arms, rough masses for the things"}


# ---------------------------------------------------------------------------------------------- holds
#: a mass measured with ``heft`` needs only this margin
LIGHT_MEASURED = 0.95
#: a load this close to a body's middle line (m) is taken as on it: the cameras place it to a few millimetres
ON_AXIS = 0.01


def _mass(obs, name: str) -> float:
    if name in getattr(obs, "measured", {}):
        return obs.measured[name]
    if name not in obs.mass:
        raise KeyError(f"{name!r} is not a movable thing with a mass")
    return obs.mass[name]


def _light(obs, name: str) -> float:
    """The margin on a thing's mass: small once it has been weighed."""
    return LIGHT_MEASURED if name in getattr(obs, "measured", {}) else LIGHT


class NoModel(KeyError):
    """A hold ``check_hold`` has no model for.  It is told as unchecked (``"holds": "unknown"`` with the reason), not
    as an error and not as a verdict: the other holds of the call still answer."""


def _unchecked(h: dict, why: NoModel) -> dict:
    return {"hold": f"{h.get('kind')} by {h.get('by')}", "holds": "unknown", "note": why.args[0]}


def _holder(case, obs, h: dict, tol: float = TOLERANCE_M) -> tuple[str, float, float]:
    """(kind, downward force or friction capacity N, twist capacity N m) a hold gives, at rough figures, keeping the
    body within ``tol`` of where it was."""
    kind, by = h.get("kind"), h.get("by", "")
    arms = {a.slot: a for a in case.inst.scene.arms}
    if kind in ("press", "grip", "hold"):
        if by not in arms:
            raise KeyError(f"a {kind} is done by an arm, not {by!r}")
        cap = CAPACITY[arms[by].model]
        give = cap["hand_stiffness_n_per_m"] * tol
        if kind == "press":
            return kind, cap["press_n"] + give, 0.0
        if kind == "hold":
            return kind, HOLD_N[arms[by].model], 0.0
        return kind, min(2 * MU * cap["grip_n"], give), cap["grip_twist_nm"]
    if kind in ("weight", "stop"):
        own = next((x for x in case.inst.scene.hinges if x.body == by.partition(":")[0]), None)
        if own is not None and own.propped and not own.stiffness:
            # a thing on a pivot with no spring, which rests against what is in its way (Con-03's hooked stay, the
            # borrow version's prop): "it stays where it is only while something holds it, so it is no stop" is not
            # true of it and would read as a verdict against the prop.  The check has no model for it, and says so
            is_a = {"stop": "a free thing set on the table, kept in place by friction on its weight",
                    "weight": "a free thing set on the body, holding it down by its own weight"}[kind]
            raise NoModel(f"{by} turns on a hinge of its own, and check_hold has no model for that: a {kind} here is "
                          f"{is_a}. So this check cannot say whether {by} holds a body, which is no verdict against "
                          f"{by}: judge it from where its parts reach (locate, measure) and which way the body it "
                          f"holds pushes on it")
        if own is not None:
            raise KeyError(f"{by} turns on a hinge: it stays where it is only while something holds it, so it is no "
                           f"{kind}; a {kind} is a free thing set on the table")
        return kind, _mass(obs, by) * G, 0.0
    raise KeyError(f"a hold's kind is press, grip, hold (a handle), weight or stop, not {kind!r}")


def check_hold(case, obs, body: str, load: Optional[dict], holds: list[dict]) -> dict:
    sc = case.inst.scene
    body = body.partition(":")[0]  # a part names its body
    hinge = next((h for h in sc.hinges if h.body == body), None)
    if hinge is not None:
        return _check_hinged(case, obs, hinge, holds)
    return _check_resting(case, obs, body, load or {}, holds)


def _check_resting(case, obs, body: str, load: dict, holds: list[dict]) -> dict:
    """A body on the table under a load: lift-off, tipping about each footprint edge, sliding."""
    np = kin.np
    box = obs.locate(body)
    m_body = _mass(obs, body)
    at = load.get("at", box["centre"])
    f = load.get("force", [0.0, 0.0, 0.0])
    lo, hi = box["box_min"], box["box_max"]
    edges = {"west end (x min)": (0, lo[0], 1), "east end (x max)": (0, hi[0], -1),
             "south side (y min)": (1, lo[1], 1), "north side (y max)": (1, hi[1], -1)}
    # a load within ON_AXIS of the body's middle line, across its narrow side, is on that line
    at = list(at)
    narrow = 1 if (hi[1] - lo[1]) < (hi[0] - lo[0]) else 0
    mid = (lo[narrow] + hi[narrow]) / 2
    if abs(at[narrow] - mid) < ON_AXIS:
        at[narrow] = mid
    items = [("its own weight", "weight", m_body * G, 0.0, box["centre"][:2], _light(obs, body))]
    for h in holds:
        try:
            kind, cap, twist = _holder(case, obs, h)
        except NoModel as e:  # one verdict covers all the holds: with one of them unchecked there is none
            return {"body": body, "holds": "unknown", "unchecked": [_unchecked(h, e)],
                    "note": "one of the holds could not be checked, so whether the body stays put is not known"}
        p = h.get("at")
        if p is None:
            p = obs.locate(h["by"])["centre"] if kind == "weight" else box["centre"]
        if not (lo[0] - 0.01 <= p[0] <= hi[0] + 0.01 and lo[1] - 0.01 <= p[1] <= hi[1] + 0.01):
            return {"error": f"the {kind} by {h.get('by')} at {p[:2]} is not over {body} "
                             f"(x {lo[0]:.3f}..{hi[0]:.3f}, y {lo[1]:.3f}..{hi[1]:.3f})"}
        blocked = _no_room(case, obs, body, kind, h.get("by"), p)
        if blocked:
            return {"error": blocked}
        items.append((f"{kind} by {h.get('by')}", kind, cap, twist, p[:2],
                      _light(obs, h["by"]) if kind == "weight" else 1.0))

    def verdict(mass_k: float, load_k: float) -> dict:
        up = f[2] * load_k
        k_of = lambda k, light: (light if mass_k < 1.0 else 1.0) if k == "weight" else 1.0  # noqa: E731
        down = sum(c * k_of(k, li) for _, k, c, _, _, li in items)
        twist = sum(t for _, _, _, t, _, _ in items)
        out = {"lift_off": {"down_n": round(down, 1), "up_n": round(up, 1), "holds": down >= up}}
        worst = None
        for name, (axis, coord, sign) in edges.items():
            lever = lambda p: max(0.0, sign * (p[axis] - coord))  # noqa: E731 -- distance in from that edge
            keep = sum(c * k_of(k, li) * lever(p) for _, k, c, _, p, li in items) + twist
            tip = up * lever(at)
            ratio = keep / tip if tip > 1e-9 else math.inf
            if worst is None or ratio < worst[1]:
                worst = (name, ratio, keep, tip)
        out["tipping"] = {"weakest_edge": worst[0], "keeping_nm": round(worst[2], 2), "tipping_nm": round(worst[3], 2),
                          "holds": worst[1] >= 1.0}
        side = math.hypot(f[0], f[1]) * load_k
        normal = max(0.0, down - up)
        grips = sum(c for _, k, c, _, _, _ in items if k == "grip")
        out["sliding"] = {"push_n": round(side, 1), "friction_n": round(MU * normal + grips, 1),
                          "holds": side <= MU * normal + grips}
        out["holds"] = all(v["holds"] for v in out.values() if isinstance(v, dict))
        return out

    return {"body": body, "load": {"at": [round(v, 3) for v in at], "force_n": f},
            "holds_counted": [n for n, *_ in items],
            "masses_used_kg": {body: round(m_body, 2), **{h["by"]: round(_mass(obs, h["by"]), 2) for h in holds
                                                         if h.get("kind") == "weight"}},
            "with_rough_masses": verdict(1.0, 1.0),
            f"with_margin (rough masses x{LIGHT}, weighed ones x{LIGHT_MEASURED}, load x{HEAVY})": verdict(LIGHT, HEAVY),
            "tolerance_m": TOLERANCE_M,
            "note": "quasi-static, rigid body on the table; a hand gives like a spring: within the tolerance a press "
                    "resists its rated push plus stiffness x tolerance, a grip only stiffness x tolerance"}


def _no_room(case, obs, body: str, kind: str, by: str, p) -> Optional[str]:
    """Whether a press (a hand's fingertips) or a weight (its footprint, laid along the body or across it, either)
    fits at ``p`` on ``body``'s top: nothing standing up from it there.  For a weight that fits neither way, the
    nearest place along the body where it does is named."""
    if kind not in ("press", "grip", "weight"):
        return None
    from ..robots import collision as col

    sc = case.inst.scene
    surface = obs.locate(f"{body}:body")["top_z"] if f"{body}:body" in _parts(obs, body) else obs.locate(body)["top_z"]
    standing = []  # (name, x0, y0, x1, y1) of what stands up from the surface
    for o in sc.objects:
        if not o.dynamic or o.name == by:
            continue
        pose = obs.state["objects"].get(o.name, o.pose)
        for b in col.boxes_of(o, pose):
            pts = col.box_points(b, 3)
            if float(pts[:, 2].max()) > surface + 0.01:
                standing.append((b.name.replace(".", " "), pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(),
                                 pts[:, 1].max()))

    def hit(half, q):
        return next((n for n, x0, y0, x1, y1 in standing
                     if x0 - half[0] < q[0] < x1 + half[0] and y0 - half[1] < q[1] < y1 + half[1]), None)

    if kind != "weight":
        what = hit([HAND_HALF, HAND_HALF], p)
        return f"no room for a hand to {kind} at {[round(v, 3) for v in p[:2]]}: {what} stands up next to it" \
            if what else None
    size = sorted(obs.locate(by).get("size_along_own_axes", [0.05, 0.05, 0.05])[:2], reverse=True)
    lo_, hi_ = obs.locate(body)["box_min"], obs.locate(body)["box_max"]
    along = 0 if hi_[0] - lo_[0] >= hi_[1] - lo_[1] else 1
    ways = {"along the body": [size[0] / 2, size[1] / 2] if along == 0 else [size[1] / 2, size[0] / 2],
            "across the body": [size[1] / 2, size[0] / 2] if along == 0 else [size[0] / 2, size[1] / 2]}
    if any(hit(h, p) is None for h in ways.values()):
        return None
    what = hit(ways["along the body"], p)
    msg = f"no room for {by} at {[round(v, 3) for v in p[:2]]}, laid along or across: it runs into {what}"
    for k in range(1, 25):  # the nearest place along the body, on it, where it fits one way or the other
        for sgn in (-1, 1):
            q = list(p)
            q[along] += sgn * 0.005 * k
            if not lo_[along] <= q[along] <= hi_[along]:
                continue
            for way, h in ways.items():
                if hit(h, q) is None:
                    return msg + f"; it fits laid {way} at {[round(v, 3) for v in q[:2]]}"
    return msg


def _parts(obs, body: str) -> list:
    from .observe import PARTS

    return [f"{body}:{k}" for k in PARTS.get(body, {})]


# ---------------------------------------------------------------------------------------------- hinged bodies
#: the hinged body's material is sampled this finely (m), and a hold counts as on it within this of a sample
GRID_M, ON_BODY_M = 0.01, 0.015
#: a hold read off a point may put the body this far (deg) past a stop and still count as at the stop
STOP_SLACK_DEG = 5.0


class _Hinge:
    """A hinge's line and its body's material, closed (as built: angle 0)."""

    def __init__(self, case, obs, hinge):
        np = kin.np
        from ..robots import collision as col

        self.h = hinge
        self.a = np.array(hinge.axis, float) / np.linalg.norm(hinge.axis)
        self.opens = hinge.sign * self.a  # turning about this, right-handed, opens it
        self.p0 = np.array(hinge.pivot, float)
        o = case.inst.scene.obj(hinge.body)
        self.parts, weights = [], []
        for b in col.boxes_of(o, o.pose):
            n = [max(2, min(60, int(math.ceil(2 * h / GRID_M)) + 1)) for h in b.half]
            g = np.stack(np.meshgrid(*[np.linspace(-h, h, k) for h, k in zip(b.half, n)], indexing="ij"), -1)
            pts = b.center + g.reshape(-1, 3) @ b.rot.T
            self.parts.append((b.name.partition(".")[2] or b.name, pts))
            weights.append(float(np.prod(2 * b.half)) / len(pts))
        allpts = np.concatenate([q for _, q in self.parts])
        w = np.concatenate([np.full(len(q), wk) for (_, q), wk in zip(self.parts, weights)])
        self.centroid = (allpts * w[:, None]).sum(0) / w.sum()
        try:
            self.mass = _mass(obs, hinge.body)
        except KeyError:
            self.mass = o.mass_kg
        # how far open it is now (the snapshot's pose against the closed one)
        now = obs.state["objects"].get(o.name, o.pose)
        d = kin.pose_matrix(now)[:3, :3] @ kin.pose_matrix(o.pose)[:3, :3].T
        u = self._square(np.array([1.0, 0.0, 0.0]) if abs(self.a[0]) < 0.9 else np.array([0.0, 1.0, 0.0]))
        self.now_deg = round(self._angle(u, d @ u), 1) + 0.0  # no -0.0

    def _square(self, v):
        """``v``'s part square to the hinge line (``v`` one vector or rows of them)."""
        return v - kin.np.outer(v @ self.a, self.a).reshape(v.shape)

    def _angle(self, u, v) -> float:
        """Degrees from ``u`` to ``v`` (both square to the line), positive the way it opens."""
        np = kin.np
        return math.degrees(math.atan2(float(np.cross(u, v) @ self.opens), float(u @ v)))

    def turned(self, v, deg: float):
        """``v`` (from the hinge line) turned ``deg`` the way the body opens."""
        np = kin.np
        k, t = self.opens, math.radians(deg)
        return v * math.cos(t) + np.cross(k, v) * math.sin(t) + k * (k @ v) * (1 - math.cos(t))

    def place(self, at, open_deg: Optional[float] = None):
        """Where on the body a hand at ``at`` is: (part, radius m, angle open deg, the point with the body that far
        open), or an error.  ``at`` is taken as a point of the body held ``open_deg`` open; without it, as a point
        of the body wherever it is held, which says how far open that is."""
        np = kin.np
        d = np.array(at, float) - self.p0
        t = float(d @ self.a)
        sq = d - t * self.a
        r = float(np.linalg.norm(sq))
        found = []  # (angle open, part, the closed point's square offset)
        for name, pts in self.parts:
            dq = pts - self.p0
            tq = dq @ self.a
            sqq = dq - np.outer(tq, self.a)
            rq = np.linalg.norm(sqq, axis=1)
            near = np.hypot(rq - r, tq - t) < ON_BODY_M + GRID_M / 2
            for i in np.flatnonzero(near)[:: max(1, int(near.sum()) // 50)]:
                ang = self._angle(sqq[i], sq) if r > 1e-6 else 0.0
                found.append((ang, name, sqq[i]))
        reach = max(float(np.linalg.norm(self._square(q - self.p0), axis=1).max()) for _, q in self.parts)
        if not found:
            return {"error": f"nothing of {self.h.body} is {r:.3f} m from its hinge line at that place along it (it "
                             f"reaches {reach:.3f} m out): a hand there does not touch it"}
        lo, hi = -STOP_SLACK_DEG, self.h.open_max_deg + STOP_SLACK_DEG
        if open_deg is not None:
            deg = float(open_deg)
            # the material at that place along the line and that far out: as it is now, or as it is held
            ang, name, sqq = min(found, key=lambda f: min(abs(f[0] - self.now_deg), abs(f[0] - deg)))
            if not 0.0 <= deg <= self.h.open_max_deg:
                return {"error": f"{self.h.body} opens 0 to {self.h.open_max_deg:g} deg, not {deg:g}"}
        else:
            ok = [f for f in found if lo <= f[0] <= hi]
            if not ok:
                return {"error": f"with a hand there {self.h.body} would be {found[0][0]:.0f} deg open, past its stops "
                                 f"(0 to {self.h.open_max_deg:g} deg)"}
            ang, name, sqq = min(ok, key=lambda f: abs(f[0] - self.now_deg))
            deg = min(max(ang, 0.0), self.h.open_max_deg)
        held = self.p0 + t * self.a + self.turned(sqq, deg)
        return {"part": name, "r": r, "deg": deg, "held": held}

    def closing_nm(self, deg: float, mass_k: float = 1.0) -> tuple[float, float, float]:
        """(spring, its own weight, total) turning it shut at ``deg`` open, N m; negative turns it further open."""
        np = kin.np
        spring = self.h.stiffness * math.radians(deg + self.h.preload_deg)
        c = self.p0 + self.turned(self.centroid - self.p0, deg)
        weight = -float(np.cross(c - self.p0, np.array([0.0, 0.0, -self.mass * mass_k * G])) @ self.opens)
        return spring, weight, spring + weight

    def axis_text(self) -> str:
        a = self.a
        if abs(a[2]) > 0.9:
            return "vertical"
        if abs(a[2]) < 0.1:
            return "level, along " + ("x (west-east)" if abs(a[0]) >= abs(a[1]) else "y (south-north)")
        return f"slanted, along {[round(float(v), 2) for v in a]}"


def _ends(case, body: str):
    """What hinged ``body`` is at its 0 deg stop and at its far stop, in the task's words, where the task gives them
    (``TaskSetup.hinge_ends``).  The replies about such a body name the stop it turns to, not "shut" or "open": those
    count the hinge's turn from 0 deg, and Con-04's bleed valve is open at 0 deg."""
    return getattr(getattr(case, "setup", None), "hinge_ends", {}).get(body.partition(":")[0])


def _hinge_facts(case, obs, h) -> dict:
    hg = _Hinge(case, obs, h)
    ends = _ends(case, h.body)
    d = {"thing": h.body, "kind": f"hinge, {hg.axis_text()} axis", "axis_through": [round(v, 3) for v in h.pivot]}
    if ends:
        d["at_0_deg"], d[f"at_{h.open_max_deg:g}_deg"] = ends
    if hg.axis_text() == "vertical":
        d["opens"] = "turning " + ("counter-clockwise" if hg.opens[2] > 0 else "clockwise") + " seen from above"
    else:
        rise = float(kin.np.cross(hg.opens, hg.centroid - hg.p0)[2])
        d["opens"] = "its free side rising" if rise > 0 else "its free side falling"
    d.update({"spring_nm_per_rad": h.stiffness, "spring_preload_deg": h.preload_deg,
              "opens_at_most_deg": h.open_max_deg, "now_open_deg": round(hg.now_deg, 1)})
    turns = {}
    for deg in sorted({0.0, round(h.open_max_deg / 2), float(h.open_max_deg)}):
        spring, weight, total = hg.closing_nm(deg)
        turns[f"{deg:g} deg open"] = {"spring_nm": round(spring, 2), "own_weight_nm": round(weight, 2),
                                      "closing_nm": round(total, 2)}
    d["turning_it_back_to_0_deg" if ends else "turning_it_shut"] = turns
    d["note"] = ("spring = stiffness x (angle open + preload) in rad; own weight from its rough mass; a negative "
                 "closing torque turns it further open")
    if ends:
        d["note"] += ("; here 'open' counts its turn from the 0 deg stop and 'closing' is the turn back to that stop, "
                      "whatever the thing is there (at_0_deg says)")
    return d


def _check_hinged(case, obs, hinge, holds: list[dict]) -> dict:
    """A body on a hinge held at some angle open: by an arm on it, by a stop or prop set in its way, or by a weight set
    on it, against its spring and its own weight."""
    np = kin.np
    hg = _Hinge(case, obs, hinge)
    ends = _ends(case, hinge.body)
    if not holds:
        spring, weight, total = hg.closing_nm(hg.now_deg)
        if ends:
            to = f"back to its 0 deg stop, where {ends[0]}" if total >= 0 else \
                f"on to its {hinge.open_max_deg:g} deg stop, where {ends[1]}"
            return {"body": hinge.body, "holds": False,
                    "note": f"nothing holds it: its spring and its weight turn it {to} "
                            f"({abs(total):.2f} N m, at {hg.now_deg:.0f} deg now)"}
        return {"body": hinge.body, "holds": False,
                "note": f"nothing holds it: its spring and its weight turn it {'shut' if total >= 0 else 'open'} "
                        f"({total:.2f} N m at {hg.now_deg:.0f} deg open)"}
    out = []
    for h in holds:
        try:
            kind, cap, _ = _holder(case, obs, h)
        except NoModel as e:
            out.append(_unchecked(h, e))
            continue
        at = h.get("at") or obs.locate(h["by"])["centre"]
        pl = hg.place(at, h.get("open_deg"))
        if "error" in pl:
            out.append({"hold": f"{kind} by {h.get('by')}", "error": pl["error"]})
            continue
        r, deg, held = pl["r"], pl["deg"], pl["held"]
        # the way the held point moves as the body shuts: a stop or prop is pushed that way
        shut = -np.cross(hg.opens, held - hg.p0)
        shut = shut / (np.linalg.norm(shut) or 1.0)

        def verdict(mass_k, load_k):
            _, _, total = hg.closing_nm(deg)
            need = total * load_k
            push = shut if need >= 0 else -shut
            if kind == "stop":
                light = (_light(obs, h["by"]) if mass_k < 1.0 else 1.0)
                across, down = math.hypot(push[0], push[1]), max(0.0, -float(push[2]))
                if across <= MU * down + 1e-9:
                    resist = math.inf  # pushed down into the table: it does not slide
                else:
                    resist = MU * cap * light / (across - MU * down) * r
            elif kind == "weight":
                # the margin on its mass: 0.95 once it has been weighed, as the reply's heading says (as for a stop,
                # and a weight on a body on the table)
                light = (_light(obs, h["by"]) if mass_k < 1.0 else 1.0)
                w = -float(np.cross(held - hg.p0, np.array([0.0, 0.0, -cap * light])) @ hg.opens)  # turning it shut
                resist = max(0.0, -w if need >= 0 else w)
            else:
                resist = cap * r
            return {"closing_nm": round(need, 2),
                    "resisting_nm": "any (pushed into the table)" if resist == math.inf else round(resist, 2),
                    "holds": resist >= abs(need)}

        out.append({"hold": f"{kind} by {h.get('by')}", "on_part": pl["part"], "radius_from_hinge_m": round(r, 3),
                    "open_deg": round(deg, 1), "with_rough_masses": verdict(1.0, 1.0),
                    f"with_margin (rough masses x{LIGHT}, weighed ones x{LIGHT_MEASURED}, load x{HEAVY})":
                        verdict(LIGHT, HEAVY)})
    back, on = ("back to its 0 deg stop", "further from that stop") if ends else ("shut", "further open")
    res = {"body": hinge.body, "hinge": f"{hg.axis_text()} axis through {[round(v, 3) for v in hinge.pivot]}"}
    if ends:  # open_deg counts from the 0 deg stop: what the thing is at each stop
        res["stops"] = {"0 deg": ends[0], f"{hinge.open_max_deg:g} deg": ends[1]}
    res["holds"] = out
    res["note"] = (f"closing = the spring and the body's own weight turning it {back} at that angle (negative: turning "
                   f"it {on}); a hand resists by its rated push at its distance from the hinge line; a stop "
                   "or prop on the table by sliding friction on its weight plus whatever pushes it down (whether it "
                   "stands or tips is not checked); the body rests where the hold is")
    return res


# ---------------------------------------------------------------------------------------------- heft
def heft(case, obs, slot: str, thing: str) -> dict:
    """Lift ``thing`` a little with ``slot`` and read the joint torques."""
    np = kin.np
    arm = case.inst.scene.arm(slot)
    m = get_model(arm.model)
    box = obs.locate(thing)
    top = (box["centre"][0], box["centre"][1], box["top_z"] - 0.01)
    r = obs.check_reach(slot, top, "any")
    ok = [k for k, v in r["reachable"].items() if v.startswith("yes")]
    if not ok:
        return {"arm": slot, "thing": thing, "result": "the arm does not reach it from above", "reach": r["reachable"]}
    true = next(o.mass_kg for o in case.inst.scene.objects if o.name == thing)
    cap = CAPACITY[arm.model]["payload_kg"]
    if true > cap:
        return {"arm": slot, "thing": thing, "result": f"too heavy to lift: the joint torques reached their limits "
                                                      f"before it left the table (rated payload {cap:g} kg)"}
    base = kin.transform(arm.pos, kin.rpy_matrix(0.0, 0.0, math.radians(arm.yaw_deg)))
    from ..policies.script import tool_quat
    target = np.linalg.inv(base) @ kin.pose_matrix((*top, *tool_quat((0.0, 0.0, -1.0), (1.0, 0.0, 0.0))))
    q = kin.ik(arm.model, target, m.home, iters=300)
    rng = random.Random(f"heft/{case.inst.spec.seed_id}/{thing}/{slot}")
    measured = true * (1 + rng.gauss(0, HEFT_SD))
    torques = None
    if q is not None:
        _, jac = kin.tcp_and_jacobian(arm.model, q)
        w = np.zeros(6)
        w[:3] = base[:3, :3].T @ np.array([0.0, 0.0, -measured * G])
        torques = [round(float(t), 2) for t in jac.T @ w]
    if not hasattr(obs, "measured"):
        obs.measured = {}
    obs.measured[thing] = round(measured, 2)
    return {"arm": slot, "thing": thing, "mass_kg": round(measured, 2),
            "joint_torque_change_nm": torques, "rated_payload_kg": cap,
            "note": "lifted 2 cm and put back where it was; the mass is from the joint torques, to about 3 %"}
