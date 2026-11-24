"""What the agent is told, and what its queries answer -- only what the team could sense.

* **The scene:** the goal, what each thing on the table is with a rough mass (``MASS_SPREAD``), the parts of it a
  camera could pick out, marked places, and the few facts that do not show (a door's spring, a tight fit);
* **the arms:** each one's robot and where it is mounted (its own calibration), and from the snapshot its joint
  angles, hand pose, finger opening and joint torques;
* **locate:** a thing's (or a part's) bounding box from the cameras: the truth with a few millimetres' error, in
  place of a segmentation-plus-depth pipeline;
* **check_reach:** the arm's own kinematics -- whether inverse kinematics puts its hand at a point, pointing a
  given way.  Nothing about the scene: no collisions, no payload.

Never shown: the task card, roles, the scoring regions, the reach rules, the scripted reference.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Optional

from ..engines.base import quat_rotate
from ..robots import collision as col
from ..robots import get_model
from ..robots import kinematics as kin
from .cases import BLOCK_TEXT, Case

#: a rough mass is the true one times a factor in this range
MASS_SPREAD = (0.75, 1.3)
#: how far a located box is off: its centre (m, each axis), its turn (deg)
LOCATE_SD_M, LOCATE_SD_DEG = 0.004, 1.5
#: the parts of a body the cameras can pick out, by body name: shown name -> the body's parts it covers
PARTS = {"door_w": {"handle": ("handle",), "slab": ("slab",)}, "door_e": {"handle": ("handle",), "slab": ("slab",)},
         "toolbox": {"pull": ("pull",), "body": ("body",)},
         "strip": {"body": ("body", "top_w", "top_e", "top_mid"), "plug_w": ("plug_w",), "plug_e": ("plug_e",)}}
#: how the hand points, for ``check_reach``: name -> pitch below level (deg) of a hand pointing away from its base;
#: "down" is straight down
APPROACHES = {"down": 90.0, "steep": 60.0, "tilted": 40.0, "level": 0.0}
IK_SEEDS = 10
#: ``can_reach``'s margins: how far out and in they are looked for, in how many halvings, with how many IK seeds
MARGIN_OUT_M, MARGIN_IN_M, MARGIN_STEPS, MARGIN_SEEDS = 0.30, 0.50, 5, 4
#: ``measure`` lists at most this many pieces of one thing
MAX_PIECES = 12
#: an arm's links (not its hand) keep this far from the table, as the program check asks (``clearance.ENV_GAP``)
TABLE_GAP = 0.01

COMPASS = {0: "east (+x)", 90: "north (+y)", 180: "west (-x)", -90: "south (-y)", 270: "south (-y)", -180: "west (-x)"}
FRAME_TEXT = ("World frame: metres, x points east, y north, z up; the table's centre is at x = 0, y = 0, its top at "
              "z = 0.75, and it is 3.2 m long (x from -1.6 to 1.6) and 1.8 m wide (y from -0.9 to 0.9).")


def _r(v, n: int = 3):
    return [round(float(x), n) for x in v]


def _facing(yaw_deg: float) -> str:
    return COMPASS.get(int(round(yaw_deg)), f"{yaw_deg:.0f} deg from east")


class Observer:
    def __init__(self, case: Case, snapshot_dir: Path):
        self.case = case
        self.dir = Path(snapshot_dir)
        self.state = json.loads((self.dir / "state.json").read_text())
        rng = case.rng()
        # the same thing gets the same rough mass in every version of a task
        bodies = [o for o in case.inst.scene.objects if o.dynamic] + list(case.inst.scene.cloths)
        self.mass = {o.name: round(o.mass_kg * random.Random(f"mass/{case.inst.spec.seed_id}/{o.name}").uniform(*MASS_SPREAD),
                                   2 if o.mass_kg < 1 else 1) for o in bodies}
        self.cloths = {c.name: c for c in case.inst.scene.cloths}
        self._noise = {}
        self._rng = rng

    # ------------------------------------------------------------------ the scene
    def _pose(self, name: str):
        if name in self.state["objects"]:
            return self.state["objects"][name]
        return self.case.inst.scene.obj(name).pose

    def things(self) -> list[dict]:
        s, sc = self.case.setup, self.case.inst.scene
        out = []
        for name, what in s.bodies.items():
            if name in self.cloths:
                d = {"name": name, "what": what, "mass": f"about {self.mass[name]} kg, soft cloth",
                     "parts": list(self.cloths[name].keys)}
            else:
                o = sc.obj(name)
                d = {"name": name, "what": what, "mass": f"about {self.mass[name]} kg" if o.dynamic else "fixed"}
                if self._parts(name):
                    d["parts"] = list(self._parts(name))
            out.append(d)
        for name in self.case.blocks:
            out.append({"name": name, "what": BLOCK_TEXT, "mass": f"about {self.mass[name]} kg"})
        for name, (what, _) in s.groups.items():
            out.append({"name": name, "what": what, "mass": "fixed"})
        for name, what in s.places.items():
            out.append({"name": name, "what": what + " (painted on the table, flat)", "mass": "-"})
        return out

    def arm_list(self) -> list[dict]:
        out = []
        for a in self.case.inst.scene.arms:
            m = get_model(a.model)
            out.append({"arm": a.slot, "robot": m.display_name, "joints": m.arm_dof,
                        "fingers": f"two parallel fingers, opening up to {m.grip_open_m * 100:.0f} cm apart"
                        if m.grip_open_m else "parallel gripper",
                        "base": _r(a.pos), "faces": _facing(a.yaw_deg)})
        return out

    def briefing(self) -> dict:
        return {"goal": self.case.inst.instruction, "frame": FRAME_TEXT, "arms": self.arm_list(),
                "things": self.things(), "notes": list(self.case.setup.notes),
                "cameras": self.views()}

    def views(self) -> dict:
        names = {"main": "high over the south edge, looking down the whole table",
                 "top": "straight down from above (image up = north, right = east)",
                 "front": "low, from the south-east, at the middle of the table",
                 "west": "from beyond the west end", "east": "from beyond the east end",
                 "north": "from beyond the north edge, looking south"}
        out = {v: names.get(v, "") for v in self.state["views"] if not v.startswith("wrist_")}
        out.update({v: f"on the {v[6:]} arm's hand" for v in self.state["views"] if v.startswith("wrist_")})
        return out

    def image(self, view: str) -> Path:
        if view not in self.state["views"]:
            raise KeyError(f"no camera {view!r}; cameras: {', '.join(self.state['views'])}")
        return self.dir / f"{view}.jpg"

    # ------------------------------------------------------------------ the arms
    def _base(self, slot: str):
        a = self.case.inst.scene.arm(slot)
        return kin.transform(a.pos, kin.rpy_matrix(0.0, 0.0, math.radians(a.yaw_deg)))

    def proprio(self, slot: str) -> dict:
        a = self.case.inst.scene.arm(slot)
        m = get_model(a.model)
        st = self.state["arms"][slot]
        t = self._base(slot) @ kin.tcp(a.model, st["q"])
        z, y = t[:3, 2], t[:3, 1]
        d = {"arm": slot, "joint_angles_rad": _r(st["q"]), "joint_limits_rad": [_r(lim, 2) for lim in m.arm_limits],
             "hand_point": _r(t[:3, 3]), "hand_points_along": _r(z, 2), "fingers_open_along": _r(y, 2),
             "finger_opening_m": round(st["gripper"] * (m.grip_open_m or 0.0), 3),
             "joint_torques_nm": _r(st["torque"], 2)}
        if st.get("torque_limit"):
            d["joint_torque_limits_nm"] = _r(st["torque_limit"], 0)
        if a.model == "panda":
            d["torque_note"] = "the arm's own weight is compensated: torques show only outside loads (about 0 at rest)"
        return d

    # ------------------------------------------------------------------ perception
    def _jitter(self, key: str):
        if key not in self._noise:
            g = self._rng.gauss
            self._noise[key] = ([g(0, LOCATE_SD_M) for _ in range(3)], g(0, LOCATE_SD_DEG))
        return self._noise[key]

    def _parts(self, body: str) -> dict:
        return self.case.setup.parts.get(body) or PARTS.get(body, {})

    def _cloth_boxes(self, name: str, part: str) -> list:
        """A cloth as it lies (a flat box), or one of its named points (a 2 cm cube)."""
        np = kin.np
        c = self.cloths[name]
        if part:
            if part not in c.keys:
                raise KeyError(f"{name} has no part {part!r}; parts: {', '.join(c.keys)}")
            return [col.Box(f"{name}.{part}", np.array(c.key_point(part)), np.eye(3), np.array([0.01, 0.01, 0.01]))]
        rot = kin.rpy_matrix(0.0, 0.0, math.radians(c.yaw_deg))
        return [col.Box(name, np.array(c.pos), rot, np.array([c.size[0] / 2, c.size[1] / 2, 0.003]))]

    def _boxes(self, name: str) -> list:
        sc, s = self.case.inst.scene, self.case.setup
        body, _, part = name.partition(":")
        if body in self.cloths and body in s.bodies:
            return self._cloth_boxes(body, part)
        if body in s.groups:
            return [b for p in s.groups[body][1] for b in col.boxes_of(sc.obj(p), sc.obj(p).pose)]
        if any(r.name == body for r in sc.regions) and body in s.places:
            r = sc.region(body)
            np = kin.np
            return [col.Box(body, np.array(r.center), np.eye(3), np.array(r.half))]
        shown = set(s.bodies) | set(self.case.blocks)
        if body not in shown:
            raise KeyError(f"nothing called {body!r} is in view; things: {', '.join(t['name'] for t in self.things())}")
        boxes = col.boxes_of(sc.obj(body), self._pose(body))
        if part:
            covers = self._parts(body).get(part)
            if not covers:
                raise KeyError(f"{body} has no part {part!r} the cameras pick out; parts: "
                               f"{', '.join(self._parts(body)) or 'none'}")
            boxes = [b for b in boxes if b.name in {f"{body}.{p}" for p in covers}]
        return boxes

    def locate(self, name: str) -> dict:
        np = kin.np
        boxes = self._boxes(name)
        corners = []
        for b in boxes:
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz in (-1, 1):
                        corners.append(b.center + b.rot @ (b.half * np.array([sx, sy, sz])))
        pts = np.array(corners)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        dxyz, dyaw = self._jitter(name)
        out = {"name": name, "centre": _r((lo + hi) / 2 + dxyz), "box_min": _r(lo + dxyz), "box_max": _r(hi + dxyz),
               "top_z": round(float(hi[2] + dxyz[2]), 3)}
        if len(boxes) == 1 and boxes[0].half[2] > 0:
            b = boxes[0]
            x = b.rot[:, 0]
            out["size_along_own_axes"] = _r(2 * b.half)
            out["yaw_deg"] = round(math.degrees(math.atan2(x[1], x[0])) + dyaw, 1)
        return out

    # ------------------------------------------------------------------ self-model
    def reaches(self, slot: str, point, cache: dict) -> bool:
        """Whether ``slot``'s tool point gets to ``point`` any way round (``check_reach``), remembered in ``cache``."""
        key = (slot, tuple(round(v, 3) for v in point))
        if key not in cache:
            r = self.check_reach(slot, point, "any")
            cache[key] = any(v.startswith("yes") for v in r["reachable"].values())
        return cache[key]

    def check_reach(self, slot: str, xyz, approach: str = "any", across_deg: Optional[float] = None) -> dict:
        """Whether ``slot``'s hand gets its tool point (between the fingertips) to ``xyz`` pointing ``approach``."""
        from ..policies.script import radial_tool, tool_quat

        np = kin.np
        a = self.case.inst.scene.arm(slot)
        m = get_model(a.model)
        base = self._base(slot)
        inv = np.linalg.inv(base)
        dx, dy = xyz[0] - a.pos[0], xyz[1] - a.pos[1]
        horiz = math.hypot(dx, dy)
        names = list(APPROACHES) if approach == "any" else [approach]
        if any(n not in APPROACHES for n in names):
            raise KeyError(f"approach must be one of {', '.join(APPROACHES)} or any")
        rng = np.random.default_rng(0)
        lo = np.array([l for l, _ in m.arm_limits])
        hi = np.array([h for _, h in m.arm_limits])
        seeds = [np.array(m.home)] + [rng.uniform(lo, hi) for _ in range(IK_SEEDS - 1)]
        result = {}
        for n in names:
            pitch = APPROACHES[n]
            if pitch >= 90.0:
                acr = math.radians(across_deg if across_deg is not None else math.degrees(math.atan2(dx, -dy)))
                quats = [tool_quat((0.0, 0.0, -1.0), (math.cos(acr + k * math.pi / 2), math.sin(acr + k * math.pi / 2), 0.0))
                         for k in range(2)]
            else:
                quats = [radial_tool(a.pos, xyz, pitch)]
            ok, hit = None, None
            for q in quats:
                target = inv @ kin.pose_matrix((*xyz, *q))
                for s in seeds:
                    sol = kin.ik(a.model, target, s, iters=250)
                    if sol is None:
                        continue
                    hit = self._touches(a, sol, base, xyz)
                    if hit is None:
                        ok = sol
                        break
                if ok is not None:
                    break
            if ok is None:
                result[n] = f"no: every way there brings the arm against {hit}" if hit else "no"
            else:
                margin = float(np.min(np.minimum(ok - lo, hi - ok)))
                result[n] = "yes" + (" (a joint within 0.1 rad of its limit)" if margin < 0.1 else "")
        return {"arm": slot, "point": _r(xyz), "horizontal_distance_from_base_m": round(horiz, 3),
                "height_above_table_m": round(xyz[2] - 0.75, 3), "reachable": result,
                "note": "the arm kept 1 cm clear of the table and out of fixed things; not checked: the path there, "
                        "other arms, movable things, payload"}

    def reach_once(self, slot: str, xyz, approach: str, tool_len: float = 0.0, seeds: int = IK_SEEDS,
                   across_deg: Optional[float] = None) -> tuple[bool, str]:
        """Whether ``slot`` gets a point to ``xyz`` pointing ``approach``: its tool point, or with ``tool_len`` the tip
        of a straight tool that long held in its hand along the way the hand points.  (ok, what it says)."""
        from ..policies.script import radial_tool, tool_quat

        np = kin.np
        a = self.case.inst.scene.arm(slot)
        m = get_model(a.model)
        base = self._base(slot)
        inv = np.linalg.inv(base)
        dx, dy = xyz[0] - a.pos[0], xyz[1] - a.pos[1]
        pitch = APPROACHES[approach]
        if pitch < 90.0 and math.hypot(dx, dy) < 0.02:
            return False, "no: the point is over the base, where a hand cannot point away from it"
        if pitch >= 90.0:
            acr = math.radians(across_deg if across_deg is not None else math.degrees(math.atan2(dx, -dy)))
            quats = [tool_quat((0.0, 0.0, -1.0), (math.cos(acr + k * math.pi / 2), math.sin(acr + k * math.pi / 2), 0.0))
                     for k in range(2)]
        else:
            quats = [radial_tool(a.pos, xyz, pitch)]
        rng = np.random.default_rng(0)
        lo = np.array([l for l, _ in m.arm_limits])
        hi = np.array([h for _, h in m.arm_limits])
        starts = [np.array(m.home)] + [rng.uniform(lo, hi) for _ in range(seeds - 1)]
        hit = None
        for q in quats:
            pose = kin.pose_matrix((*xyz, *q))
            if tool_len:
                pose[:3, 3] -= tool_len * pose[:3, 2]  # the hand holds the tool's far end back along its pointing
            target = inv @ pose
            for s in starts:
                sol = kin.ik(a.model, target, s, iters=250)
                if sol is None:
                    continue
                hit = self._touches(a, sol, base, pose[:3, 3])
                if hit is None:
                    margin = float(np.min(np.minimum(sol - lo, hi - sol)))
                    return True, "yes" + (" (a joint within 0.1 rad of its limit)" if margin < 0.1 else "")
        return False, f"no: every way there brings the arm against {hit}" if hit else "no"

    def can_reach(self, slot: str, xyz, approach: str = "any", tool_len: float = 0.0) -> dict:
        """``check_reach`` with margins: for the first way that works, how much farther out (along the line from the
        arm's base, at the same height) it would still reach; if none works, how far short it falls along that line.
        Margins to ``MARGIN_STEPS`` halvings (about 1 cm out, 2 cm in)."""
        names = list(APPROACHES) if approach == "any" else [approach]
        if any(n not in APPROACHES for n in names):
            raise KeyError(f"point must be one of {', '.join(APPROACHES)} or any")
        a = self.case.inst.scene.arm(slot)
        dx, dy = xyz[0] - a.pos[0], xyz[1] - a.pos[1]
        horiz = math.hypot(dx, dy)
        ux, uy = (dx / horiz, dy / horiz) if horiz > 1e-6 else (1.0, 0.0)

        def along(d):  # the point moved d outward (d < 0: toward the base)
            return (xyz[0] + ux * d, xyz[1] + uy * d, xyz[2])

        def works(p, ways):
            return any(self.reach_once(slot, p, w, tool_len, seeds=MARGIN_SEEDS)[0] for w in ways)

        result = {n: self.reach_once(slot, xyz, n, tool_len)[1] for n in names}
        good = [n for n in names if result[n].startswith("yes")]
        if good:
            lo, hi = 0.0, MARGIN_OUT_M
            if works(along(hi), good[:1]):
                margin = f"at least {hi:.2f} m to spare"
            else:
                for _ in range(MARGIN_STEPS):
                    mid = (lo + hi) / 2
                    lo, hi = (mid, hi) if works(along(mid), good[:1]) else (lo, mid)
                margin = f"about {lo:.2f} m to spare outward, pointing {good[0]}"
        else:
            lo, hi = 0.0, min(MARGIN_IN_M, horiz)
            if not works(along(-hi), names):
                margin = f"short by more than {hi:.2f} m along the line to its base (or blocked)"
            else:
                for _ in range(MARGIN_STEPS):
                    mid = (lo + hi) / 2
                    lo, hi = (lo, mid) if works(along(-mid), names) else (mid, hi)
                margin = f"about {hi:.2f} m short: the nearest point on the line to its base it reaches"
        out = {"arm": slot, "point": _r(xyz), "horizontal_distance_from_base_m": round(horiz, 3),
               "height_above_table_m": round(xyz[2] - 0.75, 3), "reachable": result, "margin": margin,
               "note": "the arm kept 1 cm clear of the table and out of fixed things; not checked: the path there, "
                       "other arms, movable things, payload"}
        if tool_len:
            out["holding"] = f"a straight tool {tool_len:.2f} m long, its tip at the point"
        return out

    def measure(self, name: str) -> dict:
        """A thing's (or a part's) sizes: each piece's size along its own axes and where it is, a few mm off."""
        loc = self.locate(name)
        pieces = []
        for b in self._boxes(name):
            dxyz, _ = self._jitter(f"measure/{b.name}")
            pieces.append({"piece": b.name.split(".", 1)[-1], "size_along_own_axes": _r(kin.np.maximum(2 * b.half + 0.5 * kin.np.array(dxyz), 0.001)),
                           "centre": _r(b.center + kin.np.array(dxyz))})
        lo, hi = loc["box_min"], loc["box_max"]
        return {"name": name, "world_box_size_xyz": _r([h - l for h, l in zip(hi, lo)]), "top_z": loc["top_z"],
                "pieces": pieces[:MAX_PIECES],
                **({"more_pieces": len(pieces) - MAX_PIECES} if len(pieces) > MAX_PIECES else {})}

    def _touches(self, arm, q, base, point=None) -> Optional[str]:
        """What the arm at ``q`` runs into, if anything: its links within ``TABLE_GAP`` of the table, or into a
        fixed body; its hand may touch the table and a fixed body the point is on, not pass through another."""
        sc = self.case.inst.scene
        shown = {p: g for g, (_, parts) in self.case.setup.groups.items() for p in parts}
        fixed = [(shown.get(o.name, o.name), b) for o in sc.objects if not o.dynamic for b in col.boxes_of(o, o.pose)]
        table = col.table_box(sc)
        p = kin.np.array([point]) if point is not None else None
        for c in col.arm_capsules(arm.model, q, 1.0, base):
            hand = c.name in ("hand", "finger", "camera")
            if c.on_table:
                continue
            if not hand and col.capsule_box(c, table) < TABLE_GAP:
                return "the table"
            for name, b in fixed:
                if hand and (p is None or float(col.point_box(p, b).min()) < 0.03):
                    continue
                if col.capsule_box(c, b) < 0.0:
                    return f"the {name}"
        return None
