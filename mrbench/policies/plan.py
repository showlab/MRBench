"""Turn a choreography (``policies/script.py``) into joint trajectories, before anything moves.

Each arm is planned on its own (the arms' paths do not depend on one another).
Every move is sampled along its path -- a straight blend between poses, or the
script's own path, both on minimum-jerk timing -- and each sample is solved with
inverse kinematics seeded by the one before, so the joints move continuously.

* **Free moves:** a grasp may be taken either way round (half a turn about the
  approach); the way that leaves the wrist camera further forward (from the
  base) is preferred.  If the straight path loses the solution, the
  end pose is solved from many seeds and the arm goes there in joint space.
* **Looking ahead to a grip:** which joint solution a hand grips with decides
  whether it can follow the path it then has to keep to (pulling a door round its
  hinge, lifting a box over a cabinet).  So a free move with a grip ahead tries
  the solutions of its end pose -- both ways round, from many seeds -- and keeps
  the first from which everything after it can be followed without another jump:
  the moves that bring the hand in, the grip, and the held stretch to the release.
  If none does, the move goes the plain way and the next one tries -- so a change of
  joint solution happens as early as it can, while the hand is still out of the
  way.  The move that ends at the grip must come straight in along its path: a hand
  that swings round in joint space to its grip can knock what it goes for off its place.
* A waypoint that nothing reaches is an error here, before the engine runs.
* **A time bound, when asked for** (``plan(..., timeout_s=)``; the agent's program check sets one): the search for
  joint solutions has no bound of its own, and a held stretch of several joint-space moves can keep it going for
  hours.  Past the bound the plan stops with ``PlanTimeout`` -- no plan in time, which says nothing of reach.
  Without it (the scripted references, ``mrbench validate``) planning has no bound.

Engine-independent (needs numpy).
"""

from __future__ import annotations

import math
import time
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Optional

from ..core.errors import MRBenchError
from ..engines.base import Pose7, quat_mul
from ..robots import get_model
from ..robots import kinematics as kin
from .script import Clip, Grip, Move, Open, Phase, Release, Unclip, blend, smooth

#: joint trajectories are sampled this often (s)
SAMPLE_DT = 0.05
#: seeds tried for an end pose: when a straight path loses it, or to find a grip the held stretch can follow from
RESTARTS = 48
#: when none of those finds a solution: seeds this many, this spread (rad) round the configuration the arm is in
#: (from their own generator, so a plan that solves without them is unchanged)
LOCAL_RESTARTS, LOCAL_SPREAD, LOCAL_SEED = 24, 0.35, 7
#: how many of a joint-space move's nearest solutions a hold may end in (``_ArmPlanner.held_ok``)
HELD_BRANCHES = 4
#: two ways of moving that differ in joint travel by less than this (rad) count as the same: then the one that
#: leaves the wrist camera further forward is taken
TIE_RAD = 0.25
#: a move that turns any joint further than this (rad) is reported (``Plan.warnings``): it is usually a sign that
#: the scene asks for an awkward hand, and the scene should change
TURN_WARN_RAD = 2.0
HALF_TURN = (0.0, 0.0, 0.0, 1.0)


class PlanError(MRBenchError):
    pass


class PlanTimeout(PlanError):
    """The search ran past ``plan``'s ``timeout_s``: no plan found in time (not a verdict that a pose is out of reach)."""


@dataclass
class ArmPlan:
    times: list[float] = field(default_factory=list)
    qs: list[tuple[float, ...]] = field(default_factory=list)

    def at(self, t: float) -> tuple[float, ...]:
        """Joint targets at time ``t``: linear between samples."""
        i = bisect_right(self.times, t)
        if i == 0:
            return self.qs[0]
        if i >= len(self.times):
            return self.qs[-1]
        t0, t1 = self.times[i - 1], self.times[i]
        a = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
        return tuple(x + a * (y - x) for x, y in zip(self.qs[i - 1], self.qs[i]))


@dataclass
class Plan:
    arms: dict[str, ArmPlan]
    #: (time, phase index, note) at each phase's start
    phases: list[tuple[float, int, str]]
    #: (time, action) for every Grip, Release, Open, Clip and Unclip
    events: list[tuple[float, object]]
    #: planned tool point of every arm at the end of each phase (world frame), for comparing with what happened
    tcp_at_phase_end: list[dict[str, Pose7]]
    duration: float
    #: moves that turn a joint further than TURN_WARN_RAD
    warnings: list[str] = field(default_factory=list)


def _turned(pose: Pose7, flip: bool) -> Pose7:
    return (*pose[:3], *quat_mul(pose[3:], HALF_TURN)) if flip else pose  # type: ignore[return-value]


@dataclass
class _Step:
    """One move of one arm: when it runs and the sample fractions along it."""

    t0: float
    seconds: float
    phase: int
    note: str
    move: Move
    ss: list[float]


class _ArmPlanner:
    def __init__(self, arm, rng, deadline: Optional[float] = None):
        np = kin.np
        self.deadline = deadline  # time.monotonic() past which the search stops (None: no bound)
        self.slot, self.model = arm.slot, arm.model
        self.base = kin.transform(arm.pos, kin.rpy_matrix(0.0, 0.0, math.radians(arm.yaw_deg)))
        self.inv = np.linalg.inv(self.base)
        m = get_model(arm.model)
        self.lo, self.hi = np.array([a for a, _ in m.arm_limits]), np.array([b for _, b in m.arm_limits])
        self.home = np.array(m.home)
        self.rng = rng
        cam = m.wrist_camera
        self.camera = None if cam is None else np.array([cam.pos[0], cam.pos[1], cam.pos[2] + m.tcp_offset_m, 1.0])

    @staticmethod
    def travel(q0, q1) -> float:
        """How far the arm turns between two configurations: its furthest-turning joint (rad)."""
        return float(max(abs(a - b) for a, b in zip(q0, q1)))

    def rank(self, q0, q1) -> tuple:
        """Order for choosing among ways of moving: least joint travel first; within TIE_RAD of each other,
        the wrist camera further forward."""
        t = self.travel(q0, q1)
        return (round(t / TIE_RAD), self.camera_behind(q1), t)

    def camera_behind(self, q) -> float:
        """How far (cm, rounded) the wrist camera is nearer the arm's base, on the table, than the tool point: of a
        hand's two ways round, the smaller keeps the camera in front (or, for a hand tilted out from the base,
        on top of it rather than back toward the base)."""
        if self.camera is None:
            return 0.0
        np = kin.np
        ee = kin.fk(self.model, q)
        cam, tcp = ee @ self.camera, kin.tcp(self.model, q)[:3, 3]
        return round(100 * (float(np.hypot(*tcp[:2])) - float(np.hypot(*cam[:2]))))

    def world(self, q) -> Pose7:
        return kin.matrix_pose(self.base @ kin.tcp(self.model, q))

    def in_time(self) -> None:
        """Raise ``PlanTimeout`` once past the deadline (checked where the search branches: ``solutions``, ``held_ok``)."""
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise PlanTimeout("the planner's search ran out of time")

    def poses(self, step: _Step, start: Pose7, flip: bool) -> list[Pose7]:
        mv = step.move
        if callable(mv.to):
            return [_turned(mv.to(s), flip) for s in step.ss]
        return [blend(start, _turned(mv.to, flip), s) for s in step.ss]

    def follow(self, q0, poses) -> Optional[list]:
        """IK along ``poses``, each seeded by the one before; None if it loses the solution."""
        out, q = [], kin.np.array(q0)
        for p in poses:
            q = kin.ik(self.model, self.inv @ kin.pose_matrix(p), q)
            if q is None:
                return None
            out.append(tuple(float(v) for v in q))
        return out

    def solutions(self, pose: Pose7, q0) -> list:
        """Joint solutions of ``pose`` from ``q0`` and from random seeds, nearest-and-homeliest first."""
        self.in_time()
        np = kin.np
        target = self.inv @ kin.pose_matrix(pose)
        seeds = [np.array(q0)] + [self.rng.uniform(self.lo, self.hi) for _ in range(RESTARTS)]
        found = [q for q in (kin.ik(self.model, target, sd) for sd in seeds) if q is not None]
        if not found:  # random seeds can all stall at a limit (a Piper's six joints): try round where the arm is
            local = np.random.default_rng(LOCAL_SEED)
            near = [np.clip(np.array(q0) + local.normal(0.0, LOCAL_SPREAD, len(q0)), self.lo, self.hi)
                    for _ in range(LOCAL_RESTARTS)]
            found = [q for q in (kin.ik(self.model, target, sd) for sd in near) if q is not None]
        found.sort(key=lambda q: float(np.max(np.abs(q - np.array(q0))) + 0.5 * np.mean(np.abs(q - self.home))))
        unique: list = []
        for q in found:
            if all(np.max(np.abs(q - u)) > 1e-3 for u in unique):
                unique.append(q)
        return [tuple(float(v) for v in q) for q in unique]

    def held_ok(self, q, flip: bool, held: list[_Step]) -> bool:
        """From configuration ``q``, can the hand follow every move of ``held`` (the way in, then the held
        stretch) without changing joint solution?  A move made in joint space may end in any of its nearest few
        solutions: the nearest is tried first, the next ones only if what follows cannot be done from it."""
        self.in_time()
        for k, st in enumerate(held):
            if st.move.joints and not callable(st.move.to):  # made in joint space: only its end must be reachable
                ends = self.solutions(_turned(st.move.to, flip), q)
                return any(self.held_ok(e, flip, held[k + 1:]) for e in ends[:HELD_BRANCHES])
            path = self.follow(q, self.poses(st, self.world(q), flip))
            if path is None:
                return False
            q = path[-1]
        return True

    def plan_straight(self, q, flip, step: _Step, held_after: Optional[list[_Step]]):
        """The move that ends at a grip: along its path only (either way round), and from where it ends the
        held stretch must be followable.  (path, flip); path None if no such path."""
        start = self.world(q)
        found = []
        for f in _ways(flip, step):
            path = self.follow(q, self.poses(step, start, f))
            if path is not None and (held_after is None or self.held_ok(path[-1], f, held_after)):
                found.append((path, f))
        if not found:
            return None, flip
        return min(found, key=lambda pf: self.rank(q, pf[0][-1]))

    def plan_free(self, q, flip, step: _Step, held_after: Optional[list[_Step]], can_defer: bool = False):
        """A move with nothing in the hand.  Returns (path, flip); path None if nothing works.  With
        ``can_defer``, a move that finds no end configuration for everything ahead goes the plain way."""
        start = self.world(q)
        tries = []
        for f in _ways(flip, step):
            path = self.follow(q, self.poses(step, start, f))
            if path is not None:
                tries.append((path, f))
        # of the two ways round, the one that turns the arm least (and, as near as that, keeps the camera in front)
        if held_after is None:  # nothing to grip at the end: the straight path if it works
            if tries:
                return min(tries, key=lambda pf: self.rank(q, pf[0][-1]))
            if callable(step.move.to):
                return None, flip
            ends = [(e, f) for f in _ways(flip, step) for e in self.solutions(_turned(step.move.to, f), q)[:1]]
        else:  # the end of this move is a grip: its configuration must let the held stretch be followed
            for path, f in sorted(tries, key=lambda pf: self.rank(q, pf[0][-1])):
                if self.held_ok(path[-1], f, held_after):
                    return path, f
            if callable(step.move.to):
                return None, flip
            ends = [(e, f) for f in _ways(flip, step) for e in self.solutions(_turned(step.move.to, f), q)
                    if self.held_ok(e, f, held_after)]
            if not ends and can_defer:  # leave the choice to the next move in
                return self.plan_free(q, flip, step, None)
        if not ends:
            return None, flip
        end, f = min(ends, key=lambda ef: self.rank(q, ef[0]))
        return [tuple(a + s * (b - a) for a, b in zip(q, end)) for s in step.ss], f  # in joint space

    def plan_joints(self, q, flip, step: _Step, ahead: Optional[list[_Step]], fixed_turn: bool):
        """A move made in joint space: to the nearest configuration (of either way round, unless the hand holds
        something) from which ``ahead`` can be followed.  (path, flip); path None if the pose is out of reach."""
        ends, nearest = [], []
        for f in ((flip,) if fixed_turn else _ways(flip, step)):
            sols = self.solutions(_turned(step.move.to, f), q)
            nearest += [(e, f) for e in sols[:1]]
            for e in sols:
                if ahead is None or self.held_ok(e, f, ahead):
                    ends.append((e, f))
                    break  # the nearest that works, this way round
        ends = ends or nearest  # nothing lets what follows be followed: the nearest, and leave it to the next move
        if not ends:
            return None, flip
        end, f = min(ends, key=lambda ef: self.rank(q, ef[0]))
        return [tuple(a + s * (b - a) for a, b in zip(q, end)) for s in step.ss], f


def _ways(flip: bool, step: _Step) -> tuple[bool, ...]:
    """The ways round a move may go: the one it pins (``Move.turn``; ``allow_flip=False`` is ``turn=False``), or
    both, the present one first."""
    turn = step.move.turn if step.move.turn is not None else (None if step.move.allow_flip else False)
    return (turn,) if turn is not None else (flip, not flip)


def _rest_of_hold(items: list, k: int) -> list[_Step]:
    """The moves after move ``k`` up to the next release (what a hand holding something still has to do)."""
    out = []
    for later in items[k + 1:]:
        if isinstance(later, Release):
            break
        if isinstance(later, _Step):
            out.append(later)
    return out


def _ahead(items: list, k: int) -> Optional[list[_Step]]:
    """The moves after free move ``k`` up to the release that follows the next grip -- the ones its end
    configuration must let the hand follow -- or None if no grip is ahead before a release."""
    out: list[_Step] = []
    gripped = False
    for later in items[k + 1:]:
        if isinstance(later, Grip):
            gripped = True
        elif isinstance(later, Release):
            return out if gripped else None
        elif isinstance(later, _Step):
            out.append(later)
    return out if gripped else None


def plan(inst, script: list[Phase], dt: float = SAMPLE_DT, seed: int = 0, timeout_s: Optional[float] = None) -> Plan:
    np = kin.np
    if np is None:
        raise PlanError("planning a scripted reference needs numpy")
    rng = np.random.default_rng(seed)
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    t, phases, events = 0.0, [], []
    per_arm: dict[str, list] = {a.slot: [] for a in inst.scene.arms}  # slot -> [_Step | Grip | Release]
    for i, ph in enumerate(script):
        phases.append((t, i, ph.note))
        n = max(2, math.ceil(ph.seconds / dt))
        ss = [smooth(k / n) for k in range(1, n + 1)]
        for act in ph.actions:
            if isinstance(act, (Open, Clip, Unclip)):  # (a tool's clip: the hand holding it goes on as it was)
                events.append((t, act))
            elif isinstance(act, (Grip, Release)):
                events.append((t, act))
                per_arm[act.slot].append(act)
            else:
                per_arm[act.slot].append(_Step(t, ph.seconds, i, ph.note, act, ss))
        t += ph.seconds
    duration = t

    out: dict[str, ArmPlan] = {}
    warnings: list[str] = []
    planners = {a.slot: _ArmPlanner(a, rng, deadline) for a in inst.scene.arms}
    for arm in inst.scene.arms:
        ap = planners[arm.slot]
        q = tuple(float(v) for v in ap.home)
        flip, held = False, False
        arm_plan = ArmPlan([0.0], [q])
        items = per_arm[arm.slot]
        for k, item in enumerate(items):
            if isinstance(item, (Grip, Release)):
                held = isinstance(item, Grip)
                continue
            joints = item.move.joints and not callable(item.move.to)
            last_before_grip = k + 1 < len(items) and isinstance(items[k + 1], Grip)
            try:
                if joints and not last_before_grip:
                    ahead = _rest_of_hold(items, k) if held else _ahead(items, k)
                    path, flip = ap.plan_joints(q, flip, item, ahead or None, fixed_turn=held)
                elif held:
                    path = ap.follow(q, ap.poses(item, ap.world(q), flip))
                else:
                    held_after = _ahead(items, k)
                    if last_before_grip:  # the hand closes in on what it takes along its path, never swinging round
                        path, flip = ap.plan_straight(q, flip, item, held_after)
                        if path is None:
                            raise PlanError(f"{inst.spec.id}: phase {item.phase} ({item.note or 'no note'}): "
                                            f"{arm.slot} cannot come straight in to its grip")
                    else:
                        path, flip = ap.plan_free(q, flip, item, held_after, can_defer=True)
            except PlanTimeout:
                raise PlanTimeout(f"{inst.spec.id}: phase {item.phase} ({item.note or 'no note'}): {arm.slot}: the "
                                  f"planner's search ran out of time ({timeout_s:.0f} s) -- no plan found in time, "
                                  "which is not a verdict that the poses are out of reach") from None
            if path is None:
                mv = item.move
                where = "its path" if callable(mv.to) else f"{[round(v, 3) for v in mv.to[:3]]}"
                raise PlanError(f"{inst.spec.id}: phase {item.phase} ({item.note or 'no note'}): "
                                f"{arm.slot} cannot reach {where}")
            n = len(item.ss)
            for j, qk in enumerate(path, start=1):
                arm_plan.times.append(item.t0 + item.seconds * j / n)
                arm_plan.qs.append(qk)
            turned = [abs(a - b) for a, b in zip(q, path[-1])]
            if max(turned) > TURN_WARN_RAD:
                j = turned.index(max(turned))
                warnings.append(f"phase {item.phase} ({item.note or 'no note'}): {arm.slot} turns its joint {j + 1} "
                                f"by {math.degrees(max(turned)):.0f} degrees")
            q = path[-1]
        arm_plan.times.append(duration)
        arm_plan.qs.append(q)
        out[arm.slot] = arm_plan

    ends = []
    for _, nxt in zip(phases, phases[1:] + [(duration, len(phases), "")]):
        ends.append({s: planners[s].world(out[s].at(nxt[0] - 1e-9)) for s in out})
    return Plan(out, phases, events, ends, duration, warnings)
