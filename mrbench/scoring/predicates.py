"""Success is judged from the object trajectory only, never from what a policy claims.

Two layers:

* **State predicates** (``Pred``) look at one Frame: ``InRegion``, ``OnTop``, ...
  Combine them with ``&``, ``|``, ``~``, ``all_of``, ``any_of``.
* **Temporal formulas** (``Temporal``) look at the whole list of frames and
  report the index at which they first become true: ``Eventually``,
  ``HoldsFor``, ``HoldsAtEnd``, ``Always``, ``Before``, ``AllOf``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import reduce
from typing import Optional, Sequence

from ..engines.base import Frame, SceneSpec, quat_rotate, quat_yaw_deg, transform

# --------------------------------------------------------------------------- state


class Pred:
    def __call__(self, frame: Frame, scene: SceneSpec) -> bool:
        raise NotImplementedError

    def __and__(self, other: "Pred") -> "Pred":
        return _And(self, other)

    def __or__(self, other: "Pred") -> "Pred":
        return _Or(self, other)

    def __invert__(self) -> "Pred":
        return _Not(self)


@dataclass(frozen=True)
class _And(Pred):
    a: Pred
    b: Pred

    def __call__(self, frame, scene):
        return self.a(frame, scene) and self.b(frame, scene)


@dataclass(frozen=True)
class _Or(Pred):
    a: Pred
    b: Pred

    def __call__(self, frame, scene):
        return self.a(frame, scene) or self.b(frame, scene)


@dataclass(frozen=True)
class _Not(Pred):
    a: Pred

    def __call__(self, frame, scene):
        return not self.a(frame, scene)


def all_of(*ps: Pred) -> Pred:
    return reduce(lambda a, b: a & b, ps)


def any_of(*ps: Pred) -> Pred:
    return reduce(lambda a, b: a | b, ps)


def tilt_deg(quat_wxyz: Sequence[float]) -> float:
    """Angle between the body z axis and world z."""
    _, x, y, _ = quat_wxyz
    cos_t = 1.0 - 2.0 * (x * x + y * y)
    return math.degrees(math.acos(min(max(cos_t, -1.0), 1.0)))


@dataclass(frozen=True)
class InRegion(Pred):
    obj: str
    region: str

    def __call__(self, frame, scene):
        p = frame.pose(self.obj, scene)
        r = scene.region(self.region)
        return all(abs(p[i] - r.center[i]) <= r.half[i] for i in range(3))


@dataclass(frozen=True)
class InBox(Pred):
    """Object centre within ``half`` of ``center`` on each axis: a box the scene does not name as a region (inside a
    covered channel)."""

    obj: str
    center: tuple[float, float, float]
    half: tuple[float, float, float]

    def __call__(self, frame, scene):
        p = frame.pose(self.obj, scene)
        return all(abs(p[i] - self.center[i]) <= self.half[i] for i in range(3))


@dataclass(frozen=True)
class EndIn(Pred):
    """One end of a long body lies in place: the middle of its -x (``sign`` -1) or +x (+1) end along its own x (its
    length from its spec, so a shorter body set down where a longer one goes does not pass), in region ``region``; or,
    with ``body``, within ``half`` of ``at`` in that body's frame (where it is now: inside a channel that lies a little
    off its own place)."""

    obj: str
    sign: float
    region: Optional[str] = None
    body: Optional[str] = None
    at: tuple[float, float, float] = (0.0, 0.0, 0.0)
    half: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __call__(self, frame, scene):
        p = frame.pose(self.obj, scene)
        q = transform(p, (self.sign * scene.obj(self.obj).size[0] / 2, 0.0, 0.0))
        if self.body is None:
            r = scene.region(self.region)
            return all(abs(q[i] - r.center[i]) <= r.half[i] for i in range(3))
        b = frame.pose(self.body, scene)
        loc = quat_rotate((b[3], -b[4], -b[5], -b[6]), tuple(q[i] - b[i] for i in range(3)))
        return all(abs(loc[i] - self.at[i]) <= self.half[i] for i in range(3))


@dataclass(frozen=True)
class Lifted(Pred):
    """Object centre at least ``dz`` above its initial height."""

    obj: str
    dz: float

    def __call__(self, frame, scene):
        return frame.pose(self.obj, scene)[2] >= scene.obj(self.obj).pos[2] + self.dz


@dataclass(frozen=True)
class Upright(Pred):
    obj: str
    max_deg: float = 15.0

    def __call__(self, frame, scene):
        return tilt_deg(frame.pose(self.obj, scene)[3:]) <= self.max_deg


@dataclass(frozen=True)
class Still(Pred):
    """A body within ``max_move`` of where it started and turned (about any axis) by at most ``max_deg``."""

    obj: str
    max_move: float = 0.01
    max_deg: float = 3.0

    def __call__(self, frame, scene):
        p, p0 = frame.pose(self.obj, scene), scene.obj(self.obj).pose
        if math.dist(p[:3], p0[:3]) > self.max_move:
            return False
        dot = min(1.0, abs(sum(a * b for a, b in zip(p[3:], p0[3:]))))
        return math.degrees(2 * math.acos(dot)) <= self.max_deg


@dataclass(frozen=True)
class OnBody(Pred):
    """``item`` rests on ``body``, wherever ``body`` has gone: its centre, in ``body``'s frame, within ``half_xy`` of
    the body's middle and between ``z_lo`` and ``z_hi`` above the body's origin (a dish on a tray that slides)."""

    item: str
    body: str
    half_xy: tuple
    z_lo: float
    z_hi: float

    def __call__(self, frame, scene):
        p, b = frame.pose(self.item, scene), frame.pose(self.body, scene)
        rel = quat_rotate((b[3], -b[4], -b[5], -b[6]), [x - y for x, y in zip(p[:3], b[:3])])
        return abs(rel[0]) <= self.half_xy[0] and abs(rel[1]) <= self.half_xy[1] and self.z_lo <= rel[2] <= self.z_hi


@dataclass(frozen=True)
class OnTop(Pred):
    """``top`` rests on ``bottom``: centred within ``xy_tol`` and touching (heights from the specs)."""

    top: str
    bottom: str
    xy_tol: float = 0.02
    z_tol: float = 0.01

    def __call__(self, frame, scene):
        a, b = frame.pose(self.top, scene), frame.pose(self.bottom, scene)
        gap = (scene.obj(self.top).height + scene.obj(self.bottom).height) / 2
        return math.dist(a[:2], b[:2]) <= self.xy_tol and abs((a[2] - b[2]) - gap) <= self.z_tol


@dataclass(frozen=True)
class InsideCylinder(Pred):
    """``item`` (any shape) is inside the open cylinder ``container`` (axis assumed vertical)."""

    item: str
    container: str

    def __call__(self, frame, scene):
        p, c = frame.pose(self.item, scene), frame.pose(self.container, scene)
        cr, ch = scene.obj(self.container).size[:2]
        ir = scene.obj(self.item).size[0] if scene.obj(self.item).shape != "box" else 0.0
        bottom = c[2] - ch / 2
        return math.dist(p[:2], c[:2]) <= max(cr - ir, 0.0) + 1e-3 and bottom <= p[2] <= c[2] + ch / 2


@dataclass(frozen=True)
class Inserted(Pred):
    """Peg axis within ``xy_tol`` of the socket centre and its bottom ``depth`` below the socket top."""

    peg: str
    socket: str
    xy_tol: float
    depth: float

    def __call__(self, frame, scene):
        p, s = frame.pose(self.peg, scene), frame.pose(self.socket, scene)
        peg_bottom = p[2] - scene.obj(self.peg).height / 2
        socket_top = s[2] + scene.obj(self.socket).height / 2
        return math.dist(p[:2], s[:2]) <= self.xy_tol and peg_bottom <= socket_top - self.depth


@dataclass(frozen=True)
class Opened(Pred):
    """A hinged body has turned at least ``min_deg`` from where it started, about whatever axis (a door, a lid)."""

    obj: str
    min_deg: float

    def __call__(self, frame, scene):
        q, q0 = frame.pose(self.obj, scene)[3:], scene.obj(self.obj).pose[3:]
        half_cos = min(1.0, abs(sum(a * b for a, b in zip(q, q0))))  # |q . q0| = cos(turn / 2)
        return 2.0 * math.degrees(math.acos(half_cos)) >= self.min_deg


@dataclass(frozen=True)
class Level(Pred):
    """A body's x axis within ``max_deg`` of level (a leaf, a beam)."""

    obj: str
    max_deg: float

    def __call__(self, frame, scene):
        x = quat_rotate(frame.pose(self.obj, scene)[3:], (1.0, 0.0, 0.0))
        return abs(math.degrees(math.asin(max(-1.0, min(1.0, x[2]))))) <= self.max_deg


@dataclass(frozen=True)
class ThroughSlots(Pred):
    """A pin -- a key along its body's y, ``length`` long, ``section`` (across, thick) -- runs through every slot.
    A slot is (body, its middle in the body's frame, its hole's size along the body's x and z): the key covers the
    slot's middle with ``reach`` to spare each way, and where it passes the slot its cross-section lies inside the
    hole (``tol`` for contact).  Through two slots on two bodies, the key pins them together."""

    pin: str
    length: float
    section: tuple
    slots: tuple
    reach: float = 0.01
    tol: float = 0.003

    def __call__(self, frame, scene):
        p = frame.pose(self.pin, scene)
        axis = quat_rotate(p[3:], (0.0, 1.0, 0.0))
        for body, centre, hx, hz in self.slots:
            b = frame.pose(body, scene)
            c = transform(b, centre)
            t = sum((ci - pi) * a for ci, pi, a in zip(c, p[:3], axis))
            if abs(t) > self.length / 2 - self.reach:
                return False
            on_axis = [pi + t * a for pi, a in zip(p[:3], axis)]
            back = (b[3], -b[4], -b[5], -b[6])
            rel = quat_rotate(back, [o - ci for o, ci in zip(on_axis, c)])
            if abs(rel[0]) > (hx - self.section[0]) / 2 + self.tol or abs(rel[2]) > (hz - self.section[1]) / 2 + self.tol:
                return False
        return True


@dataclass(frozen=True)
class ClothAbove(Pred):
    """Every mark of a cloth (or only its named ``keys``) at least ``dz`` above the table top."""

    cloth: str
    dz: float
    keys: tuple[str, ...] = ()

    def __call__(self, frame, scene):
        marks = frame.cloths[self.cloth]
        if self.keys:
            spec = scene.cloth(self.cloth)
            marks = [marks[spec.key(k)] for k in self.keys]
        return all(p[2] >= scene.table_top_z + self.dz for p in marks)


@dataclass(frozen=True)
class DrapedOver(Pred):
    """A cloth hangs over a horizontal bar -- part ``part`` of body ``obj``, a box along the
    part's y: nothing of it above the bar's top, its highest mark at the bar's top (within a
    mark-spacing), at least ``min_side`` of its marks on each side of the bar, and all of it
    within the bar's length; with ``min_along``, its marks spread at least that far along the
    bar (hung spread out, not bunched up where it was held)."""

    cloth: str
    obj: str
    part: str
    min_side: float = 0.3
    tol: float = 0.02
    min_along: float = 0.0

    def __call__(self, frame, scene):
        pose = frame.pose(self.obj, scene)
        part = scene.obj(self.obj).part(self.part)
        centre = transform(pose, part.pos)
        h = math.radians(part.yaw_deg)
        axis = quat_rotate(pose[3:], (-math.sin(h), math.cos(h), 0.0))  # the part's y, in the world
        across = (axis[1], -axis[0], 0.0)  # horizontal, square to the bar
        n = math.hypot(*across[:2])
        if n < 0.5:  # the bar is not lying level enough to hang anything on
            return False
        across = (across[0] / n, across[1] / n, 0.0)
        top = max(part.size[0], part.size[2]) / 2
        half = part.size[1] / 2
        ups, sides, along = [], [], []
        for p in frame.cloths[self.cloth]:
            d = tuple(a - b for a, b in zip(p, centre))
            a = sum(x * y for x, y in zip(d, axis))
            if abs(a) > half + self.tol:
                return False
            along.append(a)
            ups.append(d[2])
            sides.append(sum(x * y for x, y in zip(d, across)))
        spec = scene.cloth(self.cloth)
        # the fold may fall between two marks: the highest one can sit a mark-spacing below the top
        if max(ups) > top + 2 * spec.cell + self.tol or max(ups) < top - spec.cell * spec.mark_every - self.tol:
            return False
        if max(along) - min(along) < self.min_along:
            return False
        k = self.min_side * len(sides)
        return sum(s < 0 for s in sides) >= k and sum(s > 0 for s in sides) >= k


@dataclass(frozen=True)
class ClothOver(Pred):
    """A long cloth (a cord) lies over a horizontal bar -- part ``part`` of body ``obj``, a box along the part's y --
    and hangs down on both sides of it: one of its marks on the bar's top, within its length (``tol`` more) and
    within a mark-spacing of it across, and on each side, further than that across, marks below the bar's top.
    (``DrapedOver`` is for a towel hung wholly on a bar.)"""

    cloth: str
    obj: str
    part: str
    tol: float = 0.01

    def __call__(self, frame, scene):
        pose = frame.pose(self.obj, scene)
        part = scene.obj(self.obj).part(self.part)
        centre = transform(pose, part.pos)
        h = math.radians(part.yaw_deg)
        axis = quat_rotate(pose[3:], (-math.sin(h), math.cos(h), 0.0))
        across = (axis[1], -axis[0], 0.0)
        n = math.hypot(*across[:2])
        if n < 0.5:
            return False
        across = (across[0] / n, across[1] / n, 0.0)
        top = part.size[2] / 2
        half = part.size[1] / 2
        spec = scene.cloth(self.cloth)
        spacing = spec.cell * spec.mark_every
        on, below = False, {-1: False, 1: False}
        for p in frame.cloths[self.cloth]:
            d = tuple(a - b for a, b in zip(p, centre))
            a, c, z = sum(x * y for x, y in zip(d, axis)), sum(x * y for x, y in zip(d, across)), d[2]
            if abs(c) <= spacing and abs(a) <= half + self.tol and top - spacing / 2 <= z <= top + 2 * spec.cell + self.tol:
                on = True
            elif abs(c) > spacing and z < top:
                below[1 if c > 0 else -1] = True
        return on and below[-1] and below[1]


@dataclass(frozen=True)
class ChainStands(Pred):
    """Things standing in a line along x that will fall one onto the next: of ``names``, those upright (within
    ``max_tilt``) with their middles within ``across`` of y = ``line_y`` and between the ends, in order along x, the
    first at ``x0`` and
    the last at ``x1`` (within ``end_tol``; each measured at its foot, the middle of its bottom face, which a domino
    leaning as it is pushed keeps where it stood while its middle goes on by half its height times the lean's sine:
    1.5 cm at 14 degrees), the gap between each one's faces and the next's less than its height (falling, it reaches
    that far) by ``margin``.  A domino line with a place left empty is broken; one with a plank
    tall enough to fall across the empty place is not.  (``max_tilt`` as the tilt past which a domino counts as
    fallen: checked just before the first one falls, that one is already leaning, pushed.)"""

    names: tuple[str, ...]
    line_y: float
    x0: float
    x1: float
    across: float = 0.03
    max_tilt: float = 15.0
    end_tol: float = 0.02
    margin: float = 0.01
    #: if set: each one, falling about its far bottom edge, also strikes the next one's face at least this far below
    #: the next one's top (one that comes down on the next one's top corner, or just under it, presses it down instead
    #: of tipping it on)
    below_top: Optional[float] = None

    def __call__(self, frame, scene):
        up = []
        for nm in self.names:
            p = frame.pose(nm, scene)
            if abs(p[1] - self.line_y) > self.across or tilt_deg(p[3:]) > self.max_tilt \
                    or not self.x0 - self.end_tol <= p[0] <= self.x1 + self.end_tol:
                continue
            o = scene.obj(nm)
            yaw = math.radians(quat_yaw_deg(p[3:]))
            thick = abs(math.cos(yaw)) * o.size[0] + abs(math.sin(yaw)) * o.size[1]  # along x
            foot = p[0] + quat_rotate(p[3:], (0.0, 0.0, -o.height / 2))[0]
            up.append((p[0], thick, o.height, foot))
        up.sort()
        if not up or abs(up[0][3] - self.x0) > self.end_tol or abs(up[-1][3] - self.x1) > self.end_tol:
            return False
        for a, b in zip(up, up[1:]):
            gap = (b[0] - b[1] / 2) - (a[0] + a[1] / 2)
            if gap >= a[2] - self.margin:
                return False
            if self.below_top is not None and math.sqrt(max(0.0, a[2] ** 2 - gap ** 2)) > b[2] - self.below_top:
                return False
        return True


@dataclass(frozen=True)
class DownOnLine(Pred):
    """Every one of ``names`` that is on the line (its middle within ``across`` of y = ``line_y``, between ``x0`` and
    ``x1``) has tilted past ``min_deg``."""

    names: tuple[str, ...]
    line_y: float
    x0: float
    x1: float
    min_deg: float = 15.0
    across: float = 0.08

    def __call__(self, frame, scene):
        for nm in self.names:
            p = frame.pose(nm, scene)
            if abs(p[1] - self.line_y) <= self.across and self.x0 - 0.05 <= p[0] <= self.x1 + 0.15 \
                    and tilt_deg(p[3:]) < self.min_deg:
                return False
        return True


@dataclass(frozen=True)
class FallenAt(Pred):
    """One of ``names`` that stood at ``spot`` has fallen: its foot (the middle of its bottom face, which stays within
    half its thickness of where it stood however far it tips) within ``tol`` (x, y, z) of ``spot``, and tilted past
    ``min_deg``.  A domino line is set off when the domino on its first place falls -- not when one waiting off the
    line, to be stood there, is knocked over."""

    names: tuple[str, ...]
    spot: tuple[float, float, float]
    tol: tuple[float, float, float]
    min_deg: float = 15.0

    def __call__(self, frame, scene):
        for nm in self.names:
            p = frame.pose(nm, scene)
            if tilt_deg(p[3:]) <= self.min_deg:
                continue
            down = quat_rotate(p[3:], (0.0, 0.0, -scene.obj(nm).height / 2))
            if all(abs(p[i] + down[i] - self.spot[i]) <= self.tol[i] for i in range(3)):
                return True
        return False


@dataclass(frozen=True)
class AtLeast(Pred):
    """At least ``k`` of ``preds`` hold."""

    k: int
    preds: tuple[Pred, ...]

    def __call__(self, frame, scene):
        return sum(p(frame, scene) for p in self.preds) >= self.k


def Stacked(names: Sequence[str], base_region: str, xy_tol: float = 0.02) -> Pred:
    """``names[0]`` in ``base_region`` and every next object resting on the previous one."""
    preds = [InRegion(names[0], base_region)]
    preds += [OnTop(top, bottom, xy_tol=xy_tol) for bottom, top in zip(names, names[1:])]
    return all_of(*preds)


# ------------------------------------------------------------------------ temporal


class Temporal:
    def first(self, frames: Sequence[Frame], scene: SceneSpec) -> Optional[int]:
        """Index of the frame at which the formula is first satisfied, or None."""
        raise NotImplementedError

    def __call__(self, frames, scene) -> bool:
        return self.first(frames, scene) is not None


@dataclass(frozen=True)
class Eventually(Temporal):
    p: Pred

    def first(self, frames, scene):
        return next((i for i, f in enumerate(frames) if self.p(f, scene)), None)


@dataclass(frozen=True)
class HoldsFor(Temporal):
    """``p`` holds without interruption for at least ``seconds``."""

    p: Pred
    seconds: float

    def first(self, frames, scene):
        start = None
        for i, f in enumerate(frames):
            if self.p(f, scene):
                start = i if start is None else start
                if f.t - frames[start].t >= self.seconds - 1e-9:
                    return i
            else:
                start = None
        return None


def _into_last_stretch(frames, ok, seconds) -> Optional[int]:
    """Index of the frame ``seconds`` into the run's last unbroken stretch of frames that ``ok`` passes -- the one that
    reaches the last frame -- or None if the last frame fails or the stretch is shorter."""
    k = len(frames)
    while k > 0 and ok(frames[k - 1]):
        k -= 1
    if k == len(frames):
        return None
    t0 = frames[k].t
    return next((i for i in range(k, len(frames)) if frames[i].t - t0 >= seconds - 1e-9), None)


@dataclass(frozen=True)
class HoldsAtEnd(Temporal):
    """``p`` holds when the run ends and has held without a break through its last ``seconds`` at least: a ball left
    in its cradle, not one held there a second and then carried off, which ``HoldsFor`` passes.  The run is the frames
    given (a script is played to its end, never cut at the first success), so this can be judged only once it is over.

    Satisfied ``seconds`` into that last stretch (as ``LeftAt``), not at the last frame: where ``p`` is never left
    again, at the frame ``HoldsFor(p, seconds)`` is, and never before it.  It implies ``HoldsFor(p, seconds)``.
    (``LeftAt`` asks, besides, that one body lie still through the stretch.)"""

    p: Pred
    seconds: float = 0.0

    def first(self, frames, scene):
        return _into_last_stretch(frames, lambda f: self.p(f, scene), self.seconds)


@dataclass(frozen=True)
class Always(Temporal):
    """``p`` holds in every frame; satisfied at the last frame."""

    p: Pred

    def first(self, frames, scene):
        return len(frames) - 1 if frames and all(self.p(f, scene) for f in frames) else None


@dataclass(frozen=True)
class Before(Temporal):
    """Both happen and ``a`` is first satisfied no later than ``b``."""

    a: Temporal
    b: Temporal

    def first(self, frames, scene):
        ia, ib = self.a.first(frames, scene), self.b.first(frames, scene)
        return ib if ia is not None and ib is not None and ia <= ib else None


@dataclass(frozen=True)
class SoonAfter(Temporal):
    """``then`` holds no more than ``seconds`` after the last frame in which ``before`` held: a ball let go on a
    track is in the pot within seconds; one carried along takes far longer."""

    before: Pred
    then: Pred
    seconds: float

    def first(self, frames, scene):
        last = None
        for i, f in enumerate(frames):
            if self.before(f, scene):
                last = i
            elif last is not None and self.then(f, scene) and f.t - frames[last].t <= self.seconds + 1e-9:
                return i
        return None


@dataclass(frozen=True)
class AtOnset(Temporal):
    """``cond`` held in the last frame before ``event`` first held (the first frame of ``event``, or None): a domino
    line was whole when it was set off."""

    cond: Pred
    event: Pred

    def first(self, frames, scene):
        for i, f in enumerate(frames):
            if self.event(f, scene):
                return i if i > 0 and self.cond(frames[i - 1], scene) else None
        return None


@dataclass(frozen=True)
class Then(Temporal):
    """``b`` holds without a break for ``seconds``, from a frame after one in which ``a`` held: a door, once opened,
    shut again.  ``b`` holding before ``a`` ever did (the door shut at the start) does not count."""

    a: Pred
    b: Pred
    seconds: float = 0.0

    def first(self, frames, scene):
        seen, start = False, None
        for i, f in enumerate(frames):
            if seen and self.b(f, scene):
                start = i if start is None else start
                if f.t - frames[start].t >= self.seconds - 1e-9:
                    return i
            else:
                start = None
            seen = seen or self.a(f, scene)
        return None


@dataclass(frozen=True)
class LeftAt(Temporal):
    """``obj`` came to rest where ``where`` holds and was left there to the end of the run, ``seconds`` at least: in
    every frame of that last stretch ``where`` holds and the body lies within ``max_move`` of where it ends.  Satisfied
    ``seconds`` into the stretch.  A thing set down on its way and never taken on; one taken on later does not count."""

    obj: str
    where: Pred
    seconds: float
    max_move: float = 0.01

    def first(self, frames, scene):
        if not frames:
            return None
        end = frames[-1].pose(self.obj, scene)[:3]
        return _into_last_stretch(
            frames, lambda f: self.where(f, scene) and math.dist(f.pose(self.obj, scene)[:3], end) <= self.max_move,
            self.seconds)


@dataclass(frozen=True)
class Unless(Temporal):
    """``p``, unless ``done`` was satisfied first: what goes wrong once a task is done (a pipe that held its pressure
    for its test, then its valves let go) is no failure."""

    p: Temporal
    done: Temporal

    def first(self, frames, scene):
        i = self.p.first(frames, scene)
        if i is None:
            return None
        j = self.done.first(frames, scene)
        return i if j is None or j > i else None


@dataclass(frozen=True)
class AllOf(Temporal):
    parts: tuple[Temporal, ...]

    def first(self, frames, scene):
        idx = [t.first(frames, scene) for t in self.parts]
        return None if any(i is None for i in idx) else max(idx)  # type: ignore[type-var]


def all_temporal(*parts: Temporal) -> AllOf:
    return AllOf(tuple(parts))
