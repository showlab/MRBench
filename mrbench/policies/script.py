"""A scripted reference, written as choreography: phase by phase, where each arm's
tool point goes and when it grips or lets go.  Engine-independent; the Isaac engine
plays it with inverse kinematics (``engines/isaac/puppeteer.py``).

It is privileged: it reads the scene's true
geometry, and a grip *attaches* what is between the fingers (a body, or the
patch of a cloth under them) instead of holding it by friction.  It shows that a
task can be done and what doing it looks like; it is never a result.

Poses are (x y z qw qx qy qz) in the world frame.  A tool frame's z points out
through the fingers and its y is the way they open (``RobotModel.tcp_offset_m``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Union

from ..engines.base import Pose7, Quat, Vec3

Path = Callable[[float], Pose7]  # s in [0, 1] -> where the tool point is


@dataclass(frozen=True)
class Move:
    """The tool point goes to ``to`` (smoothly, from wherever it is), or along ``to(s)``.  With
    ``joints``, the arm goes there the shortest way in joint space -- every joint straight from
    where it is to the nearest configuration that puts the tool point at ``to`` -- rather than the
    tool point along a straight line: for moving across, where the hand's path does not matter."""

    slot: str
    to: Union[Pose7, Path]
    joints: bool = False
    #: which way round the hand goes: None, whichever the planner finds turns the arm least (a parallel grip works
    #: either way); False as ``to`` has it, True turned half round about its pointing -- where the side its wrist
    #: camera ends up on matters (the planner keeps the way round it chose for a while, so a hand coming from
    #: elsewhere may have the camera on the other side)
    turn: Optional[bool] = None
    #: the same switch under another name: False is ``turn=False`` (the hand as ``to`` has it, for a grasp
    #: whose wrist camera must face one side of a narrow opening); True, the default, leaves it to ``turn``
    allow_flip: bool = True

    def __post_init__(self):
        if not self.allow_flip and self.turn:
            raise ValueError(f"{self.slot}: allow_flip=False keeps the hand as `to` has it, turn=True turns it round")


@dataclass(frozen=True)
class Grip:
    """Close on ``target`` (a body or a cloth), ``width`` apart, and take it.  With ``attach``
    False the hand only works on it -- presses it down, pushes it -- and nothing is attached:
    what it does to the body is physics alone."""

    slot: str
    target: str
    width: float = 0.0
    attach: bool = True


@dataclass(frozen=True)
class Release:
    slot: str


@dataclass(frozen=True)
class Open:
    """Open (or close) the fingers to ``width`` apart without taking anything: shaping the hand
    before it goes in, so a finger fits a narrow gap (behind a door handle)."""

    slot: str
    width: float


@dataclass(frozen=True)
class Clip:
    """A tool held in a hand takes ``target`` in its clip: the tool's two parts ``jaw_1`` and ``jaw_2``, pressed round
    it (a sponge mop's clip round its sponge).  Privileged, as a grip is: ``target`` is fixed to the tool -- if it is
    within a grip's reach of the point between the jaws and no wider across them than they stand apart; else nothing
    is taken (``collision.clip_jaws``)."""

    tool: str
    target: str


@dataclass(frozen=True)
class Unclip:
    """The tool's clip lets go of what it holds (the hand on the tool works its catch)."""

    tool: str


@dataclass(frozen=True)
class Phase:
    seconds: float
    actions: tuple[Union[Move, Grip, Release, Open, Clip, Unclip], ...] = ()
    #: what the video's caption says while it plays
    note: str = ""


Script = list[Phase]


def phase(seconds: float, note: str, *actions) -> Phase:
    return Phase(seconds, tuple(actions), note)


def _norm(v: Sequence[float]) -> Vec3:
    n = math.sqrt(sum(x * x for x in v))
    return (v[0] / n, v[1] / n, v[2] / n)


def _cross(a, b) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def tool_quat(approach: Vec3, across: Vec3) -> Quat:
    """The tool orientation whose z is ``approach`` and whose y (the fingers' travel) is
    ``across``, made square to it."""
    z = _norm(approach)
    d = sum(a * b for a, b in zip(across, z))
    y = _norm(tuple(a - d * b for a, b in zip(across, z)))
    x = _cross(y, z)
    m = [[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]]  # columns x y z
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return (0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s)
    i = max(range(3), key=lambda k: m[k][k])
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + m[i][i] - m[j][j] - m[k][k]) * 2
    q = [0.0, 0.0, 0.0, 0.0]
    q[0] = (m[k][j] - m[j][k]) / s
    q[i + 1] = 0.25 * s
    q[j + 1] = (m[j][i] + m[i][j]) / s
    q[k + 1] = (m[k][i] + m[i][k]) / s
    return tuple(q)  # type: ignore[return-value]


def radial_tool(base: Sequence[float], point: Sequence[float], pitch_deg: float, across=None) -> Quat:
    """A hand reaching out from its arm's base: pointing ``pitch_deg`` below level, horizontally away from ``base``
    toward ``point``, its fingers travelling along ``across`` (made square to it) -- by default level and square to
    the reach.  (Far out, a Piper reaches a point only with its hand pointing about this way.)"""
    rx, ry = point[0] - base[0], point[1] - base[1]
    n = math.hypot(rx, ry) or 1.0
    p = math.radians(pitch_deg)
    approach = (rx / n * math.cos(p), ry / n * math.cos(p), -math.sin(p))
    return tool_quat(approach, across if across is not None else (-ry / n, rx / n, 0.0))


def carry(tool: Pose7, body_from: Pose7, body_to: Pose7) -> Pose7:
    """Where a hand holding a body (rigidly, as a privileged grip does) at ``tool`` while the body is at ``body_from``
    must be for the body to be at ``body_to``."""
    from ..engines.base import quat_mul, quat_rotate

    back = (body_from[3], -body_from[4], -body_from[5], -body_from[6])
    rel_p = quat_rotate(back, [t - b for t, b in zip(tool[:3], body_from[:3])])
    rel_q = quat_mul(back, tool[3:])
    pos = [b + r for b, r in zip(body_to[:3], quat_rotate(body_to[3:], rel_p))]
    return (*pos, *quat_mul(body_to[3:], rel_q))  # type: ignore[return-value]


def smooth(s: float) -> float:
    """Minimum-jerk timing: starts and stops at rest."""
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10 - 15 * s + 6 * s * s)


def slerp(a: Sequence[float], b: Sequence[float], s: float) -> Quat:
    d = sum(x * y for x, y in zip(a, b))
    if d < 0:
        b, d = [-x for x in b], -d
    if d > 0.9995:
        q = [x + s * (y - x) for x, y in zip(a, b)]
    else:
        th = math.acos(d)
        q = [(math.sin((1 - s) * th) * x + math.sin(s * th) * y) / math.sin(th) for x, y in zip(a, b)]
    n = math.sqrt(sum(x * x for x in q))
    return tuple(x / n for x in q)  # type: ignore[return-value]


def blend(a: Pose7, b: Pose7, s: float) -> Pose7:
    """Between two poses: straight in position, slerp in orientation."""
    return (*(x + s * (y - x) for x, y in zip(a[:3], b[:3])), *slerp(a[3:], b[3:], s))  # type: ignore[return-value]


def offset(pose: Pose7, dx=0.0, dy=0.0, dz=0.0) -> Pose7:
    """The same orientation, moved in the world frame."""
    return (pose[0] + dx, pose[1] + dy, pose[2] + dz, *pose[3:])  # type: ignore[return-value]


def back_off(pose: Pose7, d: float) -> Pose7:
    """``d`` back along the tool's approach (its z): where the hand waits before closing in."""
    from ..engines.base import quat_rotate

    z = quat_rotate(pose[3:], (0.0, 0.0, 1.0))
    return offset(pose, -d * z[0], -d * z[1], -d * z[2])


def home_pose(inst, slot: str) -> Pose7:
    """Where an arm's tool point is at its home pose, in the world (needs numpy)."""
    from ..robots import get_model
    from ..robots import kinematics as kin

    arm = inst.scene.arm(slot)
    base = kin.transform(arm.pos, kin.rpy_matrix(0.0, 0.0, math.radians(arm.yaw_deg)))
    return kin.matrix_pose(base @ kin.tcp(arm.model, get_model(arm.model).home))
