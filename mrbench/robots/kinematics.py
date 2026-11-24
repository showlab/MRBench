"""Forward and inverse kinematics of each arm, engine-independent (numpy).

The chains are the robots' own descriptions: the Panda's from the standard
``franka_description`` URDF (the one Isaac Lab's Franka USD is made from), the
Piper's and the SO-101's from their pinned URDFs (``mrbench assets fetch``).
``tests/test_kinematics.py`` checks the Panda's against poses measured in Isaac.

Poses are 4x4 homogeneous matrices in the arm's base frame unless a function says otherwise.
Needs numpy (the ``dev`` and ``isaac`` extras); the rest of the core does not.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Sequence

try:
    import numpy as np
except ImportError:  # the core runs without numpy; only planning a scripted reference needs it
    np = None  # type: ignore[assignment]

from ..core.errors import AssetMissing
from .models import RobotModel, get_model


def rpy_matrix(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def transform(xyz=(0.0, 0.0, 0.0), rot: Optional[np.ndarray] = None) -> np.ndarray:
    t = np.eye(4)
    t[:3, 3] = xyz
    if rot is not None:
        t[:3, :3] = rot
    return t


def quat_to_matrix(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def matrix_to_quat(m: np.ndarray) -> tuple[float, float, float, float]:
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = (0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
    else:
        i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2
        v = [0.0, 0.0, 0.0, 0.0]
        v[0] = (m[k, j] - m[j, k]) / s
        v[i + 1] = 0.25 * s
        v[j + 1] = (m[j, i] + m[i, j]) / s
        v[k + 1] = (m[k, i] + m[i, k]) / s
        q = tuple(v)
    return q if q[0] >= 0 else tuple(-x for x in q)  # type: ignore[return-value]


def pose_matrix(pose: Sequence[float]) -> np.ndarray:
    """(x y z qw qx qy qz) -> 4x4."""
    return transform(pose[:3], quat_to_matrix(pose[3:7]))


def matrix_pose(m: np.ndarray) -> tuple:
    return (*(float(v) for v in m[:3, 3]), *matrix_to_quat(m[:3, :3]))


@dataclass(frozen=True)
class Joint:
    origin: np.ndarray  # 4x4 from the parent link
    axis: Optional[np.ndarray]  # None: fixed


#: The Panda as in franka_description's panda_arm URDF, up to panda_hand (link8 turned -pi/4 about z).
_PANDA = [
    ((0, 0, 0.333), (0, 0, 0)), ((0, 0, 0), (-math.pi / 2, 0, 0)), ((0, -0.316, 0), (math.pi / 2, 0, 0)),
    ((0.0825, 0, 0), (math.pi / 2, 0, 0)), ((-0.0825, 0.384, 0), (-math.pi / 2, 0, 0)),
    ((0, 0, 0), (math.pi / 2, 0, 0)), ((0.088, 0, 0), (math.pi / 2, 0, 0)),
]


def _panda_chain() -> list[Joint]:
    chain = [Joint(transform(xyz, rpy_matrix(*rpy)), np.array([0.0, 0.0, 1.0])) for xyz, rpy in _PANDA]
    chain.append(Joint(transform((0, 0, 0.107), rpy_matrix(0, 0, -math.pi / 4)), None))  # flange -> panda_hand
    return chain


def _urdf_chain(model: RobotModel) -> list[Joint]:
    """The joints from the base to ``ee_link``, read from the fetched URDF."""
    from .fetch import require

    try:
        root = ET.parse(require(model.name)).getroot()
    except Exception as e:  # not fetched
        raise AssetMissing(f"{model.name}: URDF not fetched ({e}); run `mrbench assets fetch`") from e
    by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
    path, link = [], model.ee_link
    while link in by_child:
        j = by_child[link]
        path.append(j)
        link = j.find("parent").get("link")
    chain = []
    for j in reversed(path):
        o = j.find("origin")
        xyz = [float(v) for v in (o.get("xyz") if o is not None else "0 0 0").split()]
        rpy = [float(v) for v in (o.get("rpy") if o is not None and o.get("rpy") else "0 0 0").split()]
        moving = j.get("type") in ("revolute", "continuous")
        axis = np.array([float(v) for v in j.find("axis").get("xyz").split()]) if moving else None
        chain.append(Joint(transform(xyz, rpy_matrix(*rpy)), axis))
    if sum(j.axis is not None for j in chain) != model.arm_dof:
        raise AssetMissing(f"{model.name}: the URDF chain to {model.ee_link} does not have {model.arm_dof} joints")
    return chain


@lru_cache(maxsize=None)
def chain(name: str) -> tuple[Joint, ...]:
    model = get_model(name)
    return tuple(_panda_chain() if name == "panda" else _urdf_chain(model))


def _rot(axis: np.ndarray, a: float) -> np.ndarray:
    x, y, z = axis
    c, s = math.cos(a), math.sin(a)
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + s * k + (1 - c) * k @ k


def fk(name: str, q: Sequence[float]) -> np.ndarray:
    """``ee_link`` in the base frame."""
    t, i = np.eye(4), 0
    for j in chain(name):
        t = t @ j.origin
        if j.axis is not None:
            t = t @ transform(rot=_rot(j.axis, q[i]))
            i += 1
    return t


def frames(name: str, q: Sequence[float]) -> list[np.ndarray]:
    """Every joint's frame along the chain (where it turns, base frame), then ``ee_link``'s."""
    t, i, out = np.eye(4), 0, []
    for j in chain(name):
        t = t @ j.origin
        if j.axis is not None:
            out.append(t.copy())
            t = t @ transform(rot=_rot(j.axis, q[i]))
            i += 1
    out.append(t)
    return out


def tcp(name: str, q: Sequence[float]) -> np.ndarray:
    """The tool point (between the fingertips) in the base frame."""
    return fk(name, q) @ transform((0.0, 0.0, get_model(name).tcp_offset_m))


def pose_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """6-vector taking ``a`` to ``b``: position, then rotation as an axis times angle (world frame)."""
    r = b[:3, :3] @ a[:3, :3].T
    angle = math.acos(max(-1.0, min(1.0, (np.trace(r) - 1) / 2)))
    if angle < 1e-9:
        w = np.zeros(3)
    elif angle > math.pi - 1e-6:  # half a turn: the axis is the eigenvector of +1
        vals, vecs = np.linalg.eig(r)
        w = np.real(vecs[:, np.argmin(abs(vals - 1))]) * angle
    else:
        w = angle / (2 * math.sin(angle)) * np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    return np.concatenate([b[:3, 3] - a[:3, 3], w])


def tcp_and_jacobian(name: str, q: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """The tool point and its geometric Jacobian (6 x n: linear, then angular), in one pass."""
    t, i = np.eye(4), 0
    axes, origins = [], []
    for j in chain(name):
        t = t @ j.origin
        if j.axis is not None:
            axes.append(t[:3, :3] @ j.axis)
            origins.append(t[:3, 3].copy())
            t = t @ transform(rot=_rot(j.axis, q[i]))
            i += 1
    t = t @ transform((0.0, 0.0, get_model(name).tcp_offset_m))
    p = t[:3, 3]
    jac = np.stack([np.concatenate([np.cross(a, p - o), a]) for a, o in zip(axes, origins)], axis=1)
    return t, jac


def ik(name: str, target: np.ndarray, seed: Sequence[float], iters: int = 400, damping: float = 0.02,
       centering: float = 0.1, pos_tol: float = 1e-3, rot_tol_deg: float = 0.5) -> Optional[np.ndarray]:
    """Joint angles that put the tool point at ``target`` (base frame), starting from ``seed``:
    damped least squares inside the joint limits, spare freedom leaning toward the arm's home
    posture (elbow up, wrist over the hand).  None if it does not converge."""
    m = get_model(name)
    lo, hi = np.array([a for a, _ in m.arm_limits]), np.array([b for _, b in m.arm_limits])
    margin = 0.01
    q = np.clip(np.array(seed, dtype=float), lo + margin, hi - margin)
    rest = np.clip(np.array(m.home, dtype=float), lo + margin, hi - margin)
    for _ in range(iters):
        now, j = tcp_and_jacobian(name, q)
        err = pose_error(now, target)
        if np.linalg.norm(err[:3]) < pos_tol and math.degrees(np.linalg.norm(err[3:])) < rot_tol_deg:
            return q
        inv = j.T @ np.linalg.inv(j @ j.T + damping ** 2 * np.eye(6))
        dq = inv @ err
        if j.shape[1] > 6:
            dq += (np.eye(j.shape[1]) - inv @ j) @ (centering * (rest - q))
        step = np.max(np.abs(dq))
        if step > 0.2:
            dq *= 0.2 / step
        q = np.clip(q + dq, lo + margin, hi - margin)
    return None
