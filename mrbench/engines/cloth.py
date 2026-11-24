"""Cloth: a rectangular sheet (a towel) that is not a rigid body.

A cloth is simulated as a grid of points ``cell`` apart, two triangles a
cell.  Its state is where a sparse subset of those points, the *marks*, are:
that is what a Frame logs and what scoring reads.  Named *keys* (its ends and
corners) are marks too.  On Isaac the grid is a PhysX particle cloth
(``engines/isaac/cloth.py``); the mock engine only moves it when told to.

Own frame: ``u`` along its length, ``v`` along its width, centred, lying flat
(turned by ``yaw_deg`` about the world z axis).  Nothing here imports an engine.

A cloth can be *tied* to rigid bodies (``ClothTie``): some of its points held fixed in a body's frame, where they
are when everything starts -- a lantern hung on a cord, a ring at a cord's end.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class ClothTie:
    """Points of a cloth held fixed in a dynamic body's frame: the point nearest (u, v) -- ``u`` from the cloth's
    ``l`` end, ``v`` across from its middle -- or with ``v`` None every point across the cloth at ``u``.  The body
    does not collide with the cloth (a hanging one with nothing)."""

    body: str
    u: float
    v: Optional[float] = None
    #: instead, the body hangs from the point, as far below it as it lay from it (beside it while the cloth lies too
    #: low), moved there every step: weightless, it neither falls nor pulls the cloth.  (A PhysX attachment moves a
    #: cloth with a body, not the body with the cloth: a 5 g lantern tied to a cord stays on the table while the
    #: cord, lifted, stretches down to it.)
    hang: bool = False


@dataclass(frozen=True)
class ClothSpec:
    name: str
    #: length (along u) and width (along v), m
    size: tuple[float, float]
    #: centre as it lies flat; z is its mid-plane, ``radius`` above what it lies on
    pos: Vec3
    yaw_deg: float = 0.0
    color: tuple[float, float, float] = (0.92, 0.92, 0.92)
    mass_kg: float = 0.05
    #: grid step; each point is a particle ``cell / 2`` thick each way
    cell: float = 0.01
    #: every this many grid steps each way is a mark
    mark_every: int = 5
    #: PhysX particle-cloth springs, and friction against everything
    stretch_stiffness: float = 1.0e4
    bend_stiffness: float = 2.0
    shear_stiffness: float = 100.0
    spring_damping: float = 5.0e-3
    friction: float = 0.8
    #: particle-solver passes per step (a long thin cloth -- a cord -- needs many: with 16 a 1.9 m cord of 1 cm cells
    #: hangs 18% longer than it is)
    iterations: int = 16
    keys: tuple[str, ...] = field(default=("end_l", "end_r", "corner_ll", "corner_lr", "corner_rl", "corner_rr"))
    ties: tuple[ClothTie, ...] = ()

    @property
    def radius(self) -> float:
        return self.cell / 2

    @cached_property
    def grid(self) -> tuple[int, int]:
        """Points along u and along v."""
        return round(self.size[0] / self.cell) + 1, round(self.size[1] / self.cell) + 1

    @cached_property
    def nodes(self) -> tuple[tuple[float, float], ...]:
        """(u, v) of every point, u-major: node ``i * nv + j``."""
        nu, nv = self.grid
        lu, lv = self.size
        return tuple((-lu / 2 + lu * i / (nu - 1), -lv / 2 + lv * j / (nv - 1)) for i in range(nu) for j in range(nv))

    @cached_property
    def triangles(self) -> tuple[tuple[int, int, int], ...]:
        nu, nv = self.grid
        out = []
        for i in range(nu - 1):
            for j in range(nv - 1):
                a, b, c, d = i * nv + j, (i + 1) * nv + j, (i + 1) * nv + j + 1, i * nv + j + 1
                out += [(a, b, c), (a, c, d)]
        return tuple(out)

    def _key_node(self, key: str) -> int:
        nu, nv = self.grid
        i_of = {"l": 0, "r": nu - 1}
        if key.startswith("end_"):
            return i_of[key[-1]] * nv + nv // 2
        if key.startswith("corner_"):  # corner_<end><side>: end l/r along u, side l/r along v
            return i_of[key[-2]] * nv + (0 if key[-1] == "l" else nv - 1)
        raise KeyError(key)

    @cached_property
    def marks(self) -> tuple[int, ...]:
        """Node numbers of the marks: every ``mark_every``-th point each way, the edges, the keys."""
        nu, nv = self.grid
        k = self.mark_every
        us = sorted({*range(0, nu, k), nu - 1})
        vs = sorted({*range(0, nv, k), nv - 1})
        out = {i * nv + j for i in us for j in vs} | {self._key_node(key) for key in self.keys}
        return tuple(sorted(out))

    def tie_nodes(self, tie: ClothTie) -> tuple[int, ...]:
        """The points a tie holds (``nodes`` indices)."""
        nu, nv = self.grid
        i = min(range(nu), key=lambda k: abs(self.nodes[k * nv][0] - (tie.u - self.size[0] / 2)))
        if tie.v is None:
            return tuple(i * nv + j for j in range(nv))
        return (i * nv + min(range(nv), key=lambda j: abs(self.nodes[j][1] - tie.v)),)

    def key(self, name: str) -> int:
        """Where a key point is in ``marks`` (and in a frame's list of mark positions)."""
        return self.marks.index(self._key_node(name))

    def at(self, uv: tuple[float, float]) -> Vec3:
        """A point of the flat cloth, in the world."""
        c, s = math.cos(math.radians(self.yaw_deg)), math.sin(math.radians(self.yaw_deg))
        u, v = uv
        return (self.pos[0] + c * u - s * v, self.pos[1] + s * u + c * v, self.pos[2])

    def flat_points(self) -> list[Vec3]:
        """Every node where it starts: lying flat."""
        return [self.at(uv) for uv in self.nodes]

    def key_point(self, name: str) -> Vec3:
        return self.at(self.nodes[self._key_node(name)])
