"""What every engine consumes (SceneSpec) and produces (Frame).

Coordinates are in the world frame, metres, z up; the table top sits at
``table_top_z``.  Quaternions are (w, x, y, z), matching Isaac Lab.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Mapping, Optional, Sequence

from ..core.errors import ConfigError
from .cloth import ClothSpec, ClothTie  # also re-exported: a scene is built from base alone

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # w x y z
Pose7 = tuple[float, float, float, float, float, float, float]  # x y z qw qx qy qz
RGB = tuple[float, float, float]


def yaw_quat(yaw_deg: float) -> Quat:
    h = math.radians(yaw_deg) / 2
    return (math.cos(h), 0.0, 0.0, math.sin(h))


def axis_quat(axis: Vec3, deg: float) -> Quat:
    n = math.sqrt(sum(a * a for a in axis))
    h = math.radians(deg) / 2
    return (math.cos(h), *(math.sin(h) * a / n for a in axis))  # type: ignore[return-value]


def z_to(axis: Sequence[float]) -> Quat:
    """The shortest turn that takes the z axis to ``axis`` (a unit vector or not)."""
    n = math.sqrt(sum(c * c for c in axis))
    ax, ay, az = (c / n for c in axis)
    if az > 1.0 - 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    if az < -1.0 + 1e-9:
        return (0.0, 1.0, 0.0, 0.0)  # half a turn about x
    # about z x axis = (-ay, ax, 0), by acos(az)
    return axis_quat((-ay, ax, 0.0), math.degrees(math.acos(az)))


def quat_mul(a: Sequence[float], b: Sequence[float]) -> Quat:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def quat_rotate(q: Sequence[float], v: Sequence[float]) -> Vec3:
    w, x, y, z = q
    _, rx, ry, rz = quat_mul(quat_mul(q, (0.0, *v)), (w, -x, -y, -z))
    return (rx, ry, rz)


def quat_yaw_deg(q: Sequence[float]) -> float:
    """Heading of the body x axis about world z."""
    w, x, y, z = q
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def pose7(pos: Vec3, yaw_deg: float = 0.0, quat: Optional[Quat] = None) -> Pose7:
    return (*pos, *(quat if quat is not None else yaw_quat(yaw_deg)))  # type: ignore[return-value]


def transform(pose: Sequence[float], local: Sequence[float]) -> Vec3:
    """A point given in a body's frame, in the world."""
    r = quat_rotate(pose[3:7], local)
    return (pose[0] + r[0], pose[1] + r[1], pose[2] + r[2])


@dataclass(frozen=True)
class ArmPlacement:
    slot: str  # unique name of this arm inside the scene, e.g. "panda", "piper_left"
    model: str  # key of mrbench.robots.MODELS
    pos: Vec3  # base position
    yaw_deg: float


@dataclass(frozen=True)
class Part:
    """One shape of a compound body, in the body's frame.  Cylinders stand along the body's z."""

    name: str
    shape: Literal["box", "cylinder", "sphere"]
    size: tuple[float, ...]
    pos: Vec3 = (0.0, 0.0, 0.0)
    yaw_deg: float = 0.0  # about the body's z axis
    color: Optional[RGB] = None  # None: the body's colour
    #: this part's own friction, as ``ObjectSpec.friction`` (a block's rubber feet, its sides as the rest); None: the
    #: default material
    friction: Optional[float] = None
    friction_combine: Literal["average", "min", "multiply", "max"] = "average"


@dataclass(frozen=True)
class ObjectSpec:
    """A primitive object, or with ``parts`` a compound one (a door with its handle, a
    coat hanger).  ``dynamic=False`` makes it a static collider (bin walls, rack posts)."""

    name: str
    shape: Literal["box", "cylinder", "sphere", "compound"]
    #: box and compound: (lx, ly, lz), a compound's the box round its parts; cylinder: (radius, height); sphere: (radius,)
    size: tuple[float, ...]
    pos: Vec3  # centre (a compound's: its frame's origin)
    yaw_deg: float = 0.0
    color: RGB = (0.7, 0.7, 0.7)
    mass_kg: float = 0.05
    dynamic: bool = True
    parts: tuple[Part, ...] = ()
    #: the start orientation when it is more than a turn about z (a hanger on its rail); overrides yaw_deg
    quat: Optional[Quat] = None
    #: drawn as clear glass tinted ``color`` (a guard, a partition the cameras see through); it collides as any body
    glass: bool = False
    #: its friction coefficient (static and dynamic), and how it combines with what it touches ("min" makes a slick
    #: surface slick against anything); None: PhysX's default, 0.5.  Primitive bodies only: a compound's parts take
    #: their own (``Part.friction``).
    friction: Optional[float] = None
    friction_combine: Literal["average", "min", "multiply", "max"] = "average"

    @property
    def height(self) -> float:
        return {"box": lambda s: s[2], "compound": lambda s: s[2], "cylinder": lambda s: s[1],
                "sphere": lambda s: 2 * s[0]}[self.shape](self.size)

    @property
    def pose(self) -> Pose7:
        return pose7(self.pos, self.yaw_deg, self.quat)

    def part(self, name: str) -> Part:
        try:
            return next(p for p in self.parts if p.name == name)
        except StopIteration:
            raise ConfigError(f"{self.name} has no part {name!r}") from None


@dataclass(frozen=True)
class RegionSpec:
    """An axis-aligned box used by predicates; rendered as a flat, non-colliding marker."""

    name: str
    center: Vec3
    half: Vec3
    color: RGB = (0.9, 0.9, 0.3)
    visible: bool = True


@dataclass(frozen=True)
class HingeSpec:
    """A spring-loaded revolute joint between a dynamic body and the world (a cabinet door, a lid, a leaf).

    The axis runs through ``pivot`` along ``axis`` (world frame; vertical by default).  The
    body starts closed (angle 0) and opens by turning ``sign`` x (0 .. ``open_max_deg``) about
    ``axis`` (right-handed).  A torsion spring of ``stiffness`` N*m/rad pulls it ``preload_deg``
    past closed, so a door that is let go swings shut and stays pressed against its stop.  With
    a level axis gravity turns the body too: a leaf with no spring hangs on whichever limit it
    falls toward.
    """

    name: str
    body: str
    pivot: Vec3
    sign: Literal[1, -1]
    open_max_deg: float = 110.0
    stiffness: float = 1.0  # N*m/rad
    damping: float = 0.1  # N*m*s/rad
    preload_deg: float = 10.0
    axis: Vec3 = (0.0, 0.0, 1.0)
    #: the clearance check's model of it let go (the engines simulate it anyway): back at its stop, or -- propped --
    #: swinging back only as far as the bodies in its way let it, and pushed on by a held body that comes into it (a
    #: block set in a door's way to keep it open)
    propped: bool = False


@dataclass(frozen=True)
class FitSpec:
    """A tight fit: ``body`` sits in ``base`` where it starts (a plug's pins in its socket), free
    only to slide out along the base's z, held there by a stiff spring -- the socket's contacts
    gripping the pins.  Pulled out further than ``travel_m`` -- that is, with more than
    ``break_force_n`` -- it lets go, and the pins slide out of the socket.

    On Isaac it is a PhysX prismatic joint whose linear drive is the spring, removed by the
    engine once the body has come out ``travel_m`` (a PhysX joint's own break force does not
    break on the GPU pipeline); the mock engine ignores it."""

    name: str
    body: str
    base: str
    break_force_n: float
    travel_m: float = 0.005


@dataclass(frozen=True)
class Look:
    """How a scene looks apart from its task (``scenery`` names every choice): the table's top and how it
    stands, the floor, the room round the table, the light, and which props stand about on the table.
    Nothing here changes what the task is; the props stand where no arm reaches."""

    #: the top's finish: "wood:oak", "stone:white", "cloth:blue", ... (``scenery.TABLES``)
    table: str = "wood:oak"
    #: "legs": a top ``top_m`` thick on four legs; "frame": on a frame of aluminium profiles; "block": a solid
    #: block down to the floor
    table_form: str = "legs"
    top_m: float = 0.04
    #: a floor finish (``scenery.FLOORS``), or "room": the room's own floor
    floor: str = "room"
    #: "room": walls, a window and a ceiling light round the table; "studio": a photo studio's backdrop
    #: all round, seen far off; "plain": a flat grey floor and an even sky
    room: str = "room"
    #: "none", or a furnished corner of the room behind the table, drawn only and out of every arm's reach
    #: (``scenery.backdrop``): "kitchen", "workshop" or "home"
    backdrop: str = "none"
    #: ``scenery.LIGHTS``
    light: str = "studio"
    #: a prop set (``scenery.CLUTTER``) or "none", and how many of its groups at most
    clutter: str = "none"
    clutter_n: int = 0
    #: "card": the task's bodies in the colours its impl gives them; "random": the bodies the impl names as painted
    #: (``metadata["paint"]``) in colours drawn from the seed, and the words that name them changed to match
    #: (``scenery.paint``)
    colors: str = "card"
    #: how many distractors to stand within the arms' reach (``scenery.distractors``): bodies the task does not
    #: need, which the team sees and may move
    distract: int = 0


@dataclass(frozen=True)
class PropSpec:
    """Something standing on the table that the task does not use (a mug, a stack of books): a rigid body
    that can be knocked over, out of every arm's reach (``scenery.place_props``).  ``kind`` names it in
    ``scenery.PROPS``, which says what it is made of."""

    name: str
    kind: str
    #: the centre of its upright bounding box
    pos: Vec3
    yaw_deg: float
    #: its upright bounding box
    size: Vec3
    mass_kg: float

    @property
    def pose(self) -> Pose7:
        return pose7(self.pos, self.yaw_deg)


@dataclass(frozen=True)
class SceneSpec:
    arms: tuple[ArmPlacement, ...]
    objects: tuple[ObjectSpec, ...]
    regions: tuple[RegionSpec, ...] = ()
    table_size: Vec3 = (3.2, 1.8, 0.75)
    table_top_z: float = 0.75
    hinges: tuple[HingeSpec, ...] = ()
    cloths: tuple[ClothSpec, ...] = ()
    fits: tuple[FitSpec, ...] = ()
    look: Look = Look()
    props: tuple[PropSpec, ...] = ()

    def __post_init__(self):
        names = [a.slot for a in self.arms] + [o.name for o in self.objects] + [r.name for r in self.regions]
        names += [h.name for h in self.hinges] + [c.name for c in self.cloths] + [f.name for f in self.fits]
        names += [p.name for p in self.props]
        dup = {n for n in names if names.count(n) > 1}
        if dup:
            raise ConfigError(f"duplicate names in scene: {sorted(dup)}")

    def arm(self, slot: str) -> ArmPlacement:
        return next(a for a in self.arms if a.slot == slot)

    def obj(self, name: str) -> ObjectSpec:
        try:
            return next(o for o in self.objects if o.name == name)
        except StopIteration:
            raise ConfigError(f"no object {name!r} in scene") from None

    def region(self, name: str) -> RegionSpec:
        try:
            return next(r for r in self.regions if r.name == name)
        except StopIteration:
            raise ConfigError(f"no region {name!r} in scene") from None

    def cloth(self, name: str) -> ClothSpec:
        try:
            return next(c for c in self.cloths if c.name == name)
        except StopIteration:
            raise ConfigError(f"no cloth {name!r} in scene") from None

    def point(self, name: str) -> Vec3:
        """Initial centre of an object, region or cloth by name; ``<cloth>:<key>`` is a key point of a cloth."""
        if any(r.name == name for r in self.regions):
            return self.region(name).center
        cloth, _, key = name.partition(":")
        if any(c.name == cloth for c in self.cloths):
            return self.cloth(cloth).key_point(key) if key else self.cloth(cloth).pos
        return self.obj(name).pos

    @property
    def dynamic_objects(self) -> tuple[ObjectSpec, ...]:
        return tuple(o for o in self.objects if o.dynamic)


@dataclass(frozen=True)
class ArmCommand:
    """Joint-space command for one arm: absolute arm joint targets + gripper opening in [0, 1]."""

    q: tuple[float, ...]
    gripper: float = 1.0


@dataclass
class Frame:
    t: float
    objects: dict[str, Pose7]  # dynamic objects only; static ones keep their spec pose
    arms: dict[str, dict] = field(default_factory=dict)  # slot -> {"q": [...], "gripper": float}
    cloths: dict[str, list[Vec3]] = field(default_factory=dict)  # cloth -> its marks, in ClothSpec.marks order
    props: dict[str, Pose7] = field(default_factory=dict)  # every prop (SceneSpec.props)

    def pose(self, name: str, scene: SceneSpec) -> Pose7:
        if name in self.objects:
            return self.objects[name]
        return scene.obj(name).pose

    def to_json(self) -> dict:
        d = {"t": round(self.t, 4), "objects": {k: [round(v, 5) for v in p] for k, p in self.objects.items()},
             "arms": self.arms}
        if self.cloths:
            d["cloths"] = {k: [[round(v, 4) for v in p] for p in pts] for k, pts in self.cloths.items()}
        if self.props:
            d["props"] = {k: [round(v, 5) for v in p] for k, p in self.props.items()}
        return d


class Engine(ABC):
    """Steps a SceneSpec under joint-space commands and reports Frames."""

    name: str = "base"
    #: seconds of simulated time per ``step`` call
    control_dt: float = 0.1

    @abstractmethod
    def load(self, scene: SceneSpec, seed: int) -> Frame:
        """Build (or reset to) ``scene`` and return the first frame."""

    @abstractmethod
    def step(self, commands: Mapping[str, ArmCommand]) -> Frame:
        """Apply commands for one control period and return the resulting frame."""

    def close(self) -> None:
        pass

    def set_object_pose(self, name: str, pose: Pose7) -> Optional[Frame]:
        """Teleport a dynamic object (used by oracle replays); engines may refuse."""
        raise NotImplementedError(f"{self.name} engine does not support teleporting objects")

    def set_cloth_points(self, name: str, points: Sequence[Vec3]) -> Optional[Frame]:
        """Put every point of a cloth (``ClothSpec.nodes`` order) somewhere, at rest."""
        raise NotImplementedError(f"{self.name} engine does not support moving cloth")
