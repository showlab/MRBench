"""How a scene looks apart from its task: the table, the floor, the room, the light, and props on the table.

    mrbench looks                                  # every choice and the named looks
    mrbench render Con-04 --look kitchen              # a named look
    mrbench demo Rea-01 --look random --seed 3        # drawn from the seed
    mrbench render Con-01 --look default,table=tile:stone,clutter=desk:6,light=warm

A look (``engines.base.Look``) is a set of names; this module says what each name
is, and ``dress`` puts a look on a task instance, standing its props where no arm
works (``props.place_props``).  The task -- its bodies, regions, success and
scripted reference -- is the same in every look.  The default is a realistic room:
an oak top on aluminium legs in Isaac's Simple_Room, lit by a photo studio's
light; ``plain`` is a flat look.  Engine-independent: the Isaac
engine turns the names into materials, lights and bodies
(``engines/isaac/environment.py``).
"""

from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass
from typing import Optional, Union

from ..core.errors import ConfigError
from ..engines.base import RGB, Look
from .backdrop import BACKDROPS
from .paint import COLOURINGS, paint, repaint_text
from .patterns import BENCH_M, TILE_M
from .props import CLUTTER, PROPS, place_props

__all__ = ["Finish", "Light", "TABLES", "FLOORS", "ROOMS", "FORMS", "LIGHTS", "LOOKS", "CLUTTER", "PROPS",
           "BACKDROPS", "COLOURINGS", "parse_look", "random_look", "check_look", "dress", "describe", "paint", "repaint_text"]


@dataclass(frozen=True)
class Finish:
    """A surface: a photographed texture set (``downloads.TEXTURES``) or a drawn one (``pattern:<name>``,
    ``patterns.PATTERNS``) repeated ``repeat`` times a metre and multiplied by ``tint`` (linear RGB); or, with
    no ``maps``, a plain sRGB ``colour``, rough or metallic."""

    maps: str = ""
    repeat: float = 1.0
    tint: RGB = (1.0, 1.0, 1.0)
    colour: RGB = (0.5, 0.5, 0.5)
    roughness: float = 0.5
    metallic: bool = False


#: linen dyed for a tablecloth: the colour each comes out (linear).  Poly Haven's rough_linen photograph is a blue
#: linen, its mean (linear) _LINEN, so each is tinted by its colour over that (tinted by the colour itself, every
#: cloth would come out bluish)
_LINEN = (0.288, 0.412, 0.617)
_CLOTHS = {"white": (0.92, 0.92, 0.90), "cream": (0.90, 0.80, 0.60), "red": (0.62, 0.08, 0.07),
           "blue": (0.10, 0.22, 0.52), "green": (0.14, 0.36, 0.16), "grey": (0.30, 0.30, 0.32)}

TABLES: dict[str, Finish] = {
    "wood:oak": Finish("oak_veneer_01", 1.5),
    # the oak photograph darkened
    "wood:walnut": Finish("oak_veneer_01", 3.0, tint=(0.42, 0.30, 0.24)),
    "wood:ash": Finish("ash_veneer", 3.0),
    "wood:plywood": Finish("plywood", 2.0),
    # Poly Haven's marble_01: a floor of sandstone-coloured tiles, laid on the table as a tiled counter
    "tile:stone": Finish("marble_01", 1.0),
    "stone:white": Finish(colour=(0.90, 0.89, 0.86), roughness=0.15),
    "laminate:white": Finish(colour=(0.88, 0.88, 0.86), roughness=0.35),
    "laminate:grey": Finish(colour=(0.52, 0.53, 0.55), roughness=0.45),
    "matte:black": Finish(colour=(0.07, 0.07, 0.075), roughness=0.7),
    # brushed: polished (roughness 0.32) it mirrors the dim room and renders near black
    "metal:steel": Finish(colour=(0.70, 0.71, 0.72), roughness=0.55, metallic=True),
    # a workbench's steel top, worn: brushed, faintly smudged, finely scratched (drawn; the workshop's)
    "metal:workbench": Finish("pattern:workbench_steel", 1 / BENCH_M, metallic=True),
    # a lab's tops, drawn: an optical breadboard's M6 holes on a 25 mm grid, or 40 mm aluminium profiles with
    # their T-slots (the collider stays flat)
    "breadboard:black": Finish("pattern:breadboard_black", 1 / TILE_M),
    "breadboard:steel": Finish("pattern:breadboard_steel", 1 / TILE_M, metallic=True),
    # anodised: satin, not a mirror
    "extrusion": Finish("pattern:extrusion", 1 / TILE_M),
    # a flat brown slab
    "flat:wood": Finish(colour=(0.58, 0.47, 0.37), roughness=0.6),
    **{f"cloth:{c}": Finish("rough_linen", 8.0, tint=tuple(round(a / m, 3) for a, m in zip(t, _LINEN)))
       for c, t in _CLOTHS.items()},
}
FLOORS: dict[str, Finish] = {
    "concrete": Finish("concrete_floor_01", 1.0),
    "wood": Finish("laminate_floor_02", 0.5),
    "parquet": Finish("herringbone_parquet", 0.5),
    "carpet": Finish("dirty_carpet", 0.5),
    "tiles": Finish("large_grey_tiles", 0.5),
    "terrazzo": Finish("terrazzo_tiles", 0.5),
    "plain": Finish(colour=(0.55, 0.55, 0.57), roughness=0.7),
}
ROOMS = ("room", "studio", "plain")
#: "legs": a top on four round aluminium legs; "frame": on a frame of aluminium profiles (square legs and rails
#: under the top's edges); "block": a solid block down to the floor
FORMS = ("legs", "frame", "block")


@dataclass(frozen=True)
class Light:
    """The studio light: its strength times the usual, its colour temperature (K; 6500 is white) and how far
    it is turned about the vertical (degrees)."""

    scale: float = 1.0
    kelvin: float = 6500.0
    turn_deg: float = 0.0


LIGHTS: dict[str, Light] = {
    "studio": Light(),
    "warm": Light(0.9, 3800.0),
    "cool": Light(1.1, 8500.0),
    "dim": Light(0.45),
    "bright": Light(1.7),
    "evening": Light(0.6, 3200.0, 120.0),
    # a lab's: bright, a little cool, so that a black breadboard's holes show
    "lab": Light(1.9, 6000.0),
}

#: how many groups of props a set brings when the look does not say
CLUTTER_N = 2

#: the named looks: none of them brings props (``clutter=SET:N`` puts them on)
LOOKS: dict[str, Look] = {
    "default": Look(),
    "plain": Look(table="flat:wood", table_form="block", floor="plain", room="plain"),
    "studio": Look(table="wood:ash", floor="concrete", room="studio"),
    "office": Look(table="laminate:white", floor="carpet"),
    "walnut": Look(table="wood:walnut", floor="parquet", light="warm"),
    # a white top in a kitchen's corner -- cabinets, shelves, a fridge -- so that the table does not stand in an
    # empty room: the varied scenes' room at seed 0
    "kitchen": Look(table="stone:white", floor="tiles", light="bright", backdrop="kitchen"),
    # a workbench under a pegboard of tools behind the table: the varied scenes' room at seed 1
    # (``agent.cases.SEED_LOOKS``) and every card's look, the borrow configurations' props being factory things
    "workshop": Look(table="metal:workbench", table_form="block", floor="concrete", light="cool", backdrop="workshop"),
    "picnic": Look(table="cloth:red", floor="wood", light="warm"),
    "evening": Look(table="wood:plywood", top_m=0.06, floor="terrazzo", light="evening"),
    "dark": Look(table="matte:black", floor="concrete", room="studio", light="dim"),
    "lab": Look(table="breadboard:black", table_form="frame", floor="concrete", light="lab"),
    # a sideboard, a bookcase, an armchair behind the table: the varied scenes' room at seed 2 (with a cream cloth)
    "home": Look(table="wood:walnut", floor="parquet", light="warm", backdrop="home"),
    "rig": Look(table="extrusion", table_form="frame", top_m=0.04, floor="concrete"),
}


def check_look(look: Look) -> Look:
    """``look``, if every name in it is one this module knows."""
    def bad(what, value, have):
        raise ConfigError(f"no {what} {value!r}; have {', '.join(sorted(have))}")

    if look.table not in TABLES:
        bad("table finish", look.table, TABLES)
    if look.table_form not in FORMS:
        bad("table form", look.table_form, FORMS)
    if not 0.015 <= look.top_m <= 0.12:
        raise ConfigError(f"a table top {look.top_m} m thick: keep it 0.015-0.12 m")
    if look.floor != "room" and look.floor not in FLOORS:
        bad("floor", look.floor, [*FLOORS, "room"])
    if look.floor == "room" and look.room != "room":
        raise ConfigError(f"floor=room needs room=room (the {look.room!r} room has no floor of its own)")
    if look.room not in ROOMS:
        bad("room", look.room, ROOMS)
    if look.backdrop not in BACKDROPS:
        bad("backdrop", look.backdrop, BACKDROPS)
    if look.light not in LIGHTS:
        bad("light", look.light, LIGHTS)
    if look.clutter != "none" and look.clutter not in CLUTTER:
        bad("prop set", look.clutter, [*CLUTTER, "none"])
    if look.clutter_n < 0:
        raise ConfigError("clutter_n must be >= 0")
    if look.colors not in COLOURINGS:
        bad("colouring", look.colors, COLOURINGS)
    if look.distract < 0:
        raise ConfigError("distract must be >= 0")
    return look


def random_look(seed: int) -> Look:
    """A look drawn from ``seed`` alone -- the same for every task at one seed; it brings no props."""
    rng = random.Random(f"look/{seed}")
    room = rng.choices(ROOMS, weights=(6, 3, 1))[0]
    floors = [*FLOORS, "room"] if room == "room" else list(FLOORS)
    return check_look(Look(table=rng.choice(sorted(TABLES)), table_form=rng.choices(FORMS, weights=(4, 2, 1))[0],
                           top_m=rng.choice((0.03, 0.04, 0.06)), floor=rng.choice(floors), room=room,
                           light=rng.choice(sorted(LIGHTS))))


#: the keys a look's text may set, and the Look field each sets
_KEYS = {"table": "table", "form": "table_form", "top": "top_m", "floor": "floor", "room": "room",
         "backdrop": "backdrop", "light": "light", "clutter": "clutter", "colors": "colors", "distract": "distract"}


def parse_look(text: Optional[str], seed: int = 0) -> Look:
    """A look from text: a named look, or ``random`` (drawn from ``seed``), then ``key=value`` changes:

        kitchen
        random
        default,table=cloth:blue,floor=parquet,light=warm,clutter=desk:6,form=block,top=0.06

    ``clutter`` is a prop set and how many of its groups (``desk`` alone brings ``CLUTTER_N``; ``none`` none)."""
    parts = [p.strip() for p in (text or "default").split(",") if p.strip()]
    if not parts or "=" in parts[0]:
        parts.insert(0, "default")
    name, changes = parts[0], parts[1:]
    if name == "random":
        look = random_look(seed)
    elif name in LOOKS:
        look = LOOKS[name]
    else:
        raise ConfigError(f"no look {name!r}; have {', '.join(LOOKS)} or random")
    for c in changes:
        key, sep, value = c.partition("=")
        if not sep or key not in _KEYS:
            raise ConfigError(f"cannot read {c!r} in a look: give key=value, the keys {', '.join(_KEYS)}")
        if key == "clutter":
            kind, _, n = value.partition(":")
            look = dataclasses.replace(look, clutter=kind, clutter_n=0 if kind == "none" else int(n or CLUTTER_N))
        elif key == "top":
            look = dataclasses.replace(look, top_m=float(value))
        elif key == "distract":
            look = dataclasses.replace(look, distract=int(value))
        else:
            look = dataclasses.replace(look, **{_KEYS[key]: value})
        if key == "room" and value != "room" and look.floor == "room":  # a studio has no floor of its own
            look = dataclasses.replace(look, floor="concrete" if value == "studio" else "plain")
    return check_look(look)


def describe(look: Look) -> str:
    form = {"legs": f"on legs, its top {look.top_m * 100:.0f} cm", "frame": f"on a frame of aluminium profiles, "
            f"its top {look.top_m * 100:.0f} cm", "block": "a solid block"}[look.table_form]
    floor = "the room's floor" if look.floor == "room" else f"a {look.floor} floor"
    props = "no props" if look.clutter == "none" or not look.clutter_n else \
        f"up to {look.clutter_n} group{'s' if look.clutter_n > 1 else ''} of {look.clutter} props"
    room = {"room": "in a room", "studio": "in a photo studio", "plain": "in plain surroundings"}[look.room]
    if look.backdrop != "none":
        room += f" with a {look.backdrop} behind the table"
    colours = ", the task's things in random colours" if look.colors == "random" else ""
    distract = f", {look.distract} distractor{'s' if look.distract > 1 else ''} within reach" if look.distract else ""
    return f"{look.table} table ({form}), {floor}, {room}, {look.light} light, {props}{colours}{distract}"


def dress(inst, look: Union[Look, str, None], seed: Optional[int] = None):
    """``inst`` in ``look`` (a Look, or text for ``parse_look``): its scene with the look, and the look's props
    stood where no arm works -- where, drawn from the task, level and seed."""
    from ..tasks.instance import TaskInstance

    assert isinstance(inst, TaskInstance)
    seed = inst.seed if seed is None else seed
    look = check_look(look) if isinstance(look, Look) else parse_look(look, seed)
    rng = random.Random(f"{inst.spec.seed_id}/{inst.level}/{seed}/props")
    props = place_props(inst.scene, look, rng)
    inst = dataclasses.replace(inst, scene=dataclasses.replace(inst.scene, look=look, props=props))
    if look.colors == "random":
        inst = paint(inst, random.Random(f"{inst.spec.seed_id}/{inst.level}/{seed}/colours"))
    if look.distract:
        from .distractors import place

        inst = place(inst, look.distract, random.Random(f"{inst.spec.seed_id}/{inst.level}/{seed}/distractors"))
    return inst
