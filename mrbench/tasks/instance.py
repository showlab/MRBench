"""A TaskInstance is one (task, level, seed): a concrete scene plus how it is scored."""

from __future__ import annotations

import dataclasses
import importlib
from dataclasses import dataclass, field
from typing import Any, Sequence, Union

from ..core.errors import ConfigError
from ..engines.base import Pose7, SceneSpec, Vec3
from ..scoring.predicates import Temporal
from .loader import get
from .spec import LEVELS, VARIANTS, TaskSpec


@dataclass(frozen=True)
class Milestone:
    name: str
    formula: Temporal


@dataclass(frozen=True)
class TaskInstance:
    spec: TaskSpec
    level: str
    seed: int
    scene: SceneSpec
    instruction: str
    success: Temporal
    milestones: tuple[Milestone, ...]
    #: (arm slot, object/region name) pairs the arm must be able to reach -- a design rule
    reach: tuple[tuple[str, str], ...] = ()
    #: pairs the arm must NOT reach; this is what forces the arms to cooperate
    out_of_reach: tuple[tuple[str, str], ...] = ()
    #: pairs of arms that must not reach any point in common (no handing over between them)
    apart: tuple[tuple[str, str], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    #: which version of the task (``spec.VARIANTS``): its ``script`` and ``keyframes`` branch on it.  A borrow
    #: version names its props and distractors in ``metadata["borrow"]`` ({"props": [...], "decoys": [...]}, the
    #: n-th decoy standing in for the n-th prop)
    variant: str = "exact"

    @property
    def knob_value(self):
        return self.spec.knob.levels[self.level]


#: A keyframe maps dynamic object names to poses, and cloth names to the position
#: of every point (``ClothSpec.nodes`` order).  An impl's ``keyframes(inst)``
#: returns a sequence of them that, replayed on the mock engine, must satisfy the
#: success formula -- proof that the formula is achievable and consistent.
Keyframes = list[dict[str, Union[Pose7, Sequence[Vec3]]]]


def _module(spec: TaskSpec):
    try:
        return importlib.import_module(spec.impl)
    except ModuleNotFoundError as e:
        raise ConfigError(f"{spec.id}: impl module {spec.impl!r} not found") from e


def build(task_id: str, level: str, seed: int, look=None, variant: str = "exact", decoy: bool = False) -> TaskInstance:
    """One (task, level, seed), dressed in ``look`` (a ``Look``, or text for ``scenery.parse_look``) or, without
    one, in its card's look.  ``variant`` picks the configuration, the team on the table (``spec.VARIANTS``); only
    those on the card's ``variants`` are built.  The impl builds the exact version with ``build(level, seed)`` and any
    other with ``build(level, seed, variant=...)``.  With ``decoy`` (a borrow version only) the props and the
    distractors swap names (``swap_decoys``): the scripted reference then takes the distractor."""
    spec = get(task_id)
    if level not in LEVELS:
        raise ConfigError(f"level must be one of {LEVELS}, got {level!r}")
    if variant not in VARIANTS:
        raise ConfigError(f"variant must be one of {VARIANTS}, got {variant!r}")
    if variant not in spec.variants:
        raise ConfigError(f"{task_id} has no {variant} version yet (built: {', '.join(spec.variants)})")
    if spec.status == "planned":
        raise ConfigError(f"{task_id} is only planned; it has no implementation yet")
    if decoy and variant != "borrow":
        raise ConfigError("only a borrow version has a distractor to swap in (decoy)")
    mod = _module(spec)
    inst = mod.build(level, seed) if variant == "exact" else mod.build(level, seed, variant=variant)
    if hasattr(mod, "own_scene"):  # what the task adds when built as itself (``tasks.build``)
        inst = mod.own_scene(inst)
    if inst.spec is not spec:
        raise ConfigError(f"{spec.impl}.build returned an instance for {inst.spec.id}")
    if inst.variant != variant:
        inst = dataclasses.replace(inst, variant=variant)
    if variant == "borrow":
        gone = spec.borrow.remove
        if any(a.slot == gone for a in inst.scene.arms):
            raise ConfigError(f"{spec.impl}.build left the {gone} arm in its borrow version")
        if decoy:
            inst = swap_decoys(inst)
    look = spec.look if look is None else look
    if look != "default":
        from ..scenery import dress

        inst = dress(inst, look, seed)
    return inst


def tag(task_id: str, variant: str = "exact", decoy: bool = False) -> str:
    """The name a version's pictures and reports are written under: "Con-01", "Con-01-borrow", "Con-01-borrow-decoy"."""
    return task_id + ("" if variant == "exact" else f"-{variant}") + ("-decoy" if decoy else "")


def swap_decoys(inst: TaskInstance) -> TaskInstance:
    """The borrow version with each prop and its distractor (``metadata["borrow"]``) swapping names -- and whatever
    is tied to them by name (a hinge's body, a fit's bodies, a cloth's ties) going with the body it belongs to -- so
    that a scripted reference written for the prop takes the distractor.  The success formula is unchanged."""
    b = inst.metadata.get("borrow") or {}
    props, decoys = list(b.get("props", ())), list(b.get("decoys", ()))
    if not props or len(props) != len(decoys):
        raise ConfigError(f"{inst.spec.id}: its borrow version must name as many distractors as props, got "
                          f"{props} and {decoys}")
    swap = {**dict(zip(props, decoys)), **dict(zip(decoys, props))}
    sc = inst.scene
    names = {o.name for o in sc.objects}
    missing = sorted(set(swap) - names)
    if missing:
        raise ConfigError(f"{inst.spec.id}: no {', '.join(missing)} in its borrow version's scene")
    rn = lambda n: swap.get(n, n)  # noqa: E731
    scene = dataclasses.replace(
        sc, objects=tuple(dataclasses.replace(o, name=rn(o.name)) for o in sc.objects),
        hinges=tuple(dataclasses.replace(h, body=rn(h.body)) for h in sc.hinges),
        fits=tuple(dataclasses.replace(f, body=rn(f.body), base=rn(f.base)) for f in sc.fits),
        cloths=tuple(dataclasses.replace(c, ties=tuple(dataclasses.replace(t, body=rn(t.body)) for t in c.ties))
                     for c in sc.cloths))
    return dataclasses.replace(inst, scene=scene, metadata={**inst.metadata, "decoy": True})


def keyframes(inst: TaskInstance) -> Keyframes:
    return _module(inst.spec).keyframes(inst)
