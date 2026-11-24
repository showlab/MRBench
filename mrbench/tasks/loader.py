"""Strict loader for task cards.  Unknown or missing fields are errors, not warnings."""

from __future__ import annotations

import threading
from functools import lru_cache
from importlib import resources

import yaml

from ..core.errors import ConfigError
from .spec import LEVELS, STATUSES, SUITES, VARIANTS, ArmSlot, Borrow, Knob, TaskSpec

_SUITE_KEYS = {"suite", "title", "tasks"}
_TASK_REQUIRED = {"id", "name", "summary", "arms", "knob", "horizon_s", "success",
                  "milestones", "status", "impl"}
_TASK_OPTIONAL = {"todo", "look", "facts", "borrow", "variants", "was", "horizons"}
_ARM_KEYS = {"mount", "model", "role"}
_KNOB_KEYS = {"field", "description", "levels", "unit"}
_BORROW_KEYS = {"replaces", "prop", "distractor"}
_BORROW_OPTIONAL = {"remove"}


def _check_keys(where: str, got: dict, required: set, optional: set = frozenset()) -> None:
    missing, unknown = required - got.keys(), got.keys() - required - optional
    if missing or unknown:
        raise ConfigError(f"{where}: missing {sorted(missing)}, unknown {sorted(unknown)}")


def _parse_task(suite: str, d: dict, where: str) -> TaskSpec:
    _check_keys(where, d, _TASK_REQUIRED, _TASK_OPTIONAL)
    where = f"{where} {d['id']}"
    for a in d["arms"]:
        _check_keys(f"{where} arms", a, _ARM_KEYS)
    _check_keys(f"{where} knob", d["knob"], _KNOB_KEYS)
    if set(d["knob"]["levels"]) != set(LEVELS):
        raise ConfigError(f"{where}: knob levels must be exactly {LEVELS}, got {sorted(d['knob']['levels'])}")
    if d["status"] not in STATUSES:
        raise ConfigError(f"{where}: status {d['status']!r} not in {STATUSES}")
    if suite not in SUITES:
        raise ConfigError(f"{where}: suite {suite!r} not in {SUITES}")
    if "borrow" in d:
        _check_keys(f"{where} borrow", d["borrow"], _BORROW_KEYS, _BORROW_OPTIONAL)
    variants = tuple(d.get("variants", ("exact",)))
    if "exact" not in variants or not set(variants) <= set(VARIANTS):
        raise ConfigError(f"{where}: variants {list(variants)} must include 'exact' and come from {VARIANTS}")
    remove = d.get("borrow", {}).get("remove")
    if remove is not None and remove not in {a["mount"] for a in d["arms"]}:
        raise ConfigError(f"{where}: borrow removes {remove!r}, which is not one of its arms")
    if "borrow" in variants and remove is None:
        raise ConfigError(f"{where}: a built borrow version must say which arm it takes away (borrow: remove:)")
    horizons = d.get("horizons", {})
    if not isinstance(horizons, dict) or not set(horizons) <= set(VARIANTS) - {"exact"}:
        raise ConfigError(f"{where}: horizons {horizons!r} must map versions other than exact (whose horizon is "
                          f"horizon_s) from {VARIANTS} to seconds")
    if any(float(v) <= 0 for v in horizons.values()):
        raise ConfigError(f"{where}: horizons must be positive, got {horizons!r}")
    if "look" in d:
        from ..scenery import parse_look

        parse_look(d["look"])  # a look that does not parse is the card's error
    return TaskSpec(
        id=d["id"], name=d["name"], suite=suite, summary=" ".join(d["summary"].split()),
        arms=tuple(ArmSlot(**a) for a in d["arms"]),
        knob=Knob(**d["knob"]), horizon_s=float(d["horizon_s"]), success=" ".join(d["success"].split()),
        milestones=tuple(d["milestones"]), status=d["status"], impl=d["impl"], todo=tuple(d.get("todo", ())),
        look=d.get("look", "default"), facts=tuple(" ".join(f.split()) for f in d.get("facts", ())),
        borrow=Borrow(**{k: " ".join(v.split()) for k, v in d["borrow"].items()}) if "borrow" in d else None,
        variants=tuple(v for v in VARIANTS if v in variants), was=d.get("was"),
        horizons=tuple((v, float(horizons[v])) for v in VARIANTS if v in horizons),
    )


#: the cards are read once: two threads reading them at once would make two copies of every spec, and ``build``
#: checks that an impl's instance carries the very spec it was asked for
_LOCK = threading.Lock()
#: card directories a plugin adds (``add_cards``), read after the package's own
_EXTRA: list = []


def add_cards(directory) -> None:
    """Read the ``*.yaml`` task cards of ``directory`` too (``core/plugins.py``); call it before any card is read."""
    if _read_cards.cache_info().currsize:
        raise ConfigError(f"cards from {directory} added after the cards were read")
    _EXTRA.append(directory)


def all_specs() -> dict[str, TaskSpec]:
    from ..core import plugins

    plugins.load()
    with _LOCK:
        return _read_cards()


@lru_cache(maxsize=1)
def _read_cards() -> dict[str, TaskSpec]:
    specs: dict[str, TaskSpec] = {}
    every = [c for cards in [resources.files("mrbench.tasks").joinpath("cards"), *_EXTRA] for c in cards.iterdir()
             if c.name.endswith(".yaml")]
    for card in sorted(every, key=lambda c: c.name):  # by name across the folders: a plugin's cards among the rest
        _read_card(card, specs)
    return specs


def _read_card(card, specs: dict[str, TaskSpec]) -> None:
    doc = yaml.safe_load(card.read_text())
    _check_keys(card.name, doc, _SUITE_KEYS)
    for d in doc["tasks"]:
        spec = _parse_task(doc["suite"], d, card.name)
        if spec.id in specs:
            raise ConfigError(f"duplicate task id {spec.id} in {card.name}")
        specs[spec.id] = spec


def get(task_id: str) -> TaskSpec:
    try:
        return all_specs()[task_id]
    except KeyError:
        raise ConfigError(f"unknown task {task_id!r}; available: {', '.join(all_specs())}") from None


def ids() -> list[str]:
    return list(all_specs())
