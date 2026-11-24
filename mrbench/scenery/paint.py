"""The task's things in colours drawn from the seed (``Look.colors = "random"``), and the words that name them.

A task's impl names what may be painted in ``metadata["paint"]``: a body, a cloth or a marked place -> the colour
word its instruction and descriptions use for it ("box": "orange").  Things that share a word share a colour.  Each
word is given a colour from ``PALETTE``, no two the same; a compound body's parts in its own colour change with it,
parts with a colour of their own (a handle, a bar) do not.  The words change in the instruction, and ``repaint_text``
changes them in whatever else the team is told (the agent's briefing).
Things with a colour of their own -- an apple, a tomato, coffee, wood, metal -- are not named, and keep it.
Engine-independent.
"""

from __future__ import annotations

import dataclasses
import random
import re
from typing import Mapping, Optional

from ..core.errors import ConfigError

#: a look's ``colors``
COLOURINGS = ("card", "random")


def repaint_text(text: str, words: Optional[Mapping[str, str]]) -> str:
    """``text`` with every whole colour word in ``words`` replaced by its new one, all at once (two words may swap)."""
    if not words or not text:
        return text
    pattern = re.compile(r"\b(" + "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True)) + r")\b")
    return pattern.sub(lambda m: words[m.group(1)], text)


def paint(inst, rng: random.Random):
    """``inst`` with the things its impl names as painted in colours drawn from ``rng``; ``metadata["repaint"]`` says
    which word became which."""
    from ..tasks.impl._common import COLORS, PALETTE

    named = inst.metadata.get("paint") or {}
    if not named:
        return inst
    sc = inst.scene
    have = {o.name for o in sc.objects} | {r.name for r in sc.regions} | {c.name for c in sc.cloths}
    missing = sorted(set(named) - have)
    if missing:
        raise ConfigError(f"{inst.spec.id}: paint names {', '.join(missing)}, which is not in the scene")
    old = sorted(set(named.values()))
    words = dict(zip(old, rng.sample(PALETTE, len(old))))
    rgb = {name: COLORS[words[w]] for name, w in named.items()}

    def body(o):
        if o.name not in rgb:
            return o
        parts = tuple(dataclasses.replace(p, color=rgb[o.name]) if p.color == o.color else p for p in o.parts)
        return dataclasses.replace(o, color=rgb[o.name], parts=parts)

    scene = dataclasses.replace(
        sc, objects=tuple(body(o) for o in sc.objects),
        regions=tuple(dataclasses.replace(r, color=rgb[r.name]) if r.name in rgb else r for r in sc.regions),
        cloths=tuple(dataclasses.replace(c, color=rgb[c.name]) if c.name in rgb else c for c in sc.cloths))
    return dataclasses.replace(inst, scene=scene, instruction=repaint_text(inst.instruction, words),
                               metadata={**inst.metadata, "repaint": words})
