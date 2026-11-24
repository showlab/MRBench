"""Turn a trajectory into an EpisodeScore.  A pure function of (instance, frames)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Sequence

from ..engines.base import Frame

if TYPE_CHECKING:
    from ..tasks.instance import TaskInstance


@dataclass
class EpisodeScore:
    success: bool
    #: fraction of milestones reached
    progress: float
    milestones_hit: list[str] = field(default_factory=list)
    #: simulated time (s) at which success / each milestone was first reached
    first_hit_s: dict[str, float] = field(default_factory=dict)
    #: the jobs' own failure signals that showed (``tasks/signals.py``), and when each first did
    signals_hit: list[str] = field(default_factory=list)
    signals_first_s: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)


def score(inst: "TaskInstance", frames: Sequence[Frame]) -> EpisodeScore:
    from ..tasks import signals

    scene = inst.scene
    hit, times = [], {}
    for m in inst.milestones:
        i = m.formula.first(frames, scene)
        if i is not None:
            hit.append(m.name)
            times[m.name] = round(frames[i].t, 3)
    i = inst.success.first(frames, scene)
    if i is not None:
        times["success"] = round(frames[i].t, 3)
    shown, shown_s = [], {}
    for s in signals.of(inst):
        k = s.formula.first(frames, scene)
        if k is not None:
            shown.append(s.name)
            shown_s[s.name] = round(frames[k].t, 3)
    n = len(inst.milestones)
    return EpisodeScore(success=i is not None, progress=len(hit) / n if n else float(i is not None),
                        milestones_hit=hit, first_hit_s=times, signals_hit=shown, signals_first_s=shown_s)
