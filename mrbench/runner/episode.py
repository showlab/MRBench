"""Run one episode: build -> load -> step until the horizon -> score."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from ..engines.base import Engine, Frame
from ..policies import Policy, make_policy
from ..scoring import EpisodeScore, score
from ..tasks.instance import TaskInstance, keyframes

LOG_SCHEMA = "mrbench-log/1"


@dataclass
class EpisodeResult(EpisodeScore):
    task: str = ""
    level: str = ""
    seed: int = 0
    engine: str = ""
    policy: str = ""
    steps: int = 0
    sim_time_s: float = 0.0
    wall_time_s: float = 0.0
    frames: list[dict] = field(default_factory=list)

    def to_json(self, keep_frames: bool = False) -> dict:
        d = {"schema": LOG_SCHEMA, **super().to_json()}
        if not keep_frames:
            d.pop("frames")
        return d


def _finish(inst: TaskInstance, frames: list[Frame], engine: Engine, policy: str, t0: float) -> EpisodeResult:
    s = score(inst, frames)
    return EpisodeResult(**s.__dict__, task=inst.spec.id, level=inst.level, seed=inst.seed, engine=engine.name,
                         policy=policy, steps=len(frames) - 1, sim_time_s=round(frames[-1].t, 3),
                         wall_time_s=round(time.time() - t0, 3), frames=[f.to_json() for f in frames])


def run_episode(inst: TaskInstance, engine: Engine, policy: Policy, horizon_s: Optional[float] = None,
                policy_name: str = "") -> EpisodeResult:
    t0 = time.time()
    frames = [engine.load(inst.scene, inst.seed)]
    policy.reset(inst, inst.seed)
    steps = int(round((horizon_s or inst.spec.horizon(inst.variant)) / engine.control_dt))
    for _ in range(steps):
        frames.append(engine.step(policy.act(frames[-1])))
    return _finish(inst, frames, engine, policy_name or type(policy).__name__, t0)


def replay_keyframes(inst: TaskInstance, engine: Engine, hold_s: float = 1.5,
                     final_hold_s: float = 3.0) -> EpisodeResult:
    """Teleport objects through the impl's oracle keyframes, holding each for ``hold_s``
    and the last one for ``final_hold_s`` (longer than any HoldsFor in a success formula).

    A state-level check that the success formula is achievable; it says nothing
    about whether the robots can physically do it.
    """
    t0 = time.time()
    frames = [engine.load(inst.scene, inst.seed)]
    hold = make_policy("do_nothing")
    hold.reset(inst, inst.seed)
    kfs = keyframes(inst)
    cloths = {c.name for c in inst.scene.cloths}
    for i, kf in enumerate(kfs):
        for name, value in kf.items():
            if name in cloths:
                engine.set_cloth_points(name, value)
            else:
                engine.set_object_pose(name, value)
        dur = final_hold_s if i == len(kfs) - 1 else hold_s
        for _ in range(max(1, int(round(dur / engine.control_dt)))):
            frames.append(engine.step(hold.act(frames[-1])))
    return _finish(inst, frames, engine, "oracle_keyframes", t0)
