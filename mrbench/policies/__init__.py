"""Policies map a Frame to one ArmCommand per arm.

The policies registered here are negative controls.  Any other policy (scripted,
learned or an LLM agent) implements the same two methods and registers itself.
"""

from __future__ import annotations

import random
from typing import Mapping

from ..core.registry import Registry
from ..engines.base import ArmCommand, Frame
from ..robots import get_model

POLICIES: Registry = Registry("policy")


class Policy:
    def reset(self, inst, seed: int) -> None:
        self.inst = inst

    def act(self, frame: Frame) -> Mapping[str, ArmCommand]:
        raise NotImplementedError


@POLICIES.register("do_nothing")
class DoNothing(Policy):
    """Hold the home pose with open grippers.  Must never succeed (a design rule)."""

    def act(self, frame):
        return {a.slot: ArmCommand(get_model(a.model).home, 1.0) for a in self.inst.scene.arms}


@POLICIES.register("random")
class RandomJoints(Policy):
    """Uniform random joint targets within limits; a floor for any learned policy."""

    def reset(self, inst, seed):
        super().reset(inst, seed)
        self.rng = random.Random(seed)

    def act(self, frame):
        out = {}
        for a in self.inst.scene.arms:
            m = get_model(a.model)
            out[a.slot] = ArmCommand(tuple(self.rng.uniform(lo, hi) for lo, hi in m.arm_limits), self.rng.random())
        return out


def make_policy(name: str, **kwargs) -> Policy:
    return POLICIES.get(name)(**kwargs)
