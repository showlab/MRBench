"""A physics-free engine for tests and CI.

Arms jump to their (clipped) joint targets; objects and cloths stay where they
are unless teleported, props stay where they stand, and hinges have no springs.
It exists to exercise task building, scoring, logging and the CLI end to end
without a GPU -- never to produce benchmark numbers.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from ..core.errors import ConfigError
from ..robots import get_model
from .base import ArmCommand, Engine, Frame, Pose7, SceneSpec, Vec3


class MockEngine(Engine):
    name = "mock"

    def __init__(self, control_dt: float = 0.1, **_ignored):
        self.control_dt = control_dt
        self._scene: SceneSpec | None = None
        self._frame: Frame | None = None

    def load(self, scene: SceneSpec, seed: int) -> Frame:
        self._scene = scene
        arms = {}
        for a in scene.arms:
            m = get_model(a.model)
            arms[a.slot] = {"q": list(m.home), "gripper": 1.0}
        cloths = {c.name: [c.flat_points()[i] for i in c.marks] for c in scene.cloths}
        self._frame = Frame(t=0.0, objects={o.name: o.pose for o in scene.dynamic_objects}, arms=arms, cloths=cloths,
                            props={p.name: p.pose for p in scene.props})
        return self._snapshot()

    def step(self, commands: Mapping[str, ArmCommand]) -> Frame:
        if self._frame is None or self._scene is None:
            raise ConfigError("MockEngine.step called before load")
        for slot, cmd in commands.items():
            m = get_model(self._scene.arm(slot).model)
            self._frame.arms[slot] = {"q": list(m.clip_arm(cmd.q)), "gripper": min(max(cmd.gripper, 0.0), 1.0)}
        self._frame.t += self.control_dt
        return self._snapshot()

    def set_object_pose(self, name: str, pose: Pose7) -> Frame:
        if self._frame is None or name not in self._frame.objects:
            raise ConfigError(f"no dynamic object {name!r} to teleport")
        self._frame.objects[name] = tuple(pose)  # type: ignore[assignment]
        return self._snapshot()

    def set_cloth_points(self, name: str, points: Sequence[Vec3]) -> Frame:
        if self._frame is None or self._scene is None or name not in self._frame.cloths:
            raise ConfigError(f"no cloth {name!r} to move")
        spec = self._scene.cloth(name)
        if len(points) != len(spec.nodes):
            raise ConfigError(f"{name}: {len(points)} points given, the cloth has {len(spec.nodes)}")
        self._frame.cloths[name] = [tuple(points[i]) for i in spec.marks]  # type: ignore[misc]
        return self._snapshot()

    def _snapshot(self) -> Frame:
        f = self._frame
        assert f is not None
        return Frame(t=f.t, objects=dict(f.objects), arms={k: {"q": list(v["q"]), "gripper": v["gripper"]}
                                                          for k, v in f.arms.items()},
                     cloths={k: list(v) for k, v in f.cloths.items()}, props=dict(f.props))
