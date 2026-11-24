"""The engine-independent half of `mrbench render`: goal poses, the settle report, the views."""

import pytest

from mrbench import render
from mrbench.engines.mock import MockEngine
from mrbench.runner.episode import replay_keyframes
from mrbench.tasks import all_specs, build


@pytest.mark.parametrize("tid", sorted(t for t, s in all_specs().items() if s.status != "planned"))
def test_goal_poses_are_the_oracles_end_state(tid):
    inst = build(tid, "hard", 0)
    final = replay_keyframes(inst, MockEngine()).frames[-1]
    for name, value in render.goal_poses(inst).items():
        if name in final["objects"]:
            assert final["objects"][name] == pytest.approx(list(value), abs=1e-4)
        else:  # a cloth: the frame keeps its marks
            marks = [v for i in inst.scene.cloth(name).marks for v in value[i]]
            assert [v for p in final["cloths"][name] for v in p] == pytest.approx(marks, abs=1e-3)


def test_settle_report_flags_what_moved_and_what_sagged():
    inst = build("Rea-01", "easy", 0, "kitchen,clutter=kitchen:2")
    first, last = MockEngine().load(inst.scene, 0), MockEngine().load(inst.scene, 0)
    assert render.settle_report(inst, first, last)["flags"] == []

    x, y, z, *q = first.objects["apple_0"]
    last.objects["apple_0"] = (x + 0.05, y, z - 0.10, *q)
    last.arms["west"]["q"][1] += 0.2
    assert inst.scene.props
    prop = inst.scene.props[0]
    last.props[prop.name] = (prop.pos[0] + 0.03, *prop.pose[1:])
    flags = render.settle_report(inst, first, last)["flags"]
    assert any(f.startswith("apple_0 moved") for f in flags)
    assert "apple_0 below the table top" in flags
    assert any(f.startswith("west is 0.20 rad off home at joint2") for f in flags)
    assert f"{prop.name} ({prop.kind}) moved 3.0 cm" in flags


def test_every_view_is_rendered():
    assert set(render.default_views()) == set(render.VIEW_NAMES)


