import importlib
import json

import pytest

from mrbench import batch
from mrbench.tasks import build


def test_a_surplus_scene_is_the_exact_one_with_the_cases_fourth_arm():
    from mrbench.agent.cases import SETUPS

    with_surplus = [t for t in SETUPS if "surplus" in SETUPS[t].expect]
    todo = batch.scenes(with_surplus, ["surplus"], ["easy"], [0])
    assert [s.task for s in todo] == with_surplus and todo[0].name == f"{with_surplus[0]}-surplus-easy-s0"
    inst, _ = batch.build_scene(todo[0])
    exact = build(todo[0].task, "easy", 0)
    assert [a.slot for a in inst.scene.arms][:-1] == [a.slot for a in exact.scene.arms]
    assert inst.scene.arms[-1].slot == "north" and inst.scene.objects == exact.scene.objects


def test_randomizing_follows_the_seed():
    inst = build("Rea-02", "easy", 4)
    a, info_a = batch.randomize(inst, "random", "random")
    b, info_b = batch.randomize(inst, "random", "random")
    assert info_a == info_b and a.scene == b.scene  # the same seed draws the same
    lo, hi = batch.MASS_RANGE
    for o, o0 in zip(a.scene.objects, inst.scene.objects):
        if o.dynamic:
            assert lo * o0.mass_kg - 1e-4 <= o.mass_kg <= hi * o0.mass_kg + 1e-4
        else:
            assert o == o0
    combo = {arm.slot: arm.model for arm in a.scene.arms}
    assert combo in batch.robot_choices("Rea-02")  # a combination that can do it and still needs every arm
    same, info = batch.randomize(inst)
    assert same.scene == inst.scene and info == {}



def test_sizes_drawn_follow_the_seed_and_are_put_back():
    from mrbench.tasks.impl import rea02_tomato_flume as m

    r0 = m.TOMATO_R
    with batch.sized("Rea-02", 4, "random") as a:
        tomato = build("Rea-02", "easy", 4).scene.obj("tomato")
    with batch.sized("Rea-02", 4, "random") as b:
        pass
    lo, hi = batch.SIZES["Rea-02"][0]
    assert a == b and lo <= a["size_scale"] <= hi  # the same seed draws the same
    assert tomato.size[0] == pytest.approx(r0 * a["size_scale"], rel=1e-3)
    assert m.TOMATO_R == r0  # put back
    with batch.sized("Con-01", 4, "random") as none:  # nothing of Con-01's is drawn
        assert none == {}


@pytest.mark.parametrize("task", sorted(batch.SIZES))
def test_each_task_builds_and_scripts_with_its_size_drawn(task):
    (lo, hi), _ = batch.SIZES[task]
    with batch.sized(task, 1, "random") as drawn:
        inst = build(task, "hard", 1)
        assert importlib.import_module(inst.spec.impl).script(inst)
    assert lo <= drawn["size_scale"] <= hi

def test_the_summary_counts_and_lists_failures(tmp_path):
    for seed, ok in ((0, True), (1, False), (2, True)):
        rep = {"scene": f"Rea-03-easy-s{seed}", "task": "Rea-03", "variant": "exact", "level": "easy", "seed": seed,
               "success": ok, "milestones_hit": ["west_gap_filled"], "missed_grips": []}
        (tmp_path / f"Rea-03-easy-s{seed}.json").write_text(json.dumps(rep))
    text = batch.summarize(tmp_path)
    assert "| Rea-03 | exact | easy | 3 | 2 | 67% |" in text
    assert "failures: 1 of 3" in text and "Rea-03-easy-s1" in text
    assert json.loads((tmp_path / "summary.json").read_text())["failures"] == ["Rea-03-easy-s1"]


def test_a_scene_whose_isaac_did_not_start_is_played_once_more(tmp_path, monkeypatch):
    from mrbench import render

    calls = []

    def fake_run_children(jobs, gpus, logs, timeout_s):
        calls.append([name for name, _ in jobs])
        logs.mkdir(parents=True, exist_ok=True)
        if len(calls) == 1:  # the first time, one scene's Kit cannot make its shared memory
            (logs / f"{jobs[0][0]}.log").write_text(f"... Assertion (false) failed: {batch.KIT_NO_SHM} named carb-x\n")
            return {jobs[0][0]}
        return set()

    monkeypatch.setattr(render, "run_children", fake_run_children)
    todo = batch.scenes(["Rea-03"], ["exact"], ["easy"], [0, 1])
    batch.run(todo, tmp_path, ["0"])
    assert calls == [[s.name for s in todo], [todo[0].name]]
