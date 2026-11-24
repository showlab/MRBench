"""Rea-01's borrow configuration keeps the exact one's goal and physical objects, and is registered."""
from dataclasses import replace
import pytest
from mrbench.tasks.impl import rea01_relay as task

@pytest.mark.parametrize('level', ['easy', 'hard'])
@pytest.mark.parametrize('seed', range(5))
def test_borrow_preserves_work_and_uses_present_arms(level, seed):
    exact = task.build(level, seed)
    inst = replace(task.build(level, seed, variant='borrow'), variant='borrow')
    assert inst.instruction == exact.instruction
    assert inst.success == exact.success
    assert [m.name for m in inst.milestones] == [m.name for m in exact.milestones]
    for obj in exact.scene.objects:
        assert inst.scene.obj(obj.name) == obj
    assert {a.slot for a in inst.scene.arms} == {'west', 'east'}
    policy = task.script(inst)
    assert all(a.slot in {'west', 'east'} for ph in policy for a in ph.actions)
    assert sum(ph.seconds for ph in policy) + .5 <= inst.spec.horizon_s
    assert 'borrow' in inst.spec.variants  # registered: built by mrbench.tasks.build as well
    from mrbench.tasks import build
    assert build('Rea-01', level, seed, variant='borrow').metadata['borrow'] == inst.metadata['borrow']
